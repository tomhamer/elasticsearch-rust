//! Segment merging (Lucene's `SegmentMerger`).
//!
//! Merges several immutable segments into one new segment: deleted docs are
//! dropped, doc ids are renumbered contiguously (segment order, then doc
//! order), and the term dictionary, postings, fast fields and stored docs
//! are rewritten. The old segment files are left in place; the caller swaps
//! the merged segment in and deletes them.
//!
//! How each part is produced:
//!
//! * **Terms + postings**: a streaming k-way merge over the sorted source
//!   dictionaries ([`TermCursor`], no materialization of the inputs) yields
//!   the merged term list with each term's contributing lists; the lists
//!   are then re-encoded in parallel over contiguous runs of terms
//!   (balanced by postings volume). Re-encoding is block-wise: whole
//!   128-doc blocks are bulk-decoded (SIMD), shifted by the segment's doc
//!   base (or remapped through the deletion map) and re-packed. The runs'
//!   buffers are written to the file as-is (no concatenation).
//! * **Stored fields**: LZ4 chunks of segments without deletions are copied
//!   verbatim (straight from the input mapping into the output file); only
//!   chunks containing deleted docs are decompressed, filtered and
//!   recompressed (in parallel).
//! * **Fast fields / norms**: columns rebuilt over the live docs; norms of
//!   segments without deletions are copied with one `memcpy` per segment.
//! * **Vectors** are not merged at all: text segments hold only a
//!   `doc -> PointId` column, which is rewritten with the new doc ids.
//!   Vectors and HNSW graphs stay in the
//!   [`VectorStore`](crate::vector::store::VectorStore).
//!
//! Postings, fast fields, the store and the point map are produced
//! concurrently, and the output file is streamed: the store section is
//! written (from the input mappings) while postings are still encoding,
//! the checksum is computed on the way, and the footer records the
//! sections' offsets. Inputs are first read once in bulk
//! ([`Directory::prefetch`]).

use crate::codec::directory::{Directory, FileSlice};
use crate::codec::fastfields::{ord_width, KeywordReader, MultiKeywordReader, MISSING_ORD, TAG_FEATURES, TAG_KEYWORD, TAG_MULTI_KEYWORD};
use crate::codec::postings::{write_postings_uncached, BlockNorms, PostingsReader, BLOCK_DOCS};
use crate::codec::store::{encode_store_table, StoreReader};
use crate::codec::terms::{TermCursor, TermDictWriter};
use crate::codec::varint::{read_vint, write_vint};
use crate::codec::{
    encode_extra, segment_file_name, SegmentFileWriter, EXTRA_FIELD_STATS, EXTRA_GEO, EXTRA_NESTED, SECTION_EXTRA, SECTION_FAST, SECTION_POSTINGS, SECTION_STORE,
    SECTION_TERMS, SECTION_VECTORS,
};
use crate::error::{Result, RusticError};
use crate::fast_fields::FastFields;
use crate::postings::BitSet;
use crate::segment::{FieldAgg, Segment};
use crate::term_dictionary::PostingsLocation;
use crate::types::DocId;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Marks a deleted doc in a [`DocMap`] row.
const DELETED: DocId = DocId::MAX;

/// The doc-id remapping used by a merge: `remap[seg_idx][old_doc]` is the
/// new id of a live doc, [`DELETED`] for a deleted one. 4 bytes per doc, and
/// no row at all for a segment without deletions (its `base` maps it).
pub struct DocMap {
    remap: Vec<Vec<DocId>>,
    /// `Some(base)` when segment `i` has no deletions: `new = base + old`.
    bases: Vec<Option<DocId>>,
    total: u32,
    /// New ids follow an index sort, not the input order: `new_id` is not
    /// monotonic within a segment (`order[new] = (segment, old)`).
    order: Option<Vec<(usize, DocId)>>,
    // Must release after all mapping/order buffers, including drop-on-error.
    _memory: Option<crate::admission::Reservation>,
}

impl DocMap {
    /// New id for `(seg_idx, old_doc)`, or `None` if the doc is deleted.
    pub fn new_id(&self, seg_idx: usize, old_doc: DocId) -> Option<DocId> {
        match self.bases[seg_idx] {
            Some(base) => Some(base + old_doc),
            None => live(self.remap[seg_idx][old_doc as usize]),
        }
    }

    /// Total number of live docs across all segments.
    pub fn num_docs(&self) -> u32 {
        self.total
    }

    /// The constant offset of segment `seg_idx`'s doc ids, when it has no
    /// deletions (every doc maps to `base + old`).
    pub fn base(&self, seg_idx: usize) -> Option<DocId> {
        self.bases[seg_idx]
    }

    /// Whether new ids follow an index sort (see [`DocMap::from_order`]).
    pub fn is_sorted(&self) -> bool {
        self.order.is_some()
    }

    /// The doc map of a sorting merge: `order[new] = (segment, old)` lists
    /// the docs to keep in their output order (docs not listed are dropped).
    pub fn from_order(segments: &[Arc<Segment>], order: Vec<(usize, DocId)>) -> DocMap {
        let mut remap: Vec<Vec<DocId>> = segments.iter().map(|s| vec![DELETED; s.num_docs as usize]).collect();
        for (new, &(i, old)) in order.iter().enumerate() {
            remap[i][old as usize] = new as DocId;
        }
        DocMap { remap, _memory: None, bases: vec![None; segments.len()], total: order.len() as u32, order: Some(order) }
    }
}

/// Build the doc-id remapping from each segment's deletion bitmap.
///
/// `deletions()` is snapshotted once per segment so the whole merge sees a
/// consistent view even if deletions land concurrently.
pub fn build_doc_map(segments: &[Arc<Segment>]) -> DocMap {
    let dels: Vec<Arc<BitSet>> = segments.iter().map(|s| s.deletions()).collect();
    build_doc_map_snapshot(segments, &dels)
}

/// [`build_doc_map`] over given deletion snapshots (`dels[i]` for
/// `segments[i]`): a background merge and the install that follows it
/// must agree on the doc map although deletes land in between.
pub fn build_doc_map_snapshot(segments: &[Arc<Segment>], dels: &[Arc<BitSet>]) -> DocMap {
    let mut remap = Vec::with_capacity(segments.len());
    let mut bases = Vec::with_capacity(segments.len());
    let mut total = 0u32;
    for (seg, dels) in segments.iter().zip(dels) {
        let base = total;
        let live = (0..seg.num_docs).filter(|&d| !dels.get(d)).count() as u32;
        if live == seg.num_docs {
            bases.push(Some(base));
            remap.push(Vec::new());
            total += live;
            continue;
        }
        let mut row = Vec::with_capacity(seg.num_docs as usize);
        for d in 0..seg.num_docs {
            if !dels.get(d) {
                row.push(total);
                total += 1;
            } else {
                row.push(DELETED);
            }
        }
        bases.push(None);
        remap.push(row);
    }
    DocMap { remap, _memory: None, bases, total, order: None }
}

/// Merge encoders reserve before building maps. The reservation follows the
/// map through every parallel stage and is released on error/unwind as well.
pub(crate) fn admitted_doc_map(segments: &[Arc<Segment>], dels: Option<&[Arc<BitSet>]>, order: Option<Vec<(usize, DocId)>>) -> Result<DocMap> {
    admitted_doc_map_with_budget(segments, dels, order, &crate::admission::shared())
}

fn admitted_doc_map_with_budget(segments: &[Arc<Segment>], dels: Option<&[Arc<BitSet>]>, order: Option<Vec<(usize, DocId)>>, budget: &Arc<crate::admission::Budget>) -> Result<DocMap> {
    let metadata = (segments.len() as u64).checked_mul(128).ok_or_else(|| RusticError::Memory("merge map size overflow".into()))?;
    let mut memory = budget.reserve(metadata, "merge document map metadata")?;
    let snapshots = dels.is_none().then(|| segments.iter().map(|s| s.deletions()).collect::<Vec<_>>());
    let dels = dels.or(snapshots.as_deref()).expect("deletion snapshots");
    if dels.len() != segments.len() { return Err(RusticError::Corrupted("merge deletion snapshot count differs".into())); }
    let mut bytes = 0u64;
    for (segment, deleted) in segments.iter().zip(dels) {
        // A clean unsorted segment uses only its base, no per-document row.
        if order.is_some() || deleted.count() != 0 {
            bytes = bytes.checked_add(u64::from(segment.num_docs) * 4).ok_or_else(|| RusticError::Memory("merge map size overflow".into()))?;
        }
    }
    if let Some(order) = &order {
        bytes = bytes.checked_add((order.capacity() as u64).saturating_mul(std::mem::size_of::<(usize, DocId)>() as u64))
            .ok_or_else(|| RusticError::Memory("merge order size overflow".into()))?;
    }
    memory.grow(bytes, "merge document map rows")?;
    let mut map = match order {
        Some(order) => DocMap::from_order(segments, order),
        None => build_doc_map_snapshot(segments, dels),
    };
    map._memory = Some(memory);
    Ok(map)
}

/// A [`DocMap`] row entry as an id, `None` when deleted.
#[inline]
fn live(new: DocId) -> Option<DocId> {
    (new != DELETED).then_some(new)
}

/// Field id of a term key (`2-byte LE field id + term bytes`).
#[inline]
fn key_field(key: &[u8]) -> u16 {
    u16::from_le_bytes([key[0], key.get(1).copied().unwrap_or(0)])
}

/// Merge `segments` into one new segment with id `new_id`, dropping deleted
/// docs and renumbering doc ids contiguously (segment order, then doc
/// order). Returns the opened merged segment; the caller swaps it in and
/// deletes the old files.
pub fn merge_segments(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32) -> Result<Segment> {
    merge_segments_with(dir, segments, new_id, None)
}

/// [`merge_segments`] with an explicit output order (an index sort, see
/// [`DocMap::from_order`]); `None`: segment order, then doc order.
pub fn merge_segments_with(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, order: Option<Vec<(usize, DocId)>>) -> Result<Segment> {
    // Merges run under an index's write lock: encode on the dedicated pool
    // (see `crate::encode_pool`).
    // Merges encode on the merge pool (see `crate::merge_pool`).
    crate::on_pool(crate::merge_pool(), || merge_segments_in_pool(dir, segments, new_id, order, true))
}

/// [`merge_segments_with`] for a seal (an index-sorted flush / refresh
/// rewriting a fresh segment in sort order): never throttled and not counted
/// by the adaptive merge throttle (ES rate-limits only merges, never
/// flush / refresh writes).
pub(crate) fn seal_segments_with(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, order: Option<Vec<(usize, DocId)>>) -> Result<Segment> {
    crate::on_encode_pool(|| merge_segments_in_pool(dir, segments, new_id, order, false))
}

/// Merge with the docs deleted in `dels` (one deletion snapshot per
/// input, see [`build_doc_map_snapshot`]) dropped, in input order.
pub fn merge_segments_snapshot(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, dels: &[Arc<BitSet>]) -> Result<Segment> {
    let doc_map = admitted_doc_map(segments, Some(dels), None)?;
    crate::on_pool(crate::merge_pool(), || merge_segments_mapped(dir, segments, new_id, doc_map, true, true, MergeAdmission::Background(std::time::Duration::ZERO)))
}

/// Encode a background merge without publishing vector ownership. The index
/// validates the inputs and carries over deletes before attaching the result.
/// `throttled`: a merge (rate-limited, feeds the adaptive throttle, and
/// admitted as background work that waits for room); `false`: a seal's sort
/// or coalescing rewrite (never throttled, and admitted like any other
/// foreground write: a seal under [`crate::admission::forced`] replay is
/// charged, never refused).
pub(crate) fn merge_segments_snapshot_detached(pool: &rayon::ThreadPool, dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, dels: &[Arc<BitSet>], order: Option<Vec<(usize, DocId)>>, throttled: bool) -> Result<Segment> {
    let doc_map = admitted_doc_map(segments, Some(dels), order)?;
    let admit = if throttled { MergeAdmission::Background(MERGE_ADMISSION_WAIT) } else { MergeAdmission::Foreground };
    crate::on_pool(pool, || merge_segments_mapped(dir, segments, new_id, doc_map, false, throttled, admit))
}

fn merge_segments_in_pool(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, order: Option<Vec<(usize, DocId)>>, throttled: bool) -> Result<Segment> {
    let doc_map = admitted_doc_map(segments, None, order)?;
    // A seal (`throttled == false`) is foreground work, a merge background.
    let admit = if throttled { MergeAdmission::Background(std::time::Duration::ZERO) } else { MergeAdmission::Foreground };
    merge_segments_mapped(dir, segments, new_id, doc_map, true, throttled, admit)
}

/// How a merge's encoders admit their working memory.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MergeAdmission {
    /// A merge: [`crate::admission::Budget::reserve_background`], waiting up
    /// to this long for room ([`MERGE_ADMISSION_WAIT`] for detached merges,
    /// zero under an index lock).
    Background(std::time::Duration),
    /// A seal's rewrite: ordinary [`crate::admission::Budget::reserve`] (no
    /// background share cap, no wait; forced under replay).
    Foreground,
}
impl MergeAdmission {
    fn reserve(self, budget: &Arc<crate::admission::Budget>, bytes: u64, label: &str) -> Result<crate::admission::Reservation> {
        match self {
            Self::Background(wait) => budget.reserve_background(bytes, label, wait),
            Self::Foreground => budget.reserve(bytes, label),
        }
    }
}

/// Merge write throttling, adaptive like ES (Lucene
/// `ConcurrentMergeScheduler` auto I/O throttle): merged segment bytes are
/// written at most at the current node-wide rate. The rate starts at
/// 20 MiB/s; each merge that starts with fewer than 2 merges active
/// (running or wanted) lowers it by 10%, with more than twice the
/// processors active raises it by 20%, else holds it ([`begin_merge_io`],
/// ES 9 `ThreadPoolMergeExecutorService`), and the
/// once-a-second [`check_merge_backlog`] raises it while the backlog holds
/// and halves it under search load once the backlog shrinks, within
/// 5 MiB/s .. 10 GiB/s. As in Lucene, only merges of at least
/// [`MIN_BIG_MERGE_BYTES`] are limited (and move the rate) and force merges
/// are not limited; merges while no indexing happened for
/// [`IDLE_INDEXING`] run unthrottled. `RUSTIC_MERGE_MAX_BYTES_PER_SEC` (at
/// startup) or [`set_merge_io_limit`] (runtime) fix the rate instead
/// (`0`: unlimited). Independently, while the node searches, every merge
/// (force merges included) also takes the IO-pressure rate
/// ([`PRESSURE_RATE`]).
static MERGE_IO_LIMIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
/// [`MERGE_IO_LIMIT`] before the environment was read.
const UNSET: u64 = u64::MAX;
/// No fixed limit: adaptive.
const ADAPTIVE: u64 = u64::MAX - 1;
const AUTO_START: u64 = 20 << 20;
const AUTO_MIN: u64 = 5 << 20;
const AUTO_MAX: u64 = 10 << 30;
/// No indexing for this long: merges run unthrottled (ES does not throttle
/// merges on an idle node either: nothing competes for the disk).
pub const IDLE_INDEXING: std::time::Duration = std::time::Duration::from_secs(5);
static AUTO_RATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(AUTO_START);
static ACTIVE_MERGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static LAST_INDEXING: parking_lot::Mutex<Option<std::time::Instant>> = parking_lot::const_mutex(None);

/// Set a fixed merge write rate limit in bytes per second (`0`: unlimited).
pub fn set_merge_io_limit(bytes_per_sec: u64) {
    MERGE_IO_LIMIT.store(bytes_per_sec.min(ADAPTIVE - 1), std::sync::atomic::Ordering::Relaxed);
}

/// Return to the adaptive merge rate.
pub fn set_merge_io_adaptive() {
    MERGE_IO_LIMIT.store(ADAPTIVE, std::sync::atomic::Ordering::Relaxed);
}

/// Indexing happened now (the node is not idle for merge throttling).
pub fn note_indexing() {
    *LAST_INDEXING.lock() = Some(std::time::Instant::now());
}

fn indexing_idle() -> bool {
    LAST_INDEXING.lock().is_none_or(|last| last.elapsed() >= IDLE_INDEXING)
}

fn fixed_limit() -> u64 {
    let current = MERGE_IO_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
    if current != UNSET {
        return current;
    }
    let initial = std::env::var("RUSTIC_MERGE_MAX_BYTES_PER_SEC").ok().and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(ADAPTIVE, |v| v.min(ADAPTIVE - 1));
    let _ = MERGE_IO_LIMIT.compare_exchange(UNSET, initial, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed);
    MERGE_IO_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// The merge write rate limit now, in bytes per second (`0`: unlimited):
/// the fixed limit if one is set, else unlimited while indexing is idle or
/// nothing searches, else the adaptive rate.
pub fn merge_io_limit() -> u64 {
    match fixed_limit() {
        // ES's auto-throttle: while the node indexes, big merges write at
        // the adaptive rate (from 20 MB/s, moved by the merges active, see
        // [`es_next_rate`]), searches or not, so a bulk load defers most of
        // its merging rather than rewriting its segments at full speed
        // beside the indexing (an eql load ran ~30x ES's merges during the
        // bulk). Idle indexing: unlimited, as on an idle ES node. While the
        // node searches the IO-pressure rate bounds it further.
        ADAPTIVE if indexing_idle() => 0,
        ADAPTIVE => AUTO_RATE.load(std::sync::atomic::Ordering::Relaxed),
        fixed => fixed,
    }
}

/// Whether the adaptive rate is at its maximum (ES
/// `ThreadPoolMergeExecutorService.usingMaxTargetIORateBytesPerSec`): ES
/// throttles indexing behind a merge backlog only once its merges' IO is
/// no longer throttled.
pub fn merge_auto_rate_at_max() -> bool {
    AUTO_RATE.load(std::sync::atomic::Ordering::Relaxed) >= AUTO_MAX
}

/// The adaptive rate (ES `total_auto_throttle_in_bytes`), whether or not it
/// applies right now.
pub fn merge_auto_throttle_rate() -> u64 {
    AUTO_RATE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Merges the copies' policies want that have not started (their merge
/// slots are busy): the node is merge-backlogged while any is pending
/// (Lucene `ConcurrentMergeScheduler`: more merges than its threads).
static PENDING_MERGES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// A copy's pending merges went from `old` to `new` (see [`PENDING_MERGES`]).
pub fn note_pending_merges(old: usize, new: usize) {
    PENDING_MERGES.fetch_add(new as i64 - old as i64, std::sync::atomic::Ordering::AcqRel);
}

fn merges_backlogged() -> bool {
    PENDING_MERGES.load(std::sync::atomic::Ordering::Acquire) > 0
}

/// [`PENDING_MERGES`] at the last backlog check.
static LAST_PENDING: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// A periodic backlog check (about once a second): while merges are
/// pending and the backlog is not shrinking under search load, the rate
/// rises 20% per check, so a backlog reaches hundreds of MB/s within
/// seconds even when few merges start. While the node searches and the
/// backlog is gone or shrinking, the rate halves per check (merges yield
/// the disk and CPU to searches quickly).
pub fn check_merge_backlog() {
    let pending = PENDING_MERGES.load(std::sync::atomic::Ordering::Acquire);
    let last = LAST_PENDING.swap(pending, std::sync::atomic::Ordering::AcqRel);
    let rate = next_checked_rate(AUTO_RATE.load(std::sync::atomic::Ordering::Relaxed), pending, last, crate::searches_active());
    AUTO_RATE.store(rate, std::sync::atomic::Ordering::Relaxed);
}

/// [`check_merge_backlog`]'s controller (pure).
fn next_checked_rate(rate: u64, pending: i64, last: i64, searching: bool) -> u64 {
    if searching && (pending <= 0 || pending < last) {
        return (rate / 2).clamp(AUTO_MIN, AUTO_MAX);
    }
    // Without searches the rate moves only as ES 9 moves it, per merge
    // started ([`es_next_rate`]): a backlog during a bulk load is ES's
    // deferred merging, not a reason to merge at full speed.
    if pending > 0 && searching { next_auto_rate(rate, true) } else { rate }
}

/// Lucene `ConcurrentMergeScheduler.MIN_BIG_MERGE_MB`: merges whose
/// inputs are smaller are not rate limited (they still count as active
/// merges for the rate, [`begin_merge_io`]): throttling them held the flood
/// of small merges a bulk load makes to the rate.
pub const MIN_BIG_MERGE_BYTES: u64 = 50 << 20;

/// A running merge ([`begin_merge_io`]).
struct ActiveMerge;
impl Drop for ActiveMerge {
    fn drop(&mut self) {
        ACTIVE_MERGES.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        pace_release();
    }
}

/// ES 9's `ThreadPoolMergeExecutorService` limits for the adaptive rate:
/// with fewer than this many IO-throttled merges active (running or
/// waiting), each new one lowers the rate by 10%.
const THROTTLE_FLOOR_MERGES: i64 = 2;

/// The count past which each new merge raises the rate by 20%: twice the
/// merge pool's threads (ES: twice `allocatedProcessors`, the merge pool's
/// maximum).
fn throttle_ceil_merges() -> i64 {
    static N: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    *N.get_or_init(|| 2 * std::thread::available_parallelism().map_or(2, |n| n.get()) as i64)
}

/// The rate after a merge starts with `active` merges active (running,
/// this one included, or wanted by the policies and not started), as ES 9
/// adapts it on each submitted merge (`newTargetIORateBytesPerSec`): down
/// 10% below [`THROTTLE_FLOOR_MERGES`], up 20% past
/// [`throttle_ceil_merges`], else held.
fn es_next_rate(rate: u64, active: i64, ceil: i64) -> u64 {
    if active < THROTTLE_FLOOR_MERGES {
        next_auto_rate(rate, false)
    } else if active > ceil {
        next_auto_rate(rate, true)
    } else {
        rate
    }
}

/// A merge of `bytes` input bytes starts (`None`: a vector rebuild, which
/// never moves the rate): the rate adapts to the merges active, as ES 9
/// does ([`es_next_rate`]). Lucene's `ConcurrentMergeScheduler` rule
/// (ES's scheduler before 9) lowered it whenever a merge started without
/// a similar-sized one running, so in a bulk load, with merges always
/// active but rarely of similar sizes, the rate kept falling to its floor
/// and large merges ran at a few MB/s.
fn begin_merge_io(bytes: Option<u64>) -> ActiveMerge {
    let running = ACTIVE_MERGES.fetch_add(1, std::sync::atomic::Ordering::AcqRel) as i64 + 1;
    if bytes.is_some() {
        let active = running + PENDING_MERGES.load(std::sync::atomic::Ordering::Acquire).max(0);
        let ceil = throttle_ceil_merges();
        let _ = AUTO_RATE.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |rate| Some(es_next_rate(rate, active, ceil)));
    }
    ActiveMerge
}

/// The adaptive throttle's controller (pure; the node-wide rate is a thin
/// wrapper around it): a merge starting behind a backlog raises the rate by
/// 20%, one starting without lowers it by 10%, within the bounds.
fn next_auto_rate(rate: u64, backlogged: bool) -> u64 {
    let next = if backlogged { rate.saturating_add(rate / 5) } else { rate.saturating_sub(rate / 11) };
    next.clamp(AUTO_MIN, AUTO_MAX)
}

/// Search pressure on running background merges (Lucene
/// `MergeRateLimiter` under the auto IO throttle, here driven by the search
/// pool): a write rate in bytes per second, stepped every
/// [`SEARCH_RATE_STEP_MS`] from a smoothed search queue depth. It halves
/// only in a genuine search burst (a sustained queue, see
/// [`next_search_state`]), doubles back as soon as the queue is about
/// empty, between [`SEARCH_RATE_FLOOR`] (Lucene's minimum: a merge never
/// stalls under constant search load) and [`SEARCH_RATE_CEILING`] (at the
/// ceiling it adds no limit). Steady search load, whose queue waits are
/// momentary, never throttles merges.
static SEARCH_RATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(SEARCH_RATE_CEILING);
/// The smoothed queue depth, in thousandths.
static SEARCH_QUEUE_AVG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SEARCH_RATE_AT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const SEARCH_RATE_FLOOR: u64 = 5 << 20;
const SEARCH_RATE_CEILING: u64 = 1 << 30;
const SEARCH_RATE_STEP_MS: u64 = 20;
/// Smoothed queue depth from which merges yield; with a merge backlog
/// (merges falling behind, which Lucene's auto throttle answers by raising
/// the rate) only a larger burst holds them back.
const SEARCH_BURST: u64 = 8_000;
const SEARCH_BURST_BACKLOGGED: u64 = 30_000;

/// [`SEARCH_RATE`]'s controller (pure): `(rate, smoothed depth in
/// thousandths)` after one step that sampled `queued` searches waiting
/// (`backlogged`: merges pending). The depth is an exponential average
/// (weight 1/4 per step, about 80 ms), so momentary queue waits average
/// out; the rate halves while it is a burst, doubles while it is below one,
/// and holds in between.
fn next_search_state(rate: u64, avg: u64, queued: usize, backlogged: bool) -> (u64, u64) {
    let avg = (avg * 3 + (queued as u64).saturating_mul(1000)) / 4;
    let burst = if backlogged { SEARCH_BURST_BACKLOGGED } else { SEARCH_BURST };
    let next = if avg >= burst {
        rate / 2
    } else if avg < 1000 {
        rate.saturating_mul(2)
    } else {
        rate
    };
    (next.clamp(SEARCH_RATE_FLOOR, SEARCH_RATE_CEILING), avg)
}

/// The search-pressure rate now (stepped at most every
/// [`SEARCH_RATE_STEP_MS`] by whichever merge write gets there first: two
/// atomic loads otherwise); `None` at the ceiling (no limit).
fn search_rate() -> Option<u64> {
    use std::sync::atomic::Ordering::Relaxed;
    let now = crate::now_ms();
    let at = SEARCH_RATE_AT.load(Relaxed);
    if now.saturating_sub(at) >= SEARCH_RATE_STEP_MS && SEARCH_RATE_AT.compare_exchange(at, now, Relaxed, Relaxed).is_ok() {
        let (queued, _active) = crate::search_load();
        let (rate, avg) = next_search_state(SEARCH_RATE.load(Relaxed), SEARCH_QUEUE_AVG.load(Relaxed), queued, merges_backlogged());
        SEARCH_QUEUE_AVG.store(avg, Relaxed);
        SEARCH_RATE.store(rate, Relaxed);
    }
    let r = SEARCH_RATE.load(Relaxed);
    (r < SEARCH_RATE_CEILING).then_some(r)
}

/// IO pressure on merges while the node searches: a per-merge write rate in
/// bytes per second, stepped every [`PRESSURE_STEP_MS`] from the share of
/// time some task stalled on IO (Linux PSI `/proc/pressure/io`, `some`),
/// smoothed. It applies to every throttled merge write (small, big and
/// force merges, vector rebuilds) and only while searches ran within the
/// last second ([`crate::searches_active`]): without search load it is off,
/// so bulk loads and force merges with nobody searching run at full speed.
///
/// The search-queue rate ([`search_rate`]) misses this case: a search
/// waiting on a page fault behind merge IO is running, not queued. With a
/// 1-replica wiki index larger than the page cache, big and unthrottled
/// force merges saturated the disk and search p99 rose 2-5x while the
/// search queue stayed short.
///
/// The controller ([`next_pressure_state`]) is AIMD: a sustained stall share
/// of [`PRESSURE_HIGH_PPM`] engages at [`PRESSURE_ENGAGE`] and halves per
/// step down to [`PRESSURE_FLOOR`] (a merge never stalls); below
/// [`PRESSURE_LOW_PPM`] it grows 50% per step back to the ceiling (no limit).
/// `RUSTIC_MERGE_IO_PRESSURE` (at startup): `off`, or `HIGH,LOW` in parts
/// per thousand (default `100,30`).
static PRESSURE_RATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(PRESSURE_CEILING);
/// The smoothed stall share, in parts per million.
static PRESSURE_SHARE_AVG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PRESSURE_AT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// PSI `some` total (microseconds) at the last step.
static PRESSURE_STALL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const PRESSURE_STEP_MS: u64 = 100;
const PRESSURE_HIGH_PPM: u64 = 100_000;
const PRESSURE_LOW_PPM: u64 = 30_000;
const PRESSURE_ENGAGE: u64 = 64 << 20;
const PRESSURE_FLOOR: u64 = 8 << 20;
const PRESSURE_CEILING: u64 = 1 << 30;

/// `(high, low)` stall-share thresholds in parts per million, or `None`
/// when `RUSTIC_MERGE_IO_PRESSURE=off` (read once).
fn pressure_thresholds() -> Option<(u64, u64)> {
    static T: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *T.get_or_init(|| parse_pressure_setting(std::env::var("RUSTIC_MERGE_IO_PRESSURE").ok().as_deref()))
}

/// [`pressure_thresholds`] from the setting's value (pure).
fn parse_pressure_setting(value: Option<&str>) -> Option<(u64, u64)> {
    let default = Some((PRESSURE_HIGH_PPM, PRESSURE_LOW_PPM));
    let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) else { return default };
    if v.eq_ignore_ascii_case("off") || v == "0" {
        return None;
    }
    let mut parts = v.split(',').map(|p| p.trim().parse::<u64>().ok());
    match (parts.next().flatten(), parts.next().flatten(), parts.next()) {
        (Some(high), Some(low), None) if low < high && high <= 1000 => Some((high * 1000, low * 1000)),
        _ => default,
    }
}

/// The PSI `some` total (microseconds) from `/proc/pressure/io` text (pure).
fn psi_some_total_us(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("some "))?;
    line.split_whitespace().find_map(|f| f.strip_prefix("total=")).and_then(|t| t.parse().ok())
}

/// The node's cumulative IO stall (PSI `some`, microseconds), `None` where
/// the kernel does not report it (not Linux, PSI disabled); a failed read
/// is not retried.
#[cfg(target_os = "linux")]
fn io_stall_total_us() -> Option<u64> {
    static AVAILABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
    if !AVAILABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let total = std::fs::read_to_string("/proc/pressure/io").ok().as_deref().and_then(psi_some_total_us);
    if total.is_none() {
        AVAILABLE.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    total
}

#[cfg(not(target_os = "linux"))]
fn io_stall_total_us() -> Option<u64> {
    None
}

/// [`PRESSURE_RATE`]'s controller (pure): `(rate, smoothed share in ppm)`
/// after one step that measured a stall share of `share` ppm, with
/// thresholds `(high, low)`. The share is an exponential average (weight
/// 1/4 per step, about 400 ms). Not searching: no limit.
fn next_pressure_state(rate: u64, avg: u64, share: u64, searching: bool, (high, low): (u64, u64)) -> (u64, u64) {
    let avg = (avg * 3 + share.min(1_000_000)) / 4;
    if !searching {
        return (PRESSURE_CEILING, avg);
    }
    let next = if avg >= high {
        if rate >= PRESSURE_CEILING { PRESSURE_ENGAGE } else { rate / 2 }
    } else if avg < low {
        rate.saturating_add(rate / 2)
    } else {
        rate
    };
    (next.clamp(PRESSURE_FLOOR, PRESSURE_CEILING), avg)
}

/// The IO-pressure rate now (stepped at most every [`PRESSURE_STEP_MS`] by
/// whichever merge write gets there first); `None`: no limit.
fn pressure_rate() -> Option<u64> {
    use std::sync::atomic::Ordering::Relaxed;
    let thresholds = pressure_thresholds()?;
    let now = crate::now_ms();
    let at = PRESSURE_AT.load(Relaxed);
    if now.saturating_sub(at) >= PRESSURE_STEP_MS && PRESSURE_AT.compare_exchange(at, now, Relaxed, Relaxed).is_ok() {
        if let Some(total) = io_stall_total_us() {
            let last = PRESSURE_STALL_US.swap(total, Relaxed);
            // Microseconds stalled per millisecond elapsed = ppm x 1/1000.
            let share = if at == 0 || last == 0 { 0 } else { total.saturating_sub(last).saturating_mul(1000) / (now - at).max(1) };
            let (rate, avg) = next_pressure_state(PRESSURE_RATE.load(Relaxed), PRESSURE_SHARE_AVG.load(Relaxed), share, crate::searches_active(), thresholds);
            PRESSURE_SHARE_AVG.store(avg, Relaxed);
            PRESSURE_RATE.store(rate, Relaxed);
        }
    }
    let r = PRESSURE_RATE.load(Relaxed);
    (r < PRESSURE_CEILING).then_some(r)
}

/// The IO-pressure merge rate in bytes per second (`0`: no limit), for stats.
pub fn merge_pressure_rate() -> u64 {
    let r = PRESSURE_RATE.load(std::sync::atomic::Ordering::Relaxed);
    if r >= PRESSURE_CEILING { 0 } else { r }
}

/// `limit` (`0`: unlimited) further bounded by `pressure` (pure).
fn with_pressure(limit: u64, pressure: Option<u64>) -> u64 {
    match pressure {
        Some(p) if limit == 0 || p < limit => p,
        _ => limit,
    }
}

/// The rate a merge write runs at now (`0`: unlimited): [`merge_io_limit`],
/// and for a background merge (`search_aware`) also the search-pressure
/// rate, whichever is lower.
fn write_limit(search_aware: bool) -> u64 {
    let base = merge_io_limit();
    match search_aware.then(search_rate).flatten() {
        Some(s) if base == 0 || s < base => s,
        _ => base,
    }
}

/// CPU pacing of background merges in a search burst (the write throttle
/// paces only their output, not the encoding): merge-pool threads call
/// [`merge_pace`] at cheap points (postings per term, stored-field chunks,
/// columns, point maps). While the smoothed search queue is a burst
/// ([`SEARCH_BURST`], backlog or not) one thread at a time runs, sleeping
/// in proportion to the search-pressure rate (down to a quarter of its
/// time), and the others park at their next check, each going on for one
/// slice per [`PACE_PARK_MAX`] so merges still progress. Force merges (own
/// pool), seals (encode pool) and merges with the auto throttle off are
/// never paced.
static PACE_RUNNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Work units (postings, docs, values) between pacing checks.
const PACE_UNITS: u64 = 1 << 16;
/// A runner not seen for this long gives up its turn.
const PACE_STALE_MS: u64 = 100;
const PACE_PARK_MAX: std::time::Duration = std::time::Duration::from_millis(250);

thread_local! {
    // Captured explicitly at every parallel task boundary: Rayon workers may
    // execute tasks from several indexes with different throttle settings.
    static PACING_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// `(work since the last check, when this thread last resumed)`.
    static PACE: std::cell::Cell<(u64, Option<std::time::Instant>)> = const { std::cell::Cell::new((0, None)) };
}

/// A task-local policy, restored even when an encoder unwinds. Never disable
/// another index's pacing merely because an unthrottled merge is running.
fn with_pacing<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool, Option<(u64, Option<std::time::Instant>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(progress) = self.1 {
                pace_release();
                PACING_ENABLED.with(|p| p.set(self.0));
                PACE.with(|p| p.set(progress));
            }
        }
    }
    let old = PACING_ENABLED.with(|p| p.replace(enabled));
    // Same-policy child tasks must retain accumulated work: resetting every
    // small posting run could otherwise prevent reaching PACE_UNITS at all.
    let progress = (old != enabled).then(|| PACE.with(|p| p.replace((0, None))));
    let _restore = Restore(old, progress);
    f()
}

/// Rayon does not inherit thread locals when a child is stolen. Capture the
/// submitting task's policy and establish it on both child execution threads.
fn paced_join<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    let enabled = PACING_ENABLED.with(|p| p.get());
    let admission = crate::admission::Context::capture();
    rayon::join(|| admission.run(|| with_pacing(enabled, a)), || admission.run(|| with_pacing(enabled, b)))
}

/// `units` of merge work done on this thread (see [`PACE_RUNNER`]).
#[inline]
pub(crate) fn merge_pace(units: u64) {
    PACE.with(|p| {
        let (work, at) = p.get();
        let work = work + units;
        if work < PACE_UNITS {
            p.set((work, at));
            return;
        }
        p.set((0, at));
        pace_check(p);
    });
}

/// Whether the search queue is a burst merges yield their CPU to.
fn search_burst() -> bool {
    let _ = search_rate();
    SEARCH_QUEUE_AVG.load(std::sync::atomic::Ordering::Relaxed) >= SEARCH_BURST
}

/// Time merge threads slept for the IO rate limit and for search
/// pressure, node-wide (the per-merge summary reports the change over a
/// merge; concurrent merges share them).
static THROTTLE_SLEEP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PACE_SLEEP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cold]
fn pace_check(p: &std::cell::Cell<(u64, Option<std::time::Instant>)>) {
    pace_check_with(p, search_burst, || SEARCH_RATE.load(std::sync::atomic::Ordering::Relaxed))
}

// Inject only the pressure observations, not the scheduler/parking behavior.
// This lets tests exercise real merge workers without perturbing node-global
// search counters shared by other concurrent tests.
fn pace_check_with(p: &std::cell::Cell<(u64, Option<std::time::Instant>)>,
    searching: impl Fn() -> bool, rate: impl Fn() -> u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(me) = crate::merge_pool().current_thread_index().map(|i| i as u64 + 1) else { return };
    if !PACING_ENABLED.with(|p| p.get()) || !searching() {
        p.set((0, None));
        return;
    }
    let now = std::time::Instant::now();
    let resumed = p.get().1.unwrap_or(now);
    // The turn: `owner << 40 | last seen ms`; free when 0 or stale.
    let take = |cur: u64| -> bool {
        let ms = crate::now_ms();
        let mine = cur >> 40 == me;
        (cur == 0 || mine || ms.saturating_sub(cur & ((1 << 40) - 1)) >= PACE_STALE_MS)
            && PACE_RUNNER.compare_exchange(cur, me << 40 | ms, Relaxed, Relaxed).is_ok()
    };
    if take(PACE_RUNNER.load(Relaxed)) {
        // Sleep `k` times the time worked: k = log2(ceiling / rate), at most 3.
        let rate = rate().max(1);
        let k = (SEARCH_RATE_CEILING / rate).max(1).ilog2().min(3);
        let nap = now.saturating_duration_since(resumed).saturating_mul(k).min(std::time::Duration::from_millis(PACE_STALE_MS / 2));
        if !nap.is_zero() {
            std::thread::sleep(nap);
            PACE_SLEEP_NS.fetch_add(nap.as_nanos() as u64, Relaxed);
        }
    } else {
        let parked = std::time::Instant::now();
        while parked.elapsed() < PACE_PARK_MAX && searching() && !take(PACE_RUNNER.load(Relaxed)) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        PACE_SLEEP_NS.fetch_add(parked.elapsed().as_nanos() as u64, Relaxed);
    }
    p.set((0, Some(std::time::Instant::now())));
}

/// This thread's pacing turn ends (its merge finished).
fn pace_release() {
    let Some(me) = crate::merge_pool().current_thread_index().map(|i| i as u64 + 1) else { return };
    let cur = PACE_RUNNER.load(std::sync::atomic::Ordering::Relaxed);
    if cur >> 40 == me {
        let _ = PACE_RUNNER.compare_exchange(cur, 0, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Rate-limit one merge's output: each merge writes at most at the current
/// rate ([`write_limit`]; ES's auto throttle limits every merge thread to
/// its target rate, ramping it with the backlog, never switching it off),
/// sleeping (on the merge's own thread) when its bucket runs dry, in slices
/// short enough that a merge speeds up again as soon as searches stop
/// queueing. Up to one second of burst at the current rate. Force merges
/// (their own pool) never take the search-queue rate. Every throttled write,
/// force merges included, takes the IO-pressure rate while the node
/// searches ([`pressure_rate`]).
fn throttle_merge_write(bucket: &mut Option<(std::time::Instant, f64)>, bytes: usize, io: bool) -> u64 {
    if bytes == 0 {
        return 0;
    }
    let mut slept = 0u64;
    let search_aware = crate::merge_pool().current_thread_index().is_some();
    // `io`: the IO limit applies (a big merge, a vector rebuild); a small
    // merge takes the search pressure only.
    let limit_now = || {
        let base = if io { write_limit(search_aware) } else { search_aware.then(search_rate).flatten().unwrap_or(0) };
        with_pressure(base, pressure_rate())
    };
    let limit = limit_now();
    if limit == 0 {
        *bucket = None;
        return 0;
    }
    let rate = limit as f64;
    let now = std::time::Instant::now();
    let (last, available) = bucket.get_or_insert((now, rate));
    let refill = now.saturating_duration_since(*last).as_secs_f64() * rate;
    *available = (*available + refill).min(rate) - bytes as f64;
    *last = now;
    let started = now;
    while *available < 0.0 && started.elapsed() < std::time::Duration::from_secs(5) {
        let rate = match limit_now() {
            0 => break,
            r => r as f64,
        };
        let nap = std::time::Duration::from_secs_f64(-*available / rate).min(std::time::Duration::from_millis(50));
        std::thread::sleep(nap);
        THROTTLE_SLEEP_NS.fetch_add(nap.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        slept += nap.as_nanos() as u64;
        let now = std::time::Instant::now();
        *available = (*available + now.saturating_duration_since(*last).as_secs_f64() * rate).min(rate);
        *last = now;
    }
    slept
}

/// Merge output written in slices of this size, so a large field body
/// yields to searches progressively rather than in one long sleep.
const THROTTLE_SLICE: usize = 1 << 20;

/// A vector-store rebuild's output (a DiskBBQ seal or merge, see
/// `VectorStore::rebuild`), throttled like a merge's: the IO limit, and on
/// the merge pool the search-pressure rate.
pub(crate) fn throttled_output<'a>(inner: Box<dyn crate::codec::directory::OutputFile + 'a>) -> Box<dyn crate::codec::directory::OutputFile + 'a> {
    Box::new(ThrottledOutput { inner, throttled: true, io: true, bucket: None, slept: None })
}

/// A vector-store rebuild's CPU work, paced like its writes: `work(bytes)`
/// after each clustering pass charges the merge's token bucket (it sleeps
/// on this thread under the IO limit or search pressure), and the rebuild
/// counts as a running merge for the adaptive rate.
pub(crate) struct MergeWork {
    _active: ActiveMerge,
    bucket: Option<(std::time::Instant, f64)>,
}

impl MergeWork {
    pub(crate) fn begin() -> Self {
        MergeWork { _active: begin_merge_io(None), bucket: None }
    }

    pub(crate) fn work(&mut self, bytes: usize) {
        let mut left = bytes;
        while left > 0 {
            let slice = left.min(THROTTLE_SLICE);
            throttle_merge_write(&mut self.bucket, slice, true);
            left -= slice;
        }
    }
}

/// The merged segment's output, throttled ([`throttle_merge_write`]).
struct ThrottledOutput<'a> {
    inner: Box<dyn crate::codec::directory::OutputFile + 'a>,
    /// `false`: a seal's output (never throttled).
    throttled: bool,
    /// The IO rate limit applies: a big merge ([`MIN_BIG_MERGE_BYTES`]) or
    /// a vector rebuild. A small merge's writes take the search pressure only.
    io: bool,
    /// This merge's token bucket ([`throttle_merge_write`]).
    bucket: Option<(std::time::Instant, f64)>,
    /// This output's own throttle sleeps (ns), for its merge's summary.
    slept: Option<Arc<std::sync::atomic::AtomicU64>>,
}

/// Test probe: a throttled (merge) write ran on an encode-pool thread.
#[cfg(test)]
static THROTTLED_ON_ENCODE_POOL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl crate::codec::directory::OutputFile for ThrottledOutput<'_> {
    fn write_all(&mut self, data: &[u8]) -> Result<()> {
        #[cfg(test)]
        if self.throttled && crate::on_encode_worker() {
            THROTTLED_ON_ENCODE_POOL.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if !self.throttled {
            return self.inner.write_all(data);
        }
        for slice in data.chunks(THROTTLE_SLICE) {
            let slept = throttle_merge_write(&mut self.bucket, slice.len(), self.io);
            if let (Some(acc), true) = (&self.slept, slept > 0) {
                acc.fetch_add(slept, std::sync::atomic::Ordering::Relaxed);
                MERGE_OWN_THROTTLE_NS.fetch_add(slept, std::sync::atomic::Ordering::Relaxed);
            }
            self.inner.write_all(slice)?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        self.inner.finish()
    }
}

#[inline]
fn profile_stage<R>(enabled: bool, stage: &'static str, f: impl FnOnce() -> R) -> R {
    let started = enabled.then(std::time::Instant::now);
    let result = f();
    if let Some(started) = started {
        tracing::debug!(target: "merge_profile", stage, elapsed_us=started.elapsed().as_micros() as u64, "segment merge stage (spans overlap)");
    }
    result
}

/// Run `f`, adding its wall time to `into` (ms), for the merge summary.
#[inline]
fn timed<R>(into: &std::sync::atomic::AtomicU64, f: impl FnOnce() -> R) -> R {
    let t = std::time::Instant::now();
    let r = f();
    into.fetch_add(t.elapsed().as_millis() as u64, std::sync::atomic::Ordering::Relaxed);
    r
}

/// `admit`: how the encoders admit their memory (see [`MergeAdmission`]).
/// Node-wide totals of the merges run (seal rewrites excluded): what a
/// merge microbench or a profile reads to tell how many bytes were merged
/// and where the time went (stage times overlap; see `merge_summary`).
#[derive(Default)]
pub struct MergeTotals {
    pub merges: std::sync::atomic::AtomicU64,
    /// Merges the IO rate limit applied to ([`MIN_BIG_MERGE_BYTES`]).
    pub io_limited: std::sync::atomic::AtomicU64,
    pub input_bytes: std::sync::atomic::AtomicU64,
    pub docs: std::sync::atomic::AtomicU64,
    pub wall_ns: std::sync::atomic::AtomicU64,
    /// norms, postings, store, fast fields, point map, extra, finish.
    pub stage_ms: [std::sync::atomic::AtomicU64; 7],
}

pub static MERGE_TOTALS: MergeTotals = MergeTotals {
    merges: std::sync::atomic::AtomicU64::new(0),
    io_limited: std::sync::atomic::AtomicU64::new(0),
    input_bytes: std::sync::atomic::AtomicU64::new(0),
    docs: std::sync::atomic::AtomicU64::new(0),
    wall_ns: std::sync::atomic::AtomicU64::new(0),
    stage_ms: [const { std::sync::atomic::AtomicU64::new(0) }; 7],
};

/// Throttle sleeps of merges' own output writes (each counted once, by the
/// merge that slept), node-wide, in nanoseconds: ES's
/// `merges.total_throttled_time`.
pub static MERGE_OWN_THROTTLE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Merge throttle and pacing sleeps so far, node-wide, in nanoseconds.
pub fn merge_sleep_ns() -> (u64, u64) {
    (THROTTLE_SLEEP_NS.load(std::sync::atomic::Ordering::Relaxed), PACE_SLEEP_NS.load(std::sync::atomic::Ordering::Relaxed))
}

fn merge_segments_mapped(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, doc_map: DocMap, attach_vectors: bool, throttled: bool, admit: MergeAdmission) -> Result<Segment> {
    // Only merges count toward (and adapt) the merge throttle.
    let input_bytes: u64 = segments.iter().map(|s| s.segment_bytes()).sum();
    let _active = throttled.then(|| begin_merge_io(Some(input_bytes)));
    with_pacing(throttled, || merge_segments_mapped_inner(dir, segments, new_id, doc_map, attach_vectors, throttled, admit))
}

fn merge_segments_mapped_inner(dir: &dyn Directory, segments: &[Arc<Segment>], new_id: u32, doc_map: DocMap, attach_vectors: bool, throttled: bool, admit: MergeAdmission) -> Result<Segment> {
    let profile = tracing::enabled!(target: "merge_profile", tracing::Level::DEBUG);
    // Per-stage wall times (stages overlap) and the node's merge sleeps over
    // this merge, for the one-line summary at the end.
    let started = std::time::Instant::now();
    let (throttle0, pace0) = (THROTTLE_SLEEP_NS.load(std::sync::atomic::Ordering::Relaxed), PACE_SLEEP_NS.load(std::sync::atomic::Ordering::Relaxed));
    let stage_ms: [std::sync::atomic::AtomicU64; 7] = Default::default();
    let [t_norms, t_postings, t_store, t_fast, t_points, t_extra, t_finish] = &stage_ms;
    let input_bytes: u64 = segments.iter().map(|s| s.segment_bytes()).sum();
    // Every input byte is read: warm the inputs with bulk reads first.
    {
        use rayon::prelude::*;
        segments.par_iter().for_each(|s| dir.prefetch(&segment_file_name(s.id)));
    }
    let num_docs = doc_map.num_docs();

    // Norms first: postings record per-block `min_norm` impacts.
    let merged_norms = timed(t_norms, || profile_stage(profile, "norms", || merge_norms(segments, &doc_map)))?;

    // The output is streamed: postings are written a window at a time while
    // the store (mostly chunks borrowed from the inputs), fast fields and
    // point map are built; the footer records where each section landed.
    // One format version for the header and every postings run, whichever
    // worker encodes it.
    let version = if segments.iter().any(|s| !s.shape_fields().is_empty()) { crate::codec::shape_write_version() } else { crate::codec::write_version() };
    // Lucene rate-limits only big merges, and force merges not at all
    // (`forceMergeMBPerSec` is unlimited); they run on their own pool.
    let forced = crate::force_merge_pool().current_thread_index().is_some();
    let io = throttled && input_bytes >= MIN_BIG_MERGE_BYTES && !forced;
    let own_sleep = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut w = SegmentFileWriter::with_version(Box::new(ThrottledOutput { inner: dir.create_output_unsynced(&segment_file_name(new_id))?, throttled, io, bucket: None, slept: Some(own_sleep.clone()) }), version)?;
    let (postings, rest) = paced_join(
        || {
            timed(t_postings, || profile_stage(profile, "postings", || {
                w.begin_section(SECTION_POSTINGS)?;
                // The merged dictionary's blocks spill to a scratch file
                // as they fill; the terms section copies them back.
                static NEXT_TERMS_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let scratch = MergeScratch { dir, name: format!(".merge-terms-{new_id}-{}-{}.tmp", std::process::id(), NEXT_TERMS_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) };
                let mut spill_out = dir.create_scratch_output(&scratch.name)?;
                let mut spilled = 0usize;
                let postings = stream_postings(segments, &doc_map, &merged_norms, POSTINGS_WINDOW_TERMS, version >= crate::codec::IMPACTS_SINCE, version >= crate::codec::INLINE_POSTINGS_SINCE, version >= crate::codec::COMPACT_TERMS_SINCE, version >= crate::codec::INLINE_POSITIONS_SINCE, version >= crate::codec::PACKED_TERMS_SINCE, version >= crate::codec::INT_TERMS_SINCE, admit, |b| w.write(b),
                    Some(&mut |b: &[u8]| { spilled += b.len(); spill_out.write_all(b) }))?;
                spill_out.finish()?;
                let head = if spilled > 0 { Some(dir.open_read(&scratch.name)?) } else { None };
                Ok::<_, crate::error::RusticError>((postings, head, scratch))
            }))
        },
        || -> Result<_> {
            // The `[extra]` entries other than the field statistics (geo
            // indexes, shape trees, nested ordinals: a geo-heavy segment's
            // longest stage) build alongside the postings; the statistics,
            // which need the merged postings' totals, are added after.
            let ((store, (fast, vectors)), extra) = paced_join(
                || paced_join(
                    || timed(t_store, || profile_stage(profile, "store", || if doc_map.is_sorted() { merge_store_sorted_versioned(dir, new_id, segments, &doc_map, version, &crate::admission::shared()) } else { merge_store(segments, &doc_map, version) })),
                    || {
                        paced_join(
                            || timed(t_fast, || profile_stage(profile, "fast_fields", || spool_fast_fields(dir, new_id, segments, &doc_map, &merged_norms, admit, version))),
                            || timed(t_points, || profile_stage(profile, "point_map", || merge_point_map(segments, &doc_map))),
                        )
                    },
                ),
                || timed(t_extra, || profile_stage(profile, "extra", || merge_extra(segments, &doc_map, &merged_norms, &BTreeMap::new(), version))),
            );
            Ok((store?, fast?, vectors?, extra?))
        },
    );
    let (postings, terms_head, _terms_scratch) = postings?;
    let (mut store, fast, vectors, extra) = rest?;
    w.begin_section(SECTION_STORE)?;
    for piece in store.pieces() {
        w.write(piece)?;
    }
    let uids = store.uids.take();
    drop(store);
    w.begin_section(SECTION_FAST)?;
    w.write(&fast.directory)?;
    for chunk in fast.bodies.as_slice().chunks(1024 * 1024) {
        w.write(chunk)?;
    }
    drop(fast);
    w.begin_section(SECTION_VECTORS)?;
    w.write(&vectors)?;
    // The statistics entries, then the other entries' encoded bodies (no
    // copy of the large geo / shape entries): one `[extra]` container.
    let stats = encode_extra(&field_stats_entries(segments, &doc_map, &merged_norms, &postings.field_ttf));
    let uid_entry = uids.map(|(f, col)| encode_extra(&[(crate::codec::EXTRA_UID_COLUMN, f, col)])).unwrap_or_default();
    let count = |e: &[u8]| if e.is_empty() { 0 } else { u32::from_le_bytes(e[..4].try_into().unwrap()) };
    let entries = count(&stats) + count(&extra) + count(&uid_entry);
    if entries > 0 {
        w.begin_section(SECTION_EXTRA)?;
        w.write(&entries.to_le_bytes())?;
        if !stats.is_empty() { w.write(&stats[4..])?; }
        if !extra.is_empty() { w.write(&extra[4..])?; }
        if !uid_entry.is_empty() { w.write(&uid_entry[4..])?; }
    }
    drop(extra);
    w.begin_section(SECTION_TERMS)?;
    if let Some(head) = &terms_head {
        for chunk in head.as_slice().chunks(1024 * 1024) {
            w.write(chunk)?;
        }
    }
    w.write(&postings.terms)?;
    timed(t_finish, || profile_stage(profile, "finish", || w.finish(num_docs)))?;
    {
        use std::sync::atomic::Ordering::Relaxed;
        let ms: Vec<u64> = stage_ms.iter().map(|t| t.load(Relaxed)).collect();
        let wall = started.elapsed().as_millis() as u64;
        // This merge's own throttle sleeps; `node_throttle_sleep_ms` is every
        // merge's over the same time (concurrent merges each count it).
        let throttle = own_sleep.load(Relaxed) / 1_000_000;
        let node_throttle = THROTTLE_SLEEP_NS.load(Relaxed).saturating_sub(throttle0) / 1_000_000;
        let pace = PACE_SLEEP_NS.load(Relaxed).saturating_sub(pace0) / 1_000_000;
        macro_rules! summary { ($lvl:ident) => {
            tracing::$lvl!(target: "merge_summary", segment = new_id, inputs = segments.len(), docs = num_docs, input_bytes, wall_ms = wall,
                norms_ms = ms[0], postings_ms = ms[1], store_ms = ms[2], fast_fields_ms = ms[3], point_map_ms = ms[4], extra_ms = ms[5], finish_ms = ms[6],
                throttle_sleep_ms = throttle, node_throttle_sleep_ms = node_throttle, pace_sleep_ms = pace, throttled, version, "segment merge (stage times overlap; pace and node sleeps are node-wide)")
        } }
        if wall >= 1000 || num_docs >= 1_000_000 { summary!(info) } else { summary!(debug) }
        if throttled {
            let t = &MERGE_TOTALS;
            t.merges.fetch_add(1, Relaxed);
            t.input_bytes.fetch_add(input_bytes, Relaxed);
            t.docs.fetch_add(num_docs as u64, Relaxed);
            t.wall_ns.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
            if io { t.io_limited.fetch_add(1, Relaxed); }
            for (acc, ms) in t.stage_ms.iter().zip(&ms) { acc.fetch_add(*ms, Relaxed); }
        }
    }

    if !attach_vectors {
        return profile_stage(profile, "open", || Segment::open_detached(dir, new_id));
    }
    // Inside one store (an index), attaching moves point ownership to the
    // merged segment: O(docs) table writes, no vector work.
    let vstore = segments.first().and_then(|s| s.vector_store().cloned());
    let same_store = vstore
        .as_ref()
        .is_some_and(|st| segments.iter().all(|s| s.vector_store().is_some_and(|o| Arc::ptr_eq(o, st))));
    profile_stage(profile, "open", || match vstore {
        Some(st) if same_store => Segment::open_with_store(dir, new_id, &st),
        _ => Segment::open(dir, new_id),
    })
}

/// The merged `[extra]` section: nested path ordinals (blocks stay
/// contiguous: segments are concatenated in order and a block's docs are
/// deleted together) and every geo field's points, renumbered.
/// The `[extra]` field-statistics entries of a merge ([`merge_extra`]
/// builds the same ones when given the merged totals).
fn field_stats_entries(segments: &[Arc<Segment>], _doc_map: &DocMap, norms: &MergedNorms, field_ttf: &BTreeMap<u16, u64>) -> Vec<(u8, u16, Vec<u8>)> {
    let mut entries = Vec::new();
    for (&field, &sum_ttf) in field_ttf {
        if !segments.iter().any(|s| s.field_agg(field).is_some()) {
            continue;
        }
        let doc_count = match norms.get(&field) {
            Some((bytes, _)) => bytes.iter().filter(|&&b| b != 0).count() as u64,
            None => segments.iter().filter_map(|s| s.live_field_doc_count(field)).sum(),
        };
        if doc_count > 0 {
            entries.push((EXTRA_FIELD_STATS, field, FieldAgg { doc_count, sum_ttf }.encode()));
        }
    }
    entries
}

fn merge_extra(segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, field_ttf: &BTreeMap<u16, u64>, version: u32) -> Result<crate::admission::Allocation<Vec<u8>>> {
    merge_extra_with_budget(segments, doc_map, norms, field_ttf, version, &crate::admission::shared())
}

fn merge_extra_with_budget(segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, field_ttf: &BTreeMap<u16, u64>, version: u32, budget: &Arc<crate::admission::Budget>) -> Result<crate::admission::Allocation<Vec<u8>>> {
    // Includes nested ord/path collectors, geo point/tree builders and both
    // per-entry encodings and the final combined output, including growth overlap.
    let mut peak = (doc_map.num_docs() as u64).saturating_mul(64).saturating_add((field_ttf.len() as u64).saturating_mul(2048)).saturating_add(64 << 10);
    for segment in segments {
        if let Some(nested) = segment.nested_info() {
            for (path, _) in nested.paths() {
                peak = peak.saturating_add((path.len() as u64).saturating_mul(16).saturating_add(512));
            }
        }
        for field in segment.geo_field_ids() {
            let count = segment.geo_index(field).map_or(0, |g| g.values().len());
            peak = peak.saturating_add((count as u64).saturating_mul(512)).saturating_add(2048);
        }
        // Shape doc values: the doc map (16 bytes a doc) and the copy; the
        // encoded tree parts (their input size again).
        for field in segment.shape_fields() {
            if let Some(d) = segment.shape_data(field) {
                peak = peak.saturating_add(d.tree_bytes() as u64).saturating_add(d.value_bytes() as u64).saturating_add(segment.num_docs as u64 * 16);
            }
        }
    }
    // Shape trees: one part's triangles and docs (32 bytes each) at a time,
    // and its sort and encoding.
    let all_tris: u64 = segments.iter().flat_map(|s| s.shape_fields().into_iter().filter_map(move |f| s.shape_data(f))).map(|d| d.tree.iter().map(|p| p.size()).sum::<u64>()).sum();
    peak = peak.saturating_add(all_tris.min(MERGE_PART_TRIANGLES as u64).saturating_mul(32 + 24));
    let mut memory = budget.reserve(peak,"merge nested and geo workspace")?;
    let mut entries: Vec<(u8, u16, Vec<u8>)> = Vec::new();
    // Collection statistics of the surviving docs: `sumTotalTermFreq` from
    // the merged postings, `docCount` from the merged norms (fields with
    // norms) or the sources' live docs.
    for (&field, &sum_ttf) in field_ttf {
        if !segments.iter().any(|s| s.field_agg(field).is_some()) {
            continue;
        }
        let doc_count = match norms.get(&field) {
            Some((bytes, _)) => bytes.iter().filter(|&&b| b != 0).count() as u64,
            None => segments.iter().filter_map(|s| s.live_field_doc_count(field)).sum(),
        };
        if doc_count > 0 {
            entries.push((EXTRA_FIELD_STATS, field, FieldAgg { doc_count, sum_ttf }.encode()));
        }
    }
    // Column-stored fields: docs any input stored without them keep
    // coming back from the merged columns.
    if version >= crate::codec::COLUMN_STORED_SINCE {
        let mut fields: Vec<u16> = segments.iter().flat_map(|s| crate::segment::column_stored_fields(s).iter().copied()).collect();
        fields.sort_unstable();
        fields.dedup();
        for f in fields {
            entries.push((crate::codec::EXTRA_COLUMN_STORED, f, Vec::new()));
        }
    }
    if version >= crate::codec::KEYWORD_FROM_POSTINGS_SINCE {
        for f in keyword_from_postings_union(segments) {
            entries.push((crate::codec::EXTRA_KEYWORD_FROM_POSTINGS, f, Vec::new()));
        }
    }
    // A synthetic id field stays derived (its inputs hold no postings or
    // stored values of it).
    if let Some((field, sid)) = segments.iter().find_map(|s| crate::segment::synthetic_id_field(s)) {
        entries.push((crate::codec::EXTRA_SYNTHETIC_ID, field, crate::synthetic_id::encode_extra(&sid)));
    }
    let parts = segments.iter().enumerate().map(|(i, s)| {
        let f: Box<dyn Fn(DocId) -> Option<DocId> + '_> = Box::new(move |old| doc_map.new_id(i, old));
        (s.nested_info(), s.num_docs, f)
    });
    // A sorting merge moves nested blocks as units, so the per-doc path
    // ordinals stay consistent through any doc map.
    if let Some(n) = crate::nested::NestedInfo::merge(parts, doc_map.num_docs()) {
        let mut b = Vec::new();
        n.encode(&mut b);
        entries.push((EXTRA_NESTED, 0, b));
    }
    let mut fields: Vec<u16> = segments.iter().flat_map(|s| s.geo_fields()).collect();
    fields.sort_unstable();
    fields.dedup();
    for field in fields {
        let mut pts = Vec::new();
        for (i, s) in segments.iter().enumerate() {
            if let Some(g) = s.geo_index(field) {
                for p in g.values().iter() {
                    if let Some(new) = doc_map.new_id(i, p.doc) {
                        pts.push(crate::geo::GeoPointEntry { doc: new, ..p });
                    }
                }
            }
        }
        if !pts.is_empty() {
            pts.sort_by_key(|p| p.doc);
            let mut b = Vec::new();
            let index = crate::geo::GeoIndex::build(pts);
            if version >= crate::codec::GEO_COMPACT_SINCE {
                index.encode_compact(&mut b);
                entries.push((crate::codec::EXTRA_GEO2, field, b));
            } else {
                index.encode(&mut b);
                entries.push((EXTRA_GEO, field, b));
            }
        }
    }
    merge_shapes(segments, doc_map, version, &mut entries)?;
    let bytes = encode_extra(&entries);
    drop(entries);
    memory.shrink(memory.bytes().saturating_sub(bytes.capacity() as u64));
    Ok(crate::admission::Allocation::new(bytes,memory))
}

/// The `geo_shape` structures of the merged segment: every surviving doc's
/// triangles (renumbered) in a new tree, and its doc value copied as is
/// (a doc value does not depend on the doc's number). Only at segment
/// format v25 or later: older targets refuse shape data.
/// Triangles per rebuilt tree part in a merge (1 GiB of triangles).
const MERGE_PART_TRIANGLES: usize = 32 << 20;

/// The pool that rebuilds a merge's triangle trees: half the cores
/// (`RUSTIC_SHAPE_MERGE_THREADS` overrides). A tree rebuild is a sort of
/// every triangle; on a force merge's own pool (an eighth of the cores) it
/// dominated the merge.
fn shape_merge_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let n = std::env::var("RUSTIC_SHAPE_MERGE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or((cores / 2).max(1));
        rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("shape-merge-{i}")).build().expect("shape merge pool")
    })
}

fn merge_shapes(segments: &[Arc<Segment>], doc_map: &DocMap, version: u32, entries: &mut Vec<(u8, u16, Vec<u8>)>) -> Result<()> {
    if segments.iter().all(|s| s.shape_fields().is_empty()) {
        return Ok(());
    }
    shape_merge_pool().install(|| merge_shapes_in_pool(segments, doc_map, version, entries))
}

fn merge_shapes_in_pool(segments: &[Arc<Segment>], doc_map: &DocMap, version: u32, entries: &mut Vec<(u8, u16, Vec<u8>)>) -> Result<()> {
    use crate::fieldtypes::shape_bkd::{build_parts_with, Entry};
    let mut fields: Vec<u16> = segments.iter().flat_map(|s| s.shape_fields()).collect();
    fields.sort_unstable();
    fields.dedup();
    if !fields.is_empty() && version < crate::codec::SHAPE_TREE_SINCE {
        return Err(crate::error::RusticError::Unsupported(format!("geo_shape fields need segment format v{} (merging into v{version})", crate::codec::SHAPE_TREE_SINCE)));
    }
    let num_docs = doc_map.num_docs();
    for field in fields {
        // The tree is rebuilt in parts of at most `MERGE_PART_TRIANGLES`
        // (a doc's triangles may span parts: the walks combine docs over
        // every part), so a merge holds one part's triangles at a time.
        let total: u64 = segments.iter().filter_map(|s| s.shape_data(field)).map(|d| d.tree.iter().map(|p| p.size()).sum::<u64>()).sum();
        let mut tris: Vec<Entry> = Vec::with_capacity(total.min(MERGE_PART_TRIANGLES as u64) as usize);
        let mut values: Vec<Option<(usize, DocId)>> = Vec::new();
        // Inputs of one triangle a doc merge into parts of one triangle a
        // doc (a doc's triangles may otherwise split across parts).
        let single = segments.iter().filter_map(|s| s.shape_data(field)).all(|d| d.tree.iter().all(|p| p.one_triangle_per_doc()));
        for (i, s) in segments.iter().enumerate() {
            let Some(d) = s.shape_data(field) else { continue };
            // Leaves decode and renumber in parallel, a batch at a time.
            use rayon::prelude::*;
            const LEAF_BATCH: u32 = 4096;
            for p in &d.tree {
                let mut from = 0;
                while from < p.leaf_count() {
                    let to = (from + LEAF_BATCH).min(p.leaf_count());
                    let batch: Vec<Vec<Entry>> = (from..to)
                        .into_par_iter()
                        .map_init(crate::fieldtypes::shape_bkd::LeafBuf::default, |buf, leaf| {
                            p.read_leaf(leaf, buf);
                            buf.docs.iter().zip(&buf.tris).filter_map(|(&old, t)| doc_map.new_id(i, old).map(|new| Entry { tri: *t, doc: new })).collect()
                        })
                        .collect();
                    from = to;
                    for b in batch {
                        tris.extend_from_slice(&b);
                        if tris.len() >= MERGE_PART_TRIANGLES {
                            for part in build_parts_with(&mut tris, single) {
                                entries.push((crate::codec::EXTRA_SHAPE_TREE, field, part));
                            }
                            tris.clear();
                        }
                    }
                }
            }
            if let Some(col) = &d.values {
                if values.is_empty() {
                    values = vec![None; num_docs as usize];
                }
                for old in 0..s.num_docs {
                    if let Some(new) = doc_map.new_id(i, old) {
                        if col.has_value(old) {
                            values[new as usize] = Some((i, old));
                        }
                    }
                }
            }
        }
        if !tris.is_empty() {
            for part in build_parts_with(&mut tris, single) {
                entries.push((crate::codec::EXTRA_SHAPE_TREE, field, part));
            }
        }
        drop(tris);
        if !values.is_empty() {
            let parts = crate::fieldtypes::shape_dv::encode_column(num_docs, |doc| {
                let (i, old) = values[doc as usize]?;
                segments[i].shape_data(field)?.values.as_ref()?.get(old)
            });
            for part in parts {
                entries.push((crate::codec::EXTRA_SHAPE_DV, field, part));
            }
        }
    }
    Ok(())
}

/// Per norms field: the norm byte of every new doc and the exact length
/// total over the live docs.
type MergedNorms = crate::admission::Allocation<BTreeMap<u16, (Vec<u8>, u64)>>;

fn merge_norms(segments: &[Arc<Segment>], doc_map: &DocMap) -> Result<MergedNorms> {
    merge_norms_with_budget(segments, doc_map, &crate::admission::shared())
}

fn merge_norms_with_budget(segments: &[Arc<Segment>], doc_map: &DocMap, budget: &Arc<crate::admission::Budget>) -> Result<MergedNorms> {
    let num_docs = doc_map.num_docs() as usize;
    let mut memory = budget.reserve(0, "merge norms")?;
    let mut merged: BTreeMap<u16, (Vec<u8>, u64)> = BTreeMap::new();
    for (i, seg) in segments.iter().enumerate() {
        let reader = seg.fast_fields();
        for (field, tag) in reader.fields() {
            if tag != 6 {
                continue;
            }
            let Some(c) = reader.get_norms(field) else { continue };
            if !merged.contains_key(&field) {
                // A new exact-size norm column plus conservative BTreeMap
                // node/split metadata. Admit before either allocation; keep
                // the guard across parallel postings/fast-fields consumers.
                memory.grow((num_docs as u64).checked_add(2048)
                    .ok_or_else(|| RusticError::Memory("merge norm size overflow".into()))?, "merge norm column")?;
                merged.insert(field, (vec![0u8; num_docs], 0));
            }
            let entry = merged.get_mut(&field).expect("norm field inserted");
            match doc_map.base(i) {
                Some(base) => {
                    // No deletions: one copy of the segment's norm bytes.
                    let src = c.lengths_bytes();
                    let n = seg.num_docs as usize;
                    if src.len() >= n {
                        entry.0[base as usize..base as usize + n].copy_from_slice(&src[..n]);
                    } else {
                        for old in 0..seg.num_docs {
                            entry.0[(base + old) as usize] = c.get_byte(old);
                        }
                    }
                    entry.1 += c.total();
                }
                None => {
                    // Exact total of the live docs: the segment total minus
                    // the (decoded) lengths of its deleted docs.
                    let mut total = c.total();
                    for old in 0..seg.num_docs {
                        match doc_map.new_id(i, old) {
                            Some(new) => entry.0[new as usize] = c.get_byte(old),
                            None => total = total.saturating_sub(c.get(old) as u64),
                        }
                    }
                    entry.1 += total;
                }
            }
        }
    }
    Ok(crate::admission::Allocation::new(merged, memory))
}

// ---- terms + postings -----------------------------------------------------

/// The merged term list: every distinct key with its contributing
/// `(segment, location)` lists, in key order (contributors in segment order).
struct MergedTerms {
    keys: Vec<u8>,
    /// `keys[key_ends[t-1]..key_ends[t]]` is term `t`'s key.
    key_ends: Vec<usize>,
    contrib: Vec<(u32, PostingsLocation)>,
    /// `contrib[contrib_ends[t-1]..contrib_ends[t]]` are term `t`'s lists.
    contrib_ends: Vec<usize>,
    _memory: crate::admission::Reservation,
}

impl MergedTerms {
    fn len(&self) -> usize {
        self.key_ends.len()
    }

    fn key(&self, t: usize) -> &[u8] {
        let s = if t == 0 { 0 } else { self.key_ends[t - 1] };
        &self.keys[s..self.key_ends[t]]
    }

    fn contributors(&self, t: usize) -> &[(u32, PostingsLocation)] {
        let s = if t == 0 { 0 } else { self.contrib_ends[t - 1] };
        &self.contrib[s..self.contrib_ends[t]]
    }
}

/// Streaming k-way merge of the segments' term dictionaries, produced a
/// window of terms at a time.
struct TermMerger<'a> {
    cursors: Vec<TermCursor<'a>>,
    live: Vec<bool>,
    key_memory: Vec<crate::admission::Reservation>,
    budget: Arc<crate::admission::Budget>,
    _metadata: crate::admission::Reservation,
}

impl<'a> TermMerger<'a> {
    fn new(segments: &'a [Arc<Segment>]) -> Result<Self> {
        Self::with_budget(segments, &crate::admission::shared())
    }

    fn with_budget(segments: &'a [Arc<Segment>], budget: &Arc<crate::admission::Budget>) -> Result<Self> {
        let metadata = budget.reserve((segments.len() as u64).saturating_mul((std::mem::size_of::<TermCursor<'a>>() + std::mem::size_of::<crate::admission::Reservation>() + 1) as u64), "merge term cursors")?;
        let mut merger = TermMerger {
            cursors: Vec::with_capacity(segments.len()),
            live: Vec::with_capacity(segments.len()),
            key_memory: Vec::with_capacity(segments.len()),
            budget: budget.clone(),
            _metadata: metadata,
        };
        for segment in segments {
            merger.cursors.push(segment.term_dict().cursor());
            merger.key_memory.push(budget.reserve(0, "merge cursor key")?);
            let i = merger.cursors.len() - 1;
            merger.live.push(advance_merge_cursor(&mut merger.cursors[i], &mut merger.key_memory[i])?);
        }
        Ok(merger)
    }

    /// The next (up to) `max_terms` merged terms; empty when done.
    fn next_window(&mut self, max_terms: usize) -> Result<MergedTerms> {
        let (cursors, live, key_memory) = (&mut self.cursors, &mut self.live, &mut self.key_memory);
        let mut out = MergedTerms { keys: Vec::new(), key_ends: Vec::new(), contrib: Vec::new(), contrib_ends: Vec::new(), _memory: self.budget.reserve(0, "merged term window")? };
        while out.len() < max_terms {
            // Smallest current key (lowest segment on ties).
            let mut min: Option<usize> = None;
            for i in 0..cursors.len() {
                if live[i] {
                    match min {
                        Some(m) if cursors[i].key() >= cursors[m].key() => {}
                        _ => min = Some(i),
                    }
                }
            }
            let Some(m) = min else { break };
            let start = out.keys.len();
            admit_merge_append(&mut out.keys, &mut out._memory, cursors[m].key().len())?;
            admit_merge_append(&mut out.key_ends, &mut out._memory, 1)?;
            admit_merge_append(&mut out.contrib_ends, &mut out._memory, 1)?;
            out.keys.extend_from_slice(cursors[m].key());
            out.key_ends.push(out.keys.len());
            for i in m..cursors.len() {
                if live[i] && cursors[i].key() == &out.keys[start..] {
                    admit_merge_append(&mut out.contrib, &mut out._memory, 1)?;
                    out.contrib.push((i as u32, cursors[i].loc()));
                    live[i] = advance_merge_cursor(&mut cursors[i], &mut key_memory[i])?;
                }
            }
            out.contrib_ends.push(out.contrib.len());
        }
        Ok(out)
    }
}

/// Keep the largest cursor buffer charged. The callback's conservative
/// key bound is doubled to cover old/replacement overlap during Vec growth.
fn advance_merge_cursor(cursor: &mut TermCursor<'_>, memory: &mut crate::admission::Reservation) -> Result<bool> {
    cursor.advance_guarded(|bytes| {
        let peak = bytes.checked_mul(2).ok_or_else(|| RusticError::Memory("merge term key size overflow".into()))?;
        if peak > memory.bytes() { memory.grow(peak - memory.bytes(), "merge term cursor key")?; }
        Ok(())
    })
}

fn admit_merge_append<T>(values: &mut Vec<T>, memory: &mut crate::admission::Reservation, additional: usize) -> Result<()> {
    let needed = values.len().checked_add(additional).ok_or_else(|| RusticError::Memory("merge term length overflow".into()))?;
    if needed > values.capacity() {
        let old = values.capacity();
        let next = needed.max(old.checked_mul(2).ok_or_else(|| RusticError::Memory("merge term capacity overflow".into()))?);
        let bytes = next.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| RusticError::Memory("merge term buffer size overflow".into()))?;
        memory.grow(bytes as u64, "merge term window growth overlap")?;
        values.reserve_exact(next - values.len());
        memory.shrink((old * std::mem::size_of::<T>()) as u64);
    }
    Ok(())
}

/// Terms merged and encoded per window of [`stream_postings`]: bounds the
/// term list and encoded postings held at once, whatever the segment size.
const POSTINGS_WINDOW_TERMS: usize = 1 << 20;

/// The merged terms section, after [`stream_postings`] wrote the postings.
struct MergedPostings {
    /// Sum of the merged terms' total term frequencies per field.
    field_ttf: BTreeMap<u16, u64>,
    terms: Vec<u8>,
    _memory: crate::admission::Reservation,
}

/// Reused per-run decode/encode buffers.
#[derive(Default)]
struct ListBuf {
    docs: Vec<DocId>,
    tfs: Vec<u32>,
    positions: Vec<u32>,
    block_pos: Vec<u32>,
}

impl ListBuf {
    /// Order the collected postings by (new) doc id (a sorting merge
    /// appends them in input order).
    fn sort_by_doc(&mut self, has_pos: bool) {
        if self.docs.windows(2).all(|w| w[0] < w[1]) {
            return;
        }
        let mut idx: Vec<u32> = (0..self.docs.len() as u32).collect();
        idx.sort_unstable_by_key(|&i| self.docs[i as usize]);
        if has_pos {
            let mut starts = Vec::with_capacity(self.tfs.len());
            let mut acc = 0usize;
            for &tf in &self.tfs {
                starts.push(acc);
                acc += tf as usize;
            }
            let mut pos = Vec::with_capacity(self.positions.len());
            for &i in &idx {
                let s = starts[i as usize];
                pos.extend_from_slice(&self.positions[s..s + self.tfs[i as usize] as usize]);
            }
            self.positions = pos;
        }
        self.docs = idx.iter().map(|&i| self.docs[i as usize]).collect();
        self.tfs = idx.iter().map(|&i| self.tfs[i as usize]).collect();
    }

    /// Append the live docs of one contributing list, renumbered.
    fn append(&mut self, r: &PostingsReader, seg: usize, doc_map: &DocMap) -> Result<()> {
        let mut decoded_freq = 0u64;
        let has_pos = r.has_positions();
        let (mut d, mut t) = ([0u32; BLOCK_DOCS], [1u32; BLOCK_DOCS]);
        let nfull = r.num_full_blocks();
        for b in 0..=nfull {
            let n = if b < nfull {
                r.decode_docs(b, &mut d);
                if r.decode_tfs(b, &mut t) {
                    t.fill(1);
                }
                BLOCK_DOCS
            } else {
                if r.tail_len() == 0 {
                    break;
                }
                r.decode_tail(&mut d, &mut t)
            };
            decoded_freq = t[..n].iter().try_fold(decoded_freq, |sum, &tf| sum.checked_add(tf as u64)).ok_or_else(|| RusticError::Corrupted("merge postings frequency overflow".into()))?;
            if decoded_freq > r.total_term_freq() {
                return Err(RusticError::Corrupted("merge postings frequencies exceed header".into()));
            }
            let count: usize = if has_pos { t[..n].iter().map(|&x| x as usize).sum() } else { 0 };
            match doc_map.base(seg) {
                Some(base) => {
                    self.docs.extend(d[..n].iter().map(|&x| x + base));
                    self.tfs.extend_from_slice(&t[..n]);
                    // Every position is kept: no per-block scratch.
                    r.append_block_positions(b, count, &mut self.positions);
                }
                None => {
                    if has_pos {
                        // Exactly one block's positions (see `term_run_peak`).
                        if self.block_pos.capacity() < count {
                            self.block_pos = Vec::with_capacity(count);
                        }
                        r.decode_block_positions(b, count, &mut self.block_pos);
                    }
                    let row = &doc_map.remap[seg];
                    let mut p = 0usize;
                    for i in 0..n {
                        let tf = t[i] as usize;
                        if let Some(new) = live(row[d[i] as usize]) {
                            self.docs.push(new);
                            self.tfs.push(t[i]);
                            if has_pos {
                                self.positions.extend_from_slice(&self.block_pos[p..p + tf]);
                            }
                        }
                        p += tf;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A term's `(doc_freq, total_term_freq)` summed over its contributors, from
/// the dictionaries (each list header is checked against them on decode).
fn term_counts(terms: &MergedTerms, t: usize) -> Result<(u64, u64)> {
    terms.contributors(t).iter().try_fold((0u64, 0u64), |(df, ttf), (_, loc)| {
        df.checked_add(loc.doc_freq as u64).zip(ttf.checked_add(loc.total_term_freq))
            .ok_or_else(|| RusticError::Memory("merge postings run size overflow".into()))
    })
}

/// Peak working memory of [`encode_term_run`] over `range`, with the bound
/// of its encoded output (allocated once, so it never grows past it).
/// The decode buffers are reused across the run's terms and sized exactly
/// for each term (4 bytes per doc for doc ids and for tfs, 4 per position),
/// so they peak at the run's largest counts, not their sum. Added to them:
/// one block of positions scratch when an input has deletions, a sorting
/// merge's reorder scratch (an index and position starts, 12 bytes per doc,
/// plus the reordered positions), the encoder's positions region and the
/// run's location table.
///
/// Positions are counted only for terms whose lists carry them (as
/// [`encode_term_run`] decodes them). A list without positions decodes into
/// its doc and tf buffers only: its total term frequency is not a position
/// count. For a `rank_features` / sparse-vector field it is a sum of
/// weights (up to `0x7F7F` per doc), and counting it as positions sized a
/// small index's merge at about a gigabyte.
fn term_run_peak(terms: &MergedTerms, range: std::ops::Range<usize>, segments: &[Arc<Segment>], doc_map: &DocMap) -> Result<(u64, u64)> {
    let overflow = || RusticError::Memory("merge postings run size overflow".into());
    // `max_ttf`: the largest position count among positional terms.
    let (mut max_df, mut max_ttf, mut encoded, mut remaps) = (0u64, 0u64, 0u64, false);
    for t in range.clone() {
        let (df, ttf) = term_counts(terms, t)?;
        let positions = term_has_positions(terms, t, segments);
        max_df = max_df.max(df);
        if positions {
            max_ttf = max_ttf.max(ttf);
        }
        encoded = encoded.checked_add(crate::codec::postings::encoded_postings_bound(df, ttf, doc_map.num_docs() as u64, positions)).ok_or_else(overflow)?;
        remaps |= terms.contributors(t).iter().any(|(seg, _)| doc_map.base(*seg as usize).is_none());
    }
    let mut work = max_df.saturating_mul(8).saturating_add(max_ttf.saturating_mul(4));
    if remaps {
        work = work.saturating_add(max_ttf.saturating_mul(4));
    }
    if doc_map.is_sorted() {
        work = work.saturating_add(max_df.saturating_mul(12)).saturating_add(max_ttf.saturating_mul(4));
    }
    work = work.saturating_add(crate::codec::postings::positions_region_bound(max_ttf, max_df));
    let locs = (range.len() as u64).saturating_mul(std::mem::size_of::<Option<PostingsLocation>>() as u64);
    let peak = (64u64 << 10).checked_add(work).and_then(|n| n.checked_add(encoded)).and_then(|n| n.checked_add(locs)).ok_or_else(overflow)?;
    Ok((peak, encoded))
}

/// Whether any contributing list of term `t` carries positions (what
/// [`encode_term_run`] decodes and writes positions for).
fn term_has_positions(terms: &MergedTerms, t: usize, segments: &[Arc<Segment>]) -> bool {
    // An inline single-doc list carries its position in the entry, if any.
    terms.contributors(t).iter().any(|(seg, loc)| if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 {
        crate::term_dictionary::inline_position(loc.offset).is_some()
    } else {
        segments[*seg as usize].postings_at(loc).is_some_and(|r| r.has_positions())
    })
}

/// Replace `v` (empty) by an exact allocation of `n` when it is smaller:
/// no geometric overshoot and no old/new overlap.
fn exact_capacity<T>(v: &mut Vec<T>, n: u64) {
    if (v.capacity() as u64) < n {
        *v = Vec::new();
        v.reserve_exact(n as usize);
    }
}

/// Re-encode terms `range` of `terms` into one postings buffer; returns the
/// buffer and per term its `(offset, doc_freq, total_term_freq)` within it
/// (`None` when every doc of the term was deleted).
#[cfg(test)]
fn encode_term_run_with_budget(terms: &MergedTerms, range: std::ops::Range<usize>, segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, impacts: bool, budget: &Arc<crate::admission::Budget>) -> Result<crate::admission::Allocation<(Vec<u8>, Vec<Option<PostingsLocation>>)>> {
    let (peak, encoded) = term_run_peak(terms, range.clone(), segments, doc_map)?;
    let memory = budget.reserve(peak, "merge postings decode and encode workspace")?;
    encode_term_run(terms, range, segments, doc_map, norms, impacts, false, false, memory, encoded)
}

/// [`encode_term_run_with_budget`] within `memory`, admitted for
/// [`term_run_peak`]'s `peak`; `encoded` is its output bound.
#[allow(clippy::too_many_arguments)]
/// `inline`: format [`crate::codec::INLINE_POSTINGS_SINCE`]+, single-doc
/// lists without positions go into the dictionary entry.
fn encode_term_run(terms: &MergedTerms, range: std::ops::Range<usize>, segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, impacts: bool, inline: bool, inline_positions: bool, mut memory: crate::admission::Reservation, encoded: u64) -> Result<crate::admission::Allocation<(Vec<u8>, Vec<Option<PostingsLocation>>)>> {
    // Allocated (not touched) at its bound once: no reallocation copies.
    let mut out = Vec::with_capacity(encoded as usize);
    let mut locs = Vec::with_capacity(range.len());
    let mut buf = ListBuf::default();
    let mut cur_field = u32::MAX;
    let mut block_norms = BlockNorms::Absent;
    for t in range {
        let key = terms.key(t);
        let field = key_field(key);
        if field as u32 != cur_field {
            cur_field = field as u32;
            block_norms = match norms.get(&field) {
                Some((bytes, _)) => BlockNorms::Bytes(bytes),
                None => BlockNorms::Absent,
            };
        }
        // One input's inline single-doc list (every `_id` of v16 inputs):
        // remap its doc, no postings opened or decoded.
        if let [(seg, loc)] = terms.contributors(t) {
            if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 {
                merge_pace(17);
                let old = crate::term_dictionary::inline_doc(loc.offset);
                let position = crate::term_dictionary::inline_position(loc.offset);
                match doc_map.new_id(*seg as usize, old) {
                    None => locs.push(None),
                    Some(new) if inline && (position.is_none() || inline_positions) => locs.push(Some(PostingsLocation { offset: crate::term_dictionary::inline_offset(new, position), doc_freq: 1, total_term_freq: 1 })),
                    Some(new) => {
                        let offset = out.len() as u64;
                        write_postings_uncached(&[new], &[1], position.as_ref().map(std::slice::from_ref), block_norms, impacts, &mut out);
                        locs.push(Some(PostingsLocation { offset, doc_freq: 1, total_term_freq: 1 }));
                    }
                }
                continue;
            }
        }
        buf.docs.clear();
        buf.tfs.clear();
        buf.positions.clear();
        let (df, ttf) = term_counts(terms, t)?;
        exact_capacity(&mut buf.docs, df);
        exact_capacity(&mut buf.tfs, df);
        let mut has_positions = false;
        // Paced by the work done: docs, and positions only when there are
        // any (a feature field's ttf is a sum of weights).
        merge_pace(16 + df + if term_has_positions(terms, t, segments) { ttf } else { 0 });
        for (seg, loc) in terms.contributors(t) {
            let Some(r) = segments[*seg as usize].postings_at(loc) else { continue };
            if r.doc_freq() != loc.doc_freq || r.total_term_freq() != loc.total_term_freq {
                return Err(RusticError::Corrupted("merge postings dictionary/header counters differ".into()));
            }
            if r.has_positions() && !has_positions {
                exact_capacity(&mut buf.positions, ttf);
            }
            has_positions |= r.has_positions();
            buf.append(&r, *seg as usize, doc_map)?;
        }
        if doc_map.is_sorted() {
            buf.sort_by_doc(has_positions);
        }
        if buf.docs.is_empty() {
            locs.push(None); // every doc of this term was deleted
            continue;
        }
        debug_assert!(buf.docs.windows(2).all(|w| w[0] < w[1]));
        if let Some(at) = crate::segment::inline_singleton(inline, inline_positions, &buf.docs, &buf.tfs, has_positions.then_some(buf.positions.as_slice())) {
            locs.push(Some(PostingsLocation { offset: at, doc_freq: 1, total_term_freq: 1 }));
            continue;
        }
        let offset = out.len() as u64;
        write_postings_uncached(
            &buf.docs,
            &buf.tfs,
            has_positions.then_some(buf.positions.as_slice()),
            block_norms,
            impacts,
            &mut out,
        );
        locs.push(Some(PostingsLocation {
            offset,
            doc_freq: buf.docs.len() as u32,
            total_term_freq: buf.tfs.iter().map(|&x| x as u64).sum(),
        }));
    }
    drop(buf);
    // Return the unused part of the bound (in place for large buffers).
    out.shrink_to_fit();
    let retained = (out.capacity() as u64).saturating_add((locs.capacity() as u64).saturating_mul(std::mem::size_of::<Option<PostingsLocation>>() as u64));
    memory.shrink(memory.bytes().saturating_sub(retained));
    Ok(crate::admission::Allocation::new((out, locs), memory))
}

/// Closed term-dictionary blocks a merge buffers before writing them to
/// its spill file ([`stream_postings`]).
const TERMS_SPILL_BYTES: usize = 1 << 20;

/// Share of the background budget one batch of parallel postings runs may
/// hold: runs beyond it wait for the batch before them to be written out,
/// so a merge's postings never crowd out foreground admissions.
const POSTINGS_BATCH_DIVISOR: u64 = 8;

/// How long a background merge waits for budget room (other owners
/// releasing) before it gives up with a memory error.
pub(crate) const MERGE_ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Group consecutive runs into batches of at most `cap` admitted bytes
/// (a run larger than `cap` is a batch of its own).
fn run_batches(peaks: &[u64], cap: u64) -> Vec<std::ops::Range<usize>> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < peaks.len() {
        let (mut end, mut sum) = (start, 0u64);
        while end < peaks.len() && (end == start || sum.saturating_add(peaks[end]) <= cap) {
            sum = sum.saturating_add(peaks[end]);
            end += 1;
        }
        batches.push(start..end);
        start = end;
    }
    batches
}

/// Merge and encode the postings into `out` (the open postings section), a
/// window of terms at a time; returns the term dictionary and field totals.
/// Each window's lists are re-encoded in parallel over runs of roughly equal
/// work and written in term order, so the section's bytes do not depend on
/// the window or run boundaries.
fn stream_postings(segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, window_terms: usize, impacts: bool, inline: bool, compact: bool, inline_positions: bool, packed: bool, int_keys: bool, admit: MergeAdmission, mut out: impl FnMut(&[u8]) -> Result<()>, mut spill: Option<&mut dyn FnMut(&[u8]) -> Result<()>>) -> Result<MergedPostings> {
    use rayon::prelude::*;
    let budget = crate::admission::shared();
    let mut memory = admit.reserve(&budget, 4096, "merge term dictionary")?;
    let mut merger = TermMerger::new(segments)?;
    let mut dict = TermDictWriter::with_capacity(0).tagged(inline).compact(compact).inline_positions(inline_positions).packed(packed).int_keys(packed && int_keys);
    let mut base = 0u64;
    let mut field_ttf: BTreeMap<u16, u64> = BTreeMap::new();
    loop {
        let terms = merger.next_window(window_terms)?;
        let n = terms.len();
        if n == 0 {
            break;
        }
        let _runs_memory = admit.reserve(&budget, (n as u64).saturating_mul(64), "merge postings run descriptors")?;
        // Runs of roughly equal work (postings volume plus a per-term
        // constant), several per thread for load balancing.
        // (Positions only where the term's lists carry them, as above.)
        let weight = |t: usize| -> u64 {
            let positional = term_has_positions(&terms, t, segments);
            16 + terms.contributors(t).iter().map(|(_, l)| (if positional { l.total_term_freq } else { 0 }) + l.doc_freq as u64).sum::<u64>()
        };
        let total: u64 = (0..n).map(weight).sum();
        let runs_wanted = (rayon::current_num_threads() * 4).max(1) as u64;
        let target = (total / runs_wanted).max(1 << 16);
        let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
        let (mut start, mut acc) = (0usize, 0u64);
        for t in 0..n {
            acc += weight(t);
            if acc >= target {
                runs.push(start..t + 1);
                start = t + 1;
                acc = 0;
            }
        }
        if start < n {
            runs.push(start..n);
        }
        let pacing = PACING_ENABLED.with(|p| p.get());
        let admission = crate::admission::Context::capture();
        let peaks: Vec<(u64, u64)> = runs.iter().map(|r| term_run_peak(&terms, r.clone(), segments, doc_map)).collect::<Result<_>>()?;
        let totals: Vec<u64> = peaks.iter().map(|&(peak, _)| peak).collect();
        // Admit a batch of runs at once, waiting for room rather than failing
        // part way; each run then works within its share.
        for batch in run_batches(&totals, (budget.background_limit() / POSTINGS_BATCH_DIVISOR).max(1)) {
        let mut batch_memory = admit.reserve(&budget, totals[batch.clone()].iter().sum(), "merge postings decode and encode workspace")?;
        let shares: Vec<_> = peaks[batch.clone()].iter().map(|&(peak, encoded)| (batch_memory.split(peak).expect("the batch admits each run"), encoded)).collect();
        let encoded: Vec<crate::admission::Allocation<(Vec<u8>, Vec<Option<PostingsLocation>>)>> =
            runs[batch.clone()].par_iter().zip(shares).map(|(r, (memory, encoded))| admission.run(|| with_pacing(pacing, || encode_term_run(&terms, r.clone(), segments, doc_map, norms, impacts, inline, inline_positions, memory, encoded)))).collect::<Result<_>>()?;
        for (run, encoded) in runs[batch].iter().zip(encoded) {
            let (bytes, locs) = &*encoded;
            for (t, loc) in run.clone().zip(locs.iter().copied()) {
                if let Some(loc) = loc {
                    let field = key_field(terms.key(t));
                    if !field_ttf.contains_key(&field) { memory.grow(2048, "merge field totals")?; }
                    // Writer retains per term at most the key suffix and 35
                    // bytes of lengths/counters, a block's first key (1 in
                    // 16 terms) and its index entries, at twice its length
                    // for geometric growth, and finish appends in place.
                    if spill.is_none() {
                        memory.grow((terms.key(t).len() as u64).saturating_mul(3).saturating_add(96), "merge term dictionary growth")?;
                    }
                    *field_ttf.entry(field).or_default() += loc.total_term_freq;
                    let offset = if loc.offset & crate::term_dictionary::INLINE_POSTINGS != 0 { loc.offset } else { base + loc.offset };
                    dict.add(terms.key(t), &PostingsLocation { offset, ..loc });
                }
            }
            out(&bytes)?;
            base += bytes.len() as u64;
            // The dictionary's closed blocks go to the spill as they fill: a
            // merge of segments with many unique terms (dynamic text fields,
            // one value per doc) held its whole merged dictionary until the
            // end, reserving GBs of the background share and refusing every
            // other merge meanwhile.
            // What the writer holds is then charged as measured (its
            // unspilled blocks, the first keys and the block index), not
            // per term as when the whole dictionary stays in memory.
            if let Some(spill) = spill.as_mut() {
                if let Some(blocks) = dict.take_spill(TERMS_SPILL_BYTES) {
                    spill(&blocks)?;
                }
                let retained = (dict.retained_bytes() as u64).saturating_add((field_ttf.len() as u64).saturating_mul(2048)).saturating_add(4096);
                if retained > memory.bytes() {
                    memory.grow(retained - memory.bytes(), "merge term dictionary growth")?;
                } else {
                    memory.shrink(memory.bytes() - retained);
                }
            }
        }
        }
    }
    let terms = dict.finish(doc_map.num_docs());
    let retained = (terms.capacity() as u64).saturating_add((field_ttf.len() as u64).saturating_mul(2048));
    memory.shrink(memory.bytes().saturating_sub(retained));
    Ok(MergedPostings { field_ttf, terms, _memory: memory })
}

// ---- fast fields ------------------------------------------------------------

/// A numeric column written without a points index (doc-values skipper
/// only) stays that way in the merged segment. A column left without one
/// only because its docs all held one value (the writer's constant-column
/// rule) does not decide that: when another input has the points index the
/// merged column keeps it (else a segment of all-zero `population` would
/// strip the points, and with them value-ordered sorts and script top-k,
/// from every merge it joins). A doc-values-only field has no input with
/// points, so it stays without them.
fn carry_skipper<'a, 'r: 'a>(readers: impl IntoIterator<Item = &'a crate::codec::fastfields::FastFieldsReader<'r>>, ff: &mut FastFields, field: u16) {
    fn state<T: crate::codec::fastfields::FastValue>(c: &crate::codec::fastfields::ColumnReader<'_, T>) -> (bool, bool, bool) {
        (c.has_points(), c.skips_points(), c.skips_points() && !c.written_constant())
    }
    let (mut points, mut skips, mut on_purpose) = (false, false, false);
    for r in readers {
        let s = r.get_i64(field).map(|c| state(&c)).or_else(|| r.get_u64(field).map(|c| state(&c))).or_else(|| r.get_f64(field).map(|c| state(&c)));
        if let Some((p, k, o)) = s {
            points |= p;
            skips |= k;
            on_purpose |= o;
        }
    }
    if on_purpose || (skips && !points) {
        ff.skip_points(field);
    }
}

/// Spool encoded columns to the directory rather than keeping their total
/// output in anonymous memory. The mapped bodies are copied into the final
/// section after its small column directory. No on-disk format changes.
struct MergedFastFields {
    directory: Vec<u8>,
    bodies: FileSlice,
    _memory: crate::admission::Reservation,
}

struct MergeScratch<'a> {
    dir: &'a dyn Directory,
    name: String,
}

impl Drop for MergeScratch<'_> {
    fn drop(&mut self) {
        let _ = self.dir.delete(&self.name);
    }
}

/// `version`: the merged file's format, as its header.
fn spool_fast_fields(dir: &dyn Directory, new_id: u32, segments: &[Arc<Segment>], doc_map: &DocMap, norms: &MergedNorms, admit: MergeAdmission, version: u32) -> Result<MergedFastFields> {
    static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scratch = MergeScratch { dir, name: format!(".merge-fast-{new_id}-{}-{sequence}.tmp", std::process::id()) };
    let count = segments.iter().map(|s| s.fast_fields().field_count() as u64).fold(0u64,u64::saturating_add);
    let mut memory = crate::admission::shared().reserve(count.saturating_mul(512).saturating_add(4096), "merge fast field spool directory")?;
    let mut output = dir.create_scratch_output(&scratch.name)?;
    let mut entries = Vec::new();
    let mut offset = 0usize;
    // Consecutive bodies of one column concatenate (a column streamed in
    // pieces, see `merge_keyword_by_ordinal`).
    let mut emit = |field: u16, tag: u8, body: &[u8]| -> Result<()> {
        output.write_all(&body)?;
        match entries.last_mut() {
            Some((f, t, _, len)) if *f == field && *t == tag => *len += body.len(),
            _ => entries.push((field, tag, offset, body.len())),
        }
        offset += body.len();
        Ok(())
    };
    if doc_map.is_sorted() {
        stream_fast_fields_sorted(segments, doc_map, norms, admit, version, &mut emit)?;
    } else {
        stream_fast_fields(segments, doc_map, norms, true, admit, version, &mut emit)?;
    }
    output.finish()?;
    let bodies = dir.open_read(&scratch.name)?;
    entries.sort_unstable_by_key(|&(field, tag, _, _)| (field, tag));
    let mut directory = Vec::new();
    write_vint(entries.len() as u64, &mut directory);
    for (field, tag, offset, len) in entries {
        directory.extend_from_slice(&field.to_le_bytes());
        directory.push(tag);
        write_vint(offset as u64, &mut directory);
        write_vint(len as u64, &mut directory);
    }
    // FileSlice keeps the mapping alive after the scratch name is removed.
    drop(scratch);
    memory.shrink(memory.bytes().saturating_sub(directory.capacity() as u64));
    Ok(MergedFastFields { directory, bodies, _memory: memory })
}

#[cfg(test)]
fn merge_fast_fields_impl(segments: &[Arc<Segment>], doc_map: &DocMap, merged_norms: &MergedNorms, keywords_by_ordinal: bool) -> Vec<u8> {
    let mut columns: Vec<(u16, u8, Vec<u8>)> = Vec::new();
    stream_fast_fields(segments, doc_map, merged_norms, keywords_by_ordinal, MergeAdmission::Background(std::time::Duration::ZERO), crate::codec::write_version(), &mut |f, t, b| {
        match columns.last_mut() {
            Some((lf, lt, body)) if *lf == f && *lt == t => body.extend_from_slice(b),
            _ => columns.push((f, t, b.to_vec())),
        }
        Ok(())
    }).unwrap();
    crate::codec::fastfields::encode_fast_fields_with(&FastFields::new(0), doc_map.num_docs(), columns)
}

fn emit_fast_columns(ff: &FastFields, num_docs: u32, version: u32, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>) -> Result<()> {
    for (field, tag, body) in crate::codec::fastfields::encode_fast_field_columns_versioned(ff, num_docs, version) {
        emit(field, tag, &body)?;
    }
    Ok(())
}

fn fast_merge_metadata(segments: &[Arc<Segment>]) -> Result<crate::admission::Reservation> {
    let columns = segments.iter().try_fold(0u64, |sum, s| sum.checked_add(s.fast_fields().field_count() as u64).ok_or_else(|| RusticError::Memory("merge column count overflow".into())))?;
    crate::admission::shared().reserve(columns.saturating_mul(512).saturating_add((segments.len() as u64).saturating_mul(512)).saturating_add(4096), "merge fast field metadata")
}

/// Bound one column's mutable collector, sorting/encoding scratch and encoded
/// output before any of them grow. Count borrowed values without materializing
/// per-document strings or vectors. Per unit, from the collectors and encoders:
/// - single numeric (dense by doc): values at 8 bytes per doc with doubling
///   (16), the encoded body (at most 13 bytes per doc: value, points entry,
///   presence and rank words) with doubling (26), the points sort pairs (16)
///   and the compact doc list (4): 64 per doc;
/// - multi numeric: values 8 per value and offsets 4 per doc, doubled, and the
///   body (offsets and values again) doubled: 16 per doc and 34 per value;
/// - bool: bits and body at 1/8 byte per doc, doubled; norms: lengths and
///   bytes (5 per doc) doubled and the body: 12 per doc;
/// - keyword by ordinal ([`merge_keyword_by_ordinal`]): per input term a used
///   bit, a 4-byte map entry and an 8-byte source of each merged term (the
///   table itself streams out through a 1 MiB buffer, its bytes borrowed
///   from the inputs); per doc a 4-byte ordinal, and for multi-valued
///   columns offsets and ordinals (4 + 4 per value) doubled;
/// - keyword rebuilt as a column (sorting merges): the string table, the
///   value -> docs map (key copy, entry and doc list) and the sorted term
///   refs, 5 bytes per value byte and 192 per value, plus 32 per doc.
fn fast_column_memory(segments: &[Arc<Segment>], doc_map: &DocMap, field: u16, out: u8, by_ordinal: bool, budget: &Arc<crate::admission::Budget>, admit: MergeAdmission) -> Result<crate::admission::Reservation> {
    let n = doc_map.num_docs() as u64;
    let per_doc: u64 = match out {
        1..=3 => 64,
        7..=9 => 16,
        4 => 1,
        6 => 12,
        5 | 10 if by_ordinal => if out == 5 { 4 } else { 8 },
        5 | 10 => 32,
        _ => 0,
    };
    // (Plus a streaming keyword table's 1 MiB write buffer.)
    let mut bytes = n.saturating_mul(per_doc).saturating_add(if by_ordinal { (64 << 10) + (1 << 20) } else { 64 << 10 });
    let mut add = |count: u64, per: u64, text: u64, per_byte: u64| -> Result<()> {
        bytes = bytes.checked_add(count.saturating_mul(per)).and_then(|b| b.checked_add(text.saturating_mul(per_byte))).ok_or_else(|| RusticError::Memory("merge fast field workspace overflow".into()))?;
        Ok(())
    };
    for (i, segment) in segments.iter().enumerate() {
        let reader = segment.fast_fields();
        let live_docs = || (0..segment.num_docs).filter(|&doc| doc_map.new_id(i, doc).is_some());
        match out {
            7 | 8 | 9 => {
                // A single-valued source promoted into the multi column.
                let single = match out { 7 => reader.get_i64(field).map(|c| c.present_docs().filter(|&doc| doc_map.new_id(i, doc).is_some()).count()), 8 => reader.get_u64(field).map(|c| c.present_docs().filter(|&doc| doc_map.new_id(i, doc).is_some()).count()), _ => reader.get_f64(field).map(|c| c.present_docs().filter(|&doc| doc_map.new_id(i, doc).is_some()).count()) };
                add(single.unwrap_or(0) as u64, 34, 0, 0)?;
                match out {
                    7 => if let Some(c) = reader.get_multi_i64(field) { for doc in live_docs() { add(c.values(doc).count() as u64, 34, 0, 0)?; } },
                    8 => if let Some(c) = reader.get_multi_u64(field) { for doc in live_docs() { add(c.values(doc).count() as u64, 34, 0, 0)?; } },
                    _ => if let Some(c) = reader.get_multi_f64(field) { for doc in live_docs() { add(c.values(doc).count() as u64, 34, 0, 0)?; } },
                }
            }
            5 | 10 if by_ordinal => {
                // Input terms (dead ones included: ordinal maps cover them).
                let multi = out == 10;
                if let Some(c) = reader.get_keyword(field) {
                    add(c.num_terms() as u64, 16, 0, 0)?;
                }
                if let Some(c) = reader.get_multi_keyword(field) {
                    add(c.num_terms() as u64, 16, 0, 0)?;
                    if multi { for doc in live_docs() { add(c.ords(doc).count() as u64, 8, 0, 0)?; } }
                }
                if multi {
                    if let Some(c) = reader.get_keyword(field) { add(live_docs().filter(|&doc| c.get(doc).is_some()).count() as u64, 8, 0, 0)?; }
                }
            }
            5 | 10 => {
                if let Some(c) = reader.get_keyword(field) {
                    for doc in live_docs() { if let Some(v) = c.get(doc) { add(1, 192, v.len() as u64, 5)?; } }
                }
                if let Some(c) = reader.get_multi_keyword(field) {
                    for doc in live_docs() { for ord in c.ords(doc) { add(1, 192, c.term(ord).map_or(0, |v| v.len()) as u64, 5)?; } }
                }
                if let Some(c) = reader.get_features(field) {
                    // About 16 bytes a value (`token\0` + 8 hex digits).
                    for doc in live_docs() { let n = c.count(doc) as u64; add(n, 192, n * 16, 5)?; }
                }
            }
            _ => {}
        }
    }
    admit.reserve(budget, bytes, "merge fast field column workspace")
}

/// `keywords_by_ordinal` is false only for the test reference implementation.
fn stream_fast_fields(segments: &[Arc<Segment>], doc_map: &DocMap, merged_norms: &MergedNorms, keywords_by_ordinal: bool, admit: MergeAdmission,
    version: u32, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>) -> Result<()> {
    let _metadata = fast_merge_metadata(segments)?;
    let profile = tracing::enabled!(target: "merge_profile", tracing::Level::DEBUG);
    let started = profile.then(std::time::Instant::now);
    let num_docs = doc_map.num_docs();
    let mut columns: Vec<(u16, u8)> = Vec::new();
    for seg in segments {
        for (field, tag) in seg.fast_fields().fields() {
            if !columns.iter().any(|(f, t)| *f == field && *t == tag) {
                columns.push((field, tag));
            }
        }
    }
    // A field single-valued in one segment and multi-valued in another
    // merges into its multi-valued column only: readers prefer the single
    // column, so keeping both would hide the multi-valued segment's docs.
    let promote = |field: u16, tag: u8| -> u8 {
        let multi = match tag {
            1 => 7,
            2 => 8,
            3 => 9,
            5 => 10,
            // Feature values merge with the multi-valued keyword column of
            // the field (the encoder picks the output layout).
            TAG_FEATURES => return 10,
            _ => return tag,
        };
        if columns.iter().any(|&(f, t)| f == field && t == multi) { multi } else { tag }
    };
    // (field, output tag) -> source tags. A segment holds at most one of a
    // field's source columns (its writer promotes in place), so visiting
    // segments in order keeps a promoted multi column's docs ascending.
    let mut groups: Vec<(u16, u8, Vec<u8>)> = Vec::new();
    for &(f, t) in &columns {
        let out = promote(f, t);
        match groups.iter_mut().find(|(gf, go, _)| *gf == f && *go == out) {
            Some(g) => g.2.push(t),
            None => groups.push((f, out, vec![t])),
        }
    }
    groups.sort_unstable();
    let from_postings = keyword_from_postings_union(segments);
    for (field, out, tags) in groups {
        // No column where an input has none: the merged segment derives
        // the field's values from its (merged) postings and stored fields.
        if from_postings.contains(&field) && (out == TAG_KEYWORD || out == TAG_MULTI_KEYWORD) {
            continue;
        }
        merge_pace(num_docs as u64);
        // Feature columns hold no term table to map ordinals through.
        let by_ordinal = keywords_by_ordinal && (out == TAG_KEYWORD || out == TAG_MULTI_KEYWORD) && !tags.contains(&TAG_FEATURES);
        let _column_memory = fast_column_memory(segments, doc_map, field, out, by_ordinal, &crate::admission::shared(), admit)?;
        let mut ff = FastFields::new(0);
        let inputs: Vec<_> = segments.iter().map(|seg| seg.fast_fields()).collect();
        carry_skipper(&inputs, &mut ff, field);

        if by_ordinal {
            merge_keyword_by_ordinal(segments, doc_map, field, out, version, emit)?;
            continue;
        }
        // Feature columns only: their tables and packed pairs merge
        // directly, without decoding each value to its string.
        if !tags.is_empty() && tags.iter().all(|&t| t == TAG_FEATURES) && version >= crate::codec::FEATURES_COLUMN_SINCE {
            let readers: Vec<_> = segments.iter().map(|s| s.fast_fields()).collect();
            let inputs: Vec<_> = readers.iter().map(|r| r.get_features(field)).collect();
            if let Some(body) = crate::codec::fastfields::merge_features(&inputs, &|i, old| doc_map.new_id(i, old), num_docs) {
                emit(field, TAG_FEATURES, &body)?;
            }
            continue;
        }
        if out == 6 {
            // Norms: already merged (one copy per segment without deletions).
            if let Some((bytes, total)) = merged_norms.get(&field).filter(|(b, _)| !b.is_empty()) {
                let col = ff.norms_mut(field);
                for (new, &b) in bytes.iter().enumerate() {
                    col.set_byte(new as DocId, b);
                }
                col.set_total(*total);
            }
            emit_fast_columns(&ff, num_docs, version, emit)?;
            continue;
        }
        for (i, seg) in segments.iter().enumerate() {
            let reader = seg.fast_fields();
            for &tag in &tags {
            let live = (0..seg.num_docs).filter_map(|old| doc_map.new_id(i, old).map(|new| (old, new)));
            macro_rules! numeric {
                ($get:ident, $single:ident, $multi:ident) => {
                    if let Some(c) = reader.$get(field) {
                        let mut values = c.present_docs().filter_map(|old| {
                            doc_map.new_id(i, old).map(|new| (new, c.raw_value(old)))
                        });
                        // Preserve omission of columns whose values were all
                        // deleted, and avoid looking up the target per value.
                        if let Some((first, value)) = values.next() {
                            if out == tag {
                                let target = ff.$single(field);
                                target.set(first, value);
                                for (new, value) in values { target.set(new, value); }
                            } else {
                                let target = ff.$multi(field);
                                target.push(first, value);
                                for (new, value) in values { target.push(new, value); }
                            }
                        }
                    }
                };
            }
            match tag {
                1 => numeric!(get_i64, i64_column_mut, multi_i64_column_mut),
                2 => numeric!(get_u64, u64_column_mut, multi_u64_column_mut),
                3 => numeric!(get_f64, f64_column_mut, multi_f64_column_mut),
                4 => {
                    if let Some(c) = reader.get_bool(field) {
                        let mut live = live.peekable();
                        if live.peek().is_some() {
                            let target = ff.bool_column_mut(field);
                            // A column that does not know its presence makes
                            // the output's unknown too (its docs read false).
                            if !c.presence_known() {
                                target.set_presence_unknown();
                            }
                            for (old, new) in live {
                                if c.has_value(old) {
                                    target.set(new, c.get(old));
                                }
                            }
                        }
                    }
                }
                5 => {
                    if let Some(c) = reader.get_keyword(field) {
                        for (old, new) in live {
                            if let Some(v) = c.get(old) {
                                if out == tag {
                                    ff.keyword_column_mut(field).set(new, &v);
                                } else {
                                    ff.multi_keyword_column_mut(field).push(new, &v);
                                }
                            }
                        }
                    }
                }
                7 => {
                    if let Some(c) = reader.get_multi_i64(field) {
                        for (old, new) in live {
                            for v in c.values(old) {
                                ff.multi_i64_column_mut(field).push(new, v);
                            }
                        }
                    }
                }
                8 => {
                    if let Some(c) = reader.get_multi_u64(field) {
                        for (old, new) in live {
                            for v in c.values(old) {
                                ff.multi_u64_column_mut(field).push(new, v);
                            }
                        }
                    }
                }
                9 => {
                    if let Some(c) = reader.get_multi_f64(field) {
                        for (old, new) in live {
                            for v in c.values(old) {
                                ff.multi_f64_column_mut(field).push(new, v);
                            }
                        }
                    }
                }
                10 => {
                    if let Some(c) = reader.get_multi_keyword(field) {
                        for (old, new) in live {
                            for v in c.values(old) {
                                ff.multi_keyword_column_mut(field).push(new, v);
                            }
                        }
                    }
                }
                TAG_FEATURES => {
                    if let Some(c) = reader.get_features(field) {
                        for (old, new) in live {
                            for v in c.values(old) {
                                ff.multi_keyword_column_mut(field).push(new, &v);
                            }
                        }
                    }
                }
                _ => {}
            }
            }
        }
        emit_fast_columns(&ff, num_docs, version, emit)?;
    }
    if let Some(started) = started {
        tracing::debug!(target: "merge_profile", stage="fast_collect", elapsed_us=started.elapsed().as_micros() as u64, "segment merge stage (spans overlap)");
    }
    // The encoder builds disk indexes from raw values/presence. Building
    // the mutable columns' query indexes here only repeats scans and sorts.
    Ok(())
}

/// Keyword fields some input has no column for ([`crate::codec::EXTRA_KEYWORD_FROM_POSTINGS`]).
fn keyword_from_postings_union(segments: &[Arc<Segment>]) -> Vec<u16> {
    let mut v: Vec<u16> = segments.iter().flat_map(|s| crate::segment::keyword_from_postings_fields(s)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// One segment's source column of a keyword field being merged (a merged
/// multi-valued column also takes the docs of single-valued segments).
enum KeywordSource<'a> {
    Single(KeywordReader<'a>),
    Multi(MultiKeywordReader<'a>),
}

impl<'a> KeywordSource<'a> {
    fn num_terms(&self) -> usize {
        match self {
            KeywordSource::Single(r) => r.num_terms(),
            KeywordSource::Multi(r) => r.num_terms(),
        }
    }

    fn term(&self, ord: u32) -> std::borrow::Cow<'a, [u8]> {
        match self {
            KeywordSource::Single(r) => r.term_bytes(ord).unwrap_or_default(),
            KeywordSource::Multi(r) => std::borrow::Cow::Borrowed(r.term(ord).unwrap_or_default().as_bytes()),
        }
    }

    /// Calls `f` on the ordinals of `doc`, ascending.
    fn for_each_ord(&self, doc: DocId, mut f: impl FnMut(u32)) {
        match self {
            KeywordSource::Single(r) => {
                let o = r.ord(doc);
                if o != MISSING_ORD {
                    f(o);
                }
            }
            KeywordSource::Multi(r) => r.ords(doc).for_each(f),
        }
    }
}

/// Keyword doc values of `field` merged by ordinal (Lucene's `OrdinalMap`):
/// the output term table is a k-way merge of the inputs' sorted tables, and
/// every doc's ordinals are remapped through a per-segment `old -> new`
/// table. That holds 4 bytes per input term and the output column, where
/// rebuilding a keyword column held a string and a doc list per unique value
/// to sort and re-encode (several GB for tens of millions of values). The
/// bytes match encoding the rebuilt column: terms only deleted docs used are
/// dropped, the remap is monotonic (per-doc ordinals stay ascending and
/// distinct), and no column is written when no live doc has a value.
/// `out`: [`TAG_KEYWORD`] or [`TAG_MULTI_KEYWORD`]; docs in output order
/// (input order, or a sorting merge's order: the term table and the remap
/// do not depend on doc order).
/// Returns the column's on-disk tag (a single-valued column may be written
/// sparse, see [`crate::codec::fastfields::write_keyword_ords`]) and body.
/// Fast-field columns a sorting merge builds at once ([`stream_fast_fields_sorted`]).
const FAST_COLUMN_BATCH: usize = 8;

/// [`merge_keyword_by_ordinal`] visits only the docs with a value when the
/// inputs hold at most one per this many output docs.
const SPARSE_KEYWORD_MERGE_RATIO: u64 = 4;

fn merge_keyword_by_ordinal(segments: &[Arc<Segment>], doc_map: &DocMap, field: u16, out: u8, version: u32, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>) -> Result<()> {
    use std::cmp::Reverse;
    let readers: Vec<_> = segments.iter().map(|s| s.fast_fields()).collect();
    let sources: Vec<Option<KeywordSource<'_>>> = readers
        .iter()
        .map(|r| match out {
            TAG_KEYWORD => r.get_keyword(field).map(KeywordSource::Single),
            _ => r.get_multi_keyword(field).map(KeywordSource::Multi).or_else(|| r.get_keyword(field).map(KeywordSource::Single)),
        })
        .collect();
    let live_docs = |i: usize| (0..segments[i].num_docs).filter(move |&old| doc_map.new_id(i, old).is_some());
    let num_docs = doc_map.num_docs();
    // A sparse single-valued field (most of github_archive's ~900 fields
    // fill under a tenth of the docs): only the docs with a value are
    // visited, in each input's doc order, as `(new doc, old ordinal)`, and
    // the output ordinals are scattered into place; walking every output
    // doc of every such column (three times) was most of a merge.
    let sparse: Option<(Vec<Vec<(DocId, u32)>>, crate::admission::Reservation)> = (out == TAG_KEYWORD)
        .then(|| sources.iter().map(|s| match s { None => Some(None), Some(KeywordSource::Single(r)) => Some(Some(r)), Some(KeywordSource::Multi(_)) => None }).collect::<Option<Vec<_>>>())
        .flatten()
        .and_then(|singles| {
            let present: u64 = singles.iter().flatten().map(|r| u64::from(r.present_count())).sum();
            if present.saturating_mul(SPARSE_KEYWORD_MERGE_RATIO) > u64::from(num_docs) {
                return None;
            }
            let bytes = present.saturating_mul(8).saturating_add(u64::from(num_docs).saturating_mul(4));
            let memory = crate::admission::shared().reserve(bytes, "sparse keyword merge").ok()?;
            let pairs = singles.iter().enumerate().map(|(i, r)| {
                let mut v = Vec::new();
                if let Some(r) = r {
                    r.for_each_present(|old, o| if let Some(new) = doc_map.new_id(i, old) { v.push((new, o)) });
                }
                v
            }).collect();
            Some((pairs, memory))
        });
    // Every output doc as `(segment, old)`, in output order: the sorted
    // order of a sorting merge (seal of an index-sorted core), else each
    // segment's live docs in turn.
    let for_each_doc = |f: &mut dyn FnMut(usize, DocId)| match doc_map.order.as_deref() {
        Some(order) => for &(i, old) in order { f(i, old) },
        None => for i in 0..segments.len() { for old in live_docs(i) { f(i, old) } },
    };
    // Ordinals live docs use, per segment.
    let used: Vec<Vec<u64>> = sources
        .iter()
        .enumerate()
        .map(|(i, src)| {
            let Some(src) = src else { return Vec::new() };
            let mut bits = vec![0u64; src.num_terms().div_ceil(64)];
            match &sparse {
                Some((pairs, _)) => pairs[i].iter().for_each(|&(_, o)| bits[o as usize / 64] |= 1 << (o % 64)),
                None => for old in live_docs(i) {
                    src.for_each_ord(old, |o| bits[o as usize / 64] |= 1 << (o % 64));
                },
            }
            bits
        })
        .collect();
    let next_used = |i: usize, from: u32| -> Option<u32> {
        let bits = &used[i];
        let mut w = from as usize / 64;
        let mut word = *bits.get(w)? & (!0u64 << (from % 64));
        loop {
            if word != 0 {
                return Some((w * 64) as u32 + word.trailing_zeros());
            }
            w += 1;
            word = *bits.get(w)?;
        }
    };
    let mut maps: Vec<Vec<u32>> = sources.iter().map(|s| s.as_ref().map_or_else(Vec::new, |s| vec![MISSING_ORD; s.num_terms()])).collect();
    let mut heap = std::collections::BinaryHeap::new();
    for (i, src) in sources.iter().enumerate() {
        if let (Some(src), Some(o)) = (src, next_used(i, 0)) {
            heap.push(Reverse((src.term(o), i, o)));
        }
    }
    // The merged term table is not held in memory (a geo_shape column holds
    // every doc's encoded shape: gigabytes per merge): the merge records
    // which input term each new term comes from, and the table is written
    // out in pieces after the ordinals are known (the column's tag
    // depends on them).
    let mut firsts: Vec<(u32, u32)> = Vec::new();
    let mut last: Option<std::borrow::Cow<'_, [u8]>> = None;
    while let Some(Reverse((term, i, o))) = heap.pop() {
        if last.as_deref() != Some(&*term) {
            firsts.push((i as u32, o));
            last = Some(term);
        }
        maps[i][o as usize] = firsts.len() as u32 - 1;
        if let Some(n) = next_used(i, o + 1) {
            heap.push(Reverse((sources[i].as_ref().unwrap().term(n), i, n)));
        }
    }
    let num_terms = firsts.len() as u32;
    if num_terms == 0 {
        return Ok(());
    }
    // `vint(num_terms)`, then each term as `vint(len) bytes`, then the
    // ordinals part: emitted as consecutive pieces of one column.
    // A single-valued column's table is front-coded from v30 (as the
    // column encoder writes it): the same bytes, written in pieces.
    let front_coded = out == TAG_KEYWORD && version >= crate::codec::FRONT_CODED_TERMS_SINCE;
    let packed_table = front_coded && version >= crate::codec::PACKED_KEYWORD_TABLES_SINCE;
    let emit_table = |tag: u8, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>| -> Result<()> {
        let mut buf = Vec::with_capacity(1 << 20);
        let mut fc = crate::codec::fastfields::FcTableWriter::new(packed_table);
        if front_coded {
            fc.header(&mut buf, num_terms as usize);
        } else {
            write_vint(num_terms as u64, &mut buf);
        }
        for &(i, o) in firsts.iter() {
            let term = sources[i as usize].as_ref().unwrap().term(o);
            if front_coded {
                fc.push(&mut buf, &term);
            } else {
                write_vint(term.len() as u64, &mut buf);
                buf.extend_from_slice(&term);
            }
            if buf.len() >= 1 << 20 {
                emit(field, tag, &buf)?;
                buf.clear();
            }
        }
        if front_coded {
            fc.finish(&mut buf);
        }
        if !buf.is_empty() {
            emit(field, tag, &buf)?;
        }
        Ok(())
    };
    if out == TAG_KEYWORD {
        let sparse_ok = version >= crate::codec::SPARSE_KEYWORD_SINCE;
        let mut body = Vec::with_capacity(num_docs as usize * ord_width(num_terms as usize, num_docs) + 16);
        let blocked_ok = version >= crate::codec::BLOCK_PACKED_SINCE;
        let blocked_missing_ok = version >= crate::codec::BLOCKED_MISSING_KEYWORD_SINCE;
        let constant_ok = version >= crate::codec::CONSTANT_KEYWORD_COLUMN_SINCE;
        if let Some((pairs, _memory)) = &sparse {
            let mut ords = vec![MISSING_ORD; num_docs as usize];
            for (i, p) in pairs.iter().enumerate() {
                for &(new, o) in p {
                    ords[new as usize] = maps[i][o as usize];
                }
            }
            let tag = crate::codec::fastfields::write_keyword_ords_vmc(&mut body, num_terms as usize, num_docs, sparse_ok, blocked_ok, blocked_missing_ok, constant_ok, &mut |f: &mut dyn FnMut(u32)| {
                ords.iter().for_each(|&o| f(o));
            });
            emit_table(tag, emit)?;
            return emit(field, tag, &body);
        }
        let tag = crate::codec::fastfields::write_keyword_ords_vmc(&mut body, num_terms as usize, num_docs, sparse_ok, blocked_ok, blocked_missing_ok, constant_ok, &mut |f: &mut dyn FnMut(u32)| {
            for_each_doc(&mut |i, old| {
                let mut new = MISSING_ORD;
                if let Some(src) = &sources[i] {
                    src.for_each_ord(old, |o| new = maps[i][o as usize]);
                }
                f(new);
            });
        });
        emit_table(tag, emit)?;
        return emit(field, tag, &body);
    }
    emit_table(TAG_MULTI_KEYWORD, emit)?;
    let total: u32 = sources
        .iter()
        .enumerate()
        .map(|(i, src)| src.as_ref().map_or(0, |src| live_docs(i).map(|old| { let mut n = 0u32; src.for_each_ord(old, |_| n += 1); n }).sum::<u32>()))
        .sum();
    let packed = version >= crate::codec::PACKED_MULTI_KEYWORD_SINCE;
    let mut body = Vec::with_capacity(if packed { num_docs as usize * 4 + total as usize * 2 + 32 } else { num_docs as usize * 8 + 16 });
    crate::codec::fastfields::write_multi_keyword_ords(
        &mut body,
        num_terms as usize,
        num_docs,
        total,
        packed,
        &mut |f: &mut dyn FnMut(u32)| {
            for_each_doc(&mut |i, old| {
                let mut n = 0u32;
                if let Some(src) = &sources[i] {
                    src.for_each_ord(old, |_| n += 1);
                }
                f(n);
            });
        },
        &mut |f: &mut dyn FnMut(u32)| {
            for_each_doc(&mut |i, old| {
                if let Some(src) = &sources[i] {
                    src.for_each_ord(old, |o| f(maps[i][o as usize]));
                }
            });
        },
    );
    emit(field, TAG_MULTI_KEYWORD, &body)
}

// ---- stored fields ------------------------------------------------------------

/// The merged store section: compressed chunks (borrowed from the inputs
/// or recompressed) followed by the chunk table.
struct MergedStore<'a> {
    chunks: Vec<StoreChunk<'a>>,
    table: Vec<u8>,
    /// The merged docs' `_id`s for the id column (format
    /// [`crate::codec::UID_COLUMN_SINCE`]+; their stored docs hold none):
    /// the id field and the column.
    uids: Option<(u16, Vec<u8>)>,
    // Retained chunk/table metadata; temporary merge buffers hold their own guards.
    _metadata: Option<crate::admission::Reservation>,
}

enum StoreChunk<'a> {
    Copied(&'a [u8]),
    Rewritten(crate::admission::Allocation<Vec<u8>>),
    Mapped(FileSlice),
}

impl<'a> MergedStore<'a> {
    fn pieces(&self) -> impl Iterator<Item = &[u8]> {
        self.chunks.iter().map(|c| match c {
            StoreChunk::Copied(b) => *b,
            StoreChunk::Rewritten(b) => b.as_slice(),
            StoreChunk::Mapped(b) => b.as_slice(),
        }).chain(std::iter::once(self.table.as_slice()))
    }
}

fn store_plan_bytes(chunks: usize) -> Result<u64> {
    // Includes simultaneous plan, parallel result collector, final chunk
    // descriptors, input/output tables and table encoder growth overlap.
    // 128 bytes per chunk covers 3 maximal u64 varints and Vec reallocations.
    let per_chunk = 2 * std::mem::size_of::<(usize, usize, bool)>()
        + 4 * std::mem::size_of::<(StoreChunk<'_>, usize)>()
        + 2 * std::mem::size_of::<StoreChunk<'_>>()
        + 4 * std::mem::size_of::<(usize, usize)>()
        + std::mem::size_of::<(usize, usize, usize)>() + 128;
    (chunks as u64).checked_mul(per_chunk as u64).and_then(|n| n.checked_add(1024))
        .ok_or_else(|| RusticError::Memory("merge store plan size overflow".into()))
}

#[cfg(test)]
fn stored_rewrite_memory(budget: &Arc<crate::admission::Budget>, raw_bytes: usize) -> Result<crate::admission::Reservation> {
    stored_rewrite_memory_for(budget, raw_bytes, false)
}

/// Scratch of a zstd compression context at [`crate::codec::store::BEST_ZSTD_LEVEL`]
/// for a chunk of at most [`crate::codec::store::BEST_CHUNK_MAX_BYTES`] plus one
/// doc (window, hash and chain tables, sized down to the input).
const ZSTD_SCRATCH: u64 = 4 << 20;

/// Memory of rewriting one stored chunk of `raw_bytes` (compressed with
/// zstd when `zstd`).
fn stored_rewrite_memory_for(budget: &Arc<crate::admission::Budget>, raw_bytes: usize, zstd: bool) -> Result<crate::admission::Reservation> {
    // LZ4 worst-case compressed size < n + n/255 + 32 (zstd's bound is
    // similar). Four times n covers raw+kept+encoded, with fixed
    // compressor/table scratch separate.
    let scratch = if zstd { ZSTD_SCRATCH } else { 128 << 10 };
    let bytes = (raw_bytes as u64).checked_mul(4).and_then(|n| n.checked_add(scratch))
        .ok_or_else(|| RusticError::Memory("merge store workspace overflow".into()))?;
    budget.reserve(bytes, "merge stored rewrite")
}

use crate::codec::uid_column::{Cursor as IdCursor, IdList};

/// How a store merge treats the docs' `_id`s: from format
/// [`crate::codec::UID_COLUMN_SINCE`] (every input chunk compact) they
/// move into the output's id column ([`crate::codec::uid_column`]), taken
/// out of the stored docs that hold them (binary ids) or read off an
/// input's column; below it, an input column's ids go back into the docs.
struct IdPlan {
    out: bool,
    /// Per input: the id field of its column.
    fields: Vec<Option<u16>>,
    /// The id field (an input column's, or the first binary id met).
    field: Option<u16>,
    seen: std::sync::OnceLock<u16>,
}

impl IdPlan {
    fn new(segments: &[Arc<Segment>], version: u32) -> Self {
        let fields: Vec<Option<u16>> = segments.iter().map(|s| s.uid_column_field()).collect();
        let compact = segments.iter().all(|s| s.store().is_ok_and(|r| (0..r.num_chunks()).all(|c| r.chunk_compact(c))));
        IdPlan { out: version >= crate::codec::UID_COLUMN_SINCE && compact, field: fields.iter().flatten().next().copied(), fields, seen: std::sync::OnceLock::new() }
    }

    /// Whether a chunk of input `i` (holding binary ids: `uid_chunk`) is
    /// rewritten for its ids.
    fn rewrites(&self, i: usize, uid_chunk: bool) -> bool {
        if self.out { self.fields[i].is_none() && uid_chunk } else { self.fields[i].is_some() }
    }

    /// A reader of input `i`'s id column.
    fn cursor<'s>(&self, segments: &'s [Arc<Segment>], i: usize) -> Option<IdCursor<'s>> {
        self.fields[i]?;
        segments[i].uid_column().map(IdCursor::new)
    }

    fn saw(&self, field: u16) {
        let _ = self.seen.set(field);
    }

    /// The output's id column of the merged docs' `ids`.
    fn finish(&self, ids: &IdList, budget: &Arc<crate::admission::Budget>) -> Result<Option<(u16, Vec<u8>)>> {
        if !self.out || !ids.any() {
            return Ok(None);
        }
        let Some(field) = self.field.or_else(|| self.seen.get().copied()) else { return Ok(None) };
        // The id references and the encoded column (about 2 bytes a doc).
        let _memory = budget.reserve(ids.len() as u64 * 24 + 4096, "merge id column")?;
        Ok(Some((field, crate::codec::uid_column::encode(&ids.refs()))))
    }
}

fn merge_store<'a>(segments: &'a [Arc<Segment>], doc_map: &DocMap, version: u32) -> Result<MergedStore<'a>> {
    merge_store_with_budget(segments, doc_map, version, &crate::admission::shared())
}

/// Chunks are copied verbatim, flags and all, unless they hold a deleted
/// doc, are compact while `version` predates
/// [`crate::codec::COMPACT_STORED_SINCE`] (then they are re-encoded plain),
/// or are zstd while `version` predates [`crate::codec::ZSTD_STORED_SINCE`]
/// (then they are recompressed LZ4). A rewritten chunk keeps its zstd
/// compression otherwise.
fn merge_store_with_budget<'a>(segments: &'a [Arc<Segment>], doc_map: &DocMap, version: u32, budget: &Arc<crate::admission::Budget>) -> Result<MergedStore<'a>> {
    let compact_out = version >= crate::codec::COMPACT_STORED_SINCE;
    let zstd_out = version >= crate::codec::ZSTD_STORED_SINCE;
    let uid_out = version >= crate::codec::UID_STORED_SINCE;
    let ids = IdPlan::new(segments, version);
    use rayon::prelude::*;
    // Chunk plans in output order: (segment, chunk) and whether it needs a
    // rewrite (it holds a deleted doc).
    let reader_bytes = (segments.len() as u64).checked_mul((std::mem::size_of::<StoreReader<'a>>() * 2) as u64)
        .ok_or_else(|| RusticError::Memory("merge reader metadata overflow".into()))?;
    let mut metadata = budget.reserve(reader_bytes, "merge store readers")?;
    let mut readers = Vec::with_capacity(segments.len());
    for segment in segments { readers.push(segment.store()?); }
    let chunks = readers.iter().try_fold(0usize, |n, reader| n.checked_add(reader.num_chunks()))
        .ok_or_else(|| RusticError::Memory("merge chunk count overflow".into()))?;
    metadata.grow(store_plan_bytes(chunks)?, "merge store plan and output metadata")?;
    let mut plan: Vec<(usize, usize, bool)> = Vec::with_capacity(chunks);
    for (i, r) in readers.iter().enumerate() {
        let row = &doc_map.remap[i];
        for c in 0..r.num_chunks() {
            let (start, n) = r.chunk_docs(c);
            let dirty = (doc_map.base(i).is_none() && row[start as usize..(start + n) as usize].contains(&DELETED))
                || (!compact_out && r.chunk_compact(c))
                || (!zstd_out && r.chunk_zstd(c))
                || (!uid_out && r.chunk_compact(c) && crate::codec::store::chunk_is_uid(r.raw_chunk(c)))
                || ids.rewrites(i, r.chunk_compact(c) && crate::codec::store::chunk_is_uid(r.raw_chunk(c)));
            plan.push((i, c, dirty));
        }
    }
    let rewrite = |&(i, c, dirty): &(usize, usize, bool)| -> Result<(StoreChunk<'a>, usize, IdList)> {
        let r = &readers[i];
        let (start, n) = r.chunk_docs(c);
        let mut chunk_ids = IdList::default();
        let mut cursor = ids.cursor(segments, i);
        if !dirty {
            // Every doc is live: the ids of the input's column (or none).
            if ids.out {
                for d in start..start + n {
                    chunk_ids.push(match &mut cursor { Some(cur) => cur.get(d)?, None => None });
                }
            }
            return Ok((StoreChunk::Copied(r.raw_chunk(c)), n as usize, chunk_ids));
        }
        merge_pace(PACE_UNITS / 8);
        // Raw input, kept bytes, compression output and compressor scratch
        // coexist. Read the LZ4 size header before any decoder allocation.
        let raw_bytes = r.decoded_chunk_bytes(c).ok_or_else(|| RusticError::Corrupted("stored chunk size missing".into()))?;
        let zstd = zstd_out && r.chunk_zstd(c);
        let mut memory = stored_rewrite_memory_for(budget, raw_bytes, zstd)?;
        let raw = r.decompress(c)?;
        let row = &doc_map.remap[i];
        let downgrade = !compact_out && r.chunk_compact(c);
        let mut kept = Vec::with_capacity(if downgrade { raw.len() * 2 } else { raw.len() });
        let mut live = 0usize;
        let mut pos = 0usize;
        for d in 0..n {
            let at = pos;
            let len = read_vint(&raw, &mut pos)? as usize;
            pos += len;
            if pos > raw.len() {
                return Err(RusticError::Corrupted("stored doc out of range".into()));
            }
            // A segment without deletions has no row (its base maps it).
            if doc_map.base(i).is_some() || row[(start + d) as usize] != DELETED {
                let body = &raw[pos - len..pos];
                // The id: into the output's column (taken out of the doc
                // or read off the input's), or back into the doc.
                let joined;
                let body = match (ids.out, &mut cursor) {
                    (true, Some(cur)) => { chunk_ids.push(cur.get(start + d)?); body }
                    (true, None) if r.chunk_compact(c) => {
                        let mut framed = Vec::with_capacity(len + 4);
                        let found = crate::codec::store::split_uid(body, &mut framed)?;
                        chunk_ids.push(found.as_ref().map(|(_, id)| id.as_slice()));
                        if let Some((f, _)) = found { ids.saw(f); }
                        kept.extend_from_slice(&framed);
                        live += 1;
                        continue;
                    }
                    (true, None) => { chunk_ids.push(None); body }
                    (false, Some(cur)) => match cur.get(start + d)? {
                        Some(id) => {
                            let mut framed = Vec::with_capacity(len + id.len() + 8);
                            crate::codec::store::join_uid(body, ids.fields[i].unwrap_or(0), id, &mut framed)?;
                            let mut p = 0;
                            let l = read_vint(&framed, &mut p)? as usize;
                            joined = framed[p..p + l].to_vec();
                            &joined[..]
                        }
                        None => body,
                    },
                    (false, None) => body,
                };
                if downgrade {
                    crate::codec::store::append_compact_as_plain(body, &mut kept)?;
                } else {
                    write_vint(body.len() as u64, &mut kept);
                    kept.extend_from_slice(body);
                }
                live += 1;
            }
        }
        if live == 0 {
            drop(kept);
            drop(raw);
            memory.shrink(memory.bytes());
            return Ok((StoreChunk::Rewritten(crate::admission::Allocation::new(Vec::new(), memory)), 0, chunk_ids));
        }
        let compact = r.chunk_compact(c) && !downgrade;
        // Binary ids written back as keywords below v31.
        let kept = if compact && !uid_out && crate::codec::store::chunk_is_uid(r.raw_chunk(c)) { crate::codec::store::compact_chunk_without_uid(&kept)? } else { kept };
        let encoded = crate::codec::store::compress_chunk_flagged(&kept, compact, zstd, uid_out);
        drop(kept);
        drop(raw);
        // Keep the surviving compressed output admitted until its final file
        // write, including while parallel sibling tasks are still running.
        memory.shrink(memory.bytes().saturating_sub(encoded.capacity() as u64));
        Ok((StoreChunk::Rewritten(crate::admission::Allocation::new(encoded, memory)), live, chunk_ids))
    };
    let admission = crate::admission::Context::capture();
    let any_dirty = plan.iter().any(|p| p.2);
    let built: Vec<(StoreChunk<'a>, usize, IdList)> = if any_dirty {
        let pacing = PACING_ENABLED.with(|p| p.get());
        plan.par_iter().map(|part| admission.run(|| with_pacing(pacing, || rewrite(part)))).collect::<Result<_>>()?
    } else {
        plan.iter().map(rewrite).collect::<Result<_>>()?
    };
    let mut chunks = Vec::with_capacity(built.len());
    let mut table = Vec::with_capacity(built.len());
    let mut merged_ids = IdList::default();
    for (chunk, docs, chunk_ids) in built {
        merged_ids.extend(&chunk_ids);
        if docs == 0 {
            continue;
        }
        let len = match &chunk {
            StoreChunk::Copied(b) => b.len(),
            StoreChunk::Rewritten(b) => b.len(),
            StoreChunk::Mapped(b) => b.len(),
        };
        table.push((len, docs));
        chunks.push(chunk);
    }
    let encoded_table = encode_store_table(&table);
    drop(table);
    drop(plan);
    drop(readers);
    let uids = ids.finish(&merged_ids, budget)?;
    drop(merged_ids);
    let retained = (chunks.capacity() * std::mem::size_of::<StoreChunk<'a>>()) as u64 + encoded_table.capacity() as u64;
    metadata.shrink(metadata.bytes().saturating_sub(retained));
    Ok(MergedStore { chunks, table: encoded_table, uids, _metadata: Some(metadata) })
}

/// [`merge_store`] for a sorting merge: docs are re-chunked in output
/// order. Each input chunk is decompressed once while its segment's docs
/// are consumed in order (inputs that are themselves sorted), at worst once
/// per segment when they are not.
#[cfg(test)]
fn merge_store_sorted<'a>(dir: &dyn Directory, new_id: u32, segments: &'a [Arc<Segment>], doc_map: &DocMap) -> Result<MergedStore<'a>> {
    merge_store_sorted_with_budget(dir, new_id, segments, doc_map, &crate::admission::shared())
}

#[cfg(test)]
fn merge_store_sorted_with_budget<'a>(dir: &dyn Directory, new_id: u32, segments: &'a [Arc<Segment>], doc_map: &DocMap,
    budget: &Arc<crate::admission::Budget>) -> Result<MergedStore<'a>> {
    merge_store_sorted_versioned(dir, new_id, segments, doc_map, crate::codec::VERSION, budget)
}

/// [`merge_store_sorted_with_budget`] writing segment format `version`: when
/// an input holds zstd chunks ([`crate::codec::store::StoredCompression::Best`])
/// and `version` allows them, the output is re-chunked and compressed the
/// same way, else as LZ4 chunks of the default size.
fn merge_store_sorted_versioned<'a>(dir: &dyn Directory, new_id: u32, segments: &'a [Arc<Segment>], doc_map: &DocMap, version: u32,
    budget: &Arc<crate::admission::Budget>) -> Result<MergedStore<'a>> {
    use crate::codec::store::StoredCompression;
    let order = doc_map.order.as_deref().unwrap_or(&[]);
    // Reader descriptors, per-segment traversal state, and the bounded
    // 64-chunk raw/compressed parallel collectors (including overlap).
    let initial = (segments.len() as u64).checked_mul(256).and_then(|n| n.checked_add(32768))
        .ok_or_else(|| RusticError::Memory("sorted merge metadata size overflow".into()))?;
    let mut metadata = budget.reserve(initial, "sorted merge metadata")?;
    let mut readers: Vec<StoreReader<'a>> = Vec::with_capacity(segments.len());
    for segment in segments { readers.push(segment.store()?); }
    let mode = if readers.iter().any(|r| r.any_zstd()) { StoredCompression::Best } else { StoredCompression::Fast };
    // Small chunks for large docs only where fetches want them (vectors).
    let small_chunks = segments.iter().any(|s| s.has_vector_data());
    let zstd = mode.zstd_at(version);
    let compact_out = version >= crate::codec::COMPACT_STORED_SINCE;
    let uid_out = version >= crate::codec::UID_STORED_SINCE;
    let ids = IdPlan::new(segments, version);
    let mut cursors: Vec<Option<IdCursor<'a>>> = (0..segments.len()).map(|i| ids.cursor(segments, i)).collect();
    let mut merged_ids = IdList::default();
    let mut traversal = 0u64;
    for reader in &readers {
        traversal = traversal.checked_add(u64::from(reader.num_docs()) * 4)
            .and_then(|n| n.checked_add((reader.num_chunks() as u64).checked_mul(std::mem::size_of::<Option<crate::admission::Allocation<(Vec<u8>, Vec<(usize, usize)>)>>>() as u64)?))
            .ok_or_else(|| RusticError::Memory("sorted merge traversal size overflow".into()))?;
    }
    metadata.grow(traversal, "sorted merge traversal metadata")?;
    // Per segment: chunk of each doc, and the decompressed chunks with each
    // doc's entry range (`vint len, bytes`).
    let chunk_of: Vec<Vec<u32>> = readers
        .iter()
        .map(|r| {
            let mut v = vec![0u32; r.num_docs() as usize];
            for c in 0..r.num_chunks() {
                let (start, n) = r.chunk_docs(c);
                for d in start..start + n {
                    v[d as usize] = c as u32;
                }
            }
            v
        })
        .collect();
    // Whether a segment's docs are consumed in ascending order (then a
    // chunk can be released once passed).
    let mut last: Vec<i64> = vec![-1; segments.len()];
    let mut monotonic = vec![true; segments.len()];
    for &(i, old) in order {
        if (old as i64) < last[i] {
            monotonic[i] = false;
        }
        last[i] = old as i64;
    }
    type Decoded = crate::admission::Allocation<(Vec<u8>, Vec<(usize, usize)>)>;
    let mut cache: Vec<Vec<Option<Decoded>>> = readers.iter().map(|r| (0..r.num_chunks()).map(|_| None).collect()).collect();
    let mut cur_chunk: Vec<Option<usize>> = vec![None; segments.len()];
    static NEXT_STORE_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT_STORE_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scratch = MergeScratch { dir, name: format!(".merge-store-{new_id}-{}-{sequence}.tmp", std::process::id()) };
    let mut output = dir.create_scratch_output(&scratch.name)?;
    let mut table = Vec::new();
    // Parallel compression uses a bounded batch instead of retaining the whole
    // uncompressed merged store. Bodies are spooled as batches complete.
    let mut raw_chunks: Vec<crate::admission::Allocation<Vec<u8>>> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut cur = StoreBuffer::new()?;
    let mut n_cur = 0u32;
    // Docs of the open chunk over SIZED_DOC_BYTES (the chunk rule's average).
    let mut sized_cur = 0usize;
    for &(i, old) in order {
        merge_pace(16);
        let c = chunk_of[i][old as usize] as usize;
        if monotonic[i] {
            if let Some(prev) = cur_chunk[i] {
                if prev != c {
                    cache[i][prev] = None;
                }
            }
            cur_chunk[i] = Some(c);
        }
        if cache[i][c].is_none() {
            let (start, n) = readers[i].chunk_docs(c);
            let raw_bytes = readers[i].decoded_chunk_bytes(c).ok_or_else(|| RusticError::Corrupted("stored chunk size missing".into()))?;
            let bytes = (raw_bytes as u64).checked_add(u64::from(n) * std::mem::size_of::<(usize, usize)>() as u64)
                .ok_or_else(|| RusticError::Memory("sorted merge decoded size overflow".into()))?;
            let decoded_memory = crate::admission::shared().reserve(bytes, "sorted merge decoded cache")?;
            let raw = readers[i].decompress(c)?;
            let _ = start;
            let mut ranges = Vec::with_capacity(n as usize);
            let mut pos = 0usize;
            for _ in 0..n {
                let at = pos;
                let len = read_vint(&raw, &mut pos)? as usize;
                pos = pos.checked_add(len).ok_or_else(|| RusticError::Corrupted("stored doc length overflow".into()))?;
                if pos > raw.len() {
                    return Err(RusticError::Corrupted("stored doc out of range".into()));
                }
                ranges.push((at, pos));
            }
            cache[i][c] = Some(crate::admission::Allocation::new((raw, ranges), decoded_memory));
        }
        let (raw, ranges) = &**cache[i][c].as_ref().expect("decoded above");
        let (start, _) = readers[i].chunk_docs(c);
        let (a, b) = ranges[(old - start) as usize];
        // Re-chunked docs of several inputs share output chunks, written in
        // the compact encoding (binary ids from v31) when the output format
        // has it, as unsorted merges and seals write them; else in bincode.
        let in_compact = readers[i].chunk_compact(c);
        let in_uid = in_compact && crate::codec::store::chunk_is_uid(readers[i].raw_chunk(c));
        let mut p = a;
        let len = read_vint(raw, &mut p)? as usize;
        let body = &raw[p..p + len];
        // The id: into the output's column (taken out of the doc or read
        // off the input's), or back into the doc.
        let joined;
        let (body, a, b, raw): (&[u8], usize, usize, &[u8]) = match (ids.out, &mut cursors[i]) {
            (true, Some(cur)) => { merged_ids.push(cur.get(old)?); (body, a, b, raw) }
            (true, None) if in_compact => {
                let mut framed = Vec::with_capacity(len + 4);
                let found = crate::codec::store::split_uid(body, &mut framed)?;
                merged_ids.push(found.as_ref().map(|(_, id)| id.as_slice()));
                if let Some((f, _)) = found { ids.saw(f); }
                joined = framed;
                let mut q = 0;
                let l = read_vint(&joined, &mut q)? as usize;
                (&joined[q..q + l], 0, joined.len(), &joined[..])
            }
            (true, None) => { merged_ids.push(None); (body, a, b, raw) }
            (false, Some(cur)) => match cur.get(old)? {
                Some(id) => {
                    let mut framed = Vec::with_capacity(len + id.len() + 8);
                    crate::codec::store::join_uid(body, ids.fields[i].unwrap_or(0), id, &mut framed)?;
                    joined = framed;
                    let mut q = 0;
                    let l = read_vint(&joined, &mut q)? as usize;
                    (&joined[q..q + l], 0, joined.len(), &joined[..])
                }
                None => (body, a, b, raw),
            },
            (false, None) => (body, a, b, raw),
        };
        let len = body.len();
        let before = cur.bytes.len();
        if compact_out && in_compact && (uid_out || !in_uid) {
            append_store_bytes(&mut cur.bytes, &mut cur.memory, &raw[a..b])?;
        } else if compact_out {
            // A bincode doc, or binary ids written below v31: re-encoded.
            let doc = crate::codec::store::decode_doc(body, in_compact)?;
            let mut one = Vec::with_capacity(len + 16);
            crate::codec::store::encode_doc_compact(doc.fields.iter(), &mut one);
            let mut framed = Vec::with_capacity(one.len() + 5);
            crate::codec::varint::write_vint(one.len() as u64, &mut framed);
            framed.extend_from_slice(&one);
            append_store_bytes(&mut cur.bytes, &mut cur.memory, &framed)?;
        } else if in_compact {
            let mut framed = Vec::with_capacity(len * 2 + 16);
            crate::codec::store::append_compact_as_plain(body, &mut framed)?;
            append_store_bytes(&mut cur.bytes, &mut cur.memory, &framed)?;
        } else {
            append_store_bytes(&mut cur.bytes, &mut cur.memory, &raw[a..b])?;
        }
        n_cur += 1;
        if cur.bytes.len() - before > crate::codec::store::SIZED_DOC_BYTES + 1 {
            sized_cur += 1;
        }
        if mode.chunk_full_for(n_cur as usize, sized_cur, cur.bytes.len(), small_chunks) {
            raw_chunks.push(std::mem::replace(&mut cur, StoreBuffer::new()?).into_allocation());
            counts.push(n_cur);
            n_cur = 0;
            sized_cur = 0;
            // Best chunks are 15x larger: a smaller batch bounds the
            // uncompressed bytes held at once.
            if raw_chunks.len() >= if zstd { 8 } else { 64 } {
                flush_store_batch(&mut raw_chunks, &mut counts, &mut table, &mut metadata, output.as_mut(), zstd, compact_out, uid_out)?;
            }
        }
    }
    if n_cur > 0 {
        raw_chunks.push(cur.into_allocation());
        counts.push(n_cur);
    } else {
        drop(cur);
    }
    flush_store_batch(&mut raw_chunks, &mut counts, &mut table, &mut metadata, output.as_mut(), zstd, compact_out, uid_out)?;
    output.finish()?;
    drop(cursors);
    let uids = ids.finish(&merged_ids, budget)?;
    drop(merged_ids);
    let bodies = dir.open_read(&scratch.name)?;
    drop(scratch);
    // Encoder builds a triple table plus a geometrically growing varint
    // output. Reserve their overlap before encoding, retaining final bytes.
    let encode_bytes = (table.len() as u64).checked_mul(128).and_then(|n| n.checked_add(1024))
        .ok_or_else(|| RusticError::Memory("sorted merge table size overflow".into()))?;
    metadata.grow(encode_bytes, "sorted merge table encoding")?;
    let encoded = encode_store_table(&table);
    let chunks = vec![StoreChunk::Mapped(bodies)];
    drop(table); drop(cache); drop(chunk_of); drop(cur_chunk);
    drop(last); drop(monotonic); drop(readers); drop(raw_chunks); drop(counts);
    let retained = encoded.capacity() as u64 + (chunks.capacity() * std::mem::size_of::<StoreChunk<'a>>()) as u64;
    metadata.shrink(metadata.bytes().saturating_sub(retained));
    Ok(MergedStore { chunks, table: encoded, uids, _metadata: Some(metadata) })
}

/// Column streaming for a sorting merge: columns are filled in output
/// doc order (multi-valued columns need ascending docs).
fn stream_fast_fields_sorted(segments: &[Arc<Segment>], doc_map: &DocMap, merged_norms: &MergedNorms, admit: MergeAdmission,
    version: u32, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>) -> Result<()> {
    let _metadata = fast_merge_metadata(segments)?;
    let num_docs = doc_map.num_docs();
    let order = doc_map.order.as_deref().unwrap_or(&[]);
    let readers: Vec<_> = segments.iter().map(|s| s.fast_fields()).collect();
    let mut columns: Vec<(u16, u8)> = Vec::new();
    for r in &readers {
        for (field, tag) in r.fields() {
            if !columns.iter().any(|(f, t)| *f == field && *t == tag) {
                columns.push((field, tag));
            }
        }
    }
    let multi_of = |tag: u8| -> Option<u8> {
        match tag {
            1 => Some(7),
            2 => Some(8),
            3 => Some(9),
            5 => Some(10),
            _ => None,
        }
    };
    let promote = |field: u16, tag: u8| -> u8 {
        if tag == TAG_FEATURES {
            return 10;
        }
        match multi_of(tag) {
            Some(m) if columns.iter().any(|&(f, t)| f == field && t == m) => m,
            _ => tag,
        }
    };
    let mut groups: Vec<(u16, u8)> = columns.iter().map(|&(f, t)| (f, promote(f, t))).collect();
    groups.sort_unstable();
    groups.dedup();
    let from_postings = keyword_from_postings_union(segments);
    // Each column is independent: they are built a few at a time on the
    // merge pool (the merge's other stages leave its threads idle while
    // one thread walked ~900 columns of github_archive), each into its own
    // buffer, then written in order (the same bytes as one at a time).
    let one = |field: u16, out: u8, emit: &mut dyn FnMut(u16, u8, &[u8]) -> Result<()>| -> Result<()> {
            if from_postings.contains(&field) && (out == TAG_KEYWORD || out == TAG_MULTI_KEYWORD) {
                return Ok(());
            }
            merge_pace(num_docs as u64);
            // Keywords merge by ordinal here too (the term table and the remap
            // do not depend on doc order): rebuilding a column re-interned every
            // doc's string, most of an index-sorted refresh's seal.
            // Feature columns (`sparse_vector` doc values, promoted to the
            // multi-valued keyword group) hold no term table to remap: their
            // packed pairs merge in the sorted doc order, or, beside a plain
            // keyword column of the field (or below v29), rebuild per doc.
            let features: Vec<_> = readers.iter().map(|r| r.get_features(field)).collect();
            if out == TAG_MULTI_KEYWORD && features.iter().any(Option::is_some) {
                let _column_memory = fast_column_memory(segments, doc_map, field, out, false, &crate::admission::shared(), admit)?;
                let plain = readers.iter().any(|r| r.get_keyword(field).is_some() || r.get_multi_keyword(field).is_some());
                if !plain && version >= crate::codec::FEATURES_COLUMN_SINCE {
                    let mut new_of: Vec<Vec<Option<DocId>>> = segments.iter().map(|s| vec![None; s.num_docs as usize]).collect();
                    for (new, &(i, old)) in order.iter().enumerate() {
                        if let Some(slot) = new_of.get_mut(i).and_then(|v| v.get_mut(old as usize)) {
                            *slot = Some(new as DocId);
                        }
                    }
                    let new_id = |i: usize, old: DocId| new_of.get(i).and_then(|v| v.get(old as usize).copied().flatten());
                    if let Some(body) = crate::codec::fastfields::merge_features(&features, &new_id, num_docs) {
                        emit(field, TAG_FEATURES, &body)?;
                    }
                    return Ok(());
                }
                let mut ff = FastFields::new(0);
                for (new, &(i, old)) in order.iter().enumerate() {
                    let r = &readers[i];
                    let mut vals: Vec<String> = r.get_keyword(field).and_then(|c| c.get(old)).map(|v| v.into_owned()).into_iter().collect();
                    if let Some(c) = r.get_multi_keyword(field) {
                        vals.extend(c.values(old).into_iter().map(str::to_string));
                    }
                    if let Some(c) = &features[i] {
                        vals.extend(c.values(old));
                    }
                    for v in vals {
                        ff.multi_keyword_column_mut(field).push(new as DocId, &v);
                    }
                }
                emit_fast_columns(&ff, num_docs, version, emit)?;
                return Ok(());
            }
            let by_ordinal = out == TAG_KEYWORD || out == TAG_MULTI_KEYWORD;
            let _column_memory = fast_column_memory(segments, doc_map, field, out, by_ordinal, &crate::admission::shared(), admit)?;
            if by_ordinal {
                merge_keyword_by_ordinal(segments, doc_map, field, out, version, emit)?;
                return Ok(());
            }
            let mut ff = FastFields::new(0);
            carry_skipper(&readers, &mut ff, field);

            if out == 6 {
                if let Some((bytes, total)) = merged_norms.get(&field).filter(|(b, _)| !b.is_empty()) {
                    let col = ff.norms_mut(field);
                    for (new, &b) in bytes.iter().enumerate() {
                        col.set_byte(new as DocId, b);
                    }
                    col.set_total(*total);
                }
                emit_fast_columns(&ff, num_docs, version, emit)?;
                return Ok(());
            }
            // Column readers opened once per segment (not per doc).
            macro_rules! sorted_numeric {
                ($get:ident, $get_multi:ident, $single:ident, $multi:ident) => {{
                    let cols: Vec<_> = readers.iter().map(|r| (r.$get(field), r.$get_multi(field))).collect();
                    // A sparse single-valued field: its values from each input's
                    // docs with one, put in output order (not a read of every
                    // output doc of every input).
                    let present: u64 = cols.iter().map(|(s, m)| match (s, m) { (None, None) => 0, (Some(c), None) => c.value_doc_count(), _ => u64::MAX }).fold(0u64, u64::saturating_add);
                    if !matches!(out, 7..=9) && present.saturating_mul(SPARSE_KEYWORD_MERGE_RATIO) <= u64::from(num_docs) {
                        let mut pairs = Vec::with_capacity(present as usize);
                        for (i, (single, _)) in cols.iter().enumerate() {
                            let Some(c) = single else { continue };
                            for old in c.present_docs() {
                                if let (Some(new), Some(v)) = (doc_map.new_id(i, old), c.get(old)) {
                                    pairs.push((new, v));
                                }
                            }
                        }
                        if !pairs.is_empty() {
                            pairs.sort_unstable_by_key(|&(new, _)| new);
                            let target = ff.$single(field);
                            for (new, v) in pairs { target.set(new, v); }
                        }
                        emit_fast_columns(&ff, num_docs, version, emit)?;
                        return Ok(());
                    }
                    // Created at the first value, as a rebuilt column is (no
                    // column when no output doc has one).
                    let has = |i: usize, old: DocId| cols[i].0.as_ref().is_some_and(|c| c.get(old).is_some())
                        || cols[i].1.as_ref().is_some_and(|c| c.values(old).next().is_some());
                    if let Some(start) = order.iter().position(|&(i, old)| has(i, old)) {
                        if matches!(out, 7..=9) {
                            let target = ff.$multi(field);
                            for (new, &(i, old)) in order.iter().enumerate().skip(start) {
                                let (single, multi) = &cols[i];
                                if let Some(v) = single.as_ref().and_then(|c| c.get(old)) { target.push(new as DocId, v); }
                                if let Some(c) = multi { for v in c.values(old) { target.push(new as DocId, v); } }
                            }
                        } else {
                            let target = ff.$single(field);
                            for (new, &(i, old)) in order.iter().enumerate().skip(start) {
                                if let Some(v) = cols[i].0.as_ref().and_then(|c| c.get(old)) { target.set(new as DocId, v); }
                            }
                        }
                    }
                }};
            }
            match out {
                1 | 7 => sorted_numeric!(get_i64, get_multi_i64, i64_column_mut, multi_i64_column_mut),
                2 | 8 => sorted_numeric!(get_u64, get_multi_u64, u64_column_mut, multi_u64_column_mut),
                3 | 9 => sorted_numeric!(get_f64, get_multi_f64, f64_column_mut, multi_f64_column_mut),
                4 => {
                    let cols: Vec<_> = readers.iter().map(|r| r.get_bool(field)).collect();
                    // The column looked up once, not per doc (created as before:
                    // when some output doc comes from an input that has one).
                    if order.iter().any(|&(i, _)| cols[i].is_some()) {
                        let target = ff.bool_column_mut(field);
                        if cols.iter().flatten().any(|c| !c.presence_known()) {
                            target.set_presence_unknown();
                        }
                        for (new, &(i, old)) in order.iter().enumerate() {
                            if let Some(c) = &cols[i] {
                                if c.has_value(old) {
                                    target.set(new as DocId, c.get(old));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            emit_fast_columns(&ff, num_docs, version, emit)?;
        Ok(())
    };
    let enabled = PACING_ENABLED.with(|p| p.get());
    let admission = crate::admission::Context::capture();
    for batch in groups.chunks(FAST_COLUMN_BATCH) {
        use rayon::prelude::*;
        let built: Vec<Result<Vec<(u16, u8, crate::admission::Allocation<Vec<u8>>)>>> = batch.par_iter().map(|&(field, out)| admission.run(|| with_pacing(enabled, || {
            let mut pieces = Vec::new();
            one(field, out, &mut |f, t, b| {
                let memory = crate::admission::shared().reserve((b.len() as u64).saturating_add(64), "buffered fast column")?;
                pieces.push((f, t, crate::admission::Allocation::new(b.to_vec(), memory)));
                Ok(())
            })?;
            Ok(pieces)
        }))).collect();
        for pieces in built {
            for (f, t, b) in pieces? {
                emit(f, t, &b)?;
            }
        }
    }
    Ok(())
}

/// Consume raw chunks so each allocation is released as compression finishes.
// Field drop order matters on every `?`/unwind path: free bytes before
// returning their reservation to a concurrently allocating merge worker.
struct StoreBuffer {
    bytes: Vec<u8>,
    memory: crate::admission::Reservation,
}
impl StoreBuffer {
    fn new() -> Result<Self> {
        Ok(Self { bytes: Vec::new(), memory: crate::admission::shared().reserve(0, "sorted merge raw chunk")? })
    }
    fn into_allocation(self) -> crate::admission::Allocation<Vec<u8>> {
        crate::admission::Allocation::new(self.bytes, self.memory)
    }
}

/// Admit the old/new buffer overlap before Vec reallocation. An allocation
/// refusal leaves both the bytes and existing reservation unchanged.
fn append_store_bytes(out: &mut Vec<u8>, memory: &mut crate::admission::Reservation, bytes: &[u8]) -> Result<()> {
    let needed = out.len().checked_add(bytes.len()).ok_or_else(|| RusticError::Memory("sorted merge chunk size overflow".into()))?;
    if needed > out.capacity() {
        let old = out.capacity();
        let next = needed.max(old.checked_mul(2).ok_or_else(|| RusticError::Memory("sorted merge chunk capacity overflow".into()))?);
        memory.grow(next as u64, "sorted merge raw chunk growth overlap")?;
        out.reserve_exact(next - out.len());
        memory.shrink(old as u64);
    }
    out.extend_from_slice(bytes);
    Ok(())
}

fn flush_store_batch(raw: &mut Vec<crate::admission::Allocation<Vec<u8>>>, counts: &mut Vec<u32>, table: &mut Vec<(usize, usize)>, metadata: &mut crate::admission::Reservation,
    output: &mut dyn crate::codec::directory::OutputFile, zstd: bool, compact: bool, uid: bool) -> Result<()> {
    use rayon::prelude::*;
    let admission = crate::admission::Context::capture();
    let needed = table.len().checked_add(raw.len()).ok_or_else(|| RusticError::Memory("sorted merge table length overflow".into()))?;
    if needed > table.capacity() {
        let old = table.capacity();
        let next = needed.max(old.checked_mul(2).ok_or_else(|| RusticError::Memory("sorted merge table capacity overflow".into()))?);
        let bytes = next.checked_mul(std::mem::size_of::<(usize, usize)>()).ok_or_else(|| RusticError::Memory("sorted merge table bytes overflow".into()))?;
        metadata.grow(bytes as u64, "sorted merge table growth overlap")?;
        table.reserve_exact(next - table.len());
        metadata.shrink((old * std::mem::size_of::<(usize, usize)>()) as u64);
    }
    let compressed: Vec<crate::admission::Allocation<Vec<u8>>> = std::mem::take(raw).into_par_iter()
        .map(|chunk| -> Result<_> {
            let _admission=admission.enter();
            let mut memory = stored_rewrite_memory_for(&crate::admission::shared(), chunk.len(), zstd)?;
            // Compact docs from v18 (binary ids from v31), else bincode
            // (compact inputs re-encoded on append).
            let encoded = crate::codec::store::compress_chunk_flagged(&chunk, compact, zstd, uid);
            drop(chunk);
            memory.shrink(memory.bytes().saturating_sub(encoded.capacity() as u64));
            Ok(crate::admission::Allocation::new(encoded, memory))
        }).collect::<Result<_>>()?;
    for (chunk, docs) in compressed.into_iter().zip(counts.drain(..)) {
        output.write_all(&chunk)?;
        table.push((chunk.len(), docs as usize));
    }
    Ok(())
}

// ---- vectors -------------------------------------------------------------------

/// Rewrite the `doc -> PointId` map with remapped doc ids. Vectors and HNSW
/// graphs live in the vector store, keyed by stable point ids, so a text
/// merge never copies a vector or touches a graph.
fn merge_point_map(segments: &[Arc<Segment>], doc_map: &DocMap) -> Result<crate::admission::Allocation<Vec<u8>>> {
    merge_point_map_with_budget(segments,doc_map,&crate::admission::shared())
}

fn merge_point_map_with_budget(segments: &[Arc<Segment>], doc_map: &DocMap, budget:&Arc<crate::admission::Budget>) -> Result<crate::admission::Allocation<Vec<u8>>> {
    let num_docs = doc_map.num_docs();
    let fields=segments.iter().map(|s|s.point_map().field_count() as u64).fold(0u64,u64::saturating_add);
    // Input u64 columns plus encoded output/reallocation overlap and directory.
    let peak=fields.saturating_mul((num_docs as u64).saturating_mul(40).saturating_add(512)).saturating_add(4096);
    let mut memory=budget.reserve(peak,"merge point map columns and encoding")?;
    let mut vector_fields: Vec<u16> = segments.iter().flat_map(|s| s.point_map().fields()).collect();
    vector_fields.sort_unstable();
    vector_fields.dedup();
    let mut point_cols: Vec<(u16, Vec<u64>)> = Vec::with_capacity(vector_fields.len());
    for field in vector_fields {
        let mut col = vec![crate::codec::vectors::NO_POINT; num_docs as usize];
        for (seg_idx, seg) in segments.iter().enumerate() {
            merge_pace(seg.num_docs as u64);
            for (old, p) in seg.point_map().iter(field) {
                if let Some(new) = doc_map.new_id(seg_idx, old) {
                    col[new as usize] = p;
                }
            }
        }
        point_cols.push((field, col));
    }
    let bytes=crate::codec::vectors::encode_point_map(&point_cols);
    drop(point_cols);
    memory.shrink(memory.bytes().saturating_sub(bytes.capacity() as u64));
    Ok(crate::admission::Allocation::new(bytes,memory))
}

/// A tiered merge policy, in the spirit of Lucene's `TieredMergePolicy`.
///
/// Segments are grouped into tiers by live-doc count (a tier holds segments
/// whose live doc counts are within a factor of `max_segments_per_tier` of
/// each other); an over-full tier proposes merging its smallest
/// `max_segments_per_tier` members. Independently, any segment whose
/// deleted-doc ratio exceeds `deletes_pct_allowed` is proposed for a
/// solo rewrite. No proposal ever exceeds `max_merged_docs` live docs.
pub struct MergePolicy {
    pub max_segments_per_tier: usize,
    /// Segments merged at once from a tier over `max_segments_per_tier`
    /// (ES `max_merge_at_once`); `0`: `max_segments_per_tier`.
    pub max_merge_at_once: usize,
    pub floor_segment_docs: u32,
    pub max_merged_docs: u32,
    pub deletes_pct_allowed: f32,
    /// Segments a copy keeps at most once its proposals ran (`0`: no
    /// budget): past it the smallest merge, so a copy's tail of tiny seals
    /// collapses while indexing runs rather than waiting for a full tier.
    pub max_segment_count: usize,
    /// Byte-sized Lucene `TieredMergePolicy` ([`MergePolicy::find_merges_tiered`],
    /// what ES runs) instead of the doc-count tiers above. `None`: the
    /// doc-count policy.
    pub tiered: Option<TieredBytes>,
    /// Lucene's `LogByteSizeMergePolicy` ([`MergePolicy::find_merges_log_bytes`]):
    /// what ES runs for an index whose mapping has a `@timestamp` date field
    /// (`index.merge.policy.type: time_based`, or `unset` with the field).
    /// Takes precedence over `tiered`.
    pub log_bytes: Option<LogBytes>,
}

/// The settings of [`MergePolicy::find_merges_log_bytes`] (ES's time-based
/// policy: `LogByteSizeMergePolicy` with `merge_factor`, `floor_segment` as
/// its min merge size and the time-based `max_merged_segment`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogBytes {
    /// `index.merge.policy.merge_factor` (ES default 32).
    pub merge_factor: usize,
    /// `index.merge.policy.floor_segment` (Lucene `minMergeSize`).
    pub min_merge_bytes: u64,
    /// `index.merge.policy.max_merged_segment`, else 100gb (Lucene `maxMergeSize`).
    pub max_merge_bytes: u64,
}

/// The byte settings of [`MergePolicy::find_merges_tiered`] (ES
/// `index.merge.policy.*`, Lucene `TieredMergePolicy`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TieredBytes {
    /// `segments_per_tier`.
    pub segments_per_tier: f64,
    /// `floor_segment`.
    pub floor_segment_bytes: u64,
    /// `max_merged_segment`.
    pub max_merged_segment_bytes: u64,
}

/// One segment as [`MergePolicy::find_merges_tiered`] sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentSize {
    pub id: u32,
    pub docs: u32,
    pub live: u32,
    /// Its file's bytes (every section).
    pub bytes: u64,
    /// Claimed by a running merge: counted toward the index size, never
    /// proposed again.
    pub merging: bool,
    /// Visibility class (searchable or not yet refreshed): every segment
    /// counts toward one allowance, as all of Lucene's do, but a merge only
    /// joins segments of one class.
    pub class: u8,
}

impl Default for MergePolicy {
    fn default() -> Self {
        Self {
            max_segments_per_tier: 10,
            max_merge_at_once: 0,
            floor_segment_docs: 10_000,
            max_merged_docs: 50_000_000,
            deletes_pct_allowed: 0.2,
            max_segment_count: 0,
            tiered: None,
            log_bytes: None,
        }
    }
}

/// How costly a merge is for what it gains (lower goes first): its live
/// documents, scaled down by the square of its live ratio, so a merge that
/// reclaims many deletions ranks with smaller ones (Lucene's
/// TieredMergePolicy weighs `nonDelRatio^2` likewise).
pub fn merge_weight(live: u64, docs: u64) -> u64 {
    if docs == 0 {
        return 0;
    }
    let ratio = live as f64 / docs as f64;
    (live as f64 * ratio * ratio) as u64
}

/// A set of segments proposed for one merge.
pub struct MergeCandidate {
    pub segment_ids: Vec<u32>,
}

impl MergePolicy {
    /// Lucene's `TieredMergePolicy.findMerges` (natural merges), as ES 9
    /// configures it (`tiered`; deletes from `deletes_pct_allowed`, the
    /// merge width from `max_merge_at_once`), over every segment of one
    /// visibility class, those already merging included. Proposals come
    /// in Lucene's order (best-scored first):
    ///
    /// * sizes are file bytes pro-rated by live docs; segments over half
    ///   `max_merged_segment` with few deletes are out of consideration;
    /// * the allowed segment count is `segments_per_tier` per level, levels
    ///   growing by the merge factor from `floor_segment`; while the
    ///   eligible segments exceed it (or their deletes exceed
    ///   `deletes_pct_allowed`), the best merge of up to `merge factor`
    ///   consecutive segments (largest first) is proposed, scored by skew,
    ///   size^0.05 and the live ratio squared;
    /// * a merge must grow its largest input by half (unless reclaiming its
    ///   deletes), so segments never cascade into one big segment, and at
    ///   most one max-sized merge is proposed per call.
    /// Lucene 10.5's `LogMergePolicy.findMerges` as `LogByteSizeMergePolicy`
    /// runs it (ES's time-based policy), over `segments` in index order, the
    /// ones already merging included (a window holding one is skipped):
    ///
    /// * a segment's level is `log(size) / log(merge_factor)`, size its file
    ///   bytes pro-rated by live docs (`calibrateSizeByDeletes`), at least 1;
    /// * from the left, each round takes the highest level still to place,
    ///   and the rightmost segment within `LEVEL_LOG_SPAN` (0.75) of it, or
    ///   twice that when the level is below the floor's
    ///   (`log(min_merge_bytes)`), bounds the run;
    /// * the run is cut into windows of `merge_factor` adjacent segments; a
    ///   window stops before the segment that would take it past
    ///   `max_merge_bytes` (a segment alone past it is skipped). A full
    ///   window still under `min_merge_bytes` keeps taking segments while
    ///   it stays under it. Windows of one segment are not merged.
    ///
    /// Merges are of adjacent segments, so a time-ordered index stays time
    /// ordered (ES picks this policy for `@timestamp` data). With one search
    /// slice (`targetSearchConcurrency` 1) the doc cap is the index's docs.
    pub fn find_merges_log_bytes(&self, segments: &[SegmentSize]) -> Vec<MergeCandidate> {
        const LEVEL_LOG_SPAN: f64 = 0.75;
        let Some(l) = self.log_bytes else { return Vec::new() };
        let merge_factor = l.merge_factor.max(2);
        let (min_merge, max_merge) = (l.min_merge_bytes, l.max_merge_bytes);
        let norm = (merge_factor as f32 as f64).ln() as f32;
        let size = |s: &SegmentSize| -> u64 { if s.docs == 0 { 0 } else { (s.bytes as u128 * s.live as u128 / s.docs as u128) as u64 } };
        let size_docs = |s: &SegmentSize| u64::from(s.live);
        let total_docs: u64 = segments.iter().map(size_docs).sum();
        // Lucene: `Math.min(maxMergeDocs, ceilDiv(totalDocCount, 1))`.
        let max_merge_docs = total_docs.min(i32::MAX as u64);
        let levels: Vec<f32> = segments.iter().map(|s| ((size(s).max(1) as f64).ln() as f32) / norm).collect();
        // Lucene computes the floor in double (`Math.log(minMergeSize) / norm`).
        let level_floor: f32 = if min_merge == 0 { 0.0 } else { ((min_merge as f64).ln() / f64::from(norm)) as f32 };
        let n = segments.len();
        let mut max_levels = vec![-1.0f32; n + 1];
        for i in (0..n).rev() { max_levels[i] = levels[i].max(max_levels[i + 1]); }
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < n {
            let max_level = max_levels[start];
            let level_bottom = if max_level > level_floor { (max_level as f64 - LEVEL_LOG_SPAN) as f32 } else { (max_level as f64 - 2.0 * LEVEL_LOG_SPAN) as f32 };
            // The rightmost segment at or above the bottom (`start - 1` when none).
            let mut upto = n as isize - 1;
            while upto >= start as isize && levels[upto as usize] < level_bottom { upto -= 1; }
            let mut end = start + merge_factor;
            while (end as isize) <= 1 + upto {
                let mut any_merging = false;
                let (mut merge_size, mut merge_docs) = (0u64, 0u64);
                for i in start..end {
                    let s = &segments[i];
                    if s.merging { any_merging = true; break; }
                    let (seg_size, seg_docs) = (size(s), size_docs(s));
                    if merge_size + seg_size > max_merge || merge_docs + seg_docs > max_merge_docs {
                        // Full: a segment alone too large is a singleton (skipped).
                        end = if i == start { i + 1 } else { i };
                        break;
                    }
                    merge_size += seg_size;
                    merge_docs += seg_docs;
                }
                if end - start >= merge_factor && min_merge < max_merge && merge_size < min_merge && !any_merging {
                    // A full window still under the floor keeps packing.
                    while (end as isize) < 1 + upto {
                        let s = &segments[end];
                        if s.merging { any_merging = true; break; }
                        let (seg_size, seg_docs) = (size(s), size_docs(s));
                        if merge_size + seg_size > min_merge || merge_docs + seg_docs > max_merge_docs { break; }
                        merge_size += seg_size;
                        merge_docs += seg_docs;
                        end += 1;
                    }
                }
                if !any_merging && end - start > 1 {
                    out.push(MergeCandidate { segment_ids: segments[start..end].iter().map(|s| s.id).collect() });
                }
                start = end;
                end = start + merge_factor;
            }
            start = (1 + upto).max(start as isize) as usize;
        }
        out
    }

    pub fn find_merges_tiered(&self, segments: &[SegmentSize]) -> Vec<MergeCandidate> {
        self.find_merges_tiered_up_to(segments, usize::MAX)
    }

    /// [`MergePolicy::find_merges_tiered`]'s first `limit` proposals (the
    /// same ones, in the same order). Each proposal scans every eligible
    /// segment, so all of them cost O(n^2 / merge factor): with a backlog of
    /// ten thousand tiny segments (a burst of refreshes the merges fell
    /// behind) that is seconds per call under the core lock, called for
    /// every merge started and finished, which slows merging further as the
    /// backlog grows. A merge needs only the best proposal, and the backlog
    /// count only whether it passes `max_merge_count`.
    pub fn find_merges_tiered_up_to(&self, segments: &[SegmentSize], limit: usize) -> Vec<MergeCandidate> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(t) = self.tiered else { return Vec::new() };
        let segs_per_tier = t.segments_per_tier.max(2.0);
        let max_bytes = t.max_merged_segment_bytes.max(1);
        let floor = t.floor_segment_bytes;
        let deletes_pct = self.deletes_pct_allowed as f64 * 100.0;
        let at_once = if self.max_merge_at_once == 0 { self.max_segments_per_tier } else { self.max_merge_at_once };
        let merge_factor = at_once.min(segs_per_tier as usize).max(2);
        // (size, segment): `size` pro-rated by live docs (Lucene `size()`).
        let pro = |s: &SegmentSize| if s.docs == 0 { 0 } else { (s.bytes as u128 * s.live as u128 / s.docs as u128) as u64 };
        let mut sorted: Vec<(u64, SegmentSize)> = segments.iter().filter(|s| s.docs > 0).map(|s| (pro(s), *s)).collect();
        sorted.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
        let (mut tot_bytes, mut min_bytes, mut total_del, mut total_max, mut merging_bytes) = (0u64, u64::MAX, 0u64, 0u64, 0u64);
        let mut eligible: Vec<(u64, SegmentSize)> = Vec::new();
        for &(size, s) in &sorted {
            let del = (s.docs - s.live.min(s.docs)) as u64;
            if s.merging {
                merging_bytes += size;
                total_max += s.live as u64;
            } else {
                total_del += del;
                total_max += s.docs as u64;
                eligible.push((size, s));
            }
            min_bytes = min_bytes.min(size);
            tot_bytes += size;
        }
        if eligible.is_empty() {
            return Vec::new();
        }
        let mut allowed_del = (deletes_pct * total_max as f64 / 100.0) as i64;
        let total_del_pct = if total_max == 0 { 0.0 } else { 100.0 * total_del as f64 / total_max as f64 };
        // Too-large segments leave the count (and the size) under budget.
        eligible.retain(|(size, s)| {
            let seg_del_pct = 100.0 * (s.docs - s.live.min(s.docs)) as f64 / s.docs as f64;
            let too_big = *size > max_bytes / 2 && (total_del_pct <= deletes_pct || seg_del_pct <= deletes_pct);
            if too_big {
                tot_bytes -= *size;
                allowed_del -= (s.docs - s.live.min(s.docs)) as i64;
            }
            !too_big
        });
        let allowed_del = allowed_del.max(0) as u64;
        let mut level = min_bytes.max(floor).max(1);
        let mut left = tot_bytes as f64;
        let mut allowed = 0f64;
        loop {
            let count = left / level as f64;
            if count < segs_per_tier || level == max_bytes {
                allowed += count.ceil();
                break;
            }
            allowed += segs_per_tier;
            left -= segs_per_tier * level as f64;
            level = max_bytes.min(level.saturating_mul(merge_factor as u64));
        }
        let allowed = allowed.max(segs_per_tier) as usize;
        let max_merge_running = merging_bytes >= max_bytes;
        let floored = |b: u64| b.max(floor);

        let mut out = Vec::new();
        let mut taken: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut have_large = false;
        loop {
            eligible.retain(|(_, s)| !taken.contains(&s.id));
            if eligible.is_empty() {
                break;
            }
            let remaining_del: u64 = eligible.iter().map(|(_, s)| (s.docs - s.live.min(s.docs)) as u64).sum();
            if eligible.len() <= allowed && remaining_del <= allowed_del {
                break;
            }
            let mut best: Option<(f64, Vec<usize>, bool)> = None;
            // Classes whose remaining windows are all short (Lucene stops
            // there; with one class this is its `break`).
            let mut done: [bool; 2] = [false; 2];
            // A candidate of the class was the best so far (one class: `best.is_some()`).
            let mut had_best: [bool; 2] = [false; 2];
            for start in 0..eligible.len() {
                if done[(eligible[start].1.class & 1) as usize] {
                    if done.iter().all(|&d| d) { break; }
                    continue;
                }
                let mut cand: Vec<usize> = Vec::new();
                let mut too_large = false;
                let mut bytes = 0u64;
                let mut idx = start;
                let class = eligible[start].1.class;
                while idx < eligible.len() && cand.len() < merge_factor && bytes < max_bytes {
                    if eligible[idx].1.class != class {
                        idx += 1;
                        continue;
                    }
                    let seg = eligible[idx].0;
                    if bytes + seg > max_bytes {
                        too_large = true;
                        if !cand.is_empty() {
                            idx += 1;
                            continue;
                        }
                    }
                    cand.push(idx);
                    bytes += seg;
                    idx += 1;
                }
                let (first_size, first) = eligible[cand[0]];
                let first_del = (first.docs - first.live.min(first.docs)) as f64;
                if !too_large && (bytes as f64) < first_size as f64 * 1.5 && first_del < first.docs as f64 * deletes_pct / 100.0 {
                    continue;
                }
                if cand.len() == 1 && first.live >= first.docs {
                    continue;
                }
                if had_best[(class & 1) as usize] && !too_large && cand.len() < merge_factor {
                    done[(class & 1) as usize] = true;
                    if done.iter().all(|&d| d) || !eligible[start..].iter().any(|(_, s)| s.class != class) { break; }
                    continue;
                }
                // Lucene `score`: skew, then size^0.05, then the live ratio^2.
                let (mut after, mut after_floored, mut before) = (0u64, 0u64, 0u64);
                for &i in &cand {
                    let (size, s) = eligible[i];
                    after += size;
                    after_floored += floored(size);
                    before += s.bytes;
                }
                let skew = if too_large { 1.0 / merge_factor as f64 } else { floored(first_size) as f64 / after_floored.max(1) as f64 };
                let live_ratio = if before == 0 { 1.0 } else { after as f64 / before as f64 };
                let score = skew * (after as f64).powf(0.05) * live_ratio * live_ratio;
                if best.as_ref().is_none_or(|(b, _, _)| score < *b) && (!too_large || !max_merge_running) {
                    had_best[(class & 1) as usize] = true;
                    best = Some((score, cand, too_large));
                }
            }
            let Some((_, cand, too_large)) = best else { break };
            let ids: Vec<u32> = cand.iter().map(|&i| eligible[i].1.id).collect();
            if !have_large || !too_large {
                have_large |= too_large;
                out.push(MergeCandidate { segment_ids: ids.clone() });
                if out.len() >= limit {
                    break;
                }
            }
            taken.extend(ids);
        }
        out
    }

    /// Find merge candidates among `segments` (id, num_docs, num_live_docs).
    /// Deterministic: candidates are ordered by their smallest segment id.
    pub fn find_merges(&self, segments: &[(u32, u32, u32)]) -> Vec<MergeCandidate> {
        let mut out: Vec<MergeCandidate> = Vec::new();

        // The floor groups tiny segments into the lowest tier; it must not
        // exclude them from merging (bulk/recovery writers can be very small).
        let mut eligible: Vec<(u32, u32, u32)> = segments
            .iter()
            .copied()
            .filter(|(_, docs, _)| *docs > 0)
            .collect();
        eligible.sort_by_key(|(id, _, live)| ((*live).max(self.floor_segment_docs), *live, *id));
        let mut i = 0;
        while i < eligible.len() {
            let base = eligible[i].2.max(self.floor_segment_docs);
            let cap = base.saturating_mul(self.max_segments_per_tier as u32);
            let mut tier: Vec<(u32, u32, u32)> = Vec::new();
            while i < eligible.len() && eligible[i].2 <= cap {
                tier.push(eligible[i]);
                i += 1;
            }
            if tier.len() > self.max_segments_per_tier {
                // `eligible` is sorted by live docs, so the first
                // `max_merge_at_once` of the tier are the smallest.
                let at_once = if self.max_merge_at_once == 0 { self.max_segments_per_tier } else { self.max_merge_at_once };
                let chosen: Vec<u32> = tier
                    .iter()
                    .take(at_once)
                    .map(|(id, _, _)| *id)
                    .collect();
                let live: u32 = tier
                    .iter()
                    .take(at_once)
                    .map(|(_, _, live)| *live)
                    .sum();
                if live <= self.max_merged_docs {
                    out.push(MergeCandidate {
                        segment_ids: chosen,
                    });
                }
            }
        }

        // Over the segment budget: the smallest segments no tier proposal
        // took, as many as bring the count to it (at least two). Never one
        // much larger than the rest together past the first tier (it would
        // be rewritten for little gain).
        let after = eligible.len() - out.iter().map(|c| c.segment_ids.len() - 1).sum::<usize>();
        if self.max_segment_count > 0 && after > self.max_segment_count {
            let at_once = if self.max_merge_at_once == 0 { self.max_segments_per_tier } else { self.max_merge_at_once };
            let want = (after - self.max_segment_count + 1).clamp(2, at_once.max(2));
            let free: Vec<(u32, u32, u32)> = eligible.iter().copied()
                .filter(|(id, _, _)| !out.iter().any(|c| c.segment_ids.contains(id))).take(want).collect();
            let live: u64 = free.iter().map(|s| s.2 as u64).sum();
            let top = free.iter().map(|s| s.2 as u64).max().unwrap_or(0);
            let small = top <= self.floor_segment_docs as u64 * self.max_segments_per_tier as u64 || top <= live - top;
            if free.len() >= 2 && small && live <= self.max_merged_docs as u64 {
                out.push(MergeCandidate { segment_ids: free.iter().map(|s| s.0).collect() });
            }
        }

        // Solo rewrites for segments with too many deletions.
        for (id, num_docs, live) in segments {
            if *num_docs == 0 {
                continue;
            }
            let deleted_ratio = (*num_docs - *live) as f32 / *num_docs as f32;
            if deleted_ratio > self.deletes_pct_allowed && *live <= self.max_merged_docs {
                out.push(MergeCandidate {
                    segment_ids: vec![*id],
                });
            }
        }

        // Cheapest merges first ([`merge_weight`]; Lucene's TieredMergePolicy
        // scores smaller and delete-heavier merges better): a burst of tiny
        // segments (a large backlog sealed at once) collapses before a large
        // tier is rewritten; ties by the smallest segment id (deterministic).
        let by_id: std::collections::HashMap<u32, (u32, u32)> = segments.iter().map(|(id, docs, live)| (*id, (*docs, *live))).collect();
        let weight = |c: &MergeCandidate| {
            let (docs, live) = c.segment_ids.iter().filter_map(|id| by_id.get(id)).fold((0u64, 0u64), |(d, l), (docs, live)| (d + *docs as u64, l + *live as u64));
            merge_weight(live, docs)
        };
        out.sort_by_key(|c| (weight(c), c.segment_ids.iter().min().copied().unwrap_or(u32::MAX)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiered() -> MergePolicy {
        MergePolicy {
            max_merge_at_once: 16,
            deletes_pct_allowed: 0.2,
            tiered: Some(TieredBytes { segments_per_tier: 8.0, floor_segment_bytes: 16 << 20, max_merged_segment_bytes: 5 << 30 }),
            ..Default::default()
        }
    }

    fn seg(id: u32, mb: u64) -> SegmentSize {
        SegmentSize { id, docs: 1000, live: 1000, bytes: mb << 20, merging: false, class: 0 }
    }

    /// ES's time-based policy (`LogByteSizeMergePolicy`: merge factor 32,
    /// 16mb floor, 100gb max).
    fn log_bytes(factor: usize) -> MergePolicy {
        MergePolicy { log_bytes: Some(LogBytes { merge_factor: factor, min_merge_bytes: 16 << 20, max_merge_bytes: 100 << 30 }), ..Default::default() }
    }
    fn ids(c: &[MergeCandidate]) -> Vec<Vec<u32>> { c.iter().map(|m| m.segment_ids.clone()).collect() }

    /// Lucene `LogMergePolicy.findMerges`: windows of `merge_factor`
    /// adjacent segments, a short tail left alone; a window holding a
    /// merging segment skipped.
    #[test]
    fn log_bytes_merges_adjacent_windows_of_the_merge_factor() {
        let p = log_bytes(32);
        // 40 one-megabyte flushes (under the floor: one level).
        let segs: Vec<_> = (0..40).map(|i| seg(i, 1)).collect();
        assert_eq!(ids(&p.find_merges_log_bytes(&segs)), vec![(0..32).collect::<Vec<u32>>()]);
        // 64 of them: two windows, in order.
        let segs: Vec<_> = (0..64).map(|i| seg(i, 1)).collect();
        assert_eq!(ids(&p.find_merges_log_bytes(&segs)), vec![(0..32).collect::<Vec<u32>>(), (32..64).collect()]);
        // A merging segment blocks its window only.
        let mut segs = segs;
        segs[5].merging = true;
        assert_eq!(ids(&p.find_merges_log_bytes(&segs)), vec![(32..64).collect::<Vec<u32>>()]);
        // Fewer than the factor: nothing.
        let segs: Vec<_> = (0..31).map(|i| seg(i, 1)).collect();
        assert!(p.find_merges_log_bytes(&segs).is_empty());
    }

    /// Levels: a big segment and small ones are not merged together; the
    /// small ones to its right form their own windows.
    #[test]
    fn log_bytes_levels_keep_sizes_apart() {
        let p = log_bytes(4);
        // 3 GB, then eight 100 MB segments: level of 3 GB is far above.
        let mut segs = vec![seg(0, 3 << 10)];
        segs.extend((1..9).map(|i| seg(i, 100)));
        assert_eq!(ids(&p.find_merges_log_bytes(&segs)), vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
    }

    /// A window stops before the segment that would take it past the max
    /// merged size; a segment alone past it is never merged.
    #[test]
    fn log_bytes_respects_the_max_merged_size() {
        let p = MergePolicy { log_bytes: Some(LogBytes { merge_factor: 2, min_merge_bytes: 16 << 20, max_merge_bytes: 100 << 30 }), ..Default::default() };
        let gb = |id, g: u64| seg(id, g << 10);
        assert_eq!(ids(&p.find_merges_log_bytes(&[gb(0, 40), gb(1, 40), gb(2, 40), gb(3, 40)])), vec![vec![0, 1], vec![2, 3]]);
        assert!(p.find_merges_log_bytes(&[gb(0, 60), gb(1, 60), gb(2, 60)]).is_empty());
        assert!(p.find_merges_log_bytes(&[gb(0, 120), gb(1, 1)]).is_empty());
    }

    /// A full window still under the floor keeps taking segments while it
    /// stays under it.
    #[test]
    fn log_bytes_packs_a_window_under_the_floor() {
        let p = log_bytes(4);
        let segs: Vec<_> = (0..10).map(|i| seg(i, 1)).collect();
        assert_eq!(ids(&p.find_merges_log_bytes(&segs)), vec![(0..10).collect::<Vec<u32>>()]);
        // Deleted docs pro-rate the size (Lucene `calibrateSizeByDeletes`).
        let half = SegmentSize { live: 500, ..seg(0, 2) };
        assert_eq!(p.find_merges_log_bytes(&[half, seg(1, 1), seg(2, 1), seg(3, 1)]).len(), 1);
    }

    /// The merge schedule of a long append-only load (1 GB flushes to
    /// 4.8 TB, merges applied as proposed, outputs in place of their inputs):
    /// the time-based policy ends near ES's count on github_archive (125),
    /// the tiered one with thousands of segments capped at 5 GB.
    #[test]
    fn log_bytes_schedule_ends_with_far_fewer_segments_than_tiered_at_terabytes() {
        let run = |p: &MergePolicy, log: bool| -> usize {
            let mut segs: Vec<SegmentSize> = Vec::new();
            let mut next = 0u32;
            for _ in 0..4800 {
                segs.push(SegmentSize { id: next, docs: 1_000_000, live: 1_000_000, bytes: 1 << 30, merging: false, class: 0 });
                next += 1;
                loop {
                    let found = if log { p.find_merges_log_bytes(&segs) } else { p.find_merges_tiered_up_to(&segs, 1) };
                    let Some(m) = found.into_iter().next() else { break };
                    let pos = segs.iter().position(|s| m.segment_ids.contains(&s.id)).unwrap();
                    let (bytes, docs) = segs.iter().filter(|s| m.segment_ids.contains(&s.id)).fold((0u64, 0u32), |a, s| (a.0 + s.bytes, a.1.saturating_add(s.docs)));
                    segs.retain(|s| !m.segment_ids.contains(&s.id));
                    segs.insert(pos.min(segs.len()), SegmentSize { id: next, docs, live: docs, bytes, merging: false, class: 0 });
                    next += 1;
                }
            }
            segs.len()
        };
        let log = run(&log_bytes(32), true);
        let tiered = run(&tiered(), false);
        eprintln!("4.8 TB of 1 GB flushes: time-based {log} segments, tiered {tiered}");
        assert!(log <= 150, "time-based policy: {log} segments");
        assert!(tiered >= 900, "tiered policy: {tiered} segments");
    }

    /// Lucene TieredMergePolicy (ES 9 defaults): flush-sized segments merge
    /// eight at a time; the result never cascades into one segment, since
    /// an index of a few levels stays within its allowed count; too-large
    /// segments, claimed ones and merges that barely grow their largest
    /// input are never proposed.
    /// Searchable and not-yet-refreshed segments share one allowance (as
    /// Lucene counts every flushed segment): each class alone is within
    /// budget, both together are not, and merges stay within a class.
    #[test]
    fn tiered_allowance_counts_both_visibility_classes() {
        let p = tiered();
        let mut visible: Vec<_> = (1..=7).map(|i| seg(i, 1000)).collect();
        visible.extend((8..=14).map(|i| seg(i, 128)));
        visible.extend((15..=21).map(|i| seg(i, 16)));
        let hidden: Vec<_> = (22..=27).map(|i| SegmentSize { class: 1, ..seg(i, 16) }).collect();
        assert!(p.find_merges_tiered(&visible).is_empty());
        assert!(p.find_merges_tiered(&hidden).is_empty());
        let all: Vec<_> = visible.iter().chain(&hidden).copied().collect();
        let m = p.find_merges_tiered(&all);
        assert!(!m.is_empty(), "27 segments exceed one allowance");
        for c in &m {
            let classes: std::collections::HashSet<u8> = c.segment_ids.iter().map(|id| all.iter().find(|s| s.id == *id).unwrap().class).collect();
            assert_eq!(classes.len(), 1, "{:?}", c.segment_ids);
        }
        // One class alone: proposals equal the single-class policy's.
        let mut flushes: Vec<_> = (1..=20).map(|i| seg(i, 10)).collect();
        let before = p.find_merges_tiered(&flushes);
        flushes.iter_mut().for_each(|s| s.class = 1);
        let ids = |m: &[MergeCandidate]| m.iter().map(|c| c.segment_ids.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&p.find_merges_tiered(&flushes)), ids(&before));
    }

    #[test]
    fn tiered_policy_matches_lucene() {
        let p = tiered();
        // 20 flushes of 10 MB: one merge of the eight largest (ties by id).
        let flushes: Vec<_> = (1..=20).map(|i| seg(i, 10)).collect();
        let m = p.find_merges_tiered(&flushes);
        assert!(!m.is_empty());
        assert_eq!(m[0].segment_ids.len(), 8);
        // A settled index: levels of up to 8 segments each, nothing to do.
        let mut settled: Vec<_> = (1..=7).map(|i| seg(i, 1000)).collect();
        settled.extend((8..=14).map(|i| seg(i, 128)));
        settled.extend((15..=21).map(|i| seg(i, 16)));
        assert!(p.find_merges_tiered(&settled).is_empty(), "{} segments stay", settled.len());
        // One big segment and two small ones are within the budget: no
        // rewrite of the big one (ES keeps many segments, never one).
        assert!(p.find_merges_tiered(&[seg(1, 4000), seg(2, 20), seg(3, 20)]).is_empty());
        // Over the budget, the merge leaves the 3 GB segment (over half the
        // max) alone and does not rewrite the large one for small gains.
        let mut over: Vec<_> = vec![seg(1, 3000), seg(2, 900)];
        over.extend((3..=22).map(|i| seg(i, 5)));
        let m = p.find_merges_tiered(&over);
        assert!(!m.is_empty(), "21 eligible segments exceed the 15 allowed");
        for c in m {
            assert!(!c.segment_ids.contains(&1), "{:?}", c.segment_ids);
            assert!(!c.segment_ids.contains(&2), "{:?}", c.segment_ids);
        }
        // Claimed segments count toward the size but are never proposed.
        let mut claimed = flushes.clone();
        for s in claimed.iter_mut().take(8) { s.merging = true; }
        for c in p.find_merges_tiered(&claimed) {
            assert!(c.segment_ids.iter().all(|id| *id > 8), "{:?}", c.segment_ids);
        }
        // Deletes past deletes_pct_allowed are reclaimed by a rewrite even
        // within the segment budget.
        let mut deleted: Vec<_> = (1..=3).map(|i| seg(i, 100)).collect();
        deleted[0].live = 300;
        let m = p.find_merges_tiered(&deleted);
        assert!(m.iter().any(|c| c.segment_ids.contains(&1)), "{:?}", m.iter().map(|c| c.segment_ids.clone()).collect::<Vec<_>>());
        // No proposal ever exceeds max_merged_segment.
        let big: Vec<_> = (1..=30).map(|i| seg(i, 700)).collect();
        for c in p.find_merges_tiered(&big) {
            let bytes: u64 = c.segment_ids.iter().map(|id| big[*id as usize - 1].bytes).sum();
            assert!(bytes <= 5 << 30, "{bytes}");
        }
        // The doc-count policy is unchanged for policies without `tiered`.
        assert!(MergePolicy::default().find_merges_tiered(&flushes).is_empty());
    }

    /// chaos seed 3: a primary's merges fell behind a burst
    /// of tiny refresh segments and planning every proposal over 11,000 of
    /// them, three times per merge under the core lock, kept them behind
    /// for 20 minutes. A bounded plan is the full plan's prefix.
    #[test]
    fn a_bounded_tiered_plan_is_the_prefix_of_the_full_plan() {
        let p = tiered();
        let tiny = |id: u32| SegmentSize { id, docs: 1 + id % 7, live: 1 + id % 7, bytes: 600 + (id as u64 % 13) * 97, merging: false, class: 0 };
        let backlog: Vec<_> = (1..=1500).map(tiny).collect();
        let full = p.find_merges_tiered(&backlog);
        assert!(full.len() > 64, "{} proposals", full.len());
        for limit in [1, 2, 7, 64] {
            let bounded = p.find_merges_tiered_up_to(&backlog, limit);
            assert_eq!(bounded.len(), limit);
            assert_eq!(bounded.iter().map(|c| &c.segment_ids).collect::<Vec<_>>(), full.iter().take(limit).map(|c| &c.segment_ids).collect::<Vec<_>>());
        }
        assert!(p.find_merges_tiered_up_to(&backlog, 0).is_empty());
        assert_eq!(p.find_merges_tiered_up_to(&backlog, usize::MAX).len(), full.len());
        // The best proposal over a 20,000-segment backlog is one pass.
        let huge: Vec<_> = (1..=20_000).map(tiny).collect();
        assert_eq!(p.find_merges_tiered_up_to(&huge, 1).len(), 1);
    }

    /// A v14 merge writes a sparse keyword column sparse, from dense and
    /// sparse inputs alike, with every doc's value (and deletions dropped);
    /// a v13 merge of the same inputs writes it dense with the same values.
    #[test]
    fn merges_mix_compact_and_bincode_stored_chunks() {
        let dir = RamDirectory::new();
        let schema = schema();
        let doc_of = |seg: u32, i: u32| {
            let mut doc = Document::new();
            doc.add_text(0, &format!("seg{seg} doc{i} common words"));
            doc.add_f64(2, i as f64 * -0.5);
            doc.add_bool(3, i % 3 == 0);
            doc.add_keyword(4, &format!("cat{}", i % 5));
            doc.add_i64(5, i as i64 - 100);
            doc
        };
        // The version is read when a stored chunk opens, so it is set
        // before the docs are added.
        let build = |id: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(id);
            for i in 0..300 { w.add_document(&doc_of(id, i), &schema); }
            let seg = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(seg)
        };
        let segments = vec![build(1, 17), build(2, 18)];
        let flags = |seg: &Segment| { let r = seg.store().unwrap(); (0..r.num_chunks()).map(|c| r.chunk_compact(c)).collect::<Vec<_>>() };
        assert!(flags(&segments[0]).iter().all(|&f| !f));
        assert!(flags(&segments[1]).iter().all(|&f| f));
        let fields = |d: Document| format!("{:?}", d.fields);
        for i in [0u32, 299] {
            assert_eq!(fields(segments[1].get_doc(i).unwrap().unwrap()), fields(doc_of(2, i)));
        }
        // Deletions in both inputs: their dirty chunks are rewritten, the
        // rest copied verbatim, each keeping its encoding at v18. A v17
        // merge (writes held below the floor) re-encodes compact chunks.
        segments[0].delete_doc(5).unwrap();
        segments[1].delete_doc(200).unwrap();
        let mut expect = Vec::new();
        for (seg, del) in [(1u32, 5u32), (2, 200)] {
            for i in 0..300 { if i != del { expect.push(fields(doc_of(seg, i))); } }
        }
        for version in [18u32, 17] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let merged = merge_segments(&dir, &segments, 10 + version).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            let f = flags(&merged);
            if version >= 18 { assert!(f.contains(&true) && f.contains(&false), "{f:?}"); } else { assert!(f.iter().all(|&x| !x), "{f:?}"); }
            assert_eq!(merged.num_docs as usize, expect.len());
            for (i, want) in expect.iter().enumerate() {
                assert_eq!(&fields(merged.get_doc(i as DocId).unwrap().unwrap()), want, "v{version} doc {i}");
            }
            let mut n = 0;
            merged.store().unwrap().for_each(|i, d| { assert_eq!(fields(d), expect[i as usize]); n += 1; }).unwrap();
            assert_eq!(n, expect.len());
        }
        // A sorted merge interleaves both inputs' docs into shared chunks,
        // written compact from v18 (bincode docs re-encoded), else plain.
        let fresh = vec![build(3, 17), build(4, 18)];
        let order: Vec<(usize, DocId)> = (0..300u32).rev().flat_map(|d| [(0usize, d), (1, d)]).collect();
        let map = DocMap::from_order(&fresh, order.clone());
        let store = merge_store_sorted_with_budget(&dir, 30, &fresh, &map, &crate::admission::shared()).unwrap();
        let bytes = store.pieces().collect::<Vec<_>>().concat();
        let r = StoreReader::open(&bytes).unwrap();
        assert_eq!(r.num_docs(), 600);
        let compact = crate::codec::VERSION >= crate::codec::COMPACT_STORED_SINCE;
        assert!((0..r.num_chunks()).all(|c| r.chunk_compact(c) == compact));
        for (new, &(i, old)) in order.iter().enumerate() {
            assert_eq!(fields(r.get(new as DocId).unwrap().unwrap()), fields(doc_of(3 + i as u32, old)), "sorted doc {new}");
        }
    }

    /// Sparse single-valued keyword and long fields (most docs without a
    /// value, as most of github_archive's fields) and a bool merge to the
    /// same per-doc values whether the merge visits only the docs with a
    /// value or every doc: an input-order merge and a sorting merge
    /// (interleaved inputs), deletions dropped, a dense keyword beside them.
    #[test]
    fn sparse_columns_merge_to_the_same_values_in_any_order() {
        let dir = RamDirectory::new();
        let schema = vec![Field::new(0, "k", FieldType::Keyword), Field::new(1, "n", FieldType::I64), Field::new(2, "b", FieldType::Bool), Field::new(3, "dense", FieldType::Keyword)];
        let kw = |seg: u32, d: u32| ((d * 7 + seg) % 23 == 0).then(|| format!("v{}", (d * 13 + seg * 5) % 37));
        let num = |seg: u32, d: u32| ((d + seg * 3) % 17 == 0).then(|| i64::from(d) * 1_000 - i64::from(seg) * 77);
        let boolean = |seg: u32, d: u32| ((d + seg) % 5 == 0).then(|| d % 2 == 0);
        let dense = |seg: u32, d: u32| format!("d{}", (d + seg) % 9);
        let sizes = [3_000u32, 2_000, 2_500];
        let segments: Vec<Arc<Segment>> = sizes.iter().enumerate().map(|(i, &n)| {
            let seg = i as u32 + 1;
            let mut w = SegmentWriter::new(seg);
            for d in 0..n {
                let mut doc = Document::new();
                if let Some(v) = kw(seg, d) { doc.add_keyword(0, &v); }
                if let Some(v) = num(seg, d) { doc.add_i64(1, v); }
                if let Some(v) = boolean(seg, d) { doc.add_bool(2, v); }
                doc.add_keyword(3, &dense(seg, d));
                w.add_document(&doc, &schema);
            }
            Arc::new(w.write_to(&dir).unwrap())
        }).collect();
        for d in [0u32, 23, 46, 69, 1_000] { segments[1].delete_doc(d).unwrap(); }
        let deleted = |i: usize, d: u32| i == 1 && [0u32, 23, 46, 69, 1_000].contains(&d);
        let live: Vec<(usize, DocId)> = segments.iter().enumerate().flat_map(|(i, s)| (0..s.num_docs).filter(move |&d| !deleted(i, d)).map(move |d| (i, d))).collect();
        // A sorting merge's order: the inputs interleaved.
        let mut sorted = live.clone();
        sorted.sort_by_key(|&(i, d)| ((u64::from(d) * 7_919 + i as u64 * 104_729) % 10_007, i, d));
        for (id, order) in [(10u32, None), (11, Some(sorted.clone()))] {
            let expect = order.clone().unwrap_or_else(|| live.clone());
            let merged = merge_segments_with(&dir, &segments, id, order).unwrap();
            assert_eq!(merged.num_docs as usize, expect.len());
            let ff = merged.fast_fields();
            let k = merged.keyword_reader(0).unwrap();
            let dk = merged.keyword_reader(3).unwrap();
            let n = ff.get_i64(1).unwrap();
            let b = ff.get_bool(2).unwrap();
            for (new, &(i, old)) in expect.iter().enumerate() {
                let seg = i as u32 + 1;
                let new = new as DocId;
                assert_eq!(k.get(new).map(|v| v.into_owned()), kw(seg, old), "keyword doc {new} ({i}, {old}) id {id}");
                assert_eq!(n.get(new), num(seg, old), "long doc {new} id {id}");
                assert_eq!(dk.get(new).map(|v| v.into_owned()), Some(dense(seg, old)), "dense doc {new}");
                if let Some(v) = boolean(seg, old) { assert_eq!(b.get(new), v, "bool doc {new} id {id}"); }
            }
        }
    }

    #[test]
    fn merges_write_sparse_keyword_columns_from_v14() {
        let dir = RamDirectory::new();
        let schema = vec![Field::new(0, "k", FieldType::Keyword)];
        let value = |seg: u32, d: u32| (d % 40 == seg).then(|| format!("t{}", (d + seg) % 5));
        let build = |id: u32, version: u32| {
            let mut w = SegmentWriter::new(id);
            for d in 0..2000u32 {
                let mut doc = Document::new();
                if let Some(v) = value(id, d) { doc.add_keyword(0, &v); }
                w.add_document(&doc, &schema);
            }
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let seg = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(seg)
        };
        let segments = vec![build(1, 13), build(2, 14)];
        segments[1].delete_doc(2).unwrap();
        let sparse = |seg: &Segment| seg.fast_fields().keyword_column(0).unwrap().1 == crate::codec::fastfields::OrdLayout::Sparse;
        assert!(!sparse(&segments[0]) && sparse(&segments[1]));
        for (version, id) in [(14u32, 10u32), (13, 11)] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let merged = merge_segments(&dir, &segments, id).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            assert_eq!(sparse(&merged), version >= 14, "v{version}");
            let k = merged.keyword_reader(0).unwrap();
            let mut expect = Vec::new();
            for (i, seg) in [1u32, 2].into_iter().enumerate() {
                for d in 0..2000u32 {
                    if i == 1 && d == 2 { continue; }
                    expect.push(value(seg, d));
                }
            }
            assert_eq!(merged.num_docs as usize, expect.len());
            for (d, v) in expect.iter().enumerate() {
                assert_eq!(k.get(d as DocId).map(|v| v.into_owned()), v.clone(), "v{version} doc {d}");
            }
        }
    }

    /// Feature columns (sparse_vector doc values) through the index-sorted
    /// fast-field merge: their packed pairs follow the sort order, value for
    /// value; beside a plain keyword column of the field (or below v29) they
    /// rebuild per doc. Nothing is dropped either way.
    #[test]
    fn sorted_merges_keep_feature_columns() {
        use crate::fieldtypes::features::{encode, quantize};
        let dir = RamDirectory::new();
        let schema = vec![Field::new(0, "sv", FieldType::Keyword).with_features(true), Field::new(1, "k", FieldType::Keyword)];
        let value = |seg: u32, d: u32| -> Vec<String> {
            (0..(d % 6)).map(|k| encode(&format!("t{}", (seg * 31 + d * 7 + k * 3) % 50), quantize(0.05 + ((seg + d + k) % 300) as f32 / 41.0))).collect()
        };
        let build = |id: u32, plain: bool| {
            let mut w = SegmentWriter::new(id);
            for d in 0..400u32 {
                let mut doc = Document::new();
                for v in value(id, d) { doc.add_keyword(0, &v); }
                if plain && d == 3 { doc.add_keyword(0, "not-a-feature"); }
                doc.add_keyword(1, &format!("k{}", d % 9));
                w.add_document(&doc, &schema);
            }
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::FEATURES_COLUMN_SINCE));
            let seg = w.write_to(&dir).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            Arc::new(seg)
        };
        for (plain, version) in [(false, crate::codec::FEATURES_COLUMN_SINCE), (true, crate::codec::FEATURES_COLUMN_SINCE), (false, crate::codec::FEATURES_COLUMN_SINCE - 1)] {
            let segments = vec![build(1, false), build(2, plain)];
            assert!(segments[0].fast_fields().get_features(0).is_some());
            segments[0].delete_doc(10).unwrap();
            // Interleaved, descending: a real sort order across both inputs.
            let order: Vec<(usize, DocId)> = (0..400u32).rev().flat_map(|d| [(1usize, d), (0, d)]).filter(|&(i, d)| !(i == 0 && d == 10)).collect();
            let map = DocMap::from_order(&segments, order.clone());
            let norms = merge_norms(&segments, &map).unwrap();
            let mut cols: Vec<(u16, u8, Vec<u8>)> = Vec::new();
            stream_fast_fields_sorted(&segments, &map, &norms, MergeAdmission::Foreground, version, &mut |f, t, b| { cols.push((f, t, b.to_vec())); Ok(()) }).unwrap();
            let col = cols.iter().find(|c| c.0 == 0).expect("the feature column is kept");
            let values: Box<dyn Fn(DocId) -> Vec<String>> = if col.1 == crate::codec::fastfields::TAG_FEATURES {
                let r = crate::codec::fastfields::FeaturesReader::open(&col.2).unwrap();
                Box::new(move |d| r.values(d))
            } else {
                assert_eq!(col.1, TAG_MULTI_KEYWORD);
                let r = MultiKeywordReader::open(&col.2).unwrap();
                Box::new(move |d| r.values(d).into_iter().map(str::to_string).collect())
            };
            assert_eq!(col.1 == crate::codec::fastfields::TAG_FEATURES, !plain && version >= crate::codec::FEATURES_COLUMN_SINCE, "plain {plain} v{version}");
            for (new, &(i, old)) in order.iter().enumerate() {
                let mut want = value(i as u32 + 1, old);
                if plain && i == 1 && old == 3 { want.push("not-a-feature".to_string()); }
                want.sort();
                want.dedup();
                assert_eq!(values(new as DocId), want, "plain {plain} v{version} doc {new}");
            }
            // Plain keywords still take the ordinal path.
            assert!(cols.iter().any(|c| c.0 == 1 && c.1 != crate::codec::fastfields::TAG_FEATURES));
        }
    }

    /// An index-sorted merge writes its re-chunked docs as unsorted merges
    /// and seals do: compact docs, binary ids from v31, 1024-doc / 64 KB
    /// chunks; below v31 the ids as keywords, below v18 bincode. Every doc
    /// reads back in the merge order.
    #[test]
    fn sorted_merges_write_compact_chunks_with_binary_ids() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword), Field::new(1, "body", FieldType::Text), Field::new(2, "n", FieldType::I64)];
        let doc = |seg: u32, d: u32| {
            let mut doc = Document::new();
            doc.add_keyword(0, &format!("AaG{}{:05}vX0F9q8YlnEvG", seg, d));
            doc.add_text(1, &format!("{{\"message\":\"GET /english/images/{} HTTP/1.0\",\"n\":{d}}}", d % 37));
            doc.add_i64(2, (d * 2 + seg) as i64);
            doc
        };
        let fields = |d: &Document| format!("{:?}", d.fields.iter().filter(|(f, _)| *f != 2).collect::<Vec<_>>());
        let mut sizes = Vec::new();
        for version in [crate::codec::VERSION, crate::codec::UID_STORED_SINCE - 1, crate::codec::COMPACT_STORED_SINCE - 1] {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
            let dir = RamDirectory::new();
            let segments: Vec<Arc<Segment>> = (1..=2u32).map(|seg| {
                let mut w = SegmentWriter::new(seg);
                for d in 0..1500 { w.add_document(&doc(seg, d), &schema); }
                Arc::new(w.write_to(&dir).unwrap())
            }).collect();
            let order: Vec<(usize, DocId)> = (0..1500u32).rev().flat_map(|d| [(1usize, d), (0, d)]).collect();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let merged = merge_segments_with(&dir, &segments, 9, Some(order.clone())).unwrap();
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            let store = merged.store().unwrap();
            for c in 0..store.num_chunks() {
                let raw = store.raw_chunk(c);
                assert_eq!(crate::codec::store::chunk_is_compact(raw), version >= crate::codec::COMPACT_STORED_SINCE, "v{version} chunk {c}");
                assert_eq!(crate::codec::store::chunk_is_uid(raw), version >= crate::codec::UID_STORED_SINCE, "v{version} chunk {c}");
            }
            assert!(store.num_chunks() < 3000 / 64, "v{version}: {} chunks", store.num_chunks());
            for (new, &(i, old)) in order.iter().enumerate() {
                assert_eq!(fields(&merged.get_doc(new as DocId).unwrap().unwrap()), fields(&doc(i as u32 + 1, old)), "v{version} doc {new}");
            }
            sizes.push((0..store.num_chunks()).map(|c| store.raw_chunk(c).len()).sum::<usize>());
        }
        assert!(sizes[0] < sizes[2], "v31 {} vs bincode {}", sizes[0], sizes[2]);
    }

    /// v35 merges move the docs' `_id`s into an id column: sorted and
    /// unsorted merges, with deleted docs, of sealed segments (binary ids
    /// in the stored docs), of a merged segment (ids in its column) with a
    /// sealed one, and nested-like docs without an id. Every doc reads back
    /// its id (get, batched reads, `keyword_value`), the stored docs hold
    /// none, and the store shrinks; a merge below v35 puts the ids back.
    #[test]
    fn merges_move_ids_into_a_column() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword), Field::new(1, "body", FieldType::Text), Field::new(2, "n", FieldType::I64)];
        let id = |seg: u32, d: u32| {
            use base64::Engine;
            let ts = 1_700_000_000_000u64 + (seg as u64) * 1_000_000 + d as u64 / 4;
            let seq = (seg as u64) << 20 | d as u64;
            let mut b = [0u8; 15];
            b[0] = seq as u8; b[1] = (seq >> 16) as u8;
            b[2..6].copy_from_slice(&[(ts >> 16) as u8, (ts >> 24) as u8, (ts >> 32) as u8, (ts >> 40) as u8]);
            b[6..12].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
            b[12..].copy_from_slice(&[(ts >> 8) as u8, (seq >> 8) as u8, ts as u8]);
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
        };
        let doc = |seg: u32, d: u32| {
            let mut doc = Document::new();
            // Every 50th doc has no id (as a nested child).
            if d % 50 != 7 { doc.add_keyword(0, &id(seg, d)); }
            doc.add_text(1, &format!("GET /english/images/{} HTTP/1.0", d % 37));
            doc.add_i64(2, (d * 2 + seg) as i64);
            doc
        };
        let want_id = |seg: u32, d: u32| (d % 50 != 7).then(|| id(seg, d));
        let seal = |dir: &RamDirectory, seg: u32, n: u32| {
            let mut w = SegmentWriter::new(seg);
            for d in 0..n { w.add_document(&doc(seg, d), &schema); }
            Arc::new(w.write_to(dir).unwrap())
        };
        let check = |merged: &Segment, expect: &[(u32, u32)], column: bool, what: &str| {
            assert_eq!(merged.uid_column_field().is_some(), column, "{what}");
            let store = merged.store().unwrap();
            let leaf = crate::query::reader::LeafReader::Segment(merged);
            let mut batched = vec![None; expect.len()];
            let all: Vec<DocId> = (0..expect.len() as DocId).collect();
            merged.get_docs_batched(&all, false, |i, d| batched[i] = d.get(0).and_then(|v| v.as_text()).map(str::to_string)).unwrap();
            for (new, &(seg, d)) in expect.iter().enumerate() {
                let got = merged.get_doc(new as DocId).unwrap().unwrap();
                assert_eq!(got.get(0).and_then(|v| v.as_text()).map(str::to_string), want_id(seg, d), "{what} doc {new}");
                assert_eq!(got.get(2).and_then(|v| v.as_i64()), Some((d * 2 + seg) as i64), "{what} doc {new}");
                assert_eq!(batched[new], want_id(seg, d), "{what} batched {new}");
                assert_eq!(leaf.keyword_value(0, new as DocId), want_id(seg, d), "{what} keyword_value {new}");
                let raw = store.get(new as DocId).unwrap().unwrap();
                assert_eq!(raw.get(0).is_some(), !column && want_id(seg, d).is_some(), "{what} stored doc {new}");
            }
        };
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
        let dir = RamDirectory::new();
        let a = seal(&dir, 1, 1300);
        let b = seal(&dir, 2, 900);
        for d in [0u32, 5, 600, 1299] { a.delete_doc(d).unwrap(); }
        let live = |seg: u32, n: u32, dead: &[u32]| (0..n).filter(|d| !dead.contains(d)).map(move |d| (seg, d)).collect::<Vec<_>>();
        let unsorted: Vec<(u32, u32)> = live(1, 1300, &[0, 5, 600, 1299]).into_iter().chain(live(2, 900, &[])).collect();
        let m1 = Arc::new(merge_segments_with(&dir, &[a.clone(), b.clone()], 10, None).unwrap());
        check(&m1, &unsorted, true, "unsorted");
        // Sorted: interleaved, descending.
        let order: Vec<(usize, DocId)> = (0..1300u32).rev().flat_map(|d| [(1usize, d), (0, d)]).filter(|&(i, d)| (i == 0 && ![0, 5, 600, 1299].contains(&d)) || (i == 1 && d < 900)).collect();
        let m2 = merge_segments_with(&dir, &[a.clone(), b.clone()], 11, Some(order.clone())).unwrap();
        let sorted: Vec<(u32, u32)> = order.iter().map(|&(i, d)| (i as u32 + 1, d)).collect();
        check(&m2, &sorted, true, "sorted");
        // The store is smaller with the column (same docs).
        let size = |s: &Segment| { let st = s.store().unwrap(); (0..st.num_chunks()).map(|c| st.raw_chunk(c).len()).sum::<usize>() };
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::UID_COLUMN_SINCE - 1));
        let old = merge_segments_with(&dir, &[a.clone(), b.clone()], 12, None).unwrap();
        check(&old, &unsorted, false, "below v35");
        let col = m1.uid_column().unwrap().byte_len();
        assert!(col < 2 * 2196, "column {col} bytes");
        assert!(size(&m1) + col + 2 * 2196 < size(&old), "store {} + column {col} vs stored ids {}", size(&m1), size(&old));
        // A merged segment (column) with a sealed one (stored ids), with
        // deletes in the merged one; then below v35 the ids go back.
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::VERSION));
        let c = seal(&dir, 3, 400);
        for d in [1u32, 2, 3, 2000] { m1.delete_doc(d).unwrap(); }
        let mixed: Vec<(u32, u32)> = unsorted.iter().enumerate().filter(|(i, _)| ![1usize, 2, 3, 2000].contains(i)).map(|(_, &x)| x).chain(live(3, 400, &[])).collect();
        let m3 = merge_segments_with(&dir, &[m1.clone(), c.clone()], 13, None).unwrap();
        check(&m3, &mixed, true, "column + sealed");
        let order3: Vec<(usize, DocId)> = (0..400u32).flat_map(|d| [(1usize, d), (0, d)]).filter(|&(i, d)| i == 1 || ![1, 2, 3].contains(&d)).collect();
        let m4 = merge_segments_with(&dir, &[m1.clone(), c.clone()], 14, Some(order3.clone())).unwrap();
        let sorted3: Vec<(u32, u32)> = order3.iter().map(|&(i, d)| if i == 1 { (3, d) } else { unsorted[d as usize] }).collect();
        check(&m4, &sorted3, true, "sorted column + sealed");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::UID_COLUMN_SINCE - 1));
        let m5 = merge_segments_with(&dir, &[m1.clone(), c.clone()], 15, None).unwrap();
        check(&m5, &mixed, false, "column + sealed below v35");
        let m6 = merge_segments_with(&dir, &[m1.clone(), c.clone()], 16, Some(order3)).unwrap();
        check(&m6, &sorted3, false, "sorted column + sealed below v35");
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
    }

    /// Packed term blocks (v32) read as unpacked ones (v31) term for term,
    /// and a merge of a v31 and a v32 segment writes every term, posting and
    /// inline doc of both: hex ids, base64 ids, keywords and text.
    #[test]
    fn packed_term_blocks_through_seal_and_mixed_merge() {
        let schema = vec![Field::new(0, "_id", FieldType::Keyword), Field::new(1, "eid", FieldType::Keyword), Field::new(2, "body", FieldType::Text), Field::new(3, "k", FieldType::Keyword)];
        let doc = |seg: u32, d: u32| {
            let h = ((seg as u64) << 40 | d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut doc = Document::new();
            doc.add_keyword(0, &format!("AaG{:016x}", h).replace('a', "_").replace('3', "Q"));
            doc.add_keyword(1, &format!("{:016x}", h));
            doc.add_text(2, &format!("get english images {} http", d % 13));
            doc.add_keyword(3, &format!("cat {}", d % 5));
            doc
        };
        let terms = |seg: &Segment| -> Vec<(Vec<u8>, u32)> {
            let mut c = seg.term_dict().cursor();
            let mut out = Vec::new();
            while c.advance().unwrap() { out.push((c.key().to_vec(), c.loc().doc_freq)); }
            out
        };
        let dir = RamDirectory::new();
        let build = |seg: u32, version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let mut w = SegmentWriter::new(seg);
            for d in 0..2500 { w.add_document(&doc(seg, d), &schema); }
            let s = Arc::new(w.write_to(&dir).unwrap());
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            s
        };
        let (old, new) = (build(1, crate::codec::PACKED_TERMS_SINCE - 1), build(1, crate::codec::PACKED_TERMS_SINCE));
        assert_eq!(terms(&old), terms(&new));
        assert!(new.term_dict().byte_len() * 10 < old.term_dict().byte_len() * 9, "{} vs {}", new.term_dict().byte_len(), old.term_dict().byte_len());
        for d in (0..2500).step_by(7) {
            let want = doc(1, d);
            for (f, v) in &want.fields {
                if let crate::types::FieldValue::Keyword(t) = v {
                    let term = crate::types::Term::new_text(*f, t);
                    assert_eq!(new.term_dict().get(&term), old.term_dict().get(&term), "{t}");
                    assert!(new.term_dict().get(&term).is_some());
                }
            }
        }
        let other = build(2, crate::codec::PACKED_TERMS_SINCE - 1);
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::PACKED_TERMS_SINCE));
        let merged = merge_segments(&dir, &[new.clone(), other.clone()], 9).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::PACKED_TERMS_SINCE - 1));
        let reference = merge_segments(&dir, &[old.clone(), other.clone()], 10).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        assert_eq!(terms(&merged), terms(&reference));
        assert_eq!(terms(&merged).len(), 2 * 2500 * 2 + 13 + 4 + 5);
        for (seg, base) in [(1u32, 0u32), (2, 2500)] {
            for d in (0..2500).step_by(11) {
                let id = format!("AaG{:016x}", ((seg as u64) << 40 | d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).replace('a', "_").replace('3', "Q");
                let loc = merged.term_dict().get(&crate::types::Term::new_text(0, &id)).expect("merged id");
                assert_eq!(crate::term_dictionary::inline_doc(loc.offset), base + d, "{id}");
            }
        }
        // v33: the merge streams packed keyword term tables; every doc's
        // hex keyword reads back as from the v32 merge, and the column is
        // smaller.
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::PACKED_KEYWORD_TABLES_SINCE));
        let packed = merge_segments(&dir, &[new.clone(), other.clone()], 11).unwrap();
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        let (a, b) = (packed.keyword_reader(1).expect("eid column"), merged.keyword_reader(1).expect("eid column"));
        for d in 0..5000 {
            assert_eq!(a.get(d), b.get(d), "doc {d}");
        }
        let fast = |s: &Segment| s.disk_usage().unwrap().fields.get(&1).map_or(0, |f| f.doc_values + f.doc_values_terms);
        assert!(fast(&packed) * 10 < fast(&merged) * 8, "{} vs {}", fast(&packed), fast(&merged));
    }

    /// Sorted segments merge k ways, in the stable sort's order (ties:
    /// the earlier segment), under a reservation of the order alone.
    #[test]
    fn sorted_segments_merge_k_ways() {
        use crate::index_sort::{IndexSort,IndexSortField,IndexSortKind};
        let dir=RamDirectory::new();
        // Field 5 ascends in each segment; equal values across them tie.
        let segments=[build_segment(&dir,1,0,300),build_segment(&dir,2,0,200),build_segment(&dir,3,1,150)];
        for reverse in [false, true] {
            let sort=IndexSort::new(vec![IndexSortField{field:5,kind:IndexSortKind::Long,reverse,max_mode:false,missing_first:false}]);
            let leaves:Vec<_>=segments.iter().map(|s|crate::query::reader::LeafReader::Segment(s)).collect();
            let docs:Vec<(usize,DocId)>=segments.iter().enumerate().flat_map(|(i,s)|(0..s.num_docs).map(move|d|(i,d))).collect();
            let want=sort.sort_docs_keyed(&leaves,docs.clone());
            // Ascending input is sorted; descending is not (keyed path).
            assert_eq!(sort.sort_docs(&leaves,docs),want,"reverse {reverse}");
        }
        let sort=IndexSort::new(vec![IndexSortField{field:5,kind:IndexSortKind::Long,reverse:false,max_mode:false,missing_first:false}]);
        let budget=crate::admission::Budget::new(1<<30);
        let memory=sort.reserve_workspace_with_budget(&segments,&budget).unwrap();
        assert!(memory.bytes() < 650 * 32 + 64 * 1024, "{}", memory.bytes());
    }

    #[test]
    fn sorting_workspace_denies_before_keys_and_covers_retained_order() {
        use crate::index_sort::{IndexSort,IndexSortField,IndexSortKind};
        let dir=RamDirectory::new();let segments=[build_segment(&dir,1,0,100)];
        let sort=IndexSort::new(vec![IndexSortField{field:4,kind:IndexSortKind::Keyword,reverse:false,max_mode:false,missing_first:false}]);
        let budget=crate::admission::Budget::new(1);
        assert!(matches!(sort.reserve_workspace_with_budget(&segments,&budget),Err(RusticError::Memory(_))));
        assert_eq!(budget.used(),0);budget.set_limit(1<<20);
        let mut memory=sort.reserve_workspace_with_budget(&segments,&budget).unwrap();
        let order=sort.sort_docs(&[crate::query::reader::LeafReader::Segment(&segments[0])],(0..100).map(|d|(0,d)).collect());
        assert_eq!(order.len(),100);
        let bytes=(order.capacity()*std::mem::size_of::<(usize,DocId)>()) as u64;
        memory.shrink(memory.bytes().saturating_sub(bytes));
        let order=crate::admission::Allocation::new(order,memory);
        assert_eq!(budget.used(),bytes);drop(order);assert_eq!(budget.used(),0);
    }

    #[test]
    fn point_map_workspace_refusal_releases_without_output() {
        let dir=RamDirectory::new();let segments=[build_segment(&dir,1,0,10)];
        let map=build_doc_map(&segments);let budget=crate::admission::Budget::new(1);
        assert!(matches!(merge_point_map_with_budget(&segments,&map,&budget),Err(RusticError::Memory(_))));
        assert_eq!(budget.used(),0);budget.set_limit(1<<20);
        let output=merge_point_map_with_budget(&segments,&map,&budget).unwrap();
        assert_eq!(budget.used(),output.capacity() as u64);drop(output);assert_eq!(budget.used(),0);
    }

    #[test]
    fn merge_extra_denies_before_build_and_retains_final_output() {
        let dir=RamDirectory::new();let segments=[build_segment(&dir,1,0,100)];
        let map=build_doc_map(&segments);let norms=merge_norms(&segments,&map).unwrap();
        let fields=BTreeMap::from([(0u16,100u64)]);
        let budget=crate::admission::Budget::new(1);
        assert!(matches!(merge_extra_with_budget(&segments,&map,&norms,&fields,crate::codec::VERSION,&budget),Err(RusticError::Memory(_))));
        assert_eq!(budget.used(),0);
        budget.set_limit(1<<24);
        let extra=merge_extra_with_budget(&segments,&map,&norms,&fields,crate::codec::VERSION,&budget).unwrap();
        assert_eq!(budget.used(),extra.capacity() as u64);
        drop(extra);assert_eq!(budget.used(),0);
    }

    #[test]
    fn fast_field_column_preflight_refuses_before_collector_growth() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir,1,0,100)];
        let map = build_doc_map(&segments);
        let budget = crate::admission::Budget::new(1);
        for (field, tag) in segments[0].fast_fields().fields() {
            assert!(matches!(fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Background(std::time::Duration::ZERO)),Err(RusticError::Memory(_))));
            assert_eq!(budget.used(),0);
        }
        budget.set_limit(1 << 24);
        let (field,tag)=segments[0].fast_fields().fields()[0];
        let memory=fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Background(std::time::Duration::ZERO)).unwrap();
        assert!(budget.used()>=64<<10 && budget.used()==memory.bytes());
        drop(memory);assert_eq!(budget.used(),0);
    }

    /// A seal's rewrite is foreground work: it may use the whole budget
    /// (not just the background share) and never waits; under forced replay
    /// even a merge's background admission is charged instead of refused.
    #[test]
    fn seal_admission_is_foreground_and_replay_is_never_refused() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir,1,0,100)];
        let map = build_doc_map(&segments);
        let (field,tag)=segments[0].fast_fields().fields()[0];
        let budget = crate::admission::Budget::new(1 << 24);
        let bytes = fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Foreground).unwrap().bytes();
        // Fits the limit, not the background three quarters of it.
        budget.set_limit(bytes + bytes / 8);
        assert!(bytes > budget.background_limit());
        assert!(matches!(fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Background(std::time::Duration::ZERO)),Err(RusticError::Memory(_))));
        let seal = fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Foreground).unwrap();
        assert_eq!(budget.used(), bytes);
        // Over the limit: a replayed seal (or merge) is charged, not refused.
        let started = std::time::Instant::now();
        let replay = crate::admission::forced(|| fast_column_memory(&segments,&map,field,tag,false,&budget,MergeAdmission::Background(MERGE_ADMISSION_WAIT))).unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert_eq!(budget.used(), 2 * bytes);
        drop(replay); drop(seal); assert_eq!(budget.used(), 0);
    }

    /// A feature (sparse-vector / rank_features) field's frequencies carry
    /// weights of up to 0x7F7F per doc and its lists have no positions: the
    /// merge workspace is sized by its docs, not by its total frequency as
    /// if it were a position count (that sized a small index's merge at
    /// about a gigabyte and failed it).
    #[test]
    fn feature_field_merges_size_their_workspace_by_docs_not_weights() {
        let schema = vec![crate::types::Field::new(0, "tokens", crate::types::FieldType::Keyword).with_features(true)];
        let dir = RamDirectory::new();
        let segments: Vec<Arc<Segment>> = (0..2u32).map(|seg| {
            let mut writer = SegmentWriter::new(seg + 1);
            for _ in 0..2000 {
                let mut doc = Document::new();
                doc.add_keyword(0, &crate::fieldtypes::features::encode("t", 30.0));
                writer.add_document(&doc, &schema);
            }
            Arc::new(writer.write_to(&dir).unwrap())
        }).collect();
        let map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &map).unwrap();
        let terms = TermMerger::new(&segments).unwrap().next_window(100).unwrap();
        let (df, ttf) = term_counts(&terms, 0).unwrap();
        assert_eq!(df, 4000);
        assert!(ttf > 4000 * 10_000, "weights are carried as large frequencies: {ttf}");
        assert!(!term_has_positions(&terms, 0, &segments));
        let (peak, _) = term_run_peak(&terms, 0..terms.len(), &segments, &map).unwrap();
        assert!(peak < 1 << 20, "a 4000-doc feature term's merge workspace is sized at {peak} bytes");
        let budget = crate::admission::Budget::new(4 << 20);
        let encoded = encode_term_run_with_budget(&terms, 0..terms.len(), &segments, &map, &norms, true, &budget).unwrap();
        let loc = encoded.1[0].expect("the term has live docs");
        assert_eq!((loc.doc_freq as u64, loc.total_term_freq), (df, ttf), "the merged list keeps every doc and weight");
    }

    #[test]
    fn postings_run_refuses_before_decode_and_retains_encoded_output() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir, 1, 0, 10)];
        let map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &map).unwrap();
        let terms = TermMerger::new(&segments).unwrap().next_window(100).unwrap();
        let budget = crate::admission::Budget::new(1);
        assert!(matches!(encode_term_run_with_budget(&terms, 0..terms.len(), &segments, &map, &norms, true, &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(),0);
        budget.set_limit(1 << 24);
        let encoded = encode_term_run_with_budget(&terms, 0..terms.len(), &segments, &map, &norms, true, &budget).unwrap();
        assert!(!encoded.0.is_empty());
        assert_eq!(budget.used(), encoded.0.capacity() as u64 + (encoded.1.capacity() * std::mem::size_of::<Option<PostingsLocation>>()) as u64);
        drop(encoded); assert_eq!(budget.used(),0);
    }

    #[test]
    fn merged_term_window_admission_retains_owner_and_refuses_before_growth() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir, 1, 0, 10)];
        let budget = crate::admission::Budget::new(1 << 20);
        let mut merger = TermMerger::with_budget(&segments, &budget).unwrap();
        let cursors = budget.used();
        budget.set_limit(cursors);
        assert!(matches!(merger.next_window(10), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), cursors);
        budget.set_limit(1 << 20);
        let terms = merger.next_window(10).unwrap();
        assert!(terms.len() > 0);
        drop(merger);
        assert!(budget.used() > 0, "window owns its buffers independently of cursors");
        drop(terms);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn sorted_store_chunk_growth_admits_reallocation_overlap_before_mutation() {
        let budget = crate::admission::Budget::new(8);
        let mut bytes = Vec::new();
        let mut memory = budget.reserve(0, "test").unwrap();
        append_store_bytes(&mut bytes, &mut memory, &[1, 2, 3, 4]).unwrap();
        assert_eq!(budget.used(), 4);
        assert!(matches!(append_store_bytes(&mut bytes, &mut memory, &[5]), Err(RusticError::Memory(_))));
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert_eq!(budget.used(), 4);
        budget.set_limit(12); // old4 + replacement8, not just final8
        append_store_bytes(&mut bytes, &mut memory, &[5]).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4, 5]);
        assert_eq!(budget.used(), 8);
        let owner = crate::admission::Allocation::new(bytes, memory);
        assert_eq!(budget.used(), 8);
        drop(owner);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn merge_norm_columns_refuse_before_growth_and_release_with_owner() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir, 1, 0, 100)];
        let map = build_doc_map(&segments);
        let budget = crate::admission::Budget::new(2148); // first 100-byte column + metadata, not the second
        assert!(matches!(merge_norms_with_budget(&segments, &map, &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), 0, "failed norm merge releases admitted prefix");
        budget.set_limit(1 << 20);
        let norms = merge_norms_with_budget(&segments, &map, &budget).unwrap();
        assert!(!norms.is_empty());
        assert!(norms.values().all(|(column, _)| column.len() == 100));
        assert!(budget.used() >= 100 * norms.len() as u64);
        drop(norms);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn sorted_store_metadata_denial_and_retained_table_lifetime() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir, 1, 0, 100)];
        let map = DocMap::from_order(&segments, (0..100).rev().map(|d| (0, d)).collect());
        let budget = crate::admission::Budget::new(32768 + 256);
        assert!(matches!(merge_store_sorted_with_budget(&dir, 22, &segments, &map, &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), 0, "traversal denial releases reader metadata");
        budget.set_limit(1 << 20);
        let store = merge_store_sorted_with_budget(&dir, 23, &segments, &map, &budget).unwrap();
        assert!(budget.used() > 0, "final encoded table retains admission");
        let bytes = store.pieces().collect::<Vec<_>>().concat();
        assert_eq!(StoreReader::open(&bytes).unwrap().num_docs(), 100);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn merge_store_metadata_refuses_growth_and_follows_encoded_output() {
        let dir = RamDirectory::new();
        let segments = [build_segment(&dir, 1, 0, 100)];
        let map = build_doc_map(&segments);
        let budget = crate::admission::Budget::new(0);
        assert!(matches!(merge_store_with_budget(&segments, &map, crate::codec::VERSION, &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), 0);
        budget.set_limit(1 << 20);
        let store = merge_store_with_budget(&segments, &map, crate::codec::VERSION, &budget).unwrap();
        assert!(budget.used() > 0, "returned chunk/table metadata stays admitted until write");
        let bytes = store.pieces().collect::<Vec<_>>().concat();
        assert_eq!(StoreReader::open(&bytes).unwrap().num_docs(), 100);
        drop(store);
        assert_eq!(budget.used(), 0);
        assert!(store_plan_bytes(usize::MAX).is_err());
    }

    #[test]
    fn merge_map_admission_retains_owner_and_rejects_deleted_rows() {
        let dir = RamDirectory::new();
        let segment = build_segment(&dir, 1, 0, 100);
        let segments = [segment.clone()];
        let budget = crate::admission::Budget::new(128);
        let map = admitted_doc_map_with_budget(&segments, None, None, &budget).unwrap();
        assert_eq!(map.num_docs(), 100);
        assert_eq!(budget.used(), 128, "clean segment only needs constant mapping metadata");
        drop(map);
        assert_eq!(budget.used(), 0);
        segment.delete_doc(0).unwrap();
        assert!(matches!(admitted_doc_map_with_budget(&segments, None, None, &budget), Err(RusticError::Memory(_))));
        assert_eq!(budget.used(), 0, "row refusal releases preliminary metadata reservation");
        budget.set_limit(528);
        let map = admitted_doc_map_with_budget(&segments, None, None, &budget).unwrap();
        assert_eq!(map.new_id(0, 0), None);
        assert_eq!(map.new_id(0, 1), Some(0));
        assert_eq!(budget.used(), 528);
        drop(map);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn stored_rewrite_admission_is_before_decode_and_output_lived() {
        let bytes = (128 << 10) + 4096 * 4;
        let budget = crate::admission::Budget::new(bytes);
        let mut workspace = stored_rewrite_memory(&budget, 4096).unwrap();
        assert!(stored_rewrite_memory(&budget, 1).is_err(), "parallel decode must not exceed shared budget");
        workspace.shrink(workspace.bytes() - 100);
        let encoded = crate::admission::Allocation::new(vec![0u8; 100], workspace);
        assert_eq!(budget.used(), 100, "output remains charged after temporary decode is gone");
        drop(encoded);
        assert_eq!(budget.used(), 0);
        assert!(stored_rewrite_memory(&budget, usize::MAX).is_err());
        assert_eq!(budget.used(), 0, "refusal cannot leak budget");
    }

    #[test]
    fn mixed_merge_policies_are_isolated_and_restore_worker_state() {
        // Keep an unthrottled job alive while another index runs throttled
        // work. Previously its node-global guard exempted the other index.
        let ready = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let (release, wait) = std::sync::mpsc::channel::<()>();
            let ready = &ready;
            scope.spawn(move || with_pacing(false, || {
                assert!(!PACING_ENABLED.with(|p| p.get()));
                ready.wait();
                // Dropping the sender also releases this worker if an
                // assertion on the other thread unwinds.
                let _ = wait.recv();
            }));
            ready.wait();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
            pool.install(|| with_pacing(true, || {
                let check = || {
                    assert!(PACING_ENABLED.with(|p| p.get()));
                    // Both policies make progress; exempt work cannot consume
                    // or reset the surrounding throttled task's work budget.
                    merge_pace(7);
                    let before = PACE.with(|p| p.get());
                    with_pacing(true, || merge_pace(3));
                    assert_eq!(PACE.with(|p| p.get().0), before.0 + 3,
                        "same-policy small tasks accumulate toward the pacing check");
                    let before = PACE.with(|p| p.get());
                    with_pacing(false, || {
                        assert!(!PACING_ENABLED.with(|p| p.get()));
                        merge_pace(11);
                    });
                    assert_eq!(PACE.with(|p| p.get()), before);
                    assert!(PACING_ENABLED.with(|p| p.get()));
                };
                paced_join(check, check);
                with_pacing(false, || {
                    paced_join(|| assert!(!PACING_ENABLED.with(|p| p.get())),
                        || assert!(!PACING_ENABLED.with(|p| p.get())));
                });
                assert!(PACING_ENABLED.with(|p| p.get()));
            }));
            drop(release);
        });
        assert!(PACING_ENABLED.with(|p| p.get()));
        let result = std::panic::catch_unwind(|| with_pacing(false, || panic!("encoder failed")));
        assert!(result.is_err());
        assert!(PACING_ENABLED.with(|p| p.get()));
    }

    #[test]
    fn real_merge_worker_yields_under_pressure_and_resumes_after_exempt_work() {
        crate::merge_pool().install(|| with_pacing(true, || {
            assert!(crate::merge_pool().current_thread_index().is_some());
            PACE.with(|p| {
                // A real burst check uses the actual turn/park/sleep path. Only
                // its pressure samples are injected to avoid global test races.
                p.set((0, Some(std::time::Instant::now() - std::time::Duration::from_millis(20))));
                let samples = std::cell::Cell::new(0);
                pace_check_with(p, || { samples.set(samples.get() + 1); true }, || SEARCH_RATE_FLOOR);
                assert!(samples.get() > 0, "real worker bypassed burst admission");
                assert!(p.get().1.is_some(), "paced work resumed after its bounded yield");
                with_pacing(false, || {
                    pace_check_with(p, || panic!("exempt work must not consult pressure"), || SEARCH_RATE_FLOOR);
                    assert!(p.get().1.is_none());
                });
                assert!(PACING_ENABLED.with(|enabled| enabled.get()));
                // Restored work can finish immediately once the burst clears.
                pace_check_with(p, || false, || SEARCH_RATE_CEILING);
                assert!(p.get().1.is_none());
                pace_release();
            });
        }));
    }

    /// ES-style adaptive merge throttle, tested on its pure controller (the
    /// node-wide rate other tests' merges change is only a wrapper):
    /// backlogged merge starts raise the rate, others lower it, within the
    /// bounds.
    #[test]
    fn adaptive_merge_throttle_follows_the_merge_backlog() {
        let mut rate = AUTO_START;
        for _ in 0..5 { rate = next_auto_rate(rate, true); }
        assert!(rate > AUTO_START, "a backlog raises the rate: {rate}");
        let raised = rate;
        for _ in 0..200 { rate = next_auto_rate(rate, true); }
        assert_eq!(rate, AUTO_MAX, "capped");
        rate = raised;
        rate = next_auto_rate(rate, false);
        assert!(rate < raised, "keeping up lowers it");
        for _ in 0..500 { rate = next_auto_rate(rate, false); }
        assert_eq!(rate, AUTO_MIN, "floored");
    }
    /// The backlog check ramps the rate every second while merges are
    /// pending (slots busy), even when no merge starts: from the floor to
    /// hundreds of MB/s within seconds of checks.
    /// Under search load the rate falls fast once the backlog is gone or
    /// shrinking; it still rises while a backlog grows.
    #[test]
    fn the_checked_rate_yields_to_searches_as_the_backlog_shrinks() {
        let high = 800u64 << 20;
        assert_eq!(next_checked_rate(high, 0, 0, true), high / 2, "no backlog while searching: halved");
        assert_eq!(next_checked_rate(high, 3, 5, true), high / 2, "shrinking while searching: halved");
        assert!(next_checked_rate(high, 6, 5, true) > high, "growing: still raised");
        assert_eq!(next_checked_rate(high, 0, 0, false), high, "idle searches: left to the merge starts");
        let mut rate = AUTO_MAX;
        for _ in 0..12 { rate = next_checked_rate(rate, 0, 0, true); }
        assert!(rate <= 4 << 20 || rate == AUTO_MIN, "12 checks left {rate}");
    }

    /// Running merges yield only in a genuine search burst: steady load
    /// with momentary queue waits never throttles them; a sustained queue
    /// (the post-preload crunch) brings the rate to the floor within a few
    /// hundred milliseconds, never below it; a merge backlog needs a larger
    /// burst; the rate climbs back to the ceiling once the queue clears.
    /// IO pressure paces merges only while the node searches: no limit
    /// without searches whatever the stall share; a sustained stall engages
    /// and reaches the floor within a second, never below it; a single
    /// short stall does not engage; in between it holds; once the stall
    /// clears the rate is back to no limit within three seconds.
    #[test]
    fn io_pressure_paces_merges_only_while_searching() {
        let t = (PRESSURE_HIGH_PPM, PRESSURE_LOW_PPM);
        let step = |st: (u64, u64), share: u64, searching: bool| next_pressure_state(st.0, st.1, share, searching, t);
        let mut st = (PRESSURE_CEILING, 0);
        for _ in 0..100 {
            st = step(st, 900_000, false);
        }
        assert_eq!(st.0, PRESSURE_CEILING, "bulk load without searches: full speed");
        // Searches start under the same stall: engages, then halves.
        st = step(st, 900_000, true);
        assert_eq!(st.0, PRESSURE_ENGAGE);
        let mut steps = 1;
        while st.0 > PRESSURE_FLOOR {
            st = step(st, 900_000, true);
            steps += 1;
        }
        assert!(steps as u64 * PRESSURE_STEP_MS <= 1000, "{steps} steps to the floor");
        for _ in 0..100 {
            st = step(st, 1_000_000, true);
        }
        assert_eq!(st.0, PRESSURE_FLOOR, "a merge never stalls");
        // Cleared: the average decays, then the rate grows back.
        let mut steps = 0;
        while st.0 < PRESSURE_CEILING {
            st = step(st, 0, true);
            steps += 1;
        }
        assert!(steps as u64 * PRESSURE_STEP_MS <= 3000, "{steps} steps back to no limit");
        // One 100 ms window at 20% stall from calm: no engagement.
        let st1 = step((PRESSURE_CEILING, 0), 200_000, true);
        assert_eq!(st1.0, PRESSURE_CEILING);
        // Between the thresholds the rate holds.
        let mut st = (32 << 20, 50_000);
        for _ in 0..50 {
            st = step(st, 50_000, true);
        }
        assert_eq!(st.0, 32 << 20);
        // Searches stop: no limit at once.
        assert_eq!(step((PRESSURE_FLOOR, 900_000), 900_000, false).0, PRESSURE_CEILING);
    }

    #[test]
    fn io_pressure_setting_and_psi_parsing() {
        assert_eq!(parse_pressure_setting(None), Some((PRESSURE_HIGH_PPM, PRESSURE_LOW_PPM)));
        assert_eq!(parse_pressure_setting(Some("off")), None);
        assert_eq!(parse_pressure_setting(Some("0")), None);
        assert_eq!(parse_pressure_setting(Some("200, 50")), Some((200_000, 50_000)));
        assert_eq!(parse_pressure_setting(Some("50,200")), Some((PRESSURE_HIGH_PPM, PRESSURE_LOW_PPM)), "low must be below high");
        assert_eq!(parse_pressure_setting(Some("junk")), Some((PRESSURE_HIGH_PPM, PRESSURE_LOW_PPM)));
        let psi = "some avg10=1.52 avg60=0.80 avg300=0.21 total=123456789\nfull avg10=0.90 avg60=0.40 avg300=0.10 total=98765\n";
        assert_eq!(psi_some_total_us(psi), Some(123_456_789));
        assert_eq!(psi_some_total_us("full avg10=0.00 total=5\n"), None);
        assert_eq!(psi_some_total_us(""), None);
    }

    /// The IO-pressure rate bounds every merge write limit: an unlimited
    /// one (force and small merges), or a higher IO limit; a lower IO
    /// limit stays; no pressure leaves the limit as it was.
    #[test]
    fn io_pressure_bounds_every_merge_write_limit() {
        assert_eq!(with_pressure(0, Some(8 << 20)), 8 << 20, "force / small merge under pressure");
        assert_eq!(with_pressure(0, None), 0, "force merge without searches: unlimited");
        assert_eq!(with_pressure(200 << 20, Some(16 << 20)), 16 << 20);
        assert_eq!(with_pressure(10 << 20, Some(64 << 20)), 10 << 20);
        assert_eq!(with_pressure(10 << 20, None), 10 << 20);
    }

    #[test]
    fn the_search_pressure_rate_yields_only_in_bursts() {
        let step = |st: (u64, u64), q: usize, b: bool| next_search_state(st.0, st.1, q, b);
        // Steady load: a queued search in one sample of twenty.
        let mut st = (SEARCH_RATE_CEILING, 0);
        for i in 0..2000 {
            st = step(st, if i % 20 == 0 { 1 } else { 0 }, false);
        }
        assert_eq!(st.0, SEARCH_RATE_CEILING, "steady load: no limit");
        // A burst: a queue of 121.
        let mut steps = 0;
        while st.0 > SEARCH_RATE_FLOOR {
            st = step(st, 121, false);
            steps += 1;
        }
        assert!(steps as u64 * SEARCH_RATE_STEP_MS <= 300, "{steps} steps to the floor");
        for _ in 0..1000 {
            st = step(st, 121, false);
        }
        assert_eq!(st.0, SEARCH_RATE_FLOOR, "a merge never stalls");
        assert!(SEARCH_RATE_FLOOR >= 5 << 20, "Lucene's minimum");
        // Cleared: the average decays (~340 ms), then the rate doubles back.
        let mut steps = 0;
        while st.0 < SEARCH_RATE_CEILING {
            st = step(st, 0, false);
            steps += 1;
        }
        assert!(steps as u64 * SEARCH_RATE_STEP_MS <= 600, "{steps} steps back to the ceiling");
        // A backlog: a moderate queue no longer holds merges back, a large
        // one still does.
        let mut st = (SEARCH_RATE_CEILING, 0);
        for _ in 0..100 {
            st = step(st, 12, true);
        }
        assert_eq!(st.0, SEARCH_RATE_CEILING, "backlogged, queue 12");
        for _ in 0..100 {
            st = step(st, 12, false);
        }
        assert_eq!(st.0, SEARCH_RATE_FLOOR, "not backlogged, queue 12");
        for _ in 0..100 {
            st = step(st, 60, true);
        }
        assert_eq!(st.0, SEARCH_RATE_FLOOR, "backlogged, queue 60");
    }

    /// While the node searches, pending merges ramp the rate per check
    /// (merges must not fall behind the segments searches read); without
    /// searches a backlog leaves the rate to the merge starts, as ES 9's
    /// auto-throttle (a bulk load defers its merging).
    #[test]
    fn pending_merges_ramp_the_merge_rate_per_check_only_while_searching() {
        note_pending_merges(0, 3);
        assert!(merges_backlogged());
        note_pending_merges(3, 0);
        let mut rate = AUTO_START;
        for _ in 0..25 { rate = next_checked_rate(rate, 3, 3, true); }
        assert!(rate >= 400 << 20, "25 backlogged checks under search left the rate at {rate} B/s");
        let mut rate = AUTO_START;
        for _ in 0..25 { rate = next_checked_rate(rate, 3, 3, false); }
        assert_eq!(rate, AUTO_START, "a bulk load's backlog holds the rate");
    }
    use crate::codec::directory::RamDirectory;
    use crate::segment::SegmentWriter;
    use crate::types::{Document, Field, FieldType};

    /// R7: merges encode (and sleep in the throttle) on the merge pool,
    /// never on encode-pool threads seals need.
    #[test]
    fn merges_never_throttle_on_encode_pool_threads() {
        let dir = RamDirectory::new();
        let segments: Vec<_> = (0..3u32).map(|id| build_segment(&dir, id, id, 200)).collect();
        let merged = merge_segments(&dir, &segments, 70).unwrap();
        assert_eq!(merged.num_docs, 600);
        let snapshot = merge_segments_snapshot(&dir, &segments, 71, &segments.iter().map(|s| s.deletions()).collect::<Vec<_>>()).unwrap();
        assert_eq!(snapshot.num_docs, 600);
        assert!(!THROTTLED_ON_ENCODE_POOL.load(std::sync::atomic::Ordering::Relaxed), "a merge wrote (and could sleep) on an encode-pool thread");
    }

    /// B6: a seal's sort rewrite is never rate-limited like a merge (ES    /// B6: a seal's sort rewrite is never rate-limited like a merge (ES
    /// throttles merges only), even under a tight fixed merge limit.
    #[test]
    fn seal_rewrites_are_never_throttled() {
        let dir = RamDirectory::new();
        let seg = build_segment(&dir, 1, 0, 5000);
        const LIMIT: u64 = 4 << 10;
        set_merge_io_limit(LIMIT);
        let t0 = std::time::Instant::now();
        let sealed = seal_segments_with(&dir, std::slice::from_ref(&seg), 90, None);
        let took = t0.elapsed();
        set_merge_io_adaptive();
        let sealed = sealed.unwrap();
        let throttled = std::time::Duration::from_secs_f64(sealed.file_bytes() as f64 / LIMIT as f64);
        assert!(throttled >= std::time::Duration::from_secs(2), "fixture too small to show throttling: {throttled:?}");
        assert!(took < throttled / 2, "the seal was throttled: {took:?} (merge rate would take {throttled:?})");
        assert_eq!(sealed.num_docs, 5000);
    }

    /// As Lucene's `ConcurrentMergeScheduler`, a merge of less than
    /// [`MIN_BIG_MERGE_BYTES`] is never IO rate limited (here under a
    /// fixed limit that would take seconds), and it leaves the adaptive
    /// rate where it was.
    #[test]
    fn small_merges_are_not_rate_limited() {
        let dir = RamDirectory::new();
        let segments: Vec<_> = (0..3u32).map(|id| build_segment(&dir, 200 + id, id, 3000)).collect();
        let input: u64 = segments.iter().map(|s| s.segment_bytes()).sum();
        assert!(input < MIN_BIG_MERGE_BYTES);
        const LIMIT: u64 = 4 << 10;
        set_merge_io_limit(LIMIT);
        let t0 = std::time::Instant::now();
        let merged = merge_segments(&dir, &segments, 290);
        let took = t0.elapsed();
        set_merge_io_adaptive();
        let merged = merged.unwrap();
        let throttled = std::time::Duration::from_secs_f64(merged.file_bytes() as f64 / LIMIT as f64);
        assert!(throttled >= std::time::Duration::from_secs(2), "fixture too small to show throttling: {throttled:?}");
        assert!(took < throttled / 2, "a small merge was rate limited: {took:?} (the limit would take {throttled:?})");
        assert_eq!(merged.num_docs, 9000);
    }

    /// ES 9's adaptive merge rate: down 10% with fewer than two merges
    /// active, up 20% past twice the processors, held in between (the old
    /// rule lowered it on every merge start without a backlog).
    #[test]
    fn the_rate_follows_es_active_merge_limits() {
        let r = 20u64 << 20;
        assert_eq!(es_next_rate(r, 1, 32), next_auto_rate(r, false), "a lone merge: down 10%");
        assert_eq!(es_next_rate(r, 2, 32), r, "two active: held");
        assert_eq!(es_next_rate(r, 32, 32), r, "at the ceiling: held");
        assert_eq!(es_next_rate(r, 33, 32), next_auto_rate(r, true), "past the ceiling: up 20%");
        let mut rate = r;
        for _ in 0..100 { rate = es_next_rate(rate, 5, 32); }
        assert_eq!(rate, r, "a steady bulk load's merges do not drain the rate");
    }

    fn schema() -> Vec<Field> {
        vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "body", FieldType::Text),
            Field::new(2, "price", FieldType::F64),
            Field::new(3, "in_stock", FieldType::Bool),
            Field::new(4, "category", FieldType::Keyword),
            Field::new(5, "count", FieldType::I64),
        ]
    }

    /// Build a segment of `n` docs with overlapping and distinct terms.
    /// `seg` shifts the vocabulary so segments share some terms.
    /// A merge of segments with many unique terms (a value per doc, as
    /// dynamic text fields give) spills its dictionary as it grows: under a
    /// budget far below the whole dictionary's accounting it completes,
    /// every term reads back, and once it is done its working memory is
    /// back on the budget (the joins track's 10M-doc load held GBs of
    /// "merge term dictionary" and refused every other merge).
    #[test]
    /// c22 cohere_vector: resident vector rows held 106 GB of a budget whose
    /// background share is 100 GB, and every merge's first reservation (a
    /// 4 KB term dictionary) was refused, so merges stopped for good. Resident
    /// data past the share must not refuse a merge: it completes, charges
    /// its working memory meanwhile, and gives it all back.
    fn a_merge_completes_with_resident_data_past_the_background_share() {
        let schema = vec![Field::new(0, "v", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let segments: Vec<Arc<Segment>> = (0..4u32).map(|s| {
            let mut writer = SegmentWriter::new(s);
            for i in 0..2_000u32 {
                let mut doc = Document::new();
                doc.add_keyword(0, &format!("value {s}_{i:05}"));
                writer.add_document(&doc, &schema);
            }
            Arc::new(writer.write_to(&dir).unwrap())
        }).collect();
        let doc_map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &doc_map).unwrap();
        let _private = crate::admission::private_shared(64 << 20);
        let shared = crate::admission::shared();
        let resident = shared.reserve(63 << 20, "resident vector rows").unwrap();
        assert!(shared.used() > shared.background_limit(), "resident data past the background share");
        // The merge's postings (term dictionary included) stream on this
        // thread under background admission, as a detached merge's do.
        let admit = MergeAdmission::Background(std::time::Duration::ZERO);
        let postings = stream_postings(&segments, &doc_map, &norms, 50_000, false, false, false, false, false, false, admit, |_| Ok(()), None)
            .expect("resident data does not refuse a merge");
        assert!(shared.background_used() > 0, "the merge's working memory is charged as background work");
        drop(postings);
        assert_eq!((shared.used(), shared.background_used()), (63 << 20, 0), "the merge gave its memory back");
        drop(resident);
    }

    #[test]
    fn unique_term_merge_spills_its_dictionary_within_a_small_budget() {
        let schema = vec![Field::new(0, "v", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let per = 100_000u32;
        let segments: Vec<Arc<Segment>> = (0..4u32).map(|s| {
            let mut writer = SegmentWriter::new(s);
            for i in 0..per {
                let mut doc = Document::new();
                doc.add_keyword(0, &format!("text with value {s}_{i:07}"));
                writer.add_document(&doc, &schema);
            }
            Arc::new(writer.write_to(&dir).unwrap())
        }).collect();
        let doc_map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &doc_map).unwrap();
        // The postings stream on this thread, a window of 50k terms at a
        // time, under a budget far below the whole dictionary's accounting
        // (about 70 MB for these terms).
        let _budget = crate::admission::private_shared(48 << 20);
        let admit = MergeAdmission::Background(std::time::Duration::ZERO);
        let unspilled = stream_postings(&segments, &doc_map, &norms, 50_000, false, false, false, false, false, false, admit, |_| Ok(()), None);
        assert!(matches!(unspilled, Err(RusticError::Memory(_))), "the whole dictionary does not fit this budget");
        assert_eq!(crate::admission::shared().used(), 0, "a refused merge gives its memory back");
        let mut spilled = Vec::new();
        let postings = stream_postings(&segments, &doc_map, &norms, 50_000, false, false, false, false, false, false, admit, |_| Ok(()),
            Some(&mut |b: &[u8]| { spilled.extend_from_slice(b); Ok(()) })).expect("the merge fits by spilling its dictionary");
        assert!(!spilled.is_empty(), "the dictionary spilled");
        assert!(crate::admission::shared().used() < 8 << 20, "the spilled dictionary is not held: {}", crate::admission::shared().used());
        spilled.extend_from_slice(&postings.terms);
        drop(postings);
        assert_eq!(crate::admission::shared().used(), 0, "merge memory left on the budget");
        let dict = crate::codec::terms::TermDictReader::open(crate::codec::directory::FileSlice::from_vec(spilled)).unwrap();
        let terms = dict.iter_all().unwrap();
        assert_eq!(terms.len(), 4 * per as usize);
        assert!(terms.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn build_segment(dir: &RamDirectory, id: u32, seg: u32, n: u32) -> Arc<Segment> {
        let schema = schema();
        let mut writer = SegmentWriter::new(id);
        for i in 0..n {
            let mut doc = Document::new();
            // Overlapping terms (same in every segment) + distinct terms.
            doc.add_text(0, "common alpha beta gamma delta");
            doc.add_text(1, &format!("common alpha word{} seg{}term{}", i % 7, seg, i));
            doc.add_f64(2, (seg * 1000 + i) as f64 * 0.5);
            doc.add_bool(3, (seg + i) % 2 == 0);
            doc.add_keyword(4, &format!("cat{}", (seg * 3 + i) % 5));
            doc.add_i64(5, (seg * 1000 + i) as i64);
            writer.add_document(&doc, &schema);
        }
        Arc::new(writer.write_to(dir).unwrap())
    }

    fn collect(mut p: PostingsReader) -> Vec<(DocId, u32, Vec<u32>)> {
        let mut v = Vec::new();
        while let Some(d) = p.next() {
            v.push((d, p.term_freq().unwrap(), p.positions().to_vec()));
        }
        v
    }

    /// Brute-force expected postings for `key` over the live docs.
    fn expected_postings(
        key: &[u8],
        segments: &[Arc<Segment>],
        doc_map: &DocMap,
    ) -> Vec<(DocId, u32, Vec<u32>)> {
        let mut out = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            let Some(loc) = seg.term_dict().lookup(key).unwrap() else {
                continue;
            };
            let mut pr = seg.postings_at(&loc).unwrap();
            while let Some(old) = pr.next() {
                if let Some(new) = doc_map.new_id(i, old) {
                    out.push((new, pr.term_freq().unwrap(), pr.positions().to_vec()));
                }
            }
        }
        out
    }

    /// The merged segment's collection statistics count only surviving
    /// docs (a merge drops deleted docs, so Lucene's `docCount` and
    /// `sumTotalTermFreq` shrink with them), for fields with and without
    /// norms.
    #[test]
    fn merge_recomputes_field_stats_from_live_docs() {
        let dir = RamDirectory::new();
        let mut segments = Vec::new();
        for id in 0..3u32 {
            let seg = build_segment(&dir, id, id, 100);
            for d in (0..100).step_by(5) {
                seg.delete_doc(d).unwrap();
            }
            segments.push(seg);
        }
        // Deleted docs still count before the merge.
        assert_eq!(segments[0].field_agg(4).unwrap().doc_count, 100);
        assert_eq!(segments[0].field_agg(0).unwrap(), crate::segment::FieldAgg { doc_count: 100, sum_ttf: 500 });
        let live: u64 = segments.iter().map(|s| s.num_live_docs() as u64).sum();
        let merged = merge_segments(&dir, &segments, 50).unwrap();
        assert_eq!(merged.num_docs as u64, live);
        // text with norms: 5 tokens per doc in field 0, 5 in field 1
        assert_eq!(merged.field_agg(0).unwrap(), crate::segment::FieldAgg { doc_count: live, sum_ttf: live * 5 });
        assert_eq!(merged.field_agg(1).unwrap(), crate::segment::FieldAgg { doc_count: live, sum_ttf: live * 4 });
        // keyword (no norms): one value per doc
        assert_eq!(merged.field_agg(4).unwrap(), crate::segment::FieldAgg { doc_count: live, sum_ttf: live });
        // numeric fields carry no collection statistics
        assert!(merged.field_agg(5).is_none());
        // a second merge round trips
        let again = merge_segments(&dir, &[Arc::new(merged)], 51).unwrap();
        assert_eq!(again.field_agg(4).unwrap().doc_count, live);
    }

    #[test]
    fn merge_three_segments_with_deletes() {
        let dir = RamDirectory::new();
        let mut segments = Vec::new();
        for id in 0..3u32 {
            let seg = build_segment(&dir, id, id, 300);
            // Delete a scattered subset.
            for d in (0..300).step_by(7) {
                seg.delete_doc(d + id).unwrap();
            }
            segments.push(seg);
        }
        let doc_map = build_doc_map(&segments);
        let total_live: u32 = segments.iter().map(|s| s.num_live_docs()).sum();
        assert_eq!(doc_map.num_docs(), total_live);

        let merged = merge_segments(&dir, &segments, 99).unwrap();
        // (a) doc count
        assert_eq!(merged.num_docs, total_live);
        assert_eq!(merged.num_live_docs(), total_live);

        // Union of all term keys.
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for seg in &segments {
            for (k, _) in seg.term_dict().iter_all().unwrap() {
                if !keys.contains(&k) {
                    keys.push(k);
                }
            }
        }
        keys.sort();
        assert!(!keys.is_empty());

        // (b) postings: doc_freq, docs, term_freqs, positions.
        for key in &keys {
            let expected = expected_postings(key, &segments, &doc_map);
            if expected.is_empty() {
                assert!(merged.term_dict().lookup(key).unwrap().is_none(), "{key:?}");
                continue;
            }
            let loc = merged.term_dict().lookup(key).unwrap().unwrap();
            assert_eq!(loc.doc_freq as usize, expected.len(), "doc_freq {key:?}");
            let total_tf: u64 = expected.iter().map(|(_, tf, _)| *tf as u64).sum();
            assert_eq!(loc.total_term_freq, total_tf, "ttf {key:?}");
            let got = collect(merged.postings_at(&loc).unwrap());
            assert_eq!(got, expected, "postings {key:?}");
        }

        // (c) fast fields + norms.
        let ff = merged.fast_fields();
        let mut new_doc = 0u32;
        for (i, seg) in segments.iter().enumerate() {
            let src = seg.fast_fields();
            for old in 0..seg.num_docs {
                let Some(new) = doc_map.new_id(i, old) else {
                    continue;
                };
                assert_eq!(new, new_doc);
                new_doc += 1;
                assert_eq!(ff.get_f64(2).unwrap().get(new), src.get_f64(2).unwrap().get(old), "f64 {new}");
                assert_eq!(ff.get_i64(5).unwrap().get(new), src.get_i64(5).unwrap().get(old), "i64 {new}");
                assert_eq!(ff.get_bool(3).unwrap().get(new), src.get_bool(3).unwrap().get(old), "bool {new}");
                assert_eq!(ff.get_keyword(4).unwrap().get(new), src.get_keyword(4).unwrap().get(old), "kw {new}");
                assert_eq!(ff.get_norms(0).unwrap().get(new), src.get_norms(0).unwrap().get(old), "norms0 {new}");
                assert_eq!(ff.get_norms(1).unwrap().get(new), src.get_norms(1).unwrap().get(old), "norms1 {new}");
            }
        }
        assert_eq!(new_doc, merged.num_docs);

        // (d) stored docs.
        let mut expected_docs = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            seg.store()
                .unwrap()
                .for_each(|old, doc| {
                    if doc_map.new_id(i, old).is_some() {
                        expected_docs.push(doc);
                    }
                })
                .unwrap();
        }
        assert_eq!(expected_docs.len() as u32, merged.num_docs);
        for (new, expected) in expected_docs.iter().enumerate() {
            let got = merged.get_doc(new as DocId).unwrap().unwrap();
            assert_eq!(format!("{got:?}"), format!("{expected:?}"), "stored doc {new}");
        }
    }

    /// Feature (sparse_vector / rank_features) postings keep their
    /// block-max data through a merge: every skip entry's `max_tf` (the
    /// block's largest quantized weight, which MaxScore and block skipping
    /// bound scores by) equals indexing the same docs into one segment.
    #[test]
    fn merged_feature_postings_keep_block_max() {
        let schema = vec![Field::new(0, "fv", FieldType::Keyword).with_features(true)];
        let doc = |g: u32| {
            let mut d = Document::new();
            for t in 0..(3 + g % 5) {
                let tok = format!("t{}", (g * 7 + t * 13) % 40);
                let w = ((g.wrapping_mul(2654435761) >> (t + 3)) % 1000) as f32 / 37.0 + 0.01;
                d.add_keyword(0, &crate::fieldtypes::features::encode(&tok, w));
            }
            d
        };
        let dir = RamDirectory::new();
        let sizes = [900u32, 333, 1500];
        let mut segments = Vec::new();
        let mut g = 0u32;
        let mut fresh = SegmentWriter::new(99);
        for (id, &n) in sizes.iter().enumerate() {
            let mut w = SegmentWriter::new(id as u32);
            for _ in 0..n {
                w.add_document(&doc(g), &schema);
                fresh.add_document(&doc(g), &schema);
                g += 1;
            }
            segments.push(Arc::new(w.write_to(&dir).unwrap()));
        }
        let merged = merge_segments(&dir, &segments, 77).unwrap();
        let fresh = fresh.write_to(&dir).unwrap();
        let (a, b) = (merged.term_dict().iter_all().unwrap(), fresh.term_dict().iter_all().unwrap());
        assert_eq!(a.len(), b.len());
        let mut blocks = 0;
        for ((ka, la), (kb, lb)) in a.iter().zip(&b) {
            assert_eq!(ka, kb);
            let (pa, pb) = (merged.postings_at(la).unwrap(), fresh.postings_at(lb).unwrap());
            assert_eq!(pa.num_full_blocks(), pb.num_full_blocks(), "{ka:?}");
            for blk in 0..pa.num_full_blocks() {
                let (sa, sb) = (pa.skip(blk), pb.skip(blk));
                assert_eq!((sa.last_doc, sa.max_tf), (sb.last_doc, sb.max_tf), "{ka:?} block {blk}");
                assert!(sa.max_tf != u32::MAX && sa.max_tf > 1, "{ka:?} block {blk}: no usable bound");
                blocks += 1;
            }
            assert_eq!(collect(pa), collect(pb), "postings {ka:?}");
        }
        assert!(blocks > 50, "{blocks}");
    }

    /// The merged segment must equal re-indexing the same live docs into
    /// one segment: every term's postings (docs, tfs, positions, skip
    /// impacts), every fast-field column and norms, and every stored doc.
    #[test]
    fn merge_equals_reindex_of_live_docs() {
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "body", FieldType::Text),
            Field::new(2, "price", FieldType::F64),
            Field::new(3, "flag", FieldType::Bool),
            Field::new(4, "cat", FieldType::Keyword),
            Field::new(5, "n", FieldType::I64),
            Field::new(6, "multi", FieldType::I64),
        ];
        let doc = |seg: u32, i: u32| {
            let mut d = Document::new();
            let g = seg * 10_000 + i;
            d.add_text(0, &format!("title{} common", g % 13));
            // Dense terms (multi-block lists), repeated terms (tf > 1) and
            // segment-local terms.
            let mut body = format!("common w{} w{} x{} seg{}", g % 3, g % 3, g % 17, seg);
            if g % 2 == 0 {
                body.push_str(" even even even");
            }
            d.add_text(1, &body);
            d.add_f64(2, (g % 977) as f64 * 0.25);
            d.add_bool(3, g % 3 == 0);
            d.add_keyword(4, &format!("c{}", g % 6));
            d.add_i64(5, g as i64 - 5000);
            for k in 0..(g % 3) {
                d.add_i64(6, (g * 10 + k) as i64);
            }
            d
        };
        let dir = RamDirectory::new();
        // Segment sizes straddle block (128) and store chunk (64) boundaries.
        let sizes = [700u32, 333, 1000, 129];
        let mut segments = Vec::new();
        for (id, &n) in sizes.iter().enumerate() {
            let mut w = SegmentWriter::new(id as u32);
            for i in 0..n {
                w.add_document(&doc(id as u32, i), &schema);
            }
            let seg = Arc::new(w.write_to(&dir).unwrap());
            match id {
                0 => {} // no deletions: constant-offset fast paths
                1 => (0..n).step_by(5).for_each(|d| seg.delete_doc(d).unwrap()),
                2 => (64..200).chain([3, 999]).for_each(|d| seg.delete_doc(d).unwrap()), // whole chunks deleted
                _ => {}
            }
            segments.push(seg);
        }
        let merged = merge_segments(&dir, &segments, 77).unwrap();

        // Re-index the live docs (from the stored copies) in merge order.
        let mut w = SegmentWriter::new(78);
        for seg in &segments {
            for d in 0..seg.num_docs {
                if !seg.is_deleted(d) {
                    w.add_document(&seg.get_doc(d).unwrap().unwrap(), &schema);
                }
            }
        }
        let fresh = w.write_to(&dir).unwrap();
        assert_eq!(merged.num_docs, fresh.num_docs);

        // Terms and postings.
        let a = merged.term_dict().iter_all().unwrap();
        let b = fresh.term_dict().iter_all().unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(merged.term_dict().num_terms(), fresh.term_dict().num_terms());
        for ((ka, la), (kb, lb)) in a.iter().zip(&b) {
            assert_eq!(ka, kb);
            assert_eq!((la.doc_freq, la.total_term_freq), (lb.doc_freq, lb.total_term_freq), "{ka:?}");
            let (pa, pb) = (merged.postings_at(la).unwrap(), fresh.postings_at(lb).unwrap());
            assert_eq!(pa.num_full_blocks(), pb.num_full_blocks());
            assert_eq!(pa.has_norm_impacts(), pb.has_norm_impacts());
            for blk in 0..pa.num_full_blocks() {
                let (sa, sb) = (pa.skip(blk), pb.skip(blk));
                assert_eq!((sa.last_doc, sa.max_tf, sa.min_norm), (sb.last_doc, sb.max_tf, sb.min_norm), "{ka:?}");
            }
            assert_eq!(collect(pa), collect(pb), "postings {ka:?}");
            assert!(merged.term_dict().lookup(ka).unwrap().is_some());
        }

        // Fast fields and norms.
        let (fa, fb) = (merged.fast_fields(), fresh.fast_fields());
        assert_eq!(fa.fields(), fb.fields());
        for d in 0..merged.num_docs {
            assert_eq!(fa.get_f64(2).unwrap().get(d), fb.get_f64(2).unwrap().get(d));
            assert_eq!(fa.get_bool(3).unwrap().get(d), fb.get_bool(3).unwrap().get(d));
            assert_eq!(fa.get_keyword(4).unwrap().get(d), fb.get_keyword(4).unwrap().get(d));
            assert_eq!(fa.get_i64(5).unwrap().get(d), fb.get_i64(5).unwrap().get(d));
            assert_eq!(fa.get_multi_i64(6).unwrap().values_vec(d), fb.get_multi_i64(6).unwrap().values_vec(d));
            for f in [0u16, 1] {
                assert_eq!(fa.get_norms(f).unwrap().get_byte(d), fb.get_norms(f).unwrap().get_byte(d));
            }
        }
        for f in [0u16, 1] {
            assert_eq!(fa.get_norms(f).unwrap().total(), fb.get_norms(f).unwrap().total());
        }
        assert_eq!(fa.get_f64(2).unwrap().range_query(10.0, 50.0), fb.get_f64(2).unwrap().range_query(10.0, 50.0));

        // Stored docs, by random access and in bulk.
        for d in 0..merged.num_docs {
            assert_eq!(format!("{:?}", merged.get_doc(d).unwrap().unwrap()), format!("{:?}", fresh.get_doc(d).unwrap().unwrap()), "doc {d}");
        }
        assert!(merged.get_doc(merged.num_docs).unwrap().is_none());
        let mut n = 0;
        merged.store().unwrap().for_each(|_, _| n += 1).unwrap();
        assert_eq!(n, merged.num_docs);
        // The file passes its checksum.
        let data = dir.open_read(&segment_file_name(77)).unwrap();
        crate::codec::open_segment_file(data, true).unwrap();
    }

    #[test]
    fn merge_on_disk_streams_a_valid_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = crate::codec::directory::MmapDirectory::open(tmp.path()).unwrap();
        let schema = schema();
        let mut segments = Vec::new();
        for id in 0..3u32 {
            let mut w = SegmentWriter::new(id);
            for i in 0..150u32 {
                let mut d = Document::new();
                d.add_text(1, &format!("common w{} s{id}", i % 9));
                d.add_f64(2, i as f64);
                w.add_document(&d, &schema);
            }
            segments.push(Arc::new(w.write_to(&dir).unwrap()));
        }
        segments[1].delete_doc(3).unwrap();
        let merged = merge_segments(&dir, &segments, 10).unwrap();
        assert_eq!(merged.num_docs, 449);
        assert!(!tmp.path().join("seg_00000010.rstc.tmp").exists());
        assert!(dir.list().unwrap().iter().all(|name| !name.starts_with(".merge-fast-")));
        crate::codec::open_segment_file(dir.open_read(&segment_file_name(10)).unwrap(), true).unwrap();
        let reopened = Segment::open(&dir, 10).unwrap();
        assert_eq!(reopened.term_dict().doc_freq(&crate::types::Term::new_text(1, "common")), 449);
        assert_eq!(reopened.get_doc(448).unwrap().unwrap().get(2).unwrap().as_f64(), Some(149.0));
    }

    #[test]
    fn sorted_store_spools_multiple_batches_in_requested_order() {
        let input = RamDirectory::new();
        let segments = vec![build_segment(&input, 0, 0, 5000), build_segment(&input, 1, 1, 5000)];
        // Reverse/interleave to exercise both reordering and chunk-cache reuse.
        let order: Vec<_> = (0..5000u32).rev().flat_map(|d| [(0, d), (1, d)])
            .filter(|&(_, d)| d != 5).collect();
        let map = DocMap::from_order(&segments, order.clone());
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = crate::codec::directory::MmapDirectory::open(tmp.path()).unwrap();
        let store = merge_store_sorted(&dir, 9, &segments, &map).unwrap();
        assert!(dir.list().unwrap().is_empty());
        let bytes = store.pieces().collect::<Vec<_>>().concat();
        let reader = StoreReader::open(&bytes).unwrap();
        assert_eq!(reader.num_docs(), order.len() as u32);
        for (new, &(segment, old)) in order.iter().enumerate() {
            let doc = reader.get(new as u32).unwrap().unwrap();
            assert_eq!(doc.get(2).unwrap().as_f64(), Some((segment as u32 * 1000 + old) as f64 * 0.5));
        }
    }

    #[test]
    fn spooled_columns_match_buffered_bytes_and_cleanup_after_read_failure() {
        #[derive(Debug, Default)]
        struct FailRead(RamDirectory);
        impl Directory for FailRead {
            fn open_read(&self, _: &str) -> Result<FileSlice> {
                Err(std::io::Error::other("injected scratch read failure").into())
            }
            fn write(&self, name: &str, data: &[u8]) -> Result<()> { self.0.write(name, data) }
            fn exists(&self, name: &str) -> bool { self.0.exists(name) }
            fn delete(&self, name: &str) -> Result<()> { self.0.delete(name) }
            fn list(&self) -> Result<Vec<String>> { self.0.list() }
            fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.0.list_admitted(temporary) }
        }
        let dir = RamDirectory::new();
        let segments = vec![build_segment(&dir, 0, 0, 300), build_segment(&dir, 1, 1, 300)];
        segments[0].delete_doc(5).unwrap();
        let map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &map).unwrap();
        let expected = merge_fast_fields_impl(&segments, &map, &norms, true);
        let spooled = spool_fast_fields(&dir, 9, &segments, &map, &norms, MergeAdmission::Background(std::time::Duration::ZERO), crate::codec::write_version()).unwrap();
        assert_eq!([spooled.directory.as_slice(), spooled.bodies.as_slice()].concat(), expected);
        assert!(dir.list().unwrap().iter().all(|name| !name.starts_with(".merge-fast-")));
        let failed = FailRead::default();
        assert!(spool_fast_fields(&failed, 9, &segments, &map, &norms, MergeAdmission::Background(std::time::Duration::ZERO), crate::codec::write_version()).is_err());
        assert!(failed.list().unwrap().is_empty());
    }

    #[test]
    fn merge_multi_valued_fast_fields_with_deletes() {
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "prices", FieldType::I64),
            Field::new(2, "tags", FieldType::Keyword),
            Field::new(3, "weights", FieldType::F64),
            Field::new(4, "ids", FieldType::U64),
        ];
        let words = ["apple", "banana", "cherry", "date", "elder"];
        let dir = RamDirectory::new();
        let mut segments = Vec::new();
        for seg_id in 0..2u32 {
            let mut w = SegmentWriter::new(seg_id);
            for i in 0..120u32 {
                let mut d = Document::new();
                d.add_text(0, &format!("doc {i}"));
                // 0..=3 values; the first doc of each segment has exactly
                // one so the field starts single-valued and gets promoted.
                let n = if i == 0 { 1 } else { (i % 4) as usize };
                for k in 0..n {
                    let v = (seg_id * 1000 + i * 10 + k as u32) as i64;
                    d.add_i64(1, v);
                    d.add_keyword(2, words[((i + k as u32) % 5) as usize]);
                    d.add_f64(3, v as f64 * 0.5);
                    d.add_u64(4, v as u64 * 2);
                }
                w.add_document(&d, &schema);
            }
            let seg = w.write_to(&dir).unwrap();
            for del in (seg_id..120).step_by(5) {
                seg.delete_doc(del).unwrap();
            }
            segments.push(Arc::new(seg));
        }
        let doc_map = build_doc_map(&segments);
        let merged = merge_segments(&dir, &segments, 50).unwrap();
        assert_eq!(merged.num_docs, doc_map.num_docs());
        let ff = merged.fast_fields();
        assert!(ff.get_i64(1).is_none());
        let mi = ff.get_multi_i64(1).unwrap();
        let mk = ff.get_multi_keyword(2).unwrap();
        let mf = ff.get_multi_f64(3).unwrap();
        let mu = ff.get_multi_u64(4).unwrap();
        assert_eq!(mi.num_docs(), merged.num_docs);

        // Brute force: every live doc's values, with remapped ids.
        let mut expected_ords: Vec<u64> = vec![0; 5];
        for (i, seg) in segments.iter().enumerate() {
            let src = seg.fast_fields();
            let si = src.get_multi_i64(1).unwrap();
            let sk = src.get_multi_keyword(2).unwrap();
            let sf = src.get_multi_f64(3).unwrap();
            let su = src.get_multi_u64(4).unwrap();
            for old in 0..seg.num_docs {
                let Some(new) = doc_map.new_id(i, old) else {
                    continue;
                };
                assert_eq!(mi.values_vec(new), si.values_vec(old), "i64 {new}");
                assert_eq!(mk.values(new), sk.values(old), "kw {new}");
                assert_eq!(mf.values_vec(new), sf.values_vec(old), "f64 {new}");
                assert_eq!(mu.values_vec(new), su.values_vec(old), "u64 {new}");
                for v in sk.values(old) {
                    expected_ords[words.iter().position(|w| *w == v).unwrap()] += 1;
                }
            }
        }
        // Term table is the union of the live docs' terms; here all 5 survive.
        assert_eq!(mk.num_terms(), 5);
        assert_eq!(mk.ordinal_counts(|_| true), expected_ords);
        // Range on the merged column agrees with brute force.
        let expected_range: Vec<DocId> = (0..merged.num_docs)
            .filter(|&d| mi.values_vec(d).iter().any(|v| *v >= 500 && *v <= 1100))
            .collect();
        assert_eq!(mi.range_query(500, 1100), expected_range);
        assert!(!expected_range.is_empty());
        // Stored docs still line up.
        let total_values: usize = (0..merged.num_docs).map(|d| mi.count(d) as usize).sum();
        assert_eq!(total_values, mi.total_values());
    }

    /// A field single-valued (and sparse) in one flushed segment and
    /// multi-valued in another keeps every doc's values through a merge, in
    /// either segment order; docs without the field stay absent (not the
    /// padding 0), and geo points survive.
    #[test]
    fn merge_mixed_single_and_multi_valued_segments() {
        use crate::query::reader::LeafReader;
        let schema = vec![
            Field::new(0, "title", FieldType::Text),
            Field::new(1, "n", FieldType::I64),
            Field::new(2, "tag", FieldType::Keyword),
            Field::new(3, "f", FieldType::F64),
            Field::new(4, "g", FieldType::GeoPoint),
        ];
        // Per segment: doc values (None = field missing); `multi` segments
        // give doc 1 two values so the writer promotes the column.
        let build = |dir: &RamDirectory, id: u32, multi: bool| -> Arc<Segment> {
            let mut w = SegmentWriter::new(id);
            for i in 0..4u32 {
                let mut d = Document::new();
                d.add_text(0, "doc");
                if i != 2 {
                    let base = id as i64 * 100 + i as i64;
                    d.add_i64(1, base);
                    d.add_keyword(2, &format!("t{base}"));
                    d.add_f64(3, base as f64 - 0.5);
                    d.add_geo_point(4, i as f64, 10.0);
                    if multi && i == 1 {
                        d.add_i64(1, base + 50);
                        d.add_keyword(2, &format!("u{base}"));
                        d.add_f64(3, base as f64 + 49.5);
                    }
                }
                w.add_document(&d, &schema);
            }
            Arc::new(w.write_to(dir).unwrap())
        };
        for multi_first in [false, true] {
            let dir = RamDirectory::new();
            let segments = vec![build(&dir, 0, multi_first), build(&dir, 1, !multi_first)];
            let merged = merge_segments(&dir, &segments, 9).unwrap();
            let m = LeafReader::Segment(&merged);
            for (si, seg) in segments.iter().enumerate() {
                let s = LeafReader::Segment(seg);
                for old in 0..4u32 {
                    let new = si as u32 * 4 + old;
                    assert_eq!(m.i64_values(1, new), s.i64_values(1, old), "i64 doc {new} multi_first={multi_first}");
                    assert_eq!(m.keyword_values(2, new), s.keyword_values(2, old), "kw doc {new}");
                    assert_eq!(m.f64_values(3, new), s.f64_values(3, old), "f64 doc {new}");
                    assert_eq!(m.geo_values(4, new), s.geo_values(4, old), "geo doc {new}");
                }
            }
            let present = vec![0, 1, 3, 4, 5, 7];
            for f in [1, 2, 3] {
                assert_eq!(m.value_presence(f).unwrap(), present, "field {f} multi_first={multi_first}");
            }
            // `exists` reads single-valued numeric columns without listing
            // docs: the same docs as `value_presence`.
            assert!(LeafReader::Segment(&segments[usize::from(multi_first)]).numeric_presence(1).is_some());
            for leaf in [LeafReader::Segment(&merged), LeafReader::Segment(&segments[0]), LeafReader::Segment(&segments[1])] {
                assert!(leaf.numeric_presence(2).is_none(), "keyword columns list their docs");
                for f in [1u16, 2, 3, 9] {
                    if let Some(p) = leaf.numeric_presence(f) {
                        let docs: Vec<u32> = match p {
                            None => (0..leaf.num_docs()).collect(),
                            Some(words) => (0..leaf.num_docs()).filter(|&d| words.get(d as usize / 64).is_some_and(|w| (*w >> (d % 64)) & 1 == 1)).collect(),
                        };
                        assert_eq!(Some(docs), leaf.value_presence(f), "field {f} multi_first={multi_first}");
                    }
                }
            }
            // The missing docs (2 and 6) are not read as 0.
            assert_eq!(m.range_f64(3, -1.0, 1.0), vec![0, 1]);
            assert_eq!(m.range_i64(1, 0, 2), vec![0, 1]);
            let shape = crate::geo::GeoShape::BoundingBox { top: 5.0, left: 0.0, bottom: -5.0, right: 20.0 };
            assert_eq!(m.geo_docs(4, &shape), present);
        }
    }

    /// A numeric column written without a points index (doc-values
    /// skipper only, `Field::points: false`) stays that way through a
    /// merge, and answers ranges like the indexed one.
    #[test]
    fn keyword_columns_merged_by_ordinal_match_rebuilt_columns() {
        // Single- and multi-valued keyword columns, a single-valued segment
        // promoted into a multi-valued merge, segments without the field,
        // missing values, terms only deleted docs used, a field whose every
        // value is deleted, shared and segment-local terms, non-ASCII.
        let schema = vec![
            Field::new(0, "id", FieldType::Keyword),
            Field::new(1, "tag", FieldType::Keyword),
            Field::new(2, "mixed", FieldType::Keyword),
            Field::new(3, "gone", FieldType::Keyword),
            Field::new(4, "n", FieldType::I64),
        ];
        let dir = RamDirectory::new();
        let mut segments = Vec::new();
        for (id, n) in [(0u32, 300u32), (1, 257), (2, 140), (3, 90)] {
            let mut w = SegmentWriter::new(id);
            for i in 0..n {
                let mut d = Document::new();
                d.add_i64(4, i as i64);
                if id != 2 && i % 7 != 3 {
                    d.add_keyword(0, &format!("id-{id}-{i:05}-é"));
                }
                if i % 4 != 0 {
                    d.add_keyword(1, &format!("t{}", (i * 31 + id) % 23));
                    if id == 1 && i % 3 == 0 {
                        d.add_keyword(1, &format!("t{}", (i * 7) % 23));
                        d.add_keyword(1, &format!("only-{id}-{i}"));
                    }
                }
                // Single-valued in segments 0 and 2, multi-valued in 1.
                d.add_keyword(2, &format!("m{}", i % 11));
                if id == 1 {
                    d.add_keyword(2, &format!("m{}", (i + 5) % 11));
                }
                if id == 3 && i < 10 {
                    d.add_keyword(3, &format!("g{i}"));
                }
                w.add_document(&d, &schema);
            }
            let seg = Arc::new(w.write_to(&dir).unwrap());
            match id {
                0 => {}
                1 => (0..n).step_by(3).for_each(|d| seg.delete_doc(d).unwrap()),
                2 => (20..90).for_each(|d| seg.delete_doc(d).unwrap()),
                _ => (0..10).for_each(|d| seg.delete_doc(d).unwrap()),
            }
            segments.push(seg);
        }
        let doc_map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &doc_map).unwrap();
        let by_ordinal = merge_fast_fields_impl(&segments, &doc_map, &norms, true);
        let rebuilt = merge_fast_fields_impl(&segments, &doc_map, &norms, false);
        assert!(by_ordinal == rebuilt, "keyword merge by ordinal differs from the rebuilt columns");
        let r = crate::codec::fastfields::FastFieldsReader::open(&by_ordinal).unwrap();
        assert!(r.get_keyword(3).is_none() && r.get_multi_keyword(3).is_none(), "a column of deleted values only is omitted");
        assert!(r.get_multi_keyword(2).is_some() && r.get_keyword(2).is_none(), "mixed columns merge into the multi-valued one");
        // Written as v26, multi-valued columns are bit-packed: still the
        // rebuilt columns' bytes, and the same values as the plain layout.
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(crate::codec::PACKED_MULTI_KEYWORD_SINCE));
        let packed_by_ordinal = merge_fast_fields_impl(&segments, &doc_map, &norms, true);
        let packed_rebuilt = merge_fast_fields_impl(&segments, &doc_map, &norms, false);
        crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
        assert!(packed_by_ordinal == packed_rebuilt, "packed keyword merge by ordinal differs from the rebuilt columns");
        let p = crate::codec::fastfields::FastFieldsReader::open(&packed_by_ordinal).unwrap();
        for field in [1u16, 2] {
            let (a, b) = (r.get_multi_keyword(field).unwrap(), p.get_multi_keyword(field).unwrap());
            assert!(!a.is_packed() && b.is_packed());
            for d in 0..doc_map.num_docs() {
                assert_eq!(a.values(d), b.values(d), "field {field} doc {d}");
            }
        }
        // Without deletions too (every segment through its base).
        let all: Vec<_> = segments.iter().take(1).cloned().collect();
        let doc_map = build_doc_map(&all);
        let norms = merge_norms(&all, &doc_map).unwrap();
        assert!(merge_fast_fields_impl(&all, &doc_map, &norms, true) == merge_fast_fields_impl(&all, &doc_map, &norms, false));
    }

    /// Keyword tables larger than the 1 MiB write buffer (a geo_shape
    /// column's encoded shapes) stream out in several pieces of one column:
    /// the same bytes as the rebuilt columns, single- and multi-valued.
    #[test]
    fn large_keyword_tables_stream_in_pieces() {
        let schema = vec![Field::new(0, "shape", FieldType::Keyword), Field::new(1, "multi", FieldType::Keyword)];
        let dir = RamDirectory::new();
        let mut segments = Vec::new();
        for id in 0..3u32 {
            let mut w = SegmentWriter::new(id);
            for i in 0..12u32 {
                let mut d = Document::new();
                let big = format!("{id}-{i}-{}", "x".repeat(150_000 + (i as usize * 7919) % 50_000));
                d.add_keyword(0, &big);
                d.add_keyword(1, &format!("a{}{}", i % 3, "y".repeat(120_000)));
                d.add_keyword(1, &format!("b{id}{i}{}", "z".repeat(90_000)));
                w.add_document(&d, &schema);
            }
            let seg = Arc::new(w.write_to(&dir).unwrap());
            if id == 1 {
                seg.delete_doc(4).unwrap();
            }
            segments.push(seg);
        }
        let doc_map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &doc_map).unwrap();
        let by_ordinal = merge_fast_fields_impl(&segments, &doc_map, &norms, true);
        let rebuilt = merge_fast_fields_impl(&segments, &doc_map, &norms, false);
        assert!(by_ordinal == rebuilt, "streamed keyword tables differ from the rebuilt columns");
        let r = crate::codec::fastfields::FastFieldsReader::open(&by_ordinal).unwrap();
        assert_eq!(r.get_keyword(0).unwrap().num_terms(), 35);
    }

    #[test]
    fn streamed_postings_do_not_depend_on_window_size() {
        let dir = RamDirectory::new();
        let segments: Vec<_> = (0..4u32).map(|i| build_segment(&dir, i, i, 150 + i * 97)).collect();
        segments[1].delete_doc(3).unwrap();
        (40..90).for_each(|d| segments[2].delete_doc(d).unwrap());
        let doc_map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &doc_map).unwrap();
        let run = |window: usize| {
            let mut bytes = Vec::new();
            let p = stream_postings(&segments, &doc_map, &norms, window, false, false, false, false, false, false, MergeAdmission::Background(std::time::Duration::ZERO), |b| {
                bytes.extend_from_slice(b);
                Ok(())
            }, None)
            .unwrap();
            (bytes, p.terms, p.field_ttf)
        };
        let whole = run(usize::MAX);
        assert!(!whole.0.is_empty() && !whole.1.is_empty());
        for window in [1, 7, 64] {
            assert!(run(window) == whole, "window {window}");
        }
    }

    #[test]
    fn doc_map_keeps_no_row_for_segments_without_deletions() {
        let schema = vec![Field::new(0, "n", FieldType::I64)];
        let dir = RamDirectory::new();
        let segments: Vec<_> = (0..3u32)
            .map(|id| {
                let mut w = SegmentWriter::new(id);
                for i in 0..100 {
                    let mut d = Document::new();
                    d.add_i64(0, i);
                    w.add_document(&d, &schema);
                }
                Arc::new(w.write_to(&dir).unwrap())
            })
            .collect();
        segments[1].delete_doc(0).unwrap();
        segments[1].delete_doc(99).unwrap();
        let m = build_doc_map(&segments);
        assert_eq!(m.num_docs(), 298);
        assert!(m.remap[0].is_empty() && m.remap[2].is_empty() && m.remap[1].len() == 100);
        assert_eq!((m.base(0), m.base(1), m.base(2)), (Some(0), None, Some(198)));
        assert_eq!((m.new_id(0, 99), m.new_id(1, 0), m.new_id(1, 1), m.new_id(1, 99), m.new_id(2, 0)), (Some(99), None, Some(100), None, Some(198)));
    }

    #[test]
    fn merge_keeps_skipper_columns() {
        let dir = RamDirectory::new();
        let schema = vec![Field::new(5, "ts", FieldType::I64).with_points(false), Field::new(6, "n", FieldType::I64)];
        let mut segments = Vec::new();
        for id in 0..2u32 {
            let mut w = SegmentWriter::new(id);
            for i in 0..300u32 {
                let mut doc = Document::new();
                doc.add_i64(5, 10_000 - (id * 300 + i) as i64);
                doc.add_i64(6, (id * 300 + i) as i64);
                w.add_document(&doc, &schema);
            }
            let seg = Arc::new(w.write_to(&dir).unwrap());
            assert!(seg.fast_fields().get_i64(5).unwrap().skips_points());
            segments.push(seg);
        }
        segments[0].delete_doc(3).unwrap();
        let merged = merge_segments(&dir, &segments, 9).unwrap();
        let (ts, n) = (merged.fast_fields().get_i64(5).unwrap(), merged.fast_fields().get_i64(6).unwrap());
        assert!(ts.skips_points() && !ts.has_points());
        assert!(!n.skips_points() && n.has_points());
        assert_eq!(ts.count_range(9_500, 9_800), (0..merged.num_docs).filter(|&d| ts.get(d).is_some_and(|v| (9_500..=9_800).contains(&v))).count() as u64);
    }

    /// A segment whose docs all hold one value is written without the
    /// points index for that shape alone; merged with a segment that has
    /// the index, the merged column has it again (value-ordered sorts and
    /// script top-k read it). A doc-values-only field stays without, even
    /// when one of its inputs is constant.
    #[test]
    fn merge_restores_points_dropped_for_a_constant_segment() {
        let dir = RamDirectory::new();
        let schema = vec![Field::new(5, "ts", FieldType::I64).with_points(false), Field::new(6, "n", FieldType::I64)];
        let value = |id: u32, i: u32| if id == 0 { 0i64 } else { (i * 7 % 300) as i64 };
        let mut segments = Vec::new();
        for id in 0..2u32 {
            let mut w = SegmentWriter::new(id);
            for i in 0..300u32 {
                let mut doc = Document::new();
                doc.add_i64(5, if id == 0 { 7 } else { i as i64 });
                doc.add_i64(6, value(id, i));
                w.add_document(&doc, &schema);
            }
            segments.push(Arc::new(w.write_to(&dir).unwrap()));
        }
        let n0 = segments[0].fast_fields().get_i64(6).unwrap();
        assert!(n0.skips_points() && n0.written_constant());
        assert!(segments[1].fast_fields().get_i64(6).unwrap().has_points());
        let merged = merge_segments(&dir, &segments, 9).unwrap();
        let (ts, n) = (merged.fast_fields().get_i64(5).unwrap(), merged.fast_fields().get_i64(6).unwrap());
        assert!(ts.skips_points() && !ts.has_points());
        assert!(!n.skips_points() && n.has_points() && n.counts_from_points());
        assert_eq!(n.points_min_max(), Some((0, 299)));
        let mut desc = Vec::new();
        n.visit_desc(|d, v| { desc.push((v, d)); true });
        let mut expect: Vec<(i64, DocId)> = (0..2u32).flat_map(|id| (0..300u32).map(move |i| (value(id, i), id * 300 + i))).collect();
        expect.sort_by(|a, b| b.0.cmp(&a.0));
        assert_eq!(desc.iter().map(|p| p.0).collect::<Vec<_>>(), expect.iter().map(|p| p.0).collect::<Vec<_>>());
        assert_eq!(n.count_range(0, 0), (0..merged.num_docs).filter(|&d| n.get(d) == Some(0)).count() as u64);
    }

    #[test]
    fn merge_single_segment_no_deletes_is_identity() {
        let dir = RamDirectory::new();
        let seg = build_segment(&dir, 0, 0, 300);
        let merged = merge_segments(&dir, std::slice::from_ref(&seg), 7).unwrap();
        assert_eq!(merged.num_docs, seg.num_docs);

        // (e) term/postings streams are identical (compared by iteration).
        let src_terms = seg.term_dict().iter_all().unwrap();
        let merged_terms = merged.term_dict().iter_all().unwrap();
        assert_eq!(src_terms.len(), merged_terms.len());
        for ((k1, l1), (k2, l2)) in src_terms.iter().zip(&merged_terms) {
            assert_eq!(k1, k2);
            assert_eq!(l1.doc_freq, l2.doc_freq);
            assert_eq!(l1.total_term_freq, l2.total_term_freq);
            let a = collect(seg.postings_at(l1).unwrap());
            let b = collect(merged.postings_at(l2).unwrap());
            assert_eq!(a, b, "postings {k1:?}");
        }
        // Fast fields and store too.
        for new in 0..merged.num_docs {
            assert_eq!(
                merged.fast_fields().get_f64(2).unwrap().get(new),
                seg.fast_fields().get_f64(2).unwrap().get(new)
            );
            assert_eq!(
                format!("{:?}", merged.get_doc(new).unwrap().unwrap()),
                format!("{:?}", seg.get_doc(new).unwrap().unwrap())
            );
        }
    }

    #[test]
    fn merge_all_deleted_yields_empty_segment() {
        let dir = RamDirectory::new();
        let seg = build_segment(&dir, 0, 0, 50);
        for d in 0..50 {
            seg.delete_doc(d).unwrap();
        }
        let merged = merge_segments(&dir, std::slice::from_ref(&seg), 8).unwrap();
        assert_eq!(merged.num_docs, 0);
        assert_eq!(merged.num_live_docs(), 0);
        assert_eq!(merged.term_dict().num_terms(), 0);
        assert!(merged.fast_fields().fields().is_empty());
        assert!(merged.get_doc(0).unwrap().is_none());
    }

    #[test]
    fn doc_map_renumbers_contiguously() {
        let dir = RamDirectory::new();
        let s0 = build_segment(&dir, 0, 0, 10);
        let s1 = build_segment(&dir, 1, 1, 10);
        s0.delete_doc(0).unwrap();
        s0.delete_doc(3).unwrap();
        s1.delete_doc(9).unwrap();
        let m = build_doc_map(&[s0.clone(), s1.clone()]);
        assert_eq!(m.num_docs(), 17);
        assert_eq!(m.new_id(0, 0), None);
        assert_eq!(m.new_id(0, 1), Some(0));
        assert_eq!(m.new_id(0, 3), None);
        assert_eq!(m.new_id(0, 4), Some(2));
        assert_eq!(m.new_id(0, 9), Some(7));
        assert_eq!(m.new_id(1, 0), Some(8));
        assert_eq!(m.new_id(1, 9), None);
    }

    #[test]
    fn merge_policy_tiering() {
        let p = MergePolicy {
            max_segments_per_tier: 3,
            max_merge_at_once: 0,
            floor_segment_docs: 100,
            max_merged_docs: 1_000_000,
            deletes_pct_allowed: 0.2,
            max_segment_count: 0,
            tiered: None,
            log_bytes: None,
        };
        // 4 segments in one tier (all within 3x of 1000) -> merge the 3 smallest.
        let segs = vec![
            (1u32, 1000, 1000),
            (2, 1100, 1100),
            (3, 1200, 1200),
            (4, 1300, 1300),
        ];
        let merges = p.find_merges(&segs);
        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].segment_ids, vec![1, 2, 3]);

        // Exactly max_segments_per_tier in a tier -> no merge.
        let segs = vec![(1u32, 1000, 1000), (2, 1100, 1100), (3, 1200, 1200)];
        assert!(p.find_merges(&segs).is_empty());

        // Below the floor belongs to the floor tier, rather than escaping compaction.
        let segs = vec![(1u32, 90, 90), (2, 90, 90), (3, 90, 90), (4, 90, 90)];
        assert_eq!(p.find_merges(&segs)[0].segment_ids, vec![1, 2, 3]);

        // Two tiers: 4 small + 2 large.
        let segs = vec![
            (1u32, 1000, 1000),
            (2, 1000, 1000),
            (3, 1000, 1000),
            (4, 1000, 1000),
            (5, 100_000, 100_000),
            (6, 100_000, 100_000),
        ];
        let merges = p.find_merges(&segs);
        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].segment_ids, vec![1, 2, 3]);
    }

    /// Delete weighting: a solo rewrite reclaiming most of a segment ranks
    /// with small merges.
    #[test]
    fn merge_weight_favours_reclaiming_deletes() {
        assert!(merge_weight(100_000, 400_000) < merge_weight(10_000, 10_000), "75% deleted: {}", merge_weight(100_000, 400_000));
        assert!(merge_weight(900_000, 1_000_000) > merge_weight(100_000, 100_000));
        assert_eq!(merge_weight(0, 0), 0);
    }

    /// Smaller merges are proposed first: a burst of tiny segments goes
    /// before a large over-full tier.
    #[test]
    fn merge_policy_proposes_the_smallest_merges_first() {
        let p = MergePolicy { max_segments_per_tier: 3, max_merge_at_once: 0, floor_segment_docs: 100, max_merged_docs: 10_000_000, deletes_pct_allowed: 0.2, max_segment_count: 0, tiered: None, log_bytes: None };
        let mut segs: Vec<(u32, u32, u32)> = (1..=4).map(|id| (id, 100_000, 100_000)).collect();
        segs.extend((10..30).map(|id| (id, 50, 50)));
        let merges = p.find_merges(&segs);
        assert!(merges.len() >= 2, "{}", merges.len());
        assert!(merges[0].segment_ids.iter().all(|id| *id >= 10), "{:?}", merges[0].segment_ids);
    }

    #[test]
    fn merge_policy_deletes_trigger() {
        let p = MergePolicy {
            max_segments_per_tier: 10,
            max_merge_at_once: 0,
            floor_segment_docs: 100,
            max_merged_docs: 1_000_000,
            deletes_pct_allowed: 0.2,
            max_segment_count: 0,
            tiered: None,
            log_bytes: None,
        };
        // 30% deleted -> solo rewrite proposed.
        let segs = vec![(1u32, 1000, 700)];
        let merges = p.find_merges(&segs);
        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].segment_ids, vec![1]);

        // 10% deleted -> nothing.
        let segs = vec![(1u32, 1000, 900)];
        assert!(p.find_merges(&segs).is_empty());

        // Deleted ratio high but live docs exceed max_merged_docs -> nothing.
        let p = MergePolicy {
            max_merged_docs: 500,
            ..p
        };
        let segs = vec![(1u32, 1000, 700)];
        assert!(p.find_merges(&segs).is_empty());
    }

    /// Past the segment budget the smallest segments no tier took merge,
    /// enough to reach it; a large one is not rewritten with a few tiny
    /// ones.
    #[test]
    fn merge_policy_segment_budget_collapses_the_tiny_tail() {
        let p = MergePolicy { max_segments_per_tier: 8, max_merge_at_once: 16, floor_segment_docs: 5_000, max_merged_docs: 10_000_000, deletes_pct_allowed: 0.2, max_segment_count: 10, tiered: None, log_bytes: None };
        // Four large segments and eight of 4k..10k docs: no tier is over full.
        let mut segs: Vec<(u32, u32, u32)> = (1..=4).map(|id| (id, 400_000, 400_000)).collect();
        segs.extend((5..=12).map(|id| (id, 4_000 + id * 500, 4_000 + id * 500)));
        assert!(MergePolicy { max_segment_count: 0, ..p }.find_merges(&segs).is_empty());
        let m = p.find_merges(&segs);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].segment_ids, vec![5, 6, 7], "12 segments: the three smallest merge to reach 10");
        // A large one is never the bulk of a budget merge.
        let segs = vec![(1, 400_000, 400_000), (2, 300_000, 300_000), (3, 4_000, 4_000)];
        assert!(MergePolicy { max_segment_count: 2, ..p }.find_merges(&segs).is_empty());
    }

    #[test]
    fn merge_policy_max_merged_docs_cap() {
        let p = MergePolicy {
            max_segments_per_tier: 3,
            max_merge_at_once: 0,
            floor_segment_docs: 100,
            max_merged_docs: 2500,
            deletes_pct_allowed: 0.2,
            max_segment_count: 0,
            tiered: None,
            log_bytes: None,
        };
        // 4 segments of 1000 live docs: the 3 smallest would total 3000 > 2500.
        let segs = vec![
            (1u32, 1000, 1000),
            (2, 1000, 1000),
            (3, 1000, 1000),
            (4, 1000, 1000),
        ];
        assert!(p.find_merges(&segs).is_empty());
    }
}

#[cfg(test)]
mod shape_merge_tests {
    use super::*;
    use crate::codec::directory::RamDirectory;
    use crate::fieldtypes::shape::{self, CentroidCalc, ParseOpts};
    use crate::fieldtypes::tessellator::geometry_triangles;
    use crate::segment::SegmentWriter;
    use crate::types::{Document, Field, FieldType};
    use std::collections::HashMap;

    fn wkt(i: u32) -> String {
        match i % 3 {
            0 => format!("POINT ({} {})", i % 170, i % 80),
            1 => format!("LINESTRING (0 0, {} 1, 2 {})", 1 + i % 5, 1 + i % 7),
            _ => format!("POLYGON ((0 0, {} 0, {} {}, 0 {}, 0 0))", 1 + i % 9, 1 + i % 9, 1 + i % 3, 1 + i % 3),
        }
    }

    /// Doc `i`'s triangles and doc value (`None`: every tenth doc has no value).
    fn expected(i: u32) -> Option<(Vec<[i32; 7]>, Vec<u8>)> {
        if i % 10 == 9 {
            return None;
        }
        let g = shape::parse_wkt(&wkt(i), &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
        let mut t = Vec::new();
        geometry_triangles(&g, &mut t).unwrap();
        let mut c = CentroidCalc::default();
        c.add(&g);
        let v = crate::fieldtypes::shape_field::doc_value(&t, &c).unwrap();
        Some((t, v))
    }

    /// 300 docs of `geo_shape` field 0 (indexed, with doc values).
    fn build(dir: &RamDirectory, id: u32) -> Arc<Segment> {
        let schema = vec![Field::new(0, "s", FieldType::Keyword).with_shape(true)];
        let mut w = SegmentWriter::new(id);
        for i in 0..300u32 {
            let mut d = Document::new();
            if i % 10 != 9 {
                let g = shape::parse_wkt(&wkt(i), &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
                let mut t = Vec::new();
                geometry_triangles(&g, &mut t).unwrap();
                d.add_shape_triangles(0, t);
                let mut c = CentroidCalc::default();
                c.add(&g);
                d.set_shape_centroid(0, c);
            }
            w.add_document(&d, &schema);
        }
        Arc::new(w.write_to(dir).unwrap())
    }

    fn tree_of(parts: &[crate::fieldtypes::shape_bkd::ShapeBkd]) -> HashMap<DocId, Vec<[i32; 7]>> {
        let mut docs: HashMap<DocId, Vec<[i32; 7]>> = HashMap::new();
        for p in parts {
            p.for_each(&mut |d, t| docs.entry(d).or_default().push(*t));
        }
        for v in docs.values_mut() {
            v.sort_unstable();
        }
        docs
    }

    /// A segment holds every doc's triangles in its tree and its value in
    /// the doc values; a merge renumbers both (deleted docs dropped) and
    /// keeps every value as written; a merge into a format before v23
    /// refuses shape data.
    #[test]
    fn tree_and_values_written_and_merged() {
        let dir = RamDirectory::new();
        let a = build(&dir, 1);
        let b = build(&dir, 2);
        let data = a.shape_data(0).expect("shape data written");
        let tree = tree_of(&data.tree);
        for i in 0..300u32 {
            match expected(i) {
                None => {
                    assert!(!tree.contains_key(&i));
                    assert!(data.values.as_ref().unwrap().get(i).is_none());
                }
                Some((mut t, v)) => {
                    t.sort_unstable();
                    assert_eq!(tree.get(&i), Some(&t), "{i}");
                    assert_eq!(data.values.as_ref().unwrap().get(i).map(|x| x.to_es_bytes()), Some(v), "{i}");
                }
            }
        }
        b.delete_doc(5).unwrap();
        b.delete_doc(6).unwrap();
        let segments = [a.clone(), b.clone()];
        let map = build_doc_map(&segments);
        let norms = merge_norms(&segments, &map).unwrap();
        let extra = merge_extra_with_budget(&segments, &map, &norms, &BTreeMap::new(), crate::codec::VERSION, &crate::admission::shared()).unwrap();
        let mut merged = crate::fieldtypes::shape_field::ShapeFieldData::default();
        for (t, f, bytes) in crate::codec::decode_extra(&extra).unwrap() {
            if f != 0 {
                continue;
            }
            let slice = crate::codec::directory::FileSlice::from_vec(bytes.to_vec());
            if t == crate::codec::EXTRA_SHAPE_TREE {
                merged.tree.push(crate::fieldtypes::shape_bkd::ShapeBkd::open(slice).unwrap());
            } else if t == crate::codec::EXTRA_SHAPE_DV {
                merged.values.get_or_insert_with(Default::default).push_part(crate::fieldtypes::shape_dv::DvPart::open(slice).unwrap());
            }
        }
        let tree = tree_of(&merged.tree);
        for (si, deleted) in [(0usize, &[][..]), (1usize, &[5u32, 6][..])] {
            for i in 0..300u32 {
                let new_id = map.new_id(si, i);
                if deleted.contains(&i) {
                    assert!(new_id.is_none());
                    continue;
                }
                let d = new_id.unwrap();
                match expected(i) {
                    None => assert!(!tree.contains_key(&d) && merged.values.as_ref().unwrap().get(d).is_none()),
                    Some((mut t, v)) => {
                        t.sort_unstable();
                        assert_eq!(tree.get(&d), Some(&t), "{si} {i}");
                        assert_eq!(merged.values.as_ref().unwrap().get(d).map(|x| x.to_es_bytes()), Some(v), "{si} {i}");
                    }
                }
            }
        }
        // Into a format before shapes' own: refused.
        assert!(merge_extra_with_budget(&segments, &map, &norms, &BTreeMap::new(), crate::codec::SHAPE_TREE_SINCE - 1, &crate::admission::shared()).is_err());
    }
}
