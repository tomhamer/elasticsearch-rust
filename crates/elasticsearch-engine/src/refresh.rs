//! Scheduled refresh (ES `index.refresh_interval`, default `1s`, `-1`
//! disables) and search-idle shards (`index.search.idle.after`, default
//! `30s`).
//!
//! One process-wide scheduler thread walks every open shard copy: a copy
//! whose reader is stale (writes not yet visible to searches, or data
//! sealed in the background since its last refresh) is refreshed once its
//! interval has passed since the last refresh. As in ES, a copy whose index does
//! not set `index.refresh_interval` explicitly and that has not been
//! searched for `index.search.idle.after` is *search-idle*: scheduled
//! refreshes are skipped (bulk indexing does not pay for them) and the next
//! search refreshes it first ([`crate::shard::Shard::before_search`]).

use crate::index::IndexShard;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Scheduler tick.
const TICK: Duration = Duration::from_millis(20);
/// ES default `index.search.idle.after`.
const DEFAULT_IDLE_AFTER_MS: i64 = 30_000;

fn now_ms() -> u64 {
    static BASE: OnceLock<Instant> = OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Per shard copy refresh bookkeeping.
pub struct RefreshState {
    last_refresh: AtomicU64,
    last_search: AtomicU64,
    last_write: AtomicU64,
    /// Writes noted so far (a generation).
    writes: AtomicU64,
    /// The write generation the last completed refresh covers: writes
    /// noted before it started. Dirty while `writes` is past it, so a
    /// write landing while a refresh runs keeps the copy dirty.
    published: AtomicU64,
    /// A scheduled refresh was skipped because the copy was search-idle.
    pub(crate) skipped: AtomicBool,
    /// The copy is search-idle (as of the scheduler's last tick): a search
    /// on it refreshes first when it holds writes not yet visible, even
    /// before a tick had the chance to skip a refresh.
    pub(crate) idle: AtomicBool,
}

impl Default for RefreshState {
    fn default() -> Self {
        let now = now_ms();
        // A new copy counts as just searched (ES initialises
        // `lastSearcherAccess` when the shard is created).
        RefreshState { last_refresh: AtomicU64::new(now), last_search: AtomicU64::new(now), last_write: AtomicU64::new(now), writes: AtomicU64::new(0), published: AtomicU64::new(0), skipped: AtomicBool::new(false), idle: AtomicBool::new(false) }
    }
}

impl RefreshState {
    /// A write completed (called once its documents are where a refresh
    /// that starts after this finds them).
    pub fn note_write(&self) {
        self.last_write.store(now_ms(), Ordering::Relaxed);
        self.writes.fetch_add(1, Ordering::AcqRel);
    }

    /// A refresh is starting (before it takes the completed writes): returns
    /// the write generation it will cover, for [`RefreshState::refreshed`].
    pub fn note_refresh(&self) -> u64 {
        self.skipped.store(false, Ordering::Release);
        self.last_refresh.store(now_ms(), Ordering::Relaxed);
        self.writes.load(Ordering::Acquire)
    }

    /// The refresh started at `generation` published: the copy is clean
    /// unless a write was noted since it started.
    pub fn refreshed(&self, generation: u64) {
        self.published.fetch_max(generation, Ordering::AcqRel);
    }

    /// Time since the last write to the copy.
    pub fn write_idle(&self) -> Duration {
        Duration::from_millis(now_ms().saturating_sub(self.last_write.load(Ordering::Relaxed)))
    }

    pub fn note_search(&self) {
        self.last_search.store(now_ms(), Ordering::Relaxed);
        self.idle.store(false, Ordering::SeqCst);
    }

    /// Whether a scheduled refresh was skipped while search-idle (and
    /// clears it).
    pub fn take_skipped(&self) -> bool {
        self.skipped.load(Ordering::Acquire) && self.skipped.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.writes.load(Ordering::Acquire) > self.published.load(Ordering::Acquire)
    }
}

fn registry() -> &'static Mutex<Vec<Weak<IndexShard>>> {
    static R: OnceLock<Mutex<Vec<Weak<IndexShard>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// A pool of background workers draining one queue of shard copies.
fn spawn_workers(name: &'static str, n: usize, pass: fn(&Weak<IndexShard>)) -> std::sync::mpsc::SyncSender<Weak<IndexShard>> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Weak<IndexShard>>(4096);
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..n {
        let rx = rx.clone();
        let _ = std::thread::Builder::new().name(format!("{name}-{i}")).spawn(move || loop {
            let next = rx.lock().recv();
            let Ok(w) = next else { return };
            pass(&w);
        });
    }
    tx
}

/// Background refresh / seal / repair workers (ES's `refresh` and `flush`
/// threads, `max(1, min(5, cores / 2))`): never behind a merge, however
/// long it runs.
fn seal_workers() -> &'static std::sync::mpsc::SyncSender<Weak<IndexShard>> {
    static W: OnceLock<std::sync::mpsc::SyncSender<Weak<IndexShard>>> = OnceLock::new();
    W.get_or_init(|| {
        let n = crate::processors::get().max(1).div_euclid(2).clamp(1, 5);
        spawn_workers("es-refresh-seal", n, seal_pass)
    })
}

/// Background merge workers: a merge waits on its own worker (throttled),
/// and every copy runs up to its `max_thread_count` merges (ES: one merge
/// scheduler per shard), so the node has as many workers as cores (the
/// merges' encoding shares the niced merge pool of that size).
fn merge_workers() -> &'static std::sync::mpsc::SyncSender<Weak<IndexShard>> {
    static W: OnceLock<std::sync::mpsc::SyncSender<Weak<IndexShard>>> = OnceLock::new();
    W.get_or_init(|| {
        let n = crate::processors::get().max(1);
        spawn_workers("es-merge", n, merge_pass)
    })
}

/// One refresh / seal pass over a copy: a deferred refresh, vector repair,
/// then a due seal. The copy is only referenced (upgraded) to detach and to
/// install a job, never while it encodes: dropping the copy meanwhile
/// releases its shard lock at once (the result is dropped).
fn seal_pass(w: &Weak<IndexShard>) {
    let live = |w: &Weak<IndexShard>| w.upgrade().filter(|s| !s.is_closed());
    let Some(s) = live(w) else { return };
    s.shard.bg.queued.store(false, Ordering::Release);
    // A refresh the scheduler could not take without waiting goes first
    // (visibility latency); this worker may wait for the core, the
    // scheduler does not.
    if s.shard.bg.want_refresh.swap(false, Ordering::AcqRel) {
        s.shard.refresh_deferred();
    }
    if s.shard.bg.want_repair.swap(false, Ordering::AcqRel) {
        s.shard.repair_vectors();
    }
    let seal = if s.shard.bg.want_seal.swap(false, Ordering::AcqRel) { s.shard.start_seal() } else { None };
    // A backlog left unclaimed: another worker starts the next seal job of
    // this copy while this one encodes (jobs seal disjoint memtables).
    if seal.is_some() && s.shard.bg.want_seal.load(Ordering::Acquire) {
        dispatch_background(&s);
    }
    drop(s);
    if let Some(task) = seal {
        let done = task.run();
        if let Some(s) = live(w) {
            s.shard.finish_seal(done);
            // New segments: their merges go to the merge workers.
            dispatch_background(&s);
        }
    }
}

/// One merge pass over a copy: merge until the policy is satisfied.
fn merge_pass(w: &Weak<IndexShard>) {
    let live = |w: &Weak<IndexShard>| w.upgrade().filter(|s| !s.is_closed());
    let Some(s) = live(w) else { return };
    s.shard.bg.merge_queued.store(false, Ordering::Release);
    drop(s);
    loop {
        let Some(s) = live(w) else { return };
        if !s.shard.bg.want_merge.swap(false, Ordering::AcqRel) {
            return;
        }
        let Some(task) = s.shard.start_merge() else { return };
        // A backlog: another worker starts a concurrent merge of this copy.
        if s.shard.bg.want_merge.load(Ordering::Acquire) {
            dispatch_background(&s);
        }
        drop(s);
        let done = task.run();
        let Some(s) = live(w) else { return };
        if s.shard.finish_merge(done) {
            s.shard.bg.want_merge.store(true, Ordering::Release);
        }
    }
}

/// Hand a copy with due background work to its workers: refresh / seal /
/// repair work and merges each go to their own workers (once queued, a
/// copy is not queued again on that side until a worker picks it up).
fn dispatch_background(s: &Arc<IndexShard>) {
    let bg = &s.shard.bg;
    let seal_due = bg.want_seal.load(Ordering::Acquire) || bg.want_refresh.load(Ordering::Acquire) || bg.want_repair.load(Ordering::Acquire);
    if seal_due && !bg.queued.swap(true, Ordering::AcqRel) && seal_workers().try_send(Arc::downgrade(s)).is_err() {
        bg.queued.store(false, Ordering::Release);
    }
    if bg.want_merge.load(Ordering::Acquire) && !bg.merge_queued.swap(true, Ordering::AcqRel) && merge_workers().try_send(Arc::downgrade(s)).is_err() {
        bg.merge_queued.store(false, Ordering::Release);
    }
}

/// Put a shard copy under the scheduler (starting it on first use).
pub fn register(shard: &Arc<IndexShard>) {
    static STARTED: OnceLock<()> = OnceLock::new();
    shard.shard.bg.served.store(true, Ordering::Release);
    registry().lock().push(Arc::downgrade(shard));
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("es-refresh".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            tick();
        });
    });
}

/// Scheduler ticks so far.
static TICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Ticks between two indexing-buffer checks (about a second, ES
/// `indices.memory.interval`).
const BUFFER_CHECK_TICKS: u64 = 50;

/// A write saw the node's memtables over the indexing buffer: the next
/// tick checks the buffer instead of waiting for the interval.
static BUFFER_CHECK_DUE: AtomicBool = AtomicBool::new(false);

/// Ask the scheduler to check the indexing buffer at its next tick.
pub(crate) fn request_indexing_buffer_check() {
    BUFFER_CHECK_DUE.store(true, Ordering::Release);
}

/// ES `IndexingMemoryController`: given every copy's (unsealed,
/// unclaimed) indexing-buffer bytes, the copies whose buffers to write
/// out: while the node's bytes not already being written (unclaimed) are
/// over `buffer`, the largest unclaimed buffer goes (its bytes leave the
/// total), then the check repeats. Copies under their share are never
/// written out for being over some per-copy size. Also whether the node's
/// whole buffer (unclaimed plus being written) is past 1.5x the buffer: ES
/// then throttles the copies it writes out, until it is not.
pub(crate) fn buffers_to_write(sizes: &[(u64, u64)], buffer: u64) -> (Vec<usize>, bool) {
    let throttle = sizes.iter().map(|s| s.0).sum::<u64>() > buffer.saturating_add(buffer / 2);
    let mut total: u64 = sizes.iter().map(|s| s.1).sum();
    if total <= buffer {
        return (Vec::new(), throttle);
    }
    let mut order: Vec<usize> = (0..sizes.len()).filter(|&i| sizes[i].1 > 0).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sizes[i].1));
    let mut picked = Vec::new();
    for i in order {
        if total <= buffer {
            break;
        }
        picked.push(i);
        total = total.saturating_sub(sizes[i].1);
    }
    (picked, throttle)
}

/// A copy's throttle after a buffer check: on once written out past 1.5x,
/// kept while the node stays past 1.5x, off below.
pub(crate) fn next_throttle(throttled: bool, picked: bool, over: bool) -> bool {
    over && (throttled || picked)
}

/// The node-wide indexing-buffer check over the live copies (never waits
/// on a copy's core: a busy one counts as holding nothing this time).
pub(crate) fn check_indexing_buffer(copies: &[Arc<IndexShard>], buffer: u64) {
    let sizes: Vec<(u64, u64)> = copies.iter().map(|c| c.shard.indexing_buffer_bytes().unwrap_or((0, 0))).collect();
    let (picked, throttle) = buffers_to_write(&sizes, buffer);
    for (i, copy) in copies.iter().enumerate() {
        // ES `activateThrottling` on the copies written out while the node
        // is past 1.5x its buffer (one indexing thread each); a throttled
        // copy stays throttled (its buffer just claimed, it shows little
        // unclaimed) until the node's whole buffer is under 1.5x.
        let now = copy.shard.bg.memory_throttled.load(Ordering::Acquire);
        copy.shard.set_memory_throttled(next_throttle(now, picked.contains(&i), throttle));
    }
    for i in picked {
        copies[i].shard.write_indexing_buffer();
        dispatch_background(&copies[i]);
    }
}

fn tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
    // Copies are upgraded one at a time: a tick waiting on one copy's
    // lock must not keep every other (possibly dropped) copy alive.
    let shards: Vec<Weak<IndexShard>> = {
        let mut r = registry().lock();
        r.retain(|w| w.strong_count() > 0);
        r.clone()
    };
    let now = now_ms();
    if TICKS.load(Ordering::Relaxed) % BUFFER_CHECK_TICKS == 0 {
        // The merge throttle's backlog check (Lucene updates its I/O rate
        // as merges fall behind).
        rustic_core::merge::check_merge_backlog();
    }
    if TICKS.load(Ordering::Relaxed) % BUFFER_CHECK_TICKS == 0 || BUFFER_CHECK_DUE.swap(false, Ordering::AcqRel) {
        let live: Vec<Arc<IndexShard>> = shards.iter().filter_map(Weak::upgrade).filter(|s| !s.is_closed()).collect();
        check_indexing_buffer(&live, crate::memory::indexing_buffer());
    }
    for w in shards {
        let Some(s) = w.upgrade() else { continue };
        if s.is_closed() {
            continue;
        }
        // Dynamic merge settings, about once a second.
        if TICKS.load(Ordering::Relaxed) % 50 == 0 {
            s.sync_merge_config();
        }
        dispatch_background(&s);
        // `refresh=wait_for` listeners waiting past their limit get a
        // forced refresh (on a worker: this thread never waits on a copy).
        if s.shard.refresh_listeners_expired() {
            s.shard.bg.want_refresh.store(true, Ordering::Release);
            dispatch_background(&s);
            continue;
        }
        let interval = s.refresh_interval_ms();
        scheduled_refresh(&s, now, interval);
    }
}

/// The scheduled refresh of one copy (ES `IndexShard.scheduledRefresh`):
/// due once `interval` (`-1`: never) has passed since its last refresh,
/// when the reader searches use is stale: writes were noted since the last
/// refresh, or a publish would show something without any new write
/// (memtables or segments sealed in the background after the last refresh,
/// pending deletes): ES refreshes whenever the reader is not current
/// (`maybeRefresh`), not only after writes. A search-idle copy skips it
/// (the next search refreshes first), unless `refresh=wait_for` listeners
/// wait, as in ES. The copy's core is never waited on here.
fn scheduled_refresh(s: &Arc<IndexShard>, now: u64, interval: i64) {
    let st = &s.shard.sched;
    // Search-idle (ES `IndexShard.isSearchIdle`): no explicit refresh
    // interval and no search for `index.search.idle.after`.
    let idle = s.setting("index.refresh_interval").is_none() && {
        let idle_after = s.setting("index.search.idle.after").and_then(|v| crate::date::parse_time_value(&v)).unwrap_or(DEFAULT_IDLE_AFTER_MS);
        now.saturating_sub(st.last_search.load(Ordering::Relaxed)) >= idle_after.max(0) as u64
    };
    st.idle.store(idle, Ordering::SeqCst);
    if interval < 0 || now.saturating_sub(st.last_refresh.load(Ordering::Relaxed)) < interval as u64 {
        return;
    }
    if !st.is_dirty() && !s.shard.reader_stale() {
        return;
    }
    // A search-idle copy still refreshes for waiting `refresh=wait_for`
    // listeners (as ES does).
    if idle && !s.shard.listeners.pending() {
        st.skipped.store(true, Ordering::Release);
        return;
    }
    // Never wait on one copy's core here (a flush or merge install can
    // hold it): try, and hand a busy copy to a background worker, which
    // refreshes it as soon as the core is free. Other copies keep
    // refreshing on this thread meanwhile.
    let t0 = Instant::now();
    match s.shard.try_publish() {
        Some(Ok(true)) => {
            s.shard.counters.refresh_total.fetch_add(1, Ordering::Relaxed);
            crate::shard::Counters::add_time(&s.shard.counters.refresh_time_nanos, t0.elapsed());
        }
        Some(Ok(false)) => {}
        Some(Err(e)) => tracing::warn!("scheduled refresh of [{}][{}] failed: {e}", s.name, s.id),
        None => {
            s.shard.bg.want_refresh.store(true, Ordering::Release);
            dispatch_background(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ES `IndexingMemoryController`: only while the node is over its
    /// buffer, the largest buffers are written out until it is not; a copy
    /// is never written out for its own size alone.
    #[test]
    fn the_indexing_buffer_writes_out_the_largest_copies_until_under_budget() {
        // (unsealed, unclaimed) per copy.
        assert!(buffers_to_write(&[(60, 60), (30, 30), (5, 5)], 100).0.is_empty(), "under the buffer: nothing");
        assert_eq!(buffers_to_write(&[(40, 40), (70, 70), (5, 5)], 100), (vec![1], false));
        assert_eq!(buffers_to_write(&[(80, 80), (70, 70), (60, 60)], 100), (vec![0, 1], true), "past 1.5x: throttled");
        // Bytes already being written (claimed by seals in flight) do not
        // count for the write-out (ES: used minus writing): no needless
        // small write-out; they do count for the throttle.
        assert_eq!(buffers_to_write(&[(90, 0), (30, 30)], 100), (vec![], false));
        assert_eq!(buffers_to_write(&[(200, 0)], 100), (vec![], true), "nothing left to claim, still throttled");
        // The throttle holds across checks while the node stays past 1.5x
        // (the throttled copy's buffer is being written: nothing unclaimed).
        let (picked, over) = buffers_to_write(&[(160, 160), (10, 10)], 100);
        assert_eq!((picked.clone(), over), (vec![0], true));
        let first = next_throttle(false, picked.contains(&0), over);
        assert!(first);
        let (picked, over) = buffers_to_write(&[(160, 0), (10, 10)], 100);
        assert!(next_throttle(first, picked.contains(&0), over), "unthrottled while the node is past 1.5x");
        let (picked, over) = buffers_to_write(&[(20, 0), (10, 10)], 100);
        assert!(!next_throttle(true, picked.contains(&0), over), "still throttled under 1.5x");
    }

    /// ES IMC throttling: past 1.5x the node's buffer the copies written
    /// out index one request at a time (their writes take the indexing
    /// throttle), lifted once the node is under its buffer.
    #[test]
    fn copies_written_out_past_one_and_a_half_buffers_are_throttled() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("throttled", &serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
        for i in 0..50 {
            svc.write(vec![crate::index::WriteReq::Index { id: Some(i.to_string()), source: serde_json::json!({"t": "some text to buffer"}).to_string(), routing: None, cond: Default::default() }], false).remove(0).unwrap();
        }
        let copy = svc.shard(0).unwrap();
        // Far past 1.5x a one-byte buffer: written out and throttled.
        check_indexing_buffer(std::slice::from_ref(&copy), 1);
        assert!(copy.shard.bg.memory_throttled.load(Ordering::Acquire), "not throttled past 1.5x the buffer");
        assert!(copy.shard.merge_throttle().is_some(), "a write to the copy does not take the indexing throttle");
        check_indexing_buffer(std::slice::from_ref(&copy), u64::MAX / 4);
        assert!(!copy.shard.bg.memory_throttled.load(Ordering::Acquire), "the throttle was not lifted under the buffer");
    }

    /// Four copies under sustained indexing with a 1 s refresh: once
    /// indexing stops, the scheduler's merges bring every copy below 50
    /// segments within a bounded time (each copy merges with its own slots).
    #[test]
    fn copies_converge_to_tiered_segment_counts_after_indexing_stops() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("converge", &serde_json::json!({"settings": {"number_of_shards": 4, "number_of_replicas": 0, "refresh_interval": "1s"}})).unwrap();
        let text = "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(8);
        let t0 = Instant::now();
        let mut n = 0u64;
        while t0.elapsed() < Duration::from_secs(4) {
            let reqs = (0..200).map(|_| { n += 1; crate::index::WriteReq::Index { id: Some(n.to_string()), source: serde_json::json!({"t": format!("{text} {n}")}).to_string(), routing: None, cond: Default::default() } }).collect();
            for r in svc.write(reqs, false) { r.unwrap(); }
            // Many small segments, as refreshes with concurrent writers make.
            for copy in svc.local_shards() { copy.shard.refresh().unwrap(); copy.shard.seal_all().unwrap(); }
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let counts: Vec<usize> = svc.local_shards().iter().map(|c| c.shard.core.read().num_segments()).collect();
            if counts.iter().all(|c| *c < 50) { break; }
            assert!(Instant::now() < deadline, "segments per copy after 60 s: {counts:?}");
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// A large publish on a served search-idle copy: the refresh the first
    /// search triggers writes the backlog out itself (as a Lucene refresh
    /// writes its own segments), and every search arriving during that
    /// refresh waits for it and then reads segments, not the memtables.
    #[test]
    fn concurrent_first_searches_on_an_idle_copy_wait_for_one_sealed_refresh() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("idle-backlog", &serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}})).unwrap();
        for i in 0..300 {
            svc.write(vec![crate::index::WriteReq::Index { id: Some(i.to_string()), source: serde_json::json!({"t": "backlog text"}).to_string(), routing: None, cond: Default::default() }], false).remove(0).unwrap();
        }
        let copy = svc.shard(0).unwrap();
        copy.shard.bg.large_publish_override.store(1, Ordering::Release);
        copy.shard.sched.skipped.store(true, Ordering::Release);
        let searchers: Vec<_> = (0..4).map(|_| {
            let copy = copy.clone();
            std::thread::spawn(move || {
                copy.shard.before_search();
                copy.shard.reader_view().num_live_docs()
            })
        }).collect();
        let seen: Vec<u64> = searchers.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(seen.iter().all(|n| *n == 300), "a first search read the old view: {seen:?}");
        let core = copy.shard.core.read();
        assert_eq!(core.num_published_memtables(), 0, "the backlog was published as memtables");
        assert!(core.num_segments() >= 1);
    }

    /// M9: a refresh / seal pass never runs merges (they go to the merge
    /// workers), so a long merge can never hold back refreshes and seals.
    #[test]
    fn a_seal_pass_leaves_merges_to_the_merge_workers() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("passes", &serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
        let copy = svc.shard(0).unwrap();
        // Marked queued on the merge side, so the scheduler's own dispatch
        // leaves its merge flag alone during the test.
        copy.shard.bg.merge_queued.store(true, Ordering::Release);
        copy.shard.bg.want_merge.store(true, Ordering::Release);
        copy.shard.bg.want_refresh.store(true, Ordering::Release);
        seal_pass(&Arc::downgrade(&copy));
        assert!(!copy.shard.bg.want_refresh.load(Ordering::Acquire), "the refresh ran");
        assert!(copy.shard.bg.want_merge.load(Ordering::Acquire), "merges are not run on a refresh / seal worker");
    }

    fn write_docs(svc: &crate::index::IndexService, ids: std::ops::Range<u32>) {
        for i in ids {
            svc.write(vec![crate::index::WriteReq::Index { id: Some(i.to_string()), source: serde_json::json!({"t": "stale reader text"}).to_string(), routing: None, cond: Default::default() }], false).remove(0).unwrap();
        }
    }

    /// Data sealed after the copy's last refresh with no write noted since
    /// (the copy is clean, its reader stale).
    fn sealed_but_unpublished(copy: &Arc<IndexShard>) {
        copy.shard.seal_all().unwrap();
        let st = &copy.shard.sched;
        let generation = st.note_refresh();
        st.refreshed(generation);
        st.skipped.store(false, Ordering::Release);
        assert!(!st.is_dirty());
        // (`reader_stale` reads `false` while a merge install holds the core.)
        let stale = (0..200).any(|_| copy.shard.reader_stale() || { std::thread::sleep(Duration::from_millis(5)); false });
        assert!(stale, "sealed data waits for a publish");
    }

    /// ES `scheduledRefresh` refreshes whenever the reader is stale, not
    /// only after writes: data sealed after the last refresh, with no write
    /// noted since, is published by the next scheduled refresh.
    #[test]
    fn a_scheduled_refresh_publishes_a_stale_reader_without_a_new_write() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        // No scheduled refresh of its own (the test runs it).
        let svc = node.create_index("stale-active", &serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}})).unwrap();
        let copy = svc.shard(0).unwrap();
        write_docs(&svc, 0..20);
        copy.shard.refresh().unwrap();
        write_docs(&svc, 20..40);
        sealed_but_unpublished(&copy);
        assert_eq!(copy.shard.reader_view().num_live_docs(), 20);
        scheduled_refresh(&copy, now_ms() + 1, 0);
        // Published here, or (core busy) by a background worker.
        let deadline = Instant::now() + Duration::from_secs(10);
        while copy.shard.reader_view().num_live_docs() < 40 {
            assert!(Instant::now() < deadline, "the stale reader was never refreshed");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!copy.shard.reader_stale());
    }

    /// A search-idle copy with a stale reader skips the scheduled refresh,
    /// as ES does, and its next search refreshes it first.
    #[test]
    fn a_search_idle_copy_with_a_stale_reader_refreshes_on_its_next_search() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let svc = node.create_index("stale-idle", &serde_json::json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.search.idle.after": "0ms"}})).unwrap();
        let copy = svc.shard(0).unwrap();
        write_docs(&svc, 0..20);
        copy.shard.refresh().unwrap();
        write_docs(&svc, 20..40);
        sealed_but_unpublished(&copy);
        scheduled_refresh(&copy, now_ms() + 1, 0);
        assert!(copy.shard.sched.idle.load(Ordering::SeqCst));
        assert_eq!(copy.shard.reader_view().num_live_docs(), 20, "a search-idle copy was refreshed by the scheduler");
        copy.shard.before_search();
        assert_eq!(copy.shard.reader_view().num_live_docs(), 40, "the search did not refresh the stale copy first");
    }

    /// R1: a refresh clears the dirty flag only for the writes it covers: a
    /// write noted after it started (mid-refresh) keeps the copy dirty, so
    /// the next scheduled refresh publishes it even on an idle index.
    #[test]
    fn a_write_landing_mid_refresh_keeps_the_copy_dirty() {
        let st = RefreshState::default();
        assert!(!st.is_dirty());
        st.note_write();
        assert!(st.is_dirty());
        let generation = st.note_refresh();
        st.note_write(); // completes while the refresh runs
        st.refreshed(generation);
        assert!(st.is_dirty(), "the mid-refresh write lost its dirty flag");
        let generation = st.note_refresh();
        st.refreshed(generation);
        assert!(!st.is_dirty());
        // A refresh that did not publish leaves it dirty.
        st.note_write();
        let _unpublished = st.note_refresh();
        assert!(st.is_dirty());
    }
}
