//! # rustic-core
//!
//! Core data structures for the Rustic search engine.
//!
//! This crate provides the fundamental building blocks:
//! - Term dictionary: front-coded SSTable, binary search on 8-byte prefixes
//! - Postings lists with efficient skip data
//! - Fast fields (doc values) with columnar storage
//! - Inverted index with segment-based storage
//!
//! ## Design goals vs Tantivy
//!
//! 1. **Term dictionary**: block-indexed SSTable, O(1) empty dictionary,
//!    ~100ns lookups (Tantivy: ~5ms cold on the empty path, issue #766)
//! 2. **Range queries**: a 1-D points index (doc ids sorted by value) plus
//!    256-value block min/max over columnar fast fields (Tantivy range queries are ~5x slower than text, issue #2266)
//! 3. **Memory**: one mmap-backed file per segment (plus `.del` and
//!    vector `.rvec` sidecars), zero-copy readers
//! 4. **Postings**: 128-doc bit-packed blocks with one skip entry per block
//!    (single level, with block-max impacts; Lucene 10.2 got 33% from
//!    moving to 2-level skip data)
//! 5. **Durability**: atomic segment files + `meta.json` commit points

pub mod accounting;
pub mod admission;
pub mod analysis;
pub mod automaton;
pub mod cancel;
pub mod codec;
pub mod doc;
pub mod error;
pub mod fast_fields;
pub mod fielddata;
pub mod fieldtypes;
pub mod fdlibm;
pub mod geo;
pub mod index;
pub mod index_sort;
pub mod merge;
pub mod nested;
pub mod platform;
pub mod postings;
pub mod query;
pub mod reader_view;
pub mod segment;
pub mod sloppy_math;
pub mod synthetic_id;
pub mod term_dictionary;
pub mod transfer;
pub mod types;
pub mod uid;
pub mod vector;

/// The pool segment flushes and merges encode on. Callers hold an index's
/// write lock while they seal, and the global rayon pool also runs work
/// that blocks on such locks (an ES layer's parallel bulk writes), so
/// encoding there can starve and deadlock: it gets its own workers.
///
/// Its size follows search load (ES gets this implicitly: indexing threads
/// do their own flushing): every core while the node has not searched
/// recently ([`searches_active`]), half of them while it has, so searches
/// keep the other half. Work already on one of the two pools stays there
/// (nested encode work never hops pools).
pub fn encode_pool() -> &'static rayon::ThreadPool {
    let (full, shared) = (encode_pool_full(), encode_pool_shared());
    if full.current_thread_index().is_some() {
        return full;
    }
    if shared.current_thread_index().is_some() {
        return shared;
    }
    if let Some(foreground) = FOREGROUND_POOL.get().filter(|p| p.current_thread_index().is_some()) {
        return foreground;
    }
    if FOREGROUND.with(|f| f.get()) {
        return encode_pool_foreground();
    }
    if searches_active() { shared } else { full }
}

thread_local! {
    /// Set by [`foreground_encode`] on the calling thread.
    static FOREGROUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static FOREGROUND_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// The encode pool of foreground seals: the encode work a flush or a
/// refresh waits for itself (a Lucene flush or refresh writes its segments
/// on the indexing or refreshing thread, at that thread's priority). Half
/// the cores, at normal priority: the niced background pools yield to
/// every indexing and search thread, so a flush beside them waited many
/// seconds for its own seal (a 47 MB memtable took 5.9 s on host3).
fn encode_pool_foreground() -> &'static rayon::ThreadPool {
    FOREGROUND_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads((encode_threads() / 2).max(2))
            .thread_name(|i| format!("rustic-encode-f-{i}"))
            .stack_size(16 << 20)
            .build()
            .expect("foreground encode pool")
    })
}

/// Whether the calling thread runs under [`foreground_encode`] (to carry
/// it into threads it spawns).
pub(crate) fn foreground_encode_active() -> bool {
    FOREGROUND.with(|f| f.get())
}

/// Run `f` with its encode work ([`encode_pool`]) on the foreground encode
/// pool, at normal priority: for seals a caller waits on (flush, refresh).
pub fn foreground_encode<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) { FOREGROUND.with(|c| c.set(self.0)); }
    }
    let _restore = Restore(FOREGROUND.with(|c| c.replace(true)));
    f()
}

/// The encode pool sized for an idle search side: `RUSTIC_ENCODE_THREADS`,
/// else every core.
fn encode_pool_full() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| build_encode_pool("rustic-encode", encode_threads()))
}

/// The encode pool used while searches run: half of [`encode_pool_full`]
/// (ES runs flush and merge work on at most half the processors).
fn encode_pool_shared() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| build_encode_pool("rustic-encode-s", (encode_threads() / 2).max(2)))
}

/// Workers of the full encode pool.
pub fn encode_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::env::var("RUSTIC_ENCODE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(cores)
    })
}

/// Whether the current thread is a worker of either encode pool.
pub fn on_encode_worker() -> bool {
    encode_pool_full().current_thread_index().is_some() || encode_pool_shared().current_thread_index().is_some()
        || FOREGROUND_POOL.get().is_some_and(|p| p.current_thread_index().is_some())
}

fn build_encode_pool(name: &'static str, n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(move |i| format!("{name}-{i}"))
        // Headroom: a worker waiting inside one encode job runs other
        // queued jobs nested on its stack.
        .stack_size(16 << 20)
        // Seals: below searches, above merges.
        .start_handler(|_| lower_thread_priority(5))
        .build()
        .expect("encode pool")
}

/// Milliseconds since the first call (a monotonic clock for activity marks).
pub(crate) fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

static LAST_SEARCH_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How long after the last search the node still counts as searching.
const SEARCH_ACTIVE_MS: u64 = 1_000;

/// A search started on this node (cheap: one relaxed store).
pub fn note_search() {
    LAST_SEARCH_MS.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
}

/// The node searched within the last second ([`note_search`]).
pub fn searches_active() -> bool {
    let last = LAST_SEARCH_MS.load(std::sync::atomic::Ordering::Relaxed);
    last != 0 && now_ms().saturating_sub(last) < SEARCH_ACTIVE_MS
}

/// The search pool's queued and running request counters (the node's
/// search executor registers them once; see [`search_load`]).
static SEARCH_LOAD: std::sync::OnceLock<(std::sync::Arc<std::sync::atomic::AtomicUsize>, std::sync::Arc<std::sync::atomic::AtomicUsize>)> = std::sync::OnceLock::new();

/// Register the search pool's `(queued, active)` counters: running merges
/// throttle their writes while searches queue (`merge::throttle_merge_write`).
/// The first registration of the process wins.
pub fn register_search_load(queued: std::sync::Arc<std::sync::atomic::AtomicUsize>, active: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let _ = SEARCH_LOAD.set((queued, active));
}

/// Searches queued and running on the node's search pool now (two relaxed
/// loads; `(0, 0)` before a pool registered).
pub(crate) fn search_load() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    SEARCH_LOAD.get().map_or((0, 0), |(q, a)| (q.load(Relaxed), a.load(Relaxed)))
}

/// Linux `SCHED_BATCH` for the calling thread (best effort): the scheduler
/// treats it as CPU-bound and does not let its wakeups preempt
/// interactive work (searches), on top of its `nice`.
pub fn batch_thread_scheduling() {
    #[cfg(target_os = "linux")]
    {
        #[repr(C)]
        struct SchedParam {
            sched_priority: i32,
        }
        unsafe extern "C" {
            fn sched_setscheduler(pid: i32, policy: i32, param: *const SchedParam) -> i32;
        }
        const SCHED_BATCH: i32 = 3;
        if std::env::var_os("RUSTIC_MERGE_NO_BATCH").is_none() {
            // SAFETY: a plain syscall wrapper on the calling thread (pid 0); a
            // failure leaves the policy as it was.
            unsafe { let _ = sched_setscheduler(0, SCHED_BATCH, &SchedParam { sched_priority: 0 }); }
        }
    }
}

/// Run `f` on [`encode_pool`] and block until it is done, **without**
/// letting the caller steal other work meanwhile. Called from a worker of
/// another rayon pool, `ThreadPool::install` keeps that worker stealing
/// jobs of its own pool while it waits; a stolen job that needs a lock the
/// caller holds (e.g. another bulk write to the same shard) then deadlocks
/// it. Such callers wait on a plain scoped thread instead.
pub fn on_encode_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    on_pool(encode_pool(), f)
}

/// The pool background merges encode on (ES / Lucene
/// `ConcurrentMergeScheduler`: dedicated merge threads, throttled by sleeping
/// on themselves): a throttled merge never occupies an [`encode_pool`]
/// thread that seals (which indexing and refresh wait on) need. As many
/// threads as ES's `index.merge.scheduler.max_thread_count` default
/// (`max(1, min(4, cores / 2))`).
/// Lower the calling thread's CPU priority (Linux `nice`, per thread): the
/// background encode and merge threads yield the cores to searches and
/// indexing when the node is busy, and use them fully when it is not.
/// `RUSTIC_BACKGROUND_NICE` overrides the increment (`0`: unchanged).
pub fn lower_thread_priority(default_nice: i32) {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
        }
        let nice = std::env::var("RUSTIC_BACKGROUND_NICE").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(default_nice);
        if nice != 0 {
            // PRIO_PROCESS with who = 0 on Linux addresses the calling thread.
            // SAFETY: a plain syscall wrapper; a failure leaves the priority as is.
            unsafe { let _ = setpriority(0, 0, nice.clamp(-20, 19)); }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = default_nice;
}

pub fn merge_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        // Every copy runs its own merges (ES: a merge scheduler per shard,
        // `max_thread_count` each), the node at most half its cores at
        // once (the engine's node-wide merge permits), niced.
        let n = std::env::var("RUSTIC_MERGE_POOL_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0)
            .unwrap_or_else(|| (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rustic-merge-{i}"))
            .stack_size(16 << 20)
            // Merges are the most deferrable work (ES throttles them too):
            // niced and batch-scheduled below searches; their writes also
            // slow down while searches queue (`merge::throttle_merge_write`).
            .start_handler(|_| {
                lower_thread_priority(10);
                batch_thread_scheduling();
            })
            .build()
            .expect("merge pool")
    })
}

/// The pool force merges encode on, apart from [`encode_pool`] so that seals
/// (which indexing waits on) and background merges keep their workers while
/// a large force merge runs (ES runs force merges on its own `force_merge`
/// pool, `max(1, cores / 8)` threads): `RUSTIC_FORCE_MERGE_THREADS`, else
/// that size, so a force merge takes one thread per shard at a time.
pub fn force_merge_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let n = std::env::var("RUSTIC_FORCE_MERGE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or((cores / 8).max(1));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rustic-force-merge-{i}"))
            .start_handler(|_| lower_thread_priority(10))
            .build()
            .expect("force merge pool")
    })
}

/// [`on_encode_pool`] for any pool.
pub fn on_pool<R: Send>(pool: &rayon::ThreadPool, f: impl FnOnce() -> R + Send) -> R {
    let admission = admission::Context::capture();
    let f = move || admission.run(f);
    // A test's per-thread write version follows the segment work it hands
    // to a pool (flushes and merges run there).
    #[cfg(test)]
    let f = {
        let forced = codec::TEST_WRITE_VERSION.with(|v| v.get());
        move || {
            let prev = codec::TEST_WRITE_VERSION.with(|v| v.replace(forced));
            let out = f();
            codec::TEST_WRITE_VERSION.with(|v| v.set(prev));
            out
        }
    };
    if pool.current_thread_index().is_some() || rayon::current_thread_index().is_none() {
        return pool.install(f);
    }
    std::thread::scope(|s| s.spawn(|| pool.install(f)).join().expect("encode thread panicked"))
}

pub use error::RusticError;
pub use index::Index;
pub use reader_view::{ReaderView, ViewSlot};
pub use types::*;

/// Tests that intentionally occupy every global Rayon worker must not run
/// against each other: neither can reach its release step in that setup.
#[cfg(test)]
pub(crate) fn global_rayon_test_gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GATE.lock().unwrap_or_else(|poison| poison.into_inner())
}

#[cfg(test)]
mod encode_pool_tests {
    #[test]
    fn nested_pool_handoffs_keep_lease_and_owner_and_restore_idle_worker() {
        use crate::admission::{Budget, Lease, Owner};
        let budget = Budget::new(128);
        let lease = Lease::reserve(budget.clone(), 96, "test plan").unwrap();
        let owner = Owner::new();
        let first = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let second = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let allocation = {
            let _owner = owner.enter();
            let _lease = lease.enter();
            super::on_pool(&first, || super::on_pool(&second, || budget.reserve(64, "nested worker").unwrap()))
        };
        assert_eq!(owner.bytes().current, 64);
        assert_eq!(budget.used(), 96, "child uses reserved credits once");
        let ordinary = second.install(|| budget.reserve(32, "idle worker").unwrap());
        assert_eq!(owner.bytes().current, 64, "worker scope must restore unrelated caller");
        assert_eq!(budget.used(), 128);
        drop(ordinary);
        lease.complete().unwrap();
        assert_eq!(budget.used(), 64);
        drop(allocation);
        assert_eq!(budget.used(), 0);
        assert_eq!(owner.bytes().current, 0);
    }
    #[test]
    fn encode_pool_follows_search_load_and_nested_work_stays_on_its_pool() {
        let full = super::encode_threads();
        super::note_search();
        assert!(super::searches_active());
        let pool = super::encode_pool();
        assert_eq!(pool.current_num_threads(), (full / 2).max(2), "searching: the half-size pool");
        // Work on a pool keeps that pool for its nested encode work.
        let nested = pool.install(|| super::encode_pool().current_num_threads());
        assert_eq!(nested, pool.current_num_threads());
        assert!(pool.install(super::on_encode_worker));
        assert!(!super::on_encode_worker());
    }
}

#[cfg(test)]
mod foreground_encode_tests {
    fn worker_name() -> String {
        std::thread::current().name().unwrap_or_default().to_string()
    }

    /// A flush's or refresh's own seal encodes on the normal-priority
    /// foreground pool; other encode work stays on the niced pools, and
    /// the flag ends with the call.
    #[test]
    fn foreground_encode_runs_on_the_foreground_pool() {
        let name = super::foreground_encode(|| super::on_encode_pool(worker_name));
        assert!(name.starts_with("rustic-encode-f-"), "{name}");
        // Nested encode work stays on the pool it runs on.
        let nested = super::foreground_encode(|| super::on_encode_pool(|| super::on_encode_pool(worker_name)));
        assert!(nested.starts_with("rustic-encode-f-"), "{nested}");
        let background = super::on_encode_pool(worker_name);
        assert!(background.starts_with("rustic-encode"), "{background}");
        assert!(!background.starts_with("rustic-encode-f-"), "{background}");
        assert!(!super::foreground_encode(super::on_encode_worker), "the caller is not an encode worker");
    }
}
