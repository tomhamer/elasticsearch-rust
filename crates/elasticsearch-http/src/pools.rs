//! Bounded executors for engine work (ES thread pools): search, write,
//! flush (refresh / flush / force merge), replica (replication), recovery
//! (peer recovery source / target work), management (everything else,
//! ES's `generic`) and search_coordination (a retriever's plan phase,
//! which waits on child searches; see [`Pool::SearchCoordination`]).
//!
//! Every blocking engine call made for a REST request ([`crate::blocking`])
//! runs on the executor its route selects ([`pool_for`]). A user-facing
//! executor with a full queue rejects with 429
//! `es_rejected_execution_exception`, as ES does; the replica, flush and
//! management queues are unbounded (ES never rejects replication / recovery
//! work, and generic tasks scale), and replica work has its own threads, so
//! it never queues behind user searches or writes.
//!
//! Sizes follow ES's defaults (`allocated_processors` = available CPUs):
//! search `cpus * 3 / 2 + 1` threads / queue 1000, write `cpus` (at least 4)
//! / queue 10000, flush `max(1, min(5, cpus / 2))`, replica `max(4, cpus)`,
//! recovery `max(2, cpus / 2)`, management `max(128, 4 * cpus)`. Normal loads never reach a queue limit.

use elasticsearch_engine::error::EsError;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The executor a piece of engine work runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Search,
    Write,
    Flush,
    Replica,
    Recovery,
    Management,
    Refresh,
    ForceMerge,
    /// ES `search_coordination`: work that coordinates searches and waits
    /// for them, a retriever's plan phase (its child searches). Its child
    /// searches' shard phases run on `search` as any search's do; no
    /// `search` thread ever waits on this pool, so the wait cannot
    /// deadlock (it did while the plan phase held a `search` thread:
    /// every `search` thread waited on shard phases queued behind them).
    /// Its workers are not executor work for [`on_executor`]: what they
    /// wait on is queued normally, with its backpressure. A worker waiting
    /// on a child ([`blocking`]) lets a spare stand in.
    SearchCoordination,
}

impl Pool {
    pub const ALL: [Pool; 9] = [Pool::Search, Pool::Write, Pool::Flush, Pool::Replica, Pool::Recovery, Pool::Management, Pool::Refresh, Pool::ForceMerge, Pool::SearchCoordination];

    /// The ES thread-pool name.
    pub fn name(self) -> &'static str {
        match self {
            Pool::Search => "search",
            Pool::Write => "write",
            Pool::Flush => "flush",
            Pool::Replica => "replica",
            Pool::Recovery => "recovery",
            Pool::Management => "management",
            Pool::Refresh => "refresh",
            Pool::ForceMerge => "force_merge",
            Pool::SearchCoordination => "search_coordination",
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
    /// Set on executor worker threads.
    static ON_EXECUTOR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The executor whose worker this thread is.
    static CURRENT: std::cell::Cell<Option<Pool>> = const { std::cell::Cell::new(None) };
}

/// An executor's job queue: many consumers, and an idle worker holds no
/// lock while it waits (a spare can always take the next job).
struct JobQueue {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    ready: std::sync::Condvar,
    capacity: Option<usize>,
    /// Bumped (under `jobs`) by [`JobQueue::wake_all`]: every idle worker
    /// that has not seen this generation yet collects once.
    collect: AtomicU64,
}

/// What an idle worker's wait on its queue ended with.
enum Next {
    Job(Job),
    /// [`wake_idle`]: run `memory::idle_collect`, then wait again.
    Collect,
    /// The timeout passed with no job (spares only).
    Idle,
}

impl JobQueue {
    /// `Err(())`: full.
    fn push(&self, job: Job) -> Result<(), ()> {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        if self.capacity.is_some_and(|c| jobs.len() >= c) {
            return Err(());
        }
        jobs.push_back(job);
        drop(jobs);
        self.ready.notify_one();
        Ok(())
    }

    /// The next job, or a collect request this worker (last at
    /// generation `seen`) has not run yet; with `timeout`, [`Next::Idle`]
    /// when neither came in time.
    fn pop(&self, timeout: Option<std::time::Duration>, seen: &mut u64) -> Next {
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(job) = jobs.pop_front() {
                return Next::Job(job);
            }
            // Read under `jobs`, as `wake_all` bumps it: a worker either
            // sees the new generation here or is waiting when it notifies.
            let generation = self.collect.load(Ordering::Acquire);
            if generation != *seen {
                *seen = generation;
                return Next::Collect;
            }
            match timeout {
                None => jobs = self.ready.wait(jobs).unwrap_or_else(|e| e.into_inner()),
                Some(t) => {
                    let (next, waited) = self.ready.wait_timeout(jobs, t).unwrap_or_else(|e| e.into_inner());
                    jobs = next;
                    if waited.timed_out() {
                        return jobs.pop_front().map_or(Next::Idle, Next::Job);
                    }
                }
            }
        }
    }

    /// Wake every idle worker of this queue to collect once (no job, so
    /// nothing shows in the queue or completed counts).
    fn wake_all(&self) {
        let jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        self.collect.fetch_add(1, Ordering::Release);
        drop(jobs);
        self.ready.notify_all();
    }

    fn generation(&self) -> u64 {
        self.collect.load(Ordering::Acquire)
    }
}


/// One executor: fixed worker threads draining a (bounded or unbounded)
/// queue.
pub struct Executor {
    pool: Pool,
    threads: usize,
    capacity: Option<usize>,
    queue: Arc<JobQueue>,
    queued: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    completed: Arc<AtomicU64>,
    /// EWMA of one job's execution time in nanoseconds (f64 bits; search
    /// pool only), as ES's `TaskExecutionTimeTrackingEsThreadPoolExecutor`
    /// tracks it for adaptive replica selection.
    service_ns: Arc<AtomicU64>,
    rejected: AtomicU64,
    /// Nested work admitted against this executor's queue and running off
    /// it ([`Executor::admit`]).
    nested: AtomicUsize,
    /// Workers parked in [`blocking`] (each wants a spare to stand in).
    parked: AtomicUsize,
    /// Live spare worker threads (reused; an idle one exits once more
    /// spares live than workers are parked).
    spares: AtomicUsize,
}

/// Point-in-time figures of one executor (`_nodes/stats` `thread_pool`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutorStats {
    pub threads: usize,
    pub queue: usize,
    pub active: usize,
    pub rejected: u64,
    pub completed: u64,
    pub queue_size: Option<usize>,
}

impl Executor {
    fn start(pool: Pool, threads: usize, capacity: Option<usize>) -> Executor {
        let queue = Arc::new(JobQueue { jobs: Mutex::new(std::collections::VecDeque::new()), ready: std::sync::Condvar::new(), capacity, collect: AtomicU64::new(0) });
        let queued = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let service_ns = Arc::new(AtomicU64::new(0));
        if pool == Pool::Search {
            // Running merges throttle their writes while searches queue.
            elasticsearch_engine::rustic_core::register_search_load(queued.clone(), active.clone());
        }
        for i in 0..threads {
            let (queue, queued, active, completed, service_ns) = (queue.clone(), queued.clone(), active.clone(), completed.clone(), service_ns.clone());
            let spawned = std::thread::Builder::new().name(format!("es-{}-{i}", pool.name())).spawn(move || {
                ON_EXECUTOR.with(|on| on.set(pool.waits_nested()));
                CURRENT.with(|c| c.set(Some(pool)));
                let mut seen = queue.generation();
                loop {
                    let job = match queue.pop(None, &mut seen) {
                        Next::Job(job) => job,
                        // Woken while idle (`wake_idle`): collect, wait again.
                        Next::Collect | Next::Idle => {
                            elasticsearch_engine::memory::idle_collect();
                            continue;
                        }
                    };
                    queued.fetch_sub(1, Ordering::Relaxed);
                    active.fetch_add(1, Ordering::Relaxed);
                    let started = (pool == Pool::Search).then(std::time::Instant::now);
                    // A panicking task must not take its worker down with it; its
                    // caller sees the dropped result channel.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    if let Some(started) = started { record_service(&service_ns, started.elapsed()); }
                    active.fetch_sub(1, Ordering::Relaxed);
                    completed.fetch_add(1, Ordering::Relaxed);
                    // Between jobs: this worker's freed pages go back to the OS
                    // while the parent breaker is over its limit.
                    elasticsearch_engine::memory::idle_collect();
                }
            });
            if let Err(e) = spawned {
                tracing::error!("could not start {} pool worker: {e}", pool.name());
            }
        }
        Executor { pool, threads, capacity, queue, queued, active, completed, service_ns, rejected: AtomicU64::new(0), nested: AtomicUsize::new(0), parked: AtomicUsize::new(0), spares: AtomicUsize::new(0) }
    }

    /// A worker parks: make sure a spare stands in (reusing an idle one;
    /// at most four per regular worker).
    fn park(&'static self) {
        let parked = self.parked.fetch_add(1, Ordering::AcqRel) + 1;
        // Parked work is not active work (`stats().active` stays within
        // the pool size: the spare's job counts instead).
        self.active.fetch_sub(1, Ordering::Relaxed);
        loop {
            let live = self.spares.load(Ordering::Acquire);
            if live >= parked || live >= self.threads * 4 {
                return;
            }
            if self.spares.compare_exchange(live, live + 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                break;
            }
        }
        let spawned = std::thread::Builder::new().name(format!("es-{}-spare", self.pool.name())).spawn(move || {
            ON_EXECUTOR.with(|on| on.set(self.pool.waits_nested()));
            CURRENT.with(|c| c.set(Some(self.pool)));
            let mut idle_since = std::time::Instant::now();
            let mut seen = self.queue.generation();
            loop {
                match self.queue.pop(Some(std::time::Duration::from_millis(100)), &mut seen) {
                    Next::Collect => elasticsearch_engine::memory::idle_collect(),
                    Next::Job(job) => {
                        self.queued.fetch_sub(1, Ordering::Relaxed);
                        self.active.fetch_add(1, Ordering::Relaxed);
                        let started = (self.pool == Pool::Search).then(std::time::Instant::now);
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        if let Some(started) = started { record_service(&self.service_ns, started.elapsed()); }
                        self.active.fetch_sub(1, Ordering::Relaxed);
                        self.completed.fetch_add(1, Ordering::Relaxed);
                        idle_since = std::time::Instant::now();
                        // As the regular workers: freed pages back between jobs while
                        // the parent breaker is over its limit.
                        elasticsearch_engine::memory::idle_collect();
                    }
                    Next::Idle => {
                        // Exit only while more spares live than workers are
                        // parked (after a short reuse window): a parked
                        // worker always keeps its stand-in, however long the
                        // pause in traffic.
                        let live = self.spares.load(Ordering::Acquire);
                        if live > self.parked.load(Ordering::Acquire) && idle_since.elapsed() >= std::time::Duration::from_millis(500) {
                            if self.spares.compare_exchange(live, live - 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                                // A worker parked meanwhile (it saw this spare
                                // still live and started none): stay for it.
                                if self.parked.load(Ordering::Acquire) > live - 1 {
                                    self.spares.fetch_add(1, Ordering::AcqRel);
                                    idle_since = std::time::Instant::now();
                                    continue;
                                }
                                return;
                            }
                        }
                    }
                }
            }
        });
        if spawned.is_err() {
            self.spares.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn unpark(&self) {
        self.parked.fetch_sub(1, Ordering::AcqRel);
        self.active.fetch_add(1, Ordering::Relaxed);
    }

    fn submit(&self, job: Job) -> Result<(), EsError> {
        self.queued.fetch_add(1, Ordering::Relaxed);
        match self.queue.push(job) {
            Ok(()) => Ok(()),
            Err(()) => {
                self.queued.fetch_sub(1, Ordering::Relaxed);
                self.rejected.fetch_add(1, Ordering::Relaxed);
                Err(rejected(self.pool, self.capacity.unwrap_or(0)))
            }
        }
    }

    /// Admission of work this executor's queue would bound but that runs
    /// off it (a nested sub-request: queueing it could deadlock its
    /// waiting caller): rejected (429) while the queue plus the nested work
    /// already admitted reach its capacity, as a submit would be, so nested
    /// writes keep the write pool's backpressure, also among themselves.
    /// The admission counts until the returned guard drops.
    pub fn admit(&'static self) -> Result<NestedAdmission, EsError> {
        if let Some(capacity) = self.capacity {
            let nested = self.nested.fetch_add(1, Ordering::AcqRel);
            if self.queued.load(Ordering::Relaxed) + nested >= capacity {
                self.nested.fetch_sub(1, Ordering::AcqRel);
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(rejected(self.pool, capacity));
            }
            return Ok(NestedAdmission(Some(self)));
        }
        Ok(NestedAdmission(None))
    }

    /// The EWMA of one job's execution time in nanoseconds; 0 before the
    /// first job completes (and for pools other than search).
    pub fn service_time_ewma_ns(&self) -> f64 {
        f64::from_bits(self.service_ns.load(Ordering::Relaxed))
    }

    pub fn stats(&self) -> ExecutorStats {
        ExecutorStats {
            threads: self.threads,
            queue: self.queued.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            queue_size: self.capacity,
        }
    }
}

/// Fold one job's execution time into `cell` (ES's EWMA, alpha 0.3; the
/// first sample is taken as is).
fn record_service(cell: &AtomicU64, elapsed: std::time::Duration) {
    let sample = elapsed.as_nanos() as f64;
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        let old = f64::from_bits(current);
        let next = if old == 0.0 { sample } else { old * 0.7 + sample * 0.3 }.to_bits();
        match cell.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

/// Run `f`, which waits for something other than CPU (a cluster commit),
/// without taking a worker from the executor this thread belongs to: a
/// spare worker drains its queue meanwhile (Java's managed blocking). Off
/// an executor, `f` just runs.
pub fn blocking<T>(f: impl FnOnce() -> T) -> T {
    let Some(pool) = CURRENT.with(|c| c.get()) else { return f() };
    struct Parked(&'static Executor);
    impl Drop for Parked {
        fn drop(&mut self) {
            self.0.unpark();
        }
    }
    let executor = executor(pool);
    executor.park();
    let _parked = Parked(executor);
    f()
}

/// Nested work counted against an executor's queue ([`Executor::admit`]).
pub struct NestedAdmission(Option<&'static Executor>);

impl Drop for NestedAdmission {
    fn drop(&mut self) {
        if let Some(executor) = self.0 {
            executor.nested.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// ES's rejection of a task on a full executor queue (HTTP 429).
pub fn rejected(pool: Pool, capacity: usize) -> EsError {
    EsError::new(429, "es_rejected_execution_exception", format!(
        "rejected execution of task on EsThreadPoolExecutor[name = {}, queue capacity = {capacity}]", pool.name()))
}

/// ES allocated processors (`node.processors`, else the processors the
/// process may use: CPU affinity and cgroup quota, as the JVM's
/// `availableProcessors`): every executor is sized from it, as ES sizes its
/// thread pools.
fn cpus() -> usize {
    elasticsearch_engine::processors::get().max(1)
}

/// Wake every executor's idle workers, spares included, so each runs
/// `memory::idle_collect` once: a collect generation every waiting worker
/// checks, and a broadcast (no jobs: nothing in the queue or completed
/// counts). Busy workers collect after their job anyway. Registered as a
/// breaker wake hook.
pub fn wake_idle() {
    for pool in Pool::ALL {
        executor(pool).queue.wake_all();
    }
}

/// The executor of `pool` (started on first use).
pub fn executor(pool: Pool) -> &'static Executor {
    static EXECUTORS: OnceLock<[Executor; 9]> = OnceLock::new();
    let executors = EXECUTORS.get_or_init(|| {
        let n = cpus();
        [
            Executor::start(Pool::Search, n * 3 / 2 + 1, Some(1000)),
            Executor::start(Pool::Write, n.max(4), Some(10_000)),
            Executor::start(Pool::Flush, (n / 2).clamp(1, 5), None),
            Executor::start(Pool::Replica, n.max(4), None),
            Executor::start(Pool::Recovery, (n / 2).max(2), None),
            Executor::start(Pool::Management, (4 * n).max(128), None),
            // ES: `refresh` half the processors (at most 10), `force_merge`
            // `max(1, cores / 8)`: a long force merge never holds the
            // threads `_refresh` and `_flush` need.
            Executor::start(Pool::Refresh, (n / 2).clamp(1, 10), None),
            Executor::start(Pool::ForceMerge, (n / 8).max(1), None),
            // ES: `search_coordination` half the processors, queue 1000.
            // Its workers wait on child searches; parked in such a wait
            // ([`blocking`]) each lets a spare stand in (up to four per
            // worker), so retriever coordination is not limited to the
            // pool's size while the search pool does the work.
            Executor::start(Pool::SearchCoordination, n.div_ceil(2).max(1), Some(1000)),
        ]
    });
    let i = Pool::ALL.iter().position(|p| *p == pool).unwrap_or(5);
    &executors[i]
}

/// Run `f` on `pool`'s executor and wait for it without blocking the async
/// runtime. A full user-facing queue rejects at once (429). Called from an
/// executor thread (a nested synchronous request), `f` runs on the
/// runtime's blocking threads instead.
pub async fn run<F, T>(pool: Pool, f: F) -> Result<T, EsError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Engine work running on an executor can issue a nested request
    // synchronously (the cluster front's `runtime.block_on(dispatch(..))`
    // for sub-searches, lookups, ES|QL fan-out). Queueing that behind the
    // same bounded executor its caller occupies could deadlock it, so nested
    // work runs on the runtime's blocking threads, as all work did before.
    if ON_EXECUTOR.with(|on| on.get()) {
        return run_unpooled(f).await;
    }
    // A coordination nested in coordination (a child search that is itself
    // a retriever search) must not queue behind the worker waiting for it.
    if pool == Pool::SearchCoordination && on_coordination() {
        let admission=elasticsearch_engine::rustic_core::admission::Context::capture();
        return tokio::task::spawn_blocking(move || {
            let _admission=admission.enter();
            let _unmarked = ExecutorMark::clear();
            f()
        }).await.map_err(|e| EsError::internal(format!("task failed: {e}")));
    }
    let admission=elasticsearch_engine::rustic_core::admission::Context::capture();
    let (tx, rx) = tokio::sync::oneshot::channel();
    executor(pool).submit(Box::new(move || {
        let _admission=admission.enter();
        let _ = tx.send(f());
    }))?;
    rx.await.map_err(|_| EsError::internal("task failed: its worker dropped it"))
}

/// Whether the current thread is an executor worker (engine work of a
/// request holds it).
pub fn on_executor() -> bool {
    ON_EXECUTOR.with(|on| on.get())
}

/// Whether the current thread is a `search_coordination` worker (or its
/// spare): one whose synchronous waits on child searches are by design
/// (their work is queued on the executors normally).
pub fn on_coordination() -> bool {
    CURRENT.with(|c| c.get()) == Some(Pool::SearchCoordination)
}

/// Whether a synchronous sub-request issued from this thread (a cluster
/// call the thread waits for) must run as nested work, off the bounded
/// executors and marked so on every node it reaches: any thread but a
/// `search_coordination` worker's (and an async runtime worker's, which
/// [`on_executor`] reports when it runs engine work). An executor thread
/// waiting on work queued behind it on its own (or a peer's) executor
/// deadlocks, also when the fan-out moves the work to other tasks or
/// threads, which do not inherit [`on_executor`].
pub fn sub_requests_nested() -> bool {
    if on_coordination() {
        return false;
    }
    on_executor() || tokio::runtime::Handle::try_current().is_err()
}

/// Run `f` off every executor, on the runtime's blocking threads: the
/// engine work of a sub-request that a thread of some executor (on this
/// node or another) is waiting for. Queueing it behind a bounded executor
/// could deadlock: two nodes' search threads each waiting for sub-requests
/// queued behind the other's.
pub async fn run_unpooled<F, T>(f: F) -> Result<T, EsError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Marked as engine work: its own nested requests (and their forwards to
    // other nodes) run unpooled too.
    let admission=elasticsearch_engine::rustic_core::admission::Context::capture();
    tokio::task::spawn_blocking(move || {
        let _admission=admission.enter();
        let _marked = ExecutorMark::set();
        f()
    }).await.map_err(|e| EsError::internal(format!("task failed: {e}")))
}

/// Marks the current thread as engine work until dropped, restoring the
/// previous mark even when the work panics (the blocking thread is reused
/// by later tasks, which must not inherit the mark).
struct ExecutorMark(bool);

impl ExecutorMark {
    fn set() -> ExecutorMark {
        ExecutorMark(ON_EXECUTOR.with(|on| on.replace(true)))
    }
    fn clear() -> ExecutorMark {
        ExecutorMark(ON_EXECUTOR.with(|on| on.replace(false)))
    }
}

impl Pool {
    /// Whether this executor's workers count as executor work for
    /// [`on_executor`] (their sub-requests run nested): all but
    /// `search_coordination`'s.
    fn waits_nested(self) -> bool {
        self != Pool::SearchCoordination
    }
}

impl Drop for ExecutorMark {
    fn drop(&mut self) {
        let was = self.0;
        ON_EXECUTOR.with(|on| on.set(was));
    }
}

/// The executor a REST request's engine work runs on (ES's per-action
/// thread pools): searches and reads on `search`, document writes on
/// `write`, refresh / flush / force merge on `flush`, the rest on
/// `management`.
pub fn pool_for(method: &axum::http::Method, segs: &[String]) -> Pool {
    const SEARCH: &[&str] = &[
        "_search", "_msearch", "_count", "_pit", "_knn_search", "_mvt", "_terms_enum", "_validate", "_explain", "_field_caps",
        "_query", "_sql", "_eql", "_async_search", "_mget", "_termvectors", "_mtermvectors", "_rank_eval", "_source",
    ];
    // By-query and reindex tasks run on `management` (ES's generic pool): a
    // few throttled ones must not hold every write thread.
    const WRITE: &[&str] = &["_bulk", "_create", "_update"];
    let endpoint = segs.iter().find(|s| s.starts_with('_') && s.as_str() != "_all").map(String::as_str).unwrap_or("");
    if endpoint == "_flush" {
        Pool::Flush
    } else if endpoint == "_refresh" {
        Pool::Refresh
    } else if endpoint == "_forcemerge" {
        Pool::ForceMerge
    } else if WRITE.contains(&endpoint) {
        Pool::Write
    } else if endpoint == "_doc" {
        if *method == axum::http::Method::GET || *method == axum::http::Method::HEAD { Pool::Search } else { Pool::Write }
    } else if SEARCH.contains(&endpoint) {
        Pool::Search
    } else {
        Pool::Management
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Combined review M29: one wake reaches every idle worker of a queue
    /// (not one per notify, nor drained by the first woken), each exactly
    /// once, and a job queued meanwhile still runs.
    #[test]
    fn a_wake_reaches_every_idle_worker_once() {
        let queue = Arc::new(JobQueue { jobs: Mutex::new(std::collections::VecDeque::new()), ready: std::sync::Condvar::new(), capacity: None, collect: AtomicU64::new(0) });
        let (tx, rx) = mpsc::channel::<&'static str>();
        let workers = 6;
        for _ in 0..workers {
            let (queue, tx) = (queue.clone(), tx.clone());
            std::thread::spawn(move || {
                let mut seen = queue.generation();
                loop {
                    match queue.pop(None, &mut seen) {
                        Next::Collect => { let _ = tx.send("collect"); }
                        Next::Job(job) => { job(); let _ = tx.send("job"); }
                        Next::Idle => unreachable!("no timeout"),
                    }
                }
            });
        }
        // Let every worker reach its wait first.
        std::thread::sleep(std::time::Duration::from_millis(100));
        queue.wake_all();
        let wait = std::time::Duration::from_secs(5);
        for _ in 0..workers {
            assert_eq!(rx.recv_timeout(wait).expect("every idle worker collects"), "collect");
        }
        assert!(rx.recv_timeout(std::time::Duration::from_millis(200)).is_err(), "each worker collects once per wake");
        queue.push(Box::new(|| {})).unwrap();
        assert_eq!(rx.recv_timeout(wait).unwrap(), "job");
        // A second wake: everyone again.
        queue.wake_all();
        for _ in 0..workers {
            assert_eq!(rx.recv_timeout(wait).unwrap(), "collect");
        }
    }

    /// B9: nested writes run off the write executor but keep its
    /// backpressure: rejected while its queue is full.
    #[test]
    fn nested_work_is_admitted_only_while_the_queue_has_room() {
        let executor: &'static Executor = Box::leak(Box::new(Executor::start(Pool::Write, 1, Some(1))));
        // Nested work alone: one admission fills a capacity of one.
        let first = executor.admit().expect("room for one");
        assert!(executor.admit().is_err(), "nested work is not counted against the queue");
        drop(first);
        assert!(executor.admit().is_ok());
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        executor.submit(Box::new(move || { let _ = started_tx.send(()); let _ = hold_rx.recv(); })).unwrap();
        started_rx.recv().unwrap();
        executor.submit(Box::new(|| {})).unwrap();
        let rejected = executor.admit().err().expect("admitted past a full queue");
        assert_eq!(rejected.status, 429);
        assert_eq!(executor.stats().rejected, 2);
        drop(hold_tx);
    }

    /// B9: a sub-request whose work panics leaves its (reused) blocking
    /// thread unmarked: later blocking tasks there are not taken for
    /// executor work.
    #[test]
    fn a_panicking_unpooled_task_restores_the_executor_mark() {
        let rt = tokio::runtime::Builder::new_current_thread().max_blocking_threads(1).enable_all().build().unwrap();
        rt.block_on(async {
            assert!(run_unpooled(|| -> () { panic!("injected sub-request panic") }).await.is_err());
            let marked = tokio::task::spawn_blocking(on_executor).await.unwrap();
            assert!(!marked, "the panicked task's executor mark leaked to the next task on its thread");
            assert!(run_unpooled(on_executor).await.unwrap(), "unpooled work is still marked");
        });
    }

    #[test]
    fn executor_handoff_preserves_recovery_credits_and_shard_owner() {
        use elasticsearch_engine::rustic_core::admission::{Budget,Lease,Owner};
        let budget=Budget::new(100);let owner=Owner::new();
        let lease=Lease::reserve(budget.clone(),80,"recovery plan").unwrap();
        let rt=tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let allocations={let _owner=owner.enter();let _lease=lease.enter();rt.block_on(async {
            let pooled_budget=budget.clone();
            let pooled=run(Pool::Recovery,move || pooled_budget.reserve(30,"pooled open").unwrap()).await.unwrap();
            let unpooled_budget=budget.clone();
            let unpooled=run_unpooled(move || unpooled_budget.reserve(40,"nested open").unwrap()).await.unwrap();
            (pooled,unpooled)
        })};
        assert_eq!(owner.bytes().current,70);assert_eq!(budget.used(),80);
        lease.complete().unwrap();drop(lease);assert_eq!(budget.used(),70);
        drop(allocations);assert_eq!(owner.bytes().current,0);assert_eq!(budget.used(),0);
    }
    #[test]
    fn routes_select_pools() {
        let segs = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["idx", "_search"])), Pool::Search);
        assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["_bulk"])), Pool::Write);
        assert_eq!(pool_for(&axum::http::Method::PUT, &segs(&["idx", "_doc", "1"])), Pool::Write);
        assert_eq!(pool_for(&axum::http::Method::GET, &segs(&["idx", "_doc", "1"])), Pool::Search);
        assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["idx", "_refresh"])), Pool::Refresh);
        assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["idx", "_flush"])), Pool::Flush);
        assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["idx", "_forcemerge"])), Pool::ForceMerge);
        assert_eq!(pool_for(&axum::http::Method::GET, &segs(&["_cluster", "health"])), Pool::Management);
        for task in ["_reindex", "_update_by_query", "_delete_by_query"] {
            assert_eq!(pool_for(&axum::http::Method::POST, &segs(&["idx", task])), Pool::Management, "{task}");
        }
    }

    /// Retriever deadlock (2026-11-08): `search_coordination` workers wait
    /// on pooled work, not nested; `search` workers' synchronous
    /// sub-requests are nested; coordination nested in coordination does
    /// not queue behind its waiting worker.
    #[test]
    fn coordination_workers_wait_on_pooled_work() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let handle = rt.handle().clone();
        rt.block_on(async move {
            let (marked, nested, coordinating) = run(Pool::SearchCoordination, || (on_executor(), sub_requests_nested(), on_coordination())).await.unwrap();
            assert!(!marked && !nested && coordinating, "coordination worker: marked {marked}, nested {nested}");
            let (marked, nested) = run(Pool::Search, || (on_executor(), sub_requests_nested())).await.unwrap();
            assert!(marked && nested, "search worker: marked {marked}, nested {nested}");
            // Every coordination worker (and spare) busy waiting on a nested
            // coordination: each still completes.
            let threads = executor(Pool::SearchCoordination).stats().threads;
            let waits = (0..threads * 6).map(|_| {
                let handle = handle.clone();
                run(Pool::SearchCoordination, move || {
                    blocking(|| handle.block_on(run(Pool::SearchCoordination, || {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        (on_executor(), on_coordination())
                    })))
                })
            }).collect::<Vec<_>>();
            let done = tokio::time::timeout(std::time::Duration::from_secs(30), futures::future::join_all(waits)).await.expect("nested coordination deadlocked");
            for inner in done {
                assert_eq!(inner.unwrap().unwrap(), (false, false));
            }
        });
    }

    #[test]
    fn a_full_bounded_queue_rejects_with_429() {
        let executor = Executor::start(Pool::Search, 1, Some(1));
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (started_tx, started_rx) = mpsc::channel::<()>();
        executor.submit(Box::new(move || {
            started_tx.send(()).unwrap();
            let _ = release_rx.recv();
        })).unwrap();
        started_rx.recv().unwrap();
        // The worker is busy: one task fits the queue, the next is rejected.
        executor.submit(Box::new(|| {})).unwrap();
        let err = executor.submit(Box::new(|| {})).unwrap_err();
        assert_eq!((err.status, err.kind.as_str()), (429, "es_rejected_execution_exception"));
        assert_eq!(executor.stats().rejected, 1);
        release_tx.send(()).unwrap();
    }
}
