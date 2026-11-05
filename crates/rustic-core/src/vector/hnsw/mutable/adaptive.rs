//! Optional intra-query distance parallelism. Frontier/visited/heap updates stay
//! on the caller, in their original order. No speculative graph expansion.
use std::sync::{OnceLock, atomic::{AtomicUsize, Ordering}};
use rayon::prelude::*;

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static HELPERS: AtomicUsize = AtomicUsize::new(0);
struct Executor { pool: rayon::ThreadPool, capacity: usize, workers: usize }
static EXECUTOR: OnceLock<Option<Executor>> = OnceLock::new();
fn executor() -> Option<&'static Executor> {
    EXECUTOR.get_or_init(|| {
        let workers = std::env::var("RUSTIC_HNSW_QUERY_WORKERS").ok()?.parse::<usize>().ok()?.min(4);
        let capacity = std::thread::available_parallelism().ok()?.get();
        if workers < 2 || capacity <= workers { return None; }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers)
            .thread_name(|i| format!("hnsw-distance-{i}")).build().ok()?;
        Some(Executor { pool, capacity, workers })
    }).as_ref()
}

pub(super) struct QueryWork(bool);
impl QueryWork {
    pub(super) fn new() -> Self {
        let enabled = executor().is_some();
        if enabled { ACTIVE.fetch_add(1, Ordering::AcqRel); }
        Self(enabled)
    }
}
impl Drop for QueryWork {
    fn drop(&mut self) { if self.0 { ACTIVE.fetch_sub(1, Ordering::AcqRel); } }
}
struct Lease(usize);
impl Drop for Lease { fn drop(&mut self) { HELPERS.fetch_sub(self.0, Ordering::AcqRel); } }
fn reserve(active: usize, capacity: usize, workers: usize, used: &AtomicUsize) -> bool {
    used.fetch_update(Ordering::AcqRel, Ordering::Acquire,
        |n| (active.saturating_add(n).saturating_add(workers) <= capacity).then_some(n + workers)).is_ok()
}

pub(super) fn eligible(rows: usize, dim: usize) -> bool {
    executor().is_some_and(|e| rows >= 16 && dim >= 256 && rayon::current_thread_index().is_none()
        && ACTIVE.load(Ordering::Acquire).saturating_add(HELPERS.load(Ordering::Acquire)).saturating_add(e.workers) <= e.capacity)
}

/// Nonblocking admission, reconsidered for each neighborhood. Small batches and
/// busy callers take the unchanged serial path. Avoid cross-pool Rayon installs:
/// its waiting worker could steal unrelated writes while holding a read view.
pub(super) fn calculate<T: Sync, R: Send>(rows: &[T], dim: usize,
    evaluate: impl Fn(&[T]) -> Vec<R> + Send + Sync) -> Option<Vec<R>> {
    let executor = executor()?;
    if !eligible(rows.len(), dim) { return None; }
    if !reserve(ACTIVE.load(Ordering::Acquire), executor.capacity, executor.workers, &HELPERS) { return None; }
    let _lease = Lease(executor.workers);
    // Multiples of four preserve the plain path's SIMD batch grouping exactly.
    let chunk = rows.len().div_ceil(executor.workers).div_ceil(4) * 4;
    Some(executor.pool.install(|| rows.par_chunks(chunk).map(evaluate).collect::<Vec<_>>()
        .into_iter().flatten().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_backs_off_without_waiting_or_oversubscribing() {
        let used = AtomicUsize::new(0);
        assert!(reserve(1, 8, 4, &used));
        assert!(!reserve(4, 8, 4, &used));
        assert_eq!(used.load(Ordering::Relaxed), 4);
        used.fetch_sub(4, Ordering::Relaxed);
        assert!(reserve(4, 8, 4, &used));
        assert!(!reserve(8, 8, 2, &AtomicUsize::new(0)));
    }
}
