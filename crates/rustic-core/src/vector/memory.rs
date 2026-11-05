//! Process-wide admission for optional vector-column heap copies.
//! Reservations live as long as the cached allocation, including old segments
//! still referenced by searches during optimizer publication.
use std::sync::{OnceLock, atomic::{AtomicUsize, Ordering}};

pub(super) struct CacheBudget {
    limit: usize,
    used: AtomicUsize,
}
impl CacheBudget {
    fn new(limit: usize) -> Self { Self { limit, used: AtomicUsize::new(0) } }
    pub(super) fn reserve(&self, bytes: usize) -> Option<Reservation<'_>> {
        self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes).filter(|next| *next <= self.limit)
        }).ok()?;
        Some(Reservation { budget: self, bytes })
    }
    pub(super) fn used(&self) -> usize { self.used.load(Ordering::Relaxed) }
}
pub(super) struct Reservation<'a> { budget: &'a CacheBudget, bytes: usize }
impl Drop for Reservation<'_> {
    fn drop(&mut self) { self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel); }
}
pub(super) fn column_budget() -> &'static CacheBudget {
    static BUDGET: OnceLock<CacheBudget> = OnceLock::new();
    BUDGET.get_or_init(|| {
        // Bytes; zero disables optional copies. This bounds the column cache,
        // not all vector/index/merge memory. Admission safety is per process.
        let limit = std::env::var("RUSTIC_VECTOR_COLUMN_CACHE_BYTES").ok()
            .and_then(|value| value.parse().ok()).unwrap_or(256 * 1024 * 1024);
        CacheBudget::new(limit)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_is_bounded_and_releases_only_when_the_owner_drops() {
        let budget = CacheBudget::new(100);
        let first = budget.reserve(60).unwrap();
        assert!(budget.reserve(41).is_none());
        let second = budget.reserve(40).unwrap();
        assert!(budget.reserve(1).is_none());
        drop(first);
        let third = budget.reserve(60).unwrap();
        assert_eq!(budget.used.load(Ordering::Acquire), 100);
        drop(second); drop(third);
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
        assert!(CacheBudget::new(0).reserve(1).is_none());
        assert!(budget.reserve(usize::MAX).is_none());
    }
    #[test]
    fn concurrent_admission_cannot_oversubscribe() {
        let budget = CacheBudget::new(64);
        let barrier = std::sync::Barrier::new(16);
        std::thread::scope(|scope| {
            for _ in 0..16 {
                let budget = &budget; let barrier = &barrier;
                scope.spawn(move || {
                    let reservation = budget.reserve(16);
                    barrier.wait();
                    assert!(budget.used.load(Ordering::Acquire) <= 64);
                    barrier.wait();
                    drop(reservation);
                });
            }
        });
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }
}
