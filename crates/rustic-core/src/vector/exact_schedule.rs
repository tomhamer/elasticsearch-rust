//! Document-range assignment for parallel exact vector search.
//!
//! A fixed number of query workers pull disjoint ranges from one task cursor.
//! This is the task-range option of Vespa's match scheduler, not adaptive range
//! donation. The shared search pool bounds executing threads across queries;
//! no worker blocks while reserving another worker from that pool.
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ExactTaskRanges {
    documents: Range<usize>,
    tasks: usize,
    next: AtomicUsize,
}

impl ExactTaskRanges {
    pub fn new(documents: Range<usize>, tasks: usize) -> Self {
        assert!(documents.start <= documents.end);
        assert!(tasks > 0);
        Self { documents, tasks, next: AtomicUsize::new(0) }
    }

    /// Limit the query's worker count when few matching documents need scoring.
    /// The minimum work is a caller policy, not a hard-coded Vespa default.
    pub fn workers(max_threads: usize, matching: usize, min_per_thread: usize) -> usize {
        assert!(max_threads > 0 && min_per_thread > 0);
        max_threads.min((matching / min_per_thread).max(1))
    }

    pub fn claim(&self) -> Option<Range<usize>> {
        // Saturate at the task count: repeated calls after exhaustion cannot
        // wrap the cursor and hand out duplicate ranges.
        let task = self.next.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
            |next| (next < self.tasks).then(|| next + 1)).ok()?;
        let width = self.documents.end - self.documents.start;
        let start = self.documents.start + width / self.tasks * task + (width % self.tasks).min(task);
        let end = start + width / self.tasks + usize::from(task < width % self.tasks);
        Some(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_claims_cover_each_document_once() {
        for (start, end, tasks) in [(17, 511, 13), (10, 13, 32), (3, 3, 5), (0, 10007, 19)] {
            let ranges = ExactTaskRanges::new(start..end, tasks);
            let mut claimed = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8).map(|_| {
                    let ranges = &ranges;
                    scope.spawn(move || std::iter::from_fn(|| ranges.claim()).collect::<Vec<_>>())
                }).collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
            });
            assert_eq!(claimed.len(), tasks);
            claimed.sort_by_key(|range| (range.start, range.end));
            assert_eq!(claimed.into_iter().flatten().collect::<Vec<_>>(), (start..end).collect::<Vec<_>>());
            assert!(ranges.claim().is_none());
        }
    }

    #[test]
    fn minimum_work_caps_workers() {
        assert_eq!(ExactTaskRanges::workers(32, 0, 1024), 1);
        assert_eq!(ExactTaskRanges::workers(32, 2047, 1024), 1);
        assert_eq!(ExactTaskRanges::workers(32, 2048, 1024), 2);
        assert_eq!(ExactTaskRanges::workers(32, usize::MAX, 1024), 32);
    }
}
