//! Shared competitive threshold for partitioned exact search.
//!
//! Workers retain independent execution ranges, but publish accepted hits to
//! one bounded query heap. ES scores (not Vespa raw angular scores) determine
//! admission, preserving public float-score ties and document-id ordering.
//! Angular evaluation is deliberately full-distance: Vespa's angular
//! calc_with_limit also ignores its limit. A stale threshold only does extra
//! work; it cannot reject a competitive hit.
use super::Similarity;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

#[derive(Clone, Copy)]
struct Hit { id: u32, score: f32 }
impl PartialEq for Hit { fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal } }
impl Eq for Hit {}
impl PartialOrd for Hit { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Worst hit at the root: low score, then larger document id.
        other.score.total_cmp(&self.score).then(self.id.cmp(&other.id))
    }
}

pub struct SharedExactTopK {
    k: usize,
    minimum_score: AtomicU32,
    hits: Mutex<BinaryHeap<Hit>>,
}
impl SharedExactTopK {
    pub fn new(k: usize) -> Self {
        Self { k, minimum_score: AtomicU32::new(f32::NEG_INFINITY.to_bits()), hits: Mutex::new(BinaryHeap::new()) }
    }

    pub fn offer(&self, id: u32, score: f32) {
        if self.k == 0 || score.is_nan() || score < f32::from_bits(self.minimum_score.load(Relaxed)) { return; }
        let hit = Hit { id, score };
        let mut hits = self.hits.lock();
        if hits.len() < self.k { hits.push(hit); }
        else if hit < *hits.peek().unwrap() {
            *hits.peek_mut().unwrap() = hit;
        }
        if hits.len() == self.k {
            self.minimum_score.store(hits.peek().unwrap().score.to_bits(), Relaxed);
        }
    }

    /// Input ranges must be disjoint and ids unique within the query.
    pub fn scan<'a>(&self, query: &[f32], similarity: Similarity,
        rows: impl Iterator<Item = (u32, &'a [f32])>) -> u64 {
        if self.k == 0 { return 0; }
        let mut scanned = 0;
        let query_norm = if similarity == Similarity::Cosine { super::distance::norm(query) } else { 0.0 };
        for (id, row) in rows {
            scanned += 1;
            let score = if similarity == Similarity::Cosine {
                let cosine = if query_norm == 0.0 { 0.0 } else {
                    let (product, squared_norm) = super::distance::dot_and_sq_norm(query, row);
                    if squared_norm == 0.0 { 0.0 } else { product / (query_norm * squared_norm.sqrt()) }
                };
                (1.0 + cosine) / 2.0
            } else { similarity.score(query, row) };
            self.offer(id, score);
        }
        scanned
    }

    pub fn finish(&self) -> Vec<(u32, f32)> {
        let mut hits: Vec<_> = self.hits.lock().iter().map(|hit| (hit.id, hit.score)).collect();
        hits.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_threshold_retains_ties_and_total_order() {
        for k in [0, 1, 10, 10000] {
            let collector = SharedExactTopK::new(k);
            std::thread::scope(|scope| {
                for part in 0..8 {
                    let collector = &collector;
                    scope.spawn(move || {
                        for id in (part..4096).step_by(8) { collector.offer(id, (id % 13) as f32); }
                        collector.offer(10000 + part, f32::NAN);
                    });
                }
            });
            let mut expected: Vec<_> = (0..4096).map(|id| (id, (id % 13) as f32)).collect();
            expected.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            expected.truncate(k);
            assert_eq!(collector.finish(), expected);
        }
    }
}
