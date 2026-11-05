//! Sparse stable-point directory. Text segments normally enumerate nearby point
//! IDs, so query preparation hashes once per page run, then directly indexes it.
//! Keep full generation-bearing handles; stable point IDs are not graph slots.
use std::collections::HashMap;
use crate::error::{Result, RusticError};
use super::NodeHandle;

const SHIFT: u32 = 8;
const SIZE: usize = 1 << SHIFT;
const MASK: u64 = SIZE as u64 - 1;

#[derive(Default)]
pub(super) struct PointHandles {
    pages: HashMap<u64, Page>,
    revision: u64,
}
struct Page {
    rows: Vec<Option<NodeHandle>>,
    // Query projection needs only slot + 1, not the generation-bearing handle.
    // Zero is missing; retain full handles above for ownership/mutation checks.
    nodes: Box<[u32; SIZE]>,
    live: usize,
}
impl PointHandles {
    pub(super) fn revision(&self) -> u64 { self.revision }
    /// Heap bytes of the directory's pages.
    pub(super) fn heap_bytes(&self) -> usize {
        self.pages.values().map(|p| std::mem::size_of::<Page>() + p.rows.capacity() * std::mem::size_of::<Option<NodeHandle>>() + SIZE * 4).sum::<usize>()
            + self.pages.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<Page>())
    }
    fn changed(&mut self) { self.revision = self.revision.checked_add(1).expect("point identity revision exhausted"); }
    pub(super) fn extend(&mut self, entries: impl Iterator<Item = (u64, NodeHandle)> + Clone) -> Result<()> {
        // Allocate all missing pages before publishing any identity. The caller
        // holds the write guard throughout reservation and insertion.
        for (point, handle) in entries.clone() {
            if handle.slot() == u32::MAX {
                return Err(RusticError::InvalidQuery("graph slot exceeds node-ID range".into()));
            }
            let page = point >> SHIFT;
            if !self.pages.contains_key(&page) {
                self.pages.try_reserve(1).map_err(|_| RusticError::Memory("point directory".into()))?;
                let mut rows = Vec::new();
                rows.try_reserve_exact(SIZE).map_err(|_| RusticError::Memory("point directory page".into()))?;
                rows.resize(SIZE, None);
                self.pages.insert(page, Page { rows, nodes: Box::new([0; SIZE]), live: 0 });
            }
        }
        self.changed();
        for (point, handle) in entries {
            let page = self.pages.get_mut(&(point >> SHIFT)).expect("reserved point page");
            let row = &mut page.rows[(point & MASK) as usize];
            if row.is_none() { page.live += 1; }
            *row = Some(handle);
            page.nodes[(point & MASK) as usize] = handle.slot() + 1;
        }
        Ok(())
    }
    pub(super) fn get(&self, point: &u64) -> Option<&NodeHandle> {
        self.pages.get(&(point >> SHIFT))?.rows[(point & MASK) as usize].as_ref()
    }
    pub(super) fn remove(&mut self, point: &u64) {
        self.changed();
        let key = point >> SHIFT;
        if let Some(page) = self.pages.get_mut(&key) {
            if page.rows[(point & MASK) as usize].take().is_some() { page.live -= 1; }
            page.nodes[(point & MASK) as usize] = 0;
            if page.live == 0 { self.pages.remove(&key); }
        }
    }
    /// Every point with a handle (unordered).
    pub(super) fn points(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.pages.values().map(|page| page.live).sum());
        for (key, page) in &self.pages {
            for (offset, row) in page.rows.iter().enumerate() {
                if row.is_some() { out.push((*key << SHIFT) | offset as u64); }
            }
        }
        out
    }
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&u64, &NodeHandle) -> bool) {
        self.changed();
        self.pages.retain(|key, page| {
            for (offset, row) in page.rows.iter_mut().enumerate() {
                if row.as_ref().is_some_and(|handle| !keep(&((*key << SHIFT) | offset as u64), handle)) {
                    *row = None;
                    page.nodes[offset] = 0;
                    page.live -= 1;
                }
            }
            page.live != 0
        });
    }
    /// Stream eligible points through the compact node directory. The caller
    /// holds this directory's read guard together with a graph/source view.
    pub(super) fn visit_nodes(&self, points: impl IntoIterator<Item = u64>, mut visit: impl FnMut(u32)) {
        let mut previous = None;
        let mut page = None;
        for point in points {
            let key = point >> SHIFT;
            if previous != Some(key) {
                page = self.pages.get(&key).map(|page| &page.nodes);
                previous = Some(key);
            }
            if let Some(nodes) = page {
                let node = nodes[(point & MASK) as usize];
                if node != 0 { visit(node); }
            }
        }
    }
    pub(super) fn visit(&self, points: &[u64], mut visit: impl FnMut(NodeHandle)) {
        let mut previous = None;
        let mut page = None;
        for &point in points {
            let key = point >> SHIFT;
            if previous != Some(key) {
                page = self.pages.get(&key);
                previous = Some(key);
            }
            if let Some(handle) = page.and_then(|page| page.rows[(point & MASK) as usize]) {
                visit(handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::{hnsw::MutableHnsw, HnswParams, Similarity};

    #[test]
    fn sparse_pages_preserve_lookup_order_removal_and_replacement() {
        let graph = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
        let first = graph.reserve_insert(1).unwrap();
        let second = graph.reserve_insert(2).unwrap();
        let a = first.handle();
        let b = second.handle();
        let mut map = PointHandles::default();
        map.extend([(0, a), (255, b), (256, a), (u64::MAX, b)].into_iter()).unwrap();
        assert_eq!(map.pages.len(), 3, "sparse keys must not allocate their gaps");
        let mut found = Vec::new();
        map.visit(&[u64::MAX, 255, 254, 256, 0, 255], |h| found.push(h));
        assert_eq!(found, vec![b,b,a,a,b]);
        let mut compact = Vec::new();
        map.visit_nodes([u64::MAX, 255, 254, 256, 0, 255], |n| compact.push(n));
        assert_eq!(compact, found.iter().map(|h| h.slot()+1).collect::<Vec<_>>());
        map.extend(std::iter::once((0, b))).unwrap();
        assert_eq!(map.get(&0), Some(&b));
        let mut compact = Vec::new();
        map.visit_nodes([0], |n| compact.push(n));
        assert_eq!(compact, vec![b.slot()+1]);
        assert_eq!(map.pages[&0].live, 2);
        map.remove(&255);
        map.remove(&255);
        assert!(map.get(&255).is_none());
        map.retain(|key, _| *key == u64::MAX);
        assert_eq!(map.pages.len(), 1);
        assert_eq!(map.get(&u64::MAX), Some(&b));
        let mut compact = Vec::new();
        map.visit_nodes([0, 255, 256, u64::MAX], |n| compact.push(n));
        assert_eq!(compact, vec![b.slot()+1]);
        map.remove(&u64::MAX);
        assert!(map.pages.is_empty());
    }
    #[test]
    #[ignore = "EC2-only diagnostic lookup benchmark; not an end-to-end ANN result"]
    fn point_directory_lookup_cost_at_two_million_points() {
        use std::{hint::black_box, time::Instant};
        let graph = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
        let reservation = graph.reserve_insert(0).unwrap();
        let handle = reservation.handle();
        let mut pages = PointHandles::default();
        pages.extend((0..2_000_000).map(|point| (point, handle))).unwrap();
        let hash: HashMap<_, _> = (0..2_000_000u64).map(|point| (point, handle)).collect();
        for stride in [1usize, 2, 10] {
            let points: Vec<_> = (0..2_000_000u64).step_by(stride).collect();
            let mut baseline_ms = Vec::new();
            let mut paged_ms = Vec::new();
            for round in 0..8 {
                // Alternate measurement order to avoid giving one table every
                // first pass. Both produce exactly the same eligibility bit.
                for paged in if round % 2 == 0 { [false, true] } else { [true, false] } {
                    let mut mask = [0u64; 1];
                    let begin = Instant::now();
                    if paged {
                        pages.visit(black_box(&points), |h| mask[0] |= 1 << (h.slot() + 1));
                    } else {
                        for point in black_box(&points) {
                            if let Some(h) = hash.get(point) { mask[0] |= 1 << (h.slot() + 1); }
                        }
                    }
                    let elapsed = begin.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(black_box(mask), [2]);
                    if paged { paged_ms.push(elapsed); } else { baseline_ms.push(elapsed); }
                }
            }
            eprintln!("point-directory diagnostic: points={} stride={} hash_ms={:?} paged_ms={:?}", points.len(), stride, baseline_ms, paged_ms);
        }
    }

    #[test]
    #[ignore = "EC2-only projection microbenchmark; not ANN acceptance"]
    fn compact_stream_projection_at_two_million_points() {
        use std::{hint::black_box, time::Instant};
        let graph = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
        let reservation = graph.reserve_insert(0).unwrap();
        let handle = reservation.handle();
        let mut directory = PointHandles::default();
        directory.extend((0..2_000_000).map(|point| (point, handle))).unwrap();
        for stride in [2usize, 10] {
            let mut baseline = Vec::new();
            let mut compact = Vec::new();
            for round in 0..12 {
                for candidate in if round % 2 == 0 { [false, true] } else { [true, false] } {
                    let start = Instant::now();
                    let mut sum = 0u64;
                    let points = (0..2_000_000u64).step_by(stride).map(black_box);
                    if candidate {
                        directory.visit_nodes(points, |node| sum += black_box(node) as u64);
                    } else {
                        let points: Vec<_> = points.collect();
                        directory.visit(&points, |h| sum += black_box(h.slot()+1) as u64);
                    }
                    let elapsed = start.elapsed().as_secs_f64()*1000.;
                    assert_eq!(sum, (2_000_000/stride) as u64 * (handle.slot()+1) as u64);
                    if candidate { compact.push(elapsed); } else { baseline.push(elapsed); }
                }
            }
            eprintln!("compact-stream diagnostic: stride={stride} full_handle_vec_ms={baseline:?} compact_stream_ms={compact:?}");
        }
    }

}
