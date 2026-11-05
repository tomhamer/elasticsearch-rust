//! Bounded exact-key cache for ES text-document to graph-node projection.
//! No query hash collisions, document masks omitted, or graph epochs retained.
use std::sync::Arc;
pub(crate) type ProjectionKey = Vec<(u64, Vec<u64>)>;
#[derive(Default)]
pub(super) struct ProjectionCache { entries: Vec<Entry>, bytes: usize, maps: Vec<NodeMap>, map_bytes: usize }
struct Entry { key: ProjectionKey, _key_memory:crate::admission::Reservation, revision: u64, nodes: u32, complete: bool, words: Arc<crate::admission::Allocation<Vec<u64>>>, matched: usize, bytes: usize }
struct NodeMap { segment: u64, revision: u64, nodes: u32, mapping: Arc<crate::admission::Allocation<Vec<u32>>> }
const MAX_BYTES: usize = 16 * 1024 * 1024;
impl ProjectionCache {
    #[cfg(test)]
    fn insert(&mut self,key:ProjectionKey,revision:u64,nodes:u32,words:Arc<Vec<u64>>,matched:usize) {
        let bytes=key.iter().map(|(_,w)|w.len() as u64*8+32).sum::<u64>()+256;
        let memory=crate::admission::shared().reserve(bytes,"test projection key").unwrap();
        let wm=crate::admission::shared().reserve(words.len() as u64*8+256,"test projection words").unwrap();
        self.insert_admitted(key,memory,revision,nodes,false,Arc::new(crate::admission::Allocation::new((*words).clone(),wm)),matched);
    }
    #[cfg(test)]
    fn insert_node_map(&mut self,segment:u64,revision:u64,nodes:u32,mapping:Arc<Vec<u32>>) {
        let memory=crate::admission::shared().reserve(mapping.len() as u64*4+256,"test projection mapping").unwrap();
        self.insert_node_map_admitted(segment,revision,nodes,Arc::new(crate::admission::Allocation::new((*mapping).clone(),memory)));
    }
    // Independent bound: structural translations survive mask eviction, but
    // never identity mutation or graph-bound changes. No graph epoch is held.
    pub(super) fn node_map(&mut self, segment: u64, revision: u64, nodes: u32) -> Option<Arc<crate::admission::Allocation<Vec<u32>>>> {
        self.maps.retain(|m| m.revision == revision && m.nodes == nodes);
        self.map_bytes = self.maps.iter().map(|m| m.mapping.len() * 4).sum();
        let i = self.maps.iter().position(|m| m.segment == segment)?;
        let entry = self.maps.remove(i);
        let result = entry.mapping.clone();
        self.maps.push(entry);
        Some(result)
    }
    pub(super) fn insert_node_map_admitted(&mut self, segment: u64, revision: u64, nodes: u32, mapping: Arc<crate::admission::Allocation<Vec<u32>>>) {
        let bytes = mapping.len().saturating_mul(4);
        if bytes > MAX_BYTES || self.node_map(segment, revision, nodes).is_some() { return; }
        while !self.maps.is_empty() && (self.maps.len() >= 1024 || self.map_bytes.saturating_add(bytes) > MAX_BYTES) {
            self.map_bytes -= self.maps.remove(0).mapping.len() * 4;
        }
        self.map_bytes += bytes;
        self.maps.push(NodeMap { segment, revision, nodes, mapping });
    }
    pub(super) fn map_fits(docs: usize) -> bool { docs <= MAX_BYTES / 4 }
    pub(super) fn get(&mut self, key: &ProjectionKey, revision: u64, nodes: u32) -> Option<(Arc<crate::admission::Allocation<Vec<u64>>>, usize)> {
        // A complete entry (every masked doc with a vector had a node) also
        // survives the graph's node bound changing: its nodes are docs the
        // key's masks see live, which no repair removes while such a view
        // exists (a shrink: the last slot removed by a delete repair), and
        // a slot the graph gains is a point of none of the key's docs (each
        // already had its node), which its words, sized at the old bound,
        // reject. Under concurrent indexing the graph grows between any two
        // queries: evicting on growth rebuilt the projection on every
        // query (a node-map lookup per doc of every segment).
        self.entries.retain(|e| e.revision == revision && (e.nodes == nodes || e.complete));
        self.bytes = self.entries.iter().map(|e| e.bytes).sum();
        let i = self.entries.iter().position(|e| &e.key == key)?;
        let entry = self.entries.remove(i);
        let result = (entry.words.clone(), entry.matched);
        self.entries.push(entry);
        Some(result)
    }
    pub(super) fn insert_admitted(&mut self, key: ProjectionKey, key_memory:crate::admission::Reservation, revision: u64, nodes: u32, complete: bool, words: Arc<crate::admission::Allocation<Vec<u64>>>, matched: usize) {
        let bytes = words.len().saturating_mul(8).saturating_add(key.iter().map(|(_, w)| w.len().saturating_mul(8).saturating_add(32)).sum::<usize>());
        if bytes > MAX_BYTES { return; }
        if self.get(&key, revision, nodes).is_some() { return; }
        // Replaces an entry kept only through the shrink rule.
        if let Some(i) = self.entries.iter().position(|e| e.key == key) { self.bytes -= self.entries.remove(i).bytes; }
        while !self.entries.is_empty() && (self.entries.len() >= 8 || self.bytes.saturating_add(bytes) > MAX_BYTES) {
            self.bytes -= self.entries.remove(0).bytes;
        }
        self.bytes += bytes;
        self.entries.push(Entry { key, _key_memory:key_memory, revision, nodes, complete, words, matched, bytes });
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_result_admission_survives_cache_eviction() {
        let budget=crate::admission::Budget::new(4096);
        let key_memory=budget.reserve(256,"key").unwrap();
        let word_memory=budget.reserve(264,"words").unwrap();
        let words=Arc::new(crate::admission::Allocation::new(vec![6],word_memory));
        let mut cache=ProjectionCache::default();
        cache.insert_admitted(vec![(7,vec![3])],key_memory,1,10,false,words.clone(),2);
        let reader=cache.get(&vec![(7,vec![3])],1,10).unwrap().0;
        drop(words);drop(cache);
        assert_eq!(budget.used(),264,"escaped query still owns result");
        assert_eq!(reader[0],6);drop(reader);assert_eq!(budget.used(),0);
    }

    #[test]
    fn structural_maps_are_bounded_and_invalidated_independently_of_masks() {
        let mut cache = ProjectionCache::default();
        let map = Arc::new(vec![3, 0, 2]);
        cache.insert_node_map(1, 5, 8, map.clone());
        for n in 0..20 { cache.insert(vec![(n,vec![1])], 5, 8, Arc::new(vec![2]), 1); }
        assert_eq!(cache.node_map(1,5,8).unwrap().as_slice(),map.as_slice());
        assert!(cache.node_map(2, 5, 8).is_none());
        assert!(cache.node_map(1, 6, 8).is_none());
        cache.insert_node_map(1, 6, 8, map.clone());
        assert!(cache.node_map(1, 6, 9).is_none());
        for n in 0..6 { cache.insert_node_map(n, 6, 9, Arc::new(vec![1; MAX_BYTES / 16])); }
        assert_eq!(cache.map_bytes, MAX_BYTES);
        assert!(cache.node_map(0, 6, 9).is_none());
        assert!(!ProjectionCache::map_fits(MAX_BYTES / 4 + 1));
    }
    #[test]
    fn exact_masks_segment_identity_and_graph_revision_gate_reuse() {
        let mut c = ProjectionCache::default();
        let key = vec![(7, vec![3])];
        let words = Arc::new(vec![6]);
        c.insert(key.clone(), 1, 10, words.clone(), 2);
        assert_eq!(c.get(&key,1,10).unwrap().0.as_slice(),words.as_slice());
        assert!(c.get(&vec![(7, vec![1])], 1, 10).is_none()); // deletion/filter change
        assert!(c.get(&vec![(8, vec![3])], 1, 10).is_none()); // merged/reopened segment
        assert!(c.get(&key, 2, 10).is_none()); // slot removal/reuse, even with same node bound
        c.insert(key.clone(), 2, 10, words, 2);
        assert!(c.get(&key, 2, 11).is_none()); // graph grew
    }
    #[test]
    fn complete_entries_survive_graph_growth_incomplete_ones_do_not() {
        let mut c = ProjectionCache::default();
        let memory = || crate::admission::shared().reserve(512, "test projection").unwrap();
        let words = || Arc::new(crate::admission::Allocation::new(vec![6], memory()));
        c.insert_admitted(vec![(7, vec![3])], memory(), 1, 10, true, words(), 2);
        c.insert_admitted(vec![(8, vec![3])], memory(), 1, 10, false, words(), 2);
        assert_eq!(c.get(&vec![(7, vec![3])], 1, 12).unwrap().0.as_slice(), &[6]);
        assert!(c.get(&vec![(8, vec![3])], 1, 12).is_none(), "a doc without a node may have gained one");
        assert!(c.get(&vec![(7, vec![3])], 2, 12).is_none(), "projection epoch change");
    }
    #[test]
    fn bounded_entries_and_oversized_filters() {
        let mut c = ProjectionCache::default();
        for n in 0..20 { c.insert(vec![(n,vec![1])], 1, 10, Arc::new(vec![2]), 1); }
        assert_eq!(c.entries.len(),8);
        assert!(c.get(&vec![(0,vec![1])],1,10).is_none());
        c.insert(vec![],1,10,Arc::new(vec![0; MAX_BYTES / 8 + 1]),0);
        assert_eq!(c.entries.len(),8);
        assert!(c.bytes <= MAX_BYTES);
    }
}
