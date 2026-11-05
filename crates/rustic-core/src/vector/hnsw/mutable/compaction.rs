//! Vespa-style level/link owner sweeps: relocate arrays without graph search.
//! Caller runs this at a commit/maintenance boundary, never per query.
use super::*;

struct NoVectors;
impl MutableVectorSource for NoVectors {
    fn get(&self, _: u64) -> Option<&[f32]> { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rows(Vec<Vec<f32>>);
    impl MutableVectorSource for Rows {
        fn get(&self, key: u64) -> Option<&[f32]> { self.0.get(key as usize).map(Vec::as_slice) }
    }

    #[test]
    fn relocation_preserves_topology_results_and_pinned_old_arrays() {
        let source = Rows((0..40).map(|i| vec![i as f32, (i % 7) as f32]).collect());
        let graph = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
        for key in 0..40 { graph.insert(key, &source).unwrap(); }
        let before = graph.encode();
        let expected = graph.search(&[13., 6.], 10, 40, &source, |_| true);
        let held = graph.snapshot();
        let nodes: Vec<_> = held.handles().map(|h| held.node(h).unwrap()).collect();
        let old_values: Vec<_> = nodes.iter().map(|node| node.links(0).unwrap().to_vec()).collect();
        let mut levels: Vec<_> = nodes.iter().map(|node| node.levels.buffer_id()).collect();
        levels.sort_unstable(); levels.dedup();
        let mut links: Vec<_> = nodes.iter().flat_map(|node| node.level_refs().iter()
            .map(|owner| LinksRef::from_raw(owner.load(Acquire)))
            .filter(|r| r.is_valid()).map(|r| r.buffer_id())).collect();
        links.sort_unstable(); links.dedup();
        // Fail link staging after level owners have moved. The completed
        // moves must survive, unfinished buffers must reactivate, and retry
        // must work without graph reconstruction or lost reader ownership.
        graph.control.links.test_allocation_failure(Some(1));
        assert!(graph.compact_selected(&mut graph.control.writer.lock(), &levels, &links).is_err());
        graph.control.links.test_allocation_failure(None);
        assert_eq!(graph.encode(), before);
        let generation = graph.mutation_generation();
        let moved = graph.compact_selected(&mut graph.control.writer.lock(), &levels, &links).unwrap();
        assert!(graph.mutation_generation() > generation, "compaction bumps the mutation generation");
        assert_eq!(moved, levels.len() + links.len());
        assert_eq!(graph.encode(), before);
        let after = graph.search(&[13., 6.], 10, 40, &source, |_| true);
        assert_eq!(after.hits, expected.hits);
        assert_eq!(after.vector_ops, expected.vector_ops);
        for (node, values) in nodes.iter().zip(old_values) {
            assert_eq!(node.links(0).unwrap(), values);
        }
        assert!(graph.storage_memory().levels.held_payload_bytes > 0);
        assert!(graph.storage_memory().links.held_payload_bytes > 0);
    }
}

impl MutableHnsw {
    /// Select fragmented buffers using the pinned default strategy and sweep
    /// their owners. Returns the number of retired buffers. Readers/prepared
    /// operations retain old arrays through the shared generation domain.
    /// No vectors are read, no distances computed and no nodes reinserted.
    pub fn compact_storage(&self) -> Result<usize> {
        let mut writer = self.control.writer.lock();
        self.control.admit_write(&mut writer)?;
        let levels = self.control.levels.writer().default_compaction_buffers();
        let links = self.control.links.writer().default_compaction_buffers();
        self.compact_selected(&mut writer, &levels, &links)
    }

    /// Sweep the owners of `levels` / `links` buffers. The topology writer
    /// is handed to waiting graph writers every [`CAPTURE_CHUNK_NODES`]
    /// owners (a fresh snapshot is taken after each hand-off): compacting
    /// buffers take no new arrays, so interleaved writes never add owners
    /// to them and the sweep still finds every one.
    fn compact_selected(&self, writer: &mut parking_lot::MutexGuard<'_, Writer>, levels: &[u32], links: &[u32]) -> Result<usize> {
        if levels.is_empty() && links.is_empty() { return Ok(0); }
        self.control.levels.writer().start_compaction(levels).map_err(crate::vector::hnsw::mutable::failure_error)?;
        if let Err(error) = self.control.links.writer().start_compaction(links) {
            for &id in levels { let _ = self.control.levels.writer().abort_compaction(id); }
            return Err(crate::vector::hnsw::mutable::failure_error(error));
        }
        self.control.mark_maintenance_pending(writer);
        let mut finished_levels = 0;
        let mut finished_links = 0;
        let result = (|| -> Result<usize> {
            let mut view = self.try_snapshot()?;
            // Same owner traversal as HnswIndex::compact_level_arrays and
            // compact_link_arrays. No O(N) temporary list of owners is needed.
            let mut id = 1;
            while id < view.len {
                if id % CAPTURE_CHUNK_NODES == 0 {
                    drop(view);
                    parking_lot::MutexGuard::bump(writer);
                    view = self.try_snapshot()?;
                    if id >= view.len { break; }
                }
                let current = id;
                id += 1;
                let Some(node) = view.current_node(current) else { continue };
                if levels.contains(&node.levels.buffer_id()) {
                    self.relocate_levels(&view, node.handle(), writer)?;
                }
            }
            let mut id = 1;
            while id < view.len {
                if id % CAPTURE_CHUNK_NODES == 0 {
                    drop(view);
                    parking_lot::MutexGuard::bump(writer);
                    view = self.try_snapshot()?;
                    if id >= view.len { break; }
                }
                let current = id;
                id += 1;
                let Some(node) = view.current_node(current) else { continue };
                let mut overlay = AdjacencyOverlay::new(&view, &NoVectors, self.similarity, self.params.m, None);
                for (level, owner) in node.level_refs().iter().enumerate() {
                    let reference = LinksRef::from_raw(owner.load(Acquire));
                    if reference.is_valid() && links.contains(&reference.buffer_id()) {
                        overlay.set(node.handle(), level, node.links(level).expect("live link owner").to_vec());
                    }
                }
                if !overlay.changed.is_empty() {
                    self.publish_overlay(&view, overlay, Publication::RelocateLinks, writer)?;
                }
            }
            for &id in levels {
                self.control.levels.writer().finish_compaction(id).map_err(crate::vector::hnsw::mutable::failure_error)?;
                finished_levels += 1;
            }
            for &id in links {
                self.control.links.writer().finish_compaction(id).map_err(crate::vector::hnsw::mutable::failure_error)?;
                finished_links += 1;
            }
            Ok(finished_levels + finished_links)
        })();
        if result.is_err() {
            // Published moves remain valid. Reactivate only unfinished buffers;
            // abort cannot allocate because start_compaction prepaid its credit.
            for &id in &levels[finished_levels..] {
                if self.control.levels.writer().abort_compaction(id).is_err() { self.control.poison(); }
            }
            for &id in &links[finished_links..] {
                if self.control.links.writer().abort_compaction(id).is_err() { self.control.poison(); }
            }
        }
        self.control.finish_mutation(writer);
        result
    }

    pub(super) fn relocate_levels(&self, view: &MutableHnswSnapshot, id: NodeHandle, writer: &mut Writer) -> Result<()> {
        let node = view.node(id).ok_or_else(|| failure("compaction owner absent"))?;
        let values: Vec<_> = node.level_refs().iter().map(|r| r.load(Acquire)).collect();
        let store_view = self.control.levels.view(&view.guard).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let token = unsafe { store_view.retirement_token(node.levels) }.map_err(crate::vector::hnsw::mutable::failure_error)?;
        self.control.mark_maintenance_pending(writer);
        let mut batch = self.control.levels.writer().stage_batch(1, &[token]).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let staged = batch.stage(&values).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let batch = batch.finish_staging().map_err(crate::vector::hnsw::mutable::failure_error)?;
        let replacement = batch.resolve(staged).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let transition = unsafe { self.control.nodes.preflight_replace_levels(internal_id(id), node.levels, replacement) }
            .map_err(crate::vector::hnsw::mutable::failure_error)?;
        let mut poison = PoisonOnUnwind { control: &self.control, armed: true };
        let batch = batch.begin_publish();
        transition.commit();
        unsafe { batch.finish_after_publish() };
        poison.armed = false;
        self.control.note_mutation();
        Ok(())
    }
}
