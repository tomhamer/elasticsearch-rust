//! Lazy metadata reads retain the same acquired-ref/generation contract.
use super::*;
use super::node_table::NodeRecord;

struct Rows([[f32; 2]; 4]);
impl MutableVectorSource for Rows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.0.get(key as usize).map(|row| row.as_slice())
    }
}
fn graph(similarity: Similarity) -> MutableHnsw {
    MutableHnsw::new(
        2,
        similarity,
        HnswParams {
            m: 2,
            ef_construction: 16,
            threads: 1,
            ..HnswParams::default()
        },
    )
}
fn install(g: &MutableHnsw, key: u64, inverse_norm: f32) -> NodeHandle {
    let mut permit = g.reserve_insert(key).unwrap();
    let id = permit.handle();
    g.test_install_node(id, key, inverse_norm, vec![vec![]])
        .unwrap();
    permit.pending = false;
    id
}
fn force_table_growth(g: &MutableHnsw) {
    // Grow the real table under the graph writer. Extra vacant capacity is not
    // a published node, and does not change graph/source bounds or topology.
    let mut writer = g.control.writer.lock();
    g.control.admit_write(&mut writer).unwrap();
    let capacity = g.control.nodes.memory_usage().capacity;
    g.control.mark_maintenance_pending(&mut writer);
    g.control
        .nodes
        .ensure_len(u32::try_from(capacity + 1).unwrap())
        .unwrap();
    g.control.finish_mutation(&mut writer);
}

#[test]
fn lazy_record_and_captured_entry_keep_identity_after_growth_and_removal() {
    assert_eq!(std::mem::size_of::<NodeRecord>(), 32);
    assert_eq!(std::mem::size_of::<NodeView<'_>>(), 24);
    let g = graph(Similarity::Cosine);
    let rows = Rows([[3.0, 4.0], [4.0, 3.0], [0.0, 1.0], [1.0, 0.0]]);
    let a = install(&g, 0, 0.2);
    let b = install(&g, 1, 0.2);
    g.test_replace_links(a, 0, &[internal_id(b)]).unwrap();
    g.test_replace_links(b, 0, &[internal_id(a)]).unwrap();
    g.test_set_entry(a).unwrap();
    let view = g.snapshot();
    let acquired = view.current_node(internal_id(a)).unwrap();
    let candidate = Candidate::new(0.0, acquired);
    let captured = view.entry_identity.unwrap();
    force_table_growth(&g);
    let current = view.current_node(internal_id(a)).unwrap();
    assert!(!std::ptr::eq(&*acquired.record, &*current.record));
    assert!(g.remove(a, &rows).unwrap());
    assert!(view.current_node(internal_id(a)).is_none());
    force_table_growth(&g); // copy the retired identity as well as live rows
    let retired_copy = g
        .control
        .nodes
        .read(&view.guard)
        .unwrap()
        .get(internal_id(a))
        .unwrap();
    assert!(!std::ptr::eq(&*current.record, &*retired_copy));
    assert!(g.collect_reclaimed().is_empty());
    // Read scalar fields only after growth/removal. The borrowed published old
    // record and fresh current record represent the same guarded incarnation.
    let from_candidate = view.candidate_node(candidate);
    let from_entry = view.acquired_node(a).unwrap();
    assert!(std::ptr::eq(&*from_candidate.record, &*retired_copy));
    assert!(std::ptr::eq(&*from_entry.record, &*retired_copy));
    for node in [acquired, from_candidate, from_entry] {
        assert_eq!(node.handle(), a);
        assert_eq!(node.key(), captured.source_key);
        assert_eq!(node.inverse_norm().to_bits(), captured.norm.to_bits());
        assert_eq!(
            node.record.publication_after_acquire(),
            captured.publication
        );
        assert_eq!(node.levels, acquired.levels);
        assert_eq!(node.links(0), Some([internal_id(b)].as_slice()));
    }
    let result = view.search(&[3.0, 4.0], 2, 8, &rows, |_| true);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].0, 1);
    assert_eq!(result.missing_source_lookups, 0);
    assert_eq!(result.entry_fallback_scanned, 0);
    drop(view);
    let tokens = g.collect_reclaimed();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].handle(), a);
    drop(tokens);
    let next = g.reserve_insert(2).unwrap();
    assert_eq!(next.handle().slot(), a.slot());
    assert_ne!(next.handle().generation(), a.generation());
}

#[test]
fn current_lookup_never_pairs_reserved_old_copy_with_new_published_norm() {
    let g = graph(Similarity::Cosine);
    let mut permit = g.reserve_insert(0).unwrap();
    let id = permit.handle();
    let guard = g.control.domain.take_guard().unwrap();
    let old_record = g
        .control
        .nodes
        .read(&guard)
        .unwrap()
        .get(internal_id(id))
        .unwrap();
    assert!(!old_record.levels_ref().is_valid());
    force_table_growth(&g);
    g.test_install_node(id, 0, 0.125, vec![vec![]]).unwrap();
    permit.pending = false;
    assert!(!old_record.levels_ref().is_valid());
    assert_eq!(old_record.metadata().norm, 0.0);
    let view = g.snapshot();
    let current = view.current_node(internal_id(id)).unwrap();
    assert!(!std::ptr::eq(&*old_record, &*current.record));
    assert_eq!(current.handle(), id);
    assert_eq!(current.key(), 0);
    assert_eq!(current.inverse_norm(), 0.125);
    assert!(view.visible(current));
    let encoded = g.encode();
    let reopened = MutableHnsw::open(&encoded).unwrap();
    assert_eq!(reopened.encode(), encoded);
    let reopened_view = reopened.snapshot();
    assert_eq!(
        reopened_view
            .current_node(internal_id(id))
            .unwrap()
            .inverse_norm(),
        0.125
    );
}

#[test]
fn later_publication_reusing_low_id_is_neither_source_read_nor_bridge() {
    let g = graph(Similarity::L2);
    let rows = Rows([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [15.0, 0.0]]);
    let removed = install(&g, 0, 0.0);
    let entry = install(&g, 1, 0.0);
    let old_target = install(&g, 2, 0.0);
    g.test_set_entry(entry).unwrap();
    assert!(g.remove(removed, &rows).unwrap());
    let tokens = g.collect_reclaimed();
    assert_eq!(tokens.len(), 1);
    drop(tokens);
    let old = g.snapshot();
    let future = install(&g, 3, 0.0);
    assert_eq!(future.slot(), removed.slot());
    assert_ne!(future.generation(), removed.generation());
    assert!(internal_id(future) < old.len);
    g.test_replace_links(entry, 0, &[internal_id(future)])
        .unwrap();
    g.test_replace_links(future, 0, &[internal_id(entry), internal_id(old_target)])
        .unwrap();
    g.test_replace_links(old_target, 0, &[internal_id(future)])
        .unwrap();
    let node = old.current_node(internal_id(future)).unwrap();
    assert!(!old.visible(node));
    struct OldRows<'a>(&'a Rows);
    impl MutableVectorSource for OldRows<'_> {
        fn get(&self, key: u64) -> Option<&[f32]> {
            assert!(key == 1 || key == 2, "source read outside old visibility");
            self.0.get(key)
        }
    }
    let source = OldRows(&rows);
    for mode in [
        MutableSearchMode::Plain,
        MutableSearchMode::VespaFilterFirst(VespaFilterFirstParams {
            filter_estimate: Some(VespaFilterEstimate { size: 3, count: 2 }),
            ..VespaFilterFirstParams::default()
        }),
    ] {
        let result = old
            .search_mode(&[20.0, 0.0], 3, 8, &source, |_| true, mode)
            .unwrap();
        assert_eq!(result.hits.iter().map(|hit| hit.0).collect::<Vec<_>>(), [1]);
        assert_eq!(result.missing_source_lookups, 0);
        assert_eq!(result.entry_fallback_scanned, 0);
    }
}
