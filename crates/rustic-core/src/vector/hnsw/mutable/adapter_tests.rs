//! Causal adapter regressions: retained entry acquisition and checkpoint migration.
use super::*;

struct FixedRows([[f32; 2]; 3]);
impl MutableVectorSource for FixedRows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.0.get(key as usize).map(|row| row.as_slice())
    }
}
fn fixture() -> (MutableHnsw, FixedRows) {
    (
        MutableHnsw::new(
            2,
            Similarity::L2,
            HnswParams {
                m: 2,
                ef_construction: 16,
                seed: 7,
                threads: 1,
            },
        ),
        FixedRows([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
}
fn install(graph: &MutableHnsw, key: u64, level: usize) -> NodeHandle {
    let mut permit = graph.reserve_insert(key).unwrap();
    let handle = permit.handle();
    graph
        .test_install_node(handle, key, 0.0, vec![vec![]; level + 1])
        .unwrap();
    // The test-only installation consumes the writer's pending record.
    permit.pending = false;
    handle
}
fn connect(graph: &MutableHnsw, a: NodeHandle, b: NodeHandle, level: usize) {
    graph
        .test_replace_links(a, level, &[internal_id(b)])
        .unwrap();
    graph
        .test_replace_links(b, level, &[internal_id(a)])
        .unwrap();
}

#[test]
fn entry_acquisition_retries_when_writer_replaces_and_removes_loaded_entry() {
    let (graph, source) = fixture();
    let a = install(&graph, 0, 1);
    let b = install(&graph, 1, 0);
    connect(&graph, a, b, 0);
    graph.test_set_entry(a).unwrap();
    let mut loads = 0;
    let view = graph
        .try_snapshot_with_hook(|packed| {
            loads += 1;
            if loads == 1 {
                assert_eq!(packed, packed_entry(internal_id(a), 1));
                assert!(graph.remove(a, &source).unwrap());
            }
        })
        .unwrap();
    assert_eq!(loads, 2, "the obsolete tuple must be retried");
    assert_eq!(view.entry, packed_entry(internal_id(b), 0));
    let node = view.acquired_node(b).unwrap();
    assert_eq!(node.levels, view.entry_levels);
    assert_eq!(node.key(), 1);
    assert!(view.current_node(internal_id(a)).is_none());
    let result = view.search(&[1.0, 0.0], 1, 4, &source, |_| true);
    assert_eq!(result.hits, vec![(1, 1.0)]);
    assert_eq!(result.missing_source_lookups, 0);
    assert_eq!(result.entry_fallback_scanned, 0);
}

#[test]
fn acquired_entry_remains_traversable_after_removal_until_reader_drains() {
    let (graph, source) = fixture();
    let a = install(&graph, 0, 1);
    let b = install(&graph, 1, 0);
    connect(&graph, a, b, 0);
    graph.test_set_entry(a).unwrap();
    let view = graph.snapshot();
    let old_levels = view.entry_levels;
    assert!(graph.remove(a, &source).unwrap());
    assert!(view.current_node(internal_id(a)).is_none());
    let retained = view.acquired_node(a).unwrap();
    assert_eq!(retained.levels, old_levels);
    assert_eq!(retained.links(0), Some([internal_id(b)].as_slice()));
    assert!(graph.collect_reclaimed().is_empty());
    // Current topology is searchable via the retained entry. This does not
    // promise PIT repeatability or return the removed document as a live hit.
    let result = view.search(&[1.0, 0.0], 2, 4, &source, |_| true);
    assert_eq!(result.hits, vec![(1, 1.0)]);
    assert_eq!(result.missing_source_lookups, 0);
    assert_eq!(result.entry_fallback_scanned, 0);
    drop(view);
    let tokens = graph.collect_reclaimed();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].handle(), a);
    drop(tokens);
    let permit = graph.reserve_insert(2).unwrap();
    assert_eq!(permit.handle().slot(), a.slot());
    assert_ne!(permit.handle().generation(), a.generation());
}

#[test]
fn removal_replaces_entry_at_highest_nonempty_level_and_codec_preserves_it() {
    let (graph, source) = fixture();
    let a = install(&graph, 0, 3);
    let b = install(&graph, 1, 2);
    let c = install(&graph, 2, 0);
    connect(&graph, a, b, 1);
    connect(&graph, a, c, 0);
    graph.test_set_entry(a).unwrap();
    assert!(graph.remove(a, &source).unwrap());
    let view = graph.snapshot();
    assert_eq!(view.entry, packed_entry(internal_id(b), 1));
    assert_eq!(view.current_node(internal_id(b)).unwrap().level(), 2);
    drop(view);
    let bytes = graph.encode();
    assert_eq!(&bytes[..8], b"RMHG\x04\0\0\0");
    let reopened = MutableHnsw::open(&bytes).unwrap();
    assert_eq!(reopened.snapshot().entry, packed_entry(internal_id(b), 1));
    assert_eq!(reopened.encode(), bytes);
}

// Handwritten legacy record, independent of both the new and old encoders.
// Two L2 nodes, reciprocal level-zero edges in legacy zero-based public slots.
fn legacy_graph(entry: u64, first_neighbor: u32) -> Vec<u8> {
    fn u32_(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }
    fn u64_(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }
    let mut bytes = b"RMHG\x03\0\0\0".to_vec();
    for value in [2, 2, 2] {
        u32_(&mut bytes, value);
    } // dim, L2, M
    for value in [16, 7, entry] {
        u64_(&mut bytes, value);
    }
    u32_(&mut bytes, 2); // allocated slots
    u64_(&mut bytes, 2); // publication
    u32_(&mut bytes, 1); // has source watermark
    u64_(&mut bytes, 1);
    for slot in 0..2u32 {
        u32_(&mut bytes, 1); // incarnation
        u32_(&mut bytes, 1); // present
        u64_(&mut bytes, slot as u64); // source key
        u64_(&mut bytes, slot as u64 + 1); // born
        u32_(&mut bytes, 0); // L2 inverse norm
        u32_(&mut bytes, 0); // maximum level
        u32_(&mut bytes, 0); // no visibility retirement
        u64_(&mut bytes, 0);
        u32_(&mut bytes, 1); // neighbor count
        u32_(&mut bytes, if slot == 0 { first_neighbor } else { 0 });
    }
    bytes
}

#[test]
fn rmhg3_migration_converts_slot_links_and_validates_entry_incarnation() {
    let (_, source) = fixture();
    let legacy = legacy_graph(1u64 << 32, 1);
    let graph = MutableHnsw::open(&legacy).unwrap();
    let view = graph.snapshot();
    assert_eq!(view.entry, packed_entry(1, 0));
    assert_eq!(view.current_node(1).unwrap().links(0), Some([2].as_slice()));
    assert_eq!(view.current_node(2).unwrap().links(0), Some([1].as_slice()));
    let result = view.search(&[1.0, 0.0], 2, 4, &source, |_| true);
    assert_eq!(result.hits, vec![(1, 1.0), (0, 0.5)]);
    assert_eq!(result.missing_source_lookups, 0);
    drop(view);
    let canonical = graph.encode();
    assert_eq!(&canonical[..8], b"RMHG\x04\0\0\0");
    assert_eq!(MutableHnsw::open(&canonical).unwrap().encode(), canonical);
    assert!(MutableHnsw::open(&legacy_graph(2u64 << 32, 1)).is_err());
    assert!(MutableHnsw::open(&legacy_graph(1u64 << 32, 2)).is_err());
}
