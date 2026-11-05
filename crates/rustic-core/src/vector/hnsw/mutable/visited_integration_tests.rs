use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};
use visited::Policy;

const HIGH_KEY: u64 = 16_383;
struct SparseRows;
impl MutableVectorSource for SparseRows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        match key {
            0 => Some(&[0.0]),
            HIGH_KEY => Some(&[1.0]),
            _ => None,
        }
    }
}

// Reserve/cancel intermediate IDs without acknowledging their source tokens.
// Only two nodes are published, but the live high node gives a real sparse
// graph-ID universe large enough to exercise both Vespa visited policies.
fn sparse_graph() -> (MutableHnsw, NodeHandle) {
    let g = MutableHnsw::new(
        1,
        Similarity::L2,
        HnswParams {
            m: 1,
            ef_construction: 16,
            ..HnswParams::default()
        },
    );
    g.insert(0, &SparseRows).unwrap();
    for key in 1..HIGH_KEY {
        drop(g.reserve_insert(key).unwrap());
    }
    let high = g.insert(HIGH_KEY, &SparseRows).unwrap();
    g.test_set_entry(high).unwrap();
    (g, high)
}

#[test]
fn plain_visited_policy_changes_storage_without_changing_results_or_work() {
    let (g, high) = sparse_graph();
    let view = g.snapshot();
    assert_eq!(view.len, 16_385);
    // Both filters accept exactly source key zero. The compact bitvector's
    // absent high document bits are false; the padded bitvector adds zeros.
    // They legitimately have different size/count ratios, and therefore
    // choose different visited representations for exactly the same query.
    let compact_filter = VespaFilterEstimate { size: 1, count: 1 };
    let padded_filter = VespaFilterEstimate {
        size: 16_384,
        count: 1,
    };
    assert_eq!(
        view.layer_visited(0, 1, Some(FilterEstimate { size: 1, count: 1 }))
            .policy(),
        Policy::Hash
    );
    assert_eq!(
        view.layer_visited(
            0,
            1,
            Some(FilterEstimate {
                size: 16_384,
                count: 1
            })
        )
        .policy(),
        Policy::Bitmap
    );
    let hash = view
        .search_with_filter_estimate(&[0.0], 1, 1, &SparseRows, |key| key == 0, compact_filter)
        .unwrap();
    let bitmap = view
        .search_with_filter_estimate(&[0.0], 1, 1, &SparseRows, |key| key == 0, padded_filter)
        .unwrap();
    let compatibility = view.search(&[0.0], 1, 1, &SparseRows, |key| key == 0);
    assert_eq!(hash.hits, vec![(0, 1.0)]);
    assert_eq!(hash.hits, bitmap.hits);
    assert_eq!(hash.hits, compatibility.hits);
    assert_eq!(hash.vector_ops, bitmap.vector_ops);
    assert_eq!(hash.vector_ops, compatibility.vector_ops);
    assert_eq!(
        hash.missing_source_lookups + bitmap.missing_source_lookups,
        0
    );
    assert_eq!(
        hash.entry_fallback_scanned + bitmap.entry_fallback_scanned,
        0
    );
    // The only entry has an internal ID far above compact_filter.size. Its
    // traversal to accepted key zero must not be truncated by document bounds.
    assert_eq!(view.entry as u32, internal_id(high));
    assert!(internal_id(high) > compact_filter.size);
}

#[test]
fn construction_visited_degree_and_trailing_removal_use_current_node_bound() {
    let (g, high) = sparse_graph();
    let view = g.snapshot();
    // At limit/128 = 128, level zero estimates 2*14+100 = 128,
    // while an upper level estimates 1*14+100 = 114.
    let mut lower = view.layer_visited(0, 14, None);
    let mut upper = view.layer_visited(1, 14, None);
    assert_eq!(lower.policy(), Policy::Bitmap);
    assert_eq!(upper.policy(), Policy::Hash);
    assert!(lower.insert(internal_id(high)));
    assert!(
        upper.insert(internal_id(high)),
        "each layer needs fresh marks"
    );
    assert!(!lower.insert(internal_id(high)));
    assert!(lower.storage_usage().bitmap_bytes > 2_000);
    drop(lower);
    drop(upper);
    drop(view);
    g.remove(high, &SparseRows).unwrap();
    let smaller = g.snapshot();
    assert_eq!(
        g.num_nodes(),
        16_384,
        "peak allocator slots remain allocated"
    );
    assert_eq!(
        smaller.len, 2,
        "trailing unpublished IDs do not size visited storage"
    );
    let mut fresh = smaller.layer_visited(0, 14, None);
    assert_eq!(fresh.policy(), Policy::Bitmap);
    assert_eq!(fresh.storage_usage().bitmap_bytes, 8);
    assert!(fresh.insert(1));
    assert!(!fresh.insert(internal_id(high)));
}

#[test]
fn plain_visited_state_unwinds_and_restarts_cleanly_after_predicate_panic() {
    let g = MutableHnsw::new(1, Similarity::L2, HnswParams::default());
    g.insert(0, &SparseRows).unwrap();
    let estimate = VespaFilterEstimate { size: 1, count: 1 };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            g.search_with_filter_estimate(
                &[0.0],
                1,
                1,
                &SparseRows,
                |_| panic!("intentional predicate panic"),
                estimate,
            )
        }))
        .is_err()
    );
    // The tracker is owned by the unwound layer stack, not TLS. The remaining
    // scratch heaps also release every guard-bound reference on unwind.
    MUTABLE_SCRATCH.with(|scratch| {
        let scratch = scratch.borrow();
        assert!(scratch.cand.is_empty() && scratch.top.is_empty());
        assert!(scratch.out.is_empty() && scratch.selected.is_empty());
    });
    for _ in 0..2 {
        let result = g
            .search_with_filter_estimate(&[0.0], 1, 1, &SparseRows, |_| true, estimate)
            .unwrap();
        assert_eq!(result.hits, vec![(0, 1.0)]);
        assert_eq!(result.missing_source_lookups, 0);
    }
}

#[test]
fn explicit_plain_filter_statistics_validate_and_do_not_count_inside_query() {
    struct NoReads;
    impl MutableVectorSource for NoReads {
        fn get(&self, _: u64) -> Option<&[f32]> {
            panic!("invalid statistics must fail before source access")
        }
    }
    let g = MutableHnsw::new(1, Similarity::L2, HnswParams::default());
    g.insert(0, &SparseRows).unwrap();
    assert!(
        g.search_with_filter_estimate(
            &[0.0],
            1,
            1,
            &NoReads,
            |_| true,
            VespaFilterEstimate { size: 0, count: 1 }
        )
        .is_err()
    );
    let view = g.snapshot();
    assert!(
        view.search_with_filter_estimate(
            &[0.0],
            1,
            1,
            &NoReads,
            |_| true,
            VespaFilterEstimate { size: 1, count: 2 }
        )
        .is_err()
    );
    let empty = view
        .search_with_filter_estimate(
            &[0.0],
            1,
            1,
            &SparseRows,
            |_| false,
            VespaFilterEstimate { size: 0, count: 0 },
        )
        .unwrap();
    assert!(empty.hits.is_empty());
    assert_eq!(empty.missing_source_lookups, 0);
    assert!(
        view.search_with_filter_estimate(
            &[0.0],
            0,
            1,
            &NoReads,
            |_| true,
            VespaFilterEstimate { size: 1, count: 1 }
        )
        .unwrap()
        .hits
        .is_empty()
    );
}

#[test]
fn explicit_filter_document_domain_survives_node_id_reuse() {
    struct Source;
    impl MutableVectorSource for Source {
        fn get(&self, key: u64) -> Option<&[f32]> {
            match key {
                7 => Some(&[0.0]),
                1_000_000 => Some(&[1.0]),
                _ => None,
            }
        }
    }
    let g = MutableHnsw::new(1, Similarity::L2, HnswParams::default());
    let old = g.insert(7, &Source).unwrap();
    g.remove(old, &Source).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let token = loop {
        if let Some(token) = g.collect_reclaimed().pop() {
            break token;
        }
        assert!(Instant::now() < deadline, "source token did not drain");
        std::thread::yield_now();
    };
    assert_eq!(token.handle(), old);
    drop(token);
    let new = g.insert(1_000_000, &Source).unwrap();
    assert_eq!(new.slot(), old.slot());
    assert_ne!(new.generation(), old.generation());
    let view = g.snapshot();
    assert_eq!(view.len, 2);
    let result = view
        .search_with_filter_estimate(
            &[1.0],
            1,
            4,
            &Source,
            |key| key == 1_000_000,
            VespaFilterEstimate {
                size: 1_000_001,
                count: 1,
            },
        )
        .unwrap();
    assert_eq!(result.hits, vec![(1_000_000, 1.0)]);
    assert_eq!(result.missing_source_lookups, 0);
}
