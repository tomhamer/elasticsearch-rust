//! Causal checks for pinned Vespa completion/reconnection/Plain cutoff loops.
//! Pin: 0521b00640ef934b359d2fc356f97d84499dd17d, hnsw_index.cpp.
use super::*;
use std::cmp::Ordering;

struct Rows(Vec<Vec<f32>>);
impl MutableVectorSource for Rows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.0.get(key as usize).map(Vec::as_slice)
    }
}
fn graph(dim: u32, m: usize) -> MutableHnsw {
    MutableHnsw::new(
        dim,
        Similarity::L2,
        HnswParams {
            m,
            ef_construction: 16,
            threads: 1,
            ..HnswParams::default()
        },
    )
}
fn install(g: &MutableHnsw, key: u64) -> NodeHandle {
    let mut permit = g.reserve_insert(key).unwrap();
    let id = permit.handle();
    g.test_install_node(id, key, 0.0, vec![vec![]]).unwrap();
    permit.pending = false;
    id
}
fn save(view: &MutableHnswSnapshot, handle: NodeHandle) -> SavedNeighbor {
    SavedNeighbor {
        handle,
        levels: view.node(handle).unwrap().levels,
    }
}
fn set_links(g: &MutableHnsw, ids: &[NodeHandle], adjacency: &[&[usize]]) {
    for (index, links) in adjacency.iter().enumerate() {
        g.test_replace_links(
            ids[index],
            0,
            &links
                .iter()
                .map(|&i| internal_id(ids[i]))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
fn completion_preserves_real_nearest_first_proposal_not_handle_order() {
    let rows = Rows(vec![vec![0.0, 2.0], vec![1.0, 0.0], vec![0.0, 0.0]]);
    let mut g = graph(2, 2);
    g.min_size_before_two_phase = 0;
    let a = install(&g, 0); // farther from the incoming row, smaller handle
    let b = install(&g, 1); // nearer, larger handle
    set_links(&g, &[a, b], &[&[1], &[0]]);
    g.test_set_entry(a).unwrap();
    let prepared = g.prepare(g.reserve_insert(2).unwrap(), &rows).unwrap();
    let PreparedConnections::Neighbors(ref connections) = prepared.preparation else {
        panic!("expected actual prepared neighbors");
    };
    assert_eq!(
        connections[0].iter().map(|n| n.handle).collect::<Vec<_>>(),
        [b, a]
    );
    let inserted = g.complete(prepared, &rows).unwrap();
    let view = g.snapshot();
    assert_eq!(
        view.node(inserted).unwrap().links(0),
        Some([internal_id(b), internal_id(a)].as_slice())
    );
    assert_eq!(
        view.node(a).unwrap().links(0),
        Some([internal_id(b), internal_id(inserted)].as_slice())
    );
    assert_eq!(
        view.node(b).unwrap().links(0),
        Some([internal_id(a), internal_id(inserted)].as_slice())
    );
}

#[test]
fn proposal_filter_keeps_survivor_order_while_rejecting_removed_or_moved_refs() {
    let rows = Rows((0..4).map(|i| vec![i as f32]).collect());
    let g = graph(1, 4);
    let ids: Vec<_> = (0..4).map(|key| install(&g, key)).collect();
    let held = g.snapshot();
    let mut connections = vec![
        [3, 1, 2, 0].map(|i| save(&held, ids[i])).to_vec(),
        vec![save(&held, ids[3])], // this node has no upper level
    ];
    assert!(g.remove(ids[1], &rows).unwrap());
    g.test_move_levels(ids[2]).unwrap(); // same identity, different LevelsRef
    struct Survivors<'a>(&'a Rows);
    impl MutableVectorSource for Survivors<'_> {
        fn get(&self, key: u64) -> Option<&[f32]> {
            assert!(key == 0 || key == 3, "stale proposal read a source row");
            self.0.get(key)
        }
    }
    let current = g.snapshot();
    g.filter_proposal(&current, &Survivors(&rows), &mut connections)
        .unwrap();
    assert_eq!(
        connections[0].iter().map(|n| n.handle).collect::<Vec<_>>(),
        [ids[3], ids[0]]
    );
    assert!(connections[1].is_empty());
    assert!(
        g.collect_reclaimed().is_empty(),
        "held proposal delays reuse"
    );
}

#[test]
fn reconnect_pair_comparison_has_no_handle_or_signed_zero_tiebreaker() {
    let low = NodeHandle::new(0, 1);
    let high = NodeHandle::new(19, 7);
    let a = (0.0, high, low);
    let b = (-0.0, low, high);
    assert_eq!(reconnect_pair_cmp(&a, &b), Ordering::Equal);
    assert_eq!(reconnect_pair_cmp(&b, &a), Ordering::Equal);
    assert_eq!(
        reconnect_pair_cmp(&(2.0, high, low), &(2.0, low, high)),
        Ordering::Equal
    );
    assert_eq!(reconnect_pair_cmp(&(-1.0, high, low), &b), Ordering::Less);
    assert_eq!(reconnect_pair_cmp(&(1.0, low, high), &a), Ordering::Greater);
    let mut pairs = [(1.0, low, high), a, b, (-1.0, high, low)];
    pairs.sort_unstable_by(reconnect_pair_cmp);
    assert_eq!(pairs[0].0, -1.0);
    assert_eq!(pairs[3].0, 1.0);
    assert_eq!(reconnect_pair_cmp(&pairs[1], &pairs[2]), Ordering::Equal);
    // Equivalent pairs deliberately have no specified Rust/C++ tied order.
}

#[test]
fn reconnect_uses_nearest_pairs_capacity_and_isolated_node_exception() {
    let rows = Rows([0.0, 1.0, 3.0, 10.0].map(|x| vec![x]).to_vec());
    let g = graph(1, 1);
    let ids: Vec<_> = (0..4).map(|key| install(&g, key)).collect();
    let view = g.snapshot();
    let mut overlay = AdjacencyOverlay::new(&view, &rows, Similarity::L2, 1, None);
    // Exercise reconnect directly on an isolated cluster in reverse ID order.
    // Distinct pair distances fix the policy's order without a tie-layout claim.
    overlay
        .reconnect(&[ids[3], ids[2], ids[1], ids[0]], 0)
        .unwrap();
    let expected = [vec![1], vec![0, 2], vec![1, 3], vec![2]];
    for (i, neighbors) in expected.iter().enumerate() {
        assert_eq!(
            overlay.links(ids[i], 0).unwrap(),
            &neighbors
                .iter()
                .map(|&j| internal_id(ids[j]))
                .collect::<Vec<_>>()
        );
    }
    // B and C each get a second edge only because the other endpoint is still
    // isolated. The later A-C pair is skipped; no endpoint exceeds 2*m.
}

#[test]
fn delete_repair_keeps_an_edge_across_a_node_whose_pair_was_taken_by_a_doomed_neighbor() {
    // LL(-1) - L(0) - X(1) - {R(2), C(3.9)}: X is deleted first, C (a
    // document deleted in the same batch, still live in the graph) next.
    // Removing X: R is isolated, so the pinned pass pairs R with its
    // nearest cluster node C (3.61 < 4) and then skips L-R (R already got
    // its edge) and L-C. L kept only LL, so nothing crossed X's position
    // but R-C; removing C then left R unreachable from the left side.
    let rows = Rows(vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0], vec![3.9]]);
    let g = graph(1, 2);
    let ids: Vec<_> = (0..5).map(|key| install(&g, key)).collect();
    let (ll, l, x, r, c) = (0, 1, 2, 3, 4);
    set_links(&g, &ids, &[&[l], &[ll, x], &[l, r, c], &[x], &[x]]);
    g.test_set_entry(ids[ll]).unwrap();
    assert!(g.remove(ids[x], &rows).unwrap());
    let view = g.snapshot();
    let links = |i: usize| view.node(ids[i]).unwrap().links(0).unwrap().to_vec();
    assert!(links(l).contains(&internal_id(ids[r])), "L gained no edge across X: {:?}", links(l));
    assert!(links(r).contains(&internal_id(ids[l])));
    assert!(links(r).contains(&internal_id(ids[c])), "the pinned nearest pair is still added first");
    drop(view);
    assert!(g.remove(ids[c], &rows).unwrap());
    // From the left end, the right side is still reachable.
    let result = g.search(&[2.0], 1, 4, &rows, |_| true);
    assert_eq!(result.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![r as u64]);
}

#[test]
fn plain_full_seed_traverses_equal_or_filtered_bridge_before_first_overflow() {
    for (bridge, filter_bridge) in [(-1.0, false), (2.0, true)] {
        let rows = Rows(vec![vec![1.0], vec![bridge], vec![0.0]]);
        let g = graph(1, 2);
        let ids: Vec<_> = (0..3).map(|key| install(&g, key)).collect();
        set_links(&g, &ids, &[&[1], &[0, 2], &[1]]);
        g.test_set_entry(ids[0]).unwrap();
        let result = g.search(&[0.0], 1, 1, &rows, |key| !filter_bridge || key != 1);
        assert_eq!(result.hits, vec![(2, 1.0)]);
        assert_eq!(result.vector_ops, 4); // seed, bridge, exact target, final score
        assert_eq!(result.missing_source_lookups, 0);
        assert_eq!(result.entry_fallback_scanned, 0);
    }
}

#[test]
fn plain_overflow_cutoff_stays_strict_across_four_row_batches() {
    let rows = Rows(
        [1.0, -1.0, 1.0, 2.0, 2.0, -1.0, 0.0]
            .map(|x| vec![x])
            .to_vec(),
    );
    let g = graph(1, 3);
    let ids: Vec<_> = (0..7).map(|key| install(&g, key)).collect();
    set_links(
        &g,
        &ids,
        &[&[1, 2, 3, 4, 5], &[0], &[0], &[0], &[0], &[0, 6], &[5]],
    );
    g.test_set_entry(ids[0]).unwrap();
    struct NoExactTarget<'a>(&'a Rows);
    impl MutableVectorSource for NoExactTarget<'_> {
        fn get(&self, key: u64) -> Option<&[f32]> {
            assert_ne!(
                key, 6,
                "post-overflow equal bridge was incorrectly expanded"
            );
            self.0.get(key)
        }
    }
    let result = g.search(&[0.0], 1, 1, &NoExactTarget(&rows), |_| true);
    assert_eq!(result.hits.len(), 1);
    assert!(
        result.hits[0].0 <= 1,
        "only seed or first equal neighbor may survive"
    );
    assert_eq!(result.hits[0].1, 0.5);
    assert_eq!(result.vector_ops, 7); // seed + five neighbors + final score
    assert_eq!(result.missing_source_lookups, 0);
    assert_eq!(result.entry_fallback_scanned, 0);
}

#[test]
fn final_score_operations_include_candidates_discarded_by_top_k() {
    let rows = Rows(vec![vec![0.0], vec![1.0], vec![2.0]]);
    let g = graph(1, 2);
    let ids: Vec<_> = (0..3).map(|key| install(&g, key)).collect();
    set_links(&g, &ids, &[&[1, 2], &[0, 2], &[0, 1]]);
    g.test_set_entry(ids[0]).unwrap();
    let result = g.search(&[0.0], 2, 3, &rows, |_| true);
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.final_score_ops, 3);
    assert!(result.vector_ops >= result.final_score_ops);
}
