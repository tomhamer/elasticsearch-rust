use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::AcqRel};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// A checkpoint captured with an admission in flight leaves that key out
/// while its watermark covers later keys that completed first: the open's
/// catch-up backfills the missing key below the watermark. Ordinary
/// reservations still must increase, a backfill of a reserved key fails,
/// and the watermark stays the highest key issued.
#[test]
fn backfill_reserves_a_missing_key_below_the_watermark() {
    let source = rows(8, 4);
    let graph = MutableHnsw::new(4, Similarity::L2, HnswParams::default());
    for key in [0u64, 1, 2, 5] { graph.insert(key, &source).unwrap(); }
    let err = graph.reserve_insert(3).err().expect("ordinary keys must increase").to_string();
    assert!(err.contains("monotonically"), "{err}");
    let reservation = graph.reserve_insert_backfill(3).unwrap();
    assert!(graph.reserve_insert_backfill(3).is_err(), "a reserved key is not reserved twice");
    let prepared = graph.prepare(reservation, &source).unwrap();
    graph.complete(prepared, &source).unwrap();
    assert_eq!(graph.active_nodes(), 5);
    // The watermark stayed 5: 4 is still below it for ordinary inserts.
    assert!(graph.reserve_insert(4).is_err());
    graph.insert(6, &source).unwrap();
    assert_eq!(graph.active_nodes(), 6);
}

/// `complete_batch` publishes proposals in order under one writer hold. A
/// failing completion leaves the earlier ones published (in `done`), and
/// the failed and later reservations release when the caller drops them
/// (with the writer released: their drop takes it).
#[test]
fn complete_batch_publishes_in_order_and_releases_the_rest_on_failure() {
    let source = rows(16, 4);
    let g = graph_with_bootstrap(4, 2);
    for key in 0..4 { g.insert(key, &source).unwrap(); }
    let mut proposals: Vec<_> = (4..8).map(|key| prepare(&g, key, &source)).collect();
    let mut done = Vec::new();
    g.complete_batch(&mut proposals, &source, &mut done).unwrap();
    assert_eq!(done.len(), 4);
    assert_eq!(g.active_nodes(), 8);
    drop(proposals);
    assert_eq!(g.stats().pending_reservations, 0);
    invariants(&g);
    // The source lacks key 10's row: 8 and 9 publish, 10 and 11 stay pending.
    let mut proposals: Vec<_> = (8..12).map(|key| prepare(&g, key, &source)).collect();
    let partial = Rows { rows: source.rows[..10].to_vec() };
    let mut done = Vec::new();
    assert!(g.complete_batch(&mut proposals, &partial, &mut done).is_err());
    assert_eq!(done.len(), 2);
    assert_eq!(g.active_nodes(), 10);
    drop(proposals);
    assert_eq!(g.stats().pending_reservations, 0);
    invariants(&g);
    g.insert(12, &source).unwrap();
    assert_eq!(g.active_nodes(), 11);
}

#[test]
fn commit_batch_defers_maintenance_but_not_publication() {
    let source = rows(8, 8);
    let graph = MutableHnsw::new(8, Similarity::Cosine, HnswParams::default());
    let batch = graph.commit_batch().unwrap();
    let generation = graph.control.domain.current_generation();
    for key in 0..8 {
        let inner = graph.commit_batch().unwrap();
        graph.insert(key, &source).unwrap();
        drop(inner);
        assert_eq!(graph.active_nodes(), key as u32 + 1);
        assert_eq!(graph.control.domain.current_generation(), generation);
    }
    drop(batch);
    assert!(graph.control.domain.current_generation() > generation);
    assert_eq!(graph.active_nodes(), 8);
}

#[test]
fn commit_batch_unwind_drains_maintenance_and_leaves_committed_row() {
    let source = rows(2, 8);
    let graph = MutableHnsw::new(8, Similarity::Cosine, HnswParams::default());
    let generation = graph.control.domain.current_generation();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _batch = graph.commit_batch().unwrap();
        graph.insert(0, &source).unwrap();
        panic!("caller fails after publication");
    }));
    assert!(failed.is_err());
    assert_eq!(graph.active_nodes(), 1);
    assert!(graph.control.domain.current_generation() > generation);
    graph.insert(1, &source).unwrap();
    assert_eq!(graph.active_nodes(), 2);
}

#[test]
fn commit_batch_keeps_pinned_reader_backpressure() {
    let source = rows(16, 8);
    let graph = MutableHnsw::with_limits(8, Similarity::L2, HnswParams::default(),
        MutableHnswLimits { max_retired_adjacency_bytes: 1, ..Default::default() });
    graph.insert(0, &source).unwrap();
    let _reader = graph.snapshot();
    let _batch = graph.commit_batch().unwrap();
    let mut blocked = false;
    for key in 1..16 {
        if graph.insert(key, &source).is_err() {
            blocked = true;
            break;
        }
    }
    assert!(blocked, "a commit batch must not bypass pinned-reader memory limits");
    assert!(graph.stats().retired_adjacency_bytes >= 1);
}

#[test]
fn bound_mutation_distance_matches_angular_normalization_and_nonfinite_rejection() {
    for dims in [1, 3, 4, 7, 8, 15, 16, 31, 32, 96, 127, 128, 769] {
        let mut source = rows(6, dims);
        source.rows.push(vec![0.0; dims]);
        source.rows.push(vec![-0.0; dims]);
        source.rows.push(vec![f32::MIN_POSITIVE; dims]);
        source.rows.push(vec![f32::MAX; dims]);
        for metric in [Similarity::Cosine, Similarity::DotProduct, Similarity::L2, Similarity::MaxInnerProduct] {
            for left in &source.rows {
                let bound = BoundMutationDistance::new(left, metric);
                for right in &source.rows {
                    let expected = if cfg!(feature = "mutable-hnsw-vespa-angular") && metric == Similarity::Cosine {
                        let squared = f64::from(dot(left, left)) * f64::from(dot(right, right));
                        1.0 - f64::from(dot(left, right)) / if squared > 0.0 { squared.sqrt() } else { 1.0 }
                    } else { f64::from(metric.distance(left, right)) };
                    let raw = bound.distance_value(right);
                    assert!(raw.to_bits() == expected.to_bits() || (raw.is_nan() && expected.is_nan()), "raw preparation kernel dims={dims}, metric={metric:?}");
                    let actual = bound.distance(right);
                    if expected.is_finite() {
                        assert_eq!(actual.unwrap().to_bits(), expected.to_bits(), "dims={dims}, metric={metric:?}");
                    } else {
                        assert!(actual.is_err());
                    }
                }
            }
        }
    }
}

#[test]
fn mutation_pruning_resolves_its_owner_row_once() {
    struct CountedRows { source: Rows, reads: Vec<AtomicUsize> }
    impl MutableVectorSource for CountedRows {
        fn get(&self, key: u64) -> Option<&[f32]> {
            self.reads.get(key as usize)?.fetch_add(1, Relaxed);
            self.source.get(key)
        }
    }
    let source = CountedRows { source: rows(6, 96), reads: (0..6).map(|_| AtomicUsize::new(0)).collect() };
    let graph = MutableHnsw::new(96, Similarity::Cosine, HnswParams { m: 2, ..Default::default() });
    let handles: Vec<_> = (0..6).map(|key| graph.insert(key, &source).unwrap()).collect();
    let _writer = graph.control.writer.lock();
    let view = graph.snapshot();
    let mut overlay = AdjacencyOverlay::new(&view, &source, Similarity::Cosine, 2, None);
    overlay.set(handles[0], 0, handles[1..].iter().copied().map(internal_id).collect());
    for reads in &source.reads { reads.store(0, Relaxed); }
    overlay.shrink(handles[0], 0).unwrap();
    assert_eq!(source.reads[0].load(Relaxed), 1);
    assert!(overlay.links(handles[0], 0).unwrap().len() <= 4);
}

struct Rows {
    rows: Vec<Vec<f32>>,
}
impl MutableVectorSource for Rows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.rows.get(key as usize).map(Vec::as_slice)
    }
}
fn rows(n: usize, dim: usize) -> Rows {
    let mut rng = XorShift64Star::new(0x4873_AB79);
    Rows {
        rows: (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                    .collect()
            })
            .collect(),
    }
}
fn graph(dim: usize) -> MutableHnsw {
    graph_with_bootstrap(dim, 32)
}
fn graph_with_bootstrap(dim: usize, threshold: u32) -> MutableHnsw {
    // Small private threshold exercises both modes in bounded lifecycle
    // fixtures. Production callers always get the pinned default 10,000.
    let mut graph = MutableHnsw::new(dim as u32, Similarity::L2, HnswParams::default());
    graph.min_size_before_two_phase = threshold;
    graph
}
fn exact(source: &Rows, query: &[f32], k: usize, accept: impl Fn(u64) -> bool) -> Vec<(u64, f32)> {
    let mut out: Vec<_> = source
        .rows
        .iter()
        .enumerate()
        .filter(|(key, _)| accept(*key as u64))
        .map(|(key, row)| (key as u64, Similarity::L2.score(query, row)))
        .collect();
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out
}
fn entry_handle(graph: &MutableHnsw) -> NodeHandle {
    let view = graph.snapshot();
    view.current_node(view.entry as u32)
        .expect("fixture has a current entry")
        .handle()
}
fn invariants(graph: &MutableHnsw) {
    let view = graph.snapshot();
    let mut active = 0;
    for id in view.handles() {
        let node = view.node(id).unwrap();
        assert!(view.visible(node));
        active += 1;
        assert_eq!(
            id.generation(),
            view.control
                .nodes
                .read(&view.guard)
                .unwrap()
                .get(internal_id(id))
                .unwrap()
                .metadata()
                .incarnation
        );
        for level in 0..=node.level() {
            let neighbors = node.links(level).unwrap();
            assert!(
                neighbors.len()
                    <= if level == 0 {
                        graph.params.m * 2
                    } else {
                        graph.params.m
                    }
            );
            for &neighbor in neighbors {
                assert_ne!(internal_id(id), neighbor);
                let other = view.current_node(neighbor).unwrap();
                assert!(view.live(other.handle()));
                assert!(other.level() >= level);
                assert!(
                    other.links(level).unwrap().contains(&internal_id(id)),
                    "{id:?}->{neighbor:?} is not reciprocal"
                );
                assert_eq!(neighbors.iter().filter(|&&n| n == neighbor).count(), 1);
            }
        }
    }
    assert_eq!(graph.active_nodes(), active);
    if active == 0 {
        assert_eq!(view.entry, NO_ENTRY);
    } else if view.entry != NO_ENTRY {
        // Removing an isolated entry may deliberately leave a disconnected
        // graph without an entry. A present packed entry must be valid.
        let entry = view.current_node(view.entry as u32).unwrap();
        assert!(view.live(entry.handle()));
        assert!(entry.level() >= (view.entry >> 32) as usize);
    }
}
fn reclaimed(g: &MutableHnsw, expected: usize) -> Vec<ReclaimedNode> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut tokens = Vec::new();
    while tokens.len() < expected && Instant::now() < deadline {
        tokens.extend(g.collect_reclaimed());
        std::thread::yield_now();
    }
    assert_eq!(
        tokens.len(),
        expected,
        "generation reclamation did not drain"
    );
    tokens
}
fn prepare(g: &MutableHnsw, key: u64, source: &Rows) -> PreparedMutableInsert {
    g.prepare(g.reserve_insert(key).unwrap(), source).unwrap()
}
struct CountedRows<'a> {
    rows: &'a Rows,
    reads: Cell<usize>,
}
impl MutableVectorSource for CountedRows<'_> {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.reads.set(self.reads.get() + 1);
        self.rows.get(key)
    }
}

#[test]
fn mutable_bootstrap_marker_survives_crossing_threshold_without_prepare_draw_or_search() {
    assert_eq!(
        MutableHnsw::new(4, Similarity::L2, HnswParams::default()).min_size_before_two_phase,
        10_000
    );
    let source = rows(4, 4);
    let g = graph_with_bootstrap(4, 2);
    g.insert(0, &source).unwrap();
    let traced = CountedRows {
        rows: &source,
        reads: Cell::new(0),
    };
    let before_draws = g.level_generator.draws();
    let marker = g.prepare(g.reserve_insert(1).unwrap(), &traced).unwrap();
    assert!(marker.is_bootstrap());
    assert_eq!(g.level_generator.draws(), before_draws);
    assert_eq!(
        traced.reads.get(),
        1,
        "only the incoming row may be validated"
    );
    assert_eq!(marker.estimated_bytes(), std::mem::size_of_val(&marker));

    g.insert(2, &source).unwrap(); // Cross threshold while the marker waits.
    assert_eq!(g.active_nodes(), 2);
    let before_draws = g.level_generator.draws();
    let before_reads = traced.reads.get();
    let id = g.complete(marker, &traced).unwrap();
    assert_eq!(g.level_generator.draws(), before_draws + 1);
    assert!(
        traced.reads.get() > before_reads + 1,
        "marker completion did not search"
    );
    let view = g.snapshot();
    assert!(!view.node(id).unwrap().links(0).unwrap().is_empty());
    invariants(&g);
}

#[test]
fn mutable_empty_graph_marker_burst_prepares_sequentially_at_completion() {
    let source = rows(8, 4);
    let g = graph_with_bootstrap(4, 2);
    let traced = CountedRows {
        rows: &source,
        reads: Cell::new(0),
    };
    let markers: Vec<_> = (0..8)
        .map(|key| g.prepare(g.reserve_insert(key).unwrap(), &traced).unwrap())
        .collect();
    assert!(markers.iter().all(PreparedMutableInsert::is_bootstrap));
    assert_eq!(g.level_generator.draws(), 0);
    assert_eq!(traced.reads.get(), 8, "only incoming rows may be validated");
    for (i, marker) in markers.into_iter().enumerate() {
        let before_reads = traced.reads.get();
        let id = g.complete(marker, &traced).unwrap();
        assert_eq!(g.level_generator.draws(), i as u64 + 1);
        let view = g.snapshot();
        if i > 0 {
            assert!(traced.reads.get() > before_reads + 1);
            assert!(!view.node(id).unwrap().links(0).unwrap().is_empty());
        }
        invariants(&g);
    }
    assert_eq!(g.active_nodes(), 8);
}

#[test]
fn mutable_prepared_proposal_stays_two_phase_after_active_count_drops() {
    let source = rows(5, 4);
    let g = graph_with_bootstrap(4, 2);
    let old: Vec<_> = (0..4).map(|key| g.insert(key, &source).unwrap()).collect();
    let proposal = prepare(&g, 4, &source);
    let PreparedConnections::Neighbors(ref connections) = proposal.preparation else {
        panic!("expected prepared neighbors above threshold");
    };
    let level = connections.len() - 1;
    let draws = g.level_generator.draws();
    for &id in &old[..3] {
        g.remove(id, &source).unwrap();
    }
    assert_eq!(g.active_nodes(), 1);
    let id = g.complete(proposal, &source).unwrap();
    assert_eq!(g.level_generator.draws(), draws, "completion redrew level");
    assert_eq!(g.snapshot().node(id).unwrap().level(), level);
    invariants(&g);
}

#[test]
fn mutable_all_stale_prepared_neighbors_are_not_reprepared() {
    let source = rows(6, 4);
    let g = graph_with_bootstrap(4, 2);
    let old: Vec<_> = (0..4).map(|key| g.insert(key, &source).unwrap()).collect();
    let proposal = prepare(&g, 4, &source);
    let PreparedConnections::Neighbors(ref connections) = proposal.preparation else {
        panic!("expected prepared neighbors");
    };
    assert!(!connections[0].is_empty());
    for id in old {
        g.remove(id, &source).unwrap();
    }
    // Keep a new graph node present: a generic all-stale retry would connect
    // the old proposal to this node and consume another level draw.
    g.insert(5, &source).unwrap();
    let draws = g.level_generator.draws();
    let traced = CountedRows {
        rows: &source,
        reads: Cell::new(0),
    };
    let id = g.complete(proposal, &traced).unwrap();
    assert_eq!(g.level_generator.draws(), draws);
    assert_eq!(traced.reads.get(), 1, "all-stale completion searched again");
    let view = g.snapshot();
    let node = view.node(id).unwrap();
    assert!((0..=node.level()).all(|level| node.links(level).unwrap().is_empty()));
    assert_eq!(node.key(), 4);
    invariants(&g);
}

#[test]
fn mutable_live_neighbor_missing_from_source_is_an_error_in_both_prepare_modes() {
    struct IncomingOnly<'a> {
        key: u64,
        row: &'a [f32],
    }
    impl MutableVectorSource for IncomingOnly<'_> {
        fn get(&self, key: u64) -> Option<&[f32]> {
            (key == self.key).then_some(self.row)
        }
    }
    let source = rows(4, 4);
    for existing in [1u64, 2] {
        let g = graph_with_bootstrap(4, 2);
        for key in 0..existing {
            g.insert(key, &source).unwrap();
        }
        let proposal = prepare(&g, existing, &source);
        assert_eq!(proposal.is_bootstrap(), existing < 2);
        let missing = IncomingOnly {
            key: existing,
            row: &source.rows[existing as usize],
        };
        assert!(g.complete(proposal, &missing).is_err());
        assert_eq!(g.active_nodes(), existing as u32);
        assert_eq!(g.stats().pending_reservations, 0);
        invariants(&g);

        if existing == 2 {
            let missing = IncomingOnly {
                key: 3,
                row: &source.rows[3],
            };
            assert!(g.prepare(g.reserve_insert(3).unwrap(), &missing).is_err());
            assert_eq!(g.active_nodes(), 2);
            assert_eq!(g.stats().pending_reservations, 0);
        }
    }
}

#[test]
fn mutable_fixed_level_stream_ignores_legacy_seed_and_preserves_codec_metadata() {
    let source = rows(33, 4);
    let make = |seed| {
        MutableHnsw::new(
            4,
            Similarity::L2,
            HnswParams {
                seed,
                ..HnswParams::default()
            },
        )
    };
    let a = make(1);
    let b = make(9_876_543);
    for key in 0..32 {
        let left = a.insert(key, &source).unwrap();
        let right = b.insert(key, &source).unwrap();
        assert_eq!(
            a.snapshot().node(left).unwrap().level(),
            b.snapshot().node(right).unwrap().level()
        );
    }
    assert_eq!(a.level_generator.draws(), 32);
    assert_eq!(b.level_generator.draws(), 32);
    let bytes = b.encode();
    let reopened = MutableHnsw::open(&bytes).unwrap();
    assert_eq!(reopened.params.seed, 9_876_543);
    assert_eq!(reopened.encode(), bytes);
    assert_eq!(reopened.level_generator.draws(), 0);
    assert_eq!(
        reopened
            .search(&source.rows[7], 10, 32, &source, |_| true)
            .hits,
        b.search(&source.rows[7], 10, 32, &source, |_| true).hits
    );
    let expected_level = level_generator::VespaLevelGenerator::new(reopened.params.m as u32)
        .unwrap()
        .next_level()
        .unwrap();
    let new = reopened.insert(32, &source).unwrap();
    assert_eq!(reopened.level_generator.draws(), 1);
    assert_eq!(
        reopened.snapshot().node(new).unwrap().level(),
        expected_level as usize
    );
}

#[test]
fn mutable_construction_carries_all_upper_level_seeds_after_diversification() {
    let source = Rows {
        rows: vec![vec![1.0], vec![2.0], vec![0.1]],
    };
    let g = MutableHnsw::new(
        1,
        Similarity::L2,
        HnswParams {
            m: 2,
            ef_construction: 2,
            ..HnswParams::default()
        },
    );
    // Upper A--B, lower B--C, with A isolated on the lower level. A is
    // nearest at the upper level; reaching C requires carrying B as well.
    // This is a reciprocal topology fixture, not an implementation mirror.
    let mut ids = Vec::new();
    for key in 0..3 {
        let mut permit = g.reserve_insert(key).unwrap();
        let id = permit.handle();
        g.test_install_node(id, key, 0.0, vec![Vec::new(), Vec::new()])
            .unwrap();
        permit.pending = false;
        ids.push(id);
    }
    g.test_set_entry(ids[0]).unwrap();
    g.test_replace_links(ids[0], 1, &[internal_id(ids[1])])
        .unwrap();
    g.test_replace_links(ids[1], 1, &[internal_id(ids[0])])
        .unwrap();
    g.test_replace_links(ids[1], 0, &[internal_id(ids[2])])
        .unwrap();
    g.test_replace_links(ids[2], 0, &[internal_id(ids[1])])
        .unwrap();
    let view = g.snapshot();
    let d = SourceDistance::new(&view, &source, &[0.0]);
    let mut scratch = MutableScratch::default();
    scratch
        .top
        .push(Candidate::new(d.dist(ids[0]), view.node(ids[0]).unwrap()));
    view.construction_beam(&d, 1, 2, &mut scratch);
    assert_eq!(
        scratch.out.iter().map(|c| c.id).collect::<Vec<_>>(),
        ids[..2]
            .iter()
            .copied()
            .map(internal_id)
            .collect::<Vec<_>>()
    );
    select_diverse(
        &view,
        &source,
        Similarity::L2,
        &scratch.out,
        None,
        1,
        &mut scratch.selected,
    );
    assert_eq!(scratch.selected.len(), 1);
    assert_eq!(scratch.selected[0].id, internal_id(ids[0]));
    assert_eq!(
        scratch.top.len(),
        2,
        "edge selection must not consume the beam"
    );
    view.construction_beam(&d, 0, 2, &mut scratch);
    assert_eq!(scratch.out[0].id, internal_id(ids[2]));
    let mut reset = MutableScratch::default();
    view.beam(&d, ids[0], d.dist(ids[0]), 0, 2, &|_, _| true, &mut reset);
    assert!(
        !reset.out.iter().any(|c| c.id == internal_id(ids[2])),
        "single-seed control cannot reach C"
    );
    let proposal = (0..128)
        .find_map(|_| {
            let proposal = g.prepare_connections(&[0.0], (ids[0], 0), &source).unwrap();
            (proposal.len() == 2).then_some(proposal)
        })
        .expect("fixed m=2 stream must include a level-one sample");
    assert_eq!(proposal.len(), 2);
    assert!(
        proposal[0].iter().any(|saved| saved.handle == ids[2]),
        "actual preparation must carry the full beam"
    );
}
#[test]
fn mutable_construction_bound_tightens_only_after_overflow() {
    let source = Rows {
        rows: vec![vec![1.0], vec![1.0], vec![0.0]],
    };
    let g = graph(1);
    let ids: Vec<_> = (0..3).map(|key| g.insert(key, &source).unwrap()).collect();
    g.test_replace_links(ids[0], 0, &[internal_id(ids[1])])
        .unwrap();
    g.test_replace_links(ids[1], 0, &[internal_id(ids[2])])
        .unwrap();
    g.test_replace_links(ids[2], 0, &[]).unwrap();
    let view = g.snapshot();
    let d = SourceDistance::new(&view, &source, &[0.0]);
    let mut scratch = MutableScratch::default();
    scratch
        .top
        .push(Candidate::new(d.dist(ids[0]), view.node(ids[0]).unwrap()));
    // ef=1 is already full; the equally distant bridge still enters before
    // the first overflow establishes the bound, exposing the exact match.
    view.construction_beam(&d, 0, 1, &mut scratch);
    assert_eq!(scratch.out[0].id, internal_id(ids[2]));
    assert_eq!(scratch.top.len(), 1);
}
#[test]
fn mutable_construction_specializations_preserve_candidates_and_distances() {
    use super::visited::{BitmapVisited, HashVisited, LayerVisited};
    let source = rows(192, 12);
    let g = graph(12);
    let ids: Vec<_> = (0..192).map(|key| g.insert(key, &source).unwrap()).collect();
    let view = g.snapshot();
    for query in [0, 17, 191] {
        let d = SourceDistance::new(&view, &source, &source.rows[query]);
        for ef in [1, 16, 192] {
            let mut bitmap = BitmapVisited::default();
            let mut hash = HashVisited::default();
            bitmap.reset(view.len, view.len);
            hash.reset(view.len, view.len);
            let mut a = MutableScratch::default();
            let mut b = MutableScratch::default();
            let entry = Candidate::new(d.dist(ids[0]), view.node(ids[0]).unwrap());
            a.top.push(entry);
            b.top.push(entry);
            view.construction_beam_typed(&d, 0, ef, &mut a, &mut bitmap);
            view.construction_beam_typed(&d, 0, ef, &mut b, &mut hash);
            let signature = |s: &MutableScratch| s.out.iter()
                .map(|c| (c.id, c.distance.to_bits())).collect::<Vec<_>>();
            assert_eq!(signature(&a), signature(&b));
        }
    }
}
#[test]
fn mutable_incremental_publication_matches_exact_and_preserves_filters() {
    let source = rows(192, 12);
    let g = graph(12);
    for batch in 0..6 {
        for key in batch * 32..(batch + 1) * 32 {
            g.insert(key, &source).unwrap();
        }
        let limit = (batch + 1) * 32;
        for q in [0, 17, limit as usize - 1] {
            let got = g.search(&source.rows[q], 10, 192, &source, |key| {
                key < limit && key % 3 != 1
            });
            assert_eq!(
                got.hits,
                exact(&source, &source.rows[q], 10, |key| key < limit
                    && key % 3 != 1)
            );
            assert!(got.vector_ops > 0);
        }
        invariants(&g);
    }
}
#[test]
fn mutable_failed_rows_and_wrong_graph_proposals_do_not_publish_nodes() {
    let source = rows(16, 4);
    let g = graph(4);
    let wrong_dimension = Rows {
        rows: vec![vec![1.0; 3]],
    };
    assert!(g.insert(0, &wrong_dimension).is_err());
    let nan = Rows {
        rows: vec![vec![0.0; 4], vec![f32::NAN; 4]],
    };
    assert!(g.insert(1, &nan).is_err());
    let proposal = prepare(&g, 2, &source);
    let other = graph(4);
    assert!(other.complete(proposal, &source).is_err());
    let reservation = g.reserve_insert(3).unwrap();
    assert!(other.prepare(reservation, &source).is_err());
    let proposal = prepare(&g, 4, &source);
    let missing = Rows { rows: Vec::new() };
    assert!(g.complete(proposal, &missing).is_err());
    assert_eq!(g.active_nodes(), 0);
    assert_eq!(g.stats().pending_reservations, 0);
    drop(reclaimed(&g, 5));
    assert_eq!(g.stats().retired_slots, 0);
    assert!(g.reserve_insert(4).is_err());
    g.insert(5, &source).unwrap();
    assert_eq!(
        g.search(&source.rows[5], 1, 10, &source, |_| true).hits[0].0,
        5
    );
}
#[test]
fn mutable_old_view_bounds_future_edges_and_accept_controls_update_visibility() {
    let source = rows(160, 8);
    let g = graph(8);
    let ids: Vec<_> = (0..80).map(|key| g.insert(key, &source).unwrap()).collect();
    // Reuse a lower ID beneath the old published bound, so the publication
    // check (rather than only the ID-bound check) must reject the new row.
    g.remove(ids[0], &source).unwrap();
    drop(reclaimed(&g, 1));
    let reserved = g.reserve_insert(80).unwrap();
    let replacement = reserved.handle();
    assert_eq!(replacement.slot(), ids[0].slot());
    let old_view = g.snapshot();
    assert!(internal_id(replacement) < old_view.len);
    g.complete(g.prepare(reserved, &source).unwrap(), &source)
        .unwrap();
    let future = old_view.current_node(internal_id(replacement)).unwrap();
    assert!(!old_view.visible(future));
    for key in 81..160 {
        g.insert(key, &source).unwrap();
    }
    assert!(
        old_view
            .search(&source.rows[82], 80, 160, &source, |_| true)
            .hits
            .iter()
            .all(|&(key, _)| key < 80)
    );
    assert_eq!(
        g.search(&source.rows[2], 1, 160, &source, |key| key < 80)
            .hits[0]
            .0,
        2
    );
    assert_eq!(
        g.search(&source.rows[82], 1, 160, &source, |key| key != 2)
            .hits[0]
            .0,
        82
    );
    invariants(&g);
}
#[test]
fn mutable_delete_repairs_reciprocal_links_and_replaces_entry_without_rebuild() {
    let source = rows(256, 12);
    let g = graph(12);
    let ids: Vec<_> = (0..256)
        .map(|key| g.insert(key, &source).unwrap())
        .collect();
    let entry = entry_handle(&g);
    let entry_key = g.snapshot().node(entry).unwrap().key();
    let builds = HnswBuilder::builds_on_this_thread();
    assert!(g.remove(entry, &source).unwrap());
    assert!(!g.remove(entry, &source).unwrap());
    let mut deleted = vec![false; 256];
    deleted[entry_key as usize] = true;
    for key in (0..256).step_by(5) {
        g.remove(ids[key], &source).unwrap();
        deleted[key] = true;
    }
    invariants(&g);
    for q in [3, 37, 179] {
        assert_eq!(
            g.search(&source.rows[q], 10, 256, &source, |_| true).hits,
            exact(&source, &source.rows[q], 10, |key| !deleted[key as usize])
        );
    }
    assert_eq!(HnswBuilder::builds_on_this_thread(), builds);
    for id in ids {
        g.remove(id, &source).unwrap();
    }
    assert!(
        g.search(&source.rows[0], 10, 256, &source, |_| true)
            .hits
            .is_empty()
    );
    invariants(&g);
    drop(reclaimed(&g, 256));
    assert_eq!(g.stats().reusable_slots, 256);
    assert_eq!(g.stats().retired_slots, 0);
}
#[test]
fn mutable_prepared_lease_delays_reuse_and_revalidates_removed_neighbors() {
    let source = rows(194, 8);
    let g = graph(8);
    let old: Vec<_> = (0..64).map(|key| g.insert(key, &source).unwrap()).collect();
    let reservations: Vec<_> = (64..=128)
        .map(|key| g.reserve_insert(key).unwrap())
        .collect();
    let mut proposals: Vec<_> = reservations
        .into_iter()
        .map(|r| g.prepare(r, &source).unwrap())
        .collect();
    let stale = proposals.pop().unwrap();
    let PreparedConnections::Neighbors(ref connections) = stale.preparation else {
        panic!("fixture must exercise a real prepared neighbor list");
    };
    assert!(!connections[0].is_empty());
    let stale_id = stale.handle();
    for &id in &old {
        g.remove(id, &source).unwrap();
    }
    assert!(g.collect_reclaimed().is_empty());
    // These earlier valid reservations must remain legal despite watermark128.
    for proposal in proposals {
        g.complete(proposal, &source).unwrap();
    }
    // The transferred stale proposal still withholds all old node IDs.
    let stale = std::thread::spawn(move || stale).join().unwrap();
    assert!(g.collect_reclaimed().is_empty());
    g.complete(stale, &source).unwrap();
    drop(reclaimed(&g, 64));
    let replacements: Vec<_> = (129..193)
        .map(|key| g.insert(key, &source).unwrap())
        .collect();
    assert!(replacements.iter().all(|new| {
        old.iter()
            .any(|id| id.slot() == new.slot() && id.generation() != new.generation())
    }));
    assert!(old.iter().all(|&id| !g.remove(id, &source).unwrap()));
    invariants(&g);
    // All proposals lost their old neighbors; pinned completion does not
    // invent a retry to reconnect them. This remains a lifetime/identity
    // test, not a recall claim for an adversarial disconnected topology.
    assert_eq!(g.snapshot().node(stale_id).unwrap().key(), 128);
}
#[test]
fn mutable_epoch_reader_retains_old_neighbor_array_and_does_not_block_commit() {
    let source = Arc::new(rows(160, 8));
    let g = Arc::new(graph(8));
    let ids: Vec<_> = (0..64)
        .map(|key| g.insert(key, source.as_ref()).unwrap())
        .collect();
    let view = g.snapshot();
    let old = view.node(ids[0]).unwrap().links(0).unwrap();
    let saved = old.to_vec();
    assert!(!saved.is_empty());
    let (tx, rx) = mpsc::channel();
    let writer_graph = g.clone();
    let writer_source = source.clone();
    let id = ids[0];
    let writer = std::thread::spawn(move || {
        writer_graph.remove(id, writer_source.as_ref()).unwrap();
        for key in 64..160 {
            writer_graph.insert(key, writer_source.as_ref()).unwrap();
        }
        tx.send(()).unwrap();
    });
    let completion = rx.recv_timeout(Duration::from_secs(10));
    assert_eq!(old, saved.as_slice());
    assert!(g.collect_reclaimed().is_empty());
    drop(view);
    writer.join().unwrap();
    completion.unwrap();
    drop(reclaimed(&g, 1));
    invariants(&g);
}

#[test]
fn mutable_query_can_pause_in_source_while_writer_removes_entry() {
    struct PausedSource {
        rows: Arc<Rows>,
        once: AtomicBool,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl MutableVectorSource for PausedSource {
        fn get(&self, key: u64) -> Option<&[f32]> {
            if !self.once.swap(true, AcqRel) {
                self.entered.send(()).unwrap();
                self.release.lock().recv().unwrap();
            }
            self.rows.get(key)
        }
    }
    let source = Arc::new(rows(96, 8));
    let g = Arc::new(graph(8));
    for key in 0..96 {
        g.insert(key, source.as_ref()).unwrap();
    }
    let entry = entry_handle(&g);
    let entry_key = g.snapshot().node(entry).unwrap().key();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let paused = PausedSource {
        rows: source.clone(),
        once: AtomicBool::new(false),
        entered: entered_tx,
        release: Mutex::new(release_rx),
    };
    let reader_graph = g.clone();
    let reader = std::thread::spawn(move || {
        reader_graph.search(&paused.rows.rows[0], 10, 96, &paused, |_| true)
    });
    let entered = entered_rx.recv_timeout(Duration::from_secs(10));
    let removed = if entered.is_ok() {
        g.remove(entry, source.as_ref())
    } else {
        Ok(false)
    };
    release_tx.send(()).unwrap();
    let result = reader.join().unwrap();
    entered.unwrap();
    assert!(removed.unwrap());
    assert!(result.hits.iter().all(|&(key, _)| key != entry_key));
    invariants(&g);
}
#[test]
fn mutable_query_traverses_acquired_levels_after_slot_invalidation() {
    struct PauseThird {
        rows: Arc<Rows>,
        calls: AtomicUsize,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl MutableVectorSource for PauseThird {
        fn get(&self, key: u64) -> Option<&[f32]> {
            if self.calls.fetch_add(1, Relaxed) == 2 {
                self.entered.send(()).unwrap();
                self.release
                    .lock()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            self.rows.get(key)
        }
    }
    let source = Arc::new(Rows {
        rows: (0..4).map(|i| vec![i as f32]).collect(),
    });
    let g = Arc::new(MutableHnsw::new(
        1,
        Similarity::L2,
        HnswParams {
            m: 1,
            ef_construction: 16,
            ..HnswParams::default()
        },
    ));
    for key in 0..4 {
        g.insert(key, source.as_ref()).unwrap();
    }
    let entry = entry_handle(&g);
    let entry_key = g.snapshot().node(entry).unwrap().key();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let paused = PauseThird {
        rows: source.clone(),
        calls: AtomicUsize::new(0),
        entered: entered_tx,
        release: Mutex::new(release_rx),
    };
    let query_graph = g.clone();
    let reader = std::thread::spawn(move || query_graph.search(&[2.0], 4, 16, &paused, |_| true));
    // Level zero only: lookup 1 resolves the entry, lookup 2 computes its
    // distance, lookup 3 acquires the first beam candidate's source row.
    let entered = entered_rx.recv_timeout(Duration::from_secs(10));
    let removed = if entered.is_ok() {
        g.remove(entry, source.as_ref())
    } else {
        Ok(false)
    };
    let _ = release_tx.send(());
    let result = reader.join().unwrap();
    entered.unwrap();
    assert!(removed.unwrap());
    assert_eq!(
        result.hits,
        exact(source.as_ref(), &[2.0], 4, |key| key != entry_key)
    );
    assert_eq!(result.missing_source_lookups, 0);
    invariants(&g);
}
/// Hand-built level-0 chain for the removal-bridge tests: rows at 1-D
/// positions, nodes installed at level zero only, links set exactly.
fn bridge_chain(positions: &[f32], adjacency: &[&[usize]]) -> (Arc<Rows>, Arc<MutableHnsw>, Vec<NodeHandle>) {
    let source = Arc::new(Rows { rows: positions.iter().map(|&p| vec![p]).collect() });
    let g = Arc::new(MutableHnsw::new(1, Similarity::L2, HnswParams { m: 2, ef_construction: 16, threads: 1, ..HnswParams::default() }));
    let ids: Vec<_> = (0..positions.len() as u64).map(|key| {
        let mut permit = g.reserve_insert(key).unwrap();
        let id = permit.handle();
        g.test_install_node(id, key, 0.0, vec![vec![]]).unwrap();
        permit.pending = false;
        id
    }).collect();
    for (index, links) in adjacency.iter().enumerate() {
        let links: Vec<u32> = links.iter().map(|&i| internal_id(ids[i])).collect();
        g.test_replace_links(ids[index], 0, &links).unwrap();
    }
    g.test_set_entry(ids[0]).unwrap();
    (source, g, ids)
}

#[test]
fn removed_node_stays_a_bridge_only_for_readers_admitted_before_removal() {
    // 0 - 1 - 2 - 3 - 4; removing 2 reconnects 1 and 3.
    let (source, g, ids) = bridge_chain(&[0.0, 1.0, 2.0, 3.0, 4.0],
        &[&[1], &[0, 2], &[1, 3], &[2, 4], &[3]]);
    let removed = internal_id(ids[2]);
    let before = g.snapshot();
    let old_links = before.node(ids[2]).unwrap().links(0).unwrap().to_vec();
    assert!(before.removed_bridge(removed).is_none(), "a live node is not a bridge");
    assert!(g.remove(ids[2], source.as_ref()).unwrap());
    assert!(before.current_node(removed).is_none());
    let bridge = before.removed_bridge(removed).expect("an older live reader keeps the removed node traversable");
    assert_eq!(bridge.links(0).unwrap(), old_links.as_slice());
    assert_eq!(bridge.key(), 2);
    // A reader admitted after the removal never sees the old links, so it
    // must not dereference the retired owner either.
    let after = g.snapshot();
    assert!(after.removed_bridge(removed).is_none());
    // The bridge is never a hit for either view.
    for view in [&before, &after] {
        let hits = view.search(&[2.0], 5, 8, source.as_ref(), |_| true).hits;
        assert!(hits.iter().all(|&(key, _)| key != 2), "{hits:?}");
    }
    drop(before);
    drop(after);
    drop(reclaimed(&g, 1));
    // Once reclaimed, the stale removal record is ignored by new readers.
    let fresh = g.snapshot();
    assert!(fresh.removed_bridge(removed).is_none());
}

#[test]
fn live_query_holding_old_links_reaches_nodes_past_a_concurrently_removed_neighbor() {
    // Chain 0 - 1 - 2 - 3 - 4 plus a far node 5 hanging off 1. The query sits
    // at 4, so everything useful lies past node 2. The reader pauses while
    // expanding node 1 (in the prefetch of neighbor 5, AFTER it loaded 1's
    // link list [0, 5, 2]); the writer then removes 2, rewriting 1's links
    // and invalidating 2. Resuming, the reader meets the removed 2 through its
    // old list: without a bridge it can never reach 3 or 4.
    struct PauseOnPrefetch {
        rows: Arc<Rows>,
        key: u64,
        once: AtomicBool,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl MutableVectorSource for PauseOnPrefetch {
        fn prefetch_node(&self, _handle: NodeHandle, key: u64) {
            if key == self.key && !self.once.swap(true, AcqRel) {
                self.entered.send(()).unwrap();
                self.release.lock().recv_timeout(Duration::from_secs(10)).unwrap();
            }
        }
        fn get(&self, key: u64) -> Option<&[f32]> {
            self.rows.get(key)
        }
    }
    let (source, g, ids) = bridge_chain(&[0.0, 1.0, 2.0, 3.0, 4.0, -10.0],
        &[&[1], &[0, 5, 2], &[1, 3], &[2, 4], &[3], &[1]]);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let paused = PauseOnPrefetch {
        rows: source.clone(),
        key: 5,
        once: AtomicBool::new(false),
        entered: entered_tx,
        release: Mutex::new(release_rx),
    };
    let reader_graph = g.clone();
    let reader = std::thread::spawn(move || reader_graph.search(&[4.0], 5, 8, &paused, |_| true));
    let entered = entered_rx.recv_timeout(Duration::from_secs(10));
    let removed = if entered.is_ok() { g.remove(ids[2], source.as_ref()) } else { Ok(false) };
    let _ = release_tx.send(());
    let result = reader.join().unwrap();
    entered.unwrap();
    assert!(removed.unwrap());
    // The rewrite really dropped 2 from 1's list (the reader used a stale list).
    assert!(!g.snapshot().node(ids[1]).unwrap().links(0).unwrap().contains(&internal_id(ids[2])));
    assert_eq!(result.hits, exact(source.as_ref(), &[4.0], 5, |key| key != 2));
    assert_eq!(result.missing_source_lookups, 0);
}

#[test]
fn upper_layer_descent_steps_through_a_concurrently_removed_node() {
    // Level 1: E(0) - X(2) - T(4), plus a far node P(-10) hanging off E.
    // Level 0: E is isolated, X - T. The query sits at T. The reader pauses
    // resolving P while descending level 1 from E (after loading E's level-1
    // links [P, X]); the writer removes X meanwhile. Without the bridge the
    // descent stays at E and the level-0 beam from the isolated E finds only
    // E; through the bridge it reaches T.
    struct PauseOnGet {
        rows: Arc<Rows>,
        key: u64,
        once: AtomicBool,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl MutableVectorSource for PauseOnGet {
        fn get(&self, key: u64) -> Option<&[f32]> {
            if key == self.key && !self.once.swap(true, AcqRel) {
                self.entered.send(()).unwrap();
                self.release.lock().recv_timeout(Duration::from_secs(10)).unwrap();
            }
            self.rows.get(key)
        }
    }
    let source = Arc::new(Rows { rows: vec![vec![0.0], vec![2.0], vec![4.0], vec![-10.0]] });
    let g = Arc::new(MutableHnsw::new(1, Similarity::L2, HnswParams { m: 2, ef_construction: 16, threads: 1, ..HnswParams::default() }));
    let ids: Vec<_> = (0..4u64).map(|key| {
        let mut permit = g.reserve_insert(key).unwrap();
        let id = permit.handle();
        g.test_install_node(id, key, 0.0, vec![vec![], vec![]]).unwrap();
        permit.pending = false;
        id
    }).collect();
    let (e, x, t, p) = (ids[0], ids[1], ids[2], ids[3]);
    let link = |from: NodeHandle, level: usize, to: &[NodeHandle]| {
        g.test_replace_links(from, level, &to.iter().map(|h| internal_id(*h)).collect::<Vec<_>>()).unwrap();
    };
    link(e, 1, &[p, x]);
    link(x, 1, &[e, t]);
    link(t, 1, &[x]);
    link(p, 1, &[e]);
    link(x, 0, &[t]);
    link(t, 0, &[x]);
    g.test_set_entry(e).unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let paused = PauseOnGet { rows: source.clone(), key: 3, once: AtomicBool::new(false), entered: entered_tx, release: Mutex::new(release_rx) };
    let reader_graph = g.clone();
    let reader = std::thread::spawn(move || reader_graph.search(&[4.0], 1, 1, &paused, |_| true));
    let entered = entered_rx.recv_timeout(Duration::from_secs(10));
    let removed = if entered.is_ok() { g.remove(x, source.as_ref()) } else { Ok(false) };
    let _ = release_tx.send(());
    let result = reader.join().unwrap();
    entered.unwrap();
    assert!(removed.unwrap());
    assert_eq!(result.hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2], "the descent reached T through the removed X");
}

#[test]
fn mutable_parallel_prepares_out_of_order_commits_remain_consistent() {
    let source = rows(384, 12);
    let g = graph(12);
    for key in 0..64 {
        g.insert(key, &source).unwrap();
    }
    let mut batches: [Vec<InsertReservation>; 4] = std::array::from_fn(|_| Vec::new());
    for key in 64..384 {
        batches[key as usize % 4].push(g.reserve_insert(key).unwrap());
    }
    std::thread::scope(|scope| {
        for (worker, reservations) in batches.into_iter().enumerate() {
            let g = &g;
            let source = &source;
            scope.spawn(move || {
                for reservation in reservations.into_iter().rev() {
                    let proposal = g.prepare(reservation, source).unwrap();
                    g.complete(proposal, source).unwrap();
                    assert!(
                        g.search(&source.rows[worker], 5, 100, source, |key| key < 64)
                            .hits
                            .iter()
                            .all(|&(key, _)| key < 64)
                    );
                }
            });
        }
    });
    assert_eq!(g.active_nodes(), 384);
    assert_eq!(g.stats().pending_reservations, 0);
    invariants(&g);
    for key in [0, 99, 200, 383] {
        assert_eq!(
            g.search(&source.rows[key], 10, 384, &source, |_| true).hits,
            exact(&source, &source.rows[key], 10, |_| true)
        );
    }
}
#[test]
fn mutable_checkpoint_roundtrip_and_corruption_checks_preserve_mutability() {
    let source = rows(120, 8);
    let g = graph(8);
    let ids: Vec<_> = (0..80).map(|key| g.insert(key, &source).unwrap()).collect();
    for key in (0..80).step_by(7) {
        g.remove(ids[key], &source).unwrap();
    }
    g.retire_after(ids[1], 9);
    let unused = g.reserve_insert(80).unwrap();
    drop(unused);
    let encoded = g.encode();
    let restored = MutableHnsw::open(&encoded).unwrap();
    assert_eq!(restored.encode(), encoded);
    assert!(restored.reserve_insert(80).is_err());
    for key in [2, 18, 77] {
        assert_eq!(
            restored
                .search(&source.rows[key], 10, 100, &source, |_| true)
                .hits,
            g.search(&source.rows[key], 10, 100, &source, |_| true).hits
        );
    }
    assert_eq!(restored.reclaim_through(8, 10, &source).unwrap(), 0);
    assert_eq!(restored.reclaim_through(9, 10, &source).unwrap(), 1);
    for key in 81..100 {
        restored.insert(key, &source).unwrap();
    }
    invariants(&restored);
    for len in [0, 7, 16, 47, encoded.len() - 1] {
        assert!(MutableHnsw::open(&encoded[..len]).is_err());
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(MutableHnsw::open(&trailing).is_err());
    let mut bad_entry = encoded.clone();
    bad_entry[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(MutableHnsw::open(&bad_entry).is_err());
    let mut bad_generation = encoded.clone();
    bad_generation[68..72].copy_from_slice(&0u32.to_le_bytes());
    assert!(MutableHnsw::open(&bad_generation).is_err());
    let mut bad_watermark = encoded.clone();
    bad_watermark[60..68].copy_from_slice(&0u64.to_le_bytes());
    assert!(MutableHnsw::open(&bad_watermark).is_err());
}
#[test]
fn mutable_drained_checkpoint_cannot_reset_issued_key_watermark() {
    let source = rows(1, 4);
    for committed in [false, true] {
        let g = graph(4);
        if committed {
            let id = g.insert(0, &source).unwrap();
            g.remove(id, &source).unwrap();
        } else {
            drop(g.reserve_insert(0).unwrap());
        }
        drop(reclaimed(&g, 1));
        assert_eq!(g.active_nodes(), 0);
        let encoded = g.encode();
        let restored = MutableHnsw::open(&encoded).unwrap();
        assert!(restored.reserve_insert(0).is_err());
        let mut missing_watermark = encoded;
        missing_watermark[56..60].copy_from_slice(&0u32.to_le_bytes());
        missing_watermark[60..68].copy_from_slice(&0u64.to_le_bytes());
        assert!(MutableHnsw::open(&missing_watermark).is_err());
    }
    let mut impossible_empty = graph(4).encode();
    impossible_empty[48..56].copy_from_slice(&1u64.to_le_bytes());
    assert!(MutableHnsw::open(&impossible_empty).is_err());
}

#[test]
fn mutable_source_relocation_requires_no_vector_copy_or_graph_rebuild() {
    let source = rows(128, 8);
    let g = graph(8);
    for key in 0..128 {
        g.insert(key, &source).unwrap();
    }
    let baseline = g.search(&source.rows[17], 10, 128, &source, |_| true).hits;
    let relocated = Rows {
        rows: source.rows.clone(),
    };
    let builds = HnswBuilder::builds_on_this_thread();
    drop(source);
    assert_eq!(
        g.search(&relocated.rows[17], 10, 128, &relocated, |_| true)
            .hits,
        baseline
    );
    assert_eq!(HnswBuilder::builds_on_this_thread(), builds);
}
#[test]
fn mutable_remove_missing_neighbor_source_leaves_topology_unchanged() {
    let source = rows(64, 8);
    let g = graph(8);
    for key in 0..64 {
        g.insert(key, &source).unwrap();
    }
    let entry = entry_handle(&g);
    let before = g.encode();
    assert!(g.remove(entry, &Rows { rows: Vec::new() }).is_err());
    assert_eq!(g.encode(), before);
    invariants(&g);
}
#[test]
fn mutable_all_similarity_scores_match_exact_and_zero_cosine_is_safe() {
    let mut source = rows(96, 12);
    for row in &mut source.rows {
        let n = norm(row);
        for x in row {
            *x /= n;
        }
    }
    source.rows[0].fill(0.0);
    for similarity in [
        Similarity::Cosine,
        Similarity::DotProduct,
        Similarity::MaxInnerProduct,
        Similarity::L2,
    ] {
        let g = MutableHnsw::new(12, similarity, HnswParams::default());
        for key in 0..96 {
            g.insert(key, &source).unwrap();
        }
        for q in [1, 17, 88] {
            let mut expected: Vec<_> = source
                .rows
                .iter()
                .enumerate()
                .map(|(key, row)| (key as u64, similarity.score(&source.rows[q], row)))
                .collect();
            expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            expected.truncate(10);
            assert_eq!(
                g.search(&source.rows[q], 10, 96, &source, |_| true).hits,
                expected
            );
        }
        assert!(
            g.search(&source.rows[0], 10, 96, &source, |_| true)
                .hits
                .iter()
                .all(|(_, score)| score.is_finite())
        );
        invariants(&g);
    }
}
#[test]
fn mutable_duplicate_reservations_and_removed_keys_are_rejected() {
    let source = rows(4, 8);
    let g = graph(8);
    let a = g.reserve_insert(0).unwrap();
    assert!(g.reserve_insert(0).is_err());
    let id = g.complete(g.prepare(a, &source).unwrap(), &source).unwrap();
    assert!(g.insert(0, &source).is_err());
    assert!(g.remove(id, &source).unwrap());
    drop(reclaimed(&g, 1));
    assert!(g.insert(0, &source).is_err());
    let next = g.insert(1, &source).unwrap();
    assert_eq!(next.slot(), id.slot());
    assert_eq!(next.generation(), id.generation() + 1);
    let restored = MutableHnsw::open(&g.encode()).unwrap();
    assert!(restored.insert(0, &source).is_err());
    assert!(restored.insert(1, &source).is_err());
    invariants(&restored);
}
#[test]
fn mutable_epoch_and_source_ack_both_precede_slot_reuse() {
    let source = rows(4, 8);
    let g = MutableHnsw::with_limits(
        8,
        Similarity::L2,
        HnswParams::default(),
        MutableHnswLimits {
            max_slots: 1,
            max_pending: 1,
            max_retired: 2,
            ..MutableHnswLimits::default()
        },
    );
    let old = g.insert(0, &source).unwrap();
    let view = g.snapshot();
    let held = view.node(old).unwrap();
    assert!(g.remove(old, &source).unwrap());
    assert_eq!(held.key(), 0);
    assert!(g.reserve_insert(1).is_err());
    assert!(g.collect_reclaimed().is_empty());
    drop(view);
    let token = reclaimed(&g, 1).pop().unwrap();
    assert_eq!(token.handle(), old);
    assert_eq!(token.key(), 0);
    assert!(g.reserve_insert(1).is_err());
    drop(token);
    let fresh = g.insert(1, &source).unwrap();
    assert_eq!(fresh.slot(), old.slot());
    assert_eq!(fresh.generation(), old.generation() + 1);
    assert!(!g.remove(old, &source).unwrap());
    assert_eq!(
        g.search(&source.rows[1], 1, 4, &source, |_| true).hits[0].0,
        1
    );
}
#[test]
fn mutable_compact_edges_retain_acquired_levels_until_reader_drains() {
    let source = rows(8, 8);
    let g = graph(8);
    let old = g.insert(0, &source).unwrap();
    let other = g.insert(1, &source).unwrap();
    let view = g.snapshot();
    let candidate = Candidate::new(0.0, view.node(old).unwrap());
    let saved = view.candidate_node(candidate).links(0).unwrap().to_vec();
    assert_eq!(saved, vec![internal_id(other)]);
    g.remove(old, &source).unwrap();
    assert!(view.node(old).is_none());
    // The candidate acquired a levels identity before removal, so it can
    // continue traversing after the current slot has become invalid.
    assert_eq!(view.candidate_node(candidate).links(0).unwrap(), saved);
    let newer = g.snapshot();
    assert!(
        !newer
            .node(other)
            .unwrap()
            .links(0)
            .unwrap()
            .contains(&internal_id(old))
    );
    let interim = g.insert(2, &source).unwrap();
    assert_ne!(interim.slot(), old.slot());
    assert!(g.collect_reclaimed().is_empty());
    drop(newer);
    drop(view);
    drop(reclaimed(&g, 1));
    let fresh = g.insert(3, &source).unwrap();
    assert_eq!(fresh.slot(), old.slot());
    assert_ne!(fresh, old);
    assert_eq!(g.stats().neighbor_size, 4);
    assert_eq!(std::mem::size_of::<Candidate>(), 16);
    invariants(&g);
    let restored = MutableHnsw::open(&g.encode()).unwrap();
    invariants(&restored);
    assert_eq!(
        restored
            .search(&source.rows[3], 1, 8, &source, |_| true)
            .hits[0]
            .0,
        3
    );
}
#[test]
fn mutable_compact_checkpoint_rejects_old_layout_duplicate_and_absent_links() {
    let source = rows(4, 8);
    let g = graph(8);
    let a = g.insert(0, &source).unwrap();
    let b = g.insert(1, &source).unwrap();
    let encoded = g.encode();
    let mut previous = encoded.clone();
    previous[4] = 2;
    assert!(MutableHnsw::open(&previous).is_err());
    for end in 0..encoded.len() {
        assert!(MutableHnsw::open(&encoded[..end]).is_err());
    }
    let view = g.snapshot();
    g.test_replace_links(a, 0, &[internal_id(b), internal_id(b)])
        .unwrap();
    assert!(MutableHnsw::open(&g.encode()).is_err());
    g.test_replace_links(a, 0, &[view.len]).unwrap();
    assert!(MutableHnsw::open(&g.encode()).is_err());
    g.test_replace_links(a, 0, &[internal_id(a)]).unwrap();
    assert!(MutableHnsw::open(&g.encode()).is_err());
    g.test_replace_links(a, 0, &[internal_id(b)]).unwrap();
    g.test_replace_links(b, 0, &[]).unwrap();
    assert!(MutableHnsw::open(&g.encode()).is_err());
}
#[test]
fn mutable_removed_node_arrays_count_as_retired_until_reader_drains() {
    let source = rows(4, 8);
    let g = graph(8);
    let a = g.insert(0, &source).unwrap();
    let b = g.insert(1, &source).unwrap();
    let view = g.snapshot();
    let node = view.node(a).unwrap();
    let bytes: usize = (0..=node.level())
        .map(|level| node.links(level).unwrap().len() * std::mem::size_of::<u32>())
        .sum();
    assert!(bytes >= 4);
    g.remove(a, &source).unwrap();
    assert!(g.stats().retired_adjacency_bytes >= bytes);
    assert_eq!(node.links(0).unwrap()[0], internal_id(b));
    drop(view);
    drop(reclaimed(&g, 1));
    let deadline = Instant::now() + Duration::from_secs(10);
    while g.stats().retired_adjacency_bytes != 0 && Instant::now() < deadline {
        drop(g.collect_reclaimed());
        std::thread::yield_now();
    }
    assert_eq!(g.stats().retired_adjacency_bytes, 0);
    assert_eq!(g.stats().live_adjacency_bytes, 0); // The survivor is isolated.
}
#[test]
fn mutable_later_prepare_does_not_retain_already_retired_generation() {
    let source = rows(4, 8);
    let g = graph(8);
    let old = g.insert(0, &source).unwrap();
    g.remove(old, &source).unwrap();
    let prepared = prepare(&g, 1, &source);
    let tokens = reclaimed(&g, 1);
    assert_eq!(tokens[0].handle(), old);
    drop(tokens);
    let fresh = g.reserve_insert(2).unwrap();
    assert_eq!(fresh.handle().slot(), old.slot());
    g.complete(prepared, &source).unwrap();
    g.complete(g.prepare(fresh, &source).unwrap(), &source)
        .unwrap();
    invariants(&g);
}
#[test]
fn mutable_pending_and_retired_limits_apply_backpressure_then_recover() {
    let source = rows(8, 8);
    let g = MutableHnsw::with_limits(
        8,
        Similarity::L2,
        HnswParams::default(),
        MutableHnswLimits {
            max_slots: 3,
            max_pending: 1,
            max_retired: 2,
            ..MutableHnswLimits::default()
        },
    );
    let first = g.reserve_insert(0).unwrap();
    assert!(g.reserve_insert(1).is_err());
    drop(first);
    let next = g.reserve_insert(1).unwrap();
    drop(next);
    assert!(g.reserve_insert(2).is_err());
    assert_eq!(g.stats().pending_reservations, 0);
    assert_eq!(g.stats().retired_slots, 2);
    drop(reclaimed(&g, 2));
    let a = g.insert(2, &source).unwrap();
    let b = g.insert(3, &source).unwrap();
    assert!(g.retire_after(a, 7));
    assert_eq!(g.reclaim_through(6, 1, &source).unwrap(), 0);
    assert_eq!(
        g.search(&source.rows[2], 1, 8, &source, |_| true).hits[0].0,
        2
    );
    assert_eq!(g.reclaim_through(7, 1, &source).unwrap(), 1);
    assert_eq!(g.stats().visibility_retained, 0);
    drop(reclaimed(&g, 1));
    assert!(g.remove(b, &source).unwrap());
    drop(reclaimed(&g, 1));
    assert_eq!(g.stats().retired_slots, 0);
}
#[test]
fn mutable_exhausted_generation_is_never_wrapped() {
    let source = rows(4, 8);
    let g = MutableHnsw::with_limits(
        8,
        Similarity::L2,
        HnswParams::default(),
        MutableHnswLimits {
            max_slots: 1,
            max_pending: 1,
            max_retired: 2,
            ..MutableHnswLimits::default()
        },
    );
    let first = g.reserve_insert(0).unwrap();
    drop(first);
    drop(reclaimed(&g, 1));
    // Exercise the real terminal counter value without billions of cycles.
    g.test_set_vacant_incarnation(0, u32::MAX - 1).unwrap();
    let last = g.insert(1, &source).unwrap();
    assert_eq!(last.generation(), u32::MAX);
    g.remove(last, &source).unwrap();
    drop(reclaimed(&g, 1));
    assert_eq!(g.stats().exhausted_slots, 1);
    assert_eq!(g.stats().reusable_slots, 0);
    assert!(g.reserve_insert(2).is_err());
    let reopened = MutableHnsw::open(&g.encode()).unwrap();
    assert_eq!(reopened.stats().exhausted_slots, 1);
}

#[test]
fn mutable_held_reader_bounds_retired_adjacency_growth() {
    let source = rows(16, 8);
    let g = MutableHnsw::with_limits(
        8,
        Similarity::L2,
        HnswParams::default(),
        MutableHnswLimits {
            max_retired_adjacency_bytes: 1,
            ..MutableHnswLimits::default()
        },
    );
    g.insert(0, &source).unwrap();
    let held = g.snapshot();
    let mut blocked = None;
    for key in 1..16 {
        if g.insert(key, &source).is_err() {
            blocked = Some(key);
            break;
        }
    }
    let blocked = blocked.expect("held immutable arrays must cause byte backpressure");
    assert!(g.stats().retired_adjacency_bytes >= 1);
    let before = g.encode();
    let entry = entry_handle(&g);
    assert!(g.remove(entry, &source).is_err());
    assert_eq!(g.encode(), before);
    drop(held);
    // Admission can reject before issuing a reservation. Drain only issued leases.
    let retired = g.stats().retired_slots;
    drop(reclaimed(&g, retired));
    let deadline = Instant::now() + Duration::from_secs(10);
    while g.stats().retired_adjacency_bytes != 0 && Instant::now() < deadline {
        drop(g.collect_reclaimed());
        std::thread::yield_now();
    }
    assert_eq!(g.stats().retired_adjacency_bytes, 0);
    g.insert(blocked + 1, &source).unwrap();
    invariants(&g);
}

struct OwnedRow {
    values: Vec<f32>,
    drops: Arc<AtomicUsize>,
}
impl Drop for OwnedRow {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Relaxed);
    }
}
#[derive(Clone, Default)]
struct DirectRows {
    slots: Vec<Option<(NodeHandle, u64, Arc<OwnedRow>)>>,
}
impl MutableVectorSource for DirectRows {
    fn get(&self, _: u64) -> Option<&[f32]> {
        panic!("direct source must not hash/lookup monotonic keys")
    }
    fn get_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        let (expected, expected_key, row) = self.slots.get(handle.slot() as usize)?.as_ref()?;
        (*expected == handle && *expected_key == key).then_some(row.values.as_slice())
    }
}
impl DirectRows {
    fn install(&mut self, permit: &InsertReservation, row: Vec<f32>, drops: &Arc<AtomicUsize>) {
        let slot = permit.handle().slot() as usize;
        self.slots
            .resize_with(self.slots.len().max(slot + 1), || None);
        assert!(self.slots[slot].is_none());
        self.slots[slot] = Some((
            permit.handle(),
            permit.key(),
            Arc::new(OwnedRow {
                values: row,
                drops: drops.clone(),
            }),
        ));
    }
    fn reclaim(&mut self, token: &ReclaimedNode) {
        let slot = &mut self.slots[token.handle().slot() as usize];
        if slot
            .as_ref()
            .is_some_and(|(handle, key, _)| *handle == token.handle() && *key == token.key())
        {
            *slot = None;
        }
    }
}
#[test]
fn mutable_transferred_prepare_withholds_source_tokens_through_cancel() {
    let g = graph(4);
    let mut source = DirectRows::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let first = g.reserve_insert(0).unwrap();
    let old = first.handle();
    source.install(&first, vec![1.0; 4], &drops);
    g.complete(g.prepare(first, &source).unwrap(), &source)
        .unwrap();
    let pending = g.reserve_insert(1).unwrap();
    let pending_id = pending.handle();
    source.install(&pending, vec![2.0; 4], &drops);
    let prepared = g.prepare(pending, &source).unwrap();
    // Compile-time and runtime proof that no thread-bound epoch Guard is
    // smuggled through the worker-to-writer preparation handoff.
    let prepared = std::thread::spawn(move || prepared).join().unwrap();
    g.remove(old, &source).unwrap();
    let lease_generation = prepared._generation.generation();
    assert!(g.control.domain.current_generation() > lease_generation);
    assert!(g.control.domain.update_oldest_used_generation() <= lease_generation);
    assert!(
        g.collect_reclaimed().is_empty(),
        "proposal must withhold source token"
    );
    assert_eq!(drops.load(Relaxed), 0);
    assert!(source.get_node(old, 0).is_some());
    drop(prepared); // Cancels its own reservation, then releases the read lease.
    let tokens = reclaimed(&g, 2);
    assert!(tokens.iter().any(|t| t.handle() == old));
    assert!(tokens.iter().any(|t| t.handle() == pending_id));
    for token in &tokens {
        source.reclaim(token);
    }
    assert_eq!(drops.load(Relaxed), 2);
    assert_eq!(g.stats().reusable_slots, 0);
    drop(tokens);
    assert_eq!(g.stats().reusable_slots, 2);
}

#[test]
fn mutable_source_lease_retains_old_root_without_pinning_reused_graph_slot() {
    let g = graph(4);
    let mut source = DirectRows::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let first = g.reserve_insert(0).unwrap();
    let old = first.handle();
    source.install(&first, vec![1.0; 4], &drops);
    g.complete(g.prepare(first, &source).unwrap(), &source)
        .unwrap();
    let old_root = source.clone();
    g.remove(old, &source).unwrap();
    let tokens = reclaimed(&g, 1);
    for token in &tokens {
        source.reclaim(token);
    }
    drop(tokens);
    assert_eq!(drops.load(Relaxed), 0); // Independent source root still owns old row.
    let next = g.reserve_insert(1).unwrap();
    assert_eq!(next.handle().slot(), old.slot());
    assert_ne!(next.handle(), old);
    source.install(&next, vec![2.0; 4], &drops);
    let fresh = g
        .complete(g.prepare(next, &source).unwrap(), &source)
        .unwrap();
    assert!(old_root.get_node(fresh, 1).is_none());
    assert!(source.get_node(old, 0).is_none());
    assert_eq!(old_root.get_node(old, 0).unwrap(), &[1.0; 4]);
    assert_eq!(
        g.search_with_handles(&[2.0; 4], 1, 4, &source, |handle, key| source
            .get_node(handle, key)
            .is_some())
            .hits[0]
            .0,
        1
    );
    drop(old_root);
    assert_eq!(drops.load(Relaxed), 1);
}
#[test]
fn mutable_failed_prepare_returns_source_lease_and_does_not_reuse_early() {
    let g = graph(4);
    let other = graph(4);
    let mut source = DirectRows::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let permit = g.reserve_insert(0).unwrap();
    let handle = permit.handle();
    source.install(&permit, vec![f32::NAN; 4], &drops);
    assert!(g.prepare(permit, &source).is_err());
    assert_eq!(g.stats().pending_reservations, 0);
    let token = reclaimed(&g, 1).pop().unwrap();
    assert!(g.owns_reclaimed(&token));
    assert!(!other.owns_reclaimed(&token));
    assert_eq!(g.stats().reusable_slots, 0);
    source.reclaim(&token);
    assert_eq!(drops.load(Relaxed), 1);
    drop(token);
    let fresh = g.reserve_insert(1).unwrap();
    assert_eq!(fresh.handle().slot(), handle.slot());
    source.install(&fresh, vec![1.0; 4], &drops);
    g.complete(g.prepare(fresh, &source).unwrap(), &source)
        .unwrap();
    assert_eq!(g.search(&[1.0; 4], 1, 4, &source, |_| true).hits[0].0, 1);
}

#[test]
fn mutable_fixed_cardinality_churn_reclaims_slots_keys_and_source_rows() {
    let n = 32;
    let rounds = 20;
    let data = rows(n * rounds, 8);
    let g = graph(8);
    let mut source = DirectRows::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let mut live = Vec::new();
    for round in 0..rounds {
        for id in live.drain(..) {
            g.remove(id, &source).unwrap();
        }
        if round > 0 {
            let tokens = reclaimed(&g, n);
            for token in &tokens {
                source.reclaim(token);
            }
            drop(tokens);
        }
        for key in round * n..(round + 1) * n {
            let permit = g.reserve_insert(key as u64).unwrap();
            source.install(&permit, data.rows[key].clone(), &drops);
            live.push(
                g.complete(g.prepare(permit, &source).unwrap(), &source)
                    .unwrap(),
            );
        }
        let stats = g.stats();
        assert_eq!(stats.allocated_slots, n as u32);
        assert_eq!(stats.pending_reservations, 0);
        assert_eq!(stats.retired_slots, 0);
        assert_eq!(source.slots.len(), n);
        assert_eq!(source.slots.iter().filter(|row| row.is_some()).count(), n);
        assert_eq!(drops.load(Relaxed), round * n);
        invariants(&g);
        let query = &data.rows[round * n];
        assert_eq!(
            g.search(query, 10, n, &source, |_| true).hits,
            exact(&data, query, 10, |key| key >= (round * n) as u64
                && key < ((round + 1) * n) as u64)
        );
    }
    assert_eq!(
        g.stats().issued_key_watermark,
        Some((n * rounds - 1) as u64)
    );
}

#[test]
fn mutable_prepared_neighbor_rejects_changed_levels_identity_without_research() {
    let source = rows(3, 4);
    let g = graph_with_bootstrap(4, 1);
    let first = g.insert(0, &source).unwrap();
    let proposal = prepare(&g, 1, &source);
    let PreparedConnections::Neighbors(ref connections) = proposal.preparation else {
        panic!("threshold requires a real prepared proposal");
    };
    assert_eq!(connections[0].len(), 1);
    let saved = connections[0][0];
    assert_eq!(saved.handle, first);
    let held = g.snapshot();
    let acquired = Candidate::new(0.0, held.node(first).unwrap());
    let old_links = held.candidate_node(acquired).links(0).unwrap().to_vec();
    g.test_move_levels(first).unwrap();
    assert_ne!(g.snapshot().node(first).unwrap().levels, saved.levels);
    assert_eq!(held.candidate_node(acquired).links(0).unwrap(), old_links);
    let draws = g.level_generator.draws();
    let traced = CountedRows {
        rows: &source,
        reads: Cell::new(0),
    };
    let inserted = g.complete(proposal, &traced).unwrap();
    assert_eq!(
        g.level_generator.draws(),
        draws,
        "stale ref must not redraw a level"
    );
    assert_eq!(traced.reads.get(), 1, "stale ref must not trigger search");
    let view = g.snapshot();
    let node = view.node(inserted).unwrap();
    assert!((0..=node.level()).all(|level| node.links(level).unwrap().is_empty()));
    assert_eq!(view.node(first).unwrap().handle(), first);
    invariants(&g);
}

#[test]
fn mutable_candidates_are_compact_and_order_only_by_numeric_distance() {
    let source = rows(2, 4);
    let g = graph(4);
    let a = g.insert(0, &source).unwrap();
    let b = g.insert(1, &source).unwrap();
    let view = g.snapshot();
    assert_eq!(std::mem::size_of::<Candidate>(), 16);
    let ca = Candidate::new(1.0, view.node(a).unwrap());
    let cb = Candidate::new(1.0, view.node(b).unwrap());
    assert_eq!(ca.cmp(&cb), std::cmp::Ordering::Equal);
    assert_eq!(ca, cb, "node IDs and storage refs are not tie breakers");
    let minus = Candidate::new(-0.0, view.node(a).unwrap());
    let plus = Candidate::new(0.0, view.node(b).unwrap());
    assert_eq!(minus.cmp(&plus), std::cmp::Ordering::Equal);
    assert_eq!(minus, plus);
    assert!(Candidate::new(2.0, view.node(a).unwrap()) > cb);
}

#[test]
#[ignore = "run separately with RUSTIC_HNSW_QUERY_WORKERS=4 and --test-threads=1 on EC2"]
fn adaptive_distances_preserve_serial_arithmetic_and_order() {
    for similarity in [Similarity::L2, Similarity::Cosine, Similarity::DotProduct] {
        let source = rows(35, 768);
        let graph = MutableHnsw::new(768, similarity, HnswParams::default());
        let handles: Vec<_> = (0..35).map(|key| graph.insert(key, &source).unwrap()).collect();
        let view = graph.snapshot();
        let d = SourceDistance::new(&view, &source, &source.rows[0]);
        let resolved: Vec<_> = handles.iter().map(|&id| d.resolve(id).unwrap()).collect();
        for batched in [false, true] {
            let parallel = d.parallel_distances(&resolved, batched).expect("parallel path must execute");
            let mut serial = Vec::new();
            for group in resolved.chunks(4) {
                if batched && group.len() == 4 && d.metric != Metric::L2 {
                    let products = dot_four(d.query, [group[0].row, group[1].row, group[2].row, group[3].row]);
                    for i in 0..4 { serial.push(d.from_dot(&group[i], products[i])); }
                } else { for row in group { serial.push(d.distance_to(row)); } }
            }
            assert_eq!(parallel, serial, "{similarity:?}, batched={batched}");
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_proposals_preserve_mutable_lifecycle_and_existing_component() {
    let source = rows(48, 8);
    let g = graph(8);
    let original = g.insert(0, &source).unwrap();
    let before = g.snapshot();
    // Directed ring intentionally proposes neighbors not yet published. CPU
    // bridging must connect the batch to the pre-existing shard component.
    let count = 32usize;
    let edges: Vec<_> = (0..count).flat_map(|i| [((i + 1) % count) as u32, ((i + 2) % count) as u32]).collect();
    let proposals = HnswGraph::from_knn_rows(count, 2, &edges, 0);
    let mut reservations: Vec<_> = (1..=count).map(|key| Some(g.reserve_insert(key as u64).unwrap())).collect();
    let handles: Vec<_> = reservations.iter().map(|r| r.as_ref().unwrap().handle()).collect();
    for ordinal in proposals.mutable_insertion_order() {
        let reservation = reservations[ordinal as usize].take().unwrap();
        let prepared = g.prepare_gpu(reservation, &source, &proposals, ordinal, &handles).unwrap();
        g.complete(prepared, &source).unwrap();
    }
    invariants(&g);
    assert_eq!(g.active_nodes(), count as u32 + 1);
    assert_eq!(before.handles().count(), 1, "older view must exclude new publications");
    for key in 0..=count {
        assert_eq!(g.search(&source.rows[key], 1, 128, &source, |_| true).hits[0].0, key as u64);
    }
    drop(before);
    g.remove(original, &source).unwrap();
    g.remove(handles[9], &source).unwrap();
    let restored = MutableHnsw::open(&g.encode()).unwrap();
    invariants(&restored);
    restored.insert(40, &source).unwrap();
    assert_eq!(restored.search(&source.rows[40], 1, 128, &source, |_| true).hits[0].0, 40);
    // A rejected GPU ordinal cancels the reservation, without a visible node.
    let count_before = restored.active_nodes();
    let rejected = restored.reserve_insert(41).unwrap();
    assert!(restored.prepare_gpu(rejected, &source, &proposals, 0, &handles).is_err());
    assert_eq!(restored.active_nodes(), count_before);
    invariants(&restored);
}

/// A chunked capture copies the graph exactly as of its capture point while
/// graph writers remove and insert nodes between (and during) its chunks.
#[test]
fn chunked_capture_is_exact_while_writers_change_the_graph() {
    let n = CAPTURE_CHUNK_NODES as usize * 2 + 500;
    let source = rows(n + 300, 4);
    let g = MutableHnsw::new(4, Similarity::L2, HnswParams::default());
    let handles: Vec<_> = (0..n as u64).map(|key| g.insert(key, &source).unwrap()).collect();
    // No concurrent change: exact by construction.
    let reference = g.capture_snapshot().unwrap();
    let before = g.mutation_generation();
    let (pending, ()) = g.begin_capture(|_, _| Ok(())).unwrap();
    let captured = std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            for &handle in handles.iter().rev().step_by(7).take(400) {
                g.remove(handle, &source).unwrap();
            }
            for key in n as u64..n as u64 + 300 {
                g.insert(key, &source).unwrap();
            }
        });
        let captured = g.finish_capture(pending).unwrap();
        writer.join().unwrap();
        captured
    });
    assert!(g.mutation_generation() > before, "topology changes bump the mutation generation");
    let expected: Vec<_> = reference.visible_identities().collect();
    assert_eq!(captured.visible_identities().collect::<Vec<_>>(), expected);
    for (_, handle) in &expected {
        let (a, b) = (reference.node(*handle).unwrap(), captured.node(*handle).unwrap());
        assert_eq!(a.level(), b.level());
        for level in 0..=a.level() {
            assert_eq!(a.links(level), b.links(level), "links of {handle:?} at level {level}");
        }
    }
}

/// Delete repair on collinear rows, where diversity selection makes the
/// layer-0 graph a chain: removing every third node (the engine tests'
/// `knn_deletes_are_invisible_at_refresh_before_graph_repair` and
/// `older_view_keeps_live_knn_hits_across_deletes_and_repair` corpus) must
/// keep every survivor reachable, so a beam wider than the graph finds all
/// of them, in distance order.
#[test]
fn removal_repair_keeps_a_collinear_chain_connected() {
    struct Line(Vec<Vec<f32>>);
    impl MutableVectorSource for Line {
        fn get(&self, key: u64) -> Option<&[f32]> { self.0.get(key as usize).map(Vec::as_slice) }
    }
    let source = Line((0..30).map(|i| vec![i as f32, 1.0]).collect());
    let g = MutableHnsw::new(2, Similarity::L2, HnswParams::default());
    let ids: Vec<_> = (0..30).map(|key| g.insert(key, &source).unwrap()).collect();
    let all = g.search(&[0., 1.], 30, 60, &source, |_| true);
    assert_eq!(all.hits.iter().map(|h| h.0).collect::<Vec<_>>(), (0..30).collect::<Vec<u64>>());
    for key in (0..30).step_by(3) {
        assert!(g.remove(ids[key], &source).unwrap());
    }
    let survivors: Vec<u64> = (0..30).filter(|key| key % 3 != 0).collect();
    let hits = g.search(&[0., 1.], 30, 60, &source, |_| true);
    assert_eq!(hits.hits.iter().map(|h| h.0).collect::<Vec<_>>(), survivors, "a survivor became unreachable after delete repair");
    // From the far end too (another entry descent).
    let far = g.search(&[29., 1.], 30, 60, &source, |_| true);
    let mut keys: Vec<u64> = far.hits.iter().map(|h| h.0).collect();
    keys.sort_unstable();
    assert_eq!(keys, survivors);
}

/// Microbench (ignored; `cargo test --release -p rustic-core --lib
/// knn_k100_nc1000_ops_vs_lucene_reference -- --ignored --nocapture`,
/// `RUSTIC_BENCH_DOCS`, default 200,000): the dense_vector track's shape
/// (96-d dot product, m=32, ef_construction=100), k=100 / ef=1000. Prints
/// our traversal's vector ops, latency and recall@100, and those of a
/// Lucene `HnswGraphSearcher` reimplementation run over the SAME graph
/// (greedy topK=1 per upper layer, then layer 0 with topK=ef and Lucene's
/// stop rule: the best candidate is worse than the worst result once the
/// results are full). Equal ops on the same graph put the remaining gap
/// to ES in graph construction; different ops, in the search loop.
///
/// Profiling: the brute-force truth is computed before the `built` line,
/// and from it our search alone runs for `RUSTIC_BENCH_SEARCH_SECS`
/// (default 5) before the `search done` line (attach `perf -p` in
/// between); the Lucene reference loop runs after it.
#[test]
#[ignore]
fn knn_k100_nc1000_ops_vs_lucene_reference() {
    use std::cmp::Ordering as CmpOrdering;
    use std::collections::{BinaryHeap, HashSet};
    let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let dim = 96;
    let mut rng = XorShift64Star::new(0x5EED);
    let mut unit = || {
        let mut v: Vec<f32> = (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v { *x /= n; }
        v
    };
    let source = Rows { rows: (0..docs).map(|_| unit()).collect() };
    let queries: Vec<Vec<f32>> = (0..100).map(|_| unit()).collect();
    let g = MutableHnsw::new(dim as u32, Similarity::DotProduct, HnswParams { m: 32, ef_construction: 100, ..HnswParams::default() });
    let started = Instant::now();
    for key in 0..docs as u64 { g.insert(key, &source).unwrap(); }
    let build_time = started.elapsed();
    {
        // Graph stats: nodes and average degree per level (capacity 2m on
        // layer 0, m above; Vespa inserts link up to m on every layer).
        let view = g.snapshot();
        let mut per_level: Vec<(usize, usize)> = Vec::new();
        for handle in view.handles() {
            let node = view.node(handle).unwrap();
            for level in 0..=node.level() {
                if per_level.len() <= level { per_level.resize(level + 1, (0, 0)); }
                per_level[level].0 += 1;
                per_level[level].1 += node.links(level).map_or(0, <[u32]>::len);
            }
        }
        for (level, (nodes, links)) in per_level.iter().enumerate() {
            eprintln!("level {level}: {nodes} nodes, average degree {:.1}", *links as f64 / (*nodes).max(1) as f64);
        }
    }
    let dist = |q: &[f32], key: u64| -> f64 { -f64::from(dot(q, &source.rows[key as usize])) };
    // Brute-force truth, outside every timed / profiled window.
    let truths: Vec<HashSet<u64>> = queries.iter().map(|q| {
        let mut all: Vec<(f64, u64)> = (0..docs as u64).map(|k| (dist(q, k), k)).collect();
        all.select_nth_unstable_by(99, |a, b| a.0.total_cmp(&b.0));
        all.into_iter().take(100).map(|(_, k)| k).collect()
    }).collect();
    eprintln!("built {docs} x {dim}d (m=32, efc=100) in {build_time:?}; truth ready, searching");
    #[derive(Clone, Copy, PartialEq)]
    struct D(f64, u32);
    impl Eq for D {}
    impl PartialOrd for D { fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> { Some(self.cmp(o)) } }
    impl Ord for D { fn cmp(&self, o: &Self) -> CmpOrdering { self.0.total_cmp(&o.0).then(self.1.cmp(&o.1)) } }
    let view = g.snapshot();
    // Our search alone, repeated over the queries for the profiling window.
    let window = std::time::Duration::from_secs(std::env::var("RUSTIC_BENCH_SEARCH_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(5));
    let (mut ours_ops, mut ours_recall, mut ours_queries) = (0u64, 0usize, 0u64);
    let mut latencies: Vec<std::time::Duration> = Vec::new();
    let window_start = Instant::now();
    while ours_queries < queries.len() as u64 || window_start.elapsed() < window {
        let i = (ours_queries % queries.len() as u64) as usize;
        let t = Instant::now();
        let ours = view.search(&queries[i], 100, 1000, &source, |_| true);
        latencies.push(t.elapsed());
        if ours_queries < queries.len() as u64 {
            ours_ops += ours.vector_ops - ours.final_score_ops;
            ours_recall += ours.hits.iter().filter(|(k, _)| truths[i].contains(k)).count();
        }
        ours_queries += 1;
    }
    let searched = window_start.elapsed();
    latencies.sort();
    let n = queries.len() as u64;
    eprintln!("search done: {ours_queries} queries in {searched:?}");
    eprintln!("ours:   {:>7} traversal ops/query, recall@100 {:.4}, mean {:?}, p50 {:?}, p99 {:?}/query (search only)",
        ours_ops / n, ours_recall as f64 / (100.0 * n as f64), searched / ours_queries as u32,
        latencies[latencies.len() / 2], latencies[latencies.len() * 99 / 100]);
    let key_of = |id: u32| view.current_node(id).unwrap().key();
    let links = |id: u32, level: usize| -> Vec<u32> { view.current_node(id).and_then(|n| n.links(level).map(<[u32]>::to_vec)).unwrap_or_default() };
    // Lucene searchLevel: results (max-heap of distance, worst on top) of
    // size topk, candidates (min-heap); stop when the best candidate is
    // worse than the worst result and results are full.
    let search_level = |q: &[f32], eps: &[D], level: usize, topk: usize, ops: &mut u64| -> Vec<D> {
        // A bitmap (Lucene's FixedBitSet), not a hashed set.
        let mut visited = vec![0u64; (view.len as usize).div_ceil(64)];
        let mut insert = |n: u32| { let (w, b) = (n as usize / 64, 1u64 << (n % 64)); let fresh = visited[w] & b == 0; visited[w] |= b; fresh };
        for e in eps { insert(e.1); }
        let mut results: BinaryHeap<D> = eps.iter().copied().collect();
        let mut candidates: BinaryHeap<std::cmp::Reverse<D>> = eps.iter().copied().map(std::cmp::Reverse).collect();
        while results.len() > topk { results.pop(); }
        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            if results.len() >= topk && c.0 > results.peek().unwrap().0 { break; }
            for n in links(c.1, level) {
                if !insert(n) { continue; }
                *ops += 1;
                let d = D(dist(q, key_of(n)), n);
                if results.len() < topk || d.0 < results.peek().unwrap().0 {
                    candidates.push(std::cmp::Reverse(d));
                    results.push(d);
                    if results.len() > topk { results.pop(); }
                }
            }
        }
        results.into_vec()
    };
    let (mut ref_ops, mut ref_recall, mut ref_time) = (0u64, 0usize, std::time::Duration::ZERO);
    for (q, want) in queries.iter().zip(&truths) {
        let t = Instant::now();
        let mut ops = 1u64;
        let entry = view.entry as u32;
        let top = (view.entry >> 32) as usize;
        let mut eps = vec![D(dist(q, key_of(entry)), entry)];
        for level in (1..=top).rev() {
            let r = search_level(q, &eps, level, 1, &mut ops);
            eps = vec![*r.iter().min().unwrap()];
        }
        let mut r = search_level(q, &eps, 0, 1000, &mut ops);
        r.sort();
        ref_time += t.elapsed();
        ref_ops += ops;
        ref_recall += r.iter().take(100).filter(|d| want.contains(&key_of(d.1))).count();
    }
    eprintln!("lucene: {:>7} traversal ops/query, recall@100 {:.4}, {:?}/query (reference loop, not tuned)", ref_ops / n, ref_recall as f64 / (100.0 * n as f64), ref_time / n as u32);
}

/// Contiguous rows (`key * dim`), prefetched when asked: the layout of a
/// fixed-stride vector store, for comparing against the others.
struct DenseRows { data: Vec<f32>, dim: usize }
impl MutableVectorSource for DenseRows {
    fn get(&self, key: u64) -> Option<&[f32]> {
        let start = key as usize * self.dim;
        self.data.get(start..start + self.dim)
    }
    fn prefetch_row(&self, _: NodeHandle, key: u64) {
        let start = key as usize * self.dim;
        if start < self.data.len() { prefetch(self.data[start..].as_ptr().cast::<u8>(), self.dim * 4); }
    }
    // The bench inserts key `k` into slot `k` (id `k + 1`).
    fn prefetch_slot(&self, id: u32) {
        if id > 0 { self.prefetch_row(NodeHandle::new(0, 0), u64::from(id - 1)); }
    }
}

/// int8 codes in one fixed-stride array by key (= slot), scored as the
/// production bound scorer scores them.
struct DenseCodes { words: Vec<f32>, stride: usize, epochs: Vec<u64>, dim: usize, owner: Arc<MutableVectorOwner> }
struct DenseCodesScorer<'a> { codes: &'a DenseCodes, scorers: std::cell::RefCell<Vec<(u64, crate::vector::es_quant::EsQuantScorer)>>, query: Vec<f32> }
impl MutableQueryScorer for DenseCodesScorer<'_> {
    fn score(&self, _: NodeHandle, key: u64) -> Option<f32> {
        let key = key as usize;
        let epoch = *self.codes.epochs.get(key)?;
        let mut scorers = self.scorers.borrow_mut();
        let at = match scorers.iter().position(|(e, _)| *e == epoch) {
            Some(at) => at,
            None => { scorers.push((epoch, self.codes.owner.code_scorer_for_test(epoch, &self.query)?)); scorers.len() - 1 }
        };
        let row = &self.codes.words[key * self.codes.stride..(key + 1) * self.codes.stride];
        // Trailing alignment padding is not part of the row.
        let len = 4 + (self.codes.dim).div_ceil(4);
        scorers[at].1.score_words(&row[..len], self.codes.dim)
    }
}
impl MutableVectorSource for DenseCodes {
    fn get(&self, _: u64) -> Option<&[f32]> { None }
    fn bind_query(&self, query: &[f32]) -> Option<Box<dyn MutableQueryScorer + '_>> {
        Some(Box::new(DenseCodesScorer { codes: self, scorers: Default::default(), query: query.to_vec() }))
    }
    fn prefetch_row(&self, _: NodeHandle, key: u64) {
        let start = key as usize * self.stride;
        if start < self.words.len() { prefetch(self.words[start..].as_ptr().cast::<u8>(), (4 + self.dim.div_ceil(4)) * 4); }
    }
    fn prefetch_slot(&self, id: u32) {
        if id > 0 { self.prefetch_row(NodeHandle::new(0, 0), u64::from(id - 1)); }
    }
}

/// Microbench (ignored): the cost of one HNSW step (one scored neighbour)
/// on the knn_ops shape (96-d dot product, m=32, ef_construction=100,
/// k=100, ef=1000), by vector source:
/// - `vecs`: one heap allocation per row, no prefetch (the knn_ops bench's
///   test source: a step is two serial misses, its header then its data);
/// - `dense`: one contiguous fixed-stride array, prefetched per neighbourhood;
/// - `owner`: the production source (`MutableVectorOwner`, f32 rows);
/// - `int8`: the production source of an int8_hnsw field (traversal on
///   7-bit codes, no rescore), its own graph.
/// The f32 sources search the same graph, so their traversal ops are equal;
/// `compute` is the dot products of the same rows gathered contiguously
/// (in cache): what is left of a step is graph bookkeeping and misses.
/// `cargo test --release -p rustic-core --lib knn_ns_per_step_by_source --
/// --ignored --nocapture`; `RUSTIC_BENCH_DOCS` (default 200,000; try
/// 1,000,000), `RUSTIC_BENCH_SOURCES` (default `vecs,dense,owner,int8`).
#[test]
#[ignore]
fn knn_ns_per_step_by_source() {
    use rayon::prelude::*;
    use std::collections::HashSet;
    // Room for a 1M-row owner (the test default budget is 256 MB).
    crate::admission::configure_shared(24 << 30);
    let docs: usize = std::env::var("RUSTIC_BENCH_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000);
    // `RUSTIC_BENCH_EF`: the beam width (default 1000, num_candidates).
    let bench_ef: usize = std::env::var("RUSTIC_BENCH_EF").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let sources = std::env::var("RUSTIC_BENCH_SOURCES").unwrap_or_else(|_| "vecs,dense,owner,int8".into());
    let dim = 96;
    let mut rng = XorShift64Star::new(0x5EED);
    let mut unit = || {
        let mut v: Vec<f32> = (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v { *x /= n; }
        v
    };
    let vecs = Rows { rows: (0..docs).map(|_| unit()).collect() };
    let queries: Vec<Vec<f32>> = (0..100).map(|_| unit()).collect();
    let dense = DenseRows { data: vecs.rows.iter().flatten().copied().collect(), dim };
    // `RUSTIC_BENCH_DUMP=<dir>`: write the corpus and queries (little-endian
    // f32, row after row) for running other engines on the same data.
    if let Ok(out) = std::env::var("RUSTIC_BENCH_DUMP") {
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(format!("{out}/vectors.f32"), bytes(&dense.data)).unwrap();
        std::fs::write(format!("{out}/queries.f32"), bytes(&queries.concat())).unwrap();
        eprintln!("dumped {docs} vectors and {} queries to {out}", queries.len());
        return;
    }
    let truths: Vec<HashSet<u64>> = queries.iter().map(|q| {
        let mut all: Vec<(f32, u64)> = (0..docs as u64).map(|k| (-dot(q, &vecs.rows[k as usize]), k)).collect();
        all.select_nth_unstable_by(99, |a, b| a.0.total_cmp(&b.0));
        all.into_iter().take(100).map(|(_, k)| k).collect()
    }).collect();
    let params = HnswParams { m: 32, ef_construction: 100, ..HnswParams::default() };
    // Build through a production owner, a chunk of reservations prepared in
    // parallel and completed in order (as `MutableIngestField::stage`).
    // `RUSTIC_BENCH_CACHE=<dir>`: reuse a built graph and its rows
    // (checkpoint files per kind and size), so variants search one graph.
    let cache = std::env::var("RUSTIC_BENCH_CACHE").ok();
    let build_fresh = |quantized: bool| -> (Arc<MutableHnsw>, Arc<MutableVectorOwner>, Duration) {
        let graph = Arc::new(MutableHnsw::new(dim as u32, Similarity::DotProduct, params));
        let owner = Arc::new(if quantized {
            MutableVectorOwner::new_quantized(graph.clone(), Some(crate::vector::es_quant::EsQuant { bits: 7, osq: true, confidence_interval: None })).unwrap()
        } else { MutableVectorOwner::new(graph.clone()).unwrap() });
        let started = Instant::now();
        for chunk in (0..docs).collect::<Vec<_>>().chunks(512) {
            let rows: Vec<&[f32]> = chunk.iter().map(|&k| vecs.rows[k].as_slice()).collect();
            let codes = owner.prepare_codes(&rows).unwrap();
            let reservations: Vec<_> = chunk.iter().zip(&codes).map(|(&k, code)| {
                let reservation = graph.reserve_insert(k as u64).unwrap();
                owner.install_prepared(&reservation, &vecs.rows[k], code.as_ref()).unwrap();
                reservation
            }).collect();
            let prepared: Vec<_> = reservations.into_par_iter().map(|reservation| {
                let read = owner.pin().unwrap();
                graph.prepare(reservation, &read).unwrap()
            }).collect();
            for proposal in prepared {
                let read = owner.pin().unwrap();
                graph.complete(proposal, &read).unwrap();
            }
            owner.commit().unwrap();
        }
        let took = started.elapsed();
        (graph, owner, took)
    };
    let build = |quantized: bool| -> (Arc<MutableHnsw>, Arc<MutableVectorOwner>, Duration) {
        let kind = if quantized { "int8" } else { "f32" };
        let files = cache.as_ref().map(|dir| (format!("{dir}/{kind}.{docs}.graph"), format!("{dir}/{kind}.{docs}.rows")));
        if let Some((g, r)) = &files {
            if let (Ok(graph), Ok(rows)) = (std::fs::read(g), std::fs::read(r)) {
                let started = Instant::now();
                let (graph, owner) = MutableVectorOwner::restore_checkpoint(&graph, &rows).unwrap();
                eprintln!("restored {kind} {g} in {:?}", started.elapsed());
                return (graph, Arc::new(owner), Duration::ZERO);
            }
        }
        let built = build_fresh(quantized);
        if let Some((g, r)) = &files {
            let checkpoint = built.1.capture_checkpoint().unwrap();
            let mut rows = Vec::new();
            checkpoint.write_rows(&mut rows).unwrap();
            std::fs::write(g, checkpoint.graph_bytes().unwrap()).unwrap();
            std::fs::write(r, rows).unwrap();
        }
        built
    };
    let run = |name: &str, view: &MutableHnswSnapshot, source: &dyn Fn(&[f32]) -> MutableSearchResult| {
        let _ = view;
        // Warm once, then time every query.
        for q in queries.iter().take(10) { let _ = source(q); }
        let (mut ops, mut recall, mut time) = (0u64, 0usize, Duration::ZERO);
        for (q, want) in queries.iter().zip(&truths) {
            let t = Instant::now();
            let r = source(q);
            time += t.elapsed();
            ops += r.vector_ops - r.final_score_ops;
            recall += r.hits.iter().filter(|(k, _)| want.contains(k)).count();
        }
        let n = queries.len() as u64;
        eprintln!("{name:6} {:>7} ops/query, {:6.1} ns/step, {:?}/query, recall@100 {:.4}",
            ops / n, time.as_nanos() as f64 / ops.max(1) as f64, time / n as u32, recall as f64 / (100.0 * n as f64));
        // `RUSTIC_BENCH_PROFILE=<name>`: keep searching for 5 s (perf window).
        if std::env::var("RUSTIC_BENCH_PROFILE").is_ok_and(|p| p == name) {
            eprintln!("profiling {name}");
            let started = Instant::now();
            let mut i = 0;
            while started.elapsed() < Duration::from_secs(5) { let _ = source(&queries[i % queries.len()]); i += 1; }
            eprintln!("profiled {name}: {i} queries");
        }
    };
    let wants = |s: &str| sources.split(',').any(|x| x.trim() == s);
    if wants("vecs") || wants("dense") || wants("owner") {
        let (graph, owner, took) = build(false);
        eprintln!("built {docs} x {dim}d f32 (m=32, efc=100) in {took:?}");
        let view = graph.snapshot();
        let degrees: Vec<usize> = (1..view.len).filter_map(|id| view.current_node(id)).map(|n| n.links(0).map_or(0, <[u32]>::len)).collect();
        eprintln!("level-0 degree: mean {:.1}, max {}", degrees.iter().sum::<usize>() as f64 / degrees.len().max(1) as f64, degrees.iter().max().unwrap_or(&0));
        if wants("vecs") { run("vecs", &view, &|q| view.search(q, 100, bench_ef, &vecs, |_| true)); }
        if wants("dense") { run("dense", &view, &|q| view.search(q, 100, bench_ef, &dense, |_| true)); }
        if wants("owner") {
            let (owned, read) = owner.read().unwrap();
            run("owner", &owned, &|q| owned.search(q, 100, bench_ef, &read, |_| true));
            // As production searches: a prepared node filter (every node),
            // `accept` exactly its bitmap.
            let all = vec![u64::MAX; (owned.len as usize).div_ceil(64)];
            run("ownerF", &owned, &|q| owned.search_with_exact_node_filter(q, 100, bench_ef, &read, |_, _| true,
                MutableSearchMode::VespaPlain(VespaPlainParams::default()), &all).unwrap());
        }
        // Pure compute: the dot products of one query's scored rows,
        // gathered contiguously (in cache) in visiting order.
        let scored: std::cell::RefCell<Vec<u64>> = Default::default();
        struct Recording<'a> { rows: &'a DenseRows, scored: &'a std::cell::RefCell<Vec<u64>> }
        impl MutableVectorSource for Recording<'_> {
            fn get(&self, key: u64) -> Option<&[f32]> { self.scored.borrow_mut().push(key); self.rows.get(key) }
        }
        let _ = view.search(&queries[0], 100, bench_ef, &Recording { rows: &dense, scored: &scored }, |_| true);
        let gathered: Vec<f32> = scored.borrow().iter().flat_map(|&k| dense.get(k).unwrap().iter().copied()).collect();
        let steps = scored.borrow().len().max(1);
        let t = Instant::now();
        let mut sink = 0f32;
        for _ in 0..100 { for row in gathered.chunks_exact(dim) { sink += dot(&queries[0], row); } }
        eprintln!("compute {:6.1} ns/step (96-d dot over in-cache rows; {steps} steps, sink {sink:.1})",
            t.elapsed().as_nanos() as f64 / (100 * steps) as f64);
    }
    if wants("int8") {
        let (graph, owner, took) = build(true);
        eprintln!("built {docs} x {dim}d int8_hnsw (m=32, efc=100) in {took:?}");
        let (view, read) = owner.read().unwrap();
        let _ = graph;
        run("int8", &view, &|q| view.search(q, 100, bench_ef, &read, |_| true));
        let all = vec![u64::MAX; (view.len as usize).div_ceil(64)];
        run("int8F", &view, &|q| view.search_with_exact_node_filter(q, 100, bench_ef, &read, |_, _| true,
            MutableSearchMode::VespaPlain(VespaPlainParams::default()), &all).unwrap());
        // The same codes in one fixed-stride array by slot (key `k` is in
        // slot `k`): what a dense code store would read per step.
        let mut stride = 0;
        let mut epochs = Vec::with_capacity(docs);
        let mut words = Vec::new();
        for handle in view.handles() {
            let key = view.node(handle).unwrap().key() as usize;
            let (epoch, row) = owner.code_words_for_test(handle, key as u64).unwrap();
            if stride == 0 { stride = row.len().div_ceil(16) * 16; words = vec![0f32; docs * stride]; epochs = vec![0u64; docs]; }
            words[key * stride..key * stride + row.len()].copy_from_slice(&row);
            epochs[key] = epoch;
        }
        let codes = DenseCodes { words, stride, epochs, dim, owner: owner.clone() };
        run("int8d", &view, &|q| view.search(q, 100, bench_ef, &codes, |_| true));
    }
}

/// Delete repair keeps every live node reachable at level 0 from the entry
/// point: collinear 2-d chains (the diversity heuristic keeps about two
/// links per node, so each delete opens a gap that only the repair's
/// reconnect closes) and random 8-d points, in shuffled insertion orders,
/// with every third node (chains) or a random 30% (points) deleted.
#[test]
fn delete_repair_keeps_every_live_node_reachable() {
    fn reachable(g: &MutableHnsw) -> (usize, usize) {
        let view = g.snapshot();
        let live = (1..view.len).filter(|&id| view.current_node(id).is_some_and(|n| view.visible(n))).count();
        let mut seen = vec![false; view.len as usize];
        let mut stack = vec![view.entry as u32];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if id == 0 || id >= view.len || std::mem::replace(&mut seen[id as usize], true) { continue; }
            let Some(n) = view.current_node(id).filter(|n| view.visible(*n)) else { continue };
            count += 1;
            stack.extend_from_slice(n.links(0).unwrap_or(&[]));
        }
        (count, live)
    }
    for seed in 0..120u64 {
        let mut rng = XorShift64Star::new(0x9E37_79B9 ^ seed.wrapping_mul(0x1000_0001));
        let chain = seed % 2 == 0;
        let n = if chain { [30usize, 100, 300][(seed / 2 % 3) as usize] } else { 600 };
        let dim = if chain { 2 } else { 8 };
        // Keys must increase: key k holds the point at position order[k].
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..order.len()).rev() { order.swap(i, (rng.next_f64() * (i + 1) as f64) as usize % (i + 1)); }
        let source = Rows { rows: order.iter().map(|&p| if chain { vec![p as f32, 1.0] } else { (0..dim).map(|_| (rng.next_f64() * 2.0 - 1.0) as f32).collect() }).collect() };
        let g = graph_with_bootstrap(dim, if seed % 4 < 2 { 10_000 } else { 16 });
        // Insert one by one, or as bulk staging does: a batch of rows
        // prepared against the same graph, then completed in order.
        let handles: Vec<NodeHandle> = if seed % 3 == 0 {
            (0..n as u64).map(|key| g.insert(key, &source).unwrap()).collect()
        } else {
            let mut handles = Vec::with_capacity(n);
            for batch in (0..n as u64).collect::<Vec<_>>().chunks(if seed % 3 == 1 { 30 } else { 64 }) {
                let reservations: Vec<_> = batch.iter().map(|&key| g.reserve_insert(key).unwrap()).collect();
                let prepared: Vec<_> = reservations.into_iter().map(|r| g.prepare(r, &source).unwrap()).collect();
                handles.extend(prepared.into_iter().map(|p| g.complete(p, &source).unwrap()));
            }
            handles
        };
        // Chains: every third position; points: a random 30%.
        let mut gone: Vec<usize> = if chain { (0..n).filter(|&k| order[k] % 3 == 0).collect() } else { (0..n).filter(|_| rng.next_f64() < 0.3).collect() };
        for i in (1..gone.len()).rev() { gone.swap(i, (rng.next_f64() * (i + 1) as f64) as usize % (i + 1)); }
        for &key in &gone { assert!(g.remove(handles[key], &source).unwrap()); }
        let (count, live) = reachable(&g);
        assert_eq!(live, n - gone.len());
        assert_eq!(count, live, "seed {seed} ({}, n {n}): {} of {live} live nodes reachable after repair", if chain { "chain" } else { "points" }, count);
        let hits = g.search(&source.rows[0], n, 4 * n, &source, |_| true).hits;
        assert_eq!(hits.len(), live, "seed {seed}: search returns every live node");
    }
}
