//! DiskBBQ (`bbq_disk`) through [`Index`]: one shared index per field,
//! trained and grown by maintenance, searched with `f32` rescoring; deletes,
//! filters, background splits and compaction, reopen with catch-up,
//! retained views, recovery by files, and resident memory.

use std::collections::HashSet;

use crate::index::{Index, KnnHit, KnnRequest};
use crate::query::query::Query;
use crate::types::*;
use crate::vector::ivf::{test_corpus, IvfOptions};
use crate::vector::quantize::Quantization;
use crate::vector::store::{OptimizerPolicy, PlannerConfig, VectorFieldOptions};
use crate::vector::Similarity;

const F_ID: u16 = 0;
const F_TAG: u16 = 1;
const F_V: u16 = 2;
const F_RARE: u16 = 3;
const DIM: usize = 32;
const K: usize = 10;

/// `bbq_disk` as the engine maps it: no graph, 1-bit codes, 3x rescoring.
fn options(cluster_size: u32, flat_index_threshold: i32) -> VectorFieldOptions {
    VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_osq(true).flat().with_oversample(3.0)
        .with_ivf(Some(IvfOptions { cluster_size, flat_index_threshold, ..IvfOptions::default() }))
}

fn schema(cluster_size: u32, flat_index_threshold: i32) -> Vec<Field> {
    vec![
        Field::new(F_ID, "id", FieldType::I64),
        Field::new(F_TAG, "tag", FieldType::Keyword),
        Field::new(F_V, "v", FieldType::Vector { dim: DIM as u16 }).with_similarity("l2_norm").with_vector_options(Some(options(cluster_size, flat_index_threshold))),
        Field::new(F_RARE, "rare", FieldType::Keyword),
    ]
}

/// Chunks of more than 100 vectors are sealed (count driven, never idle).
fn policy() -> OptimizerPolicy {
    OptimizerPolicy {
        indexing_threshold: 100,
        indexing_threshold_kb: usize::MAX,
        seal_full_scan_threshold: usize::MAX,
        seal_idle: None,
        max_segments: 8,
        payload_merge: crate::vector::store::PayloadMerge::Capped,
        ..OptimizerPolicy::default()
    }
}

struct Corpus {
    rows: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn corpus(n: usize) -> Corpus {
    let mut rows = test_corpus(21, n + 12, DIM, 24);
    let queries = rows.split_off(n);
    Corpus { rows, queries }
}

fn add(idx: &mut Index, c: &Corpus, range: std::ops::Range<usize>) {
    for i in range {
        let mut d = Document::new();
        d.add_i64(F_ID, i as i64);
        d.add_keyword(F_TAG, if i % 2 == 0 { "even" } else { "odd" });
        d.add_vector(F_V, c.rows[i].clone());
        d.add_keyword(F_RARE, if i % 251 == 0 { "yes" } else { "no" });
        idx.add_document(&d).unwrap();
    }
    idx.refresh().unwrap();
}

fn id_of(idx: &Index, h: &KnnHit) -> usize {
    idx.get_doc(&DocAddress::new(h.leaf as u32, h.doc)).unwrap().get(F_ID).unwrap().as_i64().unwrap() as usize
}

fn exact(c: &Corpus, q: &[f32], live: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut all: Vec<(usize, f32)> = (0..c.rows.len()).filter(|&i| live(i)).map(|i| (i, Similarity::L2.score(q, &c.rows[i]))).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(K).map(|x| x.0).collect()
}

fn recall(got: &[usize], want: &[usize]) -> f64 {
    let want: HashSet<usize> = want.iter().copied().collect();
    got.iter().filter(|g| want.contains(g)).count() as f64 / want.len().max(1) as f64
}

fn ids(idx: &Index, q: &[f32], cfg: &PlannerConfig) -> Vec<(usize, f32)> {
    let mut req = KnnRequest::new(F_V, q.to_vec(), K);
    req.num_candidates = 15;
    idx.knn_search_with(&req, cfg).unwrap().0.iter().map(|h| (id_of(idx, h), h.score)).collect()
}

/// Mean recall@K (num_candidates 1.5 K, ES's default), checking every hit
/// is live and accepted and scored with its `f32` similarity.
fn mean_recall(idx: &Index, c: &Corpus, cfg: &PlannerConfig, filter: Option<Query>, live: impl Fn(usize) -> bool + Copy) -> f64 {
    let mut total = 0.0;
    for q in &c.queries {
        let mut req = KnnRequest::new(F_V, q.clone(), K);
        req.num_candidates = 15;
        req.filter = filter.clone();
        let (hits, stats) = idx.knn_search_with(&req, cfg).unwrap();
        assert!(stats.graph_segments <= 1, "one shared index: {stats:?}");
        let got: Vec<usize> = hits.iter().map(|h| id_of(idx, h)).collect();
        assert!(got.iter().all(|&i| live(i)), "only live, accepted docs");
        for (h, &i) in hits.iter().zip(&got) {
            assert_eq!(h.score, Similarity::L2.score(q, &c.rows[i]), "rescored with f32");
        }
        total += recall(&got, &exact(c, q, live));
    }
    total / c.queries.len() as f64
}

#[test]
fn maintenance_trains_one_shared_index_and_grows_it_incrementally() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(6000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..1500);
    // Before any pass the rows are searched flat (and exactly).
    assert!(idx.vector_store().ivf_state(F_V).is_none());
    let cfg = PlannerConfig::default();
    assert!(mean_recall(&idx, &Corpus { rows: c.rows[..1500].to_vec(), queries: c.queries.clone() }, &cfg, None, |i| i < 1500) >= 0.99);
    let r = idx.optimize_vectors().unwrap();
    assert!(r.ivf_steps >= 1, "trained: {r:?}");
    let first = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(first.indexed().count(), 1500);
    let trained = first.num_centroids();
    // More rows: inserted into the same index, never re-clustered; the
    // payload seals and merges do not touch it.
    for b in 1..4 {
        add(&mut idx, &c, b * 1500..(b + 1) * 1500);
        idx.optimize_vectors().unwrap();
    }
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(state.indexed().count(), 6000);
    eprintln!("IVF_GROWTH trained={trained} now={} batches={}", state.num_centroids(), state.num_batches());
    for s in idx.vector_store().segments(F_V) {
        assert!(state.covers(s.id), "segment {} is searched through the index", s.id);
        assert!(!s.has_int8() && !s.has_int4() && !s.has_graph(), "no per-segment index or codes");
    }
    // (Recall at ES's default ratio is tested on the index itself with
    // ES's cluster size; small clusters here want a wider visit.)
    let r = mean_recall(&idx, &c, &cfg.with_visit_ratio(0.2), None, |_| true);
    assert!(r >= 0.9, "recall@{K} after incremental inserts {r}");
}

#[test]
fn below_the_flat_threshold_the_field_stays_flat() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(500);
    let mut idx = Index::new("ivf", schema(64, -1), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..150);
    idx.optimize_vectors().unwrap();
    assert!(idx.vector_store().ivf_state(F_V).is_none(), "150 <= 3 x 64: searched flat");
    add(&mut idx, &c, 150..500);
    idx.optimize_vectors().unwrap();
    assert!(idx.vector_store().ivf_state(F_V).is_some(), "500 > 192: indexed");
}

#[test]
fn deletes_and_filters_respect_liveness_and_compact_away() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(3000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..3000);
    idx.optimize_vectors().unwrap();
    let before = idx.vector_store().ivf_state(F_V).unwrap().entries();
    for i in (0..3000).step_by(3) {
        idx.delete_doc(&DocAddress::new(0, i as u32)).unwrap();
    }
    idx.refresh().unwrap();
    let live = |i: usize| i % 3 != 0;
    let cfg = PlannerConfig::default().with_visit_ratio(0.2);
    assert!(mean_recall(&idx, &c, &cfg, None, live) >= 0.9);
    let r = mean_recall(&idx, &c, &cfg, Some(Query::term(F_TAG, "even")), |i| live(i) && i % 2 == 0);
    assert!(r >= 0.85, "filtered recall {r}");
    // Fewer matches than centroids: the matches are scored exactly.
    let r = mean_recall(&idx, &c, &cfg, Some(Query::term(F_RARE, "yes")), |i| live(i) && i % 251 == 0);
    assert!(r >= 0.99, "selective filter recall {r}");
    // Expunging deletes compacts the dead entries away.
    idx.force_merge_vectors(1, true).unwrap();
    let after = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(after.entries() < before * 3 / 4, "{before} -> {}", after.entries());
    assert_eq!(after.num_batches(), 1);
    assert!(mean_recall(&idx, &c, &cfg, None, live) >= 0.9);
}

#[test]
fn growth_past_twice_the_trained_rows_retrains_on_every_row() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(8000);
    let mut idx = Index::new("ivf", schema(64, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..400);
    let job = idx.vector_optimizer();
    job.index_pending().unwrap();
    let trained = idx.vector_store().ivf_state(F_V).unwrap().num_centroids();
    add(&mut idx, &c, 400..8000);
    let job = idx.vector_optimizer();
    job.index_pending().unwrap();
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(state.num_centroids() > trained, "{trained} -> {} centroids", state.num_centroids());
    assert_eq!(state.num_batches(), 1, "one fresh batch");
    assert_eq!(state.indexed().count(), 8000);
    assert!(idx.vector_store().segments(F_V).iter().all(|s| state.covers(s.id)), "the retrained index covers every segment at install");
    let r = mean_recall(&idx, &c, &PlannerConfig::default().with_visit_ratio(0.2), None, |_| true);
    assert!(r >= 0.9, "recall@{K} after the retrain {r}");
}

/// A retrain too large for one step is skipped and recorded: later
/// inserts do not list every row again until the index doubles once more.
#[test]
fn a_retrain_too_large_for_one_step_is_skipped_once() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    use crate::vector::store::{IVF_RETRAIN_ATTEMPTS, SKIP_IVF_RETRAIN};
    let c = corpus(8000);
    let mut idx = Index::new("ivf", schema(64, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..400);
    idx.vector_optimizer().index_pending().unwrap();
    let trained = idx.vector_store().ivf_state(F_V).unwrap().num_centroids();
    SKIP_IVF_RETRAIN.with(|f| f.set(true));
    IVF_RETRAIN_ATTEMPTS.with(|n| n.set(0));
    add(&mut idx, &c, 400..1000);
    idx.vector_optimizer().index_pending().unwrap();
    add(&mut idx, &c, 1000..1500);
    idx.vector_optimizer().index_pending().unwrap();
    let attempts = IVF_RETRAIN_ATTEMPTS.with(std::cell::Cell::get);
    SKIP_IVF_RETRAIN.with(|f| f.set(false));
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(attempts, 1, "the skip is recorded");
    assert_eq!(state.num_centroids(), trained, "inserted, not retrained");
    assert_eq!(state.indexed().count(), 1500);
    // Past twice the recorded size it is considered again (and fits now).
    add(&mut idx, &c, 1500..8000);
    idx.vector_optimizer().index_pending().unwrap();
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(state.num_centroids() > trained, "retrained past the next doubling");
    assert!(idx.vector_store().segments(F_V).iter().all(|s| state.covers(s.id)), "a retrained index covers every segment");
}

#[test]
fn background_maintenance_splits_growing_clusters_in_bounded_steps() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let mut c = corpus(8000);
    // Growth concentrated on a few clusters: near copies of 20 rows.
    let mut r = crate::vector::ivf::Rng::new(5);
    for i in 4000..8000 {
        let v: Vec<f32> = c.rows[i % 20].iter().map(|x| x + (r.gaussian() as f32) * 1e-3).collect();
        c.rows[i] = v;
    }
    let mut idx = Index::new("ivf", schema(64, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..4000);
    let job = idx.vector_optimizer();
    job.index_pending().unwrap();
    let trained = idx.vector_store().ivf_state(F_V).unwrap().num_centroids();
    // Short of doubling: inserted into the trained centroids.
    add(&mut idx, &c, 4000..7900);
    let job = idx.vector_optimizer();
    job.index_pending().unwrap();
    assert_eq!(idx.vector_store().ivf_state(F_V).unwrap().num_centroids(), trained, "inserts below twice the trained rows do not re-cluster");
    let mut steps = 0;
    for _ in 0..400 {
        let r = job.maintain_once().unwrap();
        if r.ivf_steps == 0 && r.vacuumed + r.merged == 0 {
            break;
        }
        assert!(r.ivf_steps <= 1, "one bounded step per pass");
        steps += r.ivf_steps;
    }
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(steps > 0 && state.num_centroids() > trained, "{steps} steps, {trained} -> {} centroids", state.num_centroids());
    assert!(state.num_batches() <= crate::vector::ivf_field::MAX_BATCHES, "{} batches", state.num_batches());
    let r = mean_recall(&idx, &c, &PlannerConfig::default().with_visit_ratio(0.2), None, |_| true);
    assert!(r >= 0.9, "recall@{K} after splits {r}");
}

#[test]
fn reopen_reads_the_index_and_catches_up_rows_past_its_checkpoint() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(4000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..3000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let committed = idx.vector_store().ivf_state(F_V).unwrap().manifest();
    // Rows past the checkpoint: committed (as chunks), not yet indexed.
    add(&mut idx, &c, 3000..4000);
    idx.commit().unwrap();
    let cfg = PlannerConfig::default();
    let before: Vec<_> = c.queries.iter().map(|q| ids(&idx, q, &cfg)).collect();
    drop(idx);
    let mut idx = Index::open(dir.path()).unwrap();
    idx.optimizer_policy = policy();
    let state = idx.vector_store().ivf_state(F_V).expect("the committed index is reopened");
    assert_eq!(state.manifest(), committed, "read back, not rebuilt");
    assert_eq!(state.indexed().count(), 3000);
    // Unindexed rows are searched flat meanwhile: same results.
    let after: Vec<_> = c.queries.iter().map(|q| ids(&idx, q, &cfg)).collect();
    assert_eq!(after, before);
    let wide = cfg.with_visit_ratio(0.2);
    assert!(mean_recall(&idx, &c, &wide, None, |_| true) >= 0.9);
    // Catch-up: the next pass indexes the rest.
    idx.optimize_vectors().unwrap();
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(state.indexed().count(), 4000);
    assert!(idx.vector_store().segments(F_V).iter().all(|s| state.covers(s.id)));
    assert!(mean_recall(&idx, &c, &wide, None, |_| true) >= 0.9);
    // The replaced files go at the commit; nothing unreferenced is left.
    idx.commit().unwrap();
    let live = idx.vector_store().ivf_files().unwrap();
    let on_disk: HashSet<String> = std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .filter(|n| crate::vector::ivf_field::is_ivf_file(n)).collect();
    assert_eq!(&on_disk, &*live);
}

#[test]
fn retained_views_keep_their_index_version() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(4000);
    let mut idx = Index::new("ivf", schema(64, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    let captured = idx.vector_store().capture_read_store().unwrap();
    let cfg = PlannerConfig::default();
    let search = |store: &crate::vector::store::VectorStore| -> Vec<Vec<(u64, f32)>> {
        c.queries.iter().map(|q| store.try_search(F_V, q, K, 15, None, &cfg).unwrap().0.iter().map(|h| (h.point, h.score)).collect()).collect()
    };
    let before = search(&*captured);
    // Deletes, inserts, splits and compaction on the live index.
    for i in (0..2000).step_by(2) {
        idx.delete_doc(&DocAddress::new(0, i as u32)).unwrap();
    }
    add(&mut idx, &c, 2000..4000);
    idx.force_merge_vectors(1, false).unwrap();
    let live = idx.vector_store().ivf_state(F_V).unwrap();
    assert_ne!(live.manifest(), captured.ivf_state(F_V).unwrap().manifest(), "the live index moved on");
    assert_eq!(search(&*captured), before, "the view searches the index it captured");
}

#[test]
fn file_recovery_ships_the_index_and_the_target_opens_it_in_place() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(3000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..3000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let snapshot = idx.snapshot_commit().unwrap();
    let manifest = idx.vector_store().ivf_state(F_V).unwrap().manifest();
    // The engine's peer recovery pins the commit (`pin_commit`).
    let pinned: Vec<String> = idx.pin_commit().unwrap().iter().map(|f| f.name.clone()).collect();
    for name in manifest.files() {
        assert!(snapshot.manifest().get(&name).is_some(), "{name} shipped");
        assert!(pinned.contains(&name), "{name} pinned for file recovery");
    }
    let target = tempfile::tempdir().unwrap();
    let dst = crate::codec::directory::MmapDirectory::open(target.path()).unwrap();
    snapshot.copy_to(&dst).unwrap();
    let mut copy = Index::open(target.path()).unwrap();
    copy.optimizer_policy = policy();
    assert_eq!(copy.vector_store().ivf_state(F_V).unwrap().manifest(), manifest, "opened, not rebuilt");
    let cfg = PlannerConfig::default();
    for q in &c.queries {
        assert_eq!(ids(&copy, q, &cfg), ids(&idx, q, &cfg));
    }
    assert_eq!(copy.optimize_vectors().unwrap().ivf_steps, 0, "nothing to train or insert on the target");
}

/// Only centroids and small per-cluster metadata (plus one bit a vector)
/// are resident; posting lists stay in the mapping.
#[test]
fn resident_index_memory_is_about_the_centroids() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(24_000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    let mut seen = Vec::new();
    for b in 0..4 {
        add(&mut idx, &c, b * 6000..(b + 1) * 6000);
        idx.optimize_vectors().unwrap();
        let state = idx.vector_store().ivf_state(F_V).unwrap();
        let n = (b + 1) * 6000;
        let centroid_bytes = state.num_centroids() * DIM * 4;
        let heap = state.heap_bytes();
        eprintln!("IVF_HEAP n={n} centroids={} centroid_bytes={centroid_bytes} heap={heap} postings_mapped={}", state.num_centroids(), state.posting_bytes());
        // Centroids, run tables and lists, and one bit per point.
        assert!(heap <= centroid_bytes * 3 / 2 + n / 8 + 64 * 1024, "heap {heap} for {centroid_bytes} bytes of centroids at n={n}");
        assert!(state.posting_bytes() > 4 * heap, "posting lists are not resident: {} mapped vs {heap} heap", state.posting_bytes());
        // Charged to the accounting breaker (other tests charge too).
        assert!(crate::accounting::bytes(crate::accounting::Category::VectorGraphs) >= heap as u64);
        seen.push((n, heap, centroid_bytes));
    }
    // Grows with the centroids, not with the vectors' size.
    let (n0, h0, _) = seen[0];
    let (n3, h3, _) = seen[3];
    assert!((h3 as f64 / n3 as f64) <= 1.5 * (h0 as f64 / n0 as f64) + 1.0, "{seen:?}");
    // A cold reopen (fresh mappings) searches as before.
    idx.commit().unwrap();
    let cfg = PlannerConfig::default();
    let before: Vec<_> = c.queries.iter().map(|q| ids(&idx, q, &cfg)).collect();
    drop(idx);
    let idx = Index::open(dir.path()).unwrap();
    let after: Vec<_> = c.queries.iter().map(|q| ids(&idx, q, &cfg)).collect();
    assert_eq!(after, before);
}

/// A payload swap (vacuum) landing while searches run: every search sees
/// either side of it whole, never a candidate whose segment it misses.
#[test]
fn a_concurrent_segment_swap_loses_no_hits() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(3000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..3000);
    idx.optimize_vectors().unwrap();
    let store = idx.vector_store().clone();
    let cfg = PlannerConfig::default();
    let search = |q: &[f32]| -> Vec<(u64, f32)> { store.try_search(F_V, q, K, 15, None, &cfg).unwrap().0.iter().map(|h| (h.point, h.score)).collect() };
    let expected: Vec<Vec<(u64, f32)>> = c.queries.iter().map(|q| search(q)).collect();
    assert!(expected.iter().all(|e| e.len() == K));
    let job = idx.vector_optimizer();
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..30 {
                let id = store.segments(F_V)[0].id;
                job.rewrite_segment(id).unwrap();
            }
            done.store(true, std::sync::atomic::Ordering::Release);
        });
        let mut rounds = 0;
        while !done.load(std::sync::atomic::Ordering::Acquire) || rounds == 0 {
            for (q, want) in c.queries.iter().zip(&expected) {
                assert_eq!(&search(q), want, "a swap mid-search changed the hits");
            }
            rounds += 1;
        }
    });
    let state = store.ivf_state(F_V).unwrap();
    // Payload IDs change on rewrite, while immutable IVF postings retain
    // stable point IDs. The segment-ID covered set is only an optimization;
    // uncached replacement segments fall back to checking indexed points.
    assert!(store.segments(F_V).iter().all(|s| s.points().iter().all(|&point| state.indexed().contains(point))),
        "every replacement point remains indexed");
    assert_eq!(state.indexed().count(), 3000, "rewriting payloads neither loses nor duplicates indexed points");
    for (q, want) in c.queries.iter().zip(&expected) {
        assert_eq!(&search(q), want, "final replacement preserves every exact hit and score");
    }
}

/// A vacuum or merge with no new rows (no insert step after it) keeps the
/// swapped-in segment covered: queries never scan it row by row, and the
/// hits are unchanged.
#[test]
fn payload_vacuums_and_merges_keep_segments_covered() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(3000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..1500);
    idx.vector_optimizer().index_pending().unwrap();
    add(&mut idx, &c, 1500..3000);
    idx.vector_optimizer().index_pending().unwrap();
    let store = idx.vector_store().clone();
    let cfg = PlannerConfig::default().with_visit_ratio(0.2);
    let search = || -> (Vec<Vec<(u64, f32)>>, usize) {
        let mut flat = 0;
        let hits = c.queries.iter().map(|q| {
            let (hits, stats) = store.try_search(F_V, q, K, 15, None, &cfg).unwrap();
            flat += stats.exact_segments;
            hits.iter().map(|h| (h.point, h.score)).collect()
        }).collect();
        (hits, flat)
    };
    let segments = store.segments(F_V);
    assert!(segments.len() >= 2, "two sealed segments to merge: {}", segments.len());
    let state = store.ivf_state(F_V).unwrap();
    assert!(segments.iter().all(|s| state.covers(s.id)));
    let (before, flat) = search();
    assert_eq!(flat, 0, "covered segments are searched only through the index");
    let job = idx.vector_optimizer();
    // A vacuum.
    job.rewrite_segment(segments[0].id).unwrap();
    let state = store.ivf_state(F_V).unwrap();
    assert!(store.segments(F_V).iter().all(|s| state.covers(s.id)), "the vacuumed segment stays covered");
    assert_eq!(search(), (before.clone(), 0), "no row-by-row scan after a vacuum");
    // A merge of every segment.
    let ids: Vec<u32> = store.segments(F_V).iter().map(|s| s.id).collect();
    job.merge_segments(&ids).unwrap();
    assert_eq!(store.segments(F_V).len(), 1);
    let state = store.ivf_state(F_V).unwrap();
    assert!(store.segments(F_V).iter().all(|s| state.covers(s.id)), "the merged segment is covered");
    assert_eq!(search(), (before, 0), "no row-by-row scan after a merge");
}

/// More than 2 x cluster_size identical vectors: the cluster cannot be
/// split; it is marked, and drops and compaction still run (force merges
/// and expunges finish with one batch).
#[test]
fn an_unsplittable_cluster_does_not_stop_maintenance() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let mut c = corpus(2000);
    let same = c.rows[0].clone();
    let mut rows = vec![same; 800];
    rows.extend(c.rows.drain(..));
    let c = Corpus { rows, queries: c.queries };
    let mut idx = Index::new("ivf", schema(64, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2800);
    idx.optimize_vectors().unwrap();
    for i in (800..2800).step_by(3) {
        idx.delete_doc(&DocAddress::new(0, i as u32)).unwrap();
    }
    idx.refresh().unwrap();
    let job = idx.vector_optimizer();
    let mut passes = 0;
    while job.maintain_once().unwrap() != Default::default() {
        passes += 1;
        assert!(passes < 400, "maintenance never settles");
    }
    let before = idx.vector_store().ivf_state(F_V).unwrap().entries();
    let r = idx.force_merge_vectors(1, true).unwrap();
    assert!(r.ivf_steps <= 3, "one compaction, not one per step: {r:?}");
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(state.num_batches(), 1, "expunge compacted the batches");
    assert!(state.entries() <= before);
    assert!(mean_recall(&idx, &c, &PlannerConfig::default().with_visit_ratio(0.2), None, |i| i < 800 || (i - 800) % 3 != 0) >= 0.5);
}

/// A field that leaves `bbq_disk` loses its index at the next step, and a
/// build cannot install an index for options that changed meanwhile.
#[test]
fn a_mapping_change_retires_the_index() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(2000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    assert!(idx.vector_store().ivf_state(F_V).is_some());
    let job = idx.vector_optimizer();
    idx.vector_store().set_field_options(F_V, VectorFieldOptions::quantized_hnsw(Quantization::Binary).flat());
    job.run().unwrap();
    assert!(idx.vector_store().ivf_state(F_V).is_none(), "no longer bbq_disk: no index");
}

/// New code parameters (`bits` 1 -> 4): the stale index serves until a
/// replacement, trained in maintenance, takes its place; the next commit
/// retires the stale files.
#[test]
fn new_code_parameters_rebuild_the_index_while_the_old_one_serves() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(2000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let old = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(old.params().bits, 1);
    let job = idx.vector_optimizer();
    let four = VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_osq(true).flat().with_oversample(3.0)
        .with_ivf(Some(IvfOptions { cluster_size: 128, flat_index_threshold: 0, bits: 4, ..IvfOptions::default() }));
    idx.vector_store().set_field_options(F_V, four);
    let cfg = PlannerConfig::default().with_visit_ratio(0.2);
    assert!(std::sync::Arc::ptr_eq(&idx.vector_store().ivf_state(F_V).unwrap(), &old), "the stale index still serves");
    assert!(mean_recall(&idx, &c, &cfg, None, |_| true) >= 0.9);
    assert!(job.run().unwrap().ivf_steps >= 1);
    let new = idx.vector_store().ivf_state(F_V).unwrap();
    assert_eq!(new.params().bits, 4);
    assert!(new.indexed().count() >= 2000, "every row is in the replacement");
    assert!(mean_recall(&idx, &c, &cfg, None, |_| true) >= 0.9);
    let stale: Vec<String> = old.manifest().files().into_iter().filter(|n| !new.manifest().files().contains(n)).collect();
    assert!(!stale.is_empty());
    idx.commit().unwrap();
    for name in &stale {
        assert!(!dir.path().join(name).exists(), "{name} retired by the commit");
    }
    drop(idx);
    let idx = Index::open(dir.path()).unwrap();
    assert_eq!(idx.vector_store().ivf_state(F_V).unwrap().params().bits, 4, "the replacement is what the commit lists");
}

/// An index whose files cannot be read: the field is searched flat, the
/// commit point's files survive the orphan sweep (so it can still be
/// pinned and recovered), and the next commit retires them.
#[test]
fn an_unreadable_index_keeps_the_commit_recoverable() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(2000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let manifest = idx.vector_store().ivf_state(F_V).unwrap().manifest();
    drop(idx);
    let batch = dir.path().join(crate::vector::ivf_field::batch_file_name(F_V, manifest.batches[0]));
    std::fs::write(&batch, b"damaged").unwrap();
    let mut idx = Index::open(dir.path()).unwrap();
    idx.optimizer_policy = policy();
    assert!(idx.vector_store().ivf_state(F_V).is_none());
    idx.sweep_orphan_files().unwrap();
    for name in manifest.files() {
        assert!(dir.path().join(&name).exists(), "{name} kept while meta.json lists it");
    }
    idx.snapshot_commit().unwrap();
    assert!(mean_recall(&idx, &c, &PlannerConfig::default(), None, |_| true) >= 0.99, "searched flat");
    assert!(idx.vector_store().has_uncommitted_changes());
    idx.commit().unwrap();
    for name in manifest.files() {
        assert!(!dir.path().join(&name).exists(), "{name} retired by the next commit");
    }
}

/// A build voided by a mapping change (bits 1 -> 4) retires its files;
/// when the mapping changes back, the next step must not reuse their
/// numbers, or the commit would delete a file the index lists.
#[test]
fn a_voided_build_never_lends_its_file_numbers_to_a_live_index() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(3000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let store = idx.vector_store().clone();
    let base = store.ivf_state(F_V).unwrap();
    let one = options(128, 0);
    let four = VectorFieldOptions::quantized_hnsw(Quantization::Binary).with_osq(true).flat().with_oversample(3.0)
        .with_ivf(Some(IvfOptions { cluster_size: 128, flat_index_threshold: 0, bits: 4, ..IvfOptions::default() }));
    // New rows; their insert step is built, then the mapping changes: void.
    add(&mut idx, &c, 2000..2500);
    let during = store.clone();
    crate::vector::store::AFTER_IVF_BUILD.with(|hook| *hook.borrow_mut() = Some(Box::new(move || during.set_field_options(F_V, four))));
    let job = idx.vector_optimizer();
    job.index_pending().unwrap();
    crate::vector::store::AFTER_IVF_BUILD.with(|hook| hook.borrow_mut().take());
    assert!(std::sync::Arc::ptr_eq(&store.ivf_state(F_V).unwrap(), &base), "the voided build was not installed");
    // Back to 1 bit: the same index takes the rows, in new files.
    store.set_field_options(F_V, one);
    job.index_pending().unwrap();
    let state = store.ivf_state(F_V).unwrap();
    assert_eq!(state.indexed().count(), 2500);
    idx.commit().unwrap();
    for name in state.manifest().files() {
        assert!(dir.path().join(&name).exists(), "{name} is listed by the commit and must not be deleted");
    }
    let manifest = state.manifest();
    drop(state);
    drop(base);
    drop(job);
    drop(store);
    drop(idx);
    let idx = Index::open(dir.path()).unwrap();
    assert_eq!(idx.vector_store().ivf_state(F_V).expect("the committed index reopens").manifest(), manifest);
}

/// An index that cannot be opened for want of memory does not fail the
/// index open: the field is searched flat, commits keep its files, and the
/// next maintenance step opens it (not a retrain).
#[test]
fn an_index_refused_memory_at_open_is_deferred_not_fatal() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(2000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let manifest = idx.vector_store().ivf_state(F_V).unwrap().manifest();
    drop(idx);
    crate::vector::store::FAIL_IVF_OPEN.with(|f| f.set(true));
    let opened = Index::open(dir.path());
    crate::vector::store::FAIL_IVF_OPEN.with(|f| f.set(false));
    let mut idx = opened.expect("a refused IVF open does not fail the index open");
    idx.optimizer_policy = policy();
    assert!(idx.vector_store().ivf_state(F_V).is_none());
    assert!(mean_recall(&idx, &c, &PlannerConfig::default(), None, |_| true) >= 0.99, "searched flat");
    let live = idx.vector_store().ivf_files().unwrap();
    assert!(manifest.files().iter().all(|n| live.contains(n)), "deferred files stay live");
    idx.sweep_orphan_files().unwrap();
    idx.commit().unwrap();
    for name in manifest.files() {
        assert!(dir.path().join(&name).exists(), "{name} kept while deferred");
    }
    // Maintenance opens the deferred index (the same files), then indexes nothing new.
    idx.optimize_vectors().unwrap();
    let state = idx.vector_store().ivf_state(F_V).expect("opened by maintenance");
    assert_eq!(state.manifest(), manifest, "opened, not retrained");
    assert!(idx.vector_store().segments(F_V).iter().all(|s| state.covers(s.id)));
    drop(state);
    drop(idx);
    let idx = Index::open(dir.path()).unwrap();
    assert_eq!(idx.vector_store().ivf_state(F_V).unwrap().manifest(), manifest, "the commit kept listing it");
}

/// A deferred open that maintenance retries in vain backs off: the next
/// rounds skip it until its retry is due.
#[test]
fn a_deferred_open_backs_off_between_failed_retries() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    use crate::vector::store::{FAIL_IVF_OPEN, IVF_OPEN_ATTEMPTS};
    let c = corpus(2000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..2000);
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    drop(idx);
    FAIL_IVF_OPEN.with(|f| f.set(true));
    let mut idx = Index::open(dir.path()).unwrap();
    idx.optimizer_policy = policy();
    IVF_OPEN_ATTEMPTS.with(|n| n.set(0));
    let job = idx.vector_optimizer();
    for _ in 0..5 {
        job.maintain_once().unwrap();
    }
    let attempts = IVF_OPEN_ATTEMPTS.with(std::cell::Cell::get);
    FAIL_IVF_OPEN.with(|f| f.set(false));
    assert_eq!(attempts, 1, "one retry, then backed off");
    assert!(idx.vector_store().ivf_state(F_V).is_none());
}

/// The default visit ratio is ES's: computed from the ES-adjusted
/// `num_candidates` (`max(num_candidates, ceil(k * oversample))`) and the
/// field's live vectors (dead ones excluded).
#[test]
fn the_default_visit_ratio_uses_es_candidates_and_live_vectors() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    use crate::vector::ivf::{dynamic_visit_ratio, segment_size_cap};
    let c = corpus(6000);
    let mut idx = Index::new("ivf", schema(128, 0), IndexSettings::default());
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..6000);
    idx.optimize_vectors().unwrap();
    for i in (0..6000).step_by(2) {
        idx.delete_doc(&DocAddress::new(0, i as u32)).unwrap();
    }
    idx.refresh().unwrap();
    let ops = |cfg: &PlannerConfig| -> u64 {
        c.queries.iter().map(|q| {
            let mut req = KnnRequest::new(F_V, q.clone(), K);
            req.num_candidates = 15;
            idx.knn_search_with(&req, cfg).unwrap().1.vector_ops
        }).sum()
    };
    let es = dynamic_visit_ratio(30, K).min(segment_size_cap(3000));
    let default = PlannerConfig::default();
    assert_eq!(ops(&default), ops(&default.with_visit_ratio(es)), "the default ratio is ES's for 30 candidates over 3000 live vectors");
}

/// Doc ids to addresses over every leaf (ids are never reused).
fn addresses(idx: &Index) -> std::collections::HashMap<usize, DocAddress> {
    let mut out = std::collections::HashMap::new();
    let leaves = idx.leaves();
    for (l, leaf) in leaves.iter().enumerate() {
        for d in 0..leaf.num_docs() {
            let addr = DocAddress::new(l as u32, d);
            if let Ok(doc) = idx.get_doc(&addr) {
                if let Some(id) = doc.get(F_ID).and_then(|v| v.as_i64()) {
                    out.insert(id as usize, addr);
                }
            }
        }
    }
    out
}

/// Update churn (deletes plus new vectors, maintenance between rounds):
/// recall holds, garbage stays bounded, and a reopen neither re-inserts
/// dead points nor leaves segments to flat scans.
#[test]
fn churn_and_reopen_keep_the_index_lean() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let c = corpus(6000);
    let dir = tempfile::tempdir().unwrap();
    let mut idx = Index::create(dir.path(), "ivf", schema(128, 0), IndexSettings::default()).unwrap();
    idx.optimizer_policy = policy();
    add(&mut idx, &c, 0..3000);
    idx.optimize_vectors().unwrap();
    let mut live: Vec<usize> = (0..3000).collect();
    for round in 0..5 {
        let addr = addresses(&idx);
        let gone: Vec<usize> = live.iter().copied().filter(|i| i % 5 == round % 5).take(600).collect();
        for i in &gone {
            idx.delete_doc(&addr[i]).unwrap();
        }
        live.retain(|i| !gone.contains(i));
        add(&mut idx, &c, 3000 + round * 600..3000 + (round + 1) * 600);
        live.extend(3000 + round * 600..3000 + (round + 1) * 600);
        idx.optimize_vectors().unwrap();
    }
    let alive: HashSet<usize> = live.iter().copied().collect();
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(state.entries() as f64 <= 2.6 * alive.len() as f64, "garbage bounded: {} entries for {} live", state.entries(), alive.len());
    let wide = PlannerConfig::default().with_visit_ratio(0.2);
    assert!(mean_recall(&idx, &c, &wide, None, |i| alive.contains(&i)) >= 0.9);
    idx.commit().unwrap();
    let entries = state.entries();
    drop(state);
    drop(idx);
    let mut idx = Index::open(dir.path()).unwrap();
    idx.optimizer_policy = policy();
    idx.optimize_vectors().unwrap();
    let state = idx.vector_store().ivf_state(F_V).unwrap();
    assert!(state.entries() <= entries, "no dead point re-inserted: {entries} -> {}", state.entries());
    assert!(idx.vector_store().segments(F_V).iter().all(|s| state.covers(s.id)), "every segment covered after one pass");
    assert!(mean_recall(&idx, &c, &wide, None, |i| alive.contains(&i)) >= 0.9);
}
