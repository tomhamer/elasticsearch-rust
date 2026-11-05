//! End-to-end tests of the vector store through [`Index`].

use std::collections::HashSet;

use crate::codec::directory::{Directory, RamDirectory};
use crate::index::{Index, KnnHit, KnnRequest};
use crate::query::query::Query;
use crate::query::reader::LeafReader;
use crate::query::searcher::{SearchOptions, Searcher};
use crate::segment::SegmentWriter;
use crate::types::*;
use crate::vector::hnsw::{HnswBuilder, HnswParams};
use crate::vector::store::{OptimizerPolicy, PlannerConfig, VectorStore};
use crate::vector::Similarity;

const DIM: usize = 16;
const F_ID: u16 = 0;
const F_TAG: u16 = 1;
const F_EMB: u16 = 2;
const F_EMB2: u16 = 3;

#[test]
fn retained_leaf_view_keeps_masks_vectors_and_owners_after_delete_and_merge() {
    let mut index = new_index();
    add_batch(&mut index, 0..100);
    add_batch(&mut index, 100..200);
    let captured = crate::segment::Segment::capture_read_view(index.segments()).unwrap();
    let store = captured[0].vector_store().unwrap();
    assert!(captured.iter().all(|leaf| std::sync::Arc::ptr_eq(leaf.vector_store().unwrap(), store)));
    assert!(!std::sync::Arc::ptr_eq(store, index.vector_store()));
    let query = rvec(37, DIM);
    let before = store.try_search(F_EMB, &query, 10, 100, None, &PlannerConfig::exact()).unwrap().0;
    let vector = captured[0].vector(F_EMB, 37).unwrap();
    let owner = store.owner(before[0].point);
    index.delete_doc(&DocAddress::new(0, 37)).unwrap();
    add_batch(&mut index, 200..300);
    index.force_merge().unwrap();
    assert!(!captured[0].is_deleted(37));
    assert_eq!(captured[0].num_live_docs(), 100);
    assert_eq!(captured[0].vector(F_EMB, 37).unwrap(), vector);
    assert_eq!(store.owner(before[0].point), owner);
    assert_eq!(store.try_search(F_EMB, &query, 10, 100, None, &PlannerConfig::exact()).unwrap().0, before);
    let newest = crate::segment::Segment::capture_read_view(index.segments()).unwrap();
    assert_eq!(newest.iter().map(|leaf| leaf.num_live_docs()).sum::<u32>(), 299);
}

#[test]
fn parallel_sparse_exact_preserves_scores_ties_and_operation_counts() {
    use crate::vector::segment::{SealSpec, VectorSegment};
    let mut vectors: Vec<Vec<f32>> = (0..12_000).map(|i| rvec(i, 128)).collect();
    for vector in vectors.iter_mut().step_by(200) { vector.fill(0.0); }
    let rows: Vec<_> = vectors.iter().enumerate().map(|(i, v)| (i as u64, v.as_slice())).collect();
    let ordinals: Vec<u32> = (0..12_000).step_by(2).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for similarity in [Similarity::Cosine, Similarity::L2, Similarity::DotProduct] {
        let directory = RamDirectory::new();
        let segment = VectorSegment::write_with(&directory, 0, F_EMB, similarity, 128, &rows, SealSpec::chunk()).unwrap();
        for query in [rvec(42, 128), vec![0.0; 128]] {
            for k in [0, 1, 10, 500] {
                let expected = crate::vector::flat::exact_top_k(&query, similarity, k,
                    ordinals.iter().map(|&o| (o, segment.vector(o))));
                // External callers use the dedicated search pool; Rayon
                // callers below must take the safe sequential fallback.
                let before = crate::vector::hnsw::vector_ops();
                assert_eq!(segment.search_exact(&query, k, ordinals.iter().copied()), expected);
                assert_eq!(crate::vector::hnsw::vector_ops() - before,
                    if k == 0 { 0 } else { ordinals.len() as u64 });
                pool.install(|| {
                    let before = crate::vector::hnsw::vector_ops();
                    let actual = segment.search_exact(&query, k, ordinals.iter().copied());
                    assert_eq!(actual, expected, "{similarity:?} k={k}");
                    assert_eq!(crate::vector::hnsw::vector_ops() - before,
                        if k == 0 { 0 } else { ordinals.len() as u64 });
                });
            }
        }
    }
}

fn schema() -> Vec<Field> {
    vec![
        Field::new(F_ID, "id", FieldType::I64),
        Field::new(F_TAG, "tag", FieldType::Keyword),
        Field::new(F_EMB, "emb", FieldType::Vector { dim: DIM as u16 }).with_similarity("cosine"),
        Field::new(F_EMB2, "emb2", FieldType::Vector { dim: 4 }).with_similarity("l2_norm"),
    ]
}

/// Deterministic pseudo-random vector for `seed`.
fn rvec(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn doc(i: i64) -> Document {
    let mut d = Document::new();
    d.add_i64(F_ID, i);
    d.add_keyword(F_TAG, if i % 10 == 0 { "ten" } else { "other" });
    d.add_vector(F_EMB, rvec(i as u64, DIM));
    if i % 3 == 0 {
        d.add_vector(F_EMB2, vec![i as f32, 0.0, 0.0, 0.0]);
    }
    d
}

/// Small graphs in debug builds (HNSW construction is ~20x slower there).
fn policy() -> OptimizerPolicy {
    OptimizerPolicy {
        indexing_threshold: 1_000_000,
        // Count-driven in these tests: no byte / full-scan / idle sealing.
        indexing_threshold_kb: usize::MAX,
        seal_full_scan_threshold: usize::MAX,
        seal_idle: None,
        vacuum_deleted_ratio: 0.2,
        max_segments: 8,
        // These tests pin the count-capped behaviour they were written for.
        payload_merge: crate::vector::store::PayloadMerge::Capped,
        hnsw: HnswParams { m: 12, ef_construction: 64, ..HnswParams::default() },
        quantize: false,
    }
}

/// [`schema`] with every vector field explicitly mapped to f32 HNSW, the
/// mapping the mutable per-field graph owner requires.
fn supported_schema() -> Vec<Field> {
    schema().into_iter().map(|mut field| {
        if matches!(field.field_type, FieldType::Vector { .. }) {
            let mut options = crate::vector::store::VectorFieldOptions::f32_hnsw();
            options.m = Some(policy().hnsw.m);
            options.ef_construction = Some(policy().hnsw.ef_construction);
            field.vector_options = Some(options);
        }
        field
    }).collect()
}

fn activate_test_index(mut idx: Index) -> Index {
    idx.optimizer_policy = policy();
    idx.enable_mutable_vectors().unwrap();
    idx
}

fn new_index() -> Index {
    activate_test_index(Index::new("v", supported_schema(), IndexSettings::default()))
}

fn create_index(path: &std::path::Path, name: &str) -> Index {
    activate_test_index(Index::create(path, name, supported_schema(), IndexSettings::default()).unwrap())
}

fn add_batch(idx: &mut Index, range: std::ops::Range<i64>) {
    for i in range {
        idx.add_document(&doc(i)).unwrap();
    }
    idx.refresh().unwrap();
}

/// Stable id of a hit (the stored `id` field).
fn id_of(idx: &Index, h: &KnnHit) -> i64 {
    idx.get_doc(&DocAddress::new(h.leaf as u32, h.doc)).unwrap().get(F_ID).unwrap().as_i64().unwrap()
}

fn ids_scores(idx: &Index, hits: &[KnnHit]) -> Vec<(i64, f32)> {
    hits.iter().map(|h| (id_of(idx, h), h.score)).collect()
}

/// Brute-force top-k ids over `live` ids.
fn brute(live: &[i64], q: &[f32], k: usize) -> Vec<i64> {
    let mut all: Vec<(i64, f32)> = live.iter().map(|&i| (i, Similarity::Cosine.score(q, &rvec(i as u64, DIM)))).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|x| x.0).collect()
}

fn recall(got: &[i64], want: &[i64]) -> f32 {
    got.iter().filter(|g| want.contains(g)).count() as f32 / want.len().max(1) as f32
}

fn batch() -> i64 {
    if cfg!(debug_assertions) { 300 } else { 1500 }
}

#[test]
fn text_merge_builds_no_graph_and_preserves_knn() {
    let mut idx = new_index();
    let n = batch();
    // Two batches sealed into one payload segment, two more left as chunks;
    // all four text segments share the field's single mutable graph.
    add_batch(&mut idx, 0..n);
    add_batch(&mut idx, n..2 * n);
    idx.optimizer_policy.indexing_threshold = (n as usize) + 1;
    let r = idx.optimize_vectors().unwrap();
    assert_eq!(r.sealed, 1);
    add_batch(&mut idx, 2 * n..3 * n);
    add_batch(&mut idx, 3 * n..4 * n);
    for i in (0..4 * n).step_by(7) {
        let leaf = (i / n) as u32;
        idx.delete_doc(&DocAddress::new(leaf, (i % n) as u32)).unwrap();
    }
    assert_eq!(idx.num_segments(), 4);
    let queries: Vec<Vec<f32>> = (0..8).map(|q| rvec(10_000 + q, DIM)).collect();
    let cfg = PlannerConfig { full_scan_threshold: 50, full_scan_fraction: 0.0, ..PlannerConfig::default() };
    let before: Vec<_> = queries
        .iter()
        .map(|q| {
            let (h, st) = idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &cfg).unwrap();
            assert_eq!(st.graph_segments, 1, "sealed segment searched through the graph");
            ids_scores(&idx, &h)
        })
        .collect();
    // Per-leaf Query::Knn path too.
    let leaf_path = |idx: &Index| -> Vec<Vec<i64>> {
        queries
            .iter()
            .map(|q| {
                let td = Searcher::new(idx.leaves()).search(&Query::knn(F_EMB, q.clone(), 10), &SearchOptions { top_k: 10, ..Default::default() });
                td.hits.iter().map(|h| id_of(idx, &KnnHit { leaf: h.leaf, doc: h.doc, score: h.score })).collect()
            })
            .collect()
    };
    let leaf_before = leaf_path(&idx);

    let builds = HnswBuilder::builds_on_this_thread();
    let inserts = HnswBuilder::inserts_on_this_thread();
    let files_before: HashSet<String> = idx.directory().list().unwrap().into_iter().filter(|f| f.ends_with(".rvec")).collect();
    idx.force_merge().unwrap();
    assert_eq!(idx.num_segments(), 1);
    assert_eq!(HnswBuilder::builds_on_this_thread(), builds, "text merge must not build a graph");
    assert_eq!(HnswBuilder::inserts_on_this_thread(), inserts);
    let files_after: HashSet<String> = idx.directory().list().unwrap().into_iter().filter(|f| f.ends_with(".rvec")).collect();
    assert_eq!(files_before, files_after, "text merge must not write vector files");

    let after: Vec<_> = queries
        .iter()
        .map(|q| ids_scores(&idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &cfg).unwrap().0))
        .collect();
    assert_eq!(before, after, "knn results identical across a text merge");
    assert_eq!(leaf_before, leaf_path(&idx));
    assert!(before.iter().flatten().all(|(id, _)| id % 7 != 0), "deleted docs never returned");
}

#[test]
fn deletes_hide_vectors_on_chunks_sealed_and_memtable() {
    let mut idx = new_index();
    add_batch(&mut idx, 0..200);
    idx.optimizer_policy.indexing_threshold = 100;
    idx.optimize_vectors().unwrap(); // sealed
    add_batch(&mut idx, 200..400); // chunk
    for i in 400..450 {
        idx.add_document(&doc(i)).unwrap(); // memtable
    }
    // Query exactly at doc 50 (sealed), 250 (chunk), 420 (memtable).
    for (target, leaf, d) in [(50i64, 0u32, 50u32), (250, 1, 50), (420, 2, 20)] {
        let q = rvec(target as u64, DIM);
        let hits = idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 3)).unwrap();
        assert_eq!(id_of(&idx, &hits[0]), target);
        idx.delete_doc(&DocAddress::new(leaf, d)).unwrap();
        let hits = idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 3)).unwrap();
        assert!(hits.iter().all(|h| id_of(&idx, h) != target), "deleted {target} still returned");
        assert_eq!(hits.len(), 3);
        // Leaf path (Segment::knn / memtable) agrees.
        let leaves = idx.leaves();
        let live = leaves[leaf as usize].live_docs();
        assert!(leaves[leaf as usize].knn(F_EMB, &q, 3, 50, live).unwrap().iter().all(|(dd, _)| *dd != d));
    }
    // Point liveness in the store.
    let seg0 = &idx.segments()[0];
    let p = seg0.point(F_EMB, 50).unwrap();
    assert!(!idx.vector_store().is_alive(p));
    assert!(idx.vector_store().is_alive(seg0.point(F_EMB, 51).unwrap()));
    assert_eq!(seg0.vector(F_EMB, 51).unwrap(), rvec(51, DIM));
}

#[test]
fn optimizer_background_phases_index_first_and_bound_rebuilds() {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 10;
    idx.optimizer_policy.max_segments = 100;
    for batch in 0..6 {
        add_batch(&mut idx, batch * 60..(batch + 1) * 60);
        let report = idx.vector_optimizer().index_pending().unwrap();
        assert_eq!((report.sealed, report.vacuumed, report.merged), (2, 0, 0));
    }
    let emb2_vacuum_input = idx.vector_store().segments(F_EMB2)[0].id;
    // Both fields have a segment with >20% deletions. Fresh chunks and
    // excess graph segments must not hide the priority of new indexing.
    for d in 0..15 {
        idx.delete_doc(&DocAddress::new(0, d)).unwrap();
    }
    add_batch(&mut idx, 360..420);
    idx.optimizer_policy.max_segments = 2;
    let job = idx.vector_optimizer();
    let before = job.backlog();
    assert_eq!((before.unsealed_segments, before.unsealed_vectors, before.searchable_unsealed_vectors), (2, 80, 80));
    assert_eq!(before.graph_segments, 0, "payload segments never carry a graph");
    assert_eq!(before.sealed_segments, 12);
    let request = KnnRequest::new(F_EMB, rvec(99, DIM), 30);
    let exact = |idx: &Index| ids_scores(idx, &idx.knn_search_with(&request, &PlannerConfig::exact()).unwrap().0);
    let expected = exact(&idx);

    let indexed = job.index_pending().unwrap();
    assert_eq!((indexed.sealed, indexed.vacuumed, indexed.merged), (2, 0, 0));
    assert_eq!(job.backlog().unsealed_vectors, 0);
    assert_eq!(job.backlog().sealed_segments, 14, "indexing does not drain unrelated payload merge work");
    assert_eq!(job.backlog().graph_segments, 0);
    assert_eq!(exact(&idx), expected);

    let maintained = job.maintain_once().unwrap();
    assert_eq!((maintained.sealed, maintained.vacuumed, maintained.merged), (0, 1, 0));
    // The second field's vacuum remains pending: allowance is per shard,
    // not per field. Its first 20-row segment still contains five dead rows.
    assert!(idx.vector_store().segments(F_EMB2).iter().any(|s| s.id == emb2_vacuum_input && s.len() == 20));
    assert_eq!(exact(&idx), expected);
    let maintained = job.maintain_once().unwrap();
    assert_eq!((maintained.sealed, maintained.vacuumed, maintained.merged), (0, 1, 0),
        "second field gets its vacuum before the first field's pending merge");
    assert!(idx.vector_store().segments(F_EMB2).iter().all(|s| s.id != emb2_vacuum_input));
    assert_eq!(exact(&idx), expected);

    // The existing full API still drains the remaining work across fields.
    let rest = job.run().unwrap();
    assert!(rest.vacuumed + rest.merged > 0);
    assert_eq!(exact(&idx), expected);
    assert_eq!(job.run().unwrap(), Default::default());
}

#[test]
fn optimizer_background_maintenance_does_not_take_fresh_indexing_work() {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 10;
    add_batch(&mut idx, 0..60);
    let job = idx.vector_optimizer();
    let before = job.backlog();
    assert_eq!(before.unsealed_vectors, 80);
    assert_eq!(job.maintain_once().unwrap(), Default::default());
    assert_eq!(job.backlog(), before);
    assert_eq!(job.index_pending().unwrap().sealed, 2);
    assert_eq!(job.backlog().unsealed_vectors, 0);
}

#[test]
fn optimizer_background_indexing_preserves_hidden_vectors_and_pending_deletes() {
    let mut idx = new_index();
    idx.set_nrt(true);
    idx.set_auto_seal(false);
    idx.optimizer_policy.indexing_threshold = 10;
    add_batch(&mut idx, 0..30);
    idx.vector_optimizer().index_pending().unwrap();
    let request = KnnRequest::new(F_EMB, rvec(44, DIM), 100);
    let exact = |idx: &Index| ids_scores(idx, &idx.knn_search_with(&request, &PlannerConfig::exact()).unwrap().0);
    let visible = exact(&idx);
    assert_eq!(visible.len(), 30);

    idx.delete_doc(&DocAddress::new(0, 0)).unwrap();
    for i in 30..60 {
        idx.add_document(&doc(i)).unwrap();
    }
    idx.seal().unwrap(); // Encoded, but still hidden until refresh.
    let job = idx.vector_optimizer();
    let backlog = job.backlog();
    // emb2's first ten rows remain below the strict >10 seal threshold;
    // they are visible. The new 40 vector rows are hidden.
    assert_eq!(backlog.unsealed_vectors, 50);
    assert_eq!(backlog.searchable_unsealed_vectors, 10);
    assert_eq!(job.index_pending().unwrap().sealed, 2);
    assert_eq!(job.backlog().unsealed_vectors, 0);
    job.maintain_once().unwrap();
    assert_eq!(exact(&idx), visible, "background indexing must not publish writes or deletes");

    idx.publish().unwrap();
    let hits = exact(&idx);
    let ids: HashSet<_> = hits.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, (1i64..60).collect::<HashSet<_>>(), "refresh reveals new vectors and the pending deletion");
}

/// Pause after vector files have been written but before the owning text
/// segment is installed. Optimizer work must be able to run during this
/// window without treating the in-flight vectors as abandoned/deleted.
#[derive(Debug)]
struct PausingVectorDirectory {
    inner: RamDirectory,
    text: String,
    deletion: String,
    state: parking_lot::Mutex<(bool, bool)>, // reached text write, released
    wake: parking_lot::Condvar,
    pause: std::sync::atomic::AtomicBool,
    failure: std::sync::atomic::AtomicU8,
    vector_writes: std::sync::atomic::AtomicUsize,
}

impl PausingVectorDirectory {
    fn new(id: u32, failure: u8) -> Self {
        Self {
            inner: RamDirectory::new(), text: crate::codec::segment_file_name(id), deletion: crate::codec::deletes_file_name(id),
            state: parking_lot::Mutex::new((false, false)), wake: parking_lot::Condvar::new(),
            pause: std::sync::atomic::AtomicBool::new(true), failure: std::sync::atomic::AtomicU8::new(failure),
            vector_writes: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn wait_until_blocked(&self) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut state = self.state.lock();
        while !state.0 {
            let now = std::time::Instant::now();
            if now >= deadline { return false; }
            self.wake.wait_for(&mut state, deadline - now);
        }
        true
    }

    fn release(&self) {
        self.state.lock().1 = true;
        self.wake.notify_all();
    }

    fn injected() -> crate::error::RusticError {
        std::io::Error::other("injected vector flush failure").into()
    }
}

impl crate::codec::directory::Directory for PausingVectorDirectory {
    fn open_read(&self, name: &str) -> crate::error::Result<crate::codec::directory::FileSlice> {
        if name == self.text && self.failure.load(std::sync::atomic::Ordering::Relaxed) == 2 { return Err(Self::injected()); }
        self.inner.open_read(name)
    }

    fn write(&self, name: &str, data: &[u8]) -> crate::error::Result<()> {
        use std::sync::atomic::Ordering;
        if name == self.text && self.pause.swap(false, Ordering::AcqRel) {
            let mut state = self.state.lock();
            state.0 = true;
            self.wake.notify_all();
            while !state.1 { self.wake.wait(&mut state); }
        }
        self.inner.write(name, data)?;
        let fail = self.failure.load(Ordering::Relaxed);
        let second_vector = name.ends_with(".rvec") && self.vector_writes.fetch_add(1, Ordering::Relaxed) == 1;
        if (name == self.text && (fail == 1 || fail == 5)) || (name == self.deletion && fail == 3) || (second_vector && fail == 4) {
            return Err(Self::injected());
        }
        Ok(())
    }

    fn exists(&self, name: &str) -> bool { self.inner.exists(name) }
    fn delete(&self, name: &str) -> crate::error::Result<()> {
        if name.ends_with(".rvec") && self.failure.load(std::sync::atomic::Ordering::Relaxed) == 5 { return Err(Self::injected()); }
        self.inner.delete(name)
    }
    fn list(&self) -> crate::error::Result<Vec<String>> { self.inner.list() }
    fn list_admitted(&self, temporary: bool) -> crate::error::Result<crate::codec::directory::AdmittedNames> { self.inner.list_admitted(temporary) }
}

/// Also release the writer if an assertion panics, rather than hanging the
/// scoped thread while reporting the original regression.
struct ReleaseVectorWriter<'a>(&'a PausingVectorDirectory);
impl Drop for ReleaseVectorWriter<'_> {
    fn drop(&mut self) { self.0.release(); }
}

#[test]
fn vector_append_survives_optimizer_while_owning_segment_is_encoding() {
    use crate::codec::directory::Directory;
    for hidden in [false, true] {
        let dir = PausingVectorDirectory::new(7, 0);
        let store = VectorStore::new();
        for field in [F_EMB, F_EMB2] { store.set_field_options(field,
            crate::vector::store::VectorFieldOptions::f32_hnsw().flat()); }
        let mut writer = SegmentWriter::new(7);
        for i in 0..64 { writer.add_document(&doc(i), &schema()); }
        writer.delete_doc(3);
        writer.delete_doc_pending(4);
        let policy = OptimizerPolicy { indexing_threshold: 10, ..policy() };
        let segment = std::thread::scope(|scope| {
            let _release = ReleaseVectorWriter(&dir);
            let writing = scope.spawn(|| writer.write_to_store_ref(&dir, Some(&store), hidden));
            assert!(dir.wait_until_blocked(), "writer did not reach the text-write barrier");
            assert_eq!(dir.list().unwrap().iter().filter(|n| n.ends_with(".rvec")).count(), 2);
            // Provisional files cannot enter a durable commit manifest. An
            // options update while encoding must apply at the later seal.
            assert!(store.commit_point().unwrap().segments.is_empty());
            store.set_field_options(F_EMB2, crate::vector::store::VectorFieldOptions::default().flat());
            let indexed = store.index_pending(&dir, &policy).unwrap();
            let optimized = store.optimize(&dir, &policy).unwrap();
            let merged = store.force_merge(&dir, &policy, 1).unwrap();
            assert!(store.try_search(F_EMB, &rvec(1, DIM), 100, 100, None, &PlannerConfig::exact()).unwrap().0.is_empty(),
                "unfinished text documents are not searchable");
            dir.release();
            let segment = writing.join().unwrap().unwrap();
            assert_eq!(indexed.dropped + optimized.dropped + merged.dropped, 0,
                "an optimizer must not discard in-flight append rows as dead");
            segment
        });
        assert!(store.has_uncommitted_changes(), "publication after a checkpoint must dirty the vector state");
        assert_eq!(segment.vector(F_EMB, 0), Some(rvec(0, DIM)));
        let ids = || store.try_search(F_EMB, &rvec(11, DIM), 100, 100, None, &PlannerConfig::exact()).unwrap().0
            .into_iter().map(|h| h.doc).collect::<HashSet<_>>();
        let visible: HashSet<u32> = (0..64).filter(|&d| d != 3).collect();
        assert_eq!(ids(), if hidden { HashSet::new() } else { visible.clone() });
        store.index_pending(&dir, &policy).unwrap();
        assert!(store.segments(F_EMB2).iter().all(|s| s.is_flat()), "seal reads current shared field options");
        assert_eq!(ids(), if hidden { HashSet::new() } else { visible });
        segment.publish_deletions().unwrap();
        if hidden { store.reveal(&segment); }
        assert_eq!(ids(), (0..64).filter(|&d| d != 3 && d != 4).collect::<HashSet<_>>());
    }
}

#[test]
fn failed_vector_append_releases_provisional_chunks_and_can_retry() {
    use crate::codec::directory::Directory;
    use std::sync::atomic::Ordering;
    // Fail after writing the text, opening it, persisting deletes, or
    // writing the second vector field. Earlier fields must not leak.
    for failure in 1..=4 {
        let dir = PausingVectorDirectory::new(9, failure);
        dir.pause.store(false, Ordering::Relaxed);
        let store = VectorStore::new();
        let mut writer = SegmentWriter::new(9);
        for i in 0..64 { writer.add_document(&doc(i), &schema()); }
        writer.delete_doc(3);
        assert!(writer.write_to_store_ref(&dir, Some(&store), false).is_err(), "failure point {failure}");
        assert!(store.segment_ids().is_empty(), "failed text write registered vector ownership at failure point {failure}");
        assert!(dir.list().unwrap().iter().all(|n| !n.ends_with(".rvec")), "provisional vector files leaked at failure point {failure}");
        assert!(store.commit_point().unwrap().segments.is_empty());
        let next = store.next_point_id();
        dir.failure.store(0, Ordering::Relaxed);
        let segment = writer.write_to_store_ref(&dir, Some(&store), false).unwrap();
        assert!(segment.point(F_EMB, 0).unwrap() >= next, "failed point IDs are not reused");
        assert_eq!(segment.vector(F_EMB, 0), Some(rvec(0, DIM)));
        assert_eq!(store.num_alive(F_EMB), 63);
        assert_eq!(store.segment_ids().len(), 2);
        assert_eq!(dir.list().unwrap().iter().filter(|n| n.ends_with(".rvec")).count(), 2);
    }
}

#[test]
fn failed_vector_append_cleanup_requeues_failed_unlinks_for_commit() {
    use std::sync::atomic::Ordering;
    let dir = PausingVectorDirectory::new(11, 5);
    dir.pause.store(false, Ordering::Relaxed);
    let store = VectorStore::new();
    let mut writer = SegmentWriter::new(11);
    for i in 0..64 { writer.add_document(&doc(i), &schema()); }
    assert!(writer.write_to_store_ref(&dir, Some(&store), false).is_err());
    assert!(store.segment_ids().is_empty());
    assert!(store.has_uncommitted_changes(), "failed cleanup must request a future commit");
    let checkpoint = store.commit_point().unwrap();
    assert!(checkpoint.segments.is_empty());
    assert_eq!(checkpoint.obsolete.len(), 2, "both provisional files are queued for commit cleanup");
    dir.failure.store(0, Ordering::Relaxed);
    // Same obsolete-file cleanup used by Index::commit after meta is durable.
    for id in checkpoint.obsolete { dir.delete(&crate::codec::vector_segment_file_name(id)).unwrap(); }
    assert!(dir.list().unwrap().iter().all(|name| !name.ends_with(".rvec")));
}

#[test]
fn optimizer_indexing_vacuum_and_merge_triggers() {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 250;
    idx.optimizer_policy.max_segments = 3;
    // Indexing: one 200-doc chunk stays appendable; the second pushes the
    // chunks over the threshold and they are sealed together.
    add_batch(&mut idx, 0..200);
    assert_eq!(idx.optimize_vectors().unwrap(), Default::default());
    add_batch(&mut idx, 200..400);
    let b = HnswBuilder::builds_on_this_thread();
    let r = idx.optimize_vectors().unwrap();
    assert_eq!((r.sealed, r.vectors_indexed), (1, 400));
    assert_eq!(HnswBuilder::builds_on_this_thread(), b, "sealing payloads builds no graph");
    let store = idx.vector_store().clone();
    let sealed: Vec<_> = store.segments(F_EMB).into_iter().filter(|s| s.is_sealed()).collect();
    assert_eq!(sealed.len(), 1);
    assert!(store.segments(F_EMB).iter().all(|s| s.is_sealed()), "chunks replaced");
    assert!(store.segments(F_EMB2).iter().all(|s| !s.is_sealed()), "emb2 below threshold stays appendable");

    // Vacuum: delete 25% (> 20%) of the sealed segment.
    let q = rvec(77_777, DIM);
    for d in (0..200u32).step_by(2) {
        idx.delete_doc(&DocAddress::new(0, d)).unwrap();
    }
    let live: Vec<i64> = (0..400).filter(|i| !(i < &200 && i % 2 == 0)).collect();
    let exact_before = ids_scores(&idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &PlannerConfig::exact()).unwrap().0);
    let r = idx.optimize_vectors().unwrap();
    assert_eq!((r.vacuumed, r.vectors_indexed), (1, 300));
    let seg = store.segments(F_EMB).into_iter().find(|s| s.is_sealed()).unwrap();
    assert_eq!(seg.len(), 300, "dead points dropped");
    let exact_after = ids_scores(&idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &PlannerConfig::exact()).unwrap().0);
    assert_eq!(exact_before, exact_after, "vacuum preserves live results");
    assert_eq!(exact_after.iter().map(|x| x.0).collect::<Vec<_>>(), brute(&live, &q, 10));
    // Below the ratio: nothing.
    idx.delete_doc(&DocAddress::new(0, 1)).unwrap();
    assert_eq!(idx.optimize_vectors().unwrap().vacuumed, 0);

    // Merge: sealed segments of 300 (largest), then 260, 260, 260 -> with
    // max_segments = 3 the two smallest are combined; the largest never.
    for b in 0..3 {
        add_batch(&mut idx, 1000 + b * 260..1260 + b * 260);
        idx.optimizer_policy.max_segments = 100;
        idx.optimize_vectors().unwrap();
    }
    let sealed_ids = |s: &VectorStore| -> Vec<(u32, u32)> {
        s.segments(F_EMB).iter().filter(|x| x.is_sealed()).map(|x| (x.id, x.len())).collect()
    };
    let largest = sealed_ids(&store).into_iter().max_by_key(|x| x.1).unwrap();
    assert_eq!(sealed_ids(&store).len(), 4);
    idx.optimizer_policy.max_segments = 3;
    let inserts = HnswBuilder::inserts_on_this_thread();
    let r = idx.optimize_vectors().unwrap();
    assert_eq!(r.merged, 1);
    // Payload compaction copies both 260-row inputs and inserts nothing
    // into the mutable graph, which already holds every row.
    assert_eq!(HnswBuilder::inserts_on_this_thread() - inserts, 0, "payload merging must not rebuild the mutable graph");
    assert_eq!(r.vectors_indexed, 520);
    let now = sealed_ids(&store);
    assert_eq!(now.len(), 3);
    assert!(now.contains(&largest), "largest segment untouched");
    assert!(now.iter().any(|x| x.1 == 520));
    // Recall is intact after all of it.
    let all_live: Vec<i64> = live.iter().copied().filter(|&i| i != 3).chain(1000..1780).collect();
    let got: Vec<i64> = idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 10)).unwrap().iter().map(|h| id_of(&idx, h)).collect();
    assert!(recall(&got, &brute(&all_live, &q, 10)) >= 0.8);
    // Obsolete files are deleted by the next commit.
    let before = idx.directory().list().unwrap().iter().filter(|f| f.ends_with(".rvec")).count();
    idx.commit().unwrap();
    let after = idx.directory().list().unwrap().iter().filter(|f| f.ends_with(".rvec")).count();
    assert!(after < before);
    assert_eq!(after, store.segment_ids().len());
}

#[test]
fn merge_rule_extends_to_satisfy_doubling() {
    // Sizes 50, 200, 200, 200 with max 3: {50, 200} violates "largest input
    // <= sum of the others", so the plan extends to {50, 200, 200}.
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 10;
    idx.optimizer_policy.max_segments = 3;
    for (a, b) in [(0, 50), (50, 250), (250, 450), (450, 650)] {
        add_batch(&mut idx, a..b);
        idx.optimizer_policy.max_segments = 100;
        idx.optimize_vectors().unwrap();
    }
    // sizes: 50, 200, 200, 200 -> {50, 200}: 200 > 50; {50,200,200}: 200 <= 250 ok.
    idx.optimizer_policy.max_segments = 3;
    let before: Vec<u32> = idx.vector_store().segments(F_EMB).iter().map(|s| s.len()).collect();
    assert_eq!(before, vec![50, 200, 200, 200]);
    let r = idx.optimize_vectors().unwrap();
    assert!(r.merged >= 1); // emb2 may merge too
    let mut sizes: Vec<u32> = idx.vector_store().segments(F_EMB).iter().map(|s| s.len()).collect();
    sizes.sort();
    assert_eq!(sizes, vec![200, 450]);
}

#[test]
fn deletes_only_merges_smallest_above_max_segments() {
    // Deletes-only never merges for count below the ceiling, but over
    // max_segments it merges the smallest segments (never the largest)
    // instead of letting the payload segment count grow without bound.
    let mut idx = new_index();
    idx.optimizer_policy.payload_merge = crate::vector::store::PayloadMerge::DeletesOnly;
    idx.optimizer_policy.indexing_threshold = 10;
    idx.optimizer_policy.max_segments = 100;
    for (a, b) in [(0, 50), (50, 250), (250, 450), (450, 650)] {
        add_batch(&mut idx, a..b);
        idx.optimize_vectors().unwrap();
    }
    let sizes = |idx: &Index| -> Vec<u32> {
        let mut v: Vec<u32> = idx.vector_store().segments(F_EMB).iter().filter(|s| s.is_sealed()).map(|s| s.len()).collect();
        v.sort();
        v
    };
    assert_eq!(sizes(&idx), vec![50, 200, 200, 200], "below the ceiling nothing merges");
    idx.optimizer_policy.max_segments = 3;
    let r = idx.optimize_vectors().unwrap();
    assert!(r.merged >= 1);
    assert_eq!(sizes(&idx), vec![200, 450], "the three smallest merged, the largest left alone");
}

#[test]
fn merge_rule_skips_without_doubling_until_hard_cap() {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 5;
    idx.optimizer_policy.max_segments = 100;
    let mut start = 0;
    for size in [10i64, 50, 250, 1250] {
        add_batch(&mut idx, start..start + size);
        idx.optimize_vectors().unwrap();
        start += size;
    }
    let sizes = |idx: &Index| -> Vec<u32> {
        let mut v: Vec<u32> = idx.vector_store().segments(F_EMB).iter().map(|s| s.len()).collect();
        v.sort();
        v
    };
    assert_eq!(sizes(&idx), vec![10, 50, 250, 1250]);
    // 4 > max 2, but no prefix of {10, 50, 250} doubles: within the hard
    // cap (2 * max) nothing is merged.
    idx.optimizer_policy.max_segments = 2;
    idx.optimize_vectors().unwrap();
    assert_eq!(sizes(&idx), vec![10, 50, 250, 1250]);
    // A fifth segment exceeds the hard cap: the two smallest are merged.
    idx.optimizer_policy.max_segments = 100;
    add_batch(&mut idx, start..start + 10);
    idx.optimize_vectors().unwrap();
    idx.optimizer_policy.max_segments = 2;
    idx.optimize_vectors().unwrap();
    assert_eq!(sizes(&idx), vec![20, 50, 250, 1250]);
}

#[test]
fn planner_exact_vs_graph_across_selectivities() {
    let n: u64 = if cfg!(debug_assertions) { 2000 } else { 6000 };
    let mut index = new_index();
    add_batch(&mut index, 0..n as i64);
    index.optimizer_policy.indexing_threshold = 10;
    index.optimize_vectors().unwrap();
    let store = index.vector_store().clone();
    let seg = index.segments()[0].clone();
    assert!(store.segments(F_EMB)[0].is_sealed());
    let cfg = PlannerConfig { full_scan_threshold: 100, full_scan_fraction: 0.05, ..PlannerConfig::default() };
    for (pct, expect_graph) in [(1u64, false), (10, true), (50, true), (100, true)] {
        let keep = |d: u64| (d * 7919) % 100 < pct;
        let mut total_recall = 0.0;
        let nq = 10;
        for qi in 0..nq {
            let q = rvec(50_000 + qi, DIM);
            let docs: Vec<u32> = (0..n as u32).filter(|&d| keep(d as u64)).collect();
            let points: Vec<u64> = docs.iter().map(|&d| seg.point(F_EMB, d).unwrap()).collect();
            let (hits, st) = store.try_search(F_EMB, &q, 10, 64, Some(&points), &cfg).unwrap();
            assert_eq!(st.graph_segments + st.exact_segments, 1);
            assert_eq!(st.graph_segments == 1, expect_graph, "{pct}%: {st:?}");
            assert!(hits.iter().all(|h| keep(h.doc as u64)), "{pct}%: filter violated");
            let live: Vec<i64> = docs.iter().map(|&d| d as i64).collect();
            let want = brute(&live, &q, 10);
            let got: Vec<i64> = hits.iter().map(|h| h.doc as i64).collect();
            if !expect_graph {
                assert_eq!(got, want, "exact path must equal brute force");
            }
            total_recall += recall(&got, &want);
            // Leaf-level path agrees on correctness.
            let leaf = seg.knn_with(F_EMB, &q, 10, 64, |d| keep(d as u64), &cfg).unwrap();
            assert!(leaf.iter().all(|(d, _)| keep(*d as u64)));
        }
        let avg = total_recall / nq as f32;
        assert!(avg >= 0.9, "{pct}% selectivity recall {avg}");
    }
    // Unfiltered goes through the graph; forced exact never does.
    let q = rvec(1, DIM);
    assert_eq!(store.try_search(F_EMB, &q, 10, 64, None, &cfg).unwrap().1.graph_segments, 1);
    let (h, st) = store.try_search(F_EMB, &q, 10, 64, None, &PlannerConfig::exact()).unwrap();
    assert_eq!((st.graph_segments, st.exact_segments), (0, 1));
    assert_eq!(h[0].doc, 1);
    // A filter accepting fewer than k points still returns all of them.
    let few: Vec<u64> = (0..3).map(|d| seg.point(F_EMB, d * 500).unwrap()).collect();
    let (h, _) = store.try_search(F_EMB, &q, 10, 64, Some(&few), &PlannerConfig { full_scan_threshold: 0, full_scan_fraction: 0.0, ..PlannerConfig::default() }).unwrap();
    assert_eq!(h.len(), 3);
}

#[test]
fn index_knn_prefilter_query_and_threshold() {
    let mut idx = new_index();
    add_batch(&mut idx, 0..300);
    idx.optimizer_policy.indexing_threshold = 100;
    idx.optimize_vectors().unwrap();
    add_batch(&mut idx, 300..600);
    for i in 600..650 {
        idx.add_document(&doc(i)).unwrap();
    }
    let q = rvec(123, DIM);
    let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
    req.filter = Some(Query::term(F_TAG, "ten"));
    let hits = idx.knn_search(&req).unwrap();
    let ids: Vec<i64> = hits.iter().map(|h| id_of(&idx, h)).collect();
    let live: Vec<i64> = (0..650).filter(|i| i % 10 == 0).collect();
    assert_eq!(ids, brute(&live, &q, 10), "60 matching points per segment -> exact");
    assert!(hits.iter().any(|h| h.leaf == 2), "memtable participates");
    // similarity_threshold + boost.
    let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
    let top = idx.knn_search(&req).unwrap();
    req.similarity_threshold = Some(top[4].score);
    req.boost = 2.0;
    let thr = idx.knn_search(&req).unwrap();
    assert_eq!(thr.len(), 5);
    assert!((thr[0].score - 2.0 * top[0].score).abs() < 1e-6);
}

#[test]
fn hybrid_bool_should_sums_text_and_knn() {
    let mut idx = new_index();
    add_batch(&mut idx, 0..200);
    let q = rvec(40, DIM); // doc 40 has tag "ten" and is the nearest
    let text = Query::term(F_TAG, "ten");
    let knn = KnnRequest::new(F_EMB, q.clone(), 5);
    let hybrid = idx.hybrid_search(Some(&text), std::slice::from_ref(&knn), 30).unwrap();
    assert_eq!(id_of(&idx, &hybrid[0]), 40);
    let text_hits = Searcher::new(idx.leaves()).search(&text, &SearchOptions { top_k: 100, ..Default::default() });
    let text_score = text_hits.hits.iter().find(|h| h.doc == 40).unwrap().score;
    let knn_hits = idx.knn_search(&knn).unwrap();
    assert!((hybrid[0].score - (text_score + knn_hits[0].score)).abs() < 1e-5);
    // A knn-only doc (not tagged "ten") scores its knn score alone.
    let knn_only = knn_hits.iter().find(|h| id_of(&idx, h) % 10 != 0).unwrap();
    let in_hybrid = hybrid.iter().find(|h| h.leaf == knn_only.leaf && h.doc == knn_only.doc).unwrap();
    assert!((in_hybrid.score - knn_only.score).abs() < 1e-6);
    // Query::Knn inside a bool should (per-leaf scorer path) also sums.
    let bq = Query::Bool {
        must: vec![],
        should: vec![text.clone(), Query::knn(F_EMB, q.clone(), 5)],
        must_not: vec![],
        filter: vec![],
        minimum_should_match: Default::default(),
    };
    let td = Searcher::new(idx.leaves()).search(&bq, &SearchOptions { top_k: 1, ..Default::default() });
    assert_eq!(td.hits[0].doc, 40);
    assert!((td.hits[0].score - hybrid[0].score).abs() < 1e-4);
    // Fusion helpers over the two lists.
    let key = |h: &KnnHit| (h.leaf, h.doc);
    let tl: Vec<_> = text_hits.hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
    let kl: Vec<_> = knn_hits.iter().map(|h| (key(h), h.score)).collect();
    let fused = crate::vector::fusion::rrf(&[tl, kl], 60.0, 100);
    assert_eq!(fused[0].0, (0, 40));
}

#[test]
fn persistence_roundtrip_of_vector_store() {
    let dir = tempfile::tempdir().unwrap();
    let queries: Vec<Vec<f32>> = (0..5).map(|q| rvec(90_000 + q, DIM)).collect();
    let (before, before2, seg_ids, next_point) = {
        let mut idx = create_index(dir.path(), "p");
        idx.optimizer_policy = OptimizerPolicy { indexing_threshold: 250, ..policy() };
        add_batch(&mut idx, 0..200);
        add_batch(&mut idx, 200..400);
        idx.optimize_vectors().unwrap(); // seal
        add_batch(&mut idx, 400..600); // chunk
        idx.delete_doc(&DocAddress::new(0, 5)).unwrap();
        idx.delete_doc(&DocAddress::new(2, 7)).unwrap();
        idx.commit().unwrap();
        let before: Vec<_> = queries.iter().map(|q| ids_scores(&idx, &idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 10)).unwrap())).collect();
        let before2 = ids_scores(&idx, &idx.knn_search(&KnnRequest::new(F_EMB2, vec![300.2, 0.0, 0.0, 0.0], 3)).unwrap());
        (before, before2, idx.vector_store().segment_ids(), idx.vector_store().next_point_id())
    };
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["vector_store"]["segments"].as_array().unwrap().len(), seg_ids.len());
    let on_disk: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".rvec"))
        .collect();
    assert_eq!(on_disk.len(), seg_ids.len(), "sealed-away chunks were deleted at commit");

    let idx = Index::open(dir.path()).unwrap();
    assert_eq!(idx.vector_store().segment_ids(), seg_ids);
    assert_eq!(idx.vector_store().next_point_id(), next_point);
    let after: Vec<_> = queries.iter().map(|q| ids_scores(&idx, &idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 10)).unwrap())).collect();
    assert_eq!(before, after);
    assert!(after.iter().flatten().all(|(id, _)| *id != 5 && *id != 407), "deletes survive reopen");
    assert_eq!(ids_scores(&idx, &idx.knn_search(&KnnRequest::new(F_EMB2, vec![300.2, 0.0, 0.0, 0.0], 3)).unwrap()), before2);
    assert_eq!(before2[0].0, 300, "second named vector field (l2) searchable");
}

#[test]
fn recall_across_several_sealed_segments() {
    let mut idx = new_index();
    let n = batch();
    idx.optimizer_policy.indexing_threshold = n as usize - 1;
    for b in 0..4 {
        add_batch(&mut idx, b * n..(b + 1) * n);
        idx.optimize_vectors().unwrap();
    }
    let sealed = idx.vector_store().segments(F_EMB).iter().filter(|s| s.is_sealed()).count();
    assert!(idx.vector_store().segments(F_EMB).iter().all(|s| !s.has_graph()),
        "text/vector payload sealing must never build a second graph");
    assert_eq!(sealed, 4);
    let live: Vec<i64> = (0..4 * n).collect();
    let cfg = PlannerConfig { full_scan_threshold: 0, full_scan_fraction: 0.0, ..PlannerConfig::default() };
    let nq = 30;
    let mut total = 0.0;
    for qi in 0..nq {
        let q = rvec(70_000 + qi, DIM);
        let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
        req.num_candidates = 100;
        let (hits, st) = idx.knn_search_with(&req, &cfg).unwrap();
        assert_eq!(st.graph_segments + st.fallbacks, 1,
            "mutable search traverses one graph across all text segments");
        let got: Vec<i64> = hits.iter().map(|h| id_of(&idx, h)).collect();
        total += recall(&got, &brute(&live, &q, 10));
    }
    let avg = total / nq as f32;
    assert!(avg >= 0.9, "recall@10 across 4 vector segments = {avg}");
}

#[test]
fn standalone_segments_merge_through_private_stores() {
    // Segments written without an Index get private stores over the
    // directory; merging them still moves no vectors and keeps knn working.
    let dir = RamDirectory::new();
    let schema = schema();
    let mut segs = Vec::new();
    for s in 0..2i64 {
        let mut w = SegmentWriter::new(s as u32);
        for i in 0..100 {
            w.add_document(&doc(s * 100 + i), &schema);
        }
        segs.push(std::sync::Arc::new(w.write_to(&dir).unwrap()));
    }
    let b = HnswBuilder::builds_on_this_thread();
    let merged = crate::merge::merge_segments(&dir, &segs, 9).unwrap();
    assert_eq!(HnswBuilder::builds_on_this_thread(), b);
    let q = rvec(150, DIM);
    assert_eq!(merged.knn(F_EMB, &q, 1, 10, |_| true).unwrap()[0].0, 150);
    assert_eq!(merged.vector(F_EMB, 150).unwrap(), q);
    assert_eq!(LeafReader::Segment(&merged).knn(F_EMB2, &[3.0, 0.0, 0.0, 0.0], 1, 10, |_| true).unwrap()[0].0, 3);
}

/// One sealed segment of `n` docs; tag "even"/"odd" = 50 % selectivity.
fn half_filter_index(n: i64) -> Index {
    let mut idx = new_index();
    for i in 0..n {
        let mut d = Document::new();
        d.add_i64(F_ID, i);
        d.add_keyword(F_TAG, if i % 2 == 0 { "even" } else { "odd" });
        d.add_vector(F_EMB, rvec(i as u64, DIM));
        idx.add_document(&d).unwrap();
    }
    idx.refresh().unwrap();
    idx.optimizer_policy.indexing_threshold = n as usize - 1;
    assert_eq!(idx.optimize_vectors().unwrap().sealed, 1);
    idx
}

fn median_us(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn filtered_knn_bitset_paths_agree_and_are_exact_enough() {
    use crate::vector::store::{DocFilter, PointBits, PointFilter};
    let n = if cfg!(debug_assertions) { 2000 } else { 10_000 };
    let mut idx = half_filter_index(n);
    // A deleted matching doc must never come back.
    idx.delete_doc(&DocAddress::new(0, 42)).unwrap();
    let live: Vec<i64> = (0..n).filter(|i| i % 2 == 0 && *i != 42).collect();
    let store = idx.vector_store().clone();
    let seg = idx.segments()[0].clone();
    let pts: Vec<u64> = live.iter().map(|&i| seg.point(F_EMB, i as u32).unwrap()).collect();
    let bits = PointBits::from_points(&pts, store.next_point_id());
    let mut words = vec![0u64; (n as usize).div_ceil(64)];
    for &i in &live {
        words[i as usize / 64] |= 1 << (i % 64);
    }
    let mut df = DocFilter::new();
    df.add_segment(seg.id, words);
    assert_eq!((bits.len(), df.len()), (live.len(), live.len()));
    // Small corpus: lower the full-scan threshold so 50 % goes through the graph.
    let cfg = PlannerConfig { full_scan_threshold: 100, ..PlannerConfig::default() };
    let mut tot = [0.0f32; 3];
    for qi in 0..20 {
        let q = rvec(55_000 + qi, DIM);
        let want = brute(&live, &q, 10);
        let ids = |hits: Vec<crate::vector::store::PointHit>| -> Vec<i64> { hits.iter().map(|h| h.doc as i64).collect() };
        let (a, sa) = store.try_search(F_EMB, &q, 10, 100, Some(&pts), &cfg).unwrap();
        let (b, _) = store.try_search_filtered(F_EMB, &q, 10, 100, Some(PointFilter::Points(&bits)), &cfg).unwrap();
        let (c, sc) = store.try_search_filtered(F_EMB, &q, 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap();
        assert_eq!((sa.graph_segments, sc.graph_segments), (1, 1), "50 % -> graph");
        let (a, b, c) = (ids(a), ids(b), ids(c));
        assert_eq!(a, b, "slice and PointBits filters are the same search");
        assert_eq!(a, c, "PointBits and DocFilter accept the same points");
        let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
        req.num_candidates = 100;
        req.filter = Some(Query::term(F_TAG, "even"));
        let (h, _) = idx.knn_search_with(&req, &cfg).unwrap();
        let d: Vec<i64> = h.iter().map(|h| id_of(&idx, h)).collect();
        assert_eq!(d, c, "Index path matches the store path");
        assert!(d.iter().all(|i| i % 2 == 0 && *i != 42));
        for (j, got) in [&a, &c, &d].into_iter().enumerate() {
            tot[j] += recall(got, &want);
        }
    }
    for r in tot {
        assert!(r / 20.0 >= 0.9, "filtered recall {}", r / 20.0);
    }
    // Filter matching nothing -> no hits.
    let mut req = KnnRequest::new(F_EMB, rvec(3, DIM), 5);
    req.filter = Some(Query::term(F_TAG, "nope"));
    assert!(idx.knn_search_with(&req, &cfg).unwrap().0.is_empty());
    // Selective explicit set: exact over just the matches.
    let few: Vec<i64> = live.iter().copied().take(30).collect();
    let fp: Vec<u64> = few.iter().map(|&i| seg.point(F_EMB, i as u32).unwrap()).collect();
    let (h, st) = store.try_search(F_EMB, &rvec(3, DIM), 5, 100, Some(&fp), &cfg).unwrap();
    assert_eq!(st.exact_segments, 1);
    assert_eq!(h.iter().map(|h| h.doc as i64).collect::<Vec<_>>(), brute(&few, &rvec(3, DIM), 5));
}

/// The ordinal filter (doc bitset resolved to per-segment point ordinals)
/// is the same search as the doc filter: identical hits on the two-hop
/// graph path and on the exact path, with deleted docs excluded. Both the
/// word-shift fast path (identity point map, contiguous points) and the
/// per-doc path are exercised.
#[test]
fn ordinal_filter_matches_doc_filter() {
    use crate::query::query::Bound;
    use crate::vector::store::{DocFilter, PointFilter};
    let n: i64 = if cfg!(debug_assertions) { 3000 } else { 12_000 };
    let mut idx = half_filter_index(n);
    idx.delete_doc(&DocAddress::new(0, 7)).unwrap();
    idx.delete_doc(&DocAddress::new(0, 8)).unwrap();
    let store = idx.vector_store().clone();
    let seg = idx.segments()[0].clone();
    assert_eq!(seg.point_identity_base(F_EMB), Some(seg.point(F_EMB, 0).unwrap()), "points assigned in doc order");
    let ids = |hits: Vec<crate::vector::store::PointHit>| -> Vec<i64> { hits.iter().map(|h| h.doc as i64).collect() };
    // 20 % (ids < n / 5): two-hop graph range; 2 %: exact range.
    for (frac, exact) in [(5i64, false), (50, true)] {
        let hi = n / frac;
        let live: Vec<i64> = (0..hi).filter(|i| *i != 7 && *i != 8).collect();
        let mut words = vec![0u64; (n as usize).div_ceil(64)];
        for &i in &live {
            words[i as usize / 64] |= 1 << (i % 64);
        }
        let mut df = DocFilter::new();
        df.add_segment(seg.id, words);
        let cfg = PlannerConfig { full_scan_threshold: 100, ..PlannerConfig::default() };
        let fast = store.ord_filter(F_EMB, df.clone(), |sid| {
            (sid == seg.id).then(|| (seg.point_map().column(F_EMB).unwrap(), seg.point_identity_base(F_EMB)))
        });
        let slow = store.ord_filter(F_EMB, df.clone(), |sid| (sid == seg.id).then(|| (seg.point_map().column(F_EMB).unwrap(), None)));
        assert_eq!((fast.len(), slow.len()), (live.len(), live.len()), "dead points left out");
        assert_eq!(format!("{fast:?}"), format!("{slow:?}"), "shift and per-doc builds agree");
        for qi in 0..10 {
            let q = rvec(91_000 + qi, DIM);
            let (a, sa) = store.try_search_filtered(F_EMB, &q, 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap();
            let (b, sb) = store.try_search_filtered(F_EMB, &q, 10, 100, Some(PointFilter::Ords(&fast)), &cfg).unwrap();
            assert_eq!(sa.exact_segments == 1, exact, "{frac}: planner path");
            assert_eq!((sa.exact_segments, sa.graph_segments), (sb.exact_segments, sb.graph_segments));
            assert_eq!(ids(a), ids(b.clone()), "doc and ordinal filters are the same search");
            assert!(ids(b.clone()).iter().all(|d| live.contains(d)));
            // The Index path: a numeric range pre-filter resolves to ordinals.
            let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
            req.num_candidates = 100;
            req.filter = Some(Query::RangeI64 { field: F_ID, lower: Bound::Unbounded, upper: Bound::Excluded(hi) });
            let (h, _) = idx.knn_search_with(&req, &cfg).unwrap();
            let d: Vec<i64> = h.iter().map(|h| id_of(&idx, h)).collect();
            assert_eq!(d, ids(b), "Index path matches the store path");
            assert!(recall(&d, &brute(&live, &q, 10)) >= 0.8);
        }
    }
}

/// Filtered k-NN at 50 % selectivity must cost about what the unfiltered
/// search costs (Lucene-style: the filter is a bitset consulted per visited
/// node), not O(matching points) of preprocessing.
#[test]
fn filtered_knn_at_half_selectivity_is_near_unfiltered_latency() {
    use crate::vector::store::{DocFilter, PointFilter};
    let n = if cfg!(debug_assertions) { 4000 } else { 50_000 };
    let idx = half_filter_index(n);
    let store = idx.vector_store().clone();
    let seg = idx.segments()[0].clone();
    let mut words = vec![0u64; (n as usize).div_ceil(64)];
    for i in (0..n).step_by(2) {
        words[i as usize / 64] |= 1 << (i % 64);
    }
    let mut df = DocFilter::new();
    df.add_segment(seg.id, words);
    let cfg = PlannerConfig { full_scan_threshold: 100, ..PlannerConfig::default() };
    let (_, st) = store.try_search_filtered(F_EMB, &rvec(1, DIM), 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap();
    assert_eq!((st.graph_segments, st.fallbacks), (1, 0), "50 % filter searched through the graph");
    let qs: Vec<Vec<f32>> = (0..32).map(|q| rvec(77_000 + q, DIM)).collect();
    let reps = 15;
    let mut qi = 0;
    let mut next = || {
        qi = (qi + 1) % qs.len();
        &qs[qi]
    };
    // Warm up (graph decode, thread-local scratch).
    for q in &qs {
        store.try_search_filtered(F_EMB, q, 10, 100, None, &cfg).unwrap();
        store.try_search_filtered(F_EMB, q, 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap();
    }
    let mut un = 0.0;
    let mut fi = 0.0;
    for _ in 0..qs.len() {
        un += median_us(|| drop(store.try_search_filtered(F_EMB, next(), 10, 100, None, &cfg).unwrap()), reps);
        fi += median_us(|| drop(store.try_search_filtered(F_EMB, next(), 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap()), reps);
    }
    let (un, fi) = (un / qs.len() as f64, fi / qs.len() as f64);
    eprintln!("[filtered-knn] n={n} unfiltered {un:.1}us  50% DocFilter {fi:.1}us  ratio {:.2}", fi / un);
    // 2x plus a small absolute slack for timer noise on a shared machine.
    assert!(fi <= 2.0 * un + 20.0, "50 % filtered {fi:.1}us vs unfiltered {un:.1}us");
}

/// Timing report only (old and new APIs): `cargo test --release -p
/// rustic-core filtered_knn_timing_report -- --ignored --nocapture`.
#[test]
#[ignore]
fn filtered_knn_timing_report() {
    let n = 200_000;
    let idx = half_filter_index(n);
    let store = idx.vector_store().clone();
    let seg = idx.segments()[0].clone();
    let pts: Vec<u64> = (0..n).step_by(2).map(|i| seg.point(F_EMB, i as u32).unwrap()).collect();
    let cfg = PlannerConfig::default();
    let qs: Vec<Vec<f32>> = (0..50).map(|q| rvec(88_000 + q, DIM)).collect();
    let mut req: Vec<KnnRequest> = qs.iter().map(|q| KnnRequest::new(F_EMB, q.clone(), 10)).collect();
    for r in &mut req {
        r.num_candidates = 100;
    }
    let mut freq = req.clone();
    for r in &mut freq {
        r.filter = Some(Query::term(F_TAG, "even"));
    }
    let time = |f: &mut dyn FnMut(usize)| {
        for i in 0..qs.len() {
            f(i);
        }
        let mut v: Vec<f64> = Vec::new();
        for _ in 0..5 {
            for i in 0..qs.len() {
                let t = std::time::Instant::now();
                f(i);
                v.push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let un = time(&mut |i| drop(store.try_search(F_EMB, &qs[i], 10, 100, None, &cfg).unwrap()));
    let sl = time(&mut |i| drop(store.try_search(F_EMB, &qs[i], 10, 100, Some(&pts), &cfg).unwrap()));
    let iu = time(&mut |i| drop(idx.knn_search_with(&req[i], &cfg).unwrap()));
    let ifl = time(&mut |i| drop(idx.knn_search_with(&freq[i], &cfg).unwrap()));
    eprintln!("[timing] n={n} p50: store unfiltered {un:.0}us | store 50% slice {sl:.0}us | index unfiltered {iu:.0}us | index 50% filter {ifl:.0}us");
    {
        use crate::vector::store::{DocFilter, PointBits, PointFilter};
        let mut words = vec![0u64; (n as usize).div_ceil(64)];
        for i in (0..n).step_by(2) {
            words[i as usize / 64] |= 1 << (i % 64);
        }
        let mut df = DocFilter::new();
        df.add_segment(seg.id, words);
        let pb = PointBits::from_points(&pts, store.next_point_id());
        let dd = time(&mut |i| drop(store.try_search_filtered(F_EMB, &qs[i], 10, 100, Some(PointFilter::Docs(&df)), &cfg).unwrap()));
        let bb = time(&mut |i| drop(store.try_search_filtered(F_EMB, &qs[i], 10, 100, Some(PointFilter::Points(&pb)), &cfg).unwrap()));
        let un2 = time(&mut |i| drop(store.try_search_filtered(F_EMB, &qs[i], 10, 100, None, &cfg).unwrap()));
        let fp = time(&mut |_| drop(std::hint::black_box(PointBits::from_points(&pts, store.next_point_id()))));
        eprintln!("[timing] prebuilt: unfiltered {un2:.0}us | 50% DocFilter {dd:.0}us | 50% PointBits {bb:.0}us | PointBits::from_points(100k) {fp:.0}us");
    }
}

/// `hybrid_search_with` (one scorer per leaf over the sorted k-NN-only
/// docs, shared context) equals the per-doc `score_at` reference, across
/// sealed segments and the memtable, with filters, paging and options.
#[test]
fn hybrid_search_with_matches_score_at_reference() {
    use crate::query::query::{Bound, SearchContext};
    use crate::query::searcher::TrackTotalHits;
    let mut idx = new_index();
    add_batch(&mut idx, 0..300);
    idx.refresh().unwrap();
    add_batch(&mut idx, 300..600);
    idx.refresh().unwrap();
    idx.optimize_vectors().unwrap();
    add_batch(&mut idx, 600..700); // memtable leaf
    let leaves = idx.leaves();
    for filter in [None, Some(Query::RangeI64 { field: F_ID, lower: Bound::Excluded(100), upper: Bound::Included(650) })] {
        let tag = Query::term(F_TAG, "ten");
        let text = match &filter {
            Some(f) => Query::bool_query().must(tag).filter(f.clone()).build(),
            None => tag,
        };
        for qi in 0..5u64 {
            let mut knn = KnnRequest::new(F_EMB, rvec(1000 + qi, DIM), 20);
            knn.filter = filter.clone();
            // Reference: text top-k + k-NN, text score of every k-NN doc via score_at.
            let top_k = 15;
            let text_top = Searcher::new(leaves.clone()).search(&text, &SearchOptions { top_k, ..Default::default() });
            let ctx = SearchContext::for_query(&text, &leaves, crate::query::bm25::Bm25Params::default());
            let mut want: std::collections::HashMap<(usize, DocId), f32> = text_top.hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
            for h in idx.knn_search(&knn).unwrap() {
                let base = match want.get(&(h.leaf, h.doc)) {
                    Some(s) => *s,
                    None => text.score_at(&leaves[h.leaf], &ctx, h.doc).unwrap_or(0.0),
                };
                want.insert((h.leaf, h.doc), base + h.score);
            }
            let mut want: Vec<KnnHit> = want.into_iter().map(|((leaf, doc), score)| KnnHit { leaf, doc, score }).collect();
            want.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.leaf.cmp(&b.leaf)).then(a.doc.cmp(&b.doc)));
            want.truncate(top_k);
            let got = idx.hybrid_search(Some(&text), std::slice::from_ref(&knn), top_k).unwrap();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!((g.leaf, g.doc), (w.leaf, w.doc));
                assert!((g.score - w.score).abs() < 1e-5, "{} vs {}", g.score, w.score);
            }
            // Paging + options: hits from..from+size of the same ranking.
            let opts = SearchOptions { top_k: 5, from: 3, track_total_hits: TrackTotalHits::False, ..Default::default() };
            let page = idx.hybrid_search_with(Some(&text), std::slice::from_ref(&knn), &opts).unwrap();
            let full = idx.hybrid_search(Some(&text), std::slice::from_ref(&knn), 8).unwrap();
            assert_eq!(page, full[3..8].to_vec());
        }
    }
}

#[test]
fn hybrid_search_applies_knn_prefilter_during_traversal() {
    let n = if cfg!(debug_assertions) { 2000 } else { 10_000 };
    let idx = half_filter_index(n);
    let q = rvec(7, DIM); // doc 7 is odd: the unfiltered nearest, filtered out below
    let mut knn = KnnRequest::new(F_EMB, q.clone(), 10);
    knn.num_candidates = 100;
    knn.filter = Some(Query::term(F_TAG, "even"));
    let knn_hits = idx.knn_search(&knn).unwrap();
    assert_eq!(knn_hits.len(), 10, "pre-filter still fills k (no post-filter shrinkage)");
    let knn_ids: Vec<i64> = knn_hits.iter().map(|h| id_of(&idx, h)).collect();
    assert!(knn_ids.iter().all(|i| i % 2 == 0));
    let live: Vec<i64> = (0..n).filter(|i| i % 2 == 0).collect();
    assert!(recall(&knn_ids, &brute(&live, &q, 10)) >= 0.8);
    // Text "odd" + filtered knn: knn contributes only even docs, text only odd.
    let text = Query::term(F_TAG, "odd");
    let hybrid = idx.hybrid_search(Some(&text), std::slice::from_ref(&knn), 40).unwrap();
    let hy: Vec<(i64, f32)> = ids_scores(&idx, &hybrid);
    for id in &knn_ids {
        let s = hy.iter().find(|(i, _)| i == id).expect("every filtered knn hit is in the hybrid result").1;
        let k = knn_hits.iter().find(|h| id_of(&idx, h) == *id).unwrap().score;
        assert!((s - k).abs() < 1e-6, "even doc scores its knn score only");
    }
    let seven = hy.iter().find(|(i, _)| *i == 7);
    if let Some((_, s)) = seven {
        let text_only = Searcher::new(idx.leaves()).search(&text, &SearchOptions { top_k: n as usize, ..Default::default() });
        let ts = text_only.hits.iter().find(|h| h.doc == 7).unwrap().score;
        assert!((s - ts).abs() < 1e-5, "doc 7 is not a knn hit (filtered during traversal): text score only");
    }
    // RRF over the two lists keeps the knn side filtered.
    let tl: Vec<_> = Searcher::new(idx.leaves())
        .search(&text, &SearchOptions { top_k: 10, ..Default::default() })
        .hits
        .iter()
        .map(|h| ((h.leaf, h.doc), h.score))
        .collect();
    let kl: Vec<_> = knn_hits.iter().map(|h| ((h.leaf, h.doc), h.score)).collect();
    let fused = crate::vector::fusion::rrf(&[tl, kl], 60.0, 20);
    assert_eq!(fused.len(), 20);
    let even_in_fused = fused.iter().filter(|((_, d), _)| d % 2 == 0).count();
    assert_eq!(even_in_fused, 10, "all 10 filtered knn hits fused, none from outside the filter");
}

// ---------------------------------------------------------------------------
// Per-field quantization policy (ES `index_options`).

#[test]
fn per_field_options_drive_seal_time_quantization_and_search() {
    use crate::vector::quantize::Quantization;
    use crate::vector::store::VectorFieldOptions;
    const F_I4: u16 = 4;
    const F_I8F: u16 = 5;
    const F_BBQ: u16 = 6;
    let dim = 32u16;
    let mut schema = supported_schema();
    schema.push(
        Field::new(F_I4, "i4", FieldType::Vector { dim })
            .with_similarity("l2_norm")
            .with_vector_options(Some(VectorFieldOptions::quantized_hnsw(Quantization::Int4))),
    );
    schema.push(
        Field::new(F_I8F, "i8f", FieldType::Vector { dim })
            .with_similarity("dot_product")
            .with_vector_options(Some(VectorFieldOptions::quantized_hnsw(Quantization::Int8).flat().with_oversample(2.0))),
    );
    schema.push(
        Field::new(F_BBQ, "bbq", FieldType::Vector { dim })
            .with_similarity("cosine")
            .with_vector_options(Some(VectorFieldOptions::quantized_hnsw(Quantization::Binary))),
    );
    let mut idx = activate_test_index(Index::new("q", schema, IndexSettings::default()));
    idx.optimizer_policy = OptimizerPolicy { indexing_threshold: 100, ..policy() };
    assert_eq!(idx.vector_field_options(F_I4).quantization, Quantization::Int4, "options registered at open");
    assert_eq!(idx.vector_field_options(F_EMB).quantization, Quantization::None);
    let n = 600i64;
    let unit = |v: Vec<f32>| -> Vec<f32> {
        let nn = crate::vector::norm(&v);
        v.iter().map(|x| x / nn).collect()
    };
    for i in 0..n {
        let mut d = Document::new();
        d.add_i64(F_ID, i);
        d.add_keyword(F_TAG, "x");
        d.add_vector(F_I4, rvec(i as u64 + 1, dim as usize));
        d.add_vector(F_I8F, unit(rvec(i as u64 + 1, dim as usize)));
        d.add_vector(F_BBQ, rvec(i as u64 + 1, dim as usize));
        idx.add_document(&d).unwrap();
    }
    idx.refresh().unwrap();
    let r = idx.optimize_vectors().unwrap();
    assert_eq!(r.sealed, 3);
    let store = idx.vector_store();
    let seg = |f: u16| store.segments(f).into_iter().find(|s| s.is_sealed()).expect("sealed");
    // int4_hnsw: the payload holds originals only; graph and int4 codes
    // belong to the field's mutable owner (and its checkpoint).
    let s4 = seg(F_I4);
    assert!(!s4.has_graph() && !s4.has_int4() && !s4.has_int8(), "codes belong to the mutable owner checkpoint");
    // int8_flat: no graph, stored int8, still sealed (not re-sealed).
    let s8 = seg(F_I8F);
    assert!(!s8.has_graph() && s8.is_flat() && s8.has_int8() && !s8.has_int4());
    // bbq_hnsw: no payload graph, nothing stored (codes live in the owner).
    let sb = seg(F_BBQ);
    assert!(!sb.has_graph());
    assert!(!sb.has_int4() && !sb.has_int8());
    assert_eq!(idx.optimize_vectors().unwrap(), crate::vector::store::OptimizeReport::default(), "flat sealed segments are not re-sealed");

    let brute_l2 = |q: &[f32], f: u16| -> Vec<i64> {
        let sim = if f == F_I4 { Similarity::L2 } else if f == F_I8F { Similarity::DotProduct } else { Similarity::Cosine };
        let mut all: Vec<(i64, f32)> = (0..n)
            .map(|i| {
                let v = rvec(i as u64 + 1, dim as usize);
                let v = if f == F_I8F { unit(v) } else { v };
                (i, sim.score(q, &v))
            })
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.into_iter().take(10).map(|x| x.0).collect()
    };
    let cfg_graph = PlannerConfig { full_scan_threshold: 0, full_scan_fraction: 0.0, ..PlannerConfig::default() };
    let mut rec = [0.0f32; 3];
    for qi in 0..20u64 {
        let q = rvec(50_000 + qi, dim as usize);
        // int4 without rescoring (the field default): quantized scores.
        let mut req = KnnRequest::new(F_I4, q.clone(), 10);
        req.num_candidates = 100;
        let (hits, st) = idx.knn_search_with(&req, &cfg_graph).unwrap();
        assert_eq!(st.graph_segments, 1);
        let got: Vec<i64> = hits.iter().map(|h| id_of(&idx, h)).collect();
        rec[0] += recall(&got, &brute_l2(&q, F_I4));
        // A per-request oversample override rescores with f32: exact scores.
        let (hits2, _) = idx.knn_search_with(&req, &cfg_graph.with_oversample(3.0)).unwrap();
        for h in &hits2 {
            let i = id_of(&idx, h);
            let exact = Similarity::L2.score(&q, &rvec(i as u64 + 1, dim as usize));
            assert!((h.score - exact).abs() < 1e-5, "rescored score is exact");
        }
        let got2: Vec<i64> = hits2.iter().map(|h| id_of(&idx, h)).collect();
        rec[1] += recall(&got2, &brute_l2(&q, F_I4));
        // int8_flat: brute force over the codes + 2x rescore (field default).
        let uq = unit(q.clone());
        let (hits3, st3) = idx.knn_search_with(&KnnRequest::new(F_I8F, uq.clone(), 10), &cfg_graph).unwrap();
        assert_eq!((st3.graph_segments, st3.exact_segments), (0, 1), "flat segments never traverse a graph");
        let got3: Vec<i64> = hits3.iter().map(|h| id_of(&idx, h)).collect();
        rec[2] += recall(&got3, &brute_l2(&uq, F_I8F));
        // bbq: searchable, rescored 3x by default.
        let (hb, _) = idx.knn_search_with(&KnnRequest::new(F_BBQ, q.clone(), 10), &cfg_graph).unwrap();
        assert_eq!(hb.len(), 10);
        // Forced exact search is plain f32 brute force.
        let (he, _) = idx.knn_search_with(&KnnRequest::new(F_I4, q.clone(), 10), &PlannerConfig::exact()).unwrap();
        assert_eq!(he.iter().map(|h| id_of(&idx, h)).collect::<Vec<_>>(), brute_l2(&q, F_I4));
    }
    let rec = rec.map(|r| r / 20.0);
    assert!(rec[0] >= 0.6, "int4 no rescore recall {}", rec[0]);
    assert!(rec[1] >= 0.9 && rec[1] >= rec[0], "int4 rescored recall {} vs raw {}", rec[1], rec[0]);
    assert!(rec[2] >= 0.9, "int8_flat rescored recall {}", rec[2]);

    // Changing a field's options takes effect on the next search without a
    // rebuild: oversample 0 on the int8_flat field returns quantized scores.
    idx.schema.iter_mut().find(|f| f.id == F_I8F).unwrap().vector_options = Some(VectorFieldOptions::quantized_hnsw(Quantization::Int8).flat().with_oversample(0.0));
    idx.sync_vector_field_options();
    let uq = unit(rvec(50_001, dim as usize));
    let (raw, _) = idx.knn_search_with(&KnnRequest::new(F_I8F, uq.clone(), 5), &cfg_graph).unwrap();
    let any_inexact = raw.iter().any(|h| {
        let exact = Similarity::DotProduct.score(&uq, &unit(rvec(id_of(&idx, h) as u64 + 1, dim as usize)));
        (h.score - exact).abs() > 1e-6
    });
    assert!(any_inexact, "without rescoring the int8 estimates are returned");
}

/// Recall@10 of every quantization with and without rescoring on a synthetic
/// 50k x 128d set. Prints a table; run with
/// `cargo test --release -p rustic-core quantization_recall_report -- --ignored --nocapture`.


// ---------------------------------------------------------------------------
// Sealing thresholds, force merge, graph reuse through the store.

#[test]
fn seal_threshold_is_size_based() {
    let d = OptimizerPolicy::default();
    // 20,000 KiB of f32: ~5k vectors at 1024 dims, capped by the planner's
    // full-scan threshold (10k) at small dims.
    assert_eq!(d.seal_threshold(1024), 5000);
    assert_eq!(d.seal_threshold(128), 10_000);
    assert_eq!(OptimizerPolicy { seal_full_scan_threshold: usize::MAX, ..d.clone() }.seal_threshold(128), 40_000);
    assert_eq!(OptimizerPolicy { indexing_threshold: 300, ..d.clone() }.seal_threshold(1024), 300);
    assert_eq!(d.seal_idle, Some(std::time::Duration::from_secs(5)));

    // Same number of vectors: the 1024-dim field crosses its byte threshold
    // and is sealed, the 128-dim one stays appendable.
    const F_SMALL: u16 = 1;
    const F_BIG: u16 = 2;
    let schema = vec![
        Field::new(F_ID, "id", FieldType::I64),
        Field::new(F_SMALL, "small", FieldType::Vector { dim: 128 }).with_similarity("cosine"),
        Field::new(F_BIG, "big", FieldType::Vector { dim: 1024 }).with_similarity("cosine"),
    ];
    let schema = schema.into_iter().map(|mut field| {
        if matches!(field.field_type, FieldType::Vector { .. }) {
            field.vector_options = Some(crate::vector::store::VectorFieldOptions::f32_hnsw());
        }
        field
    }).collect();
    let mut idx = activate_test_index(Index::new("s", schema, IndexSettings::default()));
    // 1000 KiB: 250 vectors at 1024 dims, 2000 at 128.
    idx.optimizer_policy = OptimizerPolicy { indexing_threshold: usize::MAX, indexing_threshold_kb: 1000, ..policy() };
    let n = if cfg!(debug_assertions) { 300 } else { 1000 };
    for i in 0..n {
        let mut d = Document::new();
        d.add_i64(F_ID, i);
        d.add_vector(F_SMALL, rvec(i as u64, 128));
        d.add_vector(F_BIG, rvec(i as u64, 1024));
        idx.add_document(&d).unwrap();
    }
    idx.refresh().unwrap();
    let r = idx.optimize_vectors().unwrap();
    assert_eq!(r.sealed, 1);
    let store = idx.vector_store().clone();
    assert!(store.segments(F_BIG).iter().all(|s| s.is_sealed() && !s.has_graph()));
    assert!(store.segments(F_SMALL).iter().all(|s| !s.is_sealed()));
    // The full-scan cap seals the 128-dim tail once it is past it.
    idx.optimizer_policy.seal_full_scan_threshold = (n as usize) - 1;
    assert_eq!(idx.optimize_vectors().unwrap().sealed, 1);
    assert!(store.segments(F_SMALL).iter().all(|s| s.is_sealed()));
}

#[test]
fn idle_tail_is_sealed_and_folded_into_small_sealed_segments() {
    let mut idx = new_index();
    let idle = std::time::Duration::from_millis(1000);
    idx.optimizer_policy.seal_idle = Some(idle);
    // Just appended: below every threshold and not idle yet (checked only
    // when this thread was not starved past the idle time by other tests).
    let not_idle_yet = |idx: &Index, appended: std::time::Instant| {
        let r = idx.optimize_vectors().unwrap();
        if appended.elapsed() < idle / 2 {
            assert_eq!(r, Default::default());
        }
    };
    let t = std::time::Instant::now();
    add_batch(&mut idx, 0..100);
    not_idle_yet(&idx, t);
    std::thread::sleep(idle + std::time::Duration::from_millis(50));
    idx.optimize_vectors().unwrap();
    let store = idx.vector_store().clone();
    assert!(store.segments(F_EMB2).iter().all(|s| s.is_sealed()), "emb2 tail sealed after going idle too");
    assert!(store.segments(F_EMB).iter().all(|s| s.is_sealed()) && store.segments(F_EMB).len() == 1);
    // Mutable topology is built at admission, never again by idle sealing.
    let ins = HnswBuilder::inserts_on_this_thread();
    let t = std::time::Instant::now();
    add_batch(&mut idx, 100..200);
    let admitted = HnswBuilder::inserts_on_this_thread();
    let emb2_new = (100..200).filter(|i| i % 3 == 0).count() as u64;
    assert_eq!(admitted - ins, 100 + emb2_new);
    not_idle_yet(&idx, t);
    std::thread::sleep(idle + std::time::Duration::from_millis(50));
    idx.optimize_vectors().unwrap();
    assert_eq!(HnswBuilder::inserts_on_this_thread() - admitted, 0);
    let segs = store.segments(F_EMB);
    assert_eq!((segs.len(), segs[0].len()), (1, 200));
    assert!(segs[0].is_sealed());
    let live: Vec<i64> = (0..200).collect();
    let q = rvec(4242, DIM);
    let got: Vec<i64> = idx.knn_search(&KnnRequest::new(F_EMB, q.clone(), 10)).unwrap().iter().map(|h| id_of(&idx, h)).collect();
    assert_eq!(got, brute(&live, &q, 10));
}

#[test]
fn small_sealed_segments_are_merged_below_max_segments() {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = 40;
    idx.optimizer_policy.max_segments = 100;
    for b in 0..3 {
        add_batch(&mut idx, b * 50..(b + 1) * 50);
        idx.optimize_vectors().unwrap();
    }
    let sizes = |idx: &Index| -> Vec<u32> { idx.vector_store().segments(F_EMB).iter().map(|s| s.len()).collect() };
    assert_eq!(sizes(&idx), vec![50, 50, 50]);
    // A larger threshold makes all three "small": they are combined even
    // though the field is far below max_segments, seeded with one graph.
    idx.optimizer_policy.indexing_threshold = 1000;
    let ins = HnswBuilder::inserts_on_this_thread();
    let r = idx.optimize_vectors().unwrap();
    assert!(r.merged >= 1);
    assert_eq!(sizes(&idx), vec![150]);
    let emb2: u64 = (0..150).filter(|i| i % 3 == 0).count() as u64;
    let emb2_seed = (0..50).filter(|i| i % 3 == 0).count() as u64;
    assert!(HnswBuilder::inserts_on_this_thread() - ins <= 100 + emb2 - emb2_seed.min(emb2));
}

/// `n` docs per batch, each batch sealed into its own segment, plus an
/// appendable chunk and some deletes.
fn multi_segment_index(batches: i64, n: i64) -> Index {
    let mut idx = new_index();
    idx.optimizer_policy.indexing_threshold = n as usize - 1;
    for b in 0..batches {
        add_batch(&mut idx, b * n..(b + 1) * n);
        idx.optimize_vectors().unwrap();
    }
    add_batch(&mut idx, batches * n..batches * n + n / 2); // chunk
    for i in (0..batches * n).step_by(11) {
        idx.delete_doc(&DocAddress::new((i / n) as u32, (i % n) as u32)).unwrap();
    }
    idx
}

#[test]
fn vector_search_pool_and_nested_rayon_paths_match_with_deletes() {
    let idx = multi_segment_index(4, 400);
    assert!(idx.vector_store().segments(F_EMB).len() > 1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for q in [rvec(12345, DIM), vec![0.0; DIM]] {
        let request = KnnRequest::new(F_EMB, q, 100);
        let parallel = idx.knn_search_with(&request, &PlannerConfig::exact()).unwrap();
        let nested = pool.install(|| idx.knn_search_with(&request, &PlannerConfig::exact()).unwrap());
        assert_eq!(ids_scores(&idx, &parallel.0), ids_scores(&idx, &nested.0));
        assert_eq!(parallel.0.len(), 100);
    }
}

#[test]
fn force_merge_vectors_to_one_and_to_n() {
    let n = if cfg!(debug_assertions) { 300 } else { 1500 };
    for target in [1usize, 2] {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = create_index(dir.path(), "fm");
        idx.optimizer_policy = policy();
        idx.optimizer_policy.indexing_threshold = n as usize - 1;
        for b in 0..4 {
            add_batch(&mut idx, b * n..(b + 1) * n);
            idx.optimize_vectors().unwrap();
        }
        add_batch(&mut idx, 4 * n..4 * n + n / 2);
        for i in (0..4 * n).step_by(11) {
            idx.delete_doc(&DocAddress::new((i / n) as u32, (i % n) as u32)).unwrap();
        }
        let store = idx.vector_store().clone();
        assert_eq!(store.segments(F_EMB).iter().filter(|s| s.is_sealed()).count(), 4);
        assert_eq!(store.segments(F_EMB).iter().filter(|s| !s.is_sealed()).count(), 1);
        let qs: Vec<Vec<f32>> = (0..10).map(|q| rvec(31_000 + q, DIM)).collect();
        let exact = |idx: &Index| -> Vec<Vec<(i64, f32)>> {
            qs.iter().map(|q| ids_scores(idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &PlannerConfig::exact()).unwrap().0)).collect()
        };
        let before = exact(&idx);
        let ins = HnswBuilder::inserts_on_this_thread();
        let r = idx.force_merge_vectors(target, false).unwrap();
        let inserted = HnswBuilder::inserts_on_this_thread() - ins;
        let segs = store.segments(F_EMB);
        assert_eq!(segs.len(), target, "target {target}: {r:?}");
        assert!(segs.iter().all(|s| s.is_sealed()), "no appendable chunk left");
        assert!(segs.iter().all(|s| !s.has_graph()),
            "mutable maintenance must not rebuild segmented graphs");
        assert!(store.segments(F_EMB2).iter().all(|s| s.is_sealed()) && store.segments(F_EMB2).len() <= target);
        // Payload compaction keeps every live row in the one mutable graph.
        assert!(store.num_alive(F_EMB) > 0);
        assert_eq!(inserted, 0, "payload merge cannot rebuild HNSW");
        assert_eq!(exact(&idx), before, "exact results unchanged");
        // Approximate search over the merged graph(s) keeps recall.
        let live: Vec<i64> = (0..4 * n + n / 2).filter(|i| !(*i < 4 * n && i % 11 == 0)).collect();
        let cfg = PlannerConfig { full_scan_threshold: 0, full_scan_fraction: 0.0, ..PlannerConfig::default() };
        let mut tot = 0.0;
        for q in &qs {
            let mut req = KnnRequest::new(F_EMB, q.clone(), 10);
            req.num_candidates = 100;
            let (h, st) = idx.knn_search_with(&req, &cfg).unwrap();
            assert_eq!(st.graph_segments + st.fallbacks, 1, "one mutable graph whatever the payload segment count");
            tot += recall(&h.iter().map(|h| id_of(&idx, h)).collect::<Vec<_>>(), &brute(&live, q, 10));
        }
        assert!(tot / qs.len() as f32 >= 0.9, "recall after force merge {}", tot / qs.len() as f32);
        // Idempotent; the commit drops the replaced files; reopen agrees.
        assert_eq!(idx.force_merge_vectors(target, false).unwrap(), Default::default());
        idx.commit().unwrap();
        let rvec_files = std::fs::read_dir(dir.path()).unwrap().filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".rvec")).count();
        assert_eq!(rvec_files, store.segment_ids().len());
        drop(idx);
        let idx = Index::open(dir.path()).unwrap();
        assert_eq!(idx.vector_store().segments(F_EMB).len(), target);
        assert_eq!(exact(&idx), before);
    }
}

/// Lucene's forced merge rewrites a segment holding deletions whatever their
/// share: a force-merged store keeps no dead rows (below the routine vacuum
/// ratio, an update-heavy shard kept up to that share of dead f32 rows).
#[test]
fn force_merge_drops_dead_rows_below_the_vacuum_ratio() {
    let n = if cfg!(debug_assertions) { 300 } else { 1500 };
    let idx = multi_segment_index(2, n);
    let store = idx.vector_store().clone();
    let dead_before: usize = store.segments(F_EMB).iter().filter(|s| s.is_sealed())
        .map(|s| s.points().iter().filter(|&&p| !store.is_alive(p)).count()).sum();
    assert!(dead_before > 0, "the fixture deletes ~9% per segment, below the vacuum ratio");
    // Two sealed segments and a chunk to two: the chunk merges into the
    // smaller sealed segment, the other is not merged, yet its ~9% dead rows
    // (below the 20-30% vacuum ratio) must go too.
    idx.force_merge_vectors(2, false).unwrap();
    let segs = store.segments(F_EMB);
    assert_eq!(segs.len(), 2);
    for s in &segs {
        assert!(s.points().iter().all(|&p| store.is_alive(p)), "no dead row left after a force merge");
    }
    assert_eq!(idx.force_merge_vectors(2, false).unwrap(), Default::default(), "idempotent");
}

#[test]
fn expunge_deletes_rebuilds_segments_with_dead_points() {
    let n = if cfg!(debug_assertions) { 300 } else { 1500 };
    let idx = multi_segment_index(2, n);
    let store = idx.vector_store().clone();
    let q = rvec(8, DIM);
    let before = ids_scores(&idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &PlannerConfig::exact()).unwrap().0);
    let ins = HnswBuilder::inserts_on_this_thread();
    let r = idx.force_merge_vectors(1, true).unwrap();
    // ~9 % dead per sealed segment: each is vacuumed reusing its own graph.
    assert_eq!(r.vacuumed, 2);
    assert_eq!(HnswBuilder::inserts_on_this_thread() - ins, 0, "vacuum below 20 % dead inserts nothing");
    for s in store.segments(F_EMB).iter().filter(|s| s.is_sealed()) {
        assert_eq!(store.num_alive(F_EMB) >= s.len() as usize, true);
        assert!(s.points().iter().all(|&p| store.is_alive(p)), "dead points dropped");
    }
    assert!(store.segments(F_EMB).iter().any(|s| !s.is_sealed()), "expunge leaves chunks alone");
    assert_eq!(ids_scores(&idx, &idx.knn_search_with(&KnnRequest::new(F_EMB, q, 10), &PlannerConfig::exact()).unwrap().0), before);
}



/// `Query::Knn` through a `Searcher` over an index's leaves searches the
/// shared vector store once (not once per segment leaf) and returns exactly
/// what the index-level `knn_search` does: sealed graph segments, append
/// chunks and the memtable, with deletions, unfiltered and with broad and
/// selective filters.
#[test]
fn query_knn_matches_index_knn_across_leaves() {
    let mut idx = new_index();
    let n = batch();
    add_batch(&mut idx, 0..n);
    add_batch(&mut idx, n..2 * n);
    idx.optimizer_policy.indexing_threshold = (n as usize) + 1;
    assert_eq!(idx.optimize_vectors().unwrap().sealed, 1);
    add_batch(&mut idx, 2 * n..3 * n); // append chunk
    for i in 3 * n..3 * n + 40 {
        idx.add_document(&doc(i)).unwrap(); // memtable
    }
    for i in (0..3 * n).step_by(11) {
        idx.delete_doc(&DocAddress::new((i / n) as u32, (i % n) as u32)).unwrap();
    }
    assert!(idx.leaves().len() >= 4);
    let filters = [None, Some(Query::term(F_TAG, "other")), Some(Query::term(F_TAG, "ten"))];
    for (qi, filter) in filters.iter().enumerate() {
        for q in 0..6 {
            let v = rvec(20_000 + q, DIM);
            let mut req = KnnRequest::new(F_EMB, v.clone(), 10);
            req.filter = filter.clone();
            let want = ids_scores(&idx, &idx.knn_search(&req).unwrap());
            let knn = Query::Knn {
                field: F_EMB,
                vector: v,
                k: 10,
                num_candidates: req.num_candidates,
                filter: filter.clone().map(Box::new),
                similarity_threshold: None,
                boost: 1.0,
                oversample: None,
                visit_percentage: None,
            };
            let td = Searcher::new(idx.leaves()).search(&knn, &SearchOptions { top_k: 10, ..Default::default() });
            let got: Vec<(i64, f32)> = td.hits.iter().map(|h| (id_of(&idx, &KnnHit { leaf: h.leaf, doc: h.doc, score: h.score }), h.score)).collect();
            assert_eq!(got.len(), want.len(), "filter {qi} query {q}");
            for ((gi, gs), (wi, ws)) in got.iter().zip(&want) {
                assert_eq!(gi, wi, "filter {qi} query {q}: {got:?} vs {want:?}");
                assert!((gs - ws).abs() < 1e-5, "filter {qi} query {q}: {got:?} vs {want:?}");
            }
            assert!(got.iter().all(|(id, _)| id % 11 != 0 || *id >= 3 * n), "deleted doc returned");
            if qi == 2 {
                assert!(got.iter().all(|(id, _)| id % 10 == 0), "selective filter violated");
            }
        }
    }
}

#[test]
fn full_scan_limit_scales_with_dimension() {
    let cfg = PlannerConfig::default();
    assert_eq!(cfg.full_scan_limit(1024), 2_500, "10,000 KB of 4 KiB vectors");
    assert_eq!(cfg.full_scan_limit(128), 10_000, "count bound wins for small vectors");
    assert_eq!(PlannerConfig::exact().full_scan_limit(4096), usize::MAX);
    // A sealed segment under the count bound but over the byte bound goes
    // through the graph instead of being brute forced.
    let mut idx = new_index();
    add_batch(&mut idx, 0..200);
    idx.optimizer_policy.indexing_threshold = 100;
    assert_eq!(idx.optimize_vectors().unwrap().sealed, 1);
    let q = rvec(31_337, DIM);
    let (_, st) = idx.knn_search_with(&KnnRequest::new(F_EMB, q.clone(), 10), &PlannerConfig::default()).unwrap();
    assert_eq!((st.exact_segments, st.graph_segments), (0, 1),
        "mutable planning uses the shard's Vespa policy, not a text-segment size threshold");
    let tight = PlannerConfig { full_scan_threshold_kb: 1, ..PlannerConfig::default() }; // 16 dims: 16 vectors
    let (hits, st) = idx.knn_search_with(&KnnRequest::new(F_EMB, q, 10), &tight).unwrap();
    assert_eq!((st.exact_segments, st.graph_segments), (0, 1), "byte bound sends it to the graph");
    assert_eq!(hits.len(), 10);
}

/// The vector optimizer runs detached from the index (callers release the
/// index lock during graph builds), so commits interleave with its segment
/// swaps: every commit point must reference only files that exist, and the
/// reopened index must hold every vector.
#[test]
fn commits_interleaved_with_detached_optimizer_stay_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = create_index(dir.path(), "c");
    idx.optimizer_policy = OptimizerPolicy { indexing_threshold: 150, ..policy() };
    for b in 0..6 {
        add_batch(&mut idx, b * 150..(b + 1) * 150);
    }
    idx.commit().unwrap();
    let listed_files_exist = || {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        let vs = &meta["vector_store"];
        let ids: Vec<u64> = vs["segments"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        for id in &ids {
            let f = crate::codec::vector_segment_file_name(*id as u32);
            assert!(dir.path().join(&f).exists(), "commit point references deleted {f}");
            assert!(*id < vs["next_segment_id"].as_u64().unwrap(), "next_segment_id must exceed listed id {id}");
        }
    };
    let job = idx.vector_optimizer();
    let mut commits = 0;
    std::thread::scope(|s| {
        let h = s.spawn(|| job.run().unwrap());
        while !h.is_finished() {
            idx.commit().unwrap();
            listed_files_exist();
            commits += 1;
        }
        assert!(h.join().unwrap().sealed >= 1);
    });
    // Interleaving is best effort (the build may finish before the first
    // commit); consistency must hold either way.
    let _ = commits;
    idx.commit().unwrap();
    listed_files_exist();
    let reopened = Index::open(dir.path()).unwrap();
    let stored: u32 = reopened.vector_store().segments(F_EMB).iter().map(|s| s.len()).sum();
    assert_eq!(stored, 900, "every vector survives");
    let q = rvec(42, DIM);
    let hits = reopened.knn_search(&KnnRequest::new(F_EMB, q, 1)).unwrap();
    assert_eq!(id_of(&reopened, &hits[0]), 42);
}

/// A seal after the last commit is tracked until the next commit, so a
/// host can commit it (cluster idle flush) instead of redoing it on reopen.
#[test]
fn seal_is_uncommitted_until_the_next_commit() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = create_index(dir.path(), "u");
    idx.optimizer_policy = OptimizerPolicy { indexing_threshold: 100, ..policy() };
    add_batch(&mut idx, 0..300);
    idx.commit().unwrap();
    assert!(!idx.vector_store().has_uncommitted_changes());
    assert_eq!(idx.optimize_vectors().unwrap().sealed, 1);
    assert!(idx.vector_store().has_uncommitted_changes(), "the seal is not in the commit point yet");
    idx.commit().unwrap();
    assert!(!idx.vector_store().has_uncommitted_changes());
    let mut reopened = Index::open(dir.path()).unwrap();
    reopened.optimizer_policy = idx.optimizer_policy.clone();
    assert_eq!(reopened.optimize_vectors().unwrap().sealed, 0, "nothing left to re-seal after reopen");
}

#[test]
fn payload_partitions_follow_point_owners_through_reopen_delete_and_merge() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = create_index(dir.path(), "payload");
    idx.schema[F_EMB as usize].vector_options = Some(crate::vector::store::VectorFieldOptions::f32_hnsw());
    idx.optimizer_policy = OptimizerPolicy { indexing_threshold: 1024, ..policy() };
    add_batch(&mut idx, 0..2048);
    idx.commit().unwrap();
    idx.optimize_vectors().unwrap();
    idx.commit().unwrap();
    let covered = |index: &Index| {
        // Preserve the payload eligibility/owner regression without depending
        // on the removed per-segment graph overlay. Exercise the shared graph
        // against independently enumerated current document owners.
        let mut accepted = HashSet::new();
        for leaf in index.segments() {
            let ff = leaf.fast_fields();
            let column = ff.get_keyword(F_TAG).unwrap();
            for (doc, point) in leaf.point_map().iter(F_EMB) {
                if !leaf.is_deleted(doc) && column.get(doc).as_deref() == Some("ten") { accepted.insert(point); }
            }
        }
        let mut request = KnnRequest::new(F_EMB, rvec(37, DIM), 10);
        request.num_candidates = 100;
        request.filter = Some(Query::term(F_TAG, "ten"));
        let hits = index.knn_search(&request).unwrap();
        assert_eq!(hits.len(), accepted.len().min(10));
        for hit in hits {
            let point = index.segments()[hit.leaf].point(F_EMB, hit.doc).unwrap();
            assert!(accepted.contains(&point), "filter must resolve through current point ownership");
        }
    };
    covered(&idx);
    let mut reopened = Index::open(dir.path()).unwrap();
    reopened.optimizer_policy = idx.optimizer_policy.clone();
    covered(&reopened);
    // Another independent index has no overlay belonging to this index.
    let mut other = new_index();
    other.schema[F_EMB as usize].vector_options = Some(crate::vector::store::VectorFieldOptions::f32_hnsw());
    other.optimizer_policy.indexing_threshold = 1024;
    add_batch(&mut other, 10000..12048);
    other.commit().unwrap();
    other.optimize_vectors().unwrap();
    covered(&other);
    let independent = other.knn_search(&KnnRequest::new(F_EMB, rvec(10000, DIM), 1)).unwrap();
    assert_eq!(id_of(&other, &independent[0]), 10000);
    let hit = reopened.knn_search(&KnnRequest::new(F_EMB, rvec(0, DIM), 1)).unwrap();
    assert_eq!(id_of(&reopened, &hit[0]), 0);
    reopened.delete_doc(&DocAddress::new(hit[0].leaf as u32, hit[0].doc)).unwrap();
    reopened.commit().unwrap();
    let after = reopened.knn_search(&KnnRequest::new(F_EMB, rvec(0, DIM), 10)).unwrap();
    assert!(after.iter().all(|h| id_of(&reopened, h) != 0));
    add_batch(&mut reopened, 2048..4096);
    reopened.commit().unwrap();
    reopened.force_merge_vectors(1, false).unwrap();
    reopened.commit().unwrap();
    let merged = Index::open(dir.path()).unwrap();
    assert_eq!(merged.vector_store().segments(F_EMB).len(), 1);
    covered(&merged);
    let after = merged.knn_search(&KnnRequest::new(F_EMB, rvec(0, DIM), 10)).unwrap();
    assert!(after.iter().all(|h| id_of(&merged, h) != 0));
}


#[test]
fn ingestion_point_reservation_survives_retry_and_publishes_only_once() {
    let dir = RamDirectory::new();
    let store = VectorStore::new();
    let reservation = store.reserve_points(2).unwrap();
    assert_eq!(reservation.points(), 0..2);
    assert_eq!(store.next_point_id(), 2);
    assert!(store.segment_ids().is_empty());
    let values = [rvec(0, DIM), rvec(1, DIM)];
    let rows = [(0, values[0].as_slice()), (1, values[1].as_slice())];
    let other = VectorStore::new();
    assert!(other.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows, &reservation).is_err());
    assert!(store.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows[..1], &reservation).is_err());
    let first = store.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows, &reservation).unwrap();
    assert_eq!(first.points(), &[0, 1]);
    assert!(store.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows, &reservation).is_err());
    drop(first);
    assert!(dir.list().unwrap().is_empty());
    let mut retry = store.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows, &reservation).unwrap();
    assert_eq!(retry.points(), &[0, 1]);
    // Produce a real text point map with the same IDs, in an independent
    // directory so its standalone vector chunk cannot overwrite this one.
    let text = RamDirectory::new();
    let mut writer = SegmentWriter::new(7);
    for value in &values {
        let mut document = Document::new();
        document.add_vector(F_EMB, value.clone());
        writer.add_document(&document, &schema());
    }
    let segment = writer.write_to(&text).unwrap();
    assert_eq!(segment.point(F_EMB, 0), Some(0));
    assert_eq!(segment.point(F_EMB, 1), Some(1));
    store.attach_prepared(&segment, std::slice::from_mut(&mut retry), false);
    drop(retry);
    assert_eq!(store.num_alive(F_EMB), 2);
    assert!(store.prepare_reserved_append(&dir, F_EMB, Similarity::Cosine, DIM as u32, &rows, &reservation).is_err());
    assert_eq!(store.next_point_id(), 2, "retry must preserve ingestion identities");
}

#[test]
fn point_reservations_do_not_wrap_or_cross_captured_store_allocators() {
    let dir = RamDirectory::new();
    // Sparse storage separates the persisted allocator watermark from
    // resident pages. Opening an empty historical store must allocate no
    // dense table, but overflowing its allocator still refuses atomically.
    let historical=VectorStore::open(&dir,&[],u64::MAX-1,0).unwrap();
    assert_eq!(historical.next_point_id(),u64::MAX-1);
    assert!(historical.segment_ids().is_empty());
    assert!(matches!(historical.reserve_points(2),Err(crate::error::RusticError::Memory(_))));
    assert_eq!(historical.next_point_id(),u64::MAX-1,"refusal cannot consume or wrap IDs");
    assert!(historical.segment_ids().is_empty());
    // Exact zero-allocation overflow refusal is also checked with a private
    // budget in store::point_table_admission_tests.
    let store = VectorStore::new();
    let reservation = store.reserve_points(1).unwrap();
    let captured = store.capture_read_store().unwrap();
    assert!(captured.prepare_reserved_append(&dir, F_EMB, Similarity::L2, 2, &[(0, &[1., 2.])], &reservation).is_err());
}


#[test]
fn writer_ingestion_ids_survive_append_renumber_failed_flush_and_chunk_coalescing() {
    use std::sync::atomic::Ordering;
    let dir = PausingVectorDirectory::new(9, 1);
    dir.pause.store(false, Ordering::Relaxed);
    let store = VectorStore::new();
    let mut writer = SegmentWriter::new(77);
    for i in 0..10 { writer.add_document(&doc(i), &schema()); }
    writer.reserve_vector_points(&store).unwrap();
    let before = writer.reserved_vector_points(F_EMB);
    assert_eq!(before.len(), 10);
    for i in 10..20 { writer.add_document(&doc(i), &schema()); }
    writer.reserve_vector_points(&store).unwrap();
    let expected = writer.reserved_vector_points(F_EMB);
    assert_eq!(&expected[..10], &before);
    assert_eq!(expected.len(), 20);
    let allocated = store.next_point_id();
    writer.reserve_vector_points(&store).unwrap();
    assert_eq!(store.next_point_id(), allocated, "reservation is idempotent without new rows");
    assert!(writer.reserve_vector_points(&VectorStore::new()).is_err());
    writer.id = 9;
    writer.delete_doc(3);
    assert!(writer.write_to_store_ref(&dir, Some(&store), false).is_err());
    assert!(store.segment_ids().is_empty());
    assert_eq!(writer.reserved_vector_points(F_EMB), expected);
    dir.failure.store(0, Ordering::Relaxed);
    let segment = writer.write_to_store_ref(&dir, Some(&store), false).unwrap();
    for (doc, point) in expected { assert_eq!(segment.point(F_EMB, doc), Some(point)); }
    assert_eq!(store.next_point_id(), allocated, "flush reuses ingestion IDs");
    assert_eq!(store.segment_ids().len(), 2, "one chunk per field, not per ingestion batch");
    assert_eq!(store.num_alive(F_EMB), 19);
}

#[test]
fn writer_reserved_nested_identities_follow_child_first_document_order() {
    let store = VectorStore::new();
    let mut writer = SegmentWriter::new(0);
    let mut parent = doc(2);
    parent.add_nested("child", doc(1));
    writer.add_document(&parent, &schema());
    writer.reserve_vector_points(&store).unwrap();
    let ids = writer.reserved_vector_points(F_EMB);
    assert_eq!(ids.iter().map(|(doc, _)| *doc).collect::<Vec<_>>(), vec![0, 1]);
    let dir = RamDirectory::new();
    let segment = writer.write_to_store(&dir, Some(&store)).unwrap();
    for (doc, point) in ids { assert_eq!(segment.point(F_EMB, doc), Some(point)); }
    assert_eq!(segment.vector(F_EMB, 0), Some(rvec(1, DIM)));
    assert_eq!(segment.vector(F_EMB, 1), Some(rvec(2, DIM)));
}

#[test]
fn detached_vector_seal_reconciles_refresh_and_concurrent_deletes() {
    for publish_during in [false, true] {
        let mut index = new_index();
        index.set_nrt(true);
        index.set_auto_seal(false);
        for i in 0..20 { index.add_document(&doc(i)).unwrap(); }
        let job = index.begin_seal(true).unwrap().expect("vector encoding must detach");
        let sealed = job.run();
        // Change visibility after the encoder captured its snapshot, before
        // installation. Also retain an unrelated newly written document.
        if publish_during { index.publish().unwrap(); }
        let deleted = index.all_leaves().iter().enumerate().find_map(|(i, leaf)| {
            (0..leaf.num_docs()).find(|&d| leaf.i64_value(F_ID, d) == Some(7)).map(|d| DocAddress::new(i as u32, d))
        }).unwrap();
        index.delete_doc(&deleted).unwrap();
        index.add_document(&doc(20)).unwrap();
        index.finish_seal(sealed).unwrap();
        let request = KnnRequest::new(F_EMB, rvec(7, DIM), 30);
        let before = index.knn_search(&request).unwrap();
        assert_eq!(before.len(), if publish_during { 20 } else { 0 });
        index.publish().unwrap();
        let after = index.knn_search(&request).unwrap();
        let ids: HashSet<_> = after.iter().map(|h| id_of(&index, h)).collect();
        assert_eq!(ids, (0..21).filter(|&i| i != 7).collect());
        let job = index.begin_seal(true).unwrap().unwrap();
        index.finish_seal(job.run()).unwrap();
        index.commit().unwrap();
    }
}

// ---------------------------------------------------------------------------
// `rescore_vector` and quantized `_score` parity (B5) on the query path the
// ES engine uses (`Query::Knn` through the searcher).

#[cfg(feature = "mutable-hnsw-poc")]
#[test]
fn quantized_graph_knn_returns_f32_scores_rescored_and_code_estimates_otherwise() {
    use crate::vector::quantize::Quantization;
    use crate::vector::store::VectorFieldOptions;
    const F: u16 = 9;
    // (type, similarity, dims, estimate tolerance relative to the score,
    // f32 rank the unrescored hits must reach): 1-bit BBQ codes estimate a
    // score far more coarsely than 4/7-bit ones (ES refuses bbq below 64
    // dims for that reason and rescores it 3x by default), so a single
    // estimate is not bounded, only the ranking it produces; the estimate
    // itself is ES's OSQ `applyCorrections`.
    for (quantization, name, similarity, dims, tolerance, rank) in [(Quantization::Int8, "l2_norm", Similarity::L2, 32, 0.25, 20),
        (Quantization::Int4, "cosine", Similarity::Cosine, 32, 0.25, 30), (Quantization::Binary, "max_inner_product", Similarity::MaxInnerProduct, 64, f32::INFINITY, 100)] {
        let mut schema = supported_schema();
        schema.push(Field::new(F, "q", FieldType::Vector { dim: dims as u16 }).with_similarity(name)
            .with_vector_options(Some(VectorFieldOptions::quantized_hnsw(quantization).with_osq(true).with_oversample(0.0))));
        let mut idx = activate_test_index(Index::new("rescore", schema, IndexSettings::default()));
        for i in 0..600i64 {
            let mut d = Document::new();
            d.add_i64(F_ID, i);
            d.add_keyword(F_TAG, if i % 2 == 0 { "even" } else { "odd" });
            d.add_vector(F, rvec(i as u64 + 1, dims));
            idx.add_document(&d).unwrap();
            if i % 150 == 149 { idx.refresh().unwrap(); }
        }
        idx.refresh().unwrap();
        for qi in 0..8u64 {
            let q = rvec(90_000 + qi, dims);
            for filter in [None, Some(Box::new(Query::term(F_TAG, "even")))] {
                let run = |oversample: Option<f32>| {
                    let knn = Query::Knn { field: F, vector: q.clone(), k: 10, num_candidates: 64, filter: filter.clone(),
                        similarity_threshold: None, boost: 1.0, oversample, visit_percentage: None };
                    let result = Searcher::new(idx.leaves()).search(&knn, &SearchOptions { top_k: 10, ..Default::default() });
                    assert_eq!(result.execution_error, None);
                    assert_eq!(result.hits.len(), 10);
                    result.hits.iter().map(|h| (h.score, idx.leaves()[h.leaf].with_vector(F, h.doc, <[f32]>::to_vec).unwrap())).collect::<Vec<_>>()
                };
                for (score, row) in run(Some(3.0)) {
                    let exact = similarity.score(&q, &row);
                    assert!((score - exact).abs() <= 1e-6 * exact.abs().max(1.0), "{quantization:?}: rescored {score} is the f32 similarity {exact}");
                }
                // Unrescored, the graph returns each row's own code score
                // (its calibration epoch: a Lucene segment's quantizer), read
                // from the codes alone: a quantized estimate of the f32 one.
                assert!(idx.vector_store().mutable_query_field(F).unwrap().score_quantizer().is_some(), "quantized field");
                let raw = run(None);
                let mut inexact = false;
                for (score, row) in &raw {
                    let exact = similarity.score(&q, row);
                    assert!((score - exact).abs() <= tolerance * exact.abs().max(1.0), "{quantization:?}: estimate {score} far from {exact}");
                    inexact |= (score - exact).abs() > 1e-6;
                }
                assert!(inexact, "{quantization:?}: unrescored scores are quantized estimates");
                assert!(raw.windows(2).all(|w| w[0].0 >= w[1].0));
                // The estimates rank: most unrescored hits are among the
                // f32 best `rank` documents of the searched set.
                let mut all: Vec<f32> = (0..600i64).filter(|i| filter.is_none() || i % 2 == 0)
                    .map(|i| similarity.score(&q, &rvec(i as u64 + 1, dims))).collect();
                all.sort_by(|a, b| b.total_cmp(a));
                let bar = all[rank - 1];
                let good = raw.iter().filter(|(_, row)| similarity.score(&q, row) >= bar).count();
                assert!(good >= 7, "{quantization:?}: only {good} of 10 unrescored hits are in the f32 top {rank}");
            }
        }
    }
}
