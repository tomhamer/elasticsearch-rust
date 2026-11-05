//! The shared DiskBBQ index on its own: training, inserts, maintenance
//! (split, drop, compaction), deletes, persistence, codes and recall.

use super::*;
use crate::codec::directory::RamDirectory;
use crate::vector::ivf::{leaf_budget, test_corpus, Rng};

/// A field's rows: point `i` is `rows[i]`.
struct Field {
    dir: RamDirectory,
    sim: Similarity,
    dim: usize,
    options: IvfOptions,
    rows: Vec<Vec<f32>>,
    state: Option<IvfState>,
    dead: HashSet<u64>,
}

impl Field {
    fn new(sim: Similarity, dim: usize, options: IvfOptions) -> Self {
        Field { dir: RamDirectory::new(), sim, dim, options, rows: Vec::new(), state: None, dead: HashSet::new() }
    }

    fn params(&self) -> BuildParams {
        BuildParams::of(self.sim, self.dim, &self.options)
    }

    fn writer(&self) -> Writer<'_> {
        Writer { dir: &self.dir, field: 3, throttled: false, work: Box::leak(Box::new(|_: usize| {})), written: Vec::new() }
    }

    /// Add `rows` (train on the first call, insert after).
    fn add(&mut self, rows: Vec<Vec<f32>>) {
        let first = self.rows.len() as u64;
        self.rows.extend(rows);
        let new: Vec<(u64, &[f32])> = (first..self.rows.len() as u64).map(|p| (p, self.rows[p as usize].as_slice())).collect();
        let params = self.params();
        let mut w = self.writer();
        let state = match &self.state {
            None => {
                let (c, b, next) = train(&mut w, params, &self.options, &new, 1).unwrap();
                IvfState::assemble(3, c, vec![b], indexed_of(new.iter().map(|r| r.0)).unwrap(), HashSet::new(), next).unwrap()
            }
            Some(state) => insert(&mut w, state, &new, &[]).unwrap().state,
        };
        self.state = Some(state);
    }

    /// Maintenance steps until none is due; how many ran.
    fn maintain(&mut self, force: bool) -> usize {
        let mut steps = 0;
        for _ in 0..512 {
            let state = self.state.as_ref().unwrap();
            let dead = &self.dead;
            let kept = |p: u64| (p as usize) < self.rows.len() && !dead.contains(&p);
            let (live, garbage) = state.census(&kept).unwrap();
            let Some(action) = next_action(state, &self.options, &live, garbage, force) else { break };
            let mut w = self.writer();
            let step = match action {
                Action::Split(cs) => {
                    let groups: Vec<_> = cs.iter().map(|&c| (c, members(state, &self.rows, &kept, c))).collect();
                    let step = split_many(&mut w, state, &groups, &self.options).unwrap();
                    if step.obsolete.is_empty() && step.state.unsplittable().len() == state.unsplittable().len() { None } else { Some(step) }
                }
                Action::Drop(cs) => {
                    let groups: Vec<_> = cs.iter().map(|&c| (c, members(state, &self.rows, &kept, c))).collect();
                    drop_many(&mut w, state, &groups).unwrap()
                }
                Action::Compact(which) => Some(compact(&mut w, state, &which, &kept).unwrap()),
            };
            let Some(step) = step else { break };
            self.state = Some(step.state);
            steps += 1;
        }
        steps
    }

    fn search(&self, q: &[f32], k: usize, num_candidates: usize, ratio: f32, oversample: f32, accept: impl Fn(u64) -> bool + Sync) -> (Vec<u64>, IvfStats) {
        let state = self.state.as_ref().unwrap();
        let live = self.rows.len() - self.dead.len();
        let (hits, stats) = state.search(q, k, num_candidates, ratio, leaf_budget(k, oversample), Some(live), None, false, |p| !self.dead.contains(&p) && accept(p)).unwrap();
        // The store's rescoring.
        let keep = ((k as f32 * oversample.max(1.0)).ceil() as usize).max(k);
        let mut r: Vec<(u64, f32)> = hits.iter().take(keep).map(|&(p, _)| (p, self.sim.score(q, &self.rows[p as usize]))).collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        (r.into_iter().take(k).map(|x| x.0).collect(), stats)
    }

    fn exact(&self, q: &[f32], k: usize, accept: impl Fn(u64) -> bool) -> Vec<u64> {
        let mut all: Vec<(u64, f32)> = (0..self.rows.len() as u64).filter(|p| !self.dead.contains(p) && accept(*p)).map(|p| (p, self.sim.score(q, &self.rows[p as usize]))).collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.into_iter().take(k).map(|x| x.0).collect()
    }

    fn recall(&self, queries: &[Vec<f32>], k: usize, num_candidates: usize, ratio: f32) -> (f64, u64) {
        let mut total = 0.0;
        let mut visited = 0;
        for q in queries {
            let (got, st) = self.search(q, k, num_candidates, ratio, 3.0, |_| true);
            let want: HashSet<u64> = self.exact(q, k, |_| true).into_iter().collect();
            total += got.iter().filter(|g| want.contains(g)).count() as f64 / want.len().max(1) as f64;
            visited += st.entries_visited;
        }
        (total / queries.len() as f64, visited)
    }
}

/// The live entries of centroid `c` and their rows.
fn members<'a>(state: &IvfState, rows: &'a [Vec<f32>], kept: &dyn Fn(u64) -> bool, c: usize) -> Vec<(u64, &'a [f32])> {
    state.list_entries(c).into_iter().filter(|e| kept(e & !SPILL)).map(|e| (e, rows[(e & !SPILL) as usize].as_slice())).collect()
}

fn opts(cluster_size: u32, bits: u8) -> IvfOptions {
    IvfOptions { cluster_size, bits, flat_index_threshold: 0, ..IvfOptions::default() }
}

/// `n` corpus vectors and `queries` more from the same distribution.
fn split_corpus(seed: u64, n: usize, queries: usize, dim: usize, centres: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut all = test_corpus(seed, n + queries, dim, centres);
    let q = all.split_off(n);
    (all, q)
}

#[test]
fn training_lists_every_vector_once_plus_its_spill() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let mut f = Field::new(Similarity::L2, 32, opts(64, 1));
    f.add(test_corpus(4, 3000, 32, 12));
    let state = f.state.as_ref().unwrap();
    let mut seen = vec![0u32; 3000];
    let mut spilled = 0;
    for c in 0..state.centroids.len() {
        for e in state.list_entries(c) {
            seen[(e & !SPILL) as usize] += 1;
            spilled += usize::from(e & SPILL != 0);
        }
    }
    assert!(seen.iter().all(|&s| s == 1 || s == 2));
    assert!(spilled > 2700, "nearly every vector spills ({spilled})");
    assert_eq!(seen.iter().map(|&s| u64::from(s)).sum::<u64>(), state.entries());
    assert_eq!(state.indexed.count(), 3000);
}

/// Mean `|estimate - exact| / (|q - c| |x - c|)` of the inner product of
/// a query and the vectors of one list, per code width.
#[test]
fn quantization_error_is_bounded_and_shrinks_with_bits() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let dim = 128;
    let mut r = Rng::new(5);
    let rows: Vec<Vec<f32>> = (0..400).map(|_| (0..dim).map(|_| r.gaussian() as f32).collect()).collect();
    let queries: Vec<Vec<f32>> = (0..8).map(|_| (0..dim).map(|_| r.gaussian() as f32).collect()).collect();
    let mut errors = Vec::new();
    for (bits, bound) in [(1u8, 0.15), (2, 0.08), (4, 0.03), (7, 0.01)] {
        let mut f = Field::new(Similarity::MaxInnerProduct, dim, opts(1024, bits));
        f.add(rows.clone());
        let state = f.state.as_ref().unwrap();
        assert_eq!(state.centroids.active(), 1, "400 vectors fit one cluster of 1024");
        let c = state.centroids.vector(0).to_vec();
        let (mut sum, mut count) = (0.0, 0.0);
        for q in &queries {
            let qn = l2_sq(q, &c).sqrt();
            for (e, score) in state.score_list(q, 0) {
                let v = &rows[(e & !SPILL) as usize];
                // Invert ES's max_inner_product score transform.
                let est = if score < 1.0 { 1.0 - 1.0 / score } else { score - 1.0 };
                sum += ((est - dot(q, v)).abs() / (qn * l2_sq(v, &c).sqrt())) as f64;
                count += 1.0;
            }
        }
        let err = sum / count;
        assert!(err < bound, "{bits} bits: mean relative error {err} >= {bound}");
        errors.push(err);
    }
    assert!(errors.windows(2).all(|w| w[0] > w[1]), "error shrinks with bits: {errors:?}");
}

/// ES's `num_candidates` reaching `computeDynamicVisitRatio`
/// (`DenseVectorFieldMapper.createKnnQuery`: at least
/// `min(ceil(k * oversample), 10,000)` when rescoring).
fn es_num_candidates(k: usize, num_candidates: usize, oversample: f32) -> usize {
    if oversample > 0.0 { num_candidates.max(((k as f64 * f64::from(oversample)).ceil() as usize).min(10_000)) } else { num_candidates }
}

/// Recall@10 on a seeded 20k x 128 corpus at ES's defaults (`cluster_size`
/// 384, dynamic visit ratio, `num_candidates` 1.5 k adjusted as ES does,
/// 1 bit, oversample 3), visiting at least what ES would.
#[test]
fn recall_at_the_default_visit_ratio() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(11, 20_000, 50, 128, 64);
    let num_candidates = es_num_candidates(10, 15, 3.0);
    assert_eq!(num_candidates, 30);
    let es_visit = (2.0 * f64::from(ivf::dynamic_visit_ratio(num_candidates, 10).min(ivf::segment_size_cap(20_000))) * 20_000.0) as u64;
    for sim in [Similarity::L2, Similarity::Cosine] {
        let mut f = Field::new(sim, 128, IvfOptions::default());
        f.add(rows.clone());
        let (recall, visited) = f.recall(&queries, 10, num_candidates, 0.0);
        let state = f.state.as_ref().unwrap();
        eprintln!("IVF_RECALL sim={sim:?} centroids={} entries={} recall@10={recall:.3} visited/query={} es_budget={es_visit}", state.centroids.active(), state.entries(), visited / queries.len() as u64);
        assert!(recall >= 0.9, "{sim:?}: recall@10 {recall}");
        assert!(visited >= es_visit * queries.len() as u64, "at least ES's visit budget: {visited} < {es_visit} x {}", queries.len());
        assert!(visited < 4 * (es_visit + 1_000) * queries.len() as u64, "a few lists ({visited})");
    }
}

/// Rows added after training (inserts, then the splits maintenance does)
/// keep recall.
#[test]
fn incremental_inserts_keep_recall() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(17, 20_000, 40, 128, 64);
    let mut f = Field::new(Similarity::L2, 128, IvfOptions::default());
    f.add(rows[..4_000].to_vec());
    let trained = f.state.as_ref().unwrap().centroids.active();
    for chunk in rows[4_000..].chunks(4_000) {
        f.add(chunk.to_vec());
        f.maintain(false);
    }
    let state = f.state.as_ref().unwrap();
    assert_eq!(state.indexed.count(), 20_000);
    assert!(state.centroids.active() > trained, "maintenance split the growing clusters ({} -> {})", trained, state.centroids.active());
    assert!(state.batches.len() <= MAX_BATCHES, "{} batches", state.batches.len());
    let (live, _) = state.census(&|_| true).unwrap();
    assert!(live.iter().all(|&n| n <= SPLIT_FACTOR * 384), "no cluster above the split threshold: {:?}", live.iter().max());
    let (recall, _) = f.recall(&queries, 10, 15, 0.0);
    eprintln!("IVF_INCREMENTAL centroids={} batches={} recall@10={recall:.3}", state.centroids.active(), state.batches.len());
    assert!(recall >= 0.85, "recall@10 after inserts {recall}");
}

#[test]
fn deletes_are_skipped_then_compacted_away() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(19, 6_000, 10, 32, 16);
    let mut f = Field::new(Similarity::L2, 32, opts(128, 1));
    f.add(rows);
    let before = f.state.as_ref().unwrap().entries();
    f.dead = (0..6_000u64).filter(|p| p % 3 == 0).collect();
    for q in &queries {
        let (got, _) = f.search(q, 10, 100, 0.0, 3.0, |_| true);
        assert!(got.iter().all(|p| p % 3 != 0), "deleted vectors never returned");
    }
    assert!(f.maintain(false) > 0, "a third dead: compaction is due");
    let state = f.state.as_ref().unwrap();
    let after = state.entries();
    assert!(after < before * 3 / 4, "dead entries compacted away: {before} -> {after}");
    assert_eq!(state.indexed.count(), 6_000, "still indexed: never re-inserted");
    let (_, garbage) = state.census(&|p| p % 3 != 0).unwrap();
    assert!(garbage < 0.05, "garbage left {garbage}");
}

#[test]
fn small_clusters_are_dropped_into_their_neighbours() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, _) = split_corpus(23, 4_000, 0, 32, 16);
    let mut f = Field::new(Similarity::L2, 32, opts(128, 4));
    f.add(rows);
    let active = f.state.as_ref().unwrap().centroids.active();
    // Kill almost every vector of the first centroid.
    let first: Vec<_> = f.state.as_ref().unwrap().list_entries(0).collect();
    for e in first.iter().skip(2) {
        f.dead.insert(e & !SPILL);
    }
    f.maintain(false);
    let state = f.state.as_ref().unwrap();
    assert!(state.centroids.active() < active + 8, "{active} -> {}", state.centroids.active());
    assert!(state.generation(0).is_none(), "the emptied centroid is dropped");
    for e in first.iter().take(2) {
        let p = e & !SPILL;
        assert!((0..state.centroids.len()).any(|c| state.list_entries(c).any(|x| x & !SPILL == p)), "survivor {p} moved");
    }
}

#[test]
fn persistence_round_trip_and_unknown_versions() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(6, 2500, 5, 40, 10);
    for (sim, precondition, bits) in [(Similarity::Cosine, false, 1), (Similarity::DotProduct, true, 2), (Similarity::L2, true, 4), (Similarity::MaxInnerProduct, false, 7)] {
        let mut f = Field::new(sim, 40, IvfOptions { precondition, ..opts(128, bits) });
        f.add(rows[..1500].to_vec());
        f.add(rows[1500..].to_vec());
        let state = f.state.as_ref().unwrap();
        let m = state.manifest();
        assert_eq!(m.files().len(), 3, "centroids and two batches");
        let (c, b, indexed, next, _open_memory) = open(&f.dir, &m).unwrap().unwrap();
        assert_eq!(next, state.next_file());
        let reopened = IvfState::assemble(3, c, b, indexed, HashSet::new(), next).unwrap();
        assert_eq!(reopened.indexed.count(), 2500);
        for q in &queries {
            let (before, before_stats) = state.search(q, 10, 15, 0.0, 60, None, None, false, |_| true).unwrap();
            let (after, after_stats) = reopened.search(q, 10, 15, 0.0, 60, None, None, false, |_| true).unwrap();
            assert_eq!(before.as_slice(), after.as_slice());
            assert_eq!(before_stats, after_stats);
        }
        // A later centroid file version is ignored, not misread; a damaged
        // one is refused.
        let name = centroid_file_name(3, m.centroids);
        let mut bytes = f.dir.open_read(&name).unwrap().as_slice().to_vec();
        bytes[4..8].copy_from_slice(&(IVF_FILES_VERSION + 1).to_le_bytes());
        f.dir.write(&name, &bytes).unwrap();
        assert!(open(&f.dir, &m).unwrap().is_none());
        bytes[4..8].copy_from_slice(&IVF_FILES_VERSION.to_le_bytes());
        bytes[40] ^= 1;
        f.dir.write(&name, &bytes).unwrap();
        assert!(open(&f.dir, &m).is_err());
    }
}

/// More visit ratio visits at least as many entries and does not lose
/// recall (4-bit codes, sharp against far centroids too).
#[test]
fn visit_ratio_is_monotone() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(15, 20_000, 40, 128, 64);
    let mut f = Field::new(Similarity::L2, 128, IvfOptions { bits: 4, ..IvfOptions::default() });
    f.add(rows);
    let mut last = (0.0, 0);
    for ratio in [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0] {
        let (r, visited) = f.recall(&queries, 10, 100, ratio);
        eprintln!("IVF_RECALL visit={ratio} recall@10={r:.3} entries_visited={visited}");
        assert!(visited >= last.1, "visit {ratio}: fewer entries visited");
        assert!(r + 0.02 >= last.0, "visit {ratio}: recall {r} below {}", last.0);
        last = (r, visited);
    }
    assert!(last.0 >= 0.95, "recall@10 visiting everything {}", last.0);
    assert_eq!(last.1, f.state.as_ref().unwrap().entries() * queries.len() as u64, "visit 100% visits every list");
}

#[test]
fn filtered_search_returns_only_accepted_vectors() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let (rows, queries) = split_corpus(13, 8_000, 20, 48, 32);
    let mut f = Field::new(Similarity::L2, 48, opts(128, 1));
    f.add(rows);
    for (modulus, min_recall) in [(2u64, 0.85), (50, 0.8)] {
        let mut total = 0.0;
        for q in &queries {
            let state = f.state.as_ref().unwrap();
            let (hits, st) = state.search(q, 10, 100, 0.0, leaf_budget(10, 3.0), None, Some((8_000 / modulus) as usize), false, |p| p % modulus == 0).unwrap();
            assert!(hits.iter().all(|(p, _)| p % modulus == 0));
            assert!(st.entries_scored <= st.entries_visited);
            let (got, _) = f.search(q, 10, 100, 0.0, 3.0, |p| p % modulus == 0);
            let want: HashSet<u64> = f.exact(q, 10, |p| p % modulus == 0).into_iter().collect();
            total += got.iter().filter(|g| want.contains(g)).count() as f64 / want.len().max(1) as f64;
        }
        let r = total / queries.len() as f64;
        assert!(r >= min_recall, "1/{modulus} filter: recall {r}");
    }
}

#[test]
fn corrupt_posting_bytes_and_header_ranges_are_rejected() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let params = BuildParams::of(Similarity::L2, 8, &opts(64, 1));
    let mut run = RunOut::new(0, 1);
    run.push_raw(7, &vec![0; ivf::code_bytes(1, 8)], &[0; CORRECTION_BYTES]);
    let bytes = encode_batch(3, &params, &[run]);
    let dir = RamDirectory::new();
    let check = |bytes: &[u8]| {
        dir.write("corrupt.rivp", bytes).unwrap();
        PostingBatch::open(3, 1, dir.open_read("corrupt.rivp").unwrap(), &params).is_err()
    };
    let mut corrupt = bytes.clone();
    corrupt[56] ^= 1;
    assert!(check(&corrupt), "payload checksum must be verified");
    let mut corrupt = bytes[..bytes.len() - 8].to_vec();
    corrupt[BATCH_HEADER + 8..BATCH_HEADER + 16].copy_from_slice(&0u64.to_le_bytes());
    assert!(check(&seal(corrupt, BATCH_MAGIC)), "runs cannot overlap the header");
}

#[test]
fn overflowing_centroid_layout_is_rejected() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let params = BuildParams::of(Similarity::L2, 8, &opts(64, 1));
    let c = Centroids::new(1, params, None, vec![1], vec![0.0; 8]).unwrap();
    let bytes = c.encode(3);
    let mut body = bytes[..bytes.len() - 8].to_vec();
    body[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    body[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Centroids::decode(3, 1, &seal(body, CENTROIDS_MAGIC)).is_err());
}

#[test]
fn sparse_indexed_points_do_not_allocate_to_the_largest_id() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let high = (1u64 << 62) + 7;
    let indexed = indexed_of([3, high]).unwrap();
    assert!(indexed.contains(3));
    assert!(indexed.contains(high));
    assert!(!indexed.contains(high - 1));
    assert_eq!(indexed.count(), 2);
    assert_eq!(indexed.words.len(), 2);
    // Admission includes the BTree node/split allowance, not just payload
    // words. Equal populated word counts must cost the same regardless of
    // the largest point ID; another point in one word costs no extra node.
    let near = indexed_of([3, 71]).unwrap();
    let same_words = indexed_of([3, 4, high]).unwrap();
    assert_eq!(indexed._memory.bytes(), near._memory.bytes());
    assert_eq!(indexed._memory.bytes(), same_words._memory.bytes());
    assert_eq!(same_words.count(), 3);
    assert_eq!(same_words.words.len(), 2);
}

#[test]
fn duplicate_spills_do_not_consume_distinct_candidate_slots() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let mut collector = PointCollector::default();
    for (point, score) in [(1, 10.0), (1, 9.0), (2, 8.0), (2, 7.0), (3, 6.0), (3, 5.0)] {
        collector.offer(point, worse_key(score), 3);
    }
    assert_eq!(collector.len(), 3);
    collector.offer(3, worse_key(11.0), 3);
    collector.offer(4, worse_key(9.5), 3);
    let points: Vec<_> = collector.order.iter().map(|(_, point)| *point).collect();
    assert_eq!(points, vec![3, 1, 4]);
}

#[test]
fn all_lists_keep_distinct_candidates_with_repeated_posting_runs() {
    let _fixture = crate::vector::ivf::fixture_admission_lock();
    let mut field = Field::new(Similarity::L2, 8, IvfOptions::default());
    field.add(test_corpus(37, 64, 8, 4));
    let original = field.state.take().unwrap();
    // Simulate a duplicate-heavy set of primary/spill occurrences. The point
    // identity remains the same regardless of which posting contains it.
    let batches = vec![original.batches[0].clone(), original.batches[0].clone()];
    let state = IvfState::assemble(3, original.centroids.clone(), batches, original.indexed.clone(), HashSet::new(), original.next_file()).unwrap();
    let (hits, _) = state.search(&field.rows[0], 16, 64, 1.0, 16, Some(64), None, false, |_| true).unwrap();
    assert_eq!(hits.len(), 16);
    assert_eq!(hits.iter().map(|(point, _)| *point).collect::<HashSet<_>>().len(), 16);
}

/// The maintenance workspace is sized from what a step allocates (row
/// references, clustering arrays, entries, codes twice): 1M x 768 rows
/// train in well under half of a 16 GB budget at every code width, and the
/// estimate still covers the posting bytes a step actually encodes.
#[test]
fn maintenance_workspace_is_sized_from_the_codes() {
    for (bits, query_bits) in [(1, 4), (2, 4), (4, 4), (7, 7)] {
        let params = BuildParams { dim: 768, sim: OsqSim::Euclidean, unit: false, bits, query_bits, precondition: false };
        let clusters = cluster_bound(1_000_000, ivf::DEFAULT_VECTORS_PER_CLUSTER as usize);
        let bytes = workspace_bytes(1_000_000, &params, clusters, clusters, 0, 64);
        eprintln!("IVF_WORKSPACE bits={bits} 1Mx768={:.2} GiB", bytes as f64 / f64::from(1u32 << 30));
        assert!(bytes < 8 << 30, "{bits}-bit: {bytes} bytes for 1M x 768 rows");
        // And within a quarter of a 16 GB budget, one step takes them all.
        assert!(max_step_rows(&params, 16 << 30) >= 1_000_000, "{bits}-bit: {}", max_step_rows(&params, 16 << 30));
    }
    // Lower bound: two copies (runs, then the file) of what one step encodes.
    let (rows, _) = split_corpus(5, 3000, 0, 64, 16);
    for bits in [1u8, 4, 7] {
        let mut f = Field::new(Similarity::L2, 64, opts(64, bits));
        f.add(rows.clone());
        let state = f.state.as_ref().unwrap();
        let encoded = state.posting_bytes() as u64;
        assert!(step_row_bytes(64, bits) * 3000 >= 2 * encoded, "{bits}-bit: {} per row vs {} encoded", step_row_bytes(64, bits), encoded / 3000);
    }
}

/// The indexed set is admitted at about its real size (a byte a point for
/// dense ids), not 512 bytes a 64-point word.
#[test]
fn indexed_points_are_admitted_at_their_size() {
    let indexed = indexed_of(0..1_000_000u64).unwrap();
    assert_eq!(indexed.count(), 1_000_000);
    let words = 1_000_000u64.div_ceil(64);
    assert!(indexed.reserved_bytes() <= words * INDEXED_WORD_BYTES, "{} bytes for {words} words", indexed.reserved_bytes());
    let grown = indexed.with(1_000_000..1_000_640u64).unwrap();
    assert!(grown.reserved_bytes() <= (words + 10) * INDEXED_WORD_BYTES);
}
