//! Weighted feature terms (`sparse_vector` / `rank_features`) against a
//! brute-force oracle: the memtable, sealed segments, deletes and merges,
//! with the MaxScore top-k path (bounded total) and exhaustive counting.

use rustic_core::fieldtypes::features::{encode, quantize};
use rustic_core::query::feature::FeatureQuery;
use rustic_core::query::{Query, SearchOptions, Searcher, TrackTotalHits};
use rustic_core::{DocAddress, Document, Field, FieldType, Index, IndexSettings, Term};
use std::collections::BTreeMap;
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn weight(&mut self) -> f32 {
        // Wide range, as learned sparse encoders produce (0.01 .. ~40).
        (0.01 + (self.next() % 4000) as f32 / 100.0) * if self.below(10) == 0 { 0.05 } else { 1.0 }
    }
}

const ID: u16 = 0;
const FEAT: u16 = 1;
const VOCAB: u64 = 40;

fn schema() -> Vec<Field> {
    vec![Field::new(ID, "id", FieldType::Keyword), Field::new(FEAT, "ml.tokens", FieldType::Keyword).with_features(true)]
}

fn make(r: &mut Rng) -> BTreeMap<String, f32> {
    // Skewed vocabulary: low ids are common (long postings, many blocks).
    let n = r.below(14);
    (0..n).map(|_| (format!("t{}", r.below(VOCAB).min(r.below(VOCAB))), quantize(r.weight()))).collect()
}

fn to_doc(id: &str, feats: &BTreeMap<String, f32>) -> Document {
    let mut d = Document::new();
    d.add_keyword(ID, id);
    for (t, w) in feats {
        d.add_keyword(FEAT, &encode(t, *w));
    }
    d
}

fn oracle(live: &BTreeMap<String, BTreeMap<String, f32>>, q: &[(String, f32)]) -> BTreeMap<String, f32> {
    live.iter()
        .filter_map(|(id, f)| {
            let mut s = 0f32;
            let mut hit = false;
            for (t, w) in q {
                if let Some(dw) = f.get(t) {
                    s += w * dw;
                    hit = true;
                }
            }
            hit.then(|| (id.clone(), s))
        })
        .collect()
}

fn check(index: &Index, live: &BTreeMap<String, BTreeMap<String, f32>>, r: &mut Rng, what: &str) {
    let leaves = index.leaves();
    let searcher = Searcher::new(leaves.clone());
    for round in 0..40 {
        let n = 1 + r.below(if round % 4 == 0 { 30 } else { 8 }) as usize;
        let mut q: Vec<(String, f32)> = (0..n).map(|_| (format!("t{}", r.below(VOCAB + 3)), 0.1 + (r.below(300) as f32) / 100.0)).collect();
        q.sort_by(|a, b| a.0.cmp(&b.0));
        q.dedup_by(|a, b| a.0 == b.0);
        let want = oracle(live, &q);
        let query = Query::Custom(Arc::new(FeatureQuery::new(FEAT, q.clone(), "sparse_vector")));
        let k = 1 + r.below(20) as usize;
        let mut best: Vec<f32> = want.values().copied().collect();
        best.sort_by(|a, b| b.total_cmp(a));
        best.truncate(k);
        for track in [TrackTotalHits::UpTo(10), TrackTotalHits::True] {
            let td = searcher.search(&query, &SearchOptions { top_k: k, track_total_hits: track, ..Default::default() });
            let got: Vec<(String, f32)> = td
                .hits
                .iter()
                .map(|h| (leaves[h.leaf].get_doc(h.doc).unwrap().unwrap().get(ID).unwrap().as_text().unwrap().to_string(), h.score))
                .collect();
            assert_eq!(got.len(), best.len(), "{what} round {round} {track:?}: hit count");
            for ((id, s), b) in got.iter().zip(&best) {
                let exact = want[id];
                assert!((s - exact).abs() <= 1e-4 * exact.max(1.0), "{what} round {round}: {id} scored {s}, oracle {exact}");
                assert!((s - b).abs() <= 1e-4 * b.max(1.0), "{what} round {round} {track:?}: rank score {s} vs best {b}");
            }
            if track == TrackTotalHits::True {
                assert_eq!(td.total_hits, want.len() as u64, "{what} round {round}: total");
            } else {
                assert!(td.total_hits >= (want.len() as u64).min(10), "{what} round {round}: bounded total");
            }
            if let Some(h) = td.hits.first() {
                let e = searcher.explain(&query, &SearchOptions::default(), h.leaf, h.doc);
                assert!((e.value - h.score).abs() <= 1e-4 * h.score.max(1.0), "{what}: explain {} vs {}", e.value, h.score);
            }
        }
    }
}

fn find(index: &Index, id: &str) -> Option<DocAddress> {
    for (i, l) in index.leaves().iter().enumerate() {
        let live = l.live_docs();
        if let Some(mut p) = l.postings(&Term::new_text(ID, id)) {
            while let Some(d) = p.next() {
                if live(d) {
                    return Some(DocAddress::new(i as u32, d));
                }
            }
        }
    }
    None
}

#[test]
fn feature_terms_match_brute_force_across_segments_deletes_and_merges() {
    let mut r = Rng(7);
    let mut index = Index::new("f", schema(), IndexSettings::default());
    let mut live: BTreeMap<String, BTreeMap<String, f32>> = BTreeMap::new();
    let mut next = 0;
    for round in 0..4 {
        for _ in 0..1500 {
            next += 1;
            let (id, f) = (format!("d{next}"), make(&mut r));
            index.add_document(&to_doc(&id, &f)).unwrap();
            live.insert(id, f);
        }
        check(&index, &live, &mut r, &format!("memtable {round}"));
        index.refresh().unwrap();
        check(&index, &live, &mut r, &format!("segments {round}"));
        // Deletes and updates.
        let ids: Vec<String> = live.keys().cloned().filter(|_| r.below(8) == 0).collect();
        for id in &ids {
            let a = find(&index, id).expect("live doc is findable");
            index.delete_doc(&a).unwrap();
            live.remove(id);
            if r.below(2) == 0 {
                let f = make(&mut r);
                index.add_document(&to_doc(id, &f)).unwrap();
                live.insert(id.clone(), f);
            }
        }
        index.refresh().unwrap();
        check(&index, &live, &mut r, &format!("deletes {round}"));
    }
    index.force_merge().unwrap();
    check(&index, &live, &mut r, "merged");
}

#[test]
fn duplicate_query_tokens_keep_the_largest_weight() {
    let mut index = Index::new("f", schema(), IndexSettings::default());
    let mut f = BTreeMap::new();
    f.insert("a".to_string(), quantize(2.0));
    index.add_document(&to_doc("x", &f)).unwrap();
    index.refresh().unwrap();
    let q = FeatureQuery::new(FEAT, vec![("a".into(), 1.0), ("a".into(), 3.0)], "sparse_vector");
    let td = Searcher::new(index.leaves()).search(&Query::Custom(Arc::new(q)), &SearchOptions { top_k: 1, ..Default::default() });
    assert_eq!(td.hits.len(), 1);
    assert!((td.hits[0].score - 6.0).abs() < 1e-3);
}

#[test]
fn a_token_repeated_in_one_doc_scores_its_largest_weight() {
    let mut index = Index::new("f", schema(), IndexSettings::default());
    let mut d = Document::new();
    d.add_keyword(ID, "x");
    for w in [0.1f32, 1.1652642, 0.5] {
        d.add_keyword(FEAT, &encode("exercise", quantize(w)));
    }
    index.add_document(&d).unwrap();
    for sealed in [false, true] {
        if sealed {
            index.refresh().unwrap();
        }
        let q = FeatureQuery::new(FEAT, vec![("exercise".into(), 2.0)], "sparse_vector");
        let td = Searcher::new(index.leaves()).search(&Query::Custom(Arc::new(q)), &SearchOptions { top_k: 1, ..Default::default() });
        assert!((td.hits[0].score - 2.0 * quantize(1.1652642)).abs() < 1e-4, "sealed {sealed}: {}", td.hits[0].score);
    }
}
