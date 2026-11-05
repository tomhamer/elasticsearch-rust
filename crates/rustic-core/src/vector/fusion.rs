//! Result-list fusion for hybrid (text + vector) search, Qdrant / ES style.
//!
//! Inputs are ranked lists of `(key, score)` (best first); the key is any
//! orderable id, e.g. `(leaf, doc)`. Outputs are fused lists sorted by
//! fused score descending, ties by key ascending (deterministic).

use std::collections::HashMap;
use std::hash::Hash;

/// Reciprocal rank fusion (Cormack et al.; ES `rrf` retriever):
/// `score(d) = Σ_lists 1 / (rank_constant + rank(d))`, ranks 1-based, only
/// the first `rank_window` entries of each list participate.
/// `rank_constant` is ES's `rank_constant` (default 60).
pub fn rrf<K: Hash + Ord + Clone>(lists: &[Vec<(K, f32)>], rank_constant: f32, rank_window: usize) -> Vec<(K, f32)> {
    let mut acc: HashMap<K, f32> = HashMap::new();
    for list in lists {
        for (rank, (key, _)) in list.iter().take(rank_window).enumerate() {
            *acc.entry(key.clone()).or_insert(0.0) += 1.0 / (rank_constant + rank as f32 + 1.0);
        }
    }
    sorted(acc)
}

/// Per-list score normalization for [`linear`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// Raw scores.
    None,
    /// `(s - min) / (max - min)`; all-equal lists map to 1.0.
    MinMax,
    /// `s / ||scores||₂`.
    L2,
}

/// Weighted linear combination (ES `linear` retriever, Qdrant score
/// fusion): `score(d) = Σ_i weights[i] * norm_i(score_i(d))`; a doc absent
/// from a list contributes 0 for it. Missing weights default to 1.
pub fn linear<K: Hash + Ord + Clone>(lists: &[Vec<(K, f32)>], weights: &[f32], normalization: Normalization) -> Vec<(K, f32)> {
    let mut acc: HashMap<K, f32> = HashMap::new();
    for (i, list) in lists.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        let norm = normalizer(list, normalization);
        for (key, s) in list {
            // ES: a list without scores (NaN) contributes 0.
            let v = norm(*s);
            *acc.entry(key.clone()).or_insert(0.0) += w * if v.is_nan() { 0.0 } else { v };
        }
    }
    sorted(acc)
}

fn normalizer<K>(list: &[(K, f32)], n: Normalization) -> Box<dyn Fn(f32) -> f32> {
    match n {
        Normalization::None => Box::new(|s| s),
        Normalization::MinMax => {
            // ES `MinMaxScoreNormalizer`: `max` starts at Java's
            // `Float.MIN_VALUE` (the smallest positive float); scores
            // within 1e-6 of each other all normalize to 1.
            let (mut min, mut max) = (f32::MAX, f32::from_bits(1));
            for x in list {
                if x.1 > max {
                    max = x.1;
                }
                if x.1 < min {
                    min = x.1;
                }
            }
            if list.iter().all(|x| x.1.is_nan()) {
                return Box::new(|s| s);
            }
            if (min - max).abs() < 1e-6 { Box::new(|_| 1.0) } else { Box::new(move |s| (s - min) / (max - min)) }
        }
        Normalization::L2 => {
            // ES `L2ScoreNormalizer`: a double sum of float squares, NaNs skipped.
            let sum: f64 = list.iter().filter(|x| !x.1.is_nan()).map(|x| (x.1 * x.1) as f64).sum();
            let norm = sum.sqrt();
            if list.iter().all(|x| x.1.is_nan()) || norm < 1e-6 { Box::new(|s| s) } else { Box::new(move |s| (s as f64 / norm) as f32) }
        }
    }
}

fn sorted<K: Ord>(acc: HashMap<K, f32>) -> Vec<(K, f32)> {
    let mut v: Vec<(K, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_matches_hand_computation() {
        let text = vec![("a", 9.0), ("b", 5.0), ("c", 1.0)];
        let knn = vec![("c", 0.99), ("a", 0.5), ("d", 0.4)];
        let fused = rrf(&[text.clone(), knn.clone()], 60.0, 10);
        let s = |r1: Option<f32>, r2: Option<f32>| r1.map_or(0.0, |r| 1.0 / (60.0 + r)) + r2.map_or(0.0, |r| 1.0 / (60.0 + r));
        let want = vec![
            ("a", s(Some(1.0), Some(2.0))),
            ("c", s(Some(3.0), Some(1.0))),
            ("b", s(Some(2.0), None)),
            ("d", s(None, Some(3.0))),
        ];
        assert_eq!(fused.len(), 4);
        for ((k, v), (wk, wv)) in fused.iter().zip(&want) {
            assert_eq!(k, wk);
            assert!((v - wv).abs() < 1e-7);
        }
        // Rank window: only the first entry of each list counts.
        let w1 = rrf(&[text, knn], 60.0, 1);
        assert_eq!(w1.iter().map(|x| x.0).collect::<Vec<_>>(), vec!["a", "c"]);
        // Raw scores are irrelevant to RRF.
        let scaled = rrf(&[vec![("x", 1000.0), ("y", 0.001)]], 60.0, 10);
        assert!(scaled[0].0 == "x" && (scaled[0].1 - 1.0 / 61.0).abs() < 1e-7);
    }

    #[test]
    fn linear_with_normalizations() {
        let text = vec![(1u32, 10.0), (2, 6.0), (3, 2.0)];
        let knn = vec![(3u32, 0.9), (4, 0.7), (1, 0.5)];
        // MinMax: text -> 1, .5, 0; knn -> 1, .5, 0.
        let f = linear(&[text.clone(), knn.clone()], &[1.0, 2.0], Normalization::MinMax);
        let get = |k: u32| f.iter().find(|x| x.0 == k).unwrap().1;
        assert!((get(1) - (1.0 + 0.0)).abs() < 1e-6);
        assert!((get(2) - 0.5).abs() < 1e-6);
        assert!((get(3) - (0.0 + 2.0)).abs() < 1e-6);
        assert!((get(4) - 1.0).abs() < 1e-6);
        assert_eq!(f[0].0, 3);
        // None: plain weighted sum.
        let f = linear(&[text.clone(), knn.clone()], &[1.0, 1.0], Normalization::None);
        assert!((f.iter().find(|x| x.0 == 1).unwrap().1 - 10.5).abs() < 1e-6);
        // L2: each list scaled to unit norm.
        let f = linear(&[text.clone()], &[1.0], Normalization::L2);
        let norm = (100.0f32 + 36.0 + 4.0).sqrt();
        assert!((f[0].1 - 10.0 / norm).abs() < 1e-6);
        // Degenerate list: all-equal MinMax maps to 1.
        let f = linear(&[vec![(7u32, 3.0), (8, 3.0)]], &[0.5], Normalization::MinMax);
        assert_eq!(f, vec![(7, 0.5), (8, 0.5)]);
    }
}
