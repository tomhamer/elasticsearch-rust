//! String distances of Lucene's spell checker (`org.apache.lucene.search.spell`)
//! and the term statistics view the suggesters read candidates from.

use rustic_core::query::LeafReader;
use rustic_core::types::Term;
use std::collections::BTreeMap;

/// `string_distance` of the term / direct-generator suggesters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distance {
    /// DirectSpellChecker.INTERNAL_LEVENSHTEIN: the fuzzy enum's boost.
    Internal,
    DamerauLevenshtein,
    Levenshtein,
    JaroWinkler,
    NGram,
}

impl Distance {
    pub fn parse(s: &str) -> Option<Distance> {
        Some(match s {
            "internal" => Distance::Internal,
            "damerau_levenshtein" | "damerauLevenshtein" => Distance::DamerauLevenshtein,
            "levenshtein" | "levenstein" => Distance::Levenshtein,
            "jaro_winkler" | "jaroWinkler" => Distance::JaroWinkler,
            "ngram" => Distance::NGram,
            _ => return None,
        })
    }

    /// Similarity in [0, 1] (not used for `Internal`).
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let x: Vec<char> = a.chars().collect();
        let y: Vec<char> = b.chars().collect();
        match self {
            // ES `LuceneLevenshteinDistance` / the fuzzy enum: normalised by the shorter term.
            Distance::Internal | Distance::DamerauLevenshtein => {
                let m = x.len().min(y.len());
                if m == 0 {
                    return if x.len() == y.len() { 1.0 } else { 0.0 };
                }
                1.0 - osa(&x, &y) as f32 / m as f32
            }
            Distance::Levenshtein => {
                let m = x.len().max(y.len());
                if m == 0 {
                    return 1.0;
                }
                1.0 - levenshtein(&x, &y) as f32 / m as f32
            }
            Distance::JaroWinkler => jaro_winkler(&x, &y),
            Distance::NGram => ngram(&x, &y, 2),
        }
    }
}

/// Optimal-string-alignment distance (Damerau with adjacent transpositions).
pub fn osa(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = v;
        }
    }
    d[n][m]
}

pub fn levenshtein(a: &[char], b: &[char]) -> usize {
    let m = b.len();
    let mut prev: Vec<usize> = (0..=m).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; m + 1];
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[m]
}

/// Lucene `JaroWinklerDistance` (threshold 0.7).
fn jaro_winkler(s1: &[char], s2: &[char]) -> f32 {
    let (max, min) = if s1.len() > s2.len() { (s1, s2) } else { (s2, s1) };
    let range = (max.len() / 2).saturating_sub(1);
    let mut match_idx = vec![-1i64; min.len()];
    let mut match_flags = vec![false; max.len()];
    let mut matches = 0usize;
    for (mi, c1) in min.iter().enumerate() {
        let lo = mi.saturating_sub(range);
        let hi = (mi + range + 1).min(max.len());
        for xi in lo..hi {
            if !match_flags[xi] && *c1 == max[xi] {
                match_idx[mi] = xi as i64;
                match_flags[xi] = true;
                matches += 1;
                break;
            }
        }
    }
    let mut ms1 = Vec::with_capacity(matches);
    let mut ms2 = Vec::with_capacity(matches);
    for (i, c) in min.iter().enumerate() {
        if match_idx[i] != -1 {
            ms1.push(*c);
        }
    }
    for (i, c) in max.iter().enumerate() {
        if match_flags[i] {
            ms2.push(*c);
        }
    }
    let transpositions = ms1.iter().zip(&ms2).filter(|(a, b)| a != b).count();
    let mut prefix = 0usize;
    for mi in 0..min.len() {
        if s1[mi] == s2[mi] {
            prefix += 1;
        } else {
            break;
        }
    }
    let m = matches as f32;
    if m == 0.0 {
        return 0.0;
    }
    let t = (transpositions / 2) as f32;
    let j = (m / s1.len() as f32 + m / s2.len() as f32 + (m - t) / m) / 3.0;
    if j < 0.7 {
        j
    } else {
        j + (0.1f32).min(1.0 / max.len() as f32) * prefix as f32 * (1.0 - j)
    }
}

/// Lucene `NGramDistance`.
fn ngram(source: &[char], target: &[char], n: usize) -> f32 {
    let (sl, tl) = (source.len(), target.len());
    if sl == 0 || tl == 0 {
        return if sl == tl { 1.0 } else { 0.0 };
    }
    if sl < n || tl < n {
        let cost = (0..sl.min(tl)).filter(|&i| source[i] == target[i]).count();
        return cost as f32 / sl.max(tl) as f32;
    }
    let nul = '\0';
    let mut sa = vec![nul; sl + n - 1];
    for (i, c) in sa.iter_mut().enumerate() {
        if i >= n - 1 {
            *c = source[i + 1 - n];
        }
    }
    let mut p: Vec<f32> = (0..=sl).map(|i| i as f32).collect();
    let mut d = vec![0f32; sl + 1];
    let mut t_j = vec![nul; n];
    for j in 1..=tl {
        if j < n {
            for x in t_j.iter_mut().take(n - j) {
                *x = nul;
            }
            for ti in (n - j)..n {
                t_j[ti] = target[ti - (n - j)];
            }
        } else {
            t_j.copy_from_slice(&target[j - n..j]);
        }
        d[0] = j as f32;
        for i in 1..=sl {
            let mut cost = 0usize;
            let mut tn = n;
            for ni in 0..n {
                if sa[i - 1 + ni] != t_j[ni] {
                    cost += 1;
                } else if sa[i - 1 + ni] == nul {
                    tn -= 1;
                }
            }
            let ec = cost as f32 / tn.max(1) as f32;
            d[i] = (d[i - 1] + 1.0).min(p[i] + 1.0).min(p[i - 1] + ec);
        }
        std::mem::swap(&mut p, &mut d);
    }
    1.0 - p[sl] / tl.max(sl) as f32
}

/// The terms dictionary of one field over a set of leaves (Lucene's
/// `IndexReader` view: deleted docs still count in the statistics).
pub struct Dict<'a, 'b> {
    pub leaves: &'b [LeafReader<'a>],
    pub field: u16,
}

impl Dict<'_, '_> {
    /// (doc_freq, total_term_freq) of `t`.
    pub fn stats(&self, t: &str) -> (u64, u64) {
        let term = Term::new(self.field, t.as_bytes().to_vec());
        let mut out = (0u64, 0u64);
        for l in self.leaves {
            if let Some(i) = l.term_info(&term) {
                out.0 += i.doc_freq as u64;
                out.1 += i.total_term_freq;
            }
        }
        out
    }

    pub fn max_doc(&self) -> u64 {
        self.leaves.iter().map(|l| l.num_docs() as u64).sum()
    }

    pub fn num_docs(&self) -> u64 {
        self.leaves.iter().map(|l| l.num_live_docs() as u64).sum()
    }

    /// Terms starting with `prefix` → (doc_freq, total_term_freq).
    pub fn prefixed(&self, prefix: &str) -> BTreeMap<String, (u64, u64)> {
        let lo = prefix.as_bytes().to_vec();
        let hi = upper_bound(&lo).unwrap_or_else(|| vec![0xff]);
        let fp = self.field.to_le_bytes();
        let mut out: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for l in self.leaves {
            let lower = if lo.is_empty() { None } else { Some(lo.as_slice()) };
            for (t, info) in l.term_range(self.field, lower, Some(hi.as_slice())) {
                // Keys carry the 2-byte field prefix.
                if t.len() < 2 || t[..2] != fp || !t[2..].starts_with(&lo) {
                    continue;
                }
                let Ok(s) = String::from_utf8(t[2..].to_vec()) else { continue };
                let e = out.entry(s).or_insert((0, 0));
                e.0 += info.doc_freq as u64;
                e.1 += info.total_term_freq;
            }
        }
        out
    }

    /// (sum of total term freqs, sum of doc freqs, unique terms or -1 when
    /// the reader has several leaves, like Lucene's `MultiTerms.size()`).
    pub fn field_totals(&self) -> (u64, u64, i64) {
        let mut ttf = 0u64;
        let mut df = 0u64;
        let mut uniq = 0i64;
        for l in self.leaves {
            let fp = self.field.to_le_bytes();
            for (t, info) in l.term_range(self.field, None, Some(&[0xff])) {
                if t.len() < 2 || t[..2] != fp {
                    continue;
                }
                ttf += info.total_term_freq;
                df += info.doc_freq as u64;
                uniq += 1;
            }
        }
        let n = self.leaves.iter().filter(|l| l.num_docs() > 0).count();
        (ttf, df, if n <= 1 { uniq } else { -1 })
    }
}

fn upper_bound(p: &[u8]) -> Option<Vec<u8>> {
    let mut v = p.to_vec();
    while let Some(last) = v.pop() {
        if last < 0xff {
            v.push(last + 1);
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances() {
        let c = |s: &str| s.chars().collect::<Vec<_>>();
        assert_eq!(osa(&c("amsterdma"), &c("amsterdam")), 1);
        assert_eq!(levenshtein(&c("amsterdma"), &c("amsterdam")), 2);
        assert!((Distance::Levenshtein.similarity("abcd", "abce") - 0.75).abs() < 1e-6);
        assert!(Distance::JaroWinkler.similarity("martha", "marhta") > 0.95);
        assert!(Distance::NGram.similarity("hello", "hello") > 0.99);
    }
}
