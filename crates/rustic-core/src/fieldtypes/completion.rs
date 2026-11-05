//! Completion (prefix suggestion) index over an FST (Lucene
//! `CompletionPostingsFormat` / `NRTSuggester`): analyzed inputs map to
//! entries carrying the doc, weight and contexts. Built per segment from
//! the completion field's doc values.

use crate::types::DocId;
use fst::{IntoStreamer, Map, MapBuilder, Streamer};

/// Separator between analyzed input tokens when separators are preserved.
pub const TOKEN_SEP: char = '\u{1f}';

/// One suggestion input of a doc.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletionEntry {
    pub doc: DocId,
    /// The input as given (what the suggester returns as `text`).
    pub input: String,
    pub weight: u64,
    /// `(context name, value)` pairs.
    pub contexts: Vec<(String, String)>,
}

/// The `simple`-analyzer form of an input: lower-cased letter runs, joined
/// by [`TOKEN_SEP`] (or concatenated when separators are not preserved),
/// truncated to `max_input_length` chars.
pub fn analyze(input: &str, preserve_separators: bool, max_input_length: usize) -> String {
    let truncated: String = input.chars().take(max_input_length).collect();
    let toks: Vec<String> = truncated.split(|c: char| !c.is_alphabetic()).filter(|t| !t.is_empty()).map(|t| t.to_lowercase()).collect();
    if preserve_separators {
        toks.join(&TOKEN_SEP.to_string())
    } else {
        toks.concat()
    }
}

/// An FST from `analyzed input ‖ 0x00 ‖ entry index` to the entry index.
pub struct CompletionIndex {
    fst: Map<Vec<u8>>,
    entries: Vec<CompletionEntry>,
    analyzed: Vec<String>,
}

/// One suggestion: the best matching input of a doc.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub doc: DocId,
    pub input: String,
    pub weight: u64,
    pub contexts: Vec<(String, String)>,
}

impl CompletionIndex {
    /// Build from entries (`analyzer` maps an input to its indexed form).
    pub fn build(entries: Vec<CompletionEntry>, analyzer: impl Fn(&str) -> String) -> Self {
        let analyzed: Vec<String> = entries.iter().map(|e| analyzer(&e.input)).collect();
        let mut keys: Vec<(Vec<u8>, u64)> = analyzed
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let mut k = a.as_bytes().to_vec();
                k.push(0);
                k.extend_from_slice(&(i as u32).to_be_bytes());
                (k, i as u64)
            })
            .collect();
        keys.sort();
        let mut b = MapBuilder::memory();
        for (k, v) in &keys {
            // Keys are unique (entry index suffix) and sorted.
            let _ = b.insert(k, *v);
        }
        let fst = Map::new(b.into_inner().unwrap_or_default()).unwrap_or_else(|_| Map::default());
        Self { fst, entries, analyzed }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries whose analyzed input starts with `prefix` (exact) or is
    /// within `fuzziness` edits of it on some prefix (`fuzzy_prefix_length`
    /// leading chars must match exactly).
    pub fn lookup(&self, prefix: &str, fuzzy: Option<(u8, usize, bool)>) -> Vec<(usize, f32)> {
        let mut out = Vec::new();
        match fuzzy {
            None => {
                let mut stream = self.fst.range().ge(prefix.as_bytes()).into_stream();
                while let Some((k, v)) = stream.next() {
                    if !k.starts_with(prefix.as_bytes()) {
                        break;
                    }
                    // Only the input part (before 0x00) must start with the prefix.
                    let input_len = k.len() - 5;
                    if input_len >= prefix.len() {
                        out.push((v as usize, 1.0));
                    }
                }
            }
            Some((edits, prefix_len, transpositions)) => {
                let q: Vec<char> = prefix.chars().collect();
                let mut stream = self.fst.stream();
                while let Some((_, v)) = stream.next() {
                    let a: Vec<char> = self.analyzed[v as usize].chars().collect();
                    if a.len() < prefix_len.min(q.len()) || a[..prefix_len.min(q.len())] != q[..prefix_len.min(q.len())] {
                        continue;
                    }
                    if let Some(d) = prefix_edit_distance(&q, &a, transpositions) {
                        if d <= edits as usize {
                            // Lucene's FuzzyCompletionQuery does not rescore: the score is the weight.
                            out.push((v as usize, 1.0));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn entry(&self, i: usize) -> &CompletionEntry {
        &self.entries[i]
    }
}

/// Minimum edit distance between `q` and any prefix of `a`.
fn prefix_edit_distance(q: &[char], a: &[char], transpositions: bool) -> Option<usize> {
    let n = q.len();
    let m = a.len();
    // dp[i][j] = distance between q[..i] and a[..j].
    let mut prev2: Vec<usize> = vec![0; m + 1];
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut best = prev[0..=m].iter().copied().min();
    for i in 1..=n {
        let mut cur = vec![i; m + 1];
        for j in 1..=m {
            let cost = if q[i - 1] == a[j - 1] { 0 } else { 1 };
            let mut v = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if transpositions && i > 1 && j > 1 && q[i - 1] == a[j - 2] && q[i - 2] == a[j - 1] {
                v = v.min(prev2[j - 2] + 1);
            }
            cur[j] = v;
        }
        best = cur.iter().copied().min();
        prev2 = std::mem::replace(&mut prev, cur);
    }
    best
}

/// Top `size` suggestions over `hits` (entry indices with their match
/// score factor), one per doc (its best input), weight desc then input
/// asc. With `skip_duplicates`, one per distinct input text.
pub fn top_suggestions(index: &CompletionIndex, hits: &[(usize, f32)], size: usize, skip_duplicates: bool) -> Vec<(Suggestion, f32)> {
    let mut best: std::collections::HashMap<DocId, (usize, f32)> = std::collections::HashMap::new();
    for &(i, f) in hits {
        let e = index.entry(i);
        let score = e.weight as f32 * f;
        match best.get(&e.doc) {
            Some(&(_, s)) if s >= score => {}
            _ => {
                best.insert(e.doc, (i, score));
            }
        }
    }
    let mut v: Vec<(usize, f32)> = best.into_values().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| index.entry(a.0).input.cmp(&index.entry(b.0).input)));
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (i, s) in v {
        let e = index.entry(i);
        if skip_duplicates && !seen.insert(e.input.clone()) {
            continue;
        }
        out.push((Suggestion { doc: e.doc, input: e.input.clone(), weight: e.weight, contexts: e.contexts.clone() }, s));
        if out.len() >= size {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(doc: DocId, input: &str, weight: u64) -> CompletionEntry {
        CompletionEntry { doc, input: input.into(), weight, contexts: vec![] }
    }

    #[test]
    fn prefix_and_fuzzy() {
        let idx = CompletionIndex::build(vec![e(0, "Nirvana", 5), e(1, "Nevermind", 3), e(1, "Nirvana Live", 7), e(2, "bar", 1)], |s| analyze(s, true, 50));
        let hits = idx.lookup(&analyze("nir", true, 50), None);
        assert_eq!(hits.len(), 2);
        let top = top_suggestions(&idx, &hits, 5, false);
        assert_eq!(top[0].0.input, "Nirvana Live");
        assert_eq!(top[1].0.input, "Nirvana");
        let hits = idx.lookup("nirv", None);
        assert_eq!(top_suggestions(&idx, &hits, 1, false).len(), 1);
        let fz = idx.lookup("nirb", Some((1, 1, true)));
        assert_eq!(fz.len(), 2);
        assert!(idx.lookup("zzz", None).is_empty());
        assert_eq!(analyze("Hello, World", true, 50), format!("hello{TOKEN_SEP}world"));
        assert_eq!(analyze("Hello, World", false, 50), "helloworld");
        assert_eq!(analyze("Hello", true, 3), "hel");
    }
}
