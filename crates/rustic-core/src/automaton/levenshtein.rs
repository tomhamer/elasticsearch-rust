//! Levenshtein (and optionally Damerau) edit-distance matching over Unicode
//! characters, with the sparse-state ("banded DP") technique used by
//! Lucene's `LevenshteinAutomata`.
//!
//! A term matches when its edit distance to the pattern is at most
//! `max_edits` (0, 1 or 2) **and** it starts with the first `prefix_len`
//! characters of the pattern (Lucene's `FuzzyQuery` semantics: the prefix
//! is never edited).

use super::TermMatcher;

/// Compute the edit distance between two strings, character by character.
///
/// With `transpositions` this is the Damerau-Levenshtein distance (adjacent
/// transpositions cost 1); otherwise the plain Levenshtein distance.
/// Full O(n*m) DP — used for tests and for the fuzzy scoring boost.
pub fn edit_distance(a: &str, b: &str, transpositions: bool) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    // Three rows are needed: transpositions look back two rows.
    let mut prev2: Vec<usize> = vec![0; m + 1];
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let mut cost = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                prev[j - 1] + 1
            };
            cost = cost.min(prev[j] + 1); // deletion from a
            cost = cost.min(cur[j - 1] + 1); // insertion into a
            if transpositions
                && i > 1
                && j > 1
                && a[i - 1] == b[j - 2]
                && a[i - 2] == b[j - 1]
            {
                cost = cost.min(prev2[j - 2] + 1); // adjacent transposition
            }
            cur[j] = cost;
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// ES `fuzziness: AUTO` rule: 0 edits for terms of length <= 2, 1 for
/// length 3..=5, 2 otherwise.
pub fn auto_fuzziness(term_len_chars: usize) -> u8 {
    match term_len_chars {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

/// Matches terms within `max_edits` (0, 1 or 2) of `pattern`, optionally
/// counting adjacent transpositions as a single edit, requiring the first
/// `prefix_len` characters of the pattern to match exactly.
///
/// `matches` is O(len * (2d+1)) via the sparse-state technique: only the
/// band of states with at most `max_edits` errors is kept.
pub struct LevenshteinMatcher {
    pattern: Vec<char>,
    max_edits: u8,
    prefix_len: usize,
    transpositions: bool,
    /// The first `prefix_len` pattern characters as bytes: the literal
    /// prefix every match must start with.
    prefix_bytes: Vec<u8>,
}

impl LevenshteinMatcher {
    /// Build a matcher. `max_edits` must be 0, 1 or 2; `prefix_len` is
    /// clamped to the pattern length.
    pub fn new(pattern: &str, max_edits: u8, prefix_len: usize, transpositions: bool) -> Self {
        assert!(max_edits <= 2, "max_edits must be 0, 1 or 2");
        let pattern: Vec<char> = pattern.chars().collect();
        let prefix_len = prefix_len.min(pattern.len());
        let mut prefix_bytes = Vec::new();
        for c in &pattern[..prefix_len] {
            prefix_bytes.extend_from_slice(&c.to_string().into_bytes());
        }
        Self {
            pattern,
            max_edits,
            prefix_len,
            transpositions,
            prefix_bytes,
        }
    }

    /// Banded dynamic programme: row `j` (text consumed so far) holds the
    /// edit distance between `text[..j]` and `pattern[..i]` only for the
    /// `2d+1` pattern positions `i` in `[j-d, j+d]`; every other cell is
    /// provably `> d` and treated as infinity. This is exactly the set of
    /// live states of the Levenshtein automaton, so the cost is
    /// O(len * (2d+1)).
    fn matches_chars(&self, text: &[char]) -> bool {
        const INF: usize = usize::MAX / 2;
        let d = self.max_edits as usize;
        let n = self.pattern.len();
        let m = text.len();
        // |m - n| edits are needed at minimum.
        if m.saturating_sub(n) > d || n.saturating_sub(m) > d {
            return false;
        }
        let width = 2 * d + 1;
        // Band cell for pattern index `i` in row `j` lives at `i - j + d`.
        let get = |row: &[usize], j: usize, i: usize| -> usize {
            let k = i + d;
            if k < j || k - j >= width {
                INF
            } else {
                row[k - j]
            }
        };
        // Row 0: distance between "" and pattern[..i] is i.
        let mut prev2: Vec<usize> = vec![INF; width];
        let mut prev: Vec<usize> = vec![INF; width];
        for i in 0..=d.min(n) {
            prev[i + d] = i;
        }
        let mut cur: Vec<usize> = vec![INF; width];
        for j in 1..=m {
            let c = text[j - 1];
            cur.iter_mut().for_each(|v| *v = INF);
            let lo = j.saturating_sub(d);
            let hi = (j + d).min(n);
            let mut best = INF;
            for i in lo..=hi {
                // Insertion into pattern (consume text char).
                let mut cost = get(&prev, j - 1, i).saturating_add(1);
                if i > 0 {
                    // Deletion from pattern.
                    cost = cost.min(get(&cur, j, i - 1).saturating_add(1));
                    // Match / substitution.
                    let sub = if self.pattern[i - 1] == c { 0 } else { 1 };
                    cost = cost.min(get(&prev, j - 1, i - 1).saturating_add(sub));
                    // Adjacent transposition.
                    if self.transpositions
                        && i > 1
                        && j > 1
                        && self.pattern[i - 1] == text[j - 2]
                        && self.pattern[i - 2] == c
                    {
                        cost = cost.min(get(&prev2, j - 2, i - 2).saturating_add(1));
                    }
                }
                cur[i + d - j] = cost;
                best = best.min(cost);
            }
            if best > d {
                return false;
            }
            std::mem::swap(&mut prev2, &mut prev);
            std::mem::swap(&mut prev, &mut cur);
        }
        get(&prev, m, n) <= d
    }
}

impl TermMatcher for LevenshteinMatcher {
    fn matches(&self, term: &[u8]) -> bool {
        let Ok(text) = std::str::from_utf8(term) else {
            return false;
        };
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < self.prefix_len || chars[..self.prefix_len] != self.pattern[..self.prefix_len] {
            return false;
        }
        self.matches_chars(&chars)
    }

    fn literal_prefix(&self) -> &[u8] {
        &self.prefix_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_strings(max_len: usize) -> Vec<String> {
        let mut out = vec![String::new()];
        for _ in 0..max_len {
            let mut next = Vec::with_capacity(out.len() * 3);
            for s in &out {
                for c in ['a', 'b', 'c'] {
                    let mut t = s.clone();
                    t.push(c);
                    next.push(t);
                }
            }
            out.extend(next);
        }
        out
    }

    #[test]
    fn exhaustive_against_edit_distance() {
        let strings = all_strings(4);
        for pattern in ["abc", "abca"] {
            for d in [0u8, 1, 2] {
                for trans in [false, true] {
                    for prefix_len in [0usize, 1] {
                        let m = LevenshteinMatcher::new(pattern, d, prefix_len, trans);
                        for s in &strings {
                            let expected = {
                                let ok_prefix =
                                    prefix_len == 0 || s.starts_with(&pattern[..prefix_len]);
                                ok_prefix && edit_distance(pattern, s, trans) <= d as usize
                            };
                            assert_eq!(
                                m.matches(s.as_bytes()),
                                expected,
                                "pattern={pattern} d={d} trans={trans} prefix_len={prefix_len} s={s}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unicode_cafe() {
        let m = LevenshteinMatcher::new("café", 1, 0, false);
        assert!(m.matches("café".as_bytes()));
        assert!(m.matches("cafe".as_bytes()));
        assert!(m.matches("cafeé".as_bytes())); // one insertion
        assert!(!m.matches("caffee".as_bytes())); // two edits
        assert!(!m.matches("cafeine".as_bytes()));
        // The prefix must be compared char-wise, not byte-wise.
        let m2 = LevenshteinMatcher::new("café", 1, 2, false);
        assert!(m2.matches("cafe".as_bytes()));
        assert!(!m2.matches("bafé".as_bytes()));
        assert_eq!(m2.literal_prefix(), "ca".as_bytes());
    }

    #[test]
    fn transpositions() {
        let m = LevenshteinMatcher::new("abcd", 1, 0, true);
        assert!(m.matches("abdc".as_bytes()));
        let m = LevenshteinMatcher::new("abcd", 1, 0, false);
        assert!(!m.matches("abdc".as_bytes()));
        assert_eq!(edit_distance("abcd", "abdc", true), 1);
        assert_eq!(edit_distance("abcd", "abdc", false), 2);
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("kitten", "sitting", false), 3);
        assert_eq!(edit_distance("", "abc", false), 3);
        assert_eq!(edit_distance("abc", "", false), 3);
        assert_eq!(edit_distance("same", "same", false), 0);
        assert_eq!(edit_distance("flaw", "lawn", true), 2);
        assert_eq!(edit_distance("flaw", "lawn", false), 2);
        assert_eq!(edit_distance("acbd", "abcd", true), 1);
        assert_eq!(edit_distance("acbd", "abcd", false), 2);
    }

    #[test]
    fn auto_fuzziness_rule() {
        assert_eq!(auto_fuzziness(0), 0);
        assert_eq!(auto_fuzziness(2), 0);
        assert_eq!(auto_fuzziness(3), 1);
        assert_eq!(auto_fuzziness(5), 1);
        assert_eq!(auto_fuzziness(6), 2);
        assert_eq!(auto_fuzziness(100), 2);
    }

    #[test]
    fn literal_prefix_bytes() {
        let m = LevenshteinMatcher::new("café", 2, 1, false);
        assert_eq!(m.literal_prefix(), "c".as_bytes());
        let m = LevenshteinMatcher::new("éa", 2, 1, false);
        assert_eq!(m.literal_prefix(), "é".as_bytes());
        let m = LevenshteinMatcher::new("abc", 1, 10, false);
        assert_eq!(m.literal_prefix(), "abc".as_bytes());
        let m = LevenshteinMatcher::new("abc", 1, 0, false);
        assert!(m.literal_prefix().is_empty());
    }
}
