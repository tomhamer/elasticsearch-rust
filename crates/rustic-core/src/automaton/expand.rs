//! Term enumeration over the on-disk term dictionary, bounded by a
//! matcher's literal prefix.
//!
//! The dictionary keys are `2-byte LE field id + term bytes`, so a
//! `range_iter` over `[prefix, prefix + 1)` visits only terms of the
//! field whose text starts with `prefix` — the same trick Lucene's
//! `PrefixTermsEnum` / `FuzzyTermsEnum` use to keep the walk bounded.

use crate::codec::terms::TermDictReader;
use crate::term_dictionary::PostingsLocation;
use crate::types::Term;

use super::levenshtein::edit_distance;
use super::TermMatcher;

/// Increment the last byte of `prefix`, carrying; `None` if the prefix is
/// empty or all `0xFF` (in which case the walk is unbounded above).
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    if prefix.is_empty() {
        return None;
    }
    let mut upper = prefix.to_vec();
    for b in upper.iter_mut().rev() {
        if *b < 0xFF {
            *b += 1;
            return Some(upper);
        }
    }
    None
}

/// Enumerate up to `max_expansions` terms of `field` matching `matcher`,
/// in dictionary (byte) order.
///
/// The walk is bounded to the matcher's literal prefix range, then each
/// candidate is filtered with `matcher.matches`.
pub fn expand_terms(
    dict: &TermDictReader,
    field: u16,
    matcher: &dyn TermMatcher,
    max_expansions: usize,
) -> Vec<(Term, PostingsLocation)> {
    let prefix = matcher.literal_prefix();
    let upper = prefix_upper_bound(prefix);
    let candidates = dict.range_iter(field, Some(prefix), upper.as_deref());
    let mut out = Vec::new();
    for (key, loc) in candidates {
        let term = Term::from_key(&key);
        if matcher.matches(&term.text) {
            out.push((term, loc));
            if out.len() >= max_expansions {
                break;
            }
        }
    }
    out
}

/// Fast path for `PrefixQuery`: no matcher, just the prefix range.
pub fn expand_prefix(
    dict: &TermDictReader,
    field: u16,
    prefix: &[u8],
    max_expansions: usize,
) -> Vec<(Term, PostingsLocation)> {
    let upper = prefix_upper_bound(prefix);
    let candidates = dict.range_iter(field, Some(prefix), upper.as_deref());
    let mut out = Vec::new();
    for (key, loc) in candidates {
        out.push((Term::from_key(&key), loc));
        if out.len() >= max_expansions {
            break;
        }
    }
    out
}

/// Enumerate up to `max_expansions` terms of `field` within `max_edits`
/// (0, 1 or 2) of `text`, requiring the first `prefix_len` characters to
/// match exactly, optionally counting adjacent transpositions.
///
/// Results are sorted by (distance ascending, doc_freq descending) — the
/// same ordering Lucene's `FuzzyTermsEnum` uses to boost closer, more
/// frequent terms. The exact distance is recomputed with
/// [`edit_distance`] for the survivors.
pub fn expand_fuzzy(
    dict: &TermDictReader,
    field: u16,
    text: &str,
    max_edits: u8,
    prefix_len: usize,
    transpositions: bool,
    max_expansions: usize,
) -> Vec<(Term, PostingsLocation, u8 /* distance */)> {
    let matcher = super::levenshtein::LevenshteinMatcher::new(text, max_edits, prefix_len, transpositions);
    let candidates = expand_terms(dict, field, &matcher, max_expansions);
    let mut scored: Vec<(Term, PostingsLocation, u8)> = candidates
        .into_iter()
        .filter_map(|(term, loc)| {
            let Ok(t) = std::str::from_utf8(&term.text) else {
                return None;
            };
            let d = edit_distance(text, t, transpositions);
            debug_assert!(d <= max_edits as usize);
            Some((term, loc, d as u8))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.2.cmp(&b.2)
            .then(b.1.doc_freq.cmp(&a.1.doc_freq))
            .then_with(|| a.0.text.cmp(&b.0.text))
    });
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::FileSlice;
    use crate::codec::terms::TermDictWriter;
    use crate::automaton::levenshtein::LevenshteinMatcher;
    use crate::automaton::regexp::RegexpMatcher;
    use crate::automaton::regexp::RegexpFlags;
    use crate::automaton::wildcard::WildcardMatcher;

    /// Build a dictionary with 2000 terms across two fields.
    ///
    /// Field 0: `word_{i:04}` for i in 0..1000 (doc_freq = i % 7 + 1).
    /// Field 1: `other_{i:04}` for i in 0..1000 (doc_freq = (i % 5) + 1).
    fn build_dict() -> TermDictReader {
        let mut keys: Vec<(Vec<u8>, PostingsLocation)> = Vec::new();
        for i in 0..1000usize {
            keys.push((
                Term::new_text(0, &format!("word_{i:04}")).key(),
                PostingsLocation {
                    offset: i as u64 * 10,
                    doc_freq: (i % 7) as u32 + 1,
                    total_term_freq: (i % 7) as u64 * 3 + 2,
                },
            ));
            keys.push((
                Term::new_text(1, &format!("other_{i:04}")).key(),
                PostingsLocation {
                    offset: (i as u64 + 10000) * 10,
                    doc_freq: (i % 5) as u32 + 1,
                    total_term_freq: (i % 5) as u64 * 2 + 1,
                },
            ));
        }
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        let mut w = TermDictWriter::new();
        for (k, loc) in &keys {
            w.add(k, loc);
        }
        TermDictReader::open(FileSlice::from_vec(w.finish(1000))).unwrap()
    }

    fn brute_force(
        dict: &TermDictReader,
        field: u16,
        matcher: &dyn TermMatcher,
    ) -> Vec<(Term, PostingsLocation)> {
        dict.range_iter(field, None, None)
            .into_iter()
            .filter(|(key, _)| {
                let term = Term::from_key(key);
                matcher.matches(&term.text)
            })
            .map(|(key, loc)| (Term::from_key(&key), loc))
            .collect()
    }

    #[test]
    fn expand_prefix_matches_brute_force() {
        let dict = build_dict();
        for prefix in [&b"word_00"[..], b"word_0", b"word_", b"word_000", b"word_999", b"word_z"] {
            let got = expand_prefix(&dict, 0, prefix, usize::MAX);
            let expected: Vec<(Term, PostingsLocation)> = dict
                .range_iter(0, None, None)
                .into_iter()
                .filter(|(key, _)| {
                    let t = Term::from_key(key);
                    t.text.starts_with(prefix)
                })
                .map(|(key, loc)| (Term::from_key(&key), loc))
                .collect();
            assert_eq!(got, expected, "prefix {prefix:?}");
        }
        // Field 1 is untouched by a field-0 prefix.
        assert!(expand_prefix(&dict, 1, b"word_", usize::MAX).is_empty());
    }

    #[test]
    fn expand_terms_wildcard_matches_brute_force() {
        let dict = build_dict();
        for pattern in ["word_0*", "word_0?0", "word_*9", "word_00?"] {
            let matcher = WildcardMatcher::new(pattern);
            let got = expand_terms(&dict, 0, &matcher, usize::MAX);
            let expected = brute_force(&dict, 0, &matcher);
            assert_eq!(got, expected, "pattern {pattern}");
        }
    }

    #[test]
    fn expand_terms_regexp_matches_brute_force() {
        let dict = build_dict();
        for pattern in ["word_0[0-9]{3}", "word_[0-9][0-9][0-9][0-9]"] {
            let matcher = RegexpMatcher::new(pattern, RegexpFlags::NONE).unwrap();
            let got = expand_terms(&dict, 0, &matcher, usize::MAX);
            let expected = brute_force(&dict, 0, &matcher);
            assert_eq!(got, expected, "pattern {pattern}");
        }
    }

    #[test]
    fn expand_terms_fuzzy_matches_brute_force() {
        let dict = build_dict();
        let matcher = LevenshteinMatcher::new("word_0001", 1, 0, false);
        let got = expand_terms(&dict, 0, &matcher, usize::MAX);
        let expected = brute_force(&dict, 0, &matcher);
        assert_eq!(got, expected);
        assert!(!got.is_empty());
    }

    #[test]
    fn max_expansions_truncates() {
        let dict = build_dict();
        let got = expand_prefix(&dict, 0, b"word_", 50);
        assert_eq!(got.len(), 50);
        // Truncation keeps dictionary order: the first 50 of the full walk.
        let full = expand_prefix(&dict, 0, b"word_", usize::MAX);
        assert_eq!(got, full[..50]);

        let matcher = WildcardMatcher::new("word_0*");
        let got = expand_terms(&dict, 0, &matcher, 10);
        assert_eq!(got.len(), 10);
        let full = expand_terms(&dict, 0, &matcher, usize::MAX);
        assert_eq!(got, full[..10]);
    }

    #[test]
    fn expand_fuzzy_sorted_by_distance_then_doc_freq() {
        let dict = build_dict();
        let got = expand_fuzzy(&dict, 0, "word_0001", 1, 0, false, usize::MAX);
        assert!(!got.is_empty());
        // All within 1 edit, and the exact term (distance 0) comes first.
        assert_eq!(got[0].0.text, b"word_0001");
        assert_eq!(got[0].2, 0);
        for i in 1..got.len() {
            let (cur, prev) = (&got[i], &got[i - 1]);
            assert!(
                cur.2 >= prev.2,
                "distance not ascending at {i}: {:?} d={} after d={}",
                cur.0, cur.2, prev.2
            );
            if cur.2 == prev.2 {
                assert!(
                    cur.1.doc_freq <= prev.1.doc_freq,
                    "doc_freq not descending within distance {}",
                    cur.2
                );
            }
        }
        // Cross-check the set against a brute-force filter.
        let matcher = LevenshteinMatcher::new("word_0001", 1, 0, false);
        let mut expected: Vec<Vec<u8>> = brute_force(&dict, 0, &matcher)
            .into_iter()
            .map(|(t, _)| t.text)
            .collect();
        let mut got_terms: Vec<Vec<u8>> = got.into_iter().map(|(t, _, _)| t.text).collect();
        got_terms.sort();
        expected.sort();
        assert_eq!(got_terms, expected);
    }

    #[test]
    fn expand_fuzzy_respects_prefix_len() {
        let dict = build_dict();
        // prefix_len 5 pins "word_": only terms starting with "word_"
        // within 1 edit of "word_0001".
        let got = expand_fuzzy(&dict, 0, "word_0001", 1, 5, false, usize::MAX);
        for (t, _, d) in &got {
            assert!(t.text.starts_with(b"word_"), "{t:?}");
            assert!(*d <= 1);
        }
        // Without the prefix constraint the set is a superset.
        let all = expand_fuzzy(&dict, 0, "word_0001", 1, 0, false, usize::MAX);
        assert!(all.len() >= got.len());
    }

    #[test]
    fn empty_prefix_walks_whole_field() {
        let dict = build_dict();
        let matcher = WildcardMatcher::new("*");
        let got = expand_terms(&dict, 0, &matcher, usize::MAX);
        assert_eq!(got.len(), 1000);
        assert_eq!(expand_prefix(&dict, 0, b"", usize::MAX).len(), 1000);
    }
}
