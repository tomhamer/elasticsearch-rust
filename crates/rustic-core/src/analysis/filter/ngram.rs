//! N-gram token filters (Lucene `NGramTokenFilter`, `EdgeNGramTokenFilter`).
//!
//! Unlike the n-gram *tokenizers*, these operate on already-produced
//! tokens: every gram keeps the source token's position and offsets.

use crate::analysis::{Token, TokenFilter};

/// Which end of the token edge n-grams are taken from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Side {
    /// Prefixes (ES default).
    #[default]
    Front,
    /// Suffixes (deprecated in ES; use `reverse` + front + `reverse`).
    Back,
}

/// Emits edge n-grams of each token, lengths `min..=max` (in characters).
///
/// Tokens shorter than `min` produce nothing (or the original token when
/// `preserve_original`). Tokens longer than `max` additionally emit the
/// original token when `preserve_original`.
#[derive(Clone, Debug)]
pub struct EdgeNGramTokenFilter {
    /// Minimum gram length (ES default 1).
    pub min: usize,
    /// Maximum gram length (ES default 2).
    pub max: usize,
    /// Front (prefix) or back (suffix) grams.
    pub side: Side,
    /// Also emit the original token when it is shorter than `min` or
    /// longer than `max`.
    pub preserve_original: bool,
}

impl Default for EdgeNGramTokenFilter {
    fn default() -> Self {
        Self { min: 1, max: 2, side: Side::Front, preserve_original: false }
    }
}

impl TokenFilter for EdgeNGramTokenFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::new();
        for t in tokens {
            let chars: Vec<char> = t.text.chars().collect();
            let n = chars.len();
            if n < self.min {
                if self.preserve_original {
                    out.push(t);
                }
                continue;
            }
            for len in self.min..=self.max.min(n) {
                let gram: String = match self.side {
                    Side::Front => chars[..len].iter().collect(),
                    Side::Back => chars[n - len..].iter().collect(),
                };
                out.push(Token { text: gram, ..t.clone() });
            }
            if self.preserve_original && n > self.max {
                out.push(t);
            }
        }
        out
    }
}

/// Emits all n-grams of each token, lengths `min..=max` (in characters),
/// shortest grams first, each at the source token's position/offsets.
#[derive(Clone, Debug)]
pub struct NGramTokenFilter {
    /// Minimum gram length (ES default 1).
    pub min: usize,
    /// Maximum gram length (ES default 2).
    pub max: usize,
    /// Also emit the original token when its length is outside `min..=max`.
    pub preserve_original: bool,
}

impl Default for NGramTokenFilter {
    fn default() -> Self {
        Self { min: 1, max: 2, preserve_original: false }
    }
}

impl TokenFilter for NGramTokenFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::new();
        for t in tokens {
            let chars: Vec<char> = t.text.chars().collect();
            let n = chars.len();
            if n < self.min {
                if self.preserve_original {
                    out.push(t);
                }
                continue;
            }
            for len in self.min..=self.max.min(n) {
                for start in 0..=n - len {
                    out.push(Token {
                        text: chars[start..start + len].iter().collect(),
                        ..t.clone()
                    });
                }
            }
            if self.preserve_original && n > self.max {
                out.push(t);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, pos: u32) -> Token {
        Token { text: text.into(), position: pos, offset_from: 10, offset_to: 10 + text.len() as u32 }
    }
    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }

    #[test]
    fn edge_front() {
        let f = EdgeNGramTokenFilter { min: 1, max: 3, ..Default::default() };
        let out = f.filter(vec![tok("quick", 4)]);
        assert_eq!(texts(&out), vec!["q", "qu", "qui"]);
        assert!(out.iter().all(|t| t.position == 4 && t.offset_from == 10 && t.offset_to == 15));
    }

    #[test]
    fn edge_back() {
        let f = EdgeNGramTokenFilter { min: 2, max: 3, side: Side::Back, ..Default::default() };
        assert_eq!(texts(&f.filter(vec![tok("quick", 0)])), vec!["ck", "ick"]);
    }

    #[test]
    fn edge_short_token() {
        let f = EdgeNGramTokenFilter { min: 3, max: 4, ..Default::default() };
        assert!(f.filter(vec![tok("ab", 0)]).is_empty());
        let f = EdgeNGramTokenFilter { min: 3, max: 4, preserve_original: true, ..Default::default() };
        assert_eq!(texts(&f.filter(vec![tok("ab", 0)])), vec!["ab"]);
    }

    #[test]
    fn edge_preserve_original_long() {
        let f = EdgeNGramTokenFilter { min: 1, max: 2, preserve_original: true, ..Default::default() };
        assert_eq!(texts(&f.filter(vec![tok("abc", 0)])), vec!["a", "ab", "abc"]);
        // Within range: no duplicate.
        assert_eq!(texts(&f.filter(vec![tok("ab", 0)])), vec!["a", "ab"]);
    }

    #[test]
    fn edge_unicode() {
        let f = EdgeNGramTokenFilter { min: 1, max: 2, ..Default::default() };
        assert_eq!(texts(&f.filter(vec![tok("éa", 0)])), vec!["é", "éa"]);
    }

    #[test]
    fn ngram_basic() {
        let f = NGramTokenFilter { min: 2, max: 3, preserve_original: false };
        let out = f.filter(vec![tok("cat", 7)]);
        assert_eq!(texts(&out), vec!["ca", "at", "cat"]);
        assert!(out.iter().all(|t| t.position == 7));
    }

    #[test]
    fn ngram_default_and_preserve() {
        let f = NGramTokenFilter::default();
        assert_eq!(texts(&f.filter(vec![tok("abc", 0)])), vec!["a", "b", "c", "ab", "bc"]);
        let f = NGramTokenFilter { preserve_original: true, ..Default::default() };
        assert_eq!(texts(&f.filter(vec![tok("abc", 0)])), vec!["a", "b", "c", "ab", "bc", "abc"]);
    }

    #[test]
    fn ngram_short() {
        let f = NGramTokenFilter { min: 3, max: 3, preserve_original: false };
        assert!(f.filter(vec![tok("ab", 0)]).is_empty());
        let f = NGramTokenFilter { min: 3, max: 3, preserve_original: true };
        assert_eq!(texts(&f.filter(vec![tok("ab", 0)])), vec!["ab"]);
    }
}
