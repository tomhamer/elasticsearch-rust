//! Compound word decomposition (Lucene `DictionaryCompoundWordTokenFilter`).

use std::collections::HashSet;

use crate::analysis::{Token, TokenFilter};

/// Splits compound words into their parts using a word list
/// (ES `dictionary_decompounder`).
///
/// For every token with at least `min_word_size` characters, every
/// substring of `min_subword_size..=max_subword_size` characters that
/// is in `word_list` (compared lowercase) is emitted as an extra token
/// *after* the original, at the same position. With
/// `only_longest_match`, only the longest dictionary word starting at
/// each character offset is emitted.
///
/// Subwords keep the original token's offsets (Lucene does the same
/// when it cannot map offsets reliably).
#[derive(Clone, Debug)]
pub struct DictionaryDecompounderFilter {
    /// Dictionary of known words (stored lowercase).
    pub word_list: HashSet<String>,
    /// Minimum token length to attempt decomposition (ES default 5).
    pub min_word_size: usize,
    /// Minimum subword length (ES default 2).
    pub min_subword_size: usize,
    /// Maximum subword length (ES default 15).
    pub max_subword_size: usize,
    /// Emit only the longest match at each offset (ES default false).
    pub only_longest_match: bool,
}

impl DictionaryDecompounderFilter {
    /// Create with ES default sizes.
    pub fn new(words: &[&str]) -> Self {
        Self {
            word_list: words.iter().map(|w| w.to_lowercase()).collect(),
            min_word_size: 5,
            min_subword_size: 2,
            max_subword_size: 15,
            only_longest_match: false,
        }
    }
}

impl TokenFilter for DictionaryDecompounderFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for t in tokens {
            let chars: Vec<char> = t.text.to_lowercase().chars().collect();
            let n = chars.len();
            out.push(t.clone());
            if n < self.min_word_size || n < self.min_subword_size {
                continue;
            }
            for i in 0..=n - self.min_subword_size {
                let mut longest: Option<String> = None;
                for j in self.min_subword_size..=self.max_subword_size {
                    if i + j > n {
                        break;
                    }
                    let sub: String = chars[i..i + j].iter().collect();
                    if self.word_list.contains(&sub) {
                        if self.only_longest_match {
                            longest = Some(sub);
                        } else {
                            out.push(Token { text: sub, ..t.clone() });
                        }
                    }
                }
                if let Some(l) = longest {
                    out.push(Token { text: l, ..t.clone() });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> Token {
        Token { text: text.into(), position: 2, offset_from: 0, offset_to: text.len() as u32 }
    }
    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }

    #[test]
    fn decompounds_german() {
        let f = DictionaryDecompounderFilter::new(&["Donau", "dampf", "schiff", "fahrt"]);
        let out = f.filter(vec![tok("Donaudampfschiff")]);
        assert_eq!(texts(&out), vec!["Donaudampfschiff", "donau", "dampf", "schiff"]);
        assert!(out.iter().all(|t| t.position == 2 && t.offset_to == 16));
    }

    #[test]
    fn respects_min_word_size() {
        let f = DictionaryDecompounderFilter::new(&["ab", "cd"]);
        assert_eq!(texts(&f.filter(vec![tok("abcd")])), vec!["abcd"]);
    }

    #[test]
    fn only_longest_match() {
        let mut f = DictionaryDecompounderFilter::new(&["fuss", "fussball", "ball"]);
        f.only_longest_match = true;
        assert_eq!(texts(&f.filter(vec![tok("fussballspiel")])), vec!["fussballspiel", "fussball", "ball"]);
        f.only_longest_match = false;
        assert_eq!(
            texts(&f.filter(vec![tok("fussballspiel")])),
            vec!["fussballspiel", "fuss", "fussball", "ball"]
        );
    }

    #[test]
    fn subword_size_bounds() {
        let mut f = DictionaryDecompounderFilter::new(&["a", "basket", "ball"]);
        f.min_subword_size = 2;
        f.max_subword_size = 4;
        assert_eq!(texts(&f.filter(vec![tok("basketball")])), vec!["basketball", "ball"]);
    }

    #[test]
    fn no_match_passthrough() {
        let f = DictionaryDecompounderFilter::new(&["xyz"]);
        assert_eq!(texts(&f.filter(vec![tok("something")])), vec!["something"]);
        assert!(f.filter(vec![]).is_empty());
    }
}
