//! Token filters: transform a token stream.
//!
//! Core filters (lowercase, stop, length, ascii folding, shingle, ...)
//! live here; larger families are split into submodules and re-exported.

use std::collections::HashSet;

pub mod cjk;
pub mod compound;
pub mod misc;
pub mod ngram;
pub mod normalization;
pub mod stem;
pub mod synonym;
pub mod word_delimiter;

pub use cjk::CjkBigramFilter;
pub use compound::DictionaryDecompounderFilter;
pub use misc::{
    ApostropheFilter, ClassicFilter, CommonGramsFilter, ElisionFilter, FingerprintFilter,
    KeepWordsFilter, KeywordMarkerFilter, KeywordRepeatFilter, LimitTokenCountFilter,
    MultiplexerFilter, RemoveDuplicatesFilter, StemmerOverrideFilter,
};
pub use ngram::{EdgeNGramTokenFilter, NGramTokenFilter, Side};
pub use normalization::{
    ArabicNormalizationFilter, CjkWidthFilter, DecimalDigitFilter, GermanNormalizationFilter,
    GreekLowerCaseFilter, HindiNormalizationFilter, IrishLowerCaseFilter, TurkishLowerCaseFilter, IndicNormalizationFilter, PersianNormalizationFilter,
    ScandinavianFoldingFilter, ScandinavianNormalizationFilter, SerbianNormalizationFilter,
    SoraniNormalizationFilter,
};
pub use stem::{
    ArabicStemFilter, EnglishPossessiveFilter, PersianStemFilter, KStemFilter, SnowballFilter, StemmerFilter,
};
pub use synonym::{SynonymFilter, SynonymGraphFilter, SynonymMap};
pub use word_delimiter::{WordDelimiterFilter, WordDelimiterGraphFilter};

use super::stemmer::porter_stem;
use super::stopwords;
use super::Token;

/// Lowercases each token's text (Unicode simple case folding).
#[derive(Clone, Debug, Default)]
pub struct LowerCaseFilter;

impl super::TokenFilter for LowerCaseFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.to_lowercase();
                t
            })
            .collect()
    }
}

/// Uppercases each token's text.
#[derive(Clone, Debug, Default)]
pub struct UpperCaseFilter;

impl super::TokenFilter for UpperCaseFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.to_uppercase();
                t
            })
            .collect()
    }
}

/// Removes tokens whose (lowercased) text is in the stop-word set.
///
/// Removed tokens are simply dropped; later tokens keep their original
/// positions, as Lucene does.
#[derive(Clone, Debug)]
pub struct StopFilter {
    /// The set of stop words (stored lowercase).
    pub words: HashSet<String>,
}

impl StopFilter {
    /// Create a stop filter from a slice of words (matched case-insensitively).
    pub fn from_slice(words: &[&str]) -> Self {
        Self {
            words: words.iter().map(|w| w.to_lowercase()).collect(),
        }
    }

    /// A stop filter using Lucene's English stop words.
    pub fn english() -> Self {
        Self::from_slice(stopwords::ENGLISH)
    }
}

impl super::TokenFilter for StopFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| !self.words.contains(&t.text.to_lowercase()))
            .collect()
    }
}

/// Keeps only tokens whose length (in characters) is in `min`..=`max`.
#[derive(Clone, Debug)]
pub struct LengthFilter {
    /// Minimum token length (characters).
    pub min: usize,
    /// Maximum token length (characters).
    pub max: usize,
}

impl super::TokenFilter for LengthFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| {
                let n = t.text.chars().count();
                (self.min..=self.max).contains(&n)
            })
            .collect()
    }
}

/// Folds common accented Latin characters to their ASCII equivalents
/// (é→e, ü→u, ß→ss, ø→o, æ→ae, ...). Covers the Latin-1 Supplement and
/// Latin Extended-A blocks.
#[derive(Clone, Debug, Default)]
pub struct AsciiFoldingFilter;

/// Map a single character to its ASCII-folded replacement (empty string
/// means the character is dropped).
fn fold_char(c: char) -> &'static str {
    match c {
        // Latin-1 Supplement
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => "A",
        'Æ' => "AE",
        'Ç' => "C",
        'È' | 'É' | 'Ê' | 'Ë' => "E",
        'Ì' | 'Í' | 'Î' | 'Ï' => "I",
        'Ð' => "D",
        'Ñ' => "N",
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => "O",
        'Ø' => "O",
        'Ù' | 'Ú' | 'Û' | 'Ü' => "U",
        'Ý' => "Y",
        'Þ' => "TH",
        'ß' => "ss",
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => "a",
        'æ' => "ae",
        'ç' => "c",
        'è' | 'é' | 'ê' | 'ë' => "e",
        'ì' | 'í' | 'î' | 'ï' => "i",
        'ð' => "d",
        'ñ' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => "o",
        'ø' => "o",
        'ù' | 'ú' | 'û' | 'ü' => "u",
        'ý' => "y",
        'þ' => "th",
        'ÿ' => "y",
        // Latin Extended-A
        'Ā' | 'ā' => "a",
        'Ă' | 'ă' => "a",
        'Ą' | 'ą' => "a",
        'Ć' | 'ć' => "c",
        'Ĉ' | 'ĉ' => "c",
        'Ċ' | 'ċ' => "c",
        'Č' | 'č' => "c",
        'Ď' | 'ď' => "d",
        'Đ' | 'đ' => "d",
        'Ē' | 'ē' => "e",
        'Ĕ' | 'ĕ' => "e",
        'Ė' | 'ė' => "e",
        'Ę' | 'ę' => "e",
        'Ě' | 'ě' => "e",
        'Ĝ' | 'ĝ' => "g",
        'Ğ' | 'ğ' => "g",
        'Ġ' | 'ġ' => "g",
        'Ģ' | 'ģ' => "g",
        'Ĥ' | 'ĥ' => "h",
        'Ħ' | 'ħ' => "h",
        'Ĩ' | 'ĩ' => "i",
        'Ī' | 'ī' => "i",
        'Ĭ' | 'ĭ' => "i",
        'Į' | 'į' => "i",
        'İ' => "i",
        'Ĳ' => "IJ",
        'ĳ' => "ij",
        'Ĵ' | 'ĵ' => "j",
        'Ķ' | 'ķ' => "k",
        'Ĺ' | 'ĺ' => "l",
        'Ļ' | 'ļ' => "l",
        'Ľ' | 'ľ' => "l",
        'Ŀ' | 'ŀ' => "l",
        'Ł' | 'ł' => "l",
        'Ń' | 'ń' => "n",
        'Ņ' | 'ņ' => "n",
        'Ň' | 'ň' => "n",
        'Ō' | 'ō' => "o",
        'Ŏ' | 'ŏ' => "o",
        'Ő' | 'ő' => "o",
        'Ŕ' | 'ŕ' => "r",
        'Ŗ' | 'ŗ' => "r",
        'Ř' | 'ř' => "r",
        'Ś' | 'ś' => "s",
        'Ŝ' | 'ŝ' => "s",
        'Ş' | 'ş' => "s",
        'Š' | 'š' => "s",
        'Ţ' | 'ţ' => "t",
        'Ť' | 'ť' => "t",
        'Ŧ' | 'ŧ' => "t",
        'Ũ' | 'ũ' => "u",
        'Ū' | 'ū' => "u",
        'Ŭ' | 'ŭ' => "u",
        'Ů' | 'ů' => "u",
        'Ű' | 'ű' => "u",
        'Ų' | 'ų' => "u",
        'Ŵ' | 'ŵ' => "w",
        'Ŷ' | 'ŷ' => "y",
        'Ÿ' => "y",
        'Ź' | 'ź' => "z",
        'Ż' | 'ż' => "z",
        'Ž' | 'ž' => "z",
        _ => "",
    }
}

impl super::TokenFilter for AsciiFoldingFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                let mut out = String::with_capacity(t.text.len());
                for c in t.text.chars() {
                    let folded = fold_char(c);
                    if folded.is_empty() {
                        out.push(c);
                    } else {
                        out.push_str(folded);
                    }
                }
                t.text = out;
                t
            })
            .collect()
    }
}

/// Strips leading and trailing whitespace from each token's text.
#[derive(Clone, Debug, Default)]
pub struct TrimFilter;

impl super::TokenFilter for TrimFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.trim().to_string();
                t
            })
            .collect()
    }
}

/// Keeps only the first occurrence of each token text within one
/// analyze call (deduplication).
#[derive(Clone, Debug, Default)]
pub struct UniqueFilter;

impl super::TokenFilter for UniqueFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut seen = HashSet::new();
        tokens
            .into_iter()
            .filter(|t| seen.insert(t.text.clone()))
            .collect()
    }
}

/// Expands the token stream into word n-grams (shingles) of
/// `min`..=`max` consecutive tokens, joined with `token_separator`
/// (Lucene `ShingleFilter`).
///
/// Each shingle takes the position of its first word and the byte
/// offsets spanning the whole shingle. With `output_unigrams` (the ES
/// default) the original tokens are emitted too; at each position the
/// unigram comes first, followed by its shingles in increasing size.
/// Position gaps (e.g. from removed stop words) are filled with
/// `filler_token` (ES default `_`) inside shingles.
#[derive(Clone, Debug)]
pub struct ShingleFilter {
    /// Minimum number of words per shingle (ES `min_shingle_size`).
    pub min: usize,
    /// Maximum number of words per shingle (ES `max_shingle_size`).
    pub max: usize,
    /// Also emit single tokens (ES `output_unigrams`).
    pub output_unigrams: bool,
    /// Emit unigrams only when no shingle can be formed
    /// (ES `output_unigrams_if_no_shingles`).
    pub output_unigrams_if_no_shingles: bool,
    /// Text used for each missing position inside a shingle.
    pub filler_token: String,
    /// String between the words of a shingle (default `" "`).
    pub token_separator: String,
}

impl ShingleFilter {
    /// Shingles of `min..=max` words, without unigrams, single-space
    /// separator, `_` filler.
    pub fn new(min: usize, max: usize) -> Self {
        Self {
            min,
            max,
            output_unigrams: false,
            output_unigrams_if_no_shingles: false,
            filler_token: "_".to_string(),
            token_separator: " ".to_string(),
        }
    }

    /// ES defaults: 2..=2 shingles plus unigrams.
    pub fn es_default() -> Self {
        Self { output_unigrams: true, ..Self::new(2, 2) }
    }
}

impl super::TokenFilter for ShingleFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let n = tokens.len();
        // Build a slot sequence where position gaps become filler slots.
        // Each slot: Some(token index) or None (filler).
        let mut slots: Vec<Option<usize>> = Vec::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            if i > 0 {
                let prev = tokens[i - 1].position;
                if t.position > prev + 1 {
                    for _ in 0..(t.position - prev - 1) {
                        slots.push(None);
                    }
                }
            }
            slots.push(Some(i));
        }
        let min = self.min.max(if self.output_unigrams { 2 } else { 1 });
        let mut shingles: Vec<(usize, usize, Token)> = Vec::new(); // (start token idx, size, shingle)
        for (si, slot) in slots.iter().enumerate() {
            let Some(start_idx) = *slot else { continue };
            for size in min..=self.max {
                if si + size > slots.len() {
                    break;
                }
                let window = &slots[si..si + size];
                // A shingle must end on a real token.
                let Some(last_idx) = window[size - 1] else { continue };
                let mut text = String::new();
                for (k, s) in window.iter().enumerate() {
                    if k > 0 {
                        text.push_str(&self.token_separator);
                    }
                    match s {
                        Some(ti) => text.push_str(&tokens[*ti].text),
                        None => text.push_str(&self.filler_token),
                    }
                }
                shingles.push((
                    start_idx,
                    size,
                    Token {
                        text,
                        position: tokens[start_idx].position,
                        offset_from: tokens[start_idx].offset_from,
                        offset_to: tokens[last_idx].offset_to,
                    },
                ));
            }
        }
        let emit_unigrams = self.output_unigrams
            || (self.output_unigrams_if_no_shingles && shingles.is_empty());
        if !emit_unigrams {
            // Without unigrams, keep the historical order: grouped by size.
            let mut by_size: Vec<(usize, usize, Token)> = shingles
                .into_iter()
                .enumerate()
                .map(|(order, (_, size, t))| (size, order, t))
                .collect();
            by_size.sort_by_key(|(size, order, _)| (*size, *order));
            return by_size.into_iter().map(|(_, _, t)| t).collect();
        }
        let mut out = Vec::with_capacity(n + shingles.len());
        let mut sh = shingles.into_iter().peekable();
        for (i, t) in tokens.into_iter().enumerate() {
            out.push(t);
            while let Some((start, _, _)) = sh.peek() {
                if *start != i {
                    break;
                }
                out.push(sh.next().unwrap().2);
            }
        }
        out
    }
}

/// Applies the Porter stemmer to each token's text.
#[derive(Clone, Debug, Default)]
pub struct PorterStemFilter;

impl super::TokenFilter for PorterStemFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = porter_stem(&t.text);
                t
            })
            .collect()
    }
}

/// Alias for [`PorterStemFilter`]: a token filter that applies the
/// Porter stemmer to each token.
pub type StemFilter = PorterStemFilter;

/// Truncates each token's text to at most `len` characters.
#[derive(Clone, Debug)]
pub struct TruncateFilter {
    /// Maximum token length (characters).
    pub len: usize,
}

impl super::TokenFilter for TruncateFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.chars().take(self.len).collect();
                t
            })
            .collect()
    }
}

/// Reverses each token's text (character order).
#[derive(Clone, Debug, Default)]
pub struct ReverseFilter;

impl super::TokenFilter for ReverseFilter {
    fn clone_box(&self) -> Box<dyn super::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = t.text.chars().rev().collect();
                t
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::TokenFilter;

    fn tok(text: &str, position: u32) -> Token {
        Token {
            text: text.to_string(),
            position,
            offset_from: 0,
            offset_to: text.len() as u32,
        }
    }

    fn texts(tokens: &[Token]) -> Vec<String> {
        tokens.iter().map(|t| t.text.clone()).collect()
    }

    // ---------- LowerCaseFilter ----------

    #[test]
    fn lowercase_basic() {
        let f = LowerCaseFilter;
        let out = f.filter(vec![tok("Hello", 0), tok("WORLD", 1)]);
        assert_eq!(texts(&out), vec!["hello", "world"]);
        assert_eq!(out[0].position, 0);
    }

    #[test]
    fn lowercase_unicode() {
        let f = LowerCaseFilter;
        let out = f.filter(vec![tok("İSTANBUL", 0)]);
        // Rust's to_lowercase maps İ -> i + combining dot above (U+0307)
        assert_eq!(out[0].text, "i\u{307}stanbul");
    }

    #[test]
    fn lowercase_empty() {
        let f = LowerCaseFilter;
        assert!(f.filter(vec![]).is_empty());
    }

    // ---------- UpperCaseFilter ----------

    #[test]
    fn uppercase_basic() {
        let f = UpperCaseFilter;
        let out = f.filter(vec![tok("hello", 0)]);
        assert_eq!(texts(&out), vec!["HELLO"]);
    }

    // ---------- StopFilter ----------

    #[test]
    fn stop_removes_and_keeps_positions() {
        let f = StopFilter::from_slice(&["the", "a"]);
        let out = f.filter(vec![
            tok("the", 0),
            tok("cat", 1),
            tok("a", 2),
            tok("dog", 3),
        ]);
        assert_eq!(texts(&out), vec!["cat", "dog"]);
        assert_eq!(
            out.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn stop_case_insensitive() {
        let f = StopFilter::from_slice(&["the"]);
        let out = f.filter(vec![tok("The", 0), tok("CAT", 1)]);
        assert_eq!(texts(&out), vec!["CAT"]);
    }

    #[test]
    fn stop_english() {
        let f = StopFilter::english();
        let out = f.filter(vec![
            tok("with", 0),
            tok("cats", 1),
            tok("and", 2),
            tok("dogs", 3),
        ]);
        assert_eq!(texts(&out), vec!["cats", "dogs"]);
    }

    #[test]
    fn stop_empty_set() {
        let f = StopFilter::from_slice(&[]);
        let in_tokens = vec![tok("a", 0)];
        assert_eq!(f.filter(in_tokens.clone()), in_tokens);
    }

    // ---------- LengthFilter ----------

    #[test]
    fn length_basic() {
        let f = LengthFilter { min: 2, max: 4 };
        let out = f.filter(vec![tok("a", 0), tok("ab", 1), tok("abcd", 2), tok("abcde", 3)]);
        assert_eq!(texts(&out), vec!["ab", "abcd"]);
    }

    #[test]
    fn length_unicode_chars() {
        // é is one character, two bytes.
        let f = LengthFilter { min: 2, max: 2 };
        let out = f.filter(vec![tok("éa", 0), tok("é", 1)]);
        assert_eq!(texts(&out), vec!["éa"]);
    }

    // ---------- AsciiFoldingFilter ----------

    #[test]
    fn ascii_folding_basic() {
        let f = AsciiFoldingFilter;
        let out = f.filter(vec![
            tok("café", 0),
            tok("naïve", 1),
            tok("straße", 2),
            tok("fjord", 3),
            tok("smørrebrød", 4),
        ]);
        assert_eq!(
            texts(&out),
            vec!["cafe", "naive", "strasse", "fjord", "smorrebrod"]
        );
    }

    #[test]
    fn ascii_folding_ligatures() {
        let f = AsciiFoldingFilter;
        let out = f.filter(vec![tok("cæsar", 0), tok("Æther", 1)]);
        // Æ -> AE (uppercase), so "Æther" -> "AEther"
        assert_eq!(texts(&out), vec!["caesar", "AEther"]);
    }

    #[test]
    fn ascii_folding_extended_a() {
        let f = AsciiFoldingFilter;
        let out = f.filter(vec![tok("żółć", 0), tok("Škoda", 1), tok("Łódź", 2)]);
        // Š -> s (lowercase), Ł -> l (lowercase)
        assert_eq!(texts(&out), vec!["zolc", "skoda", "lodz"]);
    }

    #[test]
    fn ascii_folding_passthrough() {
        let f = AsciiFoldingFilter;
        let out = f.filter(vec![tok("plain", 0), tok("日本語", 1)]);
        assert_eq!(texts(&out), vec!["plain", "日本語"]);
    }

    // ---------- TrimFilter ----------

    #[test]
    fn trim_basic() {
        let f = TrimFilter;
        let out = f.filter(vec![tok("  hello  ", 0), tok("world", 1)]);
        assert_eq!(texts(&out), vec!["hello", "world"]);
    }

    // ---------- UniqueFilter ----------

    #[test]
    fn unique_dedupes() {
        let f = UniqueFilter;
        let out = f.filter(vec![
            tok("cat", 0),
            tok("dog", 1),
            tok("cat", 2),
            tok("dog", 3),
        ]);
        assert_eq!(texts(&out), vec!["cat", "dog"]);
        assert_eq!(
            out.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn unique_case_sensitive() {
        let f = UniqueFilter;
        let out = f.filter(vec![tok("Cat", 0), tok("cat", 1)]);
        assert_eq!(texts(&out), vec!["Cat", "cat"]);
    }

    // ---------- ShingleFilter ----------

    #[test]
    fn shingle_basic() {
        let f = ShingleFilter::new(2, 2);
        let out = f.filter(vec![tok("the", 0), tok("quick", 1), tok("fox", 2)]);
        assert_eq!(texts(&out), vec!["the quick", "quick fox"]);
        assert_eq!(
            out.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn shingle_range() {
        let f = ShingleFilter::new(1, 3);
        let out = f.filter(vec![tok("a", 0), tok("b", 1), tok("c", 2)]);
        assert_eq!(
            texts(&out),
            vec!["a", "b", "c", "a b", "b c", "a b c"]
        );
    }

    #[test]
    fn shingle_fewer_tokens_than_min() {
        let f = ShingleFilter::new(2, 3);
        let out = f.filter(vec![tok("a", 0)]);
        assert!(out.is_empty());
    }

    #[test]
    fn shingle_offsets() {
        let f = ShingleFilter::new(2, 2);
        let tokens = vec![
            Token { text: "the".into(), position: 0, offset_from: 0, offset_to: 3 },
            Token { text: "fox".into(), position: 1, offset_from: 4, offset_to: 7 },
        ];
        let out = f.filter(tokens);
        assert_eq!(out[0].offset_from, 0);
        assert_eq!(out[0].offset_to, 7);
    }

    // ---------- StemFilter (Porter) ----------

    #[test]
    fn stem_basic() {
        let f = PorterStemFilter;
        let out = f.filter(vec![tok("caresses", 0), tok("ponies", 1), tok("hopping", 2)]);
        assert_eq!(texts(&out), vec!["caress", "poni", "hop"]);
    }

    #[test]
    fn stem_preserves_positions() {
        let f = PorterStemFilter;
        let out = f.filter(vec![tok("running", 5)]);
        assert_eq!(out[0].position, 5);
        assert_eq!(out[0].text, "run");
    }

    // ---------- TruncateFilter ----------

    #[test]
    fn truncate_basic() {
        let f = TruncateFilter { len: 3 };
        let out = f.filter(vec![tok("hello", 0), tok("hi", 1)]);
        assert_eq!(texts(&out), vec!["hel", "hi"]);
    }

    #[test]
    fn truncate_unicode() {
        let f = TruncateFilter { len: 1 };
        let out = f.filter(vec![tok("éa", 0)]);
        assert_eq!(texts(&out), vec!["é"]);
    }

    // ---------- ReverseFilter ----------

    #[test]
    fn reverse_basic() {
        let f = ReverseFilter;
        let out = f.filter(vec![tok("hello", 0)]);
        assert_eq!(texts(&out), vec!["olleh"]);
    }

    #[test]
    fn reverse_unicode() {
        let f = ReverseFilter;
        let out = f.filter(vec![tok("éa", 0)]);
        assert_eq!(texts(&out), vec!["aé"]);
    }

    #[test]
    fn shingle_output_unigrams() {
        let f = ShingleFilter::es_default();
        let out = f.filter(vec![tok("a", 0), tok("b", 1), tok("c", 2)]);
        assert_eq!(texts(&out), vec!["a", "a b", "b", "b c", "c"]);
        assert_eq!(out.iter().map(|t| t.position).collect::<Vec<_>>(), vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn shingle_separator_and_filler() {
        let f = ShingleFilter { token_separator: "_".into(), filler_token: "-".into(), ..ShingleFilter::new(2, 3) };
        // "b" at position 1 was removed (e.g. a stop word).
        let out = f.filter(vec![tok("a", 0), tok("c", 2), tok("d", 3)]);
        assert_eq!(texts(&out), vec!["c_d", "a_-_c"]);
        assert_eq!(out[1].position, 0);
    }

    #[test]
    fn shingle_unigrams_if_no_shingles() {
        let f = ShingleFilter { output_unigrams_if_no_shingles: true, ..ShingleFilter::new(2, 2) };
        assert_eq!(texts(&f.filter(vec![tok("solo", 0)])), vec!["solo"]);
        assert_eq!(texts(&f.filter(vec![tok("a", 0), tok("b", 1)])), vec!["a b"]);
    }
}
