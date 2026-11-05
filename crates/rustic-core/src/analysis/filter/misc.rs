//! Assorted token filters: apostrophe, classic, common grams, elision,
//! fingerprint, keep words, keyword marker/repeat, limit, multiplexer,
//! remove duplicates, stemmer override.

use std::collections::{HashMap, HashSet};

use crate::analysis::{Token, TokenFilter};

/// Is `c` an apostrophe-like character (ASCII `'` or U+2019 `’`)?
fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '\u{2019}'
}

// ---------------------------------------------------------------------
// ApostropheFilter
// ---------------------------------------------------------------------

/// Strips everything after the first apostrophe, including the
/// apostrophe (Lucene `ApostropheFilter`, used by the Turkish analyzer:
/// `Türkiye'nin` → `Türkiye`).
#[derive(Clone, Debug, Default)]
pub struct ApostropheFilter;

impl TokenFilter for ApostropheFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                if let Some(idx) = t.text.find(is_apostrophe) {
                    t.text.truncate(idx);
                }
                t
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// ClassicFilter
// ---------------------------------------------------------------------

/// Lucene `ClassicFilter`: removes a trailing English possessive `'s`
/// and removes the dots from acronyms (`I.B.M.` → `IBM`).
///
/// Any token made of letters separated by single dots (e.g. `I.B.M.`,
/// `U.S.A`) is treated as an acronym.
#[derive(Clone, Debug, Default)]
pub struct ClassicFilter;

fn is_acronym(s: &str) -> bool {
    let mut saw_dot = false;
    let mut prev_dot = true; // disallow leading dot
    for c in s.chars() {
        if c == '.' {
            if prev_dot {
                return false;
            }
            saw_dot = true;
            prev_dot = true;
        } else if c.is_alphabetic() {
            prev_dot = false;
        } else {
            return false;
        }
    }
    saw_dot
}

impl TokenFilter for ClassicFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                let chars: Vec<char> = t.text.chars().collect();
                let n = chars.len();
                if n >= 2 && is_apostrophe(chars[n - 2]) && matches!(chars[n - 1], 's' | 'S') {
                    t.text = chars[..n - 2].iter().collect();
                } else if is_acronym(&t.text) {
                    t.text = t.text.chars().filter(|c| *c != '.').collect();
                }
                t
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// CommonGramsFilter
// ---------------------------------------------------------------------

/// Lucene `CommonGramsFilter` / `CommonGramsQueryFilter`.
///
/// For every pair of adjacent tokens where at least one is a *common
/// word*, a bigram `first_second` is emitted at the position of the first
/// token (offsets spanning both). In index mode (`query_mode == false`)
/// all original tokens are kept too: `the quick` → `the`, `the_quick`,
/// `quick`. In query mode the bigram replaces the common word and the
/// unigram that follows it when another token follows: `the quick brown`
/// → `the_quick`, `quick`, `brown`; `brown the` → `brown_the`. This
/// mirrors the exact state machine of Lucene's query filter.
#[derive(Clone, Debug)]
pub struct CommonGramsFilter {
    /// Common words (stored lowercase; matched case-insensitively, like
    /// ES `ignore_case: true`).
    pub common_words: HashSet<String>,
    /// Emit query-mode output (ES `query_mode`, default false).
    pub query_mode: bool,
}

impl CommonGramsFilter {
    /// Create from a word list.
    pub fn new(words: &[&str], query_mode: bool) -> Self {
        Self { common_words: words.iter().map(|w| w.to_lowercase()).collect(), query_mode }
    }

    fn is_common(&self, t: &Token) -> bool {
        self.common_words.contains(&t.text.to_lowercase())
    }
}

impl TokenFilter for CommonGramsFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        // (token, is_gram)
        let mut stream: Vec<(Token, bool)> = Vec::with_capacity(tokens.len() * 2);
        for (i, t) in tokens.iter().enumerate() {
            stream.push((t.clone(), false));
            if let Some(next) = tokens.get(i + 1) {
                if self.is_common(t) || self.is_common(next) {
                    stream.push((
                        Token {
                            text: format!("{}_{}", t.text, next.text),
                            position: t.position,
                            offset_from: t.offset_from,
                            offset_to: next.offset_to,
                        },
                        true,
                    ));
                }
            }
        }
        if !self.query_mode {
            return stream.into_iter().map(|(t, _)| t).collect();
        }
        // Port of CommonGramsQueryFilter.
        let mut out = Vec::new();
        let mut previous: Option<(Token, bool)> = None;
        let mut previous_type_gram = false;
        for current in stream {
            if previous.is_some() && !current.1 {
                let (p, p_gram) = previous.take().unwrap();
                previous_type_gram = p_gram;
                out.push(p);
            }
            previous = Some(current);
        }
        if let Some((p, _)) = previous {
            if !previous_type_gram {
                out.push(p);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------
// ElisionFilter
// ---------------------------------------------------------------------

/// Removes elided articles before an apostrophe (Lucene `ElisionFilter`):
/// `l'avion` → `avion`, `qu'il` → `il`.
#[derive(Clone, Debug)]
pub struct ElisionFilter {
    /// Articles to strip (without the apostrophe).
    pub articles: HashSet<String>,
    /// Match articles case-sensitively (ES `articles_case`, default false).
    pub articles_case: bool,
}

impl ElisionFilter {
    /// Create from an article list.
    pub fn new(articles: &[&str], articles_case: bool) -> Self {
        let articles = articles
            .iter()
            .map(|a| if articles_case { a.to_string() } else { a.to_lowercase() })
            .collect();
        Self { articles, articles_case }
    }

    /// Lucene's default French articles (also the ES default).
    pub fn french() -> Self {
        Self::new(
            &["l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu", "puisqu"],
            false,
        )
    }

    /// Catalan articles used by Lucene's `CatalanAnalyzer`.
    pub fn catalan() -> Self {
        Self::new(&["d", "l", "m", "n", "s", "t"], false)
    }

    /// Italian articles used by Lucene's `ItalianAnalyzer`.
    pub fn italian() -> Self {
        Self::new(
            &[
                "c", "l", "all", "dall", "dell", "nell", "sull", "coll", "pell", "gl", "agl",
                "dagl", "degl", "negl", "sugl", "un", "m", "t", "s", "v", "d",
            ],
            false,
        )
    }

    /// Irish elision used by Lucene's `IrishAnalyzer`.
    pub fn irish() -> Self {
        Self::new(&["d", "m", "b"], false)
    }
}

impl TokenFilter for ElisionFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                if let Some(idx) = t.text.find(is_apostrophe) {
                    let prefix = &t.text[..idx];
                    let key = if self.articles_case { prefix.to_string() } else { prefix.to_lowercase() };
                    if self.articles.contains(&key) {
                        let apo_len = t.text[idx..].chars().next().unwrap().len_utf8();
                        t.text = t.text[idx + apo_len..].to_string();
                    }
                }
                t
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// FingerprintFilter
// ---------------------------------------------------------------------

/// Lucene `FingerprintFilter`: sorts and de-duplicates all token texts
/// and joins them with `separator` into a single token. If the result
/// would exceed `max_output_size` characters, nothing is emitted.
#[derive(Clone, Debug)]
pub struct FingerprintFilter {
    /// Join character (ES default space).
    pub separator: char,
    /// Maximum fingerprint length in characters (ES default 255).
    pub max_output_size: usize,
}

impl Default for FingerprintFilter {
    fn default() -> Self {
        Self { separator: ' ', max_output_size: 255 }
    }
}

impl TokenFilter for FingerprintFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        if tokens.is_empty() {
            return tokens;
        }
        let mut texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        texts.sort_unstable();
        texts.dedup();
        let mut text = String::new();
        for (i, s) in texts.iter().enumerate() {
            if i > 0 {
                text.push(self.separator);
            }
            text.push_str(s);
        }
        if text.chars().count() > self.max_output_size {
            return Vec::new();
        }
        let offset_from = tokens.iter().map(|t| t.offset_from).min().unwrap();
        let offset_to = tokens.iter().map(|t| t.offset_to).max().unwrap();
        vec![Token { text, position: tokens[0].position, offset_from, offset_to }]
    }
}

// ---------------------------------------------------------------------
// KeepWordsFilter
// ---------------------------------------------------------------------

/// Keeps only tokens in `keep_words` (Lucene `KeepWordFilter`).
#[derive(Clone, Debug)]
pub struct KeepWordsFilter {
    /// Words to keep.
    pub keep_words: HashSet<String>,
    /// Compare case-insensitively (ES `keep_words_case: true` lowercases
    /// the list; here both sides are lowercased).
    pub ignore_case: bool,
}

impl KeepWordsFilter {
    /// Create from a word list.
    pub fn new(words: &[&str], ignore_case: bool) -> Self {
        Self {
            keep_words: words
                .iter()
                .map(|w| if ignore_case { w.to_lowercase() } else { w.to_string() })
                .collect(),
            ignore_case,
        }
    }
}

impl TokenFilter for KeepWordsFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| {
                if self.ignore_case {
                    self.keep_words.contains(&t.text.to_lowercase())
                } else {
                    self.keep_words.contains(&t.text)
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// KeywordMarkerFilter
// ---------------------------------------------------------------------

/// Protects keywords from stemming (Lucene `KeywordMarkerFilter`).
///
/// Lucene sets a keyword *attribute* that stemmers consult. Our
/// [`Token`] has no such attribute, so this filter instead *wraps* the
/// stemmer: tokens whose text is in `keywords` pass through untouched,
/// all others go through `inner`. `inner` is applied to each token
/// individually, so it should be a 1:1 filter (any stemmer is).
pub struct KeywordMarkerFilter {
    inner: Box<dyn TokenFilter>,
    /// Protected words.
    pub keywords: HashSet<String>,
    /// Match case-insensitively (ES `ignore_case`, default false).
    pub ignore_case: bool,
}

impl KeywordMarkerFilter {
    /// Wrap `inner` so it skips `keywords`.
    pub fn wrap(inner: Box<dyn TokenFilter>, keywords: &[&str]) -> Self {
        Self { inner, keywords: keywords.iter().map(|k| k.to_string()).collect(), ignore_case: false }
    }

    /// Builder: match keywords case-insensitively.
    pub fn ignore_case(mut self, yes: bool) -> Self {
        self.ignore_case = yes;
        if yes {
            self.keywords = self.keywords.iter().map(|k| k.to_lowercase()).collect();
        }
        self
    }

    fn is_keyword(&self, t: &Token) -> bool {
        if self.ignore_case {
            self.keywords.contains(&t.text.to_lowercase())
        } else {
            self.keywords.contains(&t.text)
        }
    }
}

impl std::fmt::Debug for KeywordMarkerFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordMarkerFilter").field("keywords", &self.keywords.len()).finish()
    }
}

impl TokenFilter for KeywordMarkerFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(Self {
            inner: self.inner.clone_box(),
            keywords: self.keywords.clone(),
            ignore_case: self.ignore_case,
        })
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for t in tokens {
            if self.is_keyword(&t) {
                out.push(t);
            } else {
                out.extend(self.inner.filter(vec![t]));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------
// KeywordRepeatFilter
// ---------------------------------------------------------------------

/// Emits each token twice at the same position (Lucene
/// `KeywordRepeatFilter`), so that one copy can be stemmed while the
/// other is kept verbatim. Lucene marks the first copy as a keyword so
/// a later stemmer skips it; since we have no keyword attribute, use
/// [`KeywordRepeatFilter::with_stemmer`] to get `original, stemmed`
/// pairs directly, and follow with [`RemoveDuplicatesFilter`] to drop
/// pairs where stemming changed nothing.
pub struct KeywordRepeatFilter {
    stemmer: Option<Box<dyn TokenFilter>>,
}

impl Default for KeywordRepeatFilter {
    fn default() -> Self {
        Self { stemmer: None }
    }
}

impl KeywordRepeatFilter {
    /// Plain repeat: every token is emitted twice.
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the original token followed by `stemmer(token)` at the same
    /// position.
    pub fn with_stemmer(stemmer: Box<dyn TokenFilter>) -> Self {
        Self { stemmer: Some(stemmer) }
    }
}

impl std::fmt::Debug for KeywordRepeatFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordRepeatFilter").field("stemmer", &self.stemmer.is_some()).finish()
    }
}

impl TokenFilter for KeywordRepeatFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(Self { stemmer: self.stemmer.as_ref().map(|s| s.clone_box()) })
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len() * 2);
        for t in tokens {
            out.push(t.clone());
            match &self.stemmer {
                Some(s) => out.extend(s.filter(vec![t])),
                None => out.push(t),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------
// LimitTokenCountFilter
// ---------------------------------------------------------------------

/// Keeps only the first `max_token_count` tokens (Lucene
/// `LimitTokenCountFilter`). `consume_all` only affects streaming
/// implementations (whether the rest of the input is read); it has no
/// observable effect here and is kept for configuration compatibility.
#[derive(Clone, Debug)]
pub struct LimitTokenCountFilter {
    /// Maximum number of tokens to emit (ES default 1).
    pub max_token_count: usize,
    /// Ignored (see type docs).
    pub consume_all: bool,
}

impl Default for LimitTokenCountFilter {
    fn default() -> Self {
        Self { max_token_count: 1, consume_all: false }
    }
}

impl TokenFilter for LimitTokenCountFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, mut tokens: Vec<Token>) -> Vec<Token> {
        tokens.truncate(self.max_token_count);
        tokens
    }
}

// ---------------------------------------------------------------------
// MultiplexerFilter
// ---------------------------------------------------------------------

/// Runs each token through several filter chains in parallel and emits
/// every result at the token's position (Lucene `MultiplexerFilter`
/// semantics, ES `multiplexer`). Duplicates (same text at the same
/// position) are removed. With `preserve_original` the unmodified token
/// is emitted first.
pub struct MultiplexerFilter {
    /// The branches; each is a chain applied in order to a single token.
    pub filters: Vec<Vec<Box<dyn TokenFilter>>>,
    /// Also emit the original token (ES default true).
    pub preserve_original: bool,
}

impl MultiplexerFilter {
    /// Create a multiplexer over the given branches.
    pub fn new(filters: Vec<Vec<Box<dyn TokenFilter>>>, preserve_original: bool) -> Self {
        Self { filters, preserve_original }
    }
}

impl std::fmt::Debug for MultiplexerFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiplexerFilter")
            .field("branches", &self.filters.len())
            .field("preserve_original", &self.preserve_original)
            .finish()
    }
}

impl TokenFilter for MultiplexerFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(Self {
            filters: self
                .filters
                .iter()
                .map(|chain| chain.iter().map(|f| f.clone_box()).collect())
                .collect(),
            preserve_original: self.preserve_original,
        })
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::new();
        for t in tokens {
            let mut seen: HashSet<String> = HashSet::new();
            if self.preserve_original {
                seen.insert(t.text.clone());
                out.push(t.clone());
            }
            for chain in &self.filters {
                let mut branch = vec![t.clone()];
                for f in chain {
                    branch = f.filter(branch);
                }
                for mut b in branch {
                    b.position = t.position;
                    if seen.insert(b.text.clone()) {
                        out.push(b);
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------
// RemoveDuplicatesFilter
// ---------------------------------------------------------------------

/// Removes tokens that have the same text as an earlier token at the
/// same position (Lucene `RemoveDuplicatesTokenFilter`).
#[derive(Clone, Debug, Default)]
pub struct RemoveDuplicatesFilter;

impl TokenFilter for RemoveDuplicatesFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut seen: HashSet<(u32, String)> = HashSet::new();
        tokens.into_iter().filter(|t| seen.insert((t.position, t.text.clone()))).collect()
    }
}

// ---------------------------------------------------------------------
// StemmerOverrideFilter
// ---------------------------------------------------------------------

/// Overrides stemming for specific words (Lucene `StemmerOverrideFilter`).
///
/// Tokens found in `rules` are replaced by the mapped form. In Lucene the
/// replaced token is also marked as a keyword so later stemmers leave it
/// alone; here, pass the stemmer to [`StemmerOverrideFilter::wrap`] and
/// non-overridden tokens are stemmed by it while overridden ones are
/// protected.
pub struct StemmerOverrideFilter {
    /// `word → stem` mappings.
    pub rules: HashMap<String, String>,
    /// Match case-insensitively (ES `ignore_case`, default false).
    pub ignore_case: bool,
    inner: Option<Box<dyn TokenFilter>>,
}

impl StemmerOverrideFilter {
    /// Create from a mapping.
    pub fn new(rules: HashMap<String, String>) -> Self {
        Self { rules, ignore_case: false, inner: None }
    }

    /// Parse ES rule strings `"running, runs => run"` (comma-separated
    /// inputs on the left). Lines without `=>` are ignored.
    pub fn from_rules(rules: &[&str]) -> Self {
        let mut map = HashMap::new();
        for r in rules {
            if let Some((lhs, rhs)) = r.split_once("=>") {
                for w in lhs.split(',') {
                    let w = w.trim();
                    if !w.is_empty() {
                        map.insert(w.to_string(), rhs.trim().to_string());
                    }
                }
            }
        }
        Self::new(map)
    }

    /// Protect overridden words from `stemmer`, which handles all others.
    pub fn wrap(stemmer: Box<dyn TokenFilter>, rules: HashMap<String, String>) -> Self {
        Self { rules, ignore_case: false, inner: Some(stemmer) }
    }

    /// Builder: case-insensitive matching.
    pub fn ignore_case(mut self, yes: bool) -> Self {
        self.ignore_case = yes;
        if yes {
            self.rules = self.rules.iter().map(|(k, v)| (k.to_lowercase(), v.clone())).collect();
        }
        self
    }
}

impl std::fmt::Debug for StemmerOverrideFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StemmerOverrideFilter").field("rules", &self.rules.len()).finish()
    }
}

impl TokenFilter for StemmerOverrideFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(Self {
            rules: self.rules.clone(),
            ignore_case: self.ignore_case,
            inner: self.inner.as_ref().map(|i| i.clone_box()),
        })
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for mut t in tokens {
            let key = if self.ignore_case { t.text.to_lowercase() } else { t.text.clone() };
            if let Some(stem) = self.rules.get(&key) {
                t.text = stem.clone();
                out.push(t);
            } else if let Some(inner) = &self.inner {
                out.extend(inner.filter(vec![t]));
            } else {
                out.push(t);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::filter::{LowerCaseFilter, PorterStemFilter, UpperCaseFilter};

    fn tok(text: &str, pos: u32) -> Token {
        Token { text: text.into(), position: pos, offset_from: pos * 10, offset_to: pos * 10 + text.len() as u32 }
    }
    fn toks(words: &[&str]) -> Vec<Token> {
        words.iter().enumerate().map(|(i, w)| tok(w, i as u32)).collect()
    }
    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }
    fn positions(t: &[Token]) -> Vec<u32> {
        t.iter().map(|t| t.position).collect()
    }

    #[test]
    fn apostrophe() {
        let out = ApostropheFilter.filter(toks(&["Türkiye'nin", "İstanbul’da", "plain", "'x"]));
        assert_eq!(texts(&out), vec!["Türkiye", "İstanbul", "plain", ""]);
    }

    #[test]
    fn classic_possessive_and_acronym() {
        let out = ClassicFilter.filter(toks(&["Bob's", "I.B.M.", "U.S.A", "dog’S", "a..b", "1.2"]));
        assert_eq!(texts(&out), vec!["Bob", "IBM", "USA", "dog", "a..b", "1.2"]);
    }

    #[test]
    fn common_grams_index_mode() {
        let f = CommonGramsFilter::new(&["the", "of"], false);
        let out = f.filter(toks(&["the", "quick", "brown", "of", "the", "fox"]));
        assert_eq!(
            texts(&out),
            vec!["the", "the_quick", "quick", "brown", "brown_of", "of", "of_the", "the", "the_fox", "fox"]
        );
        assert_eq!(positions(&out), vec![0, 0, 1, 2, 2, 3, 3, 4, 4, 5]);
        assert_eq!((out[1].offset_from, out[1].offset_to), (0, 15));
    }

    #[test]
    fn common_grams_query_mode() {
        let f = CommonGramsFilter::new(&["the"], true);
        assert_eq!(texts(&f.filter(toks(&["the", "quick", "brown"]))), vec!["the_quick", "quick", "brown"]);
        assert_eq!(texts(&f.filter(toks(&["brown", "the"]))), vec!["brown_the"]);
        assert_eq!(texts(&f.filter(toks(&["the"]))), vec!["the"]);
        assert_eq!(texts(&f.filter(toks(&["quick", "brown"]))), vec!["quick", "brown"]);
        assert_eq!(texts(&f.filter(toks(&["quick", "the", "fox"]))), vec!["quick_the", "the_fox"]);
    }

    #[test]
    fn common_grams_case_insensitive_and_empty() {
        let f = CommonGramsFilter::new(&["the"], false);
        assert_eq!(texts(&f.filter(toks(&["The", "cat"]))), vec!["The", "The_cat", "cat"]);
        assert!(f.filter(vec![]).is_empty());
    }

    #[test]
    fn elision_french() {
        let f = ElisionFilter::french();
        let out = f.filter(toks(&["l'avion", "qu'il", "L’Homme", "aujourd'hui", "jusqu'à", "plain"]));
        assert_eq!(texts(&out), vec!["avion", "il", "Homme", "aujourd'hui", "à", "plain"]);
    }

    #[test]
    fn elision_case_sensitive() {
        let f = ElisionFilter::new(&["l"], true);
        assert_eq!(texts(&f.filter(toks(&["l'avion", "L'avion"]))), vec!["avion", "L'avion"]);
        let f = ElisionFilter::italian();
        assert_eq!(texts(&f.filter(toks(&["dell'arte", "un'ora"]))), vec!["arte", "ora"]);
    }

    #[test]
    fn fingerprint_sorts_dedupes_joins() {
        let f = FingerprintFilter::default();
        let out = f.filter(toks(&["zebra", "apple", "zebra", "mango"]));
        assert_eq!(texts(&out), vec!["apple mango zebra"]);
        assert_eq!(out[0].position, 0);
        assert_eq!((out[0].offset_from, out[0].offset_to), (0, 35));
    }

    #[test]
    fn fingerprint_separator_and_limit() {
        let f = FingerprintFilter { separator: '+', max_output_size: 5 };
        assert_eq!(texts(&f.filter(toks(&["b", "a"]))), vec!["a+b"]);
        assert!(f.filter(toks(&["abc", "def"])).is_empty());
        assert!(f.filter(vec![]).is_empty());
    }

    #[test]
    fn keep_words() {
        let f = KeepWordsFilter::new(&["cat", "dog"], false);
        let out = f.filter(toks(&["cat", "Dog", "bird", "dog"]));
        assert_eq!(texts(&out), vec!["cat", "dog"]);
        assert_eq!(positions(&out), vec![0, 3]);
        let f = KeepWordsFilter::new(&["Cat"], true);
        assert_eq!(texts(&f.filter(toks(&["CAT", "x"]))), vec!["CAT"]);
    }

    #[test]
    fn keyword_marker_protects_from_stemmer() {
        let f = KeywordMarkerFilter::wrap(Box::new(PorterStemFilter), &["running"]);
        let out = f.filter(toks(&["running", "jumping"]));
        assert_eq!(texts(&out), vec!["running", "jump"]);
        let f = f.ignore_case(true);
        assert_eq!(texts(&f.filter(toks(&["Running"]))), vec!["Running"]);
        let c = f.clone_box();
        assert_eq!(texts(&c.filter(toks(&["cats"]))), vec!["cat"]);
    }

    #[test]
    fn keyword_repeat_plain_and_with_stemmer() {
        let out = KeywordRepeatFilter::new().filter(toks(&["cats"]));
        assert_eq!(texts(&out), vec!["cats", "cats"]);
        assert_eq!(positions(&out), vec![0, 0]);
        let f = KeywordRepeatFilter::with_stemmer(Box::new(PorterStemFilter));
        let out = f.filter(toks(&["cats", "dog"]));
        assert_eq!(texts(&out), vec!["cats", "cat", "dog", "dog"]);
        assert_eq!(positions(&out), vec![0, 0, 1, 1]);
        let out = RemoveDuplicatesFilter.filter(out);
        assert_eq!(texts(&out), vec!["cats", "cat", "dog"]);
        let _ = f.clone_box();
    }

    #[test]
    fn limit_token_count() {
        let f = LimitTokenCountFilter { max_token_count: 2, consume_all: true };
        assert_eq!(texts(&f.filter(toks(&["a", "b", "c"]))), vec!["a", "b"]);
        assert_eq!(texts(&LimitTokenCountFilter::default().filter(toks(&["a", "b"]))), vec!["a"]);
    }

    #[test]
    fn multiplexer_branches() {
        let f = MultiplexerFilter::new(
            vec![vec![Box::new(LowerCaseFilter)], vec![Box::new(UpperCaseFilter)], vec![Box::new(PorterStemFilter)]],
            true,
        );
        let out = f.filter(toks(&["Running"]));
        assert_eq!(texts(&out), vec!["Running", "running", "RUNNING", "Run"]);
        assert_eq!(positions(&out), vec![0, 0, 0, 0]);
    }

    #[test]
    fn multiplexer_dedupes_and_no_original() {
        let f = MultiplexerFilter::new(vec![vec![Box::new(LowerCaseFilter)], vec![Box::new(LowerCaseFilter)]], false);
        let out = f.filter(toks(&["Cat", "dog"]));
        assert_eq!(texts(&out), vec!["cat", "dog"]);
        let c = f.clone_box();
        assert_eq!(texts(&c.filter(toks(&["X"]))), vec!["x"]);
    }

    #[test]
    fn remove_duplicates_same_position_only() {
        let out = RemoveDuplicatesFilter.filter(vec![tok("a", 0), tok("a", 0), tok("a", 1), tok("b", 1)]);
        assert_eq!(texts(&out), vec!["a", "a", "b"]);
        assert_eq!(positions(&out), vec![0, 1, 1]);
    }

    #[test]
    fn stemmer_override_basic_and_rules() {
        let f = StemmerOverrideFilter::from_rules(&["running, runs => run", "mice => mouse", "bad"]);
        assert_eq!(texts(&f.filter(toks(&["running", "runs", "mice", "cats"]))), vec!["run", "run", "mouse", "cats"]);
    }

    #[test]
    fn stemmer_override_wrap_and_ignore_case() {
        let mut rules = HashMap::new();
        rules.insert("skies".to_string(), "sky".to_string());
        let f = StemmerOverrideFilter::wrap(Box::new(PorterStemFilter), rules).ignore_case(true);
        assert_eq!(texts(&f.filter(toks(&["Skies", "cats"]))), vec!["sky", "cat"]);
        let c = f.clone_box();
        assert_eq!(texts(&c.filter(toks(&["skies"]))), vec!["sky"]);
    }
}
