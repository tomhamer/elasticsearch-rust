//! Text analysis: tokenizers, token filters, and prebuilt analyzers.
//!
//! This module mirrors Lucene's `analysis` package and Elasticsearch's
//! built-in analyzers. An [`Analyzer`] is a [`Tokenizer`] followed by a
//! chain of [`TokenFilter`]s.

pub mod charfilter;
pub mod filter;
pub mod stemmer;
pub mod stopwords;
pub mod tokenizer;

use filter::{AsciiFoldingFilter, LowerCaseFilter, StopFilter};
use stemmer::StemFilter;
use tokenizer::{
    KeywordTokenizer, LetterTokenizer, StandardTokenizer, WhitespaceTokenizer,
};

/// A single token produced by analysis.
///
/// `offset_from`/`offset_to` are byte offsets into the original input
/// string (half-open range). `position` is the token's position in the
/// analyzed stream; when a filter removes a token (e.g. a stop word),
/// later tokens keep their original positions, matching Lucene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// Position in the analyzed stream (0-based).
    pub position: u32,
    /// Byte offset of the first character of the token in the input.
    pub offset_from: u32,
    /// Byte offset just past the last character of the token in the input.
    pub offset_to: u32,
}

/// Splits raw text into a sequence of [`Token`]s.
pub trait Tokenizer: Send + Sync {
    /// Tokenize `text`, producing tokens with 0-based positions and
    /// byte offsets into `text`.
    fn tokenize(&self, text: &str) -> Vec<Token>;

    /// Create a new boxed instance of this tokenizer (for cloning an
    /// [`Analyzer`]).
    fn clone_box(&self) -> Box<dyn Tokenizer>;
}

/// Transforms the raw text before it reaches the tokenizer (HTML
/// stripping, character mapping, ...).
///
/// Char filters run in order, each on the previous one's output.
pub trait CharFilter: Send + Sync {
    /// Transform `text`.
    fn filter(&self, text: &str) -> String;

    /// Transform `text`, also returning Lucene `BaseCharFilter` offset
    /// corrections: `(output_offset, cumulative_diff)` pairs in byte
    /// offsets, ascending. An offset `o` of the output maps back to
    /// `o + diff` of the last pair whose `output_offset <= o` (unchanged
    /// when none applies). The default records no corrections.
    fn filter_mapped(&self, text: &str) -> (String, Vec<(u32, i64)>) {
        (self.filter(text), Vec::new())
    }

    /// Create a new boxed instance of this filter (for cloning an
    /// [`Analyzer`]).
    fn clone_box(&self) -> Box<dyn CharFilter>;
}

/// Apply one char filter's offset-correction map (`BaseCharFilter.correct`).
fn correct_offset(map: &[(u32, i64)], off: u32) -> u32 {
    let i = map.partition_point(|&(o, _)| o <= off);
    if i == 0 {
        return off;
    }
    (off as i64 + map[i - 1].1).max(0) as u32
}

/// Transforms a sequence of tokens (lowercasing, stop removal, stemming, ...).
pub trait TokenFilter: Send + Sync {
    /// Apply the filter to `tokens`.
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token>;

    /// Create a new boxed instance of this filter (for cloning an
    /// [`Analyzer`]).
    fn clone_box(&self) -> Box<dyn TokenFilter>;
}

/// A text analyzer: one tokenizer followed by zero or more token filters.
///
/// Use the prebuilt constructors ([`Analyzer::standard`], [`Analyzer::english`],
/// ...) or [`Analyzer::by_name`] for the ES built-ins, or build a custom
/// pipeline with [`Analyzer::new`] and [`Analyzer::with_filter`].
pub struct Analyzer {
    char_filters: Vec<Box<dyn CharFilter>>,
    tokenizer: Box<dyn Tokenizer>,
    filters: Vec<Box<dyn TokenFilter>>,
    /// Set only by constructors whose pipeline has a fused streaming
    /// implementation; any builder call clears it.
    fast: FastPath,
}

/// Fused streaming implementations of known pipelines (identical output to
/// the generic tokenizer + filters, without a `Vec<Token>` of owned strings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastPath {
    None,
    /// `StandardTokenizer` + `LowerCaseFilter`.
    StandardLower,
}

impl Clone for Analyzer {
    fn clone(&self) -> Self {
        Self {
            char_filters: self.char_filters.iter().map(|f| f.clone_box()).collect(),
            tokenizer: self.tokenizer.clone_box(),
            filters: self.filters.iter().map(|f| f.clone_box()).collect(),
            fast: self.fast,
        }
    }
}

impl std::fmt::Debug for Analyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analyzer")
            .field("char_filters", &self.char_filters.len())
            .field("tokenizer", &"<dyn Tokenizer>")
            .field("filters", &self.filters.len())
            .finish()
    }
}

impl Analyzer {
    /// Create an analyzer with the given tokenizer and no filters.
    pub fn new(tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            char_filters: Vec::new(),
            tokenizer,
            filters: Vec::new(),
            fast: FastPath::None,
        }
    }

    /// Builder: append a char filter (applied before tokenization, after
    /// any char filters added earlier).
    pub fn with_char_filter(mut self, f: Box<dyn CharFilter>) -> Self {
        self.fast = FastPath::None;
        self.char_filters.push(f);
        self
    }

    /// Builder: append a token filter to the pipeline.
    pub fn with_filter(mut self, f: Box<dyn TokenFilter>) -> Self {
        self.fast = FastPath::None;
        self.filters.push(f);
        self
    }

    /// Analyze `text` into a token stream: char filters, then the
    /// tokenizer, then the token filters in order.
    pub fn analyze(&self, text: &str) -> Vec<Token> {
        let mut tokens = if self.char_filters.is_empty() {
            self.tokenizer.tokenize(text)
        } else {
            let (mut filtered, m0) = self.char_filters[0].filter_mapped(text);
            let mut maps = vec![m0];
            for cf in &self.char_filters[1..] {
                let (f, m) = cf.filter_mapped(&filtered);
                filtered = f;
                maps.push(m);
            }
            let mut toks = self.tokenizer.tokenize(&filtered);
            if maps.iter().any(|m| !m.is_empty()) {
                // Lucene chains `correctOffset` from the innermost filter outwards.
                for t in &mut toks {
                    for m in maps.iter().rev() {
                        t.offset_from = correct_offset(m, t.offset_from);
                        t.offset_to = correct_offset(m, t.offset_to);
                    }
                }
            }
            toks
        };
        for f in &self.filters {
            tokens = f.filter(tokens);
        }
        tokens
    }

    /// Stream the analyzed tokens of `text` without materializing them:
    /// `f(term, position)` is called per token, where `term` is `buf`'s
    /// prefix as passed in (e.g. a field id) followed by the token text.
    /// `buf` is truncated back to its prefix afterwards. Same tokens and
    /// positions as [`Analyzer::analyze`].
    pub fn for_each_token<F: FnMut(&[u8], u32)>(&self, text: &str, buf: &mut Vec<u8>, mut f: F) {
        let base = buf.len();
        match self.fast {
            FastPath::StandardLower => tokenizer::standard_lowercase_stream(text, buf, base, &mut f),
            FastPath::None => {
                for t in self.analyze(text) {
                    buf.truncate(base);
                    buf.extend_from_slice(t.text.as_bytes());
                    f(buf, t.position);
                }
            }
        }
        buf.truncate(base);
    }

    /// ES `standard` analyzer: `StandardTokenizer` + `LowerCaseFilter`.
    pub fn standard() -> Self {
        let mut a = Self::new(Box::new(StandardTokenizer)).with_filter(Box::new(LowerCaseFilter));
        a.fast = FastPath::StandardLower;
        a
    }

    /// ES `simple` analyzer: `LetterTokenizer` + `LowerCaseFilter`.
    pub fn simple() -> Self {
        Self::new(Box::new(LetterTokenizer))
            .with_filter(Box::new(LowerCaseFilter))
    }

    /// ES `whitespace` analyzer: `WhitespaceTokenizer`, no filters.
    pub fn whitespace() -> Self {
        Self::new(Box::new(WhitespaceTokenizer))
    }

    /// ES `keyword` analyzer: the whole input as a single token.
    pub fn keyword() -> Self {
        Self::new(Box::new(KeywordTokenizer))
    }

    /// ES `stop` analyzer: standard + English stop words.
    pub fn stop() -> Self {
        Self::standard().with_filter(Box::new(StopFilter::english()))
    }

    /// ES `english` analyzer: standard tokenizer, English possessive
    /// stripping, lowercase, English stop words, Porter stemmer.
    pub fn english() -> Self {
        Self::new(Box::new(StandardTokenizer))
            .with_filter(Box::new(filter::EnglishPossessiveFilter))
            .with_filter(Box::new(LowerCaseFilter))
            .with_filter(Box::new(StopFilter::english()))
            .with_filter(Box::new(StemFilter))
    }

    /// ES `fingerprint` analyzer: standard tokenizer, lowercase, ASCII
    /// folding, stop (ES default `_none_`, i.e. no stop words), and the
    /// fingerprint filter (sorted, de-duplicated, space-joined; max 255).
    pub fn fingerprint() -> Self {
        Self::new(Box::new(StandardTokenizer))
            .with_filter(Box::new(LowerCaseFilter))
            .with_filter(Box::new(AsciiFoldingFilter))
            .with_filter(Box::new(filter::FingerprintFilter::default()))
    }

    /// Build an ES language analyzer by name (`"french"`, `"cjk"`, ...).
    ///
    /// Each follows the ES "rebuilt" analyzer definition: standard
    /// tokenizer + language-specific normalization/elision + lowercase +
    /// the language's stop words + its stemmer.
    ///
    /// * `english` uses Porter (as ES does); `german` uses
    ///   `light_german` after `german_normalization` (as ES does).
    /// * Stem-exclusion (`keyword_marker`) is not configured since ES's
    ///   default exclusion set is empty.
    pub fn language(name: &str) -> Option<Self> {
        use filter::*;
        let stop = |lang: &str| -> Box<dyn TokenFilter> {
            Box::new(StopFilter::from_slice(stopwords::for_language(lang).unwrap_or(&[])))
        };
        let stem = |n: &str| -> Box<dyn TokenFilter> {
            Box::new(StemmerFilter::new(n).expect("built-in stemmer name"))
        };
        let std_lower = || Self::new(Box::new(StandardTokenizer)).with_filter(Box::new(LowerCaseFilter));
        let a = match name {
            "english" => Self::english(),
            "arabic" => std_lower()
                .with_filter(Box::new(DecimalDigitFilter))
                .with_filter(stop("arabic"))
                .with_filter(Box::new(ArabicNormalizationFilter))
                .with_filter(Box::new(ArabicStemFilter)),
            // Lucene 10 `PersianAnalyzer` (indices created on 9.x) stems;
            // `persian_v8` is the pre-Lucene-10 chain without stemming.
            "persian" | "persian_v8" => {
                let a = std_lower()
                    .with_char_filter(Box::new(charfilter::PersianCharFilter))
                    .with_filter(Box::new(DecimalDigitFilter))
                    .with_filter(Box::new(ArabicNormalizationFilter))
                    .with_filter(Box::new(PersianNormalizationFilter))
                    .with_filter(stop("persian"));
                if name == "persian" { a.with_filter(Box::new(PersianStemFilter)) } else { a }
            }
            "sorani" => std_lower()
                .with_filter(Box::new(SoraniNormalizationFilter))
                .with_filter(Box::new(DecimalDigitFilter))
                .with_filter(stop("sorani")),
            "bengali" => std_lower()
                .with_filter(Box::new(DecimalDigitFilter))
                .with_filter(Box::new(IndicNormalizationFilter))
                .with_filter(stop("bengali")),
            "hindi" => std_lower()
                .with_filter(Box::new(DecimalDigitFilter))
                .with_filter(Box::new(IndicNormalizationFilter))
                .with_filter(Box::new(HindiNormalizationFilter))
                .with_filter(stop("hindi")),
            "cjk" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(CjkWidthFilter))
                .with_filter(Box::new(LowerCaseFilter))
                .with_filter(Box::new(CjkBigramFilter::default()))
                .with_filter(stop("cjk")),
            "catalan" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(ElisionFilter::catalan()))
                .with_filter(Box::new(LowerCaseFilter))
                .with_filter(stop("catalan")),
            "french" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(ElisionFilter::french()))
                .with_filter(Box::new(LowerCaseFilter))
                .with_filter(stop("french"))
                .with_filter(stem("light_french")),
            "italian" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(ElisionFilter::italian()))
                .with_filter(Box::new(LowerCaseFilter))
                .with_filter(stop("italian"))
                .with_filter(stem("light_italian")),
            "irish" => Self::new(Box::new(StandardTokenizer))
                .with_filter(stop("irish"))
                .with_filter(Box::new(ElisionFilter::irish()))
                .with_filter(Box::new(IrishLowerCaseFilter))
                .with_filter(stop("irish")),
            "greek" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(GreekLowerCaseFilter))
                .with_filter(stop("greek")),
            "turkish" => Self::new(Box::new(StandardTokenizer))
                .with_filter(Box::new(ApostropheFilter))
                .with_filter(Box::new(TurkishLowerCaseFilter))
                .with_filter(stop("turkish"))
                .with_filter(stem("turkish")),
            "german" => std_lower()
                .with_filter(stop("german"))
                .with_filter(Box::new(GermanNormalizationFilter))
                .with_filter(stem("light_german")),
            "spanish" => std_lower().with_filter(stop("spanish")).with_filter(stem("light_spanish")),
            "portuguese" => std_lower().with_filter(stop("portuguese")).with_filter(stem("light_portuguese")),
            "danish" | "dutch" | "finnish" | "hungarian" | "norwegian" | "romanian" | "russian"
            | "swedish" => std_lower().with_filter(stop(name)).with_filter(stem(name)),
            "armenian" | "basque" | "brazilian" | "bulgarian" | "czech" | "estonian" | "galician"
            | "indonesian" | "latvian" | "lithuanian" => std_lower().with_filter(stop(name)),
            "thai" => std_lower().with_filter(Box::new(DecimalDigitFilter)).with_filter(stop("thai")),
            "serbian" => std_lower().with_filter(Box::new(SerbianNormalizationFilter)),
            _ => return None,
        };
        Some(a)
    }

    /// Every ES language analyzer name accepted by [`Analyzer::language`].
    pub const LANGUAGE_ANALYZERS: &'static [&'static str] = &[
        "arabic", "armenian", "basque", "bengali", "brazilian", "bulgarian", "catalan", "cjk",
        "czech", "danish", "dutch", "english", "estonian", "finnish", "french", "galician",
        "german", "greek", "hindi", "hungarian", "indonesian", "irish", "italian", "latvian",
        "lithuanian", "norwegian", "persian", "portuguese", "romanian", "russian", "serbian",
        "sorani", "spanish", "swedish", "turkish", "thai",
    ];

    /// Look up a built-in analyzer by name.
    ///
    /// Recognized names: `"standard"`, `"simple"`, `"whitespace"`,
    /// `"keyword"`, `"stop"`, `"fingerprint"`, and every ES language
    /// analyzer (see [`Analyzer::language`]). `"pattern"` is not
    /// supported (no regex dependency).
    pub fn by_name(name: &str) -> Option<Self> {
        Some(match name {
            "standard" => Self::standard(),
            "simple" => Self::simple(),
            "whitespace" => Self::whitespace(),
            "keyword" => Self::keyword(),
            "stop" => Self::stop(),
            "fingerprint" => Self::fingerprint(),
            // Tokens already analyzed by the caller (see `PreTokenizedTokenizer`).
            "_pretokenized" => Self::new(Box::new(tokenizer::PreTokenizedTokenizer)),
            other => return Self::language(other),
        })
    }

    /// Resolve an ES tokenizer name with default parameters.
    ///
    /// Supported: `standard`, `whitespace`, `letter`, `lowercase`,
    /// `keyword`, `ngram`/`nGram` (1–2), `edge_ngram`/`edgeNGram` (1–2),
    /// `uax_url_email`, `path_hierarchy`, `char_group` (whitespace),
    /// `classic`. Not available: `pattern`, `simple_pattern`,
    /// `simple_pattern_split`, `thai`.
    pub fn tokenizer_by_name(name: &str) -> Option<Box<dyn Tokenizer>> {
        use tokenizer::*;
        Some(match name {
            "standard" => Box::new(StandardTokenizer),
            "whitespace" => Box::new(WhitespaceTokenizer),
            "letter" => Box::new(LetterTokenizer),
            "lowercase" => Box::new(LowercaseTokenizer),
            "keyword" => Box::new(KeywordTokenizer),
            "ngram" | "nGram" => Box::new(NGramTokenizer { min: 1, max: 2 }),
            "edge_ngram" | "edgeNGram" => Box::new(EdgeNGramTokenizer { min: 1, max: 2 }),
            "uax_url_email" => Box::new(UaxUrlEmailTokenizer),
            "path_hierarchy" | "PathHierarchy" => Box::new(PathHierarchyTokenizer::default()),
            "char_group" => Box::new(CharGroupTokenizer { whitespace: true, ..Default::default() }),
            "classic" => Box::new(ClassicTokenizer),
            _ => return None,
        })
    }

    /// Resolve an ES char filter name with default parameters.
    /// Supported: `html_strip`. (`mapping` needs rules and
    /// `pattern_replace` needs regex, so neither resolves by name.)
    pub fn char_filter_by_name(name: &str) -> Option<Box<dyn CharFilter>> {
        match name {
            "html_strip" => Some(Box::new(charfilter::HtmlStripCharFilter::new())),
            _ => None,
        }
    }

    /// Resolve an ES token filter name with default parameters.
    ///
    /// Besides the generic filters (`lowercase`, `uppercase`, `stop`
    /// (English), `asciifolding`/`ascii_folding`, `trim`, `unique`,
    /// `shingle`, `porter_stem`, `snowball` (English), `stemmer`
    /// (English = Porter), `kstem`, `truncate` (10), `reverse`, `length`,
    /// `apostrophe`, `classic`, `cjk_bigram`, `cjk_width`,
    /// `decimal_digit`, `edge_ngram`, `ngram`, `elision` (French),
    /// `fingerprint`, `keyword_repeat`, `limit`, `remove_duplicates`,
    /// `word_delimiter`, `word_delimiter_graph`, `english_possessive`/
    /// `possessive_english`), the normalizers (`arabic_normalization`,
    /// `persian_normalization`, `sorani_normalization`,
    /// `hindi_normalization`, `indic_normalization`,
    /// `german_normalization`, `scandinavian_normalization`,
    /// `scandinavian_folding`, `serbian_normalization`), the language
    /// lowercasers (`greek_lowercase`, `turkish_lowercase`,
    /// `irish_lowercase`), `arabic_stem`, and `<lang>_stop` for every
    /// bundled stop list are accepted. Filters that need configuration
    /// (`synonym`, `keep`, `common_grams`, `dictionary_decompounder`,
    /// `multiplexer`, `stemmer_override`, `keyword_marker`) do not
    /// resolve by name.
    pub fn token_filter_by_name(name: &str) -> Option<Box<dyn TokenFilter>> {
        use filter::*;
        Some(match name {
            "lowercase" => Box::new(LowerCaseFilter),
            "uppercase" => Box::new(UpperCaseFilter),
            "stop" => Box::new(StopFilter::english()),
            "asciifolding" | "ascii_folding" => Box::new(AsciiFoldingFilter),
            "trim" => Box::new(TrimFilter),
            "unique" => Box::new(UniqueFilter),
            "shingle" => Box::new(ShingleFilter::es_default()),
            "porter_stem" => Box::new(PorterStemFilter),
            "snowball" => Box::new(SnowballFilter::new("english")?),
            "stemmer" => Box::new(StemmerFilter::new("english")?),
            "kstem" => Box::new(KStemFilter),
            "truncate" => Box::new(TruncateFilter { len: 10 }),
            "reverse" => Box::new(ReverseFilter),
            "length" => Box::new(LengthFilter { min: 0, max: usize::MAX }),
            "apostrophe" => Box::new(ApostropheFilter),
            "classic" => Box::new(ClassicFilter),
            "cjk_bigram" => Box::new(CjkBigramFilter::default()),
            "cjk_width" => Box::new(CjkWidthFilter),
            "decimal_digit" => Box::new(DecimalDigitFilter),
            "edge_ngram" | "edgeNGram" => Box::new(EdgeNGramTokenFilter::default()),
            "ngram" | "nGram" => Box::new(NGramTokenFilter::default()),
            "elision" => Box::new(ElisionFilter::french()),
            "fingerprint" => Box::new(FingerprintFilter::default()),
            "keyword_repeat" => Box::new(KeywordRepeatFilter::new()),
            "limit" => Box::new(LimitTokenCountFilter::default()),
            "remove_duplicates" => Box::new(RemoveDuplicatesFilter),
            "word_delimiter" | "word_delimiter_graph" => Box::new(WordDelimiterFilter::default()),
            "english_possessive" | "possessive_english" => Box::new(EnglishPossessiveFilter),
            "arabic_normalization" => Box::new(ArabicNormalizationFilter),
            "persian_normalization" => Box::new(PersianNormalizationFilter),
            "sorani_normalization" => Box::new(SoraniNormalizationFilter),
            "hindi_normalization" => Box::new(HindiNormalizationFilter),
            "indic_normalization" => Box::new(IndicNormalizationFilter),
            "german_normalization" => Box::new(GermanNormalizationFilter),
            "scandinavian_normalization" => Box::new(ScandinavianNormalizationFilter),
            "scandinavian_folding" => Box::new(ScandinavianFoldingFilter),
            "serbian_normalization" => Box::new(SerbianNormalizationFilter),
            "greek_lowercase" => Box::new(GreekLowerCaseFilter),
            "turkish_lowercase" => Box::new(TurkishLowerCaseFilter),
            "irish_lowercase" => Box::new(IrishLowerCaseFilter),
            "arabic_stem" => Box::new(ArabicStemFilter),
            "persian_stem" => Box::new(PersianStemFilter),
            other => {
                let lang = other.strip_suffix("_stop")?;
                Box::new(StopFilter::from_slice(stopwords::for_language(lang)?))
            }
        })
    }

    /// Build a custom analyzer from ES component names (all with default
    /// parameters), e.g.
    /// `Analyzer::custom("standard", &["html_strip"], &["lowercase", "porter_stem"])`.
    /// Returns `None` if any name is unknown.
    pub fn custom(tokenizer_name: &str, char_filters: &[&str], token_filters: &[&str]) -> Option<Self> {
        let mut a = Self::new(Self::tokenizer_by_name(tokenizer_name)?);
        for cf in char_filters {
            a = a.with_char_filter(Self::char_filter_by_name(cf)?);
        }
        for tf in token_filters {
            a = a.with_filter(Self::token_filter_by_name(tf)?);
        }
        Some(a)
    }

    /// An ES normalizer (for `keyword` fields): the keyword tokenizer
    /// followed by the named token filters (e.g. `["lowercase",
    /// "asciifolding"]`). Returns `None` if any filter name is unknown.
    pub fn normalizer(filters: &[&str]) -> Option<Self> {
        Self::custom("keyword", &[], filters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn texts(tokens: &[Token]) -> Vec<String> {
        tokens.iter().map(|t| t.text.clone()).collect()
    }

    #[test]
    fn standard_analyzer() {
        let a = Analyzer::standard();
        let tokens = a.analyze("Hello, World! It's 2026.");
        assert_eq!(
            texts(&tokens),
            vec!["hello", "world", "it's", "2026"]
        );
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn simple_analyzer() {
        let a = Analyzer::simple();
        let tokens = a.analyze("Hello, World! 2026");
        assert_eq!(texts(&tokens), vec!["hello", "world"]);
    }

    #[test]
    fn whitespace_analyzer() {
        let a = Analyzer::whitespace();
        let tokens = a.analyze("  a  b\tc\n");
        assert_eq!(texts(&tokens), vec!["a", "b", "c"]);
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn keyword_analyzer() {
        let a = Analyzer::keyword();
        let tokens = a.analyze("the whole phrase");
        assert_eq!(texts(&tokens), vec!["the whole phrase"]);
        assert_eq!(tokens[0].offset_from, 0);
        assert_eq!(tokens[0].offset_to, "the whole phrase".len() as u32);
    }

    #[test]
    fn keyword_analyzer_empty() {
        let a = Analyzer::keyword();
        assert!(a.analyze("").is_empty());
    }

    #[test]
    fn stop_analyzer() {
        let a = Analyzer::stop();
        let tokens = a.analyze("The quick fox is running");
        assert_eq!(texts(&tokens), vec!["quick", "fox", "running"]);
        // Positions preserved: quick=1, fox=2, running=4
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
    }

    #[test]
    fn english_analyzer_positions() {
        let a = Analyzer::english();
        let tokens = a.analyze("The quick brown foxes are jumping over the lazy dogs");
        assert_eq!(
            texts(&tokens),
            vec!["quick", "brown", "fox", "jump", "over", "lazi", "dog"]
        );
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![1, 2, 3, 5, 6, 8, 9]
        );
    }

    #[test]
    fn by_name_all() {
        for name in [
            "standard",
            "simple",
            "whitespace",
            "keyword",
            "stop",
            "english",
        ] {
            assert!(Analyzer::by_name(name).is_some(), "{name}");
        }
        assert!(Analyzer::by_name("nope").is_none());
    }

    #[test]
    fn by_name_matches_constructors() {
        let text = "The quick brown foxes are jumping over the lazy dogs";
        for name in [
            "standard",
            "simple",
            "whitespace",
            "keyword",
            "stop",
            "english",
        ] {
            let a = Analyzer::by_name(name).unwrap();
            let b = Analyzer::by_name(name).unwrap();
            assert_eq!(a.analyze(text), b.analyze(text), "{name}");
        }
    }

    #[test]
    fn custom_pipeline() {
        use filter::LengthFilter;
        use tokenizer::NGramTokenizer;
        let a = Analyzer::new(Box::new(NGramTokenizer { min: 2, max: 3 }))
            .with_filter(Box::new(LowerCaseFilter))
            .with_filter(Box::new(LengthFilter { min: 2, max: 3 }));
        let tokens = a.analyze("cat");
        assert_eq!(texts(&tokens), vec!["ca", "at", "cat"]);
    }

    #[test]
    fn analyzer_is_cloneable() {
        let a = Analyzer::english();
        let b = a.clone();
        let text = "The quick brown foxes are jumping over the lazy dogs";
        assert_eq!(a.analyze(text), b.analyze(text));
    }

    #[test]
    fn stop_filter_keeps_positions() {
        let f = StopFilter::from_slice(&["the", "a"]);
        let tokens = f.filter(vec![
            Token { text: "the".into(), position: 0, offset_from: 0, offset_to: 3 },
            Token { text: "cat".into(), position: 1, offset_from: 4, offset_to: 7 },
            Token { text: "a".into(), position: 2, offset_from: 8, offset_to: 9 },
            Token { text: "dog".into(), position: 3, offset_from: 10, offset_to: 13 },
        ]);
        assert_eq!(texts(&tokens), vec!["cat", "dog"]);
        assert_eq!(
            tokens.iter().map(|t| t.position).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn empty_stop_set_is_noop() {
        let f = StopFilter::from_slice(&[]);
        let tokens = vec![Token {
            text: "x".into(),
            position: 0,
            offset_from: 0,
            offset_to: 1,
        }];
        assert_eq!(f.filter(tokens.clone()), tokens);
    }

    #[test]
    fn stop_filter_english_set() {
        let f = StopFilter::english();
        let tokens = f.filter(vec![
            Token { text: "with".into(), position: 0, offset_from: 0, offset_to: 4 },
            Token { text: "cats".into(), position: 1, offset_from: 5, offset_to: 9 },
        ]);
        assert_eq!(texts(&tokens), vec!["cats"]);
    }

    #[test]
    fn stop_filter_uses_lowercase_set() {
        // Stop words are matched case-insensitively against the lowercase set.
        let f = StopFilter::english();
        let tokens = f.filter(vec![
            Token { text: "The".into(), position: 0, offset_from: 0, offset_to: 3 },
            Token { text: "cat".into(), position: 1, offset_from: 4, offset_to: 7 },
        ]);
        assert_eq!(texts(&tokens), vec!["cat"]);
    }

    #[test]
    fn stop_filter_english_matches_const() {
        let f = StopFilter::english();
        for w in stopwords::ENGLISH {
            let tokens = f.filter(vec![Token {
                text: w.to_string(),
                position: 0,
                offset_from: 0,
                offset_to: w.len() as u32,
            }]);
            assert!(tokens.is_empty(), "expected stop word: {w}");
        }
    }

    #[test]
    fn stop_filter_english_count() {
        // Lucene's English stop set has 33 words.
        assert_eq!(stopwords::ENGLISH.len(), 33);
        let set: HashSet<&str> = stopwords::ENGLISH.iter().copied().collect();
        assert_eq!(set.len(), 33, "stop words must be unique");
    }

    #[test]
    fn every_language_analyzer_resolves() {
        for name in Analyzer::LANGUAGE_ANALYZERS {
            let a = Analyzer::by_name(name).unwrap_or_else(|| panic!("{name}"));
            // Must not panic on mixed input.
            let _ = a.analyze("The Quick, brown 123 éèà ДОМ 中文 العربية");
        }
        assert_eq!(Analyzer::LANGUAGE_ANALYZERS.len(), 36);
        assert!(Analyzer::by_name("fingerprint").is_some());
        assert!(Analyzer::by_name("pattern").is_none());
    }

    #[test]
    fn english_analyzer_strips_possessive() {
        let a = Analyzer::english();
        assert_eq!(texts(&a.analyze("John's dogs")), vec!["john", "dog"]);
    }

    #[test]
    fn french_analyzer_elision_stop_stem() {
        let a = Analyzer::by_name("french").unwrap();
        let toks = a.analyze("L'avion et les chevaux");
        let t = texts(&toks);
        assert!(!t.contains(&"l'avion".to_string()));
        assert!(!t.contains(&"les".to_string()));
        assert!(!t.contains(&"et".to_string()));
        assert_eq!(t.len(), 2);
        assert!(t[0].starts_with("avion"));
    }

    #[test]
    fn german_analyzer_normalizes() {
        let a = Analyzer::by_name("german").unwrap();
        let t = texts(&a.analyze("Die Häuser"));
        assert_eq!(t.len(), 1);
        assert!(!t[0].contains('ä'));
    }

    #[test]
    fn russian_and_spanish_analyzers_stem() {
        let a = Analyzer::by_name("russian").unwrap();
        assert_eq!(texts(&a.analyze("и книги")), vec![stemmer::russian::stem("книги")]);
        let a = Analyzer::by_name("spanish").unwrap();
        let t = texts(&a.analyze("los gatos"));
        assert_eq!(t, vec![stemmer::light::spanish_light("gatos")]);
    }

    #[test]
    fn cjk_analyzer_bigrams() {
        let a = Analyzer::by_name("cjk").unwrap();
        assert_eq!(texts(&a.analyze("日本語 ＴＥＳＴ")), vec!["日本", "本語", "test"]);
    }

    #[test]
    fn turkish_analyzer_apostrophe_and_dotless_i() {
        let a = Analyzer::by_name("turkish").unwrap();
        let t = texts(&a.analyze("İstanbul'da"));
        assert_eq!(t.len(), 1);
        assert!(t[0].starts_with("istanbul"));
    }

    #[test]
    fn arabic_analyzer() {
        let a = Analyzer::by_name("arabic").unwrap();
        assert_eq!(texts(&a.analyze("الكتاب")), vec!["كتاب"]);
    }

    #[test]
    fn fingerprint_analyzer() {
        let a = Analyzer::fingerprint();
        assert_eq!(texts(&a.analyze("Yes yes, Gödel said this sentence is consistent and.")),
            vec!["and consistent godel is said sentence this yes"]);
    }

    #[test]
    fn custom_analyzer_by_names() {
        let a = Analyzer::custom("standard", &["html_strip"], &["lowercase", "porter_stem"]).unwrap();
        assert_eq!(texts(&a.analyze("<b>Running</b> DOGS")), vec!["run", "dog"]);
        let a = Analyzer::custom("whitespace", &[], &["word_delimiter_graph", "lowercase"]).unwrap();
        assert_eq!(texts(&a.analyze("PowerShot SD500")), vec!["power", "shot", "sd", "500"]);
        let a = Analyzer::custom("standard", &[], &["french_stop"]).unwrap();
        assert_eq!(texts(&a.analyze("le chat")), vec!["chat"]);
        assert!(Analyzer::custom("nope", &[], &[]).is_none());
        assert!(Analyzer::custom("standard", &["nope"], &[]).is_none());
        assert!(Analyzer::custom("standard", &[], &["nope"]).is_none());
        assert!(Analyzer::custom("standard", &[], &["nope_stop"]).is_none());
    }

    #[test]
    fn all_named_components_resolve() {
        for t in ["standard", "whitespace", "letter", "lowercase", "keyword", "ngram", "edge_ngram",
            "uax_url_email", "path_hierarchy", "char_group", "classic"] {
            assert!(Analyzer::tokenizer_by_name(t).is_some(), "{t}");
        }
        for f in ["lowercase", "uppercase", "stop", "asciifolding", "trim", "unique", "shingle",
            "porter_stem", "snowball", "stemmer", "kstem", "truncate", "reverse", "length",
            "apostrophe", "classic", "cjk_bigram", "cjk_width", "decimal_digit", "edge_ngram",
            "ngram", "elision", "fingerprint", "keyword_repeat", "limit", "remove_duplicates",
            "word_delimiter", "word_delimiter_graph", "arabic_normalization",
            "persian_normalization", "sorani_normalization", "hindi_normalization",
            "indic_normalization", "german_normalization", "scandinavian_normalization",
            "scandinavian_folding", "serbian_normalization", "greek_lowercase",
            "turkish_lowercase", "irish_lowercase", "arabic_stem", "english_possessive"] {
            let filt = Analyzer::token_filter_by_name(f).unwrap_or_else(|| panic!("{f}"));
            let _ = filt.filter(vec![Token { text: "Test".into(), position: 0, offset_from: 0, offset_to: 4 }]);
        }
    }

    #[test]
    fn normalizer_keyword_field() {
        let n = Analyzer::normalizer(&["lowercase", "asciifolding"]).unwrap();
        assert_eq!(texts(&n.analyze("Crème Brûlée")), vec!["creme brulee"]);
        assert!(Analyzer::normalizer(&["synonym"]).is_none());
    }

    #[test]
    fn char_filters_run_in_order() {
        use charfilter::MappingCharFilter;
        let a = Analyzer::keyword()
            .with_char_filter(Box::new(MappingCharFilter::new(&[("a", "b")])))
            .with_char_filter(Box::new(MappingCharFilter::new(&[("b", "c")])));
        assert_eq!(texts(&a.analyze("a")), vec!["c"]);
        let b = a.clone();
        assert_eq!(texts(&b.analyze("ab")), vec!["cc"]);
    }

    #[test]
    fn streaming_matches_analyze() {
        let inputs = [
            "Hello, World! It's 2026.",
            "  leading and trailing  ",
            "ÀÉÎ ÕÜ straße ΣΑΣ Οδυσσεύς don't 中文字 abc中def ДОМ x'y '' ",
            "MiXeD-case_tokens123 and\ttabs\nnewlines",
            "",
            "日本語 ＴＥＳＴ İstanbul",
        ];
        let analyzers = [Analyzer::standard(), Analyzer::english(), Analyzer::whitespace(), Analyzer::keyword()];
        for a in &analyzers {
            for text in inputs {
                let want: Vec<(Vec<u8>, u32)> = a
                    .analyze(text)
                    .into_iter()
                    .map(|t| {
                        let mut k = vec![7u8, 0];
                        k.extend_from_slice(t.text.as_bytes());
                        (k, t.position)
                    })
                    .collect();
                let mut got = Vec::new();
                let mut buf = vec![7u8, 0];
                a.for_each_token(text, &mut buf, |k, p| got.push((k.to_vec(), p)));
                assert_eq!(buf, vec![7u8, 0]);
                assert_eq!(got, want, "{text:?}");
            }
        }
        // Random ASCII (and some UTF-8) against the generic pipeline,
        // covering 64-byte chunk boundaries and words at the text's end.
        let alphabet: &[&str] = &["a", "Z", "q", "0", "9", "'", " ", ",", "-", "\n", "_", "`", "@", "[", "{", "é", "中"];
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let a = Analyzer::standard();
        for len in 0..300usize {
            for ascii_only in [true, false] {
                let mut text = String::new();
                for _ in 0..len {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let k = (x % if ascii_only { 15 } else { 17 } as u64) as usize;
                    text.push_str(alphabet[k]);
                }
                let want: Vec<(String, u32)> = a.analyze(&text).into_iter().map(|t| (t.text, t.position)).collect();
                let mut got = Vec::new();
                let mut buf = Vec::new();
                a.for_each_token(&text, &mut buf, |k, p| got.push((String::from_utf8(k.to_vec()).unwrap(), p)));
                assert_eq!(got, want, "{text:?}");
            }
        }
        assert_eq!(Analyzer::standard().fast, FastPath::StandardLower);
        assert_eq!(Analyzer::stop().fast, FastPath::None);
        assert_eq!(Analyzer::standard().clone().fast, FastPath::StandardLower);
    }
}
