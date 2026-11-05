//! Stemming token filters: [`SnowballFilter`], [`StemmerFilter`] (ES
//! `stemmer` names), [`KStemFilter`], plus the small Lucene stemmers
//! used by language analyzers ([`ArabicStemFilter`],
//! [`EnglishPossessiveFilter`]).

use crate::analysis::stemmer;
use crate::analysis::{Token, TokenFilter};

/// A stemming function over a single (lowercased) word.
pub type StemFn = fn(&str) -> String;

fn apply(tokens: Vec<Token>, f: StemFn) -> Vec<Token> {
    tokens
        .into_iter()
        .map(|mut t| {
            t.text = f(&t.text);
            t
        })
        .collect()
}

/// Snowball `german2`: the German algorithm with `ae`/`oe`/`ue`
/// first rewritten to `ä`/`ö`/`ü` (except `ue` after `q`).
pub fn german2_stem(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut s = String::with_capacity(word.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if i + 1 < chars.len() && chars[i + 1] == 'e' {
            let repl = match c {
                'a' => Some('ä'),
                'o' => Some('ö'),
                'u' if i == 0 || chars[i - 1] != 'q' => Some('ü'),
                _ => None,
            };
            if let Some(r) = repl {
                s.push(r);
                i += 2;
                continue;
            }
        }
        s.push(c);
        i += 1;
    }
    stemmer::german::stem(&s)
}

/// Resolve a Snowball language name (case-insensitive, as accepted by
/// ES's `snowball` filter) to its stemmer.
pub fn snowball_fn(language: &str) -> Option<StemFn> {
    Some(match language.to_ascii_lowercase().as_str() {
        "english" => stemmer::english::stem,
        "porter" => stemmer::snowball_porter_stem,
        // Lucene 10 folded `german2` (ae/oe/ue → ä/ö/ü) into `german`.
        "german" | "german2" => german2_stem,
        "french" => stemmer::french::stem,
        "spanish" => stemmer::spanish::stem,
        "italian" => stemmer::italian::stem,
        "portuguese" => stemmer::portuguese::stem,
        "dutch" => stemmer::dutch::stem,
        "swedish" => stemmer::swedish::stem,
        "norwegian" => stemmer::norwegian::stem,
        "danish" => stemmer::danish::stem,
        "finnish" => stemmer::finnish::stem,
        "russian" => stemmer::russian::stem,
        "hungarian" => stemmer::hungarian::stem,
        "romanian" => stemmer::romanian::stem,
        "turkish" => stemmer::turkish::stem,
        _ => return None,
    })
}

/// Resolve an ES `stemmer` filter `language`/`name` to a stemming
/// function.
///
/// Mapping (follows ES's `StemmerTokenFilterFactory`):
///
/// | name | implementation |
/// |---|---|
/// | `english`, `porter` | Porter (1980) |
/// | `porter2` | Snowball English |
/// | `light_english`, `kstem` | KStem (port of Lucene `KStemmer` + `KStemData`, see [`stemmer::kstem`]) |
/// | `minimal_english` | Lucene `EnglishMinimalStemmer` |
/// | `possessive_english` | strip trailing `'s` |
/// | `german`, `german2`, `french`, `spanish`, `italian`, `portuguese`, `dutch`, `swedish`, `norwegian`, `danish`, `finnish`, `russian`, `hungarian`, `romanian`, `turkish` | Snowball |
/// | `light_*` / `minimal_*` | Lucene light/minimal stemmers ([`stemmer::light`]); `light_nynorsk` / `minimal_nynorsk` use the `NYNORSK` flag, `light_norwegian` / `minimal_norwegian` the `BOKMAAL` flag |
///
/// ES's camelCase spellings (`lightGerman`, `minimalNynorsk`, ...) and the
/// `light_finish` misspelling are accepted for the light/minimal names.
/// Aliases: `dutch_kp` / `kp` → Snowball Dutch, `light_portuguese`'s
/// Brazilian sibling `brazilian` → `light_portuguese`, `arabic` →
/// [`arabic_stem`].
pub fn stemmer_fn(name: &str) -> Option<StemFn> {
    use stemmer::light;
    Some(match name.to_ascii_lowercase().as_str() {
        "english" | "porter" => stemmer::porter_stem,
        "porter2" => stemmer::english::stem,
        "light_english" | "kstem" => stemmer::kstem::stem,
        "minimal_english" | "minimalenglish" => light::english_minimal,
        "possessive_english" | "possessiveenglish" => strip_possessive,
        "light_german" | "lightgerman" => light::german_light,
        "minimal_german" | "minimalgerman" => light::german_minimal,
        "light_french" | "lightfrench" => light::french_light,
        "minimal_french" | "minimalfrench" => light::french_minimal,
        "light_spanish" | "lightspanish" => light::spanish_light,
        "light_italian" | "lightitalian" => light::italian_light,
        "light_portuguese" | "lightportuguese" | "brazilian" => light::portuguese_light,
        "minimal_portuguese" | "minimalportuguese" => light::portuguese_minimal,
        "light_swedish" | "lightswedish" => light::swedish_light,
        "light_norwegian" | "lightnorwegian" => light::norwegian_light,
        "minimal_norwegian" | "minimalnorwegian" => light::norwegian_minimal,
        "light_nynorsk" | "lightnynorsk" => light::nynorsk_light,
        "minimal_nynorsk" | "minimalnynorsk" => light::nynorsk_minimal,
        "light_finnish" | "lightfinnish" | "light_finish" | "lightfinish" => light::finnish_light,
        "light_russian" | "lightrussian" => light::russian_light,
        "light_hungarian" | "lighthungarian" => light::hungarian_light,
        "dutch_kp" | "kp" => stemmer::dutch::stem,
        "arabic" => arabic_stem,
        other => return snowball_fn(other),
    })
}

/// Snowball stemmer filter (ES `snowball`, `language` default English).
#[derive(Clone, Debug)]
pub struct SnowballFilter {
    language: String,
    f: StemFn,
}

impl SnowballFilter {
    /// Create for a Snowball language name; `None` if unsupported.
    pub fn new(language: &str) -> Option<Self> {
        snowball_fn(language).map(|f| Self { language: language.to_string(), f })
    }

    /// The configured language.
    pub fn language(&self) -> &str {
        &self.language
    }
}

impl TokenFilter for SnowballFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, self.f)
    }
}

/// ES `stemmer` filter, selected by name (see [`stemmer_fn`]).
#[derive(Clone, Debug)]
pub struct StemmerFilter {
    name: String,
    f: StemFn,
}

impl StemmerFilter {
    /// Create for an ES stemmer name; `None` if unsupported.
    pub fn new(name: &str) -> Option<Self> {
        stemmer_fn(name).map(|f| Self { name: name.to_string(), f })
    }

    /// The configured stemmer name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl TokenFilter for StemmerFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, self.f)
    }
}

/// ES `kstem` filter (Lucene `KStemFilter`, see [`stemmer::kstem`]).
#[derive(Clone, Debug, Default)]
pub struct KStemFilter;

impl TokenFilter for KStemFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, stemmer::kstem::stem)
    }
}

/// Strip a trailing English possessive `'s` / `’s` (and a bare trailing
/// apostrophe after `s`, as in `dogs'`), like Lucene's
/// `EnglishPossessiveFilter`.
pub fn strip_possessive(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let apo = |c: char| c == '\'' || c == '\u{2019}' || c == '\u{FF07}';
    if n >= 2 && apo(chars[n - 2]) && matches!(chars[n - 1], 's' | 'S') {
        return chars[..n - 2].iter().collect();
    }
    word.to_string()
}

/// Lucene `EnglishPossessiveFilter` (first filter of the ES `english`
/// analyzer).
#[derive(Clone, Debug, Default)]
pub struct EnglishPossessiveFilter;

impl TokenFilter for EnglishPossessiveFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, strip_possessive)
    }
}

/// Port of Lucene's `ArabicStemmer` (light stemming, Larkey et al.):
/// strip one of the prefixes `ال وال بال كال فال لل و` (the single-letter
/// `و` only from words of 4+ letters), then strip each of the suffixes
/// `ها ان ات ون ين يه ية ه ة ي` in turn, always keeping at least two
/// letters before an affix.
pub fn arabic_stem(word: &str) -> String {
    const PREFIXES: [&str; 7] = ["ال", "وال", "بال", "كال", "فال", "لل", "و"];
    const SUFFIXES: [&str; 10] = ["ها", "ان", "ات", "ون", "ين", "يه", "ية", "ه", "ة", "ي"];
    let mut s: Vec<char> = word.chars().collect();
    for p in PREFIXES {
        let pc: Vec<char> = p.chars().collect();
        if pc.len() == 1 && s.len() < 4 {
            continue;
        }
        if s.len() >= pc.len() + 2 && s.starts_with(&pc) {
            s.drain(..pc.len());
            break;
        }
    }
    for suf in SUFFIXES {
        let sc: Vec<char> = suf.chars().collect();
        if s.len() >= sc.len() + 2 && s.ends_with(&sc) {
            s.truncate(s.len() - sc.len());
        }
    }
    s.into_iter().collect()
}

/// Lucene `ArabicStemFilter` (see [`arabic_stem`]).
#[derive(Clone, Debug, Default)]
pub struct ArabicStemFilter;

impl TokenFilter for ArabicStemFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, arabic_stem)
    }
}

/// Port of Lucene's `PersianStemmer`: strip each of the suffixes
/// `ات ان ترين تر يي ي ها ZWNJ` in turn, keeping at least two letters.
pub fn persian_stem(word: &str) -> String {
    const SUFFIXES: [&str; 8] = [
        "\u{0627}\u{062A}", "\u{0627}\u{0646}", "\u{062A}\u{0631}\u{064A}\u{0646}", "\u{062A}\u{0631}",
        "\u{064A}\u{064A}", "\u{064A}", "\u{0647}\u{0627}", "\u{200C}",
    ];
    let mut s: &str = word;
    for suf in SUFFIXES {
        if s.ends_with(suf) && s.chars().count() >= suf.chars().count() + 2 {
            s = &s[..s.len() - suf.len()];
        }
    }
    s.to_string()
}

/// Lucene `PersianStemFilter` (see [`persian_stem`]).
#[derive(Clone, Debug, Default)]
pub struct PersianStemFilter;

impl TokenFilter for PersianStemFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        apply(tokens, persian_stem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> Vec<Token> {
        vec![Token { text: text.into(), position: 4, offset_from: 0, offset_to: text.len() as u32 }]
    }

    #[test]
    fn persian_suffixes() {
        assert_eq!(persian_stem("\u{0643}\u{062A}\u{0627}\u{0628}\u{0647}\u{0627}"), "\u{0643}\u{062A}\u{0627}\u{0628}");
        assert_eq!(persian_stem("\u{0628}\u{0647}\u{062A}\u{0631}\u{064A}\u{0646}"), "\u{0628}\u{0647}");
        assert_eq!(persian_stem("\u{0627}\u{0627}\u{062A}"), "\u{0627}\u{0627}\u{062A}");
    }

    #[test]
    fn snowball_languages_resolve() {
        for l in [
            "English", "porter", "german", "German2", "french", "spanish", "italian", "portuguese", "dutch",
            "swedish", "norwegian", "danish", "finnish", "russian", "hungarian", "romanian", "turkish",
        ] {
            assert!(SnowballFilter::new(l).is_some(), "{l}");
        }
        assert!(SnowballFilter::new("klingon").is_none());
    }

    #[test]
    fn snowball_filter_stems_and_keeps_position() {
        let f = SnowballFilter::new("english").unwrap();
        assert_eq!(f.language(), "english");
        let out = f.filter(tok("generously"));
        assert_eq!(out[0].text, "generous");
        assert_eq!(out[0].position, 4);
        let f = SnowballFilter::new("russian").unwrap();
        assert_eq!(f.filter(tok("книги"))[0].text, "книг");
    }

    #[test]
    fn german2_rewrites_digraphs() {
        assert_eq!(german2_stem("haeuser"), stemmer::german::stem("häuser"));
        assert_eq!(german2_stem("schoen"), stemmer::german::stem("schön"));
        // "ue" after q is kept
        assert_eq!(german2_stem("quelle"), stemmer::german::stem("quelle"));
    }

    #[test]
    fn all_es_stemmer_names_resolve() {
        for n in [
            "english", "light_english", "minimal_english", "porter2", "possessive_english", "german", "german2",
            "light_german", "minimal_german", "french", "light_french", "minimal_french", "spanish",
            "light_spanish", "italian", "light_italian", "portuguese", "light_portuguese", "minimal_portuguese",
            "dutch", "swedish", "light_swedish", "norwegian", "light_norwegian", "minimal_norwegian", "danish",
            "finnish", "light_finnish", "russian", "light_russian", "hungarian", "light_hungarian", "romanian",
            "turkish", "kstem", "porter", "arabic", "light_nynorsk", "minimal_nynorsk",
        ] {
            let f = StemmerFilter::new(n);
            assert!(f.is_some(), "{n}");
            assert_eq!(f.unwrap().name(), n);
        }
        assert!(StemmerFilter::new("lovins").is_none());
    }

    #[test]
    fn stemmer_english_is_porter() {
        let f = StemmerFilter::new("english").unwrap();
        assert_eq!(f.filter(tok("ponies"))[0].text, "poni");
        let f = StemmerFilter::new("porter2").unwrap();
        assert_eq!(f.filter(tok("generously"))[0].text, "generous");
    }

    #[test]
    fn possessive() {
        assert_eq!(strip_possessive("bob's"), "bob");
        assert_eq!(strip_possessive("BOB’S"), "BOB");
        assert_eq!(strip_possessive("its"), "its");
        assert_eq!(strip_possessive("s"), "s");
        assert_eq!(EnglishPossessiveFilter.filter(tok("dog's"))[0].text, "dog");
    }

    #[test]
    fn arabic_light_stem() {
        assert_eq!(arabic_stem("الكتاب"), "كتاب");
        assert_eq!(arabic_stem("والكتاب"), "كتاب");
        assert_eq!(arabic_stem("بالكتاب"), "كتاب");
        assert_eq!(arabic_stem("وكتب"), "كتب");
        assert_eq!(arabic_stem("وله"), "ول"); // too short for "و", but "ه" is stripped
        assert_eq!(arabic_stem("كتابات"), "كتاب");
        assert_eq!(arabic_stem("معلمون"), "معلم");
        assert_eq!(arabic_stem("مدرسة"), "مدرس");
        assert_eq!(arabic_stem("كتابها"), "كتاب");
        assert_eq!(ArabicStemFilter.filter(tok("الكتاب"))[0].text, "كتاب");
    }

    #[test]
    fn kstem_filter() {
        assert_eq!(KStemFilter.filter(tok("cats"))[0].text, "cat");
        let _ = KStemFilter.clone_box();
    }
}
