//! Language/script normalization filters (ports of Lucene's
//! `*Normalizer` character tables): Arabic, Persian, Sorani, Hindi, Indic,
//! German, Scandinavian (normalization and folding), Serbian, CJK width,
//! and decimal digits.
//!
//! Each filter rewrites token text only; positions and offsets are kept.

use crate::analysis::{Token, TokenFilter};

/// Apply a `String -> String` text transform to every token.
fn map_text(tokens: Vec<Token>, f: impl Fn(&str) -> String) -> Vec<Token> {
    tokens
        .into_iter()
        .map(|mut t| {
            t.text = f(&t.text);
            t
        })
        .collect()
}

/// Define a unit-struct token filter whose `filter` maps each token's
/// text through `$func`.
macro_rules! text_filter {
    ($(#[$meta:meta])* $name:ident, $func:path) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default)]
        pub struct $name;

        impl TokenFilter for $name {
            fn clone_box(&self) -> Box<dyn TokenFilter> {
                Box::new(self.clone())
            }

            fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
                map_text(tokens, $func)
            }
        }
    };
}

// ---------------------------------------------------------------------
// Arabic
// ---------------------------------------------------------------------

/// Normalize Arabic text per Lucene's `ArabicNormalizer`:
/// alef variants (آ أ إ) → ا, dotless yeh ى → ي, teh marbuta ة → ه, and
/// removal of tatweel ـ and the harakat (fathatan, dammatan, kasratan,
/// fatha, damma, kasra, shadda, sukun).
pub fn arabic_normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{0622}' | '\u{0623}' | '\u{0625}' => out.push('\u{0627}'),
            '\u{0649}' => out.push('\u{064A}'),
            '\u{0629}' => out.push('\u{0647}'),
            '\u{0640}' | '\u{064B}' | '\u{064C}' | '\u{064D}' | '\u{064E}' | '\u{064F}'
            | '\u{0650}' | '\u{0651}' | '\u{0652}' => {}
            _ => out.push(c),
        }
    }
    out
}

text_filter!(
    /// ES `arabic_normalization`: see [`arabic_normalize`].
    ArabicNormalizationFilter,
    arabic_normalize
);

// ---------------------------------------------------------------------
// Persian
// ---------------------------------------------------------------------

/// Normalize Persian text per Lucene's `PersianNormalizer`:
/// Farsi yeh ی and yeh barree ے → Arabic yeh ي, keheh ک → kaf ك,
/// heh-with-yeh-above ۀ and heh goal ہ → heh ه, hamza above ٔ removed.
pub fn persian_normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{06CC}' | '\u{06D2}' => out.push('\u{064A}'),
            '\u{06A9}' => out.push('\u{0643}'),
            '\u{06C0}' | '\u{06C1}' => out.push('\u{0647}'),
            '\u{0654}' => {}
            _ => out.push(c),
        }
    }
    out
}

text_filter!(
    /// ES `persian_normalization`: see [`persian_normalize`].
    PersianNormalizationFilter,
    persian_normalize
);

// ---------------------------------------------------------------------
// Sorani (Kurdish)
// ---------------------------------------------------------------------

/// Normalize Sorani Kurdish text per Lucene's `SoraniNormalizer`:
/// yeh variants ي ى → ی, kaf ك → keheh ک, heh doachashmee ھ → ه, rreh
/// with ring above ڒ → ڕ, teh marbuta ة → ە, final heh ه → ە, heh + ZWNJ
/// → ە, ZWNJ removed, and harakat/hamza-above removed.
pub fn sorani_normalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let n = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '\u{064A}' | '\u{0649}' => out.push('\u{06CC}'),
            '\u{0643}' => out.push('\u{06A9}'),
            '\u{06BE}' => out.push('\u{0647}'),
            '\u{0692}' => out.push('\u{0695}'),
            '\u{0629}' => out.push('\u{06D5}'),
            '\u{200C}' => {
                if let Some(last) = out.last_mut() {
                    if *last == '\u{0647}' {
                        *last = '\u{06D5}';
                    }
                }
            }
            '\u{0647}' if i == n - 1 => out.push('\u{06D5}'),
            '\u{0654}' | '\u{064B}' | '\u{064C}' | '\u{064D}' | '\u{064E}' | '\u{064F}'
            | '\u{0650}' | '\u{0651}' | '\u{0652}' => {}
            _ => out.push(c),
        }
    }
    out.into_iter().collect()
}

text_filter!(
    /// ES `sorani_normalization`: see [`sorani_normalize`].
    SoraniNormalizationFilter,
    sorani_normalize
);

// ---------------------------------------------------------------------
// Hindi
// ---------------------------------------------------------------------

/// Normalize Devanagari (Hindi) text per Lucene's `HindiNormalizer`:
/// dead-n (न + virama) → anusvara, candrabindu → anusvara, nukta and
/// nukta-forms folded to their base consonants, ZWJ/ZWNJ and virama
/// removed, candra/short vowel forms → standard forms, and long vowels
/// (independent and dependent) shortened.
pub fn hindi_normalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let mapped = match c {
            // dead n -> bindu
            '\u{0928}' if i + 1 < chars.len() && chars[i + 1] == '\u{094D}' => {
                out.push('\u{0902}');
                i += 2;
                continue;
            }
            '\u{0901}' => Some('\u{0902}'),
            // nukta deletions
            '\u{093C}' | '\u{200D}' | '\u{200C}' | '\u{094D}' => None,
            '\u{0929}' => Some('\u{0928}'),
            '\u{0931}' => Some('\u{0930}'),
            '\u{0934}' => Some('\u{0933}'),
            '\u{0958}' => Some('\u{0915}'),
            '\u{0959}' => Some('\u{0916}'),
            '\u{095A}' => Some('\u{0917}'),
            '\u{095B}' => Some('\u{091C}'),
            '\u{095C}' => Some('\u{0921}'),
            '\u{095D}' => Some('\u{0922}'),
            '\u{095E}' => Some('\u{092B}'),
            '\u{095F}' => Some('\u{092F}'),
            // chandra/short -> standard
            '\u{0945}' | '\u{0946}' => Some('\u{0947}'),
            '\u{0949}' | '\u{094A}' => Some('\u{094B}'),
            '\u{090D}' | '\u{090E}' => Some('\u{090F}'),
            '\u{0911}' | '\u{0912}' => Some('\u{0913}'),
            '\u{0972}' => Some('\u{0905}'),
            // long -> short independent vowels
            '\u{0906}' => Some('\u{0905}'),
            '\u{0908}' => Some('\u{0907}'),
            '\u{090A}' => Some('\u{0909}'),
            '\u{0960}' => Some('\u{090B}'),
            '\u{0961}' => Some('\u{090C}'),
            '\u{0910}' => Some('\u{090F}'),
            '\u{0914}' => Some('\u{0913}'),
            // long -> short dependent vowels
            '\u{0940}' => Some('\u{093F}'),
            '\u{0942}' => Some('\u{0941}'),
            '\u{0944}' => Some('\u{0943}'),
            '\u{0963}' => Some('\u{0962}'),
            '\u{0948}' => Some('\u{0947}'),
            '\u{094C}' => Some('\u{094B}'),
            other => Some(other),
        };
        if let Some(m) = mapped {
            out.push(m);
        }
        i += 1;
    }
    out.into_iter().collect()
}

text_filter!(
    /// ES `hindi_normalization`: see [`hindi_normalize`].
    HindiNormalizationFilter,
    hindi_normalize
);

// ---------------------------------------------------------------------
// Indic (script-generic)
// ---------------------------------------------------------------------

/// Normalize the Unicode representation of Indic-script text (Lucene's
/// `IndicNormalizer`): composes `consonant + nukta` sequences into their
/// precomposed forms and two-part dependent vowel signs into their single
/// code points, for Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
/// Telugu, Kannada and Malayalam, and removes ZWJ/ZWNJ.
pub fn indic_normalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    for c in chars {
        if c == '\u{200C}' || c == '\u{200D}' {
            continue;
        }
        if let Some(&prev) = out.last() {
            if let Some(composed) = indic_compose(prev, c) {
                *out.last_mut().unwrap() = composed;
                continue;
            }
        }
        out.push(c);
    }
    out.into_iter().collect()
}

/// Canonical two-code-point compositions for the major Indic scripts.
fn indic_compose(a: char, b: char) -> Option<char> {
    Some(match (a, b) {
        // Devanagari nukta forms
        ('\u{0928}', '\u{093C}') => '\u{0929}',
        ('\u{0930}', '\u{093C}') => '\u{0931}',
        ('\u{0933}', '\u{093C}') => '\u{0934}',
        ('\u{0915}', '\u{093C}') => '\u{0958}',
        ('\u{0916}', '\u{093C}') => '\u{0959}',
        ('\u{0917}', '\u{093C}') => '\u{095A}',
        ('\u{091C}', '\u{093C}') => '\u{095B}',
        ('\u{0921}', '\u{093C}') => '\u{095C}',
        ('\u{0922}', '\u{093C}') => '\u{095D}',
        ('\u{092B}', '\u{093C}') => '\u{095E}',
        ('\u{092F}', '\u{093C}') => '\u{095F}',
        // Devanagari two-part vowels
        ('\u{0905}', '\u{0946}') => '\u{090D}',
        ('\u{0905}', '\u{093E}') => '\u{0906}',
        ('\u{0906}', '\u{0945}') => '\u{0911}',
        ('\u{0906}', '\u{0946}') => '\u{0912}',
        ('\u{0906}', '\u{0947}') => '\u{0913}',
        ('\u{0906}', '\u{0948}') => '\u{0914}',
        ('\u{0905}', '\u{0947}') => '\u{0910}',
        ('\u{093E}', '\u{0945}') => '\u{0949}',
        ('\u{093E}', '\u{0946}') => '\u{094A}',
        ('\u{093E}', '\u{0947}') => '\u{094B}',
        ('\u{093E}', '\u{0948}') => '\u{094C}',
        // Bengali
        ('\u{09A1}', '\u{09BC}') => '\u{09DC}',
        ('\u{09A2}', '\u{09BC}') => '\u{09DD}',
        ('\u{09AF}', '\u{09BC}') => '\u{09DF}',
        ('\u{09C7}', '\u{09BE}') => '\u{09CB}',
        ('\u{09C7}', '\u{09D7}') => '\u{09CC}',
        // Gurmukhi
        ('\u{0A32}', '\u{0A3C}') => '\u{0A33}',
        ('\u{0A38}', '\u{0A3C}') => '\u{0A36}',
        ('\u{0A16}', '\u{0A3C}') => '\u{0A59}',
        ('\u{0A17}', '\u{0A3C}') => '\u{0A5A}',
        ('\u{0A1C}', '\u{0A3C}') => '\u{0A5B}',
        ('\u{0A2B}', '\u{0A3C}') => '\u{0A5E}',
        // Oriya
        ('\u{0B21}', '\u{0B3C}') => '\u{0B5C}',
        ('\u{0B22}', '\u{0B3C}') => '\u{0B5D}',
        ('\u{0B47}', '\u{0B56}') => '\u{0B48}',
        ('\u{0B47}', '\u{0B3E}') => '\u{0B4B}',
        ('\u{0B47}', '\u{0B57}') => '\u{0B4C}',
        // Tamil
        ('\u{0B92}', '\u{0BD7}') => '\u{0B94}',
        ('\u{0BC6}', '\u{0BBE}') => '\u{0BCA}',
        ('\u{0BC7}', '\u{0BBE}') => '\u{0BCB}',
        ('\u{0BC6}', '\u{0BD7}') => '\u{0BCC}',
        // Telugu
        ('\u{0C46}', '\u{0C56}') => '\u{0C48}',
        // Kannada
        ('\u{0CBF}', '\u{0CD5}') => '\u{0CC0}',
        ('\u{0CC6}', '\u{0CD5}') => '\u{0CC7}',
        ('\u{0CC6}', '\u{0CD6}') => '\u{0CC8}',
        ('\u{0CC6}', '\u{0CC2}') => '\u{0CCA}',
        ('\u{0CCA}', '\u{0CD5}') => '\u{0CCB}',
        // Malayalam
        ('\u{0D46}', '\u{0D3E}') => '\u{0D4A}',
        ('\u{0D47}', '\u{0D3E}') => '\u{0D4B}',
        ('\u{0D46}', '\u{0D57}') => '\u{0D4C}',
        _ => return None,
    })
}

text_filter!(
    /// ES `indic_normalization`: see [`indic_normalize`].
    IndicNormalizationFilter,
    indic_normalize
);

// ---------------------------------------------------------------------
// German
// ---------------------------------------------------------------------

/// Normalize German per Lucene's `GermanNormalizationFilter`:
/// ä/ö/ü → a/o/u, ß → ss, and the digraphs ae/oe/ue → a/o/u — except
/// that `ue` is kept after `q` or another vowel (`que`, `aue`), which is
/// implemented with the same small state machine as Lucene.
pub fn german_normalize(s: &str) -> String {
    #[derive(PartialEq, Clone, Copy)]
    enum State {
        N,
        V,
        U,
    }
    let mut out = String::with_capacity(s.len());
    let mut state = State::N;
    for c in s.chars() {
        match c {
            'a' | 'o' => {
                out.push(c);
                state = State::U;
            }
            'u' => {
                out.push(c);
                state = if state == State::N { State::U } else { State::V };
            }
            'e' => {
                if state != State::U {
                    out.push(c);
                }
                state = State::V;
            }
            'i' | 'q' | 'y' => {
                out.push(c);
                state = State::V;
            }
            'ä' => {
                out.push('a');
                state = State::V;
            }
            'ö' => {
                out.push('o');
                state = State::V;
            }
            'ü' => {
                out.push('u');
                state = State::V;
            }
            'ß' => {
                out.push_str("ss");
                state = State::N;
            }
            _ => {
                out.push(c);
                state = State::N;
            }
        }
    }
    out
}

text_filter!(
    /// ES `german_normalization`: see [`german_normalize`].
    GermanNormalizationFilter,
    german_normalize
);

// ---------------------------------------------------------------------
// Scandinavian
// ---------------------------------------------------------------------

/// Lucene's `ScandinavianNormalizationFilter`: normalizes the
/// interchangeable Scandinavian characters and their ASCII digraphs to
/// the Danish/Norwegian set: ä→æ, ö→ø, aa/ao→å, ae→æ, oe/oo→ø (and the
/// uppercase equivalents).
pub fn scandinavian_normalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            'ä' => out.push('æ'),
            'Ä' => out.push('Æ'),
            'ö' => out.push('ø'),
            'Ö' => out.push('Ø'),
            'a' | 'A' if matches!(next, Some('a' | 'A' | 'o' | 'O')) => {
                out.push(if c == 'a' { 'å' } else { 'Å' });
                i += 1;
            }
            'a' | 'A' if matches!(next, Some('e' | 'E')) => {
                out.push(if c == 'a' { 'æ' } else { 'Æ' });
                i += 1;
            }
            'o' | 'O' if matches!(next, Some('e' | 'E' | 'o' | 'O')) => {
                out.push(if c == 'o' { 'ø' } else { 'Ø' });
                i += 1;
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

text_filter!(
    /// ES `scandinavian_normalization`: see [`scandinavian_normalize`].
    ScandinavianNormalizationFilter,
    scandinavian_normalize
);

/// Lucene's `ScandinavianFoldingFilter`: folds å/ä/æ → a and ö/ø → o
/// (and uppercase), and collapses the digraphs aa/ae/ao → a and oe/oo → o.
pub fn scandinavian_fold(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            'å' | 'ä' | 'æ' => out.push('a'),
            'Å' | 'Ä' | 'Æ' => out.push('A'),
            'ö' | 'ø' => out.push('o'),
            'Ö' | 'Ø' => out.push('O'),
            'a' | 'A' if matches!(next, Some('a' | 'A' | 'o' | 'O' | 'e' | 'E')) => {
                out.push(c);
                i += 1;
            }
            'o' | 'O' if matches!(next, Some('e' | 'E' | 'o' | 'O')) => {
                out.push(c);
                i += 1;
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

text_filter!(
    /// ES `scandinavian_folding`: see [`scandinavian_fold`].
    ScandinavianFoldingFilter,
    scandinavian_fold
);

// ---------------------------------------------------------------------
// Serbian
// ---------------------------------------------------------------------

/// Lucene's `SerbianNormalizationFilter` ("bald" Latin): Serbian
/// Cyrillic → Latin without diacritics (ђ→dj, љ→lj, њ→nj, џ→dz, ћ→c,
/// ч→c, ш→s, ж→z, ...) and Latin diacritics folded (ć č → c, đ → dj,
/// š → s, ž → z).
pub fn serbian_normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let m: &str = match c {
            'а' => "a", 'б' => "b", 'в' => "v", 'г' => "g", 'д' => "d", 'ђ' => "dj",
            'е' => "e", 'ж' => "z", 'з' => "z", 'и' => "i", 'ј' => "j", 'к' => "k",
            'л' => "l", 'љ' => "lj", 'м' => "m", 'н' => "n", 'њ' => "nj", 'о' => "o",
            'п' => "p", 'р' => "r", 'с' => "s", 'т' => "t", 'ћ' => "c", 'у' => "u",
            'ф' => "f", 'х' => "h", 'ц' => "c", 'ч' => "c", 'џ' => "dz", 'ш' => "s",
            'ć' | 'č' => "c", 'đ' => "dj", 'š' => "s", 'ž' => "z",
            _ => {
                out.push(c);
                continue;
            }
        };
        out.push_str(m);
    }
    out
}

text_filter!(
    /// ES `serbian_normalization`: see [`serbian_normalize`].
    SerbianNormalizationFilter,
    serbian_normalize
);

/// Lucene's `SerbianNormalizationRegularFilter`: Serbian Cyrillic →
/// Latin *keeping* diacritics (ћ→ć, ч→č, ђ→đ, ш→š, ж→ž, љ→lj, њ→nj, џ→dž).
pub fn serbian_normalize_regular(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let m: &str = match c {
            'а' => "a", 'б' => "b", 'в' => "v", 'г' => "g", 'д' => "d", 'ђ' => "đ",
            'е' => "e", 'ж' => "ž", 'з' => "z", 'и' => "i", 'ј' => "j", 'к' => "k",
            'л' => "l", 'љ' => "lj", 'м' => "m", 'н' => "n", 'њ' => "nj", 'о' => "o",
            'п' => "p", 'р' => "r", 'с' => "s", 'т' => "t", 'ћ' => "ć", 'у' => "u",
            'ф' => "f", 'х' => "h", 'ц' => "c", 'ч' => "č", 'џ' => "dž", 'ш' => "š",
            _ => {
                out.push(c);
                continue;
            }
        };
        out.push_str(m);
    }
    out
}

text_filter!(
    /// See [`serbian_normalize_regular`].
    SerbianNormalizationRegularFilter,
    serbian_normalize_regular
);

// ---------------------------------------------------------------------
// CJK width
// ---------------------------------------------------------------------

/// Lucene's `CJKWidthFilter`: fullwidth ASCII variants (U+FF01–U+FF5E)
/// → ASCII, and halfwidth Katakana (U+FF65–U+FF9F) → fullwidth Katakana,
/// combining the halfwidth voiced (ﾞ) and semi-voiced (ﾟ) sound marks
/// with the preceding kana (ｶﾞ → ガ, ﾊﾟ → パ, ｳﾞ → ヴ).
pub fn cjk_width(s: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(s.chars().count());
    for c in s.chars() {
        let cp = c as u32;
        if (0xFF01..=0xFF5E).contains(&cp) {
            out.push(char::from_u32(cp - 0xFEE0).unwrap());
        } else if (0xFF65..=0xFF9F).contains(&cp) {
            match cp {
                0xFF9E => {
                    // voiced sound mark
                    if let Some(prev) = out.last_mut() {
                        if let Some(v) = voiced(*prev) {
                            *prev = v;
                            continue;
                        }
                    }
                    out.push('\u{309B}');
                }
                0xFF9F => {
                    if let Some(prev) = out.last_mut() {
                        if let Some(v) = semi_voiced(*prev) {
                            *prev = v;
                            continue;
                        }
                    }
                    out.push('\u{309C}');
                }
                _ => out.push(halfwidth_kana_to_fullwidth(cp)),
            }
        } else {
            out.push(c);
        }
    }
    out.into_iter().collect()
}

fn halfwidth_kana_to_fullwidth(cp: u32) -> char {
    const TABLE: [char; 59] = [
        '\u{30FB}', '\u{30F2}', '\u{30A1}', '\u{30A3}', '\u{30A5}', '\u{30A7}', '\u{30A9}',
        '\u{30E3}', '\u{30E5}', '\u{30E7}', '\u{30C3}', '\u{30FC}', '\u{30A2}', '\u{30A4}',
        '\u{30A6}', '\u{30A8}', '\u{30AA}', '\u{30AB}', '\u{30AD}', '\u{30AF}', '\u{30B1}',
        '\u{30B3}', '\u{30B5}', '\u{30B7}', '\u{30B9}', '\u{30BB}', '\u{30BD}', '\u{30BF}',
        '\u{30C1}', '\u{30C4}', '\u{30C6}', '\u{30C8}', '\u{30CA}', '\u{30CB}', '\u{30CC}',
        '\u{30CD}', '\u{30CE}', '\u{30CF}', '\u{30D2}', '\u{30D5}', '\u{30D8}', '\u{30DB}',
        '\u{30DE}', '\u{30DF}', '\u{30E0}', '\u{30E1}', '\u{30E2}', '\u{30E4}', '\u{30E6}',
        '\u{30E8}', '\u{30E9}', '\u{30EA}', '\u{30EB}', '\u{30EC}', '\u{30ED}', '\u{30EF}',
        '\u{30F3}', '\u{309B}', '\u{309C}',
    ];
    TABLE[(cp - 0xFF65) as usize]
}

/// Voiced form of a fullwidth kana, if it has one (カ→ガ, ... ウ→ヴ).
fn voiced(c: char) -> Option<char> {
    let cp = c as u32;
    match cp {
        // カ..コ, サ..ソ, タ..ト (odd/even alternation: base, voiced)
        0x30AB | 0x30AD | 0x30AF | 0x30B1 | 0x30B3 | 0x30B5 | 0x30B7 | 0x30B9 | 0x30BB
        | 0x30BD | 0x30BF | 0x30C1 | 0x30C4 | 0x30C6 | 0x30C8 => char::from_u32(cp + 1),
        // ハ..ホ
        0x30CF | 0x30D2 | 0x30D5 | 0x30D8 | 0x30DB => char::from_u32(cp + 1),
        0x30A6 => Some('\u{30F4}'),
        _ => None,
    }
}

/// Semi-voiced form (ハ→パ, ...).
fn semi_voiced(c: char) -> Option<char> {
    let cp = c as u32;
    match cp {
        0x30CF | 0x30D2 | 0x30D5 | 0x30D8 | 0x30DB => char::from_u32(cp + 2),
        _ => None,
    }
}

text_filter!(
    /// ES `cjk_width`: see [`cjk_width`].
    CjkWidthFilter,
    cjk_width
);

// ---------------------------------------------------------------------
// Decimal digits
// ---------------------------------------------------------------------

/// Zero code points of the Unicode `Nd` (decimal digit) blocks handled by
/// [`decimal_digit`].
const DIGIT_ZEROS: &[u32] = &[
    0x0030, 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66,
    0x0CE6, 0x0D66, 0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946,
    0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0,
    0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0,
    0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11C50, 0x11D50,
    0x11DA0, 0x16A60, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140,
    0x1E2F0, 0x1E950, 0x1FBF0,
];

/// Map a Unicode decimal digit (any `Nd` block) to its ASCII digit.
pub fn to_ascii_digit(c: char) -> Option<char> {
    let cp = c as u32;
    for &zero in DIGIT_ZEROS {
        if cp >= zero && cp < zero + 10 {
            return char::from_u32(0x30 + (cp - zero));
        }
    }
    None
}

/// Lucene's `DecimalDigitFilter`: folds Unicode decimal digits from any
/// script (Arabic-Indic ٣, Devanagari ३, fullwidth ３, ...) to ASCII 0–9.
pub fn decimal_digit(s: &str) -> String {
    s.chars().map(|c| to_ascii_digit(c).unwrap_or(c)).collect()
}

text_filter!(
    /// ES `decimal_digit`: see [`decimal_digit`].
    DecimalDigitFilter,
    decimal_digit
);

// ---------------------------------------------------------------------
// Language-specific lowercasing
// ---------------------------------------------------------------------

/// Lucene `GreekLowerCaseFilter`: lowercases, maps final sigma ς → σ,
/// and removes tonos/dialytika (ά→α, έ→ε, ή→η, ί/ϊ/ΐ→ι, ό→ο, ύ/ϋ/ΰ→υ, ώ→ω).
pub fn greek_lowercase(s: &str) -> String {
    s.chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'ς' => 'σ',
            'ά' => 'α',
            'έ' => 'ε',
            'ή' => 'η',
            'ί' | 'ϊ' | 'ΐ' => 'ι',
            'ό' => 'ο',
            'ύ' | 'ϋ' | 'ΰ' => 'υ',
            'ώ' => 'ω',
            'ϲ' => 'σ',
            other => other,
        })
        .collect()
}

text_filter!(
    /// ES `lowercase` with `language: greek`: see [`greek_lowercase`].
    GreekLowerCaseFilter,
    greek_lowercase
);

/// Lucene `TurkishLowerCaseFilter`: `I` → `ı` (dotless), `İ` → `i`, and
/// `I` + combining dot above (U+0307) → `i`; everything else lowercased
/// normally.
pub fn turkish_lowercase(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            'I' => {
                if chars.get(i + 1) == Some(&'\u{0307}') {
                    out.push('i');
                    i += 1;
                } else {
                    out.push('ı');
                }
            }
            'İ' => out.push('i'),
            c => out.extend(c.to_lowercase()),
        }
        i += 1;
    }
    out
}

text_filter!(
    /// ES `lowercase` with `language: turkish`: see [`turkish_lowercase`].
    TurkishLowerCaseFilter,
    turkish_lowercase
);

/// Lucene `IrishLowerCaseFilter`: an initial `n` or `t` followed by an
/// uppercase vowel gets a hyphen (`nAthair` → `n-athair`, `tUisce` →
/// `t-uisce`), then the word is lowercased.
pub fn irish_lowercase(s: &str) -> String {
    let mut it = s.chars();
    let first = it.next();
    let second = it.next();
    let prefix = match (first, second) {
        (Some(f @ ('n' | 't')), Some(v)) if "AEIOUÁÉÍÓÚ".contains(v) => Some(f),
        _ => None,
    };
    match prefix {
        Some(f) => {
            let mut out = String::with_capacity(s.len() + 1);
            out.push(f);
            out.push('-');
            out.push_str(&s[1..].to_lowercase());
            out
        }
        None => s.to_lowercase(),
    }
}

text_filter!(
    /// ES `lowercase` with `language: irish`: see [`irish_lowercase`].
    IrishLowerCaseFilter,
    irish_lowercase
);

#[cfg(test)]
mod lowercase_tests {
    use super::*;

    #[test]
    fn greek() {
        assert_eq!(greek_lowercase("ΑΘΗΝΑ"), "αθηνα");
        assert_eq!(greek_lowercase("λόγος"), "λογοσ");
        assert_eq!(greek_lowercase("Ελλάδα"), "ελλαδα");
        assert_eq!(greek_lowercase("ΐΰϊ"), "ιυι");
    }

    #[test]
    fn turkish() {
        assert_eq!(turkish_lowercase("ISTANBUL"), "ıstanbul");
        assert_eq!(turkish_lowercase("İSTANBUL"), "istanbul");
        assert_eq!(turkish_lowercase("I\u{0307}"), "i");
        assert_eq!(turkish_lowercase("Ağaç"), "ağaç");
    }

    #[test]
    fn irish() {
        assert_eq!(irish_lowercase("nAthair"), "n-athair");
        assert_eq!(irish_lowercase("tUisce"), "t-uisce");
        assert_eq!(irish_lowercase("Nathair"), "nathair");
        assert_eq!(irish_lowercase("n"), "n");
        assert_eq!(irish_lowercase(""), "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> Vec<Token> {
        vec![Token { text: text.to_string(), position: 3, offset_from: 1, offset_to: 2 }]
    }

    fn run(f: &dyn TokenFilter, text: &str) -> String {
        let out = f.filter(tok(text));
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].position, out[0].offset_from, out[0].offset_to), (3, 1, 2));
        out[0].text.clone()
    }

    #[test]
    fn arabic() {
        assert_eq!(arabic_normalize("أحمد"), "احمد");
        assert_eq!(arabic_normalize("إسلام"), "اسلام");
        assert_eq!(arabic_normalize("آمن"), "امن");
        assert_eq!(arabic_normalize("مكتبة"), "مكتبه");
        assert_eq!(arabic_normalize("مصطفى"), "مصطفي");
        assert_eq!(arabic_normalize("مـــدرسة"), "مدرسه");
        assert_eq!(arabic_normalize("مُحَمَّدٌ"), "محمد");
        assert_eq!(run(&ArabicNormalizationFilter, "أ"), "ا");
    }

    #[test]
    fn persian() {
        assert_eq!(persian_normalize("كتابی"), "كتابي");
        assert_eq!(persian_normalize("کتاب"), "كتاب");
        assert_eq!(persian_normalize("خانۀ"), "خانه");
        assert_eq!(persian_normalize("ہے"), "هي");
        assert_eq!(persian_normalize("مسٔله"), "مسله");
        assert_eq!(run(&PersianNormalizationFilter, "ک"), "ك");
    }

    #[test]
    fn sorani() {
        assert_eq!(sorani_normalize("ي"), "ی");
        assert_eq!(sorani_normalize("ى"), "ی");
        assert_eq!(sorani_normalize("ك"), "ک");
        assert_eq!(sorani_normalize("ھ"), "ه");
        assert_eq!(sorani_normalize("ڒ"), "ڕ");
        assert_eq!(sorani_normalize("ة"), "ە");
        assert_eq!(sorani_normalize("ماه"), "ماە");
        assert_eq!(sorani_normalize("ماه\u{200C}"), "ماە");
        assert_eq!(sorani_normalize("هەر"), "هەر");
        assert_eq!(sorani_normalize("بَ"), "ب");
        assert_eq!(run(&SoraniNormalizationFilter, "ك"), "ک");
    }

    #[test]
    fn hindi() {
        // long vowels -> short
        assert_eq!(hindi_normalize("आ"), "अ");
        assert_eq!(hindi_normalize("ई"), "इ");
        assert_eq!(hindi_normalize("की"), "कि");
        assert_eq!(hindi_normalize("कू"), "कु");
        // nukta deletion and nukta forms
        assert_eq!(hindi_normalize("क़"), "क");
        assert_eq!(hindi_normalize("क\u{093C}"), "क");
        assert_eq!(hindi_normalize("ज़"), "ज");
        // dead n -> anusvara, candrabindu -> anusvara
        assert_eq!(hindi_normalize("हिन\u{094D}दी"), "हिंदि");
        assert_eq!(hindi_normalize("चाँद"), "चांद");
        // chandra/short -> standard
        assert_eq!(hindi_normalize("ऑ"), "ओ");
        assert_eq!(hindi_normalize("कॉ"), "को");
        // ZWJ/ZWNJ removed
        assert_eq!(hindi_normalize("क\u{200D}\u{200C}"), "क");
        assert_eq!(run(&HindiNormalizationFilter, "आ"), "अ");
    }

    #[test]
    fn indic() {
        assert_eq!(indic_normalize("क\u{093C}"), "\u{0958}");
        assert_eq!(indic_normalize("\u{093E}\u{0947}"), "\u{094B}");
        assert_eq!(indic_normalize("\u{09C7}\u{09BE}"), "\u{09CB}");
        assert_eq!(indic_normalize("\u{0BC6}\u{0BBE}"), "\u{0BCA}");
        assert_eq!(indic_normalize("\u{0D46}\u{0D3E}"), "\u{0D4A}");
        assert_eq!(indic_normalize("क\u{200D}ष"), "कष");
        assert_eq!(indic_normalize("plain"), "plain");
        assert_eq!(run(&IndicNormalizationFilter, "x"), "x");
    }

    #[test]
    fn german() {
        assert_eq!(german_normalize("schön"), "schon");
        assert_eq!(german_normalize("schoen"), "schon");
        assert_eq!(german_normalize("häuser"), "hauser");
        assert_eq!(german_normalize("haeuser"), "hauser");
        assert_eq!(german_normalize("übel"), "ubel");
        assert_eq!(german_normalize("uebel"), "ubel");
        assert_eq!(german_normalize("straße"), "strasse");
        // 'ue' after q or a vowel is kept
        assert_eq!(german_normalize("quelle"), "quelle");
        assert_eq!(german_normalize("bauer"), "bauer");
        assert_eq!(german_normalize("neue"), "neue");
        assert_eq!(run(&GermanNormalizationFilter, "ä"), "a");
    }

    #[test]
    fn scandinavian_normalization() {
        assert_eq!(scandinavian_normalize("räksmörgås"), "ræksmørgås");
        assert_eq!(scandinavian_normalize("raeksmoergaas"), "ræksmørgås");
        assert_eq!(scandinavian_normalize("raksmorgas"), "raksmorgas");
        assert_eq!(scandinavian_normalize("blåbærsyltetøj"), "blåbærsyltetøj");
        assert_eq!(scandinavian_normalize("blaabaersyltetoej"), "blåbærsyltetøj");
        assert_eq!(scandinavian_normalize("ÄÖ Aa Oo"), "ÆØ Å Ø");
        assert_eq!(scandinavian_normalize("ao"), "å");
        assert_eq!(run(&ScandinavianNormalizationFilter, "ä"), "æ");
    }

    #[test]
    fn scandinavian_folding() {
        assert_eq!(scandinavian_fold("räksmörgås"), "raksmorgas");
        assert_eq!(scandinavian_fold("ræksmørgås"), "raksmorgas");
        assert_eq!(scandinavian_fold("raeksmoergaas"), "raksmorgas");
        assert_eq!(scandinavian_fold("blåbærsyltetøj"), "blabarsyltetoj");
        assert_eq!(scandinavian_fold("ÅÄÆÖØ"), "AAAOO");
        assert_eq!(scandinavian_fold("book"), "bok");
        assert_eq!(run(&ScandinavianFoldingFilter, "ö"), "o");
    }

    #[test]
    fn serbian() {
        assert_eq!(serbian_normalize("Ђорђе"), "Ђordje");
        assert_eq!(serbian_normalize("ђорђе"), "djordje");
        assert_eq!(serbian_normalize("љубав"), "ljubav");
        assert_eq!(serbian_normalize("његош"), "njegos");
        assert_eq!(serbian_normalize("џеп"), "dzep");
        assert_eq!(serbian_normalize("ћирилица"), "cirilica");
        assert_eq!(serbian_normalize("čćđšž"), "ccdjsz");
        assert_eq!(serbian_normalize("plain"), "plain");
        assert_eq!(serbian_normalize_regular("ђорђе"), "đorđe");
        assert_eq!(serbian_normalize_regular("његош"), "njegoš");
        assert_eq!(serbian_normalize_regular("џеп"), "džep");
        assert_eq!(serbian_normalize_regular("čćđšž"), "čćđšž");
        assert_eq!(run(&SerbianNormalizationFilter, "ш"), "s");
        assert_eq!(run(&SerbianNormalizationRegularFilter, "ш"), "š");
    }

    #[test]
    fn cjk_width_fullwidth_ascii() {
        assert_eq!(cjk_width("Ｔｅｓｔ　１２３"), "Test　123");
        assert_eq!(cjk_width("ＡＢＣ！？"), "ABC!?");
    }

    #[test]
    fn cjk_width_halfwidth_kana() {
        assert_eq!(cjk_width("ｶﾞｷﾞｸﾟ"), "ガギク゜");
        assert_eq!(cjk_width("ﾀﾞﾋﾟ"), "ダピ");
        assert_eq!(cjk_width("ｳﾞ"), "ヴ");
        assert_eq!(cjk_width("ｶﾅｷﾞ･ｰ"), "カナギ・ー");
        assert_eq!(cjk_width("ﾟ"), "゜");
        assert_eq!(run(&CjkWidthFilter, "ｱ"), "ア");
    }

    #[test]
    fn decimal_digits() {
        assert_eq!(decimal_digit("٠١٢٣٤٥٦٧٨٩"), "0123456789");
        assert_eq!(decimal_digit("۱۳۹۸"), "1398");
        assert_eq!(decimal_digit("१२३"), "123");
        assert_eq!(decimal_digit("１２３"), "123");
        assert_eq!(decimal_digit("๑๒๓"), "123");
        assert_eq!(decimal_digit("abc 42"), "abc 42");
        assert_eq!(to_ascii_digit('x'), None);
        assert_eq!(run(&DecimalDigitFilter, "٤"), "4");
    }
}
