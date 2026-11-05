//! Lucene's "light" and "minimal" stemmers: line-by-line ports of the Java
//! classes in `org.apache.lucene.analysis.<lang>` (Lucene `branch_10x`, the
//! line Elasticsearch ships; the logic is identical on `main`). Most are
//! Jacques Savoy's UniNE light stemmers.
//!
//! | function | Lucene class |
//! |---|---|
//! | [`english_minimal`] | `en.EnglishMinimalStemmer` |
//! | [`german_light`] | `de.GermanLightStemmer` |
//! | [`german_minimal`] | `de.GermanMinimalStemmer` |
//! | [`french_light`] | `fr.FrenchLightStemmer` |
//! | [`french_minimal`] | `fr.FrenchMinimalStemmer` |
//! | [`spanish_light`] | `es.SpanishLightStemmer` |
//! | [`spanish_minimal`] | `es.SpanishMinimalStemmer` |
//! | [`italian_light`] | `it.ItalianLightStemmer` |
//! | [`portuguese_light`] | `pt.PortugueseLightStemmer` |
//! | [`portuguese_minimal`] | `pt.PortugueseMinimalStemmer` (the "Plural" step of `portuguese.rslp`) |
//! | [`swedish_light`] | `sv.SwedishLightStemmer` |
//! | [`swedish_minimal`] | `sv.SwedishMinimalStemmer` |
//! | [`norwegian_light`] / [`nynorsk_light`] | `no.NorwegianLightStemmer` with `BOKMAAL` / `NYNORSK` |
//! | [`norwegian_minimal`] / [`nynorsk_minimal`] | `no.NorwegianMinimalStemmer` with `BOKMAAL` / `NYNORSK` |
//! | [`finnish_light`] | `fi.FinnishLightStemmer` |
//! | [`russian_light`] | `ru.RussianLightStemmer` |
//! | [`hungarian_light`] | `hu.HungarianLightStemmer` |
//!
//! Each Java `int stem(char[] s, int len)` becomes a private
//! `fn(&mut Vec<char>, usize) -> usize` with the same control flow; the
//! public wrappers convert to/from `String`. Lucene works on UTF-16 code
//! units and these ports on Unicode scalar values, which is identical for
//! BMP text (every suffix and letter involved is BMP). `Character.isLetter`
//! is mapped to [`char::is_alphabetic`] (they differ only on letter-numbers
//! and combining marks). Like Lucene, nothing is lowercased; the functions
//! never panic.

/// `StemmerUtil.endsWith(s, len, suffix)`.
fn ends(s: &[char], len: usize, suffix: &str) -> bool {
    let n = suffix.chars().count();
    n <= len && s[len - n..len].iter().copied().eq(suffix.chars())
}

fn ends_any(s: &[char], len: usize, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|x| ends(s, len, x))
}

/// `StemmerUtil.delete(s, pos, len)`: remove `s[pos]`, shifting the rest.
fn delete(s: &mut Vec<char>, pos: usize, len: usize) -> usize {
    s.remove(pos);
    len - 1
}

/// Run a Java-style `stem(char[], len)` over `word`.
fn run(word: &str, f: impl FnOnce(&mut Vec<char>, usize) -> usize) -> String {
    let mut s: Vec<char> = word.chars().collect();
    let len = s.len();
    let n = f(&mut s, len);
    s[..n].iter().collect()
}

/// Shared accent folding of the Spanish/Italian light stemmers.
fn fold_es_it(s: &mut [char], len: usize) {
    for c in s[..len].iter_mut() {
        *c = match *c {
            'à' | 'á' | 'â' | 'ä' => 'a',
            'ò' | 'ó' | 'ô' | 'ö' => 'o',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            other => other,
        };
    }
}

// ---------------------------------------------------------------------
// English
// ---------------------------------------------------------------------

/// Lucene `EnglishMinimalStemmer` (Harman's "S-stemmer").
pub fn english_minimal(word: &str) -> String {
    run(word, |s, len| {
        if len < 3 || s[len - 1] != 's' {
            return len;
        }
        match s[len - 2] {
            'u' | 's' => len,
            'e' => {
                if len > 3 && s[len - 3] == 'i' && s[len - 4] != 'a' && s[len - 4] != 'e' {
                    s[len - 3] = 'y';
                    return len - 2;
                }
                if matches!(s[len - 3], 'i' | 'a' | 'o' | 'e') {
                    return len;
                }
                len - 1
            }
            _ => len - 1,
        }
    })
}

// ---------------------------------------------------------------------
// German
// ---------------------------------------------------------------------

fn st_ending(c: char) -> bool {
    matches!(c, 'b' | 'd' | 'f' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 't')
}

fn german_light_step1(s: &[char], len: usize) -> usize {
    if len > 5 && s[len - 3] == 'e' && s[len - 2] == 'r' && s[len - 1] == 'n' {
        return len - 3;
    }
    if len > 4 && s[len - 2] == 'e' && matches!(s[len - 1], 'm' | 'n' | 'r' | 's') {
        return len - 2;
    }
    if len > 3 && s[len - 1] == 'e' {
        return len - 1;
    }
    if len > 3 && s[len - 1] == 's' && st_ending(s[len - 2]) {
        return len - 1;
    }
    len
}

fn german_light_step2(s: &[char], len: usize) -> usize {
    if len > 5 && s[len - 3] == 'e' && s[len - 2] == 's' && s[len - 1] == 't' {
        return len - 3;
    }
    if len > 4 && s[len - 2] == 'e' && (s[len - 1] == 'r' || s[len - 1] == 'n') {
        return len - 2;
    }
    if len > 4 && s[len - 2] == 's' && s[len - 1] == 't' && st_ending(s[len - 3]) {
        return len - 2;
    }
    len
}

/// Lucene `GermanLightStemmer`.
pub fn german_light(word: &str) -> String {
    run(word, |s, len| {
        for c in s[..len].iter_mut() {
            *c = match *c {
                'ä' | 'à' | 'á' | 'â' => 'a',
                'ö' | 'ò' | 'ó' | 'ô' => 'o',
                'ï' | 'ì' | 'í' | 'î' => 'i',
                'ü' | 'ù' | 'ú' | 'û' => 'u',
                other => other,
            };
        }
        let len = german_light_step1(s, len);
        german_light_step2(s, len)
    })
}

/// Lucene `GermanMinimalStemmer`.
pub fn german_minimal(word: &str) -> String {
    run(word, |s, len| {
        if len < 5 {
            return len;
        }
        for c in s[..len].iter_mut() {
            *c = match *c {
                'ä' => 'a',
                'ö' => 'o',
                'ü' => 'u',
                other => other,
            };
        }
        if len > 6 && s[len - 3] == 'n' && s[len - 2] == 'e' && s[len - 1] == 'n' {
            return len - 3;
        }
        if len > 5 {
            let hit = match s[len - 1] {
                'n' | 's' | 'r' => s[len - 2] == 'e',
                'e' => s[len - 2] == 's',
                _ => false,
            };
            if hit {
                return len - 2;
            }
        }
        match s[len - 1] {
            'n' | 'e' | 's' | 'r' => len - 1,
            _ => len,
        }
    })
}

// ---------------------------------------------------------------------
// French
// ---------------------------------------------------------------------

fn french_norm(s: &mut Vec<char>, mut len: usize) -> usize {
    if len > 4 {
        for c in s[..len].iter_mut() {
            *c = match *c {
                'à' | 'á' | 'â' => 'a',
                'ô' => 'o',
                'è' | 'é' | 'ê' => 'e',
                'ù' | 'û' => 'u',
                'î' => 'i',
                'ç' => 'c',
                other => other,
            };
        }
        let mut ch = s[0];
        let mut i = 1;
        while i < len {
            if s[i] == ch && ch.is_alphabetic() {
                // `len = delete(s, i--, len)`; the loop's `i++` cancels the `i--`.
                len = delete(s, i, len);
            } else {
                ch = s[i];
                i += 1;
            }
        }
    }
    if len > 4 && ends(s, len, "ie") {
        len -= 2;
    }
    if len > 4 {
        if s[len - 1] == 'r' {
            len -= 1;
        }
        if s[len - 1] == 'e' {
            len -= 1;
        }
        if s[len - 1] == 'e' {
            len -= 1;
        }
        if s[len - 1] == s[len - 2] && s[len - 1].is_alphabetic() {
            len -= 1;
        }
    }
    len
}

fn french_light_impl(s: &mut Vec<char>, mut len: usize) -> usize {
    if len > 5 && s[len - 1] == 'x' {
        if s[len - 3] == 'a' && s[len - 2] == 'u' && s[len - 4] != 'e' {
            s[len - 2] = 'l';
        }
        len -= 1;
    }
    if len > 3 && s[len - 1] == 'x' {
        len -= 1;
    }
    if len > 3 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 9 && ends(s, len, "issement") {
        len -= 6;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 8 && ends(s, len, "issant") {
        len -= 4;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends(s, len, "ement") {
        len -= 4;
        if len > 3 && ends(s, len, "ive") {
            len -= 1;
            s[len - 1] = 'f';
        }
        return french_norm(s, len);
    }
    if len > 11 && ends(s, len, "ficatrice") {
        len -= 5;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 10 && ends(s, len, "ficateur") {
        len -= 4;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 9 && ends(s, len, "catrice") {
        len -= 3;
        s[len - 4] = 'q';
        s[len - 3] = 'u';
        s[len - 2] = 'e';
        return french_norm(s, len);
    }
    if len > 8 && ends(s, len, "cateur") {
        len -= 2;
        s[len - 4] = 'q';
        s[len - 3] = 'u';
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 8 && ends(s, len, "atrice") {
        len -= 4;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 7 && ends(s, len, "ateur") {
        len -= 3;
        s[len - 2] = 'e';
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends(s, len, "trice") {
        len -= 1;
        s[len - 3] = 'e';
        s[len - 2] = 'u';
        s[len - 1] = 'r';
    }
    if len > 5 && ends(s, len, "ième") {
        return french_norm(s, len - 4);
    }
    if len > 7 && ends(s, len, "teuse") {
        len -= 2;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 6 && ends(s, len, "teur") {
        len -= 1;
        s[len - 1] = 'r';
        return french_norm(s, len);
    }
    if len > 5 && ends(s, len, "euse") {
        return french_norm(s, len - 2);
    }
    if len > 8 && ends(s, len, "ère") {
        len -= 1;
        s[len - 2] = 'e';
        return french_norm(s, len);
    }
    if len > 7 && ends(s, len, "ive") {
        len -= 1;
        s[len - 1] = 'f';
        return french_norm(s, len);
    }
    if len > 4 && (ends(s, len, "folle") || ends(s, len, "molle")) {
        len -= 2;
        s[len - 1] = 'u';
        return french_norm(s, len);
    }
    if len > 9 && ends(s, len, "nnelle") {
        return french_norm(s, len - 5);
    }
    if len > 9 && ends(s, len, "nnel") {
        return french_norm(s, len - 3);
    }
    if len > 4 && ends(s, len, "ète") {
        len -= 1;
        s[len - 2] = 'e';
    }
    if len > 8 && ends(s, len, "ique") {
        len -= 4;
    }
    if len > 8 && ends(s, len, "esse") {
        return french_norm(s, len - 3);
    }
    if len > 7 && ends(s, len, "inage") {
        return french_norm(s, len - 3);
    }
    if len > 9 && ends(s, len, "isation") {
        len -= 7;
        if len > 5 && ends(s, len, "ual") {
            s[len - 2] = 'e';
        }
        return french_norm(s, len);
    }
    if len > 9 && ends(s, len, "isateur") {
        return french_norm(s, len - 7);
    }
    if len > 8 && ends(s, len, "ation") {
        return french_norm(s, len - 5);
    }
    if len > 8 && ends(s, len, "ition") {
        return french_norm(s, len - 5);
    }
    french_norm(s, len)
}

/// Lucene `FrenchLightStemmer`.
pub fn french_light(word: &str) -> String {
    run(word, french_light_impl)
}

/// Lucene `FrenchMinimalStemmer`.
pub fn french_minimal(word: &str) -> String {
    run(word, |s, mut len| {
        if len < 6 {
            return len;
        }
        if s[len - 1] == 'x' {
            if s[len - 3] == 'a' && s[len - 2] == 'u' {
                s[len - 2] = 'l';
            }
            return len - 1;
        }
        if s[len - 1] == 's' {
            len -= 1;
        }
        if s[len - 1] == 'r' {
            len -= 1;
        }
        if s[len - 1] == 'e' {
            len -= 1;
        }
        if s[len - 1] == 'é' {
            len -= 1;
        }
        if s[len - 1] == s[len - 2] && s[len - 1].is_alphabetic() {
            len -= 1;
        }
        len
    })
}

// ---------------------------------------------------------------------
// Spanish
// ---------------------------------------------------------------------

/// Lucene `SpanishLightStemmer`.
pub fn spanish_light(word: &str) -> String {
    run(word, |s, len| {
        if len < 5 {
            return len;
        }
        fold_es_it(s, len);
        match s[len - 1] {
            'o' | 'a' | 'e' => len - 1,
            's' => {
                if s[len - 2] == 'e' && s[len - 3] == 's' && s[len - 4] == 'e' {
                    return len - 2;
                }
                if s[len - 2] == 'e' && s[len - 3] == 'c' {
                    s[len - 3] = 'z';
                    return len - 2;
                }
                if matches!(s[len - 2], 'o' | 'a' | 'e') {
                    return len - 2;
                }
                len
            }
            _ => len,
        }
    })
}

/// Lucene `SpanishMinimalStemmer` (plural removal only; not exposed by
/// ES's `stemmer` filter, provided for completeness).
pub fn spanish_minimal(word: &str) -> String {
    run(word, |s, len| {
        if len < 4 || s[len - 1] != 's' {
            return len;
        }
        for c in s[..len].iter_mut() {
            *c = match *c {
                'à' | 'á' | 'â' | 'ä' => 'a',
                'ò' | 'ó' | 'ô' | 'ö' => 'o',
                'è' | 'é' | 'ê' | 'ë' => 'e',
                'ù' | 'ú' | 'û' | 'ü' => 'u',
                'ì' | 'í' | 'î' | 'ï' => 'i',
                'ñ' => 'n',
                other => other,
            };
        }
        // s[len - 1] == 's' here.
        if s[len - 2] == 'a' || s[len - 2] == 'o' {
            return len - 1;
        }
        if s[len - 2] == 'e' {
            if s[len - 3] == 's' && s[len - 4] == 'e' {
                return len - 2;
            }
            if s[len - 3] == 'c' {
                s[len - 3] = 'z';
            }
            return len - 2;
        }
        len - 1
    })
}

// ---------------------------------------------------------------------
// Italian
// ---------------------------------------------------------------------

/// Lucene `ItalianLightStemmer`.
pub fn italian_light(word: &str) -> String {
    run(word, |s, len| {
        if len < 6 {
            return len;
        }
        fold_es_it(s, len);
        let prev = s[len - 2];
        match s[len - 1] {
            'e' | 'i' => {
                if prev == 'i' || prev == 'h' {
                    len - 2
                } else {
                    len - 1
                }
            }
            'a' | 'o' => {
                if prev == 'i' {
                    len - 2
                } else {
                    len - 1
                }
            }
            _ => len,
        }
    })
}

// ---------------------------------------------------------------------
// Portuguese
// ---------------------------------------------------------------------

fn pt_remove_suffix(s: &mut [char], mut len: usize) -> usize {
    if len > 4 && ends(s, len, "es") && matches!(s[len - 3], 'r' | 's' | 'l' | 'z') {
        return len - 2;
    }
    if len > 3 && ends(s, len, "ns") {
        s[len - 2] = 'm';
        return len - 1;
    }
    if len > 4 && (ends(s, len, "eis") || ends(s, len, "éis")) {
        s[len - 3] = 'e';
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends(s, len, "ais") {
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends(s, len, "óis") {
        s[len - 3] = 'o';
        s[len - 2] = 'l';
        return len - 1;
    }
    if len > 4 && ends(s, len, "is") {
        s[len - 1] = 'l';
        return len;
    }
    if len > 3 && (ends(s, len, "ões") || ends(s, len, "ães")) {
        len -= 1;
        s[len - 2] = 'ã';
        s[len - 1] = 'o';
        return len;
    }
    if len > 6 && ends(s, len, "mente") {
        return len - 5;
    }
    if len > 3 && s[len - 1] == 's' {
        return len - 1;
    }
    len
}

fn pt_norm_feminine(s: &mut [char], len: usize) -> usize {
    if len > 7 && (ends(s, len, "inha") || ends(s, len, "iaca") || ends(s, len, "eira")) {
        s[len - 1] = 'o';
        return len;
    }
    if len > 6 {
        if ends_any(s, len, &["osa", "ica", "ida", "ada", "iva", "ama"]) {
            s[len - 1] = 'o';
            return len;
        }
        if ends(s, len, "ona") {
            s[len - 3] = 'ã';
            s[len - 2] = 'o';
            return len - 1;
        }
        if ends(s, len, "ora") {
            return len - 1;
        }
        if ends(s, len, "esa") {
            s[len - 3] = 'ê';
            return len - 1;
        }
        if ends(s, len, "na") {
            s[len - 1] = 'o';
            return len;
        }
    }
    len
}

/// Lucene `PortugueseLightStemmer`.
pub fn portuguese_light(word: &str) -> String {
    run(word, |s, mut len| {
        if len < 4 {
            return len;
        }
        len = pt_remove_suffix(s, len);
        if len > 3 && s[len - 1] == 'a' {
            len = pt_norm_feminine(s, len);
        }
        if len > 4 && matches!(s[len - 1], 'e' | 'a' | 'o') {
            len -= 1;
        }
        for c in s[..len].iter_mut() {
            *c = match *c {
                'à' | 'á' | 'â' | 'ä' | 'ã' => 'a',
                'ò' | 'ó' | 'ô' | 'ö' | 'õ' => 'o',
                'è' | 'é' | 'ê' | 'ë' => 'e',
                'ù' | 'ú' | 'û' | 'ü' => 'u',
                'ì' | 'í' | 'î' | 'ï' => 'i',
                'ç' => 'c',
                other => other,
            };
        }
        len
    })
}

/// One rule of an RSLP step: `{suffix, min, replacement, {exceptions}}`.
struct RslpRule {
    suffix: &'static str,
    min: usize,
    replacement: &'static str,
    /// Whole-word exceptions (the step header's flag is 1).
    exceptions: &'static [&'static str],
}

/// The `"Plural", 3, 1, {"s"}` step of Lucene's `pt/portuguese.rslp`,
/// verbatim (step minimum length 3, whole-word exceptions, condition:
/// the word ends in `s`).
const RSLP_PLURAL: &[RslpRule] = &[
    RslpRule { suffix: "ns", min: 1, replacement: "m", exceptions: &[] },
    RslpRule { suffix: "ões", min: 3, replacement: "ão", exceptions: &[] },
    RslpRule { suffix: "ães", min: 1, replacement: "ão", exceptions: &["mães"] },
    RslpRule { suffix: "ais", min: 1, replacement: "al", exceptions: &["cais", "mais"] },
    RslpRule { suffix: "éis", min: 2, replacement: "el", exceptions: &[] },
    RslpRule { suffix: "eis", min: 2, replacement: "el", exceptions: &[] },
    RslpRule { suffix: "óis", min: 2, replacement: "ol", exceptions: &[] },
    RslpRule {
        suffix: "is",
        min: 2,
        replacement: "il",
        exceptions: &["lápis", "cais", "mais", "crúcis", "biquínis", "pois", "depois", "dois", "leis"],
    },
    RslpRule { suffix: "les", min: 3, replacement: "l", exceptions: &[] },
    RslpRule { suffix: "res", min: 3, replacement: "r", exceptions: &["árvores"] },
    RslpRule {
        suffix: "s",
        min: 2,
        replacement: "",
        exceptions: &[
            "aliás", "pires", "lápis", "cais", "mais", "mas", "menos", "férias", "fezes", "pêsames",
            "crúcis", "gás", "atrás", "moisés", "através", "convés", "ês", "país", "após", "ambas",
            "ambos", "messias", "depois",
        ],
    },
];

/// Lucene `PortugueseMinimalStemmer`: `RSLPStemmerBase.Step.apply` for the
/// RSLP "Plural" step (first matching rule wins; `Rule.matches` is
/// `len - suffix.length >= min && endsWith`, plus the whole-word exception
/// set; `Rule.replace` overwrites the suffix with the replacement).
pub fn portuguese_minimal(word: &str) -> String {
    let s: Vec<char> = word.chars().collect();
    let len = s.len();
    if len < 3 || !ends(&s, len, "s") {
        return word.to_string();
    }
    for r in RSLP_PLURAL {
        let n = r.suffix.chars().count();
        if len >= n && len - n >= r.min && ends(&s, len, r.suffix) && !r.exceptions.contains(&word) {
            let mut out: String = s[..len - n].iter().collect();
            out.push_str(r.replacement);
            return out;
        }
    }
    word.to_string()
}

// ---------------------------------------------------------------------
// Swedish
// ---------------------------------------------------------------------

/// Lucene `SwedishLightStemmer`.
pub fn swedish_light(word: &str) -> String {
    run(word, |s, mut len| {
        if len > 4 && s[len - 1] == 's' {
            len -= 1;
        }
        if len > 7 && ends_any(s, len, &["elser", "heten"]) {
            return len - 5;
        }
        if len > 6 && ends_any(s, len, &["arne", "erna", "ande", "else", "aste", "orna", "aren"]) {
            return len - 4;
        }
        if len > 5 && ends_any(s, len, &["are", "ast", "het"]) {
            return len - 3;
        }
        if len > 4 && ends_any(s, len, &["ar", "er", "or", "en", "at", "te", "et"]) {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 't' | 'a' | 'e' | 'n') {
            return len - 1;
        }
        len
    })
}

/// Lucene `SwedishMinimalStemmer` (Lucene 9.0+; ES has no `stemmer` name
/// for it, provided for completeness).
pub fn swedish_minimal(word: &str) -> String {
    run(word, |s, mut len| {
        if len > 4 && s[len - 1] == 's' {
            len -= 1;
        }
        if len > 6 && ends_any(s, len, &["arne", "erna", "arna", "orna", "aren"]) {
            return len - 4;
        }
        if len > 5 && ends(s, len, "are") {
            return len - 3;
        }
        if len > 4 && ends_any(s, len, &["ar", "at", "er", "et", "or", "en"]) {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 'a' | 'e' | 'n') {
            return len - 1;
        }
        len
    })
}

// ---------------------------------------------------------------------
// Norwegian (Bokmål / Nynorsk)
// ---------------------------------------------------------------------

fn norwegian_light_impl(s: &[char], mut len: usize, bokmaal: bool, nynorsk: bool) -> usize {
    if len > 4 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 7
        && ((ends(s, len, "heter") && bokmaal)
            || (ends(s, len, "heten") && bokmaal)
            || (ends(s, len, "heita") && nynorsk))
    {
        return len - 5;
    }
    if len > 8 && nynorsk && ends_any(s, len, &["heiter", "leiken", "leikar"]) {
        return len - 6;
    }
    if len > 5 && (ends(s, len, "dom") || (ends(s, len, "het") && bokmaal)) {
        return len - 3;
    }
    if len > 6 && nynorsk && ends_any(s, len, &["heit", "semd", "leik"]) {
        return len - 4;
    }
    if len > 7 && ends_any(s, len, &["elser", "elsen"]) {
        return len - 5;
    }
    if len > 6
        && ((ends(s, len, "ende") && bokmaal)
            || (ends(s, len, "ande") && nynorsk)
            || ends(s, len, "else")
            || (ends(s, len, "este") && bokmaal)
            || (ends(s, len, "aste") && nynorsk)
            || (ends(s, len, "eren") && bokmaal)
            || (ends(s, len, "aren") && nynorsk))
    {
        return len - 4;
    }
    if len > 5
        && ((ends(s, len, "ere") && bokmaal)
            || (ends(s, len, "are") && nynorsk)
            || (ends(s, len, "est") && bokmaal)
            || (ends(s, len, "ast") && nynorsk)
            || ends(s, len, "ene")
            || (ends(s, len, "ane") && nynorsk))
    {
        return len - 3;
    }
    if len > 4
        && (ends(s, len, "er")
            || ends(s, len, "en")
            || ends(s, len, "et")
            || (ends(s, len, "ar") && nynorsk)
            || (ends(s, len, "st") && bokmaal)
            || ends(s, len, "te"))
    {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}

fn norwegian_minimal_impl(s: &[char], mut len: usize, nynorsk: bool) -> usize {
    if len > 4 && s[len - 1] == 's' {
        len -= 1;
    }
    if len > 5 && (ends(s, len, "ene") || (ends(s, len, "ane") && nynorsk)) {
        return len - 3;
    }
    if len > 4
        && (ends(s, len, "er") || ends(s, len, "en") || ends(s, len, "et") || (ends(s, len, "ar") && nynorsk))
    {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 'a' | 'e') {
        return len - 1;
    }
    len
}

/// Lucene `NorwegianLightStemmer(BOKMAAL)` (ES `light_norwegian`).
pub fn norwegian_light(word: &str) -> String {
    run(word, |s, len| norwegian_light_impl(s, len, true, false))
}

/// Lucene `NorwegianLightStemmer(NYNORSK)` (ES `light_nynorsk`).
pub fn nynorsk_light(word: &str) -> String {
    run(word, |s, len| norwegian_light_impl(s, len, false, true))
}

/// Lucene `NorwegianMinimalStemmer(BOKMAAL)` (ES `minimal_norwegian`).
pub fn norwegian_minimal(word: &str) -> String {
    run(word, |s, len| norwegian_minimal_impl(s, len, false))
}

/// Lucene `NorwegianMinimalStemmer(NYNORSK)` (ES `minimal_nynorsk`).
pub fn nynorsk_minimal(word: &str) -> String {
    run(word, |s, len| norwegian_minimal_impl(s, len, true))
}

// ---------------------------------------------------------------------
// Finnish
// ---------------------------------------------------------------------

fn fi_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn fi_step1(s: &[char], len: usize) -> usize {
    if len > 8 {
        if ends(s, len, "kin") {
            return fi_step1(s, len - 3);
        }
        if ends(s, len, "ko") {
            return fi_step1(s, len - 2);
        }
    }
    if len > 11 {
        if ends(s, len, "dellinen") {
            return len - 8;
        }
        if ends(s, len, "dellisuus") {
            return len - 9;
        }
    }
    len
}

fn fi_step2(s: &[char], len: usize) -> usize {
    if len > 5 {
        if ends_any(s, len, &["lla", "tse", "sti"]) {
            return len - 3;
        }
        if ends(s, len, "ni") {
            return len - 2;
        }
        if ends(s, len, "aa") {
            return len - 1;
        }
    }
    len
}

fn fi_step3(s: &mut [char], len: usize) -> usize {
    if len > 8 {
        if ends(s, len, "nnen") {
            s[len - 4] = 's';
            return len - 3;
        }
        if ends(s, len, "ntena") {
            s[len - 5] = 's';
            return len - 4;
        }
        if ends(s, len, "tten") {
            return len - 4;
        }
        if ends(s, len, "eiden") {
            return len - 5;
        }
    }
    if len > 6 {
        if ends_any(s, len, &["neen", "niin", "seen", "teen", "inen"]) {
            return len - 4;
        }
        if s[len - 3] == 'h' && fi_vowel(s[len - 2]) && s[len - 1] == 'n' {
            return len - 3;
        }
        if ends(s, len, "den") {
            s[len - 3] = 's';
            return len - 2;
        }
        if ends(s, len, "ksen") {
            s[len - 4] = 's';
            return len - 3;
        }
        if ends_any(s, len, &["ssa", "sta", "lla", "lta", "tta", "ksi", "lle"]) {
            return len - 3;
        }
    }
    if len > 5 {
        if ends_any(s, len, &["na", "ne"]) {
            return len - 2;
        }
        if ends(s, len, "nei") {
            return len - 3;
        }
    }
    if len > 4 {
        if ends_any(s, len, &["ja", "ta"]) {
            return len - 2;
        }
        if s[len - 1] == 'a' {
            return len - 1;
        }
        if s[len - 1] == 'n' && fi_vowel(s[len - 2]) {
            return len - 2;
        }
        if s[len - 1] == 'n' {
            return len - 1;
        }
    }
    len
}

fn fi_norm1(s: &mut [char], len: usize) -> usize {
    if len > 5 && ends(s, len, "hde") {
        s[len - 3] = 'k';
        s[len - 2] = 's';
        s[len - 1] = 'i';
    }
    if len > 4 && ends_any(s, len, &["ei", "at"]) {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 't' | 's' | 'j' | 'e' | 'a' | 'i') {
        return len - 1;
    }
    len
}

fn fi_norm2(s: &mut Vec<char>, mut len: usize) -> usize {
    if len > 8 && matches!(s[len - 1], 'e' | 'o' | 'u') {
        len -= 1;
    }
    if len > 4 {
        if s[len - 1] == 'i' {
            len -= 1;
        }
        if len > 4 {
            let mut ch = s[0];
            let mut i = 1;
            while i < len {
                if s[i] == ch && matches!(ch, 'k' | 'p' | 't') {
                    len = delete(s, i, len);
                } else {
                    ch = s[i];
                    i += 1;
                }
            }
        }
    }
    len
}

/// Lucene `FinnishLightStemmer`.
pub fn finnish_light(word: &str) -> String {
    run(word, |s, len| {
        if len < 4 {
            return len;
        }
        for c in s[..len].iter_mut() {
            *c = match *c {
                'ä' | 'å' => 'a',
                'ö' => 'o',
                other => other,
            };
        }
        let len = fi_step1(s, len);
        let len = fi_step2(s, len);
        let len = fi_step3(s, len);
        let len = fi_norm1(s, len);
        fi_norm2(s, len)
    })
}

// ---------------------------------------------------------------------
// Russian
// ---------------------------------------------------------------------

/// Lucene `RussianLightStemmer`.
pub fn russian_light(word: &str) -> String {
    run(word, |s, len| {
        let len = russian_remove_case(s, len);
        if len > 3 {
            match s[len - 1] {
                'ь' | 'и' => return len - 1,
                'н' if s[len - 2] == 'н' => return len - 1,
                _ => {}
            }
        }
        len
    })
}

fn russian_remove_case(s: &[char], len: usize) -> usize {
    if len > 6 && ends_any(s, len, &["иями", "оями"]) {
        return len - 4;
    }
    if len > 5
        && ends_any(
            s,
            len,
            &["иям", "иях", "оях", "ями", "оям", "оьв", "ами", "его", "ему", "ери", "ими", "ого", "ому", "ыми", "оев"],
        )
    {
        return len - 3;
    }
    if len > 4
        && ends_any(
            s,
            len,
            &[
                "ая", "яя", "ях", "юю", "ах", "ею", "их", "ия", "ию", "ьв", "ою", "ую", "ям", "ых", "ея", "ам",
                "ем", "ей", "ём", "ев", "ий", "им", "ое", "ой", "ом", "ов", "ые", "ый", "ым", "ми",
            ],
        )
    {
        return len - 2;
    }
    if len > 3 && matches!(s[len - 1], 'а' | 'е' | 'и' | 'о' | 'у' | 'й' | 'ы' | 'я' | 'ь') {
        return len - 1;
    }
    len
}

// ---------------------------------------------------------------------
// Hungarian
// ---------------------------------------------------------------------

fn hu_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn hu_remove_case(s: &[char], len: usize) -> usize {
    if len > 6 && ends(s, len, "kent") {
        return len - 4;
    }
    if len > 5 {
        if ends_any(
            s,
            len,
            &["nak", "nek", "val", "vel", "ert", "rol", "ban", "ben", "bol", "nal", "nel", "hoz", "hez", "tol"],
        ) {
            return len - 3;
        }
        if ends_any(s, len, &["al", "el"]) && !hu_vowel(s[len - 3]) && s[len - 3] == s[len - 4] {
            return len - 3;
        }
    }
    if len > 4 {
        if ends_any(s, len, &["at", "et", "ot", "va", "ve", "ra", "re", "ba", "be", "ul", "ig"]) {
            return len - 2;
        }
        if ends_any(s, len, &["on", "en"]) && !hu_vowel(s[len - 3]) {
            return len - 2;
        }
        match s[len - 1] {
            't' | 'n' => return len - 1,
            'a' | 'e' if s[len - 2] == s[len - 3] && !hu_vowel(s[len - 2]) => return len - 2,
            _ => {}
        }
    }
    len
}

fn hu_remove_possessive(s: &[char], len: usize) -> usize {
    if len > 6 {
        if !hu_vowel(s[len - 5]) && ends_any(s, len, &["atok", "otok", "etek"]) {
            return len - 4;
        }
        if ends_any(s, len, &["itek", "itok"]) {
            return len - 4;
        }
    }
    if len > 5 {
        if !hu_vowel(s[len - 4]) && ends_any(s, len, &["unk", "tok", "tek"]) {
            return len - 3;
        }
        if hu_vowel(s[len - 4]) && ends(s, len, "juk") {
            return len - 3;
        }
        if ends(s, len, "ink") {
            return len - 3;
        }
    }
    if len > 4 {
        if !hu_vowel(s[len - 3]) && ends_any(s, len, &["am", "em", "om", "ad", "ed", "od", "uk"]) {
            return len - 2;
        }
        if hu_vowel(s[len - 3]) && ends_any(s, len, &["nk", "ja", "je"]) {
            return len - 2;
        }
        if ends_any(s, len, &["im", "id", "ik"]) {
            return len - 2;
        }
    }
    if len > 3 {
        match s[len - 1] {
            'a' | 'e' if !hu_vowel(s[len - 2]) => return len - 1,
            'm' | 'd' if hu_vowel(s[len - 2]) => return len - 1,
            'i' => return len - 1,
            _ => {}
        }
    }
    len
}

/// Lucene `HungarianLightStemmer`.
pub fn hungarian_light(word: &str) -> String {
    run(word, |s, len| {
        for c in s[..len].iter_mut() {
            *c = match *c {
                'á' => 'a',
                'ë' | 'é' => 'e',
                'í' => 'i',
                'ó' | 'ő' | 'õ' | 'ö' => 'o',
                'ú' | 'ű' | 'ũ' | 'û' | 'ü' => 'u',
                other => other,
            };
        }
        let mut len = hu_remove_case(s, len);
        len = hu_remove_possessive(s, len);
        // removePlural (with its intentional fall-through)
        if len > 3 && s[len - 1] == 'k' {
            len = if matches!(s[len - 2], 'a' | 'o' | 'e') && len > 4 { len - 2 } else { len - 1 };
        }
        // normalize
        if len > 3 && matches!(s[len - 1], 'a' | 'e' | 'i' | 'o') {
            len -= 1;
        }
        len
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn check(name: &str, f: fn(&str) -> String, pairs: &[(&str, &str)]) {
        for (input, expected) in pairs {
            assert_eq!(f(input), *expected, "{name}({input})");
        }
    }

    /// Load `testdata/lucene/<file>.tsv.zst`: tab-separated rows, column 0
    /// the input, then one expected column per stemmer.
    fn load(file: &str) -> Vec<Vec<String>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/analysis/testdata/lucene")
            .join(format!("{file}.tsv.zst"));
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let text = String::from_utf8(zstd::decode_all(&raw[..]).expect("zstd")).expect("utf-8");
        text.lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.split('\t').map(str::to_string).collect())
            .collect()
    }

    /// Run `f` over every row of `file`, comparing with column `col`;
    /// `lowercase` mirrors a lowercasing tokenizer in Lucene's test chain.
    fn vocab(file: &str, col: usize, min: usize, lowercase: bool, f: fn(&str) -> String) {
        let rows = load(file);
        assert!(rows.len() >= min, "{file}: only {} pairs", rows.len());
        let mut failures = Vec::new();
        for row in &rows {
            let input = if lowercase { row[0].to_lowercase() } else { row[0].clone() };
            let got = f(&input);
            if got != row[col] {
                failures.push(format!("{:?}: expected {:?}, got {got:?}", row[0], row[col]));
            }
        }
        assert!(
            failures.is_empty(),
            "{file}[{col}]: {}/{} mismatches:\n{}",
            failures.len(),
            rows.len(),
            failures[..failures.len().min(40)].join("\n")
        );
    }

    // Vocabulary tests: Lucene's own `Test*StemFilter` data
    // (lucene/analysis/common/src/test/org/apache/lucene/analysis/...).
    // The `*testdata.zip` vocabularies are stored as zstd TSV under
    // `testdata/lucene/`; de/fr/pt hold `word\tlight\tminimal` (the light
    // and minimal zips share their inputs). Sampling: every 2nd line
    // (lines 1, 3, 5, ...; N = 2) of the de, es, fi, hu, it, pt, ru and
    // sv vocabularies; fr and all Norwegian/Swedish-minimal `.txt` files
    // are complete.

    #[test]
    fn vocab_german_light() {
        vocab("de", 1, 17_000, false, german_light);
    }
    #[test]
    fn vocab_german_minimal() {
        vocab("de", 2, 17_000, false, german_minimal);
    }
    #[test]
    fn vocab_french_light() {
        vocab("fr", 1, 20_000, false, french_light);
    }
    #[test]
    fn vocab_french_minimal() {
        vocab("fr", 2, 20_000, false, french_minimal);
    }
    #[test]
    fn vocab_spanish_light() {
        vocab("es", 1, 14_000, false, spanish_light);
    }
    #[test]
    fn vocab_italian_light() {
        vocab("it", 1, 17_000, false, italian_light);
    }
    /// `TestPortuguese*StemFilter` uses `MockTokenizer(SIMPLE, lowercase=true)`.
    #[test]
    fn vocab_portuguese_light() {
        vocab("pt", 1, 16_000, true, portuguese_light);
    }
    #[test]
    fn vocab_portuguese_minimal() {
        vocab("pt", 2, 16_000, true, portuguese_minimal);
    }
    #[test]
    fn vocab_swedish_light() {
        vocab("sv", 1, 15_000, false, swedish_light);
    }
    #[test]
    fn vocab_swedish_minimal() {
        vocab("sv_minimal", 1, 60, false, swedish_minimal);
    }
    #[test]
    fn vocab_norwegian_light() {
        vocab("no_nb_light", 1, 100, false, norwegian_light);
        vocab("no_nn_light", 1, 80, false, nynorsk_light);
    }
    #[test]
    fn vocab_norwegian_minimal() {
        vocab("no_nb_minimal", 1, 70, false, norwegian_minimal);
        vocab("no_nn_minimal", 1, 50, false, nynorsk_minimal);
    }
    #[test]
    fn vocab_finnish_light() {
        vocab("fi", 1, 25_000, false, finnish_light);
    }
    #[test]
    fn vocab_russian_light() {
        vocab("ru", 1, 24_000, false, russian_light);
    }
    #[test]
    fn vocab_hungarian_light() {
        vocab("hu", 1, 15_000, false, hungarian_light);
    }

    // checkOneTerm expectations from Lucene's Test*StemFilter classes.

    #[test]
    fn lucene_english_minimal() {
        check("english_minimal", english_minimal, &[
            ("queries", "query"), ("phrases", "phrase"), ("corpus", "corpus"), ("stress", "stress"),
            ("kings", "king"), ("panels", "panel"), ("aerodynamics", "aerodynamic"),
            ("congress", "congress"), ("serious", "serious"), ("", ""),
        ]);
    }

    #[test]
    fn lucene_german_minimal() {
        check("german_minimal", german_minimal, &[
            ("sängerinnen", "sangerin"), ("frauen", "frau"), ("kenntnisse", "kenntnis"),
            ("staates", "staat"), ("bilder", "bild"), ("boote", "boot"), ("götter", "gott"),
            ("äpfel", "apfel"), ("", ""),
        ]);
    }

    #[test]
    fn lucene_spanish_minimal() {
        check("spanish_minimal", spanish_minimal, &[
            ("actrices", "actriz"), ("niños", "nino"), ("países", "pais"), ("caragodor", "caragodor"),
            ("móviles", "movil"), ("chicas", "chica"), ("", ""),
        ]);
    }

    #[test]
    fn lucene_french_light() {
        check("french_light", french_light, &[
            ("chevaux", "cheval"), ("cheval", "cheval"), ("hiboux", "hibou"), ("hibou", "hibou"),
            ("chantés", "chant"), ("chanter", "chant"), ("chante", "chant"), ("chant", "chant"),
            ("baronnes", "baron"), ("barons", "baron"), ("baron", "baron"), ("peaux", "peau"),
            ("peau", "peau"), ("anneaux", "aneau"), ("anneau", "aneau"), ("neveux", "neveu"),
            ("neveu", "neveu"), ("affreux", "afreu"), ("affreuse", "afreu"),
            ("investissement", "investi"), ("investir", "investi"), ("assourdissant", "asourdi"),
            ("assourdir", "asourdi"), ("pratiquement", "pratiqu"), ("pratique", "pratiqu"),
            ("administrativement", "administratif"), ("administratif", "administratif"),
            ("justificatrice", "justifi"), ("justificateur", "justifi"), ("justifier", "justifi"),
            ("educatrice", "eduqu"), ("eduquer", "eduqu"), ("communicateur", "comuniqu"),
            ("communiquer", "comuniqu"), ("accompagnatrice", "acompagn"),
            ("accompagnateur", "acompagn"), ("administrateur", "administr"),
            ("administrer", "administr"), ("productrice", "product"), ("producteur", "product"),
            ("acheteuse", "achet"), ("acheteur", "achet"), ("planteur", "plant"), ("plante", "plant"),
            ("poreuse", "poreu"), ("poreux", "poreu"), ("plieuse", "plieu"),
            ("bijoutière", "bijouti"), ("bijoutier", "bijouti"), ("caissière", "caisi"),
            ("caissier", "caisi"), ("abrasive", "abrasif"), ("abrasif", "abrasif"), ("folle", "fou"),
            ("fou", "fou"), ("personnelle", "person"), ("personne", "person"),
            ("complète", "complet"), ("complet", "complet"), ("aromatique", "aromat"),
            ("faiblesse", "faibl"), ("faible", "faibl"), ("patinage", "patin"), ("patin", "patin"),
            ("sonorisation", "sono"), ("ritualisation", "rituel"), ("rituel", "rituel"),
            ("nomination", "nomin"), ("disposition", "dispos"), ("dispose", "dispos"),
            ("1234555", "1234555"), ("12333345", "12333345"), ("1234", "1234"),
            ("abcdeff", "abcdef"), ("abcccddeef", "abcdef"), ("créées", "cre"),
            ("22hh00", "22h00"), ("", ""),
        ]);
    }

    #[test]
    fn lucene_french_minimal() {
        check("french_minimal", french_minimal, &[
            ("chevaux", "cheval"), ("hiboux", "hibou"), ("chantés", "chant"), ("chanter", "chant"),
            ("chante", "chant"), ("baronnes", "baron"), ("barons", "baron"), ("baron", "baron"),
            ("1234555", "1234555"), ("12333345", "12333345"), ("1234", "1234"),
            ("abcdeff", "abcdef"), ("abcccddeef", "abcccddeef"), ("créées", "cré"),
            ("22hh00", "22hh00"), ("", ""),
        ]);
    }

    #[test]
    fn lucene_portuguese_light() {
        check("portuguese_light", portuguese_light, &[
            ("doutores", "doutor"), ("doutor", "doutor"), ("homens", "homem"), ("homem", "homem"),
            ("papéis", "papel"), ("papel", "papel"), ("normais", "normal"), ("normal", "normal"),
            ("lencóis", "lencol"), ("lencol", "lencol"), ("barris", "barril"), ("barril", "barril"),
            ("botões", "bota"), ("botão", "bota"), ("", ""),
        ]);
    }

    #[test]
    fn norwegian_flags_differ() {
        // Nynorsk-only endings are ignored by the Bokmål stemmer and vice versa.
        assert_eq!(nynorsk_light("gutane"), "gut");
        assert_eq!(norwegian_light("gutane"), "gutan");
        assert_eq!(nynorsk_minimal("gutar"), "gut");
        assert_eq!(norwegian_minimal("gutar"), "gutar");
    }

    const ALL: &[fn(&str) -> String] = &[
        english_minimal, german_light, german_minimal, french_light, french_minimal, spanish_light,
        spanish_minimal, italian_light, portuguese_light, portuguese_minimal, swedish_light,
        swedish_minimal, norwegian_light, nynorsk_light, norwegian_minimal, nynorsk_minimal,
        finnish_light, russian_light, hungarian_light,
    ];

    /// `testEmptyTerm` for every filter, plus short and non-Latin input.
    #[test]
    fn no_panic_on_tricky_input() {
        for f in ALL {
            assert_eq!(f(""), "");
            for w in ["é", "123", "ß", "ĳ", "s", "es", "ies", "xx", "aux", "eaux", "ссс", "kkkkkkkkkk", "日本語", "𝔘𝔫𝔦"] {
                let _ = f(w);
            }
        }
    }
}
