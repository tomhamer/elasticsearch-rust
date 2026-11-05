//! Snowball Norwegian (Bokmål) stemmer.
//!
//! Faithful port of `norwegian.sbl` from the Snowball project. Only R1 is
//! used:
//!
//! * **Region.** R1 is the region after the first non-vowel following a
//!   vowel, adjusted so that at least 3 characters precede it. Vowels are
//!   `a e i o u y æ å ø`. Every suffix search below is restricted to R1
//!   (longest match inside R1).
//! * **Step 1.** The longest of `a e ede ande ende ane ene hetene en heten ar
//!   er heter as es edes endes enes hetenes ens hetens ets et het ast ers s
//!   erte ert` is deleted (`erte`/`ert` become `er`); `s` is deleted after a
//!   valid s-ending (`b c d f g h j l m n o p r t v y z`, or `k` not
//!   preceded by a vowel). This is the Snowball revision bundled by Lucene
//!   10.x; Snowball 3's apostrophe and `ers` exception rules are absent.
//! * **Step 2.** If the word ends `dt` or `vt` in R1, the `t` is deleted.
//! * **Step 3.** The longest of `leg eleg ig eig lig elig els lov elov slov
//!   hetslov` in R1 is deleted.
//!
//! Words shorter than three characters are returned unchanged, as in the
//! reference implementation. The input is expected to be lowercased already;
//! anything that is not a letter the algorithm knows about passes through.

/// Vowels of the Norwegian algorithm.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'æ' | 'å' | 'ø')
}

/// Plain valid s-endings (`r` and the `k` case are handled in `valid_s_ending`).
fn is_s_ending(c: char) -> bool {
    matches!(
        c,
        'b' | 'c' | 'd' | 'f' | 'g' | 'h' | 'j' | 'l' | 'm' | 'n' | 'o' | 'p' | 't' | 'v' | 'y' | 'z'
    )
}

/// Does `w` end with `suffix`?
fn ends_with(w: &[char], suffix: &str) -> bool {
    let n = suffix.chars().count();
    n <= w.len() && w[w.len() - n..].iter().copied().eq(suffix.chars())
}

/// Longest of `suffixes` that `w` ends with and that starts at or after
/// `p1`, with its start index.
fn longest_suffix_in_r1<'a>(
    w: &[char],
    p1: usize,
    suffixes: &[&'a str],
) -> Option<(&'a str, usize)> {
    let mut best: Option<(&str, usize)> = None;
    for &s in suffixes {
        let len = s.chars().count();
        if len <= w.len() - p1 && ends_with(w, s) {
            let start = w.len() - len;
            if start >= p1 && best.is_none_or(|(_, bstart)| start < bstart) {
                best = Some((s, start));
            }
        }
    }
    best
}

/// Compute the start of R1 (`len` if absent). Caller guarantees `len >= 3`.
fn mark_r1(w: &[char]) -> usize {
    let n = w.len();
    let mut i = 0;
    while i < n && !is_vowel(w[i]) {
        i += 1;
    }
    while i < n && is_vowel(w[i]) {
        i += 1;
    }
    if i >= n {
        return n;
    }
    (i + 1).max(3)
}

/// Is the text before `start` a valid s-ending?
fn valid_s_ending(w: &[char], start: usize) -> bool {
    if start == 0 {
        return false;
    }
    let c = w[start - 1];
    if is_s_ending(c) {
        return true;
    }
    c == 'r' || (c == 'k' && start >= 2 && !is_vowel(w[start - 2]))
}

/// Step 1: main suffixes.
fn main_suffix(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &[
        "a", "e", "ede", "ande", "ende", "ane", "ene", "hetene", "en", "heten", "ar", "er",
        "heter", "as", "es", "edes", "endes", "enes", "hetenes", "ens", "hetens", "ets", "et",
        "het", "ast", "ers", "s", "erte", "ert",
    ];
    let Some((suffix, start)) = longest_suffix_in_r1(w, p1, SUFFIXES) else {
        return;
    };
    match suffix {
        "s" => {
            if valid_s_ending(w, start) {
                w.truncate(start);
            }
        }
        "erte" | "ert" => {
            w.truncate(start);
            w.extend("er".chars());
        }
        _ => w.truncate(start),
    }
}

/// Step 2: `dt`/`vt` in R1 lose their `t`.
fn consonant_pair(w: &mut Vec<char>, p1: usize) {
    if longest_suffix_in_r1(w, p1, &["dt", "vt"]).is_some() {
        w.pop();
    }
}

/// Step 3: other suffixes.
fn other_suffix(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &[
        "leg", "eleg", "ig", "eig", "lig", "elig", "els", "lov", "elov", "slov", "hetslov",
    ];
    if let Some((_, start)) = longest_suffix_in_r1(w, p1, SUFFIXES) {
        w.truncate(start);
    }
}

/// Stem a lowercased Norwegian (Bokmål) word with the Snowball Norwegian
/// algorithm.
///
/// Never panics; non-letters pass through unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    if w.len() < 3 {
        return word.to_string();
    }
    let p1 = mark_r1(&w);
    main_suffix(&mut w, p1);
    consonant_pair(&mut w, p1);
    other_suffix(&mut w, p1);
    w.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(cases: &[(&str, &str)]) {
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|&(input, expected)| {
                let got = stem(input);
                (got != expected).then(|| format!("{input:?}: expected {expected:?}, got {got:?}"))
            })
            .collect();
        assert!(failures.is_empty(), "failing vectors:\n{}", failures.join("\n"));
    }

    /// Vectors taken from the official Snowball Norwegian voc.txt / output.txt.
    #[test]
    fn official_vocabulary() {
        check(&[
            ("aabakken", "aabakk"),
            ("aabakkens", "aabakk"),
            ("aarbakke", "aarbakk"),
            ("aase", "aas"),
            ("abitanti", "abitanti"),
            ("abitazione", "abitazion"),
            ("ablegøyer", "ablegøy"),
            ("absolutt", "absolutt"),
            ("absorberes", "absorber"),
            ("adgangen", "adgang"),
            ("adkomstdokumenter", "adkomstdokument"),
            ("adkomsten", "adkomst"),
            ("adlyde", "adlyd"),
            ("adlyder", "adlyd"),
            ("administrasjonen", "administrasjon"),
            ("administrasjonsdepartementet", "administrasjonsdepartement"),
            ("administrasjonskostnadene", "administrasjonskostnad"),
            ("administrative", "administrativ"),
            ("administrativt", "administrativ"),
            ("administrere", "administrer"),
            ("administrerende", "administrer"),
            ("administreres", "administrer"),
            ("administrert", "administrer"),
            ("adopsjonspenger", "adopsjonspeng"),
            ("adresse", "adress"),
            ("adresser", "adress"),
            ("adskillelse", "adskill"),
            ("leigeobjektet", "leigeobjekt"),
            ("leiger", "leig"),
            ("leigereguleringa", "leigeregulering"),
            ("leigeregulert", "leigereguler"),
            ("leigerettane", "leigerett"),
            ("leigerettar", "leigerett"),
            ("leigt", "leigt"),
            ("leiinga", "leiing"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("ab", "ab"),
            ("'e", "'e"),
            ("123", "123"),
            ("--", "--"),
            ("'''", "'''"),
        ]);
        let _ = stem("日本語テキスト");
    }
}
