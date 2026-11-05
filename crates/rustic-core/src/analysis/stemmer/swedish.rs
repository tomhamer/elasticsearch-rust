//! Snowball Swedish stemmer.
//!
//! Faithful port of `swedish.sbl` from the Snowball project. Only R1 is used:
//!
//! * **Region.** R1 is the region after the first non-vowel following a
//!   vowel, adjusted so that at least 3 letters precede it. Vowels are
//!   `a e i o u y ä å ö`. Every suffix search below is restricted to R1
//!   (longest match inside R1).
//! * **Step 1.** The longest of `a arna erna heterna orna ad e ade ande arne
//!   are aste en anden aren heten ern ar er heter or as arnas ernas ornas es
//!   ades andes ens arens hetens erns at andet het ast` is deleted; `s` is
//!   deleted after a valid s-ending (`b c d f g h j k l m n o p r t v y`).
//!   (This is the Snowball revision bundled by Lucene 10.x; the later
//!   `et`/`ets` rules of Snowball 3 are intentionally absent.)
//! * **Step 2.** If the word ends `dd gd nn dt gt kt tt` in R1, the last
//!   letter is deleted.
//! * **Step 3.** The longest of `lig ig els` in R1 is deleted; `öst` becomes
//!   `ös` after a valid öst-ending (`i k l n p r t u v`); `fullt` becomes
//!   `full`.
//!
//! The input is expected to be lowercased already; anything that is not a
//! letter the algorithm knows about passes through untouched.

/// Vowels of the Swedish algorithm.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'ä' | 'å' | 'ö')
}

/// Valid s-ending.
fn is_s_ending(c: char) -> bool {
    matches!(
        c,
        'b' | 'c'
            | 'd'
            | 'f'
            | 'g'
            | 'h'
            | 'j'
            | 'k'
            | 'l'
            | 'm'
            | 'n'
            | 'o'
            | 'p'
            | 'r'
            | 't'
            | 'v'
            | 'y'
    )
}

/// Valid öst-ending.
fn is_ost_ending(c: char) -> bool {
    matches!(c, 'i' | 'k' | 'l' | 'n' | 'p' | 'r' | 't' | 'u' | 'v')
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

/// Compute the start of R1 (`len` if absent).
fn mark_r1(w: &[char]) -> usize {
    let n = w.len();
    if n < 3 {
        return n;
    }
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

/// Step 1: main suffixes.
fn main_suffix(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &[
        "a", "arna", "erna", "heterna", "orna", "ad", "e", "ade", "ande", "arne", "are", "aste",
        "en", "anden", "aren", "heten", "ern", "ar", "er", "heter", "or", "as", "arnas",
        "ernas", "ornas", "es", "ades", "andes", "ens", "arens", "hetens", "erns", "at",
        "andet", "het", "ast", "s",
    ];
    let Some((suffix, start)) = longest_suffix_in_r1(w, p1, SUFFIXES) else {
        return;
    };
    if suffix == "s" {
        if start > 0 && is_s_ending(w[start - 1]) {
            w.truncate(start);
        }
    } else {
        w.truncate(start);
    }
}

/// Step 2: `dd gd nn dt gt kt tt` in R1 lose their last letter.
fn consonant_pair(w: &mut Vec<char>, p1: usize) {
    if longest_suffix_in_r1(w, p1, &["dd", "gd", "nn", "dt", "gt", "kt", "tt"]).is_some() {
        w.pop();
    }
}

/// Step 3: other suffixes.
fn other_suffix(w: &mut Vec<char>, p1: usize) {
    let Some((suffix, start)) = longest_suffix_in_r1(w, p1, &["lig", "ig", "els", "öst", "fullt"])
    else {
        return;
    };
    match suffix {
        "öst" => {
            if start > 0 && is_ost_ending(w[start - 1]) {
                w.pop();
            }
        }
        "fullt" => {
            w.pop();
        }
        _ => w.truncate(start),
    }
}

/// Stem a lowercased Swedish word with the Snowball Swedish algorithm.
///
/// Never panics; non-letters pass through unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
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

    /// Vectors taken from the official Snowball Swedish voc.txt / output.txt.
    #[test]
    fn official_vocabulary() {
        check(&[
            ("aaele", "aael"),
            ("abba", "abb"),
            ("abborrar", "abborr"),
            ("abborrarna", "abborr"),
            ("abborre", "abborr"),
            ("abborrpinne", "abborrpin"),
            ("abrahams", "abraham"),
            ("abskons", "abskon"),
            ("absolut", "absolut"),
            ("ackompanjerade", "ackompanjer"),
            ("adcle", "adcl"),
            ("addra", "addr"),
            ("adeie", "adei"),
            ("adele", "adel"),
            ("adeles", "adel"),
            ("adelig", "ade"),
            ("adelns", "adeln"),
            ("adelsfröken", "adelsfrök"),
            ("adelskalendern", "adelskalend"),
            ("adelskalendrar", "adelskalendr"),
            ("adelskapen", "adelskap"),
            ("adelsmän", "adelsmän"),
            ("lycka", "lyck"),
            ("lyckades", "lyck"),
            ("lyckadt", "lyckad"),
            ("lyckan", "lyckan"),
            ("lyckans", "lyckan"),
            ("lyckats", "lyckat"),
            ("lycklig", "lyck"),
            ("lyckligare", "lyck"),
            ("lyckligaste", "lyck"),
            ("lyckliggöra", "lyckliggör"),
            ("lyckligheten", "lyck"),
            ("lyckligt", "lyck"),
            ("lyckligtvis", "lyckligtvis"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("ab", "ab"),
            ("ets", "ets"),
            ("123", "123"),
            ("--", "--"),
        ]);
        let _ = stem("日本語テキスト");
        let _ = stem("öst");
    }
}
