//! Snowball German stemmer (the standard `german` algorithm, not `german2`).
//!
//! Faithful port of `german.sbl` from the Snowball project:
//!
//! 1. **Prelude.** `u` and `y` between vowels are marked as `U` / `Y` so that
//!    they are not treated as vowels. Then `ß` is replaced by `ss`, `ae` by
//!    `ä`, `oe` by `ö` and `ue` by `ü` (except in `qu`).
//! 2. **Regions.** R1 is the region after the first non-vowel following a
//!    vowel, adjusted so that at least 3 letters precede it; R2 is the same
//!    construction applied again inside R1. Vowels are `a e i o u y ä ö ü`.
//! 3. **Step 1.** Longest of `em ern er e en es s` in R1 is deleted (`em`
//!    not after `syst`; `s` only after a valid s-ending; after deleting
//!    `e en es`, a trailing `niss` loses its final `s`).
//! 4. **Step 2.** Longest of `en er est st` in R1 is deleted (`st` only
//!    after a valid st-ending with at least 3 letters before that).
//! 5. **Step 3.** Derivational suffixes `end ung ig ik isch lich heit keit`
//!    are deleted in R2, with the follow-on deletions described in the
//!    algorithm (`ig` after `end`/`ung`, `er`/`en` after `lich`/`heit`,
//!    `lich`/`ig` after `keit`).
//! 6. **Postlude.** `U`/`Y` are lowered again and the umlauts are stripped
//!    (`ä→a`, `ö→o`, `ü→u`).
//!
//! This is the algorithm at Snowball commit `34f3612e5e8c` (the revision
//! bundled by Lucene 10.x), verified against the complete official
//! vocabulary in `testdata/snowball/german.tsv.zst`.
//!
//! The input is expected to be lowercased already; anything that is not a
//! letter the algorithm knows about simply passes through untouched.

/// Vowels of the German algorithm (the marked `U`/`Y` are deliberately not
/// vowels).
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'ä' | 'ö' | 'ü')
}

/// Valid s-ending: `b d f g h k l m n r t`.
fn is_s_ending(c: char) -> bool {
    matches!(c, 'b' | 'd' | 'f' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 'r' | 't')
}

/// Valid st-ending: the s-endings minus `r`.
fn is_st_ending(c: char) -> bool {
    c != 'r' && is_s_ending(c)
}

/// Does `w` end with `suffix`?
fn ends_with(w: &[char], suffix: &str) -> bool {
    let n = suffix.chars().count();
    n <= w.len() && w[w.len() - n..].iter().copied().eq(suffix.chars())
}

/// Returns the longest of `suffixes` that `w` ends with, together with its
/// start index. `suffixes` need not be sorted.
fn longest_suffix<'a>(w: &[char], suffixes: &[&'a str]) -> Option<(&'a str, usize)> {
    let mut best: Option<(&str, usize)> = None;
    for &s in suffixes {
        let len = s.chars().count();
        if ends_with(w, s) && best.is_none_or(|(_, start)| w.len() - len < start) {
            best = Some((s, w.len() - len));
        }
    }
    best
}

/// Prelude: mark `u`/`y` between vowels, then normalise `ß`, `ae`, `oe`, `ue`.
fn prelude(w: &mut Vec<char>) {
    // `repeat goto (v [('u'] v <- 'U') or ('y'] v <- 'Y'))`: after a
    // replacement the search resumes after the following vowel.
    let mut i = 0;
    while i + 2 < w.len() {
        if is_vowel(w[i]) && matches!(w[i + 1], 'u' | 'y') && is_vowel(w[i + 2]) {
            w[i + 1] = if w[i + 1] == 'u' { 'U' } else { 'Y' };
            i += 3;
        } else {
            i += 1;
        }
    }

    let mut out = Vec::with_capacity(w.len() + 2);
    let mut i = 0;
    while i < w.len() {
        let c = w[i];
        let next = w.get(i + 1).copied();
        match (c, next) {
            ('ß', _) => {
                out.push('s');
                out.push('s');
                i += 1;
            }
            ('a', Some('e')) => {
                out.push('ä');
                i += 2;
            }
            ('o', Some('e')) => {
                out.push('ö');
                i += 2;
            }
            ('u', Some('e')) => {
                out.push('ü');
                i += 2;
            }
            ('q', Some('u')) => {
                out.push('q');
                out.push('u');
                i += 2;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    *w = out;
}

/// Position just past the first non-vowel that follows a vowel at or after
/// `start`, or `None` if there is none.
fn region_after(w: &[char], start: usize) -> Option<usize> {
    let mut i = start;
    while i < w.len() && !is_vowel(w[i]) {
        i += 1;
    }
    if i >= w.len() {
        return None;
    }
    while i < w.len() && is_vowel(w[i]) {
        i += 1;
    }
    if i >= w.len() {
        return None;
    }
    Some(i + 1)
}

/// Compute `(p1, p2)`: the start offsets of R1 and R2 (`len` if absent).
fn mark_regions(w: &[char]) -> (usize, usize) {
    let n = w.len();
    if n < 3 {
        return (n, n);
    }
    let Some(r1) = region_after(w, 0) else {
        return (n, n);
    };
    let p1 = r1.max(3);
    let p2 = region_after(w, r1).unwrap_or(n);
    (p1, p2)
}

/// Step 1.
fn step1(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &["ern", "er", "em", "en", "es", "e", "s"];
    let Some((suffix, start)) = longest_suffix(w, SUFFIXES) else {
        return;
    };
    if start < p1 {
        return;
    }
    match suffix {
        "em" => {
            if !ends_with(&w[..start], "syst") {
                w.truncate(start);
            }
        }
        "ern" | "er" => w.truncate(start),
        "e" | "en" | "es" => {
            w.truncate(start);
            if ends_with(w, "niss") {
                w.pop();
            }
        }
        _ => {
            // "s"
            if start > 0 && is_s_ending(w[start - 1]) {
                w.truncate(start);
            }
        }
    }
}

/// Step 2.
fn step2(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &["est", "en", "er", "st"];
    let Some((suffix, start)) = longest_suffix(w, SUFFIXES) else {
        return;
    };
    if start < p1 {
        return;
    }
    match suffix {
        "st" => {
            // st_ending, then `hop 3` before it.
            if start >= 4 && is_st_ending(w[start - 1]) {
                w.truncate(start);
            }
        }
        _ => w.truncate(start),
    }
}

/// Step 3: derivational suffixes.
fn step3(w: &mut Vec<char>, p1: usize, p2: usize) {
    const SUFFIXES: &[&str] = &["end", "ung", "isch", "lich", "heit", "keit", "ig", "ik"];
    let Some((suffix, start)) = longest_suffix(w, SUFFIXES) else {
        return;
    };
    if start < p2 {
        return;
    }
    match suffix {
        "end" | "ung" => {
            w.truncate(start);
            if ends_with(w, "ig") {
                let s = w.len() - 2;
                if s >= p2 && !ends_with(&w[..s], "e") {
                    w.truncate(s);
                }
            }
        }
        "ig" | "ik" | "isch" => {
            if !ends_with(&w[..start], "e") {
                w.truncate(start);
            }
        }
        "lich" | "heit" => {
            w.truncate(start);
            if ends_with(w, "er") || ends_with(w, "en") {
                let s = w.len() - 2;
                if s >= p1 {
                    w.truncate(s);
                }
            }
        }
        _ => {
            // "keit"
            w.truncate(start);
            if let Some((_, s)) = longest_suffix(w, &["lich", "ig"]) {
                if s >= p2 {
                    w.truncate(s);
                }
            }
        }
    }
}

/// Postlude: unmark `U`/`Y` and strip umlauts.
fn postlude(w: &mut [char]) {
    for c in w.iter_mut() {
        *c = match *c {
            'Y' => 'y',
            'U' => 'u',
            'ä' => 'a',
            'ö' => 'o',
            'ü' => 'u',
            other => other,
        };
    }
}

/// Stem a lowercased German word with the Snowball German algorithm.
///
/// Never panics; non-letters pass through unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    prelude(&mut w);
    let (p1, p2) = mark_regions(&w);
    step1(&mut w, p1);
    step2(&mut w, p1);
    step3(&mut w, p1, p2);
    postlude(&mut w);
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

    /// Vectors from the official Snowball German voc.txt / output.txt
    /// (snowball-data `f08c4d63b9bc`).
    #[test]
    fn official_vocabulary() {
        check(&[
            ("aalglatten", "aalglatt"),
            ("aargauer", "aargau"),
            ("aasen", "aas"),
            ("aashöllen", "aasholl"),
            ("abarten", "abart"),
            ("abbeisst", "abbeisst"),
            ("abbildung", "abbild"),
            ("abbrechen", "abbrech"),
            ("abenddämmerung", "abenddammer"),
            ("abendessen", "abendess"),
            ("abendherren", "abendherr"),
            ("abendlichen", "abend"),
            ("abendlichte", "abendlicht"),
            ("abendländischem", "abendland"),
            ("abendröte", "abendrot"),
            ("abends", "abend"),
            ("abendschöppchen", "abendschoppch"),
            ("abendsonne", "abendsonn"),
            ("abendstern", "abend"),
            ("abenteuer", "abenteu"),
            ("abenteuerlich", "abenteu"),
            ("abenteuerliches", "abenteu"),
            ("abenteuerspielplätzen", "abenteuerspielplatz"),
            ("abenteurer", "abenteur"),
            ("aberglauben", "aberglaub"),
            ("abermaligen", "abermal"),
            ("abertausend", "abertaus"),
            ("aberwitzigen", "aberwitz"),
            ("abfallen", "abfall"),
            ("abfeuernd", "abfeuernd"),
            ("abfällige", "abfall"),
            ("abfällsel", "abfallsel"),
            ("abführung", "abfuhr"),
            ("abgebildet", "abgebildet"),
            ("abgebrochenes", "abgebroch"),
            ("abgebürstet", "abgeburstet"),
            ("abgefertigt", "abgefertigt"),
            ("abgehängen", "abgehang"),
            ("abgehärmte", "abgeharmt"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("aa", "aa"),
            ("ab", "ab"),
            ("123", "123"),
            ("--", "--"),
            ("'", "'"),
            ("'s", "'s"),
        ]);
        // Arbitrary Unicode must not panic.
        let _ = stem("日本語テキスト");
        let _ = stem("ßßß");
        let _ = stem("quue");
    }
}
