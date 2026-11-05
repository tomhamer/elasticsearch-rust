//! Snowball Dutch stemmer (Martin Porter's algorithm as documented on
//! snowballstem.org; `dutch_porter.sbl` in the Snowball sources).
//!
//! Steps:
//!
//! 1. **Prelude.** Acute and diaeresis accents are stripped from
//!    `a e i o u` (`è` is kept and counts as a vowel). An initial `y`, a `y`
//!    after a vowel, and an `i` between vowels are marked as `Y` / `I` so
//!    they are treated as consonants.
//! 2. **Regions.** R1 is the region after the first non-vowel following a
//!    vowel, adjusted so that at least 3 letters precede it; R2 is the same
//!    construction applied again inside R1. Vowels are `a e i o u y è`.
//! 3. **Step 1.** Longest of `heden en ene s se`: `heden→heid` in R1;
//!    `en`/`ene` deleted in R1 after a valid en-ending (a non-vowel, not
//!    `gem`) followed by undoubling of `kk dd tt`; `s`/`se` deleted in R1
//!    after a valid s-ending (a non-vowel other than `j`).
//! 4. **Step 2.** A final `e` in R1 after a non-vowel is deleted and the
//!    ending undoubled; the deletion is remembered for step 3b.
//! 5. **Step 3a.** `heid` in R2 not after `c` is deleted, and a preceding
//!    `en` is treated as in step 1.
//! 6. **Step 3b.** Longest of `end ing ig lijk baar bar` is removed in R2,
//!    with the follow-ups described in the algorithm (`ig` after `end`/`ing`,
//!    step 2 again after `lijk`, `bar` only if step 2 removed an `e`).
//! 7. **Step 4.** A doubled `aa ee oo uu` between a non-vowel and a final
//!    non-vowel (other than `I`) is shortened to a single vowel.
//! 8. **Postlude.** `I`/`Y` are lowered again.
//!
//! The input is expected to be lowercased already; anything that is not a
//! letter the algorithm knows about simply passes through untouched.

/// Vowels of the Dutch algorithm (`I` and `Y` are deliberately not vowels).
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'è')
}

/// Does `w` end with `suffix`?
fn ends_with(w: &[char], suffix: &str) -> bool {
    let n = suffix.chars().count();
    n <= w.len() && w[w.len() - n..].iter().copied().eq(suffix.chars())
}

/// Returns the longest of `suffixes` that `w` ends with, with its start index.
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

/// Prelude: strip accents and mark `y`/`i` as consonants where required.
fn prelude(w: &mut [char]) {
    for c in w.iter_mut() {
        *c = match *c {
            'ä' | 'á' => 'a',
            'ë' | 'é' => 'e',
            'ï' | 'í' => 'i',
            'ö' | 'ó' => 'o',
            'ü' | 'ú' => 'u',
            other => other,
        };
    }
    if w.first() == Some(&'y') {
        w[0] = 'Y';
    }
    // repeat ( gopast v  try ( ['i'] followed by v -> 'I'  or  ['y'] -> 'Y' ) )
    let n = w.len();
    let mut i = 0;
    loop {
        while i < n && !is_vowel(w[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        i += 1;
        if i < n {
            match w[i] {
                'i' => {
                    if i + 1 < n && is_vowel(w[i + 1]) {
                        w[i] = 'I';
                    }
                    i += 1;
                }
                'y' => {
                    w[i] = 'Y';
                    i += 1;
                }
                _ => {}
            }
        }
    }
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

/// Remove the last letter if the word ends in `kk`, `dd` or `tt`.
fn undouble(w: &mut Vec<char>) {
    if ends_with(w, "kk") || ends_with(w, "dd") || ends_with(w, "tt") {
        w.pop();
    }
}

/// Delete a final `e` in R1 preceded by a non-vowel, then undouble.
/// Returns whether an `e` was removed.
fn e_ending(w: &mut Vec<char>, p1: usize) -> bool {
    let n = w.len();
    if n >= 2 && w[n - 1] == 'e' && n - 1 >= p1 && !is_vowel(w[n - 2]) {
        w.pop();
        undouble(w);
        true
    } else {
        false
    }
}

/// Delete the suffix starting at `start` (an `en`/`ene`) if it is in R1 and
/// preceded by a valid en-ending, then undouble.
fn en_ending(w: &mut Vec<char>, start: usize, p1: usize) {
    if start >= p1 && start > 0 && !is_vowel(w[start - 1]) && !ends_with(&w[..start], "gem") {
        w.truncate(start);
        undouble(w);
    }
}

/// Step 1.
fn step1(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &["heden", "ene", "en", "se", "s"];
    let Some((suffix, start)) = longest_suffix(w, SUFFIXES) else {
        return;
    };
    match suffix {
        "heden" => {
            if start >= p1 {
                w.truncate(start);
                w.extend("heid".chars());
            }
        }
        "en" | "ene" => en_ending(w, start, p1),
        _ => {
            // "s" | "se"
            if start >= p1 && start > 0 && !is_vowel(w[start - 1]) && w[start - 1] != 'j' {
                w.truncate(start);
            }
        }
    }
}

/// Step 3a: `heid` in R2 not after `c`; a preceding `en` as in step 1.
fn step3a(w: &mut Vec<char>, p1: usize, p2: usize) {
    if ends_with(w, "heid") {
        let start = w.len() - 4;
        if start >= p2 && !ends_with(&w[..start], "c") {
            w.truncate(start);
            if ends_with(w, "en") {
                let s = w.len() - 2;
                en_ending(w, s, p1);
            }
        }
    }
}

/// Step 3b: derivational suffixes.
fn step3b(w: &mut Vec<char>, p1: usize, p2: usize, e_found: bool) {
    const SUFFIXES: &[&str] = &["end", "ing", "lijk", "baar", "bar", "ig"];
    let Some((suffix, start)) = longest_suffix(w, SUFFIXES) else {
        return;
    };
    if start < p2 {
        return;
    }
    match suffix {
        "end" | "ing" => {
            w.truncate(start);
            if ends_with(w, "ig") && w.len() - 2 >= p2 && !ends_with(&w[..w.len() - 2], "e") {
                let s = w.len() - 2;
                w.truncate(s);
            } else {
                undouble(w);
            }
        }
        "ig" => {
            if !ends_with(&w[..start], "e") {
                w.truncate(start);
            }
        }
        "lijk" => {
            w.truncate(start);
            e_ending(w, p1);
        }
        "baar" => w.truncate(start),
        _ => {
            // "bar"
            if e_found {
                w.truncate(start);
            }
        }
    }
}

/// Step 4: undouble a vowel in a final `C V V D` sequence.
fn step4(w: &mut Vec<char>) {
    let n = w.len();
    if n < 4 {
        return;
    }
    let d = w[n - 1];
    if is_vowel(d) || d == 'I' {
        return;
    }
    let v = w[n - 2];
    if v != w[n - 3] || !matches!(v, 'a' | 'e' | 'o' | 'u') {
        return;
    }
    if is_vowel(w[n - 4]) {
        return;
    }
    w.remove(n - 2);
}

/// Postlude: unmark `I`/`Y`.
fn postlude(w: &mut [char]) {
    for c in w.iter_mut() {
        *c = match *c {
            'Y' => 'y',
            'I' => 'i',
            other => other,
        };
    }
}

/// Stem a lowercased Dutch word with the Snowball Dutch algorithm.
///
/// Never panics; non-letters pass through unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    prelude(&mut w);
    let (p1, p2) = mark_regions(&w);
    step1(&mut w, p1);
    let e_found = e_ending(&mut w, p1);
    step3a(&mut w, p1, p2);
    step3b(&mut w, p1, p2, e_found);
    step4(&mut w);
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

    /// Vectors taken from the official Snowball Dutch voc.txt / output.txt.
    #[test]
    fn official_vocabulary() {
        check(&[
            ("á", "a"),
            ("à", "à"),
            ("aachen", "aach"),
            ("aachener", "aachener"),
            ("aalders", "aalder"),
            ("aalmoezen", "aalmoez"),
            ("aalscholvers", "aalscholver"),
            ("aalsmeer", "aalsmer"),
            ("aalsmeerbaan", "aalsmeerban"),
            ("aalsmeerse", "aalsmer"),
            ("aalten", "aalt"),
            ("aaltenaar", "aaltenar"),
            ("aaltjes", "aaltjes"),
            ("áan", "aan"),
            ("aanbaklaagjes", "aanbaklaagjes"),
            ("aanbelde", "aanbeld"),
            ("aanbellen", "aanbell"),
            ("aanbesteden", "aanbested"),
            ("aanbesteding", "aanbested"),
            ("aanbestedingen", "aanbested"),
            ("aanbestedingsdossier", "aanbestedingsdossier"),
            ("lichamelijk", "licham"),
            ("lichamelijke", "licham"),
            ("lichamelijkheden", "licham"),
            ("lichtgrijs", "lichtgrijs"),
            ("opgrijnzen", "opgrijnz"),
            ("opgrijzende", "opgrijz"),
            ("komische", "komisch"),
            ("kommandeerde", "kommandeerd"),
            ("kommetje", "kommetj"),
            ("kompressoren", "kompressor"),
            ("kón", "kon"),
            ("kòn", "kòn"),
            ("konden", "kond"),
            ("kònden", "kònden"),
            ("kondigde", "kondigd"),
            ("kondigden", "kondigd"),
            ("kondigen", "kondig"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("aa", "aa"),
            ("y", "y"),
            ("ij", "ij"),
            ("123", "123"),
            ("--", "--"),
        ]);
        let _ = stem("日本語テキスト");
        let _ = stem("iiiii");
        let _ = stem("yyyy");
    }
}
