//! Snowball Danish stemmer.
//!
//! Faithful port of `danish.sbl` from the Snowball project. Only R1 is used:
//!
//! * **Region.** If the word contains an apostrophe, R1 starts after the
//!   first one (acronym loanwords such as `pc'en`); otherwise R1 is the
//!   region after the first non-vowel following a vowel. R1 is then adjusted
//!   so that at least 3 characters precede it. Vowels are `a e i o u y æ å ø`.
//!   Every suffix search below is restricted to R1 (longest match inside R1).
//! * **Step 1.** The longest of `hed ethed ered e erede ende erende ene erne
//!   ere en heden eren er heder erer heds es endes erendes enes ernes eres
//!   ens hedens erens ers ets erets et eret` is deleted; `s` is deleted when
//!   preceded by a valid s-ending (`a b c d f g h j k l m n o p r t v y z å '`).
//! * **Step 2.** If the word ends `gd dt gt kt` in R1, the last letter is
//!   deleted.
//! * **Step 3.** A trailing `igst` loses its `st`. Then the longest of
//!   `ig lig elig els` in R1 is deleted and step 2 repeated; `løst` becomes
//!   `løs`.
//! * **Step 4.** A doubled final consonant from `b d f g k l m n p r s t`
//!   whose last letter is in R1 is undoubled.
//! * **Step 5.** A trailing apostrophe is removed.
//!
//! Words shorter than three characters are returned unchanged, as in the
//! reference implementation. The input is expected to be lowercased already;
//! anything that is not a letter the algorithm knows about passes through.

/// Vowels of the Danish algorithm.
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'æ' | 'å' | 'ø')
}

/// Valid s-ending.
fn is_s_ending(c: char) -> bool {
    matches!(
        c,
        'a' | 'b'
            | 'c'
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
            | 'z'
            | 'å'
            | '\''
    )
}

/// Consonants undoubled by step 4.
fn is_undouble_c(c: char) -> bool {
    matches!(
        c,
        'b' | 'd' | 'f' | 'g' | 'k' | 'l' | 'm' | 'n' | 'p' | 'r' | 's' | 't'
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
        if len <= w.len() - p1
            && ends_with(w, s)
            && best.is_none_or(|(_, start)| w.len() - len < start)
        {
            best = Some((s, w.len() - len));
        }
    }
    best
}

/// Compute the start of R1 (`len` if absent). Caller guarantees `len >= 3`.
fn mark_r1(w: &[char]) -> usize {
    let n = w.len();
    let p1 = if let Some(apos) = w.iter().position(|&c| c == '\'') {
        apos + 1
    } else {
        let mut i = 0;
        while i < n && !is_vowel(w[i]) {
            i += 1;
        }
        while i < n && is_vowel(w[i]) {
            i += 1;
        }
        if i < n { i + 1 } else { n }
    };
    p1.max(3)
}

/// Step 1: main suffixes.
fn main_suffix(w: &mut Vec<char>, p1: usize) {
    const SUFFIXES: &[&str] = &[
        "hed", "ethed", "ered", "e", "erede", "ende", "erende", "ene", "erne", "ere", "en",
        "heden", "eren", "er", "heder", "erer", "heds", "es", "endes", "erendes", "enes",
        "ernes", "eres", "ens", "hedens", "erens", "ers", "ets", "erets", "et", "eret", "s",
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

/// Step 2: `gd dt gt kt` in R1 lose their last letter.
fn consonant_pair(w: &mut Vec<char>, p1: usize) {
    if longest_suffix_in_r1(w, p1, &["gd", "dt", "gt", "kt"]).is_some() {
        w.pop();
    }
}

/// Step 3: other suffixes.
fn other_suffix(w: &mut Vec<char>, p1: usize) {
    if ends_with(w, "igst") {
        w.truncate(w.len() - 2);
    }
    let Some((suffix, start)) = longest_suffix_in_r1(w, p1, &["ig", "lig", "elig", "els", "løst"])
    else {
        return;
    };
    if suffix == "løst" {
        w.pop();
    } else {
        w.truncate(start);
        consonant_pair(w, p1);
    }
}

/// Step 4: undouble a final doubled consonant whose last letter is in R1.
fn undouble(w: &mut Vec<char>, p1: usize) {
    let n = w.len();
    if n >= 2 && n - 1 >= p1 && is_undouble_c(w[n - 1]) && w[n - 2] == w[n - 1] {
        w.pop();
    }
}

/// Stem a lowercased Danish word with the Snowball Danish algorithm.
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
    undouble(&mut w, p1);
    if w.last() == Some(&'\'') {
        w.pop();
    }
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

    /// Vectors taken from the official Snowball Danish voc.txt / output.txt.
    #[test]
    fn official_vocabulary() {
        check(&[
            ("aab", "aab"),
            ("aabenbaringen", "aabenbaring"),
            ("abaddon", "abaddon"),
            ("abarimbjergene", "abarimbjerg"),
            ("abarimbjerget", "abarimbjerg"),
            ("abba", "abba"),
            ("abdas", "abda"),
            ("abdeels", "abdeel"),
            ("abdis", "abdis"),
            ("abels", "abel"),
            ("aber", "aber"),
            ("abias", "abia"),
            ("abiathar", "abiathar"),
            ("abibajils", "abibajil"),
            ("abidans", "abidan"),
            ("abiezer", "abiez"),
            ("abiezers", "abiez"),
            ("abiezriten", "abiezrit"),
            ("abiezriterne", "abiezrit"),
            ("linnedbenklæderne", "linnedbenklæd"),
            ("linnedbælte", "linnedbælt"),
            ("linnede", "linned"),
            ("linnedhuer", "linnedhu"),
            ("linnedkjortel", "linnedkjortel"),
            ("linnedkjortler", "linnedkjortl"),
            ("linnedklæderne", "linnedklæd"),
            ("linnedklædning", "linnedklædning"),
            ("linnedvæveriets", "linnedvæveri"),
            ("linser", "lins"),
            ("liste", "list"),
            ("listede", "listed"),
            ("listen", "list"),
            ("listet", "list"),
            ("listiges", "list"),
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
            ("'''", "''"),
        ]);
        let _ = stem("日本語テキスト");
        let _ = stem("pc'en");
    }
}
