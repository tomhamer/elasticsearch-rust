//! Snowball Finnish stemmer.
//!
//! A faithful port of the Snowball algorithm described at
//! <https://snowballstem.org/algorithms/finnish/stemmer.html>, operating on
//! a `Vec<char>` of the (already lowercased) input word.
//!
//! Letter classes: the vowels `V` are `a e i o u y ä ö`; the restricted
//! vowels `RV` are `a e i o u ä ö`; a long vowel `LV` is one of
//! `aa ee ii oo uu ää öö`; a consonant `c` is one of `bcdfghjklmnpqrstvwxz`;
//! `Vi` means a restricted vowel followed by `i`.
//!
//! Regions: `R1` is the region after the first non-vowel following a vowel,
//! `R2` the region after the first non-vowel following a vowel in `R1`.
//!
//! Steps:
//!
//! 1. Particles: in `R1`, remove `kin kaan kään ko kö han hän pa pä` if
//!    preceded by `n`, `t` or a vowel; remove `sti` if it is in `R2`.
//! 2. Possessives: in `R1`, remove `si` (unless preceded by `k`); `ni` (and
//!    turn a resulting `kse` into `ksi`); `nsa nsä mme nne`; `an` after
//!    `ta ssa sta lla lta na`; `än` after `tä ssä stä llä ltä nä`; `en` after
//!    `lle ine`.
//! 3. Cases: in `R1`, remove `han hen hin hon hän hön` after the matching
//!    vowel; `siin den tten` after `Vi`; `seen` after `LV`; `a ä` after
//!    consonant+vowel; `tta ttä` after `e`; `ta tä ssa ssä sta stä lla llä
//!    lta ltä lle na nä ksi ine`; and `n`, together with the preceding vowel
//!    when it follows `LV` or `ie`.  Like the Snowball `among` with routine
//!    conditions, a failed `Vi`/`LV` test falls back to the next shorter
//!    suffix (so `kseen` loses `en`); any other failed condition ends the
//!    step.  A flag records whether a case ending was removed.
//! 4. Other endings: in `R2`, remove `mpi mpa mpä mmi mma mmä` (unless
//!    preceded by `po`) and `impi impa impä immi imma immä eja ejä`.
//! 5. Plurals: if step 3 removed an ending, delete a final `i` or `j` in
//!    `R1`; otherwise delete a final `t` in `R1` following a vowel, and then a
//!    final `mma` (unless after `po`) or `imma` in `R2`.
//! 6. Tidying, restricted to `R1`: drop the last letter of a final `LV`; drop
//!    a final `a ä e i` after a consonant; drop the `j` of `oj`/`uj`; drop the
//!    `o` of `jo`.  Then, on the whole word: if it ends with a double
//!    consonant followed by zero or more vowels, remove the second consonant;
//!    finally remove a trailing apostrophe.

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'ä' | 'ö')
}

fn is_restricted_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'ä' | 'ö')
}

fn is_consonant(c: char) -> bool {
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
            | 'p'
            | 'q'
            | 'r'
            | 's'
            | 't'
            | 'v'
            | 'w'
            | 'x'
            | 'z'
    )
}

fn is_aei(c: char) -> bool {
    matches!(c, 'a' | 'ä' | 'e' | 'i')
}

/// True when `w[..end]` ends with a long vowel (`aa ee ii oo uu ää öö`).
fn ends_with_long_vowel(w: &[char], end: usize) -> bool {
    end >= 2 && is_restricted_vowel(w[end - 1]) && w[end - 1] == w[end - 2]
}

/// True when `w[..end]` ends with the literal `s`.
fn ends_with_at(w: &[char], end: usize, s: &str) -> bool {
    let n = s.chars().count();
    end >= n && w[end - n..end].iter().copied().eq(s.chars())
}

/// Returns `(p1, p2)` as indices into `w`; `w.len()` means "empty region".
fn mark_regions(w: &[char]) -> (usize, usize) {
    let n = w.len();
    let mut p1 = n;
    let mut p2 = n;
    let mut i = 0;
    while i < n && !is_vowel(w[i]) {
        i += 1;
    }
    while i < n && is_vowel(w[i]) {
        i += 1;
    }
    if i < n {
        i += 1;
        p1 = i;
        while i < n && !is_vowel(w[i]) {
            i += 1;
        }
        while i < n && is_vowel(w[i]) {
            i += 1;
        }
        if i < n {
            i += 1;
            p2 = i;
        }
    }
    (p1, p2)
}

/// Longest entry of `suffixes` that `w` ends with and that lies entirely at or
/// after index `limit`; returns the suffix length.
fn longest_suffix(w: &[char], limit: usize, suffixes: &[&str]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (idx, s) in suffixes.iter().enumerate() {
        let n = s.chars().count();
        if n > w.len() || w.len() - n < limit {
            continue;
        }
        if best.is_some_and(|(_, bn)| bn >= n) {
            continue;
        }
        if w[w.len() - n..].iter().copied().eq(s.chars()) {
            best = Some((idx, n));
        }
    }
    best
}

const PARTICLES: &[&str] = &["kin", "kaan", "kään", "ko", "kö", "han", "hän", "pa", "pä", "sti"];

fn particle_etc(w: &mut Vec<char>, p1: usize, p2: usize) {
    let Some((idx, n)) = longest_suffix(w, p1, PARTICLES) else {
        return;
    };
    let start = w.len() - n;
    let ok = if PARTICLES[idx] == "sti" {
        start >= p2
    } else {
        start >= 1 && (is_vowel(w[start - 1]) || matches!(w[start - 1], 'n' | 't'))
    };
    if ok {
        w.truncate(start);
    }
}

const POSSESSIVES: &[&str] = &["si", "ni", "nsa", "nsä", "mme", "nne", "an", "än", "en"];

fn possessive(w: &mut Vec<char>, p1: usize) {
    let Some((idx, n)) = longest_suffix(w, p1, POSSESSIVES) else {
        return;
    };
    let start = w.len() - n;
    let preceded_by_any =
        |list: &[&str]| list.iter().any(|s| ends_with_at(w, start, s));
    match POSSESSIVES[idx] {
        "si" => {
            if !ends_with_at(w, start, "k") {
                w.truncate(start);
            }
        }
        "ni" => {
            w.truncate(start);
            if ends_with_at(w, w.len(), "kse") {
                let last = w.len() - 1;
                w[last] = 'i';
            }
        }
        "an" => {
            if preceded_by_any(&["ta", "ssa", "sta", "lla", "lta", "na"]) {
                w.truncate(start);
            }
        }
        "än" => {
            if preceded_by_any(&["tä", "ssä", "stä", "llä", "ltä", "nä"]) {
                w.truncate(start);
            }
        }
        "en" => {
            if preceded_by_any(&["lle", "ine"]) {
                w.truncate(start);
            }
        }
        _ => w.truncate(start),
    }
}

/// Condition attached to a case ending.
#[derive(Clone, Copy)]
enum CaseKind {
    /// `hXn`: must be preceded by the given vowel.
    After(char),
    /// Must be preceded by `Vi` (or an apostrophe); falls back when it fails.
    Vi,
    /// Must be preceded by a long vowel; falls back when it fails.
    Long,
    /// `a ä`: must be preceded by consonant + vowel.
    ConsonantVowel,
    /// `tta ttä`: must be preceded by `e`.
    AfterE,
    /// Removed unconditionally.
    Plain,
    /// `n`: also removes the preceding vowel after `LV` or `ie`.
    N,
}

/// Case endings ordered by decreasing length, so the first match is the
/// longest one (the Snowball `among` semantics).
const CASE_ENDINGS: &[(&str, CaseKind)] = &[
    ("siin", CaseKind::Vi),
    ("tten", CaseKind::Vi),
    ("seen", CaseKind::Long),
    ("han", CaseKind::After('a')),
    ("hen", CaseKind::After('e')),
    ("hin", CaseKind::After('i')),
    ("hon", CaseKind::After('o')),
    ("hän", CaseKind::After('ä')),
    ("hön", CaseKind::After('ö')),
    ("den", CaseKind::Vi),
    ("tta", CaseKind::AfterE),
    ("ttä", CaseKind::AfterE),
    ("ssa", CaseKind::Plain),
    ("ssä", CaseKind::Plain),
    ("sta", CaseKind::Plain),
    ("stä", CaseKind::Plain),
    ("lla", CaseKind::Plain),
    ("llä", CaseKind::Plain),
    ("lta", CaseKind::Plain),
    ("ltä", CaseKind::Plain),
    ("lle", CaseKind::Plain),
    ("ksi", CaseKind::Plain),
    ("ine", CaseKind::Plain),
    ("ta", CaseKind::Plain),
    ("tä", CaseKind::Plain),
    ("na", CaseKind::Plain),
    ("nä", CaseKind::Plain),
    ("a", CaseKind::ConsonantVowel),
    ("ä", CaseKind::ConsonantVowel),
    ("n", CaseKind::N),
];

/// Step 3; returns whether an ending was removed.
fn case_ending(w: &mut Vec<char>, p1: usize) -> bool {
    let mut chosen = None;
    for (suffix, kind) in CASE_ENDINGS {
        let n = suffix.chars().count();
        if n > w.len() || w.len() - n < p1 || !ends_with_at(w, w.len(), suffix) {
            continue;
        }
        let start = w.len() - n;
        let accepted = match kind {
            CaseKind::Vi => {
                (start >= 2 && w[start - 1] == 'i' && is_restricted_vowel(w[start - 2]))
                    || ends_with_at(w, start, "'")
            }
            CaseKind::Long => ends_with_long_vowel(w, start),
            _ => true,
        };
        if accepted {
            chosen = Some((start, *kind));
            break;
        }
    }
    let Some((start, kind)) = chosen else {
        return false;
    };
    let mut cut = start;
    match kind {
        CaseKind::After(v) => {
            if !(start >= 1 && w[start - 1] == v) {
                return false;
            }
        }
        CaseKind::ConsonantVowel => {
            if !(start >= 2 && is_vowel(w[start - 1]) && is_consonant(w[start - 2])) {
                return false;
            }
        }
        CaseKind::AfterE => {
            if !ends_with_at(w, start, "e") {
                return false;
            }
        }
        CaseKind::N => {
            if ends_with_long_vowel(w, start) || ends_with_at(w, start, "ie") {
                cut = start - 1;
            }
        }
        CaseKind::Vi | CaseKind::Long | CaseKind::Plain => {}
    }
    w.truncate(cut);
    true
}

const OTHER_ENDINGS: &[&str] = &[
    "mpi", "mpa", "mpä", "mmi", "mma", "mmä", "impi", "impa", "impä", "immi", "imma", "immä",
    "eja", "ejä",
];

fn other_endings(w: &mut Vec<char>, p2: usize) {
    let Some((idx, n)) = longest_suffix(w, p2, OTHER_ENDINGS) else {
        return;
    };
    let start = w.len() - n;
    let needs_not_po = idx < 6;
    if needs_not_po && ends_with_at(w, start, "po") {
        return;
    }
    w.truncate(start);
}

fn i_plural(w: &mut Vec<char>, p1: usize) {
    if w.len() > p1 && matches!(w.last(), Some('i' | 'j')) {
        w.pop();
    }
}

fn t_plural(w: &mut Vec<char>, p1: usize, p2: usize) {
    let n = w.len();
    if !(n >= 2 && n - 2 >= p1 && w[n - 1] == 't' && is_vowel(w[n - 2])) {
        return;
    }
    w.pop();
    if let Some((idx, n)) = longest_suffix(w, p2, &["mma", "imma"]) {
        let start = w.len() - n;
        if idx == 0 && ends_with_at(w, start, "po") {
            return;
        }
        w.truncate(start);
    }
}

fn tidy(w: &mut Vec<char>, p1: usize) {
    // (a) R1 ends LV: drop the last letter
    let n = w.len();
    if n >= 2 && n - 2 >= p1 && ends_with_long_vowel(w, n) {
        w.pop();
    }
    // (b) R1 ends cX, X in {a ä e i}: drop the last letter
    let n = w.len();
    if n >= 2 && n - 2 >= p1 && is_aei(w[n - 1]) && is_consonant(w[n - 2]) {
        w.pop();
    }
    // (c) R1 ends oj or uj: drop the j
    let n = w.len();
    if n >= 2 && n - 2 >= p1 && w[n - 1] == 'j' && matches!(w[n - 2], 'o' | 'u') {
        w.pop();
    }
    // (d) R1 ends jo: drop the o
    let n = w.len();
    if n >= 2 && n - 2 >= p1 && w[n - 1] == 'o' && w[n - 2] == 'j' {
        w.pop();
    }
    // (e) double consonant followed by zero or more vowels: remove the second
    let mut i = w.len();
    while i > 0 && is_vowel(w[i - 1]) {
        i -= 1;
    }
    if i >= 2 && is_consonant(w[i - 1]) && w[i - 1] == w[i - 2] {
        w.remove(i - 1);
    }
    // (f) trailing apostrophe
    if w.last() == Some(&'\'') {
        w.pop();
    }
}

/// Stem a lowercased Finnish word with the Snowball Finnish algorithm.
///
/// Never panics; words without vowels (digits, punctuation, ...) have empty
/// regions and pass through essentially unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    let (p1, p2) = mark_regions(&w);
    particle_etc(&mut w, p1, p2);
    possessive(&mut w, p1);
    let ending_removed = case_ending(&mut w, p1);
    other_endings(&mut w, p2);
    if ending_removed {
        i_plural(&mut w, p1);
    } else {
        t_plural(&mut w, p1, p2);
    }
    tidy(&mut w, p1);
    w.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(cases: &[(&str, &str)]) {
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|(input, expected)| {
                let got = stem(input);
                (got != *expected).then(|| format!("{input} -> {got} (expected {expected})"))
            })
            .collect();
        assert!(failures.is_empty(), "failing words:\n{}", failures.join("\n"));
    }

    #[test]
    fn snowball_sample_edeltaja() {
        check(&[
            ("edeltäjien", "edeltäj"),
            ("edeltäjiensä", "edeltäjie"),
            ("edeltäjiinsä", "edeltäj"),
            ("edeltäjistään", "edeltäj"),
            ("edeltäjiä", "edeltäj"),
            ("edeltäjiään", "edeltäjiä"),
            ("edeltäjä", "edeltäj"),
            ("edeltäjälleen", "edeltäj"),
            ("edeltäjän", "edeltäj"),
            ("edeltäjäni", "edeltäj"),
            ("edeltäjänsä", "edeltäj"),
            ("edeltäjänä", "edeltäj"),
            ("edeltäjässä", "edeltäj"),
            ("edeltäjästä", "edeltäj"),
            ("edeltäjästään", "edeltäj"),
            ("edeltäjät", "edeltäj"),
            ("edeltäjää", "edeltäj"),
            ("edeltäjään", "edeltäj"),
            ("edeltäjäänsä", "edeltäj"),
            ("edeltäneelle", "edeltän"),
            ("edeltäneellä", "edeltän"),
            ("edeltäneeltä", "edeltän"),
            ("edeltäneen", "edeltän"),
            ("edeltäneenä", "edeltän"),
            ("edeltäneeseen", "edeltän"),
            ("edeltäneessä", "edeltän"),
            ("edeltäneestä", "edeltän"),
            ("edeltäneet", "edeltän"),
            ("edeltäneiden", "edeltän"),
            ("edeltäneissä", "edeltän"),
            ("edeltäneitä", "edeltän"),
            ("edeltänyt", "edeltäny"),
            ("edeltänyttä", "edeltänyt"),
            ("edeltävien", "edeltäv"),
            ("edeltäviin", "edeltäv"),
            ("edeltävinä", "edeltäv"),
            ("edeltävissä", "edeltäv"),
            ("edeltävä", "edeltäv"),
            ("edeltävälle", "edeltäv"),
            ("edeltävällä", "edeltäv"),
        ]);
    }

    #[test]
    fn snowball_sample_innostu() {
        check(&[
            ("innostu", "innostu"),
            ("innostua", "innostu"),
            ("innostuessaan", "innostue"),
            ("innostui", "innostui"),
            ("innostuimme", "innostui"),
            ("innostuin", "innostu"),
            ("innostuisi", "innostui"),
            ("innostuisivat", "innostuisiv"),
            ("innostuivat", "innostuiv"),
            ("innostukseen", "innostuks"),
            ("innostuksella", "innostuks"),
            ("innostuksen", "innostuks"),
            ("innostuksensa", "innostuks"),
            ("innostuksessa", "innostuks"),
            ("innostuksessaan", "innostuks"),
            ("innostuksesta", "innostuks"),
            ("innostuksissaan", "innostuks"),
            ("innostumaan", "innostum"),
            ("innostuminen", "innostumin"),
            ("innostun", "innostu"),
            ("innostuneelle", "innostun"),
            ("innostuneempia", "innostun"),
            ("innostuneen", "innostun"),
            ("innostuneena", "innostun"),
            ("innostuneesta", "innostun"),
            ("innostuneesti", "innostun"),
            ("innostuneet", "innostun"),
            ("innostuneiden", "innostun"),
            ("innostuneiksi", "innostun"),
            ("innostunein", "innostun"),
            ("innostuneina", "innostun"),
            ("innostuneissa", "innostun"),
            ("innostuneisuus", "innostuneisuus"),
            ("innostuneita", "innostun"),
            ("innostus", "innostus"),
            ("innostusta", "innostu"),
            ("innostustaan", "innostu"),
            ("innostutaan", "innostu"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("i", "i"),
            ("n", "n"),
            ("t", "t"),
            ("tt", "t"),
            ("123", "123"),
            ("!?.,", "!?.,"),
            ("ää", "ää"),
            ("'", ""),
            ("日本語", "日本語"),
        ]);
        for s in ["kin", "sti", "ni", "kseni", "an", "seen", "siin", "ien", "mma", "oj", "jo", "\u{0}", "é", "x1y"] {
            let _ = stem(s);
        }
    }
}
