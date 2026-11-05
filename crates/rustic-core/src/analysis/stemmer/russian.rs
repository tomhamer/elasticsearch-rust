//! Snowball Russian stemmer.
//!
//! A faithful port of the Snowball algorithm described at
//! <https://snowballstem.org/algorithms/russian/stemmer.html>, operating on
//! a `Vec<char>` of the (already lowercased) input word.
//!
//! Before anything else every `ё` is mapped to `е`, exactly as the Snowball
//! source does (`goto (['ё']) <- 'е'`); the suffix tables below are written
//! with `е` only.
//!
//! Regions: `RV` is the region after the first vowel; `R1` is the region after
//! the first non-vowel following a vowel; `R2` is the region after the first
//! non-vowel following a vowel in `R1`.  All suffix searches below are
//! restricted to `RV`, including the "preceded by а or я" tests.
//!
//! Steps (all performed in `RV`):
//!
//! 1. Search for a PERFECTIVE GERUND ending; if found remove it and finish
//!    step 1.  Otherwise try to remove a REFLEXIVE ending, then search in turn
//!    for an ADJECTIVAL (adjective, optionally preceded by a participle), a
//!    VERB or a NOUN ending; remove the first one found.
//! 2. If the word ends with `и`, remove it.
//! 3. Search for a DERIVATIONAL ending (`ост`, `ость`) in `R2`; remove it.
//! 4. Undouble `н`; or, if the word ends with a SUPERLATIVE ending (`ейш`,
//!    `ейше`), remove it and then undouble `н`; or, if the word ends with `ь`,
//!    remove it.
//!
//! Non-Cyrillic input contains no vowels in the stemmer's sense, so `RV` is
//! empty and such words pass through unchanged.

const VOWELS: [char; 9] = ['а', 'е', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я'];

/// Perfective gerund endings that must be preceded by `а` or `я`.
const PERFECTIVE_GERUND_1: &[&str] = &["в", "вши", "вшись"];
/// Perfective gerund endings removed unconditionally.
const PERFECTIVE_GERUND_2: &[&str] =
    &["ив", "ивши", "ившись", "ыв", "ывши", "ывшись"];

const ADJECTIVE: &[&str] = &[
    "ее", "ие", "ые", "ое", "ими", "ыми", "ей", "ий", "ый", "ой", "ем", "им",
    "ым", "ом", "его", "ого", "ему", "ому", "их", "ых", "ую", "юю", "ая", "яя",
    "ою", "ею",
];

/// Participle endings that must be preceded by `а` or `я`.
const PARTICIPLE_1: &[&str] = &["ем", "нн", "вш", "ющ", "щ"];
/// Participle endings removed unconditionally.
const PARTICIPLE_2: &[&str] = &["ивш", "ывш", "ующ"];

const REFLEXIVE: &[&str] = &["ся", "сь"];

/// Verb endings that must be preceded by `а` or `я`.
const VERB_1: &[&str] = &[
    "ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но", "ет", "ют",
    "ны", "ть", "ешь", "нно",
];
/// Verb endings removed unconditionally.
const VERB_2: &[&str] = &[
    "ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей", "уй", "ил",
    "ыл", "им", "ым", "ен", "ило", "ыло", "ено", "ят", "ует", "уют", "ит", "ыт",
    "ены", "ить", "ыть", "ишь", "ую", "ю",
];

const NOUN: &[&str] = &[
    "а", "ев", "ов", "ие", "ье", "е", "иями", "ями", "ами", "еи", "ии", "и",
    "ией", "ей", "ой", "ий", "й", "иям", "ям", "ием", "ем", "ам", "ом", "о",
    "у", "ах", "иях", "ях", "ы", "ь", "ию", "ью", "ю", "ия", "ья", "я",
];

const SUPERLATIVE: &[&str] = &["ейш", "ейше"];
const DERIVATIONAL: &[&str] = &["ост", "ость"];

fn is_vowel(c: char) -> bool {
    VOWELS.contains(&c)
}

/// Returns `(rv, r2)` as indices into `w`; `w.len()` means "empty region".
fn mark_regions(w: &[char]) -> (usize, usize) {
    let n = w.len();
    let mut rv = n;
    let mut r2 = n;
    let mut i = 0;
    // gopast v, setmark pV
    while i < n && !is_vowel(w[i]) {
        i += 1;
    }
    if i < n {
        i += 1;
        rv = i;
        // gopast non-v (end of R1 start), gopast v, gopast non-v, setmark p2
        while i < n && is_vowel(w[i]) {
            i += 1;
        }
        if i < n {
            i += 1;
            while i < n && !is_vowel(w[i]) {
                i += 1;
            }
            if i < n {
                i += 1;
                while i < n && is_vowel(w[i]) {
                    i += 1;
                }
                if i < n {
                    i += 1;
                    r2 = i;
                }
            }
        }
    }
    (rv, r2)
}

/// Longest entry of `suffixes` that `w` ends with and that lies entirely at or
/// after index `limit`.
fn longest_suffix<'a>(w: &[char], limit: usize, suffixes: &[&'a str]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for s in suffixes {
        let n = s.chars().count();
        if n > w.len() || w.len() - n < limit {
            continue;
        }
        if best.is_some_and(|(_, bn)| bn >= n) {
            continue;
        }
        if w[w.len() - n..].iter().copied().eq(s.chars()) {
            best = Some((s, n));
        }
    }
    best.map(|(s, _)| s)
}

/// Longest match across two suffix groups; the flag is `true` when the match
/// came from `group1`.
fn longest_in_groups<'a>(
    w: &[char],
    limit: usize,
    group1: &[&'a str],
    group2: &[&'a str],
) -> Option<(&'a str, bool)> {
    let a = longest_suffix(w, limit, group1);
    let b = longest_suffix(w, limit, group2);
    match (a, b) {
        (Some(x), Some(y)) => {
            if x.chars().count() >= y.chars().count() {
                Some((x, true))
            } else {
                Some((y, false))
            }
        }
        (Some(x), None) => Some((x, true)),
        (None, Some(y)) => Some((y, false)),
        (None, None) => None,
    }
}

/// True when the character just before the final `suffix_len` characters is
/// `а` or `я` and lies within `RV`.
fn preceded_by_a_or_ya(w: &[char], rv: usize, suffix_len: usize) -> bool {
    let start = w.len() - suffix_len;
    start > rv && matches!(w[start - 1], 'а' | 'я')
}

/// Removes an ending from `group1` (only if preceded by `а`/`я` in RV) or
/// `group2`; returns whether anything was removed.
fn remove_grouped(w: &mut Vec<char>, rv: usize, group1: &[&str], group2: &[&str]) -> bool {
    let Some((suffix, conditional)) = longest_in_groups(w, rv, group1, group2) else {
        return false;
    };
    let n = suffix.chars().count();
    if conditional && !preceded_by_a_or_ya(w, rv, n) {
        return false;
    }
    w.truncate(w.len() - n);
    true
}

fn remove_plain(w: &mut Vec<char>, rv: usize, suffixes: &[&str]) -> bool {
    let Some(suffix) = longest_suffix(w, rv, suffixes) else {
        return false;
    };
    w.truncate(w.len() - suffix.chars().count());
    true
}

fn adjectival(w: &mut Vec<char>, rv: usize) -> bool {
    if !remove_plain(w, rv, ADJECTIVE) {
        return false;
    }
    // try the participle
    remove_grouped(w, rv, PARTICIPLE_1, PARTICIPLE_2);
    true
}

fn step1(w: &mut Vec<char>, rv: usize) {
    if remove_grouped(w, rv, PERFECTIVE_GERUND_1, PERFECTIVE_GERUND_2) {
        return;
    }
    remove_plain(w, rv, REFLEXIVE);
    if adjectival(w, rv) {
        return;
    }
    if remove_grouped(w, rv, VERB_1, VERB_2) {
        return;
    }
    remove_plain(w, rv, NOUN);
}

fn step2(w: &mut Vec<char>, rv: usize) {
    if w.len() > rv && w.last() == Some(&'и') {
        w.pop();
    }
}

fn derivational(w: &mut Vec<char>, rv: usize, r2: usize) {
    if let Some(suffix) = longest_suffix(w, rv, DERIVATIONAL) {
        let n = suffix.chars().count();
        if w.len() - n >= r2 {
            w.truncate(w.len() - n);
        }
    }
}

/// Removes one `н` of a final `нн` lying within RV.
fn undouble_n(w: &mut Vec<char>, rv: usize) {
    let n = w.len();
    if n >= 2 && n - 2 >= rv && w[n - 1] == 'н' && w[n - 2] == 'н' {
        w.pop();
    }
}

fn tidy_up(w: &mut Vec<char>, rv: usize) {
    let candidates: [&str; 4] = ["ейш", "ейше", "н", "ь"];
    let Some(suffix) = longest_suffix(w, rv, &candidates) else {
        return;
    };
    if SUPERLATIVE.contains(&suffix) {
        w.truncate(w.len() - suffix.chars().count());
        undouble_n(w, rv);
    } else if suffix == "н" {
        undouble_n(w, rv);
    } else {
        w.pop();
    }
}

/// Stem a lowercased Russian word with the Snowball Russian algorithm.
///
/// `ё` is normalised to `е` first.  Input containing no Cyrillic vowels (Latin
/// text, digits, punctuation, the empty string) is returned unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word
        .chars()
        .map(|c| if c == 'ё' { 'е' } else { c })
        .collect();
    let (rv, r2) = mark_regions(&w);
    step1(&mut w, rv);
    step2(&mut w, rv);
    derivational(&mut w, rv, r2);
    tidy_up(&mut w, rv);
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
    fn snowball_sample_va() {
        check(&[
            ("в", "в"),
            ("вавиловка", "вавиловк"),
            ("вагнера", "вагнер"),
            ("вагон", "вагон"),
            ("вагона", "вагон"),
            ("вагоне", "вагон"),
            ("вагонов", "вагон"),
            ("вагоном", "вагон"),
            ("вагоны", "вагон"),
            ("важная", "важн"),
            ("важнее", "важн"),
            ("важнейшие", "важн"),
            ("важнейшими", "важн"),
            ("важничал", "важнича"),
            ("важно", "важн"),
            ("важного", "важн"),
            ("важное", "важн"),
            ("важной", "важн"),
            ("важном", "важн"),
            ("важному", "важн"),
            ("важности", "важност"),
            ("важностию", "важност"),
            ("важность", "важност"),
            ("важностью", "важност"),
            ("важную", "важн"),
            ("важны", "важн"),
            ("важные", "важн"),
            ("важный", "важн"),
            ("важным", "важн"),
            ("важных", "важн"),
            ("вазах", "ваз"),
            ("вазы", "ваз"),
            ("вакса", "вакс"),
            ("вакханка", "вакханк"),
            ("вал", "вал"),
            ("валандался", "валанда"),
            ("валентина", "валентин"),
            ("валериановых", "валерианов"),
            ("валерию", "валер"),
            ("валетами", "валет"),
            ("вали", "вал"),
            ("валил", "вал"),
            ("валился", "вал"),
            ("валится", "вал"),
            ("валов", "вал"),
            ("вальдшнепа", "вальдшнеп"),
            ("вальс", "вальс"),
            ("вальса", "вальс"),
            ("вальсе", "вальс"),
            ("вальсишку", "вальсишк"),
            ("вальтера", "вальтер"),
            ("валяется", "валя"),
            ("валялась", "валя"),
            ("валялись", "валя"),
            ("валялось", "валя"),
            ("валялся", "валя"),
            ("валять", "валя"),
            ("валяются", "валя"),
            ("вам", "вам"),
            ("вами", "вам"),
        ]);
    }

    #[test]
    fn snowball_sample_pa() {
        check(&[
            ("п", "п"),
            ("па", "па"),
            ("пава", "пав"),
            ("павел", "павел"),
            ("павильон", "павильон"),
            ("павильонам", "павильон"),
            ("павла", "павл"),
            ("павлиний", "павлин"),
            ("павлиньи", "павлин"),
            ("павлиньим", "павлин"),
            ("павлович", "павлович"),
            ("павловна", "павловн"),
            ("павловне", "павловн"),
            ("павловной", "павловн"),
            ("павловну", "павловн"),
            ("павловны", "павловн"),
            ("павловцы", "павловц"),
            ("павлыч", "павлыч"),
            ("павлыча", "павлыч"),
            ("пагубная", "пагубн"),
            ("падает", "пада"),
            ("падай", "пада"),
            ("падал", "пада"),
            ("падала", "пада"),
            ("падаль", "падал"),
            ("падать", "пада"),
            ("падаю", "пада"),
            ("падают", "пада"),
            ("падающего", "пада"),
            ("падающие", "пада"),
            ("падеж", "падеж"),
            ("падение", "паден"),
            ("падением", "паден"),
            ("падении", "паден"),
            ("падений", "паден"),
            ("падения", "паден"),
            ("паденье", "паден"),
            ("паденья", "паден"),
            ("падет", "падет"),
            ("падут", "падут"),
            ("падучая", "падуч"),
            ("падчерицей", "падчериц"),
            ("падчерицы", "падчериц"),
            ("падшая", "падш"),
            ("падшей", "падш"),
            ("падшему", "падш"),
            ("падший", "падш"),
            ("падшим", "падш"),
            ("падших", "падш"),
            ("падшую", "падш"),
            ("паек", "паек"),
            ("пазухи", "пазух"),
            ("пазуху", "пазух"),
            ("пай", "па"),
            ("пакет", "пакет"),
            ("пакетом", "пакет"),
            ("пакеты", "пакет"),
            ("пакостей", "пакост"),
            ("пакостно", "пакостн"),
            ("пал", "пал"),
        ]);
    }

    #[test]
    fn yo_is_normalised_to_ye() {
        assert_eq!(stem("ёж"), "еж");
        assert_eq!(stem("её"), stem("ее"));
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("а", "а"),
            ("я", "я"),
            ("ь", "ь"),
            ("нн", "нн"),
            ("123", "123"),
            ("hello", "hello"),
            ("stemming", "stemming"),
            ("!?.,", "!?.,"),
            ("日本語", "日本語"),
            ("а1б2", "а1б2"),
        ]);
        // Every input must produce output without panicking.
        for s in ["ю", "ия", "ость", "ейше", "ннн", "аа", "ав", "яв", "\u{0}", "é"] {
            let _ = stem(s);
        }
    }
}
