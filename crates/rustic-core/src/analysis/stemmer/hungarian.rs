//! Snowball Hungarian stemmer.
//!
//! A faithful port of the Snowball algorithm described at
//! <https://snowballstem.org/algorithms/hungarian/stemmer.html>, operating on
//! a `Vec<char>` of the (already lowercased) input word.  It removes noun
//! inflections only.
//!
//! Vowels are `a e i o u á é í ó ö ő ú ü ű`.  `R1` is the region after the
//! first consonant or digraph (`cs dz dzs gy ly ny sz ty zs`) in a word that
//! begins with a vowel, or the region after the first vowel in a word that
//! begins with a consonant.  A "double consonant" is one of
//! `bb cc ccs dd ff gg ggy jj kk ll lly mm nn nny pp rr ss ssz tt tty vv zz
//! zzs`.
//!
//! Every step searches for the longest matching suffix of the whole word and
//! then requires that suffix to lie in `R1` (no fallback to a shorter one, as
//! with the Snowball `among`).  The steps, in order:
//!
//! 1. Instrumental: `al el` preceded by a double consonant: delete, then
//!    remove one letter of the double consonant.
//! 2. Frequent cases: `ban ben ba be ra re nak nek val vel tól től ról ről ból
//!    ből hoz hez höz nál nél ig at et ot öt ért képp képpen kor ul ül vá vé
//!    onként enként anként ként en on an ön n t`: delete; then a final `á`/`é`
//!    in `R1` becomes `a`/`e`.
//! 3. Special cases: `én` -> `e`; `án`, `ánként` -> `a`.
//! 4. Other cases: `astul estül stul stül`: delete; `ástul` -> `a`;
//!    `éstül` -> `e`.
//! 5. Factive: `á é` preceded by a double consonant: delete and undouble.
//! 6. Owned: `oké öké aké eké ké éi é`: delete; `áké áéi` -> `a`;
//!    `éké ééi éé` -> `e`.
//! 7. Singular owner: `ünk unk nk juk jük uk ük em om am m od ed ad öd d ja je
//!    a e o`: delete; `ánk ájuk ám ád á` -> `a`; `énk éjük ém éd é` -> `e`.
//! 8. Plural owner: `jaim jeim aim eim im jaid jeid aid eid id jai jei ai ei i
//!    jaink jeink eink aink ink jaitok jeitek aitok eitek itek jeik jaik aik
//!    eik ik`: delete; `áim áid ái áink áitok áik` -> `a`; `éim éid éi éink
//!    éitek éik` -> `e`.
//! 9. Plural: `ák` -> `a`; `ék` -> `e`; `ök ok ek ak k`: delete.

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'á' | 'é' | 'í' | 'ó' | 'ö' | 'ő' | 'ú' | 'ü' | 'ű'
    )
}

/// Digraphs that count as a single consonant when locating `R1`.
const DIGRAPHS: &[&str] = &["cs", "dz", "dzs", "gy", "ly", "ny", "sz", "ty", "zs"];

/// Double consonants (`v_double` in the Snowball source).
const DOUBLE: &[&str] = &[
    "bb", "cc", "ccs", "dd", "ff", "gg", "ggy", "jj", "kk", "ll", "lly", "mm", "nn", "nny",
    "pp", "rr", "ss", "ssz", "tt", "tty", "vv", "zz", "zzs",
];

/// What to do with a matched suffix.
#[derive(Clone, Copy)]
enum Action {
    Delete,
    ToA,
    ToE,
}

use Action::{Delete, ToA, ToE};

const CASE: &[(&str, Action)] = &[
    ("ban", Delete), ("ben", Delete), ("ba", Delete), ("be", Delete), ("ra", Delete),
    ("re", Delete), ("nak", Delete), ("nek", Delete), ("val", Delete), ("vel", Delete),
    ("tól", Delete), ("től", Delete), ("ról", Delete), ("ről", Delete), ("ból", Delete),
    ("ből", Delete), ("hoz", Delete), ("hez", Delete), ("höz", Delete), ("nál", Delete),
    ("nél", Delete), ("ig", Delete), ("at", Delete), ("et", Delete), ("ot", Delete),
    ("öt", Delete), ("ért", Delete), ("képp", Delete), ("képpen", Delete), ("kor", Delete),
    ("ul", Delete), ("ül", Delete), ("vá", Delete), ("vé", Delete), ("onként", Delete),
    ("enként", Delete), ("anként", Delete), ("ként", Delete), ("en", Delete), ("on", Delete),
    ("an", Delete), ("ön", Delete), ("n", Delete), ("t", Delete),
];

const CASE_SPECIAL: &[(&str, Action)] = &[("én", ToE), ("án", ToA), ("ánként", ToA)];

const CASE_OTHER: &[(&str, Action)] = &[
    ("astul", Delete), ("estül", Delete), ("stul", Delete), ("stül", Delete),
    ("ástul", ToA), ("éstül", ToE),
];

const OWNED: &[(&str, Action)] = &[
    ("oké", Delete), ("öké", Delete), ("aké", Delete), ("eké", Delete), ("éké", ToE),
    ("áké", ToA), ("ké", Delete), ("ééi", ToE), ("áéi", ToA), ("éi", Delete), ("éé", ToE),
    ("é", Delete),
];

const SING_OWNER: &[(&str, Action)] = &[
    ("ünk", Delete), ("unk", Delete), ("ánk", ToA), ("énk", ToE), ("nk", Delete),
    ("ájuk", ToA), ("éjük", ToE), ("juk", Delete), ("jük", Delete), ("uk", Delete),
    ("ük", Delete), ("em", Delete), ("om", Delete), ("am", Delete), ("ám", ToA), ("ém", ToE),
    ("m", Delete), ("od", Delete), ("ed", Delete), ("ad", Delete), ("öd", Delete), ("ád", ToA),
    ("éd", ToE), ("d", Delete), ("ja", Delete), ("je", Delete), ("a", Delete), ("e", Delete),
    ("o", Delete), ("á", ToA), ("é", ToE),
];

const PLUR_OWNER: &[(&str, Action)] = &[
    ("jaim", Delete), ("jeim", Delete), ("áim", ToA), ("éim", ToE), ("aim", Delete),
    ("eim", Delete), ("im", Delete), ("jaid", Delete), ("jeid", Delete), ("áid", ToA),
    ("éid", ToE), ("aid", Delete), ("eid", Delete), ("id", Delete), ("jai", Delete),
    ("jei", Delete), ("ái", ToA), ("éi", ToE), ("ai", Delete), ("ei", Delete), ("i", Delete),
    ("jaink", Delete), ("jeink", Delete), ("eink", Delete), ("aink", Delete), ("áink", ToA),
    ("éink", ToE), ("ink", Delete), ("jaitok", Delete), ("jeitek", Delete), ("aitok", Delete),
    ("eitek", Delete), ("áitok", ToA), ("éitek", ToE), ("itek", Delete), ("jeik", Delete),
    ("jaik", Delete), ("aik", Delete), ("eik", Delete), ("áik", ToA), ("éik", ToE),
    ("ik", Delete),
];

const PLURAL: &[(&str, Action)] = &[
    ("ák", ToA), ("ék", ToE), ("ök", Delete), ("ak", Delete), ("ok", Delete), ("ek", Delete),
    ("k", Delete),
];

/// True when `w[..end]` ends with the literal `s`.
fn ends_with_at(w: &[char], end: usize, s: &str) -> bool {
    let n = s.chars().count();
    end >= n && w[end - n..end].iter().copied().eq(s.chars())
}

/// True when `w[i..]` starts with the literal `s`.
fn starts_with_at(w: &[char], i: usize, s: &str) -> bool {
    let n = s.chars().count();
    i + n <= w.len() && w[i..i + n].iter().copied().eq(s.chars())
}

/// Returns `p1`, the start of `R1` (`w.len()` when `R1` is empty).
fn mark_regions(w: &[char]) -> usize {
    let n = w.len();
    let Some(&first) = w.first() else {
        return n;
    };
    if is_vowel(first) {
        // goto non-v, then skip a digraph or a single character
        let mut i = 0;
        while i < n && is_vowel(w[i]) {
            i += 1;
        }
        if i >= n {
            return n;
        }
        let digraph = DIGRAPHS
            .iter()
            .filter(|d| starts_with_at(w, i, d))
            .map(|d| d.chars().count())
            .max();
        i + digraph.unwrap_or(1)
    } else {
        // gopast v
        let mut i = 0;
        while i < n && !is_vowel(w[i]) {
            i += 1;
        }
        if i >= n { n } else { i + 1 }
    }
}

/// Longest suffix of `w` found in `table`; returns `(table index, length)`.
fn longest_suffix(w: &[char], table: &[(&str, Action)]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (idx, (s, _)) in table.iter().enumerate() {
        let n = s.chars().count();
        if n > w.len() || best.is_some_and(|(_, bn)| bn >= n) {
            continue;
        }
        if ends_with_at(w, w.len(), s) {
            best = Some((idx, n));
        }
    }
    best
}

/// `[substring] R1 among(...)` followed by the entry's action.  Returns
/// whether the word was changed.
fn apply_table(w: &mut Vec<char>, p1: usize, table: &[(&str, Action)]) -> bool {
    let Some((idx, n)) = longest_suffix(w, table) else {
        return false;
    };
    let start = w.len() - n;
    if start < p1 {
        return false;
    }
    w.truncate(start);
    match table[idx].1 {
        Delete => {}
        ToA => w.push('a'),
        ToE => w.push('e'),
    }
    true
}

/// True when `w[..end]` ends with a double consonant.
fn ends_with_double(w: &[char], end: usize) -> bool {
    DOUBLE.iter().any(|d| ends_with_at(w, end, d))
}

/// Removes the second-to-last character (`next [hop 1] delete`).
fn undouble(w: &mut Vec<char>) {
    if w.len() >= 2 {
        let i = w.len() - 2;
        w.remove(i);
    }
}

/// Steps 1 and 5: a suffix preceded by a double consonant is deleted and the
/// double consonant is undoubled.
fn remove_after_double(w: &mut Vec<char>, p1: usize, table: &[(&str, Action)]) {
    let Some((_, n)) = longest_suffix(w, table) else {
        return;
    };
    let start = w.len() - n;
    if start < p1 || !ends_with_double(w, start) {
        return;
    }
    w.truncate(start);
    undouble(w);
}

/// `v_ending`: a final `á`/`é` in `R1` becomes `a`/`e`.
fn v_ending(w: &mut [char], p1: usize) {
    let n = w.len();
    if n >= 1 && n - 1 >= p1 {
        match w[n - 1] {
            'á' => w[n - 1] = 'a',
            'é' => w[n - 1] = 'e',
            _ => {}
        }
    }
}

/// Stem a lowercased Hungarian word with the Snowball Hungarian algorithm.
///
/// Never panics; input without Hungarian vowels passes through unchanged.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    let p1 = mark_regions(&w);
    remove_after_double(&mut w, p1, &[("al", Delete), ("el", Delete)]);
    if apply_table(&mut w, p1, CASE) {
        v_ending(&mut w, p1);
    }
    apply_table(&mut w, p1, CASE_SPECIAL);
    apply_table(&mut w, p1, CASE_OTHER);
    remove_after_double(&mut w, p1, &[("á", Delete), ("é", Delete)]);
    apply_table(&mut w, p1, OWNED);
    apply_table(&mut w, p1, SING_OWNER);
    apply_table(&mut w, p1, PLUR_OWNER);
    apply_table(&mut w, p1, PLURAL);
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
    fn snowball_sample_bab() {
        check(&[
            ("babaháznak", "babaház"),
            ("babakocsi", "babakocs"),
            ("babakocsijáért", "babakocs"),
            ("babakocsit", "babakocs"),
            ("babakocsiért", "babakocs"),
            ("babból", "bab"),
            ("bab", "bab"),
            ("babgulyás", "babgulyás"),
            ("babgulyást", "babgulyás"),
            ("babona", "babon"),
            ("babonákkal", "babona"),
            ("babonás", "babonás"),
            ("babrálgatta", "babrálgatt"),
            ("babrálni", "babráln"),
            ("babrál", "babrál"),
            ("babrált", "babrál"),
            ("babrálva", "babrálv"),
            ("babusgatnak", "babusgat"),
            ("baba", "ba"),
            ("babái", "baba"),
            ("babák", "baba"),
            ("babákkal", "baba"),
            ("babázni", "babázn"),
            ("babérfa", "babérf"),
            ("babérokat", "babér"),
            ("babért", "bab"),
            ("bacchánsnők", "bacchánsnő"),
            ("badacsonyi", "badacsony"),
            ("badarság", "badarság"),
            ("badarságok", "badarság"),
            ("baedeker", "baedeker"),
            ("baglyokat", "bagly"),
            ("bagolyszemüveges", "bagolyszemüveges"),
            ("bagót", "bagó"),
            ("bajbajutott", "bajbajutot"),
            ("bajbajutottak", "bajbajutott"),
            ("bajbajutottakat", "bajbajutott"),
            ("bajbajutottakon", "bajbajutott"),
            ("bajlódjanak", "bajlód"),
            ("bajlódni", "bajlódn"),
        ]);
    }

    #[test]
    fn snowball_sample_mul() {
        check(&[
            ("muattta", "muattt"),
            ("mukkot", "muk"),
            ("mulandóság", "mulandóság"),
            ("mulandóságot", "mulandóság"),
            ("mulasszátok", "mulasszát"),
            ("mulasztanak", "mulaszt"),
            ("mulasztotta", "mulasztott"),
            ("mulasztottam", "mulasztott"),
            ("mulasztották", "mulasztotta"),
            ("mulaszt", "mulasz"),
            ("mulaszthatom", "mulaszthat"),
            ("mulasztás", "mulasztás"),
            ("mulasztásban", "mulasztás"),
            ("mulasztásból", "mulasztás"),
            ("mulasztásnál", "mulasztás"),
            ("mulasztással", "mulasztás"),
            ("mulasztásának", "mulasztás"),
            ("mulasztásánál", "mulasztás"),
            ("mulasztásáért", "mulasztás"),
            ("mulasztási", "mulasztás"),
            ("mulasztásos", "mulasztásos"),
            ("mulasztó", "mulasztó"),
            ("mulathatnánk", "mulathatna"),
            ("mulathattunk", "mulathatt"),
            ("mulatna", "mulatn"),
            ("mulat", "mul"),
            ("mulatnak", "mulat"),
            ("mulatni", "mulatn"),
            ("mulattak", "mulatt"),
            ("mulattat", "mulatt"),
            ("mulattatta", "mulattatt"),
            ("mulatott", "mulatot"),
            ("mulatozott", "mulatozot"),
            ("mulatozáshoz", "mulatozás"),
            ("mulatozást", "mulatozás"),
            ("mulatság", "mulatság"),
            ("mulatságnak", "mulatság"),
            ("mulatságot", "mulatság"),
            ("mulatságos", "mulatságos"),
            ("mulatt", "mulat"),
        ]);
    }

    #[test]
    fn r1_digraph_rule() {
        // Word beginning with a vowel: R1 starts after the first consonant or
        // digraph, so a bare "asz" has an empty R1 and nothing is removed.
        check(&[("asz", "asz"), ("aszt", "asz"), ("adzsa", "adzs")]);
    }

    #[test]
    fn robustness() {
        check(&[
            ("", ""),
            ("a", "a"),
            ("k", "k"),
            ("t", "t"),
            ("al", "al"),
            ("123", "123"),
            ("!?.,", "!?.,"),
            ("日本語", "日本語"),
            ("á", "á"),
        ]);
        for s in ["ttal", "ssz", "sszal", "é", "éé", "ánként", "jaitok", "\u{0}", "x1y", "ü"] {
            let _ = stem(s);
        }
    }
}
