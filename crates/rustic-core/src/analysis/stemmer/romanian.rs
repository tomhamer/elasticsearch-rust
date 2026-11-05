//! Snowball Romanian stemmer.
//!
//! A faithful transcription of the algorithm published at
//! <https://snowballstem.org/algorithms/romanian/stemmer.html> (Snowball 3.x
//! revision), working on a `Vec<char>`:
//!
//! 1. **Normalisation**: the legacy cedilla letters `ş` (U+015F) and `ţ`
//!    (U+0163) are rewritten to the correct comma-below letters `ș` (U+0219)
//!    and `ț` (U+021B). Both spellings are therefore stemmed identically, and
//!    the output always uses the comma-below forms (this is what the current
//!    official algorithm and its reference output do; the original 2006
//!    version recognised only the cedilla forms).
//! 2. **Prelude**: `i` and `u` between vowels are upper-cased to `I`/`U` so
//!    that they are treated as consonants.
//! 3. **Regions** (same definitions as the Spanish stemmer): `RV`, `R1`, `R2`.
//! 4. **Step 0, removal of plurals**: the longest matching plural/article
//!    ending in R1 is deleted or simplified (`ul`/`ului`, `aua`→`a`,
//!    `ea`/`ele`/`elor`→`e`, `ii`/`iua`/`iei`/`iile`/`iilor`/`ilor`→`i`,
//!    `ile`→`i` unless preceded by `ab`, `atei`→`at`, `ație`/`ația`→`ați`).
//! 5. **Step 1, combining suffixes**: `abilitate`→`abil`, `ivitate`→`iv`,
//!    `icitate`/`icator`/`iciv`/`ical`→`ic`, `ativ`/`ațiune`/`ator`→`at`,
//!    `itiv`/`ițiune`/`itor`→`it` and friends are reduced in R1, repeatedly,
//!    setting the "standard suffix removed" (continuation) flag.
//! 6. **Step 2, standard suffixes**: the longest matching derivational
//!    suffix in R2 is deleted (`iune`/`iuni` after `ț` become `t`,
//!    `ism`/`ist` family is conflated to `ist`); this also sets the flag.
//! 7. **Step 3, verb suffixes**: only when the flag is unset, the longest
//!    verb ending lying inside RV is deleted (the first group only when
//!    preceded inside RV by a consonant or `u`).
//! 8. **Step 4, final vowel**: the longest of `a`, `e`, `i`, `ie`, `ă` in RV
//!    is deleted.
//! 9. **Postlude**: `I`/`U` are lowered back to `i`/`u`.
//!
//! Non-letter characters are treated as consonants and pass through
//! untouched; the function never panics.

/// Vowels of the Romanian algorithm (`I`/`U` markers are consonants).
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'â' | 'î' | 'ă')
}

fn ends_with(s: &[char], suffix: &str) -> bool {
    let n = suffix.chars().count();
    n <= s.len() && s[s.len() - n..].iter().copied().eq(suffix.chars())
}

/// Index into `table` of the longest suffix of `s` that starts at or after
/// `limit` (Snowball `among`, optionally under `setlimit`).
fn longest_suffix<T>(s: &[char], table: &[(&str, T)], limit: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (idx, (suffix, _)) in table.iter().enumerate() {
        let n = suffix.chars().count();
        if n > s.len() || s.len() - n < limit {
            continue;
        }
        if best.is_some_and(|(best_len, _)| best_len >= n) {
            continue;
        }
        if ends_with(s, suffix) {
            best = Some((n, idx));
        }
    }
    best.map(|(_, idx)| idx)
}

fn replace_suffix(s: &mut Vec<char>, start: usize, replacement: &str) {
    s.truncate(start);
    s.extend(replacement.chars());
}

/// Position just after the first character at or after `from` satisfying
/// `pred` (Snowball `gopast`).
fn gopast(s: &[char], from: usize, pred: fn(char) -> bool) -> Option<usize> {
    (from..s.len()).find(|&i| pred(s[i])).map(|i| i + 1)
}

/// Start of RV (Spanish definition).
fn mark_rv(s: &[char]) -> usize {
    let n = s.len();
    if n < 2 {
        return n;
    }
    let rv = match (is_vowel(s[0]), is_vowel(s[1])) {
        (true, false) => gopast(s, 2, is_vowel),
        (true, true) => gopast(s, 2, |c| !is_vowel(c)),
        (false, false) => gopast(s, 2, is_vowel),
        (false, true) => (n >= 3).then_some(3),
    };
    rv.unwrap_or(n)
}

/// `gopast v gopast non-v` from `from`.
fn after_vowel_nonvowel(s: &[char], from: usize) -> Option<usize> {
    let i = gopast(s, from, is_vowel)?;
    gopast(s, i, |c| !is_vowel(c))
}

/// Compute `(pV, p1, p2)`.
fn mark_regions(s: &[char]) -> (usize, usize, usize) {
    let limit = s.len();
    let pv = mark_rv(s);
    let (p1, p2) = match after_vowel_nonvowel(s, 0) {
        Some(p1) => (p1, after_vowel_nonvowel(s, p1).unwrap_or(limit)),
        None => (limit, limit),
    };
    (pv, p1, p2)
}

#[derive(Clone, Copy)]
enum Plural {
    Delete,
    A,
    E,
    I,
    /// `ile` → `i` unless preceded by `ab`.
    IleNotAb,
    At,
    Ati,
}

const STEP_0: &[(&str, Plural)] = &[
    ("ul", Plural::Delete),
    ("ului", Plural::Delete),
    ("aua", Plural::A),
    ("ea", Plural::E),
    ("ele", Plural::E),
    ("elor", Plural::E),
    ("ii", Plural::I),
    ("iua", Plural::I),
    ("iei", Plural::I),
    ("iile", Plural::I),
    ("iilor", Plural::I),
    ("ilor", Plural::I),
    ("ile", Plural::IleNotAb),
    ("atei", Plural::At),
    ("ație", Plural::Ati),
    ("ația", Plural::Ati),
];

fn step_0(s: &mut Vec<char>, p1: usize) {
    let Some(idx) = longest_suffix(s, STEP_0, 0) else {
        return;
    };
    let (suffix, kind) = STEP_0[idx];
    let start = s.len() - suffix.chars().count();
    if start < p1 {
        return;
    }
    let replacement = match kind {
        Plural::Delete => "",
        Plural::A => "a",
        Plural::E => "e",
        Plural::I => "i",
        Plural::IleNotAb => {
            if ends_with(&s[..start], "ab") {
                return;
            }
            "i"
        }
        Plural::At => "at",
        Plural::Ati => "ați",
    };
    replace_suffix(s, start, replacement);
}

const COMBO: &[(&str, &str)] = &[
    ("abilitate", "abil"),
    ("abilitati", "abil"),
    ("abilităi", "abil"),
    ("abilități", "abil"),
    ("ibilitate", "ibil"),
    ("ivitate", "iv"),
    ("ivitati", "iv"),
    ("ivităi", "iv"),
    ("ivități", "iv"),
    ("icitate", "ic"),
    ("icitati", "ic"),
    ("icităi", "ic"),
    ("icități", "ic"),
    ("icator", "ic"),
    ("icatori", "ic"),
    ("iciv", "ic"),
    ("iciva", "ic"),
    ("icive", "ic"),
    ("icivi", "ic"),
    ("icivă", "ic"),
    ("ical", "ic"),
    ("icala", "ic"),
    ("icale", "ic"),
    ("icali", "ic"),
    ("icală", "ic"),
    ("ativ", "at"),
    ("ativa", "at"),
    ("ative", "at"),
    ("ativi", "at"),
    ("ativă", "at"),
    ("ațiune", "at"),
    ("atoare", "at"),
    ("ator", "at"),
    ("atori", "at"),
    ("ătoare", "at"),
    ("ător", "at"),
    ("ători", "at"),
    ("itiv", "it"),
    ("itiva", "it"),
    ("itive", "it"),
    ("itivi", "it"),
    ("itivă", "it"),
    ("ițiune", "it"),
    ("itoare", "it"),
    ("itor", "it"),
    ("itori", "it"),
];

/// Step 1: one reduction of a combining suffix in R1. Returns whether a
/// replacement was made.
fn combo_suffix(s: &mut Vec<char>, p1: usize) -> bool {
    let Some(idx) = longest_suffix(s, COMBO, 0) else {
        return false;
    };
    let (suffix, replacement) = COMBO[idx];
    let start = s.len() - suffix.chars().count();
    if start < p1 {
        return false;
    }
    replace_suffix(s, start, replacement);
    true
}

#[derive(Clone, Copy)]
enum Std {
    Delete,
    /// `iune`/`iuni`: delete if preceded by `ț`, which becomes `t`.
    Iune,
    /// `ism`/`ist` family → `ist`.
    Ist,
}

const STANDARD: &[(&str, Std)] = &[
    ("at", Std::Delete), ("ata", Std::Delete), ("ată", Std::Delete), ("ati", Std::Delete),
    ("ate", Std::Delete), ("ut", Std::Delete), ("uta", Std::Delete), ("ută", Std::Delete),
    ("uti", Std::Delete), ("ute", Std::Delete), ("it", Std::Delete), ("ita", Std::Delete),
    ("ită", Std::Delete), ("iti", Std::Delete), ("ite", Std::Delete), ("ic", Std::Delete),
    ("ica", Std::Delete), ("ice", Std::Delete), ("ici", Std::Delete), ("ică", Std::Delete),
    ("abil", Std::Delete), ("abila", Std::Delete), ("abile", Std::Delete),
    ("abili", Std::Delete), ("abilă", Std::Delete), ("ibil", Std::Delete),
    ("ibila", Std::Delete), ("ibile", Std::Delete), ("ibili", Std::Delete),
    ("ibilă", Std::Delete), ("oasa", Std::Delete), ("oasă", Std::Delete),
    ("oase", Std::Delete), ("os", Std::Delete), ("osi", Std::Delete), ("oși", Std::Delete),
    ("ant", Std::Delete), ("anta", Std::Delete), ("ante", Std::Delete), ("anti", Std::Delete),
    ("antă", Std::Delete), ("ator", Std::Delete), ("atori", Std::Delete),
    ("itate", Std::Delete), ("itati", Std::Delete), ("ităi", Std::Delete),
    ("ități", Std::Delete), ("iv", Std::Delete), ("iva", Std::Delete), ("ive", Std::Delete),
    ("ivi", Std::Delete), ("ivă", Std::Delete),
    ("iune", Std::Iune), ("iuni", Std::Iune),
    ("ism", Std::Ist), ("isme", Std::Ist), ("ist", Std::Ist), ("ista", Std::Ist),
    ("iste", Std::Ist), ("isti", Std::Ist), ("istă", Std::Ist), ("iști", Std::Ist),
];

/// Steps 1 and 2. Returns the "standard suffix removed" flag.
fn standard_suffix(s: &mut Vec<char>, p1: usize, p2: usize) -> bool {
    let mut removed = false;
    while combo_suffix(s, p1) {
        removed = true;
    }
    let Some(idx) = longest_suffix(s, STANDARD, 0) else {
        return removed;
    };
    let (suffix, kind) = STANDARD[idx];
    let start = s.len() - suffix.chars().count();
    if start < p2 {
        return removed;
    }
    match kind {
        Std::Delete => s.truncate(start),
        Std::Iune => {
            if start == 0 || s[start - 1] != 'ț' {
                return removed;
            }
            replace_suffix(s, start - 1, "t");
        }
        Std::Ist => replace_suffix(s, start, "ist"),
    }
    true
}

/// Verb suffixes; `true` marks the group that must be preceded inside RV by
/// a consonant or `u`.
const VERB: &[(&str, bool)] = &[
    ("are", true), ("ere", true), ("ire", true), ("âre", true),
    ("ind", true), ("ând", true), ("indu", true), ("ându", true),
    ("eze", true), ("ească", true),
    ("ez", true), ("ezi", true), ("ează", true), ("esc", true), ("ești", true), ("ește", true),
    ("ăsc", true), ("ăști", true), ("ăște", true),
    ("am", true), ("ai", true), ("au", true),
    ("eam", true), ("eai", true), ("ea", true), ("eați", true), ("eau", true),
    ("iam", true), ("iai", true), ("ia", true), ("iați", true), ("iau", true),
    ("ui", true),
    ("ași", true), ("arăm", true), ("arăți", true), ("ară", true),
    ("uși", true), ("urăm", true), ("urăți", true), ("ură", true),
    ("iși", true), ("irăm", true), ("irăți", true), ("iră", true),
    ("âi", true), ("âși", true), ("ârăm", true), ("ârăți", true), ("âră", true),
    ("asem", true), ("aseși", true), ("ase", true), ("aserăm", true), ("aserăți", true),
    ("aseră", true),
    ("isem", true), ("iseși", true), ("ise", true), ("iserăm", true), ("iserăți", true),
    ("iseră", true),
    ("âsem", true), ("âseși", true), ("âse", true), ("âserăm", true), ("âserăți", true),
    ("âseră", true),
    ("usem", true), ("useși", true), ("use", true), ("userăm", true), ("userăți", true),
    ("useră", true),
    ("ăm", false), ("ați", false), ("em", false), ("eți", false), ("im", false),
    ("iți", false), ("âm", false), ("âți", false),
    ("seși", false), ("serăm", false), ("serăți", false), ("seră", false),
    ("sei", false), ("se", false),
    ("sesem", false), ("seseși", false), ("sese", false), ("seserăm", false),
    ("seserăți", false), ("seseră", false),
];

/// Step 3: delete the longest verb suffix lying inside RV.
fn verb_suffix(s: &mut Vec<char>, pv: usize) {
    let Some(idx) = longest_suffix(s, VERB, pv) else {
        return;
    };
    let (suffix, needs_consonant_or_u) = VERB[idx];
    let start = s.len() - suffix.chars().count();
    if needs_consonant_or_u {
        // The preceding letter must itself lie inside RV.
        if start <= pv {
            return;
        }
        let prev = s[start - 1];
        if is_vowel(prev) && prev != 'u' {
            return;
        }
    }
    s.truncate(start);
}

const VOWEL: &[(&str, ())] = &[("a", ()), ("e", ()), ("i", ()), ("ie", ()), ("ă", ())];

/// Step 4: delete a final vowel suffix in RV.
fn vowel_suffix(s: &mut Vec<char>, pv: usize) {
    let Some(idx) = longest_suffix(s, VOWEL, 0) else {
        return;
    };
    let start = s.len() - VOWEL[idx].0.chars().count();
    if start >= pv {
        s.truncate(start);
    }
}

/// Stem an already-lowercased Romanian word with the Snowball Romanian
/// algorithm.
///
/// Cedilla `ş`/`ţ` and comma-below `ș`/`ț` are treated as the same letters;
/// the output uses the comma-below forms. Never panics; non-letter content is
/// returned unchanged.
pub fn stem(word: &str) -> String {
    // Normalisation.
    let mut s: Vec<char> = word
        .chars()
        .map(|c| match c {
            'ş' => 'ș',
            'ţ' => 'ț',
            _ => c,
        })
        .collect();

    // Prelude: i/u between vowels become consonants.
    for i in 1..s.len().saturating_sub(1) {
        if matches!(s[i], 'i' | 'u') && is_vowel(s[i - 1]) && is_vowel(s[i + 1]) {
            s[i] = if s[i] == 'i' { 'I' } else { 'U' };
        }
    }
    let (pv, p1, p2) = mark_regions(&s);

    step_0(&mut s, p1);
    let standard_suffix_removed = standard_suffix(&mut s, p1, p2);
    if !standard_suffix_removed {
        verb_suffix(&mut s, pv);
    }
    vowel_suffix(&mut s, pv);

    // Postlude.
    s.into_iter()
        .map(|c| match c {
            'I' => 'i',
            'U' => 'u',
            _ => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(cases: &[(&str, &str)]) {
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|&(word, expected)| {
                let got = stem(word);
                (got != expected).then(|| format!("{word}: expected {expected}, got {got}"))
            })
            .collect();
        assert!(failures.is_empty(), "failing vectors:\n{}", failures.join("\n"));
    }

    /// Vectors taken from the official Snowball Romanian sample vocabulary.
    #[test]
    fn official_sample_vocabulary() {
        check(&[
            ("abruptă", "abrupt"),
            ("abrutiza", "abrutiz"),
            ("abrutizant", "abrutiz"),
            ("abrutizanta", "abrutiz"),
            ("abrutizante", "abrutiz"),
            ("abrutizanti", "abrutiz"),
            ("abrutizantă", "abrutiz"),
            ("abrutizanţi", "abrutizanț"),
            ("abrutizanți", "abrutizanț"),
            ("abrutizare", "abrutiz"),
            ("abrutizat", "abrutiz"),
            ("abrutizata", "abrutiz"),
            ("abrutizată", "abrutiz"),
            ("abrutizaţi", "abrutiz"),
            ("abrutizați", "abrutiz"),
            ("abrutizez", "abrutiz"),
            ("absent", "absent"),
            ("absenta", "absent"),
            ("absentare", "absent"),
            ("absente", "absent"),
            ("absenteism", "absenteist"),
            ("absenteist", "absenteist"),
            ("absenteista", "absenteist"),
            ("absenteisti", "absenteist"),
            ("absenteistă", "absenteist"),
            ("absenteişti", "absenteist"),
            ("absenteiști", "absenteist"),
            ("absentez", "absent"),
            ("absenti", "absent"),
            ("absentă", "absent"),
            ("absenţa", "absenț"),
            ("absenţe", "absenț"),
            ("absenţei", "absențe"),
            ("absenţi", "absenț"),
            ("absenţă", "absenț"),
            ("absența", "absenț"),
            ("absențe", "absenț"),
            ("absenței", "absențe"),
            ("absenți", "absenț"),
            ("absență", "absenț"),
            ("ocol", "ocol"),
            ("ocolea", "ocol"),
            ("ocolească", "ocol"),
            ("ocolesc", "ocol"),
            ("ocoleşte", "ocol"),
            ("ocoleşti", "ocol"),
            ("ocolește", "ocol"),
            ("ocolești", "ocol"),
            ("ocoli", "ocol"),
            ("ocolim", "ocol"),
            ("ocolind", "ocol"),
            ("ocolire", "ocol"),
            ("ocolit", "ocol"),
            ("ocolite", "ocol"),
            ("ocolită", "ocol"),
            ("ocolişuri", "ocolișur"),
            ("ocoliţi", "ocol"),
            ("ocolișuri", "ocolișur"),
            ("ocoliți", "ocol"),
            ("ocolul", "ocol"),
            ("ocolului", "ocol"),
            ("ocoluri", "ocolur"),
            ("ocolurile", "ocolur"),
            ("ocrotesc", "ocrot"),
            ("ocroti", "ocrot"),
            ("ocrotire", "ocrot"),
            ("ocrotiri", "ocrotir"),
            ("ocrotit", "ocrot"),
            ("ocrotitoare", "ocrot"),
            ("ocrotitor", "ocrot"),
            ("ocrotitori", "ocrot"),
            ("ocrotiţi", "ocrot"),
            ("ocrotiți", "ocrot"),
            ("octet", "octet"),
            ("octetul", "octet"),
            ("octetului", "octet"),
            ("octeţi", "octeț"),
            ("octeţii", "octeț"),
            ("octeți", "octeț"),
        ]);
    }

    /// Cedilla and comma-below spellings stem identically.
    #[test]
    fn cedilla_and_comma_forms_are_equivalent() {
        for (cedilla, comma) in [
            ("ocoleşte", "ocolește"),
            ("absenţei", "absenței"),
            ("abrutizanţi", "abrutizanți"),
            ("ţară", "țară"),
        ] {
            assert_eq!(stem(cedilla), stem(comma), "{cedilla} vs {comma}");
        }
        assert_eq!(stem("ţară"), "țar");
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("ă"), "ă");
        assert_eq!(stem("ţ"), "ț");
        assert_eq!(stem("ş"), "ș");
        assert_eq!(stem("ii"), "ii");
        assert_eq!(stem("ule"), "ule");
        assert_eq!(stem("ile"), "ile");
        assert_eq!(stem("abile"), "abil");
        assert_eq!(stem("aiua"), "aiu");
        assert_eq!(stem("iubire"), "iubir");
        assert_eq!(stem("are"), "are");
        assert_eq!(stem("uire"), "uir");
        assert_eq!(stem("aiure"), "aiur");
        assert_eq!(stem("123"), "123");
        assert_eq!(stem("..."), "...");
        assert_eq!(stem("日本語"), "日本語");
        for w in ["x", "xy", "I", "U", "aIa", "iune", "țiune", "ism", "se", "ie"] {
            let _ = stem(w);
        }
    }
}
