//! Snowball Portuguese stemmer.
//!
//! A faithful transcription of the algorithm published at
//! <https://snowballstem.org/algorithms/portuguese/stemmer.html>, working on
//! a `Vec<char>`:
//!
//! 1. **Prelude**: the nasal vowels `ã` and `õ` are rewritten as `a~` and
//!    `o~`, where `~` is a separate character treated as a consonant.
//! 2. **Regions** (same definitions as the Spanish stemmer): `RV` starts
//!    after the next vowel when the second letter is a consonant, after the
//!    next consonant when the first two letters are vowels, and after the
//!    third letter otherwise; `R1` is the region after the first non-vowel
//!    following a vowel and `R2` repeats that construction inside `R1`.
//! 3. **Step 1, standard suffix removal**: the longest matching suffix from
//!    the derivational table is removed or replaced subject to its R1/R2/RV
//!    condition (`-logia`→`log`, `-ução`→`u`, `-ência`→`ente`, the
//!    `-amente`/`-mente`/`-idade`/`-iva` chains with their secondary
//!    `iv`/`at`/`os`/`ic`/`ad`/`ante`/`avel`/`ível`/`abil` removals, and
//!    `-eira(s)`→`eir`).
//! 4. **Step 2, verb suffix removal**: only when step 1 changed nothing,
//!    the longest verb ending lying wholly inside RV is deleted.
//! 5. **Step 3**: if step 1 or 2 changed the word, a final `i` preceded by
//!    `c` is deleted when in RV.
//! 6. **Step 4, residual suffix**: otherwise a final `os`, `a`, `i`, `o`,
//!    `á`, `í` or `ó` in RV is deleted.
//! 7. **Step 5, residual form**: a final `e`, `é` or `ê` in RV is deleted
//!    (together with the `u` of a then-final `gu`, or the `i` of `ci`, when
//!    that letter is in RV); a final `ç` loses its cedilla.
//! 8. **Postlude**: `a~` and `o~` are turned back into `ã` and `õ`.
//!
//! Non-letter characters are treated as consonants and pass through
//! untouched; the function never panics.

/// Vowels of the Portuguese algorithm (`ã`/`õ` have been split by the
/// prelude into a vowel followed by the consonant `~` before this is used).
fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'á' | 'é' | 'í' | 'ó' | 'ú' | 'â' | 'ê' | 'ô'
    )
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
enum Standard {
    /// Delete if in R2.
    Delete,
    /// Replace with `log` if in R2.
    Log,
    /// Replace with `u` if in R2.
    U,
    /// Replace with `ente` if in R2.
    Ente,
    /// `amente`: delete if in R1, then the `iv(at)`/`os`/`ic`/`ad` chain.
    Amente,
    /// `mente`: delete if in R2, then `ante`/`avel`/`ível`.
    Mente,
    /// `idade(s)`: delete if in R2, then `abil`/`ic`/`iv`.
    Idade,
    /// `iva`/`ivo`/`ivas`/`ivos`: delete if in R2, then `at`.
    Iva,
    /// `ira`/`iras`: replace with `ir` if in RV and preceded by `e`.
    Ira,
}

const STANDARD: &[(&str, Standard)] = &[
    ("eza", Standard::Delete),
    ("ezas", Standard::Delete),
    ("ico", Standard::Delete),
    ("ica", Standard::Delete),
    ("icos", Standard::Delete),
    ("icas", Standard::Delete),
    ("ismo", Standard::Delete),
    ("ismos", Standard::Delete),
    ("ável", Standard::Delete),
    ("ível", Standard::Delete),
    ("ista", Standard::Delete),
    ("istas", Standard::Delete),
    ("oso", Standard::Delete),
    ("osa", Standard::Delete),
    ("osos", Standard::Delete),
    ("osas", Standard::Delete),
    ("amento", Standard::Delete),
    ("amentos", Standard::Delete),
    ("imento", Standard::Delete),
    ("imentos", Standard::Delete),
    ("adora", Standard::Delete),
    ("ador", Standard::Delete),
    ("aça~o", Standard::Delete),
    ("adoras", Standard::Delete),
    ("adores", Standard::Delete),
    ("aço~es", Standard::Delete),
    ("ante", Standard::Delete),
    ("antes", Standard::Delete),
    ("ância", Standard::Delete),
    ("logia", Standard::Log),
    ("logias", Standard::Log),
    ("uça~o", Standard::U),
    ("uço~es", Standard::U),
    ("ência", Standard::Ente),
    ("ências", Standard::Ente),
    ("amente", Standard::Amente),
    ("mente", Standard::Mente),
    ("idade", Standard::Idade),
    ("idades", Standard::Idade),
    ("iva", Standard::Iva),
    ("ivo", Standard::Iva),
    ("ivas", Standard::Iva),
    ("ivos", Standard::Iva),
    ("ira", Standard::Ira),
    ("iras", Standard::Ira),
];

/// Remove the longest of `suffixes` if it lies in R2 (Snowball `try`);
/// returns the index of the removed suffix.
fn try_delete_in_r2(s: &mut Vec<char>, suffixes: &[(&str, ())], p2: usize) -> Option<usize> {
    let idx = longest_suffix(s, suffixes, 0)?;
    let start = s.len() - suffixes[idx].0.chars().count();
    if start < p2 {
        return None;
    }
    s.truncate(start);
    Some(idx)
}

/// Step 1. Returns whether the word was changed.
fn standard_suffix(s: &mut Vec<char>, pv: usize, p1: usize, p2: usize) -> bool {
    let Some(idx) = longest_suffix(s, STANDARD, 0) else {
        return false;
    };
    let (suffix, kind) = STANDARD[idx];
    let start = s.len() - suffix.chars().count();
    match kind {
        Standard::Delete | Standard::Log | Standard::U | Standard::Ente => {
            if start < p2 {
                return false;
            }
            let replacement = match kind {
                Standard::Log => "log",
                Standard::U => "u",
                Standard::Ente => "ente",
                _ => "",
            };
            replace_suffix(s, start, replacement);
        }
        Standard::Amente => {
            if start < p1 {
                return false;
            }
            s.truncate(start);
            let chain = &[("iv", ()), ("os", ()), ("ic", ()), ("ad", ())];
            if try_delete_in_r2(s, chain, p2) == Some(0) {
                try_delete_in_r2(s, &[("at", ())], p2);
            }
        }
        Standard::Mente => {
            if start < p2 {
                return false;
            }
            s.truncate(start);
            try_delete_in_r2(s, &[("ante", ()), ("avel", ()), ("ível", ())], p2);
        }
        Standard::Idade => {
            if start < p2 {
                return false;
            }
            s.truncate(start);
            try_delete_in_r2(s, &[("abil", ()), ("ic", ()), ("iv", ())], p2);
        }
        Standard::Iva => {
            if start < p2 {
                return false;
            }
            s.truncate(start);
            try_delete_in_r2(s, &[("at", ())], p2);
        }
        Standard::Ira => {
            if start < pv || start == 0 || s[start - 1] != 'e' {
                return false;
            }
            replace_suffix(s, start, "ir");
        }
    }
    true
}

const VERB: &[(&str, ())] = &[
    ("ada", ()), ("ida", ()), ("ia", ()), ("aria", ()), ("eria", ()), ("iria", ()),
    ("ará", ()), ("ara", ()), ("erá", ()), ("era", ()), ("irá", ()), ("ava", ()),
    ("asse", ()), ("esse", ()), ("isse", ()), ("aste", ()), ("este", ()), ("iste", ()),
    ("ei", ()), ("arei", ()), ("erei", ()), ("irei", ()), ("am", ()), ("iam", ()),
    ("ariam", ()), ("eriam", ()), ("iriam", ()), ("aram", ()), ("eram", ()), ("iram", ()),
    ("avam", ()), ("em", ()), ("arem", ()), ("erem", ()), ("irem", ()), ("assem", ()),
    ("essem", ()), ("issem", ()), ("ado", ()), ("ido", ()), ("ando", ()), ("endo", ()),
    ("indo", ()), ("ara~o", ()), ("era~o", ()), ("ira~o", ()), ("ar", ()), ("er", ()),
    ("ir", ()), ("as", ()), ("adas", ()), ("idas", ()), ("ias", ()), ("arias", ()),
    ("erias", ()), ("irias", ()), ("arás", ()), ("aras", ()), ("erás", ()), ("eras", ()),
    ("irás", ()), ("avas", ()), ("es", ()), ("ardes", ()), ("erdes", ()), ("irdes", ()),
    ("ares", ()), ("eres", ()), ("ires", ()), ("asses", ()), ("esses", ()), ("isses", ()),
    ("astes", ()), ("estes", ()), ("istes", ()), ("is", ()), ("ais", ()), ("eis", ()),
    ("íeis", ()), ("aríeis", ()), ("eríeis", ()), ("iríeis", ()), ("áreis", ()),
    ("areis", ()), ("éreis", ()), ("ereis", ()), ("íreis", ()), ("ireis", ()),
    ("ásseis", ()), ("ésseis", ()), ("ísseis", ()), ("áveis", ()), ("ados", ()),
    ("idos", ()), ("ámos", ()), ("amos", ()), ("íamos", ()), ("aríamos", ()),
    ("eríamos", ()), ("iríamos", ()), ("áramos", ()), ("éramos", ()), ("íramos", ()),
    ("ávamos", ()), ("emos", ()), ("aremos", ()), ("eremos", ()), ("iremos", ()),
    ("ássemos", ()), ("êssemos", ()), ("íssemos", ()), ("imos", ()), ("armos", ()),
    ("ermos", ()), ("irmos", ()), ("eu", ()), ("iu", ()), ("ou", ()), ("ira", ()),
    ("iras", ()),
];

/// Step 2: delete the longest verb suffix lying inside RV.
fn verb_suffix(s: &mut Vec<char>, pv: usize) -> bool {
    let Some(idx) = longest_suffix(s, VERB, pv) else {
        return false;
    };
    let start = s.len() - VERB[idx].0.chars().count();
    s.truncate(start);
    true
}

const RESIDUAL: &[(&str, ())] = &[
    ("os", ()), ("a", ()), ("i", ()), ("o", ()), ("á", ()), ("í", ()), ("ó", ()),
];

/// Step 4: delete a residual suffix in RV.
fn residual_suffix(s: &mut Vec<char>, pv: usize) {
    let Some(idx) = longest_suffix(s, RESIDUAL, 0) else {
        return;
    };
    let start = s.len() - RESIDUAL[idx].0.chars().count();
    if start >= pv {
        s.truncate(start);
    }
}

/// Step 5: residual form.
fn residual_form(s: &mut Vec<char>, pv: usize) {
    let Some(&last) = s.last() else {
        return;
    };
    match last {
        'e' | 'é' | 'ê' => {
            let start = s.len() - 1;
            if start < pv {
                return;
            }
            s.truncate(start);
            if (ends_with(s, "gu") || ends_with(s, "ci")) && s.len() - 1 >= pv {
                s.pop();
            }
        }
        'ç' => {
            let start = s.len() - 1;
            replace_suffix(s, start, "c");
        }
        _ => {}
    }
}

/// Stem an already-lowercased Portuguese word with the Snowball Portuguese
/// algorithm.
///
/// Never panics; non-letter content is returned unchanged.
pub fn stem(word: &str) -> String {
    // Prelude.
    let mut s: Vec<char> = Vec::with_capacity(word.len());
    for c in word.chars() {
        match c {
            'ã' => s.extend(['a', '~']),
            'õ' => s.extend(['o', '~']),
            _ => s.push(c),
        }
    }
    let (pv, p1, p2) = mark_regions(&s);

    let altered = standard_suffix(&mut s, pv, p1, p2) || verb_suffix(&mut s, pv);
    if altered {
        // Step 3.
        if ends_with(&s, "ci") && s.len() - 1 >= pv {
            s.pop();
        }
    } else {
        residual_suffix(&mut s, pv);
    }
    residual_form(&mut s, pv);

    // Postlude.
    let mut out = String::with_capacity(word.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        if i + 1 < s.len() && s[i + 1] == '~' && (c == 'a' || c == 'o') {
            out.push(if c == 'a' { 'ã' } else { 'õ' });
            i += 2;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
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

    /// Vectors taken from the official Snowball Portuguese sample vocabulary.
    #[test]
    fn official_sample_vocabulary() {
        check(&[
            ("boa", "boa"),
            ("boainain", "boainain"),
            ("boas", "boas"),
            ("bôas", "bôas"),
            ("boassu", "boassu"),
            ("boataria", "boat"),
            ("boate", "boat"),
            ("boates", "boat"),
            ("boatos", "boat"),
            ("bob", "bob"),
            ("boba", "bob"),
            ("bobagem", "bobag"),
            ("bobagens", "bobagens"),
            ("bobalhões", "bobalhõ"),
            ("bobear", "bob"),
            ("bobeira", "bobeir"),
            ("bobinho", "bobinh"),
            ("bobinhos", "bobinh"),
            ("bobo", "bob"),
            ("bobs", "bobs"),
            ("boca", "boc"),
            ("bocadas", "boc"),
            ("bocadinho", "bocadinh"),
            ("bocado", "boc"),
            ("bocaiúva", "bocaiúv"),
            ("boçal", "boçal"),
            ("bocarra", "bocarr"),
            ("bocas", "boc"),
            ("bode", "bod"),
            ("bodoque", "bodoqu"),
            ("body", "body"),
            ("boeing", "boeing"),
            ("boem", "boem"),
            ("boemia", "boem"),
            ("boêmio", "boêmi"),
            ("boêmios", "boêmi"),
            ("bogotá", "bogot"),
            ("boi", "boi"),
            ("bóia", "bói"),
            ("boiando", "boi"),
            ("quiabo", "quiab"),
            ("quicaram", "quic"),
            ("quickly", "quickly"),
            ("quieto", "quiet"),
            ("quietos", "quiet"),
            ("quilate", "quilat"),
            ("quilates", "quilat"),
            ("quilinhos", "quilinh"),
            ("quilo", "quil"),
            ("quilombo", "quilomb"),
            ("quilométricas", "quilométr"),
            ("quilométricos", "quilométr"),
            ("quilômetro", "quilômetr"),
            ("quilômetros", "quilômetr"),
            ("quilos", "quil"),
            ("química", "químic"),
            ("químicas", "químic"),
            ("químico", "químic"),
            ("químicos", "químic"),
            ("quimioterapia", "quimioterap"),
            ("quimioterápicos", "quimioteráp"),
            ("quimono", "quimon"),
            ("quincas", "quinc"),
            ("quinhão", "quinhã"),
            ("quinhentos", "quinhent"),
            ("quinn", "quinn"),
            ("quino", "quin"),
            ("quinta", "quint"),
            ("quintal", "quintal"),
            ("quintana", "quintan"),
            ("quintanilha", "quintanilh"),
            ("quintão", "quintã"),
            ("quintessência", "quintessent"),
            ("quintino", "quintin"),
            ("quinto", "quint"),
            ("quintos", "quint"),
            ("quintuplicou", "quintuplic"),
            ("quinze", "quinz"),
            ("quinzena", "quinzen"),
            ("quiosque", "quiosqu"),
        ]);
    }

    /// Suffix-chain behaviour of step 1 (`-mente`, `-idade`, `-ivo`).
    #[test]
    fn standard_suffix_chains() {
        check(&[
            ("amente", "ament"),
            ("somente", "soment"),
            ("claramente", "clar"),
            ("felizmente", "feliz"),
            ("realmente", "realment"),
            ("ativamente", "ativ"),
            ("finalidade", "final"),
            ("ativo", "ativ"),
        ]);
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("ã"), "ã");
        assert_eq!(stem("ãos"), "ãos");
        assert_eq!(stem("ção"), "çã");
        assert_eq!(stem("e"), "e");
        assert_eq!(stem("é"), "é");
        assert_eq!(stem("gue"), "gue");
        assert_eq!(stem("ci"), "ci");
        assert_eq!(stem("ce"), "ce");
        assert_eq!(stem("ç"), "c");
        assert_eq!(stem("123"), "123");
        assert_eq!(stem("..."), "...");
        assert_eq!(stem("日本語"), "日本語");
        for w in ["~", "a~", "~~", "ãã", "õõo", "x", "xy", "mente", "ira", "eira", "os"] {
            let _ = stem(w);
        }
    }
}
