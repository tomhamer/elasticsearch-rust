//! Snowball French stemmer.
//!
//! A faithful port of `algorithms/french.sbl` at Snowball commit
//! `34f3612e5e8c` (the revision in every released Lucene 10.x / ES); later
//! Snowball additions (`ni`+vowel RV rule, `-oux`, `-aise(s)`) are excluded. The input is assumed to be lowercased; the
//! stemmer operates on a `Vec<char>` and never panics.
//!
//! Steps:
//!
//! 1. **Elisions** — a leading `c' d' j' l' m' n' s' t' z'` or `qu'` is
//!    removed (unless the apostrophe ends the word).
//! 2. **Prelude** — `u` and `i` between vowels become `U`/`I`, `y` next to
//!    a vowel becomes `Y`, `u` after `q` becomes `U`, and `ë`/`ï` become
//!    `He`/`Hi`. The upper-case letters and `H` are not vowels.
//! 3. **Regions** — `RV` starts after the third letter when the word begins
//!    with two vowels or with `par`, `col` or `tap`; otherwise
//!    after the first vowel not at the start. `R1`/`R2` are the usual
//!    Snowball regions.
//! 4. **Step 1** standard suffix removal, **Step 2a** verb suffixes
//!    beginning with `i`, **Step 2b** other verb suffixes (both confined to
//!    `RV`), then **Step 3** (`Y`→`i`, `ç`→`c`) if one of those changed the
//!    word, otherwise **Step 4** residual suffix removal.
//! 5. **Step 5** undoubling (`enn onn ett ell eill`) and **Step 6**
//!    un-accenting of a final `é`/`è` followed by non-vowels.
//! 6. **Postlude** — `I U Y` back to lower case, `He`/`Hi` back to `ë`/`ï`,
//!    any other `H` dropped.

/// Lower-case French vowels (as defined by the algorithm). The marked
/// letters `U`, `I`, `Y` and the diaeresis marker `H` are not vowels.
const VOWELS: &[char] = &[
    'a', 'e', 'i', 'o', 'u', 'y', 'â', 'à', 'ë', 'é', 'ê', 'è', 'ï', 'î', 'ô', 'û', 'ù',
];

fn is_v(c: char) -> bool {
    VOWELS.contains(&c)
}

/// Stem a single lowercased French word with the Snowball algorithm.
///
/// Non-letters pass through unchanged; the function never panics, even
/// for empty input or arbitrary Unicode.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    elisions(&mut w);
    prelude(&mut w);
    let (pv, p1, p2) = mark_regions(&w);
    let mut st = St { w, pv, p1, p2 };

    let changed = st.standard_suffix() || st.i_verb_suffix() || st.verb_suffix();
    if changed {
        // Step 3: final Y -> i, final ç -> c.
        if let Some(last) = st.w.last_mut() {
            if *last == 'Y' {
                *last = 'i';
            } else if *last == 'ç' {
                *last = 'c';
            }
        }
    } else {
        st.residual_suffix();
    }
    st.un_double();
    st.un_accent();
    postlude(&st.w)
}

/// Remove a leading elision (`l'`, `qu'`, ...) when something follows it.
fn elisions(w: &mut Vec<char>) {
    let n = if w.len() >= 2 && w[1] == '\'' && "cdjlmnstz".contains(w[0]) {
        2
    } else if w.len() >= 3 && w[0] == 'q' && w[1] == 'u' && w[2] == '\'' {
        3
    } else {
        return;
    };
    if n < w.len() {
        w.drain(..n);
    }
}

/// Mark `u`/`i`/`y` that should not be treated as vowels and replace
/// `ë`/`ï` with `He`/`Hi`. Mirrors Snowball's `repeat goto (...)`.
fn prelude(w: &mut Vec<char>) {
    let mut i = 0;
    while i < w.len() {
        let n = w.len();
        if is_v(w[i]) && i + 1 < n {
            let c = w[i + 1];
            if c == 'u' && i + 2 < n && is_v(w[i + 2]) {
                w[i + 1] = 'U';
                continue;
            }
            if c == 'i' && i + 2 < n && is_v(w[i + 2]) {
                w[i + 1] = 'I';
                continue;
            }
            if c == 'y' {
                w[i + 1] = 'Y';
                continue;
            }
        }
        if w[i] == 'ë' {
            w[i] = 'H';
            w.insert(i + 1, 'e');
            continue;
        }
        if w[i] == 'ï' {
            w[i] = 'H';
            w.insert(i + 1, 'i');
            continue;
        }
        if w[i] == 'y' && i + 1 < n && is_v(w[i + 1]) {
            w[i] = 'Y';
            continue;
        }
        if w[i] == 'q' && i + 1 < n && w[i + 1] == 'u' {
            w[i + 1] = 'U';
            continue;
        }
        i += 1;
    }
}

fn starts_with(w: &[char], s: &str) -> bool {
    let mut it = w.iter();
    s.chars().all(|c| it.next() == Some(&c))
}

/// Position after the first non-vowel following a vowel at or after
/// `start`, or `w.len()` if there is none.
fn region_after(w: &[char], start: usize) -> usize {
    let n = w.len();
    let mut i = start;
    while i < n && !is_v(w[i]) {
        i += 1;
    }
    if i >= n {
        return n;
    }
    i += 1;
    while i < n && is_v(w[i]) {
        i += 1;
    }
    if i >= n {
        return n;
    }
    i + 1
}

/// Compute the start positions of `RV`, `R1` and `R2`.
fn mark_regions(w: &[char]) -> (usize, usize, usize) {
    let n = w.len();
    let pv = if n >= 3 && is_v(w[0]) && is_v(w[1]) {
        3
    } else if starts_with(w, "par") || starts_with(w, "col") || starts_with(w, "tap") {
        3
    } else {
        match (1..n).find(|&i| is_v(w[i])) {
            Some(i) => i + 1,
            None => n,
        }
    };
    let p1 = region_after(w, 0);
    let p2 = if p1 < n { region_after(w, p1) } else { n };
    (pv, p1, p2)
}

/// Undo the prelude markings.
fn postlude(w: &[char]) -> String {
    let mut out = String::with_capacity(w.len());
    let mut i = 0;
    while i < w.len() {
        match w[i] {
            'I' => out.push('i'),
            'U' => out.push('u'),
            'Y' => out.push('y'),
            'H' => {
                if i + 1 < w.len() && w[i + 1] == 'e' {
                    out.push('ë');
                    i += 1;
                } else if i + 1 < w.len() && w[i + 1] == 'i' {
                    out.push('ï');
                    i += 1;
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    out
}

/// Working state for the backward (suffix) steps.
struct St {
    w: Vec<char>,
    pv: usize,
    p1: usize,
    p2: usize,
}

/// Index of the start of `suffix` if `w[..end]` ends with it.
fn suffix_start(w: &[char], end: usize, suffix: &str) -> Option<usize> {
    let n = suffix.chars().count();
    if n > end {
        return None;
    }
    let start = end - n;
    if w[start..end].iter().copied().eq(suffix.chars()) {
        Some(start)
    } else {
        None
    }
}

/// Longest suffix of `w[..end]` among `list` whose start is `>= limit`.
fn longest<'a>(w: &[char], end: usize, limit: usize, list: &[&'a str]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for &s in list {
        if let Some(start) = suffix_start(w, end, s) {
            if start >= limit && best.is_none_or(|(b, _)| start < b) {
                best = Some((start, s));
            }
        }
    }
    best
}

impl St {
    fn len(&self) -> usize {
        self.w.len()
    }

    fn rv(&self, i: usize) -> bool {
        i >= self.pv
    }

    fn r1(&self, i: usize) -> bool {
        i >= self.p1
    }

    fn r2(&self, i: usize) -> bool {
        i >= self.p2
    }

    /// Start index of `s` if the current word ends with it.
    fn ends(&self, s: &str) -> Option<usize> {
        suffix_start(&self.w, self.len(), s)
    }

    /// Start index of `s` if `w[..end]` ends with it.
    fn ends_at(&self, end: usize, s: &str) -> Option<usize> {
        suffix_start(&self.w, end, s)
    }

    fn truncate(&mut self, i: usize) {
        self.w.truncate(i);
    }

    fn replace(&mut self, i: usize, s: &str) {
        self.w.truncate(i);
        self.w.extend(s.chars());
    }

    /// Step 1: standard suffix removal. Returns the Snowball signal: `true`
    /// when a suffix was handled, `false` when the step "failed" (which
    /// includes the `ment`-family rules that alter the word but fall
    /// through to step 2a).
    fn standard_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "ance", "iqUe", "isme", "able", "iste", "eux", "ances", "iqUes", "ismes", "ables",
            "istes", "atrice", "ateur", "ation", "atrices", "ateurs", "ations", "logie", "logies",
            "usion", "ution", "usions", "utions", "ence", "ences", "ement", "ements", "ité",
            "ités", "if", "ive", "ifs", "ives", "eaux", "aux", "euse", "euses",
            "issement", "issements", "amment", "emment", "ment", "ments",
        ];
        let Some((i, suf)) = longest(&self.w, self.len(), 0, LIST) else {
            return false;
        };
        match suf {
            "ance" | "iqUe" | "isme" | "able" | "iste" | "eux" | "ances" | "iqUes" | "ismes"
            | "ables" | "istes" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
            }
            "atrice" | "ateur" | "ation" | "atrices" | "ateurs" | "ations" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some(j) = self.ends("ic") {
                    if self.r2(j) {
                        self.truncate(j);
                    } else {
                        self.replace(j, "iqU");
                    }
                }
            }
            "logie" | "logies" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "log");
            }
            "usion" | "ution" | "usions" | "utions" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "u");
            }
            "ence" | "ences" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "ent");
            }
            "ement" | "ements" => {
                if !self.rv(i) {
                    return false;
                }
                self.truncate(i);
                let end = self.len();
                if let Some(j) = self.ends_at(end, "iv") {
                    if self.r2(j) {
                        self.truncate(j);
                        if let Some(k) = self.ends("at") {
                            if self.r2(k) {
                                self.truncate(k);
                            }
                        }
                    }
                } else if let Some(j) = self.ends_at(end, "eus") {
                    if self.r2(j) {
                        self.truncate(j);
                    } else if self.r1(j) {
                        self.replace(j, "eux");
                    }
                } else if let Some(j) = self.ends_at(end, "abl").or_else(|| self.ends_at(end, "iqU")) {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                } else if let Some(j) = self.ends_at(end, "ièr").or_else(|| self.ends_at(end, "Ièr")) {
                    if self.rv(j) {
                        self.replace(j, "i");
                    }
                }
            }
            "ité" | "ités" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                let end = self.len();
                if let Some(j) = self.ends_at(end, "abil") {
                    if self.r2(j) {
                        self.truncate(j);
                    } else {
                        self.replace(j, "abl");
                    }
                } else if let Some(j) = self.ends_at(end, "ic") {
                    if self.r2(j) {
                        self.truncate(j);
                    } else {
                        self.replace(j, "iqU");
                    }
                } else if let Some(j) = self.ends_at(end, "iv") {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                }
            }
            "if" | "ive" | "ifs" | "ives" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some(j) = self.ends("at") {
                    if self.r2(j) {
                        self.truncate(j);
                        if let Some(k) = self.ends("ic") {
                            if self.r2(k) {
                                self.truncate(k);
                            } else {
                                self.replace(k, "iqU");
                            }
                        }
                    }
                }
            }
            "eaux" => self.replace(i, "eau"),
            "aux" => {
                if !self.r1(i) {
                    return false;
                }
                self.replace(i, "al");
            }
            "euse" | "euses" => {
                if self.r2(i) {
                    self.truncate(i);
                } else if self.r1(i) {
                    self.replace(i, "eux");
                } else {
                    return false;
                }
            }
            "issement" | "issements" => {
                if !self.r1(i) || i == 0 || is_v(self.w[i - 1]) {
                    return false;
                }
                self.truncate(i);
            }
            "amment" => {
                if self.rv(i) {
                    self.replace(i, "ant");
                }
                return false;
            }
            "emment" => {
                if self.rv(i) {
                    self.replace(i, "ent");
                }
                return false;
            }
            "ment" | "ments" => {
                if i >= 1 && is_v(self.w[i - 1]) && self.rv(i - 1) {
                    self.truncate(i);
                }
                return false;
            }
            _ => return false,
        }
        true
    }

    /// Step 2a: verb suffixes beginning with `i`, searched within `RV`.
    fn i_verb_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "îmes", "ît", "îtes", "i", "ie", "ies", "ir", "ira", "irai", "iraIent", "irais",
            "irait", "iras", "irent", "irez", "iriez", "irions", "irons", "iront", "is",
            "issaIent", "issais", "issait", "issant", "issante", "issantes", "issants", "isse",
            "issent", "isses", "issez", "issiez", "issions", "issons", "it",
        ];
        let Some((i, _)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return false;
        };
        // The preceding character must also be in RV and be a non-vowel
        // other than H.
        if i == 0 || i - 1 < self.pv {
            return false;
        }
        let c = self.w[i - 1];
        if c == 'H' || is_v(c) {
            return false;
        }
        self.truncate(i);
        true
    }

    /// Step 2b: other verb suffixes, searched within `RV`.
    fn verb_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "ions", "é", "ée", "ées", "és", "èrent", "er", "era", "erai", "eraIent", "erais",
            "erait", "eras", "erez", "eriez", "erions", "erons", "eront", "ez", "iez", "âmes",
            "ât", "âtes", "a", "ai", "aIent", "ait", "ant", "ante", "antes", "ants", "as", "asse",
            "assent", "asses", "assiez", "assions", "ais",
        ];
        let Some((i, suf)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return false;
        };
        match suf {
            "ions" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
            }
            "âmes" | "ât" | "âtes" | "a" | "ai" | "aIent" | "ait" | "ant" | "ante" | "antes"
            | "ants" | "as" | "asse" | "assent" | "asses" | "assiez" | "assions" | "ais" => {
                let mut cut = i;
                if i >= 1 && self.w[i - 1] == 'e' && self.rv(i - 1) {
                    cut = i - 1;
                }
                self.truncate(cut);
            }
            _ => self.truncate(i),
        }
        true
    }

    /// Step 4: residual suffix removal.
    fn residual_suffix(&mut self) {
        let n = self.len();
        if n >= 1 && self.w[n - 1] == 's' {
            let after_hi = n >= 3 && self.w[n - 3] == 'H' && self.w[n - 2] == 'i';
            let keep = n < 2 || "aiouès".contains(self.w[n - 2]);
            if after_hi || !keep {
                self.truncate(n - 1);
            }
        }
        const LIST: &[&str] = &["ion", "ier", "ière", "Ier", "Ière", "e"];
        let Some((i, suf)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return;
        };
        match suf {
            "ion" => {
                if self.r2(i) && i >= 1 && self.rv(i - 1) && matches!(self.w[i - 1], 's' | 't') {
                    self.truncate(i);
                }
            }
            "e" => self.truncate(i),
            _ => self.replace(i, "i"),
        }
    }

    /// Step 5: undouble `enn onn ett ell eill`.
    fn un_double(&mut self) {
        if ["enn", "onn", "ett", "ell", "eill"].iter().any(|s| self.ends(s).is_some()) {
            self.w.pop();
        }
    }

    /// Step 6: un-accent a final `é`/`è` followed by at least one non-vowel.
    fn un_accent(&mut self) {
        let n = self.len();
        let mut i = n;
        while i > 0 && !is_v(self.w[i - 1]) {
            i -= 1;
        }
        if i < n && i >= 1 && matches!(self.w[i - 1], 'é' | 'è') {
            self.w[i - 1] = 'e';
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(cases: &[(&str, &str)]) {
        let mut failed = Vec::new();
        for &(input, expected) in cases {
            let got = stem(input);
            if got != expected {
                failed.push(format!("{input:?}: expected {expected:?}, got {got:?}"));
            }
        }
        assert!(failed.is_empty(), "failing vectors:\n{}", failed.join("\n"));
    }

    /// Vectors taken from the official Snowball French sample
    /// vocabulary (voc.txt / output.txt).
    #[test]
    fn official_sample_vectors() {
        check(&[
            ("continu", "continu"),
            ("continua", "continu"),
            ("continuait", "continu"),
            ("continuant", "continu"),
            ("continuation", "continu"),
            ("continue", "continu"),
            ("continuel", "continuel"),
            ("continuelle", "continuel"),
            ("continuellement", "continuel"),
            ("continuelles", "continuel"),
            ("continuels", "continuel"),
            ("continuer", "continu"),
            ("continuera", "continu"),
            ("continuerait", "continu"),
            ("continueront", "continu"),
            ("continuez", "continu"),
            ("continuité", "continu"),
            ("continuons", "continuon"),
            ("continué", "continu"),
            ("contorsions", "contors"),
            ("contour", "contour"),
            ("contournait", "contourn"),
            ("contournant", "contourn"),
            ("contourne", "contourn"),
            ("contours", "contour"),
            ("contractait", "contract"),
            ("contracter", "contract"),
            ("contractions", "contract"),
            ("contracté", "contract"),
            ("contractée", "contract"),
            ("contractés", "contract"),
            ("contradictoirement", "contradictoir"),
            ("contradictoires", "contradictoir"),
            ("contraindre", "contraindr"),
            ("contraint", "contraint"),
            ("contrainte", "contraint"),
            ("contraintes", "contraint"),
            ("contraire", "contrair"),
            ("contraires", "contrair"),
            ("contraria", "contrari"),
            ("main", "main"),
            ("mains", "main"),
            ("maintenaient", "mainten"),
            ("maintenait", "mainten"),
            ("maintenant", "mainten"),
            ("maintenir", "mainten"),
            ("maintenue", "maintenu"),
            ("maintien", "maintien"),
            ("maintint", "maintint"),
            ("maire", "mair"),
            ("maires", "mair"),
            ("mairie", "mair"),
            ("mais", "mais"),
            ("maison", "maison"),
            ("maisons", "maison"),
            ("maistre", "maistr"),
            ("maitre", "maitr"),
            ("majestueuse", "majestu"),
            ("majestueusement", "majestu"),
            ("majestueux", "majestu"),
            ("majesté", "majest"),
            ("majeur", "majeur"),
            ("majeure", "majeur"),
            ("major", "major"),
            ("majordome", "majordom"),
            ("majordomes", "majordom"),
            ("majorité", "major"),
            ("majorités", "major"),
            ("mal", "mal"),
            ("malacca", "malacc"),
            ("malade", "malad"),
            ("malades", "malad"),
            ("maladie", "malad"),
            ("maladies", "malad"),
            ("maladive", "malad"),
            ("maladresse", "maladress"),
            ("maladresses", "maladress"),
            ("maladroit", "maladroit"),
            ("maladroite", "maladroit"),
            ("maladroitement", "maladroit"),
        ]);
    }

    /// More official vectors exercising elisions, diaereses, accents and
    /// the `RV` exceptions.
    #[test]
    fn official_special_cases() {
        check(&[
            ("c'est", "est"),
            ("d'accéder", "acced"),
            ("j'aime", "aim"),
            ("l'", "l'"),
            ("l'yonne", "yon"),
            ("qu'elles", "elle"),
            ("s'achève", "achev"),
            ("n'hésite", "hésit"),
            ("aujourd'hui", "aujourd'hui"),
            ("abrégés", "abreg"),
            ("abréviations", "abrévi"),
            ("accroître", "accroîtr"),
            ("académie", "academ"),
            ("aboyèrent", "aboi"),
            ("abbés", "abbé"),
            ("accès", "acces"),
        ]);
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("é"), "é");
        assert_eq!(stem("ë"), "ë");
        assert_eq!(stem("ï"), "ï");
        assert_eq!(stem("s"), "s");
        assert_eq!(stem("qu"), "qu");
        assert_eq!(stem("qu'"), "qu'");
        assert_eq!(stem("12345"), "12345");
        assert_eq!(stem("...!?"), "...!?");
        for w in ["yy", "aie", "ni", "par", "ment", "ïs", "ëment", "日本語", "😀", "a'b'c"] {
            let _ = stem(w);
        }
    }
}
