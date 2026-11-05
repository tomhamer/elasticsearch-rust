//! Snowball Italian stemmer.
//!
//! A faithful transcription of the official Snowball Italian algorithm
//! (snowballstem.org, 3.x). The input is assumed to be lowercased; the
//! stemmer operates on a `Vec<char>` and never panics.
//!
//! Steps:
//!
//! 1. **Elisions** — a leading `d' l' m' s' t' v' all' dall' dell' gl'
//!    nell' quell' quest' sull' tutt' un'` is removed (unless the
//!    apostrophe ends the word).
//! 2. **Prelude** — acute accents become grave accents, `qu` becomes `qU`,
//!    and `u`/`i` between vowels become `U`/`I` (which are not vowels).
//! 3. **Regions** — `RV` as in the Spanish stemmer (second letter a
//!    consonant -> after the next vowel; two initial vowels -> after the
//!    next consonant; consonant-vowel -> after the third letter), except
//!    that a word beginning `divan` has `RV` after that prefix. `R1`/`R2`
//!    are the usual Snowball regions.
//! 4. **Step 0** attached pronoun removal after `ando endo` (deleted) or
//!    `ar er ir` (replaced by `e`) in `RV`.
//! 5. **Step 1** standard suffix removal; if nothing was removed,
//!    **Step 2** verb suffix removal within `RV`.
//! 6. **Step 3a** deletion of a final vowel (`a e i o à è ì ò`) in `RV`
//!    plus a preceding `i` in `RV`; **Step 3b** final `ch`/`gh` -> `c`/`g`
//!    in `RV`.
//! 7. **Postlude** — `I` and `U` back to lower case.

/// Italian vowels after the prelude (acute accents have become grave).
const VOWELS: &[char] = &['a', 'e', 'i', 'o', 'u', 'à', 'è', 'ì', 'ò', 'ù'];

fn is_v(c: char) -> bool {
    VOWELS.contains(&c)
}

/// Stem a single lowercased Italian word with the Snowball algorithm.
///
/// Non-letters pass through unchanged; the function never panics, even
/// for empty input or arbitrary Unicode.
pub fn stem(word: &str) -> String {
    let mut w: Vec<char> = word.chars().collect();
    elisions(&mut w);
    prelude(&mut w);
    let (pv, p1, p2) = mark_regions(&w);
    let mut st = St { w, pv, p1, p2 };
    st.attached_pronoun();
    let _ = st.standard_suffix() || st.verb_suffix();
    st.vowel_suffix();
    postlude(&st.w)
}

fn starts_with(w: &[char], s: &str) -> bool {
    let mut it = w.iter();
    s.chars().all(|c| it.next() == Some(&c))
}

/// Remove a leading elision (`l'`, `dell'`, ...) when something follows it.
fn elisions(w: &mut Vec<char>) {
    const PREFIXES: &[&str] = &[
        "d'", "l'", "m'", "s'", "t'", "v'", "all'", "dall'", "dell'", "gl'", "nell'", "quell'",
        "quest'", "sull'", "tutt'", "un'",
    ];
    let mut best = 0;
    for p in PREFIXES {
        let len = p.chars().count();
        if len > best && starts_with(w, p) {
            best = len;
        }
    }
    if best > 0 && best < w.len() {
        w.drain(..best);
    }
}

/// Replace acute accents with grave ones, mark `u` after `q`, and mark
/// `u`/`i` between vowels as `U`/`I`.
fn prelude(w: &mut [char]) {
    let n = w.len();
    let mut i = 0;
    while i < n {
        match w[i] {
            'á' => w[i] = 'à',
            'é' => w[i] = 'è',
            'í' => w[i] = 'ì',
            'ó' => w[i] = 'ò',
            'ú' => w[i] = 'ù',
            'q' if i + 1 < n && w[i + 1] == 'u' => {
                w[i + 1] = 'U';
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    let mut i = 0;
    while i < n {
        if is_v(w[i]) && i + 2 < n && is_v(w[i + 2]) {
            match w[i + 1] {
                'u' => {
                    w[i + 1] = 'U';
                    continue;
                }
                'i' => {
                    w[i + 1] = 'I';
                    continue;
                }
                _ => {}
            }
        }
        i += 1;
    }
}

/// Position after the first vowel at or after `start`, or `w.len()`.
fn gopast_v(w: &[char], start: usize) -> usize {
    match (start..w.len()).find(|&i| is_v(w[i])) {
        Some(i) => i + 1,
        None => w.len(),
    }
}

/// Position after the first non-vowel at or after `start`, or `w.len()`.
fn gopast_non_v(w: &[char], start: usize) -> usize {
    match (start..w.len()).find(|&i| !is_v(w[i])) {
        Some(i) => i + 1,
        None => w.len(),
    }
}

/// Position after the first non-vowel following a vowel at or after
/// `start`, or `w.len()` if there is none.
fn region_after(w: &[char], start: usize) -> usize {
    let n = w.len();
    let i = gopast_v(w, start);
    if i >= n {
        return n;
    }
    gopast_non_v(w, i)
}

/// Compute the start positions of `RV`, `R1` and `R2`.
fn mark_regions(w: &[char]) -> (usize, usize, usize) {
    let n = w.len();
    let mut pv = n;
    if n >= 2 {
        if is_v(w[0]) {
            if !is_v(w[1]) {
                if (2..n).any(|i| is_v(w[i])) {
                    pv = gopast_v(w, 2);
                }
            } else if (2..n).any(|i| !is_v(w[i])) {
                pv = gopast_non_v(w, 2);
            }
        } else if starts_with(w, "divan") {
            pv = 5;
        } else if !is_v(w[1]) {
            if (2..n).any(|i| is_v(w[i])) {
                pv = gopast_v(w, 2);
            }
        } else if n >= 3 {
            pv = 3;
        }
    }
    let p1 = region_after(w, 0);
    let p2 = if p1 < n { region_after(w, p1) } else { n };
    (pv, p1, p2)
}

/// Undo the prelude markings.
fn postlude(w: &[char]) -> String {
    w.iter()
        .map(|&c| match c {
            'I' => 'i',
            'U' => 'u',
            c => c,
        })
        .collect()
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

    fn truncate(&mut self, i: usize) {
        self.w.truncate(i);
    }

    fn replace(&mut self, i: usize, s: &str) {
        self.w.truncate(i);
        self.w.extend(s.chars());
    }

    /// Step 0: attached pronoun.
    fn attached_pronoun(&mut self) {
        const PRONOUNS: &[&str] = &[
            "ci", "gli", "la", "le", "li", "lo", "mi", "ne", "si", "ti", "vi", "sene", "gliela",
            "gliele", "glieli", "glielo", "gliene", "mela", "mele", "meli", "melo", "mene", "tela",
            "tele", "teli", "telo", "tene", "cela", "cele", "celi", "celo", "cene", "vela", "vele",
            "veli", "velo", "vene",
        ];
        const ENDINGS: &[&str] = &["ando", "endo", "ar", "er", "ir"];
        let Some((p, _)) = longest(&self.w, self.len(), 0, PRONOUNS) else {
            return;
        };
        let Some((e, ending)) = longest(&self.w, p, 0, ENDINGS) else {
            return;
        };
        if !self.rv(e) {
            return;
        }
        match ending {
            "ando" | "endo" => self.truncate(p),
            _ => self.replace(p, "e"),
        }
    }

    /// Step 1: standard suffix removal. Returns `true` if a suffix was
    /// handled.
    fn standard_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "anza", "anze", "ico", "ici", "ica", "ice", "iche", "ichi", "ismo", "ismi", "abile",
            "abili", "ibile", "ibili", "ista", "iste", "isti", "istà", "istè", "istì", "oso", "osi",
            "osa", "ose", "mente", "atrice", "atrici", "ante", "anti", "azione", "azioni", "atore",
            "atori", "logia", "logie", "uzione", "uzioni", "usione", "usioni", "enza", "enze",
            "amento", "amenti", "imento", "imenti", "amente", "ità", "ivo", "ivi", "iva", "ive",
        ];
        let Some((i, suf)) = longest(&self.w, self.len(), 0, LIST) else {
            return false;
        };
        match suf {
            "anza" | "anze" | "ico" | "ici" | "ica" | "ice" | "iche" | "ichi" | "ismo" | "ismi"
            | "abile" | "abili" | "ibile" | "ibili" | "ista" | "iste" | "isti" | "istà" | "istè"
            | "istì" | "oso" | "osi" | "osa" | "ose" | "mente" | "atrice" | "atrici" | "ante"
            | "anti" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
            }
            "azione" | "azioni" | "atore" | "atori" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some(j) = self.ends("ic") {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                }
            }
            "logia" | "logie" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "log");
            }
            "uzione" | "uzioni" | "usione" | "usioni" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "u");
            }
            "enza" | "enze" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "ente");
            }
            "amento" | "amenti" | "imento" | "imenti" => {
                if !self.rv(i) {
                    return false;
                }
                self.truncate(i);
            }
            "amente" => {
                if !self.r1(i) {
                    return false;
                }
                self.truncate(i);
                if let Some((j, pre)) = longest(&self.w, self.len(), 0, &["iv", "os", "ic", "abil"]) {
                    if self.r2(j) {
                        self.truncate(j);
                        if pre == "iv" {
                            if let Some(k) = self.ends("at") {
                                if self.r2(k) {
                                    self.truncate(k);
                                }
                            }
                        }
                    }
                }
            }
            "ità" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some((j, _)) = longest(&self.w, self.len(), 0, &["abil", "ic", "iv"]) {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                }
            }
            "ivo" | "ivi" | "iva" | "ive" => {
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
                            }
                        }
                    }
                }
            }
            _ => return false,
        }
        true
    }

    /// Step 2: verb suffixes, searched within `RV`.
    fn verb_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "ammo", "ando", "ano", "are", "arono", "asse", "assero", "assi", "assimo", "ata", "ate",
            "ati", "ato", "ava", "avamo", "avano", "avate", "avi", "avo", "emmo", "enda", "ende",
            "endi", "endo", "erà", "erai", "eranno", "ere", "erebbe", "erebbero", "erei", "eremmo",
            "eremo", "ereste", "eresti", "erete", "erò", "erono", "essero", "ete", "eva", "evamo",
            "evano", "evate", "evi", "evo", "Yamo", "iamo", "immo", "irà", "irai", "iranno", "ire",
            "irebbe", "irebbero", "irei", "iremmo", "iremo", "ireste", "iresti", "irete", "irò",
            "irono", "isca", "iscano", "isce", "isci", "isco", "iscono", "issero", "ita", "ite",
            "iti", "ito", "iva", "ivamo", "ivano", "ivate", "ivi", "ivo", "ono", "uta", "ute",
            "uti", "uto", "ar", "ir",
        ];
        let Some((i, _)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return false;
        };
        self.truncate(i);
        true
    }

    /// Steps 3a and 3b: final vowel (and preceding `i`) in `RV`, then
    /// `ch`/`gh` -> `c`/`g` in `RV`.
    fn vowel_suffix(&mut self) {
        let n = self.len();
        if n >= 1 && "aeioàèìò".contains(self.w[n - 1]) && self.rv(n - 1) {
            self.truncate(n - 1);
            if n >= 2 && self.w[n - 2] == 'i' && self.rv(n - 2) {
                self.truncate(n - 2);
            }
        }
        let n = self.len();
        if n >= 2 && self.w[n - 1] == 'h' && matches!(self.w[n - 2], 'c' | 'g') && self.rv(n - 2) {
            self.truncate(n - 1);
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

    /// Vectors taken from the official Snowball Italian sample
    /// vocabulary (voc.txt / output.txt).
    #[test]
    fn official_sample_vectors() {
        check(&[
            ("abbandonata", "abbandon"),
            ("abbandonate", "abbandon"),
            ("abbandonati", "abbandon"),
            ("abbandonato", "abbandon"),
            ("abbandonava", "abbandon"),
            ("abbandoneranno", "abbandon"),
            ("abbandonerà", "abbandon"),
            ("abbandonerò", "abbandon"),
            ("abbandono", "abband"),
            ("abbandonò", "abbandon"),
            ("abbaruffato", "abbaruff"),
            ("abbassamento", "abbass"),
            ("abbassando", "abbass"),
            ("abbassandola", "abbass"),
            ("abbassandole", "abbass"),
            ("abbassar", "abbass"),
            ("abbassare", "abbass"),
            ("abbassarono", "abbass"),
            ("abbassarsi", "abbass"),
            ("abbassassero", "abbass"),
            ("abbassato", "abbass"),
            ("abbassava", "abbass"),
            ("abbassi", "abbass"),
            ("abbassò", "abbass"),
            ("abbastanza", "abbast"),
            ("abbattendo", "abbatt"),
            ("abbattere", "abbatt"),
            ("abbattersi", "abbatt"),
            ("abbattesse", "abbattess"),
            ("abbatteva", "abbatt"),
            ("abbattevamo", "abbatt"),
            ("abbattevano", "abbatt"),
            ("abbattimento", "abbatt"),
            ("abbattuta", "abbatt"),
            ("abbattuti", "abbatt"),
            ("abbattuto", "abbatt"),
            ("abbatté", "abbatt"),
            ("abbellita", "abbell"),
            ("abbenché", "abbenc"),
            ("abbi", "abbi"),
            ("pronto", "pront"),
            ("pronuncerà", "pronunc"),
            ("pronuncia", "pronunc"),
            ("pronunciamento", "pronunc"),
            ("pronunciare", "pronunc"),
            ("pronunciarsi", "pronunc"),
            ("pronunciata", "pronunc"),
            ("pronunciate", "pronunc"),
            ("pronunciato", "pronunc"),
            ("pronunzia", "pronunz"),
            ("pronunziano", "pronunz"),
            ("pronunziare", "pronunz"),
            ("pronunziarle", "pronunz"),
            ("pronunziato", "pronunz"),
            ("pronunzio", "pronunz"),
            ("pronunziò", "pronunz"),
            ("propaga", "propag"),
            ("propagamento", "propag"),
            ("propaganda", "propagand"),
            ("propagare", "propag"),
            ("propagarla", "propag"),
            ("propagarsi", "propag"),
            ("propagasse", "propag"),
            ("propagata", "propag"),
            ("propagazione", "propag"),
            ("propaghino", "propaghin"),
            ("propalate", "propal"),
            ("propende", "prop"),
            ("propensi", "propens"),
            ("propensione", "propension"),
            ("propini", "propin"),
            ("propio", "prop"),
            ("propizio", "propiz"),
            ("propone", "propon"),
            ("proponendo", "propon"),
            ("proponendosi", "propon"),
            ("proponenti", "proponent"),
            ("proponeva", "propon"),
            ("proponevano", "propon"),
            ("proponga", "propong"),
        ]);
    }

    /// Official vectors exercising elisions, accents and the `qu` rule.
    #[test]
    fn official_special_cases() {
        check(&[
            ("l'anno", "anno"),
            ("dell'uomo", "uom"),
            ("un'idea", "ide"),
            ("quest'anno", "anno"),
            ("tutt'oggi", "oggi"),
            ("d'élite", "èlit"),
            ("l'école", "ècol"),
            ("sant'anna", "sant'ann"),
            ("c'era", "c'er"),
            ("bahá'í", "bahà'"),
            ("abakoumova", "abakoumov"),
        ]);
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("é"), "è");
        assert_eq!(stem("qu"), "qu");
        assert_eq!(stem("l'"), "l'");
        assert_eq!(stem("12345"), "12345");
        assert_eq!(stem("...!?"), "...!?");
        for w in ["ae", "aeo", "ab", "abch", "divano", "gli", "ci", "日本語", "😀", "a'b'c"] {
            let _ = stem(w);
        }
    }
}
