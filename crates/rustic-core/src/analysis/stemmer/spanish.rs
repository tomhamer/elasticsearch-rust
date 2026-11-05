//! Snowball Spanish stemmer.
//!
//! A faithful port of `algorithms/spanish.sbl` at Snowball commit
//! `34f3612e5e8c` (the revision in every released Lucene 10.x / ES); the
//! later unaccented `-acion`/`-ucion` additions are excluded. The input is assumed to be lowercased; the
//! stemmer operates on a `Vec<char>` and never panics.
//!
//! Steps:
//!
//! 1. **Regions** — `RV`: if the second letter is a consonant, the region
//!    after the next vowel; if the first two letters are vowels, the region
//!    after the next consonant; otherwise (consonant–vowel) the region
//!    after the third letter. `R1`/`R2` are the usual Snowball regions.
//! 2. **Step 0** attached pronoun removal (`me se la lo ... selos nos`)
//!    after a gerund/infinitive ending in `RV`; the accent of `iéndo ándo
//!    ár ér ír` is dropped.
//! 3. **Step 1** standard suffix removal; if nothing was removed,
//!    **Step 2a** verb suffixes beginning with `y` (after `u`), then
//!    **Step 2b** other verb suffixes, both searched within `RV`.
//! 4. **Step 3** residual suffix removal (`os a o á í ó e é`, with the
//!    `gu`→`g` rule).
//! 5. **Postlude** — acute accents are removed (`ü` and `ñ` are kept).

/// Spanish vowels (as defined by the algorithm).
const VOWELS: &[char] = &['a', 'e', 'i', 'o', 'u', 'á', 'é', 'í', 'ó', 'ú', 'ü'];

fn is_v(c: char) -> bool {
    VOWELS.contains(&c)
}

/// Stem a single lowercased Spanish word with the Snowball algorithm.
///
/// Non-letters pass through unchanged; the function never panics, even
/// for empty input or arbitrary Unicode.
pub fn stem(word: &str) -> String {
    let w: Vec<char> = word.chars().collect();
    let (pv, p1, p2) = mark_regions(&w);
    let mut st = St { w, pv, p1, p2 };
    st.attached_pronoun();
    let _ = st.standard_suffix() || st.y_verb_suffix() || st.verb_suffix();
    st.residual_suffix();
    postlude(&st.w)
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
///
/// `RV`: second letter a consonant -> after the next vowel; first two
/// letters vowels -> after the next consonant; consonant-vowel -> after
/// the third letter; otherwise the end of the word.
fn mark_regions(w: &[char]) -> (usize, usize, usize) {
    let n = w.len();
    let mut pv = n;
    if n >= 2 {
        if !is_v(w[1]) {
            if (2..n).any(|i| is_v(w[i])) {
                pv = gopast_v(w, 2);
            }
        } else if is_v(w[0]) {
            if (2..n).any(|i| !is_v(w[i])) {
                pv = gopast_non_v(w, 2);
            }
        } else if n >= 3 {
            pv = 3;
        }
    }
    let p1 = region_after(w, 0);
    let p2 = if p1 < n { region_after(w, p1) } else { n };
    (pv, p1, p2)
}

/// Remove acute accents.
fn postlude(w: &[char]) -> String {
    w.iter()
        .map(|&c| match c {
            'á' => 'a',
            'é' => 'e',
            'í' => 'i',
            'ó' => 'o',
            'ú' => 'u',
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

    /// Step 0: attached pronoun.
    fn attached_pronoun(&mut self) {
        const PRONOUNS: &[&str] = &[
            "me", "se", "sela", "selo", "selas", "selos", "la", "le", "lo", "las", "les", "los",
            "nos",
        ];
        const ENDINGS: &[&str] = &[
            "iéndo", "ándo", "ár", "ér", "ír", "ando", "iendo", "ar", "er", "ir", "yendo",
        ];
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
            "iéndo" => self.replace(e, "iendo"),
            "ándo" => self.replace(e, "ando"),
            "ár" => self.replace(e, "ar"),
            "ér" => self.replace(e, "er"),
            "ír" => self.replace(e, "ir"),
            "yendo" => {
                if e >= 1 && self.w[e - 1] == 'u' {
                    self.truncate(p);
                }
            }
            _ => self.truncate(p),
        }
    }

    /// Step 1: standard suffix removal. Returns `true` if a suffix was
    /// handled.
    fn standard_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "anza", "anzas", "ico", "ica", "icos", "icas", "ismo", "ismos", "able", "ables",
            "ible", "ibles", "ista", "istas", "oso", "osa", "osos", "osas", "amiento", "amientos",
            "imiento", "imientos", "adora", "ador", "ación", "adoras", "adores", "aciones", "ante",
            "antes", "ancia", "ancias", "logía", "logías", "ución", "uciones",
            "encia", "encias", "amente", "mente", "idad", "idades", "iva", "ivo", "ivas", "ivos",
        ];
        let Some((i, suf)) = longest(&self.w, self.len(), 0, LIST) else {
            return false;
        };
        match suf {
            "anza" | "anzas" | "ico" | "ica" | "icos" | "icas" | "ismo" | "ismos" | "able"
            | "ables" | "ible" | "ibles" | "ista" | "istas" | "oso" | "osa" | "osos" | "osas"
            | "amiento" | "amientos" | "imiento" | "imientos" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
            }
            "adora" | "ador" | "ación" | "adoras" | "adores" | "aciones" | "ante" | "antes"
            | "ancia" | "ancias" => {
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
            "logía" | "logías" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "log");
            }
            "ución" | "uciones" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "u");
            }
            "encia" | "encias" => {
                if !self.r2(i) {
                    return false;
                }
                self.replace(i, "ente");
            }
            "amente" => {
                if !self.r1(i) {
                    return false;
                }
                self.truncate(i);
                if let Some((j, pre)) = longest(&self.w, self.len(), 0, &["iv", "os", "ic", "ad"]) {
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
            "mente" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some((j, _)) = longest(&self.w, self.len(), 0, &["ante", "able", "ible"]) {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                }
            }
            "idad" | "idades" => {
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
            "iva" | "ivo" | "ivas" | "ivos" => {
                if !self.r2(i) {
                    return false;
                }
                self.truncate(i);
                if let Some(j) = self.ends("at") {
                    if self.r2(j) {
                        self.truncate(j);
                    }
                }
            }
            _ => return false,
        }
        true
    }

    /// Step 2a: verb suffixes beginning with `y`, searched within `RV` and
    /// removed only after `u`.
    fn y_verb_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "ya", "ye", "yan", "yen", "yeron", "yendo", "yo", "yó", "yas", "yes", "yais", "yamos",
        ];
        let Some((i, _)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return false;
        };
        if i >= 1 && self.w[i - 1] == 'u' {
            self.truncate(i);
            true
        } else {
            false
        }
    }

    /// Step 2b: other verb suffixes, searched within `RV`.
    fn verb_suffix(&mut self) -> bool {
        const LIST: &[&str] = &[
            "en", "es", "éis", "emos", "arían", "arías", "arán", "arás", "aríais", "aría", "aréis",
            "aríamos", "aremos", "ará", "aré", "erían", "erías", "erán", "erás", "eríais", "ería",
            "eréis", "eríamos", "eremos", "erá", "eré", "irían", "irías", "irán", "irás", "iríais",
            "iría", "iréis", "iríamos", "iremos", "irá", "iré", "aba", "ada", "ida", "ía", "ara",
            "iera", "ad", "ed", "id", "ase", "iese", "aste", "iste", "an", "aban", "ían", "aran",
            "ieran", "asen", "iesen", "aron", "ieron", "ado", "ido", "ando", "iendo", "ió", "ar",
            "er", "ir", "as", "abas", "adas", "idas", "ías", "aras", "ieras", "ases", "ieses", "ís",
            "áis", "abais", "íais", "arais", "ierais", "aseis", "ieseis", "asteis", "isteis",
            "ados", "idos", "amos", "ábamos", "íamos", "imos", "áramos", "iéramos", "iésemos",
            "ásemos",
        ];
        let Some((i, suf)) = longest(&self.w, self.len(), self.pv, LIST) else {
            return false;
        };
        let mut cut = i;
        if matches!(suf, "en" | "es" | "éis" | "emos") && self.ends_at(i, "gu").is_some() {
            cut = i - 1;
        }
        self.truncate(cut);
        true
    }

    /// Step 3: residual suffix removal.
    fn residual_suffix(&mut self) {
        const LIST: &[&str] = &["os", "a", "o", "á", "í", "ó", "e", "é"];
        let Some((i, suf)) = longest(&self.w, self.len(), 0, LIST) else {
            return;
        };
        if !self.rv(i) {
            return;
        }
        self.truncate(i);
        if matches!(suf, "e" | "é") {
            if let Some(j) = self.ends("gu") {
                if self.rv(j + 1) {
                    self.truncate(j + 1);
                }
            }
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

    /// Vectors taken from the official Snowball Spanish sample
    /// vocabulary (voc.txt / output.txt).
    #[test]
    fn official_sample_vectors() {
        check(&[
            ("che", "che"),
            ("checa", "chec"),
            ("checar", "chec"),
            ("checo", "chec"),
            ("checoslovaquia", "checoslovaqui"),
            ("chedraoui", "chedraoui"),
            ("chefs", "chefs"),
            ("cheliabinsk", "cheliabinsk"),
            ("chelo", "chel"),
            ("chemical", "chemical"),
            ("chemicalweek", "chemicalweek"),
            ("chemise", "chemis"),
            ("chepo", "chep"),
            ("cheque", "chequ"),
            ("chequeo", "cheque"),
            ("cheques", "chequ"),
            ("cheraw", "cheraw"),
            ("chesca", "chesc"),
            ("chester", "chest"),
            ("chetumal", "chetumal"),
            ("chetumaleños", "chetumaleñ"),
            ("chevrolet", "chevrolet"),
            ("cheyene", "cheyen"),
            ("cheyenne", "cheyenn"),
            ("chi", "chi"),
            ("chiapaneca", "chiapanec"),
            ("chiapas", "chiap"),
            ("chiba", "chib"),
            ("chic", "chic"),
            ("chica", "chic"),
            ("chicago", "chicag"),
            ("chicana", "chican"),
            ("chicano", "chican"),
            ("chicas", "chic"),
            ("chicharrones", "chicharron"),
            ("chichen", "chich"),
            ("chichimecas", "chichimec"),
            ("chicles", "chicl"),
            ("chico", "chic"),
            ("chicos", "chic"),
            ("torá", "tor"),
            ("tos", "tos"),
            ("toscano", "toscan"),
            ("tosferina", "tosferin"),
            ("tostado", "tost"),
            ("tota", "tot"),
            ("total", "total"),
            ("totales", "total"),
            ("totalidad", "total"),
            ("totalizó", "totaliz"),
            ("totalmente", "total"),
            ("totopos", "totop"),
            ("tottenham", "tottenham"),
            ("touché", "touch"),
            ("tour", "tour"),
            ("tovar", "tov"),
            ("toyota", "toyot"),
            ("toño", "toñ"),
            ("tpc", "tpc"),
            ("tqm", "tqm"),
            ("trabado", "trab"),
            ("trabaja", "trabaj"),
            ("trabajaba", "trabaj"),
            ("trabajaban", "trabaj"),
            ("trabajada", "trabaj"),
            ("trabajado", "trabaj"),
            ("trabajador", "trabaj"),
            ("trabajadora", "trabaj"),
            ("trabajadoras", "trabaj"),
            ("trabajadores", "trabaj"),
            ("trabajamos", "trabaj"),
            ("trabajan", "trabaj"),
            ("trabajando", "trabaj"),
            ("trabajar", "trabaj"),
            ("trabajara", "trabaj"),
            ("trabajaron", "trabaj"),
            ("trabajará", "trabaj"),
            ("trabajarán", "trabaj"),
            ("trabajemos", "trabaj"),
            ("trabajen", "trabaj"),
        ]);
    }

    /// Official vectors exercising accents and the postlude.
    #[test]
    fn official_accented_vectors() {
        check(&[("aarón", "aaron"), ("abaco", "abac"), ("abajo", "abaj"), ("abandera", "abander")]);
    }

    #[test]
    fn robustness() {
        assert_eq!(stem(""), "");
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("é"), "e");
        assert_eq!(stem("ñ"), "ñ");
        assert_eq!(stem("ü"), "ü");
        assert_eq!(stem("12345"), "12345");
        assert_eq!(stem("...!?"), "...!?");
        for w in ["ae", "aeo", "ab", "abc", "me", "yendo", "guía", "日本語", "😀", "a'b'c"] {
            let _ = stem(w);
        }
    }
}
