//! Porter stemmers.
//!
//! - [`porter_stem`]: a line-by-line port of Lucene's
//!   `org.apache.lucene.analysis.en.PorterStemmer` (the C-derived 1980
//!   Porter algorithm with Porter's later `-logi`/`-bli` tweaks). This is
//!   what Lucene's `PorterStemFilter` and ES's `porter_stem` filter /
//!   `stemmer` (`english`, `porter`) use.
//! - [`snowball_porter_stem`]: a port of Snowball's `porter.sbl` (what
//!   Lucene's `SnowballFilter("Porter")` / ES `snowball` `language: Porter`
//!   use). The two differ on a few dozen words (e.g. `as`, `apology`,
//!   `humbly`, `action`).

/// Lucene `PorterStemmer.stem`: stem an already-lowercased word.
pub fn porter_stem(word: &str) -> String {
    let mut st = LucenePorter {
        b: word.chars().collect(),
        j: 0,
        k: 0,
        k0: 0,
    };
    let len = st.b.len() as isize;
    st.k = len - 1;
    if st.k > st.k0 + 1 {
        st.step1();
        st.step2();
        st.step3();
        st.step4();
        st.step5();
        st.step6();
    }
    let end = (st.k + 1).max(0) as usize;
    st.b[..end].iter().collect()
}

struct LucenePorter {
    b: Vec<char>,
    j: isize,
    k: isize,
    k0: isize,
}

impl LucenePorter {
    fn at(&self, i: isize) -> char {
        self.b[i as usize]
    }

    fn cons(&self, i: isize) -> bool {
        match self.at(i) {
            'a' | 'e' | 'i' | 'o' | 'u' => false,
            'y' => {
                if i == self.k0 {
                    true
                } else {
                    !self.cons(i - 1)
                }
            }
            _ => true,
        }
    }

    /// Number of consonant sequences between k0 and j.
    fn m(&self) -> usize {
        let mut n = 0;
        let mut i = self.k0;
        loop {
            if i > self.j {
                return n;
            }
            if !self.cons(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            loop {
                if i > self.j {
                    return n;
                }
                if self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    fn vowelinstem(&self) -> bool {
        let mut i = self.k0;
        while i <= self.j {
            if !self.cons(i) {
                return true;
            }
            i += 1;
        }
        false
    }

    fn doublec(&self, j: isize) -> bool {
        if j < self.k0 + 1 {
            return false;
        }
        if self.at(j) != self.at(j - 1) {
            return false;
        }
        self.cons(j)
    }

    fn cvc(&self, i: isize) -> bool {
        if i < self.k0 + 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
            return false;
        }
        !matches!(self.at(i), 'w' | 'x' | 'y')
    }

    fn ends(&mut self, s: &str) -> bool {
        let sc: Vec<char> = s.chars().collect();
        let l = sc.len() as isize;
        let o = self.k - l + 1;
        if o < self.k0 {
            return false;
        }
        for (i, c) in sc.iter().enumerate() {
            if self.at(o + i as isize) != *c {
                return false;
            }
        }
        self.j = self.k - l;
        true
    }

    fn setto(&mut self, s: &str) {
        let o = (self.j + 1) as usize;
        let mut l = 0isize;
        for (i, c) in s.chars().enumerate() {
            if o + i < self.b.len() {
                self.b[o + i] = c;
            } else {
                self.b.push(c);
            }
            l += 1;
        }
        self.k = self.j + l;
    }

    fn r(&mut self, s: &str) {
        if self.m() > 0 {
            self.setto(s);
        }
    }

    fn step1(&mut self) {
        if self.at(self.k) == 's' {
            if self.ends("sses") {
                self.k -= 2;
            } else if self.ends("ies") {
                self.setto("i");
            } else if self.at(self.k - 1) != 's' {
                self.k -= 1;
            }
        }
        if self.ends("eed") {
            if self.m() > 0 {
                self.k -= 1;
            }
        } else if (self.ends("ed") || self.ends("ing")) && self.vowelinstem() {
            self.k = self.j;
            if self.ends("at") {
                self.setto("ate");
            } else if self.ends("bl") {
                self.setto("ble");
            } else if self.ends("iz") {
                self.setto("ize");
            } else if self.doublec(self.k) {
                let ch = self.at(self.k);
                self.k -= 1;
                if ch == 'l' || ch == 's' || ch == 'z' {
                    self.k += 1;
                }
            } else if self.m() == 1 && self.cvc(self.k) {
                self.setto("e");
            }
        }
    }

    fn step2(&mut self) {
        if self.ends("y") && self.vowelinstem() {
            let k = self.k as usize;
            self.b[k] = 'i';
        }
    }

    fn step3(&mut self) {
        if self.k == self.k0 {
            return;
        }
        let rules: &[(&str, &str)] = match self.at(self.k - 1) {
            'a' => &[("ational", "ate"), ("tional", "tion")],
            'c' => &[("enci", "ence"), ("anci", "ance")],
            'e' => &[("izer", "ize")],
            'l' => &[
                ("bli", "ble"),
                ("alli", "al"),
                ("entli", "ent"),
                ("eli", "e"),
                ("ousli", "ous"),
            ],
            'o' => &[("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            's' => &[
                ("alism", "al"),
                ("iveness", "ive"),
                ("fulness", "ful"),
                ("ousness", "ous"),
            ],
            't' => &[("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            'g' => &[("logi", "log")],
            _ => &[],
        };
        for (suf, rep) in rules {
            if self.ends(suf) {
                self.r(rep);
                break;
            }
        }
    }

    fn step4(&mut self) {
        let rules: &[(&str, &str)] = match self.at(self.k) {
            'e' => &[("icate", "ic"), ("ative", ""), ("alize", "al")],
            'i' => &[("iciti", "ic")],
            'l' => &[("ical", "ic"), ("ful", "")],
            's' => &[("ness", "")],
            _ => &[],
        };
        for (suf, rep) in rules {
            if self.ends(suf) {
                self.r(rep);
                break;
            }
        }
    }

    fn step5(&mut self) {
        if self.k == self.k0 {
            return;
        }
        let matched = match self.at(self.k - 1) {
            'a' => self.ends("al"),
            'c' => self.ends("ance") || self.ends("ence"),
            'e' => self.ends("er"),
            'i' => self.ends("ic"),
            'l' => self.ends("able") || self.ends("ible"),
            'n' => {
                self.ends("ant") || self.ends("ement") || self.ends("ment") || self.ends("ent")
            }
            'o' => {
                (self.ends("ion") && self.j >= 0 && matches!(self.at(self.j), 's' | 't'))
                    || self.ends("ou")
            }
            's' => self.ends("ism"),
            't' => self.ends("ate") || self.ends("iti"),
            'u' => self.ends("ous"),
            'v' => self.ends("ive"),
            'z' => self.ends("ize"),
            _ => false,
        };
        if matched && self.m() > 1 {
            self.k = self.j;
        }
    }

    fn step6(&mut self) {
        self.j = self.k;
        if self.at(self.k) == 'e' {
            let a = self.m();
            if a > 1 || (a == 1 && !self.cvc(self.k - 1)) {
                self.k -= 1;
            }
        }
        if self.at(self.k) == 'l' && self.doublec(self.k) && self.m() > 1 {
            self.k -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Snowball `porter.sbl`
// ---------------------------------------------------------------------------

fn sb_v(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn sb_v_wxy(c: char) -> bool {
    sb_v(c) || matches!(c, 'w' | 'x' | 'Y')
}

/// Snowball `shortv` evaluated backwards ending at `end` (exclusive):
/// `non-v_WXY v non-v`.
fn sb_shortv(s: &[char], end: usize) -> bool {
    end >= 3 && !sb_v_wxy(s[end - 1]) && sb_v(s[end - 2]) && !sb_v(s[end - 3])
}

fn sb_ends(s: &[char], suf: &str) -> bool {
    let sc: Vec<char> = suf.chars().collect();
    s.len() >= sc.len() && s[s.len() - sc.len()..] == sc[..]
}

/// Longest suffix of `s` among `list`.
fn sb_among<'a>(s: &[char], list: &[&'a str]) -> Option<&'a str> {
    let mut best: Option<&'a str> = None;
    for suf in list {
        if sb_ends(s, suf) && best.map_or(true, |b| suf.chars().count() > b.chars().count()) {
            best = Some(suf);
        }
    }
    best
}

fn sb_replace(s: &mut Vec<char>, suf: &str, rep: &str) {
    let n = suf.chars().count();
    s.truncate(s.len() - n);
    s.extend(rep.chars());
}

/// Snowball's Porter stemmer (`algorithms/porter.sbl`).
pub fn snowball_porter_stem(word: &str) -> String {
    let mut s: Vec<char> = word.chars().collect();
    // Mark consonant-y: initial y, and y after a vowel.
    let mut y_found = false;
    if !s.is_empty() && s[0] == 'y' {
        s[0] = 'Y';
        y_found = true;
    }
    for i in 1..s.len() {
        if s[i] == 'y' && sb_v(s[i - 1]) {
            s[i] = 'Y';
            y_found = true;
        }
    }
    // Regions.
    let n = s.len();
    let mut p1 = n;
    let mut p2 = n;
    {
        let mut i = 0;
        while i < n && !sb_v(s[i]) {
            i += 1;
        }
        while i < n && sb_v(s[i]) {
            i += 1;
        }
        if i < n {
            p1 = i + 1;
            let mut i = i + 1;
            while i < n && !sb_v(s[i]) {
                i += 1;
            }
            while i < n && sb_v(s[i]) {
                i += 1;
            }
            if i < n {
                p2 = i + 1;
            }
        }
    }
    let r1 = |s: &Vec<char>, suf: &str| s.len() - suf.chars().count() >= p1;
    let r2 = |s: &Vec<char>, suf: &str| s.len() - suf.chars().count() >= p2;

    // Step 1a
    if let Some(suf) = sb_among(&s, &["sses", "ies", "ss", "s"]) {
        match suf {
            "sses" => sb_replace(&mut s, suf, "ss"),
            "ies" => sb_replace(&mut s, suf, "i"),
            "s" => sb_replace(&mut s, suf, ""),
            _ => {}
        }
    }

    // Step 1b
    if let Some(suf) = sb_among(&s, &["eed", "ed", "ing"]) {
        if suf == "eed" {
            if r1(&s, suf) {
                sb_replace(&mut s, suf, "ee");
            }
        } else {
            let stem_len = s.len() - suf.len();
            if s[..stem_len].iter().any(|&c| sb_v(c)) {
                s.truncate(stem_len);
                let doubles = ["bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"];
                if sb_ends(&s, "at") || sb_ends(&s, "bl") || sb_ends(&s, "iz") {
                    s.push('e');
                } else if doubles.iter().any(|d| sb_ends(&s, d)) {
                    s.pop();
                } else if s.len() == p1 && sb_shortv(&s, s.len()) {
                    s.push('e');
                }
            }
        }
    }

    // Step 1c
    if let Some(&last) = s.last() {
        if (last == 'y' || last == 'Y') && s[..s.len() - 1].iter().any(|&c| sb_v(c)) {
            let l = s.len();
            s[l - 1] = 'i';
        }
    }

    // Step 2
    let step2: &[(&str, &str)] = &[
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("abli", "able"),
        ("entli", "ent"),
        ("eli", "e"),
        ("izer", "ize"),
        ("ization", "ize"),
        ("ational", "ate"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alli", "al"),
        ("alism", "al"),
        ("aliti", "al"),
        ("fulness", "ful"),
        ("ousli", "ous"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];
    let keys: Vec<&str> = step2.iter().map(|p| p.0).collect();
    if let Some(suf) = sb_among(&s, &keys) {
        if r1(&s, suf) {
            let rep = step2.iter().find(|p| p.0 == suf).map(|p| p.1).unwrap_or("");
            sb_replace(&mut s, suf, rep);
        }
    }

    // Step 3
    let step3: &[(&str, &str)] = &[
        ("alize", "al"),
        ("icate", "ic"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ative", ""),
        ("ful", ""),
        ("ness", ""),
    ];
    let keys: Vec<&str> = step3.iter().map(|p| p.0).collect();
    if let Some(suf) = sb_among(&s, &keys) {
        if r1(&s, suf) {
            let rep = step3.iter().find(|p| p.0 == suf).map(|p| p.1).unwrap_or("");
            sb_replace(&mut s, suf, rep);
        }
    }

    // Step 4
    let step4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
        "ism", "ate", "iti", "ous", "ive", "ize", "ion",
    ];
    if let Some(suf) = sb_among(&s, &step4) {
        if r2(&s, suf) {
            if suf == "ion" {
                let stem_len = s.len() - 3;
                if stem_len > 0 && matches!(s[stem_len - 1], 's' | 't') {
                    s.truncate(stem_len);
                }
            } else {
                sb_replace(&mut s, suf, "");
            }
        }
    }

    // Step 5a
    if s.last() == Some(&'e') {
        let pos = s.len() - 1;
        if pos >= p2 || (pos >= p1 && !sb_shortv(&s, pos)) {
            s.pop();
        }
    }

    // Step 5b
    if s.last() == Some(&'l') {
        let pos = s.len() - 1;
        if pos >= p2 && pos >= 1 && s[pos - 1] == 'l' {
            s.pop();
        }
    }

    if y_found {
        for c in s.iter_mut() {
            if *c == 'Y' {
                *c = 'y';
            }
        }
    }
    s.into_iter().collect()
}

/// A token filter that applies the (Lucene) Porter stemmer to each token.
#[derive(Clone, Debug, Default)]
pub struct StemFilter;

impl crate::analysis::TokenFilter for StemFilter {
    fn clone_box(&self) -> Box<dyn crate::analysis::TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<crate::analysis::Token>) -> Vec<crate::analysis::Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.text = porter_stem(&t.text);
                t
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::porter_stem;

    fn check(pairs: &[(&str, &str)]) {
        for (input, expected) in pairs {
            let got = porter_stem(input);
            assert_eq!(&got, expected, "porter_stem({input})");
        }
    }

    #[test]
    fn step_1a() {
        check(&[
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
        ]);
    }

    #[test]
    fn step_1b() {
        check(&[
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("tanned", "tan"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            ("happy", "happi"),
            ("sky", "sky"),
        ]);
    }

    #[test]
    fn step_2() {
        check(&[
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("valenci", "valenc"),
            ("hesitanci", "hesit"),
            ("digitizer", "digit"),
            ("conformabli", "conform"),
            ("radicalli", "radic"),
            ("differentli", "differ"),
            ("vileli", "vile"),
            ("analogousli", "analog"),
            ("vietnamization", "vietnam"),
            ("predication", "predic"),
            ("operator", "oper"),
            ("feudalism", "feudal"),
            ("decisiveness", "decis"),
            ("hopefulness", "hope"),
            ("callousness", "callous"),
            ("formaliti", "formal"),
            ("sensitiviti", "sensit"),
            ("sensibiliti", "sensibl"),
        ]);
    }

    #[test]
    fn step_3() {
        check(&[
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electriciti", "electr"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
        ]);
    }

    #[test]
    fn step_4() {
        check(&[
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("homologou", "homolog"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("angulariti", "angular"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
        ]);
    }

    #[test]
    fn step_5() {
        check(&[
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            ("controll", "control"),
            ("roll", "roll"),
            ("generalizations", "gener"),
        ]);
    }

    #[test]
    fn short_words_unchanged() {
        check(&[("a", "a"), ("ab", "ab"), ("go", "go")]);
    }

    #[test]
    fn empty_string() {
        assert_eq!(porter_stem(""), "");
    }
}
