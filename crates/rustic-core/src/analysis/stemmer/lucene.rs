//! Lucene's hand-written language stemmers (Lucene 10.3.2), the stems of
//! the ES language analyzers that are not Snowball: the `char[] s, int
//! len` ones translated by `tools/lucene_stemmer2rs.py` (`lc_*.rs`) run
//! on the helpers here; Latvian, Brazilian and the RSLP Galician stemmer
//! are ported here directly.

use std::sync::OnceLock;

/// Lucene `StemmerUtil.endsWith`.
pub fn ends_with(s: &[char], len: i32, suffix: &str) -> bool {
    let suf: Vec<char> = suffix.chars().collect();
    let n = suf.len() as i32;
    if n > len {
        return false;
    }
    let start = (len - n) as usize;
    s[start..len as usize] == suf[..]
}

/// Lucene `StemmerUtil.startsWith`.
pub fn starts_with(s: &[char], len: i32, prefix: &str) -> bool {
    let p: Vec<char> = prefix.chars().collect();
    let n = p.len() as i32;
    if n > len {
        return false;
    }
    s[..n as usize] == p[..]
}

/// Lucene `StemmerUtil.deleteN`: delete `n` chars at `pos`; the new length.
pub fn delete_n(s: &mut Vec<char>, pos: i32, len: i32, n: i32) -> i32 {
    let (pos, len, n) = (pos as usize, len as usize, n as usize);
    if pos + n < len {
        s.copy_within(pos + n..len, pos);
    }
    (len - n) as i32
}

/// `CharArraySet.contains(s, 0, len)` (case-sensitive).
pub fn contains(set: &[&str], s: &[char], len: i32) -> bool {
    let w: String = s[..len as usize].iter().collect();
    set.contains(&w.as_str())
}

/// Run a `stem(&mut chars, len) -> len` stemmer over a word.
fn run(word: &str, f: impl FnOnce(&mut Vec<char>, i32) -> i32) -> String {
    let mut s: Vec<char> = word.chars().collect();
    let len = s.len() as i32;
    // Oversized by a few chars: some rules write past the old length
    // (Lucene's term buffers have spare capacity).
    s.resize(s.len() + 4, '\0');
    let n = f(&mut s, len).max(0) as usize;
    s[..n.min(s.len())].iter().collect()
}

pub fn czech(word: &str) -> String {
    run(word, |s, len| super::lc_czech::CzechStemmer::default().stem(s, len))
}

pub fn bulgarian(word: &str) -> String {
    run(word, |s, len| super::lc_bulgarian::BulgarianStemmer::default().stem(s, len))
}

pub fn sorani(word: &str) -> String {
    run(word, |s, len| super::lc_sorani::SoraniStemmer::default().stem(s, len))
}

pub fn hindi(word: &str) -> String {
    run(word, |s, len| super::lc_hindi::HindiStemmer::default().stem(s, len))
}

pub fn bengali(word: &str) -> String {
    run(word, |s, len| super::lc_bengali::BengaliStemmer::default().stem(s, len))
}

/// Lucene `IndonesianStemFilter` (derivational stemming on, its default).
pub fn indonesian(word: &str) -> String {
    run(word, |s, len| super::lc_indonesian::IndonesianStemmer::default().stem(s, len, true))
}

/// Lucene `GreekStemmer` (after `GreekLowerCaseFilter`).
pub fn greek(word: &str) -> String {
    run(word, |s, len| super::lc_greek::GreekStemmer::default().stem(s, len))
}

// ---------------------------------------------------------------- Latvian

/// Lucene `LatvianStemmer.affixes`: (suffix, vowel count, palatalizes).
const LV_AFFIXES: &[(&str, i32, bool)] = &[
    ("ajiem", 3, false), ("ajai", 3, false), ("ajam", 2, false), ("ajām", 2, false), ("ajos", 2, false), ("ajās", 2, false),
    ("iem", 2, true), ("ajā", 2, false), ("ais", 2, false), ("ai", 2, false), ("ei", 2, false), ("ām", 1, false),
    ("am", 1, false), ("ēm", 1, false), ("īm", 1, false), ("im", 1, false), ("um", 1, false), ("us", 1, true),
    ("as", 1, false), ("ās", 1, false), ("es", 1, false), ("os", 1, true), ("ij", 1, false), ("īs", 1, false),
    ("ēs", 1, false), ("is", 1, false), ("ie", 1, false), ("u", 1, true), ("a", 1, true), ("i", 1, true),
    ("e", 1, false), ("ā", 1, false), ("ē", 1, false), ("ī", 1, false), ("ū", 1, false), ("o", 1, false),
    ("s", 0, false), ("š", 0, false),
];

fn lv_unpalatalize(s: &mut [char], mut len: i32) -> i32 {
    let l = len as usize;
    // The removed char: -u is 2, 5 or 6 gen. pl.
    if s[l] == 'u' {
        if ends_with(s, len, "kš") {
            len += 1;
            let l = len as usize;
            s[l - 2] = 's';
            s[l - 1] = 't';
            return len;
        }
        if ends_with(s, len, "ņņ") {
            s[l - 2] = 'n';
            s[l - 1] = 'n';
            return len;
        }
    }
    if ends_with(s, len, "pj") || ends_with(s, len, "bj") || ends_with(s, len, "mj") || ends_with(s, len, "vj") {
        return len - 1;
    }
    for (from, to) in [("šņ", ['s', 'n']), ("žņ", ['z', 'n']), ("šļ", ['s', 'l']), ("žļ", ['z', 'l']), ("ļņ", ['l', 'n']), ("ļļ", ['l', 'l'])] {
        if ends_with(s, len, from) {
            s[l - 2] = to[0];
            s[l - 1] = to[1];
            return len;
        }
    }
    match s[l - 1] {
        'č' => s[l - 1] = 'c',
        'ļ' => s[l - 1] = 'l',
        'ņ' => s[l - 1] = 'n',
        _ => {}
    }
    len
}

/// Lucene `LatvianStemmer`.
pub fn latvian(word: &str) -> String {
    run(word, |s, len| {
        let vowels = s[..len as usize].iter().filter(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'ā' | 'ī' | 'ē' | 'ū')).count() as i32;
        for (affix, vc, pal) in LV_AFFIXES {
            let n = affix.chars().count() as i32;
            if vowels > *vc && len >= n + 3 && ends_with(s, len, affix) {
                let len = len - n;
                return if *pal { lv_unpalatalize(s, len) } else { len };
            }
        }
        len
    })
}

// ---------------------------------------------------------------- Brazilian

#[derive(Clone, Copy)]
enum Region {
    R1,
    R2,
    RV,
}

#[derive(Clone, Copy)]
enum Act {
    Remove,
    Replace(&'static str),
    /// Lucene's `logias` rule computes the replacement and drops it.
    Nothing,
}

/// `step1`: (minimum CT length, suffix, region, preceding text, action).
const STEP1: &[(usize, &str, Region, &str, Act)] = &[
    (0, "uciones", Region::R2, "", Act::Replace("u")),
    (6, "imentos", Region::R2, "", Act::Remove),
    (6, "amentos", Region::R2, "", Act::Remove),
    (6, "adores", Region::R2, "", Act::Remove),
    (6, "adoras", Region::R2, "", Act::Remove),
    (6, "logias", Region::R2, "", Act::Nothing),
    (6, "encias", Region::R2, "", Act::Replace("ente")),
    (6, "amente", Region::R1, "", Act::Remove),
    (6, "idades", Region::R2, "", Act::Remove),
    (5, "acoes", Region::R2, "", Act::Remove),
    (5, "imento", Region::R2, "", Act::Remove),
    (5, "amento", Region::R2, "", Act::Remove),
    (5, "adora", Region::R2, "", Act::Remove),
    (5, "ismos", Region::R2, "", Act::Remove),
    (5, "istas", Region::R2, "", Act::Remove),
    (5, "logia", Region::R2, "", Act::Replace("log")),
    (5, "ucion", Region::R2, "", Act::Replace("u")),
    (5, "encia", Region::R2, "", Act::Replace("ente")),
    (5, "mente", Region::R2, "", Act::Remove),
    (5, "idade", Region::R2, "", Act::Remove),
    (4, "acao", Region::R2, "", Act::Remove),
    (4, "ezas", Region::R2, "", Act::Remove),
    (4, "icos", Region::R2, "", Act::Remove),
    (4, "icas", Region::R2, "", Act::Remove),
    (4, "ismo", Region::R2, "", Act::Remove),
    (4, "avel", Region::R2, "", Act::Remove),
    (4, "ivel", Region::R2, "", Act::Remove),
    (4, "ista", Region::R2, "", Act::Remove),
    (4, "osos", Region::R2, "", Act::Remove),
    (4, "osas", Region::R2, "", Act::Remove),
    (4, "ador", Region::R2, "", Act::Remove),
    (4, "ivas", Region::R2, "", Act::Remove),
    (4, "ivos", Region::R2, "", Act::Remove),
    (4, "iras", Region::RV, "e", Act::Replace("ir")),
    (3, "eza", Region::R2, "", Act::Remove),
    (3, "ico", Region::R2, "", Act::Remove),
    (3, "ica", Region::R2, "", Act::Remove),
    (3, "oso", Region::R2, "", Act::Remove),
    (3, "osa", Region::R2, "", Act::Remove),
    (3, "iva", Region::R2, "", Act::Remove),
    (3, "ivo", Region::R2, "", Act::Remove),
    (3, "ira", Region::RV, "e", Act::Replace("ir")),
];
/// `step2`: (minimum RV length, RV suffix, suffix removed from CT).
const STEP2: &[(usize, &str, &str)] = &[
    (7, "issemos", "issemos"),
    (7, "essemos", "essemos"),
    (7, "assemos", "assemos"),
    (7, "ariamos", "ariamos"),
    (7, "eriamos", "eriamos"),
    (7, "iriamos", "iriamos"),
    (6, "iremos", "iremos"),
    (6, "eremos", "eremos"),
    (6, "aremos", "aremos"),
    (6, "avamos", "avamos"),
    (6, "iramos", "iramos"),
    (6, "eramos", "eramos"),
    (6, "aramos", "aramos"),
    (6, "asseis", "asseis"),
    (6, "esseis", "esseis"),
    (6, "isseis", "isseis"),
    (6, "arieis", "arieis"),
    (6, "erieis", "erieis"),
    (6, "irieis", "irieis"),
    (5, "irmos", "irmos"),
    (5, "iamos", "iamos"),
    (5, "armos", "armos"),
    (5, "ermos", "ermos"),
    (5, "areis", "areis"),
    (5, "ereis", "ereis"),
    (5, "ireis", "ireis"),
    (5, "asses", "asses"),
    (5, "esses", "esses"),
    (5, "isses", "isses"),
    (5, "astes", "astes"),
    (5, "assem", "assem"),
    (5, "essem", "essem"),
    (5, "issem", "issem"),
    (5, "ardes", "ardes"),
    (5, "erdes", "erdes"),
    (5, "irdes", "irdes"),
    (5, "ariam", "ariam"),
    (5, "eriam", "eriam"),
    (5, "iriam", "iriam"),
    (5, "arias", "arias"),
    (5, "erias", "erias"),
    (5, "irias", "irias"),
    (5, "estes", "estes"),
    (5, "istes", "istes"),
    (5, "areis", "areis"),
    (5, "aveis", "aveis"),
    (4, "aria", "aria"),
    (4, "eria", "eria"),
    (4, "iria", "iria"),
    (4, "asse", "asse"),
    (4, "esse", "esse"),
    (4, "isse", "isse"),
    (4, "aste", "aste"),
    (4, "este", "este"),
    (4, "iste", "iste"),
    (4, "arei", "arei"),
    (4, "erei", "erei"),
    (4, "irei", "irei"),
    (4, "aram", "aram"),
    (4, "eram", "eram"),
    (4, "iram", "iram"),
    (4, "avam", "avam"),
    (4, "arem", "arem"),
    (4, "erem", "erem"),
    (4, "irem", "irem"),
    (4, "ando", "ando"),
    (4, "endo", "endo"),
    (4, "indo", "indo"),
    (4, "arao", "arao"),
    (4, "erao", "erao"),
    (4, "irao", "irao"),
    (4, "adas", "adas"),
    (4, "idas", "idas"),
    (4, "aras", "aras"),
    (4, "eras", "eras"),
    (4, "iras", "iras"),
    (4, "avas", "avas"),
    (4, "ares", "ares"),
    (4, "eres", "eres"),
    (4, "ires", "ires"),
    (4, "ados", "ados"),
    (4, "idos", "idos"),
    (4, "amos", "amos"),
    (4, "emos", "emos"),
    (4, "imos", "imos"),
    (4, "iras", "iras"),
    (4, "ieis", "ieis"),
    (3, "ada", "ada"),
    (3, "ida", "ida"),
    (3, "ara", "ara"),
    (3, "era", "era"),
    (3, "ira", "ava"),
    (3, "iam", "iam"),
    (3, "ado", "ado"),
    (3, "ido", "ido"),
    (3, "ias", "ias"),
    (3, "ais", "ais"),
    (3, "eis", "eis"),
    (3, "ira", "ira"),
    (3, "ear", "ear"),
    (2, "ia", "ia"),
    (2, "ei", "ei"),
    (2, "am", "am"),
    (2, "em", "em"),
    (2, "ar", "ar"),
    (2, "er", "er"),
    (2, "ir", "ir"),
    (2, "as", "as"),
    (2, "es", "es"),
    (2, "is", "is"),
    (2, "eu", "eu"),
    (2, "iu", "iu"),
    (2, "iu", "iu"),
    (2, "ou", "ou"),
];

fn br_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn br_r1(v: Option<&[char]>) -> Option<Vec<char>> {
    let v = v?;
    let i = v.len() as i32 - 1;
    let mut j = 0i32;
    while j < i && !br_vowel(v[j as usize]) {
        j += 1;
    }
    if j >= i {
        return None;
    }
    while j < i && br_vowel(v[j as usize]) {
        j += 1;
    }
    if j >= i {
        return None;
    }
    Some(v[(j + 1) as usize..].to_vec())
}

fn br_rv(v: &[char]) -> Option<Vec<char>> {
    let i = v.len() as i32 - 1;
    if i > 0 && !br_vowel(v[1]) {
        let mut j = 2;
        while j < i && !br_vowel(v[j as usize]) {
            j += 1;
        }
        if j < i {
            return Some(v[(j + 1) as usize..].to_vec());
        }
    }
    if i > 1 && br_vowel(v[0]) && br_vowel(v[1]) {
        let mut j = 2;
        while j < i && br_vowel(v[j as usize]) {
            j += 1;
        }
        if j < i {
            return Some(v[(j + 1) as usize..].to_vec());
        }
    }
    if i > 2 {
        return Some(v[3..].to_vec());
    }
    None
}

fn br_suffix(v: Option<&[char]>, suf: &str) -> bool {
    let Some(v) = v else { return false };
    let s: Vec<char> = suf.chars().collect();
    s.len() <= v.len() && v[v.len() - s.len()..] == s[..]
}

fn br_remove(v: &mut Vec<char>, suf: &str) {
    if br_suffix(Some(&v[..]), suf) {
        let n = suf.chars().count();
        v.truncate(v.len() - n);
    }
}

fn br_preceded(v: Option<&[char]>, suf: &str, pre: &str) -> bool {
    let Some(v) = v else { return false };
    if !br_suffix(Some(&v[..]), suf) {
        return false;
    }
    let n = suf.chars().count();
    br_suffix(Some(&v[..v.len() - n]), pre)
}

/// Lucene `BrazilianStemmer.stem`; `None` for a term it does not index
/// (the filter then keeps the token as it is).
pub fn brazilian_stem(term: &str) -> Option<String> {
    // createCT / changeTerm: lowercase, accents folded, one leading and
    // one trailing punctuation char removed.
    let mut ct: Vec<char> = Vec::new();
    for c in term.to_lowercase().chars() {
        ct.push(match c {
            'á' | 'â' | 'ã' => 'a',
            'é' | 'ê' => 'e',
            'í' => 'i',
            'ó' | 'ô' | 'õ' => 'o',
            'ú' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            c => c,
        });
    }
    if ct.len() >= 2 && matches!(ct[0], '"' | '\'' | '-' | ',' | ';' | '.' | '?' | '!') {
        ct.remove(0);
    }
    if ct.len() >= 2 && matches!(ct[ct.len() - 1], '-' | ',' | ';' | '.' | '?' | '!' | '\'' | '"') {
        ct.pop();
    }
    if !(ct.len() < 30 && ct.len() > 2) {
        return None;
    }
    if !ct.iter().all(|c| c.is_alphabetic()) {
        return Some(ct.iter().collect());
    }
    let r1 = br_r1(Some(&ct[..]));
    let r2 = br_r1(r1.as_deref());
    let rv = br_rv(&ct);
    let region = |r: Region| match r {
        Region::R1 => r1.as_deref(),
        Region::R2 => r2.as_deref(),
        Region::RV => rv.as_deref(),
    };
    // step1
    let mut altered = false;
    for (min, suf, reg, pre, act) in STEP1 {
        if ct.len() < *min {
            continue;
        }
        if br_suffix(Some(&ct[..]), suf) && br_suffix(region(*reg), suf) && (pre.is_empty() || br_preceded(Some(&ct[..]), suf, pre)) {
            match act {
                Act::Remove => br_remove(&mut ct, suf),
                Act::Replace(to) => {
                    br_remove(&mut ct, suf);
                    ct.extend(to.chars());
                }
                Act::Nothing => {}
            }
            altered = true;
            break;
        }
    }
    // step2
    if !altered {
        if let Some(rv) = rv.as_deref() {
            for (min, suf, del) in STEP2 {
                if rv.len() >= *min && br_suffix(Some(rv), suf) {
                    br_remove(&mut ct, del);
                    altered = true;
                    break;
                }
            }
        }
    }
    let rv = rv.as_deref();
    if altered {
        // step3
        if rv.is_some() && br_suffix(rv, "i") && br_preceded(rv, "i", "c") {
            br_remove(&mut ct, "i");
        }
    } else if rv.is_some() {
        // step4
        for suf in ["os", "a", "i", "o"] {
            if br_suffix(rv, suf) {
                br_remove(&mut ct, suf);
                break;
            }
        }
    }
    // step5
    if rv.is_some() && br_suffix(rv, "e") {
        if br_preceded(rv, "e", "gu") {
            br_remove(&mut ct, "e");
            br_remove(&mut ct, "u");
        } else if br_preceded(rv, "e", "ci") {
            br_remove(&mut ct, "e");
            br_remove(&mut ct, "i");
        } else {
            br_remove(&mut ct, "e");
        }
    }
    Some(ct.iter().collect())
}

/// Lucene `BrazilianStemFilter`: the stem, or the token unchanged when the
/// stemmer does not index the term.
pub fn brazilian(word: &str) -> String {
    brazilian_stem(word).unwrap_or_else(|| word.to_string())
}

// ---------------------------------------------------------------- RSLP

enum Exceptions {
    None,
    /// The whole word (`RuleWithSetExceptions`).
    Set(Vec<String>),
    /// Word endings (`RuleWithSuffixExceptions`).
    Suffix(Vec<String>),
}

struct Rule {
    suffix: String,
    min: i32,
    replacement: String,
    exceptions: Exceptions,
}

impl Rule {
    fn matches(&self, s: &[char], len: i32) -> bool {
        let n = self.suffix.chars().count() as i32;
        if !(len - n >= self.min && ends_with(s, len, &self.suffix)) {
            return false;
        }
        match &self.exceptions {
            Exceptions::None => true,
            Exceptions::Set(set) => {
                let w: String = s[..len as usize].iter().collect();
                !set.contains(&w)
            }
            Exceptions::Suffix(sufs) => !sufs.iter().any(|e| ends_with(s, len, e)),
        }
    }

    fn replace(&self, s: &mut Vec<char>, len: i32) -> i32 {
        let n = self.suffix.chars().count() as i32;
        let rep: Vec<char> = self.replacement.chars().collect();
        let at = (len - n) as usize;
        if s.len() < at + rep.len() {
            s.resize(at + rep.len() + 1, '\0');
        }
        s[at..at + rep.len()].copy_from_slice(&rep);
        len - n + rep.len() as i32
    }
}

struct Step {
    rules: Vec<Rule>,
    min: i32,
    suffixes: Vec<String>,
}

impl Step {
    fn apply(&self, s: &mut Vec<char>, len: i32) -> i32 {
        if len < self.min {
            return len;
        }
        if !self.suffixes.is_empty() && !self.suffixes.iter().any(|x| ends_with(s, len, x)) {
            return len;
        }
        for r in &self.rules {
            if r.matches(s, len) {
                return r.replace(s, len);
            }
        }
        len
    }
}


#[derive(Debug)]
enum RslpItem {
    Str(String),
    Int(i32),
    List(Vec<String>),
}

/// The fields of one RSLP line: `{"suffix", min, "replacement", {"exc", ...}}`
/// or a step header `{"Name", min, type, {"suffix", ...},`.
fn rslp_items(line: &str) -> Vec<RslpItem> {
    let chars: Vec<char> = line.chars().collect();
    let mut i = chars.iter().position(|&c| c == '{').map(|p| p + 1).unwrap_or(0);
    let mut out = Vec::new();
    let string = |i: &mut usize| -> String {
        let mut s = String::new();
        *i += 1;
        while *i < chars.len() && chars[*i] != '"' {
            s.push(chars[*i]);
            *i += 1;
        }
        *i += 1;
        s
    };
    while i < chars.len() {
        match chars[i] {
            '"' => out.push(RslpItem::Str(string(&mut i))),
            '0'..='9' => {
                let mut n = 0i32;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    n = n * 10 + chars[i].to_digit(10).unwrap() as i32;
                    i += 1;
                }
                out.push(RslpItem::Int(n));
            }
            '{' => {
                let mut list = Vec::new();
                i += 1;
                while i < chars.len() && chars[i] != '}' {
                    if chars[i] == '"' {
                        list.push(string(&mut i));
                    } else {
                        i += 1;
                    }
                }
                i += 1;
                out.push(RslpItem::List(list));
            }
            '}' => break,
            _ => i += 1,
        }
    }
    out
}

/// Lucene `RSLPStemmerBase.parse` of a steps file.
fn rslp_parse(src: &str) -> std::collections::HashMap<String, Step> {
    let mut lines = src.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    let mut steps = std::collections::HashMap::new();
    while let Some(h) = lines.next() {
        let items = rslp_items(h);
        let (name, mut min, ty, suffixes) = match &items[..] {
            [RslpItem::Str(n), RslpItem::Int(m), RslpItem::Int(t), RslpItem::List(l)] => (n.clone(), *m, *t, l.clone()),
            other => panic!("illegal RSLP step header {h}: {other:?}"),
        };
        let mut rules = Vec::new();
        for line in lines.by_ref() {
            let rule = match rslp_items(line).as_slice() {
                [RslpItem::Str(suf), RslpItem::Int(m)] => Rule { suffix: suf.clone(), min: *m, replacement: String::new(), exceptions: Exceptions::None },
                [RslpItem::Str(suf), RslpItem::Int(m), RslpItem::Str(rep)] => Rule { suffix: suf.clone(), min: *m, replacement: rep.clone(), exceptions: Exceptions::None },
                [RslpItem::Str(suf), RslpItem::Int(m), RslpItem::Str(rep), RslpItem::List(exc)] => {
                    let exceptions = if ty == 0 { Exceptions::Suffix(exc.clone()) } else { Exceptions::Set(exc.clone()) };
                    Rule { suffix: suf.clone(), min: *m, replacement: rep.clone(), exceptions }
                }
                other => panic!("illegal RSLP rule {line}: {other:?}"),
            };
            rules.push(rule);
            if line.ends_with(';') {
                break;
            }
        }
        if min == 0 {
            min = rules.iter().map(|r| r.min + r.suffix.chars().count() as i32).min().unwrap_or(i32::MAX);
        }
        steps.insert(name, Step { rules, min, suffixes });
    }
    steps
}

fn galician_steps() -> &'static std::collections::HashMap<String, Step> {
    static S: OnceLock<std::collections::HashMap<String, Step>> = OnceLock::new();
    S.get_or_init(|| rslp_parse(include_str!("galician.rslp")))
}

/// Lucene `GalicianStemmer` (RSLP-G).
pub fn galician(word: &str) -> String {
    let st = galician_steps();
    run(word, |s, len| {
        let mut len = st["Plural"].apply(s, len);
        len = st["Unification"].apply(s, len);
        len = st["Adverb"].apply(s, len);
        loop {
            let old = len;
            len = st["Augmentative"].apply(s, len);
            if len == old {
                break;
            }
        }
        let old = len;
        len = st["Noun"].apply(s, len);
        if len == old {
            len = st["Verb"].apply(s, len);
        }
        len = st["Vowel"].apply(s, len);
        for c in s[..len as usize].iter_mut() {
            *c = match *c {
                'á' => 'a',
                'é' | 'ê' => 'e',
                'í' => 'i',
                'ó' => 'o',
                'ú' => 'u',
                c => c,
            };
        }
        len
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/analysis/testdata/lucene").join(name);
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        String::from_utf8(zstd::decode_all(&raw[..]).expect("zstd")).expect("utf-8")
    }

    /// SIDE 005: every pair of Lucene 10.3.2's own tests of these stemmers
    /// (`TestCzechStemmer`, `TestBulgarianStemmer`, `TestLatvianStemmer`,
    /// `TestHindiStemmer`, `TestIndonesianStemmer`, `TestGreekStemmer`,
    /// `TestBrazilianAnalyzer`, `TestSoraniStemFilter`,
    /// `TestBengaliStemmer`), extracted into `lang_pairs.tsv.zst`:
    /// `stem` rows run the stemmer, `analyzer` rows the whole language
    /// analyzer (tokens space-joined).
    #[test]
    fn lucene_test_pairs() {
        let text = load("lang_pairs.tsv.zst");
        let mut n = 0;
        let mut failures = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let (kind, lang, input, want) = (f[0], f[1], f[2], f.get(3).copied().unwrap_or(""));
            n += 1;
            let got = match kind {
                "stem" => {
                    let stem: fn(&str) -> String = match lang {
                        "latvian" => latvian,
                        "indonesian" => indonesian,
                        "hindi" => hindi,
                        "bengali" => bengali,
                        other => panic!("{other}"),
                    };
                    // Lucene's tests run these through a whitespace tokenizer.
                    stem(input.trim())
                }
                _ => {
                    let a = crate::analysis::Analyzer::language(lang).unwrap_or_else(|| panic!("{lang}"));
                    a.analyze(input).into_iter().map(|t| t.text).collect::<Vec<_>>().join(" ")
                }
            };
            if got != want {
                failures.push(format!("{kind} {lang} {input:?}: expected {want:?}, got {got:?}"));
            }
        }
        assert!(n >= 960, "only {n} pairs");
        assert!(failures.is_empty(), "{}/{n} mismatches:\n{}", failures.len(), failures[..failures.len().min(40)].join("\n"));
    }

    /// Lucene `TestGalicianStemFilter.testVocabulary`: the RSLP-G reference
    /// vocabulary (`gltestdata.zip`, all pairs).
    #[test]
    fn galician_vocabulary() {
        let text = load("gl.tsv.zst");
        let mut n = 0;
        let mut failures = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (w, want) = line.split_once('\t').expect("pair");
            n += 1;
            let got = galician(w);
            if got != want {
                failures.push(format!("{w:?}: expected {want:?}, got {got:?}"));
            }
        }
        assert!(n >= 9_000, "only {n} pairs");
        assert!(failures.is_empty(), "{}/{n} mismatches:\n{}", failures.len(), failures[..failures.len().min(40)].join("\n"));
    }
}
