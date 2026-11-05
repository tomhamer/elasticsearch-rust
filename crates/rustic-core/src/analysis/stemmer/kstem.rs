//! KStem (Krovetz) English stemmer: a faithful port of Lucene's
//! `org.apache.lucene.analysis.en.KStemmer` (branch_10x), as used by
//! `KStemFilter` (ES `kstem` / `light_english`).
//!
//! The dictionary (`kstem_data`) is generated verbatim from `KStemmer`'s
//! tables and `KStemData1..8`. The suffix-stripping state machine (the
//! `word` buffer, the `j`/`k` indices and the `matchedEntry` cache) mirrors
//! the Java code step by step, including its quirks, so results match
//! Lucene exactly. Input is expected to be lowercased already; words with
//! anything other than `a`-`z`, or shorter than 3 / longer than 49
//! characters, are returned unchanged (unless the whole word is a
//! dictionary entry with a root).

use std::collections::HashMap;
use std::sync::OnceLock;

use super::kstem_data as data;

const MAX_WORD_LEN: usize = 50;

struct DictEntry {
    exception: bool,
    root: Option<&'static str>,
}

static DEFAULT_ENTRY: DictEntry = DictEntry { exception: false, root: None };

fn dict() -> &'static HashMap<&'static [u8], &'static DictEntry> {
    static DICT: OnceLock<HashMap<&'static [u8], &'static DictEntry>> = OnceLock::new();
    DICT.get_or_init(|| {
        let mut d: HashMap<&'static [u8], &'static DictEntry> = HashMap::with_capacity(32_000);
        let mut put = |w: &'static str, e: &'static DictEntry| {
            let prev = d.insert(w.as_bytes(), e);
            debug_assert!(prev.is_none(), "duplicate kstem entry {w}");
        };
        for w in data::EXCEPTION_WORDS {
            put(w, Box::leak(Box::new(DictEntry { exception: true, root: Some(w) })));
        }
        for (w, r) in data::DIRECT_CONFLATIONS.iter().chain(data::COUNTRY_NATIONALITY) {
            put(w, Box::leak(Box::new(DictEntry { exception: false, root: Some(r) })));
        }
        for arr in data::DATA {
            for w in *arr {
                put(w, &DEFAULT_ENTRY);
            }
        }
        for w in data::SUPPLEMENT_DICT.iter().chain(data::PROPER_NOUNS) {
            put(w, &DEFAULT_ENTRY);
        }
        d
    })
}

/// Mirror of the Java `KStemmer` instance state. `buf`/`len` emulate
/// Lucene's `OpenStringBuilder` (reads via `ch` index the backing array,
/// not just the logical length).
struct KStemmer {
    buf: Vec<u8>,
    len: usize,
    j: i32,
    k: i32,
    matched: Option<&'static DictEntry>,
}

impl KStemmer {
    fn ch(&self, i: i32) -> u8 {
        self.buf[i as usize]
    }
    fn set_ch(&mut self, i: i32, c: u8) {
        self.buf[i as usize] = c;
    }
    fn set_len(&mut self, n: i32) {
        self.len = n as usize;
    }
    fn len(&self) -> i32 {
        self.len as i32
    }
    fn write(&mut self, c: u8) {
        if self.len >= self.buf.len() {
            self.buf.resize(self.len + 1, 0);
        }
        self.buf[self.len] = c;
        self.len += 1;
    }
    fn append(&mut self, s: &str) {
        for &c in s.as_bytes() {
            self.write(c);
        }
    }
    fn penult(&self) -> u8 {
        self.ch(self.k - 1)
    }
    fn is_cons(&self, i: i32) -> bool {
        match self.ch(i) {
            b'a' | b'e' | b'i' | b'o' | b'u' => false,
            b'y' if i != 0 => !self.is_cons(i - 1),
            _ => true,
        }
    }
    fn is_vowel(&self, i: i32) -> bool {
        !self.is_cons(i)
    }
    fn ends_in_s(&mut self, s: &[u8]) -> bool {
        if s.len() as i32 > self.k {
            return false;
        }
        let r = self.len() - s.len() as i32;
        self.j = self.k;
        for (i, &c) in s.iter().enumerate() {
            if c != self.ch(r + i as i32) {
                return false;
            }
        }
        self.j = r - 1;
        true
    }
    fn ends2(&mut self, a: u8, b: u8) -> bool {
        if 2 > self.k {
            return false;
        }
        if self.ch(self.k - 1) == a && self.ch(self.k) == b {
            self.j = self.k - 2;
            return true;
        }
        false
    }
    fn ends3(&mut self, a: u8, b: u8, c: u8) -> bool {
        if 3 > self.k {
            return false;
        }
        if self.ch(self.k - 2) == a && self.ch(self.k - 1) == b && self.ch(self.k) == c {
            self.j = self.k - 3;
            return true;
        }
        false
    }
    fn ends4(&mut self, a: u8, b: u8, c: u8, d: u8) -> bool {
        if 4 > self.k {
            return false;
        }
        let k = self.k;
        if self.ch(k - 3) == a && self.ch(k - 2) == b && self.ch(k - 1) == c && self.ch(k) == d {
            self.j = k - 4;
            return true;
        }
        false
    }
    fn get(&self) -> Option<&'static DictEntry> {
        dict().get(&self.buf[..self.len]).copied()
    }
    fn word_in_dict(&mut self) -> Option<&'static DictEntry> {
        if self.matched.is_some() {
            return self.matched;
        }
        let e = self.get();
        if let Some(en) = e {
            if !en.exception {
                self.matched = e;
            }
        }
        e
    }
    fn lookup(&mut self) -> bool {
        self.matched = self.get();
        self.matched.is_some()
    }
    fn set_suffix(&mut self, s: &str) {
        self.set_len(self.j + 1);
        self.append(s);
        self.k = self.j + s.len() as i32;
    }
    fn double_c(&self, i: i32) -> bool {
        if i < 1 {
            return false;
        }
        if self.ch(i) != self.ch(i - 1) {
            return false;
        }
        self.is_cons(i)
    }
    fn vowel_in_stem(&self) -> bool {
        (0..self.j + 1).any(|i| self.is_vowel(i))
    }

    fn plural(&mut self) {
        if self.ch(self.k) != b's' {
            return;
        }
        if self.ends3(b'i', b'e', b's') {
            self.set_len(self.j + 3);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.k += 1;
            self.write(b's');
            self.set_suffix("y");
            self.lookup();
        } else if self.ends2(b'e', b's') {
            self.set_len(self.j + 2);
            self.k -= 1;
            let j = self.j;
            let try_e = j > 0 && !(self.ch(j) == b's' && self.ch(j - 1) == b's');
            if try_e && self.lookup() {
                return;
            }
            self.set_len(self.j + 1);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k += 1;
            if !try_e {
                self.lookup();
            }
        } else if self.len() > 3 && self.penult() != b's' && !self.ends3(b'o', b'u', b's') {
            self.set_len(self.k);
            self.k -= 1;
            self.lookup();
        }
    }

    fn past_tense(&mut self) {
        if self.len() <= 4 {
            return;
        }
        if self.ends3(b'i', b'e', b'd') {
            self.set_len(self.j + 3);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            self.k += 1;
            self.write(b'd');
            self.set_suffix("y");
            self.lookup();
            return;
        }
        if self.ends2(b'e', b'd') && self.vowel_in_stem() {
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if let Some(e) = self.word_in_dict() {
                if !e.exception {
                    return;
                }
            }
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.set_len(self.k);
                self.k -= 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.k);
                self.write(c);
                self.k += 1;
                self.lookup();
                return;
            }
            if self.ch(0) == b'u' && self.ch(1) == b'n' {
                self.write(b'e');
                self.write(b'd');
                self.k += 2;
                return;
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
        }
    }

    fn aspect(&mut self) {
        if self.len() <= 5 {
            return;
        }
        if self.ends3(b'i', b'n', b'g') && self.vowel_in_stem() {
            self.set_ch(self.j + 1, b'e');
            self.set_len(self.j + 2);
            self.k = self.j + 1;
            if let Some(e) = self.word_in_dict() {
                if !e.exception {
                    return;
                }
            }
            self.set_len(self.k);
            self.k -= 1;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.k -= 1;
                self.set_len(self.k + 1);
                if self.lookup() {
                    return;
                }
                let c = self.ch(self.k);
                self.write(c);
                self.k += 1;
                self.lookup();
                return;
            }
            if self.j > 0 && self.is_cons(self.j) && self.is_cons(self.j - 1) {
                self.k = self.j;
                self.set_len(self.k + 1);
                return;
            }
            self.set_len(self.j + 1);
            self.write(b'e');
            self.k = self.j + 1;
        }
    }

    fn ity_endings(&mut self) {
        let old_k = self.k;
        if self.ends3(b'i', b't', b'y') {
            let j = self.j;
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_ch(j + 1, b'i');
            self.append("ty");
            self.k = old_k;
            if j > 0 && self.ch(j - 1) == b'i' && self.ch(j) == b'l' {
                self.set_len(j - 1);
                self.append("le");
                self.k = j;
                self.lookup();
                return;
            }
            if j > 0 && self.ch(j - 1) == b'i' && self.ch(j) == b'v' {
                self.set_len(j + 1);
                self.write(b'e');
                self.k = j + 1;
                self.lookup();
                return;
            }
            if j > 0 && self.ch(j - 1) == b'a' && self.ch(j) == b'l' {
                self.set_len(j + 1);
                self.k = j;
                self.lookup();
                return;
            }
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.k = j;
        }
    }

    fn nce_endings(&mut self) {
        let old_k = self.k;
        if self.ends3(b'n', b'c', b'e') {
            let j = self.j;
            let wc = self.ch(j);
            if !(wc == b'e' || wc == b'a') {
                return;
            }
            self.set_len(j);
            self.write(b'e');
            self.k = j;
            if self.lookup() {
                return;
            }
            self.set_len(j);
            self.k = j - 1;
            if self.lookup() {
                return;
            }
            self.write(wc);
            self.append("nce");
            self.k = old_k;
        }
    }

    fn ness_endings(&mut self) {
        if self.ends4(b'n', b'e', b's', b's') {
            let j = self.j;
            self.set_len(j + 1);
            self.k = j;
            if self.ch(j) == b'i' {
                self.set_ch(j, b'y');
            }
            self.lookup();
        }
    }

    fn ism_endings(&mut self) {
        if self.ends3(b'i', b's', b'm') {
            self.set_len(self.j + 1);
            self.k = self.j;
            self.lookup();
        }
    }

    fn ment_endings(&mut self) {
        let old_k = self.k;
        if self.ends4(b'm', b'e', b'n', b't') {
            self.set_len(self.j + 1);
            self.k = self.j;
            if self.lookup() {
                return;
            }
            self.append("ment");
            self.k = old_k;
        }
    }

    fn ize_endings(&mut self) {
        let old_k = self.k;
        if self.ends3(b'i', b'z', b'e') {
            let j = self.j;
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.write(b'i');
            if self.double_c(j) {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(j - 1);
                self.write(c);
            }
            self.set_len(j + 1);
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("ize");
            self.k = old_k;
        }
    }

    fn ncy_endings(&mut self) {
        if self.ends3(b'n', b'c', b'y') {
            let j = self.j;
            if !(self.ch(j) == b'e' || self.ch(j) == b'a') {
                return;
            }
            self.set_ch(j + 2, b't');
            self.set_len(j + 3);
            self.k = j + 2;
            if self.lookup() {
                return;
            }
            self.set_ch(j + 2, b'c');
            self.write(b'e');
            self.k = j + 3;
            self.lookup();
        }
    }

    fn ble_endings(&mut self) {
        let old_k = self.k;
        if self.ends3(b'b', b'l', b'e') {
            let j = self.j;
            if !(self.ch(j) == b'a' || self.ch(j) == b'i') {
                return;
            }
            let wc = self.ch(j);
            self.set_len(j);
            self.k = j - 1;
            if self.lookup() {
                return;
            }
            if self.double_c(self.k) {
                self.set_len(self.k);
                self.k -= 1;
                if self.lookup() {
                    return;
                }
                self.k += 1;
                let c = self.ch(self.k - 1);
                self.write(c);
            }
            self.set_len(j);
            self.write(b'e');
            self.k = j;
            if self.lookup() {
                return;
            }
            self.set_len(j);
            self.append("ate");
            self.k = j + 2;
            if self.lookup() {
                return;
            }
            self.set_len(j);
            self.write(wc);
            self.append("ble");
            self.k = old_k;
        }
    }

    fn ic_endings(&mut self) {
        if self.ends2(b'i', b'c') {
            let j = self.j;
            self.set_len(j + 3);
            self.append("al");
            self.k = j + 4;
            if self.lookup() {
                return;
            }
            self.set_ch(j + 1, b'y');
            self.set_len(j + 2);
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_ch(j + 1, b'e');
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.append("ic");
            self.k = j + 2;
        }
    }

    fn ion_endings(&mut self) {
        let old_k = self.k;
        if !self.ends3(b'i', b'o', b'n') {
            return;
        }
        if self.ends_in_s(b"ization") {
            self.set_len(self.j + 3);
            self.write(b'e');
            self.k = self.j + 3;
            self.lookup();
            return;
        }
        if self.ends_in_s(b"ition") {
            let j = self.j;
            self.set_len(j + 1);
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("ition");
            self.k = old_k;
        } else if self.ends_in_s(b"ation") {
            let j = self.j;
            self.set_len(j + 3);
            self.write(b'e');
            self.k = j + 3;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("ation");
            self.k = old_k;
        }
        if self.ends_in_s(b"ication") {
            let j = self.j;
            self.set_len(j + 1);
            self.write(b'y');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("ication");
            self.k = old_k;
        }
        self.j = self.k - 3;
        let j = self.j;
        self.set_len(j + 1);
        self.write(b'e');
        self.k = j + 1;
        if self.lookup() {
            return;
        }
        self.set_len(j + 1);
        self.k = j;
        if self.lookup() {
            return;
        }
        self.set_len(j + 1);
        self.append("ion");
        self.k = old_k;
    }

    fn er_and_or_endings(&mut self) {
        let old_k = self.k;
        if self.ch(self.k) != b'r' {
            return;
        }
        if self.ends4(b'i', b'z', b'e', b'r') {
            self.set_len(self.j + 4);
            self.k = self.j + 3;
            self.lookup();
            return;
        }
        if self.ends2(b'e', b'r') || self.ends2(b'o', b'r') {
            let j = self.j;
            let wc = self.ch(j + 1);
            if self.double_c(j) {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(j - 1);
                self.write(c);
            }
            if self.ch(j) == b'i' {
                self.set_ch(j, b'y');
                self.set_len(j + 1);
                self.k = j;
                if self.lookup() {
                    return;
                }
                self.set_ch(j, b'i');
                self.write(b'e');
            }
            if self.ch(j) == b'e' {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.write(b'e');
            }
            self.set_len(j + 2);
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.write(wc);
            self.write(b'r');
            self.k = old_k;
        }
    }

    fn ly_endings(&mut self) {
        let old_k = self.k;
        if self.ends2(b'l', b'y') {
            let j = self.j;
            self.set_ch(j + 2, b'e');
            if self.lookup() {
                return;
            }
            self.set_ch(j + 2, b'y');
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            if j > 0 && self.ch(j - 1) == b'a' && self.ch(j) == b'l' {
                return;
            }
            self.append("ly");
            self.k = old_k;
            if j > 0 && self.ch(j - 1) == b'a' && self.ch(j) == b'b' {
                self.set_ch(j + 2, b'e');
                self.k = j + 2;
                return;
            }
            if self.ch(j) == b'i' {
                self.set_len(j);
                self.write(b'y');
                self.k = j;
                if self.lookup() {
                    return;
                }
                self.set_len(j);
                self.append("ily");
                self.k = old_k;
            }
            self.set_len(j + 1);
            self.k = j;
        }
    }

    fn al_endings(&mut self) {
        let old_k = self.k;
        if self.len() < 4 {
            return;
        }
        if self.ends2(b'a', b'l') {
            let j = self.j;
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            if self.double_c(j) {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                let c = self.ch(j - 1);
                self.write(c);
            }
            self.set_len(j + 1);
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("um");
            self.k = j + 2;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("al");
            self.k = old_k;
            if j > 0 && self.ch(j - 1) == b'i' && self.ch(j) == b'c' {
                self.set_len(j - 1);
                self.k = j - 2;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                self.write(b'y');
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                self.append("ic");
                self.k = j;
                self.lookup();
                return;
            }
            if self.ch(j) == b'i' {
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.append("ial");
                self.k = old_k;
                self.lookup();
            }
        }
    }

    fn ive_endings(&mut self) {
        let old_k = self.k;
        if self.ends3(b'i', b'v', b'e') {
            let j = self.j;
            self.set_len(j + 1);
            self.k = j;
            if self.lookup() {
                return;
            }
            self.write(b'e');
            self.k = j + 1;
            if self.lookup() {
                return;
            }
            self.set_len(j + 1);
            self.append("ive");
            if j > 0 && self.ch(j - 1) == b'a' && self.ch(j) == b't' {
                self.set_ch(j - 1, b'e');
                self.set_len(j);
                self.k = j - 1;
                if self.lookup() {
                    return;
                }
                self.set_len(j - 1);
                if self.lookup() {
                    return;
                }
                self.append("ative");
                self.k = old_k;
            }
            self.set_ch(j + 2, b'o');
            self.set_ch(j + 3, b'n');
            if self.lookup() {
                return;
            }
            self.set_ch(j + 2, b'v');
            self.set_ch(j + 3, b'e');
            self.k = old_k;
        }
    }

    /// `KStemmer.stem(char[], int)`: `Some(stem)` when changed.
    fn stem(term: &[u8]) -> Option<String> {
        let len = term.len();
        let k = len as i32 - 1;
        if k <= 1 || k >= MAX_WORD_LEN as i32 - 1 {
            return None;
        }
        if let Some(e) = dict().get(term) {
            return e.root.map(str::to_string);
        }
        if !term.iter().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
        let mut buf = vec![0u8; len.max(22) + 10];
        buf[..len].copy_from_slice(term);
        let mut s = KStemmer { buf, len, j: 0, k, matched: None };
        let steps: [fn(&mut KStemmer); 9] = [
            KStemmer::plural,
            KStemmer::past_tense,
            KStemmer::aspect,
            KStemmer::ity_endings,
            KStemmer::ness_endings,
            KStemmer::ion_endings,
            KStemmer::er_and_or_endings,
            KStemmer::ly_endings,
            KStemmer::al_endings,
        ];
        let tail: [fn(&mut KStemmer); 8] = [
            KStemmer::ive_endings,
            KStemmer::ize_endings,
            KStemmer::ment_endings,
            KStemmer::ble_endings,
            KStemmer::ism_endings,
            KStemmer::ic_endings,
            KStemmer::ncy_endings,
            KStemmer::nce_endings,
        ];
        'outer: {
            for f in steps {
                f(&mut s);
                if s.matched.is_some() {
                    break 'outer;
                }
            }
            s.word_in_dict();
            for f in tail {
                f(&mut s);
                if s.matched.is_some() {
                    break 'outer;
                }
            }
        }
        if let Some(e) = s.matched {
            if let Some(r) = e.root {
                return Some(r.to_string());
            }
        }
        Some(String::from_utf8_lossy(&s.buf[..s.len]).into_owned())
    }
}

/// Stem a lowercased English word exactly like Lucene's `KStemFilter`.
pub fn stem(word: &str) -> String {
    match KStemmer::stem(word.as_bytes()) {
        Some(s) => s,
        None => word.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(pairs: &[(&str, &str)]) {
        for (input, expected) in pairs {
            assert_eq!(stem(input), *expected, "kstem({input})");
        }
    }

    /// Lucene `TestKStemmer.testVocabulary`: all 12,130 pairs of
    /// `kstemTestData.zip` (`kstem_examples.txt`, no sampling), stored as
    /// `testdata/lucene/kstem.tsv.zst`.
    #[test]
    fn lucene_vocabulary() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/analysis/testdata/lucene/kstem.tsv.zst");
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let text = String::from_utf8(zstd::decode_all(&raw[..]).expect("zstd")).expect("utf-8");
        let mut n = 0;
        let mut failures = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (w, expected) = line.split_once('\t').expect("pair");
            n += 1;
            let got = stem(w);
            if got != expected {
                failures.push(format!("{w:?}: expected {expected:?}, got {got:?}"));
            }
        }
        assert!(n >= 12_000, "only {n} pairs");
        assert!(
            failures.is_empty(),
            "{}/{n} mismatches:\n{}",
            failures.len(),
            failures[..failures.len().min(40)].join("\n")
        );
    }

    #[test]
    fn dictionary_conflations() {
        check(&[("aging", "age"), ("italians", "italy"), ("does", "do"), ("aide", "aide")]);
    }

    #[test]
    fn robustness() {
        // TestKStemmer.testEmptyTerm plus the length/alphabet guards.
        check(&[("", ""), ("a", "a"), ("is", "is"), ("123", "123"), ("café", "café"), ("Cats", "Cats")]);
        let long = "a".repeat(60) + "s";
        assert_eq!(stem(&long), long);
    }
}
