//! Snowball Turkish stemmer (Evren Kapusuz Çilden's algorithm, as
//! published on snowballstem.org).
//!
//! Steps: (1) [Snowball 3.x only, disabled here; see below]; (2) skip
//! words with fewer than two syllables; (3) strip nominal verb suffixes
//! (`-ymUş`, `-yDU`, `-ysA`, `-cAsInA`, `-sUnUz`, `-lAr`, `-DUr`, ...),
//! which decides whether noun-suffix stemming continues; (4) strip noun
//! suffix chains (`-lArI`, `-ndA`, `-ndAn`, `-nUn`, `-ylA`, `-ki`, ...)
//! with vowel-harmony checks; (5) postlude: append `U` to stems ending in
//! `d`/`g` where needed and devoice final `b c d ğ` → `p ç t k`; the
//! reserved words `ad`/`soyad` are left alone.
//!
//! Target revision: Snowball `34f3612e5e8c`, the one bundled by every
//! released Lucene 10.x (and Elasticsearch). That revision has no
//! apostrophe/proper-noun step (added in Snowball 3.x; Lucene's
//! `TurkishAnalyzer` handles apostrophes with `ApostropheFilter` instead), so
//! `r_remove_proper_noun_suffix` is kept but not called. Verified against the
//! full matching snowball-data vocabulary (`vocab_tests::turkish`).
//!
//! The rule code below is a mechanical Rust translation of the code that
//! Snowball generates from `turkish.sbl` (labelled blocks stand in
//! for the generator's `try`/`raise` gotos), running on a minimal port of
//! the Snowball runtime ([`Env`]). Outputs were checked against the
//! reference implementation.

/// Minimal Snowball runtime state over a `Vec<char>`.
struct Env {
    current: Vec<char>,
    cursor: i32,
    limit: i32,
    limit_backward: i32,
    bra: i32,
    ket: i32,
    b_continue_stemming_noun_suffixes: bool,
}

const G_VOWEL: &[char] = &['a', 'e', 'i', 'o', 'u', 'ö', 'ü', 'ı'];
const G_U: &[char] = &['i', 'u', 'ü', 'ı'];
const G_VOWEL1: &[char] = &['a', 'o', 'u', 'ı'];
const G_VOWEL2: &[char] = &['e', 'i', 'ö', 'ü'];
const G_VOWEL3: &[char] = &['a', 'ı'];
const G_VOWEL4: &[char] = &['e', 'i'];
const G_VOWEL5: &[char] = &['o', 'u'];
const G_VOWEL6: &[char] = &['ö', 'ü'];
const A_0: &[(&str, i32)] = &[("m", -1), ("n", -1), ("miz", -1), ("niz", -1), ("muz", -1), ("nuz", -1), ("müz", -1), ("nüz", -1), ("mız", -1), ("nız", -1)];
const A_1: &[(&str, i32)] = &[("leri", -1), ("ları", -1)];
const A_2: &[(&str, i32)] = &[("ni", -1), ("nu", -1), ("nü", -1), ("nı", -1)];
const A_3: &[(&str, i32)] = &[("in", -1), ("un", -1), ("ün", -1), ("ın", -1)];
const A_4: &[(&str, i32)] = &[("a", -1), ("e", -1)];
const A_5: &[(&str, i32)] = &[("na", -1), ("ne", -1)];
const A_6: &[(&str, i32)] = &[("da", -1), ("ta", -1), ("de", -1), ("te", -1)];
const A_7: &[(&str, i32)] = &[("nda", -1), ("nde", -1)];
const A_8: &[(&str, i32)] = &[("dan", -1), ("tan", -1), ("den", -1), ("ten", -1)];
const A_9: &[(&str, i32)] = &[("ndan", -1), ("nden", -1)];
const A_10: &[(&str, i32)] = &[("la", -1), ("le", -1)];
const A_11: &[(&str, i32)] = &[("ca", -1), ("ce", -1)];
const A_12: &[(&str, i32)] = &[("im", -1), ("um", -1), ("üm", -1), ("ım", -1)];
const A_13: &[(&str, i32)] = &[("sin", -1), ("sun", -1), ("sün", -1), ("sın", -1)];
const A_14: &[(&str, i32)] = &[("iz", -1), ("uz", -1), ("üz", -1), ("ız", -1)];
const A_15: &[(&str, i32)] = &[("siniz", -1), ("sunuz", -1), ("sünüz", -1), ("sınız", -1)];
const A_16: &[(&str, i32)] = &[("lar", -1), ("ler", -1)];
const A_17: &[(&str, i32)] = &[("niz", -1), ("nuz", -1), ("nüz", -1), ("nız", -1)];
const A_18: &[(&str, i32)] = &[("dir", -1), ("tir", -1), ("dur", -1), ("tur", -1), ("dür", -1), ("tür", -1), ("dır", -1), ("tır", -1)];
const A_19: &[(&str, i32)] = &[("casına", -1), ("cesine", -1)];
const A_20: &[(&str, i32)] = &[("di", -1), ("ti", -1), ("dik", -1), ("tik", -1), ("duk", -1), ("tuk", -1), ("dük", -1), ("tük", -1), ("dık", -1), ("tık", -1), ("dim", -1), ("tim", -1), ("dum", -1), ("tum", -1), ("düm", -1), ("tüm", -1), ("dım", -1), ("tım", -1), ("din", -1), ("tin", -1), ("dun", -1), ("tun", -1), ("dün", -1), ("tün", -1), ("dın", -1), ("tın", -1), ("du", -1), ("tu", -1), ("dü", -1), ("tü", -1), ("dı", -1), ("tı", -1)];
const A_21: &[(&str, i32)] = &[("sa", -1), ("se", -1), ("sak", -1), ("sek", -1), ("sam", -1), ("sem", -1), ("san", -1), ("sen", -1)];
const A_22: &[(&str, i32)] = &[("miş", -1), ("muş", -1), ("müş", -1), ("mış", -1)];
const A_23: &[(&str, i32)] = &[("b", 1), ("c", 2), ("d", 3), ("ğ", 4)];
const AS_23: &[&str] = &["p", "ç", "t", "k"];

impl Env {
    fn new(word: &str) -> Self {
        let current: Vec<char> = word.chars().collect();
        let limit = current.len() as i32;
        Self {
            current,
            cursor: 0,
            limit,
            limit_backward: 0,
            bra: 0,
            ket: limit,
            b_continue_stemming_noun_suffixes: false,
        }
    }

    fn ch(&self, i: i32) -> char {
        self.current[i as usize]
    }

    fn in_grouping_b(&mut self, g: &[char]) -> bool {
        if self.cursor <= self.limit_backward || !g.contains(&self.ch(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    fn out_grouping_b(&mut self, g: &[char]) -> bool {
        if self.cursor <= self.limit_backward || g.contains(&self.ch(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    fn go_out_grouping_b(&mut self, g: &[char]) -> bool {
        while self.cursor > self.limit_backward {
            if g.contains(&self.ch(self.cursor - 1)) {
                return true;
            }
            self.cursor -= 1;
        }
        false
    }

    fn go_out_grouping(&mut self, g: &[char]) -> bool {
        while self.cursor < self.limit {
            if g.contains(&self.ch(self.cursor)) {
                return true;
            }
            self.cursor += 1;
        }
        false
    }

    fn eq_s_b(&mut self, s: &str) -> bool {
        let sc: Vec<char> = s.chars().collect();
        let len = sc.len() as i32;
        if self.cursor - self.limit_backward < len {
            return false;
        }
        let start = (self.cursor - len) as usize;
        if self.current[start..self.cursor as usize] != sc[..] {
            return false;
        }
        self.cursor -= len;
        true
    }

    /// Longest entry of `v` ending at the cursor (within the backward
    /// limit): moves the cursor before it and returns its result, or 0.
    fn find_among_b(&mut self, v: &[(&str, i32)]) -> i32 {
        let mut best: Option<(usize, i32)> = None;
        for (s, result) in v {
            let sc: Vec<char> = s.chars().collect();
            let len = sc.len();
            if (self.cursor - self.limit_backward) < len as i32 {
                continue;
            }
            let start = self.cursor as usize - len;
            if self.current[start..self.cursor as usize] == sc[..]
                && best.map(|(l, _)| len > l).unwrap_or(true)
            {
                best = Some((len, *result));
            }
        }
        match best {
            Some((len, result)) => {
                self.cursor -= len as i32;
                result
            }
            None => 0,
        }
    }

    fn slice_from(&mut self, s: &str) {
        let (bra, ket) = (self.bra as usize, self.ket as usize);
        let repl: Vec<char> = s.chars().collect();
        let adjustment = repl.len() as i32 - (ket - bra) as i32;
        self.current.splice(bra..ket, repl);
        self.limit += adjustment;
        if self.cursor >= ket as i32 {
            self.cursor += adjustment;
        } else if self.cursor > bra as i32 {
            self.cursor = bra as i32;
        }
        self.ket = self.bra + s.chars().count() as i32;
    }

    fn slice_del(&mut self) {
        self.slice_from("");
    }
}

// ---------------------------------------------------------------------
// Generated rule code (see module docs).
// ---------------------------------------------------------------------

#[allow(unused_assignments, non_snake_case, clippy::all)]
impl Env {
    fn r_check_vowel_harmony(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        v_1 = self.limit - self.cursor;
        if !self.go_out_grouping_b(G_VOWEL) {
            return false;
        }
        'loop1: loop {
            v_2 = self.limit - self.cursor;
            'try2: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'a' {
                    break 'try2;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL1) {
                    break 'try2;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try3: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'e' {
                    break 'try3;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL2) {
                    break 'try3;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try4: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ı' {
                    break 'try4;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL3) {
                    break 'try4;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try5: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'i' {
                    break 'try5;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL4) {
                    break 'try5;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try6: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'o' {
                    break 'try6;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL5) {
                    break 'try6;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try7: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ö' {
                    break 'try7;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL6) {
                    break 'try7;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            'try8: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'u' {
                    break 'try8;
                }
                self.cursor -= 1;
                if !self.go_out_grouping_b(G_VOWEL5) {
                    break 'try8;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_2;
            if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ü' {
                return false;
            }
            self.cursor -= 1;
            if !self.go_out_grouping_b(G_VOWEL6) {
                return false;
            }
            break 'loop1;
        }
        self.cursor = self.limit - v_1;
        return true;
    }

    fn r_mark_suffix_with_optional_n_consonant(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'n' {
                    break 'try2;
                }
                self.cursor -= 1;
                v_2 = self.limit - self.cursor;
                if !self.in_grouping_b(G_VOWEL) {
                    break 'try2;
                }
                self.cursor = self.limit - v_2;
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try3: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'n' {
                    break 'try3;
                }
                self.cursor -= 1;
                return false;
            }
            v_3 = self.limit - self.cursor;
            if self.cursor <= self.limit_backward {
                return false;
            }
            self.cursor -= 1;
            if !self.in_grouping_b(G_VOWEL) {
                return false;
            }
            self.cursor = self.limit - v_3;
            break 'loop1;
        }
        return true;
    }

    fn r_mark_suffix_with_optional_s_consonant(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 's' {
                    break 'try2;
                }
                self.cursor -= 1;
                v_2 = self.limit - self.cursor;
                if !self.in_grouping_b(G_VOWEL) {
                    break 'try2;
                }
                self.cursor = self.limit - v_2;
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try3: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 's' {
                    break 'try3;
                }
                self.cursor -= 1;
                return false;
            }
            v_3 = self.limit - self.cursor;
            if self.cursor <= self.limit_backward {
                return false;
            }
            self.cursor -= 1;
            if !self.in_grouping_b(G_VOWEL) {
                return false;
            }
            self.cursor = self.limit - v_3;
            break 'loop1;
        }
        return true;
    }

    fn r_mark_suffix_with_optional_y_consonant(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'y' {
                    break 'try2;
                }
                self.cursor -= 1;
                v_2 = self.limit - self.cursor;
                if !self.in_grouping_b(G_VOWEL) {
                    break 'try2;
                }
                self.cursor = self.limit - v_2;
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try3: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'y' {
                    break 'try3;
                }
                self.cursor -= 1;
                return false;
            }
            v_3 = self.limit - self.cursor;
            if self.cursor <= self.limit_backward {
                return false;
            }
            self.cursor -= 1;
            if !self.in_grouping_b(G_VOWEL) {
                return false;
            }
            self.cursor = self.limit - v_3;
            break 'loop1;
        }
        return true;
    }

    fn r_mark_suffix_with_optional_U_vowel(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                if !self.in_grouping_b(G_U) {
                    break 'try2;
                }
                v_2 = self.limit - self.cursor;
                if !self.out_grouping_b(G_VOWEL) {
                    break 'try2;
                }
                self.cursor = self.limit - v_2;
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try3: {
                if !self.in_grouping_b(G_U) {
                    break 'try3;
                }
                return false;
            }
            v_3 = self.limit - self.cursor;
            if self.cursor <= self.limit_backward {
                return false;
            }
            self.cursor -= 1;
            if !self.out_grouping_b(G_VOWEL) {
                return false;
            }
            self.cursor = self.limit - v_3;
            break 'loop1;
        }
        return true;
    }

    fn r_mark_possessives(&mut self) -> bool {
        if self.find_among_b(A_0) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_U_vowel();
    }

    fn r_mark_sU(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if !self.in_grouping_b(G_U) {
            return false;
        }
        return self.r_mark_suffix_with_optional_s_consonant();
    }

    fn r_mark_lArI(&mut self) -> bool {
        return self.find_among_b(A_1) != 0;
    }

    fn r_mark_yU(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if !self.in_grouping_b(G_U) {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_nU(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_2) != 0;
    }

    fn r_mark_nUn(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_3) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_n_consonant();
    }

    fn r_mark_yA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_4) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_nA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_5) != 0;
    }

    fn r_mark_DA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_6) != 0;
    }

    fn r_mark_ndA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_7) != 0;
    }

    fn r_mark_DAn(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_8) != 0;
    }

    fn r_mark_ndAn(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_9) != 0;
    }

    fn r_mark_ylA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_10) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_ncA(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_11) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_n_consonant();
    }

    fn r_mark_yUm(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_12) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_sUn(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_13) != 0;
    }

    fn r_mark_yUz(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_14) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_sUnUz(&mut self) -> bool {
        return self.find_among_b(A_15) != 0;
    }

    fn r_mark_lAr(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_16) != 0;
    }

    fn r_mark_nUz(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_17) != 0;
    }

    fn r_mark_DUr(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        return self.find_among_b(A_18) != 0;
    }

    fn r_mark_cAsInA(&mut self) -> bool {
        return self.find_among_b(A_19) != 0;
    }

    fn r_mark_yDU(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_20) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_ysA(&mut self) -> bool {
        if self.find_among_b(A_21) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_ymUs_(&mut self) -> bool {
        if !self.r_check_vowel_harmony() {
            return false;
        }
        if self.find_among_b(A_22) == 0 {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_mark_yken(&mut self) -> bool {
        if !self.eq_s_b("ken") {
            return false;
        }
        return self.r_mark_suffix_with_optional_y_consonant();
    }

    fn r_stem_nominal_verb_suffixes(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_10: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        let mut v_4: i32 = 0;
        let mut v_5: i32 = 0;
        let mut v_6: i32 = 0;
        let mut v_7: i32 = 0;
        let mut v_8: i32 = 0;
        let mut v_9: i32 = 0;
        self.ket = self.cursor;
        self.b_continue_stemming_noun_suffixes = true;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                'loop3: loop {
                    v_2 = self.limit - self.cursor;
                    'try4: {
                        if !self.r_mark_ymUs_() {
                            break 'try4;
                        }
                        break 'loop3;
                    }
                    self.cursor = self.limit - v_2;
                    'try5: {
                        if !self.r_mark_yDU() {
                            break 'try5;
                        }
                        break 'loop3;
                    }
                    self.cursor = self.limit - v_2;
                    'try6: {
                        if !self.r_mark_ysA() {
                            break 'try6;
                        }
                        break 'loop3;
                    }
                    self.cursor = self.limit - v_2;
                    if !self.r_mark_yken() {
                        break 'try2;
                    }
                    break 'loop3;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try7: {
                if !self.r_mark_cAsInA() {
                    break 'try7;
                }
                'loop8: loop {
                    v_3 = self.limit - self.cursor;
                    'try9: {
                        if !self.r_mark_sUnUz() {
                            break 'try9;
                        }
                        break 'loop8;
                    }
                    self.cursor = self.limit - v_3;
                    'try10: {
                        if !self.r_mark_lAr() {
                            break 'try10;
                        }
                        break 'loop8;
                    }
                    self.cursor = self.limit - v_3;
                    'try11: {
                        if !self.r_mark_yUm() {
                            break 'try11;
                        }
                        break 'loop8;
                    }
                    self.cursor = self.limit - v_3;
                    'try12: {
                        if !self.r_mark_sUn() {
                            break 'try12;
                        }
                        break 'loop8;
                    }
                    self.cursor = self.limit - v_3;
                    'try13: {
                        if !self.r_mark_yUz() {
                            break 'try13;
                        }
                        break 'loop8;
                    }
                    self.cursor = self.limit - v_3;
                    break 'loop8;
                }
                if !self.r_mark_ymUs_() {
                    break 'try7;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try14: {
                if !self.r_mark_lAr() {
                    break 'try14;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_4 = self.limit - self.cursor;
                'try15: {
                    self.ket = self.cursor;
                    'loop16: loop {
                        v_5 = self.limit - self.cursor;
                        'try17: {
                            if !self.r_mark_DUr() {
                                break 'try17;
                            }
                            break 'loop16;
                        }
                        self.cursor = self.limit - v_5;
                        'try18: {
                            if !self.r_mark_yDU() {
                                break 'try18;
                            }
                            break 'loop16;
                        }
                        self.cursor = self.limit - v_5;
                        'try19: {
                            if !self.r_mark_ysA() {
                                break 'try19;
                            }
                            break 'loop16;
                        }
                        self.cursor = self.limit - v_5;
                        if !self.r_mark_ymUs_() {
                            self.cursor = self.limit - v_4;
                            break 'try15;
                        }
                        break 'loop16;
                    }
                }
                self.b_continue_stemming_noun_suffixes = false;
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try20: {
                if !self.r_mark_nUz() {
                    break 'try20;
                }
                'loop21: loop {
                    v_6 = self.limit - self.cursor;
                    'try22: {
                        if !self.r_mark_yDU() {
                            break 'try22;
                        }
                        break 'loop21;
                    }
                    self.cursor = self.limit - v_6;
                    if !self.r_mark_ysA() {
                        break 'try20;
                    }
                    break 'loop21;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try23: {
                'loop24: loop {
                    v_7 = self.limit - self.cursor;
                    'try25: {
                        if !self.r_mark_sUnUz() {
                            break 'try25;
                        }
                        break 'loop24;
                    }
                    self.cursor = self.limit - v_7;
                    'try26: {
                        if !self.r_mark_yUz() {
                            break 'try26;
                        }
                        break 'loop24;
                    }
                    self.cursor = self.limit - v_7;
                    'try27: {
                        if !self.r_mark_sUn() {
                            break 'try27;
                        }
                        break 'loop24;
                    }
                    self.cursor = self.limit - v_7;
                    if !self.r_mark_yUm() {
                        break 'try23;
                    }
                    break 'loop24;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_8 = self.limit - self.cursor;
                'try28: {
                    self.ket = self.cursor;
                    if !self.r_mark_ymUs_() {
                        self.cursor = self.limit - v_8;
                        break 'try28;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            if !self.r_mark_DUr() {
                return false;
            }
            self.bra = self.cursor;
            self.slice_del();
            v_9 = self.limit - self.cursor;
            'try29: {
                self.ket = self.cursor;
                'loop30: loop {
                    v_10 = self.limit - self.cursor;
                    'try31: {
                        if !self.r_mark_sUnUz() {
                            break 'try31;
                        }
                        break 'loop30;
                    }
                    self.cursor = self.limit - v_10;
                    'try32: {
                        if !self.r_mark_lAr() {
                            break 'try32;
                        }
                        break 'loop30;
                    }
                    self.cursor = self.limit - v_10;
                    'try33: {
                        if !self.r_mark_yUm() {
                            break 'try33;
                        }
                        break 'loop30;
                    }
                    self.cursor = self.limit - v_10;
                    'try34: {
                        if !self.r_mark_sUn() {
                            break 'try34;
                        }
                        break 'loop30;
                    }
                    self.cursor = self.limit - v_10;
                    'try35: {
                        if !self.r_mark_yUz() {
                            break 'try35;
                        }
                        break 'loop30;
                    }
                    self.cursor = self.limit - v_10;
                    break 'loop30;
                }
                if !self.r_mark_ymUs_() {
                    self.cursor = self.limit - v_9;
                    break 'try29;
                }
            }
            break 'loop1;
        }
        self.bra = self.cursor;
        self.slice_del();
        return true;
    }

    fn r_stem_suffix_chain_before_ki(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_10: i32 = 0;
        let mut v_11: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        let mut v_4: i32 = 0;
        let mut v_5: i32 = 0;
        let mut v_6: i32 = 0;
        let mut v_7: i32 = 0;
        let mut v_8: i32 = 0;
        let mut v_9: i32 = 0;
        self.ket = self.cursor;
        if !self.eq_s_b("ki") {
            return false;
        }
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                if !self.r_mark_DA() {
                    break 'try2;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_2 = self.limit - self.cursor;
                'try3: {
                    self.ket = self.cursor;
                    'loop4: loop {
                        v_3 = self.limit - self.cursor;
                        'try5: {
                            if !self.r_mark_lAr() {
                                break 'try5;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_4 = self.limit - self.cursor;
                            'try6: {
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_4;
                                    break 'try6;
                                }
                            }
                            break 'loop4;
                        }
                        self.cursor = self.limit - v_3;
                        if !self.r_mark_possessives() {
                            self.cursor = self.limit - v_2;
                            break 'try3;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        v_5 = self.limit - self.cursor;
                        'try7: {
                            self.ket = self.cursor;
                            if !self.r_mark_lAr() {
                                self.cursor = self.limit - v_5;
                                break 'try7;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            if !self.r_stem_suffix_chain_before_ki() {
                                self.cursor = self.limit - v_5;
                                break 'try7;
                            }
                        }
                        break 'loop4;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try8: {
                if !self.r_mark_nUn() {
                    break 'try8;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_6 = self.limit - self.cursor;
                'try9: {
                    self.ket = self.cursor;
                    'loop10: loop {
                        v_7 = self.limit - self.cursor;
                        'try11: {
                            if !self.r_mark_lArI() {
                                break 'try11;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            break 'loop10;
                        }
                        self.cursor = self.limit - v_7;
                        'try12: {
                            self.ket = self.cursor;
                            'loop13: loop {
                                v_8 = self.limit - self.cursor;
                                'try14: {
                                    if !self.r_mark_possessives() {
                                        break 'try14;
                                    }
                                    break 'loop13;
                                }
                                self.cursor = self.limit - v_8;
                                if !self.r_mark_sU() {
                                    break 'try12;
                                }
                                break 'loop13;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_9 = self.limit - self.cursor;
                            'try15: {
                                self.ket = self.cursor;
                                if !self.r_mark_lAr() {
                                    self.cursor = self.limit - v_9;
                                    break 'try15;
                                }
                                self.bra = self.cursor;
                                self.slice_del();
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_9;
                                    break 'try15;
                                }
                            }
                            break 'loop10;
                        }
                        self.cursor = self.limit - v_7;
                        if !self.r_stem_suffix_chain_before_ki() {
                            self.cursor = self.limit - v_6;
                            break 'try9;
                        }
                        break 'loop10;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            if !self.r_mark_ndA() {
                return false;
            }
            'loop16: loop {
                v_10 = self.limit - self.cursor;
                'try17: {
                    if !self.r_mark_lArI() {
                        break 'try17;
                    }
                    self.bra = self.cursor;
                    self.slice_del();
                    break 'loop16;
                }
                self.cursor = self.limit - v_10;
                'try18: {
                    if !self.r_mark_sU() {
                        break 'try18;
                    }
                    self.bra = self.cursor;
                    self.slice_del();
                    v_11 = self.limit - self.cursor;
                    'try19: {
                        self.ket = self.cursor;
                        if !self.r_mark_lAr() {
                            self.cursor = self.limit - v_11;
                            break 'try19;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        if !self.r_stem_suffix_chain_before_ki() {
                            self.cursor = self.limit - v_11;
                            break 'try19;
                        }
                    }
                    break 'loop16;
                }
                self.cursor = self.limit - v_10;
                if !self.r_stem_suffix_chain_before_ki() {
                    return false;
                }
                break 'loop16;
            }
            break 'loop1;
        }
        return true;
    }

    fn r_stem_noun_suffixes(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_10: i32 = 0;
        let mut v_11: i32 = 0;
        let mut v_12: i32 = 0;
        let mut v_13: i32 = 0;
        let mut v_14: i32 = 0;
        let mut v_15: i32 = 0;
        let mut v_16: i32 = 0;
        let mut v_17: i32 = 0;
        let mut v_18: i32 = 0;
        let mut v_19: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_20: i32 = 0;
        let mut v_21: i32 = 0;
        let mut v_22: i32 = 0;
        let mut v_23: i32 = 0;
        let mut v_24: i32 = 0;
        let mut v_25: i32 = 0;
        let mut v_26: i32 = 0;
        let mut v_27: i32 = 0;
        let mut v_3: i32 = 0;
        let mut v_4: i32 = 0;
        let mut v_5: i32 = 0;
        let mut v_6: i32 = 0;
        let mut v_7: i32 = 0;
        let mut v_8: i32 = 0;
        let mut v_9: i32 = 0;
        'loop1: loop {
            v_1 = self.limit - self.cursor;
            'try2: {
                self.ket = self.cursor;
                if !self.r_mark_lAr() {
                    break 'try2;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_2 = self.limit - self.cursor;
                'try3: {
                    if !self.r_stem_suffix_chain_before_ki() {
                        self.cursor = self.limit - v_2;
                        break 'try3;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try4: {
                self.ket = self.cursor;
                if !self.r_mark_ncA() {
                    break 'try4;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_3 = self.limit - self.cursor;
                'try5: {
                    'loop6: loop {
                        v_4 = self.limit - self.cursor;
                        'try7: {
                            self.ket = self.cursor;
                            if !self.r_mark_lArI() {
                                break 'try7;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            break 'loop6;
                        }
                        self.cursor = self.limit - v_4;
                        'try8: {
                            self.ket = self.cursor;
                            'loop9: loop {
                                v_5 = self.limit - self.cursor;
                                'try10: {
                                    if !self.r_mark_possessives() {
                                        break 'try10;
                                    }
                                    break 'loop9;
                                }
                                self.cursor = self.limit - v_5;
                                if !self.r_mark_sU() {
                                    break 'try8;
                                }
                                break 'loop9;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_6 = self.limit - self.cursor;
                            'try11: {
                                self.ket = self.cursor;
                                if !self.r_mark_lAr() {
                                    self.cursor = self.limit - v_6;
                                    break 'try11;
                                }
                                self.bra = self.cursor;
                                self.slice_del();
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_6;
                                    break 'try11;
                                }
                            }
                            break 'loop6;
                        }
                        self.cursor = self.limit - v_4;
                        self.ket = self.cursor;
                        if !self.r_mark_lAr() {
                            self.cursor = self.limit - v_3;
                            break 'try5;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        if !self.r_stem_suffix_chain_before_ki() {
                            self.cursor = self.limit - v_3;
                            break 'try5;
                        }
                        break 'loop6;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try12: {
                self.ket = self.cursor;
                'loop13: loop {
                    v_7 = self.limit - self.cursor;
                    'try14: {
                        if !self.r_mark_ndA() {
                            break 'try14;
                        }
                        break 'loop13;
                    }
                    self.cursor = self.limit - v_7;
                    if !self.r_mark_nA() {
                        break 'try12;
                    }
                    break 'loop13;
                }
                'loop15: loop {
                    v_8 = self.limit - self.cursor;
                    'try16: {
                        if !self.r_mark_lArI() {
                            break 'try16;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        break 'loop15;
                    }
                    self.cursor = self.limit - v_8;
                    'try17: {
                        if !self.r_mark_sU() {
                            break 'try17;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        v_9 = self.limit - self.cursor;
                        'try18: {
                            self.ket = self.cursor;
                            if !self.r_mark_lAr() {
                                self.cursor = self.limit - v_9;
                                break 'try18;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            if !self.r_stem_suffix_chain_before_ki() {
                                self.cursor = self.limit - v_9;
                                break 'try18;
                            }
                        }
                        break 'loop15;
                    }
                    self.cursor = self.limit - v_8;
                    if !self.r_stem_suffix_chain_before_ki() {
                        break 'try12;
                    }
                    break 'loop15;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try19: {
                self.ket = self.cursor;
                'loop20: loop {
                    v_10 = self.limit - self.cursor;
                    'try21: {
                        if !self.r_mark_ndAn() {
                            break 'try21;
                        }
                        break 'loop20;
                    }
                    self.cursor = self.limit - v_10;
                    if !self.r_mark_nU() {
                        break 'try19;
                    }
                    break 'loop20;
                }
                'loop22: loop {
                    v_11 = self.limit - self.cursor;
                    'try23: {
                        if !self.r_mark_sU() {
                            break 'try23;
                        }
                        self.bra = self.cursor;
                        self.slice_del();
                        v_12 = self.limit - self.cursor;
                        'try24: {
                            self.ket = self.cursor;
                            if !self.r_mark_lAr() {
                                self.cursor = self.limit - v_12;
                                break 'try24;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            if !self.r_stem_suffix_chain_before_ki() {
                                self.cursor = self.limit - v_12;
                                break 'try24;
                            }
                        }
                        break 'loop22;
                    }
                    self.cursor = self.limit - v_11;
                    if !self.r_mark_lArI() {
                        break 'try19;
                    }
                    break 'loop22;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try25: {
                self.ket = self.cursor;
                if !self.r_mark_DAn() {
                    break 'try25;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_13 = self.limit - self.cursor;
                'try26: {
                    self.ket = self.cursor;
                    'loop27: loop {
                        v_14 = self.limit - self.cursor;
                        'try28: {
                            if !self.r_mark_possessives() {
                                break 'try28;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_15 = self.limit - self.cursor;
                            'try29: {
                                self.ket = self.cursor;
                                if !self.r_mark_lAr() {
                                    self.cursor = self.limit - v_15;
                                    break 'try29;
                                }
                                self.bra = self.cursor;
                                self.slice_del();
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_15;
                                    break 'try29;
                                }
                            }
                            break 'loop27;
                        }
                        self.cursor = self.limit - v_14;
                        'try30: {
                            if !self.r_mark_lAr() {
                                break 'try30;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_16 = self.limit - self.cursor;
                            'try31: {
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_16;
                                    break 'try31;
                                }
                            }
                            break 'loop27;
                        }
                        self.cursor = self.limit - v_14;
                        if !self.r_stem_suffix_chain_before_ki() {
                            self.cursor = self.limit - v_13;
                            break 'try26;
                        }
                        break 'loop27;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try32: {
                self.ket = self.cursor;
                'loop33: loop {
                    v_17 = self.limit - self.cursor;
                    'try34: {
                        if !self.r_mark_nUn() {
                            break 'try34;
                        }
                        break 'loop33;
                    }
                    self.cursor = self.limit - v_17;
                    if !self.r_mark_ylA() {
                        break 'try32;
                    }
                    break 'loop33;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_18 = self.limit - self.cursor;
                'try35: {
                    'loop36: loop {
                        v_19 = self.limit - self.cursor;
                        'try37: {
                            self.ket = self.cursor;
                            if !self.r_mark_lAr() {
                                break 'try37;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            if !self.r_stem_suffix_chain_before_ki() {
                                break 'try37;
                            }
                            break 'loop36;
                        }
                        self.cursor = self.limit - v_19;
                        'try38: {
                            self.ket = self.cursor;
                            'loop39: loop {
                                v_20 = self.limit - self.cursor;
                                'try40: {
                                    if !self.r_mark_possessives() {
                                        break 'try40;
                                    }
                                    break 'loop39;
                                }
                                self.cursor = self.limit - v_20;
                                if !self.r_mark_sU() {
                                    break 'try38;
                                }
                                break 'loop39;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_21 = self.limit - self.cursor;
                            'try41: {
                                self.ket = self.cursor;
                                if !self.r_mark_lAr() {
                                    self.cursor = self.limit - v_21;
                                    break 'try41;
                                }
                                self.bra = self.cursor;
                                self.slice_del();
                                if !self.r_stem_suffix_chain_before_ki() {
                                    self.cursor = self.limit - v_21;
                                    break 'try41;
                                }
                            }
                            break 'loop36;
                        }
                        self.cursor = self.limit - v_19;
                        if !self.r_stem_suffix_chain_before_ki() {
                            self.cursor = self.limit - v_18;
                            break 'try35;
                        }
                        break 'loop36;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try42: {
                self.ket = self.cursor;
                if !self.r_mark_lArI() {
                    break 'try42;
                }
                self.bra = self.cursor;
                self.slice_del();
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try43: {
                if !self.r_stem_suffix_chain_before_ki() {
                    break 'try43;
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            'try44: {
                self.ket = self.cursor;
                'loop45: loop {
                    v_22 = self.limit - self.cursor;
                    'try46: {
                        if !self.r_mark_DA() {
                            break 'try46;
                        }
                        break 'loop45;
                    }
                    self.cursor = self.limit - v_22;
                    'try47: {
                        if !self.r_mark_yU() {
                            break 'try47;
                        }
                        break 'loop45;
                    }
                    self.cursor = self.limit - v_22;
                    if !self.r_mark_yA() {
                        break 'try44;
                    }
                    break 'loop45;
                }
                self.bra = self.cursor;
                self.slice_del();
                v_23 = self.limit - self.cursor;
                'try48: {
                    self.ket = self.cursor;
                    'loop49: loop {
                        v_24 = self.limit - self.cursor;
                        'try50: {
                            if !self.r_mark_possessives() {
                                break 'try50;
                            }
                            self.bra = self.cursor;
                            self.slice_del();
                            v_25 = self.limit - self.cursor;
                            'try51: {
                                self.ket = self.cursor;
                                if !self.r_mark_lAr() {
                                    self.cursor = self.limit - v_25;
                                    break 'try51;
                                }
                            }
                            break 'loop49;
                        }
                        self.cursor = self.limit - v_24;
                        if !self.r_mark_lAr() {
                            self.cursor = self.limit - v_23;
                            break 'try48;
                        }
                        break 'loop49;
                    }
                    self.bra = self.cursor;
                    self.slice_del();
                    self.ket = self.cursor;
                    if !self.r_stem_suffix_chain_before_ki() {
                        self.cursor = self.limit - v_23;
                        break 'try48;
                    }
                }
                break 'loop1;
            }
            self.cursor = self.limit - v_1;
            self.ket = self.cursor;
            'loop52: loop {
                v_26 = self.limit - self.cursor;
                'try53: {
                    if !self.r_mark_possessives() {
                        break 'try53;
                    }
                    break 'loop52;
                }
                self.cursor = self.limit - v_26;
                if !self.r_mark_sU() {
                    return false;
                }
                break 'loop52;
            }
            self.bra = self.cursor;
            self.slice_del();
            v_27 = self.limit - self.cursor;
            'try54: {
                self.ket = self.cursor;
                if !self.r_mark_lAr() {
                    self.cursor = self.limit - v_27;
                    break 'try54;
                }
                self.bra = self.cursor;
                self.slice_del();
                if !self.r_stem_suffix_chain_before_ki() {
                    self.cursor = self.limit - v_27;
                    break 'try54;
                }
            }
            break 'loop1;
        }
        return true;
    }

    fn r_post_process_last_consonants(&mut self) -> bool {
        let mut among_var: i32 = 0;
        self.ket = self.cursor;
        among_var = self.find_among_b(A_23);
        if among_var == 0 {
            return false;
        }
        self.bra = self.cursor;
        self.slice_from(AS_23[(among_var - 1) as usize]);
        return true;
    }

    fn r_append_U_to_stems_ending_with_d_or_g(&mut self) -> bool {
        let mut v_1: i32 = 0;
        self.ket = self.cursor;
        self.bra = self.cursor;
        'loop1: loop {
            'try2: {
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'd' {
                    break 'try2;
                }
                self.cursor -= 1;
                break 'loop1;
            }
            if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'g' {
                return false;
            }
            self.cursor -= 1;
            break 'loop1;
        }
        if !self.go_out_grouping_b(G_VOWEL) {
            return false;
        }
        'loop3: loop {
            v_1 = self.limit - self.cursor;
            'try4: {
                'loop5: loop {
                    'try6: {
                        if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'a' {
                            break 'try6;
                        }
                        self.cursor -= 1;
                        break 'loop5;
                    }
                    if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ı' {
                        break 'try4;
                    }
                    self.cursor -= 1;
                    break 'loop5;
                }
                self.slice_from("ı");
                break 'loop3;
            }
            self.cursor = self.limit - v_1;
            'try7: {
                'loop8: loop {
                    'try9: {
                        if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'e' {
                            break 'try9;
                        }
                        self.cursor -= 1;
                        break 'loop8;
                    }
                    if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'i' {
                        break 'try7;
                    }
                    self.cursor -= 1;
                    break 'loop8;
                }
                self.slice_from("i");
                break 'loop3;
            }
            self.cursor = self.limit - v_1;
            'try10: {
                'loop11: loop {
                    'try12: {
                        if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'o' {
                            break 'try12;
                        }
                        self.cursor -= 1;
                        break 'loop11;
                    }
                    if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'u' {
                        break 'try10;
                    }
                    self.cursor -= 1;
                    break 'loop11;
                }
                self.slice_from("u");
                break 'loop3;
            }
            self.cursor = self.limit - v_1;
            'loop13: loop {
                'try14: {
                    if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ö' {
                        break 'try14;
                    }
                    self.cursor -= 1;
                    break 'loop13;
                }
                if self.cursor <= self.limit_backward || self.ch(self.cursor - 1) != 'ü' {
                    return false;
                }
                self.cursor -= 1;
                break 'loop13;
            }
            self.slice_from("ü");
            break 'loop3;
        }
        return true;
    }

    fn r_is_reserved_word(&mut self) -> bool {
        let mut v_1: i32 = 0;
        if !self.eq_s_b("ad") {
            return false;
        }
        v_1 = self.limit - self.cursor;
        'try1: {
            if !self.eq_s_b("soy") {
                self.cursor = self.limit - v_1;
                break 'try1;
            }
        }
        return self.cursor <= self.limit_backward;
    }

    #[allow(dead_code)]
    fn r_remove_proper_noun_suffix(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        let mut v_4: i32 = 0;
        v_1 = self.cursor;
        'try1: {
            self.bra = self.cursor;
            'loop2: loop {
                v_2 = self.cursor;
                'try3: {
                    'try4: {
                        if self.cursor == self.limit || self.ch(self.cursor) != '\'' {
                            break 'try4;
                        }
                        self.cursor += 1;
                        break 'try3;
                    }
                    self.cursor = v_2;
                    break 'loop2;
                }
                self.cursor = v_2;
                if self.cursor >= self.limit {
                    break 'try1;
                }
                self.cursor += 1;
            }
            self.ket = self.cursor;
            self.slice_del();
        }
        self.cursor = v_1;
        v_3 = self.cursor;
        'try5: {
            if self.cursor + 2 > self.limit {
                break 'try5;
            }
            self.cursor += 2;
            'loop6: loop {
                v_4 = self.cursor;
                'try7: {
                    if self.cursor == self.limit || self.ch(self.cursor) != '\'' {
                        break 'try7;
                    }
                    self.cursor += 1;
                    self.cursor = v_4;
                    break 'loop6;
                }
                self.cursor = v_4;
                if self.cursor >= self.limit {
                    break 'try5;
                }
                self.cursor += 1;
            }
            self.bra = self.cursor;
            self.cursor = self.limit;
            self.ket = self.cursor;
            self.slice_del();
        }
        self.cursor = v_3;
        return true;
    }

    fn r_more_than_one_syllable_word(&mut self) -> bool {
        let mut v_1: i32 = 0;
        v_1 = self.cursor;
        for _ in 0..2 {
            if !self.go_out_grouping(G_VOWEL) {
                return false;
            }
            self.cursor += 1;
        }
        self.cursor = v_1;
        return true;
    }

    fn r_postlude(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        let mut v_3: i32 = 0;
        self.limit_backward = self.cursor;
        self.cursor = self.limit;
        v_1 = self.limit - self.cursor;
        'try1: {
            if !self.r_is_reserved_word() {
                break 'try1;
            }
            return false;
        }
        self.cursor = self.limit - v_1;
        v_2 = self.limit - self.cursor;
        self.r_append_U_to_stems_ending_with_d_or_g();
        self.cursor = self.limit - v_2;
        v_3 = self.limit - self.cursor;
        self.r_post_process_last_consonants();
        self.cursor = self.limit - v_3;
        self.cursor = self.limit_backward;
        return true;
    }

    fn stem_impl(&mut self) -> bool {
        let mut v_1: i32 = 0;
        let mut v_2: i32 = 0;
        if !self.r_more_than_one_syllable_word() {
            return false;
        }
        self.limit_backward = self.cursor;
        self.cursor = self.limit;
        v_1 = self.limit - self.cursor;
        self.r_stem_nominal_verb_suffixes();
        self.cursor = self.limit - v_1;
        if !self.b_continue_stemming_noun_suffixes {
            return false;
        }
        v_2 = self.limit - self.cursor;
        self.r_stem_noun_suffixes();
        self.cursor = self.limit - v_2;
        self.cursor = self.limit_backward;
        return self.r_postlude();
    }

}

/// Stem a lowercased Turkish word.
pub fn stem(word: &str) -> String {
    let mut env = Env::new(word);
    env.stem_impl();
    env.current.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::stem;

    fn check(pairs: &[(&str, &str)]) {
        for (input, expected) in pairs {
            assert_eq!(stem(input), *expected, "turkish stem({input})");
        }
    }

    /// Outputs of Snowball `34f3612e5e8c` (the revision in Lucene 10.x).
    #[test]
    fn reference_vectors() {
        check(&[
            ("kitaplar", "kitap"),
            ("evlerimizden", "ev"),
            ("gözlükçüler", "gözlükçü"),
            ("çocuklarının", "çocuk"),
            ("arkadaşlarımla", "arkadaş"),
            ("kitap", "kitap"),
            ("kitabı", "kitap"),
            ("kitaplarımız", "kitap"),
            ("istanbul'da", "istanbul'"),
            ("ankara'dan", "ankara'"),
            ("geliyorum", "geliyor"),
            ("gidiyorsunuz", "gidiyor"),
            ("okuyacaklar", "okuyacak"),
            ("yazdım", "yaz"),
            ("güzelleştirmek", "güzelleştirmek"),
            ("evdeki", "ev"),
            ("masadakiler", "masadaki"),
            ("öğretmenler", "öğretmen"),
            ("öğrencilerin", "öğrenci"),
            ("bilgisayarlar", "bilgisayar"),
            ("çalışkanlık", "çalışkanlık"),
            ("mutluluk", "mutluluk"),
            ("güzellik", "güzellik"),
            ("yapılmış", "yapıl"),
            ("gelmişler", "gel"),
            ("ağaçlar", "ağaç"),
            ("kapıdan", "kapı"),
            ("denizde", "de"),
            ("okulda", "okul"),
            ("evde", "ev"),
            ("ad", "ad"),
            ("soyad", "soyad"),
            ("adlar", "ad"),
            ("gözler", "göz"),
            ("geldi", "gel"),
            ("gelecek", "gelecek"),
            ("yolcular", "yolcu"),
            ("sevgilim", "sevgil"),
            ("sözcükler", "sözcük"),
            ("kelimeler", "kelime"),
            ("dillerin", "dil"),
            ("türkçe", "türkçe"),
            ("arabalarımızdan", "araba"),
            ("evimizdekiler", "evimizdeki"),
            ("kitapçı", "kitapçı"),
            ("bakkal", "bakkal"),
            ("dolabı", "dolap"),
            ("kanadı", "kanadı"),
            ("yoğurdu", "yoğur"),
            ("ağacı", "ağaç"),
        ]);
    }

    #[test]
    fn robustness() {
        check(&[("", ""), ("a", "a"), ("ev", "ev"), ("123", "123"), ("'", "'"), ("abc", "abc")]);
        for w in ["ğğğ", "lar", "'''", "x'y'z", "ılarınızdan", "日本"] {
            let _ = stem(w);
        }
    }
}
