//! The Snowball runtime (Lucene `org.tartarus.snowball.SnowballProgram` and
//! `Among`) the translated generated stemmers run on: the word as UTF-16
//! code units' worth of chars (every Snowball alphabet here is in the BMP),
//! a cursor, forward / backward limits and the slice markers.

/// One entry of a Snowball `among` table.
pub struct Among {
    pub s: &'static [char],
    pub substring_i: i32,
    pub result: i32,
}

pub struct Env {
    current: Vec<char>,
    pub cursor: i32,
    pub limit: i32,
    pub limit_backward: i32,
    pub bra: i32,
    pub ket: i32,
}

impl Env {
    pub fn new(word: &str) -> Env {
        let current: Vec<char> = word.chars().collect();
        let n = current.len() as i32;
        Env { current, cursor: 0, limit: n, limit_backward: 0, bra: 0, ket: n }
    }

    pub fn current(&self) -> String {
        self.current.iter().collect()
    }

    /// `SnowballProgram.length`: the current word's length.
    pub fn length(&self) -> i32 {
        self.current.len() as i32
    }

    fn at(&self, i: i32) -> u32 {
        self.current[i as usize] as u32
    }

    fn member(s: &[u8], min: i32, max: i32, ch: u32) -> bool {
        let ch = ch as i32;
        if ch > max || ch < min {
            return false;
        }
        let c = (ch - min) as usize;
        s[c >> 3] & (1 << (c & 7)) != 0
    }

    pub fn in_grouping(&mut self, s: &[u8], min: i32, max: i32) -> bool {
        if self.cursor >= self.limit || !Self::member(s, min, max, self.at(self.cursor)) {
            return false;
        }
        self.cursor += 1;
        true
    }

    pub fn in_grouping_b(&mut self, s: &[u8], min: i32, max: i32) -> bool {
        if self.cursor <= self.limit_backward || !Self::member(s, min, max, self.at(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    pub fn out_grouping(&mut self, s: &[u8], min: i32, max: i32) -> bool {
        if self.cursor >= self.limit || Self::member(s, min, max, self.at(self.cursor)) {
            return false;
        }
        self.cursor += 1;
        true
    }

    pub fn out_grouping_b(&mut self, s: &[u8], min: i32, max: i32) -> bool {
        if self.cursor <= self.limit_backward || Self::member(s, min, max, self.at(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    pub fn eq_s(&mut self, s: &str) -> bool {
        let s: Vec<char> = s.chars().collect();
        let n = s.len() as i32;
        if self.limit - self.cursor < n {
            return false;
        }
        for (i, c) in s.iter().enumerate() {
            if self.current[self.cursor as usize + i] != *c {
                return false;
            }
        }
        self.cursor += n;
        true
    }

    pub fn eq_s_b(&mut self, s: &str) -> bool {
        let s: Vec<char> = s.chars().collect();
        let n = s.len() as i32;
        if self.cursor - self.limit_backward < n {
            return false;
        }
        for (i, c) in s.iter().enumerate() {
            if self.current[(self.cursor - n) as usize + i] != *c {
                return false;
            }
        }
        self.cursor -= n;
        true
    }

    pub fn find_among(&mut self, v: &[Among]) -> i32 {
        let (mut i, mut j) = (0i32, v.len() as i32);
        let (c, l) = (self.cursor, self.limit);
        let (mut common_i, mut common_j) = (0i32, 0i32);
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            let w = &v[k as usize];
            let mut i2 = common;
            while i2 < w.s.len() as i32 {
                if c + common == l {
                    diff = -1;
                    break;
                }
                diff = self.at(c + common) as i32 - w.s[i2 as usize] as i32;
                if diff != 0 {
                    break;
                }
                common += 1;
                i2 += 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[i as usize];
            if common_i >= w.s.len() as i32 {
                self.cursor = c + w.s.len() as i32;
                return w.result;
            }
            i = w.substring_i;
            if i < 0 {
                return 0;
            }
        }
    }

    pub fn find_among_b(&mut self, v: &[Among]) -> i32 {
        let (mut i, mut j) = (0i32, v.len() as i32);
        let (c, lb) = (self.cursor, self.limit_backward);
        let (mut common_i, mut common_j) = (0i32, 0i32);
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            let w = &v[k as usize];
            let mut i2 = w.s.len() as i32 - 1 - common;
            while i2 >= 0 {
                if c - common == lb {
                    diff = -1;
                    break;
                }
                diff = self.at(c - 1 - common) as i32 - w.s[i2 as usize] as i32;
                if diff != 0 {
                    break;
                }
                common += 1;
                i2 -= 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[i as usize];
            if common_i >= w.s.len() as i32 {
                self.cursor = c - w.s.len() as i32;
                return w.result;
            }
            i = w.substring_i;
            if i < 0 {
                return 0;
            }
        }
    }

    fn replace_s(&mut self, c_bra: i32, c_ket: i32, s: &str) -> i32 {
        let s: Vec<char> = s.chars().collect();
        let adjustment = s.len() as i32 - (c_ket - c_bra);
        self.current.splice(c_bra as usize..c_ket as usize, s);
        self.limit += adjustment;
        if self.cursor >= c_ket {
            self.cursor += adjustment;
        } else if self.cursor > c_bra {
            self.cursor = c_bra;
        }
        adjustment
    }

    pub fn slice_check(&self) {
        debug_assert!(self.bra >= 0 && self.bra <= self.ket && self.ket <= self.limit && self.limit as usize <= self.current.len());
    }

    pub fn slice_from(&mut self, s: &str) {
        self.slice_check();
        let (b, k) = (self.bra, self.ket);
        self.replace_s(b, k, s);
    }

    pub fn slice_del(&mut self) {
        self.slice_from("");
    }

    pub fn insert(&mut self, c_bra: i32, c_ket: i32, s: &str) {
        let adjustment = self.replace_s(c_bra, c_ket, s);
        if c_bra <= self.bra {
            self.bra += adjustment;
        }
        if c_bra <= self.ket {
            self.ket += adjustment;
        }
    }
}
