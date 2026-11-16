// Translated from Lucene 10.3.2 `BulgarianStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};


#[derive(Default)]
pub struct BulgarianStemmer {
}

impl BulgarianStemmer {
    pub fn stem(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len < 4 {
            return len;
        }
        if len > 5 && ends_with(s, len, "ища") {
            return len - 3;
        }
        len = self.remove_article(s, len);
        len = self.remove_plural(s, len);
        if len > 3 {
            if ends_with(s, len, "я") {
                len -= 1;
            }
            if ends_with(s, len, "а") || ends_with(s, len, "о") || ends_with(s, len, "е") {
                len -= 1;
            }
        }
        if len > 4 && ends_with(s, len, "ен") {
            s[(len - 2) as usize] = 'н';
            len -= 1;
        }
        if len > 5 && s[(len - 2) as usize] == 'ъ' {
            s[(len - 2) as usize] = s[(len - 1) as usize];
            len -= 1;
        }
        return len;
    }
    pub fn remove_article(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 6 && ends_with(s, len, "ият") {
            return len - 3;
        }
        if len > 5 {
            if ends_with(s, len, "ът") || ends_with(s, len, "то") || ends_with(s, len, "те") || ends_with(s, len, "та") || ends_with(s, len, "ия") {
                return len - 2;
            }
        }
        if len > 4 && ends_with(s, len, "ят") {
            return len - 2;
        }
        return len;
    }
    pub fn remove_plural(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 6 {
            if ends_with(s, len, "овци") {
                return len - 3;
            }
            if ends_with(s, len, "ове") {
                return len - 3;
            }
            if ends_with(s, len, "еве") {
                s[(len - 3) as usize] = 'й';
                return len - 2;
            }
        }
        if len > 5 {
            if ends_with(s, len, "ища") {
                return len - 3;
            }
            if ends_with(s, len, "та") {
                return len - 2;
            }
            if ends_with(s, len, "ци") {
                s[(len - 2) as usize] = 'к';
                return len - 1;
            }
            if ends_with(s, len, "зи") {
                s[(len - 2) as usize] = 'г';
                return len - 1;
            }
            if s[(len - 3) as usize] == 'е' && s[(len - 1) as usize] == 'и' {
                s[(len - 3) as usize] = 'я';
                return len - 1;
            }
        }
        if len > 4 {
            if ends_with(s, len, "си") {
                s[(len - 2) as usize] = 'х';
                return len - 1;
            }
            if ends_with(s, len, "и") {
                return len - 1;
            }
        }
        return len;
    }
}
