// Translated from Lucene 10.3.2 `CzechStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};


#[derive(Default)]
pub struct CzechStemmer {
}

impl CzechStemmer {
    pub fn stem(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        len = self.remove_case(s, len);
        len = self.remove_possessives(s, len);
        if len > 0 {
            len = self.normalize(s, len);
        }
        return len;
    }
    pub fn remove_case(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 7 && ends_with(s, len, "atech") {
            return len - 5;
        }
        if len > 6 && (ends_with(s, len, "ětem") || ends_with(s, len, "etem") || ends_with(s, len, "atům")) {
            return len - 4;
        }
        if len > 5 && (ends_with(s, len, "ech") || ends_with(s, len, "ich") || ends_with(s, len, "ích") || ends_with(s, len, "ého") || ends_with(s, len, "ěmi") || ends_with(s, len, "emi") || ends_with(s, len, "ému") || ends_with(s, len, "ěte") || ends_with(s, len, "ete") || ends_with(s, len, "ěti") || ends_with(s, len, "eti") || ends_with(s, len, "ího") || ends_with(s, len, "iho") || ends_with(s, len, "ími") || ends_with(s, len, "ímu") || ends_with(s, len, "imu") || ends_with(s, len, "ách") || ends_with(s, len, "ata") || ends_with(s, len, "aty") || ends_with(s, len, "ých") || ends_with(s, len, "ama") || ends_with(s, len, "ami") || ends_with(s, len, "ové") || ends_with(s, len, "ovi") || ends_with(s, len, "ými")) {
            return len - 3;
        }
        if len > 4 && (ends_with(s, len, "em") || ends_with(s, len, "es") || ends_with(s, len, "ém") || ends_with(s, len, "ím") || ends_with(s, len, "ům") || ends_with(s, len, "at") || ends_with(s, len, "ám") || ends_with(s, len, "os") || ends_with(s, len, "us") || ends_with(s, len, "ým") || ends_with(s, len, "mi") || ends_with(s, len, "ou")) {
            return len - 2;
        }
        if len > 3 {
            match s[(len - 1) as usize] {
                'a' | 'e' | 'i' | 'o' | 'u' | 'ů' | 'y' | 'á' | 'é' | 'í' | 'ý' | 'ě' => {
                    return len - 1;
                }
                _ => {}
            }
        }
        return len;
    }
    pub fn remove_possessives(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 5 && (ends_with(s, len, "ov") || ends_with(s, len, "in") || ends_with(s, len, "ův")) {
            return len - 2;
        }
        return len;
    }
    pub fn normalize(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if ends_with(s, len, "čt") {
            s[(len - 2) as usize] = 'c';
            s[(len - 1) as usize] = 'k';
            return len;
        }
        if ends_with(s, len, "št") {
            s[(len - 2) as usize] = 's';
            s[(len - 1) as usize] = 'k';
            return len;
        }
        match s[(len - 1) as usize] {
            'c' | 'č' => {
                s[(len - 1) as usize] = 'k';
                return len;
            }
            'z' | 'ž' => {
                s[(len - 1) as usize] = 'h';
                return len;
            }
            _ => {}
        }
        if len > 1 && s[(len - 2) as usize] == 'e' {
            s[(len - 2) as usize] = s[(len - 1) as usize];
            return len - 1;
        }
        if len > 2 && s[(len - 2) as usize] == 'ů' {
            s[(len - 2) as usize] = 'o';
            return len;
        }
        return len;
    }
}
