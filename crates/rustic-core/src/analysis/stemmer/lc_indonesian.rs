// Translated from Lucene 10.3.2 `IndonesianStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};

const REMOVED_KE: i32 = 1;
const REMOVED_PENG: i32 = 2;
const REMOVED_DI: i32 = 4;
const REMOVED_MENG: i32 = 8;
const REMOVED_TER: i32 = 16;
const REMOVED_BER: i32 = 32;
const REMOVED_PE: i32 = 64;

#[derive(Default)]
pub struct IndonesianStemmer {
    num_syllables: i32,
    flags: i32,
}

impl IndonesianStemmer {
    pub fn stem(&mut self, text: &mut Vec<char>, mut length: i32, mut stem_derivational: bool) -> i32 {
        self.flags = 0;
        self.num_syllables = 0;
        let mut i: i32 = 0;
        while i < length {
            if self.is_vowel(text[(i) as usize]) {
                self.num_syllables += 1;
            }
            i += 1;
        }
        if self.num_syllables > 2 {
            length = self.remove_particle(text, length);
        }
        if self.num_syllables > 2 {
            length = self.remove_possessive_pronoun(text, length);
        }
        if stem_derivational {
            length = self.stem_derivational(text, length);
        }
        return length;
    }
    pub fn stem_derivational(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        let mut old_length: i32 = length;
        if self.num_syllables > 2 {
            length = self.remove_first_order_prefix(text, length);
        }
        if old_length != length {
            old_length = length;
            if self.num_syllables > 2 {
                length = self.remove_suffix(text, length);
            }
            if old_length != length {
                if self.num_syllables > 2 {
                    length = self.remove_second_order_prefix(text, length);
                }
            }
        } else {
            if self.num_syllables > 2 {
                length = self.remove_second_order_prefix(text, length);
            }
            if self.num_syllables > 2 {
                length = self.remove_suffix(text, length);
            }
        }
        return length;
    }
    pub fn is_vowel(&mut self, mut ch: char) -> bool {
        match ch {
            'a' | 'e' | 'i' | 'o' | 'u' => {
                return true;
            }
            _ => {
                return false;
            }
        }
    }
    pub fn remove_particle(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        if ends_with(text, length, "kah") || ends_with(text, length, "lah") || ends_with(text, length, "pun") {
            self.num_syllables -= 1;
            return length - 3;
        }
        return length;
    }
    pub fn remove_possessive_pronoun(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        if ends_with(text, length, "ku") || ends_with(text, length, "mu") {
            self.num_syllables -= 1;
            return length - 2;
        }
        if ends_with(text, length, "nya") {
            self.num_syllables -= 1;
            return length - 3;
        }
        return length;
    }
    pub fn remove_first_order_prefix(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        if starts_with(text, length, "meng") {
            self.flags |= REMOVED_MENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 4);
        }
        if starts_with(text, length, "meny") && length > 4 && self.is_vowel(text[(4) as usize]) {
            self.flags |= REMOVED_MENG;
            text[(3) as usize] = 's';
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "men") {
            self.flags |= REMOVED_MENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "mem") {
            self.flags |= REMOVED_MENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "me") {
            self.flags |= REMOVED_MENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        if starts_with(text, length, "peng") {
            self.flags |= REMOVED_PENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 4);
        }
        if starts_with(text, length, "peny") && length > 4 && self.is_vowel(text[(4) as usize]) {
            self.flags |= REMOVED_PENG;
            text[(3) as usize] = 's';
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "peny") {
            self.flags |= REMOVED_PENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 4);
        }
        if starts_with(text, length, "pen") && length > 3 && self.is_vowel(text[(3) as usize]) {
            self.flags |= REMOVED_PENG;
            text[(2) as usize] = 't';
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        if starts_with(text, length, "pen") {
            self.flags |= REMOVED_PENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "pem") {
            self.flags |= REMOVED_PENG;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "di") {
            self.flags |= REMOVED_DI;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        if starts_with(text, length, "ter") {
            self.flags |= REMOVED_TER;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "ke") {
            self.flags |= REMOVED_KE;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        return length;
    }
    pub fn remove_second_order_prefix(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        if starts_with(text, length, "ber") {
            self.flags |= REMOVED_BER;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if length == 7 && starts_with(text, length, "belajar") {
            self.flags |= REMOVED_BER;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "be") && length > 4 && ! self.is_vowel(text[(2) as usize]) && text[(3) as usize] == 'e' && text[(4) as usize] == 'r' {
            self.flags |= REMOVED_BER;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        if starts_with(text, length, "per") {
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if length == 7 && starts_with(text, length, "pelajar") {
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 3);
        }
        if starts_with(text, length, "pe") {
            self.flags |= REMOVED_PE;
            self.num_syllables -= 1;
            return delete_n(text, 0, length, 2);
        }
        return length;
    }
    pub fn remove_suffix(&mut self, text: &mut Vec<char>, mut length: i32) -> i32 {
        if ends_with(text, length, "kan") && (self.flags & REMOVED_KE) == 0 && (self.flags & REMOVED_PENG) == 0 && (self.flags & REMOVED_PE) == 0 {
            self.num_syllables -= 1;
            return length - 3;
        }
        if ends_with(text, length, "an") && (self.flags & REMOVED_DI) == 0 && (self.flags & REMOVED_MENG) == 0 && (self.flags & REMOVED_TER) == 0 {
            self.num_syllables -= 1;
            return length - 2;
        }
        if ends_with(text, length, "i") && ! ends_with(text, length, "si") && (self.flags & REMOVED_BER) == 0 && (self.flags & REMOVED_KE) == 0 && (self.flags & REMOVED_PENG) == 0 {
            self.num_syllables -= 1;
            return length - 1;
        }
        return length;
    }
}
