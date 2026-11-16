// Translated from Lucene 10.3.2 `SoraniStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};


#[derive(Default)]
pub struct SoraniStemmer {
}

impl SoraniStemmer {
    pub fn stem(&mut self, s: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 5 && ends_with(s, len, "دا") {
            len -= 2;
        } else if len > 4 && ends_with(s, len, "نا") {
            len -= 1;
        } else if len > 6 && ends_with(s, len, "ەوە") {
            len -= 3;
        }
        if len > 6 && (ends_with(s, len, "مان") || ends_with(s, len, "یان") || ends_with(s, len, "تان")) {
            len -= 3;
        }
        if len > 6 && ends_with(s, len, "ێکی") {
            return len - 3;
        } else if len > 7 && ends_with(s, len, "یەکی") {
            return len - 4;
        }
        if len > 5 && ends_with(s, len, "ێک") {
            return len - 2;
        } else if len > 6 && ends_with(s, len, "یەک") {
            return len - 3;
        } else if len > 6 && ends_with(s, len, "ەکە") {
            return len - 3;
        } else if len > 5 && ends_with(s, len, "کە") {
            return len - 2;
        } else if len > 7 && ends_with(s, len, "ەکان") {
            return len - 4;
        } else if len > 6 && ends_with(s, len, "کان") {
            return len - 3;
        } else if len > 7 && ends_with(s, len, "یانی") {
            return len - 4;
        } else if len > 6 && ends_with(s, len, "انی") {
            return len - 3;
        } else if len > 6 && ends_with(s, len, "یان") {
            return len - 3;
        } else if len > 5 && ends_with(s, len, "ان") {
            return len - 2;
        } else if len > 7 && ends_with(s, len, "یانە") {
            return len - 4;
        } else if len > 6 && ends_with(s, len, "انە") {
            return len - 3;
        } else if len > 5 && (ends_with(s, len, "ایە") || ends_with(s, len, "ەیە")) {
            return len - 2;
        } else if len > 4 && ends_with(s, len, "ە") {
            return len - 1;
        } else if len > 4 && ends_with(s, len, "ی") {
            return len - 1;
        }
        return len;
    }
}
