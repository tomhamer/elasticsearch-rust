// Translated from Lucene 10.3.2 `BengaliStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};


#[derive(Default)]
pub struct BengaliStemmer {
}

impl BengaliStemmer {
    pub fn stem(&mut self, buffer: &mut Vec<char>, mut len: i32) -> i32 {
        if len > 9 && (ends_with(buffer, len, "িয়াছিলাম") || ends_with(buffer, len, "িতেছিলাম") || ends_with(buffer, len, "িতেছিলেন") || ends_with(buffer, len, "ইতেছিলেন") || ends_with(buffer, len, "িয়াছিলেন") || ends_with(buffer, len, "ইয়াছিলেন")) {
            return len - 8;
        }
        if (len > 8) && (ends_with(buffer, len, "িতেছিলি") || ends_with(buffer, len, "িতেছিলে") || ends_with(buffer, len, "িয়াছিলা") || ends_with(buffer, len, "িয়াছিলে") || ends_with(buffer, len, "িতেছিলা") || ends_with(buffer, len, "িয়াছিলি") || ends_with(buffer, len, "য়েদেরকে")) {
            return len - 7;
        }
        if (len > 7) && (ends_with(buffer, len, "িতেছিস") || ends_with(buffer, len, "িতেছেন") || ends_with(buffer, len, "িয়াছিস") || ends_with(buffer, len, "িয়াছেন") || ends_with(buffer, len, "েছিলাম") || ends_with(buffer, len, "েছিলেন") || ends_with(buffer, len, "েদেরকে")) {
            return len - 6;
        }
        if (len > 6) && (ends_with(buffer, len, "িতেছি") || ends_with(buffer, len, "িতেছা") || ends_with(buffer, len, "িতেছে") || ends_with(buffer, len, "ছিলাম") || ends_with(buffer, len, "ছিলেন") || ends_with(buffer, len, "িয়াছি") || ends_with(buffer, len, "িয়াছা") || ends_with(buffer, len, "িয়াছে") || ends_with(buffer, len, "েছিলে") || ends_with(buffer, len, "েছিলা") || ends_with(buffer, len, "য়েদের") || ends_with(buffer, len, "দেরকে")) {
            return len - 5;
        }
        if (len > 5) && (ends_with(buffer, len, "িলাম") || ends_with(buffer, len, "িলেন") || ends_with(buffer, len, "িতাম") || ends_with(buffer, len, "িতেন") || ends_with(buffer, len, "িবেন") || ends_with(buffer, len, "ছিলি") || ends_with(buffer, len, "ছিলে") || ends_with(buffer, len, "ছিলা") || ends_with(buffer, len, "তেছে") || ends_with(buffer, len, "িতেছ") || ends_with(buffer, len, "খানা") || ends_with(buffer, len, "খানি") || ends_with(buffer, len, "গুলো") || ends_with(buffer, len, "গুলি") || ends_with(buffer, len, "য়েরা") || ends_with(buffer, len, "েদের")) {
            return len - 4;
        }
        if (len > 4) && (ends_with(buffer, len, "লাম") || ends_with(buffer, len, "িলি") || ends_with(buffer, len, "ইলি") || ends_with(buffer, len, "িলে") || ends_with(buffer, len, "ইলে") || ends_with(buffer, len, "লেন") || ends_with(buffer, len, "িলা") || ends_with(buffer, len, "ইলা") || ends_with(buffer, len, "তাম") || ends_with(buffer, len, "িতি") || ends_with(buffer, len, "ইতি") || ends_with(buffer, len, "িতে") || ends_with(buffer, len, "ইতে") || ends_with(buffer, len, "তেন") || ends_with(buffer, len, "িতা") || ends_with(buffer, len, "িবা") || ends_with(buffer, len, "ইবা") || ends_with(buffer, len, "িবি") || ends_with(buffer, len, "ইবি") || ends_with(buffer, len, "বেন") || ends_with(buffer, len, "িবে") || ends_with(buffer, len, "ইবে") || ends_with(buffer, len, "ছেন") || ends_with(buffer, len, "য়োন") || ends_with(buffer, len, "য়ের") || ends_with(buffer, len, "েরা") || ends_with(buffer, len, "দের")) {
            return len - 3;
        }
        if (len > 3) && (ends_with(buffer, len, "িস") || ends_with(buffer, len, "েন") || ends_with(buffer, len, "লি") || ends_with(buffer, len, "লে") || ends_with(buffer, len, "লা") || ends_with(buffer, len, "তি") || ends_with(buffer, len, "তে") || ends_with(buffer, len, "তা") || ends_with(buffer, len, "বি") || ends_with(buffer, len, "বে") || ends_with(buffer, len, "বা") || ends_with(buffer, len, "ছি") || ends_with(buffer, len, "ছা") || ends_with(buffer, len, "ছে") || ends_with(buffer, len, "ুন") || ends_with(buffer, len, "ুক") || ends_with(buffer, len, "টা") || ends_with(buffer, len, "টি") || ends_with(buffer, len, "নি") || ends_with(buffer, len, "ের") || ends_with(buffer, len, "তে") || ends_with(buffer, len, "রা") || ends_with(buffer, len, "কে")) {
            return len - 2;
        }
        if (len > 2) && (ends_with(buffer, len, "ি") || ends_with(buffer, len, "ী") || ends_with(buffer, len, "া") || ends_with(buffer, len, "ো") || ends_with(buffer, len, "ে") || ends_with(buffer, len, "ব") || ends_with(buffer, len, "ত")) {
            return len - 1;
        }
        return len;
    }
}
