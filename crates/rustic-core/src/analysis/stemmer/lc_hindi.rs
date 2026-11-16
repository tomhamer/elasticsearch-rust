// Translated from Lucene 10.3.2 `HindiStemmer` by tools/lucene_stemmer2rs.py.
#![allow(unused, unused_parens, unused_mut, non_snake_case, clippy::all)]
use super::lucene::{contains, delete_n, ends_with, starts_with};


#[derive(Default)]
pub struct HindiStemmer {
}

impl HindiStemmer {
    pub fn stem(&mut self, buffer: &mut Vec<char>, mut len: i32) -> i32 {
        if (len > 6) && (ends_with(buffer, len, "ाएंगी") || ends_with(buffer, len, "ाएंगे") || ends_with(buffer, len, "ाऊंगी") || ends_with(buffer, len, "ाऊंगा") || ends_with(buffer, len, "ाइयाँ") || ends_with(buffer, len, "ाइयों") || ends_with(buffer, len, "ाइयां")) {
            return len - 5;
        }
        if (len > 5) && (ends_with(buffer, len, "ाएगी") || ends_with(buffer, len, "ाएगा") || ends_with(buffer, len, "ाओगी") || ends_with(buffer, len, "ाओगे") || ends_with(buffer, len, "एंगी") || ends_with(buffer, len, "ेंगी") || ends_with(buffer, len, "एंगे") || ends_with(buffer, len, "ेंगे") || ends_with(buffer, len, "ूंगी") || ends_with(buffer, len, "ूंगा") || ends_with(buffer, len, "ातीं") || ends_with(buffer, len, "नाओं") || ends_with(buffer, len, "नाएं") || ends_with(buffer, len, "ताओं") || ends_with(buffer, len, "ताएं") || ends_with(buffer, len, "ियाँ") || ends_with(buffer, len, "ियों") || ends_with(buffer, len, "ियां")) {
            return len - 4;
        }
        if (len > 4) && (ends_with(buffer, len, "ाकर") || ends_with(buffer, len, "ाइए") || ends_with(buffer, len, "ाईं") || ends_with(buffer, len, "ाया") || ends_with(buffer, len, "ेगी") || ends_with(buffer, len, "ेगा") || ends_with(buffer, len, "ोगी") || ends_with(buffer, len, "ोगे") || ends_with(buffer, len, "ाने") || ends_with(buffer, len, "ाना") || ends_with(buffer, len, "ाते") || ends_with(buffer, len, "ाती") || ends_with(buffer, len, "ाता") || ends_with(buffer, len, "तीं") || ends_with(buffer, len, "ाओं") || ends_with(buffer, len, "ाएं") || ends_with(buffer, len, "ुओं") || ends_with(buffer, len, "ुएं") || ends_with(buffer, len, "ुआं")) {
            return len - 3;
        }
        if (len > 3) && (ends_with(buffer, len, "कर") || ends_with(buffer, len, "ाओ") || ends_with(buffer, len, "िए") || ends_with(buffer, len, "ाई") || ends_with(buffer, len, "ाए") || ends_with(buffer, len, "ने") || ends_with(buffer, len, "नी") || ends_with(buffer, len, "ना") || ends_with(buffer, len, "ते") || ends_with(buffer, len, "ीं") || ends_with(buffer, len, "ती") || ends_with(buffer, len, "ता") || ends_with(buffer, len, "ाँ") || ends_with(buffer, len, "ां") || ends_with(buffer, len, "ों") || ends_with(buffer, len, "ें")) {
            return len - 2;
        }
        if (len > 2) && (ends_with(buffer, len, "ो") || ends_with(buffer, len, "े") || ends_with(buffer, len, "ू") || ends_with(buffer, len, "ु") || ends_with(buffer, len, "ी") || ends_with(buffer, len, "ि") || ends_with(buffer, len, "ा")) {
            return len - 1;
        }
        return len;
    }
}
