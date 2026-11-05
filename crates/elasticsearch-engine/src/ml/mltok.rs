//! The ML plugin's analysis components: the `ml_classic` and `ml_standard`
//! tokenizers (`MlClassicTokenizer`, `MlStandardTokenizer`) and the
//! `first_line_with_letters` char filter (`FirstLineWithLettersCharFilter`).
//!
//! Java reads UTF-16 code units; these ports read `char`s, which differs
//! only for supplementary-plane characters.

use rustic_core::analysis::{CharFilter, Token, Tokenizer};

/// `Character.isLetterOrDigit`.
fn letter_or_digit(c: char) -> bool {
    c.is_alphabetic() || is_digit(c)
}

/// `Character.isDigit`.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || (!c.is_ascii() && c.is_numeric())
}

/// `Character.digit(c, 16) != -1`.
fn is_hex(c: char) -> bool {
    c.is_ascii_hexdigit() || (!c.is_ascii() && c.is_numeric())
}

/// Resolve a tokenizer registered by the ML plugin.
pub fn tokenizer_by_name(name: &str) -> Option<Box<dyn Tokenizer>> {
    match name {
        "ml_classic" => Some(Box::new(MlTokenizer { standard: false })),
        "ml_standard" => Some(Box::new(MlTokenizer { standard: true })),
        _ => None,
    }
}

/// Resolve a char filter registered by the ML plugin.
pub fn char_filter_by_name(name: &str) -> Option<Box<dyn CharFilter>> {
    match name {
        "first_line_with_letters" => Some(Box::new(FirstLineWithLetters)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MlTokenizer {
    /// `ml_standard` (else `ml_classic`).
    pub standard: bool,
}

impl Tokenizer for MlTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let byte_at = |i: usize| chars.get(i).map(|x| x.0).unwrap_or(text.len()) as u32;
        let mut out = Vec::new();
        let mut i = 0usize;
        let mut pos: i64 = -1;
        loop {
            let mut skipped = 0i64;
            let mut start = 0usize;
            let mut length = 0usize;
            // `MlClassicTokenizer` does not reset this after a skipped candidate.
            let mut have_non_hex = false;
            let mut letters_before_colon = 0usize;
            let mut have_colon = false;
            let mut first_bs: i64 = -1;
            let mut first_fs: i64 = -1;
            let mut slash_count = 0usize;
            while i < chars.len() {
                let c = chars[i].1;
                i += 1;
                let accept = if self.standard {
                    letter_or_digit(c)
                        || (length > 0 && (c == '_' || c == '.' || c == '-' || c == '@' || (c == ':' && letters_before_colon == length)))
                        || c == '/'
                        || (c == '\\' && (length == 0 || (have_colon && letters_before_colon == 1) || first_bs == 0))
                } else {
                    letter_or_digit(c) || (length > 0 && (c == '_' || c == '.' || c == '-'))
                };
                if accept {
                    if length == 0 {
                        start = i - 1;
                    }
                    length += 1;
                    if self.standard {
                        if c == ':' {
                            have_colon = true;
                        } else if c == '/' {
                            slash_count += 1;
                            if first_fs == -1 {
                                first_fs = length as i64 - 1;
                            }
                        } else if c == '\\' {
                            slash_count += 1;
                            if first_bs == -1 {
                                first_bs = length as i64 - 1;
                            }
                        } else if have_colon {
                            let after = letters_before_colon as i64 + 1;
                            if first_bs != after && first_fs != after {
                                // A colon not followed by a slash: drop it and what follows.
                                length -= 2;
                                i -= 1;
                                break;
                            }
                        } else if c.is_alphabetic() {
                            letters_before_colon += 1;
                        }
                        have_non_hex = have_non_hex || (!is_hex(c) && !matches!(c, '.' | '-' | '@' | ':'));
                    } else {
                        have_non_hex = have_non_hex || (!is_hex(c) && !matches!(c, '.' | '-'));
                    }
                } else if length > 0 {
                    let valid = have_non_hex && !is_digit(chars[start].1) && (!self.standard || length > slash_count);
                    if valid {
                        break;
                    }
                    skipped += 1;
                    length = 0;
                    if self.standard {
                        have_non_hex = false;
                        letters_before_colon = 0;
                        have_colon = false;
                        first_bs = -1;
                        first_fs = -1;
                        slash_count = 0;
                    }
                }
            }
            if length == 0 {
                break;
            }
            if !have_non_hex || is_digit(chars[start].1) || (self.standard && length == slash_count) {
                break;
            }
            let strip: &[char] = if self.standard { &['_', '.', '-', '@', ':'] } else { &['_', '.', '-'] };
            while length > 0 && strip.contains(&chars[start + length - 1].1) {
                length -= 1;
            }
            pos += skipped + 1;
            out.push(Token {
                text: chars[start..start + length].iter().map(|x| x.1).collect(),
                position: pos as u32,
                offset_from: byte_at(start),
                offset_to: byte_at(start + length),
            });
        }
        out
    }

    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(*self)
    }
}

/// Keeps only the first line containing a letter; the end of that line
/// maps back to the end of the input.
#[derive(Clone, Copy, Debug)]
pub struct FirstLineWithLetters;

impl CharFilter for FirstLineWithLetters {
    fn filter(&self, text: &str) -> String {
        self.filter_mapped(text).0
    }

    fn filter_mapped(&self, text: &str) -> (String, Vec<(u32, i64)>) {
        let b = text.as_bytes();
        let mut seen_letter = false;
        let mut prev_newline: i64 = -1;
        let mut end: Option<usize> = None;
        for (idx, c) in text.char_indices() {
            if c == '\n' {
                if seen_letter {
                    end = Some(if idx > 0 && b[idx - 1] == b'\r' { idx - 1 } else { idx });
                    break;
                }
                prev_newline = idx as i64;
            } else {
                seen_letter = seen_letter || c.is_alphabetic();
            }
        }
        if !seen_letter {
            return (String::new(), Vec::new());
        }
        let end = match end {
            Some(e) => e,
            None if prev_newline == -1 => return (text.to_string(), Vec::new()),
            None => text.len(),
        };
        let start = (prev_newline + 1) as usize;
        let mut map = vec![(0u32, start as i64)];
        if end < text.len() {
            map.push(((end - start) as u32, (text.len() - end + start) as i64));
        }
        (text[start..end].to_string(), map)
    }

    fn clone_box(&self) -> Box<dyn CharFilter> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(t: &MlTokenizer, s: &str) -> Vec<(String, u32, u32, u32)> {
        t.tokenize(s).into_iter().map(|k| (k.text, k.position, k.offset_from, k.offset_to)).collect()
    }

    const LOG: &str = "[elasticsearch] [2017-12-13T10:46:30,816][INFO ][o.e.c.m.MetadataCreateIndexService] [node-0] [.watcher-history-7-2017.12.13] creating index, cause [auto(bulk api)], templates [.watch-history-7], shards [1]/[1], mappings [doc]";

    #[test]
    fn classic_matches_es() {
        let t = toks(&MlTokenizer { standard: false }, LOG);
        assert_eq!(t[0], ("elasticsearch".into(), 0, 1, 14));
        assert_eq!(t[1], ("INFO".into(), 5, 42, 46));
        assert_eq!(t[4], ("watcher-history-7-2017.12.13".into(), 8, 96, 124));
        assert_eq!(t[14], ("mappings".into(), 20, 212, 220));
        assert_eq!(t[15], ("doc".into(), 21, 222, 225));
    }

    #[test]
    fn standard_matches_es() {
        let t = toks(&MlTokenizer { standard: true }, LOG);
        assert_eq!(t[1], ("INFO".into(), 5, 42, 46));
        assert_eq!(t[13], ("shards".into(), 17, 196, 202));
        assert_eq!(t[14], ("mappings".into(), 21, 212, 220));
        let words = |s: &str| -> Vec<String> { toks(&MlTokenizer { standard: true }, s).into_iter().map(|x| x.0).collect() };
        // A colon not followed by a slash truncates the token before it; the
        // character after the colon is re-read.
        assert_eq!(words("see http://x.org/a and C:\\dir\\f and xyz:word"), ["see", "http://x.org/a", "and", "C:\\dir\\f", "and", "xyz", "word"]);
        // ES quirk: a truncated token that is then invalid (all hex) makes
        // `incrementToken` return false, which ends the stream.
        assert_eq!(words("see abc:def more"), ["see"]);
    }

    #[test]
    fn first_line_with_letters_maps_offsets() {
        let (s, m) = FirstLineWithLetters.filter_mapped("   \nfirst line\nsecond line");
        assert_eq!(s, "first line");
        assert_eq!(m, vec![(0, 4), (10, 16)]);
        assert_eq!(FirstLineWithLetters.filter_mapped("one line").0, "one line");
        assert_eq!(FirstLineWithLetters.filter_mapped("12\n34").0, "");
    }
}
