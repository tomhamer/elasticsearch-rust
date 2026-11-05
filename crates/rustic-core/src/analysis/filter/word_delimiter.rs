//! Word delimiter filter (Lucene `WordDelimiterFilter` /
//! `WordDelimiterGraphFilter`).
//!
//! Splits tokens into sub-words on intra-word delimiters (non
//! alphanumerics), case transitions and letter/digit transitions, and
//! optionally concatenates the parts back together:
//!
//! * `PowerShot` → `Power`, `Shot`
//! * `SD500` → `SD`, `500`
//! * `wi-fi` → `wi`, `fi` (+ `wifi` with `catenate_words`)
//! * `O'Neil's` → `O`, `Neil` (with `stem_english_possessive`)
//!
//! # Positions
//!
//! Sub-words after the first are placed at consecutive positions, which
//! shifts the positions of all later tokens (legacy non-graph behaviour).
//! Concatenated tokens and the preserved original sit at the position of
//! the first part. Both filters produce the same token texts, and
//! [`WordDelimiterGraphFilter`] is an alias here.

use std::collections::HashSet;

use crate::analysis::{Token, TokenFilter};

const LOWER: u8 = 1;
const UPPER: u8 = 2;
const DIGIT: u8 = 4;
const DELIM: u8 = 8;
const ALPHA: u8 = LOWER | UPPER;

fn char_type(c: char) -> u8 {
    if c.is_lowercase() {
        LOWER
    } else if c.is_uppercase() {
        UPPER
    } else if c.is_numeric() {
        DIGIT
    } else if c.is_alphabetic() {
        // Caseless letters (CJK, Arabic, ...): treat as lowercase alpha.
        LOWER
    } else {
        DELIM
    }
}

/// Word delimiter filter; see the module docs. Field names follow the
/// ES options.
#[derive(Clone, Debug)]
pub struct WordDelimiterFilter {
    /// Emit alphabetic sub-words (default true).
    pub generate_word_parts: bool,
    /// Emit numeric sub-words (default true).
    pub generate_number_parts: bool,
    /// Emit runs of alphabetic parts concatenated (`wi-fi` → `wifi`).
    pub catenate_words: bool,
    /// Emit runs of numeric parts concatenated (`500-42` → `50042`).
    pub catenate_numbers: bool,
    /// Emit all parts concatenated (`wi-fi-4000` → `wifi4000`).
    pub catenate_all: bool,
    /// Split on lower→upper transitions (`PowerShot`); default true.
    pub split_on_case_change: bool,
    /// Split on letter↔digit transitions (`SD500`); default true.
    pub split_on_numerics: bool,
    /// Also emit the original token (default false).
    pub preserve_original: bool,
    /// Strip a trailing `'s` from parts (default true).
    pub stem_english_possessive: bool,
    /// Tokens that must not be split.
    pub protected_words: HashSet<String>,
}

impl Default for WordDelimiterFilter {
    fn default() -> Self {
        Self {
            generate_word_parts: true,
            generate_number_parts: true,
            catenate_words: false,
            catenate_numbers: false,
            catenate_all: false,
            split_on_case_change: true,
            split_on_numerics: true,
            preserve_original: false,
            stem_english_possessive: true,
            protected_words: HashSet::new(),
        }
    }
}

/// ES `word_delimiter_graph`: same output texts as
/// [`WordDelimiterFilter`] (see module docs for positions).
pub type WordDelimiterGraphFilter = WordDelimiterFilter;

/// One sub-word: char range and type mask.
struct Part {
    start: usize,
    end: usize,
    ty: u8,
}

impl WordDelimiterFilter {
    /// Lucene's `WordDelimiterIterator.isBreak`.
    fn is_break(&self, last: u8, ty: u8) -> bool {
        if last & ty != 0 {
            return false;
        }
        if !self.split_on_case_change && (last & ALPHA != 0) && (ty & ALPHA != 0) {
            return false;
        }
        if last & UPPER != 0 && ty & ALPHA != 0 {
            // UPPER -> letter (e.g. "SHOT" run, or "Shot" upper->lower): no split.
            return false;
        }
        if !self.split_on_numerics
            && ((last & ALPHA != 0 && ty & DIGIT != 0) || (last & DIGIT != 0 && ty & ALPHA != 0))
        {
            return false;
        }
        true
    }

    fn split(&self, chars: &[char]) -> Vec<Part> {
        let mut parts = Vec::new();
        let mut i = 0;
        let n = chars.len();
        while i < n {
            // Skip delimiters.
            if char_type(chars[i]) == DELIM {
                i += 1;
                continue;
            }
            let start = i;
            let mut ty = char_type(chars[i]);
            let mut last = ty;
            i += 1;
            while i < n {
                let t = char_type(chars[i]);
                if t == DELIM || self.is_break(last, t) {
                    break;
                }
                ty |= t;
                last = t;
                i += 1;
            }
            let mut end = i;
            // English possessive: strip trailing 's / ’s.
            if self.stem_english_possessive && end - start >= 2 {
                let apos = chars[end - 2];
                if (apos == '\'' || apos == '\u{2019}') && matches!(chars[end - 1], 's' | 'S') {
                    end -= 2;
                }
            }
            // The possessive marker itself is a delimiter so the loop
            // above never contains it; handle "Neil's" where "'" broke
            // the run: drop a lone trailing "s" part preceded by an apostrophe.
            if self.stem_english_possessive
                && end - start == 1
                && matches!(chars[start], 's' | 'S')
                && start >= 1
                && (chars[start - 1] == '\'' || chars[start - 1] == '\u{2019}')
                && (i >= n || char_type(chars[i]) == DELIM)
            {
                continue;
            }
            if end > start {
                parts.push(Part { start, end, ty });
            }
        }
        parts
    }

    fn should_generate(&self, ty: u8) -> bool {
        (self.generate_word_parts && ty & ALPHA != 0) || (self.generate_number_parts && ty & DIGIT != 0)
    }
}

impl TokenFilter for WordDelimiterFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut shift: u32 = 0;
        for t in tokens {
            let base = t.position + shift;
            if self.protected_words.contains(&t.text) {
                out.push(Token { position: base, ..t });
                continue;
            }
            let chars: Vec<char> = t.text.chars().collect();
            let parts = self.split(&chars);
            // Single word with no delimiters: pass through unchanged.
            if parts.len() == 1 && parts[0].start == 0 && parts[0].end == chars.len() {
                out.push(Token { position: base, ..t });
                continue;
            }
            if parts.is_empty() {
                if self.preserve_original {
                    out.push(Token { position: base, ..t });
                }
                continue;
            }
            let byte_off = |ci: usize| -> u32 {
                t.offset_from + chars[..ci].iter().map(|c| c.len_utf8() as u32).sum::<u32>()
            };
            let text_of = |p: &Part| -> String { chars[p.start..p.end].iter().collect() };

            let mut emitted: Vec<Token> = Vec::new();
            let mut seen: HashSet<(u32, String)> = HashSet::new();
            let mut push = |tok: Token, emitted: &mut Vec<Token>| {
                if seen.insert((tok.position, tok.text.clone())) {
                    emitted.push(tok);
                }
            };

            if self.preserve_original {
                push(Token { position: base, ..t.clone() }, &mut emitted);
            }
            // Concatenations.
            if self.catenate_all && parts.len() > 1 {
                let text: String = parts.iter().map(text_of).collect();
                push(
                    Token { text, position: base, offset_from: byte_off(parts[0].start), offset_to: byte_off(parts.last().unwrap().end) },
                    &mut emitted,
                );
            }
            for (flag, mask) in [(self.catenate_words, ALPHA), (self.catenate_numbers, DIGIT)] {
                if !flag {
                    continue;
                }
                let mut i = 0;
                while i < parts.len() {
                    if parts[i].ty & mask == 0 {
                        i += 1;
                        continue;
                    }
                    let run_start = i;
                    while i < parts.len() && parts[i].ty & mask != 0 {
                        i += 1;
                    }
                    let run = &parts[run_start..i];
                    let run_ty = run.iter().fold(0u8, |a, p| a | p.ty);
                    if run.len() > 1 || !self.should_generate(run_ty) {
                        let text: String = run.iter().map(text_of).collect();
                        push(
                            Token {
                                text,
                                position: base + run_start as u32,
                                offset_from: byte_off(run[0].start),
                                offset_to: byte_off(run[run.len() - 1].end),
                            },
                            &mut emitted,
                        );
                    }
                }
            }
            // Individual parts.
            let mut next_pos = 0u32;
            for p in &parts {
                if self.should_generate(p.ty) {
                    push(
                        Token { text: text_of(p), position: base + next_pos, offset_from: byte_off(p.start), offset_to: byte_off(p.end) },
                        &mut emitted,
                    );
                    next_pos += 1;
                }
            }
            if next_pos > 1 {
                shift += next_pos - 1;
            }
            emitted.sort_by_key(|t| t.position);
            out.extend(emitted);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, pos: u32) -> Token {
        Token { text: text.into(), position: pos, offset_from: 0, offset_to: text.len() as u32 }
    }
    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }
    fn positions(t: &[Token]) -> Vec<u32> {
        t.iter().map(|t| t.position).collect()
    }
    fn run(f: &WordDelimiterFilter, s: &str) -> Vec<Token> {
        f.filter(vec![tok(s, 0)])
    }

    #[test]
    fn case_change() {
        let f = WordDelimiterFilter::default();
        assert_eq!(texts(&run(&f, "PowerShot")), vec!["Power", "Shot"]);
        assert_eq!(texts(&run(&f, "SHOTgun")), vec!["SHOTgun"]);
        let f = WordDelimiterFilter { split_on_case_change: false, ..Default::default() };
        assert_eq!(texts(&run(&f, "PowerShot")), vec!["PowerShot"]);
    }

    #[test]
    fn numerics() {
        let f = WordDelimiterFilter::default();
        let out = run(&f, "SD500");
        assert_eq!(texts(&out), vec!["SD", "500"]);
        assert_eq!(positions(&out), vec![0, 1]);
        assert_eq!((out[1].offset_from, out[1].offset_to), (2, 5));
        let f = WordDelimiterFilter { split_on_numerics: false, ..Default::default() };
        assert_eq!(texts(&run(&f, "SD500")), vec!["SD500"]);
    }

    #[test]
    fn delimiters_and_catenate_words() {
        let f = WordDelimiterFilter::default();
        assert_eq!(texts(&run(&f, "wi-fi")), vec!["wi", "fi"]);
        let f = WordDelimiterFilter { catenate_words: true, ..Default::default() };
        let out = run(&f, "wi-fi");
        assert_eq!(texts(&out), vec!["wifi", "wi", "fi"]);
        assert_eq!(positions(&out), vec![0, 0, 1]);
        assert_eq!((out[0].offset_from, out[0].offset_to), (0, 5));
    }

    #[test]
    fn catenate_numbers_and_all() {
        let f = WordDelimiterFilter { catenate_numbers: true, ..Default::default() };
        assert_eq!(texts(&run(&f, "500-42")), vec!["50042", "500", "42"]);
        let f = WordDelimiterFilter { catenate_all: true, ..Default::default() };
        assert_eq!(texts(&run(&f, "wi-fi-4000")), vec!["wifi4000", "wi", "fi", "4000"]);
        // catenate_words does not cross a numeric part
        let f = WordDelimiterFilter { catenate_words: true, ..Default::default() };
        assert_eq!(texts(&run(&f, "wi-fi-4000-net")), vec!["wifi", "wi", "fi", "4000", "net"]);
    }

    #[test]
    fn es_doc_example() {
        let f = WordDelimiterFilter::default();
        assert_eq!(
            texts(&run(&f, "Neil's-Super-Duper-XL500--42+AutoCoder")),
            vec!["Neil", "Super", "Duper", "XL", "500", "42", "Auto", "Coder"]
        );
    }

    #[test]
    fn possessive() {
        let f = WordDelimiterFilter::default();
        assert_eq!(texts(&run(&f, "O'Neil's")), vec!["O", "Neil"]);
        assert_eq!(texts(&run(&f, "Bob’s")), vec!["Bob"]);
        let f = WordDelimiterFilter { stem_english_possessive: false, ..Default::default() };
        assert_eq!(texts(&run(&f, "Bob's")), vec!["Bob", "s"]);
    }

    #[test]
    fn preserve_original_and_protected() {
        let f = WordDelimiterFilter { preserve_original: true, ..Default::default() };
        let out = run(&f, "wi-fi");
        assert_eq!(texts(&out), vec!["wi-fi", "wi", "fi"]);
        assert_eq!(positions(&out), vec![0, 0, 1]);
        let mut f = WordDelimiterFilter::default();
        f.protected_words.insert("wi-fi".into());
        assert_eq!(texts(&run(&f, "wi-fi")), vec!["wi-fi"]);
    }

    #[test]
    fn generate_parts_flags() {
        let f = WordDelimiterFilter { generate_number_parts: false, ..Default::default() };
        assert_eq!(texts(&run(&f, "SD500-x")), vec!["SD", "x"]);
        let f = WordDelimiterFilter { generate_word_parts: false, catenate_words: true, ..Default::default() };
        // Single-part word runs are still catenated when parts aren't generated.
        assert_eq!(texts(&run(&f, "wi-fi-500")), vec!["wifi", "500"]);
        assert_eq!(texts(&run(&f, "ab-5")), vec!["ab", "5"]);
    }

    #[test]
    fn plain_tokens_pass_through_and_shift_later_positions() {
        let f = WordDelimiterFilter::default();
        let out = f.filter(vec![tok("hello", 0), tok("wi-fi", 1), tok("world", 2)]);
        assert_eq!(texts(&out), vec!["hello", "wi", "fi", "world"]);
        assert_eq!(positions(&out), vec![0, 1, 2, 3]);
    }

    #[test]
    fn only_delimiters() {
        let f = WordDelimiterFilter::default();
        assert!(run(&f, "---").is_empty());
        let f = WordDelimiterFilter { preserve_original: true, ..Default::default() };
        assert_eq!(texts(&run(&f, "---")), vec!["---"]);
    }

    #[test]
    fn unicode_offsets() {
        let f = WordDelimiterFilter::default();
        let out = run(&f, "été-Noël");
        assert_eq!(texts(&out), vec!["été", "Noël"]);
        assert_eq!((out[1].offset_from, out[1].offset_to), (6, 11));
    }
}
