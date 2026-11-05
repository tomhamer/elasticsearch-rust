//! Tokenizers: split raw text into [`super::Token`]s.

use super::Token;

/// A character that can be part of a word in the standard tokenizer:
/// letters, digits, and the apostrophe (kept inside words, e.g. "don't").
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\''
}

/// Whether `c` is a CJK ideograph (each becomes its own token, like
/// Lucene's standard tokenizer).
fn is_ideograph(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
            | '\u{3400}'..='\u{4DBF}' // CJK Extension A
            | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
            | '\u{2A700}'..='\u{2EBEF}' // CJK Extensions C-F
            | '\u{3040}'..='\u{309F}'  // Hiragana
            | '\u{30A0}'..='\u{30FF}'  // Katakana
            | '\u{AC00}'..='\u{D7AF}'  // Hangul Syllables
    )
}

/// Lucene's standard tokenizer.
///
/// Splits on non-alphanumeric characters, but keeps `'` inside words
/// (so "don't" is one token). CJK ideographs each become their own
/// token, as Lucene's standard tokenizer does for ideographs.
/// Offsets are byte offsets into the input.
#[derive(Clone, Debug, Default)]
pub struct StandardTokenizer;

impl super::Tokenizer for StandardTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        for (start, seg) in uax29_words(text) {
            tokens.push(Token { text: seg.to_string(), position, offset_from: start as u32, offset_to: (start + seg.len()) as u32 });
            position += 1;
        }
        tokens
    }
}

/// Whether a UAX#29 word segment is a token of Lucene's `StandardTokenizer`:
/// it must contain a letter, a digit or an emoji (separators, punctuation and
/// all-underscore runs are dropped).
fn is_standard_token(seg: &str) -> bool {
    seg.chars().any(|c| c.is_alphanumeric() || (c as u32) >= 0x1F000)
}

/// Southeast Asian scripts (Thai, Lao, Myanmar, Khmer): UAX#29 breaks these
/// with a dictionary; Lucene's tokenizer keeps each run as one token.
fn is_southeast_asian(c: char) -> bool {
    matches!(c as u32, 0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1780..=0x17FF | 0xAA60..=0xAA7F | 0xA9E0..=0xA9FF)
}

/// UAX#29 word segments (`(byte offset, text)`) that are standard tokens.
/// This is what Lucene's `StandardTokenizer` produces (`foo_bar`, `3.14`,
/// `1,000`, `a:b` and `don't` stay whole; Han and Hiragana split per
/// character, Katakana and Hangul runs stay whole).
fn uax29_words(text: &str) -> Vec<(usize, &str)> {
    use unicode_segmentation::UnicodeSegmentation;
    let mut out: Vec<(usize, &str)> = Vec::new();
    let mut prev_end = usize::MAX;
    for (i, seg) in text.split_word_bound_indices() {
        if !is_standard_token(seg) {
            continue;
        }
        let sea = seg.chars().next().is_some_and(is_southeast_asian);
        if sea && prev_end == i {
            if let Some(last) = out.last_mut() {
                if last.1.chars().next_back().is_some_and(is_southeast_asian) {
                    last.1 = &text[last.0..i + seg.len()];
                    prev_end = i + seg.len();
                    continue;
                }
            }
        }
        out.push((i, seg));
        prev_end = i + seg.len();
    }
    out
}

/// Fused `StandardTokenizer` + `LowerCaseFilter`: same token boundaries and
/// positions, lowercased text written into `buf[base..]` and passed to `f`
/// (with `buf`'s prefix). ASCII words are scanned and lowercased bytewise;
/// a word containing any non-ASCII char is lowercased with
/// `str::to_lowercase` exactly like the filter.
pub(crate) fn standard_lowercase_stream<F: FnMut(&[u8], u32)>(text: &str, buf: &mut Vec<u8>, base: usize, f: &mut F) {
    if text.is_ascii() {
        standard_lowercase_ascii(text.as_bytes(), buf, base, f, 0);
        return;
    }
    // Mostly-ASCII text (most prose): UAX#29 never joins a word across
    // ASCII whitespace, so the text splits there into independent stretches.
    // ASCII stretches take the bytewise scanner; only the whitespace-bounded
    // stretch around non-ASCII bytes takes the full word-break rules.
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut position: u32 = 0;
    let mut i = 0usize;
    while i < n {
        let non_ascii = bytes[i..].iter().position(|&b| b >= 0x80).map_or(n, |p| i + p);
        // The ASCII stretch ends after the last whitespace before it.
        let ascii_end = if non_ascii == n { n } else {
            bytes[i..non_ascii].iter().rposition(|b| b.is_ascii_whitespace()).map_or(i, |p| i + p + 1)
        };
        if ascii_end > i {
            position = standard_lowercase_ascii(&bytes[i..ascii_end], buf, base, f, position);
        }
        if ascii_end >= n {
            break;
        }
        // The full-rules stretch ends at the next ASCII whitespace.
        let island_end = bytes[non_ascii..].iter().position(|b| b.is_ascii_whitespace()).map_or(n, |p| non_ascii + p);
        for (_, seg) in uax29_words(&text[ascii_end..island_end]) {
            buf.truncate(base);
            buf.extend_from_slice(seg.to_lowercase().as_bytes());
            f(buf, position);
            position += 1;
        }
        i = island_end;
    }
}

/// Word-byte mask of 8 ASCII bytes (little-endian load): bit `k` is set iff
/// byte `k` is `[0-9A-Za-z_]`. Branch-free SWAR; every byte must be < 0x80.
#[inline(always)]
fn ascii_word_mask8(w: u64) -> u64 {
    const L: u64 = 0x0101_0101_0101_0101;
    const H: u64 = 0x8080_8080_8080_8080;
    let x = w | (0x20 * L); // fold case ('A'..'Z' -> 'a'..'z')
    let alpha = (x + (0x80 - 0x61) * L) & !(x + (0x80 - 0x7b) * L);
    let digit = (w + (0x80 - 0x30) * L) & !(w + (0x80 - 0x3a) * L);
    let y = w ^ (0x5f * L);
    let under = !(y | (y + 0x7f * L));
    let hi = (alpha | digit | under) & H;
    (hi >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

/// Word-byte mask of up to 64 ASCII bytes (bit `j` = byte `j`).
#[inline(always)]
fn ascii_word_mask(chunk: &[u8]) -> u64 {
    let mut m = 0u64;
    let mut words = chunk.chunks_exact(8);
    let mut k = 0;
    for w in &mut words {
        m |= ascii_word_mask8(u64::from_le_bytes(w.try_into().unwrap())) << k;
        k += 8;
    }
    let rest = words.remainder();
    if !rest.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        // Zero padding bytes are separators.
        m |= ascii_word_mask8(u64::from_le_bytes(tail)) << k;
    }
    m
}

/// UAX#29 mid-word joiners for ASCII: the word ending at `e` (exclusive)
/// continues across `bytes[e]` into the word starting at `e + 1` when both
/// neighbours are letters (`.` `'` `:`) or both are digits (`.` `'` `,` `;`).
#[inline]
fn ascii_joins(bytes: &[u8], e: usize) -> bool {
    if e == 0 || e + 1 >= bytes.len() {
        return false;
    }
    let (l, m, r) = (bytes[e - 1], bytes[e], bytes[e + 1]);
    match m {
        b'.' | b'\'' => (l.is_ascii_alphabetic() && r.is_ascii_alphabetic()) || (l.is_ascii_digit() && r.is_ascii_digit()),
        b':' => l.is_ascii_alphabetic() && r.is_ascii_alphabetic(),
        b',' | b';' => l.is_ascii_digit() && r.is_ascii_digit(),
        _ => false,
    }
}

/// [`standard_lowercase_stream`] for all-ASCII input: word boundaries come
/// from 64-byte word-byte bitmasks (start/end transitions found with
/// trailing-zero counts, so there is no per-byte branch), and each token is
/// lowercased branch-free into `buf`.
/// Returns the position after the last token (tokens are numbered from
/// `position`).
fn standard_lowercase_ascii<F: FnMut(&[u8], u32)>(bytes: &[u8], buf: &mut Vec<u8>, base: usize, f: &mut F, mut position: u32) -> u32 {
    let n = bytes.len();
    let mut open = 0usize; // start of the current word (valid when carry == 1)
    let mut carry = 0u64; // previous byte was a word byte
    let mut joined = false; // the word ended at a joiner (`.` `'` `:` `,` `;`) that continues it
    let mut emit = |s: usize, e: usize, position: &mut u32| {
        // A run of only `_` is not a token (a run starting elsewhere has a letter or digit).
        if bytes[s] == b'_' && bytes[s..e].iter().all(|&c| c == b'_') {
            return;
        }
        let len = e - s;
        buf.truncate(base);
        buf.reserve(len);
        // SAFETY: `len` bytes were reserved past `base` and every one is
        // written before `set_len`.
        unsafe {
            let dst = buf.as_mut_ptr().add(base);
            for (j, &c) in bytes[s..e].iter().enumerate() {
                *dst.add(j) = c | (((c.wrapping_sub(b'A') < 26) as u8) << 5);
            }
            buf.set_len(base + len);
        }
        f(buf, *position);
        *position += 1;
    };
    let mut chunk_base = 0usize;
    while chunk_base < n {
        let end = (chunk_base + 64).min(n);
        let m = ascii_word_mask(&bytes[chunk_base..end]);
        let prev = (m << 1) | carry;
        let mut starts = m & !prev;
        let mut ends = !m & prev;
        if end - chunk_base < 64 {
            // Bits past the text are not transitions, except the end of a
            // word running into the text's end (handled below).
            let valid = (1u64 << (end - chunk_base)) - 1;
            starts &= valid;
            ends &= valid;
        }
        // Starts and ends alternate; consume them in position order.
        let mut events = starts | ends;
        while events != 0 {
            let j = events.trailing_zeros() as usize;
            let bit = 1u64 << j;
            if starts & bit != 0 {
                if joined {
                    joined = false; // continues the word before the joiner
                } else {
                    open = chunk_base + j;
                }
            } else {
                let e = chunk_base + j;
                if ascii_joins(bytes, e) {
                    joined = true;
                } else {
                    emit(open, e, &mut position);
                }
            }
            events &= events - 1;
        }
        let last = end - chunk_base - 1;
        carry = (m >> last) & 1;
        chunk_base = end;
    }
    if carry == 1 {
        emit(open, n, &mut position);
    }
    position
}

/// Splits on whitespace runs; whitespace is not a token.
#[derive(Clone, Debug, Default)]
pub struct WhitespaceTokenizer;

impl super::Tokenizer for WhitespaceTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            let c = text[i..].chars().next().unwrap();
            if c.is_whitespace() {
                i += c.len_utf8();
            } else {
                let start = i;
                let mut end = i;
                while end < bytes.len() {
                    let c2 = text[end..].chars().next().unwrap();
                    if c2.is_whitespace() {
                        break;
                    }
                    end += c2.len_utf8();
                }
                tokens.push(Token {
                    text: text[start..end].to_string(),
                    position,
                    offset_from: start as u32,
                    offset_to: end as u32,
                });
                position += 1;
                i = end;
            }
        }
        tokens
    }
}

/// Splits on non-letter characters; runs of letters are tokens.
/// Digits and punctuation are discarded (Lucene's LetterTokenizer).
#[derive(Clone, Debug, Default)]
pub struct LetterTokenizer;

impl super::Tokenizer for LetterTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            let c = text[i..].chars().next().unwrap();
            if c.is_alphabetic() {
                let start = i;
                let mut end = i;
                while end < bytes.len() {
                    let c2 = text[end..].chars().next().unwrap();
                    if c2.is_alphabetic() {
                        end += c2.len_utf8();
                    } else {
                        break;
                    }
                }
                tokens.push(Token {
                    text: text[start..end].to_string(),
                    position,
                    offset_from: start as u32,
                    offset_to: end as u32,
                });
                position += 1;
                i = end;
            } else {
                i += c.len_utf8();
            }
        }
        tokens
    }
}

/// Emits the entire input as a single token (no filtering).
/// An empty input produces no tokens.
#[derive(Clone, Debug, Default)]
pub struct KeywordTokenizer;

impl super::Tokenizer for KeywordTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![Token {
                text: text.to_string(),
                position: 0,
                offset_from: 0,
                offset_to: text.len() as u32,
            }]
        }
    }
}

/// Emits all n-grams of length `min`..=`max` (in characters) for each
/// whitespace-separated word. Words shorter than `min` produce nothing.
/// Offsets point into the original word.
#[derive(Clone, Debug)]
pub struct NGramTokenizer {
    /// Minimum n-gram length (characters).
    pub min: usize,
    /// Maximum n-gram length (characters).
    pub max: usize,
}

impl super::Tokenizer for NGramTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        for word in text.split_whitespace() {
            let word_start = text
                .find(word)
                .expect("word comes from text");
            let chars: Vec<char> = word.chars().collect();
            let n = chars.len();
            for len in self.min..=self.max.min(n) {
                for start in 0..=n - len {
                    let gram: String = chars[start..start + len].iter().collect();
                    let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
                    let byte_end: usize = chars[..start + len].iter().map(|c| c.len_utf8()).sum();
                    tokens.push(Token {
                        text: gram,
                        position,
                        offset_from: (word_start + byte_start) as u32,
                        offset_to: (word_start + byte_end) as u32,
                    });
                    position += 1;
                }
            }
        }
        tokens
    }
}

/// Emits prefix n-grams of length `min`..=`max` (in characters) for each
/// whitespace-separated word. Words shorter than `min` produce nothing.
/// Offsets point into the original word.
#[derive(Clone, Debug)]
pub struct EdgeNGramTokenizer {
    /// Minimum n-gram length (characters).
    pub min: usize,
    /// Maximum n-gram length (characters).
    pub max: usize,
}

impl super::Tokenizer for EdgeNGramTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        for word in text.split_whitespace() {
            let word_start = text
                .find(word)
                .expect("word comes from text");
            let chars: Vec<char> = word.chars().collect();
            let n = chars.len();
            for len in self.min..=self.max.min(n) {
                let gram: String = chars[..len].iter().collect();
                let byte_end: usize = chars[..len].iter().map(|c| c.len_utf8()).sum();
                tokens.push(Token {
                    text: gram,
                    position,
                    offset_from: word_start as u32,
                    offset_to: (word_start + byte_end) as u32,
                });
                position += 1;
            }
        }
        tokens
    }
}


/// Shared helper: scan a run of characters satisfying `pred` starting at
/// byte `start`, returning the end byte offset.
fn scan_while(text: &str, start: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut end = start;
    while end < text.len() {
        let c = text[end..].chars().next().unwrap();
        if pred(c) {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Whether `text[start..]` begins with a URL of the form `scheme://...`;
/// returns the end offset of the URL. Terminates at whitespace, `<`, `>`,
/// `"`, and drops trailing sentence punctuation (`.,;:!?)`).
fn scan_url(text: &str, start: usize) -> Option<usize> {
    let scheme_end = scan_while(text, start, |c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.');
    if scheme_end == start || !text[start..scheme_end].chars().next().unwrap().is_ascii_alphabetic() {
        return None;
    }
    if !text[scheme_end..].starts_with("://") {
        return None;
    }
    let body_start = scheme_end + 3;
    let mut end = scan_while(text, body_start, |c| {
        !c.is_whitespace() && !matches!(c, '<' | '>' | '"' | '\'' | '`' | '{' | '}' | '|' | '\\' | '^')
    });
    if end == body_start {
        return None;
    }
    // Trailing punctuation is not part of the URL.
    while end > body_start {
        let c = text[..end].chars().next_back().unwrap();
        if matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']') {
            end -= c.len_utf8();
        } else {
            break;
        }
    }
    Some(end)
}

fn is_email_local_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '%' | '\'')
}

fn is_host_char(c: char) -> bool {
    c.is_alphanumeric() || c == '-' || c == '.'
}

/// Whether `text[start..]` begins with an email address `local@domain.tld`;
/// returns its end offset. The domain must contain at least one dot and
/// end in an alphabetic label of length >= 2.
fn scan_email(text: &str, start: usize) -> Option<usize> {
    let local_end = scan_while(text, start, is_email_local_char);
    if local_end == start || !text[local_end..].starts_with('@') {
        return None;
    }
    let host_start = local_end + 1;
    let mut host_end = scan_while(text, host_start, is_host_char);
    while host_end > host_start && text[..host_end].ends_with('.') {
        host_end -= 1;
    }
    let host = &text[host_start..host_end];
    let tld = host.rsplit('.').next()?;
    if !host.contains('.') || tld.len() < 2 || !tld.chars().all(|c| c.is_alphabetic()) {
        return None;
    }
    if host.starts_with('.') || host.contains("..") {
        return None;
    }
    Some(host_end)
}

/// Like [`StandardTokenizer`], but keeps URLs (`scheme://...`) and email
/// addresses (`local@domain.tld`) as single tokens (ES `uax_url_email`).
///
/// This uses a simple scanner rather than the full UAX#29 + RFC grammar:
/// a URL is `[a-z][a-z0-9+.-]*://` followed by non-whitespace, with
/// trailing sentence punctuation trimmed; an email is a run of
/// `[alnum._%+-']` then `@` then a dotted host whose last label is
/// alphabetic.
#[derive(Clone, Debug, Default)]
pub struct UaxUrlEmailTokenizer;

impl super::Tokenizer for UaxUrlEmailTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        let mut i = 0usize;
        let mut push = |tokens: &mut Vec<Token>, start: usize, end: usize| {
            tokens.push(Token {
                text: text[start..end].to_string(),
                position,
                offset_from: start as u32,
                offset_to: end as u32,
            });
            position += 1;
        };
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            if is_ideograph(c) {
                push(&mut tokens, i, i + c.len_utf8());
                i += c.len_utf8();
            } else if is_word_char(c) || is_email_local_char(c) && c != '.' {
                if let Some(end) = scan_url(text, i) {
                    push(&mut tokens, i, end);
                    i = end;
                } else if let Some(end) = scan_email(text, i) {
                    push(&mut tokens, i, end);
                    i = end;
                } else if is_word_char(c) {
                    let end = scan_while(text, i, is_word_char);
                    push(&mut tokens, i, end);
                    i = end;
                } else {
                    i += c.len_utf8();
                }
            } else {
                i += c.len_utf8();
            }
        }
        tokens
    }
}

/// Which side of a path hierarchy to emit.
///
/// Splits a path on `delimiter` and emits every prefix (ES
/// `path_hierarchy`): `/a/b/c` → `/a`, `/a/b`, `/a/b/c`, all at position
/// 0 (they are alternative representations of the same input). With
/// `reverse`, suffixes are emitted instead (`/a/b/c` → `/a/b/c`, `b/c`,
/// `c`). `skip` drops that many leading (or trailing, when reversed)
/// components. `replacement` substitutes the delimiter in the output.
#[derive(Clone, Debug)]
pub struct PathHierarchyTokenizer {
    /// Path separator (default `/`).
    pub delimiter: char,
    /// Character to emit in place of the delimiter (default: the delimiter).
    pub replacement: Option<char>,
    /// Emit suffixes rather than prefixes.
    pub reverse: bool,
    /// Number of initial tokens to skip.
    pub skip: usize,
}

impl Default for PathHierarchyTokenizer {
    fn default() -> Self {
        Self {
            delimiter: '/',
            replacement: None,
            reverse: false,
            skip: 0,
        }
    }
}

impl super::Tokenizer for PathHierarchyTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        if text.is_empty() {
            return Vec::new();
        }
        let repl = self.replacement.unwrap_or(self.delimiter);
        let emit = |s: &str| -> String {
            if repl == self.delimiter {
                s.to_string()
            } else {
                s.chars().map(|c| if c == self.delimiter { repl } else { c }).collect()
            }
        };
        // Byte offsets of each delimiter.
        let delims: Vec<usize> = text
            .char_indices()
            .filter(|(_, c)| *c == self.delimiter)
            .map(|(i, _)| i)
            .collect();
        let mut tokens = Vec::new();
        if !self.reverse {
            // Prefix ends: after each delimiter that is not at index 0 and
            // not the final char, plus the whole string.
            let mut ends: Vec<usize> = delims
                .iter()
                .copied()
                .filter(|&d| d > 0)
                .collect();
            if text.len() > *ends.last().unwrap_or(&0) || ends.is_empty() {
                ends.push(text.len());
            }
            ends.dedup();
            // A trailing delimiter ("/a/b/") yields "/a/b" and "/a/b/"? Lucene
            // emits "/a", "/a/b", "/a/b/" — ends already contains len.
            let mut skipped = 0usize;
            for end in ends {
                if skipped < self.skip {
                    skipped += 1;
                    continue;
                }
                tokens.push(Token {
                    text: emit(&text[..end]),
                    position: 0,
                    offset_from: 0,
                    offset_to: end as u32,
                });
            }
        } else {
            // Suffix starts: 0, and after each delimiter that is not the last char.
            let mut starts: Vec<usize> = vec![0];
            for &d in &delims {
                let after = d + self.delimiter.len_utf8();
                if after < text.len() {
                    starts.push(after);
                }
            }
            starts.dedup();
            let n = starts.len();
            for (idx, start) in starts.into_iter().enumerate() {
                // skip counts from the end (the shortest suffixes).
                if n - idx <= self.skip {
                    break;
                }
                tokens.push(Token {
                    text: emit(&text[start..]),
                    position: 0,
                    offset_from: start as u32,
                    offset_to: text.len() as u32,
                });
            }
        }
        tokens
    }
}

/// Splits on a configurable set of character classes and explicit
/// characters (ES `char_group`). Characters in the set are discarded;
/// everything else forms tokens.
#[derive(Clone, Debug, Default)]
pub struct CharGroupTokenizer {
    /// Split on Unicode whitespace.
    pub whitespace: bool,
    /// Split on letters.
    pub letter: bool,
    /// Split on digits.
    pub digit: bool,
    /// Split on punctuation (ASCII punctuation and Unicode `P*` approximated
    /// by `char::is_ascii_punctuation` plus common Unicode dashes/quotes).
    pub punctuation: bool,
    /// Split on symbols (`$+<=>^`|~` and currency/math symbols).
    pub symbol: bool,
    /// Additional explicit characters to split on.
    pub chars: Vec<char>,
    /// Maximum token length in characters (longer runs are split; 0 = unlimited;
    /// ES default 255).
    pub max_token_length: usize,
}

impl CharGroupTokenizer {
    /// Build from ES-style `tokenize_on_chars` entries: `"whitespace"`,
    /// `"letter"`, `"digit"`, `"punctuation"`, `"symbol"`, or a single
    /// character (escapes `\n`, `\t`, `\r`, `\\` are understood).
    pub fn from_specs(specs: &[&str]) -> Self {
        let mut t = Self::default();
        for s in specs {
            match *s {
                "whitespace" => t.whitespace = true,
                "letter" => t.letter = true,
                "digit" => t.digit = true,
                "punctuation" => t.punctuation = true,
                "symbol" => t.symbol = true,
                r"\n" => t.chars.push('\n'),
                r"\t" => t.chars.push('\t'),
                r"\r" => t.chars.push('\r'),
                r"\\" => t.chars.push('\\'),
                other => {
                    let mut it = other.chars();
                    if let (Some(c), None) = (it.next(), it.next()) {
                        t.chars.push(c);
                    }
                }
            }
        }
        t
    }

    fn is_split(&self, c: char) -> bool {
        (self.whitespace && c.is_whitespace())
            || (self.letter && c.is_alphabetic())
            || (self.digit && c.is_numeric())
            || (self.punctuation && is_punctuation(c))
            || (self.symbol && is_symbol(c))
            || self.chars.contains(&c)
    }
}

fn is_punctuation(c: char) -> bool {
    (c.is_ascii_punctuation() && !is_symbol(c))
        || matches!(
            c,
            '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{205E}' | '«' | '»' | '¡' | '¿' | '§' | '¶' | '·'
        )
}

fn is_symbol(c: char) -> bool {
    matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~')
        || matches!(c, '\u{20A0}'..='\u{20CF}' | '\u{2100}'..='\u{214F}' | '\u{2190}'..='\u{23FF}' | '\u{2500}'..='\u{27BF}' | '©' | '®' | '°' | '±' | '¬' | '×' | '÷' | '¢' | '£' | '¥')
}

impl super::Tokenizer for CharGroupTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        let mut i = 0usize;
        let max = if self.max_token_length == 0 { usize::MAX } else { self.max_token_length };
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            if self.is_split(c) {
                i += c.len_utf8();
                continue;
            }
            let mut end = i;
            let mut count = 0usize;
            while end < text.len() && count < max {
                let c2 = text[end..].chars().next().unwrap();
                if self.is_split(c2) {
                    break;
                }
                end += c2.len_utf8();
                count += 1;
            }
            tokens.push(Token {
                text: text[i..end].to_string(),
                position,
                offset_from: i as u32,
                offset_to: end as u32,
            });
            position += 1;
            i = end;
        }
        tokens
    }
}

/// Lucene's `ClassicTokenizer` (the pre-3.1 standard
/// tokenizer): like [`StandardTokenizer`] but
///
/// * keeps acronyms with internal dots (`I.B.M.` → `I.B.M.`),
/// * keeps company names joined by `&` or `@` (`AT&T`),
/// * keeps email addresses and dotted host names (`www.example.com`),
/// * keeps hyphenated words only when they contain a digit (`SD-500`),
///   splitting plain `wi-fi` into two tokens, and
/// * keeps numbers with internal `.`/`,`/`-`/`/` (`1,000.5`).
#[derive(Clone, Debug, Default)]
pub struct ClassicTokenizer;

impl ClassicTokenizer {
    /// Scan an acronym `X.Y.Z.` (single letters separated by dots, at
    /// least two letters) starting at `start`.
    fn scan_acronym(text: &str, start: usize) -> Option<usize> {
        let mut i = start;
        let mut letters = 0;
        loop {
            let c = text[i..].chars().next()?;
            if !c.is_alphabetic() {
                break;
            }
            let after = i + c.len_utf8();
            if !text[after..].starts_with('.') {
                break;
            }
            letters += 1;
            i = after + 1;
            // Stop if the next char is not a letter followed by a dot.
            match text[i..].chars().next() {
                Some(n) if n.is_alphabetic() => {
                    let a2 = i + n.len_utf8();
                    if !text[a2..].starts_with('.') {
                        // "I.B.Mx" → acronym "I.B." then "Mx"? Lucene requires
                        // the trailing dot; stop here.
                        break;
                    }
                }
                _ => break,
            }
        }
        if letters >= 2 { Some(i) } else { None }
    }

    /// Scan a dotted host (`www.foo.com`): alnum labels joined by single
    /// dots, at least two labels, last label alphabetic.
    fn scan_host(text: &str, start: usize) -> Option<usize> {
        let end = scan_while(text, start, |c| c.is_alphanumeric() || c == '-' || c == '.');
        let mut e = end;
        while e > start && text[..e].ends_with('.') {
            e -= 1;
        }
        let host = &text[start..e];
        let labels: Vec<&str> = host.split('.').collect();
        if labels.len() < 2 || labels.iter().any(|l| l.is_empty()) {
            return None;
        }
        let last = labels.last().unwrap();
        if last.len() < 2 || !last.chars().all(|c| c.is_alphabetic()) {
            return None;
        }
        Some(e)
    }

    /// Scan a company name `A&B` / `A@B` (alnum runs joined by `&`/`@`).
    fn scan_company(text: &str, start: usize) -> Option<usize> {
        let first = scan_while(text, start, |c| c.is_alphanumeric());
        if first == start {
            return None;
        }
        let mut i = first;
        let mut joined = false;
        while i < text.len() && (text[i..].starts_with('&') || text[i..].starts_with('@')) {
            let next = scan_while(text, i + 1, |c| c.is_alphanumeric());
            if next == i + 1 {
                break;
            }
            joined = true;
            i = next;
        }
        if joined { Some(i) } else { None }
    }

    /// Scan a number with internal `.`, `,`, `-`, `/` (`1,000.5`, `12-34`),
    /// or an alnum run containing a hyphen and a digit (`SD-500`).
    fn scan_num(text: &str, start: usize) -> Option<usize> {
        let end = scan_while(text, start, |c| c.is_alphanumeric() || matches!(c, '.' | ',' | '-' | '/'));
        let mut e = end;
        while e > start && text[..e].ends_with(['.', ',', '-', '/']) {
            e -= 1;
        }
        let s = &text[start..e];
        let has_digit = s.chars().any(|c| c.is_ascii_digit());
        let has_sep = s.chars().any(|c| matches!(c, '.' | ',' | '-' | '/'));
        if has_digit && has_sep && !s.contains("..") && !s.contains("--") {
            Some(e)
        } else {
            None
        }
    }
}

impl super::Tokenizer for ClassicTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position: u32 = 0;
        let mut i = 0usize;
        while i < text.len() {
            let c = text[i..].chars().next().unwrap();
            if !c.is_alphanumeric() {
                i += c.len_utf8();
                continue;
            }
            let end = if is_ideograph(c) {
                i + c.len_utf8()
            } else if let Some(e) = scan_email(text, i) {
                e
            } else if let Some(e) = Self::scan_acronym(text, i) {
                e
            } else if let Some(e) = Self::scan_host(text, i) {
                e
            } else if let Some(e) = Self::scan_company(text, i) {
                e
            } else if let Some(e) = Self::scan_num(text, i) {
                e
            } else {
                scan_while(text, i, is_word_char)
            };
            tokens.push(Token {
                text: text[i..end].to_string(),
                position,
                offset_from: i as u32,
                offset_to: end as u32,
            });
            position += 1;
            i = end;
        }
        tokens
    }
}

/// [`LetterTokenizer`] followed by lowercasing (ES `lowercase` tokenizer).
#[derive(Clone, Debug, Default)]
pub struct LowercaseTokenizer;

impl super::Tokenizer for LowercaseTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = LetterTokenizer.tokenize(text);
        for t in &mut tokens {
            t.text = t.text.to_lowercase();
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Tokenizer;

    fn texts(tokens: &[Token]) -> Vec<String> {
        tokens.iter().map(|t| t.text.clone()).collect()
    }

    fn positions(tokens: &[Token]) -> Vec<u32> {
        tokens.iter().map(|t| t.position).collect()
    }

    // ---------- StandardTokenizer ----------

    #[test]
    fn standard_basic() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("Hello, world!");
        assert_eq!(texts(&tokens), vec!["Hello", "world"]);
        assert_eq!(positions(&tokens), vec![0, 1]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 5));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (7, 12));
    }

    #[test]
    fn standard_apostrophe_inside_word() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("don't stop");
        assert_eq!(texts(&tokens), vec!["don't", "stop"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 5));
    }

    #[test]
    fn standard_digits_with_letters() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("abc123 def 456");
        assert_eq!(texts(&tokens), vec!["abc123", "def", "456"]);
    }

    #[test]
    fn standard_cjk_each_own_token() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("你好 world");
        assert_eq!(texts(&tokens), vec!["你", "好", "world"]);
        assert_eq!(positions(&tokens), vec![0, 1, 2]);
        // Byte offsets: 你 = 0..3, 好 = 3..6, world = 7..12
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 3));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (3, 6));
        assert_eq!((tokens[2].offset_from, tokens[2].offset_to), (7, 12));
    }

    #[test]
    fn standard_mixed_cjk_latin() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("a1b2 中文");
        assert_eq!(texts(&tokens), vec!["a1b2", "中", "文"]);
    }

    #[test]
    fn standard_punctuation_only() {
        let t = StandardTokenizer;
        assert!(t.tokenize("!!! ... , ,").is_empty());
    }

    #[test]
    fn standard_empty() {
        let t = StandardTokenizer;
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn standard_leading_trailing_punct() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("  (hello)  ");
        assert_eq!(texts(&tokens), vec!["hello"]);
        // "  (hello)  " — h is at byte 3, o is at byte 7
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (3, 8));
    }

    #[test]
    fn standard_unicode_letters() {
        let t = StandardTokenizer;
        let tokens = t.tokenize("café naïve");
        assert_eq!(texts(&tokens), vec!["café", "naïve"]);
    }

    #[test]
    fn standard_uax29_joiners() {
        let t = StandardTokenizer;
        let cases: &[(&str, &[&str])] = &[
            ("GET /elasticsearch_1 HTTP/1.1", &["GET", "elasticsearch_1", "HTTP", "1.1"]),
            ("a:b 1:2 a'b 1,2 a,b 1;2 a;1 a.1", &["a:b", "1", "2", "a'b", "1,2", "a", "b", "1;2", "a", "1", "a", "1"]),
            ("_ __ a _ b 'abc' a''b a..b a._b", &["a", "b", "abc", "a", "b", "a", "b", "a", "_b"]),
            ("3.14.15 1,000,000 a.b.c x-y v1.2.3-beta_4", &["3.14.15", "1,000,000", "a.b.c", "x", "y", "v1.2.3", "beta_4"]),
            ("user@host.com $5.00 C++", &["user", "host.com", "5.00", "C"]),
            ("カタカナ ひらがな 한국어 กข", &["カタカナ", "ひ", "ら", "が", "な", "한국어", "กข"]),
        ];
        for (input, want) in cases {
            assert_eq!(texts(&t.tokenize(input)), *want, "{input}");
        }
    }

    #[test]
    fn standard_fast_path_matches_segmentation() {
        // Deterministic pseudo-random ASCII strings over the interesting alphabet.
        let alphabet = b"aAbB19_.':,; -\"/";
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..20000 {
            let mut s = String::new();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let len = (x % 40) as usize;
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.push(alphabet[(x % alphabet.len() as u64) as usize] as char);
            }
            let want: Vec<String> = uax29_words(&s).into_iter().map(|(_, w)| w.to_lowercase()).collect();
            let mut got = Vec::new();
            let mut buf = Vec::new();
            standard_lowercase_stream(&s, &mut buf, 0, &mut |b: &[u8], _| got.push(String::from_utf8(b.to_vec()).unwrap()));
            assert_eq!(got, want, "{s:?}");
        }
    }

    #[test]
    fn mixed_ascii_and_unicode_text_matches_segmentation() {
        // The whitespace-split fast path for mostly-ASCII text: same tokens
        // as UAX#29 over the whole string, around non-ASCII letters,
        // punctuation, combining marks, CJK, Thai and emoji.
        let alphabet: Vec<char> = "aAbB19_.':,; -\"/\t\néü—’\u{301}中文กขカ😀".chars().collect();
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20000 {
            let mut s = String::new();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let len = (x % 40) as usize;
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.push(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            let want: Vec<String> = uax29_words(&s).into_iter().map(|(_, w)| w.to_lowercase()).collect();
            let mut got = Vec::new();
            let mut positions = Vec::new();
            let mut buf = Vec::new();
            standard_lowercase_stream(&s, &mut buf, 0, &mut |b: &[u8], p| { got.push(String::from_utf8(b.to_vec()).unwrap()); positions.push(p); });
            assert_eq!(got, want, "{s:?}");
            assert_eq!(positions, (0..got.len() as u32).collect::<Vec<_>>(), "{s:?}");
        }
    }

    // ---------- WhitespaceTokenizer ----------

    #[test]
    fn whitespace_basic() {
        let t = WhitespaceTokenizer;
        let tokens = t.tokenize("a  b\tc\n d");
        assert_eq!(texts(&tokens), vec!["a", "b", "c", "d"]);
        assert_eq!(positions(&tokens), vec![0, 1, 2, 3]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 1));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (3, 4));
        assert_eq!((tokens[2].offset_from, tokens[2].offset_to), (5, 6));
        assert_eq!((tokens[3].offset_from, tokens[3].offset_to), (8, 9));
    }

    #[test]
    fn whitespace_keeps_punctuation() {
        let t = WhitespaceTokenizer;
        let tokens = t.tokenize("hello, world!");
        assert_eq!(texts(&tokens), vec!["hello,", "world!"]);
    }

    #[test]
    fn whitespace_empty_and_blank() {
        let t = WhitespaceTokenizer;
        assert!(t.tokenize("").is_empty());
        assert!(t.tokenize("   \t\n  ").is_empty());
    }

    // ---------- LetterTokenizer ----------

    #[test]
    fn letter_basic() {
        let t = LetterTokenizer;
        let tokens = t.tokenize("Hello, World! 2026");
        assert_eq!(texts(&tokens), vec!["Hello", "World"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 5));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (7, 12));
    }

    #[test]
    fn letter_drops_digits() {
        let t = LetterTokenizer;
        let tokens = t.tokenize("abc123def");
        assert_eq!(texts(&tokens), vec!["abc", "def"]);
    }

    #[test]
    fn letter_apostrophe_splits() {
        let t = LetterTokenizer;
        let tokens = t.tokenize("don't");
        assert_eq!(texts(&tokens), vec!["don", "t"]);
    }

    #[test]
    fn letter_unicode() {
        let t = LetterTokenizer;
        let tokens = t.tokenize("café 123 naïve");
        assert_eq!(texts(&tokens), vec!["café", "naïve"]);
    }

    #[test]
    fn letter_empty() {
        let t = LetterTokenizer;
        assert!(t.tokenize("123 !?").is_empty());
    }

    // ---------- KeywordTokenizer ----------

    #[test]
    fn keyword_whole_input() {
        let t = KeywordTokenizer;
        let tokens = t.tokenize("a b c");
        assert_eq!(texts(&tokens), vec!["a b c"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 5));
    }

    #[test]
    fn keyword_empty() {
        let t = KeywordTokenizer;
        assert!(t.tokenize("").is_empty());
    }

    // ---------- NGramTokenizer ----------

    #[test]
    fn ngram_basic() {
        let t = NGramTokenizer { min: 2, max: 3 };
        let tokens = t.tokenize("cat");
        assert_eq!(texts(&tokens), vec!["ca", "at", "cat"]);
        assert_eq!(positions(&tokens), vec![0, 1, 2]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 2));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (1, 3));
        assert_eq!((tokens[2].offset_from, tokens[2].offset_to), (0, 3));
    }

    #[test]
    fn ngram_multiple_words() {
        let t = NGramTokenizer { min: 1, max: 2 };
        let tokens = t.tokenize("ab cd");
        assert_eq!(
            texts(&tokens),
            vec!["a", "b", "ab", "c", "d", "cd"]
        );
        assert_eq!(positions(&tokens), vec![0, 1, 2, 3, 4, 5]);
        // Offsets into the original text: "ab cd"
        assert_eq!((tokens[3].offset_from, tokens[3].offset_to), (3, 4));
        assert_eq!((tokens[5].offset_from, tokens[5].offset_to), (3, 5));
    }

    #[test]
    fn ngram_word_shorter_than_min() {
        let t = NGramTokenizer { min: 3, max: 4 };
        assert!(t.tokenize("ab").is_empty());
    }

    #[test]
    fn ngram_max_capped_by_word_length() {
        let t = NGramTokenizer { min: 1, max: 10 };
        let tokens = t.tokenize("ab");
        assert_eq!(texts(&tokens), vec!["a", "b", "ab"]);
    }

    #[test]
    fn ngram_empty() {
        let t = NGramTokenizer { min: 1, max: 2 };
        assert!(t.tokenize("").is_empty());
        assert!(t.tokenize("   ").is_empty());
    }

    #[test]
    fn ngram_unicode() {
        let t = NGramTokenizer { min: 1, max: 2 };
        let tokens = t.tokenize("éa");
        assert_eq!(texts(&tokens), vec!["é", "a", "éa"]);
        // é is 2 bytes: offsets 0..2, 2..3, 0..3
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 2));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (2, 3));
        assert_eq!((tokens[2].offset_from, tokens[2].offset_to), (0, 3));
    }

    // ---------- EdgeNGramTokenizer ----------

    #[test]
    fn edge_ngram_basic() {
        let t = EdgeNGramTokenizer { min: 1, max: 3 };
        let tokens = t.tokenize("cat");
        assert_eq!(texts(&tokens), vec!["c", "ca", "cat"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 1));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (0, 2));
        assert_eq!((tokens[2].offset_from, tokens[2].offset_to), (0, 3));
    }

    #[test]
    fn edge_ngram_multiple_words() {
        let t = EdgeNGramTokenizer { min: 2, max: 3 };
        let tokens = t.tokenize("ab cd");
        assert_eq!(texts(&tokens), vec!["ab", "cd"]);
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (3, 5));
    }

    #[test]
    fn edge_ngram_word_shorter_than_min() {
        let t = EdgeNGramTokenizer { min: 3, max: 4 };
        assert!(t.tokenize("ab").is_empty());
    }

    #[test]
    fn edge_ngram_max_capped() {
        let t = EdgeNGramTokenizer { min: 1, max: 10 };
        let tokens = t.tokenize("ab");
        assert_eq!(texts(&tokens), vec!["a", "ab"]);
    }

    #[test]
    fn edge_ngram_empty() {
        let t = EdgeNGramTokenizer { min: 1, max: 2 };
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn edge_ngram_unicode() {
        let t = EdgeNGramTokenizer { min: 1, max: 2 };
        let tokens = t.tokenize("éa");
        assert_eq!(texts(&tokens), vec!["é", "éa"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 2));
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (0, 3));
    }

    // ---------- UaxUrlEmailTokenizer ----------

    #[test]
    fn uax_url_kept_whole() {
        let t = UaxUrlEmailTokenizer;
        let tokens = t.tokenize("see https://example.com/a?b=c&d=e#f, ok");
        assert_eq!(texts(&tokens), vec!["see", "https://example.com/a?b=c&d=e#f", "ok"]);
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (4, 35));
    }

    #[test]
    fn uax_email_kept_whole() {
        let t = UaxUrlEmailTokenizer;
        let tokens = t.tokenize("mail john.doe+x@mail.example.co.uk today.");
        assert_eq!(texts(&tokens), vec!["mail", "john.doe+x@mail.example.co.uk", "today"]);
    }

    #[test]
    fn uax_not_an_email() {
        let t = UaxUrlEmailTokenizer;
        assert_eq!(texts(&t.tokenize("a@b c@d.1 @x")), vec!["a", "b", "c", "d", "1", "x"]);
    }

    #[test]
    fn uax_falls_back_to_standard() {
        let t = UaxUrlEmailTokenizer;
        assert_eq!(texts(&t.tokenize("Hello, don't 你好")), vec!["Hello", "don't", "你", "好"]);
    }

    #[test]
    fn uax_url_trailing_punct_and_scheme_variants() {
        let t = UaxUrlEmailTokenizer;
        assert_eq!(
            texts(&t.tokenize("(ftp://x.org/y). git+ssh://h/p; http:// nothing")),
            vec!["ftp://x.org/y", "git+ssh://h/p", "http", "nothing"]
        );
    }

    // ---------- PathHierarchyTokenizer ----------

    #[test]
    fn path_hierarchy_basic() {
        let t = PathHierarchyTokenizer::default();
        let tokens = t.tokenize("/a/b/c");
        assert_eq!(texts(&tokens), vec!["/a", "/a/b", "/a/b/c"]);
        assert_eq!(positions(&tokens), vec![0, 0, 0]);
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (0, 4));
    }

    #[test]
    fn path_hierarchy_no_leading_delim_and_trailing() {
        let t = PathHierarchyTokenizer::default();
        assert_eq!(texts(&t.tokenize("a/b/")), vec!["a", "a/b", "a/b/"]);
        assert_eq!(texts(&t.tokenize("abc")), vec!["abc"]);
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn path_hierarchy_reverse() {
        let t = PathHierarchyTokenizer { reverse: true, ..Default::default() };
        let tokens = t.tokenize("/a/b/c");
        assert_eq!(texts(&tokens), vec!["/a/b/c", "a/b/c", "b/c", "c"]);
        assert_eq!((tokens[3].offset_from, tokens[3].offset_to), (5, 6));
    }

    #[test]
    fn path_hierarchy_skip_and_replacement() {
        let t = PathHierarchyTokenizer { delimiter: '-', replacement: Some('/'), skip: 1, reverse: false };
        assert_eq!(texts(&t.tokenize("one-two-three")), vec!["one/two", "one/two/three"]);
        let r = PathHierarchyTokenizer { reverse: true, skip: 1, ..Default::default() };
        assert_eq!(texts(&r.tokenize("/a/b/c")), vec!["/a/b/c", "a/b/c", "b/c"]);
    }

    // ---------- CharGroupTokenizer ----------

    #[test]
    fn char_group_whitespace_and_chars() {
        let t = CharGroupTokenizer::from_specs(&["whitespace", "-", r"\n"]);
        let tokens = t.tokenize("The QUICK brown-fox\njumps");
        assert_eq!(texts(&tokens), vec!["The", "QUICK", "brown", "fox", "jumps"]);
        assert_eq!(positions(&tokens), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn char_group_punctuation_symbol_digit() {
        let t = CharGroupTokenizer::from_specs(&["punctuation", "symbol", "digit", "whitespace"]);
        assert_eq!(texts(&t.tokenize("a,b$c3d «e»")), vec!["a", "b", "c", "d", "e"]);
    }

    #[test]
    fn char_group_letter_and_max_len() {
        let t = CharGroupTokenizer { letter: true, ..Default::default() };
        assert_eq!(texts(&t.tokenize("ab12cd345")), vec!["12", "345"]);
        let t = CharGroupTokenizer { whitespace: true, max_token_length: 3, ..Default::default() };
        assert_eq!(texts(&t.tokenize("abcdefg hi")), vec!["abc", "def", "g", "hi"]);
    }

    // ---------- ClassicTokenizer ----------

    #[test]
    fn classic_acronym_and_company() {
        let t = ClassicTokenizer;
        assert_eq!(texts(&t.tokenize("I.B.M. and AT&T, ok")), vec!["I.B.M.", "and", "AT&T", "ok"]);
    }

    #[test]
    fn classic_email_and_host() {
        let t = ClassicTokenizer;
        assert_eq!(
            texts(&t.tokenize("bob@example.com visits www.example.com.")),
            vec!["bob@example.com", "visits", "www.example.com"]
        );
    }

    #[test]
    fn classic_numbers_and_hyphens() {
        let t = ClassicTokenizer;
        assert_eq!(
            texts(&t.tokenize("SD-500 costs 1,000.50 wi-fi")),
            vec!["SD-500", "costs", "1,000.50", "wi", "fi"]
        );
    }

    #[test]
    fn classic_plain_and_apostrophe() {
        let t = ClassicTokenizer;
        let tokens = t.tokenize("O'Reilly's book. 中文");
        assert_eq!(texts(&tokens), vec!["O'Reilly's", "book", "中", "文"]);
        assert_eq!((tokens[0].offset_from, tokens[0].offset_to), (0, 10));
    }

    // ---------- LowercaseTokenizer ----------

    #[test]
    fn lowercase_tokenizer() {
        let t = LowercaseTokenizer;
        let tokens = t.tokenize("Hello WORLD 42 Café");
        assert_eq!(texts(&tokens), vec!["hello", "world", "café"]);
        assert_eq!((tokens[1].offset_from, tokens[1].offset_to), (6, 11));
    }
}


/// Decodes text pre-analyzed by a caller (`PRETOKENIZED_*`): records
/// `"{position}\u{1}{token}"` separated by `\u{2}`. Lets an engine hand over
/// tokens with their exact positions (stacked tokens, holes) and spaces
/// inside tokens (shingles); offsets are not kept.
#[derive(Clone, Debug, Default)]
pub struct PreTokenizedTokenizer;

/// Separator between the position and the token text.
pub const PRETOKENIZED_POS_SEP: char = '\u{1}';
/// Separator between records.
pub const PRETOKENIZED_REC_SEP: char = '\u{2}';

impl super::Tokenizer for PreTokenizedTokenizer {
    fn clone_box(&self) -> Box<dyn super::Tokenizer> {
        Box::new(self.clone())
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        text.split(PRETOKENIZED_REC_SEP)
            .filter_map(|rec| {
                let (pos, tok) = rec.split_once(PRETOKENIZED_POS_SEP)?;
                Some(Token { text: tok.to_string(), position: pos.parse().ok()?, offset_from: 0, offset_to: 0 })
            })
            .collect()
    }
}
