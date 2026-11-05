//! Java `BreakIterator` emulation over UTF-16 text (offsets are UTF-16 code
//! units, as in Lucene): sentence and word instances, ES's
//! `BoundedBreakIteratorScanner` (sentences split at word boundaries when
//! longer than `fragment_size`), and Lucene's `SplittingBreakIterator`
//! behaviour (no passage crosses a multi-value separator).

/// Character class of one UTF-16 unit (surrogates count as letters).
fn ch(u: u16) -> char {
    char::from_u32(u as u32).unwrap_or('a')
}

fn is_letter_or_digit(u: u16) -> bool {
    (0xD800..=0xDFFF).contains(&u) || ch(u).is_alphanumeric() || u == b'_' as u16
}

fn is_space(u: u16) -> bool {
    ch(u).is_whitespace() && !is_para_sep(u)
}

fn is_para_sep(u: u16) -> bool {
    matches!(u, 0x0A | 0x0D | 0x2029 | 0x2028 | 0x85 | 0x0C)
}

fn is_close(u: u16) -> bool {
    matches!(ch(u), '"' | '\'' | ')' | ']' | '}' | '»' | '’' | '”')
}

/// Sentence boundaries of `t` (always includes 0 and `t.len()`), after the
/// JDK sentence rules: a sentence ends after `.`/`?`/`!` (plus closing
/// punctuation and trailing spaces) unless a `.` is followed by a lowercase
/// continuation or not followed by a space; line/paragraph separators always
/// end a sentence.
pub fn sentence_boundaries(t: &[u16]) -> Vec<usize> {
    let n = t.len();
    let mut out = vec![0];
    let mut i = 0;
    while i < n {
        let c = t[i];
        if is_para_sep(c) {
            let mut j = i + 1;
            // CR LF is one separator.
            if c == 0x0D && j < n && t[j] == 0x0A {
                j += 1;
            }
            if j < n {
                out.push(j);
            }
            i = j;
            continue;
        }
        let term = ch(c);
        if term == '.' || term == '?' || term == '!' {
            let mut j = i + 1;
            let mut strong = term != '.';
            while j < n && matches!(ch(t[j]), '.' | '?' | '!') {
                if ch(t[j]) != '.' {
                    strong = true;
                }
                j += 1;
            }
            while j < n && is_close(t[j]) {
                j += 1;
            }
            let sp_start = j;
            while j < n && is_space(t[j]) {
                j += 1;
            }
            let had_space = j > sp_start;
            if j < n && is_para_sep(t[j]) {
                // The separator itself ends the sentence.
                i = j;
                continue;
            }
            if j >= n {
                break;
            }
            if strong {
                if had_space {
                    out.push(j);
                }
                i = j;
                continue;
            }
            // ATerm: break only if followed by space and the next letter is
            // not lowercase.
            if had_space {
                let next = ch(t[j]);
                let lower = next.is_lowercase();
                if !lower {
                    out.push(j);
                }
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    if *out.last().unwrap() != n {
        out.push(n);
    }
    out
}

/// Word boundaries of `t` (JDK word instance): runs of letters/digits (with
/// `'` between letters and `.`/`,` between digits) are words, runs of spaces
/// one segment, every other character its own segment.
pub fn word_boundaries(t: &[u16]) -> Vec<usize> {
    let n = t.len();
    let mut out = vec![0];
    let mut i = 0;
    while i < n {
        let c = t[i];
        let mut j = i + 1;
        if is_letter_or_digit(c) {
            while j < n {
                if is_letter_or_digit(t[j]) {
                    j += 1;
                    continue;
                }
                let mid = ch(t[j]);
                if j + 1 < n && is_letter_or_digit(t[j + 1]) {
                    let prev_digit = ch(t[j - 1]).is_ascii_digit();
                    let next_digit = ch(t[j + 1]).is_ascii_digit();
                    if (mid == '\'' || mid == '’') && !prev_digit && !next_digit {
                        j += 2;
                        continue;
                    }
                    if (mid == '.' || mid == ',') && prev_digit && next_digit {
                        j += 2;
                        continue;
                    }
                    if mid == '.' && !prev_digit && !next_digit {
                        // "e.g" style: JDK keeps letters around a period together.
                        j += 2;
                        continue;
                    }
                }
                break;
            }
        } else if is_space(c) {
            while j < n && is_space(t[j]) {
                j += 1;
            }
        } else if c == 0x0D && j < n && t[j] == 0x0A {
            j += 1;
        }
        out.push(j);
        i = j;
    }
    if n == 0 {
        out.push(0);
    }
    out
}

/// Last boundary `< off` (Java `preceding`), or the first boundary.
pub fn preceding(b: &[usize], off: usize) -> usize {
    match b.iter().rev().find(|&&x| x < off) {
        Some(x) => *x,
        None => b[0],
    }
}

/// First boundary `> off` (Java `following`), or the last boundary.
pub fn following(b: &[usize], off: usize) -> usize {
    match b.iter().find(|&&x| x > off) {
        Some(x) => *x,
        None => *b.last().unwrap(),
    }
}

/// The break iterator the unified highlighter uses for one field.
pub enum Scanner {
    /// One passage per value (`number_of_fragments: 0`, untokenized fields).
    Whole,
    /// Plain sentence instance (no `fragment_size`).
    Sentence,
    /// ES `BoundedBreakIteratorScanner` over sentences and words.
    BoundedSentence(usize),
    /// Word instance.
    Word,
}

/// Stateful break iterator over the joined field content.
pub struct BreakIter<'a> {
    text: &'a [u16],
    /// `(start, end)` of each value in `text` (separators excluded).
    values: &'a [(usize, usize)],
    kind: &'a Scanner,
    cache_value: usize,
    main: Vec<usize>,
    inner: Vec<usize>,
    // bounded scanner state (absolute offsets)
    window_start: isize,
    window_end: isize,
    inner_start: isize,
    inner_end: isize,
}

impl<'a> BreakIter<'a> {
    pub fn new(text: &'a [u16], values: &'a [(usize, usize)], kind: &'a Scanner) -> BreakIter<'a> {
        BreakIter { text, values, kind, cache_value: usize::MAX, main: Vec::new(), inner: Vec::new(), window_start: -1, window_end: -1, inner_start: -1, inner_end: -1 }
    }

    fn value_of(&self, off: usize) -> usize {
        for (i, (s, e)) in self.values.iter().enumerate() {
            if off >= *s && off <= *e {
                return i;
            }
        }
        self.values.len().saturating_sub(1)
    }

    fn load(&mut self, vi: usize) {
        if self.cache_value == vi {
            return;
        }
        self.cache_value = vi;
        let (s, e) = self.values[vi];
        let slice = &self.text[s..e];
        let shift = |v: Vec<usize>| v.into_iter().map(|x| x + s).collect::<Vec<_>>();
        match self.kind {
            Scanner::Whole => {
                self.main = vec![s, e];
                self.inner.clear();
            }
            Scanner::Sentence => {
                self.main = shift(sentence_boundaries(slice));
                self.inner.clear();
            }
            Scanner::BoundedSentence(_) => {
                self.main = shift(sentence_boundaries(slice));
                self.inner = shift(word_boundaries(slice));
            }
            Scanner::Word => {
                self.main = shift(word_boundaries(slice));
                self.inner.clear();
            }
        }
        self.window_start = -1;
        self.window_end = -1;
    }

    /// Lucene `preceding(off)` as called by `FieldHighlighter` with
    /// `off = match_start + 1`.
    pub fn preceding(&mut self, off: usize) -> usize {
        let vi = self.value_of(off.saturating_sub(1));
        self.load(vi);
        match self.kind {
            Scanner::BoundedSentence(max_len) => {
                let max_len = *max_len as isize;
                let o = off as isize;
                if o > self.window_start && o < self.window_end {
                    self.inner_start = self.inner_end;
                    self.inner_end = self.window_end;
                } else {
                    let ws = preceding(&self.main, off) as isize;
                    let we = following(&self.main, off - 1) as isize;
                    self.window_start = ws;
                    self.inner_start = ws;
                    self.window_end = we;
                    self.inner_end = we;
                    // A short sentence extends over the following whole sentences
                    // while the passage stays within `fragment_size`.
                    if we - ws <= max_len {
                        for &b in self.main.iter().filter(|&&b| b as isize > we) {
                            if b as isize - ws <= max_len {
                                self.inner_end = b as isize;
                            } else {
                                break;
                            }
                        }
                    }
                }
                if self.inner_end - self.inner_start > max_len {
                    // too big: find boundaries on the left first ...
                    if o - max_len > self.inner_start {
                        let p = preceding(&self.inner, (o - max_len) as usize) as isize;
                        self.inner_start = self.inner_start.max(p);
                    }
                    // ... then expand to the right with the remaining size.
                    let remaining = (max_len - (o - self.inner_start)).max(0);
                    if o + remaining < self.window_end {
                        let f = following(&self.inner, (o + remaining) as usize) as isize;
                        self.inner_end = self.window_end.min(f);
                    }
                }
                self.inner_start.max(0) as usize
            }
            _ => preceding(&self.main, off),
        }
    }

    /// Lucene `following(off)` with `off = match_start`.
    pub fn following(&mut self, off: usize) -> usize {
        let vi = self.value_of(off);
        self.load(vi);
        match self.kind {
            Scanner::BoundedSentence(_) => self.inner_end.max(0) as usize,
            _ => following(&self.main, off),
        }
    }
}

/// Lucene `SimpleBoundaryScanner.findStartOffset`.
pub fn chars_find_start(buf: &[u16], start: usize, chars: &[u16], max_scan: usize) -> usize {
    if start > buf.len() || start < 1 {
        return start;
    }
    let mut offset = start;
    let mut count = max_scan;
    while offset > 0 && count > 0 {
        if chars.contains(&buf[offset - 1]) {
            return offset;
        }
        offset -= 1;
        count -= 1;
    }
    if offset == 0 {
        return 0;
    }
    start
}

/// Lucene `SimpleBoundaryScanner.findEndOffset`.
pub fn chars_find_end(buf: &[u16], start: usize, chars: &[u16], max_scan: usize) -> usize {
    if start > buf.len() {
        return start;
    }
    let mut offset = start;
    let mut count = max_scan;
    while offset < buf.len() && count > 0 {
        if chars.contains(&buf[offset]) {
            return offset;
        }
        offset += 1;
        count -= 1;
    }
    start
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn sentences() {
        let t = u("Hello there. This is it! and more? yes. e.g. stuff. Mr. Smith");
        let b = sentence_boundaries(&t);
        assert_eq!(b[0], 0);
        assert!(b.contains(&13)); // after "Hello there. "
        assert!(b.contains(&25)); // after "This is it! "
        assert_eq!(*b.last().unwrap(), t.len());
        let t = u("The quick brown fox is brown.");
        assert_eq!(sentence_boundaries(&t), vec![0, t.len()]);
    }

    #[test]
    fn words() {
        let t = u("The quick  brown, fox's 3.14");
        let b = word_boundaries(&t);
        assert_eq!(b, vec![0, 3, 4, 9, 11, 16, 17, 18, 23, 24, 28]);
    }

    /// Captured from ES 8.19.22: `fragment_size: 50` over four 20-char sentences
    /// merges the first two into one passage.
    #[test]
    fn bounded_sentences_merge_within_fragment_size() {
        let t = u("Aaaa bbbb cccc ddd. Eeee ffff gggg hhh. Iiii jjjj kkkk lll. Mmmm nnnn oooo ppp.");
        let values = [(0usize, t.len())];
        let kind = Scanner::BoundedSentence(50);
        let mut b = BreakIter::new(&t, &values, &kind);
        assert_eq!(b.preceding(1), 0);
        assert_eq!(b.following(0), 40);
        let mut b = BreakIter::new(&t, &values, &kind);
        assert_eq!(b.preceding(41), 40);
        assert_eq!(b.following(40), t.len());
    }
}
