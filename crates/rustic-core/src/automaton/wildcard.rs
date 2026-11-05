//! Wildcard (`*` / `?`) term matching with `\` escapes, over Unicode
//! characters, using the classic backtracking-free greedy two-pointer
//! algorithm (O(n*m) worst case).

use super::TermMatcher;

/// One element of a parsed wildcard pattern.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Token {
    /// A literal character (escaped wildcards become literals).
    Lit(char),
    /// `*`: any run of characters, including empty.
    Star,
    /// `?`: exactly one character.
    Any,
}

/// Matches terms against a wildcard pattern.
///
/// Pattern syntax: `*` matches any run of characters (including empty),
/// `?` matches exactly one character, `\` escapes the next character
/// (so `\*` is a literal star). Everything else is a literal character.
pub struct WildcardMatcher {
    pattern: Vec<Token>,
    /// The characters before the first wildcard, as bytes: the literal
    /// prefix every match must start with.
    prefix_bytes: Vec<u8>,
}

impl WildcardMatcher {
    /// Build a matcher for `pattern`.
    pub fn new(pattern: &str) -> Self {
        let mut tokens = Vec::new();
        let mut it = pattern.chars().peekable();
        while let Some(c) = it.next() {
            let tok = match c {
                '\\' => match it.next() {
                    Some(n) => Token::Lit(n),
                    // Trailing backslash: treat as a literal backslash.
                    None => Token::Lit('\\'),
                },
                '*' => Token::Star,
                '?' => Token::Any,
                other => Token::Lit(other),
            };
            tokens.push(tok);
        }
        let mut prefix = String::new();
        for t in &tokens {
            match t {
                Token::Lit(c) => prefix.push(*c),
                _ => break,
            }
        }
        Self {
            pattern: tokens,
            prefix_bytes: prefix.into_bytes(),
        }
    }

    /// Greedy star matching: advance on literal/`?`, remember the last
    /// `*` position, and on mismatch rewind to it (consuming one more
    /// character of the text).
    fn matches_chars(&self, text: &[char]) -> bool {
        let (p, t) = (&self.pattern, text);
        let (mut pi, mut ti) = (0usize, 0usize);
        let (mut star, mut mark) = (usize::MAX, 0usize);
        while ti < t.len() {
            if pi < p.len() && matches!(p[pi], Token::Any) || pi < p.len() && p[pi] == Token::Lit(t[ti]) {
                pi += 1;
                ti += 1;
            } else if pi < p.len() && p[pi] == Token::Star {
                star = pi;
                mark = ti;
                pi += 1;
            } else if star != usize::MAX {
                pi = star + 1;
                mark += 1;
                ti = mark;
            } else {
                return false;
            }
        }
        while pi < p.len() && p[pi] == Token::Star {
            pi += 1;
        }
        pi == p.len()
    }
}

impl TermMatcher for WildcardMatcher {
    fn matches(&self, term: &[u8]) -> bool {
        let Ok(text) = std::str::from_utf8(term) else {
            return false;
        };
        self.matches_chars(&text.chars().collect::<Vec<_>>())
    }

    fn literal_prefix(&self) -> &[u8] {
        &self.prefix_bytes
    }
}

/// Returns the literal part of a pure prefix pattern (`foo*`: a single
/// trailing `*` and no other wildcards), or `None` otherwise. A bare `*`
/// is the empty prefix (`Some("")`), i.e. every term.
///
/// The returned slice is the raw pattern text before the `*`; escape
/// sequences in it are preserved as written.
pub fn is_prefix_pattern(p: &str) -> Option<&str> {
    let mut star_at: Option<usize> = None;
    let mut it = p.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        match c {
            '\\' => {
                it.next()?;
            }
            '?' => return None,
            '*' => {
                if star_at.is_some() {
                    return None;
                }
                star_at = Some(i);
            }
            _ => {}
        }
    }
    let idx = star_at?;
    if idx + 1 == p.len() {
        Some(&p[..idx])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str) -> WildcardMatcher {
        WildcardMatcher::new(pattern)
    }

    #[test]
    fn basic_stars() {
        assert!(m("abc").matches(b"abc"));
        assert!(!m("abc").matches(b"abcd"));
        assert!(m("a*c").matches(b"abc"));
        assert!(m("a*c").matches(b"ac"));
        assert!(m("a*c").matches(b"aXbYc"));
        assert!(!m("a*c").matches(b"ab"));
        assert!(m("*").matches(b""));
        assert!(m("*").matches(b"anything"));
        assert!(m("a**c").matches(b"ac"));
        assert!(m("a**c").matches(b"aXbYc"));
        assert!(!m("a**c").matches(b"ab"));
    }

    #[test]
    fn question_marks() {
        assert!(m("a?c").matches(b"abc"));
        assert!(!m("a?c").matches(b"ac"));
        assert!(m("*?").matches(b"a"));
        assert!(!m("*?").matches(b""));
        assert!(m("?*?").matches(b"ab"));
        assert!(!m("?*?").matches(b"a"));
    }

    #[test]
    fn escapes() {
        assert!(m(r"a\*c").matches(b"a*c"));
        assert!(!m(r"a\*c").matches(b"abc"));
        assert!(m(r"a\?c").matches(b"a?c"));
        assert!(!m(r"a\?c").matches(b"abc"));
        assert!(m(r"\\").matches(b"\\"));
    }

    #[test]
    fn unicode() {
        assert!(m("café*").matches("caféine".as_bytes()));
        assert!(m("café?").matches("caféx".as_bytes()));
        assert!(!m("café?").matches("café".as_bytes()));
    }

    #[test]
    fn literal_prefix() {
        assert_eq!(m("abc*def").literal_prefix(), b"abc");
        assert_eq!(m("a?c").literal_prefix(), b"a");
        assert_eq!(m(r"a\*c*").literal_prefix(), b"a*c");
        assert!(m("*abc").literal_prefix().is_empty());
        assert!(m("?").literal_prefix().is_empty());
        assert_eq!(m("plain").literal_prefix(), b"plain");
    }

    #[test]
    fn prefix_pattern_detection() {
        assert_eq!(is_prefix_pattern("foo*"), Some("foo"));
        assert_eq!(is_prefix_pattern("foo"), None);
        assert_eq!(is_prefix_pattern("*foo"), None);
        assert_eq!(is_prefix_pattern("fo*o*"), None);
        assert_eq!(is_prefix_pattern("fo?o*"), None);
        assert_eq!(is_prefix_pattern("**"), None);
        assert_eq!(is_prefix_pattern(r"f\oo*"), Some(r"f\oo"));
        assert_eq!(is_prefix_pattern("*"), Some(""));
    }
}
