//! Whole-term regular expression matching (Lucene/ES `RegexpQuery`
//! semantics: the pattern must match the *entire* term), built on
//! `regex-automata`'s meta-regex engine.

use super::TermMatcher;
use regex_automata::meta;
use regex_syntax::hir::{Hir, HirKind, Look};
use regex_syntax::ParserBuilder;

/// Flags for [`RegexpMatcher`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RegexpFlags {
    /// Match case-insensitively (as in Lucene's `RegExp` with the
    /// `CASE_INSENSITIVE` flag).
    pub case_insensitive: bool,
}

impl RegexpFlags {
    /// No flags set.
    pub const NONE: Self = Self {
        case_insensitive: false,
    };
}

/// Matches terms against a regular expression anchored at both ends.
pub struct RegexpMatcher {
    re: meta::Regex,
    prefix: Vec<u8>,
}

impl RegexpMatcher {
    /// Compile `pattern` with the given `flags`.
    pub fn new(pattern: &str, flags: RegexpFlags) -> Result<Self, String> {
        check_supported(pattern)?;
        let mut builder = ParserBuilder::new();
        builder.case_insensitive(flags.case_insensitive);
        let hir = builder.build().parse(pattern).map_err(|e| e.to_string())?;
        let prefix = if flags.case_insensitive {
            Vec::new()
        } else {
            literal_prefix(&hir)
        };
        // Lucene regexp semantics are whole-term: anchor at both ends.
        let anchored = Hir::concat(vec![Hir::look(Look::Start), hir, Hir::look(Look::End)]);
        let re = meta::Builder::new()
            .build_from_hir(&anchored)
            .map_err(|e| e.to_string())?;
        Ok(Self { re, prefix })
    }

    /// The literal prefix every match must start with (empty when
    /// case-insensitive, since the prefix would need case folding).
    pub fn literal_prefix(&self) -> &[u8] {
        &self.prefix
    }
}

impl std::fmt::Debug for RegexpMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegexpMatcher")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl TermMatcher for RegexpMatcher {
    fn matches(&self, term: &[u8]) -> bool {
        self.re.is_match(term)
    }

    fn literal_prefix(&self) -> &[u8] {
        &self.prefix
    }
}

/// Walk the HIR concatenation from the start, collecting `Literal` bytes
/// until the first non-literal node.
///
/// `regex-syntax` merges adjacent literals, so the leading literal (if
/// any) of a concatenation is a single node; a capture group wrapping a
/// literal is unwrapped (`(ab)c` -> `ab` + `c`).
fn literal_prefix(hir: &Hir) -> Vec<u8> {
    let mut out = Vec::new();
    let mut node = hir;
    loop {
        match node.kind() {
            HirKind::Literal(lit) => {
                out.extend_from_slice(&lit.0);
                break;
            }
            HirKind::Capture(cap) => node = &cap.sub,
            HirKind::Concat(parts) => {
                // Collect leading literals; descend into the first
                // non-literal only if it is a capture (which may itself
                // start with a literal).
                let mut i = 0;
                while i < parts.len() {
                    match parts[i].kind() {
                        HirKind::Literal(lit) => {
                            out.extend_from_slice(&lit.0);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                if i < parts.len() {
                    if let HirKind::Capture(_) = parts[i].kind() {
                        out.extend_from_slice(&literal_prefix(&parts[i]));
                    }
                }
                break;
            }
            _ => break,
        }
    }
    out
}

fn check_supported(pattern: &str) -> Result<(), String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut in_class = false;
    let mut escaped = false;
    for (i, &c) in chars.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '[' => in_class = true,
            ']' if in_class => in_class = false,
            '~' if !in_class => {
                return Err("unsupported regexp operator: ~ complement".into());
            }
            '<' if !in_class => {
                if is_interval(&chars[i + 1..]) {
                    return Err("unsupported regexp operator: <n-m> interval".into());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// True if `chars` starts with a Lucene interval: `<digits-digits>`.
fn is_interval(chars: &[char]) -> bool {
    let mut i = 0;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i >= chars.len() || chars[i] != '-' {
        return false;
    }
    i += 1;
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    i > start && i < chars.len() && chars[i] == '>'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str) -> RegexpMatcher {
        RegexpMatcher::new(pattern, RegexpFlags::NONE).unwrap()
    }

    #[test]
    fn anchoring() {
        assert!(m("ab").matches(b"ab"));
        assert!(!m("ab").matches(b"abc"));
        assert!(!m("ab").matches(b"xab"));
        assert!(m("a.c").matches(b"abc"));
        assert!(!m("a.c").matches(b"abbc"));
        assert!(m("a+").matches(b"aaaa"));
        assert!(m("a*").matches(b""));
        assert!(m("a(b|c)d").matches(b"abd"));
        assert!(m("a(b|c)d").matches(b"acd"));
        assert!(!m("a(b|c)d").matches(b"add"));
    }

    #[test]
    fn case_insensitive() {
        let m = RegexpMatcher::new("abc", RegexpFlags { case_insensitive: true }).unwrap();
        assert!(m.matches(b"abc"));
        assert!(m.matches(b"ABC"));
        assert!(m.matches(b"aBc"));
        assert!(!m.matches(b"abcd"));
        // No literal prefix when case-insensitive.
        assert!(m.literal_prefix().is_empty());
    }

    #[test]
    fn literal_prefix_extraction() {
        assert_eq!(m("abc.*").literal_prefix(), b"abc");
        assert_eq!(m("a(b|c)").literal_prefix(), b"a");
        assert!(m(".*").literal_prefix().is_empty());
        assert!(m("(a|b)c").literal_prefix().is_empty());
        assert_eq!(m("ab[cd]e").literal_prefix(), b"ab");
        assert_eq!(m("a.c").literal_prefix(), b"a");
    }

    #[test]
    fn unsupported_operators() {
        let e = RegexpMatcher::new("a~b", RegexpFlags::NONE).unwrap_err();
        assert!(e.contains("unsupported regexp operator"), "{e}");
        let e = RegexpMatcher::new("a<1-100>b", RegexpFlags::NONE).unwrap_err();
        assert!(e.contains("unsupported regexp operator"), "{e}");
        // Inside a character class these are fine.
        assert!(RegexpMatcher::new(r"a[~<>1-9]b", RegexpFlags::NONE).is_ok());
        // Escaped operators are fine.
        assert!(RegexpMatcher::new(r"a\~b", RegexpFlags::NONE).is_ok());
        // A plain `<` that is not an interval is fine.
        assert!(RegexpMatcher::new(r"a<b", RegexpFlags::NONE).is_ok());
    }

    #[test]
    fn invalid_pattern_errors() {
        assert!(RegexpMatcher::new("a(", RegexpFlags::NONE).is_err());
        assert!(RegexpMatcher::new("a[", RegexpFlags::NONE).is_err());
    }
}
