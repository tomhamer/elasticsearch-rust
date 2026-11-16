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
    /// Further operands of a Lucene intersection (`a&b`, `a&~(b)`): each
    /// term must also match every `(false, re)` and none of the
    /// `(true, re)` (complement), term by term ([`intersection`]).
    and: Vec<(bool, meta::Regex)>,
}

/// Marker of a pattern built by [`intersection`].
const AND_MARK: char = '\u{0}';

/// One term pattern for a Lucene intersection of `operands` (`(negated,
/// regex)`): the conjunction and complements apply to each term, as in
/// Lucene's `RegExp` automaton, never to whole documents.
pub fn intersection(operands: &[(bool, String)]) -> String {
    let mut out = String::new();
    for (neg, re) in operands {
        out.push(AND_MARK);
        out.push(if *neg { '-' } else { '+' });
        out.push_str(re);
    }
    out
}

/// The operands of an [`intersection`] pattern (`None`: a plain pattern).
pub fn intersection_operands(pattern: &str) -> Option<Vec<(bool, &str)>> {
    let rest = pattern.strip_prefix(AND_MARK)?;
    Some(rest.split(AND_MARK).map(|op| match op.strip_prefix('-') {
        Some(re) => (true, re),
        None => (false, op.strip_prefix('+').unwrap_or(op)),
    }).collect())
}

impl RegexpMatcher {
    /// Compile `pattern` with the given `flags` (a plain pattern, or an
    /// [`intersection`] of operands).
    pub fn new(pattern: &str, flags: RegexpFlags) -> Result<Self, String> {
        if let Some(ops) = intersection_operands(pattern) {
            let first = ops.iter().position(|(neg, _)| !neg);
            let mut m = match first {
                Some(i) => Self::new(ops[i].1, flags)?,
                // Only complements: every term is a candidate.
                None => Self::new("(?s:.*)", flags)?,
            };
            for (i, (neg, re)) in ops.iter().enumerate() {
                if Some(i) == first {
                    continue;
                }
                let sub = Self::new(re, flags)?;
                m.and.push((*neg, sub.re));
            }
            return Ok(m);
        }
        check_supported(pattern)?;
        let hir = parse(pattern, flags.case_insensitive)?;
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
        Ok(Self { re, prefix, and: Vec::new() })
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
        self.re.is_match(term) && self.and.iter().all(|(neg, re)| re.is_match(term) != *neg)
    }

    fn literal_prefix(&self) -> &[u8] {
        &self.prefix
    }
}

/// Parse with Lucene's semantics: `.` matches any character, line breaks
/// included (Lucene `RegExp` has no line-oriented mode).
fn parse(pattern: &str, case_insensitive: bool) -> Result<Hir, String> {
    let mut builder = ParserBuilder::new();
    builder.case_insensitive(case_insensitive).dot_matches_new_line(true);
    builder.build().parse(pattern).map_err(|e| e.to_string())
}

/// Literal strings every whole-term match of `pattern` must contain (for
/// an n-gram prefilter), read from the parsed pattern: literal runs of the
/// mandatory concatenation, descending into groups and repetitions of at
/// least one; alternations, classes and optional parts contribute
/// nothing. A repetition count (`{3,5}`) is never mistaken for text.
pub fn required_literals(pattern: &str) -> Result<Vec<String>, String> {
    check_supported(pattern)?;
    let hir = parse(pattern, false)?;
    fn walk(h: &Hir, out: &mut Vec<Vec<u8>>) {
        match h.kind() {
            HirKind::Literal(l) => out.push(l.0.to_vec()),
            HirKind::Capture(c) => walk(&c.sub, out),
            HirKind::Repetition(r) if r.min >= 1 => walk(&r.sub, out),
            HirKind::Concat(parts) => {
                let mut run: Vec<u8> = Vec::new();
                for p in parts {
                    if let HirKind::Literal(l) = p.kind() {
                        run.extend_from_slice(&l.0);
                    } else {
                        if !run.is_empty() {
                            out.push(std::mem::take(&mut run));
                        }
                        walk(p, out);
                    }
                }
                if !run.is_empty() {
                    out.push(run);
                }
            }
            _ => {}
        }
    }
    let mut runs = Vec::new();
    walk(&hir, &mut runs);
    Ok(runs.into_iter().filter_map(|r| String::from_utf8(r).ok()).collect())
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

    /// #491/#492: `.` spans line breaks (Lucene), and the prefilter
    /// literals come from the parsed pattern: repetition counts are not text.
    #[test]
    fn newlines_and_required_literals() {
        assert!(m("first.*last").matches(b"first\nlast"));
        assert_eq!(required_literals("serial[0-9]{3,5}").unwrap(), vec!["serial"]);
        assert_eq!(required_literals("ab(cde)+x?yz").unwrap(), vec!["ab", "cde", "yz"]);
        assert!(required_literals("(abc|def)").unwrap().is_empty());
        assert_eq!(required_literals("(abc){2}").unwrap(), vec!["abc"]);
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
