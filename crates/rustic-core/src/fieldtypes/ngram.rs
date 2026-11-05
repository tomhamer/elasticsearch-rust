//! `wildcard` field support (ES `WildcardFieldMapper`): values are indexed
//! as lower-cased 3-grams (the approximation index) and kept whole in
//! doc values for verification. A wildcard / regexp / prefix query becomes
//! "docs containing every 3-gram of the pattern's literal runs" and is then
//! verified against the doc value.

pub const NGRAM: usize = 3;

/// Lower-cased 3-grams of `s`, deduplicated. Values shorter than three
/// chars index themselves (lower-cased) so short exact matches still find
/// candidates.
pub fn ngrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.to_lowercase().chars().collect();
    let mut out: Vec<String> = if chars.len() < NGRAM {
        vec![chars.iter().collect()]
    } else {
        chars.windows(NGRAM).map(|w| w.iter().collect()).collect()
    };
    out.sort();
    out.dedup();
    out
}

/// Literal runs of a wildcard pattern (`*`, `?` split; `\x` escapes).
pub fn literal_runs(pattern: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut cur = String::new();
    let mut it = pattern.chars();
    while let Some(c) = it.next() {
        match c {
            '\\' => {
                if let Some(n) = it.next() {
                    cur.push(n);
                }
            }
            '*' | '?' => {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

/// The 3-grams every value matching `pattern` must contain (empty = no
/// constraint: the query must scan all values).
pub fn required_ngrams(pattern: &str) -> Vec<String> {
    let mut out: Vec<String> = literal_runs(pattern).iter().filter(|r| r.chars().count() >= NGRAM).flat_map(|r| ngrams(r)).collect();
    out.sort();
    out.dedup();
    out
}

/// Literal runs of a regular expression that every match must contain
/// (conservative: only top-level runs of plain characters without
/// alternation are used).
pub fn regex_required_ngrams(re: &str) -> Vec<String> {
    if re.contains('|') {
        return Vec::new();
    }
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut it = re.chars().peekable();
    let mut depth = 0i32;
    while let Some(c) = it.next() {
        let quantified = matches!(it.peek(), Some('*' | '?' | '{'));
        match c {
            '\\' => {
                let n = it.next();
                match n {
                    Some(n) if !n.is_alphanumeric() && depth == 0 && !matches!(it.peek(), Some('*' | '?' | '{')) => cur.push(n),
                    _ => runs.push(std::mem::take(&mut cur)),
                }
            }
            '(' | '[' => {
                depth += 1;
                runs.push(std::mem::take(&mut cur));
            }
            ')' | ']' => {
                depth -= 1;
                runs.push(std::mem::take(&mut cur));
            }
            '.' | '*' | '+' | '?' | '{' | '}' | '^' | '$' | '~' | '&' | '<' | '>' | '#' | '@' | '"' => runs.push(std::mem::take(&mut cur)),
            c if depth == 0 && !quantified => cur.push(c),
            _ => runs.push(std::mem::take(&mut cur)),
        }
    }
    runs.push(cur);
    let mut out: Vec<String> = runs.iter().filter(|r| r.chars().count() >= NGRAM).flat_map(|r| ngrams(r)).collect();
    out.sort();
    out.dedup();
    out
}

/// Glob match (`*` any run, `?` one char, `\` escapes).
pub fn wildcard_match(pattern: &str, value: &str, case_insensitive: bool) -> bool {
    let (p, v): (Vec<char>, Vec<char>) = if case_insensitive {
        (pattern.to_lowercase().chars().collect(), value.to_lowercase().chars().collect())
    } else {
        (pattern.chars().collect(), value.chars().collect())
    };
    // Tokenize the pattern: Some(c) literal, None = '?', and '*' markers.
    #[derive(Clone, Copy, PartialEq)]
    enum T {
        Lit(char),
        One,
        Any,
    }
    let mut toks = Vec::with_capacity(p.len());
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            '\\' if i + 1 < p.len() => {
                toks.push(T::Lit(p[i + 1]));
                i += 2;
                continue;
            }
            '*' => toks.push(T::Any),
            '?' => toks.push(T::One),
            c => toks.push(T::Lit(c)),
        }
        i += 1;
    }
    // Iterative glob with backtracking to the last '*'.
    let (mut ti, mut vi) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while vi < v.len() {
        if ti < toks.len() {
            match toks[ti] {
                T::Any => {
                    star = Some((ti, vi));
                    ti += 1;
                    continue;
                }
                T::One => {
                    ti += 1;
                    vi += 1;
                    continue;
                }
                T::Lit(c) if c == v[vi] => {
                    ti += 1;
                    vi += 1;
                    continue;
                }
                _ => {}
            }
        }
        match star {
            Some((st, sv)) => {
                ti = st + 1;
                vi = sv + 1;
                star = Some((st, sv + 1));
            }
            None => return false,
        }
    }
    while ti < toks.len() && toks[ti] == T::Any {
        ti += 1;
    }
    ti == toks.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grams_and_match() {
        assert_eq!(ngrams("Abcd"), vec!["abc", "bcd"]);
        assert_eq!(ngrams("ab"), vec!["ab"]);
        assert_eq!(required_ngrams("*foo*ba?r"), vec!["foo"]);
        assert_eq!(required_ngrams("a*b"), Vec::<String>::new());
        assert!(wildcard_match("*foo*", "xxfooyy", false));
        assert!(wildcard_match("f?o", "foo", false));
        assert!(!wildcard_match("f?o", "fooo", false));
        assert!(wildcard_match("a\\*b", "a*b", false));
        assert!(!wildcard_match("a\\*b", "axb", false));
        assert!(wildcard_match("*FOO", "barfoo", true));
        assert!(wildcard_match("*", "", false));
        assert!(wildcard_match("a*b*c", "aXXbYYc", false));
        assert!(!wildcard_match("a*b*c", "aXXbYY", false));
        assert_eq!(regex_required_ngrams("foo.*bar"), vec!["bar", "foo"]);
        assert!(regex_required_ngrams("foo|bar").is_empty());
    }
}
