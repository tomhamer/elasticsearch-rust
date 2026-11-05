//! Synonym expansion (Lucene `SynonymFilter` / `SynonymGraphFilter`,
//! Solr rule format).
//!
//! # Rule format
//!
//! * `a, b, c` — equivalent terms. With `expand` (default) each term maps
//!   to every term in the set; without it each term maps to the first.
//! * `a, b => c, d` — explicit mapping: any term on the left is replaced
//!   by every term on the right (the original is dropped). `expand` has
//!   no effect on explicit mappings.
//! * Multi-word terms are written with spaces: `sea biscuit, seabiscuit`.
//! * `#` starts a comment; blank lines are ignored; `\,`, `\=`, `\ `
//!   escapes are honoured.
//!
//! Matching is greedy, longest-input-first, on consecutive tokens (by
//! index, regardless of position gaps).

use std::collections::HashMap;

use crate::analysis::{Token, TokenFilter};

/// A synonym rule set: input phrase (as a sequence of terms) → output
/// phrases.
#[derive(Clone, Debug, Default)]
pub struct SynonymMap {
    map: HashMap<Vec<String>, Vec<Vec<String>>>,
    max_len: usize,
}

impl SynonymMap {
    /// Parse Solr-format rules. With `lenient`, malformed rules are
    /// skipped instead of returning an error.
    pub fn parse(rules: &[&str], expand: bool, lenient: bool) -> Result<Self, String> {
        let mut m = Self::default();
        for (lineno, raw) in rules.iter().enumerate() {
            let line = strip_comment(raw);
            if line.trim().is_empty() {
                continue;
            }
            let res = parse_rule(line, expand);
            match res {
                Ok(pairs) => {
                    for (input, output) in pairs {
                        m.add(input, output);
                    }
                }
                Err(e) => {
                    if !lenient {
                        return Err(format!("synonym rule {}: {e}", lineno + 1));
                    }
                }
            }
        }
        Ok(m)
    }

    /// Add one mapping `input → output`.
    pub fn add(&mut self, input: Vec<String>, output: Vec<String>) {
        if input.is_empty() || output.is_empty() {
            return;
        }
        self.max_len = self.max_len.max(input.len());
        let outs = self.map.entry(input).or_default();
        if !outs.contains(&output) {
            outs.push(output);
        }
    }

    /// Number of distinct input phrases.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no rules are present.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

fn strip_comment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

/// Split on an unescaped delimiter, honouring `\` escapes, and trimming.
fn split_unescaped(s: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                cur.push(n);
            }
        } else if c == delim {
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    parts.push(cur);
    parts.into_iter().map(|p| p.trim().to_string()).collect()
}

/// Split a term into whitespace-separated words.
fn words(term: &str) -> Vec<String> {
    term.split_whitespace().map(|w| w.to_string()).collect()
}

/// Parse one rule line into `(input, output)` pairs.
fn parse_rule(line: &str, expand: bool) -> Result<Vec<(Vec<String>, Vec<String>)>, String> {
    let sides: Vec<&str> = {
        // Split on an unescaped "=>".
        let mut v = Vec::new();
        let mut last = 0;
        let bytes = line.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'=' && bytes[i + 1] == b'>' {
                v.push(&line[last..i]);
                last = i + 2;
                i += 2;
                continue;
            }
            i += 1;
        }
        v.push(&line[last..]);
        v
    };
    let mut pairs = Vec::new();
    match sides.len() {
        1 => {
            let terms: Vec<Vec<String>> = split_unescaped(sides[0], ',')
                .iter()
                .filter(|t| !t.is_empty())
                .map(|t| words(t))
                .collect();
            if terms.is_empty() {
                return Err("empty rule".into());
            }
            if expand {
                for input in &terms {
                    for output in &terms {
                        pairs.push((input.clone(), output.clone()));
                    }
                }
            } else {
                for input in &terms {
                    pairs.push((input.clone(), terms[0].clone()));
                }
            }
        }
        2 => {
            let inputs: Vec<Vec<String>> =
                split_unescaped(sides[0], ',').iter().filter(|t| !t.is_empty()).map(|t| words(t)).collect();
            let outputs: Vec<Vec<String>> =
                split_unescaped(sides[1], ',').iter().filter(|t| !t.is_empty()).map(|t| words(t)).collect();
            if inputs.is_empty() || outputs.is_empty() {
                return Err(format!("malformed explicit mapping: {line:?}"));
            }
            for input in &inputs {
                for output in &outputs {
                    pairs.push((input.clone(), output.clone()));
                }
            }
        }
        _ => return Err(format!("more than one '=>' in rule: {line:?}")),
    }
    Ok(pairs)
}

/// Token filter applying a [`SynonymMap`] (ES `synonym`).
///
/// For an equivalent-set match the original tokens are kept and the
/// synonyms are added at the same positions; for an explicit mapping
/// (`a => b`) the input is replaced. See the module docs for how
/// multi-word outputs are positioned.
#[derive(Clone, Debug)]
pub struct SynonymFilter {
    /// The rules.
    pub synonyms: SynonymMap,
    /// Compare token text and rule terms case-insensitively.
    pub ignore_case: bool,
    /// Keep the original tokens (false only for explicit `=>` mappings,
    /// which always replace; this flag governs equivalent sets).
    keep_original: bool,
    explicit: HashMap<Vec<String>, bool>,
}

impl SynonymFilter {
    /// Build from Solr-format rules.
    pub fn parse(rules: &[&str], expand: bool, lenient: bool) -> Result<Self, String> {
        let mut explicit = HashMap::new();
        for raw in rules {
            let line = strip_comment(raw);
            if line.contains("=>") {
                if let Ok(pairs) = parse_rule(line, expand) {
                    for (input, _) in pairs {
                        explicit.insert(input, true);
                    }
                }
            }
        }
        let synonyms = SynonymMap::parse(rules, expand, lenient)?;
        Ok(Self { synonyms, ignore_case: false, keep_original: true, explicit })
    }

    /// Builder: case-insensitive matching. Rule terms (inputs and
    /// outputs) are lowercased, as Lucene does with `ignore_case`.
    pub fn ignore_case(mut self, yes: bool) -> Self {
        self.ignore_case = yes;
        if yes {
            let mut m = SynonymMap::default();
            for (k, outs) in &self.synonyms.map {
                let k2: Vec<String> = k.iter().map(|w| w.to_lowercase()).collect();
                for o in outs {
                    m.add(k2.clone(), o.iter().map(|w| w.to_lowercase()).collect());
                }
            }
            self.synonyms = m;
            self.explicit = self
                .explicit
                .iter()
                .map(|(k, v)| (k.iter().map(|w| w.to_lowercase()).collect(), *v))
                .collect();
        }
        self
    }

    fn key(&self, t: &Token) -> String {
        if self.ignore_case { t.text.to_lowercase() } else { t.text.clone() }
    }
}

impl TokenFilter for SynonymFilter {
    fn clone_box(&self) -> Box<dyn TokenFilter> {
        Box::new(self.clone())
    }

    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        if self.synonyms.is_empty() {
            return tokens;
        }
        let keys: Vec<String> = tokens.iter().map(|t| self.key(t)).collect();
        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut shift: u32 = 0; // accumulated position shift
        let mut i = 0;
        while i < tokens.len() {
            // Longest match first.
            let max = self.synonyms.max_len.min(tokens.len() - i);
            let mut matched: Option<(usize, &Vec<Vec<String>>)> = None;
            for len in (1..=max).rev() {
                let slice = &keys[i..i + len];
                if let Some(outs) = self.synonyms.map.get(slice) {
                    matched = Some((len, outs));
                    break;
                }
            }
            match matched {
                None => {
                    let mut t = tokens[i].clone();
                    t.position += shift;
                    out.push(t);
                    i += 1;
                }
                Some((len, outs)) => {
                    let input = &tokens[i..i + len];
                    let base_pos = input[0].position + shift;
                    let is_explicit = self.explicit.contains_key(&keys[i..i + len].to_vec());
                    let mut longest_out = 0usize;
                    if self.keep_original && !is_explicit {
                        for (k, t) in input.iter().enumerate() {
                            let mut t = t.clone();
                            t.position = base_pos + k as u32;
                            out.push(t);
                        }
                    }
                    for o in outs {
                        // Skip outputs identical to the input (already emitted).
                        if !is_explicit && o.len() == len && o.iter().zip(&keys[i..i + len]).all(|(a, b)| a == b) {
                            continue;
                        }
                        longest_out = longest_out.max(o.len());
                        for (k, word) in o.iter().enumerate() {
                            out.push(Token {
                                text: word.clone(),
                                position: base_pos + k as u32,
                                offset_from: input[0].offset_from,
                                offset_to: input[len - 1].offset_to,
                            });
                        }
                    }
                    if longest_out > len {
                        shift += (longest_out - len) as u32;
                    }
                    i += len;
                }
            }
        }
        // Keep stream ordered by position (stable), as Lucene emits it.
        out.sort_by_key(|t| t.position);
        out
    }
}

/// ES `synonym_graph`. Identical to [`SynonymFilter`] here: without a
/// `position_length` attribute on [`Token`] no extra graph information
/// can be carried (see the module docs).
pub type SynonymGraphFilter = SynonymFilter;

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(words: &[&str]) -> Vec<Token> {
        let mut off = 0;
        words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let t = Token { text: w.to_string(), position: i as u32, offset_from: off, offset_to: off + w.len() as u32 };
                off += w.len() as u32 + 1;
                t
            })
            .collect()
    }
    fn texts(t: &[Token]) -> Vec<&str> {
        t.iter().map(|t| t.text.as_str()).collect()
    }
    fn positions(t: &[Token]) -> Vec<u32> {
        t.iter().map(|t| t.position).collect()
    }

    #[test]
    fn parse_equivalent_expand() {
        let m = SynonymMap::parse(&["a, b, c"], true, false).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.map[&vec!["a".to_string()]].len(), 3);
    }

    #[test]
    fn parse_equivalent_no_expand() {
        let m = SynonymMap::parse(&["a, b, c"], false, false).unwrap();
        assert_eq!(m.map[&vec!["b".to_string()]], vec![vec!["a".to_string()]]);
    }

    #[test]
    fn parse_explicit_and_comments() {
        let m = SynonymMap::parse(&["# comment", "", "i-pod, i pod => ipod # trailing"], true, false).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.map[&vec!["i".to_string(), "pod".to_string()]], vec![vec!["ipod".to_string()]]);
    }

    #[test]
    fn parse_errors_and_lenient() {
        assert!(SynonymMap::parse(&["a => "], true, false).is_err());
        assert!(SynonymMap::parse(&["a => b => c"], true, false).is_err());
        let m = SynonymMap::parse(&["a => ", "x, y"], true, true).unwrap();
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn parse_escapes() {
        let m = SynonymMap::parse(&[r"a\,b, c\=>d"], true, false).unwrap();
        assert!(m.map.contains_key(&vec!["a,b".to_string()]));
        assert!(m.map.contains_key(&vec!["c=>d".to_string()]));
    }

    #[test]
    fn equivalent_set_adds_synonyms_at_same_position() {
        let f = SynonymFilter::parse(&["quick, fast, speedy"], true, false).unwrap();
        let out = f.filter(toks(&["the", "quick", "fox"]));
        assert_eq!(texts(&out), vec!["the", "quick", "fast", "speedy", "fox"]);
        assert_eq!(positions(&out), vec![0, 1, 1, 1, 2]);
        assert_eq!((out[2].offset_from, out[2].offset_to), (4, 9));
    }

    #[test]
    fn explicit_mapping_replaces() {
        let f = SynonymFilter::parse(&["i-pod, i pod => ipod"], true, false).unwrap();
        let out = f.filter(toks(&["my", "i", "pod", "and", "i-pod"]));
        assert_eq!(texts(&out), vec!["my", "ipod", "and", "ipod"]);
        assert_eq!(positions(&out), vec![0, 1, 3, 4]);
        // Output positions after a 2→1 contraction are not shifted (Lucene keeps them).
        let out = f.filter(toks(&["i", "pod", "x"]));
        assert_eq!(texts(&out), vec!["ipod", "x"]);
        assert_eq!(positions(&out), vec![0, 2]);
    }

    #[test]
    fn multi_word_output_consecutive_positions_and_shift() {
        let f = SynonymFilter::parse(&["dns => domain name system"], true, false).unwrap();
        let out = f.filter(toks(&["dns", "is", "down"]));
        assert_eq!(texts(&out), vec!["domain", "name", "system", "is", "down"]);
        assert_eq!(positions(&out), vec![0, 1, 2, 3, 4]);
        // Equivalent set with a multi-word member keeps the original.
        let f = SynonymFilter::parse(&["dns, domain name system"], true, false).unwrap();
        let out = f.filter(toks(&["dns", "is"]));
        assert_eq!(texts(&out), vec!["dns", "domain", "name", "system", "is"]);
        assert_eq!(positions(&out), vec![0, 0, 1, 2, 3]);
    }

    #[test]
    fn multi_word_input_match_is_longest_first() {
        let f = SynonymFilter::parse(&["sea biscuit, seabiscuit", "sea, ocean"], true, false).unwrap();
        let out = f.filter(toks(&["sea", "biscuit"]));
        assert_eq!(texts(&out), vec!["sea", "seabiscuit", "biscuit"]);
        assert_eq!(positions(&out), vec![0, 0, 1]);
        let out = f.filter(toks(&["sea", "shell"]));
        assert_eq!(texts(&out), vec!["sea", "ocean", "shell"]);
    }

    #[test]
    fn ignore_case() {
        let f = SynonymFilter::parse(&["Quick, Fast"], true, false).unwrap().ignore_case(true);
        assert_eq!(texts(&f.filter(toks(&["QUICK"]))), vec!["QUICK", "fast"]);
        let f = SynonymFilter::parse(&["Quick, Fast"], true, false).unwrap();
        assert_eq!(texts(&f.filter(toks(&["QUICK"]))), vec!["QUICK"]);
    }

    #[test]
    fn no_expand_maps_to_first() {
        let f = SynonymFilter::parse(&["usa, united states, america"], false, false).unwrap();
        assert_eq!(texts(&f.filter(toks(&["america"]))), vec!["america", "usa"]);
        assert_eq!(texts(&f.filter(toks(&["usa"]))), vec!["usa"]);
    }

    #[test]
    fn empty_rules_passthrough_and_clone() {
        let f = SynonymFilter::parse(&[], true, false).unwrap();
        let input = toks(&["a", "b"]);
        assert_eq!(f.filter(input.clone()), input);
        let g: SynonymGraphFilter = SynonymFilter::parse(&["a, b"], true, false).unwrap();
        assert_eq!(texts(&g.clone_box().filter(toks(&["a"]))), vec!["a", "b"]);
    }
}
