//! Lucene `RegExp` front end for the `regexp` query.

use super::dsl::QueryContext;
use crate::error::{EsError, Result};
use rustic_core::automaton::regexp::{RegexpFlags, RegexpMatcher};
use rustic_core::query::{MultiTermRewrite, Query};

#[derive(Clone, Copy)]
struct Flags {
    intersection: bool,
    complement: bool,
    empty: bool,
    anystring: bool,
}

fn parse_flags(spec: Option<&str>) -> Result<Flags> {
    let mut f = Flags { intersection: true, complement: true, empty: true, anystring: true };
    let Some(spec) = spec else { return Ok(f) };
    f = Flags { intersection: false, complement: false, empty: false, anystring: false };
    for part in spec.split('|') {
        match part.trim() {
            "ALL" => f = Flags { intersection: true, complement: true, empty: true, anystring: true },
            "NONE" | "INTERVAL" => {}
            "INTERSECTION" => f.intersection = true,
            "COMPLEMENT" => f.complement = true,
            "EMPTY" => f.empty = true,
            "ANYSTRING" => f.anystring = true,
            other => return Err(EsError::illegal_argument(format!("Unknown regexp flag [{other}]"))),
        }
    }
    Ok(f)
}

/// Split at top-level `&` (outside groups, classes, quotes and escapes).
fn split_top(pattern: &str, sep: char) -> Vec<String> {
    let (mut parts, mut cur) = (Vec::new(), String::new());
    let (mut depth, mut in_class, mut in_quote, mut esc) = (0i32, false, false, false);
    for c in pattern.chars() {
        if esc {
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if in_quote {
            in_quote = c != '"';
        } else if in_class {
            in_class = c != ']';
        } else {
            match c {
                '"' => in_quote = true,
                '[' => in_class = true,
                '(' => depth += 1,
                ')' => depth -= 1,
                c if c == sep && depth == 0 => {
                    parts.push(std::mem::take(&mut cur));
                    continue;
                }
                _ => {}
            }
        }
        cur.push(c);
    }
    parts.push(cur);
    parts
}

fn escape_lit(c: char, out: &mut String) {
    if ".?+*|{}[]()\"\\^$".contains(c) {
        out.push('\\');
    }
    out.push(c);
}

/// Lucene syntax to the core engine's syntax.
fn translate(p: &str, f: Flags) -> String {
    let (mut out, mut in_class, mut esc) = (String::new(), false, false);
    let mut it = p.chars().peekable();
    while let Some(c) = it.next() {
        if esc {
            esc = false;
            out.push(c);
            continue;
        }
        if c == '\\' {
            esc = true;
            out.push(c);
            continue;
        }
        if in_class {
            in_class = c != ']';
            out.push(c);
            continue;
        }
        match c {
            '[' => {
                in_class = true;
                out.push(c);
                // `[^...]` keeps its negation; a leading `]` is literal.
                if it.peek() == Some(&'^') {
                    out.push(it.next().unwrap());
                }
            }
            '"' => {
                for q in it.by_ref() {
                    if q == '"' {
                        break;
                    }
                    escape_lit(q, &mut out);
                }
            }
            '^' | '$' => {
                out.push('\\');
                out.push(c);
            }
            '@' if f.anystring => out.push_str("(?s:.*)"),
            '#' if f.empty => out.push_str("[^\\x00-\\x{10FFFF}]"),
            _ => out.push(c),
        }
    }
    out
}

/// A Lucene `RegExp` (all syntax flags, as ES|QL `RLIKE` compiles it) as
/// the core engine's whole-term pattern ([`RegexpMatcher`] syntax, an
/// intersection of operands where the pattern has one). `Err`: invalid.
pub fn lucene_term_pattern(pattern: &str) -> std::result::Result<String, String> {
    let f = Flags { intersection: true, complement: true, empty: true, anystring: true };
    let operands = if split_top(pattern, '|').len() > 1 { vec![pattern.to_string()] } else { split_top(pattern, '&') };
    let mut parts: Vec<(bool, String)> = Vec::new();
    for op in &operands {
        let (negated, body) = match op.strip_prefix('~') {
            Some(rest) => (true, rest),
            None => (false, op.as_str()),
        };
        // (Compiled, with the caller's memory budget, by the caller.)
        parts.push((negated, translate(body, f)));
    }
    Ok(match parts.as_slice() {
        [(false, t)] => t.clone(),
        _ => rustic_core::automaton::regexp::intersection(&parts),
    })
}

/// A Lucene `regexp` applied to whole field values (wildcard,
/// constant_keyword and flattened-key fields, which verify values rather
/// than enumerate terms): the same grammar, flags, `&` / `~` operators,
/// quoting and `@` as [`regexp_query`].
pub struct ValueMatcher {
    pos: Vec<RegexpMatcher>,
    neg: Vec<RegexpMatcher>,
    /// Literal runs every match must contain (empty when case-insensitive
    /// or not derivable): an n-gram prefilter may require them.
    pub required: Vec<String>,
}

impl ValueMatcher {
    pub fn new(pattern: &str, flags: Option<&str>, ci: bool) -> std::result::Result<ValueMatcher, String> {
        use rustic_core::automaton::TermMatcher as _;
        let f = parse_flags(flags).map_err(|e| e.to_string())?;
        let operands = if f.intersection && split_top(pattern, '|').len() <= 1 { split_top(pattern, '&') } else { vec![pattern.to_string()] };
        let mut m = ValueMatcher { pos: Vec::new(), neg: Vec::new(), required: Vec::new() };
        for op in &operands {
            let (negated, body) = match op.strip_prefix('~') {
                Some(rest) if f.complement => (true, rest),
                _ => (false, op.as_str()),
            };
            let t = translate(body, f);
            let matcher = RegexpMatcher::new(&t, RegexpFlags { case_insensitive: ci })?;
            if negated {
                m.neg.push(matcher);
            } else {
                if !ci {
                    m.required.extend(rustic_core::automaton::regexp::required_literals(&t).unwrap_or_default());
                }
                m.pos.push(matcher);
            }
        }
        Ok(m)
    }

    pub fn is_match(&self, value: &str) -> bool {
        use rustic_core::automaton::TermMatcher as _;
        self.pos.iter().all(|m| m.matches(value.as_bytes())) && !self.neg.iter().any(|m| m.matches(value.as_bytes()))
    }
}

pub fn regexp_query(field: u16, pattern: &str, flags: Option<&str>, ci: bool, rewrite: MultiTermRewrite, ctx: &QueryContext) -> Result<Query> {
    let f = parse_flags(flags)?;
    let operands = if f.intersection && !split_top(pattern, '|').len().gt(&1) { split_top(pattern, '&') } else { vec![pattern.to_string()] };
    let mut parts: Vec<(bool, String)> = Vec::new();
    for op in &operands {
        let (negated, body) = match op.strip_prefix('~') {
            Some(rest) if f.complement => (true, rest),
            _ => (false, op.as_str()),
        };
        let t = translate(body, f);
        if let Err(e) = RegexpMatcher::new(&t, RegexpFlags { case_insensitive: ci }) {
            let reason = if body.matches('(').count() > body.matches(')').count() {
                format!("expected ')' at position {}", body.chars().count())
            } else {
                e
            };
            return Err(ctx.shard_err_pub(format!("failed to create query: {reason}")));
        }
        parts.push((negated, t));
    }
    // Intersection and complement are operations of the term language
    // (Lucene `RegExp`): one matcher applies them to each term; a document
    // matches when one of its terms does.
    let pattern = match parts.as_slice() {
        [(false, t)] => t.clone(),
        _ => rustic_core::automaton::regexp::intersection(&parts),
    };
    Ok(Query::Regexp { field, pattern, case_insensitive: ci, max_expansions: usize::MAX, rewrite })
}

#[cfg(test)]
mod term_language_tests {
    use serde_json::{json, Value};

    fn ids(node: &crate::node::Node, query: Value) -> Vec<String> {
        let svc = node.get_index("re").unwrap();
        let req = crate::search::parse_request(&json!({"query": query, "size": 100}), &std::collections::HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    }

    /// Review finding 322: regexp intersection and complement apply to each
    /// term (Lucene's term language), not to whole documents: `foo.*&.*bar`
    /// needs one term matching both, `~(foo)` any term other than `foo`;
    /// keyword arrays and text tokens alike; one-term controls.
    #[test]
    fn intersection_and_complement_are_per_term() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index("re", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"k": {"type": "keyword"}, "t": {"type": "text"}}}})).unwrap();
        let svc = node.get_index("re").unwrap();
        let docs = [
            ("split", json!({"k": ["fooX", "Ybar"], "t": "fooX Ybar"})),
            ("both", json!({"k": ["foobar"], "t": "foobar"})),
            ("foo_and_bar", json!({"k": ["foo", "bar"], "t": "foo bar"})),
            ("only_foo", json!({"k": ["foo"], "t": "foo"})),
        ];
        let reqs = docs.iter().map(|(id, d)| crate::index::WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        for f in ["k", "t"] {
            assert_eq!(ids(&node, json!({"regexp": {f: "foo.*&.*bar"}})), vec!["both"], "{f}");
            assert_eq!(ids(&node, json!({"regexp": {f: "~(foo)"}})), vec!["both", "foo_and_bar", "split"], "{f}");
            assert_eq!(ids(&node, json!({"regexp": {f: "foo.*&~(foobar)"}})), vec!["foo_and_bar", "only_foo", "split"], "{f}");
            // Controls: plain patterns.
            assert_eq!(ids(&node, json!({"regexp": {f: "foo"}})), vec!["foo_and_bar", "only_foo"], "{f}");
            assert_eq!(ids(&node, json!({"regexp": {f: ".*bar"}})), vec!["both", "foo_and_bar", "split"], "{f}");
        }
        // Keyword values are case-sensitive: `Ybar` matches `.*bar`.
    }
}

#[cfg(test)]
mod multi_term_rewrite_tests {
    use serde_json::{json, Value};

    fn hits(node: &crate::node::Node, query: Value) -> (usize, Vec<String>) {
        let svc = node.get_index("mt").unwrap();
        let req = crate::search::parse_request(&json!({"query": query, "size": 1000, "track_total_hits": true}), &std::collections::HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        (v["hits"]["total"]["value"].as_u64().unwrap() as usize, ids)
    }

    /// Review findings 321 / 346 / 345: streamed multi-term rewrites give
    /// the same matches: constant-score unions of few (lazy) and many
    /// (bitset) terms, overlapping terms counted once; a top-terms rewrite
    /// keeps exactly its best n terms; a fuzzy query over long terms scores
    /// with the bounded distance.
    #[test]
    fn streamed_rewrites_keep_their_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index("mt", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"k": {"type": "keyword"}, "t": {"type": "text"}}}})).unwrap();
        let svc = node.get_index("mt").unwrap();
        let long = "q".repeat(20_000);
        let mut docs: Vec<(String, Value)> = (0..300).map(|i| (format!("d{i}"), json!({"k": [format!("term{i:03}"), format!("term{:03}", (i + 1) % 300)], "t": format!("alpha{} alpha{}", i % 7, i % 5)}))).collect();
        docs.push(("long_eq".into(), json!({"k": long.clone()})));
        docs.push(("long_one".into(), json!({"k": format!("{}r", &long[..19_999])})));
        let reqs = docs.iter().map(|(id, d)| crate::index::WriteReq::Index { id: Some(id.clone()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        // Many terms (bitset), overlapping per document.
        assert_eq!(hits(&node, json!({"prefix": {"k": "term"}})).0, 300);
        assert_eq!(hits(&node, json!({"wildcard": {"k": "term1*"}})).0, 101, "terms 100-199 and their neighbours");
        // Few terms (lazy union).
        assert_eq!(hits(&node, json!({"prefix": {"k": "term00"}})).1.len(), 11);
        assert_eq!(hits(&node, json!({"prefix": {"t": "alpha"}})).0, 300);
        // Top terms: exactly the best n terms (all at doc_freq 2: by text).
        let (n, _) = hits(&node, json!({"prefix": {"k": {"value": "term", "rewrite": "top_terms_5"}}}));
        assert_eq!(n, 6, "terms 000-004 cover docs d0-d4 and d299");
        // Fuzzy over long terms: the equal and the one-edit term.
        let (_, ids) = hits(&node, json!({"fuzzy": {"k": {"value": long, "fuzziness": 2}}}));
        assert_eq!(ids, vec!["long_eq", "long_one"]);
    }
}

/// Lucene `<n-m>` numeric intervals (outside classes, quotes and escapes)
/// as alternations of the decimal numbers they match: with `n` and `m` of
/// the same length the numbers are zero-padded to it, otherwise any
/// number of leading zeros is allowed (Lucene `RegExp.makeDecimalInterval`).
/// `None` for a malformed interval or one too large to expand.
pub(crate) fn expand_intervals(p: &str) -> Option<String> {
    const MAX: u64 = 10_000;
    let (mut out, mut in_class, mut in_quote, mut esc) = (String::new(), false, false, false);
    let chars: Vec<char> = p.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        if esc { esc = false; out.push(c); continue; }
        if c == '\\' { esc = true; out.push(c); continue; }
        if in_quote { in_quote = c != '"'; out.push(c); continue; }
        if in_class { in_class = c != ']'; out.push(c); continue; }
        match c {
            '"' => { in_quote = true; out.push(c); }
            '[' => { in_class = true; out.push(c); }
            '<' => {
                let close = chars[i..].iter().position(|x| *x == '>')? + i;
                let body: String = chars[i..close].iter().collect();
                let (lo, hi) = body.split_once('-')?;
                if lo.is_empty() || hi.is_empty() || !lo.chars().all(|d| d.is_ascii_digit()) || !hi.chars().all(|d| d.is_ascii_digit()) {
                    return None;
                }
                let (a, b): (u64, u64) = (lo.parse().ok()?, hi.parse().ok()?);
                let (a, b) = (a.min(b), a.max(b));
                if b - a > MAX { return None; }
                let width = if lo.len() == hi.len() { lo.len() } else { 0 };
                let alts: Vec<String> = (a..=b).map(|n| if width > 0 { format!("{n:0width$}") } else { n.to_string() }).collect();
                if width > 0 { out.push_str(&format!("(?:{})", alts.join("|"))); } else { out.push_str(&format!("0*(?:{})", alts.join("|"))); }
                i = close + 1;
            }
            _ => out.push(c),
        }
    }
    Some(out)
}

/// A Lucene `RegExp` (ES role index-name expressions, all syntax flags) as
/// top-level intersection operands, each `(complemented, rust regex)`.
/// `None` when the expression uses a construct this cannot express
/// faithfully (a complement or intersection below the top level, an
/// unexpandable interval): callers must not reinterpret it.
pub(crate) fn lucene_operands(p: &str) -> Option<Vec<(bool, String)>> {
    let f = Flags { intersection: true, complement: true, empty: true, anystring: true };
    let operands = if split_top(p, '|').len() > 1 { vec![p.to_string()] } else { split_top(p, '&') };
    let mut out = Vec::new();
    for op in operands {
        let (negated, body) = match op.strip_prefix('~') { Some(rest) => (true, rest.to_string()), None => (false, op.clone()) };
        let body = expand_intervals(&body)?;
        let t = translate(&body, f);
        // A remaining unescaped `~` or `&` (inside a group or alternation)
        // is a complement / intersection Rust regex would read literally.
        let (mut in_class, mut esc) = (false, false);
        for c in t.chars() {
            if esc { esc = false; continue; }
            if c == '\\' { esc = true; continue; }
            if in_class { in_class = c != ']'; continue; }
            match c { '[' => in_class = true, '~' | '&' => return None, _ => {} }
        }
        out.push((negated, t));
    }
    Some(out)
}
