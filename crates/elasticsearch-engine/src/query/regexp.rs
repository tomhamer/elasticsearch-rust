//! Lucene `RegExp` front end for the `regexp` query.

use super::dsl::QueryContext;
use crate::error::{EsError, Result};
use rustic_core::automaton::regexp::{RegexpFlags, RegexpMatcher};
use rustic_core::query::{MinShouldMatch, MultiTermRewrite, Query};

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

pub fn regexp_query(field: u16, pattern: &str, flags: Option<&str>, ci: bool, rewrite: MultiTermRewrite, ctx: &QueryContext) -> Result<Query> {
    let f = parse_flags(flags)?;
    let operands = if f.intersection && !split_top(pattern, '|').len().gt(&1) { split_top(pattern, '&') } else { vec![pattern.to_string()] };
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
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
        let q = Query::Regexp { field, pattern: t, case_insensitive: ci, max_expansions: usize::MAX, rewrite };
        if negated { neg.push(q) } else { pos.push(q) }
    }
    if neg.is_empty() && pos.len() == 1 {
        return Ok(pos.pop().unwrap());
    }
    if pos.is_empty() {
        pos.push(Query::Exists { field });
    }
    let b = Query::Bool { must: pos, should: vec![], must_not: neg, filter: vec![], minimum_should_match: MinShouldMatch::Count(0) };
    Ok(Query::ConstScore { query: Box::new(b), score: 1.0, point_set: false })
}
