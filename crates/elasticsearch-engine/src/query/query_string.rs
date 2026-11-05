//! `query_string` (Lucene classic syntax) and `simple_query_string` parsers.
//!
//! Supported: field prefixes (`title:foo`, `title.*:foo`), `AND`/`OR`/`NOT`,
//! `&&`/`||`/`!`, `+`/`-`, grouping, phrases with `~slop`, fuzzy `~N`,
//! wildcards `*`/`?`, regex `/.../`, ranges `[a TO b]` / `{a TO b}` /
//! `>=x`, boosts `^n`, `_exists_:field`, escapes.

use super::dsl::{expand_fields, match_field, parse_fuzziness, parse_msm, MatchKind, MatchOpts, QueryContext};
use crate::error::{EsError, Result};
use rustic_core::query::{Fuzziness, MinShouldMatch, MultiTermRewrite, Query};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Term { field: Option<String>, text: String, fuzzy: Option<String>, boost: f32, wildcard: bool },
    Phrase { field: Option<String>, text: String, slop: Option<u32>, boost: f32 },
    Regex { field: Option<String>, pattern: String, boost: f32 },
    Range { field: Option<String>, lo: Option<String>, hi: Option<String>, inc_lo: bool, inc_hi: bool, boost: f32 },
    Exists { field: String },
    MatchAll,
    Group { field: Option<String>, clauses: Vec<(Occur, Node)>, boost: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Occur {
    Must,
    Should,
    MustNot,
}

#[derive(Clone, Copy, PartialEq)]
enum Conj {
    None,
    And,
    Or,
}

/// Deepest `(...)` nesting a `query_string` / `simple_query_string` may
/// have. Parsing, translation and scoring all recurse per level; without a
/// bound a query of ~50k `(` (well under the body limit) overflows the
/// thread stack, which aborts the node.
pub(crate) const MAX_GROUP_DEPTH: usize = 256;

fn too_deep() -> EsError {
    EsError::parsing(format!("Cannot parse query: groups are nested more than {MAX_GROUP_DEPTH} levels deep"))
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    default_and: bool,
    depth: usize,
}


impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn starts(&self, w: &str) -> bool {
        let cs: Vec<char> = w.chars().collect();
        if self.s.len() < self.i + cs.len() || self.s[self.i..self.i + cs.len()] != cs[..] {
            return false;
        }
        self.s.get(self.i + cs.len()).map(|c| c.is_whitespace() || *c == '(' ).unwrap_or(true)
    }

    fn query(&mut self, field: Option<&str>) -> Result<Vec<(Occur, Node)>> {
        let mut clauses: Vec<(Occur, Node)> = Vec::new();
        let mut conj_prev = Conj::None;
        loop {
            self.ws();
            match self.peek() {
                None | Some(')') => break,
                _ => {}
            }
            let mut conj = Conj::None;
            if self.starts("AND") || self.peek() == Some('&') {
                conj = Conj::And;
                self.i += if self.peek() == Some('&') { 2 } else { 3 };
                self.ws();
            } else if self.starts("OR") || (self.peek() == Some('|') && self.s.get(self.i + 1) == Some(&'|')) {
                conj = Conj::Or;
                self.i += 2;
                self.ws();
            }
            let mut modifier: Option<Occur> = None;
            if self.starts("NOT") {
                modifier = Some(Occur::MustNot);
                self.i += 3;
                self.ws();
            } else if self.peek() == Some('!') {
                modifier = Some(Occur::MustNot);
                self.i += 1;
            } else if self.peek() == Some('+') {
                modifier = Some(Occur::Must);
                self.i += 1;
            } else if self.peek() == Some('-') {
                modifier = Some(Occur::MustNot);
                self.i += 1;
            }
            let Some(node) = self.clause(field)? else {
                if conj != Conj::None || modifier.is_some() {
                    return Err(EsError::parsing(format!("Cannot parse query: Encountered \"<EOF>\" at line 1, column {}.", self.i)));
                }
                break;
            };
            // Lucene addClause semantics.
            if !clauses.is_empty() && conj == Conj::And {
                let last = clauses.last_mut().unwrap();
                if last.0 == Occur::Should {
                    last.0 = Occur::Must;
                }
            }
            if !clauses.is_empty() && !self.default_and && conj == Conj::Or {
                // keep previous as is
            } else if !clauses.is_empty() && self.default_and && conj == Conj::Or {
                let last = clauses.last_mut().unwrap();
                if last.0 == Occur::Must {
                    last.0 = Occur::Should;
                }
            }
            let occur = match modifier {
                Some(m) => m,
                None => {
                    if conj == Conj::And {
                        Occur::Must
                    } else if conj == Conj::Or {
                        Occur::Should
                    } else if self.default_and {
                        Occur::Must
                    } else {
                        Occur::Should
                    }
                }
            };
            let _ = conj_prev;
            conj_prev = conj;
            clauses.push((occur, node));
        }
        Ok(clauses)
    }

    fn read_word(&mut self) -> String {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                if let Some(n) = self.s.get(self.i + 1) {
                    out.push(*n);
                    self.i += 2;
                    continue;
                }
            }
            if c.is_whitespace() || "()[]{}^~\":".contains(c) {
                break;
            }
            out.push(c);
            self.i += 1;
        }
        out
    }

    fn suffix(&mut self) -> (Option<String>, f32) {
        let mut fuzzy = None;
        let mut boost = 1.0;
        loop {
            match self.peek() {
                Some('~') => {
                    self.i += 1;
                    let mut n = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() || c == '.' {
                            n.push(c);
                            self.i += 1;
                        } else {
                            break;
                        }
                    }
                    fuzzy = Some(if n.is_empty() { "AUTO".into() } else { n });
                }
                Some('^') => {
                    self.i += 1;
                    let mut n = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() || c == '.' {
                            n.push(c);
                            self.i += 1;
                        } else {
                            break;
                        }
                    }
                    boost = n.parse().unwrap_or(1.0);
                }
                _ => break,
            }
        }
        (fuzzy, boost)
    }

    fn clause(&mut self, field: Option<&str>) -> Result<Option<Node>> {
        self.ws();
        let Some(c) = self.peek() else { return Ok(None) };
        // field prefix?
        let save = self.i;
        let mut fld: Option<String> = field.map(str::to_string);
        if c != '(' && c != '"' && c != '[' && c != '{' && c != '/' {
            let w = self.read_word();
            if self.peek() == Some(':') && !w.is_empty() {
                self.i += 1;
                if w == "_exists_" {
                    let f = self.read_word();
                    return Ok(Some(Node::Exists { field: f }));
                }
                fld = Some(w);
            } else {
                self.i = save;
            }
        }
        self.ws();
        let Some(c) = self.peek() else { return Ok(None) };
        Ok(Some(match c {
            '(' => {
                self.i += 1;
                if self.depth >= MAX_GROUP_DEPTH {
                    return Err(too_deep());
                }
                self.depth += 1;
                let clauses = self.query(fld.as_deref());
                self.depth -= 1;
                let clauses = clauses?;
                if self.peek() == Some(')') {
                    self.i += 1;
                } else {
                    return Err(EsError::parsing("Cannot parse query: Encountered \"<EOF>\", expected \")\""));
                }
                let (_, boost) = self.suffix();
                Node::Group { field: fld, clauses, boost }
            }
            '"' => {
                self.i += 1;
                let mut text = String::new();
                while let Some(c) = self.peek() {
                    self.i += 1;
                    if c == '\\' {
                        if let Some(n) = self.peek() {
                            text.push(n);
                            self.i += 1;
                        }
                        continue;
                    }
                    if c == '"' {
                        break;
                    }
                    text.push(c);
                }
                let (f, boost) = self.suffix();
                Node::Phrase { field: fld, text, slop: f.and_then(|x| x.parse::<f32>().ok()).map(|x| x as u32), boost }
            }
            '/' => {
                self.i += 1;
                let mut p = String::new();
                while let Some(c) = self.peek() {
                    self.i += 1;
                    if c == '\\' {
                        if let Some(n) = self.peek() {
                            p.push('\\');
                            p.push(n);
                            self.i += 1;
                        }
                        continue;
                    }
                    if c == '/' {
                        break;
                    }
                    p.push(c);
                }
                let (_, boost) = self.suffix();
                Node::Regex { field: fld, pattern: p, boost }
            }
            '[' | '{' => {
                let inc_lo = c == '[';
                self.i += 1;
                self.ws();
                let lo = self.read_range_bound();
                self.ws();
                if self.starts("TO") {
                    self.i += 2;
                }
                self.ws();
                let hi = self.read_range_bound();
                self.ws();
                let inc_hi = match self.peek() {
                    Some(']') => true,
                    Some('}') => false,
                    _ => return Err(EsError::parsing("Cannot parse range query: missing closing bracket")),
                };
                self.i += 1;
                let (_, boost) = self.suffix();
                Node::Range { field: fld, lo: (lo != "*").then_some(lo), hi: (hi != "*").then_some(hi), inc_lo, inc_hi, boost }
            }
            '>' | '<' => {
                self.i += 1;
                let eq = self.peek() == Some('=');
                if eq {
                    self.i += 1;
                }
                let v = self.read_word();
                if c == '>' {
                    Node::Range { field: fld, lo: Some(v), hi: None, inc_lo: eq, inc_hi: true, boost: 1.0 }
                } else {
                    Node::Range { field: fld, lo: None, hi: Some(v), inc_lo: true, inc_hi: eq, boost: 1.0 }
                }
            }
            _ => {
                let start = self.i;
                let w = self.read_word();
                if w.is_empty() {
                    self.i = start + 1;
                    return Ok(Some(Node::Group { field: None, clauses: vec![], boost: 1.0 }));
                }
                let (fuzzy, boost) = self.suffix();
                let wildcard = w.contains('*') || w.contains('?');
                Node::Term { field: fld, text: w, fuzzy, boost, wildcard }
            }
        }))
    }

    fn read_range_bound(&mut self) -> String {
        let mut out = String::new();
        if self.peek() == Some('"') {
            self.i += 1;
            while let Some(c) = self.peek() {
                self.i += 1;
                if c == '"' {
                    break;
                }
                out.push(c);
            }
            return out;
        }
        while let Some(c) = self.peek() {
            if c.is_whitespace() || c == ']' || c == '}' {
                break;
            }
            out.push(c);
            self.i += 1;
        }
        out
    }
}

struct Translate<'c, 'a> {
    ctx: &'c QueryContext<'a>,
    default_fields: Vec<(String, f32)>,
    opts: MatchOpts,
    analyze_wildcard: bool,
    allow_leading_wildcard: bool,
    tie_breaker: f32,
    /// `type: cross_fields`: a term's per-field queries blend their statistics.
    cross_fields: bool,
    time_zone: Option<String>,
}

fn boosted(q: Query, b: f32) -> Query {
    if (b - 1.0).abs() < f32::EPSILON || matches!(q, Query::MatchNone) {
        q
    } else {
        q.boost(b)
    }
}

impl<'c, 'a> Translate<'c, 'a> {
    fn fields_for(&self, field: &Option<String>) -> Vec<(String, f32)> {
        match field {
            Some(f) => expand_fields(&[f.clone()], self.ctx),
            None => self.default_fields.clone(),
        }
    }

    fn per_field(&self, field: &Option<String>, f: impl Fn(&str) -> Result<Query>) -> Result<Query> {
        let fields = self.fields_for(field);
        let mut qs = Vec::new();
        for (name, b) in &fields {
            match f(name) {
                Ok(Query::MatchNone) => {}
                Ok(q) => qs.push(boosted(q, *b)),
                Err(_) if self.opts.lenient => {}
                Err(e) => return Err(e),
            }
        }
        Ok(match qs.len() {
            0 => Query::MatchNone,
            1 => qs.pop().unwrap(),
            _ => Query::DisMax { queries: qs, tie_breaker: self.tie_breaker },
        })
    }

    fn node(&self, n: &Node) -> Result<Query> {
        match n {
            Node::Term { field, text, fuzzy, boost, wildcard } => {
                let q = if *wildcard {
                    if !self.allow_leading_wildcard && (text.starts_with('*') || text.starts_with('?')) {
                        return Err(self.ctx.shard_err_pub(format!("Failed to parse query [{text}]")).caused_by(EsError::new(400, "parse_exception", "'*' or '?' not allowed as first character in WildcardQuery")));
                    }
                    self.per_field(field, |f| {
                        let Some(rf) = self.ctx.lookup.get(f) else { return Ok(Query::MatchNone) };
                        // `ConstantFieldType.wildcardQuery`: matched against the constant value.
                        if rf.declared == crate::mapping::EsType::ConstantKeyword {
                            return super::dsl::parse_query(&serde_json::json!({"wildcard": {f: {"value": text}}}), self.ctx);
                        }
                        let Some(id) = rf.core_id else { return Ok(Query::MatchNone) };
                        if !(rf.ty.is_text() || rf.ty.is_keyword_like()) {
                            return Ok(Query::MatchNone);
                        }
                        let pat = if rf.ty.is_text() { text.to_lowercase() } else { text.clone() };
                        if pat == "*" {
                            return Ok(Query::Exists { field: id });
                        }
                        if pat.ends_with('*') && !pat[..pat.len() - 1].contains(['*', '?']) {
                            return Ok(Query::Prefix { field: id, prefix: pat[..pat.len() - 1].as_bytes().to_vec(), max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore });
                        }
                        let _ = self.analyze_wildcard;
                        Ok(Query::Wildcard { field: id, pattern: pat, max_expansions: usize::MAX, case_insensitive: false, rewrite: MultiTermRewrite::ConstantScore })
                    })?
                } else {
                    let mut o = self.opts.clone();
                    if let Some(fz) = fuzzy {
                        o.fuzziness = Some(parse_fuzziness(&Value::String(fz.clone())).unwrap_or(Fuzziness::Auto));
                    }
                    o.msm = None;
                    if self.cross_fields && fuzzy.is_none() {
                        super::cross_fields::build(&Value::String(text.clone()), &self.fields_for(field), &o, self.tie_breaker, self.ctx)?
                    } else {
                        self.per_field(field, |f| match_field(f, &Value::String(text.clone()), &o, self.ctx))?
                    }
                };
                Ok(boosted(q, *boost))
            }
            Node::Phrase { field, text, slop, boost } => {
                let mut o = self.opts.clone();
                o.kind = MatchKind::Phrase;
                o.quoted = true;
                o.fuzziness = None;
                o.msm = None;
                o.slop = slop.unwrap_or(self.opts.slop);
                let q = self.per_field(field, |f| match_field(f, &Value::String(text.clone()), &o, self.ctx))?;
                Ok(boosted(q, *boost))
            }
            Node::Regex { field, pattern, boost } => {
                // Resource limits are request errors, not field/type mismatches.
                // In particular, implicit wildcard-field leniency cannot swallow them.
                super::limits::check_regex("query_string", pattern, self.ctx.max_regex_length)?;
                let q = self.per_field(field, |f| {
                    let Some(rf) = self.ctx.lookup.get(f) else { return Ok(Query::MatchNone) };
                    let Some(id) = rf.core_id else { return Ok(Query::MatchNone) };
                    Ok(Query::Regexp { field: id, pattern: pattern.clone(), case_insensitive: false, max_expansions: usize::MAX, rewrite: MultiTermRewrite::ConstantScore })
                })?;
                Ok(boosted(q, *boost))
            }
            Node::Range { field, lo, hi, inc_lo, inc_hi, boost } => {
                let q = self.per_field(field, |f| {
                    let mut spec = Map::new();
                    if let Some(l) = lo {
                        spec.insert(if *inc_lo { "gte" } else { "gt" }.into(), json!(l));
                    }
                    if let Some(h) = hi {
                        spec.insert(if *inc_hi { "lte" } else { "lt" }.into(), json!(h));
                    }
                    if let Some(tz) = &self.time_zone {
                        spec.insert("time_zone".into(), json!(tz));
                    }
                    super::dsl::parse_query(&json!({"range": {f: spec}}), self.ctx)
                })?;
                Ok(boosted(q, *boost))
            }
            Node::Exists { field } => super::dsl::parse_query(&json!({"exists": {"field": field}}), self.ctx),
            Node::MatchAll => Ok(Query::MatchAll),
            Node::Group { field, clauses, boost } => {
                let mut must = Vec::new();
                let mut should = Vec::new();
                let mut must_not = Vec::new();
                for (occ, n) in clauses {
                    let n = match (n, field) {
                        (Node::Term { field: None, text, fuzzy, boost, wildcard }, Some(f)) => Node::Term { field: Some(f.clone()), text: text.clone(), fuzzy: fuzzy.clone(), boost: *boost, wildcard: *wildcard },
                        (Node::Phrase { field: None, text, slop, boost }, Some(f)) => Node::Phrase { field: Some(f.clone()), text: text.clone(), slop: *slop, boost: *boost },
                        (other, _) => other.clone(),
                    };
                    let q = self.node(&n)?;
                    match occ {
                        Occur::Must => must.push(q),
                        Occur::Should => should.push(q),
                        Occur::MustNot => must_not.push(q),
                    }
                }
                if must.is_empty() && should.is_empty() && must_not.is_empty() {
                    return Ok(Query::MatchNone);
                }
                if must.len() == 1 && should.is_empty() && must_not.is_empty() {
                    return Ok(boosted(must.pop().unwrap(), *boost));
                }
                if should.len() == 1 && must.is_empty() && must_not.is_empty() {
                    return Ok(boosted(should.pop().unwrap(), *boost));
                }
                if must.is_empty() && should.is_empty() {
                    must.push(Query::MatchAll);
                }
                let msm = self.opts.msm.clone().unwrap_or(MinShouldMatch::Count(0));
                Ok(boosted(Query::Bool { must, should, must_not, filter: vec![], minimum_should_match: msm }, *boost))
            }
        }
    }
}

fn common(o: &Map<String, Value>, ctx: &QueryContext, simple: bool) -> Result<(Vec<(String, f32)>, MatchOpts, bool)> {
    let mut opts = MatchOpts::default();
    let default_and = match o.get("default_operator").and_then(Value::as_str) {
        Some(s) if s.eq_ignore_ascii_case("and") => true,
        Some(s) if s.eq_ignore_ascii_case("or") => false,
        Some(s) => return Err(EsError::parsing(format!("Unknown operator [{s}]"))),
        None => false,
    };
    if let Some(a) = o.get("analyzer").and_then(Value::as_str) {
        opts.analyzer = Some(ctx.lookup.registry.analyzer(a).ok_or_else(|| EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("[query_string] analyzer [{a}] not found")))?);
    }
    opts.lenient = o.get("lenient").and_then(Value::as_bool).unwrap_or(false);
    opts.slop = o.get("phrase_slop").and_then(Value::as_u64).unwrap_or(0) as u32;
    if let Some(v) = o.get("fuzziness") {
        let _ = parse_fuzziness(v)?;
    }
    if let Some(v) = o.get("minimum_should_match") {
        opts.msm = parse_msm(v);
    }
    let specs: Vec<String> = if let Some(Value::Array(a)) = o.get("fields") {
        a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()
    } else if let Some(df) = o.get("default_field").and_then(Value::as_str) {
        // One field name, as ES reads it (no `^` weight): a literal `^`
        // is escaped from the weight split below.
        vec![if df.contains('^') { format!("{df}^1") } else { df.to_string() }]
    } else {
        super::dsl::default_specs(ctx)?
    };
    if specs.iter().any(|s| s.contains('*')) && o.get("lenient").is_none() {
        opts.lenient = true;
    }
    let _ = simple;
    Ok((expand_fields(&specs, ctx), opts, default_and))
}

/// The `query` text: ES reads it with `parser.text()`, so a number is its literal text.
fn query_text(o: &Map<String, Value>) -> Option<String> {
    match o.get("query")? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

pub fn parse_query_string(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let text = query_text(o).ok_or_else(|| EsError::parsing("[query_string] must be provided with a [query]"))?;
    let text = text.as_str();
    let (fields, opts, default_and) = common(o, ctx, false)?;
    let chars: Vec<char> = text.chars().collect();
    let mut p = Parser { s: &chars, i: 0, default_and, depth: 0 };
    let clauses = p.query(None).map_err(|e| {
        EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("Failed to parse query [{text}]")).caused_by(e)
    })?;
    if p.i < chars.len() {
        return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("Failed to parse query [{text}]"))
            .caused_by(EsError::parsing(format!("Cannot parse '{text}': Encountered \" \")\" \") \"\" at line 1, column {}.", p.i))));
    }
    let t = Translate {
        ctx,
        default_fields: fields,
        opts,
        analyze_wildcard: o.get("analyze_wildcard").and_then(Value::as_bool).unwrap_or(false),
        allow_leading_wildcard: o.get("allow_leading_wildcard").and_then(Value::as_bool).unwrap_or(true),
        tie_breaker: o.get("tie_breaker").and_then(Value::as_f64).unwrap_or(if o.get("type").and_then(Value::as_str) == Some("most_fields") { 1.0 } else { 0.0 }) as f32,
        cross_fields: o.get("type").and_then(Value::as_str) == Some("cross_fields"),
        time_zone: o.get("time_zone").and_then(Value::as_str).map(str::to_string),
    };
    let q = t.node(&Node::Group { field: None, clauses, boost: 1.0 })?;
    let boost = o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    Ok(boosted(q, boost))
}

// ---------------------------------------------------------------------------
// simple_query_string: a port of Lucene's `SimpleQueryParser`.

const F_AND: u32 = 1;
const F_NOT: u32 = 2;
const F_OR: u32 = 4;
const F_PREFIX: u32 = 8;
const F_PHRASE: u32 = 16;
const F_PRECEDENCE: u32 = 32;
const F_ESCAPE: u32 = 64;
const F_WHITESPACE: u32 = 128;
const F_FUZZY: u32 = 256;
const F_SLOP: u32 = 512;
const F_ALL: u32 = 1023;

fn parse_sqs_flags(v: Option<&Value>) -> Result<u32> {
    match v {
        None => Ok(F_ALL),
        Some(Value::Number(n)) => Ok(n.as_u64().unwrap_or(F_ALL as u64) as u32),
        Some(Value::String(s)) => {
            let mut f = 0;
            for part in s.split('|') {
                f |= match part.trim().to_ascii_uppercase().as_str() {
                    "ALL" => F_ALL,
                    "NONE" => 0,
                    "AND" => F_AND,
                    "NOT" => F_NOT,
                    "OR" => F_OR,
                    "PREFIX" => F_PREFIX,
                    "PHRASE" => F_PHRASE,
                    "PRECEDENCE" => F_PRECEDENCE,
                    "ESCAPE" => F_ESCAPE,
                    "WHITESPACE" => F_WHITESPACE,
                    "FUZZY" => F_FUZZY,
                    "NEAR" | "SLOP" => F_SLOP,
                    _ => return Err(EsError::illegal_argument(format!("Unknown simple_query_string flag [{}]", part.trim()))),
                };
            }
            Ok(f)
        }
        Some(_) => Ok(F_ALL),
    }
}

struct SqState {
    top: Option<Node>,
    current: Option<Occur>,
    previous: Option<Occur>,
    not: usize,
}

struct SimpleParser<'a> {
    data: &'a [char],
    flags: u32,
    default_op: Occur,
    /// Current `(...)` nesting, and whether [`MAX_GROUP_DEPTH`] was hit
    /// (the parser itself never fails on syntax, as in Lucene).
    depth: std::cell::Cell<usize>,
    too_deep: std::cell::Cell<bool>,
}

impl SimpleParser<'_> {
    fn on(&self, f: u32) -> bool {
        self.flags & f != 0
    }

    fn parse(&self) -> Option<Node> {
        let mut st = SqState { top: None, current: None, previous: None, not: 0 };
        self.sub_query(&mut st, 0, self.data.len());
        st.top
    }

    fn build(&self, st: &mut SqState, branch: Option<Node>) {
        if let Some(mut b) = branch {
            if st.not % 2 == 1 {
                b = Node::Group { field: None, clauses: vec![(Occur::MustNot, b), (Occur::Should, Node::MatchAll)], boost: 1.0 };
            }
            match st.top.take() {
                None => st.top = Some(b),
                Some(top) => {
                    let op = st.current.unwrap_or(self.default_op);
                    let mut clauses = if st.previous != Some(op) {
                        vec![(op, top)]
                    } else if let Node::Group { clauses, .. } = top {
                        clauses
                    } else {
                        vec![(op, top)]
                    };
                    clauses.push((op, b));
                    st.top = Some(Node::Group { field: None, clauses, boost: 1.0 });
                    st.previous = Some(op);
                    st.current = None;
                }
            }
        }
        st.not = 0;
    }

    fn token_finished(&self, c: char) -> bool {
        (c == '"' && self.on(F_PHRASE))
            || (c == '|' && self.on(F_OR))
            || (c == '+' && self.on(F_AND))
            || ((c == '(' || c == ')') && self.on(F_PRECEDENCE))
            || (c.is_whitespace() && self.on(F_WHITESPACE))
    }

    fn sub_query(&self, st: &mut SqState, mut i: usize, end: usize) {
        let d = self.data;
        while i < end {
            let c = d[i];
            if c == '(' && self.on(F_PRECEDENCE) {
                i = self.consume_sub(st, i, end);
            } else if c == ')' && self.on(F_PRECEDENCE) {
                i += 1;
            } else if c == '"' && self.on(F_PHRASE) {
                i = self.consume_phrase(st, i, end);
            } else if c == '+' && self.on(F_AND) {
                if st.current.is_none() && st.top.is_some() {
                    st.current = Some(Occur::Must);
                }
                i += 1;
            } else if c == '|' && self.on(F_OR) {
                if st.current.is_none() && st.top.is_some() {
                    st.current = Some(Occur::Should);
                }
                i += 1;
            } else if c == '-' && self.on(F_NOT) {
                st.not += 1;
                i += 1;
                continue;
            } else if c.is_whitespace() && self.on(F_WHITESPACE) {
                i += 1;
            } else {
                i = self.consume_token(st, i, end);
            }
            st.not = 0;
        }
    }

    fn consume_sub(&self, st: &mut SqState, mut i: usize, end: usize) -> usize {
        let d = self.data;
        i += 1;
        let start = i;
        let (mut prec, mut esc, mut quoted) = (1, false, false);
        while i < end {
            if !esc {
                if d[i] == '\\' && self.on(F_ESCAPE) {
                    esc = true;
                } else if d[i] == '"' && self.on(F_PHRASE) {
                    quoted = !quoted;
                } else if d[i] == '(' && !quoted {
                    prec += 1;
                } else if d[i] == ')' && !quoted {
                    prec -= 1;
                    if prec == 0 {
                        break;
                    }
                }
            } else {
                esc = false;
            }
            i += 1;
        }
        if i == end {
            start
        } else if i == start {
            i + 1
        } else {
            if self.depth.get() >= MAX_GROUP_DEPTH {
                self.too_deep.set(true);
                return end;
            }
            self.depth.set(self.depth.get() + 1);
            let mut sub = SqState { top: None, current: None, previous: None, not: 0 };
            self.sub_query(&mut sub, start, i);
            self.depth.set(self.depth.get() - 1);
            self.build(st, sub.top);
            i + 1
        }
    }

    fn fuzziness(&self, mut i: usize, end: usize) -> (u32, usize) {
        let mut digits = String::new();
        if i < end && self.data[i] == '~' {
            while i < end {
                i += 1;
                if i < end && self.data[i].is_ascii_digit() {
                    digits.push(self.data[i]);
                } else {
                    break;
                }
            }
            if let Ok(n) = digits.parse::<u32>() {
                return (n, i);
            }
        }
        (1, i)
    }

    fn consume_phrase(&self, st: &mut SqState, mut i: usize, end: usize) -> usize {
        let d = self.data;
        let (mut text, mut esc, mut has_slop) = (String::new(), false, false);
        i += 1;
        while i < end {
            if !esc {
                if d[i] == '\\' && self.on(F_ESCAPE) {
                    esc = true;
                    i += 1;
                    continue;
                } else if d[i] == '"' {
                    i += 1;
                    if i < end && d[i] == '~' {
                        has_slop = true;
                    }
                    break;
                }
            }
            esc = false;
            text.push(d[i]);
            i += 1;
        }
        if !text.is_empty() {
            let slop = if has_slop {
                let (n, ni) = self.fuzziness(i, end);
                i = ni;
                n
            } else {
                0
            };
            self.build(st, Some(Node::Phrase { field: None, text, slop: Some(slop), boost: 1.0 }));
        }
        i
    }

    fn consume_token(&self, st: &mut SqState, mut i: usize, end: usize) -> usize {
        let d = self.data;
        let (mut text, mut esc, mut prefix, mut fuzzy) = (String::new(), false, false, false);
        while i < end {
            if !esc {
                if d[i] == '\\' && self.on(F_ESCAPE) {
                    esc = true;
                    i += 1;
                    continue;
                } else if self.token_finished(d[i]) {
                    break;
                } else if !text.is_empty() && d[i] == '~' && self.on(F_FUZZY) {
                    fuzzy = true;
                    break;
                } else if d[i] == '*' && self.on(F_PREFIX) {
                    prefix = true;
                    i += 1;
                    break;
                }
            }
            esc = false;
            text.push(d[i]);
            i += 1;
        }
        if !text.is_empty() {
            let branch = if fuzzy && self.on(F_FUZZY) {
                let (n, ni) = self.fuzziness(i, end);
                i = ni;
                Node::Term { field: None, text, fuzzy: Some(n.min(2).to_string()), boost: 1.0, wildcard: false }
            } else if prefix {
                Node::Term { field: None, text: format!("{text}*"), fuzzy: None, boost: 1.0, wildcard: true }
            } else {
                Node::Term { field: None, text, fuzzy: None, boost: 1.0, wildcard: false }
            };
            self.build(st, Some(branch));
        }
        i
    }
}

/// `simple_query_string` (ES `SimpleQueryStringBuilder` over Lucene's
/// `SimpleQueryParser`); never fails on syntax.
pub fn parse_simple_query_string(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let text = query_text(o).ok_or_else(|| EsError::parsing("[simple_query_string] requires query value"))?;
    let text = text.as_str();
    let (fields, mut opts, default_and) = common(o, ctx, true)?;
    let flags = parse_sqs_flags(o.get("flags"))?;
    if let Some(v) = o.get("fuzzy_max_expansions").and_then(Value::as_u64) {
        opts.max_expansions = v as usize;
    }
    if let Some(v) = o.get("fuzzy_prefix_length").and_then(Value::as_u64) {
        opts.prefix_length = v as usize;
    }
    if let Some(v) = o.get("fuzzy_transpositions").and_then(Value::as_bool) {
        opts.transpositions = v;
    }
    opts.operator_and = default_and;
    let msm = opts.msm.take();
    let boost = o.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
    if text.trim() == "*" {
        return Ok(boosted(Query::MatchAll, boost));
    }
    let chars: Vec<char> = text.chars().collect();
    let p = SimpleParser { data: &chars, flags, default_op: if default_and { Occur::Must } else { Occur::Should }, depth: Default::default(), too_deep: Default::default() };
    let tree = p.parse();
    if p.too_deep.get() {
        return Err(too_deep());
    }
    let Some(tree) = tree else { return Ok(Query::MatchNone) };
    // Per-field queries are summed (a SHOULD boolean), i.e. dis_max with tie 1.
    let t = Translate {
        ctx,
        default_fields: fields,
        opts,
        analyze_wildcard: o.get("analyze_wildcard").and_then(Value::as_bool).unwrap_or(false),
        allow_leading_wildcard: true,
        tie_breaker: 1.0,
        cross_fields: false,
        time_zone: None,
    };
    let q = t.node(&tree)?;
    let q = match (msm, q) {
        (Some(m), Query::Bool { must, should, must_not, filter, .. }) if must.is_empty() && !should.is_empty() => Query::Bool { must, should, must_not, filter, minimum_should_match: m },
        (_, q) => q,
    };
    Ok(boosted(q, boost))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_nesting_is_bounded() {
        // On a thread with the default 2 MiB stack (as request threads have).
        std::thread::Builder::new().stack_size(2 << 20).spawn(|| {
            let nest = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n)).chars().collect::<Vec<char>>();
            let ok = nest(MAX_GROUP_DEPTH);
            assert!(Parser { s: &ok, i: 0, default_and: false, depth: 0 }.query(None).is_ok());
            for n in [MAX_GROUP_DEPTH + 1, 50_000] {
                let deep = nest(n);
                let e = Parser { s: &deep, i: 0, default_and: false, depth: 0 }.query(None).err().expect("too deep");
                assert!(e.reason.contains("nested more than"), "{}", e.reason);
                let p = SimpleParser { data: &deep, flags: u32::MAX, default_op: Occur::Should, depth: Default::default(), too_deep: Default::default() };
                p.parse();
                assert!(p.too_deep.get());
            }
            let p = SimpleParser { data: &ok, flags: u32::MAX, default_op: Occur::Should, depth: Default::default(), too_deep: Default::default() };
            assert!(p.parse().is_some() && !p.too_deep.get());
        }).unwrap().join().unwrap();
    }

    fn parse(s: &str, and: bool) -> Vec<(Occur, Node)> {
        let chars: Vec<char> = s.chars().collect();
        Parser { s: &chars, i: 0, default_and: and, depth: 0 }.query(None).unwrap()
    }

    #[test]
    fn grammar() {
        let c = parse("title:quick AND brown", false);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].0, Occur::Must);
        assert_eq!(c[1].0, Occur::Must);
        assert!(matches!(&c[0].1, Node::Term { field: Some(f), text, .. } if f == "title" && text == "quick"));
        let c = parse("a b -c +d", false);
        assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![Occur::Should, Occur::Should, Occur::MustNot, Occur::Must]);
        let c = parse("\"quick fox\"~2^3 count:[1 TO *} age:>=10 qu?ck* /jo.n/ _exists_:user fuzz~1 (x OR y)", false);
        assert!(matches!(&c[0].1, Node::Phrase { slop: Some(2), boost, .. } if *boost == 3.0));
        assert!(matches!(&c[1].1, Node::Range { lo: Some(l), hi: None, inc_lo: true, inc_hi: false, .. } if l == "1"));
        assert!(matches!(&c[2].1, Node::Range { lo: Some(l), inc_lo: true, .. } if l == "10"));
        assert!(matches!(&c[3].1, Node::Term { wildcard: true, .. }));
        assert!(matches!(&c[4].1, Node::Regex { pattern, .. } if pattern == "jo.n"));
        assert!(matches!(&c[5].1, Node::Exists { field } if field == "user"));
        assert!(matches!(&c[6].1, Node::Term { fuzzy: Some(f), .. } if f == "1"));
        assert!(matches!(&c[7].1, Node::Group { clauses, .. } if clauses.len() == 2));
        let c = parse("a b", true);
        assert_eq!(c[0].0, Occur::Must);
        let c = parse("a\\:b", false);
        assert!(matches!(&c[0].1, Node::Term { text, field: None, .. } if text == "a:b"));
    }
}
