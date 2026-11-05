//! Event Query Language (`/{index}/_eql/search`).
//!
//! Event queries (`category where condition`, `any where ...`) translate to
//! Query DSL and run as a timestamp-sorted search. Sequences
//! (`sequence [by k] [with maxspan=d] [..] [..] [until [..]]`) are matched
//! in memory with ES's state machine (a later event at stage n extends the
//! most recent partial sequence at stage n-1 with the same join key), over
//! `fetch_size` pages of the stage searches read as matching needs them
//! (`windowed_sequences`), or else over the per-stage event lists (capped
//! at 10,000 events per stage); `![..]` stages must have no matching event
//! between their neighbours. `sample` queries pick distinct events per join
//! key. Each index is searched on its own, so a failing index becomes a
//! `shard_failures` entry (`allow_partial_search_results`, default true;
//! sequences need `allow_partial_sequence_results`).

use crate::error::{EsError, Result};
use crate::esql::lexer::{lex, Dialect, Tok, Token};
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// One index's search run elsewhere: `(concrete index, search routing, search body)` →
/// response. A cluster runs it through its coordinator over every shard.
pub type RemoteSearch = Arc<dyn Fn(&str, Option<&str>, &Value) -> Result<Value> + Send + Sync>;

thread_local! {
    static REMOTE: RefCell<Option<RemoteSearch>> = const { RefCell::new(None) };
}

/// While alive, the EQL searches of this thread run through `remote`.
pub struct RemoteScope(Option<RemoteSearch>);

impl Drop for RemoteScope {
    fn drop(&mut self) {
        let prev = self.0.take();
        REMOTE.with(|r| *r.borrow_mut() = prev);
    }
}

pub fn remote_scope(remote: Option<RemoteSearch>) -> RemoteScope {
    RemoteScope(REMOTE.with(|r| std::mem::replace(&mut *r.borrow_mut(), remote)))
}

/// A remote response's shard failures as one error (the first failure's
/// reason, wrapped like a local per-index failure).
fn remote_shard_failure(v: &Value) -> Option<EsError> {
    if v["_shards"]["failed"].as_u64().unwrap_or(0) == 0 {
        return None;
    }
    let f = v["_shards"]["failures"].as_array()?.first()?;
    let reason = EsError::from_error_json(f["status"].as_u64().unwrap_or(500) as u16, &f["reason"]);
    Some(EsError::new(reason.status, "search_phase_execution_exception", "all shards failed").with("phase", "query").with("grouped", true).with("failed_shards", json!([f])).caused_by(reason))
}

fn perr(full: &str, off: usize, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "parsing_exception", format!("{}: {}", crate::esql::pos_prefix(full, off), msg.as_ref()))
}

#[derive(Clone, Debug)]
pub struct EventQuery {
    pub category: Option<String>,
    pub cond: Value,
}

#[derive(Clone, Debug)]
pub struct Stage {
    pub q: EventQuery,
    pub by: Vec<String>,
    pub runs: usize,
    /// `![...]` — the event is expected to be absent; a match emits a
    /// synthetic `{"missing": true}` event.
    pub missing: bool,
}

#[derive(Clone, Debug)]
pub enum EqlQuery {
    Event { q: EventQuery, pipes: Vec<(String, usize)> },
    Sequence { by: Vec<String>, maxspan: Option<i64>, stages: Vec<Stage>, until: Option<Stage>, pipes: Vec<(String, usize)> },
    Sample { by: Vec<String>, stages: Vec<Stage> },
}

pub struct Opts {
    pub case_sensitive: bool,
}

struct P<'a> {
    full: &'a str,
    t: Vec<Token>,
    i: usize,
    o: &'a Opts,
    /// Exact-match comparisons: (field, how errors name the comparison, offset).
    exact: Vec<(String, String, usize)>,
    nest: crate::parse_depth::Nesting,
}

impl<'a> P<'a> {
    fn peek(&self) -> &Tok {
        &self.t[self.i].tok
    }
    fn pos(&self) -> usize {
        self.t[self.i].start
    }
    fn at_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s.eq_ignore_ascii_case(k))
    }
    fn eat_kw(&mut self, k: &str) -> bool {
        if self.at_kw(k) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn err(&self, what: &str) -> EsError {
        let t = &self.t[self.i];
        let txt = if matches!(t.tok, Tok::Eof) { "<EOF>".to_string() } else { self.full[t.start..t.end].to_string() };
        perr(self.full, t.start, format!("mismatched input '{txt}' expecting {what}"))
    }
    fn tilde(&mut self) -> bool {
        // `~` is lexed as an unknown op; we pre-replace it, see `parse`.
        false
    }

    fn field(&mut self) -> Result<String> {
        match self.peek().clone() {
            Tok::Ident(s) | Tok::QIdent(s) => {
                self.i += 1;
                Ok(s)
            }
            _ => Err(self.err("a field name")),
        }
    }

    fn join_field(&mut self) -> Result<String> {
        if let Tok::Param(mut name) = self.peek().clone() {
            self.i += 1;
            if name.is_empty() { name = self.field()?; }
            while self.eat(&Tok::Op(".")) {
                name.push('.');
                name.push_str(&self.field()?);
            }
            return Ok(format!("?{name}"));
        }
        self.field()
    }

    fn fields(&mut self) -> Result<Vec<String>> {
        let mut v = vec![self.join_field()?];
        while self.eat(&Tok::Comma) {
            v.push(self.join_field()?);
        }
        Ok(v)
    }

    fn event(&mut self) -> Result<EventQuery> {
        let cat = match self.peek().clone() {
            Tok::Ident(s) | Tok::QIdent(s) | Tok::Str(s) => {
                self.i += 1;
                s
            }
            _ => return Err(self.err("an event category")),
        };
        if !self.eat_kw("where") {
            return Err(self.err("'where'"));
        }
        let cond = self.or()?;
        Ok(EventQuery { category: if cat.eq_ignore_ascii_case("any") { None } else { Some(cat) }, cond })
    }

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            let at = self.t[self.i].start;
            return Err(perr(self.full, at, format!("expression nested more than {} levels deep", self.nest.max())));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    fn or(&mut self) -> Result<Value> {
        self.nested(true, Self::or_unnested)
    }

    fn or_unnested(&mut self) -> Result<Value> {
        let mut parts = vec![self.and()?];
        while self.eat_kw("or") {
            parts.push(self.and()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { json!({"bool": {"should": parts, "minimum_should_match": 1}}) })
    }

    fn and(&mut self) -> Result<Value> {
        let mut parts = vec![self.not()?];
        while self.eat_kw("and") {
            parts.push(self.not()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { json!({"bool": {"filter": parts}}) })
    }

    fn not(&mut self) -> Result<Value> {
        self.nested(false, Self::not_unnested)
    }

    fn not_unnested(&mut self) -> Result<Value> {
        if self.eat_kw("not") {
            let q = self.not()?;
            return Ok(json!({"bool": {"must_not": [q]}}));
        }
        self.cmp()
    }

    fn literal(&mut self) -> Result<Value> {
        let neg = matches!(self.peek(), Tok::Op("-"));
        if neg {
            self.i += 1;
        }
        let v = match self.peek().clone() {
            Tok::Str(s) => json!(s),
            Tok::Int(i) => json!(if neg { -i } else { i }),
            Tok::Dec(d) => json!(if neg { -d } else { d }),
            Tok::Ident(s) if s.eq_ignore_ascii_case("true") => json!(true),
            Tok::Ident(s) if s.eq_ignore_ascii_case("false") => json!(false),
            Tok::Ident(s) if s.eq_ignore_ascii_case("null") => Value::Null,
            _ => return Err(self.err("a literal")),
        };
        self.i += 1;
        Ok(v)
    }

    /// A function argument that is not a literal: a field or a function
    /// over fields / literals, as a Painless expression.
    fn arg_expr(&mut self) -> Result<String> {
        if let (Tok::Ident(name), Tok::LParen) = (self.peek().clone(), self.t[(self.i + 1).min(self.t.len() - 1)].tok.clone()) {
            self.i += 2;
            let mut args = Vec::new();
            if !matches!(self.peek(), Tok::RParen) {
                loop {
                    let save = self.i;
                    let a = match self.literal() {
                        Ok(v) => painless_lit(&v),
                        Err(_) => {
                            self.i = save;
                            self.arg_expr()?
                        }
                    };
                    args.push(a);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            if !self.eat(&Tok::RParen) {
                return Err(self.err("')'"));
            }
            let a = |i: usize| args.get(i).cloned().unwrap_or_else(|| "null".into());
            return match name.trim_end_matches('~').to_ascii_lowercase().as_str() {
                "substring" => Ok(format!("eqlSubstring({}, {}, {})", a(0), a(1), a(2))),
                "concat" => Ok(format!("eqlConcat([{}])", args.join(", "))),
                "length" => Ok(format!("({0} == null ? null : {0}.toString().length())", a(0))),
                "string" => Ok(format!("({0} == null ? null : {0}.toString())", a(0))),
                _ => Err(self.err("a supported function")),
            };
        }
        let f = self.field()?;
        Ok(painless_field(&f))
    }

    fn literal_list(&mut self) -> Result<Vec<Value>> {
        let mut out = Vec::new();
        if self.eat(&Tok::LParen) {
            loop {
                out.push(self.literal()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            if !self.eat(&Tok::RParen) {
                return Err(self.err("')'"));
            }
        } else {
            out.push(self.literal()?);
        }
        Ok(out)
    }

    /// Compare a nullable boolean predicate without treating missing inputs as false.
    ///
    /// `pushed`: the predicate is a query on a field (ES translates it to
    /// one, not a script): ES's optimizer turns `f(..) == false` / `!= true`
    /// into `NOT f(..)` (`BooleanFunctionEqualsElimination`), translated to
    /// `bool.must_not` of that query (`ExpressionTranslators.Nots`), which
    /// also matches documents without the field. A script predicate is
    /// negated inside its null-safe script, so a missing input matches
    /// neither way.
    fn predicate_comparison(&mut self, q: Value, defined: Value, pushed: bool) -> Result<Value> {
        let op = match self.peek() {
            Tok::Op(op) if matches!(*op, "==" | "!=") => *op,
            _ => return Ok(q),
        };
        self.i += 1;
        let value = self.literal()?;
        if value.is_null() {
            return Ok(if op == "==" {
                json!({"bool": {"must_not": [defined]}})
            } else {
                defined
            });
        }
        let Some(value) = value.as_bool() else {
            return Err(self.err("a boolean or null"));
        };
        Ok(if value == (op == "==") {
            q
        } else if pushed && defined.get("exists").is_some() {
            json!({"bool": {"must_not": [q]}})
        } else {
            json!({"bool": {"filter": [defined], "must_not": [q]}})
        })
    }

    fn cmp(&mut self) -> Result<Value> {
        if self.eat(&Tok::LParen) {
            let q = self.or()?;
            if !self.eat(&Tok::RParen) {
                return Err(self.err("')'"));
            }
            return Ok(q);
        }
        if self.at_kw("true") {
            self.i += 1;
            return Ok(json!({"match_all": {}}));
        }
        if self.at_kw("false") {
            self.i += 1;
            return Ok(json!({"match_none": {}}));
        }
        // Function call?
        if let (Tok::Ident(name), Tok::LParen) = (self.peek().clone(), self.t[(self.i + 1).min(self.t.len() - 1)].tok.clone()) {
            let pos = self.pos();
            self.i += 2;
            let fpos = self.pos();
            let saved = self.i;
            let (f, source_expr) = match self.literal() {
                Ok(value) => (String::new(), Some(painless_lit(&value))),
                Err(_) => {
                    self.i = saved;
                    if matches!(self.t.get(self.i + 1).map(|t| &t.tok), Some(Tok::LParen)) {
                        (String::new(), Some(self.arg_expr()?))
                    } else {
                        (self.field()?, None)
                    }
                }
            };
            let mut args = Vec::new();
            let mut exprs: Vec<String> = Vec::new();
            while self.eat(&Tok::Comma) {
                let save = self.i;
                match self.literal() {
                    Ok(v) => {
                        exprs.push(painless_lit(&v));
                        args.push(v);
                    }
                    Err(e) => {
                        // A non-foldable argument (field / function): ES
                        // evaluates the condition with a script.
                        self.i = save;
                        match self.arg_expr() {
                            Ok(x) => exprs.push(x),
                            Err(_) => return Err(e),
                        }
                    }
                }
            }
            if !self.eat(&Tok::RParen) {
                return Err(self.err("')'"));
            }
            if source_expr.is_some() || exprs.len() != args.len() {
                let ci_name = name.ends_with('~');
                let ci = ci_name || !self.o.case_sensitive;
                let lname = name.trim_end_matches('~').to_ascii_lowercase();
                let fv = source_expr.unwrap_or_else(|| painless_field(&f));
                let norm = |x: &str| if ci { format!("({x} == null ? null : {x}.toString().toLowerCase())") } else { format!("({x} == null ? null : {x}.toString())") };
                let target = norm(&fv);
                let checks: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        let a = norm(e);
                        match lname.as_str() {
                            "startswith" => Ok(format!("eqlTest({target}, {a}, 0)")),
                            "endswith" => Ok(format!("eqlTest({target}, {a}, 1)")),
                            "stringcontains" => Ok(format!("eqlTest({target}, {a}, 2)")),
                            _ => Err(perr(self.full, pos, format!("Unknown function [{name}]"))),
                        }
                    })
                    .collect::<Result<_>>()?;
                let src = format!("{EQL_PAINLESS_HELPERS} return {};", checks.join(" || "));
                let present = std::iter::once(format!("{fv} != null"))
                    .chain(exprs.iter().map(|e| format!("({e}) != null")))
                    .collect::<Vec<_>>().join(" && ");
                let defined = json!({"script": {"script": {"source": format!("{EQL_PAINLESS_HELPERS} return {present};"), "lang": "painless"}}});
                return self.predicate_comparison(json!({"script": {"script": {"source": src, "lang": "painless"}}}), defined, false);
            }
            let end = self.t[self.i - 1].end;
            self.exact.push((f.clone(), format!("[{}]", &self.full[pos..end]), fpos));
            let ci_name = name.ends_with('~');
            let name = name.trim_end_matches('~').to_string();
            let ci = ci_name || !self.o.case_sensitive;
            let s = |v: &Value| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
            let defined = if args.iter().any(Value::is_null) {
                json!({"match_none": {}})
            } else {
                json!({"exists": {"field": f}})
            };
            let q = match name.to_ascii_lowercase().as_str() {
                "startswith" => any_of(args.iter().map(|a| json!({"prefix": { f.clone(): {"value": s(a), "case_insensitive": ci}}})).collect()),
                "endswith" => any_of(args.iter().map(|a| json!({"wildcard": { f.clone(): {"value": format!("*{}", wc_escape(&s(a))), "case_insensitive": ci}}})).collect()),
                "stringcontains" => any_of(args.iter().map(|a| json!({"wildcard": { f.clone(): {"value": format!("*{}*", wc_escape(&s(a))), "case_insensitive": ci}}})).collect()),
                "wildcard" => any_of(args.iter().map(|a| json!({"wildcard": { f.clone(): {"value": s(a), "case_insensitive": ci}}})).collect()),
                "cidrmatch" => json!({"terms": { f.clone(): args }}),
                _ => return Err(perr(self.full, pos, format!("Unknown function [{name}]"))),
            };
            let q = if args.iter().any(Value::is_null) { json!({"match_none": {}}) } else { q };
            return self.predicate_comparison(q, defined, true);
        }
        let fpos = self.pos();
        let f = self.field()?;
        let op = match self.peek().clone() {
            Tok::Op(o) => {
                self.i += 1;
                o.to_string()
            }
            Tok::Ident(s) => {
                let l = s.to_ascii_lowercase();
                if l == "not" {
                    self.i += 1;
                    if !self.at_kw("in") && !self.at_kw("in~") {
                        return Err(self.err("'in'"));
                    }
                    let ci = self.at_kw("in~");
                    self.i += 1;
                    let vals = self.literal_list()?;
                    let q = in_query(&f, &vals, ci);
                    return Ok(json!({"bool": {"must_not": [q]}}));
                }
                if matches!(l.as_str(), "in" | "in~" | "like" | "like~" | "regex" | "regex~") {
                    self.i += 1;
                    l
                } else {
                    return Err(self.err("a comparison operator"));
                }
            }
            _ => return Err(self.err("a comparison operator")),
        };
        let _ = self.tilde();
        let q = self.comparison(&f, &op)?;
        let end = self.t[self.i - 1].end;
        if matches!(op.as_str(), "<" | "<=" | ">" | ">=") {
            // Only checked for existence.
            self.exact.push((f, String::new(), fpos));
        } else {
            let label = if op == ":" { "[:]".to_string() } else { format!("[{}]", &self.full[fpos..end]) };
            self.exact.push((f, label, fpos));
        }
        Ok(q)
    }

    fn comparison(&mut self, f: &str, op: &str) -> Result<Value> {
        let f = f.to_string();
        Ok(match op {
            "in" | "in~" => in_query(&f, &self.literal_list()?, op.ends_with('~')),
            "like" | "like~" => {
                let ci = op.ends_with('~');
                any_of(self.literal_list()?.iter().map(|p| json!({"wildcard": { f.clone(): {"value": p.as_str().unwrap_or(""), "case_insensitive": ci}}})).collect())
            }
            "regex" | "regex~" => {
                let ci = op.ends_with('~');
                any_of(self.literal_list()?.iter().map(|p| json!({"regexp": { f.clone(): {"value": p.as_str().unwrap_or(""), "case_insensitive": ci}}})).collect())
            }
            ":" => {
                let vals = self.literal_list()?;
                any_of(
                    vals.iter()
                        .map(|v| match v.as_str() {
                            Some(s) if s.contains('*') || s.contains('?') => json!({"wildcard": { f.clone(): {"value": s, "case_insensitive": true}}}),
                            Some(s) => json!({"term": { f.clone(): {"value": s, "case_insensitive": true}}}),
                            None => json!({"term": { f.clone(): v }}),
                        })
                        .collect(),
                )
            }
            "==" | "!=" => {
                let v = self.literal()?;
                let q = if v.is_null() {
                    json!({"bool": {"must_not": [{"exists": {"field": f}}]}})
                } else if v.is_string() && !self.o.case_sensitive {
                    json!({"term": { f.clone(): {"value": v, "case_insensitive": true}}})
                } else {
                    json!({"term": { f.clone(): v }})
                };
                if op == "==" {
                    q
                } else if v.is_null() {
                    json!({"exists": {"field": f}})
                } else {
                    json!({"bool": {"must_not": [q]}})
                }
            }
            "<" | "<=" | ">" | ">=" => {
                let v = self.literal()?;
                let k = match op {
                    "<" => "lt",
                    "<=" => "lte",
                    ">" => "gt",
                    _ => "gte",
                };
                json!({"range": { f: { k: v }}})
            }
            other => return Err(perr(self.full, self.pos(), format!("unsupported operator [{other}]"))),
        })
    }

    /// At a `![` (rewritten to `_[` by [`parse`]) missing-event stage?
    fn at_missing(&self) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == "_") && matches!(self.t.get(self.i + 1).map(|t| &t.tok), Some(Tok::LBracket))
    }

    fn stage(&mut self) -> Result<Stage> {
        let missing = self.at_missing();
        if missing {
            self.i += 1;
        }
        if !self.eat(&Tok::LBracket) {
            return Err(self.err("'['"));
        }
        let q = self.event()?;
        if !self.eat(&Tok::RBracket) {
            return Err(self.err("']'"));
        }
        let by = if self.eat_kw("by") { self.fields()? } else { vec![] };
        let mut runs = 1;
        if self.at_kw("with") && matches!(&self.t[(self.i + 1).min(self.t.len() - 1)].tok, Tok::Ident(s) if s.eq_ignore_ascii_case("runs")) {
            self.i += 2;
            if !matches!(self.peek(), Tok::Op("=")) {
                return Err(self.err("'='"));
            }
            self.i += 1;
            runs = match self.peek().clone() {
                Tok::Int(n) if n > 0 => n as usize,
                _ => return Err(self.err("a positive integer")),
            };
            self.i += 1;
        }
        Ok(Stage { q, by, runs, missing })
    }

    fn pipes(&mut self) -> Result<Vec<(String, usize)>> {
        let mut out = Vec::new();
        while self.eat(&Tok::Op("|")) || matches!(self.peek(), Tok::Ident(s) if s == "|") {
            let name = self.field()?.to_ascii_lowercase();
            let n = match self.peek().clone() {
                Tok::Int(n) if n >= 0 => n as usize,
                _ => return Err(self.err("a number")),
            };
            self.i += 1;
            if name != "head" && name != "tail" {
                return Err(perr(self.full, self.pos(), format!("Pipe [{name}] is not supported")));
            }
            out.push((name, n));
        }
        Ok(out)
    }
}

fn wc_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('*', "\\*").replace('?', "\\?")
}

fn any_of(qs: Vec<Value>) -> Value {
    if qs.len() == 1 {
        qs.into_iter().next().unwrap()
    } else {
        json!({"bool": {"should": qs, "minimum_should_match": 1}})
    }
}

fn in_query(f: &str, vals: &[Value], ci: bool) -> Value {
    if ci {
        any_of(vals.iter().map(|v| json!({"term": { f: {"value": v, "case_insensitive": true}}})).collect())
    } else {
        json!({"terms": { f: vals }})
    }
}

fn parse_duration(p: &mut P) -> Result<i64> {
    let n = match p.peek().clone() {
        Tok::Int(n) => n,
        _ => return Err(p.err("a time span")),
    };
    p.i += 1;
    let unit = match p.peek().clone() {
        Tok::Ident(u) => {
            p.i += 1;
            u
        }
        _ => "ms".into(),
    };
    let mult = match unit.as_str() {
        "ms" => 1,
        "s" => 1000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return Err(p.err("a time unit (ms, s, m, h, d)")),
    };
    Ok(n * mult)
}

/// Parse an EQL query.
pub fn parse(query: &str, o: &Opts) -> Result<EqlQuery> {
    // Normalise `~` suffixes and pipes for the shared tokenizer.
    let mut src = String::with_capacity(query.len());
    let mut in_str = false;
    let chars: Vec<char> = query.chars().collect();
    for (ci, &c) in chars.iter().enumerate() {
        if c == '"' {
            in_str = !in_str;
        }
        // `![` (missing-event stage) → `_[` (same length, so error
        // positions stay put); the shared lexer only knows `!=`.
        if !in_str && c == '!' && chars[ci + 1..].iter().find(|x| !x.is_whitespace()) == Some(&'[') {
            src.push('_');
            continue;
        }
        if !in_str && c == '~' {
            src.push('\u{1}');
            continue;
        }
        if !in_str && c == '|' {
            src.push_str(" PIPE ");
            continue;
        }
        src.push(c);
    }
    // `in~` → ident `in~`: tokenizer only knows identifiers, so map back.
    let src = src.replace('\u{1}', "_ci_");
    let toks = lex(&src, 0, src.len(), Dialect::Esql)?;
    let toks: Vec<Token> = toks
        .into_iter()
        .map(|mut t| {
            if let Tok::Ident(s) = &t.tok {
                if let Some(base) = s.strip_suffix("_ci_") {
                    t.tok = Tok::Ident(format!("{base}~"));
                } else if s == "PIPE" {
                    t.tok = Tok::Op("|");
                }
            }
            t
        })
        .collect();
    let mut p = P { full: &src, t: toks, i: 0, o, exact: Vec::new(), nest: crate::parse_depth::Nesting::new(256) };
    let q = if p.eat_kw("sequence") {
        let mut by = Vec::new();
        let mut maxspan = None;
        loop {
            if p.eat_kw("by") {
                by = p.fields()?;
            } else if p.at_kw("with") {
                p.i += 1;
                if !p.eat_kw("maxspan") {
                    return Err(p.err("'maxspan'"));
                }
                if !matches!(p.peek(), Tok::Op("=")) {
                    return Err(p.err("'='"));
                }
                p.i += 1;
                maxspan = Some(parse_duration(&mut p)?);
            } else {
                break;
            }
        }
        let mut stages = Vec::new();
        while matches!(p.peek(), Tok::LBracket) || p.at_missing() {
            stages.push(p.stage()?);
        }
        if stages.len() < 2 {
            return Err(perr(&src, p.pos(), "A sequence requires a minimum of 2 queries, found [1]"));
        }
        let until = if p.eat_kw("until") { Some(p.stage()?) } else { None };
        let pipes = p.pipes()?;
        EqlQuery::Sequence { by, maxspan, stages, until, pipes }
    } else if p.eat_kw("sample") {
        let by = if p.eat_kw("by") { p.fields()? } else { vec![] };
        let mut stages = Vec::new();
        while matches!(p.peek(), Tok::LBracket) {
            stages.push(p.stage()?);
        }
        EqlQuery::Sample { by, stages }
    } else {
        let q = p.event()?;
        let pipes = p.pipes()?;
        EqlQuery::Event { q, pipes }
    };
    if !matches!(p.peek(), Tok::Eof) {
        return Err(p.err("<EOF>"));
    }
    EXACT.with(|e| *e.borrow_mut() = std::mem::take(&mut p.exact));
    Ok(q)
}

thread_local! {
    /// Exact-match comparisons of the last [`parse`] on this thread.
    static EXACT: std::cell::RefCell<Vec<(String, String, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Rename a field in leaf queries (`term`, `terms`, `wildcard`, `prefix`, `regexp`).
fn rename_field(q: &mut Value, from: &str, to: &str) {
    match q {
        Value::Object(m) => {
            for (k, v) in m.iter_mut() {
                if matches!(k.as_str(), "term" | "terms" | "wildcard" | "prefix" | "regexp") {
                    if let Some(o) = v.as_object_mut() {
                        if let Some(x) = o.remove(from) {
                            o.insert(to.to_string(), x);
                        }
                    }
                } else {
                    rename_field(v, from, to);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| rename_field(x, from, to)),
        _ => {}
    }
}

/// EQL's indices options: unavailable indices are ignored unless asked otherwise.
fn eql_resolve_opts(params: &HashMap<String, String>) -> ResolveOpts {
    let mut opts = ResolveOpts::from_params(params);
    if !params.contains_key("ignore_unavailable") {
        opts.ignore_unavailable = true;
    }
    opts
}

/// Exact comparisons on `text` fields read the field's keyword multi-field;
/// without one they fail verification like ES.
fn resolve_exact(node: &Node, index: &str, params: &HashMap<String, String>, full: &str, runtime: Option<&Value>, q: &mut EqlQuery) -> Result<()> {
    let uses = EXACT.with(|e| e.borrow().clone());
    if uses.is_empty() {
        return Ok(());
    }
    let targets = node.resolve(index, eql_resolve_opts(params)).unwrap_or_default();
    // Runtime fields (mapping-level and request-level) exist too.
    let mut runtime_names: Vec<String> = runtime.and_then(Value::as_object).map(|m| m.keys().cloned().collect()).unwrap_or_default();
    for t in &targets {
        if let Some(Value::Object(m)) = t.svc.meta().mappings.runtime {
            runtime_names.extend(m.keys().cloned());
        }
    }
    let exists = |f: &str| runtime_names.iter().any(|r| r == f) || targets.iter().any(|t| t.svc.lookup().get(f).is_some());
    // Like ES, which resolves only the fields a query mentions, suggestions
    // come from the query's other (existing) fields.
    let mut mentioned: Vec<String> = uses.iter().map(|u| u.0.clone()).filter(|f| exists(f)).collect();
    mentioned.sort();
    mentioned.dedup();
    for (f, label, pos) in uses {
        if !targets.is_empty() && !exists(&f) {
            let refs: Vec<&str> = mentioned.iter().map(String::as_str).collect();
            return Err(EsError::new(
                400,
                "verification_exception",
                format!("Found 1 problem\n{}: Unknown column [{f}]{}", crate::esql::pos_prefix(full, pos), crate::esql::did_you_mean(&f, &refs)),
            ));
        }
        if label.is_empty() {
            continue;
        }
        let mut text = false;
        let mut sub: Option<String> = None;
        for t in &targets {
            let lk = t.svc.lookup();
            if let Some(rf) = lk.get(&f) {
                if rf.ty.is_text() {
                    text = true;
                    sub = sub.or_else(|| rf.multi_fields.iter().find(|m| lk.get(m).is_some_and(|x| x.ty == crate::mapping::EsType::Keyword)).cloned());
                }
            }
        }
        if !text {
            continue;
        }
        let Some(sub) = sub else {
            return Err(EsError::new(
                400,
                "verification_exception",
                format!("Found 1 problem\n{}: {label} cannot operate on first argument field of data type [text]: No keyword/multi-field defined exact matches for [{f}]; define one or use MATCH/QUERY instead", crate::esql::pos_prefix(full, pos)),
            ));
        };
        let fix = |e: &mut EventQuery| rename_field(&mut e.cond, &f, &sub);
        match q {
            EqlQuery::Event { q, .. } => fix(q),
            EqlQuery::Sequence { stages, until, .. } => {
                stages.iter_mut().for_each(|s| fix(&mut s.q));
                if let Some(u) = until {
                    fix(&mut u.q);
                }
            }
            EqlQuery::Sample { stages, .. } => stages.iter_mut().for_each(|s| fix(&mut s.q)),
        }
    }
    Ok(())
}

pub struct Request {
    pub query: String,
    pub timestamp_field: String,
    pub category_field: String,
    pub tiebreaker: Option<String>,
    pub filter: Option<Value>,
    pub size: usize,
    pub fetch_size: usize,
    pub tail: bool,
    pub case_sensitive: bool,
    pub fields: Option<Value>,
    pub runtime_mappings: Option<Value>,
    /// Return the events of the shards that succeeded (default true).
    pub allow_partial_search_results: bool,
    /// Build sequences from partial results (default false: no sequences).
    pub allow_partial_sequence_results: bool,
}

impl Request {
    pub fn from_json(b: &Value) -> Result<Request> {
        // Runtime field definitions validate before anything runs.
        if let Some(Value::Object(rm)) = b.get("runtime_mappings") {
            for (name, def) in rm {
                match def {
                    Value::Object(d) if d.get("type").is_none() => {
                        return Err(EsError::action_validation(&[format!("No type specified for runtime field [{name}]").as_str()]));
                    }
                    Value::Object(_) => {}
                    other => {
                        let kind = match other {
                            Value::Array(_) => "String",
                            Value::String(_) => "String",
                            Value::Number(_) => "Integer",
                            Value::Bool(_) => "Boolean",
                            _ => "null",
                        };
                        return Err(EsError::action_validation(&[format!("Expected map for runtime field [{name}] definition but got [{kind}]").as_str()]));
                    }
                }
            }
        }
        let query = b.get("query").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["query is null or empty"]))?.to_string();
        if query.trim().is_empty() {
            return Err(EsError::action_validation(&["query is null or empty"]));
        }
        let s = |k: &str, d: &str| b.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        let size = b.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size < 0 {
            return Err(EsError::action_validation(&["size must be greater than or equal to 0"]));
        }
        let rp = s("result_position", "tail");
        if rp != "tail" && rp != "head" {
            return Err(EsError::illegal_argument(format!("No enum constant ResultPosition.{}", rp.to_uppercase())));
        }
        Ok(Request {
            query,
            timestamp_field: s("timestamp_field", "@timestamp"),
            category_field: s("event_category_field", "event.category"),
            tiebreaker: b.get("tiebreaker_field").and_then(Value::as_str).map(str::to_string),
            filter: b.get("filter").cloned(),
            size: size as usize,
            fetch_size: b.get("fetch_size").and_then(Value::as_u64).unwrap_or(1000) as usize,
            tail: rp == "tail",
            case_sensitive: b.get("case_sensitive").and_then(Value::as_bool).unwrap_or(false),
            fields: b.get("fields").cloned(),
            runtime_mappings: b.get("runtime_mappings").cloned().filter(|v| !v.is_null()),
            allow_partial_search_results: b.get("allow_partial_search_results").and_then(Value::as_bool).unwrap_or(true),
            allow_partial_sequence_results: b.get("allow_partial_sequence_results").and_then(Value::as_bool).unwrap_or(false),
        })
    }
}

struct Hit {
    index: String,
    id: String,
    /// The event's `_source`; with `raw` set, only the parts the matcher
    /// reads (timestamp, tiebreaker, join keys: see [`lazy_source`]).
    source: Value,
    /// The full `_source` text of a sequence/sample stage event and the
    /// [`crate::memory::json_bytes`] of its tree. It is parsed only when
    /// the event is returned ([`materialized`]), as ES fetches stage events
    /// without `_source` (`SourceGenerator.sourceBuilder`) and loads the
    /// sources of the results alone (`BasicQueryClient.fetchHits`).
    raw: Option<(Box<serde_json::value::RawValue>, u64)>,
    fields: Option<Value>,
    ts: i64,
    tb: Value,
    /// A `![...]` stage's placeholder (ES `Event.MISSING_EVENT`).
    missing: bool,
}

/// A returned copy of `h`, its full `_source` parsed.
fn materialized(h: &Hit) -> Result<Hit> {
    let source = match &h.raw {
        Some((raw, _)) => serde_json::from_str(raw.get()).map_err(|e| EsError::internal(e.to_string()))?,
        None => h.source.clone(),
    };
    Ok(Hit { index: h.index.clone(), id: h.id.clone(), source, raw: None, fields: h.fields.clone(), ts: h.ts, tb: h.tb.clone(), missing: h.missing })
}

/// A copy of `h` as fetched (its source kept as it is, raw or projected).
fn hit_copy(h: &Hit) -> Hit {
    Hit {
        index: h.index.clone(),
        id: h.id.clone(),
        source: h.source.clone(),
        raw: h.raw.as_ref().map(|(raw, bytes)| (raw.to_owned(), *bytes)),
        fields: h.fields.clone(),
        ts: h.ts,
        tb: h.tb.clone(),
        missing: h.missing,
    }
}

/// A returned event as the response lists it (built from its parts: a
/// `json!` of a `Value` would copy it).
fn hit_json(h: Hit) -> Value {
    if h.missing {
        return json!({"_index": "", "_id": "", "_source": {}, "missing": true});
    }
    let mut v = serde_json::Map::new();
    v.insert("_index".into(), Value::String(h.index));
    v.insert("_id".into(), Value::String(h.id));
    v.insert("_source".into(), h.source);
    if let Some(f) = h.fields {
        v.insert("fields".into(), f);
    }
    Value::Object(v)
}

/// `{"total": {"value": n, "relation": "eq"}, <key>: items}`.
fn hits_object(key: &str, items: Vec<Value>) -> Value {
    let mut m = serde_json::Map::new();
    m.insert("total".into(), json!({"value": items.len(), "relation": "eq"}));
    m.insert(key.into(), Value::Array(items));
    Value::Object(m)
}

/// A sequence (or sample) as the response lists it: its `events`, then its
/// `join_keys` (`with_keys`: also when empty).
fn sequence_json(keys: Vec<Value>, events: Vec<Hit>, with_keys: bool) -> Value {
    let mut m = serde_json::Map::new();
    if with_keys {
        m.insert("join_keys".into(), Value::Array(keys));
        m.insert("events".into(), Value::Array(events.into_iter().map(hit_json).collect()));
        return Value::Object(m);
    }
    m.insert("events".into(), Value::Array(events.into_iter().map(hit_json).collect()));
    if !keys.is_empty() {
        m.insert("join_keys".into(), Value::Array(keys));
    }
    Value::Object(m)
}

fn first_value(src: &Value, path: &str) -> Value {
    crate::highlight::source_values(src, path).into_iter().next().cloned().unwrap_or(Value::Null)
}

fn ts_millis(v: &Value) -> i64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0) as i64,
        Value::String(s) => crate::esql::func::parse_date_str(s).unwrap_or(0),
        _ => 0,
    }
}

/// Per-index search failures (one entry per failed index).
pub struct ShardFailures {
    pub entries: Vec<Value>,
    first: Option<EsError>,
}

impl ShardFailures {
    fn new() -> Self {
        Self { entries: Vec::new(), first: None }
    }
    fn push(&mut self, index: &str, e: &EsError) {
        if self.first.is_none() {
            self.first = Some(e.clone());
        }
        // A per-index search error is already wrapped as a
        // `search_phase_execution_exception`; report its shard's own reason.
        let mut obj = e.error_object(false);
        let reason = match obj.get("failed_shards").and_then(|f| f.get(0)).and_then(|f| f.get("reason")) {
            Some(r) => r.clone(),
            None => {
                if let Some(m) = obj.as_object_mut() {
                    m.remove("root_cause");
                }
                obj
            }
        };
        self.entries.push(json!({
            "shard": 0,
            "index": index,
            "node": crate::node::NODE_ID,
            "reason": reason,
        }));
    }
    fn merge(&mut self, other: &ShardFailures) {
        // The same index fails once per stage query; report it once.
        for e in &other.entries {
            if !self.entries.iter().any(|x| x["index"] == e["index"]) {
                self.entries.push(e.clone());
            }
        }
        if self.first.is_none() {
            self.first = other.first.clone();
        }
    }
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// How a stage search reads its hits: `None` parses every `_source` (event
/// queries return all they fetch); `Some(paths)` keeps each `_source` as
/// raw text and parses only what `paths` (split on `.`) reach.
type Projection<'a> = Option<&'a [Vec<String>]>;

/// ES `Optimizer.AddMandatoryJoinKeyFilter`: every non-optional join key
/// of a sequence/sample stage must exist (`IsNotNull`, an `exists`
/// filter). The matcher drops events without such a key anyway
/// ([`join_values`]); filtering them in the search keeps them out of the
/// per-stage fetch window. Runtime fields are left to the matcher.
fn mandatory_key_filters(keys: &[String], r: &Request, targets: &[crate::search::Target]) -> Vec<Value> {
    let keys: Vec<&String> = keys.iter().filter(|k| !k.starts_with('?')).collect();
    if keys.is_empty() {
        return Vec::new();
    }
    let mut runtime: Vec<String> = r.runtime_mappings.as_ref().and_then(Value::as_object).map(|m| m.keys().cloned().collect()).unwrap_or_default();
    for t in targets {
        if let Some(Value::Object(m)) = t.svc.meta().mappings.runtime {
            runtime.extend(m.keys().cloned());
        }
    }
    let mut out: Vec<Value> = Vec::new();
    for k in keys {
        let f = json!({"exists": {"field": k}});
        if !runtime.iter().any(|x| x == k) && !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn fetch(node: &Node, index: &str, params: &HashMap<String, String>, r: &Request, q: &EventQuery, size: usize, desc: bool, keys: &[String], projection: Projection<'_>) -> Result<(Vec<Hit>, ShardFailures)> {
    fetch_after(node, index, params, r, q, size, desc, keys, projection, None, &[], None)
}

/// [`fetch`] resuming at a timestamp: with `from`, only events at or after
/// it (at or before it when `desc`), through `search_after` on the
/// timestamp sort (no tiebreaker sort is set). `extra` filters are added.
#[allow(clippy::too_many_arguments)]
fn fetch_after(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    q: &EventQuery,
    size: usize,
    desc: bool,
    keys: &[String],
    projection: Projection<'_>,
    from: Option<i64>,
    extra: &[Value],
    columns: Option<&[String]>,
) -> Result<(Vec<Hit>, ShardFailures)> {
    let mut filter = vec![json!({"exists": {"field": r.timestamp_field}}), q.cond.clone()];
    if let Some(c) = &q.category {
        filter.push(json!({"term": { r.category_field.clone(): c }}));
    }
    if let Some(f) = &r.filter {
        filter.push(f.clone());
    }
    let order = if desc { "desc" } else { "asc" };
    let mut sort = vec![json!({ r.timestamp_field.clone(): {"order": order} })];
    if let Some(t) = &r.tiebreaker {
        sort.push(json!({ t.clone(): {"order": order, "missing": "_last"} }));
    }
    // EQL accepts cross-cluster expressions: plan the remote parts.
    let targets = crate::ccr::remote::resolve_local_part(node, index, eql_resolve_opts(params))?.0;
    // Nothing can match (`any where false`): no search to run.
    if targets.is_empty() || size == 0 || q.cond == json!({"match_none": {}}) {
        return Ok((vec![], ShardFailures::new()));
    }
    filter.extend(mandatory_key_filters(keys, r, &targets));
    filter.extend(extra.iter().cloned());
    let mut body = json!({"query": {"bool": {"filter": filter}}, "size": size, "sort": sort, "track_total_hits": false});
    if let Some(at) = from {
        // `search_after` is exclusive: step one millisecond back (ahead).
        debug_assert!(r.tiebreaker.is_none());
        body["search_after"] = json!([if desc { at.saturating_add(1) } else { at.saturating_sub(1) }]);
    }
    if let Some(f) = &r.fields {
        body["fields"] = f.clone();
    }
    if let Some(rm) = &r.runtime_mappings {
        body["runtime_mappings"] = rm.clone();
    }
    if let Some(cols) = columns {
        // The timestamp and join keys from their doc values; no stored
        // document is read (see `hits_of_columns`).
        body["_source"] = json!(false);
        let mut dv = vec![json!({"field": r.timestamp_field, "format": "epoch_millis"})];
        dv.extend(cols.iter().map(|c| json!(c)));
        body["docvalue_fields"] = Value::Array(dv);
    }
    let mut req = crate::search::parse_request(&body, &HashMap::new())?;
    // Local stage reads render just the columns (a remote search gets the
    // body, whose hits `hits_of_columns` reads alike).
    req.columns_only = columns.is_some();
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: Some(String::new()), pit_id: None };
    // Search each index separately so a failing index (e.g. a runtime-field
    // script that throws) is reported as a shard failure instead of failing
    // the whole request — ES runs EQL per index and merges the results.
    let mut hits: Vec<Hit> = Vec::new();
    let mut failures = ShardFailures::new();
    let mut seen: Vec<&str> = Vec::new();
    let remote = REMOTE.with(|r| r.borrow().clone());
    for t in &targets {
        if seen.contains(&t.svc.name.as_str()) {
            continue;
        }
        seen.push(t.svc.name.as_str());
        let one = match &remote {
            Some(remote) => {
                // Resolving an alias to a concrete index must not discard its
                // filter/routing or the resolved document-security constraint.
                let mut remote_body = body.clone();
                if let Some(filter) = t.effective_filter() {
                    remote_body["query"] = json!({"bool":{"filter":[body["query"].clone(),filter]}});
                }
                remote(&t.svc.name, t.routing.as_deref(), &remote_body).and_then(|v| match remote_shard_failure(&v) {
                    // A failing shard fails its index, as a local per-index search does.
                    Some(e) => Err(e),
                    None => Ok(match columns {
                        Some(cols) => hits_of_columns(v, r, cols),
                        None => hits_of_value(v, r),
                    }),
                })
            },
            None => {
                let t0 = Instant::now();
                let out = crate::search::execute(std::slice::from_ref(t), &req, &opts);
                phase_add(1, t0);
                let t0 = Instant::now();
                let parsed = out.and_then(|out| match columns {
                    Some(cols) => Ok(hits_of_column_hits(out.columns, cols)),
                    None => hits_of_text(&out.json, r, projection),
                });
                phase_add(2, t0);
                parsed
            }
        };
        match one {
            Ok(found) => hits.extend(found),
            Err(e) => {
                failures.push(&t.svc.name, &e);
            }
        }
    }
    // Merge the per-index (each already sorted) results.
    if seen.len() > 1 {
        hits.sort_by(|a, b| {
            let o = a.ts.cmp(&b.ts).then_with(|| crate::esql::cmp_vals(&crate::esql::parse::json_to_val(&a.tb), &crate::esql::parse::json_to_val(&b.tb)));
            if desc { o.reverse() } else { o }
        });
        hits.truncate(size);
    }
    Ok((hits, failures))
}

/// The hits of a `_source: false` stage search with the timestamp (epoch
/// millis) and the join-key `columns` as doc values: each hit's source holds
/// just those keys (at their full paths, which [`join_values`] reads). A
/// doc with several values of one of them, or no timestamp, is marked
/// ambiguous (`tb` set, as stage searches have no tiebreaker): its source's
/// first value is unknown, so the windowed run gives up on it.
fn hits_of_columns(mut v: Value, r: &Request, columns: &[String]) -> Vec<Hit> {
    let Value::Array(found) = v["hits"]["hits"].take() else { return Vec::new() };
    found.into_iter().map(|mut h| {
        let fields = h.get_mut("fields").map(Value::take).unwrap_or(Value::Null);
        let mut ambiguous = false;
        let mut one = |name: &str| -> Value {
            match fields.get(name) {
                None | Some(Value::Null) => Value::Null,
                Some(Value::Array(a)) if a.is_empty() => Value::Null,
                Some(Value::Array(a)) if a.len() == 1 => a[0].clone(),
                Some(_) => {
                    ambiguous = true;
                    Value::Null
                }
            }
        };
        let ts = match one(r.timestamp_field.as_str()) {
            Value::String(s) => s.parse::<i64>().ok(),
            Value::Number(n) => n.as_i64(),
            _ => None,
        };
        let mut source = serde_json::Map::new();
        for c in columns {
            let value = one(c.as_str());
            if !value.is_null() {
                source.insert(c.clone(), value);
            }
        }
        let bad = ambiguous || ts.is_none();
        Hit {
            index: h["_index"].as_str().unwrap_or("").to_string(),
            id: h["_id"].as_str().unwrap_or("").to_string(),
            source: Value::Object(source),
            raw: None,
            fields: None,
            ts: ts.unwrap_or(0),
            tb: if bad { json!("ambiguous") } else { Value::Null },
            missing: false,
        }
    }).collect()
}

/// [`hits_of_columns`] of a local search's typed hits: each one's values
/// are the timestamp's, then `columns`' (the `docvalue_fields` order).
fn hits_of_column_hits(found: Vec<crate::search::ColumnHit>, columns: &[String]) -> Vec<Hit> {
    found
        .into_iter()
        .map(|h| {
            let mut ambiguous = false;
            let mut values = h.values.into_iter();
            let mut one = |v: Option<Vec<Value>>| -> Value {
                match v {
                    None => Value::Null,
                    Some(mut a) if a.len() <= 1 => a.pop().unwrap_or(Value::Null),
                    Some(_) => {
                        ambiguous = true;
                        Value::Null
                    }
                }
            };
            let ts = match one(values.next()) {
                Value::String(s) => s.parse::<i64>().ok(),
                Value::Number(n) => n.as_i64(),
                _ => None,
            };
            let mut source = serde_json::Map::new();
            for c in columns {
                let value = one(values.next());
                if !value.is_null() {
                    source.insert(c.clone(), value);
                }
            }
            let bad = ambiguous || ts.is_none();
            Hit {
                index: h.index,
                id: h.id,
                source: Value::Object(source),
                raw: None,
                fields: None,
                ts: ts.unwrap_or(0),
                tb: if bad { json!("ambiguous") } else { Value::Null },
                missing: false,
            }
        })
        .collect()
}

/// The full `_source` of the returned events of a doc-values windowed run,
/// by (index, id): ES fetches the sources of the results alone
/// (`BasicQueryClient.fetchHits`). `None` when one cannot be read.
///
/// `wanted`: (index, id, timestamp). The search is sorted on the timestamp
/// from just before the oldest wanted one, so only the segments holding
/// their time range are visited (see `Searcher::run_sorted_leaves`).
fn fetch_sources(node: &Node, index: &str, params: &HashMap<String, String>, r: &Request, wanted: &[(String, String, i64)]) -> Result<Option<HashMap<(String, String), Value>>> {
    let mut by_index: HashMap<&str, Vec<(&str, i64)>> = HashMap::new();
    for (i, id, ts) in wanted {
        let ids = by_index.entry(i.as_str()).or_default();
        if !ids.iter().any(|(x, _)| *x == id.as_str()) {
            ids.push((id.as_str(), *ts));
        }
    }
    let targets = crate::ccr::remote::resolve_local_part(node, index, eql_resolve_opts(params))?.0;
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: Some(String::new()), pit_id: None };
    let remote = REMOTE.with(|r| r.borrow().clone());
    let mut out: HashMap<(String, String), Value> = HashMap::new();
    let mut seen: Vec<&str> = Vec::new();
    for t in &targets {
        let name = t.svc.name.as_str();
        if seen.contains(&name) {
            continue;
        }
        seen.push(name);
        let Some(ids) = by_index.get(name) else { continue };
        for batch in ids.chunks(1000) {
            let values: Vec<&str> = batch.iter().map(|(id, _)| *id).collect();
            let oldest = batch.iter().map(|(_, ts)| *ts).min().unwrap_or(i64::MIN);
            let body = json!({"query": {"ids": {"values": values}}, "size": batch.len(), "track_total_hits": false,
                "sort": [{ r.timestamp_field.clone(): {"order": "asc"} }], "search_after": [oldest.saturating_sub(1)]});
            let response = match &remote {
                Some(remote) => {
                    let mut remote_body = body.clone();
                    if let Some(filter) = t.effective_filter() {
                        remote_body["query"] = json!({"bool": {"filter": [body["query"].clone(), filter]}});
                    }
                    remote(name, t.routing.as_deref(), &remote_body)
                }
                None => crate::search::parse_request(&body, &HashMap::new())
                    .and_then(|req| crate::search::execute(std::slice::from_ref(t), &req, &opts))
                    .and_then(|o| serde_json::from_str::<Value>(&o.json).map_err(|e| EsError::internal(e.to_string()))),
            };
            let Ok(mut v) = response else { return Ok(None) };
            if remote_shard_failure(&v).is_some() {
                return Ok(None);
            }
            // Each `_source` moves out of the response (no copy).
            let Value::Array(hits) = v["hits"]["hits"].take() else { continue };
            for mut h in hits {
                let key = (h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string());
                out.insert(key, h.get_mut("_source").map(Value::take).unwrap_or(Value::Null));
            }
        }
    }
    Ok(wanted.iter().all(|(i, id, _)| out.contains_key(&(i.clone(), id.clone()))).then_some(out))
}

/// The hits of a parsed search response, each `_source` complete.
fn hits_of_value(mut v: Value, r: &Request) -> Vec<Hit> {
    let Value::Array(found) = v["hits"]["hits"].take() else { return Vec::new() };
    found.into_iter().map(|mut h| {
        let source = h["_source"].take();
        Hit {
            index: h["_index"].as_str().unwrap_or("").to_string(),
            id: h["_id"].as_str().unwrap_or("").to_string(),
            ts: ts_millis(&first_value(&source, &r.timestamp_field)),
            tb: r.tiebreaker.as_deref().map(|t| first_value(&source, t)).unwrap_or(Value::Null),
            fields: h.get_mut("fields").map(Value::take),
            source,
            raw: None,
            missing: false,
        }
    }).collect()
}

/// The hits of a search response's text. With a projection, each `_source`
/// stays raw text beside the parts the matcher reads (see [`lazy_source`]).
fn hits_of_text(json: &str, r: &Request, projection: Projection<'_>) -> Result<Vec<Hit>> {
    let internal = |e: serde_json::Error| EsError::internal(e.to_string());
    let Some(paths) = projection else {
        return Ok(hits_of_value(serde_json::from_str(json).map_err(internal)?, r));
    };
    #[derive(serde::Deserialize)]
    struct Response {
        #[serde(default)]
        hits: Hits,
    }
    #[derive(serde::Deserialize, Default)]
    struct Hits {
        #[serde(default)]
        hits: Vec<RawHit>,
    }
    #[derive(serde::Deserialize)]
    struct RawHit {
        #[serde(rename = "_index", default)]
        index: String,
        #[serde(rename = "_id", default)]
        id: String,
        #[serde(rename = "_source", default)]
        source: Option<Box<serde_json::value::RawValue>>,
        #[serde(default)]
        fields: Option<Value>,
    }
    let response: Response = serde_json::from_str(json).map_err(internal)?;
    response.hits.hits.into_iter().map(|h| {
        let (source, raw) = match h.source {
            Some(raw) => {
                let (source, bytes) = lazy_source::project(raw.get(), paths).map_err(internal)?;
                (source, Some((raw, bytes)))
            }
            None => (Value::Null, None),
        };
        Ok(Hit {
            index: h.index,
            id: h.id,
            ts: ts_millis(&first_value(&source, &r.timestamp_field)),
            tb: r.tiebreaker.as_deref().map(|t| first_value(&source, t)).unwrap_or(Value::Null),
            fields: h.fields,
            source,
            raw,
            missing: false,
        })
    }).collect()
}

/// Stage-event `_source` projection: a deserializer that builds only the
/// subtrees [`crate::highlight::source_values`] can reach for the wanted
/// paths, skipping (but measuring) the rest without allocating, so
/// `first_value` of a wanted path reads the same value from the projection
/// as from the whole source.
mod lazy_source {
    use serde::de::{DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor};
    use serde::Deserialize;
    use serde_json::{Map, Value};
    use std::fmt;

    /// The path segments each reader still has to walk below a node.
    type Rest<'p> = Vec<&'p [String]>;

    /// `text` projected onto `paths` (each split on `.`), with the
    /// [`crate::memory::json_bytes`] of the whole tree `text` decodes to.
    pub(super) fn project(text: &str, paths: &[Vec<String>]) -> serde_json::Result<(Value, u64)> {
        let mut d = serde_json::Deserializer::from_str(text);
        let out = Project(paths.iter().map(Vec::as_slice).collect()).deserialize(&mut d)?;
        d.end()?;
        Ok(out)
    }

    /// The `paths` (split on `.`) a projection reads: the timestamp, the
    /// tiebreaker and every join key (optional `?` markers dropped).
    pub(super) fn paths<'a>(fields: impl IntoIterator<Item = &'a str>) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = Vec::new();
        for f in fields {
            let p: Vec<String> = f.strip_prefix('?').unwrap_or(f).split('.').map(str::to_string).collect();
            if !out.contains(&p) {
                out.push(p);
            }
        }
        out
    }

    struct Project<'p>(Rest<'p>);

    impl<'de, 'p> DeserializeSeed<'de> for Project<'p> {
        type Value = (Value, u64);
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            // A path ends here: `source_values` reads the whole subtree.
            if self.0.iter().any(|rest| rest.is_empty()) {
                let v = Value::deserialize(d)?;
                let bytes = crate::memory::json_bytes(&v);
                return Ok((v, bytes));
            }
            d.deserialize_any(self)
        }
    }

    // Scalars above the end of every path are never read (`source_values`
    // only descends objects and arrays there): keep their place as null.
    impl<'de, 'p> Visitor<'de> for Project<'p> {
        type Value = (Value, u64);
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON value")
        }
        fn visit_bool<E: Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok((Value::Null, 32))
        }
        fn visit_i64<E: Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok((Value::Null, 32))
        }
        fn visit_u64<E: Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok((Value::Null, 32))
        }
        fn visit_f64<E: Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok((Value::Null, 32))
        }
        fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
            Ok((Value::Null, 32))
        }
        fn visit_str<E: Error>(self, s: &str) -> Result<Self::Value, E> {
            Ok((Value::Null, 32 + s.len() as u64))
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
            // Arrays are transparent to paths: each element sees the same rest.
            let mut out = Vec::new();
            let mut n = 32u64;
            while let Some((v, b)) = a.next_element_seed(Project(self.0.clone()))? {
                n = n.saturating_add(b);
                out.push(v);
            }
            Ok((Value::Array(out), n))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
            let mut out = Map::new();
            let mut n = 48u64;
            while let Some(key) = a.next_key_seed(KeySeed(&self.0))? {
                match key {
                    Key::Skip(len) => n = n.saturating_add(48 + len).saturating_add(a.next_value_seed(Measure)?),
                    Key::Keep(name, rest) => {
                        let (v, b) = a.next_value_seed(Project(rest))?;
                        n = n.saturating_add(48 + name.len() as u64).saturating_add(b);
                        // Like `Value`'s own map visitor: a repeated key keeps
                        // its first position and its last value.
                        out.insert(name, v);
                    }
                }
            }
            Ok((Value::Object(out), n))
        }
    }

    enum Key<'p> {
        /// Read by no path: the key's length.
        Skip(u64),
        /// The key and what each path reaching through it still has to walk.
        Keep(String, Rest<'p>),
    }

    /// `source_values` looks a member up as the dotted join of the next 1..n
    /// path segments: a member is read iff its key, split on `.`, is a
    /// prefix of some remaining path.
    struct KeySeed<'a, 'p>(&'a [&'p [String]]);

    impl<'de, 'a, 'p> DeserializeSeed<'de> for KeySeed<'a, 'p> {
        type Value = Key<'p>;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Key<'p>, D::Error> {
            d.deserialize_str(self)
        }
    }

    impl<'de, 'a, 'p> Visitor<'de> for KeySeed<'a, 'p> {
        type Value = Key<'p>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON object key")
        }
        fn visit_str<E: Error>(self, s: &str) -> Result<Key<'p>, E> {
            let rest: Rest<'p> = self.0.iter().filter_map(|&path| {
                let mut used = 0;
                for seg in s.split('.') {
                    if used == path.len() || path[used] != seg {
                        return None;
                    }
                    used += 1;
                }
                Some(&path[used..])
            }).collect();
            Ok(if rest.is_empty() { Key::Skip(s.len() as u64) } else { Key::Keep(s.to_owned(), rest) })
        }
    }

    /// [`crate::memory::json_bytes`] of a skipped subtree, built from nothing
    /// (as `memory::json_text_bytes`).
    struct Measure;

    impl<'de> DeserializeSeed<'de> for Measure {
        type Value = u64;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<u64, D::Error> {
            d.deserialize_any(Measure)
        }
    }

    impl<'de> Visitor<'de> for Measure {
        type Value = u64;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON value")
        }
        fn visit_bool<E: Error>(self, _: bool) -> Result<u64, E> {
            Ok(32)
        }
        fn visit_i64<E: Error>(self, _: i64) -> Result<u64, E> {
            Ok(32)
        }
        fn visit_u64<E: Error>(self, _: u64) -> Result<u64, E> {
            Ok(32)
        }
        fn visit_f64<E: Error>(self, _: f64) -> Result<u64, E> {
            Ok(32)
        }
        fn visit_unit<E: Error>(self) -> Result<u64, E> {
            Ok(32)
        }
        fn visit_str<E: Error>(self, s: &str) -> Result<u64, E> {
            Ok(32 + s.len() as u64)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<u64, A::Error> {
            let mut n = 32u64;
            while let Some(x) = a.next_element_seed(Measure)? {
                n = n.saturating_add(x);
            }
            Ok(n)
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<u64, A::Error> {
            let mut n = 48u64;
            while let Some(k) = a.next_key_seed(KeyLen)? {
                n = n.saturating_add(48 + k).saturating_add(a.next_value_seed(Measure)?);
            }
            Ok(n)
        }
    }

    struct KeyLen;

    impl<'de> DeserializeSeed<'de> for KeyLen {
        type Value = u64;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<u64, D::Error> {
            d.deserialize_str(KeyLen)
        }
    }

    impl<'de> Visitor<'de> for KeyLen {
        type Value = u64;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a JSON object key")
        }
        fn visit_str<E: Error>(self, s: &str) -> Result<u64, E> {
            Ok(s.len() as u64)
        }
    }
}

// ES inserts the request-size limit immediately before the first pipe,
// using that pipe's direction (LogicalPlanBuilder.visitStatement).
fn initial_limit(n: usize, pipes: &[(String, usize)], tail_default: bool) -> (usize, bool) {
    pipes.first().map_or((n, tail_default), |(p, k)| (n.min(*k), p == "tail"))
}

// Results are chronological here. Each subsequent pipe narrows the previous
// window, rather than selecting again from the original result set.
fn pipe_window(total: usize, size: usize, tail: bool, pipes: &[(String, usize)]) -> std::ops::Range<usize> {
    let mut start = if tail { total.saturating_sub(size) } else { 0 };
    let mut end = if tail { total } else { total.min(size) };
    for (p, k) in pipes.iter().skip(1) {
        let keep = (end - start).min(*k);
        if p == "tail" {
            start = end - keep;
        } else {
            end = start + keep;
        }
    }
    start..end
}

/// Run an EQL search; returns the response body.
pub fn search(node: &Node, index: &str, params: &HashMap<String, String>, body: &Value) -> Result<Value> {
    let start = Instant::now();
    let mut r = Request::from_json(body)?;
    // URL parameters override the body.
    for (k, slot) in [("allow_partial_search_results", &mut r.allow_partial_search_results), ("allow_partial_sequence_results", &mut r.allow_partial_sequence_results)] {
        if let Some(v) = params.get(k) {
            *slot = v.is_empty() || v == "true";
        }
    }
    let mut q = parse(&r.query, &Opts { case_sensitive: r.case_sensitive })?;
    // A required remote fails every query kind (events, sequences, samples)
    // before any local search; optional remotes are skipped per fetch.
    if crate::ccr::remote::has_remote_expression(index) {
        crate::ccr::remote::ccs_plan(node, index)?;
    }
    resolve_exact(node, index, params, &r.query, r.runtime_mappings.as_ref(), &mut q)?;
    let mut failures = ShardFailures::new();
    let hits_obj = match q {
        EqlQuery::Event { q, pipes } => {
            let (size, tail) = initial_limit(r.size, &pipes, r.tail);
            let (mut hits, f) = fetch(node, index, params, &r, &q, size, tail, &[], None)?;
            failures.merge(&f);
            if tail {
                hits.reverse();
            }
            let window = pipe_window(hits.len(), size, tail, &pipes);
            hits_object("events", hits.drain(window).map(hit_json).collect())
        }
        EqlQuery::Sequence { by, maxspan, stages, until, pipes } => {
            let (size, tail) = initial_limit(r.size, &pipes, r.tail);
            let (mut seqs, f) = sequences(node, index, params, &r, &by, maxspan, &stages, until.as_ref(), size, tail, &pipes)?;
            failures.merge(&f);
            if !failures.is_empty() && !r.allow_partial_sequence_results {
                seqs.clear();
            }
            hits_object("sequences", seqs.into_iter().map(|(k, evs)| sequence_json(k, evs, false)).collect())
        }
        EqlQuery::Sample { by, stages } => {
            let max_per_key = body.get("max_samples_per_key").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
            let (seqs, f) = samples(node, index, params, &r, &by, &stages, max_per_key)?;
            failures.merge(&f);
            hits_object("sequences", seqs.into_iter().take(r.size).map(|(k, evs)| sequence_json(k, evs, true)).collect())
        }
    };
    if !failures.is_empty() && !r.allow_partial_search_results {
        if let Some(e) = failures.first.take() {
            return Err(e);
        }
    }
    phase_add(8, start);
    let mut out = json!({"is_partial": !failures.is_empty(), "is_running": false, "took": start.elapsed().as_millis() as u64, "timed_out": false, "hits": hits_obj});
    if !failures.is_empty() {
        out["shard_failures"] = Value::Array(failures.entries);
    }
    Ok(out)
}

/// [`sequences`]' charge on the `eql_sequence` breaker (ES
/// `SequenceMatcher.trackMemory`): `store` (fetched events and their
/// ordering entries) plus `other` (in-flight and completed sequences),
/// brought up to date (grown, tripping, or shrunk) by [`SeqMemory::track`].
struct SeqMemory {
    held: crate::memory::Reservation,
    store: u64,
    other: u64,
}

impl SeqMemory {
    /// An event's ordering entry (`events`).
    const EVENT: u64 = 64;
    /// A completed sequence's entry, beside its join key.
    const DONE: u64 = 64;

    fn hit(h: &Hit) -> u64 {
        // A projected event is charged for its whole source, as if parsed.
        let source = h.raw.as_ref().map_or_else(|| crate::memory::json_bytes(&h.source), |(_, bytes)| *bytes);
        96 + (h.index.capacity() + h.id.capacity()) as u64 + source + h.fields.as_ref().map_or(0, crate::memory::json_bytes)
    }

    /// An in-flight join key with its `n` stage slots.
    fn key(k: &str, n: usize) -> u64 {
        64 + k.len() as u64 + n as u64 * 32
    }

    fn track(&mut self) -> Result<()> {
        let want = self.store + self.other;
        let held = self.held.bytes();
        if want > held {
            self.held.grow(want - held, "sequence_inflight")
        } else {
            self.held.shrink(held - want);
            Ok(())
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests (`eql_windowed_bench`): nanoseconds per phase: stage searches,
    /// their `search::execute`, their hit parsing, `fetch_sources`,
    /// materializing the returned sequences, the whole request.
    static PHASE_NS: std::cell::RefCell<[u64; 9]> = const { std::cell::RefCell::new([0; 9]) };
}

/// Adds the time since `t` to phase `i` of [`PHASE_NS`] (tests only).
#[inline]
fn phase_add(i: usize, t: Instant) {
    #[cfg(test)]
    PHASE_NS.with(|p| p.borrow_mut()[i] += t.elapsed().as_nanos() as u64);
    #[cfg(not(test))]
    let _ = (i, t);
}

/// Per-stage event cap of a sequence's (and sample's) stage searches.
const STAGE_CAP: usize = 10_000;

#[cfg(test)]
thread_local! {
    /// Tests: skip the head-window page, always matching over the full
    /// per-stage cap (the reference the page must agree with).
    static FULL_STAGES_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The per-stage page a head (`result_position: head` or a first `| head`)
/// sequence first matches over, before falling back to [`STAGE_CAP`]
/// (`None`: match over the cap directly).
///
/// Matching consumes events in timestamp order, so the sequences completed
/// strictly before every full page's last timestamp (the frontier) are
/// those the capped run completes first, in the same order: all of a
/// stage's events before its page's last timestamp are in the page, and
/// none after can complete a sequence earlier. Once `size` of them are
/// complete the head window is decided. Tail windows need the last
/// sequences of the capped set; `![...]` stages look past the sequence's
/// own events; and returned `fields`/runtime fields can fail per fetched
/// hit, so those match over the cap as before.
fn head_page(r: &Request, size: usize, tail: bool, stages: &[Stage]) -> Option<usize> {
    #[cfg(test)]
    if FULL_STAGES_ONLY.with(|f| f.get()) {
        return None;
    }
    if tail || stages.iter().any(|s| s.missing) || r.fields.is_some() || r.runtime_mappings.is_some() {
        return None;
    }
    // ES pages stage queries by `fetch_size`; leave room for the window.
    let page = r.fetch_size.max(size.saturating_mul(2)).max(64);
    (page < STAGE_CAP).then_some(page)
}

#[allow(clippy::too_many_arguments)]
fn sequences(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    by: &[String],
    maxspan: Option<i64>,
    stages: &[Stage],
    until: Option<&Stage>,
    size: usize,
    tail: bool,
    pipes: &[(String, usize)],
) -> Result<(Vec<(Vec<Value>, Vec<Hit>)>, ShardFailures)> {
    let t0 = Instant::now();
    let windowed = windowed_sequences(node, index, params, r, by, maxspan, stages, until, size, tail, pipes)?;
    phase_add(6, t0);
    if let Some(found) = windowed {
        return Ok(found);
    }
    if let Some(page) = head_page(r, size, tail, stages) {
        if let Some(found) = sequences_over(node, index, params, r, by, maxspan, stages, until, size, tail, pipes, Some(page))? {
            return Ok(found);
        }
    }
    Ok(sequences_over(node, index, params, r, by, maxspan, stages, until, size, tail, pipes, None)?.expect("a capped match is always decided"))
}

/// Match over the first `page` events of each stage (`None`: the first
/// [`STAGE_CAP`]). With a page, `None` when the page cannot decide the head
/// window (see [`head_page`]).
#[allow(clippy::too_many_arguments)]
fn sequences_over(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    by: &[String],
    maxspan: Option<i64>,
    stages: &[Stage],
    until: Option<&Stage>,
    size: usize,
    tail: bool,
    pipes: &[(String, usize)],
    page: Option<usize>,
) -> Result<Option<(Vec<(Vec<Value>, Vec<Hit>)>, ShardFailures)>> {
    sequences_over_with(node, index, params, r, by, maxspan, stages, until, size, tail, pipes, page, true)
}

#[cfg(test)]
thread_local! {
    /// Tests: the capped match reads `_source` (the reference for its
    /// doc-value reads).
    static CAPPED_SOURCE_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether `r`'s timestamp is a `date` field (no runtime field) in `t`: its
/// doc values are epoch millis, as the matcher orders events.
fn date_timestamp(t: &crate::search::Target, r: &Request) -> bool {
    let runtime = matches!(t.svc.meta().mappings.runtime, Some(Value::Object(m)) if m.contains_key(&r.timestamp_field));
    !runtime && t.svc.lookup().get(&r.timestamp_field).is_some_and(|f| f.ty == crate::mapping::EsType::Date)
}

/// The join keys a capped match can read from doc values (with the
/// timestamp) instead of `_source` (`None`: read `_source`): as the windowed
/// run's ([`column_keys`]), and only without a tiebreaker (whose value
/// orders events and is not read so), returned `fields` or runtime mappings.
fn capped_columns(node: &Node, index: &str, params: &HashMap<String, String>, r: &Request, by: &[String], stages: &[Stage], until: Option<&Stage>) -> Option<Vec<String>> {
    #[cfg(test)]
    if CAPPED_SOURCE_ONLY.with(|c| c.get()) {
        return None;
    }
    if r.fields.is_some() || r.runtime_mappings.is_some() || r.tiebreaker.is_some() || crate::ccr::remote::has_remote_expression(index) {
        return None;
    }
    let targets = node.resolve(index, eql_resolve_opts(params)).ok()?;
    if targets.is_empty() || !targets.iter().all(|t| date_timestamp(t, r)) {
        return None;
    }
    let mut keys: Vec<String> = by.to_vec();
    for s in stages.iter().filter(|s| !s.missing).chain(until) {
        keys.extend(s.by.iter().cloned());
    }
    let fields: Vec<&str> = keys.iter().map(|k| k.strip_prefix('?').unwrap_or(k)).collect();
    let kinds = key_kinds(&targets, &fields);
    let refs: Vec<&String> = keys.iter().collect();
    column_keys(&targets, &refs, &kinds)
}

/// A capped match's stage read: the timestamp and `columns` from doc values
/// when given (re-read from `_source` when a hit is ambiguous there: a
/// multi-valued key or no timestamp, whose source value only `_source`
/// tells), else `_source` projected on the matcher's paths.
#[allow(clippy::too_many_arguments)]
fn capped_read(node: &Node, index: &str, params: &HashMap<String, String>, r: &Request, q: &EventQuery, cap: usize, keys: &[String], projection: Projection<'_>, columns: Option<&[String]>) -> Result<(Vec<Hit>, ShardFailures)> {
    if let Some(cols) = columns {
        let (hs, f) = fetch_after(node, index, params, r, q, cap, false, keys, projection, None, &[], Some(cols))?;
        if hs.iter().all(|h| h.tb.is_null()) {
            return Ok((hs, f));
        }
    }
    fetch(node, index, params, r, q, cap, false, keys, projection)
}

#[allow(clippy::too_many_arguments)]
fn sequences_over_with(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    by: &[String],
    maxspan: Option<i64>,
    stages: &[Stage],
    until: Option<&Stage>,
    size: usize,
    tail: bool,
    pipes: &[(String, usize)],
    page: Option<usize>,
    use_columns: bool,
) -> Result<Option<(Vec<(Vec<Value>, Vec<Hit>)>, ShardFailures)>> {
    let cap = page.unwrap_or(STAGE_CAP);
    // Stage events from doc values (see `capped_read`); the returned ones'
    // sources are fetched at the end.
    let columns = if use_columns { capped_columns(node, index, params, r, by, stages, until) } else { None };
    let mut failures = ShardFailures::new();
    // Expand `with runs=n`. `![...]` stages take no part in the matching:
    // each is remembered with the number of real stages before it and
    // checked once a candidate sequence is complete.
    let mut expanded: Vec<&Stage> = Vec::new();
    let mut missing: Vec<(usize, &Stage)> = Vec::new();
    for s in stages {
        if s.missing {
            missing.push((expanded.len(), s));
            continue;
        }
        for _ in 0..s.runs {
            expanded.push(s);
        }
    }
    let n = expanded.len();
    // Stage events keep their `_source` as text; the matcher reads only
    // these paths and the returned events are parsed at the end.
    let mut read: Vec<&str> = vec![r.timestamp_field.as_str()];
    read.extend(r.tiebreaker.as_deref());
    read.extend(by.iter().map(String::as_str));
    for s in stages.iter().chain(until) {
        read.extend(s.by.iter().map(String::as_str));
    }
    let paths = lazy_source::paths(read);
    let projection = Some(paths.as_slice());
    // The earliest last timestamp of a full page: only events strictly
    // before it are known to match as over the whole cap.
    let mut frontier: Option<i64> = None;
    let mut bound = |hs: &[Hit]| -> bool {
        if page.is_none() || hs.len() < cap {
            return true;
        }
        // The page's order must be the matcher's (source timestamps follow
        // the search's sort); otherwise match over the cap.
        if hs.windows(2).any(|w| w[1].ts < w[0].ts) {
            return false;
        }
        let last = hs[hs.len() - 1].ts;
        frontier = Some(frontier.map_or(last, |f| f.min(last)));
        true
    };
    // ES `SequenceMatcher.trackMemory`: the fetched events, the in-flight
    // sequences and the completed ones are held on `eql_sequence` while
    // sequences are matched, the charge following their size both ways.
    let mut mem = SeqMemory { held: crate::memory::breakers().reserve_without_breaking(crate::memory::Kind::EqlSequence, 0), store: 0, other: 0 };
    // Events per distinct stage query, tagged with stage numbers.
    let mut events: Vec<(i64, Value, usize, usize, usize)> = Vec::new(); // (ts, tb, kind 0=stage/1=until, stage, idx)
    let mut store: Vec<Hit> = Vec::new();
    let mut stage_keys: Vec<Vec<String>> = Vec::new();
    for (si, s) in expanded.iter().enumerate() {
        let mut keys = by.to_vec();
        keys.extend(s.by.iter().cloned());
        // Only reuse the same syntactic stage expanded by `with runs`.
        // Separate searches can select different events at a capped, tied
        // timestamp boundary (including across replicas). Identical predicates
        // alone therefore do not make their returned event sets interchangeable.
        let first_idx = expanded.iter().position(|x| std::ptr::eq(*x, *s)).unwrap();
        if first_idx != si {
            stage_keys.push(keys);
            let prev: Vec<(i64, Value, usize, usize, usize)> = events.iter().filter(|e| e.2 == 0 && e.3 == first_idx).cloned().collect();
            for mut e in prev {
                e.3 = si;
                events.push(e);
            }
            continue;
        }
        let (hs, f) = capped_read(node, index, params, r, &s.q, cap, &keys, projection, columns.as_deref())?;
        stage_keys.push(keys);
        failures.merge(&f);
        if !bound(hs.as_slice()) {
            return Ok(None);
        }
        for h in hs {
            mem.store += SeqMemory::hit(&h) + SeqMemory::EVENT;
            events.push((h.ts, h.tb.clone(), 0, si, store.len()));
            store.push(h);
        }
        mem.track()?;
    }
    // Events of each `![...]` stage: (ts, join key).
    let mut missing_events: Vec<MissingEvents> = Vec::new();
    for (_, s) in &missing {
        let mut keys = by.to_vec();
        keys.extend(s.by.iter().cloned());
        let (hs, f) = fetch(node, index, params, r, &s.q, cap, false, &keys, projection)?;
        failures.merge(&f);
        let stage = MissingEvents::new(hs.iter().filter_map(|h| join_values(&h.source, &keys).map(|key| (h.ts, serde_json::to_string(&key).unwrap_or_default()))), !by.is_empty());
        mem.store += stage.bytes();
        missing_events.push(stage);
        mem.track()?;
    }
    let mut until_keys = by.to_vec();
    if let Some(u) = until {
        until_keys.extend(u.by.iter().cloned());
        let (hs, f) = capped_read(node, index, params, r, &u.q, cap, &until_keys, projection, columns.as_deref())?;
        failures.merge(&f);
        if !bound(hs.as_slice()) {
            return Ok(None);
        }
        for h in hs {
            mem.store += SeqMemory::hit(&h) + SeqMemory::EVENT;
            events.push((h.ts, h.tb.clone(), 1, 0, store.len()));
            store.push(h);
        }
        mem.track()?;
    }
    // Process in time order; for the same event, later stages first.
    events.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| crate::esql::cmp_vals(&crate::esql::parse::json_to_val(&a.1), &crate::esql::parse::json_to_val(&b.1)))
            // Same document matched by several stages: process the later
            // stage first so an event never pairs with itself.
            .then_with(|| (&store[a.4].index, &store[a.4].id).cmp(&(&store[b.4].index, &store[b.4].id)))
            .then_with(|| b.3.cmp(&a.3))
    });
    // Join key (sequence-level by values) → per-stage partial sequence (event indices).
    let mut partial: HashMap<String, Vec<Option<Vec<usize>>>> = HashMap::new();
    let mut done: Vec<(i64, Vec<Value>, Vec<usize>)> = Vec::new();
    for (i, (ts, _, kind, si, idx)) in events.iter().enumerate() {
        if page.is_some() && (done.len() >= size || frontier.is_some_and(|f| *ts >= f)) {
            // Decided (see `head_page`), or past what the pages cover.
            break;
        }
        if i % 1024 == 1023 {
            mem.track()?;
        }
        let h = &store[*idx];
        if *kind == 1 {
            let Some(jk) = join_values(&h.source, &until_keys) else { continue };
            let k = serde_json::to_string(&jk).unwrap_or_default();
            if let Some(slots) = partial.remove(&k) {
                mem.other = mem.other.saturating_sub(SeqMemory::key(&k, n) + slots.iter().flatten().map(|s| s.len() as u64 * 8).sum::<u64>());
            }
            continue;
        }
        let Some(jk) = join_values(&h.source, &stage_keys[*si]) else { continue };
        let k = serde_json::to_string(&jk).unwrap_or_default();
        if !partial.contains_key(&k) {
            mem.other += SeqMemory::key(&k, n);
        }
        let slots = partial.entry(k).or_insert_with(|| vec![None; n]);
        // Each sequence event is 8 bytes (an index into `store`).
        if *si == 0 && n == 1 {
            mem.other += 8 + SeqMemory::DONE + crate::memory::json_bytes(&Value::Array(jk.clone()));
            done.push((*ts, jk.clone(), vec![*idx]));
        } else if *si == 0 {
            // A new first event replaces the stage's previous start.
            let replaced = slots[0].replace(vec![*idx]).map_or(0, |s| s.len() as u64 * 8);
            mem.other = (mem.other + 8).saturating_sub(replaced);
        } else if let Some(prev) = slots[*si - 1].take() {
            let first_ts = store[prev[0]].ts;
            if let Some(ms) = maxspan {
                if ts - first_ts > ms {
                    // Dropped: past the max span.
                    mem.other = mem.other.saturating_sub(prev.len() as u64 * 8);
                    continue;
                }
            }
            let mut seq = prev;
            seq.push(*idx);
            mem.other += 8;
            if *si + 1 == n {
                // Other in-flight partial sequences of this key stay alive.
                mem.other += SeqMemory::DONE + crate::memory::json_bytes(&Value::Array(jk.clone()));
                done.push((*ts, jk.clone(), seq));
            } else if let Some(old) = slots[*si].replace(seq) {
                // An older partial sequence at this stage is superseded.
                mem.other = mem.other.saturating_sub(old.len() as u64 * 8);
            }
        }
    }
    mem.track()?;
    if page.is_some() && frontier.is_some() && done.len() < size {
        // Too few sequences before the frontier: match over the cap.
        return Ok(None);
    }
    done.sort_by_key(|(ts, _, _)| *ts);
    // A candidate survives only if no event of a `![...]` stage (with the
    // same join key) falls strictly between its neighbouring events; a
    // trailing (leading) missing stage looks after (before) the sequence,
    // within `maxspan`.
    if !missing.is_empty() {
        done.retain(|(_, jk, seq)| {
            let k = serde_json::to_string(jk).unwrap_or_default();
            let first_ts = store[seq[0]].ts;
            missing.iter().zip(&missing_events).all(|((pos, _), evs)| {
                let lo = if *pos == 0 { maxspan.map(|m| first_ts - m - 1) } else { Some(store[seq[pos - 1]].ts) };
                let hi = if *pos >= seq.len() { maxspan.map(|m| first_ts + m + 1) } else { Some(store[seq[*pos]].ts) };
                !evs.between(&k, lo, hi)
            })
        });
    }
    let placeholder = || Hit { index: String::new(), id: String::new(), source: Value::Null, raw: None, fields: None, ts: 0, tb: Value::Null, missing: true };
    // Apply all head/tail windows before parsing full event sources. Matching,
    // missing-event checks and ordering still see the complete candidate set.
    let window = pipe_window(done.len(), size, tail, pipes);
    if columns.is_some() {
        // Doc-value events hold just their keys: read the returned ones'
        // sources now (ES fetches the results' sources alone). A source that
        // cannot be read runs the match again from `_source`.
        let wanted: Vec<(String, String, i64)> = done[window.clone()]
            .iter()
            .flat_map(|(_, _, seq)| seq.iter().map(|ix| &store[*ix]).filter(|h| h.raw.is_none()).map(|h| (h.index.clone(), h.id.clone(), h.ts)))
            .collect();
        if !wanted.is_empty() {
            let Some(sources) = fetch_sources(node, index, params, r, &wanted)? else {
                drop(mem);
                return sequences_over_with(node, index, params, r, by, maxspan, stages, until, size, tail, pipes, page, false);
            };
            for (_, _, seq) in &done[window.clone()] {
                for ix in seq {
                    let h = &mut store[*ix];
                    if h.raw.is_none() {
                        if let Some(src) = sources.get(&(h.index.clone(), h.id.clone())) {
                            h.source = src.clone();
                        }
                    }
                }
            }
        }
    }
    let mut seqs = Vec::with_capacity(window.len());
    for (_, jk, seq) in done.into_iter().skip(window.start).take(window.len()) {
        let mut hits: Vec<Hit> = Vec::with_capacity(seq.len() + missing.len());
        let mut mi = missing.iter().peekable();
        for (i, ix) in seq.iter().enumerate() {
            while mi.peek().is_some_and(|(p, _)| *p == i) {
                mi.next();
                hits.push(placeholder());
            }
            hits.push(materialized(&store[*ix])?);
        }
        hits.extend(mi.map(|_| placeholder()));
        // The materialized (windowed) sequences' event copies are charged
        // too, as they are made (ES `sequence_completed`).
        mem.other += hits.iter().map(SeqMemory::hit).sum::<u64>();
        mem.track()?;
        seqs.push((jk, hits));
    }
    Ok(Some((seqs, failures)))
}

// ------------------------------------------------- windowed sequences
//
// ES (`TumblingWindow`) never reads whole stages: each stage search is a
// `fetch_size` page sorted on the timestamp, later stages start at the
// earliest in-flight sequence and are filtered on the join keys in flight,
// and matching stops as soon as `size` sequences are complete (head) or
// once the newest `fetch_size` first-stage events (tail) yield them. The
// windowed run below does the same with the state machine of
// [`sequences_over`]: a stage's events are read page by page, in matching
// order, only while an in-flight sequence can take them (a stage whose
// previous stage has nothing in flight, or only sequences past `maxspan`,
// is skipped ahead), and filtered on the in-flight join keys. Head windows
// therefore decide exactly what matching every event decides (without the
// per-stage cap); tail windows decide what ES's do: the newest page of
// first-stage events (no timestamp split across pages) matched forward
// from scratch, then the page before it, until `size` sequences are found.

/// Page size of a stage search resumed after it was not needed for a
/// while (it grows back to `fetch_size` page by page).
const WAKE_PAGE: usize = 64;

/// At most this many distinct values of a join key filter a later stage's
/// search (ES `BoxedQueryRequest.MAX_TERMS` is 128; the filter only narrows
/// what is read, so the bound is a cost choice, not a semantic one).
const MAX_KEY_TERMS: usize = 1024;

/// Keyword and ip join keys are filtered on at most this many values (ES's
/// bound): each value is a dictionary lookup in every segment, which with
/// many values and few events to skip costs more than it saves. Integer
/// keys take up to [`MAX_KEY_TERMS`] (one bitset for all values).
const MAX_TERM_KEY_TERMS: usize = 128;

#[cfg(test)]
thread_local! {
    /// Tests: never give up a windowed run for the capped match.
    static UNBOUNDED_WINDOWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: the searches of the last windowed run, and whether it
    /// finished (`Some(true)`) or gave up for the capped match.
    static WINDOW_STATS: std::cell::Cell<(usize, usize, Option<bool>)> = const { std::cell::Cell::new((0, 0, None)) };
    /// Tests: where the last windowed tail run's pages stopped early, as
    /// (page of first-stage events, 1-based; matching position).
    static WINDOW_STOP: std::cell::RefCell<Vec<(usize, MatchPos)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Tests may lift the [`WindowBudget`] and the early give-up of tail runs.
fn unbounded_windows() -> bool {
    #[cfg(test)]
    {
        UNBOUNDED_WINDOWS.with(|u| u.get())
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Searches and fetched events a windowed run may spend before it gives up
/// for the capped match (which then costs what it always did).
struct WindowBudget {
    searches: usize,
    hits: usize,
    max_searches: usize,
    max_hits: usize,
}

impl WindowBudget {
    fn new(cursors: usize, fetch_size: usize) -> Self {
        Self { searches: 0, hits: 0, max_searches: 8 + 4 * cursors, max_hits: STAGE_CAP.max(4 * fetch_size) }
    }

    /// Records one search of `hits` events; false once over budget.
    fn charge(&mut self, hits: usize) -> bool {
        self.searches += 1;
        self.hits += hits;
        unbounded_windows() || (self.searches <= self.max_searches && self.hits <= self.max_hits)
    }
}

/// The values a join-key field's `terms` filter can take without dropping
/// an event the matcher would join (see [`key_kinds`]).
#[derive(Clone, Copy, PartialEq, Debug)]
enum KeyKind {
    Keyword,
    Ip,
    Integer,
    Bool,
}

impl KeyKind {
    fn admits(self, v: &Value) -> bool {
        match self {
            // Floats could index (and print) differently from the source.
            KeyKind::Keyword => v.is_string() || v.is_boolean() || v.as_i64().is_some(),
            KeyKind::Ip => v.as_str().is_some_and(|s| crate::values::parse_ip(s).is_some()),
            // A numeric term goes through f64: exact up to 2^53.
            KeyKind::Integer => v.as_i64().is_some_and(|x| x.unsigned_abs() <= 1 << 53),
            KeyKind::Bool => v.is_boolean(),
        }
    }
}

/// The join-key fields a later stage's search may filter on, by kind: the
/// same keyword, ip, integer or boolean type in every target and no
/// runtime field (whose values the search computes differently).
fn key_kinds(targets: &[crate::search::Target], fields: &[&str]) -> HashMap<String, KeyKind> {
    let mut runtime: Vec<String> = Vec::new();
    for t in targets {
        if let Some(Value::Object(m)) = t.svc.meta().mappings.runtime {
            runtime.extend(m.keys().cloned());
        }
    }
    let mut out = HashMap::new();
    for &f in fields {
        if runtime.iter().any(|x| x == f) || out.contains_key(f) {
            continue;
        }
        let mut kind = None;
        let mut same = true;
        for t in targets {
            let k = t.svc.lookup().get(f).and_then(|rf| match rf.ty {
                crate::mapping::EsType::Keyword => Some(KeyKind::Keyword),
                crate::mapping::EsType::Ip => Some(KeyKind::Ip),
                crate::mapping::EsType::Long | crate::mapping::EsType::Integer | crate::mapping::EsType::Short | crate::mapping::EsType::Byte => Some(KeyKind::Integer),
                crate::mapping::EsType::Boolean => Some(KeyKind::Bool),
                _ => None,
            });
            match (k, kind) {
                (None, _) => same = false,
                (Some(k), None) => kind = Some(k),
                (Some(k), Some(prev)) => same &= k == prev,
            }
        }
        if let (true, Some(k)) = (same, kind) {
            out.insert(f.to_string(), k);
        }
    }
    out
}

/// The join-key fields a windowed run can read from doc values with the
/// values matching reads from `_source` (`None`: read `_source`): keyword,
/// ip, integer or boolean fields with doc values in every target, without
/// a normalizer or `null_value` (doc values hold the indexed form), nor an
/// `ignore_above` on an optional key (a long value is in the source only).
fn column_keys(targets: &[crate::search::Target], keys: &[&String], kinds: &HashMap<String, KeyKind>) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for k in keys {
        let k: &str = k.as_str();
        let optional = k.starts_with('?');
        let name = k.strip_prefix('?').unwrap_or(k);
        kinds.get(name)?;
        let ok = targets.iter().all(|t| {
            t.svc.lookup().get(name).is_some_and(|f| f.doc_values && f.normalizer.is_none() && f.null_value.is_none() && !(optional && f.ignore_above.is_some()))
        });
        if !ok {
            return None;
        }
        if !out.iter().any(|x| x == name) {
            out.push(name.to_string());
        }
    }
    Some(out)
}

/// One stage (or `until`) search's position in a windowed run.
struct Cursor<'s> {
    q: &'s EventQuery,
    /// Join-key fields of its events (the sequence `by` and the stage's).
    keys: Vec<String>,
    /// The first and last expanded stage its events feed (`with runs`);
    /// `None` for `until`.
    stages: Option<(usize, usize)>,
    /// Fetched events not yet matched, in matching order.
    buf: std::collections::VecDeque<Hit>,
    /// The next search starts at this timestamp (`None`: the first event).
    lo: Option<i64>,
    /// No events remain from `lo` on.
    exhausted: bool,
    /// Size of the next search.
    page: usize,
    /// Searched before.
    started: bool,
    /// Its buffer ran dry while no in-flight sequence could use its events.
    paused: bool,
    /// Tail windows: every event of this search from `cache.1` up to
    /// `cache.2` (`None`: the last event), in matching order, as read by
    /// the forward matches of newer first-stage pages. An older page's
    /// match reads them from here instead of searching again.
    cache: (Vec<Hit>, i64, Option<i64>),
    /// This page's contiguous key-unfiltered reads: (events, from, to
    /// (`None`: the last event)); `None` once a read was filtered or
    /// skipped ahead. They become the cache after the match.
    reads: Option<(Vec<Hit>, i64, Option<i64>)>,
    /// This page's reads reached the cache's start (so the cache extends).
    joined: bool,
    /// Tail windows' later stages and `until` keep and reuse their reads.
    caching: bool,
    /// `reads` still grows (no filtered or skipped-ahead read yet).
    reading: bool,
}

impl<'s> Cursor<'s> {
    fn new(q: &'s EventQuery, keys: Vec<String>, stages: Option<(usize, usize)>, page: usize) -> Self {
        Cursor {
            q,
            keys,
            stages,
            buf: std::collections::VecDeque::new(),
            lo: None,
            exhausted: false,
            page,
            started: false,
            paused: false,
            cache: (Vec::new(), 0, Some(0)),
            reads: None,
            joined: false,
            caching: false,
            reading: false,
        }
    }

    /// Whether the cache holds every event at and after `lo` (up to its end).
    fn cached(&self, lo: i64) -> bool {
        let (_, from, to) = &self.cache;
        lo >= *from && to.is_none_or(|to| lo < to)
    }

    /// After a tail page's match: its contiguous reads (joined with the
    /// cache when they reached it) become the cache for the older pages.
    fn keep_reads(&mut self) {
        if let Some((mut hits, from, to)) = self.reads.take() {
            if self.joined {
                let (old, _, old_to) = std::mem::replace(&mut self.cache, (Vec::new(), 0, Some(0)));
                hits.extend(old);
                self.cache = (hits, from, old_to);
            } else if to.is_none_or(|to| to > from) && hits.len() >= self.cache.0.len() {
                // Apart from the cache: keep the larger of the two.
                self.cache = (hits, from, to);
            }
        }
        self.joined = false;
    }

    /// Matching order among cursors on the same event: later stages first,
    /// `until` last (as [`sequences_over`] sorts them).
    fn prio(&self) -> i64 {
        self.stages.map_or(-1, |(a, _)| a as i64)
    }
}

/// Position of a matched event: (timestamp, index, id, cursor priority).
type MatchPos = (i64, String, String, i64);

/// Whether `h` (from a cursor of priority `prio`) comes after `p` in
/// matching order.
fn after_pos(h: &Hit, prio: i64, p: &MatchPos) -> bool {
    (h.ts, &h.index, &h.id).cmp(&(p.0, &p.1, &p.2)).then(p.3.cmp(&prio)) == std::cmp::Ordering::Greater
}

/// An in-flight join key: its values and per-stage partial sequences.
struct Partial {
    key: Vec<Value>,
    slots: Vec<Option<Vec<usize>>>,
}

/// The state machine of [`sequences_over`] fed one event at a time, with
/// the number of keys in flight at each stage.
struct Machine {
    n: usize,
    maxspan: Option<i64>,
    partial: HashMap<String, Partial>,
    live: Vec<usize>,
    /// Completed sequences, in completion order.
    done: Vec<(Vec<Value>, Vec<usize>)>,
}

impl Machine {
    fn new(n: usize, maxspan: Option<i64>) -> Self {
        Machine { n, maxspan, partial: HashMap::new(), live: vec![0; n], done: Vec::new() }
    }

    /// Keys in flight at stages `lo..=hi`.
    fn live(&self, lo: usize, hi: usize) -> usize {
        self.live[lo..=hi.min(self.n - 1)].iter().sum()
    }

    /// The latest start of a sequence in flight at stages `lo..=hi`.
    fn latest_start(&self, lo: usize, hi: usize, store: &[Hit]) -> Option<i64> {
        self.partial.values().flat_map(|p| p.slots[lo..=hi].iter().flatten()).map(|seq| store[seq[0]].ts).max()
    }

    /// Stage `si` event `idx` (key `jk`, serialized `k`): the matcher step
    /// of [`sequences_over`], with the same breaker charges.
    #[allow(clippy::too_many_arguments)]
    fn stage(&mut self, si: usize, idx: usize, ts: i64, jk: &[Value], k: &str, store: &[Hit], mem: &mut SeqMemory) {
        let n = self.n;
        if !self.partial.contains_key(k) {
            mem.other += SeqMemory::key(k, n);
        }
        let p = self.partial.entry(k.to_string()).or_insert_with(|| Partial { key: jk.to_vec(), slots: vec![None; n] });
        if si == 0 && n == 1 {
            mem.other += 8 + SeqMemory::DONE + crate::memory::json_bytes(&Value::Array(jk.to_vec()));
            self.done.push((jk.to_vec(), vec![idx]));
        } else if si == 0 {
            // A new first event replaces the stage's previous start.
            let replaced = p.slots[0].replace(vec![idx]);
            if replaced.is_none() {
                self.live[0] += 1;
            }
            mem.other = (mem.other + 8).saturating_sub(replaced.map_or(0, |s| s.len() as u64 * 8));
        } else if let Some(prev) = p.slots[si - 1].take() {
            self.live[si - 1] -= 1;
            let first_ts = store[prev[0]].ts;
            if let Some(ms) = self.maxspan {
                if ts - first_ts > ms {
                    // Dropped: past the max span.
                    mem.other = mem.other.saturating_sub(prev.len() as u64 * 8);
                    return;
                }
            }
            let mut seq = prev;
            seq.push(idx);
            mem.other += 8;
            if si + 1 == n {
                mem.other += SeqMemory::DONE + crate::memory::json_bytes(&Value::Array(jk.to_vec()));
                self.done.push((jk.to_vec(), seq));
            } else {
                match p.slots[si].replace(seq) {
                    None => self.live[si] += 1,
                    // An older partial sequence at this stage is superseded.
                    Some(old) => mem.other = mem.other.saturating_sub(old.len() as u64 * 8),
                }
            }
        }
    }

    /// An `until` event of key `k` ends the key's in-flight sequences.
    fn until(&mut self, k: &str, mem: &mut SeqMemory) {
        if let Some(p) = self.partial.remove(k) {
            for (i, s) in p.slots.iter().enumerate() {
                if s.is_some() {
                    self.live[i] -= 1;
                }
            }
            mem.other = mem.other.saturating_sub(SeqMemory::key(k, self.n) + p.slots.iter().flatten().map(|s| s.len() as u64 * 8).sum::<u64>());
        }
    }
}

/// A windowed run's searches, breaker charge and matched events.
struct Run<'a> {
    node: &'a Node,
    index: &'a str,
    params: &'a HashMap<String, String>,
    r: &'a Request,
    projection: Projection<'a>,
    failures: ShardFailures,
    mem: SeqMemory,
    budget: WindowBudget,
    /// Matched events (sequences hold indices into it).
    store: Vec<Hit>,
    kinds: HashMap<String, KeyKind>,
    /// Searches resumed after a pause (see [`Run::pass`]).
    wakes: usize,
    /// The last refill was key-filtered or came from the cache (not an
    /// unfiltered search the pause/resume rule counts).
    filtered_read: bool,
    /// Stage searches read the timestamp and these join keys from doc
    /// values instead of `_source` ([`hits_of_columns`]); the returned
    /// events' sources are fetched at the end ([`fetch_sources`]).
    columns: Option<Vec<String>>,
}

impl Run<'_> {
    /// One stage search; `None` when over budget, or when the events'
    /// source timestamps do not follow the search's order and bound (the
    /// windows then cannot be trusted).
    #[allow(clippy::too_many_arguments)]
    fn search(&mut self, q: &EventQuery, keys: &[String], size: usize, desc: bool, from: Option<i64>, extra: &[Value]) -> Result<Option<Vec<Hit>>> {
        let t0 = Instant::now();
        let (hs, f) = fetch_after(self.node, self.index, self.params, self.r, q, size, desc, keys, self.projection, from, extra, self.columns.as_deref())?;
        phase_add(0, t0);
        self.failures.merge(&f);
        #[cfg(test)]
        WINDOW_STATS.with(|s| {
            let (n, h, done) = s.get();
            s.set((n + 1, h + hs.len(), done));
        });
        if !self.budget.charge(hs.len()) {
            return Ok(None);
        }
        let ordered = hs.windows(2).all(|w| if desc { w[1].ts <= w[0].ts } else { w[0].ts <= w[1].ts });
        let bounded = from.is_none_or(|at| hs.iter().all(|h| if desc { h.ts <= at } else { h.ts >= at }));
        // Doc-values hits marked ambiguous (see `hits_of_columns`).
        let ambiguous = hs.iter().any(|h| !h.tb.is_null());
        if !ordered || !bounded || ambiguous {
            return Ok(None);
        }
        for h in &hs {
            self.mem.store += SeqMemory::hit(h) + SeqMemory::EVENT;
        }
        self.mem.track()?;
        Ok(Some(hs))
    }

    /// `terms` filters on the cursor's join keys for the keys that can
    /// still reach its stages, and the timestamp before which they hold
    /// (`None`: no filter).
    fn key_filters(&self, cursors: &[Cursor<'_>], ci: usize, m: &Machine, stage_keys: &[Vec<String>], lo: Option<i64>) -> Option<(Vec<Value>, Option<i64>)> {
        let (a, b) = cursors[ci].stages?;
        if a == 0 || ci == 0 {
            return None;
        }
        // In-flight keys at the stages feeding this cursor, and those the
        // previous stage's fetched events may still bring before its next
        // search (`horizon`); later ones are unknown.
        let feeder = &cursors[ci - 1];
        if !feeder.exhausted && feeder.lo.is_none() {
            return None;
        }
        let horizon = if feeder.exhausted { None } else { feeder.lo };
        if horizon.is_some_and(|h| lo.is_some_and(|l| h <= l)) {
            return None;
        }
        let fields = &cursors[ci].keys;
        let width = fields.len();
        if width == 0 {
            return None;
        }
        let mut keys: Vec<Vec<Value>> = m.partial.values().filter(|p| p.slots[a - 1..b].iter().any(Option::is_some)).map(|p| p.key.clone()).collect();
        keys.extend(feeder.buf.iter().filter_map(|h| join_values(&h.source, &stage_keys[a - 1])));
        if keys.is_empty() || keys.iter().any(|k| k.len() != width) {
            return None;
        }
        let mut filters = Vec::new();
        for (j, field) in fields.iter().enumerate() {
            let name = field.strip_prefix('?').unwrap_or(field);
            let Some(kind) = self.kinds.get(name) else { continue };
            let limit = if *kind == KeyKind::Integer { MAX_KEY_TERMS } else { MAX_TERM_KEY_TERMS };
            let mut seen: HashSet<String> = HashSet::new();
            let mut values: Vec<Value> = Vec::new();
            let mut ok = true;
            for k in &keys {
                let v = &k[j];
                if !kind.admits(v) {
                    ok = false;
                    break;
                }
                if seen.insert(v.to_string()) {
                    values.push(v.clone());
                    if values.len() > limit {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                filters.push(json!({"terms": { name.to_string(): values }}));
            }
        }
        (!filters.is_empty()).then_some((filters, horizon))
    }

    /// Fill cursor `ci`'s buffer with its next page (after the matching
    /// position `pos`). `None`: over budget.
    fn refill(&mut self, cursors: &mut [Cursor<'_>], ci: usize, m: &Machine, stage_keys: &[Vec<String>], pos: Option<&MatchPos>) -> Result<Option<()>> {
        let fetch_size = self.r.fetch_size;
        let q = cursors[ci].q;
        let keys = cursors[ci].keys.clone();
        let prio = cursors[ci].prio();
        let lo = match (cursors[ci].lo, pos.map(|p| p.0)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let waking = cursors[ci].paused && cursors[ci].started;
        if let Some(l) = lo.filter(|&l| cursors[ci].caching && cursors[ci].cached(l)) {
            // A newer page's match read these already: no search.
            let c = &mut cursors[ci];
            let mut hits: Vec<Hit> = c.cache.0.iter().filter(|h| h.ts >= l).map(hit_copy).collect();
            hits.sort_by(|x, y| (x.ts, &x.index, &x.id).cmp(&(y.ts, &y.index, &y.id)));
            if let Some(p) = pos {
                hits.retain(|h| after_pos(h, prio, p));
            }
            if c.reads.as_ref().is_some_and(|r| r.2 == Some(c.cache.1)) {
                c.joined = true;
            }
            c.reading = false;
            c.started = true;
            c.paused = false;
            match c.cache.2 {
                Some(to) => c.lo = Some(to),
                None => c.exhausted = true,
            }
            c.buf.extend(hits);
            // No search: not a resume the thrash rule counts.
            self.filtered_read = true;
            return Ok(Some(()));
        }
        let (extra, horizon) = self.key_filters(cursors, ci, m, stage_keys, lo).unwrap_or_default();
        // A resumed unfiltered search starts small (a dense stage next to a
        // sparse one needs a few events per resume). A key-filtered one
        // reads a full page: its events are sparse, and buffered ones carry
        // it over the following gaps (paused stages keep their buffer)
        // instead of a search per gap.
        self.filtered_read = !extra.is_empty();
        let small = waking && extra.is_empty();
        let mut size = if small { WAKE_PAGE.min(fetch_size) } else { cursors[ci].page.max(if waking { fetch_size } else { 0 }) };
        loop {
            let Some(mut hits) = self.search(q, &keys, size, false, lo, &extra)? else { return Ok(None) };
            // `next`: where the following search starts (`None`: no more).
            let mut next = None;
            if hits.len() >= size {
                // The last timestamp may continue past the page: leave it
                // to the next search.
                let last = hits[hits.len() - 1].ts;
                let keep = hits.partition_point(|h| h.ts < last);
                if keep == 0 {
                    // One timestamp fills the page.
                    size = size.saturating_mul(2);
                    continue;
                }
                hits.truncate(keep);
                next = Some(last);
            }
            if let Some(h) = horizon {
                // The key filters hold for events before the horizon only.
                hits.truncate(hits.partition_point(|x| x.ts < h));
                next = Some(next.map_or(h, |n: i64| n.min(h)));
            }
            let c = &mut cursors[ci];
            if c.caching {
                // Stop at the cache: the next refill reads from it.
                let from = c.cache.1;
                let cache_has = c.cache.2.is_none_or(|to| to > from);
                if cache_has && lo.is_some_and(|l| l < from) && next.is_none_or(|n| n > from) {
                    hits.truncate(hits.partition_point(|x| x.ts < from));
                    next = Some(from);
                }
                // Keep contiguous unfiltered reads for the older pages.
                if !extra.is_empty() {
                    c.reading = false;
                } else if c.reading && !c.joined {
                    let l = lo.unwrap_or(i64::MIN);
                    match &mut c.reads {
                        None => c.reads = Some((hits.iter().map(hit_copy).collect(), l, next)),
                        Some(r) if r.2 == Some(l) => {
                            r.0.extend(hits.iter().map(hit_copy));
                            r.2 = next;
                        }
                        Some(_) => c.reading = false,
                    }
                }
            }
            hits.sort_by(|x, y| (x.ts, &x.index, &x.id).cmp(&(y.ts, &y.index, &y.id)));
            if let Some(p) = pos {
                // Resumed mid-timestamp: what matching has passed is spent.
                hits.retain(|h| after_pos(h, prio, p));
            }
            let c = &mut cursors[ci];
            c.started = true;
            c.paused = false;
            // Pages double: from a resumed search's small page back to
            // `fetch_size`, else from `fetch_size` up to 8x (fewer searches
            // over long reads; the page size decides nothing).
            let cap = fetch_size.saturating_mul(8).min(STAGE_CAP).max(fetch_size).max(size);
            c.page = if small { size.saturating_mul(2).min(fetch_size.max(size)) } else { size.saturating_mul(2).max(fetch_size).min(cap) };
            match next {
                Some(n) => c.lo = Some(n),
                None => c.exhausted = true,
            }
            c.buf.extend(hits);
            return Ok(Some(()));
        }
    }

    /// Whether cursor `ci`'s events can change anything from here on.
    fn needed(&self, c: &Cursor<'_>, m: &Machine, head: bool) -> bool {
        match c.stages {
            None => m.live(0, m.n - 1) > 0,
            Some((0, _)) => head,
            Some((a, b)) => {
                if m.live(a - 1, b - 1) == 0 {
                    return false;
                }
                // Sequences past their max span cannot take later events.
                match m.maxspan {
                    None => true,
                    Some(ms) => m.latest_start(a - 1, b - 1, &self.store).is_some_and(|s| c.lo.is_none_or(|lo| s.saturating_add(ms) >= lo)),
                }
            }
        }
    }

    /// Match in order from the cursors' positions until `head` sequences
    /// are complete, or (tail) until the fixed first-stage events are spent
    /// and no sequence is in flight. `None`: over budget.
    ///
    /// `tail_stop`: (sequences still wanted, the last stage's cursor, the
    /// window size). A tail
    /// pass stops once that many are complete and the last stage's page is
    /// spent, as ES's tail windows stop after the last stage's page that
    /// reaches the limit (`SequenceMatcher.tailLimitReached`); the position
    /// is returned.
    #[allow(clippy::too_many_arguments)]
    fn pass(&mut self, cursors: &mut [Cursor<'_>], m: &mut Machine, stage_keys: &[Vec<String>], until_keys: &[String], head: Option<usize>, tail_stop: Option<(usize, usize, usize)>) -> Result<Option<Option<MatchPos>>> {
        let t0 = Instant::now();
        let out = self.pass_inner(cursors, m, stage_keys, until_keys, head, tail_stop);
        phase_add(7, t0);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn pass_inner(&mut self, cursors: &mut [Cursor<'_>], m: &mut Machine, stage_keys: &[Vec<String>], until_keys: &[String], head: Option<usize>, tail_stop: Option<(usize, usize, usize)>) -> Result<Option<Option<MatchPos>>> {
        let mut pos: Option<MatchPos> = None;
        let mut matched = 0usize;
        loop {
            if head.is_some_and(|size| m.done.len() >= size) {
                break;
            }
            if let Some((need, last, _)) = tail_stop {
                if m.done.len() >= need && cursors[last].buf.is_empty() && pos.is_some() {
                    self.mem.track()?;
                    return Ok(Some(pos));
                }
            }
            if head.is_none() && cursors[0].buf.is_empty() && m.live(0, m.n - 1) == 0 {
                break;
            }
            // A cursor that may hold the next event is filled first (a
            // search can come back empty: matching passed it, or the key
            // filters' horizon cut it).
            for ci in 0..cursors.len() {
                while cursors[ci].buf.is_empty() && !cursors[ci].exhausted {
                    if !self.needed(&cursors[ci], m, head.is_some()) {
                        cursors[ci].paused = true;
                        break;
                    }
                    let waking = cursors[ci].paused && cursors[ci].started;
                    // A tail page still reading on (a further page of a
                    // stage) with half the budget spent and under a quarter
                    // of the sequences found will not fit: give up now
                    // rather than at the budget (the capped match runs next).
                    if let Some((need, _, size)) = tail_stop {
                        // Found so far: by the newer pages, and by this one.
                        let found = size.saturating_sub(need) + m.done.len();
                        let poor = self.budget.hits.saturating_mul(2) > self.budget.max_hits && found.saturating_mul(4) < size;
                        if cursors[ci].started && poor && !unbounded_windows() {
                            return Ok(None);
                        }
                    }
                    if self.refill(cursors, ci, m, stage_keys, pos.as_ref())?.is_none() {
                        return Ok(None);
                    }
                    // A later stage dense next to a sparse earlier one (no
                    // or few keys) pauses and resumes once per earlier
                    // event: when the page's remaining first-stage events
                    // cannot fit the budget that way, give up now.
                    // (Key-filtered resumes read full pages and are not
                    // counted: their searches are few and selective.)
                    if waking && !self.filtered_read {
                        self.wakes += 1;
                        let left = self.budget.max_searches.saturating_sub(self.budget.searches);
                        if head.is_none() && self.wakes >= 4 && cursors[0].buf.len() > left && !unbounded_windows() {
                            return Ok(None);
                        }
                    }
                }
            }
            // The next event in matching order.
            let mut best: Option<usize> = None;
            for ci in 0..cursors.len() {
                let Some(h) = cursors[ci].buf.front() else { continue };
                let earlier = match best {
                    None => true,
                    Some(b) => {
                        let o = cursors[b].buf.front().expect("a candidate has an event");
                        (h.ts, &h.index, &h.id).cmp(&(o.ts, &o.index, &o.id)).then(cursors[b].prio().cmp(&cursors[ci].prio())) == std::cmp::Ordering::Less
                    }
                };
                if earlier {
                    best = Some(ci);
                }
            }
            let Some(ci) = best else { break };
            let prio = cursors[ci].prio();
            let stages = cursors[ci].stages;
            let h = cursors[ci].buf.pop_front().expect("the chosen cursor has an event");
            pos = Some((h.ts, h.index.clone(), h.id.clone(), prio));
            matched += 1;
            if matched % 1024 == 0 {
                self.mem.track()?;
            }
            let idx = self.store.len();
            let ts = h.ts;
            match stages {
                None => {
                    let k = join_values(&h.source, until_keys).map(|jk| serde_json::to_string(&jk).unwrap_or_default());
                    self.store.push(h);
                    if let Some(k) = k {
                        m.until(&k, &mut self.mem);
                    }
                }
                Some((a, b)) => {
                    let jk = join_values(&h.source, &stage_keys[a]);
                    self.store.push(h);
                    if let Some(jk) = jk {
                        let k = serde_json::to_string(&jk).unwrap_or_default();
                        // The same event at several stages: later stages first.
                        for si in (a..=b).rev() {
                            m.stage(si, idx, ts, &jk, &k, &self.store, &mut self.mem);
                        }
                    }
                }
            }
        }
        self.mem.track()?;
        Ok(Some(None))
    }
}

/// Sequences matched over windows of the stage searches (see the section
/// comment): head windows match from the first events and stop at `size`
/// complete sequences; tail windows take the newest `fetch_size` first-stage
/// events, match forward from them, and step back a page of first-stage
/// events at a time until `size` sequences are complete (sequences found
/// again from an older page are not repeated). `None` when the query or
/// the data does not fit (then [`sequences_over`] runs as before): `![..]`
/// stages, returned `fields` or runtime mappings, a tiebreaker, a timestamp
/// that is not a `date` field, a `with runs` first stage of a tail window,
/// or a run over its search budget.
#[allow(clippy::too_many_arguments)]
fn windowed_sequences(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    by: &[String],
    maxspan: Option<i64>,
    stages: &[Stage],
    until: Option<&Stage>,
    size: usize,
    tail: bool,
    pipes: &[(String, usize)],
) -> Result<Option<(Vec<(Vec<Value>, Vec<Hit>)>, ShardFailures)>> {
    #[cfg(test)]
    {
        WINDOW_STATS.with(|s| s.set((0, 0, None)));
        WINDOW_STOP.with(|w| w.borrow_mut().clear());
        if FULL_STAGES_ONLY.with(|f| f.get()) {
            return Ok(None);
        }
    }
    if size == 0 || r.fetch_size == 0 || stages.is_empty() || stages.iter().any(|s| s.missing) {
        return Ok(None);
    }
    if r.fields.is_some() || r.runtime_mappings.is_some() || r.tiebreaker.is_some() || crate::ccr::remote::has_remote_expression(index) {
        return Ok(None);
    }
    if tail && stages[0].runs != 1 {
        return Ok(None);
    }
    let targets = node.resolve(index, eql_resolve_opts(params)).unwrap_or_default();
    // `search_after` steps whole milliseconds on a `date` timestamp.
    if targets.is_empty() || !targets.iter().all(|t| date_timestamp(t, r)) {
        return Ok(None);
    }
    let mut stage_keys: Vec<Vec<String>> = Vec::new();
    for s in stages {
        for _ in 0..s.runs {
            let mut keys = by.to_vec();
            keys.extend(s.by.iter().cloned());
            stage_keys.push(keys);
        }
    }
    let n = stage_keys.len();
    if n < 2 {
        return Ok(None);
    }
    let mut until_keys = by.to_vec();
    if let Some(u) = until {
        until_keys.extend(u.by.iter().cloned());
    }
    // As in `sequences_over`: the matcher reads these source paths only.
    let mut read: Vec<&str> = vec![r.timestamp_field.as_str()];
    read.extend(by.iter().map(String::as_str));
    for s in stages.iter().chain(until) {
        read.extend(s.by.iter().map(String::as_str));
    }
    let paths = lazy_source::paths(read);
    let key_fields: Vec<&str> = stage_keys.iter().flatten().map(|k| k.strip_prefix('?').unwrap_or(k)).collect();
    // A head window usually completes early: its searches start small and
    // double up to `fetch_size`. Tail pages are `fetch_size` (ES's window).
    let first_page = if tail { r.fetch_size } else { size.saturating_mul(2).max(WAKE_PAGE).min(r.fetch_size) };
    let mut cursors: Vec<Cursor<'_>> = Vec::new();
    let mut first = 0;
    for s in stages {
        cursors.push(Cursor::new(&s.q, stage_keys[first].clone(), Some((first, first + s.runs - 1)), first_page));
        first += s.runs;
    }
    if let Some(u) = until {
        cursors.push(Cursor::new(&u.q, until_keys.clone(), None, first_page));
    }
    let mut run = Run {
        node,
        index,
        params,
        r,
        projection: Some(paths.as_slice()),
        failures: ShardFailures::new(),
        mem: SeqMemory { held: crate::memory::breakers().reserve_without_breaking(crate::memory::Kind::EqlSequence, 0), store: 0, other: 0 },
        budget: WindowBudget::new(cursors.len(), r.fetch_size),
        store: Vec::new(),
        kinds: key_kinds(&targets, &key_fields),
        wakes: 0,
        filtered_read: false,
        columns: None,
    };
    let all_keys: Vec<&String> = stage_keys.iter().flatten().chain(&until_keys).collect();
    // Doc values pay off when the stage events read outnumber the returned
    // ones (whose sources are then fetched by id): not for large windows.
    run.columns = column_keys(&targets, &all_keys, &run.kinds);
    if run.columns.is_some() {
        // Doc-value events are a few dozen bytes: read more before giving up.
        run.budget.max_hits = run.budget.max_hits.saturating_mul(4);
    }
    let mut done: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
    if !tail {
        let mut m = Machine::new(n, maxspan);
        if run.pass(&mut cursors, &mut m, &stage_keys, &until_keys, Some(size), None)?.is_none() {
            return Ok(None);
        }
        done = m.done;
    } else {
        // Pages of first-stage events, newest first, as ES's tail windows
        // take them: `fetch_size` events, without the oldest timestamp
        // when the page may split it. They are cut from `pending`, read
        // newest first in larger searches (whole timestamps only): `before`
        // bounds the next search (inclusive); `None` in `more` once read
        // out.
        let mut pending: std::collections::VecDeque<Hit> = std::collections::VecDeque::new();
        let mut before: Option<i64> = None;
        let mut more = true;
        // A read takes what the next page lacks plus a quarter page (so
        // it usually ends past the page's oldest timestamp), more when one
        // timestamp fills it. Reading ahead in larger steps would read
        // first-stage events of pages a low-yield run never gets to.
        let mut read_min = 0usize;
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        let mut pages = 0usize;
        loop {
            let mut page = r.fetch_size;
            let (mut events, older) = loop {
                if more && pending.len() < page {
                    let want = (page - pending.len() + r.fetch_size / 4 + 1).max(read_min);
                    let Some(mut hits) = run.search(&stages[0].q, &stage_keys[0], want, true, before, &[])? else { return Ok(None) };
                    if hits.len() < want {
                        more = false;
                    } else {
                        let oldest = hits[hits.len() - 1].ts;
                        let keep = hits.partition_point(|h| h.ts > oldest);
                        if keep == 0 {
                            read_min = want.saturating_mul(2);
                            continue;
                        }
                        hits.truncate(keep);
                        before = Some(oldest);
                    }
                    read_min = 0;
                    pending.extend(hits);
                    continue;
                }
                if pending.len() < page {
                    break (pending.drain(..).collect::<Vec<Hit>>(), None);
                }
                // The oldest timestamp may continue past the page.
                let oldest = pending[page - 1].ts;
                let keep = pending.iter().take(page).take_while(|h| h.ts > oldest).count();
                if keep == 0 {
                    page = page.saturating_mul(2);
                    continue;
                }
                let chunk: Vec<Hit> = pending.drain(..keep).collect();
                break (chunk, Some(oldest));
            };
            if events.is_empty() {
                break;
            }
            pages += 1;
            if pages > 1 {
                // Each older page searches its own stretch of the later
                // stages (newer ones come from the cache).
                run.budget.max_searches += cursors.len() + 1;
            }
            events.sort_by(|x, y| (x.ts, &x.index, &x.id).cmp(&(y.ts, &y.index, &y.id)));
            let start = events[0].ts;
            // Match forward from the page's first event, from scratch: the
            // first stage reads the page only, later stages (and `until`)
            // everything from its first timestamp on.
            for (ci, c) in cursors.iter_mut().enumerate() {
                c.buf.clear();
                c.started = false;
                c.paused = false;
                c.page = r.fetch_size;
                if ci == 0 {
                    c.exhausted = true;
                } else {
                    c.exhausted = false;
                    c.lo = Some(start);
                    c.caching = true;
                    c.reading = true;
                    c.reads = None;
                    c.joined = false;
                }
            }
            cursors[0].buf.extend(events);
            let mut m = Machine::new(n, maxspan);
            let last = stages.len() - 1;
            let Some(stopped) = run.pass(&mut cursors, &mut m, &stage_keys, &until_keys, None, Some((size - done.len(), last, size)))? else {
                return Ok(None);
            };
            #[cfg(test)]
            if let Some(p) = &stopped {
                WINDOW_STOP.with(|w| w.borrow_mut().push((pages, p.clone())));
            }
            #[cfg(not(test))]
            let _ = &stopped;
            for c in cursors.iter_mut().skip(1) {
                c.keep_reads();
            }
            for (jk, seq) in m.done {
                let last = &run.store[*seq.last().expect("a sequence has events")];
                if seen.insert((serde_json::to_string(&jk).unwrap_or_default(), last.index.clone(), last.id.clone())) {
                    done.push((jk, seq));
                }
            }
            if done.len() >= size {
                break;
            }
            // No older page. (`before` stays where `pending`'s reads
            // stopped: the older pages' events are partly read already.)
            if older.is_none() {
                break;
            }
            // Older pages match about as many sequences as these did: give
            // up early (for the capped match) when that cannot fit.
            if !unbounded_windows() {
                if done.is_empty() {
                    return Ok(None);
                }
                let per_page = run.budget.hits / pages;
                let pages_left = (size - done.len()).saturating_mul(pages).div_ceil(done.len());
                if run.budget.hits.saturating_add(pages_left.saturating_mul(per_page)) > run.budget.max_hits {
                    return Ok(None);
                }
            }
        }
    }
    // Completion order (the order `sequences_over` completes them in).
    done.sort_by(|x, y| {
        let (a, b) = (&run.store[*x.1.last().expect("events")], &run.store[*y.1.last().expect("events")]);
        (a.ts, &a.index, &a.id).cmp(&(b.ts, &b.index, &b.id))
    });
    let window = pipe_window(done.len(), size, tail, pipes);
    if run.columns.is_some() {
        // Doc-values events: read the returned ones' sources now.
        let wanted: Vec<(String, String, i64)> = done[window.clone()].iter().flat_map(|(_, seq)| seq.iter().map(|ix| (run.store[*ix].index.clone(), run.store[*ix].id.clone(), run.store[*ix].ts))).collect();
        let t0 = Instant::now();
        let fetched = fetch_sources(node, index, params, r, &wanted)?;
        phase_add(3, t0);
        let Some(mut sources) = fetched else { return Ok(None) };
        // Each source moves to its last use (a doc may be several events).
        let mut uses: HashMap<(String, String), usize> = HashMap::new();
        for (i, id, _) in &wanted {
            *uses.entry((i.clone(), id.clone())).or_default() += 1;
        }
        for (_, seq) in &done[window.clone()] {
            for ix in seq {
                let h = &mut run.store[*ix];
                let key = (std::mem::take(&mut h.index), std::mem::take(&mut h.id));
                let left = uses.get_mut(&key).map(|n| {
                    *n -= 1;
                    *n
                });
                let src = match left {
                    Some(0) => sources.remove(&key),
                    _ => sources.get(&key).cloned(),
                };
                (h.index, h.id) = key;
                if let Some(src) = src {
                    h.source = src;
                }
            }
        }
        sources.clear();
    }
    let t_mat = Instant::now();
    // A store event returned once moves into its sequence (else copies).
    let mut returned: HashMap<usize, usize> = HashMap::new();
    for (_, seq) in &done[window.clone()] {
        for ix in seq {
            *returned.entry(*ix).or_default() += 1;
        }
    }
    let mut seqs = Vec::with_capacity(window.len());
    for (jk, seq) in done.into_iter().skip(window.start).take(window.len()) {
        let hits = seq
            .iter()
            .map(|ix| match returned.get(ix) {
                Some(1) if run.store[*ix].raw.is_none() => {
                    let h = &mut run.store[*ix];
                    Ok(Hit { index: std::mem::take(&mut h.index), id: std::mem::take(&mut h.id), source: std::mem::take(&mut h.source), raw: None, fields: h.fields.take(), ts: h.ts, tb: std::mem::take(&mut h.tb), missing: h.missing })
                }
                _ => materialized(&run.store[*ix]),
            })
            .collect::<Result<Vec<Hit>>>()?;
        run.mem.other += hits.iter().map(SeqMemory::hit).sum::<u64>();
        run.mem.track()?;
        seqs.push((jk, hits));
    }
    phase_add(4, t_mat);
    #[cfg(test)]
    WINDOW_STATS.with(|s| {
        let (n, h, _) = s.get();
        s.set((n, h, Some(true)));
    });
    Ok(Some((seqs, run.failures)))
}

/// Negative stages are queried repeatedly for completed sequences. Index their
/// timestamps once instead of scanning the entire stage for each candidate.
/// With no sequence-level `by`, retain the existing key-independent semantics.
struct MissingEvents {
    keyed: bool,
    timestamps: HashMap<String, Vec<i64>>,
}

impl MissingEvents {
    fn new(events: impl IntoIterator<Item = (i64, String)>, keyed: bool) -> Self {
        let mut timestamps: HashMap<String, Vec<i64>> = HashMap::new();
        for (ts, key) in events {
            timestamps.entry(if keyed { key } else { String::new() }).or_default().push(ts);
        }
        for times in timestamps.values_mut() {
            times.sort_unstable();
        }
        Self { keyed, timestamps }
    }

    /// Estimated bytes the index holds (each key, its timestamps and the
    /// map entry), for the `eql_sequence` breaker.
    fn bytes(&self) -> u64 {
        self.timestamps.iter().map(|(k, times)| 56 + k.capacity() as u64 + 8 * times.capacity() as u64).sum()
    }

    fn between(&self, key: &str, lo: Option<i64>, hi: Option<i64>) -> bool {
        let Some(times) = self.timestamps.get(if self.keyed { key } else { "" }) else { return false };
        // Bounds are open; use comparisons rather than +/- 1 so extrema are safe.
        let start = lo.map_or(0, |lo| times.partition_point(|&ts| ts <= lo));
        times.get(start).is_some_and(|&ts| hi.is_none_or(|hi| ts < hi))
    }
}

/// Optional join keys admit null; ordinary keys reject missing/null values.
fn join_values(source: &Value, keys: &[String]) -> Option<Vec<Value>> {
    keys.iter().map(|key| {
        let optional = key.starts_with('?');
        let value = first_value(source, key.strip_prefix('?').unwrap_or(key));
        if value.is_null() && !optional { None } else { Some(value) }
    }).collect()
}

/// `sample by k [..] [..]`: for every join key present in all stages, up to
/// `max_per_key` samples of distinct events (one per stage, unordered in
/// time), ordered by join key.
fn samples(
    node: &Node,
    index: &str,
    params: &HashMap<String, String>,
    r: &Request,
    by: &[String],
    stages: &[Stage],
    max_per_key: usize,
) -> Result<(Vec<(Vec<Value>, Vec<Hit>)>, ShardFailures)> {
    let mut failures = ShardFailures::new();
    // Stage events keep their `_source` as text; grouping reads the keys.
    let paths = lazy_source::paths(std::iter::once(r.timestamp_field.as_str()).chain(r.tiebreaker.as_deref()).chain(by.iter().map(String::as_str)).chain(stages.iter().flat_map(|s| s.by.iter().map(String::as_str))));
    // Per stage: join key → events (time order).
    let mut per_stage: Vec<HashMap<String, (Vec<Value>, Vec<Hit>)>> = Vec::new();
    for s in stages {
        let mut keys = by.to_vec();
        keys.extend(s.by.iter().cloned());
        let (hs, f) = fetch(node, index, params, r, &s.q, STAGE_CAP, false, &keys, Some(paths.as_slice()))?;
        failures.merge(&f);
        let mut m: HashMap<String, (Vec<Value>, Vec<Hit>)> = HashMap::new();
        for h in hs {
            let Some(jk) = join_values(&h.source, &keys) else { continue };
            m.entry(serde_json::to_string(&jk).unwrap_or_default()).or_insert_with(|| (jk, Vec::new())).1.push(h);
        }
        per_stage.push(m);
    }
    let Some(first) = per_stage.first() else { return Ok((Vec::new(), failures)) };
    let mut keys: Vec<(&String, &Vec<Value>)> = first.iter().filter(|(k, _)| per_stage.iter().all(|m| m.contains_key(*k))).map(|(k, v)| (k, &v.0)).collect();
    keys.sort_by(|a, b| {
        a.1.iter().zip(b.1.iter()).map(|(x, y)| crate::esql::cmp_vals(&crate::esql::parse::json_to_val(x), &crate::esql::parse::json_to_val(y))).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::new();
    for (k, jk) in keys {
        let stages: Vec<&[Hit]> = per_stage.iter().map(|m| m[k].1.as_slice()).collect();
        for sample in select_samples(&stages, max_per_key) {
            out.push((jk.clone(), sample.into_iter().map(materialized).collect::<Result<Vec<_>>>()?));
        }
    }
    Ok((out, failures))
}


/// Greedy selection in stage order. Used identities only grow, so each stage
/// can advance a cursor instead of rescanning its already-used prefix for each
/// sample. Borrow identities; no ID or source allocations during probing.
fn select_samples<'a>(stages: &[&'a [Hit]], limit: usize) -> Vec<Vec<&'a Hit>> {
    let mut used: HashSet<(&str, &str)> = HashSet::new();
    let mut cursors = vec![0; stages.len()];
    let mut out = Vec::new();
    for _ in 0..limit {
        let mut sample = Vec::with_capacity(stages.len());
        for (events, cursor) in stages.iter().zip(&mut cursors) {
            while *cursor < events.len() && used.contains(&(events[*cursor].index.as_str(), events[*cursor].id.as_str())) {
                *cursor += 1;
            }
            let Some(hit) = events.get(*cursor) else { return out };
            used.insert((hit.index.as_str(), hit.id.as_str()));
            sample.push(hit);
            *cursor += 1;
        }
        out.push(sample);
    }
    out
}


/// Painless helpers of EQL conditions evaluated per document (ES
/// `InternalEqlScriptUtils`): `substring` with EQL's negative offsets,
/// `concat` skipping nothing (null makes null), string tests.
const EQL_PAINLESS_HELPERS: &str = "def eqlSubstring(def s, def a, def b) { if (s == null || a == null) { return null; } String str = s.toString(); int n = str.length(); int st = (int) a; if (st < 0) { st = n + st; if (st < 0) { st = 0; } } else if (st > n) { st = n; } int en = b == null ? n : (int) b; if (en < 0) { en = n + en; if (en < 0) { en = 0; } } else if (en > n) { en = n; } return st >= en ? '' : str.substring(st, en); } def eqlConcat(def parts) { StringBuilder sb = new StringBuilder(); for (def p : parts) { if (p == null) { return null; } sb.append(p.toString()); } return sb.toString(); } boolean eqlTest(def v, def p, int kind) { if (v == null || p == null) { return false; } String s = v.toString(); String q = p.toString(); if (kind == 0) { return s.startsWith(q); } if (kind == 1) { return s.endsWith(q); } return s.contains(q); }";

/// A Painless expression of a literal.
fn painless_lit(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

/// A Painless expression reading a field's first value (or null).
fn painless_field(f: &str) -> String {
    let q = f.replace('\'', "\\'");
    format!("(doc.containsKey('{q}') && doc['{q}'].size() > 0 ? doc['{q}'].value : null)")
}

#[cfg(test)]
mod rally_parser_tests {
    use super::*;

    #[test]
    fn separate_stages_keep_independent_searches() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let temp = tempfile::tempdir().unwrap();
        let node = Node::open(temp.path()).unwrap();
        node.create_index("events", &json!({"mappings":{"properties":{
            "@timestamp":{"type":"date"},"key":{"type":"keyword"},"other":{"type":"keyword"}
        }}})).unwrap();
        for custom in 0..3 {
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = calls.clone();
            let _scope = remote_scope(Some(Arc::new(move |_,_,_| {
                counter.fetch_add(1,Ordering::SeqCst);
                Ok(json!({"hits":{"hits":[
                    {"_index":"events","_id":"1","_source":{"@timestamp":1000,"key":"same","other":"same","payload":{"wide":[1,2,3]}},"fields":{"key":["same"]}},
                    {"_index":"events","_id":"2","_source":{"@timestamp":2000,"key":"same","other":"same","payload":{"wide":[4,5,6]}},"fields":{"key":["same"]}}
                ]}}))
            })));
            let mut body=json!({"query":"sequence [any where true] by key [any where true] by other","size":1,"fields":["key"]});
            if custom==1 { body["filter"]=json!({"match_all":{}}); }
            if custom==2 { body["runtime_mappings"]=json!({"computed":{"type":"keyword","script":"emit('x')"}}); }
            let response=search(&node,"events",&HashMap::new(),&body).unwrap();
            assert_eq!(calls.load(Ordering::SeqCst),2);
            let events=response["hits"]["sequences"][0]["events"].as_array().unwrap();
            assert_eq!(events.len(),2);
            assert_eq!(events[0]["_id"],"1");assert_eq!(events[1]["_id"],"2");
            assert_eq!(events[1]["_source"]["payload"]["wide"],json!([4,5,6]));
            assert_eq!(events[1]["fields"]["key"],json!(["same"]));
        }
    }

    fn scan_samples<'a>(stages: &[&'a [Hit]], limit: usize) -> Vec<Vec<&'a Hit>> {
        let mut used = Vec::new();
        let mut out = Vec::new();
        for _ in 0..limit {
            let mut sample: Vec<&Hit> = Vec::new();
            for events in stages {
                let Some(hit) = events.iter().find(|h| !used.contains(&(h.index.as_str(), h.id.as_str()))
                    && !sample.iter().any(|x| x.index == h.index && x.id == h.id)) else { return out };
                sample.push(hit);
            }
            used.extend(sample.iter().map(|h| (h.index.as_str(), h.id.as_str())));
            out.push(sample);
        }
        out
    }

    fn sample_hit(i: usize, stage: usize) -> Hit {
        Hit { index: format!("index-{}", i % 2), id: (i / 2).to_string(),
            source: json!({"stage":stage,"identity":i}), raw:None, fields:None, ts:0, tb:Value::Null, missing:false }
    }

    #[test]
    fn sample_cursors_match_greedy_scan_with_overlaps_duplicates_and_indexes() {
        let mut state = 17u64;
        for case in 0..1000 {
            let mut owned = Vec::new();
            for stage in 0..(case % 4 + 1) {
                let mut events = Vec::new();
                for _ in 0..(case % 19) {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    events.push(sample_hit((state >> 32) as usize % 16,stage));
                }
                owned.push(events);
            }
            let stages: Vec<_> = owned.iter().map(Vec::as_slice).collect();
            let values = |out: Vec<Vec<&Hit>>| out.into_iter().map(|sample| sample.into_iter()
                .map(|h| h.source.clone()).collect::<Vec<_>>()).collect::<Vec<_>>();
            assert_eq!(values(select_samples(&stages,case % 9)),values(scan_samples(&stages,case % 9)),"case {case}");
        }
    }

    #[test]
    fn sample_cursor_selective_workload() {
        let owned: Vec<_> = (0..3).map(|stage| (0..3000).map(|i|sample_hit(i,stage)).collect::<Vec<_>>()).collect();
        let stages: Vec<_> = owned.iter().map(Vec::as_slice).collect();
        let started = Instant::now();
        let scanned = scan_samples(&stages,200);
        let scan_time = started.elapsed();
        let started = Instant::now();
        let indexed = select_samples(&stages,200);
        let cursor_time = started.elapsed();
        assert_eq!(scanned.len(),indexed.len());
        for (a,b) in scanned.iter().flatten().zip(indexed.iter().flatten()) { assert!(std::ptr::eq(*a,*b)); }
        eprintln!("sample diagnostic: 3 overlapping stages / 3000 events / 200 samples: scan={scan_time:?}, cursors={cursor_time:?}");
    }

    #[test]
    fn missing_event_index_selective_workload() {
        let events: Vec<_> = (0..10_000).map(|i| (i as i64, format!("key-{}", i % 100))).collect();
        let queries: Vec<_> = (0..2_000).map(|i| (format!("key-{}", i % 100), 10_001 + i as i64)).collect();
        let started = Instant::now();
        let scanned: Vec<_> = queries.iter().map(|(key,lo)| std::hint::black_box(&events).iter()
            .any(|(ts,k)| k == key && *ts > *lo && *ts < *lo+10)).collect();
        let scan_time = started.elapsed();
        let started = Instant::now();
        let indexed = MissingEvents::new(events, true);
        let answers: Vec<_> = queries.iter().map(|(key,lo)| indexed.between(key,Some(*lo),Some(*lo+10))).collect();
        let index_time = started.elapsed();
        assert_eq!(answers,scanned);
        eprintln!("missing-event diagnostic: 10000 events / 2000 queries: scan={scan_time:?}, index_build_and_queries={index_time:?}");
    }

    #[test]
    fn missing_event_index_matches_scan_for_open_bounds_and_keys() {
        let events = vec![(i64::MAX, "a".into()), (0, "b".into()), (-1, "a".into()),
            (0, "a".into()), (0, "a".into()), (i64::MIN, "b".into())];
        let bounds = [None, Some(i64::MIN), Some(-1), Some(0), Some(1), Some(i64::MAX)];
        for keyed in [false, true] {
            let index = MissingEvents::new(events.clone(), keyed);
            for key in ["a", "b", "absent"] {
                for lo in bounds {
                    for hi in bounds {
                        let expected = events.iter().any(|(ts, k)| (!keyed || k == key)
                            && lo.is_none_or(|l| *ts > l) && hi.is_none_or(|h| *ts < h));
                        assert_eq!(index.between(key, lo, hi), expected, "{keyed} {key} {lo:?} {hi:?}");
                    }
                }
            }
        }
        assert!(!MissingEvents::new([], false).between("", None, None));
    }

    #[test]
    fn identical_predicates_preserve_independent_tied_event_sets() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let temp = tempfile::tempdir().unwrap();
        let node = Node::open(temp.path()).unwrap();
        node.create_index("events", &json!({"mappings":{"properties":{"@timestamp":{"type":"date"}}}})).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let _scope = remote_scope(Some(Arc::new(move |_,_,_| {
            let ids = if counter.fetch_add(1, Ordering::SeqCst) == 0 { vec!["1", "2"] } else { vec!["3"] };
            Ok(json!({"hits":{"hits":ids.into_iter().map(|id| json!({
                "_index":"events", "_id":id, "_source":{"@timestamp":1000}
            })).collect::<Vec<_>>()}}))
        })));
        // The capped match's separate stage searches from `_source` (a
        // windowed run, and the capped match's doc-value reads, read stage
        // events from doc values, which this mock does not return).
        FULL_STAGES_ONLY.with(|f| f.set(true));
        CAPPED_SOURCE_ONLY.with(|c| c.set(true));
        let response = search(&node, "events", &HashMap::new(), &json!({
            "query":"sequence [any where true] [any where true]", "size":10
        }));
        FULL_STAGES_ONLY.with(|f| f.set(false));
        CAPPED_SOURCE_ONLY.with(|c| c.set(false));
        let response = response.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let sequences = response["hits"]["sequences"].as_array().unwrap();
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0]["events"][0]["_id"], "2");
        assert_eq!(sequences[0]["events"][1]["_id"], "3");
    }

    #[test]
    fn parses_all_rally_eql_workloads() {
        // All 24 upstream Rally EQL query shapes from the full-corpus v295 track.
        let cases: Vec<Value> = serde_json::from_str(include_str!("../tests/fixtures/rally_eql_queries.json")).unwrap();
        assert_eq!(cases.len(), 24);
        let failures: Vec<_> = cases.iter().filter_map(|case| {
            parse(case["query"].as_str().unwrap(), &Opts { case_sensitive: true })
                .err().map(|error| format!("{}: {error:?}", case["name"]))
        }).collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

#[cfg(test)]
mod sequence_window_tests {
    use super::*;
    use std::sync::Mutex;

    /// Deterministic generator for the randomized cases.
    pub(super) struct Lcg(pub(super) u64);
    impl Lcg {
        pub(super) fn next(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n.max(1)
        }
    }

    /// The filters of a stage search body.
    fn filters(body: &Value) -> Vec<Value> {
        body["query"]["bool"]["filter"].as_array().cloned().unwrap_or_default()
    }

    /// A stage search over `docs` (`(id, source)`) as the engine runs it:
    /// category term, `exists` filters (optionally ignored), `terms`
    /// filters, sorted by timestamp then id (both reversed for a `desc`
    /// sort), after the `search_after` timestamp, truncated to `size`.
    /// Records each request size.
    /// An `ids` query (a windowed run's source fetch) returns those docs;
    /// `docvalue_fields` come back as `fields` (the timestamp as epoch
    /// millis text), without `_source` when it is turned off.
    pub(super) fn mock(docs: Arc<Vec<(String, Value)>>, honor_exists: bool, sizes: Arc<Mutex<Vec<usize>>>) -> RemoteSearch {
        Arc::new(move |_, _, body: &Value| {
            if let Some(ids) = body["query"]["ids"]["values"].as_array() {
                let hits: Vec<Value> = docs.iter().filter(|(id, _)| ids.iter().any(|x| x.as_str() == Some(id.as_str())))
                    .map(|(id, s)| json!({"_index": "events", "_id": id, "_source": s})).collect();
                return Ok(json!({"hits": {"hits": hits}}));
            }
            let fs = filters(body);
            let category = fs.iter().find_map(|f| f["term"]["event.category"].as_str().map(str::to_string));
            let exists: Vec<String> = fs.iter().filter_map(|f| f["exists"]["field"].as_str().map(str::to_string)).collect();
            let terms: Vec<(String, Vec<Value>)> = fs.iter().filter_map(|f| f["terms"].as_object().and_then(|o| o.iter().next()).map(|(k, v)| (k.clone(), v.as_array().cloned().unwrap_or_default()))).collect();
            let size = body["size"].as_u64().unwrap() as usize;
            let desc = body["sort"][0]["@timestamp"]["order"] == "desc";
            let after = body["search_after"][0].as_i64();
            sizes.lock().unwrap().push(size);
            let mut hits: Vec<&(String, Value)> = docs.iter().filter(|(_, s)| {
                let ts = s["@timestamp"].as_i64().unwrap_or(0);
                category.as_ref().is_none_or(|c| s["event"]["category"].as_str() == Some(c.as_str()))
                    && exists.iter().all(|f| (!honor_exists && f != "@timestamp") || !first_value(s, f).is_null())
                    && terms.iter().all(|(f, vals)| vals.contains(&first_value(s, f)))
                    && after.is_none_or(|a| if desc { ts < a } else { ts > a })
            }).collect();
            hits.sort_by(|a, b| a.1["@timestamp"].as_i64().cmp(&b.1["@timestamp"].as_i64()).then_with(|| a.0.cmp(&b.0)));
            if desc {
                hits.reverse();
            }
            hits.truncate(size);
            let columns: Vec<String> = body["docvalue_fields"].as_array().into_iter().flatten()
                .filter_map(|f| f.as_str().or_else(|| f["field"].as_str()).map(str::to_string)).collect();
            let with_source = body["_source"] != json!(false);
            Ok(json!({"hits": {"hits": hits.iter().map(|(id, s)| {
                let mut h = json!({"_index": "events", "_id": id});
                if with_source {
                    h["_source"] = s.clone();
                }
                if !columns.is_empty() {
                    let mut fields = serde_json::Map::new();
                    for c in &columns {
                        let vals: Vec<Value> = if c == "@timestamp" {
                            vec![json!(s["@timestamp"].to_string())]
                        } else {
                            crate::highlight::source_values(s, c).into_iter().cloned().collect()
                        };
                        if !vals.is_empty() {
                            fields.insert(c.clone(), Value::Array(vals));
                        }
                    }
                    h["fields"] = Value::Object(fields);
                }
                h
            }).collect::<Vec<_>>()}}))
        })
    }

    pub(super) fn run(node: &Node, body: &Value, full_only: bool) -> Result<Value> {
        FULL_STAGES_ONLY.with(|f| f.set(full_only));
        let out = search(node, "events", &HashMap::new(), body);
        FULL_STAGES_ONLY.with(|f| f.set(false));
        out.map(|mut v| {
            v.as_object_mut().unwrap().remove("took");
            v
        })
    }

    pub(super) fn events_index() -> (tempfile::TempDir, std::sync::Arc<Node>) {
        let temp = tempfile::tempdir().unwrap();
        let node = Node::open(temp.path()).unwrap();
        node.create_index("events", &json!({"mappings":{"properties":{
            "@timestamp":{"type":"date"},"k":{"type":"keyword"},"j":{"type":"keyword"},
            "event":{"properties":{"category":{"type":"keyword"}}}
        }}})).unwrap();
        (temp, node)
    }

    pub(super) const QUERIES: &[&str] = &[
        "sequence [a where true] [b where true]",
        "sequence by k [a where true] [b where true]",
        "sequence by k with maxspan=5s [a where true] [b where true] [c where true]",
        "sequence by k [a where true] [b where true] until [c where true]",
        "sequence [a where true] by k [b where true] by j",
        "sequence by ?k [a where true] [a where true]",
        "sequence by k [a where true] with runs=2 [b where true]",
        "sequence by k [any where true] [any where true] | head 3 | tail 2",
        "sequence [a where true] [b where true] | head 5",
    ];

    pub(super) fn random_docs(rng: &mut Lcg, n: usize) -> Vec<(String, Value)> {
        (0..n).map(|i| {
            let mut s = json!({"@timestamp": rng.next(60) as i64 * 1000, "event": {"category": (["a", "b", "c"][rng.next(3)])}, "payload": {"i": i, "pad": "x".repeat(rng.next(40))}});
            if rng.next(5) != 0 {
                s["k"] = json!((["x", "y", "z"][rng.next(3)]));
            }
            if rng.next(4) != 0 {
                s["j"] = json!((["x", "y"][rng.next(2)]));
            }
            (format!("{i:04}"), s)
        }).collect()
    }

    /// The head-window page decides exactly what matching over the whole
    /// per-stage cap decides: randomized event streams with timestamp ties,
    /// missing join keys, `maxspan`, `until`, per-stage keys, optional
    /// keys, `with runs`, chained pipes, and page boundaries inside ties.
    #[test]
    fn head_page_matches_capped_matching_on_random_streams() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(7);
        let (mut paged_decided, mut fell_back) = (0, 0);
        for case in 0..240 {
            let docs = Arc::new(random_docs(&mut rng, [0, 5, 80, 200, 400][case % 5]));
            let query = QUERIES[case % QUERIES.len()];
            let size = [0, 1, 3, 10, 40, 200][rng.next(6)];
            let body = json!({"query": query, "size": size, "fetch_size": ([2, 50, 1000][rng.next(3)]), "result_position": "head"});
            let sizes = Arc::new(Mutex::new(Vec::new()));
            let paged = {
                let _scope = remote_scope(Some(mock(docs.clone(), true, sizes.clone())));
                run(&node, &body, false).unwrap()
            };
            let capped = {
                let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
                run(&node, &body, true).unwrap()
            };
            assert_eq!(paged, capped, "case {case}: {query} size {size}");
            let sizes = sizes.lock().unwrap();
            if sizes.iter().any(|&s| s == STAGE_CAP) { fell_back += 1 } else { paged_decided += 1 }
        }
        // Head windows are decided by pages (windowed, or the head page
        // when a windowed run gives up) at least sometimes.
        assert!(paged_decided > 0, "paged {paged_decided} / fell back {fell_back}");
        eprintln!("head page: decided by pages {paged_decided}, fell back to the cap {fell_back}");
    }

    /// The capped match reading timestamps and join keys from doc values
    /// (the returned events' sources fetched at the end) returns exactly
    /// what it returns reading `_source`: randomized streams, head and tail
    /// windows, missing keys, `until`, `![...]` stages, optional keys.
    #[test]
    fn capped_doc_values_match_capped_source_reads() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(29);
        let mut queries: Vec<&str> = QUERIES.to_vec();
        queries.push("sequence by k [a where true] ![c where true] [b where true]");
        for case in 0..200 {
            let docs = Arc::new(random_docs(&mut rng, [0, 5, 80, 200, 400][case % 5]));
            let query = queries[case % queries.len()];
            let size = [1, 3, 10, 40, 200][rng.next(5)];
            let position = ["head", "tail"][rng.next(2)];
            let body = json!({"query": query, "size": size, "fetch_size": ([2, 50, 1000][rng.next(3)]), "result_position": position});
            let read = |source_only: bool| {
                CAPPED_SOURCE_ONLY.with(|c| c.set(source_only));
                let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
                let out = run(&node, &body, true).unwrap();
                CAPPED_SOURCE_ONLY.with(|c| c.set(false));
                out
            };
            assert_eq!(read(false), read(true), "case {case}: {query} size {size} {position}");
        }
    }

    /// `![...]` stages and returned `fields` keep matching over the cap in
    /// one pass; tail windows page (windowed) like head windows.
    #[test]
    fn head_page_only_for_plain_head_windows() {
        let (_temp, node) = events_index();
        let docs = Arc::new(random_docs(&mut Lcg(3), 300));
        for (body, page) in [
            (json!({"query": "sequence [a where true] [b where true]", "size": 3}), 1000),
            (json!({"query": "sequence [a where true] [b where true] | tail 3", "result_position": "head"}), 1000),
            (json!({"query": "sequence [a where true] ![c where true] [b where true]", "result_position": "head"}), STAGE_CAP),
            (json!({"query": "sequence [a where true] [b where true]", "result_position": "head", "fields": ["k"]}), STAGE_CAP),
        ] {
            let sizes = Arc::new(Mutex::new(Vec::new()));
            let _scope = remote_scope(Some(mock(docs.clone(), true, sizes.clone())));
            run(&node, &body, false).unwrap();
            // Windowed tail runs read the first stage a quarter page ahead.
            assert!(sizes.lock().unwrap().iter().all(|&s| s == page || (page == 1000 && s == 1251)), "{body}: {:?}", sizes.lock().unwrap());
        }
        // A plain head window over dense stages is decided by the page.
        let sizes = Arc::new(Mutex::new(Vec::new()));
        let _scope = remote_scope(Some(mock(docs.clone(), true, sizes.clone())));
        let out = run(&node, &json!({"query": "sequence [a where true] [b where true]", "size": 3, "result_position": "head"}), false).unwrap();
        assert_eq!(out["hits"]["sequences"].as_array().unwrap().len(), 3);
        // Windowed head searches start at 64 events and double.
        let sizes = sizes.lock().unwrap();
        assert!(sizes.len() <= 4 && sizes[..2] == [64usize, 64], "{sizes:?}");
    }

    /// The mandatory join-key `exists` filters (ES
    /// `AddMandatoryJoinKeyFilter`) drop only events the matcher drops:
    /// below the cap, results are the same with and without them.
    #[test]
    fn mandatory_key_filters_do_not_change_uncapped_results() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(11);
        for case in 0..120 {
            let docs = Arc::new(random_docs(&mut rng, [0, 30, 300][case % 3]));
            let query = QUERIES[case % QUERIES.len()];
            for position in ["head", "tail"] {
                let body = json!({"query": query, "size": ([1, 5, 50][rng.next(3)]), "result_position": position});
                let honored = {
                    let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
                    run(&node, &body, true).unwrap()
                };
                let ignored = {
                    let _scope = remote_scope(Some(mock(docs.clone(), false, Arc::new(Mutex::new(Vec::new())))));
                    run(&node, &body, true).unwrap()
                };
                assert_eq!(honored, ignored, "case {case}: {query} {position}");
            }
        }
        // Samples too.
        let docs = Arc::new(random_docs(&mut rng, 200));
        let body = json!({"query": "sample by k [a where true] [b where true]", "size": 50, "max_samples_per_key": 3});
        let with = { let _s = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new()))))); run(&node, &body, false).unwrap() };
        let without = { let _s = remote_scope(Some(mock(docs.clone(), false, Arc::new(Mutex::new(Vec::new()))))); run(&node, &body, false).unwrap() };
        assert_eq!(with, without);
    }

    /// Which `exists` filters each stage search carries.
    #[test]
    fn mandatory_key_filters_per_stage() {
        let (_temp, node) = events_index();
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let seen = bodies.clone();
        let _scope = remote_scope(Some(Arc::new(move |_, _, body: &Value| {
            seen.lock().unwrap().push(body.clone());
            Ok(json!({"hits": {"hits": []}}))
        })));
        let exists = |body: &Value| -> Vec<String> {
            filters(body).iter().filter_map(|f| f["exists"]["field"].as_str().map(str::to_string)).collect()
        };
        let take = || std::mem::take(&mut *bodies.lock().unwrap());
        search(&node, "events", &HashMap::new(), &json!({
            "query": "sequence by k, ?o [a where true] by j [b where true] by rt until [c where true]",
            "runtime_mappings": {"rt": {"type": "keyword"}}
        })).unwrap();
        let got: Vec<Vec<String>> = take().iter().map(exists).collect();
        assert_eq!(got, vec![vec!["@timestamp", "k", "j"], vec!["@timestamp", "k"], vec!["@timestamp", "k"]]);
        search(&node, "events", &HashMap::new(), &json!({"query": "sequence [a where true] ![c where true] by j [b where true]"})).unwrap();
        let got: Vec<Vec<String>> = take().iter().map(exists).collect();
        assert_eq!(got, vec![vec!["@timestamp"], vec!["@timestamp"], vec!["@timestamp", "j"]]);
        search(&node, "events", &HashMap::new(), &json!({"query": "sample by k [a where true] [b where true] by j"})).unwrap();
        let got: Vec<Vec<String>> = take().iter().map(exists).collect();
        assert_eq!(got, vec![vec!["@timestamp", "k"], vec!["@timestamp", "k", "j"]]);
        search(&node, "events", &HashMap::new(), &json!({"query": "a where true"})).unwrap();
        let got: Vec<Vec<String>> = take().iter().map(exists).collect();
        assert_eq!(got, vec![vec!["@timestamp"]]);
    }

    fn random_json(rng: &mut Lcg, depth: usize) -> Value {
        const SEGS: &[&str] = &["a", "b", "c", "a.b", "b.c", "a.b.c", "ab", ""];
        match if depth == 0 { rng.next(5) } else { rng.next(8) } {
            0 => Value::Null,
            1 => json!(rng.next(1000) as i64 - 500),
            2 => json!(rng.next(1000) as f64 / 7.0),
            3 => json!((["s", "t\"q\\\u{e9}\u{1F600}", "", "\n"][rng.next(4)])),
            4 => json!(rng.next(2) == 0),
            5 | 6 => {
                let mut m = serde_json::Map::new();
                for _ in 0..rng.next(5) {
                    m.insert(SEGS[rng.next(SEGS.len())].to_string(), random_json(rng, depth - 1));
                }
                Value::Object(m)
            }
            _ => Value::Array((0..rng.next(4)).map(|_| random_json(rng, depth - 1)).collect()),
        }
    }

    /// A projected `_source` yields the same values for every wanted path as
    /// the whole source, and the whole source's breaker size.
    #[test]
    fn projection_reads_what_the_full_source_reads() {
        const PATHS: &[&str] = &["a", "b", "a.b", "a.b.c", "b.c", "c.a.b", "ab", "a.c"];
        let mut rng = Lcg(5);
        for case in 0..5000 {
            let text = random_json(&mut rng, 4).to_string();
            let full: Value = serde_json::from_str(&text).unwrap();
            let wanted: Vec<&str> = (0..rng.next(4) + 1).map(|_| PATHS[rng.next(PATHS.len())]).collect();
            let (projected, bytes) = lazy_source::project(&text, &lazy_source::paths(wanted.iter().copied())).unwrap();
            for p in PATHS.iter().filter(|p| wanted.contains(p)) {
                let a: Vec<&Value> = crate::highlight::source_values(&full, p);
                let b: Vec<&Value> = crate::highlight::source_values(&projected, p);
                assert_eq!(a, b, "case {case}: {p} of {text} -> {projected}");
                assert_eq!(first_value(&full, p), first_value(&projected, p));
            }
            // What the stage event was charged when its source was parsed.
            assert_eq!(bytes, crate::memory::json_bytes(&full), "case {case}: {text}");
        }
    }

    /// Stage hits read from response text keep identity, order keys and
    /// fields, are charged as before, and materialize to the full source.
    #[test]
    fn projected_hits_match_parsed_hits() {
        let mut rng = Lcg(9);
        let hits: Vec<Value> = (0..200).map(|i| {
            let mut src = json!({"@timestamp": format!("2020-01-01T00:00:{:02}.{:03}Z", i % 60, i), "process": {"pid": i, "name": "x".repeat(i % 9)}, "noise": random_json(&mut rng, 3)});
            match i % 4 {
                0 => src["source.ip"] = json!("10.0.0.1"),
                1 => src["source"] = json!({"ip": ["10.0.0.2", "10.0.0.3"]}),
                2 => src["tb"] = json!(i),
                _ => {}
            }
            let mut h = json!({"_index": "events", "_id": i.to_string(), "_score": null, "_source": src, "sort": [i]});
            if i % 3 == 0 {
                h["fields"] = json!({"process.pid": [i]});
            }
            if i % 50 == 7 {
                h.as_object_mut().unwrap().remove("_source");
            }
            h
        }).collect();
        let text = json!({"took": 1, "hits": {"hits": hits}}).to_string();
        let r = Request::from_json(&json!({"query": "any where true", "tiebreaker_field": "tb"})).unwrap();
        let paths = lazy_source::paths(["@timestamp", "tb", "source.ip", "process.pid"]);
        let full = hits_of_text(&text, &r, None).unwrap();
        let lazy = hits_of_text(&text, &r, Some(&paths)).unwrap();
        assert_eq!(full.len(), lazy.len());
        for (a, b) in full.iter().zip(&lazy) {
            assert_eq!((&a.index, &a.id, a.ts, &a.tb, &a.fields), (&b.index, &b.id, b.ts, &b.tb, &b.fields));
            assert_eq!(join_values(&a.source, &["source.ip".into(), "?process.pid".into()]), join_values(&b.source, &["source.ip".into(), "?process.pid".into()]));
            assert_eq!(SeqMemory::hit(a), SeqMemory::hit(b));
            assert_eq!(materialized(b).unwrap().source, a.source);
            assert_eq!(hit_json(materialized(b).unwrap()), hit_json(hit_copy(a)));
        }
    }

    pub(super) fn put(node: &Node, docs: &[(String, Value)]) {
        use crate::index::WriteReq;
        use crate::shard::WriteCond;
        let svc = node.resolve("events", ResolveOpts::default()).unwrap().remove(0).svc;
        for chunk in docs.chunks(5000) {
            let w = chunk.iter().map(|(id, s)| WriteReq::Index { id: Some(id.clone()), source: s.to_string(), routing: None, cond: WriteCond::default() }).collect();
            for r in svc.write(w, true) {
                r.unwrap();
            }
        }
        svc.refresh().unwrap();
    }

    /// Through the real search path (text responses, projection, `exists`
    /// filters): paged and capped matching agree, and both return the full
    /// stored `_source` of each returned event.
    #[test]
    fn local_index_head_page_and_projection() {
        let (_temp, node) = events_index();
        let docs = random_docs(&mut Lcg(21), 600);
        put(&node, &docs);
        let by_id: HashMap<&str, &Value> = docs.iter().map(|(id, s)| (id.as_str(), s)).collect();
        for query in QUERIES {
            for size in [1, 4, 30] {
                for position in ["head", "tail"] {
                    let body = json!({"query": query, "size": size, "result_position": position});
                    let paged = run(&node, &body, false).unwrap();
                    let capped = run(&node, &body, true).unwrap();
                    assert_eq!(paged, capped, "{query} {size} {position}");
                    for s in paged["hits"]["sequences"].as_array().unwrap() {
                        for e in s["events"].as_array().unwrap() {
                            assert_eq!(&e["_source"], by_id[e["_id"].as_str().unwrap()], "{query}");
                        }
                    }
                }
            }
        }
    }

    /// Diagnostic, not an assertion of speed: the Rally `eql` shapes over a
    /// 20,000-event local index of endgame-sized (~2 KB) sources. Run with
    /// `cargo test -p elasticsearch-engine --release eql_sequence_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn eql_sequence_bench() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(1);
        let docs: Vec<(String, Value)> = (0..20_000).map(|i| {
            let mut s = json!({"@timestamp": 1_600_000_000_000i64 + i as i64 * 37, "event": {"category": (["process", "network", "file"][i % 3]), "kind": "event", "action": "start"},
                "process": {"pid": 900 + rng.next(30), "ppid": rng.next(5000), "name": format!("proc-{}", rng.next(200)), "args": (0..8).map(|a| format!("--arg{a}=value-{}", rng.next(1000))).collect::<Vec<_>>(),
                    "executable": "/usr/bin/some/long/path/to/binary", "hash": {"md5": "d41d8cd98f00b204e9800998ecf8427e", "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}},
                "host": {"name": format!("host-{}", rng.next(20)), "os": {"name": "Windows", "version": "10.0", "family": "windows", "kernel": "10.0.17763.1158"}, "ip": ["10.1.2.3", "fe80::1"]},
                "agent": {"id": "a8a1b7d0-1234-5678-9abc-def012345678", "type": "endgame", "version": "3.53.0"}, "user": {"name": (["SYSTEM", "admin", "bob"][rng.next(3)]), "domain": "NT AUTHORITY"},
                "message": "x".repeat(600)});
            if i % 3 == 1 {
                s["source"] = json!({"ip": format!("10.0.0.{}", rng.next(20))});
                s["destination"] = json!({"ip": format!("10.0.1.{}", rng.next(20))});
            }
            (format!("{i:06}"), s)
        }).collect();
        let started = Instant::now();
        put(&node, &docs);
        eprintln!("indexed {} docs in {:?}", docs.len(), started.elapsed());
        for (name, body) in [
            ("sequence_2stage_nofilter_fetch1000_size_1000", json!({"query": "sequence by source.ip, destination.ip [process where true] [process where true]", "fetch_size": 1000, "size": 1000})),
            ("sequence_2stages_baseline", json!({"query": "sequence [any where true] [any where true]"})),
            ("sequence_2stages_headThenTail", json!({"query": "sequence [any where true] [any where true] | head 200 | tail 100"})),
            ("sequence_4stage_head100_tail50", json!({"query": "sequence by source.ip, destination.ip with maxspan=5m [process where true] [process where true] [process where true] [network where true] |head 100 | tail 50", "fetch_size": 1000, "size": 100})),
            ("sequence_2stages_fewThenFew_highCardJoinKey", json!({"query": "sequence by process.ppid [process where process.pid == 910] [process where process.pid == 911]"})),
        ] {
            for full_only in [true, false] {
                let _ = run(&node, &body, full_only).unwrap();
                let started = Instant::now();
                let runs = 5;
                let mut out = Value::Null;
                for _ in 0..runs {
                    out = run(&node, &body, full_only).unwrap();
                }
                eprintln!("{name:50} {:>7} {:>10.1?}/op  sequences={}", if full_only { "capped" } else { "paged" }, started.elapsed() / runs, out["hits"]["total"]["value"]);
            }
        }
        // Stage-event parsing alone: whole sources vs the projection.
        let text = json!({"hits": {"hits": docs.iter().take(10_000).map(|(id, s)| json!({"_index": "events", "_id": id, "_source": s})).collect::<Vec<_>>()}}).to_string();
        let r = Request::from_json(&json!({"query": "any where true"})).unwrap();
        let paths = lazy_source::paths(["@timestamp", "source.ip", "destination.ip"]);
        for (label, projection) in [("full parse", None), ("projection", Some(paths.as_slice()))] {
            let started = Instant::now();
            let hits = hits_of_text(&text, &r, projection).unwrap();
            eprintln!("10,000 stage hits ({} MB): {label:12} {:?}", text.len() / 1_000_000, started.elapsed());
            assert_eq!(hits.len(), 10_000);
        }
    }
}

/// Windowed sequence runs ([`windowed_sequences`]) against a straightforward
/// reference over the whole event set: the state machine fed every event in
/// order (head), and for tail windows the same machine run from scratch from
/// each page of the newest first-stage events (as ES's tail windows do).
#[cfg(test)]
mod windowed_sequence_tests {
    use super::sequence_window_tests::{events_index, mock, put, random_docs, run, Lcg, QUERIES};
    use super::*;
    use std::sync::Mutex;

    /// One matcher input: (timestamp, id, stage or `None` for `until`, key).
    type Entry<'d> = (i64, &'d str, Option<usize>, Vec<Value>);

    fn ts_of(src: &Value) -> i64 {
        src["@timestamp"].as_i64().unwrap()
    }

    fn admits(q: &EventQuery, src: &Value) -> bool {
        // The test queries are `<category> where true`.
        assert_eq!(q.cond, json!({"match_all": {}}));
        q.category.as_ref().is_none_or(|c| src["event"]["category"].as_str() == Some(c.as_str()))
    }

    /// Matching order: timestamp, id, later stages first, `until` last.
    fn entry_cmp(a: &Entry<'_>, b: &Entry<'_>) -> std::cmp::Ordering {
        (a.0, a.1).cmp(&(b.0, b.1)).then_with(|| b.2.map_or(-1, |s| s as i64).cmp(&a.2.map_or(-1, |s| s as i64)))
    }

    fn sort_entries(es: &mut [Entry<'_>]) {
        es.sort_by(entry_cmp);
    }

    /// Two sorted entry lists in matching order.
    fn merged<'e, 'd>(a: &'e [Entry<'d>], b: &'e [Entry<'d>]) -> impl Iterator<Item = &'e Entry<'d>> {
        let (mut i, mut j) = (0, 0);
        std::iter::from_fn(move || {
            let next = match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if entry_cmp(x, y) != std::cmp::Ordering::Greater => x,
                (Some(_), Some(y)) | (None, Some(y)) => y,
                (Some(x), None) => x,
                (None, None) => return None,
            };
            if a.get(i).is_some_and(|x| std::ptr::eq(x, next)) {
                i += 1;
            } else {
                j += 1;
            }
            Some(next)
        })
    }

    /// The sequence state machine from an empty state: a stage event takes
    /// its key's latest partial sequence of the previous stage (a first
    /// stage event replaces the key's start), `maxspan` bounds the span
    /// from the first event, `until` drops the key's partial sequences.
    /// Completed sequences: (key, event ids), in completion order.
    ///
    /// With `firsts`, the number of first-stage entries: matching stops once
    /// they are spent and nothing is in flight (no later entry can complete
    /// a sequence then).
    fn machine<'e, 'd: 'e>(entries: impl IntoIterator<Item = &'e Entry<'d>>, n: usize, maxspan: Option<i64>, mut firsts: Option<usize>) -> Vec<(Vec<Value>, Vec<String>, i64)> {
        let mut slots: HashMap<String, Vec<Option<(i64, Vec<String>)>>> = HashMap::new();
        let mut live = 0usize;
        let mut out = Vec::new();
        for (ts, id, stage, key) in entries {
            if firsts == Some(0) && live == 0 {
                break;
            }
            let k = serde_json::to_string(key).unwrap();
            let Some(si) = *stage else {
                if let Some(e) = slots.remove(&k) {
                    live -= e.iter().flatten().count();
                }
                continue;
            };
            let e = slots.entry(k).or_insert_with(|| vec![None; n]);
            if si == 0 {
                if let Some(f) = firsts.as_mut() {
                    *f -= 1;
                }
                if e[0].replace((*ts, vec![id.to_string()])).is_none() {
                    live += 1;
                }
                continue;
            }
            let Some((start, mut ids)) = e[si - 1].take() else { continue };
            live -= 1;
            if maxspan.is_some_and(|ms| ts - start > ms) {
                continue;
            }
            ids.push(id.to_string());
            if si + 1 == n {
                out.push((key.clone(), ids, *ts));
            } else if e[si].replace((start, ids)).is_none() {
                live += 1;
            }
        }
        out
    }

    /// The reference result of `body` over `docs`: per sequence its join
    /// keys (when there are any) and event ids.
    fn reference(docs: &[(String, Value)], body: &Value, stops: &[(usize, MatchPos)]) -> Vec<Value> {
        let r = Request::from_json(body).unwrap();
        let EqlQuery::Sequence { by, stages, until, .. } = parse(&r.query, &Opts { case_sensitive: false }).unwrap() else { panic!("a sequence") };
        let mut expanded: Vec<(&Stage, Vec<String>)> = Vec::new();
        for s in &stages {
            assert!(!s.missing);
            for _ in 0..s.runs {
                let mut keys = by.clone();
                keys.extend(s.by.iter().cloned());
                expanded.push((s, keys));
            }
        }
        let n = expanded.len();
        let until_keys: Vec<String> = by.iter().chain(until.iter().flat_map(|u| u.by.iter())).cloned().collect();
        // Every stage's events (with their mandatory keys), stage 0 apart.
        let mut first: Vec<Entry<'_>> = Vec::new();
        let mut later: Vec<Entry<'_>> = Vec::new();
        for (id, src) in docs {
            for (si, (s, keys)) in expanded.iter().enumerate() {
                if admits(&s.q, src) {
                    if let Some(key) = join_values(src, keys) {
                        if si == 0 {
                            first.push((ts_of(src), id.as_str(), Some(0), key));
                        } else {
                            later.push((ts_of(src), id.as_str(), Some(si), key));
                        }
                    }
                }
            }
            if let Some(u) = &until {
                if admits(&u.q, src) {
                    if let Some(key) = join_values(src, &until_keys) {
                        later.push((ts_of(src), id.as_str(), None, key));
                    }
                }
            }
        }
        decide(body, n, first, later, stops)
    }

    /// The tail pages a windowed run stopped early (see [`Run::pass`]).
    fn stops() -> Vec<(usize, MatchPos)> {
        WINDOW_STOP.with(|w| w.borrow().clone())
    }

    /// Whether entry `e` comes at or before matching position `p`.
    fn matched_by(e: &Entry<'_>, p: &MatchPos) -> bool {
        match (e.0, e.1).cmp(&(p.0, p.2.as_str())) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => e.2.map_or(-1, |s| s as i64) >= p.3,
            std::cmp::Ordering::Greater => false,
        }
    }

    /// The reference decision over the matcher inputs of `body`'s stages
    /// (`first`: the first stage's, `later`: the other stages' and
    /// `until`'s): the whole-stream machine (head), or ES-style pages of
    /// the newest first-stage events (tail).
    ///
    /// `stops`: tail pages (1-based) the windowed run stopped early, and
    /// where: those pages match only the entries up to there.
    fn decide(body: &Value, n: usize, mut first: Vec<Entry<'_>>, later: Vec<Entry<'_>>, stops: &[(usize, MatchPos)]) -> Vec<Value> {
        let r = Request::from_json(body).unwrap();
        let EqlQuery::Sequence { maxspan, pipes, .. } = parse(&r.query, &Opts { case_sensitive: false }).unwrap() else { panic!("a sequence") };
        let (size, tail) = initial_limit(r.size, &pipes, r.tail);
        let mut done: Vec<(Vec<Value>, Vec<String>, i64)> = Vec::new();
        if !tail {
            let mut all: Vec<Entry<'_>> = first.iter().chain(&later).cloned().collect();
            sort_entries(&mut all);
            done = machine(&all, n, maxspan, None);
        } else {
            // Pages of the newest first-stage events (`fetch_size`, never
            // splitting a timestamp), each matched forward from its first
            // timestamp with every later-stage event from there on.
            sort_entries(&mut first);
            let mut later = later;
            sort_entries(&mut later);
            let mut rest: Vec<Entry<'_>> = first.iter().rev().cloned().collect();
            let mut seen: HashSet<(String, String)> = HashSet::new();
            let mut pages = 0usize;
            while !rest.is_empty() && done.len() < size {
                pages += 1;
                let mut page = r.fetch_size;
                let take = loop {
                    if rest.len() < page {
                        break rest.len();
                    }
                    let oldest = rest[page - 1].0;
                    let keep = rest[..page].iter().take_while(|e| e.0 > oldest).count();
                    if keep > 0 {
                        break keep;
                    }
                    page *= 2;
                };
                let mut chunk: Vec<Entry<'_>> = rest.drain(..take).collect();
                sort_entries(&mut chunk);
                let start = chunk[0].0;
                let from = later.partition_point(|e| e.0 < start);
                let stop = stops.iter().find(|(p, _)| *p == pages).map(|(_, pos)| pos);
                let entries = merged(&chunk, &later[from..]).take_while(|e| stop.is_none_or(|p| matched_by(e, p)));
                for (key, ids, ts) in machine(entries, n, maxspan, Some(chunk.len())) {
                    if seen.insert((serde_json::to_string(&key).unwrap(), ids.last().unwrap().clone())) {
                        done.push((key, ids, ts));
                    }
                }
            }
            done.sort_by(|a, b| (a.2, a.1.last()).cmp(&(b.2, b.1.last())));
        }
        let window = pipe_window(done.len(), size, tail, &pipes);
        done[window].iter().map(|(key, ids, _)| {
            let mut v = json!({"ids": ids});
            if !key.is_empty() {
                v["join_keys"] = Value::Array(key.clone());
            }
            v
        }).collect()
    }

    /// A response's sequences in the reference's shape.
    fn shape(out: &Value) -> Vec<Value> {
        out["hits"]["sequences"].as_array().unwrap().iter().map(|s| {
            let ids: Vec<Value> = s["events"].as_array().unwrap().iter().map(|e| e["_id"].clone()).collect();
            let mut v = json!({"ids": ids});
            if let Some(k) = s.get("join_keys") {
                v["join_keys"] = k.clone();
            }
            v
        }).collect()
    }

    /// Whether `body` runs windowed: a tail window over a `with runs` first
    /// stage keeps the capped match.
    fn windowable(body: &Value) -> bool {
        let r = Request::from_json(body).unwrap();
        let EqlQuery::Sequence { stages, pipes, .. } = parse(&r.query, &Opts { case_sensitive: false }).unwrap() else { panic!("a sequence") };
        let (_, tail) = initial_limit(r.size, &pipes, r.tail);
        !(tail && stages[0].runs > 1)
    }

    /// The expected response shape: the reference when windowed, else the
    /// capped match's.
    fn expected(node: &Node, docs: &[(String, Value)], body: &Value, stops: &[(usize, MatchPos)]) -> Vec<Value> {
        if windowable(body) { reference(docs, body, stops) } else { shape(&run(node, body, true).unwrap()) }
    }

    /// Run `body` windowed with the budget lifted; whether the windowed run
    /// decided it, and its (searches, fetched events).
    fn windowed(node: &Node, body: &Value) -> (Value, bool, (usize, usize), Vec<(usize, MatchPos)>) {
        UNBOUNDED_WINDOWS.with(|u| u.set(true));
        let out = run(node, body, false);
        UNBOUNDED_WINDOWS.with(|u| u.set(false));
        let (searches, hits, done) = WINDOW_STATS.with(|s| s.get());
        (out.unwrap(), done == Some(true), (searches, hits), stops())
    }

    /// Randomized event streams (timestamp ties, missing and optional join
    /// keys, per-stage keys, `maxspan`, `until`, `with runs`, chained
    /// pipes), head and tail windows, `fetch_size` pages down to 2 (pages
    /// split inside ties): the windowed run returns the reference result,
    /// and the key filters it sends are exercised.
    #[test]
    fn windowed_runs_match_the_reference() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(29);
        let (mut keyed, mut multi_page) = (0, 0);
        for case in 0..400 {
            let docs = Arc::new(random_docs(&mut rng, [0, 5, 40, 120, 250][case % 5]));
            let query = QUERIES[case % QUERIES.len()];
            let body = json!({
                "query": query,
                "size": ([1, 3, 10, 40][rng.next(4)]),
                "fetch_size": ([2, 3, 7, 50, 1000][rng.next(5)]),
                "result_position": (["head", "tail"][rng.next(2)]),
            });
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let seen = bodies.clone();
            let inner = mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())));
            let _scope = remote_scope(Some(Arc::new(move |i: &str, r: Option<&str>, b: &Value| {
                seen.lock().unwrap().push(b.clone());
                inner(i, r, b)
            })));
            let (out, decided, (searches, _), stopped) = windowed(&node, &body);
            assert_eq!(decided, windowable(&body), "case {case}: {body}");
            let searched = bodies.lock().unwrap().len();
            assert_eq!(shape(&out), expected(&node, &docs, &body, &stopped), "case {case}: {body}");
            let bodies = bodies.lock().unwrap();
            if decided {
                // Plus the returned events' source fetch.
                assert_eq!(searched, searches + usize::from(!out["hits"]["sequences"].as_array().unwrap().is_empty()));
            }
            if bodies.iter().any(|b| b["query"]["bool"]["filter"].as_array().is_some_and(|fs| fs.iter().any(|f| f.get("terms").is_some()))) {
                keyed += 1;
            }
            if searches > 3 {
                multi_page += 1;
            }
        }
        assert!(keyed > 20 && multi_page > 20, "keyed {keyed} / multi-page {multi_page}");
    }

    /// c28z regression (case 158 of [`windowed_runs_match_the_reference`]):
    /// a tail window over more first-stage events than one read holds
    /// (`fetch_size` 3, optional key, tied timestamps). The end of each
    /// tail page moved the next first-stage read back to that page's oldest
    /// timestamp, so events already read were read again and later pages
    /// held duplicates; the reads now continue where they stopped.
    #[test]
    fn tail_pages_cut_from_larger_reads_hold_each_event_once() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(29);
        for case in 0..=158 {
            // The same draws as `windowed_runs_match_the_reference`.
            let docs = Arc::new(random_docs(&mut rng, [0, 5, 40, 120, 250][case % 5]));
            let query = QUERIES[case % QUERIES.len()];
            let body = json!({
                "query": query,
                "size": ([1, 3, 10, 40][rng.next(4)]),
                "fetch_size": ([2, 3, 7, 50, 1000][rng.next(5)]),
                "result_position": (["head", "tail"][rng.next(2)]),
            });
            if case < 158 {
                continue;
            }
            assert_eq!(body, json!({"query": "sequence by ?k [a where true] [a where true]", "size": 3, "fetch_size": 3, "result_position": "tail"}));
            let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
            let (out, decided, _, stopped) = windowed(&node, &body);
            assert!(decided);
            assert_eq!(shape(&out), reference(&docs, &body, &stopped));
            // Each sequence runs forward in time.
            let ts: HashMap<&str, i64> = docs.iter().map(|(id, s)| (id.as_str(), s["@timestamp"].as_i64().unwrap())).collect();
            for seq in out["hits"]["sequences"].as_array().unwrap() {
                let at: Vec<(i64, &str)> = seq["events"].as_array().unwrap().iter().map(|e| { let id = e["_id"].as_str().unwrap(); (ts[id], id) }).collect();
                assert!(at.windows(2).all(|w| w[0] < w[1]), "{seq}");
            }
        }
    }

    /// The reference agrees with the capped match wherever that sees every
    /// event: head windows, and tail windows whose first stage fits one
    /// page.
    #[test]
    fn reference_matches_the_capped_match() {
        let (_temp, node) = events_index();
        let mut rng = Lcg(31);
        for case in 0..200 {
            let docs = Arc::new(random_docs(&mut rng, [0, 30, 120, 300][case % 4]));
            let query = QUERIES[case % QUERIES.len()];
            let body = json!({"query": query, "size": ([1, 4, 30][rng.next(3)]), "result_position": (["head", "tail"][rng.next(2)])});
            let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
            let capped = run(&node, &body, true).unwrap();
            assert_eq!(shape(&capped), reference(&docs, &body, &[]), "case {case}: {body}");
            // With the budget on, a windowed run (or its give-up) agrees too.
            assert_eq!(run(&node, &body, false).unwrap(), capped, "case {case}: {body}");
        }
    }

    /// Over budget, a windowed run gives up for the capped match, which
    /// then answers as before.
    #[test]
    fn over_budget_falls_back_to_the_capped_match() {
        let (_temp, node) = events_index();
        let docs = Arc::new(random_docs(&mut Lcg(37), 2000));
        let body = json!({"query": "sequence by k [a where true] [b where true]", "size": 1000, "fetch_size": 2, "result_position": "head"});
        let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
        let out = run(&node, &body, false).unwrap();
        assert_eq!(WINDOW_STATS.with(|s| s.get()).2, None, "the windowed run gave up");
        assert_eq!(out, run(&node, &body, true).unwrap());
    }

    /// Through the real search path (`search_after` on the timestamp sort,
    /// `terms` key filters, engine tie order, projection): windowed runs
    /// return the reference result and the full stored `_source`.
    #[test]
    fn local_index_windowed_runs_match_the_reference() {
        let (_temp, node) = events_index();
        let docs = random_docs(&mut Lcg(41), 500);
        put(&node, &docs);
        let by_id: HashMap<&str, &Value> = docs.iter().map(|(id, s)| (id.as_str(), s)).collect();
        for (qi, query) in QUERIES.iter().enumerate() {
            for (fi, fetch_size) in [2usize, 9, 1000].into_iter().enumerate() {
                for position in ["head", "tail"] {
                    let body = json!({"query": query, "size": ([2, 25][(qi + fi) % 2]), "fetch_size": fetch_size, "result_position": position});
                    let (out, decided, _, stopped) = windowed(&node, &body);
                    assert_eq!(decided, windowable(&body), "{body}");
                    assert_eq!(shape(&out), expected(&node, &docs, &body, &stopped), "{body}");
                    for s in out["hits"]["sequences"].as_array().unwrap() {
                        for e in s["events"].as_array().unwrap() {
                            assert_eq!(&e["_source"], by_id[e["_id"].as_str().unwrap()], "{body}");
                        }
                    }
                }
            }
        }
    }

    /// The Rally shapes stay narrow: a dense two-stage tail sequence
    /// (`sequence_2stages_baseline`) reads about two pages of 20,000
    /// events, and a keyed head window stops at its size.
    #[test]
    fn rally_shapes_read_few_pages() {
        let (_temp, node) = events_index();
        let docs: Arc<Vec<(String, Value)>> = Arc::new((0..20_000).map(|i| {
            (format!("{i:06}"), json!({"@timestamp": 1_000_000 + i as i64 * 10, "event": {"category": (["a", "b"][i % 2])}, "k": (["x", "y", "z"][i % 3]), "pid": (900 + i % 40)}))
        }).collect());
        let _scope = remote_scope(Some(mock(docs.clone(), true, Arc::new(Mutex::new(Vec::new())))));
        // `sequence_2stages_baseline`: every consecutive pair; the newest.
        let body = json!({"query": "sequence [any where true] [any where true]"});
        let out = run(&node, &body, false).unwrap();
        let (searches, hits, done) = WINDOW_STATS.with(|s| s.get());
        assert_eq!(done, Some(true));
        // The newest first-stage events are read a quarter page ahead.
        assert!(searches <= 4 && hits <= 2_400, "{searches} searches / {hits} events");
        assert_eq!(shape(&out), reference(&docs, &body, &stops()));
        assert_eq!(out["hits"]["sequences"][9]["events"][1]["_id"], "019999");
        // A head window stops once it has its sequences.
        let body = json!({"query": "sequence by k [a where true] [b where true] | head 5", "result_position": "head"});
        let out = run(&node, &body, false).unwrap();
        let (searches, hits, done) = WINDOW_STATS.with(|s| s.get());
        assert_eq!(done, Some(true));
        assert!(searches <= 4 && hits <= 2_100, "{searches} searches / {hits} events");
        assert_eq!(shape(&out), reference(&docs, &body, &stops()));
    }

    /// One synthetic endgame-like event of [`eql_windowed_bench`].
    struct Ev {
        ts: i64,
        /// 0 process, 1 network, 2 file.
        cat: u8,
        pid: i64,
        ppid: i64,
        name: &'static str,
        host: u32,
        /// (source, destination) address numbers.
        ip: Option<(u32, u32)>,
        system: bool,
    }

    impl Ev {
        fn ip_text(n: u32) -> String {
            format!("10.0.{}.{}", n / 50, n % 50)
        }

        /// The value matching reads for a join key (doc-value form).
        fn value(&self, field: &str) -> Value {
            let process = self.cat == 0;
            match field {
                "process.pid" if process => json!(self.pid),
                "process.ppid" if process => json!(self.ppid),
                "host.name" => json!(format!("host-{}", self.host)),
                "user.name" => json!(if self.system { "SYSTEM" } else { "bob" }),
                "source.ip" => self.ip.map_or(Value::Null, |(a, _)| json!(Self::ip_text(a))),
                "destination.ip" => self.ip.map_or(Value::Null, |(_, b)| json!(Self::ip_text(b))),
                _ => Value::Null,
            }
        }

        fn source(&self) -> Value {
            let mut s = json!({"@timestamp": self.ts, "event": {"category": (["process", "network", "file"][self.cat as usize])},
                "host": {"name": format!("host-{}", self.host), "os": {"name": "Windows", "version": "10.0"}},
                "user": {"name": (if self.system { "SYSTEM" } else { "bob" }), "domain": "NT AUTHORITY"},
                "agent": {"id": "a8a1b7d0-1234-5678-9abc-def012345678", "type": "endgame"},
                "message": "x".repeat(300)});
            if self.cat == 0 {
                s["process"] = json!({"pid": self.pid, "ppid": self.ppid, "name": self.name, "executable": "/usr/bin/some/long/path/to/binary"});
            }
            if let Some((a, b)) = self.ip {
                s["source"] = json!({"ip": Self::ip_text(a)});
                s["destination"] = json!({"ip": Self::ip_text(b)});
            }
            s
        }
    }

    /// The Rally `startsWith` 3-stage sequence over a slice of the real
    /// corpus: `EQL_CORPUS_TSV` (gzip; one process event per line:
    /// timestamp, host.name, process.name (empty: none), pid, ppid). Prints
    /// the windowed run's time, searches and sequences, and the capped
    /// match's. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn eql_corpus_slice_bench() {
        let Ok(path) = std::env::var("EQL_CORPUS_TSV") else { return };
        let runs = std::env::var("EQL_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(5usize).max(1);
        let temp = tempfile::tempdir().unwrap();
        let node = Node::open(temp.path()).unwrap();
        node.create_index("endgame", &json!({"mappings": {"properties": {
            "@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}},
            "process": {"properties": {"pid": {"type": "long"}, "ppid": {"type": "long"}, "name": {"type": "keyword"}}},
            "host": {"properties": {"name": {"type": "keyword"}}}
        }}})).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut text = String::new();
        std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(file), &mut text).unwrap();
        let svc = node.resolve("endgame", ResolveOpts::default()).unwrap().remove(0).svc;
        let mut batch = Vec::new();
        let mut n = 0usize;
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 5 { continue; }
            let mut process = serde_json::Map::new();
            if !f[2].is_empty() { process.insert("name".into(), json!(f[2])); }
            if let Ok(v) = f[3].parse::<i64>() { process.insert("pid".into(), json!(v)); }
            if let Ok(v) = f[4].parse::<i64>() { process.insert("ppid".into(), json!(v)); }
            let doc = json!({"@timestamp": f[0].parse::<i64>().unwrap(), "event": {"category": "process"}, "host": {"name": f[1]}, "process": process});
            batch.push(crate::index::WriteReq::Index { id: Some(format!("{n:08}")), source: doc.to_string(), routing: None, cond: crate::shard::WriteCond::default() });
            n += 1;
            if batch.len() == 25_000 {
                for r in svc.write(std::mem::take(&mut batch), true) { r.unwrap(); }
                svc.refresh().unwrap();
            }
        }
        for r in svc.write(batch, true) { r.unwrap(); }
        svc.refresh().unwrap();
        eprintln!("indexed {n} process events");
        let body = json!({"query": "sequence by host.name with maxspan=30m [process where startsWith(process.name, \"ssh\") == true] by process.ppid [process where startsWith(process.name, \"ssh\") == false] by process.ppid [process where startsWith(process.name, \"ssh\") == true] by process.pid | tail 100 | head 50", "fetch_size": 1000, "size": 100});
        let time = |full_only: bool| {
            let mut times = Vec::new();
            let mut out = Value::Null;
            for _ in 0..runs {
                FULL_STAGES_ONLY.with(|f| f.set(full_only));
                let t = Instant::now();
                out = search(&node, "endgame", &HashMap::new(), &body).unwrap();
                times.push(t.elapsed());
                FULL_STAGES_ONLY.with(|f| f.set(false));
            }
            times.sort();
            (times[times.len() / 2], out["hits"]["sequences"].as_array().map_or(0, Vec::len))
        };
        let (t, seqs) = time(false);
        let (searches, hits, decided) = WINDOW_STATS.with(|s| s.get());
        let (tc, cseqs) = time(true);
        eprintln!("startsWith 3stage: windowed {t:?} ({}, {searches} searches, {hits} events, {seqs} sequences)  capped {tc:?} ({cseqs} sequences)", if decided == Some(true) { "decided" } else { "gave up" });
    }

    /// Diagnostic and check, not an assertion of speed: the Rally `eql`
    /// sequence shapes over `EQL_BENCH_EVENTS` (default 1,000,000)
    /// endgame-like events in segments of `EQL_BENCH_SEGMENT_DOCS` (default
    /// 25,000), windowed and capped, each windowed result checked against
    /// the reference. Run with
    /// `cargo test -p elasticsearch-engine --release eql_windowed_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn eql_windowed_bench() {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let total = env("EQL_BENCH_EVENTS", 1_000_000);
        let per_segment = env("EQL_BENCH_SEGMENT_DOCS", 25_000).max(1);
        let runs = env("EQL_BENCH_RUNS", 5).max(1);
        let temp = tempfile::tempdir().unwrap();
        let node = Node::open(temp.path()).unwrap();
        node.create_index("endgame", &json!({"mappings": {"properties": {
            "@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}},
            "process": {"properties": {"pid": {"type": "long"}, "ppid": {"type": "long"}, "name": {"type": "keyword"}, "executable": {"type": "keyword"}}},
            "host": {"properties": {"name": {"type": "keyword"}, "os": {"properties": {"name": {"type": "keyword"}, "version": {"type": "keyword"}}}}},
            "user": {"properties": {"name": {"type": "keyword"}, "domain": {"type": "keyword"}}},
            "agent": {"properties": {"id": {"type": "keyword"}, "type": {"type": "keyword"}}},
            "source": {"properties": {"ip": {"type": "ip"}}}, "destination": {"properties": {"ip": {"type": "ip"}}},
            "message": {"type": "keyword", "index": false, "doc_values": false}
        }}})).unwrap();
        let mut rng = Lcg(77);
        let names = ["ssh", "sshd", "ssh-agent"];
        // `EQL_BENCH_RALLY=1`: join keys that recur as in the Rally corpus
        // (a few address pairs, IPs on 2% of process events, parent pids
        // from a small pool that ssh processes also take as pids), so the
        // keyed shapes complete sequences instead of matching none.
        let rally = env("EQL_BENCH_RALLY", 0) != 0;
        let (ip_pool, ip_every, ppid_pool) = if rally { (10, 50, 50) } else { (200, 500, 5_000) };
        let evs: Vec<Ev> = (0..total).map(|i| {
            let cat = [0u8, 0, 0, 0, 0, 1, 1, 1, 2, 2][i % 10];
            // Every fifth event ties with the one before.
            let ts = 1_600_000_000_000i64 + i as i64 * 11 - if i % 5 == 1 { 11 } else { 0 };
            // (Same draws as before in the default mode, so its data is unchanged.)
            let mut pid = if rng.next(100) == 0 { 910 + rng.next(10) as i64 } else if rng.next(5) == 0 { 80_001 + rng.next(20_000) as i64 } else { rng.next(80_000) as i64 };
            let name = if rng.next(50) == 0 { names[rng.next(3)] } else { ["proc-a", "proc-b", "explorer.exe", "cmd.exe"][rng.next(4)] };
            if rally && name.starts_with("ssh") && rng.next(2) == 0 {
                pid = rng.next(ppid_pool) as i64;
            }
            let ip = (cat == 1 || (cat == 0 && rng.next(ip_every) == 0)).then(|| (rng.next(ip_pool) as u32, rng.next(ip_pool) as u32));
            Ev { ts, cat, pid, ppid: rng.next(ppid_pool) as i64, name, host: rng.next(20) as u32, ip, system: rng.next(10) < 3 }
        }).collect();
        if rally {
            eprintln!("Rally-shaped join keys (EQL_BENCH_RALLY)");
        }
        let ids: Vec<String> = (0..total).map(|i| format!("{i:08}")).collect();
        let started = Instant::now();
        {
            use crate::index::WriteReq;
            use crate::shard::WriteCond;
            let svc = node.resolve("endgame", ResolveOpts::default()).unwrap().remove(0).svc;
            for (n, chunk) in evs.chunks(per_segment).enumerate() {
                let base = n * per_segment;
                for (k, part) in chunk.chunks(5_000).enumerate() {
                    let w = part.iter().enumerate().map(|(j, e)| WriteReq::Index { id: Some(ids[base + k * 5_000 + j].clone()), source: e.source().to_string(), routing: None, cond: WriteCond::default() }).collect();
                    for r in svc.write(w, true) {
                        r.unwrap();
                    }
                }
                svc.refresh().unwrap();
            }
        }
        eprintln!("indexed {total} events in segments of {per_segment} in {:?}", started.elapsed());
        type Pred = Box<dyn Fn(&Ev) -> bool>;
        let any = || -> Pred { Box::new(|_| true) };
        let process = || -> Pred { Box::new(|e| e.cat == 0) };
        let network = || -> Pred { Box::new(|e| e.cat == 1) };
        let pid = |p: i64| -> Pred { Box::new(move |e| e.cat == 0 && e.pid == p) };
        let many = || -> Pred { Box::new(|e| e.cat == 0 && e.pid > 80_000) };
        let ssh = |yes: bool| -> Pred { Box::new(move |e| e.cat == 0 && e.name.starts_with("ssh") == yes) };
        let system = |yes: bool| -> Pred { Box::new(move |e| e.cat == 0 && e.system == yes) };
        let pids = |k: i64| -> Vec<Pred> { (0..k).map(|i| pid(910 + i)).collect() };
        let stages = |k: i64| (0..k).map(|i| format!("[process where process.pid == {}]", 910 + i)).collect::<Vec<_>>().join(" ");
        let shapes: Vec<(&str, Value, Vec<Pred>)> = vec![
            ("sequence_2stages_baseline", json!({"query": "sequence [any where true] [any where true]"}), vec![any(), any()]),
            ("sequence_2stages_headThenTail", json!({"query": "sequence [any where true] [any where true] | head 200 | tail 100"}), vec![any(), any()]),
            ("sequence_2stages_tailThenHead", json!({"query": "sequence [any where true] [any where true] | tail 200 | head 100"}), vec![any(), any()]),
            ("sequence_2stages_maxspan1m", json!({"query": "sequence with maxspan=1m [any where true] [any where true]"}), vec![any(), any()]),
            ("sequence_2stages_nullJoin", json!({"query": "sequence by ?source.ip [process where true] [process where true]"}), vec![process(), process()]),
            ("sequence_3stages", json!({"query": format!("sequence {}", stages(3))}), pids(3)),
            ("sequence_5stages", json!({"query": format!("sequence {}", stages(5))}), pids(5)),
            ("sequence_10stages", json!({"query": format!("sequence {}", stages(10))}), pids(10)),
            ("sequence_2stages_fewThenFew_noJoinKey", json!({"query": "sequence [process where process.pid == 910] [process where process.pid == 911]"}), vec![pid(910), pid(911)]),
            ("sequence_2stages_fewThenMany_noJoinKey", json!({"query": "sequence [process where process.pid == 910] [process where process.pid > 80000]"}), vec![pid(910), many()]),
            ("sequence_2stages_manyThenFew_noJoinKey", json!({"query": "sequence [process where process.pid > 80000] [process where process.pid == 911]"}), vec![many(), pid(911)]),
            ("sequence_2stages_manyThenMany_noJoinKey", json!({"query": "sequence [process where process.pid > 80000] [process where process.pid > 80000]"}), vec![many(), many()]),
            ("sequence_2stages_fewThenFew_highCardJoinKey", json!({"query": "sequence by process.ppid [process where process.pid == 910] [process where process.pid == 911]"}), vec![pid(910), pid(911)]),
            ("sequence_2stages_fewThenMany_highCardJoinKey", json!({"query": "sequence by process.ppid [process where process.pid == 910] [process where process.pid > 80000]"}), vec![pid(910), many()]),
            // The Rally task of this name runs the fewThenFew query.
            ("sequence_2stages_manyThenMany_highCardJoinKey", json!({"query": "sequence by process.ppid [process where process.pid == 910] [process where process.pid == 911]"}), vec![pid(910), pid(911)]),
            ("sequence_2stages_manyThenFew_highCardJoinKey", json!({"query": "sequence by process.ppid [process where process.pid > 80000] [process where process.pid == 911]"}), vec![many(), pid(911)]),
            ("sequence_2stage_nofilter_fetch1000_size_1000", json!({"query": "sequence by source.ip, destination.ip [process where true] [process where true]", "fetch_size": 1000, "size": 1000}), vec![process(), process()]),
            ("sequence_2stage_equalityFilter_maxspan1m_fetch1000_size500", json!({"query": "sequence by source.ip, destination.ip with maxspan=1m [process where user.name != \"SYSTEM\"] [process where user.name == \"SYSTEM\"]", "fetch_size": 1000, "size": 500}), vec![system(false), system(true)]),
            ("sequence_2stage_nofilter_fetch1000_size1000_tail1000_head200", json!({"query": "sequence by source.ip, destination.ip [process where true] [process where true] | tail 1000 | head 200", "fetch_size": 1000, "size": 1000}), vec![process(), process()]),
            ("sequence_4stage_nofilter_maxspan5m_fetch1000_size100_head100_tail50", json!({"query": "sequence by source.ip, destination.ip with maxspan=5m [process where true] [process where true] [process where true] [network where true] |head 100 | tail 50", "fetch_size": 1000, "size": 100}), vec![process(), process(), process(), network()]),
            ("sequence_3stage_startsWithfilter_maxspan30m_fetch1000_size100_tail100_head50", json!({"query": "sequence by host.name with maxspan=30m [process where startsWith(process.name, \"ssh\") == true] by process.ppid [process where startsWith(process.name, \"ssh\") == false] by process.ppid [process where startsWith(process.name, \"ssh\") == true] by process.pid | tail 100 | head 50", "fetch_size": 1000, "size": 100}), vec![ssh(true), ssh(false), ssh(true)]),
        ];
        let p50 = |f: &mut dyn FnMut() -> Value| -> (std::time::Duration, Value) {
            let mut times = Vec::new();
            let mut out = Value::Null;
            for _ in 0..runs {
                let t = Instant::now();
                out = f();
                times.push(t.elapsed());
            }
            times.sort();
            (times[times.len() / 2], out)
        };
        let search_all = |body: &Value, full_only: bool| -> Value {
            FULL_STAGES_ONLY.with(|f| f.set(full_only));
            let t0 = Instant::now();
            let out = search(&node, "endgame", &HashMap::new(), body).unwrap();
            phase_add(5, t0);
            FULL_STAGES_ONLY.with(|f| f.set(false));
            out
        };
        // ES 9.5.5 p50 (ms) of each Rally `eql` task on the full 60M-event
        // corpus (i4g.8xlarge, one shard), for reference next to the
        // timings; `-` where the round's report gave none.
        let es_p50 = |name: &str| -> String {
            let ms = match name {
                "condition_noMatches" => 1.65,
                "condition_allMatches" => 15.8,
                "condition_scriptFilter" => 588.0,
                "sequence_2stages_headThenTail" => 57.0,
                "sequence_3stages" => 38.0,
                "sequence_5stages" => 63.0,
                "sequence_10stages" => 123.0,
                "sequence_2stages_fewThenFew_noJoinKey" => 25.0,
                "sequence_2stages_fewThenMany_noJoinKey" => 1766.0,
                "sequence_2stages_manyThenMany_noJoinKey" => 45.0,
                "sequence_2stages_fewThenFew_highCardJoinKey" | "sequence_2stages_manyThenMany_highCardJoinKey" => 30.0,
                "sequence_2stages_fewThenMany_highCardJoinKey" => 475.0,
                "sequence_2stage_nofilter_fetch1000_size_1000" => 11.4,
                "sequence_2stage_equalityFilter_maxspan1m_fetch1000_size500" => 38.8,
                "sequence_2stage_nofilter_fetch1000_size1000_tail1000_head200" => 7.4,
                "sequence_4stage_nofilter_maxspan5m_fetch1000_size100_head100_tail50" => 7.7,
                "sequence_3stage_startsWithfilter_maxspan30m_fetch1000_size100_tail100_head50" => 56.0,
                // baseline / tailThenHead / maxspan1m / nullJoin: 51-55 ms.
                n if n.starts_with("sequence_2stages_") && !n.contains("Join") || n == "sequence_2stages_nullJoin" => 53.0,
                _ => return "-".into(),
            };
            format!("{ms}ms")
        };
        for (name, body) in [
            ("condition_noMatches", json!({"query": "any where false"})),
            ("condition_allMatches", json!({"query": "any where true"})),
            ("condition_scriptFilter", json!({"query": "process where startsWith(concat(process.name, \" \", process.pid), \"ssh\")"})),
        ] {
            let (t, out) = p50(&mut || search_all(&body, false));
            eprintln!("{name:80} {t:>10.2?}  (ES {})  events={}", es_p50(name), out["hits"]["events"].as_array().map_or(0, Vec::len));
        }
        let mut mismatches = Vec::new();
        // `EQL_BENCH_ONLY`: the shapes whose name contains one of these
        // (`|`-separated).
        let only = std::env::var("EQL_BENCH_ONLY").ok();
        for (name, body, preds) in &shapes {
            if only.as_deref().is_some_and(|o| !o.split('|').any(|x| name.contains(x))) {
                continue;
            }
            rustic_core::query::stats::reset();
            PHASE_NS.with(|p| *p.borrow_mut() = [0; 9]);
            let (windowed_t, out) = p50(&mut || search_all(body, false));
            let ph = PHASE_NS.with(|p| *p.borrow());
            let ms = |i: usize| ph[i] as f64 / 1e6 / runs as f64;
            eprintln!("    {name} per run: stage searches {:.1}ms (execute {:.1}, parse {:.1}), fetch_sources {:.1}ms, materialize {:.1}ms, passes {:.1}ms, windowed {:.1}ms, to hits {:.1}ms, whole {:.1}ms", ms(0), ms(1), ms(2), ms(3), ms(4), ms(7), ms(6), ms(8), ms(5));
            // With `--features query-stats`: where the windowed runs' time
            // went (summed over the runs).
            if rustic_core::query::stats::ENABLED {
                let parts: Vec<String> = rustic_core::query::stats::snapshot().into_iter().filter(|(_, v, _)| *v > 0).map(|(k, v, _)| format!("{k}={v}")).collect();
                eprintln!("    {name} counters ({runs} runs): {}", parts.join(" "));
            }
            let (searches, hits, decided) = WINDOW_STATS.with(|s| s.get());
            let decided = decided == Some(true);
            let stopped = stops();
            let (capped_t, capped) = p50(&mut || search_all(body, true));
            // The reference over every event (the capped match only sees the
            // first 10,000 of each stage).
            let r = Request::from_json(body).unwrap();
            let EqlQuery::Sequence { by, stages: parsed, until, .. } = parse(&r.query, &Opts { case_sensitive: false }).unwrap() else { panic!("a sequence") };
            assert!(until.is_none());
            let keys: Vec<Vec<String>> = parsed.iter().flat_map(|s| std::iter::repeat_n(by.iter().chain(&s.by).cloned().collect::<Vec<String>>(), s.runs)).collect();
            assert_eq!(keys.len(), preds.len(), "{name}");
            let (mut first, mut later): (Vec<Entry<'_>>, Vec<Entry<'_>>) = (Vec::new(), Vec::new());
            for (i, e) in evs.iter().enumerate() {
                for (si, (pred, ks)) in preds.iter().zip(&keys).enumerate() {
                    if !pred(e) {
                        continue;
                    }
                    let key: Option<Vec<Value>> = ks.iter().map(|k| {
                        let v = e.value(k.strip_prefix('?').unwrap_or(k));
                        (!v.is_null() || k.starts_with('?')).then_some(v)
                    }).collect();
                    if let Some(key) = key {
                        if si == 0 { first.push((e.ts, ids[i].as_str(), Some(0), key)) } else { later.push((e.ts, ids[i].as_str(), Some(si), key)) }
                    }
                }
            }
            let n = keys.len();
            let expected = if decided { decide(body, n, first, later, &stopped) } else { shape(&capped) };
            let ok = shape(&out) == expected;
            if !ok {
                mismatches.push(*name);
            }
            eprintln!("{name:80} windowed {windowed_t:>10.2?} ({}, {searches} searches, {hits} events)  capped {capped_t:>10.2?}  (ES {})  sequences={}  {}",
                if decided { "decided" } else { "gave up" }, es_p50(name), out["hits"]["sequences"].as_array().map_or(0, Vec::len), if ok { "ok" } else { "MISMATCH" });
        }
        assert!(mismatches.is_empty(), "windowed results differ from the reference: {mismatches:?}");
    }
}
