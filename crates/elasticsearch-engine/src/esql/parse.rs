//! ES|QL parser: query text → [`Cmd`] pipeline.

use super::lexer::{lex, split_top, Dialect, Tok, Token};
use super::{parsing_err, Val};
use crate::error::{EsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Serializable so a shard read can carry row filters and keys to the
/// node holding the shard ([`super::remote::FastPlan`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Lit(#[serde(with = "super::partial::wire_val")] Val),
    /// Column reference by name, with its offset in the query.
    Col(String, usize),
    /// Bound column reference.
    Idx(usize),
    /// Reference to the k-th aggregate of a STATS.
    AggRef(usize),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    /// `IS NULL` (false) / `IS NOT NULL` (true).
    IsNull(Box<Expr>, bool),
    In(Box<Expr>, Vec<Expr>, bool),
    /// `LIKE` / `RLIKE` with patterns; negated; regex.
    Like(Box<Expr>, Vec<String>, bool, bool),
    /// Function call (upper-cased name), offset.
    Call(String, Vec<Expr>, usize),
    Star,
    Cast(Box<Expr>, String),
    /// `{ "option": value }` map literal (function options).
    Map(#[serde(with = "super::partial::wire_pairs")] Vec<(String, Val)>),
    /// SQL `CASE WHEN c THEN v ... ELSE e END` is mapped to `Call("CASE")`.
    Between(Box<Expr>, Box<Expr>, Box<Expr>, bool),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub expr: Expr,
    /// Per-aggregate `WHERE` filter (STATS only).
    pub filter: Option<Expr>,
    pub pos: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SortKey {
    pub expr: Expr,
    pub desc: bool,
    pub nulls_first: Option<bool>,
}

#[derive(Clone, Debug)]
pub enum Cmd {
    /// `FROM`/`TS`: index patterns, metadata fields, and `(subquery)` branches.
    From { indices: Vec<String>, metadata: Vec<String>, pos: usize, ts: bool, subqueries: Vec<Vec<Cmd>> },
    Row(Vec<Field>),
    ShowInfo,
    Where(Expr, usize),
    Eval(Vec<Field>),
    Keep(Vec<String>, usize),
    Drop(Vec<String>, usize),
    Rename(Vec<(String, String)>, usize),
    Sort(Vec<SortKey>),
    Limit(usize),
    /// `LIMIT n BY key, ...`: at most `n` rows per key.
    LimitBy(usize, Vec<Expr>),
    SqlPivot { aggregate: Expr, column: String, values: Vec<(Val, String)>, pos: usize },
    Stats { aggs: Vec<Field>, by: Vec<Field>, inline: bool, pos: usize },
    MvExpand(String, usize),
    Dissect { field: Expr, pattern: String, sep: String, pos: usize },
    Grok { field: Expr, pattern: String, pos: usize },
    /// `LOOKUP JOIN index ON field, ...` with the offsets of the index name and the join fields.
    /// `cond`: the `ON` boolean expression of an expression-based join (`on` is then empty).
    /// `fields`: the lookup index columns the rest of the pipeline can read
    /// (ES's lookup-side field extraction after `PruneColumns`), set by the
    /// planner; `None` reads every column.
    LookupJoin { index: String, index_pos: usize, on: Vec<String>, on_pos: Vec<usize>, pos: usize, cond: Option<Expr>, fields: Option<HashSet<String>> },
    /// `ENRICH [mode:]policy [ON field] [WITH [new =] field, ...]`; empty
    /// `with` = all enrich fields of the policy.
    Enrich { policy: String, mode: Option<String>, on: Option<String>, with: Vec<(String, String)>, pos: usize },
    Sample(f64),
    Fork(Vec<Vec<Cmd>>),
    /// `PROMQL [params] [name=](expr)`.
    Promql(Box<super::promql::Promql>),
    /// `TS_INFO` (true) / `METRICS_INFO` (false), with the keyword's offset.
    Info(bool, usize),
}

/// Query parameters (`params`): positional and named.
#[derive(Default, Clone, Debug)]
pub struct Params {
    pub positional: Vec<Value>,
    pub named: HashMap<String, Value>,
}

impl Params {
    pub fn from_json(v: Option<&Value>) -> Result<Params> {
        let mut p = Params::default();
        let Some(v) = v else { return Ok(p) };
        let arr = match v {
            Value::Array(a) => a.clone(),
            Value::Object(m) => m.iter().map(|(k, v)| serde_json::json!({ k.clone(): v.clone() })).collect(),
            Value::Null => vec![],
            _ => return Err(EsError::parsing("[params] must be an array")),
        };
        for item in arr {
            // Named params are also addressable by position (`?` takes them in order).
            match item {
                Value::Object(m) if m.len() == 1 => {
                    let (k, v) = m.into_iter().next().unwrap();
                    p.positional.push(v.clone());
                    p.named.insert(k, v);
                }
                other => p.positional.push(other),
            }
        }
        Ok(p)
    }
}

pub fn json_to_val(v: &Value) -> Val {
    match v {
        Value::Null => Val::Null,
        Value::Bool(b) => Val::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    Val::Int(i)
                } else {
                    Val::Long(i)
                }
            } else {
                Val::Double(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => Val::Str(s.clone()),
        Value::Array(a) => Val::from_list(a.iter().map(json_to_val).collect()),
        Value::Object(_) => Val::Str(v.to_string()),
    }
}

pub fn time_unit(s: &str) -> Option<Val> {
    let s = s.to_ascii_lowercase();
    Some(match s.as_str() {
        "millisecond" | "milliseconds" | "ms" => Val::Dur(1),
        "second" | "seconds" | "sec" | "s" => Val::Dur(1000),
        "minute" | "minutes" | "min" => Val::Dur(60_000),
        "hour" | "hours" | "h" => Val::Dur(3_600_000),
        "day" | "days" | "d" => Val::Period(0, 1),
        "week" | "weeks" | "w" => Val::Period(0, 7),
        "month" | "months" | "mo" => Val::Period(1, 0),
        "quarter" | "quarters" | "q" => Val::Period(3, 0),
        "year" | "years" | "yr" | "y" => Val::Period(12, 0),
        _ => return None,
    })
}

/// ES's `EsqlParser` expression depth limit.
const MAX_EXPRESSION_DEPTH: u32 = 400;

pub struct Parser<'a> {
    pub full: &'a str,
    pub toks: Vec<Token>,
    pub i: usize,
    pub d: Dialect,
    pub params: &'a Params,
    pub next_positional: &'a std::cell::Cell<usize>,
    nest: crate::parse_depth::Nesting,
}

fn kw(t: &Tok, k: &str) -> bool {
    matches!(t, Tok::Ident(s) if s.eq_ignore_ascii_case(k))
}

impl<'a> Parser<'a> {
    pub fn new(full: &'a str, from: usize, to: usize, d: Dialect, params: &'a Params, np: &'a std::cell::Cell<usize>) -> Result<Parser<'a>> {
        Ok(Parser { full, toks: lex(full, from, to, d)?, i: 0, d, params, next_positional: np, nest: crate::parse_depth::Nesting::new(MAX_EXPRESSION_DEPTH) })
    }

    pub fn peek(&self) -> &Tok {
        &self.toks[self.i].tok
    }
    pub fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.i + k).min(self.toks.len() - 1)].tok
    }
    pub fn pos(&self) -> usize {
        self.toks[self.i].start
    }
    pub fn at_kw(&self, k: &str) -> bool {
        kw(self.peek(), k)
    }
    pub fn eat_kw(&mut self, k: &str) -> bool {
        if self.at_kw(k) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    pub fn at_eof(&self) -> bool {
        matches!(self.peek(), Tok::Eof)
    }
    pub fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.i += 1;
            true
        } else {
            false
        }
    }
    pub fn expect(&mut self, t: &Tok, what: &str) -> Result<()> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(self.unexpected(what))
        }
    }
    pub fn unexpected(&self, what: &str) -> EsError {
        let t = &self.toks[self.i];
        let txt = if matches!(t.tok, Tok::Eof) { "<EOF>".to_string() } else { self.full[t.start..t.end].to_string() };
        parsing_err(self.full, t.start, format!("mismatched input '{txt}' expecting {what}"))
    }
    pub fn text(&self, start: usize, end_tok: usize) -> String {
        let end = if end_tok == 0 { start } else { self.toks[end_tok - 1].end };
        self.full[start..end.max(start)].trim().to_string()
    }

    /// Identifier (plain or quoted).
    pub fn ident(&mut self) -> Result<String> {
        match self.peek().clone() {
            Tok::Ident(s) | Tok::QIdent(s) => {
                self.i += 1;
                Ok(s)
            }
            Tok::Param(p) => {
                let v = self.param_value(&p)?;
                Ok(match v {
                    Value::Object(m) => m.get("identifier").or_else(|| m.get("pattern")).or_else(|| m.get("value")).and_then(Value::as_str).unwrap_or("").to_string(),
                    Value::String(s) => s,
                    other => other.to_string(),
                })
            }
            _ => Err(self.unexpected("{UNQUOTED_IDENTIFIER, QUOTED_IDENTIFIER}")),
        }
    }

    /// Dotted identifier path (`a.b`, `` `a`.b ``).
    pub fn qualified(&mut self) -> Result<String> {
        let mut s = self.ident()?;
        while matches!(self.peek(), Tok::Op(".")) {
            self.i += 1;
            s.push('.');
            s.push_str(&self.ident()?);
        }
        Ok(s)
    }

    fn param_value(&mut self, p: &str) -> Result<Value> {
        let pos = self.pos();
        self.i += 1;
        if p.is_empty() {
            let k = self.next_positional.get();
            self.next_positional.set(k + 1);
            return self.params.positional.get(k).cloned().ok_or_else(|| parsing_err(self.full, pos, format!("Not enough actual parameters {}", self.params.positional.len())));
        }
        if let Ok(n) = p.parse::<usize>() {
            return self
                .params
                .positional
                .get(n.wrapping_sub(1))
                .cloned()
                .ok_or_else(|| parsing_err(self.full, pos, format!("No parameter is defined for position {n}, did you mean any position between 1 and {}?", self.params.positional.len())));
        }
        self.params.named.get(p).cloned().ok_or_else(|| parsing_err(self.full, pos, format!("Unknown query parameter [{p}]")))
    }

    pub fn expr(&mut self) -> Result<Expr> {
        self.expr_bp(1)
    }

    fn binop(&self) -> Option<(u8, &'static str)> {
        let t = self.peek();
        let sql = self.d == Dialect::Sql;
        Some(match t {
            Tok::Ident(s) => {
                let u = s.to_ascii_uppercase();
                match u.as_str() {
                    "OR" => (1, "OR"),
                    "AND" => (2, "AND"),
                    "LIKE" | "RLIKE" | "IN" | "IS" | "BETWEEN" => (4, "CMPKW"),
                    "NOT" if matches!(self.peek_at(1), Tok::Ident(n) if ["LIKE", "RLIKE", "IN", "BETWEEN"].iter().any(|k| n.eq_ignore_ascii_case(k))) => (4, "CMPKW"),
                    _ => return None,
                }
            }
            Tok::Op(o) => match *o {
                "==" | "!=" | "<>" | "<" | "<=" | ">" | ">=" => (4, o),
                "=" if sql => (4, "=="),
                ":" if !sql => (4, ":"),
                "+" | "-" => (5, o),
                "*" | "/" | "%" => (6, o),
                "::" => (8, o),
                _ => return None,
            },
            _ => return None,
        })
    }

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            let at = self.toks[self.i].start;
            return Err(parsing_err(self.full, at, format!("ESQL statement exceeded the maximum expression depth allowed ({MAX_EXPRESSION_DEPTH})")));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    fn expr_bp(&mut self, min: u8) -> Result<Expr> {
        self.nested(true, |p| p.expr_bp_unnested(min))
    }

    fn expr_bp_unnested(&mut self, min: u8) -> Result<Expr> {
        let lhs_pos = self.pos();
        let mut lhs = self.prefix()?;
        loop {
            let Some((prec, op)) = self.binop() else { break };
            if prec >= min && !self.nest.grow() {
                let at = self.toks[self.i].start;
                return Err(parsing_err(self.full, at, format!("ESQL statement exceeded the maximum expression depth allowed ({MAX_EXPRESSION_DEPTH})")));
            }
            if prec < min {
                break;
            }
            if op == "::" {
                self.i += 1;
                let ty = self.ident()?.to_ascii_lowercase();
                if ty == "text" && self.d == Dialect::Esql {
                    return Err(parsing_err(self.full, lhs_pos, "Unsupported conversion to type [TEXT]"));
                }
                lhs = Expr::Cast(Box::new(lhs), ty);
                continue;
            }
            if op == "CMPKW" {
                let neg = self.eat_kw("NOT");
                if self.eat_kw("IS") {
                    let not = self.eat_kw("NOT");
                    if !self.eat_kw("NULL") {
                        return Err(self.unexpected("NULL"));
                    }
                    lhs = Expr::IsNull(Box::new(lhs), not);
                    continue;
                }
                if self.eat_kw("BETWEEN") {
                    let lo = self.expr_bp(5)?;
                    if !self.eat_kw("AND") {
                        return Err(self.unexpected("AND"));
                    }
                    let hi = self.expr_bp(5)?;
                    lhs = Expr::Between(Box::new(lhs), Box::new(lo), Box::new(hi), neg);
                    continue;
                }
                if self.eat_kw("IN") {
                    self.expect(&Tok::LParen, "'('")?;
                    let mut items = Vec::new();
                    if !self.eat(&Tok::RParen) {
                        loop {
                            items.push(self.expr()?);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            self.expect(&Tok::RParen, "{',', ')'}")?;
                            break;
                        }
                    }
                    lhs = Expr::In(Box::new(lhs), items, neg);
                    continue;
                }
                let rlike = if self.eat_kw("LIKE") {
                    false
                } else if self.eat_kw("RLIKE") {
                    true
                } else {
                    return Err(self.unexpected("{LIKE, RLIKE, IN}"));
                };
                let mut pats = Vec::new();
                let op = if rlike { "RLIKE" } else { "LIKE" };
                if self.eat(&Tok::LParen) {
                    loop {
                        pats.push(self.pattern_lit(op, lhs_pos)?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "')'")?;
                } else {
                    pats.push(self.pattern_lit(op, lhs_pos)?);
                }
                if self.d == Dialect::Sql && !rlike {
                    pats = pats.iter().map(|p| sql_like_to_esql(p)).collect();
                }
                if self.d == Dialect::Esql {
                    if let Some(p) = pats.iter().find(|p| super::automaton::too_complex(p, rlike)) {
                        return Err(parsing_err(self.full, lhs_pos, format!("Invalid regex pattern for RLIKE [{p}]: [Pattern was too complex to determinize]")));
                    }
                }
                lhs = Expr::Like(Box::new(lhs), pats, neg, rlike);
                continue;
            }
            self.i += 1;
            let rhs = self.expr_bp(prec + 1)?;
            let bop = match op {
                "OR" => BinOp::Or,
                "AND" => BinOp::And,
                "==" => BinOp::Eq,
                "!=" | "<>" => BinOp::Neq,
                "<" => BinOp::Lt,
                "<=" => BinOp::Le,
                ">" => BinOp::Gt,
                ">=" => BinOp::Ge,
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::Div,
                "%" => BinOp::Mod,
                ":" => {
                    // The match operator (`field:"query"`), positioned at its field.
                    lhs = Expr::Call("MATCH_OP".into(), vec![lhs, rhs], lhs_pos);
                    continue;
                }
                _ => unreachable!(),
            };
            lhs = Expr::Bin(bop, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// A LIKE/RLIKE pattern: a string literal or a string parameter.
    fn pattern_lit(&mut self, op: &str, lhs_pos: usize) -> Result<String> {
        if let Tok::Param(p) = self.peek().clone() {
            let v = self.param_value(&p)?;
            return match v {
                Value::String(s) => Ok(s),
                other => {
                    let ty = json_to_val(&other).type_name();
                    Err(parsing_err(self.full, lhs_pos, format!("Invalid pattern parameter type for {op} [?{p}]: expected string, found {ty}")))
                }
            };
        }
        self.string_lit()
    }

    fn string_lit(&mut self) -> Result<String> {
        match self.peek().clone() {
            Tok::Str(s) => {
                self.i += 1;
                Ok(s)
            }
            Tok::Param(p) => match self.param_value(&p)? {
                Value::String(s) => Ok(s),
                other => Ok(other.to_string()),
            },
            _ => Err(self.unexpected("{QUOTED_STRING}")),
        }
    }

    fn prefix(&mut self) -> Result<Expr> {
        self.nested(false, Self::prefix_unnested)
    }

    fn prefix_unnested(&mut self) -> Result<Expr> {
        if self.eat_kw("NOT") {
            return Ok(Expr::Not(Box::new(self.expr_bp(3)?)));
        }
        if matches!(self.peek(), Tok::Op("-")) {
            self.i += 1;
            let e = self.expr_bp(7)?;
            return Ok(match e {
                Expr::Lit(Val::Int(i)) => Expr::Lit(Val::Int(-i)),
                Expr::Lit(Val::Long(i)) => Expr::Lit(Val::Long(-i)),
                Expr::Lit(Val::Double(d)) => Expr::Lit(Val::Double(-d)),
                other => Expr::Neg(Box::new(other)),
            });
        }
        if matches!(self.peek(), Tok::Op("+")) {
            self.i += 1;
            return self.expr_bp(7);
        }
        self.primary()
    }

    fn number_lit(&mut self, v: Val) -> Result<Expr> {
        // Time span literal: `1 day`, `2 hours`.
        if self.d == Dialect::Esql {
            if let Tok::Ident(u) = self.peek().clone() {
                if let Some(unit) = time_unit(&u) {
                    let n = match &v {
                        Val::Int(i) | Val::Long(i) => *i,
                        _ => return Err(self.unexpected("integer before time unit")),
                    };
                    self.i += 1;
                    return Ok(Expr::Lit(match unit {
                        Val::Dur(ms) => Val::Dur(ms * n),
                        Val::Period(m, d) => Val::Period(m * n, d * n),
                        other => other,
                    }));
                }
            }
        }
        Ok(Expr::Lit(v))
    }

    fn primary(&mut self) -> Result<Expr> {
        let pos = self.pos();
        match self.peek().clone() {
            Tok::Int(i) => {
                self.i += 1;
                let v = if i >= i32::MIN as i64 && i <= i32::MAX as i64 { Val::Int(i) } else { Val::Long(i) };
                self.number_lit(v)
            }
            Tok::Dec(d) => {
                self.i += 1;
                self.number_lit(Val::Double(d))
            }
            Tok::Str(s) => {
                self.i += 1;
                Ok(Expr::Lit(Val::Str(s)))
            }
            Tok::Param(p) => {
                let v = self.param_value(&p)?;
                if let Value::Object(m) = &v {
                    if let Some(id) = m.get("identifier").and_then(Value::as_str) {
                        return Ok(Expr::Col(id.to_string(), pos));
                    }
                    if m.get("kind").and_then(Value::as_str).map(|k| k.eq_ignore_ascii_case("identifier")).unwrap_or(false) {
                        return Ok(Expr::Col(m.get("value").and_then(Value::as_str).unwrap_or("").to_string(), pos));
                    }
                    if let Some(x) = m.get("value") {
                        return Ok(Expr::Lit(json_to_val(x)));
                    }
                }
                let v = json_to_val(&v);
                self.number_lit(v)
            }
            Tok::LParen => {
                self.i += 1;
                let e = self.expr()?;
                self.expect(&Tok::RParen, "')'")?;
                Ok(e)
            }
            Tok::LBracket => {
                self.i += 1;
                let mut items = Vec::new();
                if !self.eat(&Tok::RBracket) {
                    loop {
                        match self.expr()? {
                            Expr::Lit(v) => items.push(v),
                            _ => return Err(parsing_err(self.full, pos, "array literals may only contain constants")),
                        }
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        self.expect(&Tok::RBracket, "{',', ']'}")?;
                        break;
                    }
                }
                Ok(Expr::Lit(if items.is_empty() { Val::Null } else { Val::Multi(items) }))
            }
            Tok::LBrace => {
                self.i += 1;
                let mut entries = Vec::new();
                if !self.eat(&Tok::RBrace) {
                    loop {
                        let k = self.string_lit()?;
                        if !matches!(self.peek(), Tok::Op(":")) {
                            return Err(self.unexpected("':'"));
                        }
                        self.i += 1;
                        let v = match self.expr()? {
                            Expr::Lit(v) => v,
                            _ => return Err(parsing_err(self.full, pos, "map values must be constants")),
                        };
                        entries.push((k, v));
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        self.expect(&Tok::RBrace, "{',', '}'}")?;
                        break;
                    }
                }
                Ok(Expr::Map(entries))
            }
            Tok::Ident(s) => {
                let u = s.to_ascii_uppercase();
                if matches!(self.peek_at(1), Tok::LParen) {
                    self.i += 2;
                    return self.call(u, pos);
                }
                match u.as_str() {
                    "TRUE" => {
                        self.i += 1;
                        return Ok(Expr::Lit(Val::Bool(true)));
                    }
                    "FALSE" => {
                        self.i += 1;
                        return Ok(Expr::Lit(Val::Bool(false)));
                    }
                    "NULL" => {
                        self.i += 1;
                        return Ok(Expr::Lit(Val::Null));
                    }
                    "CASE" if self.d == Dialect::Sql => {
                        self.i += 1;
                        return self.sql_case(pos);
                    }
                    "CAST" if self.d == Dialect::Sql && matches!(self.peek_at(1), Tok::LParen) => {}
                    _ => {}
                }
                let name = self.qualified()?;
                Ok(Expr::Col(name, pos))
            }
            Tok::QIdent(_) => {
                let name = self.qualified()?;
                Ok(Expr::Col(name, pos))
            }
            Tok::Op("*") => {
                self.i += 1;
                Ok(Expr::Star)
            }
            _ => Err(self.unexpected("{QUOTED_STRING, INTEGER_LITERAL, DECIMAL_LITERAL, FALSE, NOT, NULL, ?, TRUE, '+', '-', '[', '(', UNQUOTED_IDENTIFIER, QUOTED_IDENTIFIER}")),
        }
    }

    fn sql_case(&mut self, pos: usize) -> Result<Expr> {
        let mut args = Vec::new();
        let operand = if !self.at_kw("WHEN") { Some(self.expr()?) } else { None };
        while self.eat_kw("WHEN") {
            let c = self.expr()?;
            let c = match &operand {
                Some(o) => Expr::Bin(BinOp::Eq, Box::new(o.clone()), Box::new(c)),
                None => c,
            };
            if !self.eat_kw("THEN") {
                return Err(self.unexpected("THEN"));
            }
            args.push(c);
            args.push(self.expr()?);
        }
        if self.eat_kw("ELSE") {
            args.push(self.expr()?);
        }
        if !self.eat_kw("END") {
            return Err(self.unexpected("END"));
        }
        Ok(Expr::Call("CASE".into(), args, pos))
    }

    fn call(&mut self, name: String, pos: usize) -> Result<Expr> {
        let mut name = name;
        let mut args = Vec::new();
        if self.d == Dialect::Sql && name == "CAST" {
            let e = self.expr()?;
            if !self.eat_kw("AS") {
                return Err(self.unexpected("AS"));
            }
            let ty = self.ident()?.to_ascii_lowercase();
            self.expect(&Tok::RParen, "')'")?;
            return Ok(Expr::Cast(Box::new(e), ty));
        }
        if self.eat(&Tok::RParen) {
            return Ok(Expr::Call(name, args, pos));
        }
        if self.d == Dialect::Sql && self.eat_kw("DISTINCT") && name == "COUNT" {
            name = "COUNT_DISTINCT".into();
        }
        loop {
            args.push(self.expr()?);
            if self.eat(&Tok::Comma) {
                continue;
            }
            self.expect(&Tok::RParen, "{',', ')'}")?;
            break;
        }
        Ok(Expr::Call(name, args, pos))
    }

    /// `[name =] expr [WHERE cond]` list (EVAL, STATS, ROW).
    pub fn fields(&mut self, allow_where: bool, stop_kw: &[&str]) -> Result<Vec<Field>> {
        let mut out = Vec::new();
        loop {
            if self.at_eof() || stop_kw.iter().any(|k| self.at_kw(k)) {
                break;
            }
            let start = self.pos();
            let pos = start;
            let named = matches!(self.peek(), Tok::Ident(_) | Tok::QIdent(_)) && {
                // name may be dotted: a.b = ...
                let mut k = 1;
                while matches!(self.peek_at(k), Tok::Op(".")) {
                    k += 2;
                }
                matches!(self.peek_at(k), Tok::Op("="))
            };
            let name = if named {
                let n = self.qualified()?;
                self.i += 1; // '='
                Some(n)
            } else {
                None
            };
            let estart = self.pos();
            let expr = self.expr()?;
            let text = self.text(estart, self.i);
            let filter = if allow_where && self.eat_kw("WHERE") { Some(self.expr()?) } else { None };
            let name = name.unwrap_or_else(|| match &expr {
                Expr::Col(n, _) => n.clone(),
                _ => text.clone(),
            });
            out.push(Field { name, expr, filter, pos });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(out)
    }
}

/// SQL LIKE (`%`, `_`) → ES|QL LIKE (`*`, `?`).
fn sql_like_to_esql(p: &str) -> String {
    let mut out = String::new();
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '%' => out.push('*'),
            '_' => out.push('?'),
            '*' => out.push_str("\\*"),
            '?' => out.push_str("\\?"),
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            c => out.push(c),
        }
    }
    out
}

fn word_at(full: &str, from: usize, to: usize) -> (String, usize) {
    let s = &full[from..to];
    let trimmed = s.trim_start();
    let off = from + (s.len() - trimmed.len());
    let w: String = trimmed.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    let end = off + w.len();
    (w, end)
}

fn unquote_name(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('`') && t.ends_with('`') {
        t[1..t.len() - 1].replace("``", "`")
    } else {
        t.to_string()
    }
}

/// Parse a full ES|QL query (its `SET` statements are skipped).
pub fn parse_query(full: &str, params: &Params) -> Result<Vec<Cmd>> {
    Ok(parse_statement(full, params)?.1)
}

/// Offset of the first non-blank, non-comment byte at or after `from`.
fn skip_blank(full: &str, mut i: usize) -> usize {
    let b = full.as_bytes();
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
        } else if full[i..].starts_with("//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if full[i..].starts_with("/*") {
            i = full[i + 2..].find("*/").map(|p| i + 2 + p + 2).unwrap_or(b.len());
        } else {
            break;
        }
    }
    i
}

/// `statements : setCommand* singleStatement EOF`: the query's settings and
/// its command pipeline.
pub fn parse_statement(full: &str, params: &Params) -> Result<(Vec<super::settings::QuerySetting>, Vec<Cmd>)> {
    let np = std::cell::Cell::new(0usize);
    let mut sets = Vec::new();
    let mut start = 0;
    loop {
        let at = skip_blank(full, start);
        let (word, after) = word_at(full, at, full.len());
        if !word.eq_ignore_ascii_case("SET") || after == at {
            break;
        }
        // `SET name = value ;`
        let semi = split_top(full, after, full.len(), b';', Dialect::Esql)[0].1;
        let mut p = Parser::new(full, after, semi, Dialect::Esql, params, &np)?;
        let name = match p.peek().clone() {
            Tok::Ident(s) | Tok::QIdent(s) => {
                p.i += 1;
                s
            }
            _ => return Err(p.unexpected("{UNQUOTED_IDENTIFIER, QUOTED_IDENTIFIER}")),
        };
        if !matches!(p.peek(), Tok::Op("=")) {
            return Err(p.unexpected("'='"));
        }
        p.i += 1;
        let vpos = p.pos();
        let value = p.expr()?;
        if !matches!(value, Expr::Lit(_) | Expr::Map(_)) {
            return Err(parsing_err(full, vpos, "mismatched input expecting {QUOTED_STRING, INTEGER_LITERAL, DECIMAL_LITERAL, FALSE, NULL, ?, TRUE, '{'}"));
        }
        if semi >= full.len() || !p.at_eof() {
            return Err(p.unexpected("';'"));
        }
        sets.push(super::settings::QuerySetting { name, value, pos: at });
        start = semi + 1;
    }
    let cmds = parse_pipeline(full, start, full.len(), params, &np, true)?;
    Ok((sets, cmds))
}

fn parse_pipeline(full: &str, from: usize, to: usize, params: &Params, np: &std::cell::Cell<usize>, top: bool) -> Result<Vec<Cmd>> {
    let parts = split_top(full, from, to, b'|', Dialect::Esql);
    let mut cmds = Vec::new();
    for (i, (s, e)) in parts.iter().enumerate() {
        if full[*s..*e].trim().is_empty() {
            return Err(parsing_err(full, *s, "mismatched input '<EOF>' expecting a command"));
        }
        let cmd = parse_command(full, *s, *e, params, np, top && i == 0, !top || i > 0)?;
        cmds.push(cmd);
    }
    Ok(cmds)
}

fn parse_command(full: &str, s: usize, e: usize, params: &Params, np: &std::cell::Cell<usize>, source: bool, processing: bool) -> Result<Cmd> {
    let (word, after) = word_at(full, s, e);
    let w = word.to_ascii_uppercase();
    let kw_pos = after - word.len();
    let is_source = matches!(w.as_str(), "FROM" | "ROW" | "SHOW" | "TS" | "METRICS" | "PROMQL");
    if source && !is_source {
        return Err(parsing_err(full, kw_pos, format!("mismatched input '{word}' expecting {{'row', 'from', 'show'}}")));
    }
    if !source && is_source && processing {
        return Err(parsing_err(full, kw_pos, format!("mismatched input '{word}' expecting a processing command")));
    }
    let p = || Parser::new(full, after, e, Dialect::Esql, params, np);
    let finish = |p: &Parser| -> Result<()> {
        if !p.at_eof() {
            return Err(p.unexpected("<EOF>"));
        }
        Ok(())
    };
    Ok(match w.as_str() {
        "FROM" | "TS" | "METRICS" => {
            // `FROM pattern, (subquery), ... [METADATA f, ...]`.
            let mut indices: Vec<String> = Vec::new();
            let mut subqueries: Vec<Vec<Cmd>> = Vec::new();
            let mut meta_raw: Vec<String> = Vec::new();
            let mut in_meta = false;
            for (a, b) in split_top(full, after, e, b',', Dialect::Esql) {
                let seg = &full[a..b];
                let t = seg.trim();
                if in_meta {
                    meta_raw.push(t.to_string());
                    continue;
                }
                let mut index_part = seg;
                if t.starts_with('(') {
                    let open = a + seg.find('(').unwrap();
                    let close = matching_paren(full, open, b).ok_or_else(|| parsing_err(full, open, "mismatched input '<EOF>' expecting ')'"))?;
                    subqueries.push(parse_pipeline(full, open + 1, close, params, np, true)?);
                    index_part = &full[close + 1..b];
                    if index_part.trim().is_empty() {
                        continue;
                    }
                }
                let up = index_part.to_ascii_uppercase();
                let (ix, meta) = match find_kw(&up, "METADATA") {
                    Some(p) => (&index_part[..p], Some(&index_part[p + 8..])),
                    None => (index_part, None),
                };
                let ix = match find_kw(&ix.to_ascii_uppercase(), "OPTIONS") {
                    Some(p) => &ix[..p],
                    None => ix,
                };
                let ix = ix.trim().trim_matches('"');
                if !ix.is_empty() {
                    indices.push(ix.to_string());
                }
                if let Some(m) = meta {
                    in_meta = true;
                    meta_raw.push(m.trim().to_string());
                }
            }
            if indices.is_empty() && subqueries.is_empty() {
                return Err(parsing_err(full, after, "mismatched input '<EOF>' expecting {QUOTED_STRING, UNQUOTED_SOURCE}"));
            }
            let metadata: Vec<String> = meta_raw.into_iter().filter(|x| !x.is_empty()).collect();
            for m in &metadata {
                if !matches!(m.as_str(), "_id" | "_index" | "_score" | "_version" | "_ignored" | "_source" | "_index_mode" | "_tsid") {
                    return Err(parsing_err(full, after, format!("unsupported metadata field [{m}]")));
                }
            }
            Cmd::From { indices, metadata, pos: kw_pos, ts: w != "FROM", subqueries }
        }
        "PROMQL" => Cmd::Promql(Box::new(super::promql::parse(full, after, e, kw_pos, params)?)),
        "ROW" => {
            let mut p = p()?;
            let f = p.fields(false, &[])?;
            finish(&p)?;
            Cmd::Row(f)
        }
        "SHOW" => {
            let (w2, _) = word_at(full, after, e);
            if !w2.eq_ignore_ascii_case("INFO") {
                return Err(parsing_err(full, after, format!("mismatched input '{w2}' expecting 'info'")));
            }
            Cmd::ShowInfo
        }
        "WHERE" => {
            let mut p = p()?;
            let ex = p.expr()?;
            finish(&p)?;
            Cmd::Where(ex, kw_pos)
        }
        "EVAL" => {
            let mut p = p()?;
            let f = p.fields(false, &[])?;
            finish(&p)?;
            if f.is_empty() {
                return Err(p.unexpected("an expression"));
            }
            Cmd::Eval(f)
        }
        "STATS" | "INLINESTATS" | "INLINE" => {
            let (mut start, inline) = (after, w != "STATS");
            if w == "INLINE" {
                let (w2, a2) = word_at(full, after, e);
                if !w2.eq_ignore_ascii_case("STATS") {
                    return Err(parsing_err(full, after, "expecting STATS"));
                }
                start = a2;
            }
            let mut p = Parser::new(full, start, e, Dialect::Esql, params, np)?;
            let aggs = p.fields(true, &["BY"])?;
            let by = if p.eat_kw("BY") { p.fields(false, &[])? } else { vec![] };
            finish(&p)?;
            if aggs.is_empty() && by.is_empty() {
                return Err(parsing_err(full, kw_pos, "At least one aggregation or grouping expression required in [STATS]"));
            }
            Cmd::Stats { aggs, by, inline, pos: kw_pos }
        }
        "KEEP" | "PROJECT" | "DROP" => {
            let mut names: Vec<String> = Vec::new();
            for (a, b) in split_top(full, after, e, b',', Dialect::Esql) {
                let raw = full[a..b].trim();
                // `?name` / `?` / `?1`: an identifier or pattern parameter.
                if let Some(pn) = raw.strip_prefix('?') {
                    names.push(ident_param(full, a, pn, params, np)?);
                    continue;
                }
                if raw.is_empty() {
                    return Err(parsing_err(full, after, "mismatched input expecting {UNQUOTED_ID_PATTERN, QUOTED_IDENTIFIER}"));
                }
                names.push(unquote_name(raw));
            }
            if w == "DROP" { Cmd::Drop(names, kw_pos) } else { Cmd::Keep(names, kw_pos) }
        }
        "RENAME" => {
            let mut pairs = Vec::new();
            for (a, b) in split_top(full, after, e, b',', Dialect::Esql) {
                let part = &full[a..b];
                let up = part.to_ascii_uppercase();
                let name = |t: &str, off: usize| -> Result<String> {
                    match t.trim().strip_prefix('?') {
                        Some(pn) => ident_param(full, off, pn, params, np),
                        None => Ok(unquote_name(t)),
                    }
                };
                if let Some(p) = find_kw(&up, "AS") {
                    let old = name(&part[..p], a)?;
                    pairs.push((old, name(&part[p + 2..], a + p + 2)?));
                } else if let Some((new, old)) = part.split_once('=') {
                    let new = name(new, a)?;
                    pairs.push((name(old, a)?, new));
                } else {
                    return Err(parsing_err(full, a, "mismatched input expecting 'as'"));
                }
            }
            Cmd::Rename(pairs, kw_pos)
        }
        "SORT" => {
            let mut p = p()?;
            let mut keys = Vec::new();
            loop {
                let expr = p.expr()?;
                let desc = if p.eat_kw("DESC") {
                    true
                } else {
                    p.eat_kw("ASC");
                    false
                };
                let nulls_first = if p.eat_kw("NULLS") {
                    if p.eat_kw("FIRST") {
                        Some(true)
                    } else if p.eat_kw("LAST") {
                        Some(false)
                    } else {
                        return Err(p.unexpected("{FIRST, LAST}"));
                    }
                } else {
                    None
                };
                keys.push(SortKey { expr, desc, nulls_first });
                if !p.eat(&Tok::Comma) {
                    break;
                }
            }
            finish(&p)?;
            Cmd::Sort(keys)
        }
        "LIMIT" => {
            let mut p = p()?;
            let text = full[after..e].trim().to_string();
            let bad = |v: &str, ty: &str| parsing_err(full, kw_pos, format!("value of [limit {text}] must be a non negative integer, found value [{v}] type [{ty}]"));
            let n = match p.peek().clone() {
                Tok::Int(n) if n >= 0 => {
                    p.i += 1;
                    n as usize
                }
                Tok::Op("-") if matches!(p.peek_at(1), Tok::Int(_)) => {
                    let Tok::Int(n) = p.peek_at(1).clone() else { unreachable!() };
                    return Err(bad(&format!("-{n}"), if n <= i32::MAX as i64 + 1 { "integer" } else { "long" }));
                }
                Tok::Param(x) => {
                    let v = p.param_value(&x)?;
                    match &v {
                        Value::Number(n) if n.as_u64().is_some() => n.as_u64().unwrap() as usize,
                        other => return Err(bad(&format!("?{x}"), json_to_val(other).type_name())),
                    }
                }
                _ => return Err(p.unexpected("INTEGER_LITERAL")),
            };
            if p.eat_kw("BY") {
                let mut keys = vec![p.expr()?];
                while p.eat(&Tok::Comma) {
                    keys.push(p.expr()?);
                }
                finish(&p)?;
                return Ok(Cmd::LimitBy(n, keys));
            }
            finish(&p)?;
            Cmd::Limit(n)
        }
        "TS_INFO" | "METRICS_INFO" => {
            let p = p()?;
            finish(&p)?;
            Cmd::Info(w == "TS_INFO", kw_pos)
        }
        "MV_EXPAND" => {
            let mut p = p()?;
            let f = p.qualified()?;
            finish(&p)?;
            Cmd::MvExpand(f, kw_pos)
        }
        "DISSECT" | "GROK" => {
            let mut p = p()?;
            let field = p.expr_bp(5)?;
            let pattern = p.string_lit()?;
            let mut sep = String::new();
            if w == "DISSECT" && p.at_kw("APPEND_SEPARATOR") {
                p.i += 1;
                if !matches!(p.peek(), Tok::Op("=")) {
                    return Err(p.unexpected("'='"));
                }
                p.i += 1;
                sep = p.string_lit()?;
            }
            finish(&p)?;
            if w == "DISSECT" {
                Cmd::Dissect { field, pattern, sep, pos: kw_pos }
            } else {
                // One attribute may be captured several times, but only with one type.
                let mut seen: HashMap<String, (String, String)> = HashMap::new();
                let mut rest = pattern.as_str();
                while let Some(st) = rest.find("%{") {
                    let Some(en) = rest[st..].find('}') else { break };
                    let spec = &rest[st + 2..st + en];
                    let mut parts = spec.splitn(3, ':');
                    let pat = parts.next().unwrap_or("").to_string();
                    if let Some(name) = parts.next() {
                        let ty = parts.next().unwrap_or("keyword").to_string();
                        if let Some(prev) = seen.get(name) {
                            if *prev != (pat.clone(), ty.clone()) {
                                return Err(parsing_err(full, kw_pos, format!("Invalid GROK pattern [{pattern}]: the attribute [{name}] is defined multiple times with different types")));
                            }
                        }
                        seen.insert(name.to_string(), (pat, ty));
                    }
                    rest = &rest[st + en + 1..];
                }
                Cmd::Grok { field, pattern, pos: kw_pos }
            }
        }
        "LOOKUP" => {
            let (w2, a2) = word_at(full, after, e);
            if !w2.eq_ignore_ascii_case("JOIN") {
                let mut p = p()?;
                return Err(p.unexpected("JOIN"));
            }
            // The index name is an unquoted source token (`test-lookup-1`, `idx*`).
            let bytes = full.as_bytes();
            let mut i = a2;
            while i < e && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let index_pos = i;
            while i < e && !bytes[i].is_ascii_whitespace() && bytes[i] != b'|' {
                i += 1;
            }
            let index = full[index_pos..i].trim_matches('"').to_string();
            if index.is_empty() {
                return Err(parsing_err(full, index_pos, "mismatched input '<EOF>' expecting {QUOTED_STRING, UNQUOTED_SOURCE}"));
            }
            if index.contains('*') {
                return Err(parsing_err(full, index_pos, format!("invalid index pattern [{index}], * is not allowed in LOOKUP JOIN")));
            }
            let mut p = Parser::new(full, i, e, Dialect::Esql, params, np)?;
            let on_kw = p.pos();
            if !p.eat_kw("ON") {
                return Err(p.unexpected("ON"));
            }
            // `visitJoinCondition`: fields, or one boolean expression.
            let mut conds: Vec<(Expr, usize, usize, usize)> = Vec::new(); // expr, offset, first token, end token
            loop {
                let (st, st_tok) = (p.pos(), p.i);
                let x = p.expr()?;
                conds.push((x, st, st_tok, p.i));
                if !p.eat(&Tok::Comma) {
                    break;
                }
            }
            finish(&p)?;
            let src = |c: &(Expr, usize, usize, usize)| p.text(c.1, c.3);
            if matches!(conds[0].0, Expr::Col(..) | Expr::Lit(_)) {
                let mut on = Vec::new();
                let mut on_pos = Vec::new();
                for c in &conds {
                    let Expr::Col(n, _) = &c.0 else {
                        return Err(parsing_err(full, c.1, format!("JOIN ON clause must be a comma separated list of fields or a single expression, found [{}]", src(c))));
                    };
                    if on.contains(n) {
                        return Err(parsing_err(full, c.1, format!("JOIN ON clause does not support multiple fields with the same name, found multiple instances of [{n}]")));
                    }
                    on.push(n.clone());
                    on_pos.push(c.1);
                }
                Cmd::LookupJoin { index, index_pos, on, on_pos, pos: kw_pos, cond: None, fields: None }
            } else {
                if conds.len() != 1 {
                    let all: Vec<String> = conds.iter().map(src).collect();
                    return Err(parsing_err(full, on_kw, format!("JOIN ON clause with expressions only supports a single expression, found [[{}]]", all.join(", "))));
                }
                let (x, _, t0, t1) = conds.pop().unwrap();
                // Conjuncts with their source text (split at top-level AND when there is no top-level OR).
                let mut parts: Vec<(Expr, String)> = Vec::new();
                fn split_and(e: Expr, out: &mut Vec<Expr>) {
                    match e {
                        Expr::Bin(BinOp::And, a, b) => {
                            split_and(*a, out);
                            split_and(*b, out);
                        }
                        o => out.push(o),
                    }
                }
                let mut xs = Vec::new();
                split_and(x, &mut xs);
                let mut texts: Vec<String> = Vec::new();
                {
                    let (mut depth, mut start, mut has_or) = (0i32, p.toks[t0].start, false);
                    let mut cuts: Vec<(usize, usize)> = Vec::new();
                    for k in t0..t1 {
                        match &p.toks[k].tok {
                            Tok::LParen => depth += 1,
                            Tok::RParen => depth -= 1,
                            Tok::Ident(w) if depth == 0 && w.eq_ignore_ascii_case("OR") => has_or = true,
                            Tok::Ident(w) if depth == 0 && w.eq_ignore_ascii_case("AND") => {
                                cuts.push((start, p.toks[k].start));
                                start = p.toks[k].end;
                            }
                            _ => {}
                        }
                    }
                    cuts.push((start, p.toks[t1 - 1].end));
                    if !has_or && cuts.len() == xs.len() {
                        texts = cuts.iter().map(|(a, b)| full[*a..*b].trim().to_string()).collect();
                    }
                }
                for (k, c) in xs.into_iter().enumerate() {
                    let t = texts.get(k).cloned().unwrap_or_else(|| p.text(p.toks[t0].start, t1));
                    let c = match c {
                        Expr::Not(inner) => match *inner {
                            Expr::Bin(BinOp::Eq, a, b) => Expr::Bin(BinOp::Neq, a, b),
                            o => Expr::Not(Box::new(o)),
                        },
                        o => o,
                    };
                    fn bare(e: &Expr) -> bool {
                        match e {
                            Expr::Col(..) => true,
                            Expr::Bin(BinOp::And | BinOp::Or, a, b) => bare(a) || bare(b),
                            _ => false,
                        }
                    }
                    if bare(&c) {
                        return Err(parsing_err(full, on_kw, format!("JOIN ON clause only supports fields or AND of Binary Expressions at the moment, found [{t}]")));
                    }
                    parts.push((c, t));
                }
                let relating = parts.iter().any(|(c, _)| matches!(c, Expr::Bin(BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge, a, b) if matches!(**a, Expr::Col(..)) && matches!(**b, Expr::Col(..))));
                if !relating {
                    return Err(parsing_err(full, on_kw, "JOIN ON clause with expressions must contain at least one condition relating the left index and the lookup index"));
                }
                let cond = parts.into_iter().map(|(c, _)| c).reduce(|a, b| Expr::Bin(BinOp::And, Box::new(a), Box::new(b)));
                Cmd::LookupJoin { index, index_pos, on: vec![], on_pos: vec![], pos: kw_pos, cond, fields: None }
            }
        }
        "ENRICH" => {
            // Policy name: ENRICH_POLICY_NAME, optionally `mode:` qualified.
            let bytes = full.as_bytes();
            let mut i = after;
            while i < e && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let start = i;
            let raw = if i < e && bytes[i] == b'"' {
                // Quoted policy name.
                let close = full[i + 1..e].find('"').map(|k| i + 1 + k).ok_or_else(|| parsing_err(full, start, "missing ENRICH_POLICY_NAME at '<EOF>'"))?;
                i = close + 1;
                full[start + 1..close].to_string()
            } else {
                while i < e && !bytes[i].is_ascii_whitespace() && !matches!(bytes[i], b'|' | b',' | b'"' | b'<' | b'>' | b'?' | b'/' | b'\\' | b'#') {
                    i += 1;
                }
                full[start..i].to_string()
            };
            if raw.is_empty() {
                return Err(parsing_err(full, start, "missing ENRICH_POLICY_NAME at '<EOF>'"));
            }
            let (mode, policy) = match raw.split_once(':') {
                Some((m, pn)) => {
                    let mu = m.to_ascii_uppercase();
                    if !matches!(mu.as_str(), "_ANY" | "_COORDINATOR" | "_REMOTE") {
                        return Err(parsing_err(full, start, format!("Unrecognized value [{m}], ENRICH policy qualifier needs to be one of [_ANY, _COORDINATOR, _REMOTE]")));
                    }
                    (Some(m.to_ascii_lowercase()), pn.to_string())
                }
                None => (None, raw.clone()),
            };
            let mut p = Parser::new(full, i, e, Dialect::Esql, params, np)?;
            let mut on = None;
            if p.eat_kw("ON") {
                on = Some(p.qualified()?);
            }
            let mut with = Vec::new();
            if p.eat_kw("WITH") {
                loop {
                    let first = p.qualified()?;
                    if matches!(p.peek(), Tok::Op("=")) {
                        p.i += 1;
                        let src = p.qualified()?;
                        with.push((first, src));
                    } else {
                        with.push((first.clone(), first));
                    }
                    if !p.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            finish(&p)?;
            let _ = kw_pos;
            Cmd::Enrich { policy, mode, on, with, pos: start }
        }
        "SAMPLE" => {
            let mut p = p()?;
            let invalid = |v: &str| parsing_err(full, kw_pos, format!("invalid value for SAMPLE probability [{v}], expecting a number between 0 and 1, exclusive"));
            let v = match p.peek().clone() {
                Tok::Dec(d) => {
                    p.i += 1;
                    d
                }
                Tok::Int(i) => {
                    p.i += 1;
                    i as f64
                }
                Tok::Param(x) => match p.param_value(&x)? {
                    Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                    Value::String(s) => return Err(invalid(&s)),
                    other => return Err(invalid(&other.to_string())),
                },
                _ => return Err(p.unexpected("DECIMAL_LITERAL")),
            };
            finish(&p)?;
            if !(v > 0.0 && v < 1.0) {
                return Err(invalid(&super::Val::Double(v).to_text().trim_end_matches(".0").to_string()));
            }
            Cmd::Sample(v)
        }
        "FORK" => {
            let mut branches = Vec::new();
            let b = full.as_bytes();
            let mut i = after;
            while i < e {
                if b[i].is_ascii_whitespace() {
                    i += 1;
                    continue;
                }
                if b[i] != b'(' {
                    return Err(parsing_err(full, i, "mismatched input expecting '('"));
                }
                let ranges = split_top(full, i, e, b')', Dialect::Esql);
                let _ = ranges;
                // find matching paren
                let mut depth = 0;
                let mut j = i;
                let mut in_str = false;
                while j < e {
                    let c = b[j];
                    if c == b'"' {
                        in_str = !in_str;
                    } else if !in_str && c == b'(' {
                        depth += 1;
                    } else if !in_str && c == b')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    j += 1;
                }
                if j >= e {
                    return Err(parsing_err(full, i, "unbalanced parentheses in FORK"));
                }
                branches.push(parse_pipeline(full, i + 1, j, params, np, false)?);
                i = j + 1;
            }
            if branches.is_empty() {
                return Err(parsing_err(full, after, "mismatched input '<EOF>' expecting '('"));
            }
            Cmd::Fork(branches)
        }
        "CHANGE_POINT" | "FUSE" | "RERANK" | "COMPLETION" => {
            return Err(EsError::new(400, "verification_exception", format!(
                "Found 1 problem\n{}: [{w}] is not supported by this node (requires the ML / inference features)",
                super::pos_prefix(full, kw_pos)
            )));
        }
        _ => {
            return Err(parsing_err(
                full,
                kw_pos,
                format!(
                    "mismatched input '{word}' expecting {{'dissect', 'drop', 'enrich', 'eval', 'grok', 'keep', 'limit', 'mv_expand', 'rename', 'sort', 'stats', 'where', 'lookup'}}"
                ),
            ))
        }
    })
}

/// Offset of the parenthesis closing the one at `open` (strings skipped).
fn matching_paren(full: &str, open: usize, end: usize) -> Option<usize> {
    let b = full.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < end {
        match b[i] {
            b'"' => {
                if full[i..].starts_with("\"\"\"") {
                    i = full[i + 3..end].find("\"\"\"").map(|p| i + 3 + p + 2)?;
                } else {
                    i += 1;
                    while i < end && b[i] != b'"' {
                        if b[i] == b'\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                }
            }
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The name an identifier/pattern parameter (`?n`, `?`, `?1`) stands for;
/// a null or empty-list parameter is the empty name.
fn ident_param(full: &str, at: usize, pn: &str, params: &Params, np: &std::cell::Cell<usize>) -> Result<String> {
    let v = if pn.is_empty() {
        let k = np.get();
        np.set(k + 1);
        params.positional.get(k).cloned()
    } else if let Ok(n) = pn.parse::<usize>() {
        params.positional.get(n.wrapping_sub(1)).cloned()
    } else {
        params.named.get(pn).cloned()
    };
    let v = v.ok_or_else(|| parsing_err(full, at, format!("Unknown query parameter [{pn}]")))?;
    Ok(match &v {
        Value::Object(m) => match m.get("identifier").or_else(|| m.get("pattern")).or_else(|| m.get("value")) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            Some(Value::Array(a)) if a.is_empty() => String::new(),
            Some(o) => o.to_string(),
        },
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Array(a) if a.is_empty() => String::new(),
        other => other.to_string(),
    })
}

/// Position of a whole-word keyword in an upper-cased string (outside quotes).
pub fn find_kw(upper: &str, kw: &str) -> Option<usize> {
    let b = upper.as_bytes();
    let mut i = 0;
    let mut quote: Option<u8> = None;
    while i + kw.len() <= b.len() {
        let c = b[i];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'`' || c == b'\'' {
            quote = Some(c);
            i += 1;
            continue;
        }
        if upper[i..].starts_with(kw) {
            let before_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
            let after_ok = i + kw.len() >= b.len() || !(b[i + kw.len()].is_ascii_alphanumeric() || b[i + kw.len()] == b'_');
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}
