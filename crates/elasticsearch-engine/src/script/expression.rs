//! `lang: expression` (ES `lang-expression`: Lucene JavaScript
//! expressions).
//!
//! An expression is a double-valued formula over numbers, `_score`,
//! `_value`, numeric `params` (bare names) and doc values
//! (`doc['f']`, `doc['f'].min()`, `doc['geo'].lat`,
//! `doc['date'].getYear()`, `doc['date'].date.dayOfWeek`, ...) with
//! Lucene's default functions (`abs`, `ln`, `haversin`, `max`, ...).
//!
//! It compiles into an ordinary [`Program`] (so every script context that
//! runs Painless runs expressions too) whose body is a tree of unboxed
//! `f64` closures. Variables are linked against the mapping and `params`
//! when an executor first evaluates (ES "link error"s); doc values are
//! read straight from the bound columns.

use super::doc::Kind;
use super::error::{SResult, ScriptErr};
use super::runtime::{Flow, Program, Rt};
use super::value::{Key, Val};
use super::{RtType, ScriptContext};
use crate::error::EsError;
use serde_json::json;
use std::sync::Arc;

pub const LANG: &str = "expression";

/// Script contexts the expression engine handles (ES `ExpressionScriptEngine`).
pub fn supports(ctx: ScriptContext) -> bool {
    matches!(
        ctx,
        ScriptContext::Score | ScriptContext::Filter | ScriptContext::Field | ScriptContext::NumberSort | ScriptContext::Aggs | ScriptContext::BucketScript | ScriptContext::BucketSelector
    )
}

// ---------------------------------------------------------------------------
// Errors.

/// A `java.text.ParseException` (`parse_exception`) at `offset`.
fn parse_err(msg: impl Into<String>, offset: usize) -> Box<ScriptErr> {
    let mut e = ScriptErr::new("parse_exception", msg);
    e.pos = Some(offset);
    e
}

fn iae(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("illegal_argument_exception", msg)
}

/// Marks a [`ScriptErr`] as a link failure of variable `var` (rendered by
/// [`to_es_error`]).
fn link_err(var: &str, mut e: Box<ScriptErr>) -> Box<ScriptErr> {
    e.cause = Some(Box::new(("\u{0}link".to_string(), var.to_string())));
    e
}

/// ES `ExpressionScriptEngine.convertToScriptException`: `compile error` /
/// `link error` with the offending portion and a pointer at the
/// `ParseException` offset.
pub fn to_es_error(e: &ScriptErr, source: &str) -> EsError {
    let (reason, portion) = match e.cause.as_deref() {
        Some((m, var)) if m == "\u{0}link" => ("link error", var.as_str()),
        _ if e.compile => ("compile error", source),
        _ => ("runtime error", source),
    };
    let pointer = format!("{}^---- HERE", " ".repeat(if e.kind == "parse_exception" { e.pos.unwrap_or(0) } else { 0 }));
    let err = EsError::new(400, "script_exception", reason)
        .with("script_stack", json!([portion, pointer]))
        .with("script", source.to_string())
        .with("lang", LANG);
    err.caused_by(EsError::new(400, &e.kind, e.msg.clone()))
}

// ---------------------------------------------------------------------------
// Parsing.

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    /// A variable reference (text as written, start offset).
    Var(String),
    Op(&'static str),
    End,
}

struct Lexer<'a> {
    src: &'a str,
    b: &'a [u8],
    i: usize,
}

const OPS: [&str; 26] = [">>>", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "(", ")", ",", "!", "~", "*", "/", "%", "+", "-", "<", ">", "&", "^", "|", "?", ":"];

impl<'a> Lexer<'a> {
    fn err(&self, at: usize) -> Box<ScriptErr> {
        let c = self.src[at..].chars().next().map(|c| c.to_string()).unwrap_or_default();
        parse_err(format!("Invalid expression '{}': unexpected character '{c}' on line (1) position ({at})", self.src), at)
    }

    fn next(&mut self) -> SResult<(Tok, usize)> {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
        let start = self.i;
        if self.i >= self.b.len() {
            return Ok((Tok::End, start));
        }
        let c = self.b[self.i];
        if c.is_ascii_digit() || (c == b'.' && self.b.get(self.i + 1).is_some_and(u8::is_ascii_digit)) {
            return self.number().map(|t| (t, start));
        }
        if c == b'_' || c == b'$' || c.is_ascii_alphabetic() {
            return self.variable().map(|t| (t, start));
        }
        for op in OPS {
            if self.src[self.i..].starts_with(op) {
                self.i += op.len();
                return Ok((Tok::Op(op), start));
            }
        }
        Err(self.err(start))
    }

    fn number(&mut self) -> SResult<Tok> {
        let s = self.i;
        let b = self.b;
        if b[s] == b'0' && matches!(b.get(s + 1), Some(b'x' | b'X')) && b.get(s + 2).is_some_and(u8::is_ascii_hexdigit) {
            let mut j = s + 2;
            while j < b.len() && b[j].is_ascii_hexdigit() {
                j += 1;
            }
            self.i = j;
            let v = i64::from_str_radix(&self.src[s + 2..j], 16).map_err(|_| parse_err(format!("Invalid expression '{}': For input string: \"{}\"", self.src, &self.src[s + 2..j]), s))?;
            return Ok(Tok::Num(v as f64));
        }
        if b[s] == b'0' && b.get(s + 1).is_some_and(|c| (b'0'..=b'7').contains(c)) {
            let mut j = s + 1;
            while j < b.len() && (b'0'..=b'7').contains(&b[j]) {
                j += 1;
            }
            self.i = j;
            let v = i64::from_str_radix(&self.src[s + 1..j], 8).map_err(|_| parse_err(format!("Invalid expression '{}': For input string: \"{}\"", self.src, &self.src[s + 1..j]), s))?;
            return Ok(Tok::Num(v as f64));
        }
        let mut j = s;
        if b[j] == b'.' {
            j += 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
        } else {
            // INTEGER: `0` or `[1-9][0-9]*`.
            if b[j] == b'0' {
                j += 1;
            } else {
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
            }
            if j < b.len() && b[j] == b'.' {
                j += 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
            }
        }
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                k += 1;
            }
            if k < b.len() && b[k].is_ascii_digit() {
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                j = k;
            }
        }
        self.i = j;
        let text = &self.src[s..j];
        text.parse::<f64>().map(Tok::Num).map_err(|_| self.err(s))
    }

    fn ident(&mut self) -> bool {
        let b = self.b;
        if self.i < b.len() && (b[self.i] == b'_' || b[self.i] == b'$' || b[self.i].is_ascii_alphabetic()) {
            self.i += 1;
            while self.i < b.len() && (b[self.i] == b'_' || b[self.i] == b'$' || b[self.i].is_ascii_alphanumeric()) {
                self.i += 1;
            }
            true
        } else {
            false
        }
    }

    /// `[ 'str' | "str" | INTEGER ]`, or false (position unchanged).
    fn array(&mut self) -> SResult<bool> {
        let b = self.b;
        let s = self.i;
        if s >= b.len() || b[s] != b'[' {
            return Ok(false);
        }
        let mut j = s + 1;
        match b.get(j) {
            Some(&q) if q == b'\'' || q == b'"' => {
                j += 1;
                loop {
                    match b.get(j) {
                        None => return Err(self.err(s)),
                        Some(b'\\') => j += 2,
                        Some(&c) if c == q => {
                            j += 1;
                            break;
                        }
                        Some(_) => j += 1,
                    }
                }
            }
            Some(b'0') => j += 1,
            Some(c) if c.is_ascii_digit() => {
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
            }
            _ => return Ok(false),
        }
        if b.get(j) != Some(&b']') {
            return Ok(false);
        }
        self.i = j + 1;
        Ok(true)
    }

    fn variable(&mut self) -> SResult<Tok> {
        let s = self.i;
        self.ident();
        while self.array()? {}
        loop {
            let save = self.i;
            if self.i < self.b.len() && self.b[self.i] == b'.' {
                self.i += 1;
                if self.ident() {
                    while self.array()? {}
                    continue;
                }
            }
            self.i = save;
            break;
        }
        Ok(Tok::Var(self.src[s..self.i].to_string()))
    }
}

/// Lucene `JavascriptCompiler.normalizeQuotes`: double-quoted index
/// strings become single-quoted.
fn normalize_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '[' && cs.get(i + 1) == Some(&'"') {
            out.push_str("['");
            i += 2;
            while i < cs.len() && cs[i] != '"' {
                if cs[i] == '\\' && i + 1 < cs.len() {
                    if cs[i + 1] == '"' {
                        out.push('"');
                    } else {
                        out.push('\\');
                        out.push(cs[i + 1]);
                    }
                    i += 2;
                    continue;
                }
                if cs[i] == '\'' {
                    out.push_str("\\'");
                } else {
                    out.push(cs[i]);
                }
                i += 1;
            }
            out.push('\'');
            i += 1;
        } else {
            out.push(cs[i]);
            i += 1;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Func {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Floor,
    Haversin,
    Ln,
    Log10,
    Logn,
    Max,
    Min,
    Pow,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
}

impl Func {
    fn of(name: &str) -> Option<(Func, usize)> {
        Some(match name {
            "abs" => (Func::Abs, 1),
            "acos" => (Func::Acos, 1),
            "acosh" => (Func::Acosh, 1),
            "asin" => (Func::Asin, 1),
            "asinh" => (Func::Asinh, 1),
            "atan" => (Func::Atan, 1),
            "atan2" => (Func::Atan2, 2),
            "atanh" => (Func::Atanh, 1),
            "ceil" => (Func::Ceil, 1),
            "cos" => (Func::Cos, 1),
            "cosh" => (Func::Cosh, 1),
            "exp" => (Func::Exp, 1),
            "floor" => (Func::Floor, 1),
            "haversin" => (Func::Haversin, 4),
            "ln" => (Func::Ln, 1),
            "log10" => (Func::Log10, 1),
            "logn" => (Func::Logn, 2),
            "max" => (Func::Max, 2),
            "min" => (Func::Min, 2),
            "pow" => (Func::Pow, 2),
            "sin" => (Func::Sin, 1),
            "sinh" => (Func::Sinh, 1),
            "sqrt" => (Func::Sqrt, 1),
            "tan" => (Func::Tan, 1),
            "tanh" => (Func::Tanh, 1),
            _ => return None,
        })
    }

    fn unary(self) -> fn(f64) -> f64 {
        match self {
            Func::Abs => f64::abs,
            Func::Acos => f64::acos,
            // Lucene `MathUtil.acosh` / `asinh` / `atanh`.
            Func::Acosh => |a| ((a * a - 1.0).sqrt() + a).ln(),
            Func::Asinh => |a| {
                let (a, sign) = if a.to_bits() >> 63 == 1 { (a.abs(), -1.0) } else { (a, 1.0) };
                sign * ((a * a + 1.0).sqrt() + a).ln()
            },
            Func::Atanh => |a| {
                let (a, mult) = if a.to_bits() >> 63 == 1 { (a.abs(), -0.5) } else { (a, 0.5) };
                mult * ((1.0 + a) / (1.0 - a)).ln()
            },
            Func::Asin => f64::asin,
            Func::Atan => f64::atan,
            Func::Ceil => f64::ceil,
            Func::Cos => f64::cos,
            Func::Cosh => f64::cosh,
            Func::Exp => f64::exp,
            Func::Floor => f64::floor,
            Func::Ln => f64::ln,
            Func::Log10 => f64::log10,
            Func::Sin => f64::sin,
            Func::Sinh => f64::sinh,
            Func::Sqrt => f64::sqrt,
            Func::Tan => f64::tan,
            Func::Tanh => f64::tanh,
            _ => |a| a,
        }
    }

    fn binary(self) -> fn(f64, f64) -> f64 {
        match self {
            Func::Atan2 => f64::atan2,
            // `MathUtil.log(base, x)`.
            Func::Logn => |base, x| x.ln() / base.ln(),
            // `Math.max` / `Math.min`: NaN wins.
            Func::Max => |x, y| if x.is_nan() || y.is_nan() { f64::NAN } else { x.max(y) },
            Func::Min => |x, y| if x.is_nan() || y.is_nan() { f64::NAN } else { x.min(y) },
            Func::Pow => f64::powf,
            _ => |a, _| a,
        }
    }
}

/// `SloppyMath.haversinKilometers`.
fn haversin_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let h = crate::sloppy_math::haversin_sort_key(lat1, lon1, lat2, lon2);
    6_371.008_771_4 * 2.0 * crate::sloppy_math::asin(1f64.min((h * 0.5).sqrt()))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum UOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Shl,
    Shr,
    Ushr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    And,
    Or,
}

#[derive(Clone, Debug)]
enum Node {
    Num(f64),
    Var(usize),
    Call(Func, Vec<Node>),
    Unary(UOp, Box<Node>),
    Bin(BOp, Box<Node>, Box<Node>),
    Cond(Box<Node>, Box<Node>, Box<Node>),
}

struct Parser<'a> {
    lex: Lexer<'a>,
    tok: Tok,
    at: usize,
    vars: Vec<String>,
    nest: crate::parse_depth::Nesting,
}

fn binop(op: &str) -> Option<(BOp, u8)> {
    Some(match op {
        "||" => (BOp::Or, 1),
        "&&" => (BOp::And, 2),
        "|" => (BOp::BitOr, 3),
        "^" => (BOp::BitXor, 4),
        "&" => (BOp::BitAnd, 5),
        "==" => (BOp::Eq, 6),
        "!=" => (BOp::Ne, 6),
        "<" => (BOp::Lt, 7),
        "<=" => (BOp::Le, 7),
        ">" => (BOp::Gt, 7),
        ">=" => (BOp::Ge, 7),
        "<<" => (BOp::Shl, 8),
        ">>" => (BOp::Shr, 8),
        ">>>" => (BOp::Ushr, 8),
        "+" => (BOp::Add, 9),
        "-" => (BOp::Sub, 9),
        "*" => (BOp::Mul, 10),
        "/" => (BOp::Div, 10),
        "%" => (BOp::Rem, 10),
        _ => return None,
    })
}

impl<'a> Parser<'a> {
    fn bump(&mut self) -> SResult<()> {
        let (t, at) = self.lex.next()?;
        self.tok = t;
        self.at = at;
        Ok(())
    }

    fn unexpected(&self) -> Box<ScriptErr> {
        let src = self.lex.src;
        let what = match &self.tok {
            Tok::End => "<EOF>".to_string(),
            Tok::Num(_) | Tok::Var(_) | Tok::Op(_) => src[self.at..self.lex.i].to_string(),
        };
        parse_err(format!("Invalid expression '{src}': unexpected token '{what}' on line (1) position ({})", self.at), self.at)
    }

    fn expect(&mut self, op: &'static str) -> SResult<()> {
        if self.tok != Tok::Op(op) {
            return Err(self.unexpected());
        }
        self.bump()
    }

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> SResult<T>) -> SResult<T> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            return Err(parse_err(format!("Invalid expression '{}': nested more than {} levels deep", self.lex.src, self.nest.max()), self.at));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    fn expr(&mut self) -> SResult<Node> {
        self.nested(true, Self::expr_unnested)
    }

    fn expr_unnested(&mut self) -> SResult<Node> {
        let c = self.binary(1)?;
        if self.tok == Tok::Op("?") {
            self.bump()?;
            let a = self.expr()?;
            self.expect(":")?;
            let b = self.expr()?;
            return Ok(Node::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn binary(&mut self, min: u8) -> SResult<Node> {
        let mut lhs = self.unary()?;
        loop {
            let Tok::Op(op) = self.tok else { break };
            let Some((bop, prec)) = binop(op) else { break };
            if prec < min {
                break;
            }
            if !self.nest.grow() {
                return Err(parse_err(format!("Invalid expression '{}': nested more than {} levels deep", self.lex.src, self.nest.max()), self.at));
            }
            self.bump()?;
            let rhs = self.binary(prec + 1)?;
            lhs = Node::Bin(bop, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> SResult<Node> {
        self.nested(false, Self::unary_unnested)
    }

    fn unary_unnested(&mut self) -> SResult<Node> {
        let op = match self.tok {
            Tok::Op("+") => None,
            Tok::Op("-") => Some(UOp::Neg),
            Tok::Op("!") => Some(UOp::Not),
            Tok::Op("~") => Some(UOp::BitNot),
            _ => return self.primary(),
        };
        self.bump()?;
        let e = self.unary()?;
        Ok(match op {
            None => e,
            Some(op) => Node::Unary(op, Box::new(e)),
        })
    }

    fn primary(&mut self) -> SResult<Node> {
        match self.tok.clone() {
            Tok::Num(v) => {
                self.bump()?;
                Ok(Node::Num(v))
            }
            Tok::Op("(") => {
                self.bump()?;
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            Tok::Var(text) => {
                let start = self.at;
                self.bump()?;
                if self.tok != Tok::Op("(") {
                    return Ok(Node::Var(self.var(&text)));
                }
                self.bump()?;
                let mut args = Vec::new();
                if self.tok != Tok::Op(")") {
                    args.push(self.expr()?);
                    while self.tok == Tok::Op(",") {
                        self.bump()?;
                        args.push(self.expr()?);
                    }
                }
                self.expect(")")?;
                let src = self.lex.src;
                if let Some((f, arity)) = Func::of(&text) {
                    if args.len() != arity {
                        return Err(parse_err(format!("Invalid expression '{src}': Expected ({arity}) arguments for function call ({text}), but found ({}).", args.len()), start));
                    }
                    return Ok(Node::Call(f, args));
                }
                if args.is_empty() && text.contains('.') {
                    return Ok(Node::Var(self.var(&format!("{text}()"))));
                }
                Err(parse_err(format!("Invalid expression '{src}': Unrecognized function call ({text})."), start))
            }
            _ => Err(self.unexpected()),
        }
    }

    fn var(&mut self, text: &str) -> usize {
        let t = normalize_quotes(text);
        match self.vars.iter().position(|v| *v == t) {
            Some(i) => i,
            None => {
                self.vars.push(t);
                self.vars.len() - 1
            }
        }
    }
}

fn parse(src: &str) -> SResult<(Node, Vec<String>)> {
    let mut p = Parser { lex: Lexer { src, b: src.as_bytes(), i: 0 }, tok: Tok::End, at: 0, vars: Vec::new(), nest: crate::parse_depth::Nesting::new(512) };
    p.bump()?;
    let e = p.expr()?;
    if p.tok != Tok::End {
        return Err(p.unexpected());
    }
    Ok((e, p.vars))
}

/// Recognize only the bounded geo/numeric formula; preserve the parsed
/// operation order and reject arbitrary score dependencies or parameters.
pub(super) fn compound_geo_fields(source: &str) -> Option<(String, String)> {
    let (root, vars) = parse(source).ok()?;
    let body = match &root {
        Node::Bin(BOp::Mul, body, score) if matches!(score.as_ref(), Node::Var(i) if vars[*i] == "_score") => body.as_ref(),
        _ => &root,
    };
    fn call(node: &Node, f: Func) -> Option<&Node> {
        match node { Node::Call(kind, args) if *kind == f && args.len() == 1 => Some(&args[0]), _ => None }
    }
    let Node::Bin(BOp::Add, prefix, lat) = call(body, Func::Abs)? else { return None };
    let Node::Bin(BOp::Add, log, lon) = prefix.as_ref() else { return None };
    let Node::Bin(BOp::Add, absolute, one) = call(log, Func::Ln)? else { return None };
    if !matches!(one.as_ref(), Node::Num(n) if *n == 1.0) { return None; }
    let Node::Var(pop) = call(absolute, Func::Abs)? else { return None };
    let numeric = match var_parts(&vars[*pop]).as_slice() {
        [Part::Member(doc), Part::StrIndex(field)] if doc == "doc" => field.clone(),
        _ => return None,
    };
    let geo = |n: &Node, coordinate: &str| -> Option<String> {
        let Node::Var(i) = n else { return None };
        match var_parts(&vars[*i]).as_slice() {
            [Part::Member(doc), Part::StrIndex(field), Part::Member(member)] if doc == "doc" && member == coordinate => Some(field.clone()),
            _ => None,
        }
    };
    let field = geo(lon, "lon")?;
    if geo(lat, "lat")? != field { return None; }
    Some((numeric, field))
}

// ---------------------------------------------------------------------------
// Variables.

/// Lucene `VariableContext` part.
#[derive(Clone, Debug, PartialEq)]
enum Part {
    Member(String),
    Method(String),
    StrIndex(String),
    IntIndex(i64),
}

fn var_parts(v: &str) -> Vec<Part> {
    let cs: Vec<char> = v.chars().collect();
    let mut out = Vec::new();
    let member = |i: usize, out: &mut Vec<Part>| -> usize {
        let mut j = i + 1;
        while j < cs.len() && cs[j] != '[' && cs[j] != '.' && cs[j] != '(' {
            j += 1;
        }
        let name: String = cs[i..j.min(cs.len())].iter().collect();
        if j + 1 < cs.len() && cs[j] == '(' && cs[j + 1] == ')' {
            out.push(Part::Method(name));
            j + 2
        } else {
            out.push(Part::Member(name));
            j
        }
    };
    let mut i = member(0, &mut out);
    while i < cs.len() {
        if cs[i] == '[' {
            i += 1;
            if cs.get(i) == Some(&'\'') {
                i += 1;
                let mut s = String::new();
                while i < cs.len() && cs[i] != '\'' {
                    if cs[i] == '\\' && i + 1 < cs.len() {
                        i += 1;
                    }
                    s.push(cs[i]);
                    i += 1;
                }
                out.push(Part::StrIndex(s));
                i += 1; // closing quote
            } else {
                let mut j = i;
                while j < cs.len() && cs[j] != ']' {
                    j += 1;
                }
                out.push(Part::IntIndex(cs[i..j].iter().collect::<String>().parse().unwrap_or(0)));
                i = j;
            }
            i += 1; // closing bracket
        } else {
            i = member(i + 1, &mut out);
        }
    }
    out
}

/// What a doc variable reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Read {
    /// Numeric / date doc values reduced with a `MultiValueMode`.
    Min,
    Max,
    Avg,
    Median,
    Sum,
    Count,
    /// 1 when the doc has no value, else 0.
    Empty,
    GeoLat,
    GeoLon,
    GeoEmpty,
    /// `java.util.Calendar` field of the (min) date in UTC.
    Cal(CalField),
    /// Joda `ReadableDateTime` property of the (min) date in UTC.
    Joda(JodaField),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalField {
    Year,
    Month,
    DayOfMonth,
    HourOfDay,
    Minute,
    Second,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JodaField {
    CenturyOfEra,
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    Era,
    HourOfDay,
    MillisOfDay,
    MillisOfSecond,
    MinuteOfDay,
    MinuteOfHour,
    MonthOfYear,
    SecondOfDay,
    SecondOfMinute,
    WeekOfWeekyear,
    Weekyear,
    Year,
    YearOfCentury,
    YearOfEra,
}

fn joda_field(name: &str) -> Option<JodaField> {
    let n = name.strip_prefix("get").map(|r| {
        let mut c = r.chars();
        c.next().map(|f| f.to_ascii_lowercase().to_string() + c.as_str()).unwrap_or_default()
    });
    let key = n.as_deref().unwrap_or(name);
    Some(match key {
        "centuryOfEra" => JodaField::CenturyOfEra,
        "dayOfMonth" => JodaField::DayOfMonth,
        "dayOfWeek" => JodaField::DayOfWeek,
        "dayOfYear" => JodaField::DayOfYear,
        "era" => JodaField::Era,
        "hourOfDay" => JodaField::HourOfDay,
        "millisOfDay" => JodaField::MillisOfDay,
        "millisOfSecond" => JodaField::MillisOfSecond,
        "minuteOfDay" => JodaField::MinuteOfDay,
        "minuteOfHour" => JodaField::MinuteOfHour,
        "monthOfYear" => JodaField::MonthOfYear,
        "secondOfDay" => JodaField::SecondOfDay,
        "secondOfMinute" => JodaField::SecondOfMinute,
        "weekOfWeekyear" => JodaField::WeekOfWeekyear,
        "weekyear" => JodaField::Weekyear,
        "year" => JodaField::Year,
        "yearOfCentury" => JodaField::YearOfCentury,
        "yearOfEra" => JodaField::YearOfEra,
        _ => return None,
    })
}

/// A variable as written: everything that does not depend on the mapping
/// or `params`.
#[derive(Clone, Debug)]
pub struct VarSpec {
    pub text: String,
    /// `doc['field']...`: the field and its doc slot.
    doc: Option<(String, usize)>,
    /// The member / method accessed (`None`: `.value`), and whether through
    /// `.date` / `.getDate()`.
    member: Option<Part>,
    date_obj: bool,
    /// The shape is invalid whatever the mapping (deferred to link time).
    shape_err: Option<(String, Option<usize>)>,
    /// Slot in `param_names` for a bare-name parameter.
    param: usize,
}

/// How an executor reads one variable (decided at link time).
#[derive(Clone, Copy, Debug)]
pub enum Bind {
    Score,
    Value,
    Const(f64),
    Doc { slot: usize, read: Read },
}

/// Expression-specific program data ([`Program::expression`]).
#[derive(Debug)]
pub struct ExprMeta {
    pub vars: Vec<VarSpec>,
}

// ---------------------------------------------------------------------------
// Linking.

fn param_number(rt: &Rt, name: &str) -> Option<Option<f64>> {
    match &rt.params {
        Val::Map(m) => m.borrow().get(&Key::str(name)).map(|v| match v {
            Val::Int(_) | Val::Long(_) | Val::Double(_) => v.as_f64(),
            _ => None,
        }),
        _ => None,
    }
}

/// The field kind as ES's expression bindings see it.
#[derive(PartialEq)]
enum FieldClass {
    Numeric,
    Date,
    Geo,
    Other,
}

fn field_class(rt: &Rt, slot: usize) -> Option<FieldClass> {
    let f = &rt.doc.fields[slot];
    Some(match f.kind {
        Kind::Unmapped => return None,
        Kind::Long | Kind::Unsigned | Kind::Double | Kind::Bool => FieldClass::Numeric,
        Kind::Date => FieldClass::Date,
        Kind::GeoPoint => FieldClass::Geo,
        Kind::Runtime(i) => match rt.doc.runtime.as_ref().map(|r| r.fields[i].ty) {
            Some(RtType::Long | RtType::Double | RtType::Boolean) => FieldClass::Numeric,
            Some(RtType::Date) => FieldClass::Date,
            _ => FieldClass::Other,
        },
        _ => FieldClass::Other,
    })
}

fn numeric_read(field: &str, member: &Option<Part>, class: &str) -> SResult<Read> {
    Ok(match member {
        None => Read::Min,
        Some(Part::Member(m)) => match m.as_str() {
            "value" => Read::Min,
            "empty" => Read::Empty,
            "length" => Read::Count,
            _ => return Err(iae(format!("Member variable [{m}] does not exist for {class} field [{field}]."))),
        },
        Some(Part::Method(m)) => match m.as_str() {
            "getValue" | "min" => Read::Min,
            "isEmpty" => Read::Empty,
            "size" | "count" => Read::Count,
            "max" => Read::Max,
            "avg" => Read::Avg,
            "median" => Read::Median,
            "sum" => Read::Sum,
            "getYear" if class == "date" => Read::Cal(CalField::Year),
            "getMonth" if class == "date" => Read::Cal(CalField::Month),
            "getDayOfMonth" if class == "date" => Read::Cal(CalField::DayOfMonth),
            "getHourOfDay" if class == "date" => Read::Cal(CalField::HourOfDay),
            "getMinutes" if class == "date" => Read::Cal(CalField::Minute),
            "getSeconds" if class == "date" => Read::Cal(CalField::Second),
            _ => return Err(iae(format!("Member method [{m}] does not exist for {class} field [{field}]."))),
        },
        _ => Read::Min,
    })
}

fn link_one(rt: &Rt, v: &VarSpec) -> SResult<Bind> {
    let ctx = rt.prog.ctx;
    if v.text == "_score" && matches!(ctx, ScriptContext::Score | ScriptContext::Filter | ScriptContext::Aggs | ScriptContext::NumberSort) {
        return Ok(Bind::Score);
    }
    if v.text == "_value" && matches!(ctx, ScriptContext::Score | ScriptContext::Filter | ScriptContext::Aggs) {
        return Ok(Bind::Value);
    }
    if let Some(p) = param_number(rt, &v.text) {
        return p.map(Bind::Const).ok_or_else(|| parse_err(format!("Parameter [{}] must be a numeric type", v.text), 0));
    }
    if let Some((msg, off)) = &v.shape_err {
        return Err(match off {
            Some(o) => parse_err(msg.clone(), *o),
            None => iae(msg.clone()),
        });
    }
    let (field, slot) = v.doc.clone().expect("doc variable");
    // ES loads the field data before looking at the type: text without
    // fielddata fails there.
    let fielddata = rt.doc.lookup.as_ref().and_then(|l| l.get(&field)).is_some_and(|f| f.fielddata);
    if rt.doc.fields[slot].kind == Kind::Text && !fielddata {
        return Err(iae(format!(
            "Fielddata is disabled on [{field}]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [{field}] in order to load field data by uninverting the inverted index. Note that this can use significant memory."
        )));
    }
    let Some(class) = field_class(rt, slot) else {
        return Err(parse_err(format!("Field [{field}] does not exist in mappings"), 5));
    };
    let read = match class {
        FieldClass::Geo => match &v.member {
            None => return Err(iae(format!("Member variable [value] does not exist for geo field [{field}]."))),
            Some(Part::Member(m)) => match m.as_str() {
                "empty" => Read::GeoEmpty,
                "lat" => Read::GeoLat,
                "lon" => Read::GeoLon,
                _ => return Err(iae(format!("Member variable [{m}] does not exist for geo field [{field}]."))),
            },
            Some(Part::Method(m)) => match m.as_str() {
                "isEmpty" => Read::GeoEmpty,
                "getLat" => Read::GeoLat,
                "getLon" => Read::GeoLon,
                _ => return Err(iae(format!("Member method [{m}] does not exist for geo field [{field}]."))),
            },
            _ => unreachable!(),
        },
        FieldClass::Date if v.date_obj => match &v.member {
            Some(Part::Member(m)) => Read::Joda(joda_field(m).filter(|_| !m.starts_with("get")).ok_or_else(|| iae(format!("Member variable [{m}] does not exist for date object on field [{field}].")))?),
            Some(Part::Method(m)) => Read::Joda(joda_field(m).filter(|_| m.starts_with("get")).ok_or_else(|| iae(format!("Member method [{m}] does not exist for date object on field [{field}].")))?),
            _ => unreachable!(),
        },
        FieldClass::Date => numeric_read(&field, &v.member, "date")?,
        FieldClass::Numeric if !v.date_obj => {
            let f = &rt.doc.fields[slot];
            if !f.doc_values && !matches!(f.kind, Kind::Runtime(_)) {
                return Err(iae(format!("Can't load fielddata on [{field}] because fielddata is unsupported on fields of type [{}]. Use doc values instead.", kind_type_name(f.kind))));
            }
            numeric_read(&field, &v.member, "numeric")?
        }
        FieldClass::Numeric | FieldClass::Other => {
            if class == FieldClass::Other {
                return Err(parse_err(format!("Field [{field}] must be numeric, date, or geopoint"), 5));
            }
            return Err(iae(format!("Variable [{}] does not follow an allowed format of either doc['field'] or doc['field'].method()", v.text)));
        }
    };
    Ok(Bind::Doc { slot, read })
}

fn kind_type_name(k: Kind) -> &'static str {
    match k {
        Kind::Double => "double",
        Kind::Unsigned => "unsigned_long",
        Kind::Bool => "boolean",
        Kind::Date => "date",
        _ => "long",
    }
}

/// Link every variable for this executor (first evaluation).
#[cold]
fn link(rt: &mut Rt) -> SResult<()> {
    let meta = rt.prog.expression.clone().expect("expression program");
    let mut binds = Vec::with_capacity(meta.vars.len());
    for v in &meta.vars {
        binds.push(link_one(rt, v).map_err(|e| link_err(&v.text, e))?);
    }
    rt.expr_binds = binds;
    rt.expr_linked = true;
    Ok(())
}

// ---------------------------------------------------------------------------
// Evaluation.

type F = Box<dyn Fn(&mut Rt) -> f64 + Send + Sync>;

fn values_of(rt: &mut Rt, slot: usize) -> Vec<f64> {
    match rt.doc.values(slot) {
        Ok(Val::DocValues(dv)) => dv
            .values
            .iter()
            .filter_map(|v| match v {
                Val::Bool(b) => Some(*b as i64 as f64),
                Val::Date(ms) | Val::Instant(ms) => Some(*ms as f64),
                other => other.as_f64(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn reduce(vals: &[f64], read: Read) -> f64 {
    if vals.is_empty() {
        return if read == Read::Empty { 1.0 } else { 0.0 };
    }
    let n = vals.len();
    match read {
        Read::Empty => 0.0,
        Read::GeoEmpty => 1.0,
        Read::Count => n as f64,
        Read::Max => vals[n - 1],
        Read::Sum => vals.iter().sum(),
        Read::Avg => vals.iter().sum::<f64>() / n as f64,
        Read::Median => {
            if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) / 2.0
            }
        }
        Read::Cal(c) => cal(vals[0] as i64, c),
        Read::Joda(j) => joda(vals[0] as i64, j),
        _ => vals[0],
    }
}

fn cal(ms: i64, f: CalField) -> f64 {
    use chrono::{Datelike, Timelike};
    let t = crate::date::from_millis(ms);
    (match f {
        CalField::Year => t.year() as i64,
        CalField::Month => t.month0() as i64,
        CalField::DayOfMonth => t.day() as i64,
        CalField::HourOfDay => t.hour() as i64,
        CalField::Minute => t.minute() as i64,
        CalField::Second => t.second() as i64,
    }) as f64
}

fn joda(ms: i64, f: JodaField) -> f64 {
    use chrono::{Datelike, Timelike};
    let t = crate::date::from_millis(ms);
    let year = t.year() as i64;
    let year_of_era = if year <= 0 { 1 - year } else { year };
    let ms_of_day = ms.rem_euclid(86_400_000);
    (match f {
        JodaField::CenturyOfEra => year_of_era / 100,
        JodaField::DayOfMonth => t.day() as i64,
        JodaField::DayOfWeek => t.weekday().number_from_monday() as i64,
        JodaField::DayOfYear => t.ordinal() as i64,
        JodaField::Era => (year > 0) as i64,
        JodaField::HourOfDay => t.hour() as i64,
        JodaField::MillisOfDay => ms_of_day,
        JodaField::MillisOfSecond => ms.rem_euclid(1000),
        JodaField::MinuteOfDay => ms_of_day / 60_000,
        JodaField::MinuteOfHour => t.minute() as i64,
        JodaField::MonthOfYear => t.month() as i64,
        JodaField::SecondOfDay => ms_of_day / 1000,
        JodaField::SecondOfMinute => t.second() as i64,
        JodaField::WeekOfWeekyear => t.iso_week().week() as i64,
        JodaField::Weekyear => t.iso_week().year() as i64,
        JodaField::Year => year,
        JodaField::YearOfCentury => year_of_era % 100,
        JodaField::YearOfEra => year_of_era,
    }) as f64
}

#[inline]
fn read_var(rt: &mut Rt, i: usize) -> f64 {
    match rt.expr_binds[i] {
        Bind::Const(c) => c,
        Bind::Score => rt.score,
        Bind::Value => rt.value.as_f64().unwrap_or(0.0),
        Bind::Doc { slot, read } => read_doc(rt, slot, read),
    }
}

#[inline]
fn read_doc(rt: &mut Rt, slot: usize, read: Read) -> f64 {
    match read {
        Read::GeoLat | Read::GeoLon if rt.doc.expr_geo_fast(slot) => rt.doc.expr_geo_coord(slot, read == Read::GeoLat).unwrap_or(0.0),
        Read::GeoLat | Read::GeoLon | Read::GeoEmpty => {
            let p = match rt.doc.expr_geo(slot) {
                Ok(p) => p,
                Err(()) => geo_slow(rt, slot),
            };
            // ES `GeoEmptyValueSource` reads 1 for docs *with* a point (and
            // the expression's 0 for docs without one).
            match (p, read) {
                (None, _) => 0.0,
                (Some(_), Read::GeoEmpty) => 1.0,
                (Some((lat, _)), Read::GeoLat) => lat,
                (Some((_, lon)), _) => lon,
            }
        }
        _ => match rt.doc.expr_single(slot) {
            Ok(Some(v)) => match read {
                Read::Empty => 0.0,
                Read::Count => 1.0,
                Read::Cal(c) => cal(v as i64, c),
                Read::Joda(j) => joda(v as i64, j),
                _ => v,
            },
            Ok(None) => (read == Read::Empty) as i64 as f64,
            Err(()) => {
                let vals = values_of(rt, slot);
                reduce(&vals, read)
            }
        },
    }
}

#[cold]
fn geo_slow(rt: &mut Rt, slot: usize) -> Option<(f64, f64)> {
    match rt.doc.values(slot) {
        Ok(Val::DocValues(dv)) => dv.values.first().and_then(|p| match p {
            Val::Map(m) => {
                let m = m.borrow();
                Some((m.get(&Key::str("lat"))?.as_f64()?, m.get(&Key::str("lon"))?.as_f64()?))
            }
            _ => None,
        }),
        _ => None,
    }
}

#[inline(always)]
fn truthy(v: f64) -> bool {
    v != 0.0
}

/// Java `(long) d`.
#[inline(always)]
fn to_long(d: f64) -> i64 {
    if d.is_nan() { 0 } else { d as i64 }
}

/// An operand evaluated inline by its parent node: constants and
/// variables need no call of their own.
enum Opnd {
    K(f64),
    V(usize),
    F(F),
}

impl Opnd {
    fn of(n: &Node) -> Opnd {
        if let Some(c) = fold(n) {
            return Opnd::K(c);
        }
        match n {
            Node::Num(v) => Opnd::K(*v),
            Node::Var(i) => Opnd::V(*i),
            _ => Opnd::F(compile_node(n)),
        }
    }

    #[inline(always)]
    fn ev(&self, rt: &mut Rt) -> f64 {
        match self {
            Opnd::K(c) => *c,
            Opnd::V(i) => read_var(rt, *i),
            Opnd::F(f) => f(rt),
        }
    }
}

fn compile_node(n: &Node) -> F {
    if let Some(c) = fold(n) {
        return Box::new(move |_| c);
    }
    match n {
        Node::Num(v) => {
            let v = *v;
            Box::new(move |_| v)
        }
        Node::Var(i) => {
            let i = *i;
            Box::new(move |rt| read_var(rt, i))
        }
        Node::Call(f, args) => {
            let f = *f;
            match args.len() {
                1 => {
                    let a = Opnd::of(&args[0]);
                    match f {
                        // The common ones as direct calls the optimizer can inline.
                        Func::Abs => Box::new(move |rt| a.ev(rt).abs()),
                        Func::Ln => Box::new(move |rt| a.ev(rt).ln()),
                        Func::Log10 => Box::new(move |rt| a.ev(rt).log10()),
                        Func::Sqrt => Box::new(move |rt| a.ev(rt).sqrt()),
                        Func::Exp => Box::new(move |rt| a.ev(rt).exp()),
                        _ => {
                            let op = f.unary();
                            Box::new(move |rt| op(a.ev(rt)))
                        }
                    }
                }
                2 => {
                    let (a, b) = (Opnd::of(&args[0]), Opnd::of(&args[1]));
                    let op = f.binary();
                    Box::new(move |rt| {
                        let x = a.ev(rt);
                        op(x, b.ev(rt))
                    })
                }
                _ => {
                    let fs: Vec<Opnd> = args.iter().map(Opnd::of).collect();
                    Box::new(move |rt| {
                        let (a, b, c) = (fs[0].ev(rt), fs[1].ev(rt), fs[2].ev(rt));
                        haversin_km(a, b, c, fs[3].ev(rt))
                    })
                }
            }
        }
        Node::Unary(op, e) => {
            let e = Opnd::of(e);
            match op {
                UOp::Neg => Box::new(move |rt| -e.ev(rt)),
                UOp::Not => Box::new(move |rt| (!truthy(e.ev(rt))) as i64 as f64),
                UOp::BitNot => Box::new(move |rt| !to_long(e.ev(rt)) as f64),
            }
        }
        Node::Cond(c, a, b) => {
            let (c, a, b) = (Opnd::of(c), Opnd::of(a), Opnd::of(b));
            Box::new(move |rt| if truthy(c.ev(rt)) { a.ev(rt) } else { b.ev(rt) })
        }
        Node::Bin(op, a, b) => {
            let (x, y) = (Opnd::of(a), Opnd::of(b));
            macro_rules! bin {
                (|$p:ident, $q:ident| $e:expr) => {
                    Box::new(move |rt| {
                        let $p = x.ev(rt);
                        let $q = y.ev(rt);
                        $e
                    })
                };
            }
            match op {
                BOp::Add => bin!(|p, q| p + q),
                BOp::Sub => bin!(|p, q| p - q),
                BOp::Mul => bin!(|p, q| p * q),
                BOp::Div => bin!(|p, q| p / q),
                BOp::Rem => bin!(|p, q| p % q),
                BOp::Shl => bin!(|p, q| to_long(p).wrapping_shl(q as i32 as u32) as f64),
                BOp::Shr => bin!(|p, q| to_long(p).wrapping_shr(q as i32 as u32) as f64),
                BOp::Ushr => bin!(|p, q| (to_long(p) as u64).wrapping_shr(q as i32 as u32) as i64 as f64),
                BOp::Lt => bin!(|p, q| (p < q) as i64 as f64),
                BOp::Le => bin!(|p, q| (p <= q) as i64 as f64),
                BOp::Gt => bin!(|p, q| (p > q) as i64 as f64),
                BOp::Ge => bin!(|p, q| (p >= q) as i64 as f64),
                BOp::Eq => bin!(|p, q| (p == q) as i64 as f64),
                BOp::Ne => bin!(|p, q| (p != q) as i64 as f64),
                BOp::BitAnd => bin!(|p, q| (to_long(p) & to_long(q)) as f64),
                BOp::BitXor => bin!(|p, q| (to_long(p) ^ to_long(q)) as f64),
                BOp::BitOr => bin!(|p, q| (to_long(p) | to_long(q)) as f64),
                BOp::And => Box::new(move |rt| (truthy(x.ev(rt)) && truthy(y.ev(rt))) as i64 as f64),
                BOp::Or => Box::new(move |rt| (truthy(x.ev(rt)) || truthy(y.ev(rt))) as i64 as f64),
            }
        }
    }
}

/// Value of a variable-free operation subtree (`None`: not constant, or
/// a bare number which needs no folding).
fn fold(n: &Node) -> Option<f64> {
    match n {
        Node::Num(_) | Node::Var(_) => None,
        _ if has_var(n) => None,
        _ => Some(const_value(n)),
    }
}

fn const_value(n: &Node) -> f64 {
    let c = |x: &Node| Node::Num(const_value(x));
    let tree = match n {
        Node::Num(v) => return *v,
        Node::Call(f, a) => Node::Call(*f, a.iter().map(c).collect()),
        Node::Unary(op, e) => Node::Unary(*op, Box::new(c(e))),
        Node::Bin(op, a, b) => Node::Bin(*op, Box::new(c(a)), Box::new(c(b))),
        Node::Cond(a, b, d) => Node::Cond(Box::new(c(a)), Box::new(c(b)), Box::new(c(d))),
        Node::Var(_) => return 0.0,
    };
    eval_const(&tree).unwrap_or(0.0)
}

fn has_var(n: &Node) -> bool {
    match n {
        Node::Num(_) => false,
        Node::Var(_) => true,
        Node::Call(_, a) => a.iter().any(has_var),
        Node::Unary(_, e) => has_var(e),
        Node::Bin(_, a, b) => has_var(a) || has_var(b),
        Node::Cond(a, b, c) => has_var(a) || has_var(b) || has_var(c),
    }
}

/// Evaluate a node whose children are all numbers.
fn eval_const(n: &Node) -> Option<f64> {
    let num = |x: &Node| match x {
        Node::Num(v) => Some(*v),
        _ => None,
    };
    Some(match n {
        Node::Num(v) => *v,
        Node::Call(f, a) => {
            let v: Option<Vec<f64>> = a.iter().map(num).collect();
            let v = v?;
            match v.len() {
                1 => f.unary()(v[0]),
                2 => f.binary()(v[0], v[1]),
                _ => haversin_km(v[0], v[1], v[2], v[3]),
            }
        }
        Node::Unary(op, e) => {
            let e = num(e)?;
            match op {
                UOp::Neg => -e,
                UOp::Not => (!truthy(e)) as i64 as f64,
                UOp::BitNot => !to_long(e) as f64,
            }
        }
        Node::Cond(c, a, b) => {
            if truthy(num(c)?) {
                num(a)?
            } else {
                num(b)?
            }
        }
        Node::Bin(op, a, b) => {
            let (p, q) = (num(a)?, num(b)?);
            match op {
                BOp::Add => p + q,
                BOp::Sub => p - q,
                BOp::Mul => p * q,
                BOp::Div => p / q,
                BOp::Rem => p % q,
                BOp::Shl => to_long(p).wrapping_shl(q as i32 as u32) as f64,
                BOp::Shr => to_long(p).wrapping_shr(q as i32 as u32) as f64,
                BOp::Ushr => (to_long(p) as u64).wrapping_shr(q as i32 as u32) as i64 as f64,
                BOp::Lt => (p < q) as i64 as f64,
                BOp::Le => (p <= q) as i64 as f64,
                BOp::Gt => (p > q) as i64 as f64,
                BOp::Ge => (p >= q) as i64 as f64,
                BOp::Eq => (p == q) as i64 as f64,
                BOp::Ne => (p != q) as i64 as f64,
                BOp::BitAnd => (to_long(p) & to_long(q)) as f64,
                BOp::BitXor => (to_long(p) ^ to_long(q)) as f64,
                BOp::BitOr => (to_long(p) | to_long(q)) as f64,
                BOp::And => (truthy(p) && truthy(q)) as i64 as f64,
                BOp::Or => (truthy(p) || truthy(q)) as i64 as f64,
            }
        }
        Node::Var(_) => return None,
    })
}

/// Compile `src` for `ctx` into a [`Program`].
pub fn compile(src: &str, ctx: ScriptContext) -> Result<Program, Box<ScriptErr>> {
    let (root, var_texts) = parse(src).map_err(|mut e| {
        e.compile = true;
        e
    })?;
    let mut doc_fields: Vec<String> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();
    let mut vars = Vec::new();
    for t in &var_texts {
        let parts = var_parts(t);
        let mut spec = VarSpec { text: t.clone(), doc: None, member: None, date_obj: false, shape_err: None, param: 0 };
        spec.param = match param_names.iter().position(|p| p == t) {
            Some(i) => i,
            None => {
                param_names.push(t.clone());
                param_names.len() - 1
            }
        };
        match parts.first() {
            Some(Part::Member(m)) if m == "doc" => {}
            Some(Part::Member(m)) | Some(Part::Method(m)) => spec.shape_err = Some((format!("Unknown variable [{m}]"), Some(0))),
            _ => spec.shape_err = Some((format!("Unknown variable [{t}]"), Some(0))),
        }
        if spec.shape_err.is_none() {
            match parts.get(1) {
                Some(Part::StrIndex(f)) => {
                    let slot = match doc_fields.iter().position(|d| d == f) {
                        Some(i) => i,
                        None => {
                            doc_fields.push(f.clone());
                            doc_fields.len() - 1
                        }
                    };
                    spec.doc = Some((f.clone(), slot));
                }
                _ => spec.shape_err = Some(("Variable 'doc' must be used with a specific field like: doc['myfield']".into(), Some(3))),
            }
        }
        if spec.shape_err.is_none() {
            match parts.len() {
                2 => {}
                3 => match &parts[2] {
                    p @ (Part::Member(_) | Part::Method(_)) => spec.member = Some(p.clone()),
                    _ => spec.shape_err = Some(("Only member variables or member methods may be accessed on a field when not accessing the field directly".into(), None)),
                },
                4 if matches!(&parts[2], Part::Member(m) | Part::Method(m) if m == "date" || m == "getDate") && matches!(parts[3], Part::Member(_) | Part::Method(_)) => {
                    spec.member = Some(parts[3].clone());
                    spec.date_obj = true;
                }
                _ => spec.shape_err = Some((format!("Variable [{t}] does not follow an allowed format of either doc['field'] or doc['field'].method()"), None)),
            }
        }
        vars.push(spec);
    }
    let uses_score = var_texts.iter().any(|v| v == "_score");
    let f: Arc<F> = Arc::new(compile_node(&root));
    let num = f.clone();
    let num_body: Box<dyn Fn(&mut Rt) -> Option<f64> + Send + Sync> = Box::new(move |rt| {
        if !rt.expr_linked {
            return None;
        }
        Some(num(rt))
    });
    let to_val = move |v: f64| -> Val {
        match ctx {
            ScriptContext::Filter => Val::Bool(truthy(v)),
            ScriptContext::BucketSelector => Val::Bool(v == 1.0),
            _ => Val::Double(v),
        }
    };
    let ev = f.clone();
    let expr_body: super::runtime::ExprFn = Box::new(move |rt| {
        if !rt.expr_linked {
            link(rt)?;
        }
        Ok(to_val(ev(rt)))
    });
    let ev = f.clone();
    let body: super::runtime::StmtFn = Box::new(move |rt| {
        if !rt.expr_linked {
            link(rt)?;
        }
        Ok(Flow::Return(to_val(ev(rt))))
    });
    let cond = f;
    let cond_body: Option<Box<dyn Fn(&mut Rt) -> Option<bool> + Send + Sync>> = match ctx {
        ScriptContext::Filter => Some(Box::new(move |rt| if rt.expr_linked { Some(truthy(cond(rt))) } else { None })),
        _ => None,
    };
    Ok(Program {
        source: src.to_string(),
        stmt_starts: vec![0],
        ctx,
        body,
        expr_body: Some(expr_body),
        num_body: if matches!(ctx, ScriptContext::Filter | ScriptContext::BucketSelector) { None } else { Some(num_body) },
        cond_body,
        uses_score,
        result_pos: 0,
        nslots: 1,
        functions: Vec::new(),
        lambdas: Vec::new(),
        strings: Vec::new(),
        param_names,
        doc_fields,
        nbindings: 0,
        expression: Some(Arc::new(ExprMeta { vars })),
        ctx_paths: None,
    })
}

/// Link errors of `program` against `lookup` / `params` without running it
/// (ES binds variables when the search builds the script's leaf factory).
pub fn check_link(program: &Arc<Program>, params: &serde_json::Value, lookup: Option<Arc<crate::mapping::FieldLookup>>, runtime: Option<Arc<super::RuntimeFields>>) -> Result<(), EsError> {
    if program.expression.is_none() {
        return Ok(());
    }
    let mut ex = super::Executor::new(program.clone(), params, lookup, runtime);
    match link(&mut ex.rt) {
        Ok(()) => Ok(()),
        Err(e) => Err(to_es_error(&e, &program.source)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nesting_is_bounded() {
        std::thread::Builder::new().stack_size(2 << 20).spawn(|| {
            assert!(parse(&format!("{}1{}", "(".repeat(100), ")".repeat(100))).is_ok());
            assert!(parse(&vec!["1"; 500].join(" + ")).is_ok());
            for src in [format!("{}1{}", "(".repeat(50_000), ")".repeat(50_000)), format!("{}1", "-".repeat(50_000)), vec!["1"; 100_000].join(" + ")] {
                let e = parse(&src).err().expect("too deep");
                assert!(format!("{e:?}").contains("levels deep"), "{e:?}");
            }
        }).unwrap().join().unwrap();
    }

    fn eval(src: &str) -> f64 {
        let p = Arc::new(compile(src, ScriptContext::Score).unwrap());
        let mut ex = super::super::Executor::new(p, &json!({"x": 2, "y": 0.5}), None, None);
        ex.set_score(3.0);
        ex.run().unwrap().as_f64().unwrap()
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(eval("1 + 2 * 3"), 7.0);
        assert_eq!(eval("(1 + 2) * 3"), 9.0);
        assert_eq!(eval("7 / 2"), 3.5);
        assert_eq!(eval("7 % 4"), 3.0);
        assert_eq!(eval("-2 * -3"), 6.0);
        assert_eq!(eval("2 < 3 ? 10 : 20"), 10.0);
        assert_eq!(eval("1 ? 2 : 0 ? 3 : 4"), 2.0);
        assert_eq!(eval("!0 + !5"), 1.0);
        assert_eq!(eval("1 << 4 | 1"), 17.0);
        assert_eq!(eval("-1 >>> 60"), 15.0);
        assert_eq!(eval("~5"), -6.0);
        assert_eq!(eval("0x1F + 010"), 39.0);
        assert_eq!(eval("1 && 0 || 2"), 1.0);
        assert_eq!(eval("1e2 + .5"), 100.5);
        assert_eq!(eval("2 == 2.0"), 1.0);
    }

    #[test]
    fn variables_and_functions() {
        assert_eq!(eval("_score * x + y"), 6.5);
        assert_eq!(eval("max(x, _score)"), 3.0);
        assert_eq!(eval("abs(-x)"), 2.0);
        assert!((eval("ln(exp(2))") - 2.0).abs() < 1e-12);
        assert!((eval("logn(2, 8)") - 3.0).abs() < 1e-12);
        assert_eq!(eval("pow(x, 10)"), 1024.0);
        let km = eval("haversin(40.7143528, -74.0059731, 40.759011, -73.9844722)");
        assert!((km - 5.28).abs() < 0.02, "{km}");
    }

    #[test]
    fn parse_errors() {
        for (src, frag) in [("1 +", "unexpected token"), ("foo(1)", "Unrecognized function call (foo)"), ("max(1)", "Expected (2) arguments for function call (max), but found (1)"), ("1 # 2", "unexpected character")] {
            let e = compile(src, ScriptContext::Score).unwrap_err();
            assert_eq!(e.kind, "parse_exception");
            assert!(e.msg.contains(frag), "{src}: {}", e.msg);
            assert!(e.compile);
        }
    }

    #[test]
    fn variable_shapes() {
        assert_eq!(var_parts("doc['f'].min()"), vec![Part::Member("doc".into()), Part::StrIndex("f".into()), Part::Method("min".into())]);
        assert_eq!(var_parts("doc['a.b'].date.year"), vec![Part::Member("doc".into()), Part::StrIndex("a.b".into()), Part::Member("date".into()), Part::Member("year".into())]);
        assert_eq!(normalize_quotes("doc[\"f\"].value"), "doc['f'].value");
        let p = compile("doc['f'] + doc[\"f\"].value + doc['f'].max() + x", ScriptContext::Score).unwrap();
        assert_eq!(p.doc_fields, vec!["f".to_string()]);
        assert_eq!(p.expression.as_ref().unwrap().vars.len(), 4);
    }

    #[test]
    fn link_errors() {
        let p = Arc::new(compile("bogus + 1", ScriptContext::Score).unwrap());
        let mut ex = super::super::Executor::new(p.clone(), &json!({}), None, None);
        let e = ex.run().unwrap_err();
        let es = to_es_error(&e, &p.source).to_json()["error"].clone();
        assert_eq!(es["reason"], "link error");
        assert_eq!(es["caused_by"]["reason"], "Unknown variable [bogus]");
        let mut ex = super::super::Executor::new(p, &json!({"bogus": "x"}), None, None);
        let e = ex.run().unwrap_err();
        assert_eq!(e.msg, "Parameter [bogus] must be a numeric type");
    }

    #[test]
    fn dates() {
        // 2020-02-29T13:45:30.123Z, a Saturday.
        let ms = 1_582_983_930_123;
        assert_eq!(cal(ms, CalField::Year), 2020.0);
        assert_eq!(cal(ms, CalField::Month), 1.0);
        assert_eq!(cal(ms, CalField::DayOfMonth), 29.0);
        assert_eq!(joda(ms, JodaField::DayOfWeek), 6.0);
        assert_eq!(joda(ms, JodaField::MillisOfSecond), 123.0);
        assert_eq!(joda(ms, JodaField::CenturyOfEra), 20.0);
    }
}
