//! The `PROMQL` source command: a PromQL expression evaluated over time
//! series indices. The expression compiles to a time-series pipeline
//! (`TS index | WHERE <label matchers> | STATS <aggregation> BY step
//! bucket, labels`), with one `step` per evaluation step (the end of its
//! bucket).

use super::func;
use super::parse::{BinOp, Cmd, Expr, Field, Params};
use super::{parsing_err, verification_err, Val};
use crate::error::Result;
use serde_json::Value;

#[derive(Clone, Debug)]
pub enum MatchOp {
    Eq,
    Ne,
    Re,
    Nre,
}

#[derive(Clone, Debug)]
pub struct Matcher {
    pub label: String,
    pub op: MatchOp,
    pub value: String,
}

#[derive(Clone, Debug)]
pub enum PExpr {
    Num(f64),
    Str(String),
    Sel { metric: String, matchers: Vec<Matcher>, range: Option<i64>, pos: usize },
    Call { name: String, args: Vec<PExpr>, pos: usize },
    Agg { op: String, param: Option<Box<PExpr>>, expr: Box<PExpr>, by: Vec<String>, without: bool, pos: usize },
    /// `bool_mod`: the comparison's `bool` modifier (0/1 samples instead
    /// of filtering).
    Bin { op: String, l: Box<PExpr>, r: Box<PExpr>, bool_mod: bool },
    Neg(Box<PExpr>),
}

/// A parsed `PROMQL` command.
#[derive(Clone, Debug)]
pub struct Promql {
    pub index: Vec<String>,
    pub step: Option<i64>,
    pub buckets: Option<i64>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub scrape_interval: i64,
    /// Output column name (`name=(expr)`), else the expression text.
    pub name: String,
    pub expr: PExpr,
    pub pos: usize,
}

const AGG_OPS: &[&str] = &["sum", "avg", "min", "max", "count", "stddev", "stdvar", "quantile", "topk", "bottomk", "group", "count_values", "limitk", "limit_ratio"];

/// Range-vector functions and their ES|QL time-series functions.
fn range_fn(name: &str) -> Option<&'static str> {
    Some(match name {
        "rate" => "RATE",
        "irate" => "IRATE",
        "increase" => "INCREASE",
        "delta" => "DELTA",
        "idelta" => "IDELTA",
        "deriv" => "DERIV",
        "avg_over_time" => "AVG_OVER_TIME",
        "sum_over_time" => "SUM_OVER_TIME",
        "min_over_time" => "MIN_OVER_TIME",
        "max_over_time" => "MAX_OVER_TIME",
        "count_over_time" => "COUNT_OVER_TIME",
        "last_over_time" => "LAST_OVER_TIME",
        "first_over_time" => "FIRST_OVER_TIME",
        "stddev_over_time" => "STDDEV_OVER_TIME",
        "stdvar_over_time" => "VARIANCE_OVER_TIME",
        "present_over_time" => "PRESENT_OVER_TIME",
        "absent_over_time" => "ABSENT_OVER_TIME",
        "quantile_over_time" => "PERCENTILE_OVER_TIME",
        _ => return None,
    })
}

/// Instant-vector math functions and their ES|QL scalar functions.
fn math_fn(name: &str) -> Option<&'static str> {
    Some(match name {
        "abs" => "ABS",
        "ceil" => "CEIL",
        "floor" => "FLOOR",
        "round" => "ROUND",
        "sqrt" => "SQRT",
        "exp" => "EXP",
        "ln" => "LOG",
        "log10" => "LOG10",
        "log2" => "LOG2",
        "sgn" => "SIGNUM",
        "clamp_min" => "CLAMP_MIN",
        "clamp_max" => "CLAMP_MAX",
        "sin" => "SIN",
        "cos" => "COS",
        "tan" => "TAN",
        "asin" => "ASIN",
        "acos" => "ACOS",
        "atan" => "ATAN",
        "sinh" => "SINH",
        "cosh" => "COSH",
        "tanh" => "TANH",
        "deg" => "TO_DEGREES",
        "rad" => "TO_RADIANS",
        "scalar" | "vector" => "\u{0}id",
        _ => return None,
    })
}

/// A PromQL duration (`1m`, `90s`, `1h30m`) in millis.
pub fn duration(s: &str) -> Option<i64> {
    let mut total = 0i64;
    let b = s.as_bytes();
    let mut i = 0;
    if s.is_empty() {
        return None;
    }
    while i < b.len() {
        let st = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let n: i64 = s[st..i].parse().ok()?;
        let us = i;
        while i < b.len() && b[i].is_ascii_alphabetic() {
            i += 1;
        }
        let unit = match &s[us..i] {
            "ms" => 1,
            "s" => 1000,
            "m" => 60_000,
            "h" => 3_600_000,
            "d" => 86_400_000,
            "w" => 7 * 86_400_000,
            "y" => 365 * 86_400_000,
            _ => return None,
        };
        total += n * unit;
    }
    Some(total)
}

struct P<'a> {
    full: &'a str,
    b: &'a [u8],
    i: usize,
    end: usize,
    params: &'a Params,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.end {
            if self.b[self.i].is_ascii_whitespace() {
                self.i += 1;
            } else if self.b[self.i] == b'#' {
                while self.i < self.end && self.b[self.i] != b'\n' {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        (self.i < self.end).then(|| self.b[self.i])
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn err(&self, msg: impl AsRef<str>) -> crate::error::EsError {
        parsing_err(self.full, self.i, msg)
    }

    fn ident(&mut self) -> Option<String> {
        self.ws();
        let st = self.i;
        while self.i < self.end && (self.b[self.i].is_ascii_alphanumeric() || matches!(self.b[self.i], b'_' | b':' | b'.' | b'@')) {
            self.i += 1;
        }
        (self.i > st).then(|| self.full[st..self.i].to_string())
    }

    fn string(&mut self) -> Result<String> {
        self.ws();
        let q = self.b[self.i];
        let st = self.i + 1;
        let mut j = st;
        let mut out = String::new();
        while j < self.end && self.b[j] != q {
            if self.b[j] == b'\\' && j + 1 < self.end {
                j += 1;
            }
            out.push(self.b[j] as char);
            j += 1;
        }
        if j >= self.end {
            return Err(self.err("unterminated string"));
        }
        self.i = j + 1;
        Ok(out)
    }

    /// A label matcher value: a string or a query parameter (several
    /// values become a regex alternation).
    fn match_value(&mut self, op: &mut MatchOp) -> Result<String> {
        self.ws();
        if self.b.get(self.i) == Some(&b'?') {
            let st = self.i;
            self.i += 1;
            let ns = self.i;
            while self.i < self.end && (self.b[self.i].is_ascii_alphanumeric() || self.b[self.i] == b'_') {
                self.i += 1;
            }
            let name = &self.full[ns..self.i];
            let v = if let Ok(n) = name.parse::<usize>() { self.params.positional.get(n.wrapping_sub(1)).cloned() } else { self.params.named.get(name).cloned() };
            let Some(v) = v else { return Err(parsing_err(self.full, st, format!("Parameter [?{name}] value not found"))) };
            let v = match v {
                Value::Object(m) => m.get("value").cloned().unwrap_or(Value::Null),
                o => o,
            };
            return Ok(match v {
                Value::Array(a) => {
                    // Several values: an alternation regex.
                    if matches!(op, MatchOp::Eq) {
                        *op = MatchOp::Re;
                    } else if matches!(op, MatchOp::Ne) {
                        *op = MatchOp::Nre;
                    }
                    let quote = matches!(op, MatchOp::Re | MatchOp::Nre) && !a.is_empty();
                    a.iter().map(|x| {
                        let s = x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string());
                        if quote { regex::escape(&s) } else { s }
                    }).collect::<Vec<_>>().join("|")
                }
                Value::String(s) => s,
                o => o.to_string(),
            });
        }
        if matches!(self.b.get(self.i), Some(b'"') | Some(b'\'') | Some(b'`')) {
            return self.string();
        }
        Err(self.err("expected a label value"))
    }

    fn selector_rest(&mut self, metric: String, pos: usize) -> Result<PExpr> {
        let mut matchers = Vec::new();
        if self.eat(b'{') {
            loop {
                if self.eat(b'}') {
                    break;
                }
                let label = self.ident().ok_or_else(|| self.err("expected a label name"))?;
                self.ws();
                let rest = &self.full[self.i..self.end];
                let mut op = if rest.starts_with("=~") {
                    self.i += 2;
                    MatchOp::Re
                } else if rest.starts_with("!~") {
                    self.i += 2;
                    MatchOp::Nre
                } else if rest.starts_with("!=") {
                    self.i += 2;
                    MatchOp::Ne
                } else if rest.starts_with('=') {
                    self.i += 1;
                    MatchOp::Eq
                } else {
                    return Err(self.err("expected a label matcher"));
                };
                let value = self.match_value(&mut op)?;
                matchers.push(Matcher { label, op, value });
                if self.eat(b',') {
                    continue;
                }
                if !self.eat(b'}') {
                    return Err(self.err("expected '}'"));
                }
                break;
            }
        }
        let mut range = None;
        if self.eat(b'[') {
            self.ws();
            let st = self.i;
            while self.i < self.end && self.b[self.i] != b']' {
                self.i += 1;
            }
            let d = self.full[st..self.i].trim().to_string();
            range = Some(duration(&d).ok_or_else(|| parsing_err(self.full, st, format!("invalid duration [{d}]")))?);
            self.i += 1;
        }
        Ok(PExpr::Sel { metric, matchers, range, pos })
    }

    fn grouping(&mut self) -> Result<(Vec<String>, bool)> {
        self.ws();
        let save = self.i;
        let w = self.ident().unwrap_or_default();
        let without = match w.as_str() {
            "by" => false,
            "without" => true,
            _ => {
                self.i = save;
                return Ok((vec![], false));
            }
        };
        if !self.eat(b'(') {
            return Err(self.err("expected '('"));
        }
        let mut labels = Vec::new();
        loop {
            if self.eat(b')') {
                break;
            }
            labels.push(self.ident().ok_or_else(|| self.err("expected a label"))?);
            if self.eat(b',') {
                continue;
            }
            if !self.eat(b')') {
                return Err(self.err("expected ')'"));
            }
            break;
        }
        Ok((labels, without))
    }

    fn primary(&mut self) -> Result<PExpr> {
        self.ws();
        let pos = self.i;
        match self.peek() {
            Some(b'(') => {
                self.i += 1;
                let e = self.expr(0)?;
                if !self.eat(b')') {
                    return Err(self.err("expected ')'"));
                }
                Ok(e)
            }
            Some(b'-') => {
                self.i += 1;
                Ok(PExpr::Neg(Box::new(self.primary()?)))
            }
            Some(b'+') => {
                self.i += 1;
                self.primary()
            }
            Some(b'"') | Some(b'\'') => Ok(PExpr::Str(self.string()?)),
            Some(c) if c.is_ascii_digit() || c == b'.' => {
                let st = self.i;
                while self.i < self.end && (self.b[self.i].is_ascii_alphanumeric() || self.b[self.i] == b'.') {
                    self.i += 1;
                }
                let t = &self.full[st..self.i];
                match t.parse::<f64>() {
                    Ok(n) => Ok(PExpr::Num(n)),
                    Err(_) => duration(t).map(|d| PExpr::Num(d as f64 / 1000.0)).ok_or_else(|| parsing_err(self.full, st, format!("invalid number [{t}]"))),
                }
            }
            Some(b'{') => self.selector_rest(String::new(), pos),
            _ => {
                let name = self.ident().ok_or_else(|| self.err("unexpected character"))?;
                if AGG_OPS.contains(&name.as_str()) {
                    let (mut by, mut without) = self.grouping()?;
                    if self.eat(b'(') {
                        let first = self.expr(0)?;
                        let (param, e) = if self.eat(b',') { (Some(Box::new(first)), self.expr(0)?) } else { (None, first) };
                        if !self.eat(b')') {
                            return Err(self.err("expected ')'"));
                        }
                        if by.is_empty() && !without {
                            let g = self.grouping()?;
                            by = g.0;
                            without = g.1;
                        }
                        return Ok(PExpr::Agg { op: name, param, expr: Box::new(e), by, without, pos });
                    }
                }
                if self.peek() == Some(b'(') {
                    self.i += 1;
                    let mut args = Vec::new();
                    if !self.eat(b')') {
                        loop {
                            args.push(self.expr(0)?);
                            if self.eat(b',') {
                                continue;
                            }
                            if !self.eat(b')') {
                                return Err(self.err("expected ')'"));
                            }
                            break;
                        }
                    }
                    return Ok(PExpr::Call { name, args, pos });
                }
                self.selector_rest(name, pos)
            }
        }
    }

    fn binop(&mut self) -> Option<(&'static str, u8)> {
        self.ws();
        let rest = &self.full[self.i..self.end];
        for (op, p) in [("==", 1), ("!=", 1), (">=", 1), ("<=", 1), (">", 1), ("<", 1), ("+", 2), ("-", 2), ("*", 3), ("/", 3), ("%", 3), ("^", 4)] {
            if rest.starts_with(op) {
                return Some((op, p));
            }
        }
        None
    }

    fn expr(&mut self, min: u8) -> Result<PExpr> {
        let mut l = self.primary()?;
        while let Some((op, p)) = self.binop() {
            if p < min {
                break;
            }
            self.i += op.len();
            // `bool` modifier of comparisons.
            let save = self.i;
            let bool_mod = self.ident().as_deref() == Some("bool");
            if !bool_mod {
                self.i = save;
            }
            let r = self.expr(if op == "^" { p } else { p + 1 })?;
            l = PExpr::Bin { op: op.to_string(), l: Box::new(l), r: Box::new(r), bool_mod };
        }
        Ok(l)
    }
}

/// Parse `PROMQL <params> [name=](<expr>)` in `full[from..to]`.
pub fn parse(full: &str, from: usize, to: usize, kw_pos: usize, params: &Params) -> Result<Promql> {
    let mut p = P { full, b: full.as_bytes(), i: from, end: to, params };
    let mut out = Promql { index: vec!["metrics-*".into()], step: None, buckets: None, start: None, end: None, scrape_interval: 60_000, name: String::new(), expr: PExpr::Num(0.0), pos: kw_pos };
    loop {
        p.ws();
        let save = p.i;
        let Some(word) = p.ident() else { break };
        p.ws();
        if p.i < p.end && p.b[p.i] == b'=' && p.b.get(p.i + 1) != Some(&b'=') && p.b.get(p.i + 1) != Some(&b'~') {
            p.i += 1;
            p.ws();
            if p.peek() == Some(b'(') {
                // `name=(expr)`.
                p.i += 1;
                let st = p.i;
                let e = p.expr(0)?;
                if !p.eat(b')') {
                    return Err(p.err("expected ')'"));
                }
                p.ws();
                if p.i < p.end {
                    return Err(p.err(format!("mismatched input '{}' expecting <EOF>", full[p.i..p.end].trim())));
                }
                let _ = st;
                out.name = word;
                out.expr = e;
                return Ok(out);
            }
            // An option value (up to whitespace).
            let st = p.i;
            let quoted = matches!(p.b.get(p.i), Some(b'"'));
            let value = if quoted {
                p.string()?
            } else {
                while p.i < p.end && !p.b[p.i].is_ascii_whitespace() {
                    p.i += 1;
                }
                full[st..p.i].to_string()
            };
            let bad = || parsing_err(full, st, format!("invalid value [{value}] for PROMQL parameter [{word}]"));
            match word.as_str() {
                "index" => out.index = value.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
                "step" => out.step = Some(duration(&value).ok_or_else(bad)?),
                "buckets" => out.buckets = Some(value.parse().map_err(|_| bad())?),
                "scrape_interval" => out.scrape_interval = duration(&value).ok_or_else(bad)?,
                "start" | "end" => {
                    let t = super::func::parse_date_str(&value).ok_or_else(bad)?;
                    if word == "start" {
                        out.start = Some(t);
                    } else {
                        out.end = Some(t);
                    }
                }
                other => return Err(parsing_err(full, save, format!("Unknown parameter [{other}]"))),
            }
            continue;
        }
        p.i = save;
        break;
    }
    // The bare expression.
    let st = p.i;
    let e = p.expr(0)?;
    p.ws();
    if p.i < p.end {
        return Err(p.err(format!("mismatched input '{}' expecting <EOF>", full[p.i..p.end].trim())));
    }
    out.name = full[st..p.i].trim().to_string();
    out.expr = e;
    Ok(out)
}

/// What the compiled pipeline needs: label matchers, the grouping labels
/// and whether an outer aggregation exists.
struct Compiled {
    matchers: Vec<Matcher>,
    by: Vec<String>,
    aggregated: bool,
    metrics: Vec<String>,
}

fn col(n: &str) -> Expr {
    Expr::Col(n.to_string(), 0)
}

fn compile(full: &str, e: &PExpr, c: &mut Compiled, in_agg: bool) -> Result<Expr> {
    Ok(match e {
        PExpr::Num(n) => Expr::Lit(Val::Double(*n)),
        PExpr::Str(s) => Expr::Lit(Val::Str(s.clone())),
        PExpr::Sel { metric, matchers, .. } => {
            c.matchers.extend(matchers.iter().cloned());
            c.metrics.push(metric.clone());
            // An instant-vector selector outside an aggregation or range
            // function is each series' latest sample per step.
            if in_agg { col(metric) } else { Expr::Call("LAST_OVER_TIME".into(), vec![col(metric)], 0) }
        }
        PExpr::Neg(a) => Expr::Neg(Box::new(compile(full, a, c, in_agg)?)),
        PExpr::Bin { op, l, r, .. } => {
            let (a, b) = (compile(full, l, c, in_agg)?, compile(full, r, c, in_agg)?);
            let bop = match op.as_str() {
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::Div,
                "%" => BinOp::Mod,
                "==" => BinOp::Eq,
                "!=" => BinOp::Neq,
                ">" => BinOp::Gt,
                "<" => BinOp::Lt,
                ">=" => BinOp::Ge,
                "<=" => BinOp::Le,
                "^" => return Ok(Expr::Call("POW".into(), vec![a, b], 0)),
                _ => return Err(verification_err(full, 0, format!("unsupported PromQL operator [{op}]"))),
            };
            Expr::Bin(bop, Box::new(a), Box::new(b))
        }
        PExpr::Call { name, args, pos } => {
            if let Some(f) = range_fn(name) {
                // `quantile_over_time(q, v)` → `PERCENTILE_OVER_TIME(v, q * 100)`.
                // (A range function reads its selector's samples.)
                if name == "quantile_over_time" && args.len() == 2 {
                    let q = compile(full, &args[0], c, true)?;
                    let v = compile(full, &args[1], c, true)?;
                    return Ok(Expr::Call(f.into(), vec![v, Expr::Bin(BinOp::Mul, Box::new(q), Box::new(Expr::Lit(Val::Double(100.0))))], *pos));
                }
                let a: Vec<Expr> = args.iter().map(|x| compile(full, x, c, true)).collect::<Result<_>>()?;
                return Ok(Expr::Call(f.into(), a, *pos));
            }
            if let Some(f) = math_fn(name) {
                let mut a: Vec<Expr> = args.iter().map(|x| compile(full, x, c, in_agg)).collect::<Result<_>>()?;
                if f == "\u{0}id" {
                    return Ok(a.pop().unwrap_or(Expr::Lit(Val::Null)));
                }
                if f == "LOG2" {
                    return Ok(Expr::Call("LOG".into(), vec![Expr::Lit(Val::Double(2.0)), a.pop().unwrap_or(Expr::Lit(Val::Null))], *pos));
                }
                return Ok(Expr::Call(f.into(), a, *pos));
            }
            return Err(verification_err(full, *pos, format!("Unknown PromQL function [{name}]")));
        }
        PExpr::Agg { op, param, expr, by, pos, .. } => {
            c.aggregated = true;
            for l in by {
                if !c.by.contains(l) {
                    c.by.push(l.clone());
                }
            }
            let inner = compile(full, expr, c, true)?;
            let f = match op.as_str() {
                "sum" => "SUM",
                "avg" => "AVG",
                "min" => "MIN",
                "max" => "MAX",
                "count" => "COUNT",
                "stddev" => "STD_DEV",
                "stdvar" => "VARIANCE",
                "quantile" => {
                    let q = compile(full, param.as_deref().unwrap_or(&PExpr::Num(0.5)), c, true)?;
                    return Ok(Expr::Call("PERCENTILE".into(), vec![inner, Expr::Bin(BinOp::Mul, Box::new(q), Box::new(Expr::Lit(Val::Double(100.0))))], *pos));
                }
                other => return Err(verification_err(full, *pos, format!("unsupported PromQL aggregation [{other}]"))),
            };
            let call = Expr::Call(f.into(), vec![inner], *pos);
            // PromQL values are doubles.
            if f == "COUNT" {
                Expr::Cast(Box::new(call), "double".into())
            } else {
                call
            }
        }
    })
}


/// Whether a matcher selects a series without its label (the empty value).
fn accepts_empty(m: &Matcher) -> bool {
    let re_empty = || regex::Regex::new(&format!("^(?:{})$", m.value)).is_ok_and(|r| r.is_match(""));
    match m.op {
        MatchOp::Eq => m.value.is_empty(),
        MatchOp::Ne => !m.value.is_empty(),
        MatchOp::Re => re_empty(),
        MatchOp::Nre => !re_empty(),
    }
}

/// A label matcher as a row condition with PromQL's semantics: a series
/// without the label has the empty value, so a matcher that accepts `""`
/// also selects it; regexes are RE2, fully anchored.
fn matcher_condition(m: &Matcher) -> Expr {
    let lab = col(&m.label);
    let absent = Expr::IsNull(Box::new(lab.clone()), false);
    let present = Expr::IsNull(Box::new(lab.clone()), true);
    let or = |a: Expr, b: Expr| Expr::Bin(BinOp::Or, Box::new(a), Box::new(b));
    let and = |a: Expr, b: Expr| Expr::Bin(BinOp::And, Box::new(a), Box::new(b));
    let eq = |v: &str| Expr::Bin(BinOp::Eq, Box::new(lab.clone()), Box::new(Expr::Lit(Val::Str(v.to_string()))));
    let re = |v: &str| Expr::Call(func::PROMQL_RE.into(), vec![lab.clone(), Expr::Lit(Val::Str(v.to_string()))], 0);
    let matches_empty = |v: &str| regex::Regex::new(&format!("^(?:{v})$")).is_ok_and(|r| r.is_match(""));
    match m.op {
        MatchOp::Eq if m.value.is_empty() => or(absent, eq("")),
        MatchOp::Eq => eq(&m.value),
        MatchOp::Ne if m.value.is_empty() => and(present, Expr::Not(Box::new(eq("")))),
        MatchOp::Ne => or(absent, Expr::Not(Box::new(eq(&m.value)))),
        MatchOp::Re if matches_empty(&m.value) => or(absent, re(&m.value)),
        MatchOp::Re => re(&m.value),
        MatchOp::Nre if matches_empty(&m.value) => and(present, Expr::Not(Box::new(re(&m.value)))),
        MatchOp::Nre => or(absent, Expr::Not(Box::new(re(&m.value)))),
    }
}

/// Compile to an ES|QL pipeline and run it.
pub fn run(ctx: &mut super::exec::Ctx, q: &Promql) -> Result<super::exec::Table> {
    // `without(labels)`: grouped by every other label (dimension) of the
    // series; the output keeps those remaining labels.
    let mut q = q.clone();
    if has_without(&q.expr) {
        let node = ctx.node.ok_or_else(|| crate::error::EsError::internal("no node"))?;
        let dims = match super::exec::index_dimensions_as(node, &q.index, ctx.authorize.as_ref()) {
            Ok(d) => d,
            Err(e) if e.kind == "verification_exception" || e.kind == "index_not_found_exception" => Vec::new(),
            Err(e) => return Err(e),
        };
        resolve_without(&mut q.expr, &dims);
    }
    if !needs_split(&q.expr) {
        return run_single(ctx, &q);
    }
    // Comparisons, and arithmetic between operands with different
    // selectors: each vector operand runs as its own pipeline (with only
    // its own label matchers) and the results are matched per step and
    // label set.
    match eval(ctx, &q, &q.expr)? {
        PVal::Vector(v) => Ok(v.into_table(&q.name)),
        PVal::Scalar(_) => Err(verification_err(ctx.full, q.pos, "a PromQL expression without a selector must be evaluated against a series; scalar-only comparisons are not supported")),
    }
}

/// Whether any aggregation uses `without`.
fn has_without(e: &PExpr) -> bool {
    match e {
        PExpr::Agg { without, expr, param, .. } => *without || has_without(expr) || param.as_deref().is_some_and(has_without),
        PExpr::Bin { l, r, .. } => has_without(l) || has_without(r),
        PExpr::Neg(a) => has_without(a),
        PExpr::Call { args, .. } => args.iter().any(has_without),
        _ => false,
    }
}

/// Turn `without(labels)` into `by(<the other dimensions>)`.
fn resolve_without(e: &mut PExpr, dims: &[String]) {
    match e {
        PExpr::Agg { without, by, expr, param, .. } => {
            if *without {
                *by = dims.iter().filter(|d| !by.contains(d)).cloned().collect();
                *without = false;
            }
            resolve_without(expr, dims);
            if let Some(p) = param.as_deref_mut() {
                resolve_without(p, dims);
            }
        }
        PExpr::Bin { l, r, .. } => {
            resolve_without(l, dims);
            resolve_without(r, dims);
        }
        PExpr::Neg(a) => resolve_without(a, dims),
        PExpr::Call { args, .. } => args.iter_mut().for_each(|a| resolve_without(a, dims)),
        _ => {}
    }
}

fn is_comparison(op: &str) -> bool {
    matches!(op, "==" | "!=" | ">" | "<" | ">=" | "<=")
}

/// Whether the expression reads a selector.
fn has_sel(e: &PExpr) -> bool {
    match e {
        PExpr::Sel { .. } => true,
        PExpr::Agg { expr, param, .. } => has_sel(expr) || param.as_deref().is_some_and(has_sel),
        PExpr::Bin { l, r, .. } => has_sel(l) || has_sel(r),
        PExpr::Neg(a) => has_sel(a),
        PExpr::Call { args, .. } => args.iter().any(has_sel),
        _ => false,
    }
}

/// Every selector's label matchers (canonical text), in order.
fn selector_scopes(e: &PExpr, out: &mut Vec<String>) {
    match e {
        PExpr::Sel { matchers, .. } => {
            let mut m: Vec<String> = matchers.iter().map(|m| format!("{:?}\u{0}{}\u{0}{}", m.op, m.label, m.value)).collect();
            m.sort();
            out.push(m.join("\u{1}"));
        }
        PExpr::Agg { expr, param, .. } => {
            selector_scopes(expr, out);
            if let Some(p) = param.as_deref() {
                selector_scopes(p, out);
            }
        }
        PExpr::Bin { l, r, .. } => {
            selector_scopes(l, out);
            selector_scopes(r, out);
        }
        PExpr::Neg(a) => selector_scopes(a, out),
        PExpr::Call { args, .. } => args.iter().for_each(|a| selector_scopes(a, out)),
        _ => {}
    }
}

/// Whether the single-pipeline compilation would be wrong: a comparison
/// (it filters / yields 0-1 samples), or a binary operation whose operands
/// select series with different label matchers (one pipeline applies every
/// selector's matchers to all of its input).
fn needs_split(e: &PExpr) -> bool {
    match e {
        PExpr::Bin { op, l, r, .. } => {
            if is_comparison(op) || needs_split(l) || needs_split(r) {
                return true;
            }
            if has_sel(l) && has_sel(r) {
                let (mut a, mut b) = (Vec::new(), Vec::new());
                selector_scopes(l, &mut a);
                selector_scopes(r, &mut b);
                let mut all = a;
                all.extend(b);
                all.sort();
                all.dedup();
                return all.len() > 1;
            }
            false
        }
        PExpr::Agg { expr, param, .. } => needs_split(expr) || param.as_deref().is_some_and(needs_split),
        PExpr::Neg(a) => needs_split(a),
        PExpr::Call { args, .. } => args.iter().any(needs_split),
        _ => false,
    }
}

/// An evaluated PromQL value.
enum PVal {
    Scalar(f64),
    Vector(PVector),
}

/// An instant vector per step: the label columns and, per (step, labels),
/// the sample.
struct PVector {
    labels: Vec<super::exec::Column>,
    rows: Vec<(Val, Vec<Val>, f64)>,
}

impl PVector {
    fn key(step: &Val, labels: &[Val]) -> (Option<i64>, Vec<String>) {
        (step.as_i64(), labels.iter().map(|v| v.to_text()).collect())
    }

    fn into_table(self, name: &str) -> super::exec::Table {
        use super::exec::{Column, Table};
        let mut cols = vec![Column { name: name.to_string(), ty: "double".into() }, Column { name: "step".into(), ty: "date".into() }];
        cols.extend(self.labels);
        let rows = self
            .rows
            .into_iter()
            .map(|(step, labels, v)| {
                let mut r = vec![Val::Double(v), step];
                r.extend(labels);
                r
            })
            .collect();
        Table { cols, rows }
    }
}

/// Evaluate `e` (some part of the expression needs the split evaluation).
fn eval(ctx: &mut super::exec::Ctx, q: &Promql, e: &PExpr) -> Result<PVal> {
    if !has_sel(e) {
        return Ok(PVal::Scalar(scalar(ctx.full, q.pos, e)?));
    }
    if !needs_split(e) {
        // A vector operand: its own pipeline, with only its own matchers.
        let sub = Promql { expr: e.clone(), name: "\u{0}v".into(), ..q.clone() };
        let t = run_single(ctx, &sub)?;
        let labels: Vec<super::exec::Column> = t.cols.iter().skip(2).cloned().collect();
        let rows = t
            .rows
            .into_iter()
            .filter_map(|mut r| {
                let v = r[0].as_f64()?;
                let step = std::mem::replace(&mut r[1], Val::Null);
                Some((step, r.split_off(2), v))
            })
            .collect();
        return Ok(PVal::Vector(PVector { labels, rows }));
    }
    match e {
        PExpr::Neg(a) => Ok(match eval(ctx, q, a)? {
            PVal::Scalar(x) => PVal::Scalar(-x),
            PVal::Vector(mut v) => {
                v.rows.iter_mut().for_each(|r| r.2 = -r.2);
                PVal::Vector(v)
            }
        }),
        PExpr::Bin { op, l, r, bool_mod } => {
            let (a, b) = (eval(ctx, q, l)?, eval(ctx, q, r)?);
            combine(ctx.full, q.pos, op, *bool_mod, a, b)
        }
        _ => Err(verification_err(
            ctx.full,
            q.pos,
            "PromQL comparisons, and binary operations between differently filtered selectors, are supported only at the top level of the expression, not inside an aggregation or function",
        )),
    }
}

/// A selector-free (scalar) expression.
fn scalar(full: &str, pos: usize, e: &PExpr) -> Result<f64> {
    Ok(match e {
        PExpr::Num(n) => *n,
        PExpr::Neg(a) => -scalar(full, pos, a)?,
        PExpr::Bin { op, l, r, bool_mod } if !is_comparison(op) || *bool_mod => {
            let (a, b) = (scalar(full, pos, l)?, scalar(full, pos, r)?);
            if is_comparison(op) {
                if cmp(op, a, b) { 1.0 } else { 0.0 }
            } else {
                arith(full, pos, op, a, b)?
            }
        }
        PExpr::Bin { .. } => return Err(verification_err(full, pos, "comparisons between scalars must use BOOL modifier")),
        _ => return Err(verification_err(full, pos, "unsupported scalar PromQL expression")),
    })
}

fn cmp(op: &str, a: f64, b: f64) -> bool {
    match op {
        "==" => a == b,
        "!=" => a != b,
        ">" => a > b,
        "<" => a < b,
        ">=" => a >= b,
        _ => a <= b,
    }
}

fn arith(full: &str, pos: usize, op: &str, a: f64, b: f64) -> Result<f64> {
    Ok(match op {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => a / b,
        "%" => a % b,
        "^" => a.powf(b),
        other => return Err(verification_err(full, pos, format!("unsupported PromQL operator [{other}]"))),
    })
}

/// A binary operation on evaluated operands (Prometheus semantics):
/// arithmetic maps / matches samples; a comparison without `bool` keeps
/// the samples for which it holds (with the vector's value), with `bool`
/// every sample becomes 0 or 1. Two vectors match one-to-one on step and
/// their (identical) label set.
fn combine(full: &str, pos: usize, op: &str, bool_mod: bool, a: PVal, b: PVal) -> Result<PVal> {
    let comparison = is_comparison(op);
    // The sample a matched pair yields, if kept.
    let apply = |x: f64, y: f64, keep: f64| -> Result<Option<f64>> {
        Ok(if comparison {
            let holds = cmp(op, x, y);
            if bool_mod {
                Some(if holds { 1.0 } else { 0.0 })
            } else {
                holds.then_some(keep)
            }
        } else {
            Some(arith(full, pos, op, x, y)?)
        })
    };
    Ok(match (a, b) {
        (PVal::Scalar(x), PVal::Scalar(y)) => {
            if comparison && !bool_mod {
                return Err(verification_err(full, pos, "comparisons between scalars must use BOOL modifier"));
            }
            PVal::Scalar(apply(x, y, x)?.unwrap_or(f64::NAN))
        }
        (PVal::Vector(mut v), PVal::Scalar(y)) => {
            let mut rows = Vec::with_capacity(v.rows.len());
            for (s, l, x) in v.rows.drain(..) {
                if let Some(out) = apply(x, y, x)? {
                    rows.push((s, l, out));
                }
            }
            v.rows = rows;
            PVal::Vector(v)
        }
        (PVal::Scalar(x), PVal::Vector(mut v)) => {
            let mut rows = Vec::with_capacity(v.rows.len());
            for (s, l, y) in v.rows.drain(..) {
                if let Some(out) = apply(x, y, y)? {
                    rows.push((s, l, out));
                }
            }
            v.rows = rows;
            PVal::Vector(v)
        }
        (PVal::Vector(lv), PVal::Vector(rv)) => {
            // One-to-one matching on the same label set (in any order).
            let mut ln: Vec<&str> = lv.labels.iter().map(|c| c.name.as_str()).collect();
            let mut rn: Vec<&str> = rv.labels.iter().map(|c| c.name.as_str()).collect();
            ln.sort_unstable();
            rn.sort_unstable();
            if ln != rn {
                return Err(verification_err(
                    full,
                    pos,
                    format!(
                        "PromQL vector matching between operands with different label sets ([{}] and [{}]) is not supported; group both operands by the same labels",
                        ln.join(", "),
                        rn.join(", ")
                    ),
                ));
            }
            let order: Vec<usize> = lv.labels.iter().map(|c| rv.labels.iter().position(|x| x.name == c.name).unwrap_or(0)).collect();
            let mut right: std::collections::HashMap<(Option<i64>, Vec<String>), f64> = std::collections::HashMap::new();
            for (s, l, y) in &rv.rows {
                let reordered: Vec<Val> = order.iter().map(|&i| l.get(i).cloned().unwrap_or(Val::Null)).collect();
                right.insert(PVector::key(s, &reordered), *y);
            }
            let mut rows = Vec::with_capacity(lv.rows.len());
            for (s, l, x) in lv.rows {
                let Some(&y) = right.get(&PVector::key(&s, &l)) else { continue };
                if let Some(out) = apply(x, y, x)? {
                    rows.push((s, l, out));
                }
            }
            PVal::Vector(PVector { labels: lv.labels, rows })
        }
    })
}

/// The single-pipeline compilation (no comparison, and every selector of
/// a binary operation has the same matchers).
fn run_single(ctx: &mut super::exec::Ctx, q: &Promql) -> Result<super::exec::Table> {
    use super::exec::{Column, Table};
    let mut c = Compiled { matchers: vec![], by: vec![], aggregated: false, metrics: vec![] };
    let agg = compile(ctx.full, &q.expr, &mut c, false)?;
    let step = q
        .step
        .or_else(|| match (q.start, q.end, q.buckets) {
            (Some(s), Some(e), b) => Some(((e - s) / b.unwrap_or(100).max(1)).max(1)),
            _ => None,
        })
        .unwrap_or(q.scrape_interval);
    let value_ty = "double".to_string();
    let empty = |by: &[String]| -> Table {
        let mut cols = vec![Column { name: q.name.clone(), ty: value_ty.clone() }, Column { name: "step".into(), ty: "date".into() }];
        if c.aggregated {
            cols.extend(by.iter().map(|b| Column { name: b.clone(), ty: "keyword".into() }));
        } else {
            cols.push(Column { name: "_timeseries".into(), ty: "keyword".into() });
        }
        Table { cols, rows: vec![] }
    };
    // Unmapped metrics select no series.
    let node = ctx.node.ok_or_else(|| crate::error::EsError::internal("no node"))?;
    let cols = match super::exec::index_columns_as(node, &q.index, ctx.authorize.as_ref()) {
        Ok(c) => c,
        Err(e) if e.kind == "verification_exception" || e.kind == "index_not_found_exception" => return Ok(empty(&c.by)),
        Err(e) => return Err(e),
    };
    if c.metrics.iter().any(|m| !cols.iter().any(|x| &x.name == m)) {
        return Ok(empty(&c.by));
    }
    // TS index | WHERE matchers | STATS value BY bucket, labels | EVAL step | WHERE value IS NOT NULL | KEEP
    let mut cmds: Vec<Cmd> = vec![Cmd::From { indices: q.index.clone(), metadata: vec![], pos: q.pos, ts: true, subqueries: vec![] }];
    let ts = col("@timestamp");
    if let Some(s) = q.start {
        cmds.push(Cmd::Where(Expr::Bin(BinOp::Ge, Box::new(ts.clone()), Box::new(Expr::Lit(Val::Date(s)))), q.pos));
    }
    if let Some(e) = q.end {
        cmds.push(Cmd::Where(Expr::Bin(BinOp::Le, Box::new(ts.clone()), Box::new(Expr::Lit(Val::Date(e)))), q.pos));
    }
    for m in &c.matchers {
        // A label no index maps is absent from every series: its matcher
        // keeps them all when it accepts the empty value, else none.
        if !cols.iter().any(|x| x.name == m.label) {
            if !accepts_empty(m) {
                cmds.push(Cmd::Where(Expr::Lit(Val::Bool(false)), q.pos));
            }
            continue;
        }
        cmds.push(Cmd::Where(matcher_condition(m), q.pos));
    }
    let bucket = "\u{0}step".to_string();
    let mut by = vec![Field { name: bucket.clone(), expr: Expr::Call("BUCKET".into(), vec![ts.clone(), Expr::Lit(Val::Dur(step))], q.pos), filter: None, pos: q.pos }];
    for l in &c.by {
        by.push(Field { name: l.clone(), expr: col(l), filter: None, pos: q.pos });
    }
    cmds.push(Cmd::Stats { aggs: vec![Field { name: q.name.clone(), expr: agg, filter: None, pos: q.pos }], by, inline: false, pos: q.pos });
    cmds.push(Cmd::Eval(vec![Field { name: "step".into(), expr: Expr::Bin(BinOp::Add, Box::new(col(&bucket)), Box::new(Expr::Lit(Val::Dur(step)))), filter: None, pos: q.pos }]));
    cmds.push(Cmd::Where(Expr::IsNull(Box::new(col(&q.name)), true), q.pos));
    let mut keep = vec![q.name.clone(), "step".into()];
    if c.aggregated {
        keep.extend(c.by.iter().cloned());
    } else {
        keep.push("_timeseries".into());
    }
    cmds.push(Cmd::Keep(keep, q.pos));
    // PROMQL resolves its own fields (unmapped_fields does not apply).
    let saved = ctx.settings.unmapped;
    ctx.settings.unmapped = super::settings::Unmapped::Default;
    let r = super::exec::run_sub_pipeline(ctx, &cmds);
    ctx.settings.unmapped = saved;
    let mut t = r?;
    // PromQL values are doubles.
    if let Some(i) = t.cols.iter().position(|x| x.name == q.name) {
        if t.cols[i].ty != "double" {
            for r in t.rows.iter_mut() {
                r[i] = match &r[i] {
                    v if v.is_null() => Val::Null,
                    v => v.as_f64().map(Val::Double).unwrap_or(Val::Null),
                };
            }
            t.cols[i].ty = "double".into();
        }
    }
    for b in &c.by {
        if let Some(i) = t.cols.iter().position(|x| &x.name == b) {
            if t.cols[i].ty == "text" {
                t.cols[i].ty = "keyword".into();
            }
        }
    }
    Ok(t)
}

#[cfg(test)]
mod semantics_tests {
    use super::*;
    use crate::node::Node;
    use serde_json::json;

    /// A time series index `name` with keyword dimensions `dims` and double
    /// gauges `metrics`, holding `docs`; returns the hour its data starts.
    fn ts_index(node: &Node, name: &str, dims: &[&str], metrics: &[&str], docs: &[(i64, Value)], base: i64) {
        let now = crate::date::now_millis();
        let mut props = serde_json::Map::new();
        props.insert("@timestamp".into(), json!({"type": "date"}));
        for d in dims {
            props.insert(d.to_string(), json!({"type": "keyword", "time_series_dimension": true}));
        }
        for m in metrics {
            props.insert(m.to_string(), json!({"type": "double", "time_series_metric": "gauge"}));
        }
        node.create_index(name, &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": dims, "index.number_of_replicas": 0,
                "index.time_series.start_time": crate::ilm::iso_date(base - 3_600_000), "index.time_series.end_time": crate::ilm::iso_date(now + 3_600_000)},
            "mappings": {"properties": props}}))
            .unwrap();
        let svc = node.get_index(name).unwrap();
        let reqs = docs
            .iter()
            .map(|(at, d)| {
                let mut d = d.clone();
                d["@timestamp"] = json!(crate::ilm::iso_date(base + at));
                crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }
            })
            .collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }

    fn base() -> i64 {
        let now = crate::date::now_millis();
        now - now.rem_euclid(3_600_000) - 3_600_000
    }

    /// `(value, step, labels...)` rows of a PROMQL query, sorted.
    fn promql(node: &Node, q: &str) -> Result<Vec<(f64, i64, Vec<String>)>> {
        let out = crate::esql::run(node, q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() })?;
        let mut rows: Vec<(f64, i64, Vec<String>)> = out
            .table
            .rows
            .iter()
            .map(|r| (r[0].as_f64().unwrap_or(f64::NAN), r[1].as_i64().unwrap_or(0), r[2..].iter().map(|v| v.to_text()).collect()))
            .collect();
        rows.sort_by(|a, b| (a.1, &a.2).cmp(&(b.1, &b.2)));
        Ok(rows)
    }

    /// Review finding 188: `without(instance)` groups by every other label
    /// (job), keeping the jobs apart and the job label in the output; `by
    /// (job)` and the ungrouped sum are the controls.
    #[test]
    fn without_groups_by_the_remaining_labels() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let docs: Vec<(i64, Value)> = [("a", "i1", 1.0), ("a", "i2", 2.0), ("b", "i1", 10.0), ("b", "i2", 20.0)]
            .iter()
            .map(|(j, i, v)| (30_000, json!({"job": j, "instance": i, "gauge": v})))
            .collect();
        ts_index(&node, "pm188", &["job", "instance"], &["gauge"], &docs, b);
        let q = |e: &str| format!("PROMQL index=pm188 step=1m start={b} end={} v=({e})", b + 59_999);
        let without = promql(&node, &q("sum without(instance)(gauge)")).unwrap();
        let by = promql(&node, &q("sum by(job)(gauge)")).unwrap();
        assert_eq!(without.iter().map(|r| (r.0, r.2.clone())).collect::<Vec<_>>(), vec![(3.0, vec!["a".to_string()]), (30.0, vec!["b".to_string()])], "{without:?}");
        assert_eq!(without, by);
        let all = promql(&node, &q("sum(gauge)")).unwrap();
        assert_eq!(all.iter().map(|r| r.0).collect::<Vec<_>>(), vec![33.0]);
    }

    /// Review finding 189: each operand keeps its own selector matchers:
    /// errors / all is 0.1 (not 1), disjoint selectors add up, grouped
    /// operands match per label set, and identical selectors stay 1.
    #[test]
    fn binary_operands_keep_their_own_selectors() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let docs: Vec<(i64, Value)> = [("500", "i1", 10.0), ("200", "i1", 50.0), ("200", "i2", 40.0)]
            .iter()
            .map(|(s, i, v)| (30_000, json!({"status": s, "instance": i, "requests": v})))
            .collect();
        ts_index(&node, "pm189", &["status", "instance"], &["requests"], &docs, b);
        let q = |e: &str| format!("PROMQL index=pm189 step=1m start={b} end={} v=({e})", b + 59_999);
        let ratio = promql(&node, &q("sum(requests{status=\"500\"}) / sum(requests)")).unwrap();
        assert_eq!(ratio.len(), 1, "{ratio:?}");
        assert!((ratio[0].0 - 0.1).abs() < 1e-12, "{ratio:?}");
        let both = promql(&node, &q("sum(requests{status=\"500\"}) + sum(requests{status=\"200\"})")).unwrap();
        assert_eq!(both.iter().map(|r| r.0).collect::<Vec<_>>(), vec![100.0], "disjoint selectors");
        let same = promql(&node, &q("sum(requests) / sum(requests)")).unwrap();
        assert_eq!(same.iter().map(|r| r.0).collect::<Vec<_>>(), vec![1.0]);
        let grouped = promql(&node, &q("sum by(instance)(requests{status=\"500\"}) / sum by(instance)(requests)")).unwrap();
        assert_eq!(grouped.len(), 1, "only i1 has errors: {grouped:?}");
        assert_eq!(grouped[0].2, vec!["i1".to_string()]);
        assert!((grouped[0].0 - 10.0 / 60.0).abs() < 1e-12, "{grouped:?}");
        // Unsupported matching is refused, not computed wrongly.
        assert!(promql(&node, &q("sum by(instance)(requests{status=\"500\"}) / sum(requests)")).is_err());
    }

    /// Review finding 190: a comparison without `bool` keeps only the
    /// samples for which it holds, with their values; with `bool` every
    /// sample becomes 0 or 1 (doubles); a step without observations has no
    /// sample either way.
    #[test]
    fn comparisons_filter_or_yield_bool_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let docs = vec![(30_000, json!({"host": "h", "gauge": 50.0})), (90_000, json!({"host": "h", "gauge": 150.0}))];
        ts_index(&node, "pm190", &["host"], &["gauge"], &docs, b);
        let q = |e: &str| format!("PROMQL index=pm190 step=1m start={b} end={} v=({e})", b + 3 * 60_000 - 1);
        let filtered = promql(&node, &q("sum(gauge) > 100")).unwrap();
        assert_eq!(filtered.iter().map(|r| r.0).collect::<Vec<_>>(), vec![150.0], "{filtered:?}");
        let flags = promql(&node, &q("sum(gauge) > bool 100")).unwrap();
        assert_eq!(flags.iter().map(|r| r.0).collect::<Vec<_>>(), vec![0.0, 1.0], "{flags:?}");
        let lower = promql(&node, &q("sum(gauge) <= 100")).unwrap();
        assert_eq!(lower.iter().map(|r| r.0).collect::<Vec<_>>(), vec![50.0]);
        let scalar_left = promql(&node, &q("100 < sum(gauge)")).unwrap();
        assert_eq!(scalar_left.iter().map(|r| r.0).collect::<Vec<_>>(), vec![150.0]);
    }

    /// Review finding 478: a bare selector (and scalar math over it) is
    /// each series' latest sample per step; last_over_time is the control.
    #[test]
    fn bare_selectors_answer_per_series() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let docs: Vec<(i64, Value)> = vec![(10_000, json!({"host": "h1", "temperature": 20.0})), (40_000, json!({"host": "h1", "temperature": -21.0})), (30_000, json!({"host": "h2", "temperature": 30.0}))];
        ts_index(&node, "pm478", &["host"], &["temperature"], &docs, b);
        let q = |e: &str| format!("PROMQL index=pm478 step=1m start={b} end={} v=({e})", b + 59_999);
        let bare = promql(&node, &q("temperature")).unwrap();
        let control = promql(&node, &q("last_over_time(temperature[1m])")).unwrap();
        assert_eq!(bare.iter().map(|r| r.0).collect::<Vec<_>>(), vec![-21.0, 30.0], "{bare:?}");
        assert_eq!(bare.iter().map(|r| r.0).collect::<Vec<_>>(), control.iter().map(|r| r.0).collect::<Vec<_>>());
        let abs = promql(&node, &q("abs(temperature)")).unwrap();
        assert_eq!(abs.iter().map(|r| r.0).collect::<Vec<_>>(), vec![21.0, 30.0], "{abs:?}");
    }

    /// Review finding 479: a series without a label has the empty value: a
    /// matcher accepting "" selects it (`!=`, `=""`, `=~` / `!~`).
    #[test]
    fn matchers_select_series_without_the_label() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let docs: Vec<(i64, Value)> = vec![(30_000, json!({"host": "h1", "region": "us", "gauge": 10.0})), (30_000, json!({"host": "h2", "gauge": 20.0}))];
        ts_index(&node, "pm479", &["host", "region"], &["gauge"], &docs, b);
        let q = |e: &str| format!("PROMQL index=pm479 step=1m start={b} end={} v=({e})", b + 59_999);
        let sum = |e: &str| promql(&node, &q(e)).unwrap().iter().map(|r| r.0).collect::<Vec<_>>();
        assert_eq!(sum("sum(gauge{region!=\"us\"})"), vec![20.0]);
        assert_eq!(sum("sum(gauge{region=\"\"})"), vec![20.0]);
        assert_eq!(sum("sum(gauge{region=\"us\"})"), vec![10.0]);
        assert_eq!(sum("sum(gauge{region=~\"u.*|\"})"), vec![30.0]);
        assert_eq!(sum("sum(gauge{region!~\"u.*\"})"), vec![20.0]);
        assert_eq!(sum("sum(gauge{region!=\"\"})"), vec![10.0]);
        // A label no index maps.
        assert_eq!(sum("sum(gauge{zone!=\"x\"})"), vec![30.0]);
        assert!(sum("sum(gauge{zone=\"x\"})").is_empty());
    }

    /// Review finding 480: a delta-temporality counter rates its running
    /// total (5 every 10 s: rate 0.5/s, increase 10), a cumulative counter
    /// as before.
    #[test]
    fn delta_counters_rate_their_running_total() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let b = base();
        let now = crate::date::now_millis();
        node.create_index("ts480", &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["host", "temporality"], "index.number_of_replicas": 0,
                "index.time_series.temporality_field": "temporality",
                "index.time_series.start_time": crate::ilm::iso_date(b - 3_600_000), "index.time_series.end_time": crate::ilm::iso_date(now + 3_600_000)},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
                "temporality": {"type": "keyword", "time_series_dimension": true},
                "requests": {"type": "double", "time_series_metric": "counter"}}}})).unwrap();
        let svc = node.get_index("ts480").unwrap();
        let mut reqs = Vec::new();
        for (i, at) in [10_000, 20_000, 30_000].iter().enumerate() {
            reqs.push(json!({"@timestamp": crate::ilm::iso_date(b + at), "host": "d", "temporality": "delta", "requests": 5.0}));
            reqs.push(json!({"@timestamp": crate::ilm::iso_date(b + at), "host": "c", "temporality": "cumulative", "requests": 5.0 * (i + 1) as f64}));
        }
        let reqs = reqs.into_iter().map(|d| crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let out = crate::esql::run(&node, "TS ts480 | STATS r = SUM(RATE(requests)), i = SUM(INCREASE(requests)) BY host | SORT host", crate::esql::EsqlOptions::default()).unwrap();
        let rows: Vec<(f64, f64, String)> = out.table.rows.iter().map(|r| (r[0].as_f64().unwrap_or(f64::NAN), r[1].as_f64().unwrap_or(f64::NAN), r[2].to_text())).collect();
        assert_eq!(rows, vec![(0.5, 10.0, "c".to_string()), (0.5, 10.0, "d".to_string())]);
    }

}
