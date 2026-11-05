//! The `PROMQL` source command: a PromQL expression evaluated over time
//! series indices. The expression compiles to a time-series pipeline
//! (`TS index | WHERE <label matchers> | STATS <aggregation> BY step
//! bucket, labels`), with one `step` per evaluation step (the end of its
//! bucket).

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
    Bin { op: String, l: Box<PExpr>, r: Box<PExpr> },
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
            if self.ident().as_deref() != Some("bool") {
                self.i = save;
            }
            let r = self.expr(if op == "^" { p } else { p + 1 })?;
            l = PExpr::Bin { op: op.to_string(), l: Box::new(l), r: Box::new(r) };
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
            col(metric)
        }
        PExpr::Neg(a) => Expr::Neg(Box::new(compile(full, a, c, in_agg)?)),
        PExpr::Bin { op, l, r } => {
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
                if name == "quantile_over_time" && args.len() == 2 {
                    let q = compile(full, &args[0], c, in_agg)?;
                    let v = compile(full, &args[1], c, in_agg)?;
                    return Ok(Expr::Call(f.into(), vec![v, Expr::Bin(BinOp::Mul, Box::new(q), Box::new(Expr::Lit(Val::Double(100.0))))], *pos));
                }
                let a: Vec<Expr> = args.iter().map(|x| compile(full, x, c, in_agg)).collect::<Result<_>>()?;
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

/// Whether a regex is a plain literal (no metacharacters).
fn is_literal(s: &str) -> bool {
    !s.chars().any(|c| ".+*?()[]{}|^$\\".contains(c))
}

/// Compile to an ES|QL pipeline and run it.
pub fn run(ctx: &mut super::exec::Ctx, q: &Promql) -> Result<super::exec::Table> {
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
    let cols = match super::exec::index_columns(node, &q.index) {
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
        let lab = col(&m.label);
        let cond = match (&m.op, is_literal(&m.value)) {
            (MatchOp::Eq, _) => Expr::Bin(BinOp::Eq, Box::new(lab), Box::new(Expr::Lit(Val::Str(m.value.clone())))),
            (MatchOp::Ne, _) => Expr::Not(Box::new(Expr::Bin(BinOp::Eq, Box::new(lab), Box::new(Expr::Lit(Val::Str(m.value.clone())))))),
            (MatchOp::Re, true) => Expr::Bin(BinOp::Eq, Box::new(lab), Box::new(Expr::Lit(Val::Str(m.value.clone())))),
            (MatchOp::Re, false) => Expr::Like(Box::new(lab), vec![m.value.clone()], false, true),
            (MatchOp::Nre, _) => Expr::Like(Box::new(lab), vec![m.value.clone()], true, true),
        };
        cmds.push(Cmd::Where(cond, q.pos));
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
