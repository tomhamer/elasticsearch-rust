//! A small Painless subset for ingest `if` conditions and the `script`
//! processor, used when no full script engine has been registered
//! (see [`super::register_script_engine`]). Covers what ingest pipelines
//! typically do: `ctx` field reads/writes (`.`, `?.`, `[]`), locals
//! (`def x = ...`), `if/else`, `for (x : list)`, `return`, arithmetic,
//! string concatenation, comparisons, `instanceof`, ternary / elvis, list
//! and map literals, and common String/List/Map/Math methods. Anything
//! else is a `script_exception` (compile error).

use crate::error::EsError;
use serde_json::{Map, Number, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------- lexer

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    P(&'static str),
}

const PUNCTS: &[&str] = &[
    "?.", "?:", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--", "->", ".", "(", ")", "[", "]", "{", "}", ",", ";", ":", "?", "!", "<", ">", "=", "+",
    "-", "*", "/", "%",
];

fn compile_err(src: &str, msg: impl Into<String>) -> EsError {
    EsError::new(400, "script_exception", "compile error")
        .with("script_stack", Value::Array(vec![]))
        .with("script", src.to_string())
        .with("lang", "painless")
        .caused_by(EsError::illegal_argument(msg))
}

fn lex(src: &str) -> Result<Vec<Tok>, EsError> {
    let b: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < b.len() && b[i + 1] == '/' {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < b.len() && b[i + 1] == '*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        if c.is_ascii_digit() {
            let st = i;
            let mut float = false;
            while i < b.len() && (b[i].is_ascii_digit() || (b[i] == '.' && !float && i + 1 < b.len() && b[i + 1].is_ascii_digit())) {
                if b[i] == '.' {
                    float = true;
                }
                i += 1;
            }
            let s: String = b[st..i].iter().collect();
            // Type suffixes.
            if i < b.len() && matches!(b[i], 'L' | 'l') {
                i += 1;
            } else if i < b.len() && matches!(b[i], 'f' | 'F' | 'd' | 'D') {
                i += 1;
                float = true;
            }
            if float {
                out.push(Tok::Float(s.parse().map_err(|_| compile_err(src, format!("invalid number [{s}]")))?));
            } else {
                out.push(Tok::Int(s.parse().map_err(|_| compile_err(src, format!("invalid number [{s}]")))?));
            }
            continue;
        }
        if c == '\'' || c == '"' {
            let q = c;
            i += 1;
            let mut s = String::new();
            while i < b.len() && b[i] != q {
                if b[i] == '\\' && i + 1 < b.len() {
                    i += 1;
                    s.push(match b[i] {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        o => o,
                    });
                } else {
                    s.push(b[i]);
                }
                i += 1;
            }
            if i >= b.len() {
                return Err(compile_err(src, "unterminated string"));
            }
            i += 1;
            out.push(Tok::Str(s));
            continue;
        }
        if c.is_alphabetic() || c == '_' || c == '$' {
            let st = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_' || b[i] == '$') {
                i += 1;
            }
            out.push(Tok::Ident(b[st..i].iter().collect()));
            continue;
        }
        let rest: String = b[i..(i + 2).min(b.len())].iter().collect();
        match PUNCTS.iter().find(|p| rest.starts_with(**p)) {
            Some(p) => {
                out.push(Tok::P(p));
                i += p.chars().count();
            }
            None => return Err(compile_err(src, format!("unexpected character [{c}]"))),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- AST

#[derive(Clone, Debug)]
enum Expr {
    Lit(Value),
    Var(String),
    Field(Box<Expr>, String, bool),
    Index(Box<Expr>, Box<Expr>, bool),
    Call(Box<Expr>, String, Vec<Expr>, bool),
    Unary(&'static str, Box<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Elvis(Box<Expr>, Box<Expr>),
    InstanceOf(Box<Expr>, String),
    List(Vec<Expr>),
    MapLit(Vec<(Expr, Expr)>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Expr(Expr),
    Assign(Expr, Option<&'static str>, Expr),
    Decl(String, Option<Expr>),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    ForEach(String, Expr, Vec<Stmt>),
    Return(Option<Expr>),
}

struct Parser<'a> {
    t: Vec<Tok>,
    i: usize,
    src: &'a str,
    nest: crate::parse_depth::Nesting,
}

const TYPES: &[&str] = &["def", "var", "String", "int", "long", "double", "float", "boolean", "List", "Map", "Object", "Integer", "Long", "Double", "Boolean", "ArrayList", "HashMap", "short", "byte", "char", "Number"];

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.t.get(self.i + n)
    }
    fn is(&self, p: &str) -> bool {
        matches!(self.peek(), Some(Tok::P(x)) if *x == p)
    }
    fn eat(&mut self, p: &str) -> bool {
        if self.is(p) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, p: &str) -> Result<(), EsError> {
        if self.eat(p) {
            Ok(())
        } else {
            Err(self.err(format!("expected [{p}] but found [{}]", self.found())))
        }
    }
    fn found(&self) -> String {
        match self.peek() {
            None => "<EOF>".into(),
            Some(Tok::Int(n)) => n.to_string(),
            Some(Tok::Float(n)) => n.to_string(),
            Some(Tok::Str(s)) => format!("'{s}'"),
            Some(Tok::Ident(s)) => s.clone(),
            Some(Tok::P(p)) => p.to_string(),
        }
    }
    fn err(&self, m: String) -> EsError {
        compile_err(self.src, m)
    }
    fn ident(&mut self) -> Result<String, EsError> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.i += 1;
                Ok(s)
            }
            _ => Err(self.err(format!("expected identifier but found [{}]", self.found()))),
        }
    }
    fn is_ident(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(x)) if x == s)
    }

    fn program(&mut self) -> Result<Vec<Stmt>, EsError> {
        let mut out = Vec::new();
        while self.peek().is_some() {
            if self.eat(";") {
                continue;
            }
            out.push(self.stmt()?);
        }
        Ok(out)
    }

    fn block(&mut self) -> Result<Vec<Stmt>, EsError> {
        if self.eat("{") {
            let mut out = Vec::new();
            while !self.eat("}") {
                if self.peek().is_none() {
                    return Err(self.err("unterminated block".into()));
                }
                if self.eat(";") {
                    continue;
                }
                out.push(self.stmt()?);
            }
            Ok(out)
        } else {
            Ok(vec![self.stmt()?])
        }
    }

    fn is_decl(&self) -> bool {
        if let Some(Tok::Ident(t)) = self.peek() {
            if TYPES.contains(&t.as_str()) {
                // `Type name` or `Type<...> name` or `Type[] name`
                return matches!(self.peek_at(1), Some(Tok::Ident(_))) || matches!(self.peek_at(1), Some(Tok::P("<"))) || (matches!(self.peek_at(1), Some(Tok::P("["))) && matches!(self.peek_at(2), Some(Tok::P("]"))));
            }
        }
        false
    }

    fn skip_type(&mut self) -> Result<(), EsError> {
        self.ident()?;
        if self.eat("<") {
            let mut depth = 1;
            while depth > 0 {
                match self.peek() {
                    Some(Tok::P("<")) => depth += 1,
                    Some(Tok::P(">")) => depth -= 1,
                    None => return Err(self.err("bad generic type".into())),
                    _ => {}
                }
                self.i += 1;
            }
        }
        while self.is("[") && matches!(self.peek_at(1), Some(Tok::P("]"))) {
            self.i += 2;
        }
        Ok(())
    }

    fn stmt(&mut self) -> Result<Stmt, EsError> {
        self.nested(true, Self::stmt_unnested)
    }

    fn stmt_unnested(&mut self) -> Result<Stmt, EsError> {
        if self.is_ident("if") {
            self.i += 1;
            self.expect("(")?;
            let c = self.expr()?;
            self.expect(")")?;
            let then = self.block()?;
            let els = if self.is_ident("else") {
                self.i += 1;
                self.block()?
            } else {
                Vec::new()
            };
            return Ok(Stmt::If(c, then, els));
        }
        if self.is_ident("for") {
            self.i += 1;
            self.expect("(")?;
            if self.is_decl() {
                self.skip_type()?;
            }
            let name = self.ident()?;
            if !self.eat(":") {
                if !self.is_ident("in") {
                    return Err(self.err("only for-each loops are supported".into()));
                }
                self.i += 1;
            }
            let it = self.expr()?;
            self.expect(")")?;
            let body = self.block()?;
            return Ok(Stmt::ForEach(name, it, body));
        }
        if self.is_ident("return") {
            self.i += 1;
            if self.eat(";") || self.is("}") || self.peek().is_none() {
                return Ok(Stmt::Return(None));
            }
            let e = self.expr()?;
            self.eat(";");
            return Ok(Stmt::Return(Some(e)));
        }
        if self.is_decl() {
            self.skip_type()?;
            let name = self.ident()?;
            let init = if self.eat("=") { Some(self.expr()?) } else { None };
            self.end_stmt()?;
            return Ok(Stmt::Decl(name, init));
        }
        let e = self.expr()?;
        for op in ["=", "+=", "-=", "*=", "/="] {
            if self.eat(op) {
                let rhs = self.expr()?;
                self.end_stmt()?;
                let bin = match op {
                    "+=" => Some("+"),
                    "-=" => Some("-"),
                    "*=" => Some("*"),
                    "/=" => Some("/"),
                    _ => None,
                };
                return Ok(Stmt::Assign(e, bin, rhs));
            }
        }
        if self.eat("++") {
            self.end_stmt()?;
            return Ok(Stmt::Assign(e, Some("+"), Expr::Lit(Value::from(1))));
        }
        if self.eat("--") {
            self.end_stmt()?;
            return Ok(Stmt::Assign(e, Some("-"), Expr::Lit(Value::from(1))));
        }
        self.end_stmt()?;
        Ok(Stmt::Expr(e))
    }

    fn end_stmt(&mut self) -> Result<(), EsError> {
        if self.eat(";") || self.is("}") || self.peek().is_none() {
            return Ok(());
        }
        Err(self.err(format!("unexpected token [{}]", self.found())))
    }

    /// Runs `f` one nesting level deeper (see [`crate::parse_depth`]).
    fn nested<T>(&mut self, counted: bool, f: impl FnOnce(&mut Self) -> Result<T, EsError>) -> Result<T, EsError> {
        let levels = self.nest.snapshot();
        if !self.nest.enter(counted) {
            return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
        }
        let r = f(self);
        self.nest.restore(levels);
        r
    }

    fn expr(&mut self) -> Result<Expr, EsError> {
        self.nested(true, Self::expr_unnested)
    }

    fn expr_unnested(&mut self) -> Result<Expr, EsError> {
        let c = self.or()?;
        if self.eat("?:") {
            let d = self.expr()?;
            return Ok(Expr::Elvis(Box::new(c), Box::new(d)));
        }
        if self.eat("?") {
            let a = self.expr()?;
            self.expect(":")?;
            let b = self.expr()?;
            return Ok(Expr::Ternary(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }
    fn or(&mut self) -> Result<Expr, EsError> {
        let mut l = self.and()?;
        while self.eat("||") {
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.and()?;
            l = Expr::Or(Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn and(&mut self) -> Result<Expr, EsError> {
        let mut l = self.eq()?;
        while self.eat("&&") {
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.eq()?;
            l = Expr::And(Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn eq(&mut self) -> Result<Expr, EsError> {
        let mut l = self.rel()?;
        loop {
            let op = if self.eat("==") {
                "=="
            } else if self.eat("!=") {
                "!="
            } else {
                return Ok(l);
            };
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.rel()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }
    fn rel(&mut self) -> Result<Expr, EsError> {
        let mut l = self.add()?;
        loop {
            if self.is_ident("instanceof") {
                self.i += 1;
                if !self.nest.grow() {
                    return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
                }
                let t = self.ident()?;
                l = Expr::InstanceOf(Box::new(l), t);
                continue;
            }
            let op = if self.eat("<=") {
                "<="
            } else if self.eat(">=") {
                ">="
            } else if self.eat("<") {
                "<"
            } else if self.eat(">") {
                ">"
            } else {
                return Ok(l);
            };
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.add()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }
    fn add(&mut self) -> Result<Expr, EsError> {
        let mut l = self.mul()?;
        loop {
            let op = if self.eat("+") {
                "+"
            } else if self.eat("-") {
                "-"
            } else {
                return Ok(l);
            };
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.mul()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }
    fn mul(&mut self) -> Result<Expr, EsError> {
        let mut l = self.unary()?;
        loop {
            let op = if self.eat("*") {
                "*"
            } else if self.eat("/") {
                "/"
            } else if self.eat("%") {
                "%"
            } else {
                return Ok(l);
            };
            if !self.nest.grow() {
                return Err(self.err(format!("too many nested expressions or blocks (more than {} levels).", self.nest.max())));
            }
            let r = self.unary()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
    }
    fn unary(&mut self) -> Result<Expr, EsError> {
        self.nested(false, Self::unary_unnested)
    }

    fn unary_unnested(&mut self) -> Result<Expr, EsError> {
        if self.eat("!") {
            return Ok(Expr::Unary("!", Box::new(self.unary()?)));
        }
        if self.eat("-") {
            return Ok(Expr::Unary("-", Box::new(self.unary()?)));
        }
        if self.eat("+") {
            return self.unary();
        }
        // Casts: `(String) x`, `(int) x`.
        if self.is("(") {
            if let (Some(Tok::Ident(t)), Some(Tok::P(")"))) = (self.peek_at(1), self.peek_at(2)) {
                if TYPES.contains(&t.as_str()) {
                    let t = t.clone();
                    self.i += 3;
                    let e = self.unary()?;
                    return Ok(Expr::Call(Box::new(Expr::Var("$cast".into())), t, vec![e], false));
                }
            }
        }
        self.postfix()
    }
    fn postfix(&mut self) -> Result<Expr, EsError> {
        let mut e = self.primary()?;
        loop {
            let safe = if self.eat("?.") {
                true
            } else if self.eat(".") {
                false
            } else if self.is("[") {
                self.i += 1;
                let k = self.expr()?;
                self.expect("]")?;
                e = Expr::Index(Box::new(e), Box::new(k), false);
                continue;
            } else {
                return Ok(e);
            };
            let name = self.ident()?;
            if self.eat("(") {
                let args = self.args(")")?;
                e = Expr::Call(Box::new(e), name, args, safe);
            } else {
                e = Expr::Field(Box::new(e), name, safe);
            }
        }
    }
    fn args(&mut self, close: &str) -> Result<Vec<Expr>, EsError> {
        let mut out = Vec::new();
        if self.eat(close) {
            return Ok(out);
        }
        loop {
            out.push(self.expr()?);
            if self.eat(close) {
                return Ok(out);
            }
            self.expect(",")?;
        }
    }
    fn primary(&mut self) -> Result<Expr, EsError> {
        match self.peek().cloned() {
            Some(Tok::Int(n)) => {
                self.i += 1;
                Ok(Expr::Lit(Value::from(n)))
            }
            Some(Tok::Float(f)) => {
                self.i += 1;
                Ok(Expr::Lit(Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)))
            }
            Some(Tok::Str(s)) => {
                self.i += 1;
                Ok(Expr::Lit(Value::String(s)))
            }
            Some(Tok::Ident(s)) => {
                self.i += 1;
                match s.as_str() {
                    "true" => Ok(Expr::Lit(Value::Bool(true))),
                    "false" => Ok(Expr::Lit(Value::Bool(false))),
                    "null" => Ok(Expr::Lit(Value::Null)),
                    "new" => {
                        let t = self.ident()?;
                        if self.eat("<") {
                            while !self.eat(">") {
                                if self.peek().is_none() {
                                    return Err(self.err("bad generic type".into()));
                                }
                                self.i += 1;
                            }
                        }
                        self.expect("(")?;
                        let _ = self.args(")")?;
                        Ok(match t.as_str() {
                            "HashMap" | "TreeMap" | "LinkedHashMap" => Expr::MapLit(vec![]),
                            _ => Expr::List(vec![]),
                        })
                    }
                    _ => Ok(Expr::Var(s)),
                }
            }
            Some(Tok::P("(")) => {
                self.i += 1;
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            Some(Tok::P("[")) => {
                self.i += 1;
                if self.eat(":") {
                    self.expect("]")?;
                    return Ok(Expr::MapLit(vec![]));
                }
                if self.eat("]") {
                    return Ok(Expr::List(vec![]));
                }
                let first = self.expr()?;
                if self.eat(":") {
                    let v = self.expr()?;
                    let mut pairs = vec![(first, v)];
                    while self.eat(",") {
                        let k = self.expr()?;
                        self.expect(":")?;
                        let v = self.expr()?;
                        pairs.push((k, v));
                    }
                    self.expect("]")?;
                    return Ok(Expr::MapLit(pairs));
                }
                let mut items = vec![first];
                while self.eat(",") {
                    items.push(self.expr()?);
                }
                self.expect("]")?;
                Ok(Expr::List(items))
            }
            _ => Err(self.err(format!("unexpected token [{}]", self.found()))),
        }
    }
}

// ---------------------------------------------------------------- eval

fn rt_err(src: &str, kind: &str, msg: impl Into<String>) -> EsError {
    EsError::new(400, "script_exception", "runtime error")
        .with("script_stack", Value::Array(vec![]))
        .with("script", src.to_string())
        .with("lang", "painless")
        .caused_by(EsError::new(400, kind, msg))
}

enum Flow {
    Normal,
    Return(Value),
}

struct Env<'a> {
    src: &'a str,
    ctx: &'a mut Value,
    params: &'a Value,
    locals: HashMap<String, Value>,
    last: Value,
    steps: usize,
}

fn num(v: &Value) -> Option<Number> {
    match v {
        Value::Number(n) => Some(n.clone()),
        _ => None,
    }
}

fn jstr(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".into(),
        Value::Object(m) => {
            let parts: Vec<String> = m.iter().map(|(k, v)| format!("{k}={}", jstr(v))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(jstr).collect::<Vec<_>>().join(", ")),
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    return format!("{f:.1}");
                }
            }
            n.to_string()
        }
        Value::Bool(b) => b.to_string(),
    }
}

fn f2v(f: f64) -> Value {
    Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => i == j,
            _ => x.as_f64() == y.as_f64(),
        },
        _ => a == b,
    }
}

impl<'a> Env<'a> {
    fn err(&self, kind: &str, msg: impl Into<String>) -> EsError {
        rt_err(self.src, kind, msg)
    }

    fn truthy(&self, v: &Value) -> Result<bool, EsError> {
        match v {
            Value::Bool(b) => Ok(*b),
            other => Err(self.err("class_cast_exception", format!("Cannot cast {} to boolean", super::doc::type_name(other)))),
        }
    }

    fn run(&mut self, stmts: &[Stmt]) -> Result<Flow, EsError> {
        for s in stmts {
            self.steps += 1;
            if self.steps > 1_000_000 {
                return Err(self.err("circuit_breaking_exception", "The maximum number of statements that can be executed in a loop has been reached."));
            }
            match s {
                Stmt::Expr(e) => {
                    self.last = self.eval(e)?;
                }
                Stmt::Decl(n, init) => {
                    let v = match init {
                        Some(e) => self.eval(e)?,
                        None => Value::Null,
                    };
                    self.locals.insert(n.clone(), v);
                }
                Stmt::Assign(target, op, rhs) => {
                    let mut v = self.eval(rhs)?;
                    if let Some(op) = op {
                        let cur = self.eval(target)?;
                        v = self.bin(op, &cur, &v)?;
                    }
                    self.assign(target, v)?;
                }
                Stmt::If(c, a, b) => {
                    let cv = self.eval(c)?;
                    let f = if self.truthy(&cv)? { self.run(a)? } else { self.run(b)? };
                    if let Flow::Return(v) = f {
                        return Ok(Flow::Return(v));
                    }
                }
                Stmt::ForEach(name, it, body) => {
                    let items: Vec<Value> = match self.eval(it)? {
                        Value::Array(a) => a,
                        Value::Object(m) => m.keys().map(|k| Value::String(k.clone())).collect(),
                        Value::Null => return Err(self.err("null_pointer_exception", "cannot iterate over null")),
                        other => return Err(self.err("illegal_argument_exception", format!("Cannot iterate over [{}]", super::doc::type_name(&other)))),
                    };
                    for x in items {
                        self.locals.insert(name.clone(), x);
                        if let Flow::Return(v) = self.run(body)? {
                            return Ok(Flow::Return(v));
                        }
                    }
                }
                Stmt::Return(e) => {
                    let v = match e {
                        Some(e) => self.eval(e)?,
                        None => Value::Null,
                    };
                    return Ok(Flow::Return(v));
                }
            }
        }
        Ok(Flow::Normal)
    }

    /// Resolve an lvalue to (root var, key path).
    fn path_of(&mut self, e: &Expr) -> Result<(String, Vec<Value>), EsError> {
        match e {
            Expr::Var(n) => Ok((n.clone(), Vec::new())),
            Expr::Field(b, n, _) => {
                let (r, mut p) = self.path_of(b)?;
                p.push(Value::String(n.clone()));
                Ok((r, p))
            }
            Expr::Index(b, k, _) => {
                let (r, mut p) = self.path_of(b)?;
                let kv = self.eval(k)?;
                p.push(kv);
                Ok((r, p))
            }
            Expr::Call(b, m, args, _) if m == "get" && args.len() == 1 => {
                let (r, mut p) = self.path_of(b)?;
                let kv = self.eval(&args[0])?;
                p.push(kv);
                Ok((r, p))
            }
            _ => Err(self.err("illegal_argument_exception", "invalid assignment target")),
        }
    }

    fn root_mut(&mut self, name: &str) -> Result<&mut Value, EsError> {
        if name == "ctx" {
            return Ok(&mut *self.ctx);
        }
        if self.locals.contains_key(name) {
            return Ok(self.locals.get_mut(name).unwrap());
        }
        Err(self.err("illegal_argument_exception", format!("cannot resolve symbol [{name}]")))
    }

    /// Mutable reference to the value at an lvalue path (must exist).
    fn place(&mut self, e: &Expr) -> Result<&mut Value, EsError> {
        let (root, path) = self.path_of(e)?;
        let src = self.src;
        let mut cur = self.root_mut(&root)?;
        for k in &path {
            cur = step_mut(cur, k, false).map_err(|m| rt_err(src, "null_pointer_exception", m))?;
        }
        Ok(cur)
    }

    fn assign(&mut self, target: &Expr, v: Value) -> Result<(), EsError> {
        let (root, path) = self.path_of(target)?;
        if path.is_empty() {
            if root == "ctx" {
                return Err(self.err("illegal_argument_exception", "cannot assign to ctx"));
            }
            self.locals.insert(root, v);
            return Ok(());
        }
        let src = self.src;
        let mut cur = self.root_mut(&root)?;
        let (last, parents) = path.split_last().unwrap();
        for k in parents {
            cur = step_mut(cur, k, false).map_err(|m| rt_err(src, "null_pointer_exception", m))?;
        }
        match (cur, last) {
            (Value::Object(m), k) => {
                m.insert(jstr(k), v);
                Ok(())
            }
            (Value::Array(a), Value::Number(n)) => {
                let i = n.as_i64().unwrap_or(-1);
                if i < 0 || i as usize >= a.len() {
                    return Err(rt_err(src, "index_out_of_bounds_exception", format!("Index {i} out of bounds for length {}", a.len())));
                }
                a[i as usize] = v;
                Ok(())
            }
            (Value::Null, _) => Err(rt_err(src, "null_pointer_exception", "Cannot invoke \"Object.put\" because value is null")),
            (other, _) => Err(rt_err(src, "illegal_argument_exception", format!("Illegal list shortcut value [{}].", super::doc::type_name(other)))),
        }
    }

    fn lookup_var(&self, n: &str) -> Result<Value, EsError> {
        match n {
            "ctx" => Ok(self.ctx.clone()),
            "params" => Ok(self.params.clone()),
            _ => match self.locals.get(n) {
                Some(v) => Ok(v.clone()),
                None => Err(self.err("illegal_argument_exception", format!("cannot resolve symbol [{n}]"))),
            },
        }
    }

    /// Read access without cloning the whole ctx for `ctx.a.b` chains.
    fn read_path(&mut self, e: &Expr) -> Result<Option<Value>, EsError> {
        // Only plain chains rooted at ctx/params/local with non-safe or safe access.
        fn chain<'e>(e: &'e Expr, out: &mut Vec<(&'e Expr, bool)>) -> Option<&'e str> {
            match e {
                Expr::Var(n) => Some(n.as_str()),
                Expr::Field(b, _, s) | Expr::Index(b, _, s) => {
                    let r = chain(b, out)?;
                    out.push((e, *s));
                    Some(r)
                }
                _ => None,
            }
        }
        let mut steps = Vec::new();
        let Some(root) = chain(e, &mut steps) else { return Ok(None) };
        let keys: Vec<(Value, bool)> = {
            let mut ks = Vec::new();
            for (s, safe) in &steps {
                let k = match s {
                    Expr::Field(_, n, _) => Value::String(n.clone()),
                    Expr::Index(_, k, _) => self.eval(k)?,
                    _ => unreachable!(),
                };
                ks.push((k, *safe));
            }
            ks
        };
        let base: &Value = match root {
            "ctx" => &*self.ctx,
            "params" => self.params,
            n => match self.locals.get(n) {
                Some(v) => v,
                None => {
                    if keys.is_empty() {
                        return Err(self.err("illegal_argument_exception", format!("cannot resolve symbol [{n}]")));
                    }
                    return Ok(None);
                }
            },
        };
        let mut cur = base;
        for (k, safe) in &keys {
            match cur {
                Value::Null => {
                    if *safe {
                        return Ok(Some(Value::Null));
                    }
                    return Err(self.err("null_pointer_exception", format!("Cannot invoke \"Object.get(Object)\" because the value is null (accessing [{}])", jstr(k))));
                }
                Value::Object(m) => {
                    cur = match m.get(&jstr(k)) {
                        Some(v) => v,
                        None => return Ok(Some(Value::Null)),
                    }
                }
                Value::Array(a) => {
                    if let Value::String(s) = k {
                        if s == "length" || s == "size" {
                            return Ok(Some(Value::from(a.len())));
                        }
                    }
                    let i = k.as_i64().unwrap_or(-1);
                    if i < 0 || i as usize >= a.len() {
                        return Err(self.err("index_out_of_bounds_exception", format!("Index {i} out of bounds for length {}", a.len())));
                    }
                    cur = &a[i as usize];
                }
                Value::String(s) if jstr(k) == "length" => return Ok(Some(Value::from(s.chars().count()))),
                other => {
                    return Err(self.err("illegal_argument_exception", format!("Illegal list shortcut value [{}].", super::doc::type_name(other))));
                }
            }
        }
        Ok(Some(cur.clone()))
    }

    fn eval(&mut self, e: &Expr) -> Result<Value, EsError> {
        match e {
            Expr::Lit(v) => Ok(v.clone()),
            Expr::Var(n) => self.lookup_var(n),
            Expr::Field(..) | Expr::Index(..) => match self.read_path(e)? {
                Some(v) => Ok(v),
                None => {
                    let (b, k, safe) = match e {
                        Expr::Field(b, n, s) => (b, Value::String(n.clone()), *s),
                        Expr::Index(b, k, s) => {
                            let kv = self.eval(k)?;
                            (b, kv, *s)
                        }
                        _ => unreachable!(),
                    };
                    let base = self.eval(b)?;
                    match (&base, &k) {
                        (Value::Null, _) if safe => Ok(Value::Null),
                        (Value::Null, _) => Err(self.err("null_pointer_exception", "cannot access a field of null")),
                        (Value::Object(m), _) => Ok(m.get(&jstr(&k)).cloned().unwrap_or(Value::Null)),
                        (Value::Array(a), Value::Number(n)) => Ok(a.get(n.as_u64().unwrap_or(u64::MAX) as usize).cloned().unwrap_or(Value::Null)),
                        _ => Err(self.err("illegal_argument_exception", "illegal field access")),
                    }
                }
            },
            Expr::Unary(op, x) => {
                let v = self.eval(x)?;
                match *op {
                    "!" => Ok(Value::Bool(!self.truthy(&v)?)),
                    _ => match num(&v) {
                        Some(n) => Ok(match n.as_i64() {
                            Some(i) => Value::from(-i),
                            None => f2v(-n.as_f64().unwrap_or(0.0)),
                        }),
                        None => Err(self.err("class_cast_exception", "Cannot apply [-] to a non-number")),
                    },
                }
            }
            Expr::Bin(op, a, b) => {
                let x = self.eval(a)?;
                let y = self.eval(b)?;
                self.bin(op, &x, &y)
            }
            Expr::And(a, b) => {
                let x = self.eval(a)?;
                if !self.truthy(&x)? {
                    return Ok(Value::Bool(false));
                }
                let y = self.eval(b)?;
                Ok(Value::Bool(self.truthy(&y)?))
            }
            Expr::Or(a, b) => {
                let x = self.eval(a)?;
                if self.truthy(&x)? {
                    return Ok(Value::Bool(true));
                }
                let y = self.eval(b)?;
                Ok(Value::Bool(self.truthy(&y)?))
            }
            Expr::Ternary(c, a, b) => {
                let cv = self.eval(c)?;
                if self.truthy(&cv)? { self.eval(a) } else { self.eval(b) }
            }
            Expr::Elvis(a, b) => {
                let v = self.eval(a)?;
                if v.is_null() { self.eval(b) } else { Ok(v) }
            }
            Expr::InstanceOf(x, t) => {
                let v = self.eval(x)?;
                Ok(Value::Bool(match t.as_str() {
                    "String" | "CharSequence" => v.is_string(),
                    "List" | "ArrayList" | "Collection" => v.is_array(),
                    "Map" | "HashMap" => v.is_object(),
                    "Number" => v.is_number(),
                    "Integer" | "Long" | "Short" | "Byte" => v.is_i64() || v.is_u64(),
                    "Double" | "Float" => v.is_f64(),
                    "Boolean" => v.is_boolean(),
                    "Object" | "def" => !v.is_null(),
                    _ => false,
                }))
            }
            Expr::List(items) => {
                let mut out = Vec::new();
                for x in items {
                    out.push(self.eval(x)?);
                }
                Ok(Value::Array(out))
            }
            Expr::MapLit(pairs) => {
                let mut m = Map::new();
                for (k, v) in pairs {
                    let kk = self.eval(k)?;
                    let vv = self.eval(v)?;
                    m.insert(jstr(&kk), vv);
                }
                Ok(Value::Object(m))
            }
            Expr::Call(recv, name, args, safe) => self.call(recv, name, args, *safe),
        }
    }

    fn bin(&self, op: &str, x: &Value, y: &Value) -> Result<Value, EsError> {
        match op {
            "==" => return Ok(Value::Bool(values_eq(x, y))),
            "!=" => return Ok(Value::Bool(!values_eq(x, y))),
            _ => {}
        }
        if op == "+" && (x.is_string() || y.is_string()) {
            return Ok(Value::String(format!("{}{}", jstr(x), jstr(y))));
        }
        if matches!(op, "<" | ">" | "<=" | ">=") {
            if let (Value::String(a), Value::String(b)) = (x, y) {
                let o = a.cmp(b);
                return Ok(Value::Bool(match op {
                    "<" => o.is_lt(),
                    ">" => o.is_gt(),
                    "<=" => o.is_le(),
                    _ => o.is_ge(),
                }));
            }
        }
        let (Some(a), Some(b)) = (num(x), num(y)) else {
            if x.is_null() || y.is_null() {
                return Err(self.err("null_pointer_exception", format!("Cannot apply [{op}] to null")));
            }
            return Err(self.err("class_cast_exception", format!("Cannot apply [{op}] operation to types [{}] and [{}].", super::doc::type_name(x), super::doc::type_name(y))));
        };
        if let (Some(i), Some(j)) = (a.as_i64().filter(|_| !a.is_f64()), b.as_i64().filter(|_| !b.is_f64())) {
            let r = match op {
                "+" => i.checked_add(j),
                "-" => i.checked_sub(j),
                "*" => i.checked_mul(j),
                "/" => {
                    if j == 0 {
                        return Err(self.err("arithmetic_exception", "/ by zero"));
                    }
                    Some(i / j)
                }
                "%" => {
                    if j == 0 {
                        return Err(self.err("arithmetic_exception", "/ by zero"));
                    }
                    Some(i % j)
                }
                "<" => return Ok(Value::Bool(i < j)),
                ">" => return Ok(Value::Bool(i > j)),
                "<=" => return Ok(Value::Bool(i <= j)),
                ">=" => return Ok(Value::Bool(i >= j)),
                _ => None,
            };
            return r.map(Value::from).ok_or_else(|| self.err("arithmetic_exception", "integer overflow"));
        }
        let (p, q) = (a.as_f64().unwrap_or(0.0), b.as_f64().unwrap_or(0.0));
        Ok(match op {
            "+" => f2v(p + q),
            "-" => f2v(p - q),
            "*" => f2v(p * q),
            "/" => f2v(p / q),
            "%" => f2v(p % q),
            "<" => Value::Bool(p < q),
            ">" => Value::Bool(p > q),
            "<=" => Value::Bool(p <= q),
            ">=" => Value::Bool(p >= q),
            _ => return Err(self.err("illegal_argument_exception", format!("unknown operator [{op}]"))),
        })
    }

    fn call(&mut self, recv: &Expr, name: &str, args: &[Expr], safe: bool) -> Result<Value, EsError> {
        let mut av = Vec::with_capacity(args.len());
        for a in args {
            av.push(self.eval(a)?);
        }
        // Static calls and casts.
        if let Expr::Var(cls) = recv {
            if cls == "$cast" {
                return self.cast(name, av.pop().unwrap_or(Value::Null));
            }
            if !self.locals.contains_key(cls) && cls != "ctx" && cls != "params" {
                return self.static_call(cls, name, &av);
            }
        }
        // Mutating methods operate on the place itself.
        if matches!(name, "add" | "addAll" | "put" | "putAll" | "remove" | "clear" | "removeIf" | "sort") {
            let src = self.src;
            if let Ok(place) = self.place(recv) {
                return mutate(src, place, name, av);
            }
        }
        let v = self.eval(recv)?;
        if v.is_null() {
            if safe {
                return Ok(Value::Null);
            }
            return Err(self.err("null_pointer_exception", format!("Cannot invoke \"{name}()\" because value is null")));
        }
        let arg = |i: usize| av.get(i).cloned().unwrap_or(Value::Null);
        let argi = |i: usize| av.get(i).and_then(Value::as_i64).unwrap_or(0);
        let r = match (&v, name) {
            (_, "toString") => Value::String(jstr(&v)),
            (_, "equals") => Value::Bool(values_eq(&v, &arg(0))),
            (_, "hashCode") => Value::from(0),
            (Value::String(s), "contains") => Value::Bool(s.contains(&jstr(&arg(0)))),
            (Value::String(s), "startsWith") => Value::Bool(s.starts_with(&jstr(&arg(0)))),
            (Value::String(s), "endsWith") => Value::Bool(s.ends_with(&jstr(&arg(0)))),
            (Value::String(s), "equalsIgnoreCase") => Value::Bool(s.to_lowercase() == jstr(&arg(0)).to_lowercase()),
            (Value::String(s), "isEmpty") => Value::Bool(s.is_empty()),
            (Value::String(s), "length") => Value::from(s.chars().count()),
            (Value::String(s), "toLowerCase") => Value::String(s.to_lowercase()),
            (Value::String(s), "toUpperCase") => Value::String(s.to_uppercase()),
            (Value::String(s), "trim") | (Value::String(s), "strip") => Value::String(s.trim().to_string()),
            (Value::String(s), "indexOf") => Value::from(s.find(&jstr(&arg(0))).map(|b| s[..b].chars().count() as i64).unwrap_or(-1)),
            (Value::String(s), "lastIndexOf") => Value::from(s.rfind(&jstr(&arg(0))).map(|b| s[..b].chars().count() as i64).unwrap_or(-1)),
            (Value::String(s), "replace") => Value::String(s.replace(&jstr(&arg(0)), &jstr(&arg(1)))),
            (Value::String(s), "charAt") => Value::String(s.chars().nth(argi(0) as usize).map(String::from).unwrap_or_default()),
            (Value::String(s), "substring") => {
                let cs: Vec<char> = s.chars().collect();
                let a = argi(0).max(0) as usize;
                let b = if av.len() > 1 { argi(1).max(0) as usize } else { cs.len() };
                if a > b || b > cs.len() {
                    return Err(self.err("string_index_out_of_bounds_exception", format!("begin {a}, end {b}, length {}", cs.len())));
                }
                Value::String(cs[a..b].iter().collect())
            }
            (Value::String(s), "splitOnToken") => Value::Array(s.split(jstr(&arg(0)).as_str()).map(|x| Value::String(x.to_string())).collect()),
            (Value::String(s), "split") => match regex::Regex::new(&jstr(&arg(0))) {
                Ok(re) => {
                    let mut parts: Vec<Value> = re.split(s).map(|x| Value::String(x.to_string())).collect();
                    while parts.last().map(|p| p.as_str() == Some("")).unwrap_or(false) {
                        parts.pop();
                    }
                    Value::Array(parts)
                }
                Err(e) => return Err(self.err("pattern_syntax_exception", e.to_string())),
            },
            (Value::Array(a), "size") | (Value::Array(a), "length") => Value::from(a.len()),
            (Value::Array(a), "isEmpty") => Value::Bool(a.is_empty()),
            (Value::Array(a), "contains") => Value::Bool(a.iter().any(|x| values_eq(x, &arg(0)))),
            (Value::Array(a), "indexOf") => Value::from(a.iter().position(|x| values_eq(x, &arg(0))).map(|p| p as i64).unwrap_or(-1)),
            (Value::Array(a), "get") => {
                let i = argi(0);
                if i < 0 || i as usize >= a.len() {
                    return Err(self.err("index_out_of_bounds_exception", format!("Index {i} out of bounds for length {}", a.len())));
                }
                a[i as usize].clone()
            }
            (Value::Array(a), "join") => Value::String(a.iter().map(jstr).collect::<Vec<_>>().join(&jstr(&arg(0)))),
            (Value::Object(m), "containsKey") => Value::Bool(m.contains_key(&jstr(&arg(0)))),
            (Value::Object(m), "containsValue") => Value::Bool(m.values().any(|x| values_eq(x, &arg(0)))),
            (Value::Object(m), "get") => m.get(&jstr(&arg(0))).cloned().unwrap_or(Value::Null),
            (Value::Object(m), "getOrDefault") => m.get(&jstr(&arg(0))).cloned().unwrap_or(arg(1)),
            (Value::Object(m), "size") => Value::from(m.len()),
            (Value::Object(m), "isEmpty") => Value::Bool(m.is_empty()),
            (Value::Object(m), "keySet") => Value::Array(m.keys().map(|k| Value::String(k.clone())).collect()),
            (Value::Object(m), "values") => Value::Array(m.values().cloned().collect()),
            (Value::Number(n), "intValue") | (Value::Number(n), "longValue") => Value::from(n.as_f64().unwrap_or(0.0) as i64),
            (Value::Number(n), "doubleValue") | (Value::Number(n), "floatValue") => f2v(n.as_f64().unwrap_or(0.0)),
            _ => {
                return Err(self.err(
                    "illegal_argument_exception",
                    format!("dynamic method [{}, {name}/{}] not found", super::doc::type_name(&v), args.len()),
                ))
            }
        };
        Ok(r)
    }

    fn cast(&self, t: &str, v: Value) -> Result<Value, EsError> {
        Ok(match t {
            "int" | "long" | "short" | "byte" | "Integer" | "Long" => match &v {
                Value::Number(n) => Value::from(n.as_f64().unwrap_or(0.0) as i64),
                _ => return Err(self.err("class_cast_exception", format!("Cannot cast from [{}] to [{t}].", super::doc::type_name(&v)))),
            },
            "double" | "float" | "Double" => match &v {
                Value::Number(n) => f2v(n.as_f64().unwrap_or(0.0)),
                _ => return Err(self.err("class_cast_exception", format!("Cannot cast from [{}] to [{t}].", super::doc::type_name(&v)))),
            },
            _ => v,
        })
    }

    fn static_call(&self, cls: &str, name: &str, av: &[Value]) -> Result<Value, EsError> {
        let arg = |i: usize| av.get(i).cloned().unwrap_or(Value::Null);
        let f = |i: usize| av.get(i).and_then(Value::as_f64).unwrap_or(0.0);
        let parse_err = |s: &Value| self.err("number_format_exception", format!("For input string: \"{}\"", jstr(s)));
        Ok(match (cls, name) {
            ("String", "valueOf") => Value::String(jstr(&arg(0))),
            ("String", "join") => {
                let sep = jstr(&arg(0));
                let items: Vec<String> = match arg(1) {
                    Value::Array(a) => a.iter().map(jstr).collect(),
                    _ => av[1..].iter().map(jstr).collect(),
                };
                Value::String(items.join(&sep))
            }
            ("Integer", "parseInt") | ("Long", "parseLong") | ("Integer", "valueOf") | ("Long", "valueOf") => {
                let s = jstr(&arg(0));
                Value::from(s.trim().parse::<i64>().map_err(|_| parse_err(&arg(0)))?)
            }
            ("Double", "parseDouble") | ("Float", "parseFloat") | ("Double", "valueOf") => {
                let s = jstr(&arg(0));
                f2v(s.trim().parse::<f64>().map_err(|_| parse_err(&arg(0)))?)
            }
            ("Boolean", "parseBoolean") => Value::Bool(jstr(&arg(0)).eq_ignore_ascii_case("true")),
            ("Math", "abs") => match arg(0).as_i64() {
                Some(i) if !arg(0).is_f64() => Value::from(i.abs()),
                _ => f2v(f(0).abs()),
            },
            ("Math", "round") => Value::from(f(0).round() as i64),
            ("Math", "floor") => f2v(f(0).floor()),
            ("Math", "ceil") => f2v(f(0).ceil()),
            ("Math", "sqrt") => f2v(f(0).sqrt()),
            ("Math", "pow") => f2v(f(0).powf(f(1))),
            ("Math", "log") => f2v(f(0).ln()),
            ("Math", "log10") => f2v(f(0).log10()),
            ("Math", "max") | ("Math", "min") => {
                let (a, b) = (arg(0), arg(1));
                let pick_a = if name == "max" { f(0) >= f(1) } else { f(0) <= f(1) };
                if pick_a { a } else { b }
            }
            ("Objects", "isNull") => Value::Bool(arg(0).is_null()),
            ("Objects", "nonNull") => Value::Bool(!arg(0).is_null()),
            ("Objects", "equals") => Value::Bool(values_eq(&arg(0), &arg(1))),
            _ => return Err(self.err("illegal_argument_exception", format!("cannot resolve symbol [{cls}]"))),
        })
    }
}

fn step_mut<'v>(cur: &'v mut Value, k: &Value, _create: bool) -> Result<&'v mut Value, String> {
    match cur {
        Value::Object(m) => {
            let key = jstr(k);
            if !m.contains_key(&key) {
                return Err(format!("Cannot invoke \"Object.get(Object)\" because [{key}] is null"));
            }
            Ok(m.get_mut(&key).unwrap())
        }
        Value::Array(a) => {
            let i = k.as_i64().unwrap_or(-1);
            let len = a.len();
            if i < 0 || i as usize >= len {
                return Err(format!("Index {i} out of bounds for length {len}"));
            }
            Ok(&mut a[i as usize])
        }
        Value::Null => Err("Cannot invoke \"Object.get(Object)\" because value is null".into()),
        other => Err(format!("Illegal list shortcut value [{}].", super::doc::type_name(other))),
    }
}

fn mutate(src: &str, place: &mut Value, name: &str, av: Vec<Value>) -> Result<Value, EsError> {
    let mut av = av.into_iter();
    let a0 = av.next().unwrap_or(Value::Null);
    let a1 = av.next();
    match (place, name) {
        (Value::Array(a), "add") => {
            match a1 {
                Some(v) => {
                    let i = a0.as_u64().unwrap_or(0) as usize;
                    if i > a.len() {
                        return Err(rt_err(src, "index_out_of_bounds_exception", format!("Index: {i}, Size: {}", a.len())));
                    }
                    a.insert(i, v);
                }
                None => a.push(a0),
            }
            Ok(Value::Bool(true))
        }
        (Value::Array(a), "addAll") => {
            if let Value::Array(x) = a0 {
                a.extend(x);
            }
            Ok(Value::Bool(true))
        }
        (Value::Array(a), "remove") => {
            if let Some(i) = a0.as_i64().filter(|_| !a0.is_f64()) {
                if i < 0 || i as usize >= a.len() {
                    return Err(rt_err(src, "index_out_of_bounds_exception", format!("Index {i} out of bounds for length {}", a.len())));
                }
                return Ok(a.remove(i as usize));
            }
            match a.iter().position(|x| values_eq(x, &a0)) {
                Some(p) => {
                    a.remove(p);
                    Ok(Value::Bool(true))
                }
                None => Ok(Value::Bool(false)),
            }
        }
        (Value::Array(a), "clear") => {
            a.clear();
            Ok(Value::Null)
        }
        (Value::Array(a), "sort") => {
            a.sort_by(|x, y| match (x.as_f64(), y.as_f64()) {
                (Some(p), Some(q)) => p.partial_cmp(&q).unwrap_or(std::cmp::Ordering::Equal),
                _ => jstr(x).cmp(&jstr(y)),
            });
            Ok(Value::Null)
        }
        (Value::Object(m), "put") => Ok(m.insert(jstr(&a0), a1.unwrap_or(Value::Null)).unwrap_or(Value::Null)),
        (Value::Object(m), "putAll") => {
            if let Value::Object(x) = a0 {
                for (k, v) in x {
                    m.insert(k, v);
                }
            }
            Ok(Value::Null)
        }
        (Value::Object(m), "remove") => Ok(m.shift_remove(&jstr(&a0)).unwrap_or(Value::Null)),
        (Value::Object(m), "clear") => {
            m.clear();
            Ok(Value::Null)
        }
        (Value::Null, _) => Err(rt_err(src, "null_pointer_exception", format!("Cannot invoke \"{name}\" because value is null"))),
        (other, _) => Err(rt_err(src, "illegal_argument_exception", format!("dynamic method [{}, {name}/1] not found", super::doc::type_name(other)))),
    }
}

/// A compiled script.
#[derive(Clone, Debug)]
pub struct LiteScript {
    src: String,
    prog: Vec<Stmt>,
}

impl LiteScript {
    pub fn compile(src: &str) -> Result<LiteScript, EsError> {
        let toks = lex(src)?;
        let mut p = Parser { t: toks, i: 0, src, nest: crate::parse_depth::Nesting::new(512) };
        let prog = p.program()?;
        Ok(LiteScript { src: src.to_string(), prog })
    }

    /// Run against `ctx` (mutable); returns the `return` value or the value
    /// of the last expression statement.
    pub fn run(&self, ctx: &mut Value, params: &Value) -> Result<Value, EsError> {
        let mut env = Env { src: &self.src, ctx, params, locals: HashMap::new(), last: Value::Null, steps: 0 };
        match env.run(&self.prog)? {
            Flow::Return(v) => Ok(v),
            Flow::Normal => Ok(std::mem::take(&mut env.last)),
        }
    }

    pub fn condition(&self, ctx: &mut Value, params: &Value) -> Result<bool, EsError> {
        match self.run(ctx, params)? {
            Value::Bool(b) => Ok(b),
            other => Err(rt_err(&self.src, "class_cast_exception", format!("Cannot cast from [{}] to [boolean].", super::doc::type_name(&other)))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(src: &str, ctx: Value) -> (Value, Value) {
        let mut c = ctx;
        let r = LiteScript::compile(src).unwrap().run(&mut c, &json!({"p": 2})).unwrap();
        (r, c)
    }

    #[test]
    fn basics() {
        let (_, c) = run("ctx.b = ctx.a * params.p + 1; ctx.s = 'x' + ctx.a; ctx.o = [:]; ctx.o.k = [1, 2]; ctx.o.k.add(3);", json!({"a": 5}));
        assert_eq!(c, json!({"a": 5, "b": 11, "s": "x5", "o": {"k": [1, 2, 3]}}));
        let (r, _) = run("ctx.a?.b == null && ctx.x != 'y'", json!({"x": "z"}));
        assert_eq!(r, json!(true));
        let (r, _) = run("if (ctx.tags.contains('a')) { return 1 } else { return 2 }", json!({"tags": ["a"]}));
        assert_eq!(r, json!(1));
        let (_, c) = run("def total = 0; for (def x : ctx.n) { total += x; } ctx.total = total; ctx.remove('n');", json!({"n": [1, 2, 3]}));
        assert_eq!(c, json!({"total": 6}));
        let (r, _) = run("ctx.msg instanceof String ? ctx.msg.toUpperCase() : 'none'", json!({"msg": "hi"}));
        assert_eq!(r, json!("HI"));
        let e = LiteScript::compile("ctx.a.b = 1").unwrap().run(&mut json!({}), &json!({})).unwrap_err();
        assert_eq!(e.kind, "script_exception");
        assert!(LiteScript::compile("ctx.a = ").is_err());
    }
}
