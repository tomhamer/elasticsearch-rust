//! A small TorchScript interpreter standing in for ES's native
//! `pytorch_inference` process.
//!
//! Uploaded PyTorch models (`PUT _ml/trained_models/{id}/definition/{part}`)
//! are TorchScript zip archives whose `code/__torch__.py` holds the
//! serialized `forward` method. This module reads the archive, parses the
//! `forward` body and evaluates it over a dense tensor implementation, with
//! no libtorch dependency. It is designed for lightweight straight-line tensor
//! programs (constant / random / one-hot outputs); an operation outside that
//! set is reported with a clear error rather than approximated. Large neural
//! models are best served by an external inference endpoint.
//!
//! `torch.rand` draws from a deterministic generator seeded by
//! `torch.manual_seed`; like real PyTorch across platforms, the exact
//! values differ from libtorch's, but equal seeds give equal tensors.

use crate::error::{EsError, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Zip archive reading

/// The files of a zip archive, by name.
pub fn unzip(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    let bad = || EsError::illegal_argument("model definition is not a valid TorchScript archive");
    let u16_at = |o: usize| -> Option<usize> { bytes.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize) };
    let u32_at = |o: usize| -> Option<usize> { bytes.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize) };
    // End of central directory record (scan back over a trailing comment).
    let eocd = (0..bytes.len().saturating_sub(21)).rev().find(|&i| bytes[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06])).ok_or_else(bad)?;
    let entries = u16_at(eocd + 10).ok_or_else(bad)?;
    let mut cd = u32_at(eocd + 16).ok_or_else(bad)?;
    let mut out = HashMap::new();
    for _ in 0..entries {
        if u32_at(cd) != Some(0x0201_4b50) {
            return Err(bad());
        }
        let method = u16_at(cd + 10).ok_or_else(bad)?;
        let csize = u32_at(cd + 20).ok_or_else(bad)?;
        let name_len = u16_at(cd + 28).ok_or_else(bad)?;
        let extra_len = u16_at(cd + 30).ok_or_else(bad)?;
        let comment_len = u16_at(cd + 32).ok_or_else(bad)?;
        let local = u32_at(cd + 42).ok_or_else(bad)?;
        let name = String::from_utf8_lossy(bytes.get(cd + 46..cd + 46 + name_len).ok_or_else(bad)?).into_owned();
        cd += 46 + name_len + extra_len + comment_len;
        if u32_at(local) != Some(0x0403_4b50) {
            return Err(bad());
        }
        let start = local + 30 + u16_at(local + 26).ok_or_else(bad)? + u16_at(local + 28).ok_or_else(bad)?;
        let raw = bytes.get(start..start + csize).ok_or_else(bad)?;
        let data = match method {
            0 => raw.to_vec(),
            8 => {
                let mut v = Vec::new();
                flate2::read::DeflateDecoder::new(raw).read_to_end(&mut v).map_err(|_| bad())?;
                v
            }
            _ => return Err(bad()),
        };
        out.insert(name, data);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tensors

#[derive(Clone, Copy, PartialEq, Debug)]
enum DType {
    Float,
    Long,
    Bool,
}

/// TorchScript `ScalarType` codes used in `dtype=` arguments.
fn dtype_of_code(c: i64) -> DType {
    match c {
        0..=4 => DType::Long,
        11 => DType::Bool,
        _ => DType::Float,
    }
}

#[derive(Clone)]
pub struct Tensor {
    data: Rc<RefCell<Vec<f64>>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: DType,
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Tensor {
    fn new(shape: Vec<usize>, data: Vec<f64>, dtype: DType) -> Tensor {
        let strides = contiguous_strides(&shape);
        Tensor { data: Rc::new(RefCell::new(data)), offset: 0, shape, strides, dtype }
    }
    fn scalar(v: f64, dtype: DType) -> Tensor {
        Tensor::new(vec![], vec![v], dtype)
    }
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Element values in row-major order.
    fn values(&self) -> Vec<f64> {
        let d = self.data.borrow();
        let mut out = Vec::with_capacity(self.numel());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            let pos = self.offset + idx.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>();
            out.push(d[pos]);
            for k in (0..idx.len()).rev() {
                idx[k] += 1;
                if idx[k] < self.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        out
    }
    fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(unsupported("item() of a multi-element tensor"));
        }
        Ok(self.values()[0])
    }
    fn select(&self, dim: usize, i: i64) -> Result<Tensor> {
        let n = *self.shape.get(dim).ok_or_else(|| unsupported("select dimension"))? as i64;
        let i = if i < 0 { i + n } else { i };
        if i < 0 || i >= n {
            return Err(EsError::illegal_argument(format!("index {i} is out of bounds for dimension {dim} with size {n}")));
        }
        let mut t = self.clone();
        t.offset += i as usize * self.strides[dim];
        t.shape.remove(dim);
        t.strides.remove(dim);
        Ok(t)
    }
    /// Nested arrays of the values (`[batch][seq][..]` style output).
    pub fn to_json(&self) -> serde_json::Value {
        fn rec(vals: &[f64], shape: &[usize], long: bool) -> serde_json::Value {
            if shape.is_empty() {
                let v = vals[0];
                return if long { serde_json::json!(v as i64) } else { crate::values::java_double(v) };
            }
            let step = shape[1..].iter().product::<usize>().max(1);
            serde_json::Value::Array((0..shape[0]).map(|i| rec(&vals[i * step..], &shape[1..], long)).collect())
        }
        rec(&self.values(), &self.shape, self.dtype == DType::Long)
    }
    /// The output as `[batch][rows][values]` (ES `PyTorchInferenceResult`).
    pub fn as_3d(&self) -> Vec<Vec<Vec<f64>>> {
        let v = self.values();
        let (b, r, c) = match self.shape.len() {
            0 => (1, 1, 1),
            1 => (1, 1, self.shape[0]),
            2 => (self.shape[0], 1, self.shape[1]),
            _ => (self.shape[0], self.shape[1], self.shape[2..].iter().product()),
        };
        (0..b).map(|i| (0..r).map(|j| v[(i * r + j) * c..(i * r + j + 1) * c].to_vec()).collect()).collect()
    }
    /// `str(tensor)` (PyTorch's repr, for seeding).
    fn repr(&self) -> String {
        fn rec(vals: &[f64], shape: &[usize], long: bool) -> String {
            if shape.is_empty() {
                return if long { format!("{}", vals[0] as i64) } else { format!("{:.4}", vals[0]) };
            }
            let step = shape[1..].iter().product::<usize>().max(1);
            let parts: Vec<String> = (0..shape[0]).map(|i| rec(&vals[i * step..], &shape[1..], long)).collect();
            format!("[{}]", parts.join(", "))
        }
        format!("tensor({})", rec(&self.values(), &self.shape, self.dtype == DType::Long))
    }
}

// ---------------------------------------------------------------------------
// Values

#[derive(Clone)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tensor(Tensor),
    List(Vec<Val>),
    DType(DType),
    Device,
}

fn unsupported(what: &str) -> EsError {
    EsError::illegal_argument(format!("unsupported TorchScript operation [{what}] in model definition"))
}

impl Val {
    fn as_i64(&self) -> Result<i64> {
        match self {
            Val::Int(i) => Ok(*i),
            Val::Float(f) => Ok(*f as i64),
            Val::Bool(b) => Ok(*b as i64),
            Val::Tensor(t) => Ok(t.item()? as i64),
            _ => Err(unsupported("integer conversion")),
        }
    }
    fn as_f64(&self) -> Result<f64> {
        match self {
            Val::Int(i) => Ok(*i as f64),
            Val::Float(f) => Ok(*f),
            Val::Bool(b) => Ok(*b as i64 as f64),
            Val::Tensor(t) => t.item(),
            _ => Err(unsupported("number conversion")),
        }
    }
    fn tensor(&self) -> Result<Tensor> {
        match self {
            Val::Tensor(t) => Ok(t.clone()),
            Val::Int(i) => Ok(Tensor::scalar(*i as f64, DType::Long)),
            Val::Float(f) => Ok(Tensor::scalar(*f, DType::Float)),
            Val::Bool(b) => Ok(Tensor::scalar(*b as i64 as f64, DType::Bool)),
            _ => Err(unsupported("tensor conversion")),
        }
    }
    fn shape(&self) -> Result<Vec<usize>> {
        match self {
            Val::List(l) => l.iter().map(|v| v.as_i64().map(|x| x.max(0) as usize)).collect(),
            Val::Int(i) => Ok(vec![(*i).max(0) as usize]),
            _ => Err(unsupported("shape argument")),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(String),
    Str(String),
    Sym(char),
    Arrow,
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '#' {
            break;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_' || cs[i] == '.') {
                i += 1;
            }
            out.push(Tok::Ident(cs[st..i].iter().collect()));
        } else if c.is_ascii_digit() {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.' || cs[i] == 'e' || ((cs[i] == '-' || cs[i] == '+') && (cs[i - 1] == 'e'))) {
                i += 1;
            }
            out.push(Tok::Num(cs[st..i].iter().collect()));
        } else if c == '"' || c == '\'' {
            let q = c;
            i += 1;
            let st = i;
            while i < cs.len() && cs[i] != q {
                i += 1;
            }
            out.push(Tok::Str(cs[st..i.min(cs.len())].iter().collect()));
            i += 1;
        } else if c == '-' && cs.get(i + 1) == Some(&'>') {
            out.push(Tok::Arrow);
            i += 2;
        } else {
            out.push(Tok::Sym(c));
            i += 1;
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
enum Expr {
    Name(String),
    Num(String),
    Str(String),
    List(Vec<Expr>),
    Neg(Box<Expr>),
    Call(String, Vec<Expr>, Vec<(String, Expr)>),
    /// `annotate(Type, e)` / type expressions: evaluates `e`.
    Annotate(Box<Expr>),
}

struct P {
    t: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.t.get(self.i).cloned();
        self.i += 1;
        t
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<()> {
        if self.eat(c) { Ok(()) } else { Err(unsupported(&format!("syntax near token {}", self.i))) }
    }
    /// Skip a type expression (`List[Optional[Tensor]]`).
    fn skip_type(&mut self) {
        let mut depth = 0i32;
        while let Some(t) = self.peek() {
            match t {
                Tok::Sym('[') => depth += 1,
                Tok::Sym(']') => {
                    if depth == 0 {
                        return;
                    }
                    depth -= 1;
                }
                Tok::Sym(',') | Tok::Sym(')') | Tok::Sym('=') if depth == 0 => return,
                _ => {}
            }
            self.i += 1;
        }
    }
    fn expr(&mut self) -> Result<Expr> {
        if self.eat('-') {
            return Ok(Expr::Neg(Box::new(self.expr()?)));
        }
        match self.next() {
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Sym('[')) => {
                let mut items = Vec::new();
                if !self.eat(']') {
                    loop {
                        items.push(self.expr()?);
                        if self.eat(']') {
                            break;
                        }
                        self.expect(',')?;
                    }
                }
                Ok(Expr::List(items))
            }
            Some(Tok::Sym('(')) => {
                let e = self.expr()?;
                self.expect(')')?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                if !self.eat('(') {
                    return Ok(Expr::Name(name));
                }
                if name == "annotate" {
                    self.skip_type();
                    self.expect(',')?;
                    let e = self.expr()?;
                    self.expect(')')?;
                    return Ok(Expr::Annotate(Box::new(e)));
                }
                let (mut args, mut kw) = (Vec::new(), Vec::new());
                if !self.eat(')') {
                    loop {
                        if let (Some(Tok::Ident(k)), Some(Tok::Sym('='))) = (self.t.get(self.i).cloned(), self.t.get(self.i + 1).cloned()) {
                            self.i += 2;
                            kw.push((k, self.expr()?));
                        } else {
                            args.push(self.expr()?);
                        }
                        if self.eat(')') {
                            break;
                        }
                        self.expect(',')?;
                    }
                }
                Ok(Expr::Call(name, args, kw))
            }
            _ => Err(unsupported("expression syntax")),
        }
    }
}

#[derive(Clone, Debug)]
enum Stmt {
    Assign(String, Expr),
    Expr(Expr),
    Return(Expr),
    For(String, Expr, Vec<Stmt>),
}

/// A parsed `forward` method.
pub struct Program {
    params: Vec<String>,
    body: Vec<Stmt>,
}

fn indent_of(l: &str) -> usize {
    l.len() - l.trim_start().len()
}

fn parse_block(lines: &[&str], i: &mut usize, indent: usize) -> Result<Vec<Stmt>> {
    let mut out = Vec::new();
    while *i < lines.len() {
        let l = lines[*i];
        if l.trim().is_empty() {
            *i += 1;
            continue;
        }
        if indent_of(l) < indent {
            break;
        }
        let s = l.trim();
        *i += 1;
        if let Some(rest) = s.strip_prefix("return ") {
            out.push(Stmt::Return(parse_expr(rest)?));
        } else if let Some(rest) = s.strip_prefix("for ") {
            let (var, it) = rest.split_once(" in ").ok_or_else(|| unsupported("for loop"))?;
            let it = it.trim_end_matches(':');
            let body_indent = lines.get(*i).map(|l| indent_of(l)).unwrap_or(indent + 2);
            let body = parse_block(lines, i, body_indent.max(indent + 1))?;
            out.push(Stmt::For(var.trim().to_string(), parse_expr(it)?, body));
        } else if s == "pass" {
        } else {
            let toks = lex(s)?;
            // `name = expr` / `name : Type = expr`
            let eq = toks.iter().position(|t| *t == Tok::Sym('='));
            match (toks.first(), eq) {
                (Some(Tok::Ident(n)), Some(p)) if p == 1 || toks.get(1) == Some(&Tok::Sym(':')) => {
                    let mut pp = P { t: toks[p + 1..].to_vec(), i: 0 };
                    out.push(Stmt::Assign(n.clone(), pp.expr()?));
                }
                _ => out.push(Stmt::Expr(parse_expr(s)?)),
            }
        }
    }
    Ok(out)
}

fn parse_expr(s: &str) -> Result<Expr> {
    let mut p = P { t: lex(s)?, i: 0 };
    let e = p.expr()?;
    if p.i < p.t.len() {
        return Err(unsupported(&format!("trailing syntax in [{s}]")));
    }
    Ok(e)
}

impl Program {
    /// Parse the `forward` method of a TorchScript `code/__torch__.py`.
    pub fn parse(code: &str) -> Result<Program> {
        let lines: Vec<&str> = code.lines().collect();
        let start = lines.iter().position(|l| l.trim_start().starts_with("def forward(")).ok_or_else(|| unsupported("missing forward method"))?;
        let def_indent = indent_of(lines[start]);
        // The signature may span lines; it ends at the line ending in `:`.
        let mut sig = String::new();
        let mut i = start;
        while i < lines.len() {
            sig.push_str(lines[i].trim());
            sig.push(' ');
            i += 1;
            if sig.trim_end().ends_with(':') {
                break;
            }
        }
        let inner = sig.split_once('(').map(|x| x.1).unwrap_or("");
        let inner = inner.rsplit_once(")").map(|x| x.0).unwrap_or(inner);
        // Parameters: split on top-level commas.
        let mut params = Vec::new();
        let (mut depth, mut cur) = (0i32, String::new());
        for c in inner.chars() {
            match c {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                _ => {}
            }
            if c == ',' && depth == 0 {
                params.push(std::mem::take(&mut cur));
            } else {
                cur.push(c);
            }
        }
        params.push(cur);
        let params: Vec<String> = params.iter().map(|p| p.split(':').next().unwrap_or("").trim().to_string()).filter(|p| !p.is_empty() && p != "self").collect();
        let body_indent = lines.get(i).map(|l| indent_of(l)).unwrap_or(def_indent + 2);
        let body = parse_block(&lines, &mut i, body_indent.max(def_indent + 1))?;
        Ok(Program { params, body })
    }

    /// Load the program of a TorchScript archive.
    pub fn from_archive(bytes: &[u8]) -> Result<Program> {
        let files = unzip(bytes)?;
        let code = files.iter().find(|(n, _)| n.ends_with("code/__torch__.py")).map(|(_, v)| v).ok_or_else(|| unsupported("archive without code/__torch__.py"))?;
        Program::parse(&String::from_utf8_lossy(code))
    }

    /// Run `forward` over one tokenized input (batch of one).
    pub fn run(&self, input_ids: &[i64]) -> Result<Tensor> {
        let n = input_ids.len();
        let ids = Tensor::new(vec![1, n], input_ids.iter().map(|x| *x as f64).collect(), DType::Long);
        let mask = Tensor::new(vec![1, n], vec![1.0; n], DType::Long);
        let types = Tensor::new(vec![1, n], vec![0.0; n], DType::Long);
        let pos = Tensor::new(vec![1, n], (0..n).map(|x| x as f64).collect(), DType::Long);
        let args = [ids, mask, types, pos];
        let mut env: HashMap<String, Val> = HashMap::new();
        for (i, p) in self.params.iter().enumerate() {
            env.insert(p.clone(), args.get(i).cloned().map(Val::Tensor).unwrap_or(Val::None));
        }
        let mut st = Interp { env, rng: Rng(0x853c_49e6_748f_ea9b), steps: 0 };
        match st.block(&self.body)? {
            Some(Val::Tensor(t)) => Ok(t),
            Some(Val::List(l)) => l.first().map(Val::tensor).transpose()?.ok_or_else(|| unsupported("empty output")),
            _ => Err(unsupported("forward must return a tensor")),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // 24 random mantissa bits, as for a float32 uniform draw.
        ((z >> 40) as f64) / ((1u64 << 24) as f64)
    }
}

struct Interp {
    env: HashMap<String, Val>,
    rng: Rng,
    steps: u64,
}

const MAX_STEPS: u64 = 10_000_000;

impl Interp {
    fn block(&mut self, b: &[Stmt]) -> Result<Option<Val>> {
        for s in b {
            self.steps += 1;
            if self.steps > MAX_STEPS {
                return Err(unsupported("program too long"));
            }
            match s {
                Stmt::Assign(n, e) => {
                    let v = self.eval(e)?;
                    self.env.insert(n.clone(), v);
                }
                Stmt::Expr(e) => {
                    self.eval(e)?;
                }
                Stmt::Return(e) => return Ok(Some(self.eval(e)?)),
                Stmt::For(var, it, body) => {
                    let items: Vec<Val> = match it {
                        Expr::Call(f, a, _) if f == "range" => {
                            let (lo, hi) = match a.len() {
                                1 => (0, self.eval(&a[0])?.as_i64()?),
                                _ => (self.eval(&a[0])?.as_i64()?, self.eval(&a[1])?.as_i64()?),
                            };
                            (lo..hi).map(Val::Int).collect()
                        }
                        other => match self.eval(other)? {
                            Val::List(l) => l,
                            _ => return Err(unsupported("for loop iterable")),
                        },
                    };
                    for v in items {
                        self.env.insert(var.clone(), v);
                        if let Some(r) = self.block(body)? {
                            return Ok(Some(r));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    fn eval(&mut self, e: &Expr) -> Result<Val> {
        Ok(match e {
            Expr::Num(n) => {
                if n.contains('.') || n.contains('e') {
                    Val::Float(n.parse().map_err(|_| unsupported("number literal"))?)
                } else {
                    Val::Int(n.parse().map_err(|_| unsupported("number literal"))?)
                }
            }
            Expr::Str(s) => Val::Str(s.clone()),
            Expr::List(items) => Val::List(items.iter().map(|x| self.eval(x)).collect::<Result<_>>()?),
            Expr::Neg(x) => match self.eval(x)? {
                Val::Int(i) => Val::Int(-i),
                Val::Float(f) => Val::Float(-f),
                _ => return Err(unsupported("negation")),
            },
            Expr::Annotate(x) => self.eval(x)?,
            Expr::Name(n) => match n.as_str() {
                "None" => Val::None,
                "True" => Val::Bool(true),
                "False" => Val::Bool(false),
                _ => self.env.get(n).cloned().ok_or_else(|| unsupported(&format!("unknown name {n}")))?,
            },
            Expr::Call(f, args, kw) => {
                let a: Vec<Val> = args.iter().map(|x| self.eval(x)).collect::<Result<_>>()?;
                let mut k: HashMap<&str, Val> = HashMap::new();
                for (n, x) in kw {
                    k.insert(n.as_str(), self.eval(x)?);
                }
                self.call(f, a, k)?
            }
        })
    }

    fn call(&mut self, f: &str, a: Vec<Val>, k: HashMap<&str, Val>) -> Result<Val> {
        let arg = |i: usize| a.get(i).cloned().ok_or_else(|| unsupported(&format!("{f} arguments")));
        let dtype = |default: DType| match k.get("dtype") {
            Some(Val::Int(c)) => dtype_of_code(*c),
            Some(Val::DType(d)) => *d,
            _ => default,
        };
        Ok(match f {
            "int" => Val::Int(arg(0)?.as_i64()?),
            "float" => Val::Float(arg(0)?.as_f64()?),
            "bool" => Val::Bool(arg(0)?.as_f64()? != 0.0),
            "str" => Val::Str(match arg(0)? {
                Val::Tensor(t) => t.repr(),
                Val::Int(i) => i.to_string(),
                Val::Float(x) => x.to_string(),
                Val::Str(s) => s,
                _ => return Err(unsupported("str")),
            }),
            "len" | "torch.len" => Val::Int(match arg(0)? {
                Val::Tensor(t) => *t.shape.first().ok_or_else(|| unsupported("len() of a 0-d tensor"))? as i64,
                Val::List(l) => l.len() as i64,
                Val::Str(s) => s.chars().count() as i64,
                _ => return Err(unsupported("len")),
            }),
            "torch.hash" => {
                let s = match arg(0)? {
                    Val::Str(s) => s,
                    Val::Int(i) => i.to_string(),
                    _ => return Err(unsupported("hash")),
                };
                // FNV-1a, folded into a non-negative seed.
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for b in s.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
                Val::Int((h >> 1) as i64)
            }
            "torch.manual_seed" => {
                self.rng = Rng(arg(0)?.as_i64()? as u64);
                Val::None
            }
            "torch.device" => Val::Device,
            "ops.prim.device" => Val::Device,
            "ops.prim.dtype" => Val::DType(arg(0)?.tensor()?.dtype),
            "ops.prim.NumToTensor" => {
                let v = arg(0)?;
                let dt = if matches!(v, Val::Float(_)) { DType::Float } else { DType::Long };
                Val::Tensor(Tensor::scalar(v.as_f64()?, dt))
            }
            "torch.size" => {
                let t = arg(0)?.tensor()?;
                match a.get(1) {
                    Some(d) => {
                        let d = d.as_i64()?;
                        let d = if d < 0 { d + t.shape.len() as i64 } else { d };
                        Val::Int(*t.shape.get(d as usize).ok_or_else(|| unsupported("size dimension"))? as i64)
                    }
                    None => Val::List(t.shape.iter().map(|x| Val::Int(*x as i64)).collect()),
                }
            }
            "torch.dim" => Val::Int(arg(0)?.tensor()?.shape.len() as i64),
            "torch.ones" | "torch.zeros" | "torch.rand" | "torch.full" => {
                let shape = arg(0)?.shape()?;
                let n: usize = shape.iter().product();
                if n > 50_000_000 {
                    return Err(unsupported("tensor too large"));
                }
                let (data, dt) = match f {
                    "torch.ones" => (vec![1.0; n], dtype(DType::Float)),
                    "torch.zeros" => (vec![0.0; n], dtype(DType::Float)),
                    "torch.full" => (vec![arg(1)?.as_f64()?; n], dtype(DType::Float)),
                    _ => ((0..n).map(|_| self.rng.next_f64()).collect(), DType::Float),
                };
                Val::Tensor(Tensor::new(shape, data, dt))
            }
            "torch.arange" => {
                let n = arg(0)?.as_i64()?.max(0) as usize;
                Val::Tensor(Tensor::new(vec![n], (0..n).map(|x| x as f64).collect(), dtype(DType::Long)))
            }
            "torch.tensor" => {
                let v = arg(0)?;
                let default = if matches!(v, Val::Float(_)) { DType::Float } else { DType::Long };
                match v {
                    Val::List(l) => {
                        let vals: Vec<f64> = l.iter().map(Val::as_f64).collect::<Result<_>>()?;
                        Val::Tensor(Tensor::new(vec![vals.len()], vals, dtype(default)))
                    }
                    other => Val::Tensor(Tensor::scalar(other.as_f64()?, dtype(default))),
                }
            }
            "torch.add" | "torch.sub" | "torch.mul" | "torch.div" => {
                let (x, y) = (arg(0)?, arg(1)?);
                let op = |p: f64, q: f64| match f {
                    "torch.add" => p + q,
                    "torch.sub" => p - q,
                    "torch.mul" => p * q,
                    _ => p / q,
                };
                match (&x, &y) {
                    (Val::Int(p), Val::Int(q)) if f != "torch.div" => Val::Int(op(*p as f64, *q as f64) as i64),
                    (Val::Tensor(_), _) | (_, Val::Tensor(_)) => {
                        let (tx, ty) = (x.tensor()?, y.tensor()?);
                        let (vx, vy) = (tx.values(), ty.values());
                        let (shape, n) = if tx.numel() >= ty.numel() { (tx.shape.clone(), tx.numel()) } else { (ty.shape.clone(), ty.numel()) };
                        let pick = |v: &Vec<f64>, i: usize| if v.len() == 1 { v[0] } else { v[i % v.len().max(1)] };
                        let data: Vec<f64> = (0..n).map(|i| op(pick(&vx, i), pick(&vy, i))).collect();
                        let dt = if tx.dtype == DType::Float || ty.dtype == DType::Float || f == "torch.div" { DType::Float } else { DType::Long };
                        Val::Tensor(Tensor::new(shape, data, dt))
                    }
                    _ => Val::Float(op(x.as_f64()?, y.as_f64()?)),
                }
            }
            "torch.max" | "torch.min" | "torch.sum" | "torch.mean" => {
                let t = arg(0)?.tensor()?;
                if a.len() > 1 {
                    return Err(unsupported(&format!("{f} over a dimension")));
                }
                let v = t.values();
                if v.is_empty() {
                    return Err(unsupported(&format!("{f} of an empty tensor")));
                }
                let r = match f {
                    "torch.max" => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "torch.min" => v.iter().cloned().fold(f64::INFINITY, f64::min),
                    "torch.sum" => v.iter().sum(),
                    _ => v.iter().sum::<f64>() / v.len() as f64,
                };
                let dt = if f == "torch.mean" { DType::Float } else { t.dtype };
                Val::Tensor(Tensor::scalar(r, dt))
            }
            "torch.select" => {
                let t = arg(0)?.tensor()?;
                Val::Tensor(t.select(arg(1)?.as_i64()?.max(0) as usize, arg(2)?.as_i64()?)?)
            }
            "torch.index_put_" => {
                let t = arg(0)?.tensor()?;
                let idx = match arg(1)? {
                    Val::List(l) => l,
                    _ => return Err(unsupported("index_put_ indices")),
                };
                if idx.len() != 1 || t.shape.len() != 1 {
                    return Err(unsupported("multi-dimensional index_put_"));
                }
                let positions: Vec<i64> = match &idx[0] {
                    Val::Tensor(it) => it.values().into_iter().map(|x| x as i64).collect(),
                    other => vec![other.as_i64()?],
                };
                let value = arg(2)?.tensor()?.values();
                {
                    let mut d = t.data.borrow_mut();
                    for (j, p) in positions.iter().enumerate() {
                        let n = t.shape[0] as i64;
                        let p = if *p < 0 { p + n } else { *p };
                        if p < 0 || p >= n {
                            return Err(EsError::illegal_argument(format!("index {p} is out of bounds for dimension 0 with size {n}")));
                        }
                        let v = if value.len() == 1 { value[0] } else { value[j % value.len().max(1)] };
                        d[t.offset + p as usize * t.strides[0]] = v;
                    }
                }
                Val::Tensor(t)
            }
            "torch.to" => Val::Tensor(arg(0)?.tensor()?),
            _ => return Err(unsupported(f)),
        })
    }
}

// ---------------------------------------------------------------------------
// Tokenization (ES `BertTokenizer`: basic + WordPiece)

/// Token ids of `text` for a WordPiece vocabulary; unknown words map to
/// `[UNK]`. `lower`: `do_lower_case`; `special`: add `[CLS]` / `[SEP]`.
pub fn wordpiece(text: &str, vocab: &[String], lower: bool, special: bool, max_len: usize) -> (Vec<i64>, bool) {
    let index: HashMap<&str, i64> = vocab.iter().enumerate().map(|(i, t)| (t.as_str(), i as i64)).collect();
    let unk = index.get("[UNK]").copied().unwrap_or(0);
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_whitespace() || c.is_control() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        } else if c.is_ascii_punctuation() || (!c.is_ascii() && !c.is_alphanumeric()) {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            words.push(c.to_string());
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    let mut ids = Vec::new();
    for w in words {
        let w = if lower { w.to_lowercase() } else { w };
        if index.contains_key(w.as_str()) {
            ids.push(index[w.as_str()]);
            continue;
        }
        // Greedy longest-match-first WordPiece.
        let chars: Vec<char> = w.chars().collect();
        let mut start = 0;
        let mut pieces = Vec::new();
        let mut bad = false;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let sub: String = chars[start..end].iter().collect();
                let sub = if start > 0 { format!("##{sub}") } else { sub };
                if let Some(id) = index.get(sub.as_str()) {
                    found = Some(*id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                None => {
                    bad = true;
                    break;
                }
            }
        }
        if bad {
            ids.push(unk);
        } else {
            ids.extend(pieces);
        }
    }
    let budget = if special { max_len.saturating_sub(2) } else { max_len };
    let truncated = ids.len() > budget;
    ids.truncate(budget);
    if special {
        let cls = index.get("[CLS]").copied().unwrap_or(unk);
        let sep = index.get("[SEP]").copied().unwrap_or(unk);
        ids.insert(0, cls);
        ids.push(sep);
    }
    (ids, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interprets_text_expansion_model() {
        let code = "class TinyTextExpansion(Module):\n  __parameters__ = []\n  def forward(self: __torch__.TinyTextExpansion,\n    input_ids: Tensor,\n    token_type_ids: Optional[Tensor]=None) -> Tensor:\n    torch.manual_seed(torch.hash(str(input_ids)))\n    _0 = annotate(int, torch.add(torch.max(input_ids), 1))\n    weights = torch.zeros([1, _0])\n    for _1 in range(torch.len(input_ids)):\n      i = torch.select(input_ids, 0, _1)\n      _2 = torch.select(weights, 0, 0)\n      _3 = torch.tensor(1., dtype=ops.prim.dtype(_2), device=ops.prim.device(_2))\n      _4 = annotate(List[Optional[Tensor]], [i])\n      _5 = torch.index_put_(_2, _4, _3)\n    return weights\n";
        let p = Program::parse(code).unwrap();
        let out = p.run(&[2, 5, 3]).unwrap();
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.values(), vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn interprets_random_embedding() {
        let code = "class T(Module):\n  def forward(self: __torch__.T,\n    input_ids: Tensor) -> Tensor:\n    torch.manual_seed(torch.hash(str(input_ids)))\n    return torch.rand([1, 100])\n";
        let p = Program::parse(code).unwrap();
        let a = p.run(&[1, 2]).unwrap().values();
        let b = p.run(&[1, 2]).unwrap().values();
        let c = p.run(&[2, 1]).unwrap().values();
        assert_eq!(a.len(), 100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn wordpiece_tokenizes() {
        let v: Vec<String> = ["[PAD]", "[UNK]", "these", "are", "my", "words", "wash", "##ing"].iter().map(|s| s.to_string()).collect();
        assert_eq!(wordpiece("these are washing machines", &v, false, false, 512).0, vec![2, 3, 6, 7, 1]);
    }
}
