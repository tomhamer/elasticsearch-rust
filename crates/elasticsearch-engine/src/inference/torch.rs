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

/// Largest total of expanded archive members (all members together).
pub const MAX_ARCHIVE_BYTES: usize = 1 << 30;
/// Most archive entries read.
const MAX_ARCHIVE_ENTRIES: usize = 10_000;

/// The files of a zip archive, by name (at most [`MAX_ARCHIVE_BYTES`] expanded).
pub fn unzip(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    unzip_within(bytes, MAX_ARCHIVE_BYTES)
}

/// [`unzip`] with an aggregate budget for the expanded members: a member
/// is inflated through a bounded reader, never past what remains.
pub fn unzip_within(bytes: &[u8], budget: usize) -> Result<HashMap<String, Vec<u8>>> {
    let bad = || EsError::illegal_argument("model definition is not a valid TorchScript archive");
    let too_big = || EsError::illegal_argument(format!("model definition archive expands beyond the limit of [{budget}] bytes"));
    let u16_at = |o: usize| -> Option<usize> { bytes.get(o..o.checked_add(2)?).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize) };
    let u32_at = |o: usize| -> Option<usize> { bytes.get(o..o.checked_add(4)?).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize) };
    // End of central directory record (scan back over a trailing comment).
    let eocd = (0..bytes.len().saturating_sub(21)).rev().find(|&i| bytes[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06])).ok_or_else(bad)?;
    let entries = u16_at(eocd + 10).ok_or_else(bad)?;
    if entries > MAX_ARCHIVE_ENTRIES {
        return Err(bad());
    }
    let mut cd = u32_at(eocd + 16).ok_or_else(bad)?;
    let mut out = HashMap::new();
    let mut left = budget;
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(bad);
    for _ in 0..entries {
        if u32_at(cd) != Some(0x0201_4b50) {
            return Err(bad());
        }
        let method = u16_at(add(cd, 10)?).ok_or_else(bad)?;
        let csize = u32_at(add(cd, 20)?).ok_or_else(bad)?;
        let name_len = u16_at(add(cd, 28)?).ok_or_else(bad)?;
        let extra_len = u16_at(add(cd, 30)?).ok_or_else(bad)?;
        let comment_len = u16_at(add(cd, 32)?).ok_or_else(bad)?;
        let local = u32_at(add(cd, 42)?).ok_or_else(bad)?;
        let name_start = add(cd, 46)?;
        let name = String::from_utf8_lossy(bytes.get(name_start..add(name_start, name_len)?).ok_or_else(bad)?).into_owned();
        cd = add(add(add(name_start, name_len)?, extra_len)?, comment_len)?;
        if u32_at(local) != Some(0x0403_4b50) {
            return Err(bad());
        }
        let start = add(add(add(local, 30)?, u16_at(add(local, 26)?).ok_or_else(bad)?)?, u16_at(add(local, 28)?).ok_or_else(bad)?)?;
        let raw = bytes.get(start..add(start, csize)?).ok_or_else(bad)?;
        let data = match method {
            0 => {
                if raw.len() > left {
                    return Err(too_big());
                }
                raw.to_vec()
            }
            8 => {
                let mut v = Vec::new();
                // One byte past the budget tells "too big" from "exactly fits".
                flate2::read::DeflateDecoder::new(raw).take(left as u64 + 1).read_to_end(&mut v).map_err(|_| bad())?;
                if v.len() > left {
                    return Err(too_big());
                }
                v
            }
            _ => return Err(bad()),
        };
        left -= data.len();
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
    depth: usize,
}

/// Deepest expression nesting the parser accepts (and so the evaluator
/// recurses); TorchScript emitted by tracing is shallow.
pub const MAX_EXPR_DEPTH: usize = 128;
/// Deepest `for` block nesting.
pub const MAX_BLOCK_DEPTH: usize = 32;
/// Largest tensor (elements) one operation may create.
pub const MAX_TENSOR_ELEMENTS: usize = 50_000_000;
/// Elements all tensors of one run may allocate together.
pub const MAX_RUN_ELEMENTS: usize = 200_000_000;

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
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            return Err(unsupported(&format!("expressions nested deeper than {MAX_EXPR_DEPTH}")));
        }
        let e = self.expr_inner();
        self.depth -= 1;
        e
    }
    fn expr_inner(&mut self) -> Result<Expr> {
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
    parse_block_at(lines, i, indent, 0)
}

fn parse_block_at(lines: &[&str], i: &mut usize, indent: usize, depth: usize) -> Result<Vec<Stmt>> {
    if depth > MAX_BLOCK_DEPTH {
        return Err(unsupported(&format!("blocks nested deeper than {MAX_BLOCK_DEPTH}")));
    }
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
            let body = parse_block_at(lines, i, body_indent.max(indent + 1), depth + 1)?;
            out.push(Stmt::For(var.trim().to_string(), parse_expr(it)?, body));
        } else if s == "pass" {
        } else {
            let toks = lex(s)?;
            // `name = expr` / `name : Type = expr`
            let eq = toks.iter().position(|t| *t == Tok::Sym('='));
            match (toks.first(), eq) {
                (Some(Tok::Ident(n)), Some(p)) if p == 1 || toks.get(1) == Some(&Tok::Sym(':')) => {
                    let mut pp = P { t: toks[p + 1..].to_vec(), i: 0, depth: 0 };
                    out.push(Stmt::Assign(n.clone(), pp.expr()?));
                }
                _ => out.push(Stmt::Expr(parse_expr(s)?)),
            }
        }
    }
    Ok(out)
}

fn parse_expr(s: &str) -> Result<Expr> {
    let mut p = P { t: lex(s)?, i: 0, depth: 0 };
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
        let mut st = Interp { env, rng: Rng(0x853c_49e6_748f_ea9b), steps: 0, allocated: 0 };
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
    /// Tensor elements allocated so far in this run.
    allocated: usize,
}

/// Elements of a tensor of `shape`, refusing overflow and the per-tensor limit.
fn checked_numel(shape: &[usize]) -> Result<usize> {
    let n = shape.iter().try_fold(1usize, |acc, d| acc.checked_mul(*d)).ok_or_else(|| unsupported("tensor too large"))?;
    if n > MAX_TENSOR_ELEMENTS {
        return Err(unsupported("tensor too large"));
    }
    Ok(n)
}

const MAX_STEPS: u64 = 10_000_000;

impl Interp {
    /// Admit `n` more tensor elements against the per-tensor and per-run budgets.
    fn admit(&mut self, n: usize) -> Result<()> {
        if n > MAX_TENSOR_ELEMENTS {
            return Err(unsupported("tensor too large"));
        }
        self.allocated = self.allocated.checked_add(n).filter(|a| *a <= MAX_RUN_ELEMENTS).ok_or_else(|| unsupported("tensors exceed the memory budget of one inference"))?;
        Ok(())
    }

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
                            // Each iteration is a step: a range longer than the step budget cannot finish.
                            if hi.saturating_sub(lo) > MAX_STEPS as i64 {
                                return Err(unsupported("program too long"));
                            }
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
                let n = checked_numel(&shape)?;
                self.admit(n)?;
                let (data, dt) = match f {
                    "torch.ones" => (vec![1.0; n], dtype(DType::Float)),
                    "torch.zeros" => (vec![0.0; n], dtype(DType::Float)),
                    "torch.full" => (vec![arg(1)?.as_f64()?; n], dtype(DType::Float)),
                    _ => ((0..n).map(|_| self.rng.next_f64()).collect(), DType::Float),
                };
                Val::Tensor(Tensor::new(shape, data, dt))
            }
            "torch.arange" => {
                let n = usize::try_from(arg(0)?.as_i64()?.max(0)).map_err(|_| unsupported("tensor too large"))?;
                self.admit(n)?;
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
                // PyTorch semantics (finding 357): `add`/`sub` scale the
                // second operand by `alpha`, `div` honours `rounding_mode`,
                // and tensors broadcast per trailing dimension.
                let (x, y) = (arg(0)?, arg(1)?);
                let alpha = match k.get("alpha").cloned().or_else(|| a.get(2).cloned()) {
                    Some(v) if matches!(f, "torch.add" | "torch.sub") => Some(v),
                    Some(_) | None => None,
                };
                let rounding = match k.get("rounding_mode").or_else(|| if f == "torch.div" { a.get(2) } else { None }) {
                    None | Some(Val::None) => None,
                    Some(Val::Str(m)) if m == "trunc" || m == "floor" => Some(m.clone()),
                    Some(_) => return Err(unsupported("div rounding_mode")),
                };
                let scale = match &alpha {
                    None => 1.0,
                    Some(v) => v.as_f64()?,
                };
                let op = |p: f64, q: f64| match f {
                    "torch.add" => p + scale * q,
                    "torch.sub" => p - scale * q,
                    "torch.mul" => p * q,
                    _ => match rounding.as_deref() {
                        Some("trunc") => (p / q).trunc(),
                        Some(_) => (p / q).floor(),
                        None => p / q,
                    },
                };
                let alpha_float = matches!(alpha, Some(Val::Float(_)));
                match (&x, &y) {
                    (Val::Int(p), Val::Int(q)) if (f != "torch.div" || rounding.is_some()) && !alpha_float => Val::Int(op(*p as f64, *q as f64) as i64),
                    (Val::Tensor(_), _) | (_, Val::Tensor(_)) => {
                        let (tx, ty) = (x.tensor()?, y.tensor()?);
                        let shape = broadcast_shape(&tx.shape, &ty.shape)?;
                        let n = checked_numel(&shape)?;
                        self.admit(n)?;
                        let (vx, vy) = (broadcast_values(&tx, &shape), broadcast_values(&ty, &shape));
                        let data: Vec<f64> = vx.iter().zip(&vy).map(|(p, q)| op(*p, *q)).collect();
                        let float = tx.dtype == DType::Float || ty.dtype == DType::Float || alpha_float || (f == "torch.div" && rounding.is_none());
                        let dt = if float { DType::Float } else { DType::Long };
                        let data = if dt == DType::Long { data.into_iter().map(f64::trunc).collect() } else { data };
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
            "torch.to" => {
                // `to(dtype)`, `to(device, dtype)`, `to(dtype=..)`: convert
                // the values; a device-only move keeps them (finding 357).
                let t = arg(0)?.tensor()?;
                let target = k.get("dtype").cloned().or_else(|| a.iter().skip(1).find(|v| matches!(v, Val::DType(_) | Val::Int(_))).cloned());
                match target {
                    None | Some(Val::None) => Val::Tensor(t),
                    Some(v) => {
                        let dt = match v {
                            Val::DType(d) => d,
                            Val::Int(c) => dtype_of_code(c),
                            _ => return Err(unsupported("to() dtype")),
                        };
                        let vals: Vec<f64> = t.values().into_iter().map(|x| match dt {
                            DType::Long => x.trunc(),
                            DType::Bool => (x != 0.0) as i64 as f64,
                            DType::Float => x,
                        }).collect();
                        Val::Tensor(Tensor::new(t.shape.clone(), vals, dt))
                    }
                }
            }
            _ => return Err(unsupported(f)),
        })
    }
}

/// The PyTorch broadcast shape of `a` and `b`: dimensions aligned from the
/// right, each pair equal or one of them 1.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(EsError::illegal_argument(format!(
                    "The size of tensor a ({x}) must match the size of tensor b ({y}) at non-singleton dimension {i}"
                )))
            }
        };
    }
    Ok(out)
}

/// `t`'s values expanded to `shape` (a broadcast of `t.shape`), row-major.
fn broadcast_values(t: &Tensor, shape: &[usize]) -> Vec<f64> {
    let v = t.values();
    let own = contiguous_strides(&t.shape);
    let lead = shape.len() - t.shape.len();
    // Stride of each output dimension in `t`'s values (0 where broadcast).
    let strides: Vec<usize> = (0..shape.len())
        .map(|i| if i < lead || t.shape[i - lead] == 1 { 0 } else { own[i - lead] })
        .collect();
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..n {
        out.push(v[idx.iter().zip(&strides).map(|(i, s)| i * s).sum::<usize>()]);
        for k in (0..idx.len()).rev() {
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tokenization (ES `BertTokenizer`: basic + WordPiece)

/// Token ids of `text` for a WordPiece vocabulary; unknown words map to
/// `[UNK]`. `lower`: `do_lower_case`; `special`: add `[CLS]` / `[SEP]`.
/// Words longer than this become `[UNK]` without a vocabulary search (ES
/// `WordPieceTokenFilter` `max_input_chars_per_word`).
pub const MAX_INPUT_CHARS_PER_WORD: usize = 100;

#[cfg(test)]
thread_local! {
    /// Tests: vocabulary lookups made by [`wordpiece`].
    pub static LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// BERT basic tokenization + WordPiece of `text` to at most `max_len` ids
/// (with `[CLS]` / `[SEP]` when `special`); whether the text was truncated.
/// The work is bounded by the sequence limit, not the input (finding 160):
/// words are split lazily and tokenizing stops as soon as one id more than
/// fits has been produced; a word longer than
/// [`MAX_INPUT_CHARS_PER_WORD`] is `[UNK]`, so a match search costs at most
/// that many squared character steps.
pub fn wordpiece(text: &str, vocab: &[String], lower: bool, special: bool, max_len: usize) -> (Vec<i64>, bool) {
    let index: HashMap<&str, i64> = vocab.iter().enumerate().map(|(i, t)| (t.as_str(), i as i64)).collect();
    let unk = index.get("[UNK]").copied().unwrap_or(0);
    let budget = if special { max_len.saturating_sub(2) } else { max_len };
    let mut ids: Vec<i64> = Vec::new();
    let mut piece = String::new();
    let mut emit = |w: &str, ids: &mut Vec<i64>| {
        let lowered;
        let w = if lower {
            lowered = w.to_lowercase();
            lowered.as_str()
        } else {
            w
        };
        #[cfg(test)]
        LOOKUPS.with(|n| n.set(n.get() + 1));
        if let Some(id) = index.get(w) {
            ids.push(*id);
            return;
        }
        let bounds: Vec<usize> = w.char_indices().map(|(b, _)| b).chain(std::iter::once(w.len())).collect();
        let chars = bounds.len() - 1;
        if chars > MAX_INPUT_CHARS_PER_WORD {
            ids.push(unk);
            return;
        }
        // Greedy longest-match-first WordPiece.
        let mut start = 0;
        let mut pieces = Vec::new();
        while start < chars {
            let mut end = chars;
            let mut found = None;
            while start < end {
                let sub = &w[bounds[start]..bounds[end]];
                #[cfg(test)]
                LOOKUPS.with(|n| n.set(n.get() + 1));
                let hit = if start > 0 {
                    piece.clear();
                    piece.push_str("##");
                    piece.push_str(sub);
                    index.get(piece.as_str())
                } else {
                    index.get(sub)
                };
                if let Some(id) = hit {
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
                    ids.push(unk);
                    return;
                }
            }
        }
        ids.extend(pieces);
    };
    let mut cur_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if ids.len() > budget {
            break;
        }
        let sep = c.is_whitespace() || c.is_control();
        // Punctuation and (BERT `tokenize_cjk_chars`, finding 343) each
        // CJK ideograph are words of their own.
        let punct = c.is_ascii_punctuation() || (!c.is_ascii() && !c.is_alphanumeric()) || is_cjk(c);
        if sep || punct {
            if let Some(s) = cur_start.take() {
                emit(&text[s..i], &mut ids);
            }
            if punct && ids.len() <= budget {
                emit(&text[i..i + c.len_utf8()], &mut ids);
            }
        } else if cur_start.is_none() {
            cur_start = Some(i);
        }
    }
    if let Some(s) = cur_start {
        if ids.len() <= budget {
            emit(&text[s..], &mut ids);
        }
    }
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

/// BERT's CJK ideograph ranges (`BasicTokenizer.isCjkChar`).
fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// A model vocabulary (`PUT _ml/trained_models/{id}/vocabulary`): the
/// tokens, and the BPE `merges` / Unigram `scores` its tokenizer needs.
#[derive(Default)]
pub struct NlpVocab {
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub scores: Vec<f64>,
    #[cfg(feature = "candle-models")]
    built: std::sync::Mutex<HashMap<String, std::sync::Arc<tokenizers::Tokenizer>>>,
}

impl std::fmt::Debug for NlpVocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpVocab").field("tokens", &self.tokens.len()).field("merges", &self.merges.len()).field("scores", &self.scores.len()).finish()
    }
}

impl NlpVocab {
    pub fn from_body(v: &serde_json::Value) -> NlpVocab {
        let strs = |k: &str| v[k].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()).unwrap_or_default();
        NlpVocab {
            tokens: strs("vocabulary"),
            merges: strs("merges"),
            scores: v["scores"].as_array().map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect()).unwrap_or_default(),
            ..Default::default()
        }
    }

    fn id(&self, token: &str) -> Option<i64> {
        self.tokens.iter().position(|t| t == token).map(|i| i as i64)
    }
}

/// `(class, separator, unknown)` special tokens of a tokenizer family.
/// The unknown token of a tokenizer family.
pub fn unknown_token(family: &str) -> &'static str {
    specials(family).2
}

fn specials(family: &str) -> (&'static str, &'static str, &'static str) {
    match family {
        "bert" => ("[CLS]", "[SEP]", "[UNK]"),
        "mpnet" => ("<s>", "</s>", "[UNK]"),
        _ => ("<s>", "</s>", "<unk>"),
    }
}

/// Token ids of `text` for an uploaded model whose `tokenization` is
/// `{family: cfg}` and whether the input was truncated (finding 343, ES
/// `NlpTokenizer.build`): BERT / MPNet use WordPiece after BERT's basic
/// tokenization (CJK ideographs split, accents stripped when lowercasing),
/// RoBERTa byte-level BPE over the vocabulary's `merges`, XLM-RoBERTa a
/// Unigram model over its `scores`, each with its own special tokens. A
/// family this node cannot tokenize is refused, never approximated with
/// another family's rules.
pub fn tokenize(family: &str, cfg: &serde_json::Value, text: &str, v: &NlpVocab) -> Result<(Vec<i64>, bool)> {
    let lower = cfg.get("do_lower_case").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let special = cfg.get("with_special_tokens").and_then(serde_json::Value::as_bool).unwrap_or(true);
    let max_len = cfg.get("max_sequence_length").and_then(serde_json::Value::as_u64).unwrap_or(512) as usize;
    let mut ids = match family {
        "bert" | "mpnet" | "roberta" | "xlm_roberta" => subword_ids(family, cfg, text, v, lower)?,
        other => return Err(EsError::illegal_argument(format!("tokenization [{other}] is not supported for uploaded models on this node"))),
    };
    let budget = if special { max_len.saturating_sub(2) } else { max_len };
    let truncated = ids.len() > budget;
    ids.truncate(budget);
    if special {
        let (cls, sep, unk) = specials(family);
        let unk = v.id(unk).unwrap_or(0);
        ids.insert(0, v.id(cls).unwrap_or(unk));
        ids.push(v.id(sep).unwrap_or(unk));
    }
    Ok((ids, truncated))
}

/// The subword ids of `text` (no special tokens).
#[cfg(feature = "candle-models")]
fn subword_ids(family: &str, cfg: &serde_json::Value, text: &str, v: &NlpVocab, lower: bool) -> Result<Vec<i64>> {
    use serde_json::json;
    if matches!(family, "bert" | "mpnet") && v.id("[UNK]").is_none() {
        // No unknown token to map unknown words to: the plain WordPiece.
        return Ok(wordpiece(text, &v.tokens, lower, false, usize::MAX).0);
    }
    let fail = |e: String| EsError::new(500, "status_exception", format!("failed to build the [{family}] tokenizer: {e}"));
    let key = format!("{family}\u{0}{cfg}");
    let cached = v.built.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
    let tok = match cached {
        Some(t) => t,
        None => {
            let vocab_map = || serde_json::Value::Object(v.tokens.iter().enumerate().map(|(i, t)| (t.clone(), json!(i))).collect());
            let spec = match family {
                "bert" | "mpnet" => json!({
                    "version": "1.0",
                    "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": lower, "lowercase": lower},
                    "pre_tokenizer": {"type": "BertPreTokenizer"},
                    "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab_map()}
                }),
                "roberta" => {
                    if v.merges.is_empty() {
                        return Err(EsError::illegal_argument("the [roberta] tokenizer requires the vocabulary's [merges]"));
                    }
                    let prefix = cfg.get("add_prefix_space").and_then(serde_json::Value::as_bool).unwrap_or(false);
                    json!({
                        "version": "1.0",
                        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": prefix, "trim_offsets": true, "use_regex": true},
                        "model": {"type": "BPE", "dropout": null, "unk_token": v.id("<unk>").map(|_| "<unk>"), "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "vocab": vocab_map(), "merges": v.merges}
                    })
                }
                _ => {
                    if v.scores.len() != v.tokens.len() {
                        return Err(EsError::illegal_argument("the [xlm_roberta] tokenizer requires one vocabulary [scores] entry per token"));
                    }
                    let pieces: Vec<serde_json::Value> = v.tokens.iter().zip(&v.scores).map(|(t, s)| json!([t, s])).collect();
                    json!({
                        "version": "1.0",
                        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Nmt"}, {"type": "NFKC"}]},
                        "pre_tokenizer": {"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true},
                        "model": {"type": "Unigram", "unk_id": v.id("<unk>"), "vocab": pieces, "byte_fallback": false}
                    })
                }
            };
            let t: tokenizers::Tokenizer = spec.to_string().parse().map_err(|e: Box<dyn std::error::Error + Send + Sync>| fail(e.to_string()))?;
            let t = std::sync::Arc::new(t);
            v.built.lock().unwrap_or_else(|e| e.into_inner()).insert(key, t.clone());
            t
        }
    };
    let enc = tok.encode(text, false).map_err(|e| fail(e.to_string()))?;
    Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
}

/// Without the tokenizer library only BERT-style WordPiece is available.
#[cfg(not(feature = "candle-models"))]
fn subword_ids(family: &str, _cfg: &serde_json::Value, text: &str, v: &NlpVocab, lower: bool) -> Result<Vec<i64>> {
    match family {
        "bert" | "mpnet" => Ok(wordpiece(text, &v.tokens, lower, false, usize::MAX).0),
        other => Err(EsError::illegal_argument(format!("tokenization [{other}] is not supported for uploaded models on this node"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finding 160: tokenizing work is bounded by the sequence limit and
    /// the per-word cap, not by the input length.
    #[test]
    fn wordpiece_work_is_bounded() {
        let v: Vec<String> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "##a", "b"].iter().map(|s| s.to_string()).collect();
        let lookups = |text: &str, max_len: usize| {
            LOOKUPS.with(|n| n.set(0));
            let (ids, truncated) = wordpiece(text, &v, false, true, max_len);
            (ids, truncated, LOOKUPS.with(|n| n.get()))
        };
        // One long unknown word: [UNK] without a quadratic search.
        for len in [1_000usize, 100_000] {
            let (ids, _, n) = lookups(&"z".repeat(len), 16);
            assert_eq!(ids, vec![2, 1, 3]);
            assert!(n <= 2, "{len}: {n} lookups");
        }
        // At the cap, the search is at most cap^2 lookups.
        let (_, _, n) = lookups(&"z".repeat(MAX_INPUT_CHARS_PER_WORD), 16);
        assert!(n <= MAX_INPUT_CHARS_PER_WORD * MAX_INPUT_CHARS_PER_WORD + 1, "{n}");
        // Much more text than fits: the work stops at the budget.
        for words in [1_000usize, 100_000] {
            let text = vec!["b"; words].join(" ");
            let (ids, truncated, n) = lookups(&text, 8);
            assert_eq!(ids.len(), 8);
            assert!(truncated);
            assert!(n <= 8, "{words} words: {n} lookups");
        }
        // Short texts are unchanged.
        let (ids, truncated, _) = lookups("a aa b", 16);
        assert_eq!((ids, truncated), (vec![2, 4, 4, 5, 6, 3], false));
    }

    /// Finding 343: each tokenizer family uses its own rules and special
    /// tokens; BERT splits CJK ideographs and strips accents when
    /// lowercasing.
    #[cfg(feature = "candle-models")]
    #[test]
    fn tokenizer_families_follow_their_own_rules() {
        let vocab = |toks: &[&str]| NlpVocab { tokens: toks.iter().map(|s| s.to_string()).collect(), ..Default::default() };
        let cfg = |lower: bool| serde_json::json!({"do_lower_case": lower, "with_special_tokens": true, "max_sequence_length": 512});
        let bert = vocab(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "中", "文", "cafe", "noir"]);
        assert_eq!(tokenize("bert", &cfg(true), "中文 Café NOIR", &bert).unwrap().0, vec![2, 4, 5, 6, 7, 3]);
        // Cased: no lowercasing, no accent stripping.
        assert_eq!(tokenize("bert", &cfg(false), "cafe Café", &bert).unwrap().0, vec![2, 6, 1, 3]);
        let mpnet = vocab(&["<s>", "<pad>", "</s>", "[UNK]", "hello"]);
        assert_eq!(tokenize("mpnet", &cfg(true), "hello", &mpnet).unwrap().0, vec![0, 4, 2]);
        // RoBERTa: byte-level BPE with its merges, `<s>` / `</s>`.
        let mut rob = vocab(&["<s>", "<pad>", "</s>", "<unk>", "h", "i", "\u{120}", "hi", "\u{120}h", "\u{120}hi"]);
        rob.merges = vec!["\u{120} h".into(), "\u{120}h i".into(), "h i".into()];
        assert_eq!(tokenize("roberta", &cfg(false), "hi hi", &rob).unwrap().0, vec![0, 7, 9, 2]);
        // `add_prefix_space` makes the first word a space-prefixed one.
        assert_eq!(tokenize("roberta", &serde_json::json!({"add_prefix_space": true}), "hi", &rob).unwrap().0, vec![0, 9, 2]);
        // XLM-RoBERTa: Unigram over the scores, `▁` word prefix.
        let mut xlm = vocab(&["<s>", "<pad>", "</s>", "<unk>", "\u{2581}", "\u{2581}he", "llo", "\u{2581}hello", "h", "e", "l", "o"]);
        xlm.scores = vec![0.0, 0.0, 0.0, 0.0, -5.0, -3.0, -3.0, -1.0, -8.0, -8.0, -8.0, -8.0];
        assert_eq!(tokenize("xlm_roberta", &cfg(false), "hello", &xlm).unwrap().0, vec![0, 7, 2]);
        assert!(tokenize("bert_ja", &cfg(false), "x", &bert).is_err());
        // Truncation keeps room for the special tokens.
        let (ids, truncated) = tokenize("bert", &serde_json::json!({"do_lower_case": true, "max_sequence_length": 4}), "cafe noir cafe noir", &bert).unwrap();
        assert_eq!((ids, truncated), (vec![2, 6, 7, 3], true));
    }

    /// Finding 357: arithmetic broadcasts per trailing dimension, `alpha`
    /// scales, `rounding_mode` rounds and `to` converts.
    #[test]
    fn arithmetic_follows_pytorch_semantics() {
        let run = |body: &str| Program::parse(&forward(body)).unwrap().run(&[1]);
        // [2,1] + [3] -> [2,3]: out[i][j] = r[i] + j.
        let r = run("    torch.manual_seed(7)\n    return torch.rand([2, 1])\n").unwrap().values();
        let out = run("    torch.manual_seed(7)\n    a = torch.rand([2, 1])\n    return torch.add(a, torch.arange(3))\n").unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let want: Vec<f64> = (0..2).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| r[i] + j as f64).collect();
        assert_eq!(out.values(), want);
        // [1,2,1] * [1,1,2] -> [1,2,2].
        let out = run("    return torch.mul(torch.full([1, 2, 1], 3.), torch.ones([1, 1, 2]))\n").unwrap();
        assert_eq!((out.shape.clone(), out.values()), (vec![1, 2, 2], vec![3.0; 4]));
        // alpha, by keyword and by position.
        assert_eq!(run("    return torch.add(torch.arange(3), torch.ones([3]), alpha=2)\n").unwrap().values(), vec![2.0, 3.0, 4.0]);
        assert_eq!(run("    return torch.sub(torch.arange(3), torch.ones([3]), 3)\n").unwrap().values(), vec![-3.0, -2.0, -1.0]);
        assert_eq!(run("    return torch.div(torch.full([2], -7.), torch.full([2], 2.), rounding_mode=\"floor\")\n").unwrap().values(), vec![-4.0, -4.0]);
        assert_eq!(run("    return torch.div(torch.full([2], -7.), torch.full([2], 2.), rounding_mode=\"trunc\")\n").unwrap().values(), vec![-3.0, -3.0]);
        // to(dtype) converts; a long result of true division is float.
        let out = run("    return torch.to(torch.full([2], 2.7), 4)\n").unwrap();
        assert_eq!((out.values(), out.dtype), (vec![2.0, 2.0], DType::Long));
        let out = run("    return torch.to(torch.full([1], 2.7), dtype=11)\n").unwrap();
        assert_eq!((out.values(), out.dtype), (vec![1.0], DType::Bool));
        // Incompatible shapes are refused, not wrapped around.
        let e = run("    return torch.add(torch.ones([2]), torch.ones([3]))\n").err().unwrap();
        assert!(e.reason.contains("must match the size of tensor b (3)"), "{}", e.reason);
    }

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

    fn forward(body: &str) -> String {
        format!("class T(Module):\n  def forward(self: __torch__.T,\n    input_ids: Tensor) -> Tensor:\n{body}")
    }

    #[test]
    fn tensor_sizes_are_admitted_before_allocation() {
        for body in [
            "    return torch.arange(50000001)\n",
            "    return torch.zeros([4294967296, 4294967296, 4294967296])\n",
            "    return torch.ones([50000001])\n",
        ] {
            let e = Program::parse(&forward(body)).unwrap().run(&[1]).err().unwrap();
            assert!(e.reason.contains("tensor too large"), "{body}: {}", e.reason);
        }
        // Just under the per-tensor limit is fine; the per-run budget stops repetition.
        let mut st = Interp { env: HashMap::new(), rng: Rng(1), steps: 0, allocated: 0 };
        for _ in 0..MAX_RUN_ELEMENTS / MAX_TENSOR_ELEMENTS {
            st.admit(MAX_TENSOR_ELEMENTS).unwrap();
        }
        assert!(st.admit(1).unwrap_err().reason.contains("memory budget"));
        // A loop longer than the step budget fails before materialising its range.
        let p = Program::parse(&forward("    for i in range(1000000000000):\n      pass\n    return torch.zeros([1])\n")).unwrap();
        assert!(p.run(&[1]).err().unwrap().reason.contains("program too long"));
    }

    #[test]
    fn parser_nesting_is_bounded() {
        let deep = format!("    return {}1{}\n", "(".repeat(100_000), ")".repeat(100_000));
        let e = Program::parse(&forward(&deep)).err().expect("deep nesting is refused");
        assert!(e.reason.contains("nested deeper"), "{}", e.reason);
        let ok = format!("    return torch.zeros([{}1{}])\n", "(".repeat(100), ")".repeat(100));
        assert!(Program::parse(&forward(&ok)).is_ok());
        let mut body = String::new();
        for d in 0..40 {
            body.push_str(&format!("{}for i{d} in range(1):\n", " ".repeat(4 + 2 * d)));
        }
        body.push_str(&format!("{}pass\n", " ".repeat(4 + 80)));
        assert!(Program::parse(&forward(&body)).err().expect("deep blocks are refused").reason.contains("nested deeper"));
    }

    fn zip_one(name: &str, data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
        enc.write_all(data).unwrap();
        let comp = enc.finish().unwrap();
        let mut z = Vec::new();
        let le16 = |v: usize| (v as u16).to_le_bytes();
        let le32 = |v: usize| (v as u32).to_le_bytes();
        z.extend(le32(0x0403_4b50));
        z.extend([20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend(le32(comp.len()));
        z.extend(le32(data.len()));
        z.extend(le16(name.len()));
        z.extend(le16(0));
        z.extend(name.as_bytes());
        z.extend(&comp);
        let cd = z.len();
        z.extend(le32(0x0201_4b50));
        z.extend([20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend(le32(comp.len()));
        z.extend(le32(data.len()));
        z.extend(le16(name.len()));
        z.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend(le32(0));
        z.extend(name.as_bytes());
        let cd_len = z.len() - cd;
        z.extend(le32(0x0605_4b50));
        z.extend([0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend(le32(cd_len));
        z.extend(le32(cd));
        z.extend([0, 0]);
        z
    }

    #[test]
    fn archive_expansion_is_bounded() {
        let data = vec![b'a'; 4096];
        let z = zip_one("code/__torch__.py", &data);
        assert_eq!(unzip_within(&z, 4096).unwrap()["code/__torch__.py"].len(), 4096);
        let e = unzip_within(&z, 4095).unwrap_err();
        assert!(e.reason.contains("expands beyond the limit"), "{}", e.reason);
        // Malformed: truncated archives and offsets past the end are errors, not panics.
        for cut in [10, z.len() / 2, z.len() - 5] {
            assert!(unzip_within(&z[..cut], 1 << 20).is_err());
        }
        let mut bad = z.clone();
        let n = bad.len();
        bad[n - 6..n - 2].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(unzip_within(&bad, 1 << 20).is_err());
    }
}
