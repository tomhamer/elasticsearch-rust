//! Data-shaping processors written against ES 8.19.22 behaviour: `csv`
//! (port of `CsvParser`), `kv`, `json` (Jackson-style errors, duplicate-key
//! detection), `convert`, `bytes`, `fingerprint`, `sort`, `urldecode`.

use super::doc::{type_name, value_to_string, IngestDocument};
use super::jtext::{f32_json, java_double, java_tostring, nfe_float, nfe_int};
use super::processors::{ia, input, missing};
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use regex::Regex;
use serde_json::{Map, Number, Value};

/// `java.lang.ClassCastException` as ES renders it.
pub fn class_cast(from: &str, to: &str) -> EsError {
    EsError::new(500, "class_cast_exception", format!("class {from} cannot be cast to class {to} ({from} and {to} are in module java.base of loader 'bootstrap')"))
}

fn nfe(msg: String) -> EsError {
    EsError::new(500, "number_format_exception", msg)
}

// ------------------------------------------------------------ java String.split

/// `String.split(regex, limit)` semantics (limit 0 drops trailing empties).
pub fn java_split(re: &Regex, s: &str, limit: usize) -> Vec<String> {
    // `usize::MAX` stands for Java's negative limit: split everything, keep trailing empties.
    let keep_trailing = limit == usize::MAX;
    let limit = if keep_trailing { 0 } else { limit };
    let mut out: Vec<String> = Vec::new();
    let mut last = 0usize;
    for m in re.find_iter(s) {
        if limit > 0 && out.len() == limit - 1 {
            break;
        }
        if m.end() == 0 {
            // a zero-width match at the beginning never produces a leading empty string
            continue;
        }
        if m.start() == m.end() && m.start() == s.len() {
            continue;
        }
        out.push(s[last..m.start()].to_string());
        last = m.end();
    }
    if out.is_empty() && last == 0 {
        return vec![s.to_string()];
    }
    out.push(s[last..].to_string());
    if limit == 0 && !keep_trailing {
        while out.len() > 1 && out.last().map(|x| x.is_empty()).unwrap_or(false) {
            out.pop();
        }
        if out.len() == 1 && out[0].is_empty() && !s.is_empty() {
            out.clear();
        }
    }
    out
}

// ------------------------------------------------------------ csv

pub struct CsvP {
    pub field: String,
    pub targets: Vec<String>,
    pub sep: char,
    pub quote: char,
    pub trim: bool,
    pub empty_value: Option<Value>,
    pub ignore_missing: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum CsvState {
    Start,
    Unquoted,
    Quoted,
    QuotedEnd,
}

struct CsvRun<'a> {
    p: &'a CsvP,
    line: Vec<u16>,
    state: CsvState,
    builder: Vec<u16>,
    start: usize,
    cur: usize,
    header: usize,
    out: Vec<(String, Vec<u16>)>,
}

fn is_ws(c: u16) -> bool {
    c == 32 || c == 9
}

impl<'a> CsvRun<'a> {
    fn sub(&self, a: usize, b: usize) -> Vec<u16> {
        self.line[a.min(self.line.len())..b.min(self.line.len()).max(a.min(self.line.len()))].to_vec()
    }

    fn set_field(&mut self, end: usize) -> bool {
        let value = if self.builder.is_empty() {
            self.sub(self.start, end)
        } else {
            let tail = self.sub(self.start, end);
            self.builder.extend_from_slice(&tail);
            self.builder.clone()
        };
        let name = self.p.targets[self.header].clone();
        if !value.is_empty() {
            self.out.push((name, value));
        } else {
            self.out.push((name, Vec::new()));
        }
        self.header += 1;
        self.header == self.p.targets.len()
    }

    fn process_start(&mut self) -> bool {
        while self.cur < self.line.len() {
            let c = self.line[self.cur];
            if c == self.p.quote as u16 {
                self.state = CsvState::Quoted;
                self.builder.clear();
                self.start = self.cur + 1;
                return false;
            } else if c == self.p.sep as u16 {
                self.start += 1;
                self.builder.clear();
                let s = self.start;
                if self.set_field(s) {
                    return true;
                }
            } else if is_ws(c) {
                if self.p.trim {
                    self.start += 1;
                }
            } else {
                self.state = CsvState::Unquoted;
                self.builder.clear();
                return false;
            }
            self.cur += 1;
        }
        true
    }

    fn process_unquoted(&mut self) -> Result<bool> {
        let mut spaces = 0usize;
        while self.cur < self.line.len() {
            let c = self.line[self.cur];
            if c == 10 || c == 13 || c == self.p.quote as u16 {
                return Err(ia(format!("Illegal character inside unquoted field at {}", self.cur)));
            } else if c == self.p.sep as u16 {
                self.state = CsvState::Start;
                let e = self.cur - spaces;
                if self.set_field(e) {
                    return Ok(true);
                }
                self.start = self.cur + 1;
                return Ok(false);
            } else if self.p.trim && is_ws(c) {
                spaces += 1;
            } else {
                spaces = 0;
            }
            self.cur += 1;
        }
        Ok(false)
    }

    fn process_quoted(&mut self) {
        while self.cur < self.line.len() {
            if self.line[self.cur] == self.p.quote as u16 {
                self.state = CsvState::QuotedEnd;
                break;
            }
            self.cur += 1;
        }
    }

    fn process_quoted_end(&mut self) -> Result<bool> {
        let c = self.line[self.cur];
        if c == self.p.quote as u16 {
            let seg = self.sub(self.start, self.cur - 1);
            self.builder.extend_from_slice(&seg);
            self.builder.push(self.p.quote as u16);
            self.start = self.cur + 1;
            self.state = CsvState::Quoted;
            return Ok(false);
        }
        let mut should_set = true;
        while self.cur < self.line.len() {
            let c = self.line[self.cur];
            if c == self.p.sep as u16 {
                if should_set && self.set_field(self.cur - 1) {
                    return Ok(true);
                }
                self.start = self.cur + 1;
                self.state = CsvState::Start;
                return Ok(false);
            } else if is_ws(c) {
                if should_set {
                    if self.set_field(self.cur - 1) {
                        return Ok(true);
                    }
                    should_set = false;
                }
            } else {
                return Err(ia(format!("character '{}' after quoted field at {}", String::from_utf16_lossy(&[c]), self.cur)));
            }
            self.cur += 1;
        }
        Ok(true)
    }

    fn run(&mut self) -> Result<()> {
        'outer: while self.cur < self.line.len() {
            match self.state {
                CsvState::Start => {
                    if self.process_start() {
                        return Ok(());
                    }
                }
                CsvState::Unquoted => {
                    if self.process_unquoted()? {
                        return Ok(());
                    }
                }
                CsvState::Quoted => self.process_quoted(),
                CsvState::QuotedEnd => {
                    if self.process_quoted_end()? {
                        return Ok(());
                    }
                }
            }
            self.cur += 1;
            continue 'outer;
        }
        match self.state {
            CsvState::Unquoted => {
                let n = self.line.len();
                self.set_field(n);
            }
            CsvState::QuotedEnd => {
                let n = self.line.len() - 1;
                self.set_field(n);
            }
            CsvState::Quoted => return Err(ia("Unmatched quote")),
            CsvState::Start => {}
        }
        Ok(())
    }
}

impl Processor for CsvP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let mut run = CsvRun { p: self, line: s.encode_utf16().collect(), state: CsvState::Start, builder: Vec::new(), start: 0, cur: 0, header: 0, out: Vec::new() };
        let r = run.run();
        // fields set before a failure stay set, like ES
        for (k, v) in std::mem::take(&mut run.out) {
            if !v.is_empty() { doc.set(&k, Value::String(String::from_utf16(&v).map_err(|_| ia("invalid UTF16 CSV field"))?))?; } else if let Some(empty)=&self.empty_value { doc.set(&k,empty.clone())?; }
        }
        r?;
        Ok(())
    }
}

impl CsvP {
    pub(super) fn process_source(&self, doc: &mut super::source_doc::SourceDocument, empty: Option<&crate::source_json::Source>) -> PResult {
        let value = doc.get(&self.field)?.cloned();
        let Some(value) = value.filter(|v| v.text() != "null") else {
            if self.ignore_missing { return Ok(()); }
            return Err(ia(format!("field [{}] missing or null, cannot process it.", self.field)).into());
        };
        let value = crate::script::source_from_json(&value);
        if !matches!(&value,crate::script::Val::Str(_) | crate::script::Val::Utf16(_)) { return Err(ia("CSV field must be a string").into()); }
        let mut run = CsvRun { p: self, line: value.string_units(), state: CsvState::Start, builder: Vec::new(), start: 0, cur: 0, header: 0, out: Vec::new() };
        let result = run.run();
        for (key,value) in run.out {
            if value.is_empty() { if let Some(empty)=empty.filter(|v|v.text()!="null") {doc.set(&key,empty.clone())?;} }
            else {doc.set(&key,crate::script::source_to_json(&crate::script::Val::utf16(value))?)?;}
        }
        result?;
        Ok(())
    }
}

pub fn build_csv(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let targets = cfg.req_list_str("target_fields")?;
    if targets.is_empty() {
        return Err(cfg.err("target_fields", "target fields list can't be empty"));
    }
    let sep = cfg.str_or("separator", ",")?;
    let quote = cfg.str_or("quote", "\"")?;
    if sep.encode_utf16().count() != 1 {
        return Err(cfg.err("separator", "separator should be exactly one character"));
    }
    if quote.encode_utf16().count() != 1 {
        return Err(cfg.err("quote", "quote should be exactly one character"));
    }
    let trim = cfg.bool_or("trim", false)?;
    let empty_value = cfg.opt_value("empty_value").filter(|v| !v.is_null());
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(CsvP { field, targets, sep: sep.chars().next().unwrap(), quote: quote.chars().next().unwrap(), trim, empty_value, ignore_missing }))
}

// ------------------------------------------------------------ kv

pub struct KvP {
    pub field: String,
    pub field_split: Regex,
    pub value_split: Regex,
    pub value_split_src: String,
    pub target: Option<String>,
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
    pub prefix: String,
    pub trim_key: Option<String>,
    pub trim_value: Option<String>,
    pub strip_brackets: bool,
    pub ignore_missing: bool,
}

fn strip_chars(s: &str, chars: &Option<String>) -> String {
    match chars {
        Some(c) => s.trim_matches(|x| c.contains(x)).to_string(),
        None => s.to_string(),
    }
}

fn strip_brackets_java(s: &str) -> String {
    let mut cs: Vec<char> = s.chars().collect();
    if cs.last().map(|c| "])>\"'".contains(*c)).unwrap_or(false) {
        cs.pop();
    }
    if cs.first().map(|c| "([<\"'".contains(*c)).unwrap_or(false) {
        cs.remove(0);
    }
    cs.into_iter().collect()
}

impl Processor for KvP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None | Some(Value::Null) => return Err(ia(format!("field [{}] doesn't exist", self.field)).into()),
            Some(v) => v.clone(),
        };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let prefix_path = self.target.as_ref().map(|t| format!("{t}.")).unwrap_or_default();
        for kv in java_split(&self.field_split, &s, 0) {
            let parts = java_split(&self.value_split, &kv, 2);
            if parts.len() != 2 {
                return Err(ia(format!("field [{}] does not contain value_split [{}]", self.field, self.value_split_src)).into());
            }
            let key = strip_chars(&parts[0], &self.trim_key);
            let mut value = parts[1].clone();
            if self.strip_brackets {
                value = strip_brackets_java(&value);
            }
            let value = strip_chars(&value, &self.trim_value);
            if let Some(inc) = &self.include {
                if !inc.contains(&key) {
                    continue;
                }
            }
            if let Some(exc) = &self.exclude {
                if exc.contains(&key) {
                    continue;
                }
            }
            let path = format!("{prefix_path}{}{key}", self.prefix);
            if doc.has(&path) {
                doc.append(&path, Value::String(value), true)?;
            } else {
                doc.set(&path, Value::String(value))?;
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------ json

#[derive(Debug)]
pub struct JsonErr {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

struct JP<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
    allow_dup: bool,
}

fn line_col(s: &str, idx: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in s.char_indices() {
        if i >= idx {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

impl<'a> JP<'a> {
    fn err_at(&self, idx: usize, msg: String) -> JsonErr {
        let (line, col) = line_col(self.s, idx);
        JsonErr { msg, line, col }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eof_err(&self, what: &str) -> JsonErr {
        self.err_at(self.b.len(), format!("Unexpected end-of-input{what}"))
    }

    fn unexpected(&self, idx: usize, expecting: &str) -> JsonErr {
        let ch = self.s[idx..].chars().next().unwrap_or(' ');
        self.err_at(idx, format!("Unexpected character ('{}' (code {})): {}", ch, ch as u32, expecting))
    }

    fn parse_value(&mut self) -> std::result::Result<Value, JsonErr> {
        self.ws();
        if self.i >= self.b.len() {
            return Err(self.eof_err(": expected a value"));
        }
        match self.b[self.i] {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => Ok(Value::String(self.parse_string()?)),
            b'-' | b'0'..=b'9' => self.parse_number(),
            c if c.is_ascii_alphabetic() || c == b'_' || c == b'$' => {
                let start = self.i;
                let mut j = self.i;
                while j < self.b.len() && (self.b[j].is_ascii_alphanumeric() || self.b[j] == b'_' || self.b[j] == b'$') {
                    j += 1;
                }
                let word = &self.s[start..j];
                match word {
                    "true" => {
                        self.i = j;
                        Ok(Value::Bool(true))
                    }
                    "false" => {
                        self.i = j;
                        Ok(Value::Bool(false))
                    }
                    "null" => {
                        self.i = j;
                        Ok(Value::Null)
                    }
                    _ => Err(self.err_at(start, format!("Unrecognized token '{word}': was expecting (JSON String, Number, Array, Object or token 'null', 'true' or 'false')"))),
                }
            }
            _ => Err(self.unexpected(self.i, "expected a valid value (JSON String, Number, Array, Object or token 'null', 'true' or 'false')")),
        }
    }

    fn parse_string(&mut self) -> std::result::Result<String, JsonErr> {
        self.i += 1;
        let mut out = String::new();
        loop {
            if self.i >= self.b.len() {
                return Err(self.err_at(self.b.len(), "Unexpected end-of-input in VALUE_STRING".into()));
            }
            let c = self.s[self.i..].chars().next().unwrap();
            match c {
                '"' => {
                    self.i += 1;
                    return Ok(out);
                }
                '\\' => {
                    self.i += 1;
                    let e = self.s[self.i..].chars().next().ok_or_else(|| self.eof_err(" in character escape sequence"))?;
                    self.i += e.len_utf8();
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let hex = self.s.get(self.i..self.i + 4).ok_or_else(|| self.eof_err(" in character escape sequence"))?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| self.err_at(self.i, "Unexpected character in \\u escape sequence".into()))?;
                            self.i += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        other => return Err(self.err_at(self.i - 1, format!("Unrecognized character escape '{}' (code {})", other, other as u32))),
                    }
                }
                c if (c as u32) < 0x20 => {
                    return Err(self.err_at(self.i, format!("Illegal unquoted character ((CTRL-CHAR, code {})): has to be escaped using backslash to be included in string value", c as u32)));
                }
                c => {
                    out.push(c);
                    self.i += c.len_utf8();
                }
            }
        }
    }

    fn parse_number(&mut self) -> std::result::Result<Value, JsonErr> {
        let start = self.i;
        let mut j = self.i;
        if self.b[j] == b'-' {
            j += 1;
        }
        let ds = j;
        while j < self.b.len() && self.b[j].is_ascii_digit() {
            j += 1;
        }
        if j == ds {
            return Err(self.err_at(j.min(self.b.len().saturating_sub(1)), "Unexpected character in number".into()));
        }
        if self.b[ds] == b'0' && j - ds > 1 {
            return Err(self.err_at(j, "Invalid numeric value: Leading zeroes not allowed".into()));
        }
        let mut is_float = false;
        if j < self.b.len() && self.b[j] == b'.' {
            is_float = true;
            j += 1;
            let fs = j;
            while j < self.b.len() && self.b[j].is_ascii_digit() {
                j += 1;
            }
            if j == fs {
                return Err(self.err_at(j, "Unexpected character in number: Decimal point not followed by a digit".into()));
            }
        }
        if j < self.b.len() && (self.b[j] == b'e' || self.b[j] == b'E') {
            is_float = true;
            j += 1;
            if j < self.b.len() && (self.b[j] == b'+' || self.b[j] == b'-') {
                j += 1;
            }
            let es = j;
            while j < self.b.len() && self.b[j].is_ascii_digit() {
                j += 1;
            }
            if j == es {
                return Err(self.err_at(j, "Unexpected character in number: Exponent indicator not followed by a digit".into()));
            }
        }
        let txt = &self.s[start..j];
        self.i = j;
        if is_float {
            return Ok(txt.parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null));
        }
        match txt.parse::<i64>() {
            Ok(n) => Ok(Value::from(n)),
            Err(_) => Ok(txt.parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null)),
        }
    }

    fn parse_array(&mut self) -> std::result::Result<Value, JsonErr> {
        let start = self.i;
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Ok(Value::Array(out));
        }
        loop {
            out.push(self.parse_value()?);
            self.ws();
            if self.i >= self.b.len() {
                let (l, c) = line_col(self.s, start);
                return Err(self.err_at(self.b.len(), format!("Unexpected end-of-input: expected close marker for Array (start marker at [Source: (String)\"{}\"; line: {l}, column: {c}])", self.s)));
            }
            match self.b[self.i] {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Value::Array(out));
                }
                _ => return Err(self.unexpected(self.i, "was expecting comma to separate Array entries")),
            }
        }
    }

    fn parse_object(&mut self) -> std::result::Result<Value, JsonErr> {
        let start = self.i;
        self.i += 1;
        let mut out = Map::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Ok(Value::Object(out));
        }
        loop {
            self.ws();
            if self.i >= self.b.len() {
                let (l, c) = line_col(self.s, start);
                return Err(self.err_at(self.b.len(), format!("Unexpected end-of-input: was expecting closing quote for a string value (start marker at [Source: (String)\"{}\"; line: {l}, column: {c}])", self.s)));
            }
            if self.b[self.i] != b'"' {
                return Err(self.unexpected(self.i, "was expecting double-quote to start field name"));
            }
            let key = self.parse_string()?;
            self.ws();
            if self.i >= self.b.len() || self.b[self.i] != b':' {
                if self.i >= self.b.len() {
                    return Err(self.eof_err(": was expecting a colon to separate field name and value"));
                }
                return Err(self.unexpected(self.i, "was expecting a colon to separate field name and value"));
            }
            if out.contains_key(&key) && !self.allow_dup {
                // Jackson points just past the offending key
                return Err(self.err_at(self.i, format!("Duplicate field '{key}'")));
            }
            self.i += 1;
            let v = self.parse_value()?;
            out.insert(key, v);
            self.ws();
            if self.i >= self.b.len() {
                let (l, c) = line_col(self.s, start);
                return Err(self.err_at(self.b.len(), format!("Unexpected end-of-input: expected close marker for Object (start marker at [Source: (String)\"{}\"; line: {l}, column: {c}])", self.s)));
            }
            match self.b[self.i] {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Object(out));
                }
                _ => return Err(self.unexpected(self.i, "was expecting comma to separate Object entries")),
            }
        }
    }
}

/// Parse the first JSON value of `s`; returns it and whether more tokens follow
/// (`Err` inside the tuple = the trailing garbage's parse error).
pub fn parse_json_prefix(s: &str, allow_dup: bool) -> std::result::Result<(Value, Option<std::result::Result<(), JsonErr>>), JsonErr> {
    let mut p = JP { s, b: s.as_bytes(), i: 0, allow_dup };
    p.ws();
    if p.i >= p.b.len() {
        return Ok((Value::Null, None));
    }
    let v = p.parse_value()?;
    p.ws();
    if p.i >= p.b.len() {
        return Ok((v, None));
    }
    // is the next token valid?
    let mut q = JP { s, b: s.as_bytes(), i: p.i, allow_dup };
    let more = match q.parse_value() {
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    };
    Ok((v, Some(more)))
}

fn json_exception(e: &JsonErr, src: &str) -> EsError {
    let full = format!("{}\n at [Source: (String)\"{}\"; line: {}, column: {}]", e.msg, src, e.line, e.col);
    EsError::new(400, "x_content_parse_exception", format!("[{}:{}] {}", e.line, e.col, full)).caused_by(EsError::new(400, "json_parse_exception", full))
}

pub struct JsonP {
    pub field: String,
    pub target: Option<String>,
    pub add_to_root: bool,
    pub merge: bool,
    pub strict: bool,
    pub allow_dup: bool,
}

fn deep_merge(dst: &mut Map<String, Value>, src: Map<String, Value>) {
    for (k, v) in src {
        match (dst.get_mut(&k), v) {
            (Some(Value::Object(d)), Value::Object(s)) => deep_merge(d, s),
            (_, v) => {
                dst.insert(k, v);
            }
        }
    }
}

impl Processor for JsonP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = doc.require(&self.field)?.clone();
        let text = match &v {
            Value::Null => "null".to_string(),
            Value::String(s) => s.clone(),
            other => java_tostring(other),
        };
        let (parsed, more) = match parse_json_prefix(&text, self.allow_dup) {
            Ok(x) => x,
            Err(e) => return Err(json_exception(&e, &text).into()),
        };
        if self.strict {
            if let Some(m) = more {
                let msg = format!("The input {text} is not valid JSON and the strict_json_parsing parameter is true");
                return Err(match m {
                    Err(e) => ia(msg).caused_by(json_exception(&e, &text)),
                    Ok(()) => ia(msg),
                }
                .into());
            }
        }
        if self.add_to_root {
            let Value::Object(m) = parsed else {
                return Err(ia("cannot add non-map fields to root of document").into());
            };
            if self.merge {
                deep_merge(doc.ctx_map_mut(), m);
            } else {
                let ctx = doc.ctx_map_mut();
                for (k, v) in m {
                    ctx.insert(k, v);
                }
            }
        } else {
            let t = self.target.clone().unwrap_or_else(|| self.field.clone());
            doc.set(&t, parsed)?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------ convert

#[derive(Clone, Copy, PartialEq)]
pub enum ConvType {
    Integer,
    Long,
    Float,
    Double,
    Str,
    Boolean,
    Ip,
    Auto,
}

pub fn conv_name(t: ConvType) -> &'static str {
    match t {
        ConvType::Integer => "integer",
        ConvType::Long => "long",
        ConvType::Float => "float",
        ConvType::Double => "double",
        ConvType::Str => "string",
        ConvType::Boolean => "boolean",
        ConvType::Ip => "ip",
        ConvType::Auto => "auto",
    }
}

fn java_int_parse(s: &str, bits: u32) -> std::result::Result<i64, String> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit()) {
        return Err(nfe_int(s));
    }
    let v: i128 = body.parse().map_err(|_| nfe_int(s))?;
    let v = if neg { -v } else { v };
    let (lo, hi) = if bits == 32 { (i32::MIN as i128, i32::MAX as i128) } else { (i64::MIN as i128, i64::MAX as i128) };
    if v < lo || v > hi {
        return Err(nfe_int(s));
    }
    Ok(v as i64)
}

fn java_int_decode(s: &str, bits: u32) -> std::result::Result<i64, String> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let h = body.strip_prefix("0x").or(body.strip_prefix("0X")).ok_or_else(|| nfe_int(s))?;
    let v = i128::from_str_radix(h, 16).map_err(|_| nfe_int(s))?;
    let v = if neg { -v } else { v };
    let (lo, hi) = if bits == 32 { (i32::MIN as i128, i32::MAX as i128) } else { (i64::MIN as i128, i64::MAX as i128) };
    if v < lo || v > hi {
        return Err(nfe_int(s));
    }
    Ok(v as i64)
}

fn java_float_parse(s: &str) -> std::result::Result<f64, String> {
    let t = s.trim();
    let t = t.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(t);
    if t.is_empty() || t.contains(|c: char| c.is_ascii_alphabetic() && !matches!(c, 'e' | 'E')) && !matches!(t.trim_start_matches(['+', '-']), "NaN" | "Infinity") {
        return Err(nfe_float(s));
    }
    match t.trim_start_matches(['+', '-']) {
        "NaN" => return Ok(f64::NAN),
        "Infinity" => return Ok(if t.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY }),
        _ => {}
    }
    t.parse::<f64>().map_err(|_| nfe_float(s))
}

fn conv_fail(v: &Value, t: ConvType, cause: String) -> EsError {
    ia(format!("unable to convert [{}] to {}", java_tostring(v), conv_name(t))).caused_by(nfe(cause))
}

fn f64_value(f: f64) -> Value {
    Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
}

pub fn convert_one(v: &Value, t: ConvType) -> std::result::Result<Value, EsError> {
    let s = java_tostring(v);
    match t {
        ConvType::Integer | ConvType::Long => {
            let bits = if t == ConvType::Integer { 32 } else { 64 };
            let r = if s.starts_with("0x") || s.starts_with("-0x") { java_int_decode(&s, bits) } else { java_int_parse(&s, bits) };
            r.map(Value::from).map_err(|c| conv_fail(v, t, c))
        }
        ConvType::Float => java_float_parse(&s).map(|f| f32_json(f as f32)).map_err(|c| conv_fail(v, t, c)),
        ConvType::Double => java_float_parse(&s).map(f64_value).map_err(|c| conv_fail(v, t, c)),
        ConvType::Str => Ok(Value::String(s)),
        ConvType::Boolean => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            _ => match s.to_lowercase().as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Err(ia(format!("[{s}] is not a boolean value, cannot convert to boolean"))),
            },
        },
        ConvType::Ip => match v {
            Value::String(st) => super::net::inet_for_string(st).map(|_| Value::String(st.clone())).map_err(ia),
            other => Err(class_cast(type_name(other), "java.lang.String")),
        },
        ConvType::Auto => {
            let Value::String(st) = v else { return Ok(v.clone()) };
            if st.starts_with("0x") || st.starts_with("-0x") {
                if let Ok(i) = java_int_decode(st, 32).or_else(|_| java_int_decode(st, 64)) {
                    return Ok(Value::from(i));
                }
            }
            if let Ok(i) = java_int_parse(st, 32) {
                return Ok(Value::from(i));
            }
            if let Ok(i) = java_int_parse(st, 64) {
                return Ok(Value::from(i));
            }
            if let Ok(f) = java_float_parse(st) {
                if st.trim() == st && st.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')) {
                    return Ok(f64_value(f));
                }
            }
            match st.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => Ok(v.clone()),
            }
        }
    }
}

pub struct ConvertP {
    pub field: String,
    pub target: String,
    pub ty: ConvType,
    pub ignore_missing: bool,
}

impl Processor for ConvertP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(missing(&self.field).into()),
            Some(Value::Null) => return Err(ia(format!("Field [{}] is null, cannot be converted to type [{}]", self.field, conv_name(self.ty))).into()),
            Some(v) => v.clone(),
        };
        let out = match &v {
            Value::Array(a) => Value::Array(a.iter().map(|x| convert_one(x, self.ty)).collect::<std::result::Result<Vec<_>, _>>()?),
            other => convert_one(other, self.ty)?,
        };
        doc.set(&self.target, out)?;
        Ok(())
    }
}

pub fn build_convert(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let ty = cfg.req_str("type")?;
    let ty = match ty.as_str() {
        "integer" => ConvType::Integer,
        "long" => ConvType::Long,
        "float" => ConvType::Float,
        "double" => ConvType::Double,
        "string" => ConvType::Str,
        "boolean" => ConvType::Boolean,
        "ip" => ConvType::Ip,
        "auto" => ConvType::Auto,
        other => return Err(cfg.err("type", format!("type [{other}] not supported, cannot convert field."))),
    };
    let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(ConvertP { field, target, ty, ignore_missing }))
}

// ------------------------------------------------------------ AbstractStringProcessor family

pub struct StrValProc {
    pub field: String,
    pub target: String,
    pub ignore_missing: bool,
    pub f: Box<dyn Fn(&str) -> Result<Value> + Send + Sync>,
}

impl Processor for StrValProc {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let out = match v {
            Value::String(s) => (self.f)(&s)?,
            Value::Array(a) => {
                let mut out = Vec::with_capacity(a.len());
                for x in a {
                    match x {
                        Value::String(s) => out.push((self.f)(&s)?),
                        other => {
                            return Err(ia(format!("value [{}] of type [{}] in list field [{}] cannot be cast to [java.lang.String]", java_tostring(&other), type_name(&other), self.field)).into());
                        }
                    }
                }
                Value::Array(out)
            }
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        doc.set(&self.target, out)?;
        Ok(())
    }
}

pub fn build_bytes(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(StrValProc {
        field,
        target,
        ignore_missing,
        f: Box::new(|s| {
            super::util::parse_byte_size_value(s, "Ingest Field").map(Value::from)
        }),
    }))
}

/// `URLDecoder.decode(s, UTF_8)` with JDK 21's messages.
pub fn url_decode(s: &str) -> Result<String> {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' => {
                while i < b.len() && b[i] == b'%' {
                    if i + 2 >= b.len() {
                        return Err(ia("URLDecoder: Incomplete trailing escape (%) pattern"));
                    }
                    let (h1, h2) = (b[i + 1], b[i + 2]);
                    for c in [h1, h2] {
                        if !c.is_ascii_hexdigit() {
                            let ch = s[i + 1..].chars().find(|ch| !ch.is_ascii_hexdigit()).unwrap_or(c as char);
                            return Err(ia(format!("URLDecoder: Illegal hex characters in escape (%) pattern - not a hexadecimal digit: \"{}\" = {}", ch, ch as u32)));
                        }
                    }
                    out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(0));
                    i += 3;
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

// ------------------------------------------------------------ sort

pub struct SortP {
    pub field: String,
    pub target: String,
    pub desc: bool,
}

fn cmp_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if x.is_f64() || y.is_f64() {
                x.as_f64().partial_cmp(&y.as_f64()).unwrap_or(Ordering::Equal)
            } else {
                x.as_i64().cmp(&y.as_i64())
            }
        }
        (Value::String(x), Value::String(y)) => x.encode_utf16().cmp(y.encode_utf16()),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

impl Processor for SortP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = match doc.get(&self.field)? {
            None => return Err(missing(&self.field).into()),
            Some(Value::Null) => return Err(ia(format!("field [{}] is null, cannot sort.", self.field)).into()),
            Some(v) => v.clone(),
        };
        let Value::Array(mut a) = v else {
            return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.util.List]", self.field, type_name(&v))).into());
        };
        // Collections.sort compares a[i] to a[i-1] first: report that cast failure.
        for i in 1..a.len() {
            let (cur, prev) = (&a[i], &a[i - 1]);
            let comparable = |x: &Value| matches!(x, Value::Number(_) | Value::String(_) | Value::Bool(_));
            if !comparable(cur) {
                let t = type_name(cur);
                return Err(class_cast(t, "java.lang.Comparable").into());
            }
            if type_name(cur) != type_name(prev) {
                return Err(class_cast(type_name(prev), type_name(cur)).into());
            }
        }
        a.sort_by(cmp_values);
        if self.desc {
            a.reverse();
        }
        doc.set(&self.target, Value::Array(a))?;
        Ok(())
    }
}

// ------------------------------------------------------------ fingerprint

pub struct FingerprintP {
    pub fields: Vec<String>,
    pub target: String,
    pub salt: Vec<u8>,
    pub method: String,
    pub ignore_missing: bool,
}

fn fp_bytes(v: &Value) -> Result<Vec<u8>> {
    Ok(match v {
        Value::String(s) => s.as_bytes().to_vec(),
        Value::Number(n) => {
            if n.is_f64() {
                n.as_f64().unwrap_or(0.0).to_le_bytes().to_vec()
            } else {
                let i = n.as_i64().unwrap_or(0);
                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    (i as i32).to_le_bytes().to_vec()
                } else {
                    i.to_le_bytes().to_vec()
                }
            }
        }
        Value::Bool(true) => vec![1],
        Value::Bool(false) => vec![2],
        Value::Null => Vec::new(),
        other => return Err(ia(format!("cannot convert object of type [{}] to bytes", type_name(other)))),
    })
}

impl Processor for FingerprintP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        use base64::Engine;
        let mut buf = self.salt.clone();
        // ES pushes the sorted fields' values from last to first, then pops.
        let mut stack: Vec<Value> = Vec::new();
        for f in self.fields.iter().rev() {
            match doc.get(f)? {
                None | Some(Value::Null) => {
                    if !self.ignore_missing {
                        return Err(ia(format!("missing field [{f}] when calculating fingerprint")).into());
                    }
                }
                Some(v) => stack.push(v.clone()),
            }
        }
        if stack.is_empty() {
            return Ok(());
        }
        // entries are represented as 2-element marker arrays [KEY, key, value]
        enum Item {
            Val(Value),
            Entry(String, Value),
        }
        let mut st: Vec<Item> = stack.into_iter().map(Item::Val).collect();
        while let Some(it) = st.pop() {
            match it {
                Item::Entry(k, v) => {
                    buf.push(0);
                    buf.extend_from_slice(k.as_bytes());
                    st.push(Item::Val(v));
                }
                Item::Val(Value::Array(a)) => {
                    for x in a.into_iter().rev() {
                        st.push(Item::Val(x));
                    }
                }
                Item::Val(Value::Object(m)) => {
                    let mut es: Vec<(String, Value)> = m.into_iter().collect();
                    es.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
                    for (k, v) in es.into_iter().rev() {
                        st.push(Item::Entry(k, v));
                    }
                }
                Item::Val(other) => {
                    buf.push(0);
                    buf.extend_from_slice(&fp_bytes(&other)?);
                }
            }
        }
        let digest = match self.method.as_str() {
            "MD5" => super::util::md5(&buf),
            "SHA-256" => super::util::sha256(&buf),
            "SHA-512" => super::util::sha512(&buf),
            "MurmurHash3" => {
                // ES writes h1 then h2 big-endian
                let le = super::util::murmur3_128(&buf);
                let mut be = le[..8].to_vec();
                be.reverse();
                let mut h2 = le[8..].to_vec();
                h2.reverse();
                be.extend(h2);
                be
            }
            _ => super::util::sha1(&buf),
        };
        doc.set(&self.target, Value::String(base64::engine::general_purpose::STANDARD.encode(digest)))?;
        Ok(())
    }
}

pub fn build_fingerprint(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let mut fields = cfg.req_list_str("fields")?;
    if fields.is_empty() {
        return Err(cfg.err("fields", "must specify at least one field"));
    }
    fields.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    let target = cfg.str_or("target_field", "fingerprint")?;
    let salt = cfg.str_or("salt", "")?;
    let salt = if salt.trim().is_empty() { Vec::new() } else { salt.into_bytes() };
    let method = cfg.str_or("method", "SHA-1")?;
    if !matches!(method.as_str(), "MD5" | "SHA-1" | "SHA-256" | "SHA-512" | "MurmurHash3") {
        return Err(cfg.err("method", format!("[{method}] must be one of the supported hash methods [MD5,SHA-1,SHA-256,SHA-512,MurmurHash3]")));
    }
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(FingerprintP { fields, target, salt, method, ignore_missing }))
}

#[allow(dead_code)]
fn _keep(_: &str) -> String {
    value_to_string(&Value::Null) + &java_double(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn csv_run(line: &str, trim: bool, n: usize, empty: Option<Value>) -> std::result::Result<Vec<(String, Value)>, String> {
        let p = CsvP { field: "f".into(), targets: (0..n).map(|i| format!("t{i}")).collect(), sep: ',', quote: '"', trim, empty_value: empty, ignore_missing: false };
        let mut r = CsvRun { p: &p, line: line.encode_utf16().collect(), state: CsvState::Start, builder: Vec::new(), start: 0, cur: 0, header: 0, out: Vec::new() };
        r.run().map_err(|e| e.reason)?;
        Ok(r.out.into_iter().filter_map(|(k,v)| if v.is_empty() { p.empty_value.clone().map(|v|(k,v)) } else { Some((k,Value::String(String::from_utf16(&v).unwrap()))) }).collect())
    }

    #[test]
    fn csv_cases() {
        assert_eq!(csv_run("1,2,", false, 3, None).unwrap().len(), 2);
        assert_eq!(csv_run("\"a\",b", false, 3, None).unwrap()[0].1, json!("a"));
        assert_eq!(csv_run("\"a\"x,b", false, 3, None).unwrap_err(), "character 'x' after quoted field at 3");
        assert_eq!(csv_run("a\"b,c", false, 3, None).unwrap_err(), "Illegal character inside unquoted field at 1");
        assert_eq!(csv_run("\"a,b", false, 3, None).unwrap_err(), "Unmatched quote");
        assert_eq!(csv_run("a, \"b\" ,c", false, 3, None).unwrap().len(), 3);
        assert_eq!(csv_run("\"a \"\"q\"\" b\",2", false, 2, None).unwrap()[0].1, json!("a \"q\" b"));
        assert_eq!(csv_run(" 1 ,2", true, 2, None).unwrap()[0].1, json!("1"));
    }

    #[test]
    fn split_semantics() {
        let re = Regex::new(" ").unwrap();
        assert_eq!(java_split(&re, "a  b ", 0), vec!["a", "", "b"]);
        assert_eq!(java_split(&re, "", 0), vec![""]);
        let re = Regex::new("=").unwrap();
        assert_eq!(java_split(&re, "a=b=c", 2), vec!["a", "b=c"]);
        assert_eq!(java_split(&re, "a", 2), vec!["a"]);
        assert_eq!(java_split(&re, "a=", 2), vec!["a", ""]);
    }

    #[test]
    fn json_errors() {
        let e = parse_json_prefix("nope", false).unwrap_err();
        assert_eq!((e.line, e.col), (1, 1));
        assert!(e.msg.starts_with("Unrecognized token 'nope'"));
        let e = parse_json_prefix(r#"{"a":1,"a":2}"#, false).unwrap_err();
        assert_eq!(e.col, 11);
        let (_, more) = parse_json_prefix(r#"{"a":1} trailing"#, false).unwrap();
        assert_eq!(more.unwrap().unwrap_err().col, 9);
    }

    #[test]
    fn conversions() {
        assert!(convert_one(&json!("1.5"), ConvType::Integer).is_err());
        assert_eq!(convert_one(&json!("0x1F"), ConvType::Integer).unwrap(), json!(31));
        assert_eq!(convert_one(&json!("99999999999"), ConvType::Float).unwrap().as_f64().unwrap(), 1e11);
        let e = convert_one(&json!("1.2.3.4"), ConvType::Double).unwrap_err();
        assert_eq!(e.caused_by.unwrap().reason, "multiple points");
    }

    #[test]
    fn urldecode_errors() {
        assert_eq!(url_decode("a%20b+c").unwrap(), "a b c");
        assert_eq!(url_decode("%zz").unwrap_err().reason, "URLDecoder: Illegal hex characters in escape (%) pattern - not a hexadecimal digit: \"z\" = 122");
        assert_eq!(url_decode("%2").unwrap_err().reason, "URLDecoder: Incomplete trailing escape (%) pattern");
    }
}
