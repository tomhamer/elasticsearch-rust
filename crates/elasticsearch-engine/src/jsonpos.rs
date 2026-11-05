//! Position-aware request-body JSON: Jackson-compatible syntax errors, a
//! token index (JSON path -> line/column) and the resolver that turns the
//! position *hints* parsers attach to an [`EsError`] into the exact
//! `[line:col]` prefixes and `line`/`col` fields Elasticsearch reports.
//!
//! ES parses bodies with a streaming parser, so every error carries the
//! location of the token the parser was on. Our parsers work on an already
//! built `serde_json::Value`, which has no positions; instead they attach
//! hints (extras whose key starts with `@`, never rendered) naming the key
//! the error is about, and the HTTP layer resolves them against the raw
//! body text once, only when a request fails. Lines and columns count
//! bytes, 1-based, like Jackson's `UTF8StreamJsonParser`.
//!
//! Hints (see [`EsError::at_key`] and friends):
//! - `@k`  the field name the error is about
//! - `@in` an enclosing key that disambiguates `@k` (the query / agg name)
//! - `@c`  cursor: `key` (the FIELD_NAME token), `value` (first token of
//!   its value, default), `end` (closing bracket of the value), `after`
//!   (just past a scalar value), `root` (first token of the body) or
//!   `inherit` (the position where the wrapped cause failed)
//! - `@p`  prepend `[l:c] ` to the reason: `key` (the field name's start)
//!   or `cursor`
//! - `@lc` also set `line` / `col` from the cursor (ParsingException style)

use crate::error::EsError;
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub line: u64,
    pub col: u64,
}

/// What the scanner found at the end of the body after the main value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trailing {
    None,
    /// A further JSON token (ES: `Unexpected token [START_OBJECT] found after the main object.`).
    Token(&'static str, Pos),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seg {
    Key(usize, usize),
    Idx(usize),
}

/// One recorded token: a field name (with its value), an array element or the root value.
#[derive(Clone, Debug)]
struct Tok {
    /// Byte range of the key (quotes included); `None` for array elements / root.
    key: Option<(usize, usize)>,
    /// Position of the key's opening quote.
    key_pos: Pos,
    /// First byte of the value and its position.
    val_pos: Pos,
    /// Position of the value's last byte (closing bracket / quote / last digit).
    end_pos: Pos,
    /// Position just past the value.
    after_pos: Pos,
    /// Path segments of the enclosing containers (keys as byte ranges).
    depth: usize,
    path: Vec<Seg>,
    val_kind: &'static str,
}

pub struct Index<'a> {
    text: &'a [u8],
    toks: Vec<Tok>,
    root: Option<Pos>,
}

struct Scanner<'a> {
    b: &'a [u8],
    p: usize,
    line: u64,
    line_start: usize,
    record: bool,
    toks: Vec<Tok>,
    path: Vec<Seg>,
    root_pos: Option<Pos>,
    /// Open containers: (`[`/`{`, start position).
    stack: Vec<(u8, Pos)>,
    last_tok: Pos,
    /// Byte ranges of comments (stripped before `serde_json` parses the body).
    comments: Vec<(usize, usize)>,
}

/// A Jackson-shaped syntax error.
#[derive(Clone, Debug)]
pub struct SyntaxError {
    pub message: String,
    pub pos: Pos,
    pub eof: bool,
    /// Start of the last token read before the failure (ES wrappers such as
    /// `[l:c] failed to parse: ...` report the parser's current token).
    pub last_tok: Pos,
    /// A failure of a YAML / CBOR / Smile body (see [`ForeignError`]).
    pub foreign: Option<Box<ForeignError>>,
}

/// Prefix of a request body that could not be decoded from YAML / CBOR /
/// Smile: the HTTP layer replaces such a body with this marker followed by
/// the JSON-serialized [`ForeignError`], so every body parser reports the
/// format's own failure.
pub const FOREIGN_MAGIC: &[u8] = b"\0es-xcontent-error\0";

/// The decode failure of a non-JSON body, already in ES's shape.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ForeignError {
    /// The exception chain, outermost (`x_content_parse_exception`) first.
    pub chain: Vec<(String, String)>,
    pub pos: (u64, u64),
    pub last_tok: (u64, u64),
    /// Positions are byte offsets (binary formats report line `-1`).
    pub binary: bool,
    /// The body held a further document after the main one (YAML `---`):
    /// the token kind ES reports.
    pub trailing: Option<String>,
}

impl ForeignError {
    pub fn to_body(&self) -> Vec<u8> {
        let mut b = FOREIGN_MAGIC.to_vec();
        b.extend_from_slice(&serde_json::to_vec(self).unwrap_or_default());
        b
    }

    pub fn from_body(text: &[u8]) -> Option<ForeignError> {
        serde_json::from_slice(text.strip_prefix(FOREIGN_MAGIC)?).ok()
    }

    /// The request failure ES reports when no handler-specific wrapper applies.
    pub fn to_es_error(&self) -> EsError {
        match &self.trailing {
            Some(kind) => {
                let e = EsError::parsing(format!("Unexpected token [{kind}] found after the main object."));
                if self.binary { e } else { e.with("line", self.pos.0).with("col", self.pos.1) }
            }
            None => Self::chain_error(&self.chain),
        }
    }

    fn chain_error(chain: &[(String, String)]) -> EsError {
        let mut it = chain.iter().rev();
        let (t, r) = it.next().cloned().unwrap_or_default();
        let mut e = EsError::new(400, &t, r);
        for (t, r) in it {
            e = EsError::new(400, t, r.clone()).caused_by(e);
        }
        e
    }
}

impl SyntaxError {
    /// The `[l:c] failed to parse: ...` document failure (`DocumentParser`).
    pub fn doc_failure(&self, source: &[u8]) -> EsError {
        let inner = self.to_error(source);
        EsError::document_parsing(format!("[{}:{}] failed to parse: {}", self.last_tok.line, self.last_tok.col, inner.reason)).caused_by(inner)
    }

    /// A search request body failure (`RestSearchAction`): JSON syntax errors
    /// are `Failed to parse`, YAML ones carry the YAML message.
    pub fn search_failure(&self, source: &[u8]) -> EsError {
        let inner = self.to_error(source);
        match &self.foreign {
            Some(f) if f.chain.len() > 2 && !f.binary => EsError::parsing(inner.reason.clone()).with("line", self.pos.line).with("col", self.pos.col.saturating_sub(1)).caused_by(inner),
            Some(_) => inner,
            None => EsError::parsing("Failed to parse").with("line", self.last_tok.line).with("col", self.last_tok.col).caused_by(inner),
        }
    }

    /// The ES error: `x_content_parse_exception` (`x_content_e_o_f_exception`
    /// for an unexpected end of input) over the Jackson exception.
    pub fn to_error(&self, source: &[u8]) -> EsError {
        if let Some(f) = &self.foreign {
            return ForeignError::chain_error(&f.chain);
        }
        let full = format!("{}\n at [Source: {}; line: {}, column: {}]", self.message, source_desc(source), self.pos.line, self.pos.col);
        if self.eof {
            EsError::new(400, "x_content_e_o_f_exception", format!("[{}:{}] Unexpected end of file", self.pos.line, self.pos.col)).caused_by(EsError::new(400, "json_e_o_f_exception", full))
        } else {
            EsError::new(400, "x_content_parse_exception", format!("[{}:{}] {}", self.pos.line, self.pos.col, full)).caused_by(EsError::new(400, "json_parse_exception", full))
        }
    }
}

/// Jackson's `ContentReference` description of a `byte[]` source: the first
/// 500 bytes, control characters escaped, `[truncated N bytes]` suffix.
pub fn source_desc(src: &[u8]) -> String {
    const MAX: usize = 500;
    let (shown, trimmed) = if src.len() > MAX { (&src[..MAX], src.len() - MAX) } else { (src, 0) };
    let mut s = String::from("(byte[])\"");
    // `ContentReference._appendEscaped`: ISO control characters other than
    // CR and LF are escaped.
    for ch in String::from_utf8_lossy(shown).chars() {
        if ch.is_control() && ch != '\n' && ch != '\r' {
            s.push_str(&format!("\\u{:04X}", ch as u32));
        } else {
            s.push(ch);
        }
    }
    s.push('"');
    if trimmed > 0 {
        s.push_str(&format!("[truncated {trimmed} bytes]"));
    }
    s
}

const VALID_TOKENS: &str = "(JSON String, Number, Array, Object or token 'null', 'true' or 'false')";

fn char_desc(c: u32) -> String {
    if c < 0x20 || (0x7f..0xa0).contains(&c) {
        format!("(CTRL-CHAR, code {c})")
    } else if c > 255 {
        format!("'{}' (code {c} / 0x{c:x})", char::from_u32(c).unwrap_or('\u{fffd}'))
    } else {
        format!("'{}' (code {c})", char::from_u32(c).unwrap_or('\u{fffd}'))
    }
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$' || c >= 0x80
}

fn is_ident_part(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

type SResult<T> = Result<T, SyntaxError>;

impl<'a> Scanner<'a> {
    fn new(b: &'a [u8], record: bool) -> Self {
        Scanner { b, p: 0, line: 1, line_start: 0, record, toks: Vec::new(), path: Vec::new(), root_pos: None, stack: Vec::new(), last_tok: Pos { line: 1, col: 1 }, comments: Vec::new() }
    }

    fn pos_at(&self, p: usize) -> Pos {
        // `line`/`line_start` track the scanner cursor; positions are only
        // asked for at or after the current line start.
        if p >= self.line_start {
            return Pos { line: self.line, col: (p - self.line_start) as u64 + 1 };
        }
        line_col(self.b, p)
    }

    fn pos(&self) -> Pos {
        self.pos_at(self.p)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.p).copied()
    }

    fn err(&self, message: impl Into<String>, pos: Pos) -> SyntaxError {
        SyntaxError { message: message.into(), pos, eof: false, last_tok: self.last_tok, foreign: None }
    }

    fn eof_err(&self, message: impl Into<String>, eof_kind: bool) -> SyntaxError {
        SyntaxError { message: message.into(), pos: self.pos(), eof: eof_kind, last_tok: self.last_tok, foreign: None }
    }

    fn unexpected(&self, p: usize, what: &str) -> SyntaxError {
        let (c, _) = decode_char(self.b, p);
        self.err(format!("Unexpected character ({}): {what}", char_desc(c)), self.pos_at(p))
    }

    fn newline(&mut self) {
        // `\r\n` counts once; a lone `\r` is a line break too.
        if self.b[self.p] == b'\r' && self.b.get(self.p + 1) == Some(&b'\n') {
            self.p += 1;
        }
        self.p += 1;
        self.line += 1;
        self.line_start = self.p;
    }

    /// Skip whitespace and comments. `Ok(None)` at end of input.
    fn skip_ws(&mut self) -> SResult<Option<u8>> {
        loop {
            let Some(c) = self.peek() else { return Ok(None) };
            match c {
                b' ' | b'\t' => self.p += 1,
                b'\n' | b'\r' => self.newline(),
                b'/' => {
                    match self.b.get(self.p + 1) {
                        Some(b'*') => {
                            let cstart = self.p;
                            self.p += 2;
                            loop {
                                match self.peek() {
                                    None => return Err(self.eof_err("Unexpected end-of-input in a comment", true)),
                                    Some(b'*') if self.b.get(self.p + 1) == Some(&b'/') => {
                                        self.p += 2;
                                        self.comments.push((cstart, self.p));
                                        break;
                                    }
                                    Some(b'\n') | Some(b'\r') => self.newline(),
                                    Some(_) => self.p += 1,
                                }
                            }
                        }
                        Some(b'/') => {
                            let cstart = self.p;
                            while let Some(c) = self.peek() {
                                if c == b'\n' || c == b'\r' {
                                    break;
                                }
                                self.p += 1;
                            }
                            self.comments.push((cstart, self.p));
                        }
                        _ => return Ok(Some(c)),
                    }
                }
                c if c < 0x20 => {
                    let pos = self.pos_at(self.p + 1);
                    return Err(self.err(format!("Illegal character ({}): only regular white space (\\r, \\n, \\t) is allowed between tokens", char_desc(c as u32)), pos));
                }
                _ => return Ok(Some(c)),
            }
        }
    }

    /// Scan a quoted string starting at the opening quote; returns the byte range (quotes included).
    fn string(&mut self, field_name: bool) -> SResult<(usize, usize)> {
        let start = self.p;
        self.p += 1;
        loop {
            let Some(c) = self.peek() else {
                return Err(self.eof_err(if field_name { "Unexpected end-of-input in field name" } else { "Unexpected end-of-input in VALUE_STRING" }, true));
            };
            match c {
                b'"' => {
                    self.p += 1;
                    return Ok((start, self.p));
                }
                b'\\' => {
                    let Some(e) = self.b.get(self.p + 1).copied() else {
                        self.p += 1;
                        return Err(self.eof_err("Unexpected end-of-input in character escape sequence", true));
                    };
                    match e {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.p += 2,
                        b'u' => {
                            self.p += 2;
                            for _ in 0..4 {
                                match self.peek() {
                                    None => return Err(self.eof_err("Unexpected end-of-input in character escape sequence", true)),
                                    Some(h) if h.is_ascii_hexdigit() => self.p += 1,
                                    Some(_) => return Err(self.unexpected(self.p, "expected a hex-digit for character escape sequence")),
                                }
                            }
                        }
                        other => {
                            let (ch, _) = decode_char(self.b, self.p + 1);
                            let pos = self.pos_at(self.p + 1);
                            let _ = other;
                            return Err(self.err(format!("Unrecognized character escape {}", char_desc(ch)), pos));
                        }
                    }
                }
                c if c < 0x20 => {
                    let pos = self.pos();
                    return Err(self.err(format!("Illegal unquoted character ({}): has to be escaped using backslash to be included in {}", char_desc(c as u32), if field_name { "name" } else { "string value" }), pos));
                }
                c if c >= 0x80 => {
                    let n = utf8_len(c);
                    if n == 0 {
                        let pos = self.pos_at(self.p + 1);
                        return Err(self.err(format!("Invalid UTF-8 start byte 0x{c:x}"), pos));
                    }
                    for k in 1..n {
                        match self.b.get(self.p + k) {
                            None => {
                                self.p = self.b.len();
                                return Err(self.eof_err("Unexpected end-of-input in VALUE_STRING", true));
                            }
                            Some(m) if (m & 0xC0) == 0x80 => {}
                            Some(m) => {
                                let pos = self.pos_at(self.p + k + 1);
                                return Err(self.err(format!("Invalid UTF-8 middle byte 0x{m:x}"), pos));
                            }
                        }
                    }
                    self.p += n;
                }
                _ => self.p += 1,
            }
        }
    }

    fn number(&mut self) -> SResult<&'static str> {
        let mut kind = "VALUE_NUMBER_INT";
        if self.peek() == Some(b'-') {
            self.p += 1;
            match self.peek() {
                Some(c) if c.is_ascii_digit() => {}
                Some(b'I') => {
                    // `-Infinity`
                    let start = self.p - 1;
                    let end = self.ident_end(self.p);
                    let tok = String::from_utf8_lossy(&self.b[start..end]).into_owned();
                    if &self.b[self.p..end] == b"Infinity" {
                        self.p = end;
                        return Err(self.err(format!("Non-standard token '{tok}': enable `JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS` to allow"), self.pos()));
                    }
                    return Err(self.unexpected(self.p, "expected digit (0-9) to follow minus sign, for valid numeric value"));
                }
                Some(_) => return Err(self.unexpected_numeric(self.p, "expected digit (0-9) to follow minus sign, for valid numeric value")),
                None => return Err(self.eof_err("Unexpected end-of-input: expected digit (0-9) to follow minus sign, for valid numeric value", true)),
            }
        }
        if self.peek() == Some(b'0') {
            self.p += 1;
            if let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    let pos = self.pos_at(self.p);
                    return Err(self.err("Invalid numeric value: Leading zeroes not allowed", pos));
                }
            }
        } else {
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.p += 1;
            }
        }
        if self.peek() == Some(b'.') {
            kind = "VALUE_NUMBER_FLOAT";
            self.p += 1;
            match self.peek() {
                Some(c) if c.is_ascii_digit() => {
                    while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        self.p += 1;
                    }
                }
                Some(_) => return Err(self.unexpected_numeric(self.p, "Decimal point not followed by a digit")),
                None => return Err(self.eof_err("Unexpected end-of-input: Decimal point not followed by a digit", true)),
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            kind = "VALUE_NUMBER_FLOAT";
            self.p += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.p += 1;
            }
            match self.peek() {
                Some(c) if c.is_ascii_digit() => {
                    while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        self.p += 1;
                    }
                }
                Some(_) => return Err(self.unexpected_numeric(self.p, "Exponent indicator not followed by a digit")),
                None => return Err(self.eof_err("Unexpected end-of-input: Exponent indicator not followed by a digit", true)),
            }
        }
        Ok(kind)
    }

    fn unexpected_numeric(&self, p: usize, what: &str) -> SyntaxError {
        let (c, _) = decode_char(self.b, p);
        self.err(format!("Unexpected character ({}) in numeric value: {what}", char_desc(c)), self.pos_at(p))
    }

    fn ident_end(&self, from: usize) -> usize {
        let mut e = from;
        while e < self.b.len() && is_ident_part(self.b[e]) && e - from < 256 {
            e += 1;
        }
        e
    }

    /// `true` / `false` / `null`, or Jackson's "Unrecognized token" error.
    fn literal(&mut self) -> SResult<&'static str> {
        let start = self.p;
        let (word, kind): (&[u8], &'static str) = match self.b[start] {
            b't' => (b"true", "VALUE_TRUE"),
            b'f' => (b"false", "VALUE_FALSE"),
            b'n' => (b"null", "VALUE_NULL"),
            _ => (b"", ""),
        };
        let end = self.ident_end(start);
        if !word.is_empty() && &self.b[start..end] == word {
            self.p = end;
            return Ok(kind);
        }
        let tok = String::from_utf8_lossy(&self.b[start..end]).into_owned();
        Err(self.err(format!("Unrecognized token '{tok}': was expecting {VALID_TOKENS}"), self.pos_at(start)))
    }

    /// Error for a byte that cannot start a value.
    fn bad_value_start(&mut self, c: u8) -> SyntaxError {
        let p = self.p;
        match c {
            b',' | b'}' => self.unexpected(p, "expected a value"),
            b']' if self.stack.last().is_some_and(|(k, _)| *k == b'[') => self.unexpected(p, "expected a value"),
            b'+' => {
                let pos = self.pos_at(p + 1);
                self.err("Unexpected character ('+' (code 43)) in numeric value: JSON spec does not allow numbers to have plus signs: enable `JsonReadFeature.ALLOW_LEADING_PLUS_SIGN_FOR_NUMBERS` to allow", pos)
            }
            b'N' | b'I' => {
                let end = self.ident_end(p);
                let tok = String::from_utf8_lossy(&self.b[p..end]).into_owned();
                if tok == "NaN" || tok == "Infinity" {
                    let pos = self.pos_at(end);
                    return self.err(format!("Non-standard token '{tok}': enable `JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS` to allow"), pos);
                }
                self.err(format!("Unrecognized token '{tok}': was expecting {VALID_TOKENS}"), self.pos_at(p))
            }
            c if c >= 0x80 => {
                // Jackson reports the byte after the (consumed) lead byte.
                let n = utf8_len(c);
                let q = if n > 0 && p + 1 < self.b.len() { p + 1 } else { p };
                let pos = self.pos_at(q + 1);
                self.err(format!("Invalid UTF-8 start byte 0x{:x}", self.b[q]), pos)
            }
            c if is_ident_start(c) => {
                let end = self.ident_end(p);
                let tok = String::from_utf8_lossy(&self.b[p..end]).into_owned();
                self.err(format!("Unrecognized token '{tok}': was expecting {VALID_TOKENS}"), self.pos_at(p))
            }
            _ => self.unexpected(p, &format!("expected a valid value {VALID_TOKENS}")),
        }
    }

    fn eof_in_container(&self) -> SyntaxError {
        match self.stack.last() {
            Some((k, start)) => {
                let kind = if *k == b'[' { "Array" } else { "Object" };
                self.eof_err(format!("Unexpected end-of-input: expected close marker for {kind} (start marker at [Source: {}; line: {}, column: {}])", source_desc(self.b), start.line, start.col), true)
            }
            None => self.eof_err("Unexpected end-of-input", true),
        }
    }

    /// Parse one value at the cursor (after whitespace). Returns the Jackson token kind of its first token.
    fn value(&mut self, key: Option<(usize, usize)>, key_pos: Pos) -> SResult<&'static str> {
        let Some(c) = self.skip_ws()? else {
            return Err(self.eof_in_container());
        };
        let start = self.p;
        let val_pos = self.pos();
        self.last_tok = val_pos;
        let kind: &'static str = match c {
            b'{' => {
                self.object()?;
                "START_OBJECT"
            }
            b'[' => {
                self.array()?;
                "START_ARRAY"
            }
            b'"' => {
                self.string(false)?;
                "VALUE_STRING"
            }
            b'-' | b'0'..=b'9' => self.number()?,
            b't' | b'f' | b'n' => self.literal()?,
            other => return Err(self.bad_value_start(other)),
        };
        if self.record {
            let end_pos = self.pos_at(self.p - 1);
            let after_pos = self.pos_at(self.p);
            self.toks.push(Tok { key, key_pos, val_pos, end_pos, after_pos, depth: self.path.len(), path: self.path.clone(), val_kind: kind });
        }
        let _ = start;
        Ok(kind)
    }

    fn object(&mut self) -> SResult<()> {
        let open = self.pos();
        self.stack.push((b'{', open));
        self.p += 1;
        let mut keys: Vec<(usize, usize)> = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.skip_ws()? else {
                if first {
                    return Err(self.eof_in_container());
                }
                return Err(self.eof_err("Unexpected end-of-input within/between Object entries", true));
            };
            if c == b'}' && first {
                self.p += 1;
                break;
            }
            if c != b'"' {
                return Err(self.unexpected(self.p, "was expecting double-quote to start field name"));
            }
            let key_pos = self.pos();
            self.last_tok = key_pos;
            let (ks, ke) = self.string(true)?;
            // Duplicate detection (ES enables STRICT_DUPLICATE_DETECTION).
            let kb = &self.b[ks + 1..ke - 1];
            if keys.iter().any(|(a, b)| &self.b[a + 1..b - 1] == kb) {
                let pos = self.pos_at(ke);
                return Err(self.err(format!("Duplicate field '{}'", String::from_utf8_lossy(kb)), pos));
            }
            keys.push((ks, ke));
            match self.skip_ws()? {
                None => return Err(self.eof_err("Unexpected end-of-input within/between Object entries", true)),
                Some(b':') => self.p += 1,
                Some(_) => return Err(self.unexpected(self.p, "was expecting a colon to separate field name and value")),
            }
            if self.skip_ws()?.is_none() {
                return Err(self.eof_err("Unexpected end-of-input within/between Object entries", true));
            }
            self.path.push(Seg::Key(ks, ke));
            self.value(Some((ks, ke)), key_pos)?;
            self.path.pop();
            first = false;
            match self.skip_ws()? {
                None => return Err(self.eof_in_container()),
                Some(b',') => {
                    self.p += 1;
                    // Jackson reports end of input right after a comma as a plain parse error.
                    if self.skip_ws()?.is_none() {
                        return Err(self.eof_err("Unexpected end-of-input within/between Object entries", false));
                    }
                }
                Some(b'}') => {
                    self.last_tok = self.pos();
                    self.p += 1;
                    break;
                }
                Some(b']') => {
                    let pos = self.pos();
                    return Err(self.err(format!("Unexpected close marker ']': expected '}}' (for Object starting at [Source: {}; line: {}, column: {}])", source_desc(self.b), open.line, open.col), pos));
                }
                Some(_) => return Err(self.unexpected(self.p, "was expecting comma to separate Object entries")),
            }
        }
        self.stack.pop();
        Ok(())
    }

    fn array(&mut self) -> SResult<()> {
        let open = self.pos();
        self.stack.push((b'[', open));
        self.p += 1;
        let mut i = 0usize;
        loop {
            let Some(c) = self.skip_ws()? else {
                return Err(self.eof_in_container());
            };
            if c == b']' && i == 0 {
                self.p += 1;
                break;
            }
            let key_pos = self.pos();
            self.path.push(Seg::Idx(i));
            self.value(None, key_pos)?;
            self.path.pop();
            i += 1;
            match self.skip_ws()? {
                None => return Err(self.eof_in_container()),
                Some(b',') => {
                    self.p += 1;
                    if self.skip_ws()?.is_none() {
                        return Err(self.eof_err("Unexpected end-of-input within/between Array entries", false));
                    }
                }
                Some(b']') => {
                    self.last_tok = self.pos();
                    self.p += 1;
                    break;
                }
                Some(b'}') => {
                    let pos = self.pos();
                    return Err(self.err(format!("Unexpected close marker '}}': expected ']' (for Array starting at [Source: {}; line: {}, column: {}])", source_desc(self.b), open.line, open.col), pos));
                }
                Some(_) => return Err(self.unexpected(self.p, "was expecting comma to separate Array entries")),
            }
        }
        self.stack.pop();
        Ok(())
    }

    /// Scan the whole body: the main value, then what follows it.
    fn run(&mut self) -> SResult<Trailing> {
        if self.skip_ws()?.is_none() {
            return Ok(Trailing::None);
        }
        let root = self.pos();
        self.root_pos = Some(root);
        self.value(None, root)?;
        match self.skip_ws()? {
            None => Ok(Trailing::None),
            Some(c) => {
                let pos = self.pos();
                let kind = match c {
                    b'{' => "START_OBJECT",
                    b'[' => "START_ARRAY",
                    b'"' => "VALUE_STRING",
                    b'-' | b'0'..=b'9' => "VALUE_NUMBER",
                    b't' | b'f' => "VALUE_BOOLEAN",
                    b'n' => "VALUE_NULL",
                    _ => "",
                };
                if kind.is_empty() {
                    return Err(self.bad_value_start(c));
                }
                if matches!(c, b't' | b'f' | b'n') {
                    self.literal()?;
                }
                Ok(Trailing::Token(kind, pos))
            }
        }
    }
}

fn utf8_len(lead: u8) -> usize {
    match lead {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 0,
    }
}

/// Decode the UTF-8 character at `p` (`(code point, byte length)`).
fn decode_char(b: &[u8], p: usize) -> (u32, usize) {
    let c = b[p];
    let n = utf8_len(c);
    if n == 0 {
        return (c as u32, 1);
    }
    match std::str::from_utf8(&b[p..(p + n).min(b.len())]) {
        Ok(s) => s.chars().next().map(|ch| (ch as u32, n)).unwrap_or((c as u32, 1)),
        Err(_) => (c as u32, 1),
    }
}

/// 1-based `(line, col)` of byte offset `at` (columns in bytes).
pub fn line_col(text: &[u8], at: usize) -> Pos {
    let at = at.min(text.len());
    let mut line = 1u64;
    let mut ls = 0usize;
    let mut i = 0usize;
    while i < at {
        match text[i] {
            b'\n' => {
                line += 1;
                ls = i + 1;
            }
            b'\r' => {
                if text.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
                line += 1;
                ls = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    Pos { line, col: (at - ls) as u64 + 1 }
}

/// Validate `text` as a single JSON document the way Jackson (as configured
/// by ES: comments allowed, strict duplicate detection) would.
pub fn validate(text: &[u8]) -> Result<Trailing, SyntaxError> {
    if text.starts_with(FOREIGN_MAGIC) {
        return foreign_validate(text);
    }
    Scanner::new(text, false).run()
}

fn foreign_validate(text: &[u8]) -> Result<Trailing, SyntaxError> {
    let f = ForeignError::from_body(text).unwrap_or(ForeignError { chain: vec![("x_content_parse_exception".into(), "unreadable request body".into())], pos: (1, 1), last_tok: (1, 1), binary: false, trailing: None });
    let pos = Pos { line: f.pos.0, col: f.pos.1 };
    if let Some(kind) = &f.trailing {
        let kind = match kind.as_str() {
            "START_ARRAY" => "START_ARRAY",
            "VALUE_STRING" => "VALUE_STRING",
            "VALUE_NUMBER" => "VALUE_NUMBER",
            "VALUE_NULL" => "VALUE_NULL",
            _ => "START_OBJECT",
        };
        return Ok(Trailing::Token(kind, pos));
    }
    Err(SyntaxError { message: String::new(), pos, eof: false, last_tok: Pos { line: f.last_tok.0, col: f.last_tok.1 }, foreign: Some(Box::new(f)) })
}

/// Fast path of a request body parse: `Some` when `text` is plain JSON
/// (no comments, no trailing tokens, no duplicate keys), which then parses
/// exactly as [`validate`] + `serde_json` would; `None` sends the caller to
/// the full Jackson-faithful path (and its exact error).
pub fn parse_plain(text: &[u8]) -> Option<Value> {
    serde_json::from_slice::<NoDup>(text).ok().map(|v| v.0)
}

/// A JSON value whose objects have no duplicate keys.
struct NoDup(Value);

/// Member buffers of the objects [`NoDup`] is parsing on this thread (one
/// per nesting level in use), recycled across objects and documents.
type Entries = Vec<(String, Value)>;
thread_local! {
    static NODUP_ENTRIES: std::cell::RefCell<Vec<Entries>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// Recycled buffers kept per thread, and the largest one kept: a huge
/// object's buffer is freed rather than retained.
const NODUP_POOL: usize = 16;
const NODUP_MAX_RETAINED: usize = 256;
fn take_entries() -> Entries {
    NODUP_ENTRIES.try_with(|pool| pool.borrow_mut().pop()).ok().flatten().unwrap_or_default()
}
fn recycle_entries(mut entries: Entries) {
    entries.clear();
    if entries.capacity() == 0 || entries.capacity() > NODUP_MAX_RETAINED {
        return;
    }
    let _ = NODUP_ENTRIES.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < NODUP_POOL {
            pool.push(entries);
        }
    });
}

impl<'de> serde::Deserialize<'de> for NoDup {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = NoDup;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_bool<E>(self, b: bool) -> Result<NoDup, E> {
                Ok(NoDup(Value::Bool(b)))
            }
            fn visit_i64<E>(self, n: i64) -> Result<NoDup, E> {
                Ok(NoDup(Value::from(n)))
            }
            fn visit_u64<E>(self, n: u64) -> Result<NoDup, E> {
                Ok(NoDup(Value::from(n)))
            }
            fn visit_f64<E>(self, n: f64) -> Result<NoDup, E> {
                Ok(NoDup(serde_json::Number::from_f64(n).map_or(Value::Null, Value::Number)))
            }
            fn visit_str<E>(self, s: &str) -> Result<NoDup, E> {
                Ok(NoDup(Value::String(s.to_owned())))
            }
            fn visit_string<E>(self, s: String) -> Result<NoDup, E> {
                Ok(NoDup(Value::String(s)))
            }
            fn visit_unit<E>(self) -> Result<NoDup, E> {
                Ok(NoDup(Value::Null))
            }
            fn visit_none<E>(self) -> Result<NoDup, E> {
                Ok(NoDup(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut a: A) -> Result<NoDup, A::Error> {
                let mut out = Vec::with_capacity(a.size_hint().unwrap_or(0));
                while let Some(NoDup(x)) = a.next_element()? {
                    out.push(x);
                }
                Ok(NoDup(Value::Array(out)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut a: A) -> Result<NoDup, A::Error> {
                // Collect the members into a recycled buffer, then build the
                // map at its final size: one index-table allocation, no
                // rehash/regrowth per doubling. Same members, same order; a
                // duplicate is still refused (after the object instead of at
                // the key, which only moves where the fast path gives up:
                // `parse_plain` returns `None` either way).
                let mut entries = take_entries();
                while let Some(k) = a.next_key::<String>()? {
                    let NoDup(x) = a.next_value()?;
                    entries.push((k, x));
                }
                let mut out = Map::with_capacity(entries.len());
                let mut duplicate = false;
                for (k, x) in entries.drain(..) {
                    if out.insert(k, x).is_some() {
                        duplicate = true;
                        break;
                    }
                }
                recycle_entries(entries);
                if duplicate {
                    return Err(serde::de::Error::custom("duplicate key"));
                }
                Ok(NoDup(Value::Object(out)))
            }
        }
        d.deserialize_any(V)
    }
}

/// `text` with its `/* */` and `//` comments blanked out (ES allows them,
/// `serde_json` does not). Returns `None` when there are no comments.
pub fn strip_comments(text: &[u8]) -> Option<Vec<u8>> {
    let mut s = Scanner::new(text, false);
    s.run().ok()?;
    if s.comments.is_empty() {
        return None;
    }
    let mut out = text.to_vec();
    for (a, b) in &s.comments {
        for x in &mut out[*a..*b] {
            if *x != b'\n' && *x != b'\r' {
                *x = b' ';
            }
        }
    }
    Some(out)
}

/// ES's token name for a parsed value (`XContentParser.Token`).
pub fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
    }
}

/// Jackson's token name for a parsed scalar (`JsonToken`).
pub fn jackson_token(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::String(_) => "VALUE_STRING",
        Value::Number(n) => {
            if n.is_f64() {
                "VALUE_NUMBER_FLOAT"
            } else {
                "VALUE_NUMBER_INT"
            }
        }
        Value::Bool(true) => "VALUE_TRUE",
        Value::Bool(false) => "VALUE_FALSE",
        Value::Null => "VALUE_NULL",
    }
}

// ---------------------------------------------------------------------------
// Index + resolver
// ---------------------------------------------------------------------------

impl<'a> Index<'a> {
    pub fn build(text: &'a [u8]) -> Option<Index<'a>> {
        let mut s = Scanner::new(text, true);
        s.run().ok()?;
        Some(Index { text, toks: s.toks, root: s.root_pos })
    }

    fn key_of(&self, seg: &Seg) -> Option<&str> {
        match seg {
            Seg::Key(a, b) => std::str::from_utf8(&self.text[a + 1..b - 1]).ok(),
            Seg::Idx(_) => None,
        }
    }

    fn tok_key(&self, t: &Tok) -> Option<&str> {
        t.key.and_then(|(a, b)| std::str::from_utf8(&self.text[a + 1..b - 1]).ok())
    }

    /// Does the token's path have `within` as one of its last `max_up` key segments?
    fn within(&self, t: &Tok, within: &str, max_up: usize) -> bool {
        let mut n = 0;
        for seg in t.path.iter().rev() {
            if let Seg::Key(..) = seg {
                n += 1;
                if self.key_of(seg) == Some(within) {
                    return true;
                }
                if n >= max_up {
                    return false;
                }
            }
        }
        false
    }

    /// The `nth` field named `key` (first in document order), preferring one under `within`.
    fn find(&self, key: &str, within: Option<&str>, nth: usize) -> Option<&Tok> {
        let unescaped = |t: &Tok| self.tok_key(t).map(|k| unescape(k)).is_some_and(|k| k == key);
        if let Some(w) = within {
            let mut it = self.toks.iter().filter(|t| unescaped(t) && self.within(t, w, 3));
            if let Some(t) = it.nth(nth) {
                return Some(t);
            }
        }
        self.toks.iter().filter(|t| unescaped(t)).nth(nth)
    }

    /// Element `i` of the array that is the value of field `key`.
    fn find_elem(&self, key: &str, within: Option<&str>, i: usize) -> Option<&Tok> {
        let parent = self.find(key, within, 0)?;
        let depth = parent.depth + 1;
        self.toks.iter().find(|t| t.depth == depth && t.path.len() == depth && matches!(t.path.last(), Some(Seg::Idx(j)) if *j == i) && t.path[..depth - 1] == parent.path[..] && parent.key.is_some_and(|(a, b)| matches!(t.path.get(depth - 2), Some(Seg::Key(x, y)) if *x == a && *y == b) || depth == 1))
    }

    /// Position of the first token (the root value).
    pub fn root(&self) -> Option<Pos> {
        self.root
    }
}

/// Minimal JSON string unescape for key comparison (keys are compared against parsed names).
fn unescape(k: &str) -> String {
    if !k.contains('\\') {
        return k.to_string();
    }
    serde_json::from_str::<String>(&format!("\"{k}\"")).unwrap_or_else(|_| k.to_string())
}

fn hint<'e>(e: &'e EsError, k: &str) -> Option<&'e Value> {
    e.extra.iter().find(|(key, _)| key == k).map(|(_, v)| v)
}

fn hint_str<'e>(e: &'e EsError, k: &str) -> Option<&'e str> {
    hint(e, k).and_then(Value::as_str)
}

/// Resolve the position hints of `err` (and its causes) against the body `text`.
/// Errors without hints are returned unchanged (hints are always stripped).
pub fn resolve(err: EsError, text: &[u8]) -> EsError {
    resolve_mapped(err, text, None)
}

/// Where the JSON a non-JSON request body was decoded into came from: each
/// mark maps a byte offset of the (single-line) JSON text to the position
/// of the same token in the original body.
#[derive(Clone, Debug)]
pub struct SourceMap(std::sync::Arc<SourceMapData>);
#[derive(Debug, Default)]
pub struct SourceMapData {
    /// Jackson's description of the original body (`(byte[])"..."`).
    pub source: String,
    /// Binary formats: positions are byte offsets (ES prints line `-1`).
    pub binary: bool,
    /// `(json offset, line, col)` ascending by offset (`line` unused for binary).
    pub marks: Vec<(u64, u64, u64)>,
    _memory: [Option<rustic_core::admission::Reservation>; 3],
}
impl std::ops::Deref for SourceMap {
    type Target = SourceMapData;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl SourceMap {
    /// Producers admit these allocations before construction. Clones share
    /// immutable metadata and retain the same owners.
    pub fn from_admitted(source: String, binary: bool, marks: Vec<(u64, u64, u64)>, memory: [Option<rustic_core::admission::Reservation>; 3]) -> Self {
        Self(std::sync::Arc::new(SourceMapData { source, binary, marks, _memory: memory }))
    }
    /// The original-body position of a position in the transcoded JSON.
    pub fn map_pos(&self, line: u64, col: u64) -> (u64, u64) {
        let p = self.map(Pos { line, col });
        (p.line, p.col)
    }

    fn map(&self, p: Pos) -> Pos {
        let off = p.col.saturating_sub(1);
        let i = self.marks.partition_point(|m| m.0 <= off);
        match i.checked_sub(1).map(|i| self.marks[i]) {
            Some((_, l, c)) => Pos { line: l, col: c },
            None => Pos { line: 1, col: 1 },
        }
    }
}

/// [`resolve`] for a body decoded from YAML / CBOR / Smile: positions are
/// reported in the original body.
pub fn resolve_mapped(err: EsError, text: &[u8], map: Option<&SourceMap>) -> EsError {
    if !has_hints(&err) {
        return err;
    }
    let idx = Index::build(text);
    resolve_rec(err, idx.as_ref(), map).0
}

fn has_hints(e: &EsError) -> bool {
    e.extra.iter().any(|(k, _)| k.starts_with('@')) || e.caused_by.as_deref().is_some_and(has_hints)
}

fn resolve_rec(mut e: EsError, idx: Option<&Index>, map: Option<&SourceMap>) -> (EsError, Option<Pos>) {
    let mut inner_cursor = None;
    if let Some(c) = e.caused_by.take() {
        let (c, cur) = resolve_rec(*c, idx, map);
        inner_cursor = cur;
        e.caused_by = Some(Box::new(c));
    }
    let key = hint_str(&e, "@k").map(str::to_string);
    let within = hint_str(&e, "@in").map(str::to_string);
    let cursor_kind = hint_str(&e, "@c").unwrap_or("value").to_string();
    let prefix = hint_str(&e, "@p").map(str::to_string);
    let set_lc = hint(&e, "@lc").is_some();
    let src = hint(&e, "@src").is_some();
    let nth = hint(&e, "@n").and_then(Value::as_u64).unwrap_or(0) as usize;
    let has = e.extra.iter().any(|(k, _)| k.starts_with('@'));
    e.extra.retain(|(k, _)| !k.starts_with('@'));
    if !has {
        return (e, inner_cursor);
    }
    let Some(idx) = idx else { return (e, inner_cursor) };
    let elem = hint(&e, "@idx").and_then(Value::as_u64).map(|i| i as usize);
    let tok = match (key.as_deref(), elem) {
        (Some(k), Some(i)) => idx.find_elem(k, within.as_deref(), i),
        (Some(k), None) => idx.find(k, within.as_deref(), nth),
        _ => None,
    };
    let key_pos = tok.map(|t| t.key_pos);
    let cursor = match cursor_kind.as_str() {
        "inherit" => inner_cursor.or_else(|| tok.map(|t| t.val_pos)),
        "key" => key_pos,
        "end" => tok.map(|t| t.end_pos),
        "after" => tok.map(|t| t.after_pos),
        "root" => idx.root(),
        "rootend" => idx.toks.iter().find(|t| t.depth == 0).map(|t| t.end_pos),
        _ => tok.map(|t| t.val_pos),
    };
    let binary = map.is_some_and(|m| m.binary);
    let shown = |p: Pos| -> Pos { map.map_or(p, |m| m.map(p)) };
    let lc = |p: Pos| if binary { format!("-1:{}", p.col) } else { format!("{}:{}", p.line, p.col) };
    if let Some(p) = prefix.as_deref() {
        let at = if p == "key" { key_pos } else { cursor };
        if let Some(at) = at.map(shown) {
            e.reason = format!("[{}] {}", lc(at), e.reason);
        }
    }
    if e.reason.contains("@POS@") {
        let at = shown(cursor.unwrap_or(Pos { line: 1, col: 1 }));
        e.reason = e.reason.replace("@POS@", &lc(at));
    }
    if src {
        if let Some(at) = cursor.map(shown) {
            let desc = map.map_or_else(|| source_desc(idx.text), |m| m.source.clone());
            e.reason = if binary {
                format!("{}\n at [Source: {desc}; byte offset: #{}]", e.reason, at.col)
            } else {
                format!("{}\n at [Source: {desc}; line: {}, column: {}]", e.reason, at.line, at.col)
            };
        }
    }
    // Binary formats have no line (`ParsingException` omits `line`/`col`).
    if set_lc && binary {
        e.extra.retain(|(k, _)| k != "line" && k != "col");
    } else if set_lc {
        if let Some(c) = cursor.map(shown) {
            e.extra.retain(|(k, _)| k != "line" && k != "col");
            e.extra.push(("line".into(), Value::from(c.line)));
            e.extra.push(("col".into(), Value::from(c.col)));
        }
    }
    (e, cursor.or(inner_cursor))
}

/// Strip unresolved hints (bodies we cannot index, e.g. NDJSON).
pub fn strip_hints(mut e: EsError) -> EsError {
    e.extra.retain(|(k, _)| !k.starts_with('@'));
    if let Some(c) = e.caused_by.take() {
        e.caused_by = Some(Box::new(strip_hints(*c)));
    }
    e
}

// ---------------------------------------------------------------------------
// Hint constructors
// ---------------------------------------------------------------------------

impl EsError {
    /// Locate this error at field `key` (optionally the one under `within`).
    pub fn at_key(mut self, key: &str, within: Option<&str>) -> Self {
        self.extra.push(("@k".into(), Value::String(key.to_string())));
        if let Some(w) = within {
            self.extra.push(("@in".into(), Value::String(w.to_string())));
        }
        self
    }

    /// Cursor kind: `key`, `value` (default), `end`, `after`, `root`, `inherit`.
    pub fn cursor(mut self, kind: &str) -> Self {
        self.extra.push(("@c".into(), Value::String(kind.to_string())));
        self
    }

    /// Prepend `[l:c] ` of the key start (`"key"`) or the cursor (`"cursor"`) to the reason.
    pub fn prefixed(mut self, by: &str) -> Self {
        self.extra.push(("@p".into(), Value::String(by.to_string())));
        self
    }

    /// Set `line` / `col` from the cursor (ES `ParsingException`).
    pub fn with_lc(mut self) -> Self {
        self.extra.push(("@lc".into(), Value::Bool(true)));
        self
    }

    /// Append Jackson's ` at [Source: ...; line: l, column: c]` to the reason (cursor position).
    pub fn with_source(mut self) -> Self {
        self.extra.push(("@src".into(), Value::Bool(true)));
        self
    }

    /// Jackson accessor type mismatch (`parser.intValue()` on `true`, ...):
    /// `x_content_parse_exception` over `json_parse_exception`, both with
    /// the source suffix, positioned just past the token.
    pub fn jackson_type(key: &str, within: Option<&str>, v: &Value, what: &str) -> Self {
        let msg = format!("Current token ({}) {what}", jackson_token(v));
        EsError::x_content_parse(msg.clone()).at_key(key, within).cursor("after").prefixed("cursor").with_source().caused_by(EsError::new(400, "json_parse_exception", msg).at_key(key, within).cursor("after").with_source())
    }

    /// The `nth` occurrence of the key (0-based).
    pub fn nth_key(mut self, n: usize) -> Self {
        self.extra.push(("@n".into(), Value::from(n as u64)));
        self
    }

    /// `ParsingException` at the value of `key` (under `within`): line/col set.
    pub fn parsing_at_key(reason: impl Into<String>, key: &str, within: Option<&str>) -> Self {
        EsError::parsing(reason).at_key(key, within).with_lc()
    }

    /// ES `ObjectParser` unknown field, as the query parsers that wrap it
    /// report it: `parsing_exception "[l:c] [parser] unknown field [f]"`
    /// (line/col = the value) caused by the `x_content_parse_exception`.
    pub fn unknown_field_wrapped(parser: &str, field: &str, within: Option<&str>, candidates: &[&str]) -> Self {
        let msg = format!("[{parser}] unknown field [{field}]{}", suggest(field, candidates.iter().copied()));
        EsError::parsing(msg.clone()).at_key(field, within).prefixed("key").with_lc().caused_by(EsError::x_content_parse(msg).at_key(field, within).prefixed("key"))
    }

    /// ES `ObjectParser` unknown field reported bare: `x_content_parse_exception "[l:c] [parser] unknown field [f]"`.
    pub fn unknown_field_xc(parser: &str, field: &str, within: Option<&str>, candidates: &[&str]) -> Self {
        let msg = format!("[{parser}] unknown field [{field}]{}", suggest(field, candidates.iter().copied()));
        EsError::x_content_parse(msg).at_key(field, within).prefixed("key")
    }

    /// ES `ObjectParser` type mismatch: `[parser] field doesn't support values of type: START_OBJECT`.
    pub fn unsupported_type_xc(parser: &str, field: &str, v: &Value, within: Option<&str>) -> Self {
        EsError::x_content_parse(format!("[{parser}] {field} doesn't support values of type: {}", token_name(v))).at_key(field, within).prefixed("cursor")
    }

    /// ES `ObjectParser` wrapper around a failing field parser:
    /// `x_content_parse_exception "[l:c] [parser] failed to parse field [f]"`
    /// positioned where the cause failed.
    pub fn field_failure(parser: &str, field: &str, within: Option<&str>, cause: EsError) -> Self {
        EsError::x_content_parse(format!("[{parser}] failed to parse field [{field}]")).at_key(field, within).cursor("inherit").prefixed("cursor").caused_by(cause)
    }

    /// `SearchSourceBuilder`-style unknown key: `Unknown key for a VALUE_NUMBER in [f].`
    pub fn unknown_key(field: &str, v: &Value, within: Option<&str>) -> Self {
        EsError::parsing(format!("Unknown key for a {} in [{field}].", token_name(v))).at_key(field, within).with_lc()
    }

    /// `[name] query does not support [f]` at the value of `f`.
    pub fn query_unsupported(name: &str, field: &str) -> Self {
        EsError::parsing_at_key(format!("[{name}] query does not support [{field}]"), field, Some(name))
    }

    /// Unknown named object (query / aggregation / retriever / ...):
    /// `parsing_exception "<what> [name]<suggestion>"` at the value of `name`
    /// caused by `named_object_not_found_exception "[l:c] unknown field [name]"`.
    pub fn unknown_named(what: &str, name: &str, within: Option<&str>, candidates: &[&str]) -> Self {
        EsError::parsing(format!("{what} [{name}]{}", suggest(name, candidates.iter().copied())))
            .at_key(name, within)
            .with_lc()
            .caused_by(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{name}]")).at_key(name, within).prefixed("cursor"))
    }

    /// `number_format_exception` for a non-numeric string (`Integer.parseInt`).
    pub fn number_format(s: &str) -> Self {
        EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))
    }
}

// ---------------------------------------------------------------------------
// Suggestions (ES `SuggestingErrorOnUnknown`)
// ---------------------------------------------------------------------------

/// Lucene `LevenshteinDistance.getDistance`: `1 - edits / max(len)`.
pub fn similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return if n == m { 1.0 } else { 0.0 };
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur = vec![0usize; n + 1];
    for j in 1..=m {
        cur[0] = j;
        for i in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[i] = (cur[i - 1] + 1).min(prev[i] + 1).min(prev[i - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    1.0 - prev[n] as f32 / n.max(m) as f32
}

/// ` did you mean [x]?` / ` did you mean any of [x, y]?` for candidates scoring > 0.5
/// (best first, ties by name), or an empty string.
pub fn suggest<'c>(unknown: &str, candidates: impl Iterator<Item = &'c str>) -> String {
    let mut scored: Vec<(f32, &str)> = candidates.filter_map(|c| {
        let d = similarity(unknown, c);
        (d > 0.5).then_some((d, c))
    }).collect();
    if scored.is_empty() {
        return String::new();
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(b.1)));
    let names: Vec<&str> = scored.iter().map(|s| s.1).collect();
    if names.len() == 1 {
        format!(" did you mean [{}]?", names[0])
    } else {
        format!(" did you mean any of [{}]?", names.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Java `HashMap` re-serialisation (ES `parser.map()` + `builder.map()`)
// ---------------------------------------------------------------------------

fn java_string_hash(s: &str) -> i32 {
    let mut h: i32 = 0;
    for u in s.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(u as i32);
    }
    h
}

fn table_size(n: usize) -> usize {
    let mut cap = 16usize;
    while n > cap * 3 / 4 {
        cap *= 2;
    }
    cap
}

/// The keys of `m` in Java `HashMap` iteration order (as `new HashMap<>()`
/// filled in insertion order would iterate them).
pub fn java_hashmap_order<'m>(m: &'m Map<String, Value>) -> Vec<(&'m String, &'m Value)> {
    let cap = table_size(m.len());
    let mut buckets: Vec<Vec<(&String, &Value)>> = vec![Vec::new(); cap];
    for (k, v) in m {
        let h = java_string_hash(k);
        let spread = h ^ ((h as u32) >> 16) as i32;
        let b = (spread as u32 as usize) & (cap - 1);
        buckets[b].push((k, v));
    }
    buckets.into_iter().flatten().collect()
}

/// Re-serialise `v` as ES does when it round-trips a body through
/// `parser.map()` and `XContentBuilder.map()`: compact JSON with every
/// object's keys in Java `HashMap` order.
pub fn java_hashmap_json(v: &Value) -> String {
    let mut out = String::new();
    write_java(v, &mut out);
    out
}

fn write_java(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            out.push('{');
            for (i, (k, v)) in java_hashmap_order(m).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                write_java(v, out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_java(v, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(body: &str) -> (String, Pos, bool) {
        let e = validate(body.as_bytes()).unwrap_err();
        (e.message, e.pos, e.eof)
    }

    #[test]
    fn jackson_messages() {
        assert_eq!(msg(r#"{"query":{"match_all":{}},}"#), ("Unexpected character ('}' (code 125)): was expecting double-quote to start field name".into(), Pos { line: 1, col: 27 }, false));
        assert_eq!(msg(r#"{"sort":["n",]}"#).1, Pos { line: 1, col: 14 });
        assert_eq!(msg(r#"{"size":}"#).0, "Unexpected character ('}' (code 125)): expected a value");
        assert_eq!(msg(r#"{"size" 1}"#), ("Unexpected character ('1' (code 49)): was expecting a colon to separate field name and value".into(), Pos { line: 1, col: 9 }, false));
        assert_eq!(msg(r#"{"explain":tru}"#), ("Unrecognized token 'tru': was expecting (JSON String, Number, Array, Object or token 'null', 'true' or 'false')".into(), Pos { line: 1, col: 12 }, false));
        assert_eq!(msg(r#"{"size":01}"#), ("Invalid numeric value: Leading zeroes not allowed".into(), Pos { line: 1, col: 10 }, false));
        assert_eq!(msg(r#"{"size":+1}"#).1, Pos { line: 1, col: 10 });
        assert_eq!(msg(r#"{"size":.5}"#), ("Unexpected character ('.' (code 46)): expected a valid value (JSON String, Number, Array, Object or token 'null', 'true' or 'false')".into(), Pos { line: 1, col: 9 }, false));
        assert_eq!(msg(r#"{"size":1.}"#), ("Unexpected character ('}' (code 125)) in numeric value: Decimal point not followed by a digit".into(), Pos { line: 1, col: 11 }, false));
        assert_eq!(msg(r#"{"size":NaN}"#), ("Non-standard token 'NaN': enable `JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS` to allow".into(), Pos { line: 1, col: 12 }, false));
        assert_eq!(msg(r#"{"query":{"match":{"f":"abc}}}"#), ("Unexpected end-of-input in VALUE_STRING".into(), Pos { line: 1, col: 31 }, true));
        assert_eq!(msg(r#"{"query":{"match":{"f":"a\qb"}}}"#), ("Unrecognized character escape 'q' (code 113)".into(), Pos { line: 1, col: 27 }, false));
        assert_eq!(msg("{\"query\":{\"match\":{\"f\":\"a\tb\"}}}").1, Pos { line: 1, col: 26 });
        assert_eq!(msg(r#"{"size":1,"size":2}"#), ("Duplicate field 'size'".into(), Pos { line: 1, col: 17 }, false));
        assert_eq!(msg("abc").1, Pos { line: 1, col: 1 });
        assert_eq!(msg(r#"{"size":1} x"#).1, Pos { line: 1, col: 12 });
        assert_eq!(msg("{\n  \"query\": {\n    \"match_all\": {}\n  },\n}").1, Pos { line: 5, col: 1 });
        assert_eq!(msg(r#"{"sort":["n"}"#).0, "Unexpected close marker '}': expected ']' (for Array starting at [Source: (byte[])\"{\"sort\":[\"n\"}\"; line: 1, column: 9])");
        assert_eq!(msg(r#"{"sort":["n":1]}"#).0, "Unexpected character (':' (code 58)): was expecting comma to separate Array entries");
        assert_eq!(msg(r#"{"query":{"match_all":{}}"#), ("Unexpected end-of-input: expected close marker for Object (start marker at [Source: (byte[])\"{\"query\":{\"match_all\":{}}\"; line: 1, column: 1])".into(), Pos { line: 1, col: 26 }, true));
        assert_eq!(msg(r#"{"size""#), ("Unexpected end-of-input within/between Object entries".into(), Pos { line: 1, col: 8 }, true));
        assert_eq!(msg(r#"{"size":1,"#), ("Unexpected end-of-input within/between Object entries".into(), Pos { line: 1, col: 11 }, false));
        assert_eq!(msg(r#"{"si"#), ("Unexpected end-of-input in field name".into(), Pos { line: 1, col: 5 }, true));
        assert_eq!(msg(r#"{"size":-}"#).0, "Unexpected character ('}' (code 125)) in numeric value: expected digit (0-9) to follow minus sign, for valid numeric value");
        assert_eq!(msg(r#"{"size":1e}"#).0, "Unexpected character ('}' (code 125)) in numeric value: Exponent indicator not followed by a digit");
        assert_eq!(msg(r#"{"size":1x}"#), ("Unexpected character ('x' (code 120)): was expecting comma to separate Object entries".into(), Pos { line: 1, col: 10 }, false));
        assert_eq!(msg("{\"size\":\u{1}1}"), ("Illegal character ((CTRL-CHAR, code 1)): only regular white space (\\r, \\n, \\t) is allowed between tokens".into(), Pos { line: 1, col: 10 }, false));
        assert_eq!(msg(r#"{"size":1,,"from":2}"#).1, Pos { line: 1, col: 11 });
        assert_eq!(msg(r#"{"query":{"match":{"f":"a\u12"}}}"#), ("Unexpected character ('\"' (code 34)): expected a hex-digit for character escape sequence".into(), Pos { line: 1, col: 30 }, false));
        assert_eq!(msg("{\"size\":é}"), ("Invalid UTF-8 start byte 0xa9".into(), Pos { line: 1, col: 11 }, false));
        assert_eq!(msg("{\"query\":{\"match\":{\"f\":\"ééé\"}},\"size\":}").1, Pos { line: 1, col: 42 });
        assert!(validate(r#"{/* c */"size":1}"#.as_bytes()).is_ok());
        assert_eq!(validate(r#"{"size":1}{"size":2}"#.as_bytes()).unwrap(), Trailing::Token("START_OBJECT", Pos { line: 1, col: 11 }));
    }

    #[test]
    fn positions_resolve() {
        let body = br#"{"query":{"bool":{"must":[{"match_all":{"zz":1}}]}}}"#;
        let e = EsError::field_failure("bool", "must", None, EsError::unknown_field_wrapped("match_all", "zz", Some("match_all"), &["boost", "_name"]));
        let r = resolve(e, body);
        assert_eq!(r.reason, "[1:46] [bool] failed to parse field [must]");
        let c = r.caused_by.as_ref().unwrap();
        assert_eq!(c.reason, "[1:41] [match_all] unknown field [zz]");
        assert_eq!(c.extra.iter().find(|(k, _)| k == "col").unwrap().1, Value::from(46u64));
        assert!(c.extra.iter().all(|(k, _)| !k.starts_with('@')));
    }

    #[test]
    fn suggestions() {
        assert_eq!(suggest("match_al", ["match_all", "match", "match_none", "match_phrase", "bool"].into_iter()), " did you mean any of [match_all, match, match_none, match_phrase]?");
        assert_eq!(suggest("term", ["terms", "range"].into_iter()), " did you mean [terms]?");
        assert_eq!(suggest("zzz", ["boost", "_name"].into_iter()), "");
    }

    #[test]
    fn hashmap_order() {
        let v: Value = serde_json::from_str(r#"{"query":{"match_all":{}},"zz":1}"#).unwrap();
        assert_eq!(java_hashmap_json(&v), r#"{"zz":1,"query":{"match_all":{}}}"#);
        let v: Value = serde_json::from_str(r#"{"query":{"match_all":{}},"zzzzzzzz":1}"#).unwrap();
        assert_eq!(java_hashmap_json(&v), r#"{"query":{"match_all":{}},"zzzzzzzz":1}"#);
    }
}

// ---------------------------------------------------------------------------
// One-pass strict parse (document sources on the write path)
// ---------------------------------------------------------------------------

/// Parse `text` into a [`Value`] in one pass, rejecting duplicate keys
/// (ES parses sources with Jackson's `STRICT_DUPLICATE_DETECTION`). `None`
/// on any failure: the caller then re-diagnoses the text with [`validate`]
/// for ES's exact error. Whatever this accepts, [`validate`] accepts too
/// (same JSON grammar and whitespace; comments, trailing tokens, lone
/// surrogates and out-of-range numbers fail here and take the slow path),
/// so a document that parses here never needs the scanner.
pub fn parse_strict(text: &str) -> Option<Value> {
    use serde::de::DeserializeSeed;
    let mut de = serde_json::Deserializer::from_str(text);
    let v = StrictValue.deserialize(&mut de).ok()?;
    de.end().ok()?;
    Some(v)
}

struct StrictValue;

impl<'de> serde::de::DeserializeSeed<'de> for StrictValue {
    type Value = Value;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<Value, D::Error> {
        d.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> serde::de::Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> std::result::Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> std::result::Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_f64<E>(self, v: f64) -> std::result::Result<Value, E> {
        Ok(serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number))
    }

    fn visit_str<E>(self, v: &str) -> std::result::Result<Value, E> {
        Ok(Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> std::result::Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_none<E>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(v) = seq.next_element_seed(StrictValue)? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        let mut out = Map::new();
        while let Some(k) = map.next_key::<String>()? {
            let v = map.next_value_seed(StrictValue)?;
            if out.insert(k, v).is_some() {
                return Err(serde::de::Error::custom("duplicate field"));
            }
        }
        Ok(Value::Object(out))
    }
}

#[cfg(test)]
mod strict_tests {
    use super::*;

    #[test]
    fn strict_parse_matches_serde_and_defers_to_the_scanner() {
        for ok in [r#"{"a":1,"b":[1,2.5,{"c":null}],"d":"xé","e":true,"big":18446744073709551615,"neg":-3}"#, "[]", "\"s\"", " {\n\t\"a\" : 1 } "] {
            assert_eq!(parse_strict(ok), Some(serde_json::from_str::<Value>(ok).unwrap()), "{ok}");
            assert!(validate(ok.as_bytes()).is_ok());
        }
        // Duplicates, comments, trailing tokens: the scanner decides.
        for slow in [r#"{"a":1,"a":2}"#, r#"{"o":{"x":1,"x":1}}"#, "{\"a\":1} // c", "{} {}", r#"{"a":"\ud800"}"#, "{\"a\":01}", "{\"a\":1e999}"] {
            assert_eq!(parse_strict(slow), None, "{slow}");
        }
        assert!(validate(br#"{"a":1,"a":2}"#).is_err());
        // Key order is kept.
        assert_eq!(parse_strict(r#"{"z":1,"a":2}"#).unwrap().as_object().unwrap().keys().collect::<Vec<_>>(), ["z", "a"]);
    }
}

/// Decode a stored/source JSON value using Lucene's replacement semantics for
/// unpaired UTF-16 surrogates. Keep the original source bytes for storage and
/// wire output: Rust strings cannot represent these Java char values.
/// Ordinary JSON takes the existing serde fast path without allocation.
pub fn parse_source_value(text: &str) -> serde_json::Result<Value> {
    match serde_json::from_str(text) {
        Ok(value) => Ok(value),
        Err(original) => {
            let mut bytes = text.as_bytes().to_vec();
            let mut i = 0;
            let mut changed = false;
            fn unit(bytes: &[u8], i: usize) -> Option<u16> {
                if bytes.get(i..i+2)? != b"\\u" { return None; }
                let s = std::str::from_utf8(bytes.get(i+2..i+6)?).ok()?;
                u16::from_str_radix(s, 16).ok()
            }
            while i < bytes.len() {
                if bytes[i] != b'\\' { i += 1; continue; }
                match unit(&bytes, i) {
                    Some(high @ 0xD800..=0xDFFF) => {
                        if high <= 0xDBFF && matches!(unit(&bytes, i+6), Some(0xDC00..=0xDFFF)) {
                            i += 12;
                        } else {
                            // Do not collapse distinct invalid UTF-16 field names
                            // onto the same Rust key. That requires a lossless key
                            // representation, not replacement decoding.
                            let mut end = i + 6;
                            while end < bytes.len() {
                                if bytes[end] == b'\\' { end += 2; }
                                else if bytes[end] == b'"' { break; }
                                else { end += 1; }
                            }
                            end += 1;
                            while bytes.get(end).is_some_and(u8::is_ascii_whitespace) { end += 1; }
                            if bytes.get(end) == Some(&b':') { return Err(original); }
                            bytes[i+2..i+6].copy_from_slice(b"fffd");
                            changed = true;
                            i += 6;
                        }
                    }
                    Some(_) => i += 6,
                    None => i += 2, // escaped backslash must not introduce a Unicode escape
                }
            }
            if changed { serde_json::from_slice(&bytes) } else { Err(original) }
        }
    }
}

#[cfg(test)]
mod source_surrogate_tests {
    use super::*;
    #[test]
    fn source_utf16_replacement_preserves_pairs_and_literal_escapes() {
        assert_eq!(parse_source_value(r#"{"x":"\ud83d \ude02 \ud83d\ude02 \\ud83d"}"#).unwrap()["x"], "\u{fffd} \u{fffd} 😂 \\ud83d");
        assert!(parse_source_value(r#"{"x":"\ud83d", broken}"#).is_err());
        assert!(parse_source_value(r#"{"x":"\uqqqq"}"#).is_err());
        assert!(parse_source_value(r#"{"\ud800":1,"\ud801":2}"#).is_err());
    }
}

#[cfg(test)]
mod nodup_presize_tests {
    use super::*;

    fn same(text: &str) {
        let fast = parse_plain(text.as_bytes()).expect("plain JSON");
        let reference: Value = serde_json::from_str(text).unwrap();
        // Serialized text compares member order too (map equality does not).
        assert_eq!(serde_json::to_string(&fast).unwrap(), serde_json::to_string(&reference).unwrap(), "{text}");
    }

    /// Presized maps built from recycled member buffers keep every member,
    /// in source order, at every nesting level.
    #[test]
    fn presized_maps_keep_members_and_order() {
        same(r#"{"date":"2016-01-01T00:00:00","TAVG":22.9,"station":{"elevation":34.0,"name":"SHARJAH INTER. AIRP","country":"United","gsn_flag":"GSN","location":{"lon":"55.517","lat":"25.333"},"country_code":"AE","wmo_id":"41196","id":"AE000041196"},"TMIN":15.5}"#);
        same(r#"{"z":1,"a":[{"y":2,"b":{}},[],{"x":null}],"m":{"k":{"j":{"i":"deep"}}},"e":{}}"#);
        let wide = format!("{{{}}}", (0..2000).rev().map(|i| format!("\"k{i}\":{{\"v\":{i}}}")).collect::<Vec<_>>().join(","));
        same(&wide);
        for _ in 0..3 {
            same(r#"{"a":{"b":1,"c":2},"d":{"e":3}}"#);
        }
    }

    /// Duplicate keys (at any level) and syntax errors still leave the
    /// fast path, and the next document parses normally.
    #[test]
    fn duplicates_still_leave_the_fast_path() {
        assert!(parse_plain(br#"{"a":1,"a":2}"#).is_none());
        assert!(parse_plain(br#"{"o":{"x":1,"y":2,"x":3},"after":true}"#).is_none());
        assert!(parse_plain(br#"[{"a":1},{"b":1,"b":1}]"#).is_none());
        assert!(parse_plain(br#"{"a":1,"b":{"c":"#).is_none());
        same(r#"{"a":1,"b":{"a":1}}"#);
    }
}
