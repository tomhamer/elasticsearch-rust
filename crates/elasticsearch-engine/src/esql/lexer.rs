//! Tokenizer shared by ES|QL and SQL (dialect differences: SQL strings are
//! single-quoted and double quotes delimit identifiers).

use super::parsing_err;
use crate::error::Result;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    /// Quoted identifier (backticks in ES|QL, double quotes in SQL).
    QIdent(String),
    Str(String),
    Int(i64),
    Dec(f64),
    /// `?` (empty), `?name`, `?1`.
    Param(String),
    Op(&'static str),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    Esql,
    Sql,
}

const OPS: &[&str] = &["==", "!=", "<>", "<=", ">=", "::", "=~", "<", ">", "+", "-", "*", "/", "%", "=", ":", ".", ";"];

/// Tokenize `src[from..to]`; offsets stay global.
pub fn lex(full: &str, from: usize, to: usize, d: Dialect) -> Result<Vec<Token>> {
    let b = full.as_bytes();
    let mut i = from;
    let mut out = Vec::new();
    while i < to {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Comments.
        if c == b'/' && i + 1 < to && b[i + 1] == b'/' || d == Dialect::Sql && c == b'-' && i + 1 < to && b[i + 1] == b'-' {
            while i < to && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < to && b[i + 1] == b'*' {
            let end = full[i + 2..to].find("*/").map(|p| i + 2 + p + 2).unwrap_or(to);
            i = end;
            continue;
        }
        let start = i;
        let push = |out: &mut Vec<Token>, tok: Tok, end: usize| out.push(Token { tok, start, end });
        // Strings.
        let str_quote = if d == Dialect::Esql { b'"' } else { b'\'' };
        if c == str_quote {
            if d == Dialect::Esql && full[i..to].starts_with("\"\"\"") {
                let end = full[i + 3..to].find("\"\"\"").ok_or_else(|| parsing_err(full, i, "unterminated string"))?;
                let s = full[i + 3..i + 3 + end].to_string();
                i = i + 3 + end + 3;
                push(&mut out, Tok::Str(s), i);
                continue;
            }
            let mut s = String::new();
            let mut j = i + 1;
            let mut closed = false;
            while j < to {
                let ch = full[j..].chars().next().unwrap();
                if ch as u32 == str_quote as u32 {
                    if d == Dialect::Sql && j + 1 < to && b[j + 1] == b'\'' {
                        s.push('\'');
                        j += 2;
                        continue;
                    }
                    closed = true;
                    j += 1;
                    break;
                }
                if ch == '\\' && d == Dialect::Esql && j + 1 < to {
                    let n = full[j + 1..].chars().next().unwrap();
                    s.push(match n {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other,
                    });
                    j += 1 + n.len_utf8();
                    continue;
                }
                s.push(ch);
                j += ch.len_utf8();
            }
            if !closed {
                return Err(parsing_err(full, i, "token recognition error: unterminated string"));
            }
            i = j;
            push(&mut out, Tok::Str(s), i);
            continue;
        }
        let qid = if d == Dialect::Esql { b'`' } else { b'"' };
        if c == qid || (d == Dialect::Sql && c == b'`') {
            let q = c;
            let mut s = String::new();
            let mut j = i + 1;
            let mut closed = false;
            while j < to {
                if b[j] == q {
                    if j + 1 < to && b[j + 1] == q {
                        s.push(q as char);
                        j += 2;
                        continue;
                    }
                    closed = true;
                    j += 1;
                    break;
                }
                let ch = full[j..].chars().next().unwrap();
                s.push(ch);
                j += ch.len_utf8();
            }
            if !closed {
                return Err(parsing_err(full, i, "token recognition error: unterminated quoted identifier"));
            }
            i = j;
            push(&mut out, Tok::QIdent(s), i);
            continue;
        }
        if c.is_ascii_digit() || (c == b'.' && i + 1 < to && b[i + 1].is_ascii_digit()) {
            let mut j = i;
            let mut dec = false;
            while j < to && (b[j].is_ascii_digit() || b[j] == b'.' && !dec && j + 1 < to && b[j + 1].is_ascii_digit()) {
                if b[j] == b'.' {
                    dec = true;
                }
                j += 1;
            }
            if j < to && (b[j] == b'e' || b[j] == b'E') && j + 1 < to && (b[j + 1].is_ascii_digit() || (b[j + 1] == b'-' || b[j + 1] == b'+') && j + 2 < to && b[j + 2].is_ascii_digit()) {
                dec = true;
                j += 2;
                while j < to && b[j].is_ascii_digit() {
                    j += 1;
                }
            }
            let t = &full[i..j];
            let tok = if dec {
                Tok::Dec(t.parse().map_err(|_| parsing_err(full, i, format!("invalid number [{t}]")))?)
            } else {
                match t.parse::<i64>() {
                    Ok(v) => Tok::Int(v),
                    Err(_) => Tok::Dec(t.parse().map_err(|_| parsing_err(full, i, format!("invalid number [{t}]")))?),
                }
            };
            i = j;
            push(&mut out, tok, i);
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' || c == b'@' {
            let mut j = i;
            while j < to && (b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'@' || (b[j] == b'.' && j + 1 < to && (b[j + 1].is_ascii_alphabetic() || b[j + 1] == b'_' || b[j + 1] == b'@' || b[j + 1] == b'`'))) {
                j += 1;
            }
            let t = full[i..j].to_string();
            i = j;
            push(&mut out, Tok::Ident(t), i);
            continue;
        }
        if c == b'?' {
            let mut j = i + 1;
            while j < to && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            let t = full[i + 1..j].to_string();
            i = j;
            push(&mut out, Tok::Param(t), i);
            continue;
        }
        let single = match c {
            b'(' => Some(Tok::LParen),
            b')' => Some(Tok::RParen),
            b'[' => Some(Tok::LBracket),
            b']' => Some(Tok::RBracket),
            b'{' => Some(Tok::LBrace),
            b'}' => Some(Tok::RBrace),
            b',' => Some(Tok::Comma),
            _ => None,
        };
        if let Some(t) = single {
            i += 1;
            push(&mut out, t, i);
            continue;
        }
        let rest = &full[i..to];
        match OPS.iter().find(|o| rest.starts_with(**o)) {
            Some(o) => {
                i += o.len();
                push(&mut out, Tok::Op(o), i);
            }
            None => {
                let ch = rest.chars().next().unwrap();
                return Err(parsing_err(full, i, format!("token recognition error at: '{ch}'")));
            }
        }
    }
    out.push(Token { tok: Tok::Eof, start: to, end: to });
    Ok(out)
}

/// Split `src[from..to]` on `sep` at nesting depth 0 (outside strings,
/// parentheses, brackets); returns global `(start, end)` ranges.
pub fn split_top(full: &str, from: usize, to: usize, sep: u8, d: Dialect) -> Vec<(usize, usize)> {
    let b = full.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut i = from;
    let mut start = from;
    let sq = if d == Dialect::Esql { b'"' } else { b'\'' };
    let qi = if d == Dialect::Esql { b'`' } else { b'"' };
    while i < to {
        let c = b[i];
        if c == sq || c == qi {
            if d == Dialect::Esql && c == b'"' && full[i..to].starts_with("\"\"\"") {
                i = full[i + 3..to].find("\"\"\"").map(|p| i + 3 + p + 3).unwrap_or(to);
                continue;
            }
            let mut j = i + 1;
            while j < to && b[j] != c {
                if b[j] == b'\\' && d == Dialect::Esql && c == b'"' {
                    j += 1;
                }
                j += 1;
            }
            i = j + 1;
            continue;
        }
        if c == b'/' && i + 1 < to && b[i + 1] == b'/' {
            while i < to && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if c == sep && depth == 0 {
            out.push((start, i));
            start = i + 1;
        }
        i += 1;
    }
    out.push((start, to.max(start)));
    out
}
