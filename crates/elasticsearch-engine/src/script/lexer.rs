//! Painless tokenizer.

use super::error::ScriptErr;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    /// Integer literal: value and whether it carried an `L` suffix.
    Int(i64, bool),
    /// Floating literal (with an optional `f`/`d` suffix).
    Float(f64),
    Str(String),
    /// Regex literal: pattern and flag letters.
    Regex(String, String),
    /// Operator or punctuation.
    P(&'static str),
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub pos: usize,
}

const PUNCT: &[&str] = &[
    ">>>=", "<<=", ">>=", ">>>", "===", "!==", "==~", "?.", "?:", "->", "::", "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
    "^=", "<<", ">>", "=~", "(", ")", "[", "]", "{", "}", ",", ";", ".", "?", ":", "=", "<", ">", "!", "~", "+", "-", "*", "/", "%", "&", "|", "^",
];

pub fn lex(src: &str) -> Result<Vec<Token>, ScriptErr> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let start = i;
            i += 2;
            loop {
                if i + 1 >= b.len() {
                    return Err(ScriptErr::compile(start, "unterminated comment"));
                }
                if b[i] == b'*' && b[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        let start = i;
        if c == b'/' {
            let div = match out.last().map(|t: &Token| &t.tok) {
                Some(Tok::Ident(word)) => !matches!(word.as_str(),"return"|"throw"|"case"),
                Some(Tok::Str(_)|Tok::Regex(..)|Tok::Int(..)|Tok::Float(_)|Tok::P(")"|"]"|"}"|"++"|"--"))=>true,
                _=>false,
            };
            if !div {
                i += 1;
                let mut pat = String::new();
                loop {
                    if i >= b.len() || b[i] == b'\n' {
                        return Err(ScriptErr::compile(start, "unexpected character [/]. The only valid escape sequences in regexes are... unterminated regex"));
                    }
                    if b[i] == b'\\' && i + 1 < b.len() {
                        let l = utf8_len(b[i + 1]);
                        pat.push_str(&src[i..i + 1 + l]);
                        i += 1 + l;
                        continue;
                    }
                    if b[i] == b'/' {
                        i += 1;
                        break;
                    }
                    let l = utf8_len(b[i]);
                    pat.push_str(&src[i..i + l]);
                    i += l;
                }
                let fs = i;
                while i < b.len() && b"cilmsUux".contains(&b[i]) {
                    i += 1;
                }
                out.push(Token { tok: Tok::Regex(pat, src[fs..i].to_string()), pos: start });
                continue;
            }
        }
        // Painless `AFTER_DOT` mode: `list.0` / `map.0` (DOTINTEGER) after a
        // member access dot is a name, not a number.
        let after_dot = matches!(out.last(), Some(Token { tok: Tok::P(p), .. }) if *p == "." || *p == "?.");
        if after_dot && c.is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Token { tok: Tok::Ident(src[start..i].to_string()), pos: start });
            continue;
        }
        let after_operand = match out.last() {
            Some(Token { tok: Tok::Ident(id), .. }) => !matches!(id.as_str(), "return" | "else" | "new" | "in" | "instanceof" | "throw" | "case" | "do"),
            Some(Token { tok: Tok::P(p), .. }) => *p == ")" || *p == "]",
            _ => false,
        };
        if c.is_ascii_digit() || (c == b'.' && !after_operand && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            // Hex.
            if c == b'0' && matches!(b.get(i + 1), Some(b'x') | Some(b'X')) {
                i += 2;
                let s = i;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let v = i64::from_str_radix(&src[s..i], 16).map_err(|_| ScriptErr::compile(start, format!("invalid hex literal [{}]", &src[start..i])))?;
                let long = matches!(b.get(i), Some(b'l') | Some(b'L'));
                if long {
                    i += 1;
                }
                out.push(Token { tok: Tok::Int(v, long), pos: start });
                continue;
            }
            let mut is_float = false;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                is_float = true;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    is_float = true;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let text = &src[start..i];
            match b.get(i) {
                Some(b'l') | Some(b'L') if !is_float => {
                    i += 1;
                    let v = text.parse::<i64>().map_err(|_| ScriptErr::compile(start, format!("Invalid long constant [{text}].")))?;
                    out.push(Token { tok: Tok::Int(v, true), pos: start });
                }
                Some(b'f') | Some(b'F') | Some(b'd') | Some(b'D') => {
                    i += 1;
                    let v = text.parse::<f64>().map_err(|_| ScriptErr::compile(start, format!("Invalid double constant [{text}].")))?;
                    out.push(Token { tok: Tok::Float(v), pos: start });
                }
                _ if is_float => {
                    let v = text.parse::<f64>().map_err(|_| ScriptErr::compile(start, format!("Invalid double constant [{text}].")))?;
                    out.push(Token { tok: Tok::Float(v), pos: start });
                }
                _ => {
                    let v = if text.len() > 1 && text.starts_with('0') && text.bytes().all(|c| (b'0'..=b'7').contains(&c)) {
                        i64::from_str_radix(&text[1..], 8).map_err(|_| ScriptErr::compile(start, format!("Invalid int constant [{text}].")))?
                    } else {
                        text.parse::<i64>().map_err(|_| ScriptErr::compile(start, format!("Invalid int constant [{text}].")))?
                    };
                    out.push(Token { tok: Tok::Int(v, false), pos: start });
                }
            }
            if b.get(i).is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_') {
                return Err(ScriptErr::compile(i, format!("unexpected character [{}].", b[i] as char)));
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                i += 1;
            }
            out.push(Token { tok: Tok::Ident(src[start..i].to_string()), pos: start });
            continue;
        }
        if c == b'"' || c == b'\'' {
            let q = c;
            i += 1;
            let mut s = String::new();
            loop {
                if i >= b.len() {
                    return Err(ScriptErr::compile(start, "unterminated string"));
                }
                let ch = b[i];
                if ch == q {
                    i += 1;
                    break;
                }
                if ch == b'\\' && i + 1 < b.len() {
                    let e = b[i + 1];
                    match e {
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'\\' => s.push('\\'),
                        b'"' => s.push('"'),
                        b'\'' => s.push('\''),
                        b'u' => return Err(ScriptErr::compile(i, "invalid escape sequence [\\u] in string literal")),
                        _ => {
                            s.push('\\');
                            s.push(e as char);
                        }
                    }
                    i += 2;
                    continue;
                }
                // Copy one UTF-8 char.
                let ch_len = utf8_len(ch);
                s.push_str(&src[i..i + ch_len]);
                i += ch_len;
            }
            out.push(Token { tok: Tok::Str(s), pos: start });
            continue;
        }
        let rest = &src[i..];
        match PUNCT.iter().find(|p| rest.starts_with(**p)) {
            Some(p) => {
                out.push(Token { tok: Tok::P(p), pos: start });
                i += p.len();
            }
            None => {
                let ch = rest.chars().next().unwrap_or('?');
                return Err(ScriptErr::compile(start, format!("unexpected character [{ch}].")));
            }
        }
    }
    out.push(Token { tok: Tok::Eof, pos: src.len() });
    Ok(out)
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}
