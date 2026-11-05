//! XContent-style parse errors for ML request bodies.
//!
//! ES parses ML configs with `ObjectParser`s, whose errors carry the
//! Jackson location of the token being parsed (`[1:46] [detector] failed to
//! parse field [function]`). [`Pos`] trees record where every value of the
//! raw body starts and ends so the same locations can be reported.

use crate::error::EsError;

/// 1-based (line, column) of a token.
pub type Loc = (u32, u32);

/// Positions of one JSON value of the raw body.
#[derive(Debug, Clone, Default)]
pub struct Pos {
    pub start: Loc,
    pub end: Loc,
    /// Object members: key, key location, value.
    pub fields: Vec<(String, Loc, Pos)>,
    pub items: Vec<Pos>,
}

static EMPTY: Pos = Pos { start: (1, 1), end: (1, 1), fields: Vec::new(), items: Vec::new() };

impl Pos {
    /// The position of member `k` (an empty position when unknown).
    pub fn get(&self, k: &str) -> &Pos {
        self.fields.iter().find(|(n, _, _)| n == k).map(|(_, _, p)| p).unwrap_or(&EMPTY)
    }
    /// Where key `k` starts.
    pub fn key(&self, k: &str) -> Loc {
        self.fields.iter().find(|(n, _, _)| n == k).map(|(_, l, _)| *l).unwrap_or(self.start)
    }
    pub fn at(&self, i: usize) -> &Pos {
        self.items.get(i).unwrap_or(&EMPTY)
    }
    pub fn empty() -> &'static Pos {
        &EMPTY
    }
}

/// Parse the positions of a raw JSON body (`None` when it is not JSON).
pub fn positions(raw: &str) -> Pos {
    let mut p = P { c: raw.chars().collect(), i: 0, line: 1, col: 1 };
    p.ws();
    p.value().unwrap_or_default()
}

struct P {
    c: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
}

impl P {
    fn loc(&self) -> Loc {
        (self.line, self.col)
    }
    fn bump(&mut self) -> Option<char> {
        let ch = *self.c.get(self.i)?;
        self.i += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }
    fn string(&mut self) -> Option<String> {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump()? {
                '"' => return Some(s),
                '\\' => {
                    let e = self.bump()?;
                    s.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    });
                }
                ch => s.push(ch),
            }
        }
    }
    fn value(&mut self) -> Option<Pos> {
        let start = self.loc();
        match self.peek()? {
            '{' => {
                self.bump();
                let mut fields = Vec::new();
                loop {
                    self.ws();
                    match self.peek()? {
                        '}' => {
                            let end = self.loc();
                            self.bump();
                            return Some(Pos { start, end, fields, items: vec![] });
                        }
                        ',' => {
                            self.bump();
                        }
                        '"' => {
                            let kl = self.loc();
                            let k = self.string()?;
                            self.ws();
                            if self.peek()? == ':' {
                                self.bump();
                            }
                            self.ws();
                            let v = self.value()?;
                            fields.push((k, kl, v));
                        }
                        _ => return None,
                    }
                }
            }
            '[' => {
                self.bump();
                let mut items = Vec::new();
                loop {
                    self.ws();
                    match self.peek()? {
                        ']' => {
                            let end = self.loc();
                            self.bump();
                            return Some(Pos { start, end, fields: vec![], items });
                        }
                        ',' => {
                            self.bump();
                        }
                        _ => items.push(self.value()?),
                    }
                }
            }
            '"' => {
                self.string()?;
                Some(Pos { start, end: start, ..Default::default() })
            }
            _ => {
                while matches!(self.peek(), Some(c) if !matches!(c, ',' | '}' | ']') && !c.is_whitespace()) {
                    self.bump();
                }
                Some(Pos { start, end: start, ..Default::default() })
            }
        }
    }
}

pub fn fmt(l: Loc) -> String {
    format!("[{}:{}]", l.0, l.1)
}

/// `x_content_parse_exception`.
pub fn xparse(reason: impl Into<String>) -> EsError {
    EsError::new(400, "x_content_parse_exception", reason)
}

/// `[l:c] [parser] failed to parse field [field]` wrapping `cause`.
pub fn wrap(at: Loc, parser: &str, field: &str, cause: EsError) -> EsError {
    xparse(format!("{} [{parser}] failed to parse field [{field}]", fmt(at))).caused_by(cause)
}

/// `[l:c] [parser] unknown field [k]` (with ES's "did you mean" hint:
/// `SuggestingErrorOnUnknown`, Lucene `LevenshteinDistance` > 0.5, best
/// first).
pub fn unknown(at: Loc, parser: &str, k: &str, known: &[&str]) -> EsError {
    let mut msg = format!("{} [{parser}] unknown field [{k}]", fmt(at));
    let mut close: Vec<(f32, &str)> = known.iter().map(|c| (levenshtein(k, c), *c)).filter(|(d, _)| *d > 0.5).collect();
    close.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    if close.len() == 1 {
        msg.push_str(&format!(" did you mean [{}]?", close[0].1));
    } else if !close.is_empty() {
        msg.push_str(&format!(" did you mean any of [{}]?", close.iter().map(|c| c.1).collect::<Vec<_>>().join(", ")));
    }
    xparse(msg)
}

/// Lucene `LevenshteinDistance.getDistance`: 1 - edits / max length.
fn levenshtein(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return if a == b { 1.0 } else { 0.0 };
    }
    let mut prev: Vec<usize> = (0..=a.len()).collect();
    for j in 1..=b.len() {
        let mut cur = vec![j; a.len() + 1];
        for i in 1..=a.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[i] = (cur[i - 1] + 1).min(prev[i] + 1).min(prev[i - 1] + cost);
        }
        prev = cur;
    }
    1.0 - prev[a.len()] as f32 / a.len().max(b.len()) as f32
}

/// `status_exception` (400), ES's `ExceptionsHelper.badRequestException`.
pub fn bad_request(reason: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", reason)
}

/// JSON token type name as ES reports it (`VALUE_STRING`, ...).
pub fn token_name(v: &serde_json::Value) -> &'static str {
    use serde_json::Value;
    match v {
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Array(_) => "START_ARRAY",
        Value::Null => "VALUE_NULL",
        Value::Object(_) => "START_OBJECT",
    }
}
