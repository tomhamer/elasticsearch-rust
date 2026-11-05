//! Script errors and their ES `script_exception` rendering.

use crate::error::EsError;
use serde_json::json;

/// A compile or runtime failure inside a script. `kind` is the ES type of
/// the underlying Java exception (`illegal_argument_exception`, ...).
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptErr {
    pub kind: String,
    /// Resource failures bypass Java catch blocks and retain HTTP 429 metadata.
    pub breaker: Option<EsError>,
    pub msg: String,
    /// Byte offset in the source of the failing node.
    pub pos: Option<usize>,
    pub compile: bool,
    /// JDK stack frames ES prints above the script frame (`java.base/...`).
    pub frames: &'static [&'static str],
    /// Nested Java cause (`caused_by` of the exception), as (type, reason).
    pub cause: Option<Box<(String, String)>>,
}

pub type SResult<T> = std::result::Result<T, Box<ScriptErr>>;

impl ScriptErr {
    pub fn compile(pos: usize, msg: impl Into<String>) -> ScriptErr {
        ScriptErr { kind: "illegal_argument_exception".into(), msg: msg.into(), pos: Some(pos), compile: true, frames: &[], cause: None, breaker: None }
    }
    pub fn new(kind: &str, msg: impl Into<String>) -> Box<ScriptErr> {
        Box::new(ScriptErr { kind: kind.into(), msg: msg.into(), pos: None, compile: false, frames: &[], cause: None, breaker: None })
    }
    pub fn breaker(error: EsError) -> Box<ScriptErr> {
        let mut result = Self::new("circuit_breaking_exception", error.reason.clone());
        result.breaker = Some(error);
        result
    }
    /// Attach a nested Java cause.
    pub fn with_cause(mut self, kind: &str, msg: impl Into<String>) -> ScriptErr {
        self.cause = Some(Box::new((kind.to_string(), msg.into())));
        self
    }
    /// Attach the JDK frames ES shows for this exception site.
    pub fn frames(mut self: Box<Self>, frames: &'static [&'static str]) -> Box<Self> {
        self.frames = frames;
        self
    }
    pub fn at(mut self: Box<Self>, pos: usize) -> Box<Self> {
        if self.pos.is_none() {
            self.pos = Some(pos);
        }
        self
    }
}

pub fn illegal_arg(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("illegal_argument_exception", msg)
}
pub fn npe(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("null_pointer_exception", msg)
}
pub fn class_cast(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("class_cast_exception", msg)
}
pub fn illegal_state(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("illegal_state_exception", msg)
}
pub fn out_of_bounds(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("index_out_of_bounds_exception", msg)
}
pub fn unsupported(msg: impl Into<String>) -> Box<ScriptErr> {
    ScriptErr::new("unsupported_operation_exception", msg)
}

/// Source offsets of statement starts (ascending), for `script_stack`.
pub fn to_es_error(e: &ScriptErr, source: &str, stmt_starts: &[usize]) -> EsError {
    if let Some(error) = &e.breaker { return error.clone(); }
    let mut err = EsError::new(400, "script_exception", if e.compile { "compile error" } else { "runtime error" });
    let mut stack: Vec<String> = e.frames.iter().map(|f| f.to_string()).collect();
    let mut position = None;
    if let Some(off) = e.pos {
        let off = off.min(source.len());
        if e.compile {
            // ES `PainlessScriptEngine.convertToScriptException`: a window of
            // 25 chars either side of the offset, with `... ` markers.
            let chars: Vec<char> = source.chars().collect();
            let coff = source.get(..off).map(|s| s.chars().count()).unwrap_or(chars.len());
            let start = coff.saturating_sub(25);
            let end = (coff + 25).min(chars.len());
            let mut snippet = String::new();
            let mut pointer = String::new();
            if start > 0 {
                snippet.push_str("... ");
                pointer.push_str("    ");
            }
            snippet.extend(&chars[start..end]);
            if end < chars.len() {
                snippet.push_str(" ...");
            }
            pointer.push_str(&" ".repeat(coff - start));
            pointer.push_str("^---- HERE");
            stack.push(snippet);
            stack.push(pointer);
            let bstart = source.char_indices().nth(start).map(|(i, _)| i).unwrap_or(source.len());
            let bend = source.char_indices().nth(end).map(|(i, _)| i).unwrap_or(source.len());
            position = Some(json!({ "offset": off, "start": bstart, "end": bend }));
        } else {
            let start = stmt_starts.iter().copied().filter(|s| *s <= off).max().unwrap_or(0);
            let end = stmt_starts.iter().copied().filter(|s| *s > start).min().unwrap_or(source.len());
            let end = end.max(start).min(source.len());
            let snippet = safe_slice(source, start, end);
            stack.push(snippet.to_string());
            let pad = source.get(start..off).map(|s| s.chars().count()).unwrap_or(0);
            stack.push(format!("{}^---- HERE", " ".repeat(pad)));
            position = Some(json!({ "offset": off, "start": start, "end": end }));
        }
    }
    err = err.with("script_stack", stack).with("script", source.to_string()).with("lang", "painless");
    if let Some(p) = position {
        err = err.with("position", p);
    }
    // A Java exception without a message renders as `"reason": null`.
    let reason = if !e.compile && e.msg.is_empty() { crate::error::NULL_REASON.to_string() } else { e.msg.clone() };
    let mut cause = EsError::new(400, &e.kind, reason);
    if let Some(c) = &e.cause {
        cause = cause.caused_by(EsError::new(400, &c.0, c.1.clone()));
    }
    err.caused_by(cause)
}

fn safe_slice(s: &str, mut a: usize, mut b: usize) -> &str {
    while a > 0 && !s.is_char_boundary(a) {
        a -= 1;
    }
    while b < s.len() && !s.is_char_boundary(b) {
        b += 1;
    }
    &s[a..b]
}

impl From<rustic_core::RusticError> for Box<ScriptErr> {
    fn from(error: rustic_core::RusticError) -> Self {
        if matches!(error, rustic_core::RusticError::Memory(_)) {
            ScriptErr::breaker(EsError::from(error))
        } else {
            let error = EsError::from(error);
            ScriptErr::new(&error.kind, error.reason)
        }
    }
}
