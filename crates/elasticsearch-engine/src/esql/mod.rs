//! ES|QL: the piped query language (`POST /_query`).
//!
//! Pipeline: [`parse`] turns the query text into [`Cmd`]s, [`exec`] runs
//! them over an in-memory [`Table`]. `FROM` loads documents through the
//! regular search path; `WHERE` commands directly after `FROM` are pushed
//! down as Query DSL when they translate (full-text functions must), and a
//! `SORT ... | LIMIT n` prefix is pushed down as a sorted top-n search.
//! Everything else (EVAL, STATS, DISSECT, GROK, MV_EXPAND, LOOKUP JOIN, ...)
//! evaluates row by row. The SQL layer ([`crate::sql`]) compiles to the same
//! commands.

pub mod analyze;
pub mod approx;
pub mod async_id;
pub mod automaton;
pub mod exec;
pub mod format;
pub mod func;
pub mod func2;
pub mod lexer;
pub mod parse;
pub mod partial;
pub mod promql;
pub mod remote;
pub(crate) mod regex_memory;
pub mod settings;
pub mod spatial;
pub mod telemetry;
pub mod ts;
pub mod typecheck;
pub mod views;
pub mod types;

use crate::error::EsError;
use serde_json::{json, Value};
use std::cmp::Ordering;

pub use exec::{run, EsqlOptions, EsqlResult, Table};
pub use parse::{parse_query, Cmd, Expr};

/// A runtime value.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Long(i64),
    Double(f64),
    Str(String),
    /// Epoch millis.
    Date(i64),
    /// `time_duration`, millis.
    Dur(i64),
    /// `date_period`: months, days.
    Period(i64, i64),
    Multi(Vec<Val>),
}

impl Val {
    /// Heap bytes beyond the value itself.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Val::Str(s) => s.capacity(),
            Val::Multi(m) => m.capacity() * std::mem::size_of::<Val>() + m.iter().map(Val::heap_bytes).sum::<usize>(),
            _ => 0,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Val::Null)
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Int(i) | Val::Long(i) => Some(*i as f64),
            Val::Double(d) => Some(*d),
            Val::Date(d) => Some(*d as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Val::Int(i) | Val::Long(i) | Val::Date(i) => Some(*i),
            Val::Double(d) if d.is_finite() => Some(*d as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Long(_) | Val::Double(_))
    }

    /// Values of a (possibly multi-valued) value, nulls skipped.
    pub fn values(&self) -> Vec<&Val> {
        match self {
            Val::Null => vec![],
            Val::Multi(v) => v.iter().filter(|x| !x.is_null()).collect(),
            v => vec![v],
        }
    }

    /// Normalise a list of values into a single value.
    pub fn from_list(mut v: Vec<Val>) -> Val {
        v.retain(|x| !x.is_null());
        match v.len() {
            0 => Val::Null,
            1 => v.pop().unwrap(),
            _ => Val::Multi(v),
        }
    }

    /// Display form (used by TO_STRING, CONCAT, text formats).
    pub fn to_text(&self) -> String {
        match self {
            Val::Null => "null".into(),
            Val::Bool(b) => b.to_string(),
            Val::Int(i) | Val::Long(i) => i.to_string(),
            Val::Double(d) => fmt_double(*d),
            Val::Str(s) => s.clone(),
            Val::Date(d) => fmt_date(*d),
            Val::Dur(ms) => fmt_duration(*ms),
            Val::Period(m, d) => fmt_period(*m, *d),
            Val::Multi(v) => format!("[{}]", v.iter().map(Val::to_text).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Val::Null => Value::Null,
            Val::Bool(b) => json!(b),
            Val::Int(i) | Val::Long(i) => json!(i),
            Val::Double(d) => {
                if d.is_finite() {
                    json!(d)
                } else {
                    Value::Null
                }
            }
            Val::Str(s) => json!(s),
            Val::Date(d) => json!(fmt_date(*d)),
            Val::Dur(_) | Val::Period(..) => json!(self.to_text()),
            Val::Multi(v) => Value::Array(v.iter().map(Val::to_json).collect()),
        }
    }

    /// Stable grouping key.
    pub fn key(&self) -> String {
        match self {
            Val::Null => "\u{0}n".into(),
            Val::Bool(b) => format!("b{b}"),
            Val::Int(i) | Val::Long(i) => format!("i{i}"),
            Val::Double(d) => {
                if d.fract() == 0.0 && d.abs() < 9.0e15 {
                    format!("i{}", *d as i64)
                } else {
                    format!("d{d}")
                }
            }
            Val::Str(s) => format!("s{s}"),
            Val::Date(d) => format!("t{d}"),
            Val::Dur(d) => format!("u{d}"),
            Val::Period(m, d) => format!("p{m}/{d}"),
            Val::Multi(v) => {
                use std::fmt::Write;
                let mut key = String::from("m");
                for value in v {
                    let part = value.key();
                    write!(&mut key, "{}:", part.len()).expect("writing to String");
                    key.push_str(&part);
                }
                key
            }
        }
    }

    /// Type name of a single value.
    pub fn type_name(&self) -> &'static str {
        match self {
            Val::Null => "null",
            Val::Bool(_) => "boolean",
            Val::Int(_) => "integer",
            Val::Long(_) => "long",
            Val::Double(_) => "double",
            Val::Str(_) => "keyword",
            Val::Date(_) => "date",
            Val::Dur(_) => "time_duration",
            Val::Period(..) => "date_period",
            Val::Multi(v) => v.first().map(Val::type_name).unwrap_or("null"),
        }
    }
}

/// Java `Double.toString`: plain decimal for `1e-3 <= |d| < 1e7`,
/// computerized scientific notation (`1.0E10`) otherwise.
pub fn fmt_double(d: f64) -> String {
    if d.is_nan() {
        return "NaN".into();
    }
    if d.is_infinite() {
        return if d > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = d.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{d}");
        return if s.contains('.') { s } else { format!("{s}.0") };
    }
    let s = format!("{d:e}");
    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
    if m.contains('.') { format!("{m}E{e}") } else { format!("{m}.0E{e}") }
}

thread_local! {
    /// The query's `time_zone`: dates render (and calendar functions run) in it.
    static ZONE: std::cell::Cell<crate::date::Tz> = const { std::cell::Cell::new(crate::date::Tz::Fixed(0)) };
}

/// The `time_zone` in effect on this thread.
pub fn zone() -> crate::date::Tz {
    ZONE.with(|z| z.get())
}

/// Restores the previous zone when dropped.
pub struct ZoneGuard(crate::date::Tz);

impl Drop for ZoneGuard {
    fn drop(&mut self) {
        ZONE.with(|z| z.set(self.0));
    }
}

/// Make `tz` the zone of this thread until the guard drops.
pub fn set_zone(tz: crate::date::Tz) -> ZoneGuard {
    ZoneGuard(ZONE.with(|z| z.replace(tz)))
}

/// A date as ES|QL prints it: ISO-8601 millis in the query's zone (`Z` for UTC).
pub fn fmt_date(ms: i64) -> String {
    let tz = zone();
    let off = tz.offset_at(ms);
    match chrono::DateTime::from_timestamp_millis(ms + off as i64 * 1000) {
        Some(dt) if off == 0 && matches!(tz, crate::date::Tz::Fixed(0)) => dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        Some(dt) => format!("{}{}", dt.format("%Y-%m-%dT%H:%M:%S%.3f"), tz.offset_string(ms)),
        None => ms.to_string(),
    }
}

fn fmt_duration(ms: i64) -> String {
    if ms % 86_400_000 == 0 {
        format!("{}d", ms / 86_400_000)
    } else if ms % 3_600_000 == 0 {
        format!("{}h", ms / 3_600_000)
    } else if ms % 60_000 == 0 {
        format!("{}m", ms / 60_000)
    } else if ms % 1000 == 0 {
        format!("{}s", ms / 1000)
    } else {
        format!("{ms}ms")
    }
}

fn fmt_period(m: i64, d: i64) -> String {
    let mut s = "P".to_string();
    if m / 12 != 0 {
        s.push_str(&format!("{}Y", m / 12));
    }
    if m % 12 != 0 {
        s.push_str(&format!("{}M", m % 12));
    }
    if d != 0 || s == "P" {
        s.push_str(&format!("{d}D"));
    }
    s
}

/// Total order used by SORT/MIN/MAX (same-typed values; numbers mixed).
pub fn cmp_vals(a: &Val, b: &Val) -> Ordering {
    match (a, b) {
        (Val::Int(x) | Val::Long(x), Val::Int(y) | Val::Long(y)) => x.cmp(y),
        (Val::Date(x), Val::Date(y)) => x.cmp(y),
        (Val::Str(x), Val::Str(y)) => x.cmp(y),
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        (Val::Dur(x), Val::Dur(y)) => x.cmp(y),
        (Val::Multi(x), Val::Multi(y)) => {
            for (p, q) in x.iter().zip(y.iter()) {
                let o = cmp_vals(p, q);
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        (Val::Null, Val::Null) => Ordering::Equal,
        (Val::Null, _) => Ordering::Greater,
        (_, Val::Null) => Ordering::Less,
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            _ => a.to_text().cmp(&b.to_text()),
        },
    }
}

/// `line L:C:` prefix for a byte offset into the query.
pub fn pos_prefix(src: &str, off: usize) -> String {
    let off = off.min(src.len());
    let before = &src[..off];
    let line = before.matches('\n').count() + 1;
    let col = before.rfind('\n').map(|p| off - p).unwrap_or(off + 1);
    format!("line {line}:{col}")
}

pub fn parsing_err(src: &str, off: usize, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "parsing_exception", format!("{}: {}", pos_prefix(src, off), msg.as_ref()))
}

pub fn verification_err(src: &str, off: usize, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "verification_exception", format!("Found 1 problem\n{}: {}", pos_prefix(src, off), msg.as_ref()))
}

/// QL `StringUtils.findSimilar` as a message suffix: candidates with a
/// Lucene Levenshtein similarity of at least 0.5, best first
/// (`, did you mean [x]?` / `, did you mean any of [a, b]?`).
pub fn did_you_mean(name: &str, candidates: &[&str]) -> String {
    let mut scored: Vec<(f32, &str)> = candidates
        .iter()
        .map(|c| {
            let max = name.chars().count().max(c.chars().count()).max(1);
            (1.0 - levenshtein(name, c) as f32 / max as f32, *c)
        })
        .filter(|(d, _)| *d >= 0.5)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    match scored.len() {
        0 => String::new(),
        1 => format!(", did you mean [{}]?", scored[0].1),
        _ => format!(", did you mean any of [{}]?", scored.iter().map(|s| s.1).collect::<Vec<_>>().join(", ")),
    }
}

/// Levenshtein edit distance (query-rule fuzzy criteria, LEVENSHTEIN()).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests;
