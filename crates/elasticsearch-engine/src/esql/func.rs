//! ES|QL scalar functions, casts and operators.

use super::parse::BinOp;
use super::{cmp_vals, Val};
use crate::error::{EsError, Result};
use chrono::{Datelike, NaiveDate, TimeZone, Timelike};
use std::cmp::Ordering;
use std::collections::HashMap;

pub const AGGS: &[&str] = &[
    "AVG",
    "COUNT",
    "COUNT_DISTINCT",
    "MAX",
    "MEDIAN",
    "MEDIAN_ABSOLUTE_DEVIATION",
    "MIN",
    "PERCENTILE",
    "ST_CENTROID_AGG",
    "ST_EXTENT_AGG",
    "STD_DEV",
    "STDDEV_POP",
    "SUM",
    "TOP",
    "VALUES",
    "WEIGHTED_AVG",
    "VARIANCE",
    "VAR_POP",
    "SAMPLE",
    "FIRST",
    "LAST",
    "PRESENT",
    "ABSENT",
];

pub const FULL_TEXT: &[&str] = &["MATCH", "MATCH_OP", "MATCH_PHRASE", "QSTR", "KQL", "TERM", "MULTI_MATCH", "KNN"];

/// How errors name a full-text function (`[:] operator`, `[MATCH] function`).
pub fn fulltext_label(name: &str) -> String {
    if name == "MATCH_OP" { "[:] operator".into() } else { format!("[{name}] function") }
}

pub const SCALARS: &[&str] = &[
    // math
    "ABS", "ACOS", "ASIN", "ATAN", "ATAN2", "CBRT", "CEIL", "COS", "COSH", "E", "EXP", "FLOOR", "HYPOT", "LOG", "LOG10", "PI", "POW", "ROUND", "SIGNUM", "SIN", "SINH", "SQRT", "TAN", "TANH", "TAU", "COPY_SIGN", "ROUND_TO",
    // string
    "BIT_LENGTH", "BYTE_LENGTH", "CONCAT", "ENDS_WITH", "FROM_BASE64", "LEFT", "LENGTH", "LOCATE", "LTRIM", "REPEAT", "REPLACE", "REVERSE", "RIGHT", "RTRIM", "SPACE", "SPLIT", "STARTS_WITH", "SUBSTRING", "TO_BASE64", "TO_LOWER", "TO_UPPER", "TRIM", "CONTAINS", "LEVENSHTEIN", "UCASE", "LCASE", "CHAR_LENGTH",
    // date
    "DATE_DIFF", "DATE_EXTRACT", "DATE_FORMAT", "DATE_PARSE", "DATE_TRUNC", "NOW", "DAY_NAME", "MONTH_NAME",
    // conversion
    "TO_BOOLEAN", "TO_BOOL", "TO_DATETIME", "TO_DT", "TO_DOUBLE", "TO_DBL", "TO_INTEGER", "TO_INT", "TO_IP", "TO_LONG", "TO_STRING", "TO_STR", "TO_UNSIGNED_LONG", "TO_UL", "TO_ULONG", "TO_VERSION", "TO_VER", "TO_DEGREES", "TO_RADIANS", "TO_GEOPOINT", "TO_CARTESIANPOINT", "TO_GEOSHAPE", "TO_CARTESIANSHAPE", "TO_TIMEDURATION", "TO_DATEPERIOD", "TO_DATE_NANOS", "TO_KEYWORD", "TO_TEXT",
    // conditional
    "CASE", "COALESCE", "GREATEST", "LEAST",
    // multivalue
    "MV_APPEND", "MV_AVG", "MV_CONCAT", "MV_COUNT", "MV_DEDUPE", "MV_FIRST", "MV_LAST", "MV_MAX", "MV_MEDIAN", "MV_MEDIAN_ABSOLUTE_DEVIATION", "MV_MIN", "MV_PERCENTILE", "MV_SLICE", "MV_SORT", "MV_SUM", "MV_ZIP", "MV_CONTAINS",
    // ip / spatial
    "CIDR_MATCH", "IP_PREFIX", "ST_X", "ST_Y", "ST_DISTANCE", "ST_INTERSECTS", "ST_DISJOINT", "ST_WITHIN", "ST_CONTAINS",
    // grouping
    "BUCKET", "TBUCKET",
    // SQL extras
    "IFNULL", "ISNULL", "NULLIF", "IIF", "UPPER", "LOWER", "SUBSTR", "MOD", "POWER", "TRUNCATE", "YEAR", "MONTH", "DAY", "DAY_OF_MONTH", "HOUR", "MINUTE", "SECOND", "CURRENT_TIMESTAMP", "CURRENT_DATE", "NVL", "CHAR_LENGTH", "ASCII", "INSERT", "POSITION",
    // ES SQL-only scalars ([`sql_scalar`]; aliases resolve in [`sql_alias`])
    "DATETIME_FORMAT", "DATETIME_PARSE", "SIGN", "COT", "EXPM1", "RANDOM", "DEGREES", "RADIANS", "CHAR", "OCTET_LENGTH",
    "DAY_OF_WEEK", "ISO_DAY_OF_WEEK", "DAY_OF_YEAR", "WEEK_OF_YEAR", "ISO_WEEK_OF_YEAR", "QUARTER", "MINUTE_OF_DAY", "DATE_ADD", "DATE_PART",
    // scoring
    "SCORE",
];

/// Type conversion functions (`TO_*`), which may read union-type fields.
pub fn is_conversion(name: &str) -> bool {
    name.starts_with("TO_") && !matches!(name, "TO_UPPER" | "TO_LOWER" | "TO_DEGREES" | "TO_RADIANS" | "TO_BASE64" | "TO_TIMEDURATION" | "TO_DATEPERIOD")
}

pub fn known(name: &str) -> bool {
    AGGS.contains(&name) || FULL_TEXT.contains(&name) || SCALARS.contains(&name) || super::func2::NAMES.contains(&name)
}

/// Error kind of a per-row evaluation failure: the row's result is `null`
/// and the failure becomes a response warning (`evaluation of [..] failed`).
pub const EVAL_WARN: &str = "\u{0}esql_eval_warning";

/// A per-row evaluation failure carrying the exception text the warning prints.
pub fn eval_warning(exception: String) -> EsError {
    EsError::new(400, EVAL_WARN, exception)
}

/// Error kind of a query-failing evaluation error whose message names the
/// call: `{src}` in the reason is replaced by the call's source text.
pub const SRC_ERR: &str = "\u{0}esql_src_error";

fn bad_arity(name: &str) -> EsError {
    EsError::new(400, "verification_exception", format!("Found 1 problem\nfunction [{name}] called with the wrong number of arguments"))
}

pub fn ip_to_bytes(s: &str) -> Option<Vec<u8>> {
    let ip: std::net::IpAddr = s.trim().parse().ok()?;
    Some(match ip {
        std::net::IpAddr::V4(v) => v.octets().to_vec(),
        std::net::IpAddr::V6(v) => v.octets().to_vec(),
    })
}

fn cidr_match(ip: &str, cidr: &str) -> bool {
    let (net, len) = match cidr.split_once('/') {
        Some((n, l)) => (n, l.parse::<usize>().unwrap_or(0)),
        None => (cidr, usize::MAX),
    };
    let (Some(mut a), Some(mut b)) = (ip_to_bytes(ip), ip_to_bytes(net)) else { return false };
    let mut len = len;
    // ES `CIDRUtils.isBetween`: an IPv4 address and an IPv6 range (or the
    // reverse) compare in the IPv4-mapped IPv6 form; an IPv4 prefix
    // length counts from the mapped address's 96th bit.
    if a.len() != b.len() {
        let mapped = |v: &[u8]| -> Vec<u8> { let mut m = vec![0u8; 10]; m.extend([0xff, 0xff]); m.extend_from_slice(v); m };
        if a.len() == 4 {
            a = mapped(&a);
        }
        if b.len() == 4 {
            b = mapped(&b);
            len = len.saturating_add(96);
        }
    }
    let bits = len.min(a.len() * 8);
    for i in 0..bits {
        let m = 0x80u8 >> (i % 8);
        if a[i / 8] & m != b[i / 8] & m {
            return false;
        }
    }
    true
}

pub fn parse_date_str(s: &str) -> Option<i64> {
    let f = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok()?;
    // A SQL session zone reads a date string without an offset as local time.
    f.parse_with_tz(s, super::literal_zone().unwrap_or(crate::date::Tz::UTC)).ok()
}

/// Arithmetic / comparison / logic.
pub fn binop(op: BinOp, a: &Val, b: &Val) -> Val {
    use BinOp::*;
    match op {
        And => match (a, b) {
            (Val::Bool(false), _) | (_, Val::Bool(false)) => Val::Bool(false),
            (Val::Bool(true), Val::Bool(true)) => Val::Bool(true),
            _ => Val::Null,
        },
        Or => match (a, b) {
            (Val::Bool(true), _) | (_, Val::Bool(true)) => Val::Bool(true),
            (Val::Bool(false), Val::Bool(false)) => Val::Bool(false),
            _ => Val::Null,
        },
        Eq | Neq | Lt | Le | Gt | Ge => {
            if a.is_null() || b.is_null() || matches!(a, Val::Multi(_)) || matches!(b, Val::Multi(_)) {
                return Val::Null;
            }
            let o = match compare(a, b) {
                Some(o) => o,
                None => return Val::Null,
            };
            Val::Bool(match op {
                Eq => o == Ordering::Equal,
                Neq => o != Ordering::Equal,
                Lt => o == Ordering::Less,
                Le => o != Ordering::Greater,
                Gt => o == Ordering::Greater,
                _ => o != Ordering::Less,
            })
        }
        _ => arith(op, a, b),
    }
}

/// Comparison with implicit casts (string literals vs dates/numbers).
pub fn compare(a: &Val, b: &Val) -> Option<Ordering> {
    match (a, b) {
        (Val::Date(x), Val::Str(s)) => parse_date_str(s).map(|y| x.cmp(&y)),
        (Val::Str(s), Val::Date(y)) => parse_date_str(s).map(|x| x.cmp(y)),
        (Val::Nanos(x), Val::Str(s)) => parse_nanos_str(s).map(|y| x.cmp(&y)),
        (Val::Str(s), Val::Nanos(y)) => parse_nanos_str(s).map(|x| x.cmp(y)),
        (Val::ULong(x), Val::Str(s)) => s.trim().parse::<u64>().ok().map(|y| x.cmp(&y)),
        (Val::Str(s), Val::ULong(y)) => s.trim().parse::<u64>().ok().map(|x| x.cmp(y)),
        (Val::Str(x), Val::Str(y)) => Some(x.cmp(y)),
        (Val::Bool(x), Val::Bool(y)) => Some(x.cmp(y)),
        (Val::Str(s), n) | (n, Val::Str(s)) if n.is_numeric() => {
            let f: f64 = s.parse().ok()?;
            let nf = n.as_f64()?;
            let o = nf.partial_cmp(&f)?;
            Some(if matches!(a, Val::Str(_)) { o.reverse() } else { o })
        }
        (Val::Str(_), _) | (_, Val::Str(_)) => None,
        (Val::Bool(_), _) | (_, Val::Bool(_)) => None,
        _ => Some(cmp_vals(a, b)),
    }
}

fn arith(op: BinOp, a: &Val, b: &Val) -> Val {
    use BinOp::*;
    if a.is_null() || b.is_null() || matches!(a, Val::Multi(_)) || matches!(b, Val::Multi(_)) {
        return Val::Null;
    }
    if let (Val::Date(_), Val::Str(t)) = (a, b) {
        if let Some(amount) = parse_temporal(t) {
            return arith(op, a, &amount);
        }
    }
    if let (Val::Str(t), Val::Date(_)) = (a, b) {
        if let Some(amount) = parse_temporal(t) {
            return arith(op, &amount, b);
        }
    }
    match (a, b) {
        (Val::Nanos(n), Val::Dur(ms)) if matches!(op, Add | Sub) => {
            let delta = ms.checked_mul(super::NANOS_PER_MILLI);
            let r = delta.and_then(|d| if op == Add { n.checked_add(d) } else { n.checked_sub(d) });
            return r.filter(|r| *r >= 0).map(Val::Nanos).unwrap_or(Val::Null);
        }
        (Val::Dur(ms), Val::Nanos(n)) if op == Add => {
            return ms.checked_mul(super::NANOS_PER_MILLI).and_then(|d| n.checked_add(d)).filter(|r| *r >= 0).map(Val::Nanos).unwrap_or(Val::Null);
        }
        (Val::Nanos(n), Val::Period(m, days)) if matches!(op, Add | Sub) => {
            let s = if op == Add { 1 } else { -1 };
            let (ms, sub) = (n.div_euclid(super::NANOS_PER_MILLI), n.rem_euclid(super::NANOS_PER_MILLI));
            let r = add_period(ms, s * m, s * days).checked_mul(super::NANOS_PER_MILLI).and_then(|x| x.checked_add(sub));
            return r.filter(|r| *r >= 0).map(Val::Nanos).unwrap_or(Val::Null);
        }
        (Val::Date(d), Val::Dur(ms)) if matches!(op, Add | Sub) => return Val::Date(if op == Add { d + ms } else { d - ms }),
        (Val::Dur(ms), Val::Date(d)) if op == Add => return Val::Date(d + ms),
        (Val::Date(d), Val::Period(m, days)) if matches!(op, Add | Sub) => {
            let s = if op == Add { 1 } else { -1 };
            return Val::Date(add_period(*d, s * m, s * days));
        }
        (Val::Period(m, days), Val::Date(d)) if op == Add => return Val::Date(add_period(*d, *m, *days)),
        (Val::Dur(x), Val::Dur(y)) if matches!(op, Add | Sub) => return Val::Dur(if op == Add { x + y } else { x - y }),
        (Val::Period(m1, d1), Val::Period(m2, d2)) if matches!(op, Add | Sub) => {
            return if op == Add { Val::Period(m1 + m2, d1 + d2) } else { Val::Period(m1 - m2, d1 - d2) };
        }
        _ => {}
    }
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => int_op(op, *x, *y, true),
        (Val::Int(x) | Val::Long(x), Val::Int(y) | Val::Long(y)) => int_op(op, *x, *y, false),
        _ => {
            let (Some(x), Some(y)) = (num(a), num(b)) else { return Val::Null };
            let r = match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => {
                    if y == 0.0 {
                        return Val::Null;
                    }
                    x / y
                }
                Mod => {
                    if y == 0.0 {
                        return Val::Null;
                    }
                    x % y
                }
                _ => return Val::Null,
            };
            if r.is_finite() { Val::Double(r) } else { Val::Null }
        }
    }
}

fn num(v: &Val) -> Option<f64> {
    match v {
        Val::Int(i) | Val::Long(i) => Some(*i as f64),
        Val::Double(d) => Some(*d),
        Val::ULong(u) => Some(*u as f64),
        _ => None,
    }
}

fn int_op(op: BinOp, x: i64, y: i64, int: bool) -> Val {
    use BinOp::*;
    let r = match op {
        Add => x.checked_add(y),
        Sub => x.checked_sub(y),
        Mul => x.checked_mul(y),
        Div => {
            if y == 0 {
                None
            } else {
                x.checked_div(y)
            }
        }
        Mod => {
            if y == 0 {
                None
            } else {
                x.checked_rem(y)
            }
        }
        _ => None,
    };
    match r {
        Some(v) if int => {
            if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                Val::Int(v)
            } else {
                Val::Null
            }
        }
        Some(v) => Val::Long(v),
        None => Val::Null,
    }
}

/// Run a calendar computation on the local wall-clock millis of the query's
/// zone and map the local result back to an instant.
fn in_zone(ms: i64, f: impl Fn(i64) -> Option<i64>) -> Option<i64> {
    let tz = super::zone();
    if tz == crate::date::Tz::UTC {
        return f(ms);
    }
    let local = f(ms + tz.offset_at(ms) as i64 * 1000)?;
    Some(tz.local_to_utc(crate::date::from_millis(local)))
}

/// The local wall-clock millis of an instant in the query's zone.
fn local_ms(ms: i64) -> i64 {
    ms + super::zone().offset_at(ms) as i64 * 1000
}

pub fn add_period(ms: i64, months: i64, days: i64) -> i64 {
    in_zone(ms, |l| Some(add_period_utc(l, months, days))).unwrap_or(ms)
}

fn add_period_utc(ms: i64, months: i64, days: i64) -> i64 {
    let dt = chrono::Utc.timestamp_millis_opt(ms).single();
    let Some(dt) = dt else { return ms };
    let mut d = dt.naive_utc();
    if months != 0 {
        let total = d.year() as i64 * 12 + d.month0() as i64 + months;
        let (y, m0) = (total.div_euclid(12) as i32, total.rem_euclid(12) as u32);
        let last = days_in_month(y, m0 + 1);
        let day = d.day().min(last);
        if let Some(nd) = NaiveDate::from_ymd_opt(y, m0 + 1, day) {
            d = nd.and_time(d.time());
        }
    }
    d += chrono::Duration::days(days);
    d.and_utc().timestamp_millis()
}

fn days_in_month(y: i32, m: u32) -> u32 {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    (NaiveDate::from_ymd_opt(ny, nm, 1).unwrap() - NaiveDate::from_ymd_opt(y, m, 1).unwrap()).num_days() as u32
}

/// Truncate a date to an interval (in the query's zone).
pub fn date_trunc(ms: i64, iv: &Val) -> Option<i64> {
    in_zone(ms, |l| date_trunc_utc(l, iv))
}

fn date_trunc_utc(ms: i64, iv: &Val) -> Option<i64> {
    let d = chrono::Utc.timestamp_millis_opt(ms).single()?.naive_utc();
    match iv {
        Val::Period(m, 0) if *m > 0 => {
            let total = d.year() as i64 * 12 + d.month0() as i64;
            let t = total - total.rem_euclid(*m);
            let nd = NaiveDate::from_ymd_opt(t.div_euclid(12) as i32, (t.rem_euclid(12) + 1) as u32, 1)?;
            Some(nd.and_hms_opt(0, 0, 0)?.and_utc().timestamp_millis())
        }
        Val::Period(0, 7) => {
            let nd = d.date() - chrono::Duration::days(d.weekday().num_days_from_monday() as i64);
            Some(nd.and_hms_opt(0, 0, 0)?.and_utc().timestamp_millis())
        }
        Val::Period(0, days) if *days > 0 => {
            let step = days * 86_400_000;
            Some(ms - ms.rem_euclid(step))
        }
        Val::Dur(step) if *step > 0 => Some(ms - ms.rem_euclid(*step)),
        _ => None,
    }
}

fn period_millis(v: &Val) -> Option<i64> {
    match v {
        Val::Dur(ms) => Some(*ms),
        Val::Period(m, d) => Some(m * 30 * 86_400_000 + d * 86_400_000),
        _ => None,
    }
}

/// An `unsigned_long` from a double: its integer part, exact `Long` below
/// `i64::MAX`, `ULong` above; `None` when negative or beyond `u64`.
pub fn unsigned_from_f64(d: f64) -> Option<Val> {
    if !d.is_finite() || d < 0.0 || d >= 18_446_744_073_709_551_616.0 {
        return None;
    }
    Some(unsigned(d.trunc() as u64))
}

/// An `unsigned_long` value: `Long` within `i64`, `ULong` beyond.
pub fn unsigned(u: u64) -> Val {
    if u <= i64::MAX as u64 { Val::Long(u as i64) } else { Val::ULong(u) }
}

/// `TO_UNSIGNED_LONG`: exact over the whole u64 range; a negative or
/// out-of-range value is null with ES's warning.
pub fn to_unsigned_long(v: &Val) -> Result<Val> {
    let out_of_range = |x: String| eval_warning(format!("org.elasticsearch.xpack.esql.core.InvalidArgumentException: [{x}] out of [unsigned_long] range"));
    Ok(match v {
        Val::Int(i) | Val::Long(i) | Val::Date(i) => {
            if *i < 0 {
                return Err(out_of_range(i.to_string()));
            }
            Val::Long(*i)
        }
        Val::Nanos(n) => Val::Long(*n),
        Val::ULong(u) => Val::ULong(*u),
        Val::Double(d) => match unsigned_from_f64(*d) {
            Some(u) => u,
            None => return Err(out_of_range(fmt_double_plain(*d))),
        },
        Val::Bool(b) => Val::Long(*b as i64),
        Val::Str(s) => {
            let t = s.trim();
            if let Ok(u) = t.parse::<u64>() {
                unsigned(u)
            } else if let Ok(d) = t.parse::<f64>() {
                match unsigned_from_f64(d) {
                    Some(u) => u,
                    None => return Err(out_of_range(t.to_string())),
                }
            } else {
                return Err(eval_warning(format!("org.elasticsearch.xpack.esql.core.InvalidArgumentException: Cannot parse number [{t}]")));
            }
        }
        Val::Multi(m) => {
            let mut out = Vec::with_capacity(m.len());
            for x in m {
                out.push(to_unsigned_long(x)?);
            }
            Val::from_list(out)
        }
        _ => Val::Null,
    })
}

fn fmt_double_plain(d: f64) -> String {
    super::fmt_double(d)
}

/// Epoch nanos of a date string: the millisecond parse plus the
/// sub-millisecond digits of its fraction.
pub fn parse_nanos_str(s: &str) -> Option<i64> {
    let t = s.trim();
    let ms = parse_date_str(t)?;
    match rustic_core::fieldtypes::numeric::fraction_nanos(t) {
        Some(frac) => rustic_core::fieldtypes::numeric::nanos_from_parts(ms.div_euclid(1000), frac).ok(),
        None => rustic_core::fieldtypes::numeric::nanos_from_millis(ms, 0).ok(),
    }
}

/// A `date_nanos` `_source` value as the field indexes it (the mapping's
/// format, nanosecond fraction kept).
pub fn source_nanos(v: &serde_json::Value, fmt: Option<&crate::date::DateFormatter>) -> Option<i64> {
    use rustic_core::fieldtypes::numeric;
    match v {
        serde_json::Value::String(s) => {
            let ms = match fmt {
                Some(f) => f.parse(s).ok()?,
                None => parse_date_str(s)?,
            };
            match numeric::fraction_nanos(s) {
                Some(frac) => numeric::nanos_from_parts(ms.div_euclid(1000), frac).ok(),
                None => numeric::nanos_from_millis(ms, 0).ok(),
            }
        }
        serde_json::Value::Number(n) => {
            let x = n.as_f64()?;
            let ms = match fmt {
                Some(f) if f.accepts_numbers() => f.parse_number(x.trunc()).ok()?,
                _ => x.trunc() as i64,
            };
            let sub = ((x - x.trunc()) * 1e6).round() as u32;
            numeric::nanos_from_millis(ms, sub).ok()
        }
        _ => None,
    }
}

/// `TO_DATE_NANOS`: a long (or double, or unsigned long) is epoch
/// *nanoseconds*; a millisecond date scales up; a string keeps its
/// sub-millisecond digits. Dates before 1970 cannot be nanosecond dates.
pub fn to_date_nanos(v: &Val) -> Result<Val> {
    let before = || eval_warning("java.lang.IllegalArgumentException: Nanosecond dates before 1970-01-01T00:00:00.000Z are not supported.".into());
    Ok(match v {
        Val::Nanos(n) => Val::Nanos(*n),
        Val::Int(i) | Val::Long(i) => {
            if *i < 0 {
                return Err(before());
            }
            Val::Nanos(*i)
        }
        Val::ULong(u) => return Err(eval_warning(format!("org.elasticsearch.xpack.esql.core.InvalidArgumentException: [{u}] out of [long] range"))),
        Val::Double(d) => {
            if !d.is_finite() || *d < 0.0 || *d >= 9.223_372_036_854_776e18 {
                return Err(before());
            }
            Val::Nanos(d.trunc() as i64)
        }
        Val::Date(ms) => {
            if *ms < 0 {
                return Err(before());
            }
            match ms.checked_mul(super::NANOS_PER_MILLI) {
                Some(n) => Val::Nanos(n),
                None => return Err(eval_warning(format!("java.lang.IllegalArgumentException: [{}] is after 2262-04-11T23:47:16.854775807 and cannot be converted to nanoseconds", super::fmt_date(*ms)))),
            }
        }
        Val::Str(s) => match parse_nanos_str(s) {
            Some(n) => Val::Nanos(n),
            None => return Err(eval_warning(format!("java.lang.IllegalArgumentException: failed to parse date field [{}] with format [strict_date_optional_time_nanos]", s.trim()))),
        },
        Val::Multi(m) => {
            let mut out = Vec::with_capacity(m.len());
            for x in m {
                out.push(to_date_nanos(x)?);
            }
            Val::from_list(out)
        }
        _ => Val::Null,
    })
}

/// The arguments a function without its own `unsigned_long` /
/// `date_nanos` handling reads: a `ULong` as a double (as before), a
/// `Nanos` as its millisecond date.
fn legacy_args(name: &str, args: &[Val]) -> Option<Vec<Val>> {
    fn wide(v: &Val) -> bool {
        match v {
            Val::ULong(_) | Val::Nanos(_) => true,
            Val::Multi(m) => m.iter().any(wide),
            _ => false,
        }
    }
    fn narrow(v: &Val) -> Val {
        match v {
            Val::ULong(u) => Val::Double(*u as f64),
            Val::Nanos(n) => Val::Date(n.div_euclid(super::NANOS_PER_MILLI)),
            Val::Multi(m) => Val::Multi(m.iter().map(narrow).collect()),
            o => o.clone(),
        }
    }
    let aware = name.starts_with("MV_")
        || matches!(
            name,
            "TO_DATE_NANOS" | "TO_DATETIME" | "TO_DT" | "TO_LONG" | "TO_UNSIGNED_LONG" | "TO_UL" | "TO_ULONG" | "TO_DOUBLE" | "TO_DBL" | "TO_STRING" | "TO_STR" | "TO_KEYWORD"
                | "COALESCE" | "IFNULL" | "NVL" | "CASE" | "IIF" | "GREATEST" | "LEAST" | "NULLIF"
        );
    (!aware && args.iter().any(wide)).then(|| args.iter().map(narrow).collect())
}

/// Java `Math.round(double)`: the closest long, ties toward positive infinity.
pub fn java_round(d: f64) -> i64 {
    (d + 0.5).floor() as i64
}

pub fn to_long(v: &Val) -> Val {
    match v {
        Val::Int(i) | Val::Long(i) | Val::Date(i) => Val::Long(*i),
        // ES: a date_nanos as a long is its epoch nanoseconds.
        Val::Nanos(n) => Val::Long(*n),
        // ES `safeDoubleToLong`: Java `Math.round` (half up) in range.
        Val::Double(d) if d.is_finite() && d.abs() < 9.3e18 => Val::Long(java_round(*d)),
        Val::Bool(b) => Val::Long(*b as i64),
        Val::Str(s) => s.trim().parse::<i64>().map(Val::Long).or_else(|_| s.trim().parse::<f64>().map_err(|_| ()).and_then(|d| if d.is_finite() && d.abs() < 9.3e18 { Ok(Val::Long(java_round(d))) } else { Err(()) })).unwrap_or(Val::Null),
        Val::Multi(m) => Val::from_list(m.iter().map(to_long).collect()),
        _ => Val::Null,
    }
}

pub fn to_int(v: &Val) -> Val {
    match to_long(v) {
        Val::Long(i) if i >= i32::MIN as i64 && i <= i32::MAX as i64 => Val::Int(i),
        Val::Multi(m) => Val::from_list(m.iter().map(to_int).collect()),
        _ => Val::Null,
    }
}

pub fn to_double(v: &Val) -> Val {
    match v {
        Val::Int(i) | Val::Long(i) | Val::Date(i) => Val::Double(*i as f64),
        Val::ULong(u) => Val::Double(*u as f64),
        Val::Double(d) => Val::Double(*d),
        Val::Bool(b) => Val::Double(*b as i64 as f64),
        Val::Str(s) => s.trim().parse::<f64>().map(Val::Double).unwrap_or(Val::Null),
        Val::Multi(m) => Val::from_list(m.iter().map(to_double).collect()),
        _ => Val::Null,
    }
}

pub fn to_bool(v: &Val) -> Val {
    match v {
        Val::Bool(b) => Val::Bool(*b),
        Val::Int(i) | Val::Long(i) => Val::Bool(*i != 0),
        Val::Double(d) => Val::Bool(*d != 0.0),
        Val::Str(s) => Val::Bool(s.eq_ignore_ascii_case("true")),
        Val::Multi(m) => Val::from_list(m.iter().map(to_bool).collect()),
        _ => Val::Null,
    }
}

pub fn to_date(v: &Val) -> Val {
    match v {
        Val::Date(d) => Val::Date(*d),
        Val::Nanos(n) => Val::Date(n.div_euclid(super::NANOS_PER_MILLI)),
        Val::Int(i) | Val::Long(i) => Val::Date(*i),
        Val::Double(d) => Val::Date(*d as i64),
        Val::Str(s) => parse_date_str(s).map(Val::Date).unwrap_or(Val::Null),
        Val::Multi(m) => Val::from_list(m.iter().map(to_date).collect()),
        _ => Val::Null,
    }
}

pub fn to_string(v: &Val) -> Val {
    match v {
        Val::Null => Val::Null,
        Val::Multi(m) => Val::Multi(m.iter().map(to_string).collect()),
        Val::Str(s) => Val::Str(s.clone()),
        other => Val::Str(other.to_text()),
    }
}

/// `TO_AGGREGATE_METRIC_DOUBLE`: a number (or multi-value) as min/max/sum/count.
pub fn to_amd(v: &Val) -> Val {
    if let Val::Str(s) = v {
        return if super::types::Amd::parse(s).is_some() { v.clone() } else { Val::Null };
    }
    let nums: Vec<f64> = v.values().into_iter().filter_map(|x| x.as_f64()).collect();
    super::types::Amd::of_values(&nums).map(|a| Val::Str(super::types::amd_json(&a))).unwrap_or(Val::Null)
}

pub fn cast(v: &Val, ty: &str) -> Result<Val> {
    Ok(match ty {
        "aggregate_metric_double" => to_amd(v),
        "long" => to_long(v),
        "int" | "integer" | "short" | "byte" => to_int(v),
        "double" | "float" | "real" | "half_float" | "scaled_float" => to_double(v),
        "bool" | "boolean" => to_bool(v),
        "date" | "datetime" | "timestamp" => to_date(v),
        "date_nanos" => to_date_nanos(v)?,
        "string" | "keyword" | "text" | "varchar" => to_string(v),
        "ip" | "version" => to_string(v),
        "geo_point" | "cartesian_point" | "geo_shape" | "cartesian_shape" => to_spatial(v, ty),
        "unsigned_long" => to_unsigned_long(v)?,
        "time_duration" | "date_period" => match v {
            Val::Str(_) => call(if ty == "time_duration" { "TO_TIMEDURATION" } else { "TO_DATEPERIOD" }, &[v.clone()], &FnCtx::default())?,
            _ => v.clone(),
        },
        // ES SQL TIME (`SqlDataTypeConverter.conversionToTime`): datetime /
        // date in its zone, numbers as `millis % day`, booleans as 1 / 0,
        // strings ISO local times; its text otherwise.
        "time" => match v {
            Val::Null => Val::Null,
            Val::Date(ms) => Val::Str(super::sqltime::time_of_datetime(*ms)),
            Val::Int(i) | Val::Long(i) => Val::Str(super::sqltime::time_of_long(*i)),
            Val::ULong(u) => Val::Str(super::sqltime::time_of_long(*u as i64)),
            Val::Double(d) => Val::Str(super::sqltime::time_of_long(d.round() as i64)),
            Val::Bool(b) => Val::Str(super::sqltime::time_of_long(*b as i64)),
            Val::Str(s) => Val::Str(super::sqltime::time_of_string(s)?),
            Val::Multi(m) => Val::from_list(m.iter().map(|x| cast(x, ty)).collect::<Result<Vec<_>>>()?),
            _ => Val::Null,
        },
        t if t == super::exec::TIME_MILLIS => match v {
            Val::Str(s) => super::sqltime::time_millis(s).map(Val::Long).unwrap_or(Val::Null),
            Val::Multi(m) => Val::from_list(m.iter().map(|x| cast(x, ty)).collect::<Result<Vec<_>>>()?),
            _ => Val::Null,
        },
        t if t == super::exec::PERIOD_VALUE || t == super::exec::DURATION_VALUE => match v {
            Val::Str(s) => match super::sqltime::parse_iso(s) {
                Some(super::sqltime::Amount::Period(y, m)) => Val::Period(y * 12 + m, 0),
                Some(super::sqltime::Amount::Millis(ms)) => Val::Dur(ms),
                None => Val::Null,
            },
            other => other.clone(),
        },
        t if t == super::exec::INTERVAL_TEXT => match v {
            Val::Str(s) => super::sqltime::sql_text(s).map(Val::Str).unwrap_or(Val::Null),
            _ => Val::Null,
        },
        // An interval to its own type; an interval literal (its value).
        t if super::sqltime::is_interval(t) || t.starts_with(super::sqltime::LITERAL_PREFIX) => v.clone(),
        other => return Err(EsError::new(400, "parsing_exception", format!("Unsupported conversion to type [{other}]"))),
    })
}

/// Return type of a cast target.
pub fn cast_type(ty: &str) -> &'static str {
    match ty {
        "long" => "long",
        "int" | "integer" | "short" | "byte" => "integer",
        "double" | "float" | "real" | "half_float" | "scaled_float" => "double",
        "bool" | "boolean" => "boolean",
        "date" | "datetime" | "timestamp" => "date",
        "date_nanos" => "date_nanos",
        "ip" => "ip",
        "version" => "version",
        "geo_point" => "geo_point",
        "cartesian_point" => "cartesian_point",
        "geo_shape" => "geo_shape",
        "cartesian_shape" => "cartesian_shape",
        "unsigned_long" => "unsigned_long",
        "text" => "text",
        "time_duration" => "time_duration",
        "date_period" => "date_period",
        "aggregate_metric_double" => "aggregate_metric_double",
        "time" => "time",
        t if t == super::exec::TIME_MILLIS => "long",
        t if t == super::exec::PERIOD_VALUE => "date_period",
        t if t == super::exec::DURATION_VALUE => "time_duration",
        t if t.starts_with(super::sqltime::LITERAL_PREFIX) => {
            let name = &t[super::sqltime::LITERAL_PREFIX.len()..];
            super::sqltime::INTERVAL_TYPES.iter().find(|n| **n == name).copied().unwrap_or("keyword")
        }
        t => super::sqltime::INTERVAL_TYPES.iter().find(|n| **n == t).copied().unwrap_or("keyword"),
    }
}

fn spatial_target(name: &str) -> &'static str {
    match name {
        "TO_GEOPOINT" => "geo_point",
        "TO_CARTESIANPOINT" => "cartesian_point",
        "TO_GEOSHAPE" => "geo_shape",
        _ => "cartesian_shape",
    }
}

/// `TO_GEOPOINT` & co: WKT strings (or spatial values) to the target type.
fn to_spatial(v: &Val, ty: &str) -> Val {
    match v {
        Val::Str(s) => super::spatial::parse_wkt(s, ty).map(Val::Str).unwrap_or(Val::Null),
        Val::Multi(m) => Val::from_list(m.iter().map(|x| to_spatial(x, ty)).collect()),
        _ => Val::Null,
    }
}

fn s_arg(v: &Val) -> Option<String> {
    match v {
        Val::Null | Val::Multi(_) => None,
        Val::Str(s) => Some(s.clone()),
        other => Some(other.to_text()),
    }
}

fn f_arg(v: &Val) -> Option<f64> {
    match v {
        Val::Multi(_) => None,
        other => other.as_f64(),
    }
}

fn i_arg(v: &Val) -> Option<i64> {
    match v {
        Val::Multi(_) => None,
        other => other.as_i64(),
    }
}

pub fn dbl(d: f64) -> Val {
    if d.is_finite() { Val::Double(d) } else { Val::Null }
}

fn math1(args: &[Val], f: impl Fn(f64) -> f64) -> Val {
    match args.first().and_then(f_arg) {
        Some(x) => dbl(f(x)),
        None => Val::Null,
    }
}

pub fn day_name(ms: i64) -> Option<String> {
    let d = chrono::Utc.timestamp_millis_opt(local_ms(ms)).single()?;
    Some(d.format("%A").to_string())
}

/// Translate a Java date pattern into a [`crate::date::DateFormatter`].
fn java_fmt(p: &str) -> Option<crate::date::DateFormatter> {
    crate::date::DateFormatter::new(p).ok()
}

fn date_part(part: &str, ms: i64) -> Option<i64> {
    let ms = local_ms(ms);
    let d = chrono::Utc.timestamp_millis_opt(ms).single()?.naive_utc();
    Some(match part.to_ascii_lowercase().as_str() {
        "year" | "years" | "year_of_era" => d.year() as i64,
        "month_of_year" | "month" | "months" => d.month() as i64,
        "day_of_month" | "day" | "days" => d.day() as i64,
        "day_of_week" => d.weekday().number_from_monday() as i64,
        "day_of_year" => d.ordinal() as i64,
        "hour_of_day" | "hour" | "hours" => d.hour() as i64,
        "minute_of_hour" | "minute" | "minutes" => d.minute() as i64,
        "second_of_minute" | "second" | "seconds" => d.second() as i64,
        "milli_of_second" => (d.nanosecond() / 1_000_000) as i64,
        "minute_of_day" => (d.hour() * 60 + d.minute()) as i64,
        "second_of_day" => d.num_seconds_from_midnight() as i64,
        "milli_of_day" => d.num_seconds_from_midnight() as i64 * 1000 + (d.nanosecond() / 1_000_000) as i64,
        "epoch_day" => ms.div_euclid(86_400_000),
        "aligned_week_of_year" => ((d.ordinal() - 1) / 7 + 1) as i64,
        "aligned_week_of_month" => ((d.day() - 1) / 7 + 1) as i64,
        "quarter_of_year" | "quarter" => (d.month0() / 3 + 1) as i64,
        "proleptic_month" => d.year() as i64 * 12 + d.month0() as i64,
        "era" => 1,
        "ampm_of_day" => (d.hour() >= 12) as i64,
        "hour_of_ampm" => (d.hour() % 12) as i64,
        "clock_hour_of_day" => if d.hour() == 0 { 24 } else { d.hour() as i64 },
        "week_of_week_based_year" | "week" => d.iso_week().week() as i64,
        _ => return None,
    })
}

fn date_diff(unit: &str, a: i64, b: i64) -> Option<i64> {
    let ms = b - a;
    Some(match unit.to_ascii_lowercase().as_str() {
        "millisecond" | "milliseconds" | "ms" => ms,
        "second" | "seconds" | "ss" | "s" => ms / 1000,
        "minute" | "minutes" | "mi" | "n" => ms / 60_000,
        "hour" | "hours" | "hh" => ms / 3_600_000,
        "day" | "days" | "dd" | "d" | "dayofyear" | "dy" | "y" => ms / 86_400_000,
        "week" | "weeks" | "wk" | "ww" => ms / (7 * 86_400_000),
        "month" | "months" | "mm" | "m" | "quarter" | "quarters" | "qq" | "q" | "year" | "years" | "yyyy" | "yy" => {
            let da = chrono::Utc.timestamp_millis_opt(a).single()?.naive_utc();
            let db = chrono::Utc.timestamp_millis_opt(b).single()?.naive_utc();
            let mut months = (db.year() as i64 * 12 + db.month0() as i64) - (da.year() as i64 * 12 + da.month0() as i64);
            // Incomplete months don't count.
            let rest_a = (da.day(), da.num_seconds_from_midnight(), da.nanosecond());
            let rest_b = (db.day(), db.num_seconds_from_midnight(), db.nanosecond());
            if months > 0 && rest_b < rest_a {
                months -= 1;
            } else if months < 0 && rest_b > rest_a {
                months += 1;
            }
            match unit.to_ascii_lowercase().chars().next().unwrap_or('m') {
                'q' => months / 3,
                'y' => months / 12,
                _ => months,
            }
        }
        "nanosecond" | "nanoseconds" | "ns" => ms * 1_000_000,
        "microsecond" | "microseconds" | "mcs" => ms * 1000,
        _ => return None,
    })
}

fn mv_list(v: &Val) -> Vec<Val> {
    match v {
        Val::Null => vec![],
        Val::Multi(m) => m.clone(),
        o => vec![o.clone()],
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

pub fn percentile(mut v: Vec<f64>, p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p = p.clamp(0.0, 100.0) / 100.0;
    let rank = p * (v.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    Some(v[lo] + (v[hi] - v[lo]) * (rank - lo as f64))
}

pub fn mad(v: Vec<f64>) -> Option<f64> {
    let m = median(v.clone())?;
    median(v.iter().map(|x| (x - m).abs()).collect())
}

fn keep_int_type(sample: &Val, d: f64) -> Val {
    match sample {
        Val::Int(_) => Val::Int(d as i64),
        Val::Long(_) => Val::Long(d as i64),
        _ => Val::Double(d),
    }
}

fn parse_point(s: &str) -> Option<(f64, f64)> {
    let t = s.trim();
    let inner = t.strip_prefix("POINT")?.trim().strip_prefix('(')?.strip_suffix(')')?;
    let mut it = inner.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

/// A "human" date rounding of ES's `Bucket` (calendar months, weeks, days, or fixed millis).
#[derive(Clone, Copy, Debug, PartialEq)]
enum DateRounding {
    Year,
    Month,
    Week,
    Day,
    Fixed(i64),
}

fn date_round(ms: i64, r: &DateRounding) -> Option<i64> {
    match r {
        DateRounding::Year => date_trunc(ms, &Val::Period(12, 0)),
        DateRounding::Month => date_trunc(ms, &Val::Period(1, 0)),
        DateRounding::Week => date_trunc(ms, &Val::Period(0, 7)),
        DateRounding::Day => date_trunc(ms, &Val::Period(0, 1)),
        DateRounding::Fixed(step) => Some(ms - ms.rem_euclid(*step)),
    }
}

fn date_next(ms: i64, r: &DateRounding) -> i64 {
    match r {
        DateRounding::Year => add_period(ms, 12, 0),
        DateRounding::Month => add_period(ms, 1, 0),
        DateRounding::Week => ms + 7 * 86_400_000,
        DateRounding::Day => ms + 86_400_000,
        DateRounding::Fixed(step) => ms + step,
    }
}

/// `Bucket.DateRoundingPicker`: the finest human rounding that covers
/// `[from, to)` in at most `buckets` buckets (year when none does).
fn pick_date_rounding(buckets: i64, from: i64, to: i64) -> DateRounding {
    const H: i64 = 3_600_000;
    const M: i64 = 60_000;
    let candidates = [
        DateRounding::Month,
        DateRounding::Week,
        DateRounding::Day,
        DateRounding::Fixed(12 * H),
        DateRounding::Fixed(3 * H),
        DateRounding::Fixed(H),
        DateRounding::Fixed(30 * M),
        DateRounding::Fixed(10 * M),
        DateRounding::Fixed(5 * M),
        DateRounding::Fixed(M),
        DateRounding::Fixed(30_000),
        DateRounding::Fixed(10_000),
        DateRounding::Fixed(1000),
        DateRounding::Fixed(100),
        DateRounding::Fixed(50),
        DateRounding::Fixed(10),
        DateRounding::Fixed(1),
    ];
    let ok = |r: &DateRounding| -> bool {
        let Some(mut b) = date_round(from, r) else { return false };
        for _ in 0..buckets {
            b = date_next(b, r);
            if b >= to {
                return true;
            }
        }
        false
    };
    let mut prev = DateRounding::Year;
    for r in candidates {
        if ok(&r) {
            prev = r;
        } else {
            return prev;
        }
    }
    prev
}

/// A temporal amount written as a string (`"1 month"`, `"3 hours"`).
pub fn parse_temporal(s: &str) -> Option<Val> {
    let mut it = s.split_whitespace();
    let k = it.next()?.parse::<i64>().ok()?;
    let unit = super::parse::time_unit(it.next()?)?;
    if it.next().is_some() {
        return None;
    }
    Some(match unit {
        Val::Dur(ms) => Val::Dur(ms * k),
        Val::Period(m, d) => Val::Period(m * k, d * k),
        other => other,
    })
}

/// `_meta.bucket` of a `BUCKET` grouping column (`{"interval", "unit"}`),
/// from its constant arguments (the first, the field, is unused).
pub fn bucket_meta(args: &[Val], date: bool) -> Option<serde_json::Value> {
    use serde_json::json;
    let dur_meta = |ms: i64| -> serde_json::Value {
        for (u, n) in [("day", 86_400_000i64), ("hour", 3_600_000), ("minute", 60_000), ("second", 1000)] {
            if ms % n == 0 {
                return json!({"interval": ms / n, "unit": u});
            }
        }
        json!({"interval": ms, "unit": "millisecond"})
    };
    let period_meta = |m: i64, d: i64| -> serde_json::Value {
        if m != 0 && d == 0 {
            if m % 12 == 0 {
                json!({"interval": m / 12, "unit": "year"})
            } else if m % 3 == 0 && m != 0 && false {
                json!({"interval": m / 3, "unit": "quarter"})
            } else {
                json!({"interval": m, "unit": "month"})
            }
        } else if m == 0 && d % 7 == 0 && d != 0 {
            json!({"interval": d / 7, "unit": "week"})
        } else {
            json!({"interval": d, "unit": "day"})
        }
    };
    let span_meta = |v: &Val| -> Option<serde_json::Value> {
        let parsed = match v {
            Val::Str(s) if date => parse_temporal(s),
            _ => None,
        };
        match parsed.as_ref().unwrap_or(v) {
            Val::Dur(ms) => Some(dur_meta(*ms)),
            Val::Period(m, d) => Some(period_meta(*m, *d)),
            o if !date => o.as_f64().map(|w| json!({"interval": w})),
            _ => None,
        }
    };
    match args.len() {
        2 => span_meta(&args[1]),
        4 if date => {
            let from = match &args[2] {
                Val::Str(s) => parse_date_str(s),
                o => o.as_i64(),
            }?;
            let to = match &args[3] {
                Val::Str(s) => parse_date_str(s),
                o => o.as_i64(),
            }?;
            Some(match pick_date_rounding(args[1].as_i64().unwrap_or(1).max(1), from, to) {
                DateRounding::Year => json!({"interval": 1, "unit": "year"}),
                DateRounding::Month => json!({"interval": 1, "unit": "month"}),
                DateRounding::Week => json!({"interval": 1, "unit": "week"}),
                DateRounding::Day => json!({"interval": 1, "unit": "day"}),
                DateRounding::Fixed(ms) => dur_meta(ms),
            })
        }
        4 => {
            let (n, from, to) = (args[1].as_f64()?.max(1.0), args[2].as_f64()?, args[3].as_f64()?);
            let raw = ((to - from) / n).max(f64::MIN_POSITIVE);
            let mag = 10f64.powf(raw.log10().floor());
            let w = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|c| *c >= raw).unwrap_or(10.0 * mag);
            Some(json!({"interval": w}))
        }
        _ => None,
    }
}

/// BUCKET(field, span) / BUCKET(field, buckets, from, to).
fn bucket(args: &[Val]) -> Result<Val> {
    let v = &args[0];
    if v.is_null() {
        return Ok(Val::Null);
    }
    if args.len() == 2 {
        // A date span may be given as a string (`"1 month"`).
        let parsed = match (v, &args[1]) {
            (Val::Date(_), Val::Str(s)) => parse_temporal(s),
            _ => None,
        };
        let span = parsed.as_ref().unwrap_or(&args[1]);
        return Ok(match (v, span) {
            (Val::Date(d), Val::Dur(_) | Val::Period(..)) => date_trunc(*d, span).map(Val::Date).unwrap_or(Val::Null),
            (_, s) if v.as_f64().is_some() && s.as_f64().is_some() => {
                let x = v.as_f64().unwrap();
                let w = s.as_f64().unwrap();
                if w <= 0.0 {
                    return Ok(Val::Null);
                }
                Val::Double((x / w).floor() * w)
            }
            _ => Val::Null,
        });
    }
    if args.len() != 4 {
        return Err(bad_arity("BUCKET"));
    }
    let n = args[1].as_f64().unwrap_or(1.0).max(1.0);
    match v {
        Val::Date(d) => {
            let _ = n;
            let from = match &args[2] {
                Val::Str(s) => parse_date_str(s),
                o => o.as_i64(),
            };
            let to = match &args[3] {
                Val::Str(s) => parse_date_str(s),
                o => o.as_i64(),
            };
            let (Some(from), Some(to)) = (from, to) else { return Ok(Val::Null) };
            let buckets = args[1].as_i64().unwrap_or(1).max(1);
            Ok(date_round(*d, &pick_date_rounding(buckets, from, to)).map(Val::Date).unwrap_or(Val::Null))
        }
        _ => {
            let (Some(x), Some(from), Some(to)) = (v.as_f64(), args[2].as_f64(), args[3].as_f64()) else { return Ok(Val::Null) };
            let raw = ((to - from) / n).max(f64::MIN_POSITIVE);
            // Nice rounding: 1, 2, 2.5, 5 x 10^k.
            let mag = 10f64.powf(raw.log10().floor());
            let w = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|c| *c >= raw).unwrap_or(10.0 * mag);
            Ok(Val::Double((x / w).floor() * w))
        }
    }
}

#[derive(Default)]
pub struct FnCtx {
    pub(crate) regex_cache:std::cell::RefCell<super::regex_memory::Cache>,
    /// The query text (warning locations and sources).
    pub query: String,
    pub now: i64,
    /// The query's `time_zone`.
    pub tz: crate::date::Tz,
    /// `locale` of the request (`it-IT`), for month/day names.
    pub locale: Option<String>,
    /// Source (`Line l:c: ` prefix, text) of each `CASE` condition, keyed by
    /// the offset of the `CASE` call.
    pub case_conds: std::cell::RefCell<HashMap<usize, Vec<(String, String)>>>,
    /// Response warnings raised while evaluating.
    pub warnings: std::cell::RefCell<Warnings>,
}

/// Evaluation warnings, shaped like the compute engine's `Warnings`: per
/// source location a header (`... Only first 20 failures recorded.`) then
/// up to 20 exception messages.
#[derive(Default)]
pub struct Warnings {
    pub list: Vec<String>,
    counts: HashMap<String, usize>,
}

impl Warnings {
    pub const MAX: usize = 20;

    /// `location` is `Line l:c: `; `first` the header body after it.
    pub fn register(&mut self, location: &str, first: &str, exception: &str) {
        let header = format!("{location}{first}. Only first {} failures recorded.", Self::MAX);
        let n = self.counts.entry(header.clone()).or_insert(0);
        if *n >= Self::MAX {
            return;
        }
        *n += 1;
        for w in [header, format!("{location}{exception}")] {
            if !self.list.contains(&w) {
                self.list.push(w);
            }
        }
    }
}

/// Month and day names (full, then abbreviated) of a locale, `None` for English.
fn locale_names(locale: &str) -> Option<([&'static str; 12], [&'static str; 7])> {
    let lang = locale.split(['-', '_']).next().unwrap_or("").to_ascii_lowercase();
    Some(match lang.as_str() {
        "it" => (
            ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
            ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"],
        ),
        "fr" => (
            ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
            ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
        ),
        "de" => (
            ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
            ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
        ),
        "es" => (
            ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
            ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        ),
        "pt" => (
            ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
            ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"],
        ),
        "nl" => (
            ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"],
            ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"],
        ),
        _ => return None,
    })
}

/// Localize the English month/day names a date pattern printed.
fn localize_date(s: String, locale: Option<&str>) -> String {
    let Some((months, days)) = locale.and_then(locale_names) else { return s };
    const EN_MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const EN_DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    let mut out = s;
    for (en, loc) in EN_MONTHS.iter().zip(months.iter()) {
        out = out.replace(en, loc);
    }
    for (en, loc) in EN_DAYS.iter().zip(days.iter()) {
        out = out.replace(en, loc);
    }
    out
}

/// Evaluate a scalar function over evaluated arguments.
/// The trailing argument [`super::exec`] adds to a spatial relation on geo
/// types.
pub const GEO_RELATION: &str = "\u{0}geo";
/// Internal function of a bound ordering comparison (and BETWEEN) over
/// `version` / `ip` values: the operand's sortable encoding
/// ([`typed_key`]), so the comparison orders by version precedence / by the
/// 16-byte address as ES does, not by rendered text.
pub const TYPED_KEY_FN: &str = "\u{0}TYPED_KEY";

/// A `version` or `ip` value (`ty`) as a string whose byte order is the
/// type's order: versions by their sortable encoding, addresses as the
/// hex of their IPv6(-mapped) 16 bytes. Other values pass unchanged.
pub fn typed_key(v: &Val, ty: &str) -> Val {
    match v {
        Val::Multi(m) => Val::Multi(m.iter().map(|x| typed_key(x, ty)).collect()),
        Val::Str(s) if ty == "version" => Val::Str(rustic_core::fieldtypes::version::encode(s)),
        Val::Str(s) if ty == "ip" => match s.trim().parse::<std::net::IpAddr>() {
            Ok(ip) => {
                let bytes = match ip {
                    std::net::IpAddr::V4(a) => a.to_ipv6_mapped().octets(),
                    std::net::IpAddr::V6(a) => a.octets(),
                };
                Val::Str(bytes.iter().map(|b| format!("{b:02x}")).collect())
            }
            Err(_) => v.clone(),
        },
        o => o.clone(),
    }
}
/// Marker argument of a bound `ST_DISTANCE` over cartesian points: the
/// Pythagorean distance in coordinate units (ES `StDistance` on cartesian
/// types), not geographic meters.
pub const CARTESIAN_DISTANCE: &str = "\u{0}cartesian";

/// Internal: dense-vector arithmetic `(op, a, b)` (ES `Add` / `Sub` / `Mul`
/// / `Div` on `dense_vector`): coordinate by coordinate for two vectors of
/// one dimension, or each coordinate with a scalar; `null` otherwise.
pub const VECTOR_ARITH: &str = "\u{0}VECTOR_ARITH";

fn vector_arith(op: &str, a: &Val, b: &Val) -> Val {
    let coords = |v: &Val| -> Option<Vec<f64>> {
        match v {
            Val::Multi(m) => m.iter().map(Val::as_f64).collect(),
            Val::Null => None,
            other => other.as_f64().map(|x| vec![x]),
        }
    };
    let (Some(x), Some(y)) = (coords(a), coords(b)) else { return Val::Null };
    let f = |p: f64, q: f64| -> f64 {
        match op {
            "+" => p + q,
            "-" => p - q,
            "*" => p * q,
            _ => p / q,
        }
    };
    let scalar_a = !matches!(a, Val::Multi(_));
    let scalar_b = !matches!(b, Val::Multi(_));
    let out: Vec<f64> = match (scalar_a, scalar_b) {
        (false, false) if x.len() == y.len() => x.iter().zip(&y).map(|(p, q)| f(*p, *q)).collect(),
        (false, true) => x.iter().map(|p| f(*p, y[0])).collect(),
        (true, false) => y.iter().map(|q| f(x[0], *q)).collect(),
        (true, true) => vec![f(x[0], y[0])],
        _ => return Val::Null,
    };
    if out.iter().any(|v| !v.is_finite()) {
        return Val::Null;
    }
    Val::Multi(out.into_iter().map(Val::Double).collect())
}

/// Internal: SQL `DATE_DIFF(unit, start, end)` (ES SQL `DateDiff`): the
/// number of unit boundaries crossed between the two datetimes, not the
/// elapsed time divided by the unit (12:59 to 13:00 is one hour).
pub const SQL_DATE_DIFF: &str = "\u{0}SQL_DATE_DIFF";
/// Internal: a MySQL `DATE_FORMAT` pattern computed at run time, as Java.
pub const MYSQL_PATTERN_FN: &str = "\u{0}MYSQL_PATTERN";

/// A MySQL `DATE_FORMAT` pattern (ES SQL `DateFormat`) as the equivalent
/// Java `DateTimeFormatter` pattern (literal text quoted).
pub fn mysql_to_java_pattern(p: &str) -> String {
    let mut out = String::new();
    let mut lit = String::new();
    let flush = |lit: &mut String, out: &mut String| {
        if !lit.is_empty() {
            out.push('\'');
            out.push_str(&lit.replace('\'', "''"));
            out.push('\'');
            lit.clear();
        }
    };
    let mut it = p.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            lit.push(c);
            continue;
        }
        let Some(spec) = it.next() else { lit.push('%'); break };
        let java = match spec {
            'a' => "EEE",
            'b' => "MMM",
            'c' => "M",
            'd' => "dd",
            'e' => "d",
            'f' => "SSSSSS",
            'H' => "HH",
            'h' | 'I' => "hh",
            'i' => "mm",
            'j' => "DDD",
            'k' => "H",
            'l' => "h",
            'M' => "MMMM",
            'm' => "MM",
            'p' => "a",
            'r' => "hh:mm:ss a",
            'S' | 's' => "ss",
            'T' => "HH:mm:ss",
            'W' => "EEEE",
            'Y' => "yyyy",
            'y' => "yy",
            other => {
                lit.push(other);
                continue;
            }
        };
        flush(&mut lit, &mut out);
        out.push_str(java);
    }
    flush(&mut lit, &mut out);
    out
}

/// [`SQL_DATE_DIFF`]: boundary counts in UTC.
fn sql_date_diff(unit: &str, a: i64, b: i64) -> Option<i64> {
    let span = |u: i64| Some(b.div_euclid(u) - a.div_euclid(u));
    let ym = |ms: i64| -> Option<(i64, i64)> {
        let d = chrono::Utc.timestamp_millis_opt(ms).single()?.naive_utc();
        Some((d.year() as i64, d.month0() as i64))
    };
    match unit.to_ascii_lowercase().as_str() {
        "millisecond" | "milliseconds" | "ms" => Some(b - a),
        "second" | "seconds" | "ss" | "s" => span(1000),
        "minute" | "minutes" | "mi" | "n" => span(60_000),
        "hour" | "hours" | "hh" => span(3_600_000),
        "day" | "days" | "dd" | "d" | "dayofyear" | "dy" | "y" => span(86_400_000),
        "month" | "months" | "mm" | "m" => {
            let ((ya, ma), (yb, mb)) = (ym(a)?, ym(b)?);
            Some((yb * 12 + mb) - (ya * 12 + ma))
        }
        "quarter" | "quarters" | "qq" | "q" => {
            let ((ya, ma), (yb, mb)) = (ym(a)?, ym(b)?);
            Some((yb * 4 + mb / 3) - (ya * 4 + ma / 3))
        }
        "year" | "years" | "yyyy" | "yy" => Some(ym(b)?.0 - ym(a)?.0),
        _ => date_diff(unit, a, b),
    }
}

/// The canonical name of an ES SQL function alias (`DAYOFWEEK` is
/// `DAY_OF_WEEK`, `CEILING` is `CEIL`, ...); other names unchanged.
pub fn sql_alias(name: &str) -> &str {
    match name {
        "CEILING" => "CEIL",
        "RAND" => "RANDOM",
        "TRUNC" => "TRUNCATE",
        "CHARACTER_LENGTH" => "CHAR_LENGTH",
        "DATEDIFF" | "TIMESTAMPDIFF" | "TIMESTAMP_DIFF" => "DATE_DIFF",
        "DATETRUNC" => "DATE_TRUNC",
        "DATEADD" | "TIMESTAMPADD" | "TIMESTAMP_ADD" => "DATE_ADD",
        "DATEPART" => "DATE_PART",
        "DAYOFWEEK" | "DOW" => "DAY_OF_WEEK",
        "ISODAYOFWEEK" | "ISODOW" | "IDOW" => "ISO_DAY_OF_WEEK",
        "DAYOFYEAR" | "DOY" => "DAY_OF_YEAR",
        "WEEK" => "WEEK_OF_YEAR",
        "ISOWEEK" | "ISOWEEKOFYEAR" | "IW" | "IWOY" => "ISO_WEEK_OF_YEAR",
        "DAYNAME" => "DAY_NAME",
        "MONTHNAME" => "MONTH_NAME",
        "DAYOFMONTH" | "DOM" => "DAY_OF_MONTH",
        "MONTH_OF_YEAR" => "MONTH",
        "HOUR_OF_DAY" => "HOUR",
        "MINUTE_OF_HOUR" => "MINUTE",
        "SECOND_OF_MINUTE" => "SECOND",
        other => other,
    }
}

/// `java.util.Random(seed).nextDouble()` (ES SQL `RANDOM(seed)`).
pub fn java_random_double(seed: i64) -> f64 {
    const MULT: u64 = 0x5DEECE66D;
    const MASK: u64 = (1 << 48) - 1;
    let mut s = (seed as u64 ^ MULT) & MASK;
    let mut next = |bits: u32| {
        s = (s.wrapping_mul(MULT).wrapping_add(0xB)) & MASK;
        s >> (48 - bits)
    };
    let hi = next(26);
    let lo = next(27);
    ((hi << 27) + lo) as f64 * (1.0f64 / (1u64 << 53) as f64)
}

/// ES SQL `WEEK_OF_YEAR`: weeks start on Sunday and week 1 holds
/// January 1st (`WeekFields.of(SUNDAY, 1).weekOfYear()`).
fn us_week(d: &chrono::NaiveDateTime) -> i64 {
    let jan1 = chrono::NaiveDate::from_ymd_opt(d.year(), 1, 1).map(|j| j.weekday().num_days_from_sunday() as i64).unwrap_or(0);
    (d.ordinal() as i64 - 1 + jan1) / 7 + 1
}

/// The ES SQL-only scalars (ES SQL's `DateTimeFunction`s, `MathFunction`s
/// and `StringFunction`s with no ES|QL counterpart); `None`: not one.
fn sql_scalar(name: &str, args: &[Val]) -> Option<Val> {
    let a0 = args.first().cloned().unwrap_or(Val::Null);
    let date = |v: &Val| match v {
        Val::Date(d) => Some(*d),
        Val::Nanos(n) => Some(n.div_euclid(super::NANOS_PER_MILLI)),
        _ => None,
    };
    let local = |v: &Val| date(v).and_then(|ms| chrono::Utc.timestamp_millis_opt(local_ms(ms)).single()).map(|d| d.naive_utc());
    let int = |v: Option<i64>| v.map(Val::Int).unwrap_or(Val::Null);
    Some(match name {
        "SIGN" => match &a0 {
            Val::Int(i) => Val::Int(i.signum()),
            Val::Long(l) => Val::Long(l.signum()),
            v => match f_arg(v) {
                Some(x) => Val::Double(if x == 0.0 { 0.0 } else { x.signum() }),
                None => Val::Null,
            },
        },
        "COT" => math1(args, |x| 1.0 / x.tan()),
        "EXPM1" => math1(args, f64::exp_m1),
        "DEGREES" => math1(args, f64::to_degrees),
        "RADIANS" => math1(args, f64::to_radians),
        "RANDOM" => match i_arg(&a0) {
            Some(seed) => Val::Double(java_random_double(seed)),
            None => Val::Double(java_random_double(crate::date::now_millis() ^ std::process::id() as i64)),
        },
        "CHAR" => match i_arg(&a0) {
            Some(c @ 0..=255) => Val::Str(char::from(c as u8).to_string()),
            _ => Val::Null,
        },
        // INSERT(source, start, length, replacement): ES's
        // `StringBuilder.replace(start - 1, start - 1 + length, ..)`.
        "INSERT" => {
            let (Some(src), Some(start), Some(len), Some(rep)) = (s_arg(&a0), args.get(1).and_then(i_arg), args.get(2).and_then(i_arg), args.get(3).and_then(s_arg)) else { return Some(Val::Null) };
            let chars: Vec<char> = src.chars().collect();
            let from = (start - 1).max(0);
            if start - 1 > chars.len() as i64 {
                Val::Str(src)
            } else if len < 0 {
                Val::Null
            } else {
                let from = from as usize;
                let to = (from as i64 + len).min(chars.len() as i64) as usize;
                let mut out: String = chars[..from].iter().collect();
                out.push_str(&rep);
                out.extend(&chars[to.max(from)..]);
                Val::Str(out)
            }
        }
        "OCTET_LENGTH" => match s_arg(&a0) {
            Some(s) => Val::Int(s.len() as i64),
            None => Val::Null,
        },
        // Sunday is 1 (`NonIsoDateTimeExtractor.DAY_OF_WEEK`).
        "DAY_OF_WEEK" => int(local(&a0).map(|d| d.weekday().num_days_from_sunday() as i64 + 1)),
        "ISO_DAY_OF_WEEK" => int(local(&a0).map(|d| d.weekday().number_from_monday() as i64)),
        "DAY_OF_YEAR" => int(local(&a0).map(|d| d.ordinal() as i64)),
        // Weeks start on Sunday; week 1 holds January 1st
        // (`WeekFields.of(SUNDAY, 1).weekOfYear()`).
        "WEEK_OF_YEAR" => int(local(&a0).map(|d| us_week(&d))),
        "ISO_WEEK_OF_YEAR" => int(local(&a0).map(|d| d.iso_week().week() as i64)),
        "QUARTER" => int(local(&a0).map(|d| d.month0() as i64 / 3 + 1)),
        "MINUTE_OF_DAY" => int(local(&a0).map(|d| (d.hour() * 60 + d.minute()) as i64)),
        // DATE_ADD(unit, count, datetime)
        "DATE_ADD" => {
            let (Some(unit), Some(n), Some(ms)) = (s_arg(&a0), args.get(1).and_then(i_arg), args.get(2).and_then(date)) else { return Some(Val::Null) };
            let fixed = |step: i64| n.checked_mul(step).and_then(|d| ms.checked_add(d)).map(Val::Date).unwrap_or(Val::Null);
            match unit.to_ascii_lowercase().as_str() {
                "year" | "years" | "yy" | "yyyy" => Val::Date(add_period(ms, n * 12, 0)),
                "quarter" | "quarters" | "qq" | "q" => Val::Date(add_period(ms, n * 3, 0)),
                "month" | "months" | "mm" | "m" => Val::Date(add_period(ms, n, 0)),
                "dayofyear" | "dy" | "y" | "day" | "days" | "dd" | "d" | "weekday" | "weekdays" | "dw" => Val::Date(add_period(ms, 0, n)),
                "week" | "weeks" | "wk" | "ww" => Val::Date(add_period(ms, 0, n * 7)),
                "hour" | "hours" | "hh" => fixed(3_600_000),
                "minute" | "minutes" | "mi" | "n" => fixed(60_000),
                "second" | "seconds" | "ss" | "s" => fixed(1000),
                "millisecond" | "milliseconds" | "ms" => fixed(1),
                "microsecond" | "microseconds" | "mcs" => Val::Date(ms + n / 1000),
                "nanosecond" | "nanoseconds" | "ns" => Val::Date(ms + n / 1_000_000),
                _ => Val::Null,
            }
        }
        // DATE_PART(unit, datetime)
        "DATE_PART" => {
            let (Some(unit), Some(d)) = (s_arg(&a0), args.get(1).and_then(local)) else { return Some(Val::Null) };
            let ms = args.get(1).and_then(date).unwrap_or(0);
            int(match unit.to_ascii_lowercase().as_str() {
                "year" | "years" | "yy" | "yyyy" => Some(d.year() as i64),
                "quarter" | "quarters" | "qq" | "q" => Some(d.month0() as i64 / 3 + 1),
                "month" | "months" | "mm" | "m" => Some(d.month() as i64),
                "dayofyear" | "dy" | "y" => Some(d.ordinal() as i64),
                "day" | "days" | "dd" | "d" => Some(d.day() as i64),
                // ES 9.5.5: DATE_PART week is WEEK_OF_YEAR's (Sunday-start) week.
                "week" | "weeks" | "wk" | "ww" => Some(us_week(&d)),
                "weekday" | "weekdays" | "dw" => Some(d.weekday().num_days_from_sunday() as i64 + 1),
                "hour" | "hours" | "hh" => Some(d.hour() as i64),
                "minute" | "minutes" | "mi" | "n" => Some(d.minute() as i64),
                "second" | "seconds" | "ss" | "s" => Some(d.second() as i64),
                "millisecond" | "milliseconds" | "ms" => Some((d.nanosecond() / 1_000_000) as i64),
                "microsecond" | "microseconds" | "mcs" => Some((d.nanosecond() / 1000) as i64),
                "nanosecond" | "nanoseconds" | "ns" => Some(d.nanosecond() as i64),
                "tzoffset" | "tz" => Some(super::zone().offset_at(ms) as i64 / 60),
                _ => None,
            })
        }
        _ => return None,
    })
}

/// ES SQL's GREATEST / LEAST (nulls skipped), REPLACE (literal), CONCAT
/// (NULL read as ''), LENGTH (trailing blanks excluded): internal names the
/// SQL binder resolves those functions to.
/// PromQL's label regex match: RE2 (Rust `regex`), fully anchored.
pub const PROMQL_RE: &str = "\u{0}PROMQL_RE";

pub const SQL_GREATEST: &str = "\u{0}SQL_GREATEST";
pub const SQL_LEAST: &str = "\u{0}SQL_LEAST";
pub const SQL_REPLACE: &str = "\u{0}SQL_REPLACE";
pub const SQL_CONCAT: &str = "\u{0}SQL_CONCAT";
pub const SQL_LENGTH: &str = "\u{0}SQL_LENGTH";

fn sql_string_fn(name: &str, args: &[Val]) -> Option<Val> {
    let a = |i: usize| args.get(i).and_then(s_arg);
    Some(match name {
        n if n == SQL_GREATEST || n == SQL_LEAST => {
            let vals = args.iter().filter(|v| !v.is_null()).flat_map(mv_list);
            (if n == SQL_GREATEST { vals.max_by(cmp_vals) } else { vals.min_by(cmp_vals) }).unwrap_or(Val::Null)
        }
        n if n == SQL_REPLACE => match (a(0), a(1), a(2)) {
            (Some(s), Some(pat), Some(rep)) if !pat.is_empty() => Val::Str(s.replace(&pat, &rep)),
            (Some(s), Some(_), Some(_)) => Val::Str(s),
            _ => Val::Null,
        },
        n if n == SQL_CONCAT => match (a(0), a(1)) {
            (None, None) => Val::Str(String::new()),
            (x, y) => Val::Str(x.unwrap_or_default() + &y.unwrap_or_default()),
        },
        n if n == SQL_LENGTH => a(0).map(|s| Val::Int(s.trim_end_matches(' ').chars().count() as i64)).unwrap_or(Val::Null),
        _ => return None,
    })
}

pub fn call(name: &str, args: &[Val], ctx: &FnCtx) -> Result<Val> {
    if name == PROMQL_RE {
        return Ok(match (args.first().and_then(s_arg), args.get(1).and_then(s_arg)) {
            (Some(v), Some(p)) => match ctx.regex_cache.borrow_mut().get(&format!("^(?:{p})$"))? {
                Some(re) => Val::Bool(re.is_match(&v)?),
                None => Val::Null,
            },
            _ => Val::Null,
        });
    }
    if let Some(v) = sql_string_fn(name, args) {
        return Ok(v);
    }
    if let Some(v) = sql_scalar(name, args) {
        return Ok(v);
    }
    if name == SQL_DATE_DIFF {
        let ms = |v: Option<&Val>| match v {
            Some(Val::Date(d)) => Some(*d),
            Some(Val::Nanos(n)) => Some(n.div_euclid(super::NANOS_PER_MILLI)),
            _ => None,
        };
        return Ok(match (args.first().and_then(Val::as_str), ms(args.get(1)), ms(args.get(2))) {
            (Some(u), Some(a), Some(b)) => sql_date_diff(u, a, b).and_then(|v| i32::try_from(v).ok()).map(|v| Val::Int(v as i64)).unwrap_or(Val::Null),
            _ => Val::Null,
        });
    }
    if name == MYSQL_PATTERN_FN {
        return Ok(match args.first().and_then(Val::as_str) {
            Some(p) => Val::Str(mysql_to_java_pattern(p)),
            None => Val::Null,
        });
    }
    if name == VECTOR_ARITH {
        let op = args.first().and_then(Val::as_str).unwrap_or_default();
        return Ok(vector_arith(op, args.get(1).unwrap_or(&Val::Null), args.get(2).unwrap_or(&Val::Null)));
    }
    if name == TYPED_KEY_FN {
        let ty = args.get(1).and_then(Val::as_str).unwrap_or_default();
        return Ok(typed_key(args.first().unwrap_or(&Val::Null), ty));
    }
    if let Some(narrowed) = legacy_args(name, args) {
        let r = call(name, &narrowed, ctx)?;
        // A rounding of a date_nanos stays a date_nanos (its type).
        fn widen(v: Val) -> Val {
            match v {
                Val::Date(ms) => ms.checked_mul(super::NANOS_PER_MILLI).map(Val::Nanos).unwrap_or(Val::Null),
                Val::Multi(m) => Val::Multi(m.into_iter().map(widen).collect()),
                o => o,
            }
        }
        let nanos_in = args.iter().any(|a| matches!(a, Val::Nanos(_)));
        return Ok(if nanos_in && matches!(name, "DATE_TRUNC" | "BUCKET" | "TBUCKET") { widen(r) } else { r });
    }
    let a0 = args.first().cloned().unwrap_or(Val::Null);
    let n = args.len();
    let arity = |lo: usize, hi: usize| -> Result<()> { if n < lo || n > hi { Err(bad_arity(name)) } else { Ok(()) } };
    Ok(match name {
        // ---- math
        "ABS" => {
            arity(1, 1)?;
            match &a0 {
                Val::Int(i) => Val::Int(i.abs()),
                Val::Long(i) => Val::Long(i.abs()),
                Val::Double(d) => Val::Double(d.abs()),
                _ => Val::Null,
            }
        }
        "ACOS" => math1(args, f64::acos),
        "ASIN" => math1(args, f64::asin),
        "ATAN" => math1(args, f64::atan),
        "ATAN2" => match (args.first().and_then(f_arg), args.get(1).and_then(f_arg)) {
            (Some(y), Some(x)) => dbl(y.atan2(x)),
            _ => Val::Null,
        },
        "CBRT" => math1(args, f64::cbrt),
        "CEIL" | "CEILING" => match &a0 {
            Val::Double(d) => Val::Double(d.ceil()),
            o => o.clone(),
        },
        "FLOOR" => match &a0 {
            Val::Double(d) => Val::Double(d.floor()),
            o => o.clone(),
        },
        "COS" => math1(args, f64::cos),
        "COSH" => math1(args, f64::cosh),
        "SIN" => math1(args, f64::sin),
        "SINH" => math1(args, f64::sinh),
        "TAN" => math1(args, f64::tan),
        "TANH" => math1(args, f64::tanh),
        "SQRT" => match args.first().and_then(f_arg) {
            Some(x) if x >= 0.0 => Val::Double(x.sqrt()),
            _ => Val::Null,
        },
        "EXP" => math1(args, f64::exp),
        "E" => Val::Double(std::f64::consts::E),
        "PI" => Val::Double(std::f64::consts::PI),
        "TAU" => Val::Double(std::f64::consts::TAU),
        "LOG10" => match args.first().and_then(f_arg) {
            Some(x) if x > 0.0 => Val::Double(x.log10()),
            _ => Val::Null,
        },
        "LOG" => {
            arity(1, 2)?;
            if n == 1 {
                match f_arg(&a0) {
                    Some(x) if x > 0.0 => Val::Double(x.ln()),
                    _ => Val::Null,
                }
            } else {
                match (f_arg(&a0), f_arg(&args[1])) {
                    (Some(b), Some(x)) if b > 0.0 && b != 1.0 && x > 0.0 => Val::Double(x.ln() / b.ln()),
                    _ => Val::Null,
                }
            }
        }
        "POW" | "POWER" => match (args.first().and_then(f_arg), args.get(1).and_then(f_arg)) {
            (Some(b), Some(e)) => dbl(b.powf(e)),
            _ => Val::Null,
        },
        "HYPOT" => match (args.first().and_then(f_arg), args.get(1).and_then(f_arg)) {
            (Some(x), Some(y)) => dbl(x.hypot(y)),
            _ => Val::Null,
        },
        "COPY_SIGN" => match (&a0, args.get(1).and_then(f_arg)) {
            (v, Some(s)) if v.as_f64().is_some() => keep_int_type(v, v.as_f64().unwrap().abs().copysign(s)),
            _ => Val::Null,
        },
        "SIGNUM" => math1(args, |x| if x == 0.0 { 0.0 } else { x.signum() }),
        "MOD" => binop(BinOp::Mod, &a0, args.get(1).unwrap_or(&Val::Null)),
        "ROUND" | "TRUNCATE" => {
            arity(1, 2)?;
            let dec = args.get(1).and_then(i_arg).unwrap_or(0);
            let trunc = name == "TRUNCATE";
            match &a0 {
                Val::Int(_) | Val::Long(_) if dec >= 0 => a0.clone(),
                Val::Int(i) | Val::Long(i) => {
                    let p = 10i64.pow((-dec).min(18) as u32);
                    let r = if trunc { i / p * p } else { ((*i as f64 / p as f64).round() as i64) * p };
                    keep_int_type(&a0, r as f64)
                }
                Val::Double(d) => {
                    let p = 10f64.powi(dec as i32);
                    let r = if trunc { (d * p).trunc() / p } else { round_half_up(d * p) / p };
                    Val::Double(r)
                }
                _ => Val::Null,
            }
        }
        "ROUND_TO" => {
            let Some(x) = a0.as_f64() else { return Ok(Val::Null) };
            let mut pts: Vec<&Val> = args[1..].iter().collect();
            pts.sort_by(|a, b| cmp_vals(a, b));
            pts.iter().rev().find(|p| p.as_f64().map(|v| v <= x).unwrap_or(false)).map(|v| (*v).clone()).unwrap_or_else(|| pts.first().map(|v| (*v).clone()).unwrap_or(Val::Null))
        }
        // ---- string
        "LENGTH" | "CHAR_LENGTH" => s_arg(&a0).map(|s| Val::Int(s.chars().count() as i64)).unwrap_or(Val::Null),
        "BIT_LENGTH" => s_arg(&a0).map(|s| Val::Int(s.len() as i64 * 8)).unwrap_or(Val::Null),
        "BYTE_LENGTH" => s_arg(&a0).map(|s| Val::Int(s.len() as i64)).unwrap_or(Val::Null),
        "CONCAT" => {
            if n < 2 {
                return Err(bad_arity(name));
            }
            let mut out = String::new();
            for a in args {
                match s_arg(a) {
                    Some(s) => out.push_str(&s),
                    None => return Ok(Val::Null),
                }
            }
            Val::Str(out)
        }
        "TO_UPPER" | "UPPER" | "UCASE" => match &a0 {
            Val::Multi(m) => Val::Multi(m.iter().map(|x| Val::Str(x.to_text().to_uppercase())).collect()),
            v => s_arg(v).map(|s| Val::Str(s.to_uppercase())).unwrap_or(Val::Null),
        },
        "TO_LOWER" | "LOWER" | "LCASE" => match &a0 {
            Val::Multi(m) => Val::Multi(m.iter().map(|x| Val::Str(x.to_text().to_lowercase())).collect()),
            v => s_arg(v).map(|s| Val::Str(s.to_lowercase())).unwrap_or(Val::Null),
        },
        "TRIM" => s_arg(&a0).map(|s| Val::Str(s.trim().to_string())).unwrap_or(Val::Null),
        "LTRIM" => s_arg(&a0).map(|s| Val::Str(s.trim_start().to_string())).unwrap_or(Val::Null),
        "RTRIM" => s_arg(&a0).map(|s| Val::Str(s.trim_end().to_string())).unwrap_or(Val::Null),
        "LEFT" => match (s_arg(&a0), args.get(1).and_then(i_arg)) {
            (Some(s), Some(k)) => Val::Str(s.chars().take(k.max(0) as usize).collect()),
            _ => Val::Null,
        },
        "RIGHT" => match (s_arg(&a0), args.get(1).and_then(i_arg)) {
            (Some(s), Some(k)) => {
                let c: Vec<char> = s.chars().collect();
                let k = (k.max(0) as usize).min(c.len());
                Val::Str(c[c.len() - k..].iter().collect())
            }
            _ => Val::Null,
        },
        "SUBSTRING" | "SUBSTR" => {
            arity(2, 3)?;
            match (s_arg(&a0), i_arg(&args[1])) {
                (Some(s), Some(start)) => {
                    let c: Vec<char> = s.chars().collect();
                    let len = c.len() as i64;
                    let mut st = if start < 0 { len + start } else { start - 1 };
                    st = st.max(0);
                    let l = args.get(2).and_then(i_arg).unwrap_or(len);
                    if l < 0 {
                        return Ok(Val::Null);
                    }
                    let st = (st as usize).min(c.len());
                    let en = (st + l as usize).min(c.len());
                    Val::Str(c[st..en].iter().collect())
                }
                _ => Val::Null,
            }
        }
        "REPLACE" => match (s_arg(&a0), args.get(1).and_then(s_arg), args.get(2).and_then(s_arg)) {
            // Java `Matcher.replaceAll`: `$n` / `${name}` capture references
            // and `\` escapes in the replacement (ES `Replace`); an invalid
            // reference fails the row with ES's warning.
            (Some(s), Some(re), Some(rep)) => match ctx.regex_cache.borrow_mut().get(&re)? {
                Some(r) => match r.replace_java(&s, &rep)? {
                    Ok(out) => Val::Str(out),
                    Err(e) => return Err(eval_warning(e)),
                },
                None => Val::Null,
            },
            _ => Val::Null,
        },
        "SPLIT" => match (s_arg(&a0), args.get(1).and_then(s_arg)) {
            (Some(s), Some(d)) if !d.is_empty() => Val::from_list(s.split(d.as_str()).map(|x| Val::Str(x.to_string())).collect()),
            _ => Val::Null,
        },
        "STARTS_WITH" => match (s_arg(&a0), args.get(1).and_then(s_arg)) {
            (Some(s), Some(p)) => Val::Bool(s.starts_with(&p)),
            _ => Val::Null,
        },
        "ENDS_WITH" => match (s_arg(&a0), args.get(1).and_then(s_arg)) {
            (Some(s), Some(p)) => Val::Bool(s.ends_with(&p)),
            _ => Val::Null,
        },
        "CONTAINS" => match (s_arg(&a0), args.get(1).and_then(s_arg)) {
            (Some(s), Some(p)) => Val::Bool(s.contains(&p)),
            _ => Val::Null,
        },
        "LOCATE" | "POSITION" => {
            let (hay, needle) = if name == "POSITION" { (args.get(1).and_then(s_arg), s_arg(&a0)) } else { (s_arg(&a0), args.get(1).and_then(s_arg)) };
            match (hay, needle) {
                (Some(s), Some(sub)) => {
                    let start = args.get(2).and_then(i_arg).unwrap_or(1).max(1) as usize;
                    let c: Vec<char> = s.chars().collect();
                    let from: String = c.iter().skip(start - 1).collect();
                    match from.find(&sub) {
                        Some(p) => Val::Int((from[..p].chars().count() + start) as i64),
                        None => Val::Int(0),
                    }
                }
                _ => Val::Null,
            }
        }
        "REVERSE" => s_arg(&a0).map(|s| Val::Str(s.chars().rev().collect())).unwrap_or(Val::Null),
        "REPEAT" => match (s_arg(&a0), args.get(1).and_then(i_arg)) {
            (Some(s), Some(k)) if k >= 0 => Val::Str(s.repeat(k as usize)),
            _ => Val::Null,
        },
        "SPACE" => match i_arg(&a0) {
            Some(k) if k >= 0 => Val::Str(" ".repeat(k as usize)),
            _ => Val::Null,
        },
        "ASCII" => s_arg(&a0).and_then(|s| s.chars().next()).map(|c| Val::Int(c as i64)).unwrap_or(Val::Null),
        "TO_BASE64" => {
            use base64::Engine;
            s_arg(&a0).map(|s| Val::Str(base64::engine::general_purpose::STANDARD.encode(s))).unwrap_or(Val::Null)
        }
        "FROM_BASE64" => {
            use base64::Engine;
            s_arg(&a0)
                .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s.trim()).ok())
                .map(|b| Val::Str(String::from_utf8_lossy(&b).into_owned()))
                .unwrap_or(Val::Null)
        }
        "LEVENSHTEIN" => match (s_arg(&a0), args.get(1).and_then(s_arg)) {
            (Some(a), Some(b)) => Val::Int(super::levenshtein(&a, &b) as i64),
            _ => Val::Null,
        },
        // ---- date
        "NOW" | "CURRENT_TIMESTAMP" => Val::Date(ctx.now),
        // Today's start in the query's zone.
        "CURRENT_DATE" => Val::Date(in_zone(ctx.now, |l| Some(l - l.rem_euclid(86_400_000))).unwrap_or(ctx.now - ctx.now.rem_euclid(86_400_000))),
        "DATE_TRUNC" => {
            arity(2, 2)?;
            match (&a0, &args[1]) {
                (Val::Str(u), Val::Date(d)) if parse_temporal(u).is_some() => parse_temporal(u).and_then(|iv| date_trunc(*d, &iv)).map(Val::Date).unwrap_or(Val::Null),
                (Val::Str(u), Val::Date(d)) => {
                    let u = u.to_ascii_lowercase();
                    let iv = match u.trim_end_matches('s') {
                        "millennium" | "millennia" => Some(Val::Period(12_000, 0)),
                        "century" | "centurie" => Some(Val::Period(1200, 0)),
                        "decade" => Some(Val::Period(120, 0)),
                        other => super::parse::time_unit(other),
                    };
                    iv.and_then(|iv| date_trunc(*d, &iv)).map(Val::Date).unwrap_or(Val::Null)
                }
                (iv, Val::Date(d)) => date_trunc(*d, iv).map(Val::Date).unwrap_or(Val::Null),
                (Val::Date(d), iv) => date_trunc(*d, iv).map(Val::Date).unwrap_or(Val::Null),
                _ => Val::Null,
            }
        }
        "DATE_EXTRACT" => {
            arity(2, 2)?;
            match (s_arg(&a0), &args[1]) {
                (Some(p), Val::Date(d)) => date_part(&p, *d).map(Val::Long).unwrap_or(Val::Null),
                _ => Val::Null,
            }
        }
        "YEAR" | "MONTH" | "DAY" | "DAY_OF_MONTH" | "HOUR" | "MINUTE" | "SECOND" => match &a0 {
            Val::Date(d) => date_part(&name.to_ascii_lowercase(), *d).map(Val::Int).unwrap_or(Val::Null),
            _ => Val::Null,
        },
        "DATE_FORMAT" => {
            arity(1, 2)?;
            let (fmt, d) = if n == 2 { (s_arg(&a0), &args[1]) } else { (None, &a0) };
            match d {
                Val::Date(ms) => match fmt {
                    None => Val::Str(super::fmt_date(*ms)),
                    Some(f) => match java_fmt(&f) {
                        Some(df) => Val::Str(localize_date(df.format(*ms, ctx.tz), ctx.locale.as_deref())),
                        None => Val::Null,
                    },
                },
                _ => Val::Null,
            }
        }
        "DATE_PARSE" => {
            arity(1, 2)?;
            let (fmt, s) = if n == 2 { (s_arg(&a0), s_arg(&args[1])) } else { (None, s_arg(&a0)) };
            match s {
                Some(s) => match fmt {
                    None => parse_date_str(&s).map(Val::Date).unwrap_or(Val::Null),
                    Some(f) => java_fmt(&f).and_then(|df| df.parse_with_tz(&s, ctx.tz).ok()).map(Val::Date).unwrap_or(Val::Null),
                },
                None => Val::Null,
            }
        }
        "DATE_DIFF" => {
            arity(3, 3)?;
            match (s_arg(&a0), &args[1], &args[2]) {
                (Some(u), Val::Date(a), Val::Date(b)) => date_diff(&u, *a, *b).map(|v| if v >= i32::MIN as i64 && v <= i32::MAX as i64 { Val::Int(v) } else { Val::Null }).unwrap_or(Val::Null),
                _ => Val::Null,
            }
        }
        "DAY_NAME" => match &a0 {
            Val::Date(d) => day_name(*d).map(Val::Str).unwrap_or(Val::Null),
            _ => Val::Null,
        },
        "MONTH_NAME" => match &a0 {
            Val::Date(d) => chrono::Utc.timestamp_millis_opt(local_ms(*d)).single().map(|x| Val::Str(x.format("%B").to_string())).unwrap_or(Val::Null),
            _ => Val::Null,
        },
        // ---- conversion
        "TO_AGGREGATE_METRIC_DOUBLE" | "TO_AGGREGATEMETRICDOUBLE" => to_amd(&a0),
        "TO_STRING" | "TO_STR" | "TO_KEYWORD" => to_string(&a0),
        "TO_TEXT" => a0.clone(),
        "TO_LONG" | "TO_INTEGER" | "TO_INT" if n == 2 => {
            // `ToLongBase` / `ToIntegerBase`.
            let (Some(s), Some(base)) = (s_arg(&a0), i_arg(&args[1])) else { return Ok(Val::Null) };
            let mut v = s.clone();
            let parsed = if base == 10 {
                v.trim().parse::<i64>().ok().or_else(|| v.trim().parse::<f64>().ok().filter(|d| d.is_finite() && d.abs() < 9.3e18).map(|d| d as i64))
            } else {
                if base == 16 && v.starts_with("0x") {
                    v = v[2..].to_string();
                }
                if (2..=36).contains(&base) { i64::from_str_radix(&v, base as u32).ok() } else { None }
            };
            let int = name != "TO_LONG";
            match parsed {
                Some(x) if !int || (x >= i32::MIN as i64 && x <= i32::MAX as i64) => {
                    if int { Val::Int(x) } else { Val::Long(x) }
                }
                Some(x) => return Err(eval_warning(format!("org.elasticsearch.xpack.esql.core.InvalidArgumentException: [{x}] out of [integer] range"))),
                None => return Err(eval_warning(format!("org.elasticsearch.xpack.esql.core.InvalidArgumentException: Unable to convert [{v}] to number of base [{base}]"))),
            }
        }
        "TO_LONG" => to_long(&a0),
        "TO_UNSIGNED_LONG" | "TO_UL" | "TO_ULONG" => to_unsigned_long(&a0)?,
        "TO_INTEGER" | "TO_INT" => to_int(&a0),
        "TO_DOUBLE" | "TO_DBL" => to_double(&a0),
        "TO_BOOLEAN" | "TO_BOOL" => to_bool(&a0),
        "TO_DATETIME" | "TO_DT" => to_date(&a0),
        "TO_DATE_NANOS" => to_date_nanos(&a0)?,
        "TO_IP" => match &a0 {
            // An IPv4-mapped IPv6 address is the IPv4 address (Java's
            // `InetAddress`, as ES parses it).
            Val::Str(s) if ip_to_bytes(s).is_some() => Val::Str(s.trim().parse::<std::net::IpAddr>().map(|i| match i {
                std::net::IpAddr::V6(v6) => v6.to_ipv4_mapped().map(|v4| v4.to_string()).unwrap_or_else(|| v6.to_string()),
                v4 => v4.to_string(),
            }).unwrap_or_else(|_| s.clone())),
            Val::Multi(m) => Val::from_list(m.iter().filter_map(|x| x.as_str().filter(|s| ip_to_bytes(s).is_some()).map(|s| Val::Str(s.to_string()))).collect()),
            _ => Val::Null,
        },
        "TO_VERSION" | "TO_VER" => to_string(&a0),
        "TO_GEOPOINT" | "TO_CARTESIANPOINT" | "TO_GEOSHAPE" | "TO_CARTESIANSHAPE" => match &a0 {
            // Text that is not WKT fails the row (null, with ES's warning).
            Val::Str(s) => match super::spatial::parse_wkt_checked(s, spatial_target(name)) {
                Ok(v) => v.map(Val::Str).unwrap_or(Val::Null),
                Err(e) => return Err(eval_warning(format!("java.lang.IllegalArgumentException: {e}"))),
            },
            _ => to_spatial(&a0, spatial_target(name)),
        },
        "TO_DEGREES" => math1(args, f64::to_degrees),
        "TO_RADIANS" => math1(args, f64::to_radians),
        "TO_TIMEDURATION" | "TO_DATEPERIOD" => match &a0 {
            Val::Str(s) => {
                let mut it = s.split_whitespace();
                match (it.next().and_then(|x| x.parse::<i64>().ok()), it.next().and_then(super::parse::time_unit)) {
                    (Some(k), Some(Val::Dur(ms))) => Val::Dur(ms * k),
                    (Some(k), Some(Val::Period(m, d))) => Val::Period(m * k, d * k),
                    _ => Val::Null,
                }
            }
            o => o.clone(),
        },
        // ---- conditional
        "CASE" | "IIF" => {
            let mut i = 0;
            while i + 1 < n {
                if args[i] == Val::Bool(true) {
                    return Ok(args[i + 1].clone());
                }
                i += 2;
            }
            if n % 2 == 1 { args[n - 1].clone() } else { Val::Null }
        }
        "COALESCE" | "IFNULL" | "NVL" => args.iter().find(|a| !a.is_null()).cloned().unwrap_or(Val::Null),
        "ISNULL" => Val::Bool(a0.is_null()),
        "NULLIF" => {
            if args.get(1).map(|b| compare(&a0, b) == Some(Ordering::Equal)).unwrap_or(false) {
                Val::Null
            } else {
                a0.clone()
            }
        }
        "GREATEST" | "LEAST" => {
            let mut vals: Vec<Val> = Vec::new();
            for a in args {
                if a.is_null() {
                    return Ok(Val::Null);
                }
                vals.extend(mv_list(a));
            }
            let it = vals.into_iter();
            (if name == "GREATEST" { it.max_by(cmp_vals) } else { it.min_by(cmp_vals) }).unwrap_or(Val::Null)
        }
        // ---- multivalue
        "MV_COUNT" => {
            let l = mv_list(&a0);
            if l.is_empty() { Val::Null } else { Val::Int(l.len() as i64) }
        }
        "MV_FIRST" => mv_list(&a0).first().cloned().unwrap_or(Val::Null),
        "MV_LAST" => mv_list(&a0).last().cloned().unwrap_or(Val::Null),
        "MV_MIN" => mv_list(&a0).into_iter().min_by(cmp_vals).unwrap_or(Val::Null),
        "MV_MAX" => mv_list(&a0).into_iter().max_by(cmp_vals).unwrap_or(Val::Null),
        "MV_SUM" => {
            let l = mv_list(&a0);
            if l.is_empty() {
                Val::Null
            } else {
                l[1..].iter().fold(l[0].clone(), |acc, x| binop(BinOp::Add, &acc, x))
            }
        }
        "MV_AVG" => {
            let l: Vec<f64> = mv_list(&a0).iter().filter_map(Val::as_f64).collect();
            if l.is_empty() { Val::Null } else { Val::Double(l.iter().sum::<f64>() / l.len() as f64) }
        }
        "MV_MEDIAN" => {
            let l = mv_list(&a0);
            let f: Vec<f64> = l.iter().filter_map(Val::as_f64).collect();
            match median(f) {
                Some(m) => match l.first() {
                    Some(Val::Int(_)) | Some(Val::Long(_)) => keep_int_type(&l[0], m.floor()),
                    _ => Val::Double(m),
                },
                None => Val::Null,
            }
        }
        "MV_MEDIAN_ABSOLUTE_DEVIATION" => {
            let l = mv_list(&a0);
            mad(l.iter().filter_map(Val::as_f64).collect()).map(|m| match l.first() {
                Some(v @ (Val::Int(_) | Val::Long(_))) => keep_int_type(v, m.floor()),
                _ => Val::Double(m),
            }).unwrap_or(Val::Null)
        }
        "MV_PERCENTILE" => {
            let l = mv_list(&a0);
            let p = args.get(1).and_then(f_arg).unwrap_or(50.0);
            match l.first() {
                // ES `MvPercentile`: integer / long values interpolate as
                // the lower value plus the truncated increment, in their type.
                Some(v @ (Val::Int(_) | Val::Long(_))) => {
                    let mut xs: Vec<i64> = l.iter().filter_map(Val::as_i64).collect();
                    xs.sort_unstable();
                    let rank = p.clamp(0.0, 100.0) / 100.0 * (xs.len() - 1) as f64;
                    let lo = rank.floor() as usize;
                    let base = xs[lo];
                    let r = match xs.get(lo + 1) {
                        Some(&hi) => base + ((hi - base) as f64 * (rank - lo as f64)) as i64,
                        None => base,
                    };
                    keep_int_type(v, r as f64)
                }
                Some(_) => percentile(l.iter().filter_map(Val::as_f64).collect(), p).map(Val::Double).unwrap_or(Val::Null),
                None => Val::Null,
            }
        }
        "MV_DEDUPE" => {
            let mut seen = std::collections::HashSet::new();
            Val::from_list(mv_list(&a0).into_iter().filter(|x| seen.insert(x.key())).collect())
        }
        "MV_SORT" => {
            let desc = match args.get(1) {
                None | Some(Val::Null) => false,
                Some(o) => {
                    let o = o.to_text();
                    if o.eq_ignore_ascii_case("desc") {
                        true
                    } else if o.eq_ignore_ascii_case("asc") {
                        false
                    } else {
                        return Err(EsError::new(400, SRC_ERR, format!("Invalid order value in [{{src}}], expected one of [ASC, DESC] but got [{o}]")));
                    }
                }
            };
            let mut l = mv_list(&a0);
            l.sort_by(cmp_vals);
            if desc {
                l.reverse();
            }
            Val::from_list(l)
        }
        "MV_CONCAT" => {
            let l = mv_list(&a0);
            if l.is_empty() {
                Val::Null
            } else {
                Val::Str(l.iter().map(Val::to_text).collect::<Vec<_>>().join(&args.get(1).and_then(s_arg).unwrap_or_default()))
            }
        }
        "MV_SLICE" => {
            let l = mv_list(&a0);
            let len = l.len() as i64;
            let norm = |i: i64| if i < 0 { len + i } else { i };
            let s = norm(args.get(1).and_then(i_arg).unwrap_or(0));
            let e = args.get(2).and_then(i_arg).map(norm).unwrap_or(s);
            if l.is_empty() || s > e || s >= len || e < 0 {
                Val::Null
            } else {
                Val::from_list(l[s.max(0) as usize..=(e.min(len - 1) as usize)].to_vec())
            }
        }
        "MV_APPEND" => {
            let mut l = mv_list(&a0);
            l.extend(mv_list(args.get(1).unwrap_or(&Val::Null)));
            Val::from_list(l)
        }
        "MV_ZIP" => {
            let a = mv_list(&a0);
            let b = mv_list(args.get(1).unwrap_or(&Val::Null));
            let d = args.get(2).and_then(s_arg).unwrap_or_else(|| ",".into());
            if a.is_empty() && b.is_empty() {
                Val::Null
            } else {
                let n = a.len().max(b.len());
                Val::from_list(
                    (0..n)
                        .map(|i| match (a.get(i), b.get(i)) {
                            (Some(x), Some(y)) => Val::Str(format!("{}{d}{}", x.to_text(), y.to_text())),
                            (Some(x), None) | (None, Some(x)) => Val::Str(x.to_text()),
                            _ => Val::Null,
                        })
                        .collect(),
                )
            }
        }
        "MV_CONTAINS" => {
            let a = mv_list(&a0);
            let b = mv_list(args.get(1).unwrap_or(&Val::Null));
            Val::Bool(b.iter().all(|x| a.iter().any(|y| compare(x, y) == Some(Ordering::Equal))))
        }
        // ---- ip / spatial
        "CIDR_MATCH" => {
            if n < 2 {
                return Err(bad_arity(name));
            }
            let Some(ip) = s_arg(&a0) else { return Ok(Val::Null) };
            let blocks: Vec<String> = args[1..].iter().flat_map(|a| mv_list(a).into_iter().map(|x| x.to_text())).collect();
            Val::Bool(blocks.iter().any(|c| cidr_match(&ip, c)))
        }
        "IP_PREFIX" => {
            let Some(ip) = s_arg(&a0) else { return Ok(Val::Null) };
            let Some(mut bytes) = ip_to_bytes(&ip) else { return Ok(Val::Null) };
            let len = if bytes.len() == 4 { args.get(1).and_then(i_arg) } else { args.get(2).and_then(i_arg) }.unwrap_or(0).max(0) as usize;
            for i in 0..bytes.len() * 8 {
                if i >= len {
                    bytes[i / 8] &= !(0x80u8 >> (i % 8));
                }
            }
            let addr: std::net::IpAddr = if bytes.len() == 4 {
                std::net::Ipv4Addr::new(bytes[0], bytes[1], bytes[2], bytes[3]).into()
            } else {
                let mut b = [0u8; 16];
                b.copy_from_slice(&bytes);
                std::net::Ipv6Addr::from(b).into()
            };
            Val::Str(addr.to_string())
        }
        // `@ConvertEvaluator`: every point of a multi-value converted, in order.
        "ST_X" | "ST_Y" => {
            let one = |v: &Val| match s_arg(v).and_then(|s| super::spatial::point_xy(&s)) {
                Some((x, y)) => Val::Double(if name == "ST_X" { x } else { y }),
                None => Val::Null,
            };
            match &a0 {
                Val::Multi(points) => Val::from_list(points.iter().map(one).filter(|v| !matches!(v, Val::Null)).collect()),
                point => one(point),
            }
        }
        "ST_DISTANCE" => {
            // ES `StDistance` (geo): Lucene's `SloppyMath.haversinMeters`
            // over the quantized coordinates (what doc values hold); a
            // multi-valued point is at its nearest value's distance.
            let pts = |v: &Val| -> Option<Vec<(f64, f64)>> {
                match v {
                    Val::Multi(m) => m.iter().map(|x| s_arg(x).and_then(|s| parse_point(&s))).collect::<Option<Vec<_>>>().filter(|p| !p.is_empty()),
                    v => s_arg(v).and_then(|s| parse_point(&s)).map(|p| vec![p]),
                }
            };
            let cartesian = matches!(args.get(2), Some(Val::Str(m)) if m == CARTESIAN_DISTANCE);
            match (pts(&a0), args.get(1).and_then(pts)) {
                (Some(a), Some(b)) if cartesian => {
                    let mut best = f64::INFINITY;
                    for (x1, y1) in &a {
                        for (x2, y2) in &b {
                            best = best.min((x1 - x2).hypot(y1 - y2));
                        }
                    }
                    Val::Double(best)
                }
                (Some(a), Some(b)) => {
                    let mut best = f64::INFINITY;
                    for (lon1, lat1) in &a {
                        for (lon2, lat2) in &b {
                            let (p, q) = (rustic_core::geo::quantize(*lat1, *lon1), rustic_core::geo::quantize(*lat2, *lon2));
                            best = best.min(rustic_core::geo::sloppy_haversin_meters(p.0, p.1, q.0, q.1));
                        }
                    }
                    Val::Double(best)
                }
                _ => Val::Null,
            }
        }
        "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS" if matches!(args.get(2), Some(Val::Str(m)) if m == GEO_RELATION) => {
            use rustic_core::fieldtypes::shape::{Geometry, SpatialRelation};
            // `SpatialRelatesFunction` on geo types: a multi-valued
            // argument is the collection of its values.
            let geom = |v: &Val| -> Option<Geometry> {
                let one = |x: &Val| s_arg(x).and_then(|s| rustic_core::fieldtypes::shape::parse_wkt(s.trim(), &rustic_core::fieldtypes::shape::ParseOpts { geo: false, coerce: true, ignore_z: true }).ok());
                match v {
                    Val::Multi(m) => Some(Geometry::Collection(m.iter().map(one).collect::<Option<Vec<_>>>()?)),
                    v => one(v),
                }
            };
            let rel = match name {
                "ST_INTERSECTS" => SpatialRelation::Intersects,
                "ST_DISJOINT" => SpatialRelation::Disjoint,
                "ST_WITHIN" => SpatialRelation::Within,
                _ => SpatialRelation::Contains,
            };
            match (geom(&a0), args.get(1).and_then(geom)) {
                (Some(a), Some(b)) => crate::fieldtypes::geo_shape_query::geometries_relate(&a, &b, rel).map_or(Val::Null, Val::Bool),
                _ => Val::Null,
            }
        }
        "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS" => {
            // A multi-valued argument is the union of its values (ES reads
            // multi-valued points as a MultiPoint).
            let prims = |v: &Val| -> Option<rustic_core::fieldtypes::shape::Prims> {
                match v {
                    Val::Multi(m) => {
                        let mut all = rustic_core::fieldtypes::shape::Prims::default();
                        for x in m {
                            let p = super::spatial::prims(&s_arg(x)?)?;
                            all.points.extend(p.points);
                            all.lines.extend(p.lines);
                            all.polys.extend(p.polys);
                        }
                        Some(all)
                    }
                    v => super::spatial::prims(&s_arg(v)?),
                }
            };
            match (prims(&a0), args.get(1).and_then(prims)) {
                (Some(a), Some(b)) => {
                    use rustic_core::fieldtypes::shape::{relate, SpatialRelation};
                    let rel = match name {
                        "ST_INTERSECTS" => SpatialRelation::Intersects,
                        "ST_DISJOINT" => SpatialRelation::Disjoint,
                        "ST_WITHIN" => SpatialRelation::Within,
                        _ => SpatialRelation::Contains,
                    };
                    Val::Bool(relate(&a, &b, rel))
                }
                _ => Val::Null,
            }
        }
        "BUCKET" | "TBUCKET" => bucket(args)?,
        other => match super::func2::call(other, args, ctx) {
            Some(r) => r?,
            None => return Err(EsError::new(400, "verification_exception", format!("Found 1 problem\nUnknown function [{}]", other.to_lowercase()))),
        },
    })
}

fn round_half_up(x: f64) -> f64 {
    if x >= 0.0 { (x + 0.5).floor() } else { -((-x + 0.5).floor()) }
}

/// Static return type of a scalar function (None = from the values).
pub fn return_type(name: &str, arg_types: &[String]) -> Option<String> {
    // A bound geographic envelope carries a marker argument (see
    // `func2::GEO_ENVELOPE`): typed as the function of its geometry.
    let arg_types = if matches!(name, "ST_ENVELOPE" | "ST_XMIN" | "ST_XMAX" | "ST_YMIN" | "ST_YMAX") && arg_types.len() == 2 && arg_types[1] == "keyword" { &arg_types[..1] } else { arg_types };
    if name == VECTOR_ARITH {
        return Some("dense_vector".into());
    }
    if name == SQL_DATE_DIFF {
        return Some("integer".into());
    }
    if name == PROMQL_RE {
        return Some("boolean".into());
    }
    if name == SQL_REPLACE || name == SQL_CONCAT {
        return Some("keyword".into());
    }
    if name == SQL_LENGTH {
        return Some("integer".into());
    }
    if name == SQL_GREATEST || name == SQL_LEAST {
        return arg_types.iter().find(|t| t.as_str() != "null").cloned().map(|t| if t == "text" { "keyword".into() } else { t });
    }
    if name == MYSQL_PATTERN_FN {
        return Some("keyword".into());
    }
    let a0 = arg_types.first().cloned();
    let t: &str = match name {
        "INSERT" => "keyword",
        "DAY_OF_WEEK" | "ISO_DAY_OF_WEEK" | "DAY_OF_YEAR" | "WEEK_OF_YEAR" | "ISO_WEEK_OF_YEAR" | "QUARTER" | "MINUTE_OF_DAY" | "DATE_PART" | "OCTET_LENGTH" => "integer",
        "COT" | "EXPM1" | "RANDOM" | "DEGREES" | "RADIANS" => "double",
        "CHAR" => "keyword",
        "DATE_ADD" => "date",
        "SIGN" => return a0.filter(|t| matches!(t.as_str(), "integer" | "long")).or(Some("double".into())),
        "LENGTH" | "CHAR_LENGTH" | "BIT_LENGTH" | "BYTE_LENGTH" | "LOCATE" | "POSITION" | "MV_COUNT" | "DATE_DIFF" | "LEVENSHTEIN" | "ASCII" | "YEAR" | "MONTH" | "DAY" | "DAY_OF_MONTH" | "HOUR" | "MINUTE" | "SECOND" => "integer",
        "DATE_EXTRACT" => "long",
        "CONCAT" | "TO_UPPER" | "TO_LOWER" | "UPPER" | "LOWER" | "UCASE" | "LCASE" | "TRIM" | "LTRIM" | "RTRIM" | "LEFT" | "RIGHT" | "SUBSTRING" | "SUBSTR" | "REPLACE" | "SPLIT" | "REVERSE" | "REPEAT" | "SPACE" | "TO_BASE64" | "FROM_BASE64" | "DATE_FORMAT" | "DAY_NAME" | "MONTH_NAME" | "TO_STRING" | "TO_STR" | "TO_KEYWORD" | "MV_CONCAT" | "MV_ZIP" => "keyword",
        "STARTS_WITH" | "ENDS_WITH" | "CONTAINS" | "CIDR_MATCH" | "TO_BOOLEAN" | "TO_BOOL" | "ISNULL" | "MV_CONTAINS" | "ST_INTERSECTS" | "ST_DISJOINT" | "ST_WITHIN" | "ST_CONTAINS" => "boolean",
        "NOW" | "CURRENT_TIMESTAMP" | "CURRENT_DATE" | "DATE_PARSE" | "TO_DATETIME" | "TO_DT" => "date",
        "TO_DATE_NANOS" => "date_nanos",
        "TO_TEXT" => "text",
        "TO_LONG" => "long",
        "TO_UNSIGNED_LONG" | "TO_UL" | "TO_ULONG" => "unsigned_long",
        "TO_INTEGER" | "TO_INT" => "integer",
        "TO_DOUBLE" | "TO_DBL" | "ACOS" | "ASIN" | "ATAN" | "ATAN2" | "CBRT" | "COS" | "COSH" | "SIN" | "SINH" | "TAN" | "TANH" | "SQRT" | "EXP" | "E" | "PI" | "TAU" | "LOG" | "LOG10" | "POW" | "POWER" | "HYPOT" | "SIGNUM" | "MV_AVG" | "TO_DEGREES" | "TO_RADIANS" | "ST_X" | "ST_Y" | "ST_DISTANCE" => "double",
        "TO_IP" | "IP_PREFIX" => "ip",
        "TO_VERSION" | "TO_VER" => "version",
        "TO_GEOPOINT" => "geo_point",
        "TO_CARTESIANPOINT" => "cartesian_point",
        "TO_GEOSHAPE" => "geo_shape",
        "TO_CARTESIANSHAPE" => "cartesian_shape",
        "TO_TIMEDURATION" => "time_duration",
        "TO_DATEPERIOD" => "date_period",
        "DATE_TRUNC" => return arg_types.get(1).cloned().filter(|t| t.starts_with("date")).or(Some("date".into())),
        "ABS" | "CEIL" | "FLOOR" | "ROUND" | "TRUNCATE" | "MV_MIN" | "MV_MAX" | "MV_FIRST" | "MV_LAST" | "MV_DEDUPE" | "MV_SORT" | "MV_SLICE" | "MV_APPEND" | "MV_SUM" | "COPY_SIGN" | "GREATEST" | "LEAST" | "NULLIF" | "MOD" | "ROUND_TO" => return a0.map(|t| if t == "text" { "keyword".into() } else { t }),
        "COALESCE" | "IFNULL" | "NVL" => return arg_types.iter().find(|t| *t != "null").cloned().map(|t| if t == "text" { "keyword".into() } else { t }),
        "CASE" | "IIF" => {
            let n = arg_types.len();
            let mut vals: Vec<&String> = (1..n).step_by(2).map(|i| &arg_types[i]).collect();
            if n % 2 == 1 {
                vals.push(&arg_types[n - 1]);
            }
            return vals.into_iter().find(|t| *t != "null").cloned().map(|t| if t == "text" { "keyword".into() } else { t });
        }
        "BUCKET" | "TBUCKET" => return a0.map(|t| if t.starts_with("date") { t } else { "double".into() }),
        "MV_MEDIAN" | "MV_MEDIAN_ABSOLUTE_DEVIATION" | "MV_PERCENTILE" => return a0,
        // `StSimplify.dataType`: the geometry's type, `null` for a null tolerance.
        "ST_SIMPLIFY" => return if arg_types.get(1).map(String::as_str) == Some("null") { Some("null".into()) } else { a0 },
        "TO_AGGREGATE_METRIC_DOUBLE" | "TO_AGGREGATEMETRICDOUBLE" => "aggregate_metric_double",
        // Everything else: the function's declared signatures.
        _ => return super::typecheck::lookup(name).and_then(|d| super::typecheck::return_type(d, arg_types)),
    };
    Some(t.to_string())
}

#[cfg(test)]
mod exact_value_tests {
    use super::*;
    use serde_json::{json, Value};

    fn node_with(name: &str, mapping: Value, docs: &[(&str, Value)]) -> std::sync::Arc<crate::node::Node> {
        let tmp = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index(name, &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": mapping}})).unwrap();
        let svc = node.get_index(name).unwrap();
        let reqs = docs.iter().map(|(id, d)| crate::index::WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node
    }

    fn ids(node: &crate::node::Node, index: &str, query: Value) -> Vec<String> {
        let svc = node.get_index(index).unwrap();
        let req = crate::search::parse_request(&json!({"query": query, "size": 100}), &HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    }

    fn esql(node: &crate::node::Node, q: &str) -> super::super::EsqlResult {
        match super::super::run(node, q, super::super::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }) {
            Ok(r) => r,
            Err(e) => panic!("{q}: {e:?}"),
        }
    }

    /// Review finding 181: long term / terms / range predicates and
    /// query-language source reads keep adjacent values above 2^53 and
    /// both extrema exact, for JSON numbers and strings; fractional bounds
    /// round inward; exclusive bounds past the extrema match nothing.
    #[test]
    fn signed_longs_are_matched_and_read_exactly() {
        const P: i64 = 9_007_199_254_740_992;
        let node = node_with("nums", json!({"n": {"type": "long"}}), &[
            ("a", json!({"n": P})),
            ("b", json!({"n": P + 1})),
            ("mx", json!({"n": i64::MAX})),
            ("mn", json!({"n": i64::MIN})),
            ("s", json!({"n": "9007199254740995"})),
        ]);
        let v = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(ids(&node, "nums", json!({"term": {"n": P + 1}})), v(&["b"]));
        assert_eq!(ids(&node, "nums", json!({"term": {"n": (P + 1).to_string()}})), v(&["b"]));
        assert_eq!(ids(&node, "nums", json!({"term": {"n": i64::MAX}})), v(&["mx"]));
        assert_eq!(ids(&node, "nums", json!({"term": {"n": i64::MIN.to_string()}})), v(&["mn"]));
        assert_eq!(ids(&node, "nums", json!({"term": {"n": 1.5}})), Vec::<String>::new());
        assert_eq!(ids(&node, "nums", json!({"terms": {"n": [P + 1, i64::MAX]}})), v(&["b", "mx"]));
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"gt": P, "lt": P + 3}}})), v(&["b"]));
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"gte": (P + 1).to_string(), "lte": P + 1}}})), v(&["b"]));
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"gt": i64::MAX}}})), Vec::<String>::new());
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"lt": i64::MIN}}})), Vec::<String>::new());
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"gte": i64::MAX}}})), v(&["mx"]));
        assert_eq!(ids(&node, "nums", json!({"range": {"n": {"gt": 1.5, "lte": P + 3}}})), v(&["a", "b", "s"]));
        let r = esql(&node, "FROM nums | KEEP n | SORT n");
        let got: Vec<Val> = r.table.rows.iter().map(|row| row[0].clone()).collect();
        assert_eq!(got, vec![Val::Long(i64::MIN), Val::Long(P), Val::Long(P + 1), Val::Long(P + 3), Val::Long(i64::MAX)]);
        let r = esql(&node, "FROM nums | WHERE n == 9007199254740993 | KEEP n");
        assert_eq!(r.table.rows, vec![vec![Val::Long(P + 1)]]);
    }

    /// Review finding 182: unsigned_long values beyond i64 keep their exact
    /// identity through projection, grouping and string round trips;
    /// negative and out-of-range conversions are null with a warning.
    #[test]
    fn unsigned_longs_keep_their_full_range() {
        let node = node_with("u", json!({"v": {"type": "unsigned_long"}}), &[
            ("a", json!({"v": 9_223_372_036_854_775_808u64})),
            ("b", json!({"v": 9_223_372_036_854_775_809u64})),
            ("c", json!({"v": "9223372036854775809"})),
            ("m", json!({"v": u64::MAX})),
            ("s", json!({"v": 5})),
        ]);
        let r = esql(&node, "FROM u | STATS c = COUNT(*) BY v | SORT v");
        let vi = r.table.cols.iter().position(|c| c.name == "v").unwrap();
        let ci = r.table.cols.iter().position(|c| c.name == "c").unwrap();
        let got: Vec<(Val, Val)> = r.table.rows.iter().map(|row| (row[vi].clone(), row[ci].clone())).collect();
        assert_eq!(got, vec![
            (Val::Long(5), Val::Long(1)),
            (Val::ULong(9_223_372_036_854_775_808), Val::Long(1)),
            (Val::ULong(9_223_372_036_854_775_809), Val::Long(2)),
            (Val::ULong(u64::MAX), Val::Long(1)),
        ]);
        assert_eq!(crate::esql::format::cell_json(&Val::ULong(u64::MAX), "unsigned_long"), json!(u64::MAX));
        let r = esql(&node, "FROM u | WHERE v > 9223372036854775808 | EVAL s = TO_STRING(v), back = TO_UNSIGNED_LONG(s) | KEEP v, s, back | SORT v");
        assert_eq!(r.table.rows.len(), 3);
        for row in &r.table.rows {
            assert_eq!(row[0], row[2], "{row:?}");
            assert_eq!(row[1], Val::Str(row[0].to_text()));
        }
        assert_eq!(to_unsigned_long(&Val::Str("18446744073709551615".into())).unwrap(), Val::ULong(u64::MAX));
        assert!(to_unsigned_long(&Val::Long(-1)).unwrap_err().reason.contains("out of [unsigned_long] range"));
        assert!(to_unsigned_long(&Val::Str("18446744073709551616".into())).unwrap_err().reason.contains("out of [unsigned_long] range"));
        let r = esql(&node, "ROW x = TO_UNSIGNED_LONG(-1)");
        assert_eq!(r.table.rows, vec![vec![Val::Null]]);
        assert!(r.warnings.iter().any(|w| w.contains("out of [unsigned_long] range")), "{:?}", r.warnings);
    }

    /// A field mapped `date` in one index and `date_nanos` in another is one
    /// `date_nanos` column (implicit casting): the millisecond index's values
    /// widen, so they group with the other index's (one BUCKET row, as
    /// logsdb's rollover from date to date_nanos expects), and nanoseconds
    /// render as ES's `strict_date_optional_time_nanos` (3 to 9 digits).
    #[test]
    fn date_and_date_nanos_indices_union_to_one_nanos_column() {
        let node = node_with("dms", json!({"t": {"type": "date"}}), &[("a", json!({"t": "2025-09-09T14:00:00Z"}))]);
        node.create_index("dns", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": {"properties": {"t": {"type": "date_nanos"}}}})).unwrap();
        let svc = node.get_index("dns").unwrap();
        for r in svc.write(vec![crate::index::WriteReq::Index { id: Some("b".into()), source: json!({"t": "2025-09-09T14:00:01.000000500Z"}).to_string(), routing: None, cond: Default::default() }], true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let r = esql(&node, "FROM dms,dns | WHERE t >= \"2025-09-09\" | STATS min = MIN(t) BY day = BUCKET(t, 1 day) | LIMIT 10");
        assert_eq!(r.table.rows.len(), 1, "{:?}", r.table.rows);
        let texts: Vec<String> = r.table.rows[0].iter().map(|v| v.to_text()).collect();
        assert_eq!(texts, vec!["2025-09-09T14:00:00.000Z", "2025-09-09T00:00:00.000Z"]);
        let r = esql(&node, "FROM dms,dns | SORT t | KEEP t");
        let texts: Vec<String> = r.table.rows.iter().map(|row| row[0].to_text()).collect();
        assert_eq!(texts, vec!["2025-09-09T14:00:00.000Z", "2025-09-09T14:00:01.0000005Z"]);
        assert!(r.table.rows.iter().all(|row| matches!(row[0], Val::Nanos(_))), "{:?}", r.table.rows);
    }

    /// Review finding 183: date_nanos keeps nanoseconds in ES|QL: adjacent
    /// nanoseconds stay distinct (grouping, ordering, filtering, output),
    /// TO_DATE_NANOS reads a long as epoch nanoseconds and scales a
    /// millisecond date, and string round trips are exact.
    #[test]
    fn date_nanos_keep_nanosecond_resolution() {
        let node = node_with("dn", json!({"t": {"type": "date_nanos"}}), &[
            ("a", json!({"t": "2023-01-01T00:00:00.000000001Z"})),
            ("b", json!({"t": "2023-01-01T00:00:00.000000002Z"})),
            ("c", json!({"t": "2023-01-01T00:00:00.000000002Z"})),
        ]);
        let r = esql(&node, "FROM dn | STATS c = COUNT(*) BY t | SORT t");
        let ti = r.table.cols.iter().position(|c| c.name == "t").unwrap();
        let got: Vec<String> = r.table.rows.iter().map(|row| row[ti].to_text()).collect();
        assert_eq!(got, vec!["2023-01-01T00:00:00.000000001Z", "2023-01-01T00:00:00.000000002Z"]);
        let r = esql(&node, "FROM dn | WHERE t > \"2023-01-01T00:00:00.000000001Z\" | STATS c = COUNT(*)");
        assert_eq!(r.table.rows[0][0], Val::Long(2));
        let r = esql(&node, "FROM dn | SORT t DESC | LIMIT 1 | EVAL s = TO_STRING(t), back = TO_DATE_NANOS(s), l = TO_LONG(t) | KEEP t, s, back, l");
        let row = &r.table.rows[0];
        assert_eq!(row[1], Val::Str("2023-01-01T00:00:00.000000002Z".into()));
        assert_eq!(row[0], row[2]);
        assert_eq!(row[3], Val::Long(1_672_531_200_000_000_002));
        assert_eq!(to_date_nanos(&Val::Long(1_672_531_200_000_000_001)).unwrap(), Val::Nanos(1_672_531_200_000_000_001));
        assert_eq!(to_date_nanos(&Val::Date(1_672_531_200_001)).unwrap(), Val::Nanos(1_672_531_200_001_000_000));
        assert!(to_date_nanos(&Val::Long(-1)).is_err());
        let r = esql(&node, "ROW x = TO_DATE_NANOS(1672531200000000001)");
        assert_eq!(r.table.rows[0][0].to_text(), "2023-01-01T00:00:00.000000001Z");
    }
}
