//! ES SQL's TIME and INTERVAL_* types (ES 9.5.5 `SqlDataTypes`,
//! `SqlDataTypeConverter`, `Intervals`, `StringUtils`).
//!
//! * A TIME value is held as its text, `HH:mm:ss.SSS` and its offset (`Z`
//!   or `+hh:mm`), what ES renders (`ISO_TIME_WITH_MILLIS`).
//! * An interval value is held as its ISO-8601 text, as ES renders it to
//!   JSON clients (`java.time.Period.toString`: `P1Y2M`;
//!   `Duration.toString`: `PT26H3M4.5S`); text formats show ES's SQL form
//!   (`StringUtils.toString`: `+1-2`, `+1 02:03:04.5`), see [`sql_text`].
//!
//! Conversions follow `SqlDataTypeConverter.converterFor`: TIME from
//! datetime / date (the time of day in its zone), integers and rationals
//! (`millis % day`), booleans and strings (`ISO_LOCAL_TIME_OPTIONAL_TZ`);
//! TIME to strings, numbers (millis of the day) and booleans; intervals to
//! themselves and to strings (their SQL form) only. Any other pair is
//! ES's `Cannot cast [FROM] to [TO]` verification error.

use crate::error::{EsError, Result};

pub const DAY_MILLIS: i64 = 86_400_000;

/// The interval type names (`interval_year` ..), ES order.
pub const INTERVAL_TYPES: &[&str] = &[
    "interval_year", "interval_month", "interval_day", "interval_hour", "interval_minute", "interval_second", "interval_year_to_month", "interval_day_to_hour",
    "interval_day_to_minute", "interval_day_to_second", "interval_hour_to_minute", "interval_hour_to_second", "interval_minute_to_second",
];

pub fn is_interval(ty: &str) -> bool {
    INTERVAL_TYPES.contains(&ty)
}

/// The internal cast an interval literal is bound with (its value already
/// the interval): never ES's user-facing CAST checks.
pub const LITERAL_PREFIX: &str = "\u{0}lit:";

fn offset_text(secs: i32) -> String {
    if secs == 0 {
        return "Z".into();
    }
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.abs();
    let (h, m, s) = (a / 3600, (a % 3600) / 60, a % 60);
    if s != 0 {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    } else {
        format!("{sign}{h:02}:{m:02}")
    }
}

/// A TIME of `millis` within the day (any value, wrapped into the day)
/// at offset `offset_secs`.
pub fn time_text(millis: i64, offset_secs: i32) -> String {
    let m = millis.rem_euclid(DAY_MILLIS);
    let (h, mi, s, ms) = (m / 3_600_000, (m / 60_000) % 60, (m / 1000) % 60, m % 1000);
    format!("{h:02}:{mi:02}:{s:02}.{ms:03}{}", offset_text(offset_secs))
}

/// `DateUtils.asTimeOnly(long)`: `millis % day` (Java's remainder: a
/// negative value is the previous day's time), at UTC.
pub fn time_of_long(millis: i64) -> String {
    time_text(millis % DAY_MILLIS, 0)
}

/// The time of day of the instant `millis` in the session zone
/// (`ZonedDateTime.toOffsetDateTime().toOffsetTime()`).
pub fn time_of_datetime(millis: i64) -> String {
    let off = crate::esql::zone().offset_at(millis);
    time_text(millis + off as i64 * 1000, off)
}

/// A TIME's millis since midnight UTC (`OffsetTime.atDate(EPOCH).toInstant()`),
/// from its text.
pub fn time_millis(text: &str) -> Option<i64> {
    let (local, offset) = split_offset(text)?;
    let local = parse_local_time(local)?;
    Some(local - offset as i64 * 1000)
}

fn split_offset(s: &str) -> Option<(&str, i32)> {
    if let Some(l) = s.strip_suffix('Z').or_else(|| s.strip_suffix('z')) {
        return Some((l, 0));
    }
    if let Some(i) = s.rfind(['+', '-']) {
        let (l, o) = s.split_at(i);
        let sign = if o.starts_with('-') { -1 } else { 1 };
        let parts: Vec<&str> = o[1..].split(':').collect();
        let num = |x: &str| -> Option<i32> { (x.len() == 2 && x.bytes().all(|b| b.is_ascii_digit())).then(|| x.parse().ok()).flatten() };
        let secs = match parts.as_slice() {
            [h] if h.len() == 2 => num(h)? * 3600,
            [h] if h.len() == 4 => num(&h[..2])? * 3600 + num(&h[2..])? * 60,
            [h, m] => num(h)? * 3600 + num(m)? * 60,
            [h, m, sec] => num(h)? * 3600 + num(m)? * 60 + num(sec)?,
            _ => return None,
        };
        if secs > 18 * 3600 {
            return None;
        }
        return Some((l, sign * secs));
    }
    Some((s, 0))
}

/// `HH:mm[:ss[.fraction]]` (`ISO_LOCAL_TIME`) as millis of the day.
fn parse_local_time(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let two = |i: usize| -> Option<i64> { (b.len() >= i + 2 && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit()).then(|| ((b[i] - b'0') * 10 + (b[i + 1] - b'0')) as i64) };
    let h = two(0)?;
    if b.get(2) != Some(&b':') {
        return None;
    }
    let m = two(3)?;
    let mut ms = 0;
    let mut sec = 0;
    if b.len() > 5 {
        if b[5] != b':' {
            return None;
        }
        sec = two(6)?;
        if b.len() > 8 {
            if b[8] != b'.' || b.len() == 9 || b.len() > 18 || !b[9..].iter().all(u8::is_ascii_digit) {
                return None;
            }
            let frac = &s[9..];
            let digits: String = frac.chars().chain(std::iter::repeat('0')).take(3).collect();
            ms = digits.parse::<i64>().ok()?;
        }
    }
    (h < 24 && m < 60 && sec < 60).then_some(((h * 60 + m) * 60 + sec) * 1000 + ms)
}

/// `DateUtils.asTimeOnly(String)` (`ISO_LOCAL_TIME_OPTIONAL_TZ`, UTC by
/// default), or ES's cast error.
pub fn time_of_string(s: &str) -> Result<String> {
    let bad = |at: usize| EsError::new(400, "invalid_argument_exception", format!("cannot cast [{s}] to [time]: Text '{s}' could not be parsed at index {at}"));
    let (local, offset) = split_offset(s).ok_or_else(|| bad(s.len().min(8)))?;
    let millis = parse_local_time(local).ok_or_else(|| bad(first_bad(local)))?;
    Ok(time_text(millis, offset))
}

fn first_bad(local: &str) -> usize {
    let pattern = b"00:00:00";
    local.bytes().zip(pattern.iter()).position(|(c, p)| if *p == b'0' { !c.is_ascii_digit() } else { c != *p }).unwrap_or(local.len().min(8))
}

// ------------------------------------------------------------- intervals

/// A year-month interval as `Period.toString` prints it.
pub fn period_iso(years: i64, months: i64) -> String {
    if years == 0 && months == 0 {
        return "P0D".into();
    }
    let mut s = String::from("P");
    if years != 0 {
        s.push_str(&format!("{years}Y"));
    }
    if months != 0 {
        s.push_str(&format!("{months}M"));
    }
    s
}

/// A day-time interval of `millis` as `Duration.toString` prints it.
pub fn duration_iso(millis: i64) -> String {
    if millis == 0 {
        return "PT0S".into();
    }
    let seconds = millis.div_euclid(1000);
    let nanos = millis.rem_euclid(1000) * 1_000_000;
    let effective = if seconds < 0 && nanos > 0 { seconds + 1 } else { seconds };
    let hours = effective / 3600;
    let minutes = (effective % 3600) / 60;
    let secs = effective % 60;
    let mut buf = String::from("PT");
    if hours != 0 {
        buf.push_str(&format!("{hours}H"));
    }
    if minutes != 0 {
        buf.push_str(&format!("{minutes}M"));
    }
    if secs == 0 && nanos == 0 && buf.len() > 2 {
        return buf;
    }
    if seconds < 0 && nanos > 0 {
        if secs == 0 {
            buf.push_str("-0");
        } else {
            buf.push_str(&secs.to_string());
        }
    } else {
        buf.push_str(&secs.to_string());
    }
    if nanos > 0 {
        let pos = buf.len();
        let frac = if seconds < 0 { 2 * 1_000_000_000 - nanos } else { nanos + 1_000_000_000 };
        buf.push_str(&frac.to_string());
        while buf.ends_with('0') {
            buf.pop();
        }
        buf.replace_range(pos..pos + 1, ".");
    }
    buf.push('S');
    buf
}

/// An interval's ISO text read back: year-month `(years, months)` or
/// day-time millis.
pub enum Amount {
    Period(i64, i64),
    Millis(i64),
}

pub fn parse_iso(s: &str) -> Option<Amount> {
    let body = s.strip_prefix('P')?;
    if let Some(t) = body.strip_prefix('T') {
        // Duration: [nH][nM][n[.f]S]
        let mut millis: i64 = 0;
        let mut num = String::new();
        for c in t.chars() {
            match c {
                'H' => { millis += num.parse::<i64>().ok()? * 3_600_000; num.clear(); }
                'M' => { millis += num.parse::<i64>().ok()? * 60_000; num.clear(); }
                'S' => {
                    let (int, frac) = num.split_once('.').unwrap_or((&num, ""));
                    let neg = int.starts_with('-');
                    let whole: i64 = int.parse().ok()?;
                    let f: i64 = if frac.is_empty() { 0 } else { frac.chars().chain(std::iter::repeat('0')).take(3).collect::<String>().parse().ok()? };
                    millis += whole * 1000 + if neg { -f } else { f };
                    num.clear();
                }
                _ => num.push(c),
            }
        }
        return Some(Amount::Millis(millis));
    }
    let (mut y, mut m) = (0, 0);
    let mut num = String::new();
    for c in body.chars() {
        match c {
            'Y' => { y = num.parse().ok()?; num.clear(); }
            'M' => { m = num.parse().ok()?; num.clear(); }
            'D' => { num.clear(); }
            _ => num.push(c),
        }
    }
    Some(Amount::Period(y, m))
}

/// `-interval`.
pub fn negate_iso(s: &str) -> Option<String> {
    Some(match parse_iso(s)? {
        Amount::Period(y, m) => period_iso(-y, -m),
        Amount::Millis(ms) => duration_iso(-ms),
    })
}

/// ES SQL's text form of an interval (`StringUtils.toString`): `+y-m` for
/// year-month, `+d hh:mm:ss[.f]` for day-time ones.
pub fn sql_text(iso: &str) -> Option<String> {
    Some(match parse_iso(iso)? {
        Amount::Period(y, m) => {
            // `Period.isNegative`: any negative part; negated as a whole.
            let neg = y < 0 || m < 0;
            let (y, m) = if neg { (-y, -m) } else { (y, m) };
            format!("{}{y}-{m}", if neg { '-' } else { '+' })
        }
        Amount::Millis(ms) => {
            let neg = ms < 0;
            let ms = ms.abs();
            let secs = ms / 1000;
            let (d, h, mi, s) = (secs / 86_400, (secs % 86_400) / 3600, (secs % 3600) / 60, secs % 60);
            let mut out = format!("{}{d} {h:02}:{mi:02}:{s:02}", if neg { '-' } else { '+' });
            let mut frac = ms % 1000;
            if frac > 0 {
                while frac % 10 == 0 {
                    frac /= 10;
                }
                out.push_str(&format!(".{frac}"));
            }
            out
        }
    })
}

/// Interval units (`Intervals.TimeUnit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

impl Unit {
    pub fn parse(word: &str) -> Option<Unit> {
        Some(match word.to_ascii_uppercase().as_str() {
            "YEAR" | "YEARS" => Unit::Year,
            "MONTH" | "MONTHS" => Unit::Month,
            "DAY" | "DAYS" => Unit::Day,
            "HOUR" | "HOURS" => Unit::Hour,
            "MINUTE" | "MINUTES" => Unit::Minute,
            "SECOND" | "SECONDS" => Unit::Second,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Unit::Year => "YEAR",
            Unit::Month => "MONTH",
            Unit::Day => "DAY",
            Unit::Hour => "HOUR",
            Unit::Minute => "MINUTE",
            Unit::Second => "SECOND",
        }
    }

    fn millis(self) -> i64 {
        match self {
            Unit::Day => DAY_MILLIS,
            Unit::Hour => 3_600_000,
            Unit::Minute => 60_000,
            Unit::Second => 1000,
            _ => 0,
        }
    }
}

/// `Intervals.intervalType` (the declaration already checked).
pub fn interval_type(leading: Unit, trailing: Option<Unit>) -> std::result::Result<&'static str, String> {
    Ok(match (leading, trailing) {
        (Unit::Year, None) => "interval_year",
        (Unit::Month, None) => "interval_month",
        (Unit::Day, None) => "interval_day",
        (Unit::Hour, None) => "interval_hour",
        (Unit::Minute, None) => "interval_minute",
        (Unit::Second, None) => "interval_second",
        (Unit::Year, Some(Unit::Month)) => "interval_year_to_month",
        (Unit::Day, Some(Unit::Hour)) => "interval_day_to_hour",
        (Unit::Day, Some(Unit::Minute)) => "interval_day_to_minute",
        (Unit::Day, Some(Unit::Second)) => "interval_day_to_second",
        (Unit::Hour, Some(Unit::Minute)) => "interval_hour_to_minute",
        (Unit::Hour, Some(Unit::Second)) => "interval_hour_to_second",
        (Unit::Minute, Some(Unit::Second)) => "interval_minute_to_second",
        (l, Some(t)) => return Err(format!("Cannot determine datatype for combination [{}] [{}]", l.name(), t.name())),
    })
}

/// `ExpressionBuilder.visitInterval`'s declaration checks: only YEAR TO
/// MONTH, or a trailing unit smaller than the leading one.
pub fn check_declaration(leading: Unit, trailing: Option<Unit>) -> std::result::Result<(), String> {
    let Some(t) = trailing else { return Ok(()) };
    if leading == Unit::Year && t != Unit::Month {
        return Err(format!("Invalid interval declaration; YEAR trailing unit required to be MONTH, received {}", t.name()));
    }
    if t <= leading {
        let all = [Unit::Year, Unit::Month, Unit::Day, Unit::Hour, Unit::Minute, Unit::Second];
        let range: Vec<&str> = all.iter().filter(|u| **u > leading).map(|u| u.name()).collect();
        return Err(format!(
            "Invalid interval declaration; trailing unit [{}] needs to be smaller than leading unit[{}], expected one of [{}]",
            t.name(),
            leading.name(),
            range.join(", ")
        ));
    }
    Ok(())
}

/// `Intervals.of(duration, unit)`: a numeric interval literal.
pub fn of_number(v: i64, unit: Unit) -> std::result::Result<String, String> {
    let too_large = || format!("Value [{v}] cannot be used as it is too large to convert into [{}]s", unit.name());
    Ok(match unit {
        Unit::Year => period_iso(v, 0),
        Unit::Month => period_iso(0, v),
        u => duration_iso(v.checked_mul(u.millis()).ok_or_else(too_large)?),
    })
}

/// `Intervals.parseInterval`: the string form of an interval of type `ty`
/// (each type its own pattern, ES's messages).
pub fn parse_pattern(s: &str, ty: &str) -> std::result::Result<String, String> {
    // (unit, max) for numbers; separators as chars; `optional` from the
    // first optional token on.
    enum T {
        N(Unit, i64),
        Ms,
        C(char),
    }
    let (tokens, opt_from): (Vec<T>, usize) = match ty {
        "interval_year" => (vec![T::N(Unit::Year, 0)], usize::MAX),
        "interval_month" => (vec![T::N(Unit::Month, 0)], usize::MAX),
        "interval_day" => (vec![T::N(Unit::Day, 0)], usize::MAX),
        "interval_hour" => (vec![T::N(Unit::Hour, 0)], usize::MAX),
        "interval_minute" => (vec![T::N(Unit::Minute, 0)], usize::MAX),
        "interval_second" => (vec![T::N(Unit::Second, 0), T::C('.'), T::Ms], 1),
        "interval_year_to_month" => (vec![T::N(Unit::Year, 0), T::C('-'), T::N(Unit::Month, 11)], usize::MAX),
        "interval_day_to_hour" => (vec![T::N(Unit::Day, 0), T::C(' '), T::N(Unit::Hour, 23)], usize::MAX),
        "interval_day_to_minute" => (vec![T::N(Unit::Day, 0), T::C(' '), T::N(Unit::Hour, 23), T::C(':'), T::N(Unit::Minute, 59)], usize::MAX),
        "interval_day_to_second" => (
            vec![T::N(Unit::Day, 0), T::C(' '), T::N(Unit::Hour, 23), T::C(':'), T::N(Unit::Minute, 59), T::C(':'), T::N(Unit::Second, 59), T::C('.'), T::Ms],
            7,
        ),
        "interval_hour_to_minute" => (vec![T::N(Unit::Hour, 0), T::C(':'), T::N(Unit::Minute, 59)], usize::MAX),
        "interval_hour_to_second" => (vec![T::N(Unit::Hour, 0), T::C(':'), T::N(Unit::Minute, 59), T::C(':'), T::N(Unit::Second, 59), T::C('.'), T::Ms], 5),
        "interval_minute_to_second" => (vec![T::N(Unit::Minute, 0), T::C(':'), T::N(Unit::Second, 59), T::C('.'), T::Ms], 3),
        other => return Err(format!("Cannot determine datatype for [{other}]")),
    };
    let name = ty.replace('_', " ").to_ascii_uppercase();
    let invalid = format!("Invalid [{name}] value [{s}]");
    let expecting: String = tokens.iter().map(|t| match t { T::C(c) => c.to_string(), _ => "[numeric]".into() }).collect();
    let b: Vec<char> = s.chars().collect();
    if b.is_empty() {
        return Err(format!("{invalid}: incorrect format, expecting {expecting}"));
    }
    let mut start = 0;
    let mut negate = false;
    match b[0] {
        '+' => start = 1,
        '-' => {
            start = 1;
            negate = true;
        }
        _ => {}
    }
    let year_month = ty.contains("year") || ty == "interval_month";
    let (mut years, mut months, mut millis): (i64, i64, i64) = (0, 0, 0);
    let mut end = start;
    for (i, tok) in tokens.iter().enumerate() {
        end = start;
        if start >= b.len() {
            if i >= opt_from {
                break;
            }
            return Err(format!("{invalid}: incorrect format, expecting {expecting}"));
        }
        match tok {
            T::C(c) => {
                if b[start] != *c {
                    return Err(format!("{invalid}: expected [{c}] (at [{start}]) but found [{}]", b[start]));
                }
                start += 1;
            }
            T::N(..) | T::Ms => {
                while end < b.len() && b[end].is_ascii_digit() {
                    end += 1;
                }
                if end == start {
                    return Err(format!("{invalid}: expected digit (at [{end}]) but found [{}]", b[end]));
                }
                let digits: String = b[start..end].iter().collect();
                let mut v: i64 = digits.parse().map_err(|_| format!("{invalid}"))?;
                match tok {
                    T::N(unit, max) => {
                        if *max > 0 && v > *max {
                            return Err(format!("{invalid}: [{}] unit has illegal value [{v}], expected a positive number up to [{max}]", unit.name()));
                        }
                        match unit {
                            Unit::Year => years = v,
                            Unit::Month => months = v,
                            u => millis = millis.checked_add(v.checked_mul(u.millis()).ok_or_else(|| format!("Value [{v}] cannot be used as it is too large to convert into [{}]s", u.name()))?).ok_or_else(|| invalid.clone())?,
                        }
                    }
                    _ => {
                        if v > 999 {
                            return Err(format!("{invalid}: [MILLISECOND] unit has illegal value [{v}], expected a positive number up to [999]"));
                        }
                        if digits.len() < 3 {
                            v *= if digits.len() < 2 { 100 } else { 10 };
                        }
                        millis += v;
                    }
                }
                start = end;
            }
        }
    }
    if end < b.len() {
        let rest: String = b[end..].iter().collect();
        return Err(format!("{invalid}: unexpected trailing characters found [{rest}]"));
    }
    let iso = if year_month { period_iso(years, months) } else { duration_iso(millis) };
    Ok(if negate { negate_iso(&iso).unwrap_or(iso) } else { iso })
}

/// ES's type name of a SQL type in messages (`DataType.toString`: upper case).
pub fn type_label(ty: &str) -> String {
    match ty {
        "date" => "DATETIME".into(),
        "integer" | "int" => "INTEGER".into(),
        other => other.to_ascii_uppercase(),
    }
}

/// `SqlDataTypeConverter.canConvert` for the pairs these types add
/// (`None`: not one of them, the ordinary conversions decide).
pub fn can_convert(from: &str, to: &str) -> Option<bool> {
    let interval_from = is_interval(from);
    let interval_to = is_interval(to);
    if from == to && (interval_from || from == "time") {
        return Some(true);
    }
    if interval_to {
        return Some(from == "null");
    }
    if interval_from {
        return Some(matches!(to, "keyword" | "text") || to == "null");
    }
    if to == "time" {
        return Some(matches!(from, "null" | "integer" | "long" | "short" | "byte" | "unsigned_long" | "double" | "float" | "half_float" | "scaled_float" | "boolean" | "keyword" | "text" | "date" | "datetime"));
    }
    if from == "time" {
        return Some(matches!(to, "keyword" | "text" | "long" | "integer" | "short" | "byte" | "unsigned_long" | "double" | "float" | "half_float" | "scaled_float" | "boolean" | "null"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_and_sql_forms_match_java() {
        assert_eq!(period_iso(1, 2), "P1Y2M");
        assert_eq!(period_iso(0, 14), "P14M");
        assert_eq!(period_iso(-1, -2), "P-1Y-2M");
        assert_eq!(duration_iso(26 * 3_600_000 + 3 * 60_000 + 4_500), "PT26H3M4.5S");
        assert_eq!(duration_iso(-90_000), "PT-1M-30S");
        assert_eq!(duration_iso(-500), "PT-0.5S");
        assert_eq!(duration_iso(86_400_000), "PT24H");
        assert_eq!(sql_text("P1Y2M").unwrap(), "+1-2");
        assert_eq!(sql_text("P-1Y-2M").unwrap(), "-1-2");
        assert_eq!(sql_text("PT26H3M4.5S").unwrap(), "+1 02:03:04.5");
        assert_eq!(sql_text("PT-1M-30S").unwrap(), "-0 00:01:30");
        assert_eq!(time_of_long(45_296_789), "12:34:56.789Z");
        assert_eq!(time_millis("12:34:56.789+01:00"), Some(45_296_789 - 3_600_000));
    }
}
