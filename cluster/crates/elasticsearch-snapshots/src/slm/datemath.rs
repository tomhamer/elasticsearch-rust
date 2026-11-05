//! Date-math names (`<daily-snap-{now/d}>`), as resolved by ES's
//! `DateMathExpressionResolver`: static text plus `{now[±N<unit>][/<unit>][{format[|tz]}]}`
//! sections; `\{` / `\}` escape braces. Default format `yyyy.MM.dd`, UTC.

use crate::error::{Result, SnapshotError};
use chrono::{DateTime, Datelike, Duration, FixedOffset, Months, NaiveDate, TimeZone, Timelike, Utc};

fn bad(expr: &str, msg: &str) -> SnapshotError {
    SnapshotError::new("elasticsearch_parse_exception", 400, format!("invalid date math expression [{expr}]: {msg}"))
}

/// Resolve `<...>` expressions; other names are returned unchanged.
pub fn resolve(name: &str, now_millis: i64) -> Result<String> {
    let Some(inner) = name.strip_prefix('<').and_then(|n| n.strip_suffix('>')) else {
        return Ok(name.to_string());
    };
    let chars: Vec<char> = inner.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 1;
                match chars.get(i) {
                    Some(c) => out.push(*c),
                    None => return Err(bad(name, "invalid escape at the end")),
                }
                i += 1;
            }
            '{' => {
                // Find the matching close brace (one nested level for the format).
                let mut depth = 0;
                let mut j = i;
                loop {
                    match chars.get(j) {
                        Some('{') => depth += 1,
                        Some('}') => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        Some(_) => {}
                        None => return Err(bad(name, "missing closing `}` for date math format")),
                    }
                    j += 1;
                }
                let section: String = chars[i + 1..j].iter().collect();
                out.push_str(&resolve_section(name, &section, now_millis)?);
                i = j + 1;
            }
            '}' => return Err(bad(name, &format!("invalid character in placeholder at position [{i}]"))),
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

fn resolve_section(name: &str, section: &str, now_millis: i64) -> Result<String> {
    let (math, format, tz) = match section.find('{') {
        Some(p) => {
            let fmt_part = section[p + 1..].strip_suffix('}').ok_or_else(|| bad(name, "missing closing `}` for date math format"))?;
            let (fmt, tz) = match fmt_part.split_once('|') {
                Some((f, t)) => (f.to_string(), Some(t.to_string())),
                None => (fmt_part.to_string(), None),
            };
            let fmt = if fmt.is_empty() { "yyyy.MM.dd".to_string() } else { fmt };
            (section[..p].to_string(), fmt, tz)
        }
        None => (section.to_string(), "yyyy.MM.dd".to_string(), None),
    };
    let offset = match tz.as_deref() {
        None | Some("UTC") | Some("Z") | Some("Etc/UTC") | Some("GMT") => FixedOffset::east_opt(0).unwrap(),
        Some(t) => parse_offset(t).ok_or_else(|| bad(name, &format!("unsupported time zone [{t}]")))?,
    };
    let now = Utc.timestamp_millis_opt(now_millis).single().ok_or_else(|| bad(name, "invalid time"))?.with_timezone(&offset);
    let t = eval_math(name, &math, now)?;
    format_java(name, &format, t)
}

fn parse_offset(t: &str) -> Option<FixedOffset> {
    let (sign, rest) = match t.chars().next()? {
        '+' => (1, &t[1..]),
        '-' => (-1, &t[1..]),
        _ => return None,
    };
    let (h, m) = match rest.split_once(':') {
        Some((h, m)) => (h.parse::<i32>().ok()?, m.parse::<i32>().ok()?),
        None if rest.len() == 4 => (rest[..2].parse().ok()?, rest[2..].parse().ok()?),
        None => (rest.parse().ok()?, 0),
    };
    FixedOffset::east_opt(sign * (h * 3600 + m * 60))
}

fn eval_math(name: &str, math: &str, now: DateTime<FixedOffset>) -> Result<DateTime<FixedOffset>> {
    let rest = math.strip_prefix("now").ok_or_else(|| bad(name, "date math must start with `now`"))?;
    let mut t = now;
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let op = chars[i];
        i += 1;
        let mut num = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            num.push(chars[i]);
            i += 1;
        }
        let unit = *chars.get(i).ok_or_else(|| bad(name, "truncated date math"))?;
        i += 1;
        match op {
            '/' => {
                if !num.is_empty() {
                    return Err(bad(name, "rounding `/` can only be used on single unit types"));
                }
                t = round_down(name, t, unit)?;
            }
            '+' | '-' => {
                let n: i64 = if num.is_empty() { 1 } else { num.parse().map_err(|_| bad(name, "bad number"))? };
                let n = if op == '-' { -n } else { n };
                t = add(name, t, n, unit)?;
            }
            _ => return Err(bad(name, &format!("operator not supported for date math [{op}]"))),
        }
    }
    Ok(t)
}

fn add(name: &str, t: DateTime<FixedOffset>, n: i64, unit: char) -> Result<DateTime<FixedOffset>> {
    let months = |m: i64| {
        if m >= 0 { t.checked_add_months(Months::new(m as u32)) } else { t.checked_sub_months(Months::new((-m) as u32)) }
    };
    let r = match unit {
        'y' => months(n * 12),
        'M' => months(n),
        'w' => Some(t + Duration::weeks(n)),
        'd' => Some(t + Duration::days(n)),
        'h' | 'H' => Some(t + Duration::hours(n)),
        'm' => Some(t + Duration::minutes(n)),
        's' => Some(t + Duration::seconds(n)),
        _ => return Err(bad(name, &format!("unit [{unit}] not supported for date math"))),
    };
    r.ok_or_else(|| bad(name, "date out of range"))
}

fn round_down(name: &str, t: DateTime<FixedOffset>, unit: char) -> Result<DateTime<FixedOffset>> {
    let tz = *t.offset();
    let d = t.date_naive();
    let mk = |date: NaiveDate, h: u32, mi: u32, s: u32| tz.from_local_datetime(&date.and_hms_opt(h, mi, s).unwrap()).single().unwrap();
    Ok(match unit {
        'y' => mk(NaiveDate::from_ymd_opt(d.year(), 1, 1).unwrap(), 0, 0, 0),
        'M' => mk(NaiveDate::from_ymd_opt(d.year(), d.month(), 1).unwrap(), 0, 0, 0),
        'w' => mk(d - Duration::days(d.weekday().num_days_from_monday() as i64), 0, 0, 0),
        'd' => mk(d, 0, 0, 0),
        'h' | 'H' => mk(d, t.hour(), 0, 0),
        'm' => mk(d, t.hour(), t.minute(), 0),
        's' => mk(d, t.hour(), t.minute(), t.second()),
        _ => return Err(bad(name, &format!("unit [{unit}] not supported for date math"))),
    })
}

/// Java `DateTimeFormatter` pattern subset → formatted string.
fn format_java(name: &str, pattern: &str, t: DateTime<FixedOffset>) -> Result<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            let end = chars[i + 1..].iter().position(|x| *x == '\'').ok_or_else(|| bad(name, "unterminated quote in format"))?;
            if end == 0 {
                out.push('\'');
            } else {
                out.extend(&chars[i + 1..i + 1 + end]);
            }
            i += end + 2;
            continue;
        }
        if !c.is_ascii_alphabetic() {
            out.push(c);
            i += 1;
            continue;
        }
        let mut n = 1;
        while i + n < chars.len() && chars[i + n] == c {
            n += 1;
        }
        let pad = |v: i64, w: usize| format!("{v:0w$}");
        let s = match (c, n) {
            ('y' | 'u', 2) => pad((t.year() % 100) as i64, 2),
            ('y' | 'u', _) => pad(t.year() as i64, n.max(4).min(9)),
            ('M', 1) => t.month().to_string(),
            ('M', 2) => pad(t.month() as i64, 2),
            ('M', 3) => t.format("%b").to_string(),
            ('M', _) => t.format("%B").to_string(),
            ('d', _) => pad(t.day() as i64, n),
            ('D', _) => pad(t.ordinal() as i64, n),
            ('H', _) => pad(t.hour() as i64, n),
            ('h', _) => pad(((t.hour() + 11) % 12 + 1) as i64, n),
            ('m', _) => pad(t.minute() as i64, n),
            ('s', _) => pad(t.second() as i64, n),
            ('S', _) => {
                let ms = format!("{:09}", t.nanosecond());
                ms[..n.min(9)].to_string()
            }
            ('a', _) => t.format("%p").to_string(),
            ('E', 1..=3) => t.format("%a").to_string(),
            ('E', _) => t.format("%A").to_string(),
            ('e', _) => pad(t.weekday().number_from_monday() as i64, n),
            ('w', _) => pad(t.iso_week().week() as i64, n),
            ('Z', _) => t.format("%z").to_string(),
            ('X', _) | ('x', _) => {
                if t.offset().local_minus_utc() == 0 && c == 'X' { "Z".to_string() } else { t.format("%:z").to_string() }
            }
            _ => return Err(bad(name, &format!("unsupported date format pattern letter [{c}]"))),
        };
        out.push_str(&s);
        i += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(s: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis()
    }

    #[test]
    fn resolves() {
        let now = ms("2024-03-15T13:45:30Z");
        assert_eq!(resolve("plain", now).unwrap(), "plain");
        assert_eq!(resolve("<daily-snap-{now/d}>", now).unwrap(), "daily-snap-2024.03.15");
        assert_eq!(resolve("<snap-{now}>", now).unwrap(), "snap-2024.03.15");
        assert_eq!(resolve("<m-{now/M}>", now).unwrap(), "m-2024.03.01");
        assert_eq!(resolve("<y-{now/y{yyyy}}>", now).unwrap(), "y-2024");
        assert_eq!(resolve("<x-{now-1d/d}>", now).unwrap(), "x-2024.03.14");
        assert_eq!(resolve("<x-{now+1M{yyyy-MM}}>", now).unwrap(), "x-2024-04");
        assert_eq!(resolve("<h-{now{yyyy.MM.dd.HH.mm}}>", now).unwrap(), "h-2024.03.15.13.45");
        assert_eq!(resolve("<tz-{now/d{yyyy.MM.dd|+12:00}}>", now).unwrap(), "tz-2024.03.16");
        assert_eq!(resolve("<esc-\\{x\\}-{now/d}>", now).unwrap(), "esc-{x}-2024.03.15");
        assert!(resolve("<bad-{then}>", now).is_err());
        assert!(resolve("<bad-{now/d{yyyy|Mars/Base}}>", now).is_err());
        assert!(resolve("<bad-{now/d>", now).is_err());
    }
}
