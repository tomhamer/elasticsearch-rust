//! Dates: ES/Java date formats, time zones, date math and calendar rounding.
//!
//! Supports the built-in named formats ES clients use in practice
//! (`strict_date_optional_time`, `epoch_millis`, `epoch_second`,
//! `basic_date`, `date_hour_minute_second`, ...), `||`-separated format
//! lists, and custom `java.time` patterns (`yyyy-MM-dd HH:mm:ss.SSS`,
//! `dd/MMM/yyyy:HH:mm:ss Z`, optional sections `[...]`, quoted literals).
//! Values are epoch milliseconds (`i64`).

use crate::error::{EsError, Result};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime, NaiveTime, Offset, TimeZone, Timelike, Utc};

/// The ES default date format.
pub const DEFAULT_FORMAT: &str = "strict_date_optional_time||epoch_millis";

/// A time zone: fixed offset or IANA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tz {
    Fixed(i32),
    Named(chrono_tz::Tz),
}

impl Default for Tz {
    fn default() -> Self {
        Tz::Fixed(0)
    }
}

impl Tz {
    pub const UTC: Tz = Tz::Fixed(0);

    /// [`Tz::parse`] as a query `time_zone` option: ES wraps the JDK
    /// `ZoneRulesException` in an `IllegalArgumentException`.
    pub fn parse_query_option(s: &str) -> Result<Tz> {
        Tz::parse(s).map_err(|e| {
            if e.reason.starts_with("Unknown time-zone ID") {
                EsError::illegal_argument(format!("java.time.zone.ZoneRulesException: {}", e.reason)).caused_by(EsError::new(400, "zone_rules_exception", e.reason))
            } else {
                e
            }
        })
    }

    /// Parse `Z`, `UTC`, `+01:00`, `-0500`, `+3`, or an IANA id.
    pub fn parse(s: &str) -> Result<Tz> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("z") || t.eq_ignore_ascii_case("utc") || t == "GMT" || t.eq_ignore_ascii_case("etc/utc") {
            return Ok(Tz::UTC);
        }
        if let Some(off) = parse_offset_str(t) {
            return Ok(Tz::Fixed(off));
        }
        let rest = t.strip_prefix("UTC").or_else(|| t.strip_prefix("GMT"));
        if let Some(r) = rest {
            if let Some(off) = parse_offset_str(r) {
                return Ok(Tz::Fixed(off));
            }
        }
        t.parse::<chrono_tz::Tz>()
            .map(Tz::Named)
            .map_err(|_| EsError::illegal_argument(format!("Unknown time-zone ID: {s}")))
    }

    /// Offset (seconds east of UTC) in effect at the UTC instant `millis`.
    pub fn offset_at(&self, millis: i64) -> i32 {
        match self {
            Tz::Fixed(o) => *o,
            Tz::Named(tz) => {
                let utc = Utc.timestamp_millis_opt(millis).single().unwrap_or_else(|| Utc.timestamp_millis_opt(0).unwrap());
                tz.offset_from_utc_datetime(&utc.naive_utc()).fix().local_minus_utc()
            }
        }
    }

    /// UTC millis of a local wall-clock time (DST gaps resolve forward,
    /// overlaps to the earlier instant, as java.time does).
    pub fn local_to_utc(&self, local: NaiveDateTime) -> i64 {
        match self {
            Tz::Fixed(o) => local.and_utc().timestamp_millis() - *o as i64 * 1000,
            Tz::Named(tz) => match tz.from_local_datetime(&local) {
                chrono::LocalResult::Single(d) => d.timestamp_millis(),
                chrono::LocalResult::Ambiguous(a, _) => a.timestamp_millis(),
                chrono::LocalResult::None => {
                    // In a gap: shift forward by the gap length.
                    let mut l = local;
                    for _ in 0..4 {
                        l += Duration::minutes(30);
                        if let chrono::LocalResult::Single(d) | chrono::LocalResult::Ambiguous(d, _) = tz.from_local_datetime(&l) {
                            return d.timestamp_millis() - 30 * 60 * 1000;
                        }
                    }
                    local.and_utc().timestamp_millis()
                }
            },
        }
    }

    /// Local wall-clock time of a UTC instant.
    pub fn to_local(&self, millis: i64) -> NaiveDateTime {
        let off = self.offset_at(millis);
        from_millis(millis + off as i64 * 1000)
    }

    /// `Z` or `+01:00` (java `XXX`) for the offset at `millis`.
    pub fn offset_string(&self, millis: i64) -> String {
        format_offset(self.offset_at(millis), true, true)
    }
}

fn parse_offset_str(t: &str) -> Option<i32> {
    let (sign, rest) = match t.as_bytes().first()? {
        b'+' => (1, &t[1..]),
        b'-' => (-1, &t[1..]),
        _ => return None,
    };
    let digits: String = rest.chars().filter(|c| *c != ':').collect();
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (h, m) = match digits.len() {
        1 | 2 => (digits.parse::<i32>().ok()?, 0),
        3 => (digits[..1].parse::<i32>().ok()?, digits[1..].parse::<i32>().ok()?),
        4 => (digits[..2].parse::<i32>().ok()?, digits[2..].parse::<i32>().ok()?),
        _ => return None,
    };
    if h > 18 || m > 59 {
        return None;
    }
    Some(sign * (h * 3600 + m * 60))
}

/// Parse a `Z`/`+HH[:MM]` offset at the end of a date string (java
/// `DateTimeFormatterBuilder.appendOffsetId` / `appendOffset`), returning
/// the offset in seconds and the number of characters consumed.
fn parse_offset_tail(s: &str) -> Option<(i32, usize)> {
    let b = s.as_bytes();
    if b.first() == Some(&b'Z') {
        return Some((0, 1));
    }
    let sign = match b.first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let rest = &s[1..];
    let rb = rest.as_bytes();
    let h = take_digits(rb, 0, 2, 2)?;
    let mut pos = h.1;
    let mut m = 0;
    if rb.get(pos) == Some(&b':') {
        let (mm, n2) = take_digits(rb, pos + 1, 2, 2)?;
        m = mm as i32;
        pos = n2;
    } else if let Some((mm, n2)) = take_digits(rb, pos, 2, 2) {
        m = mm as i32;
        pos = n2;
    }
    Some((sign * (h.0 as i32 * 3600 + m * 60), pos + 1))
}

fn format_offset(secs: i32, colon: bool, z_for_zero: bool) -> String {
    if secs == 0 && z_for_zero {
        return "Z".into();
    }
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.abs();
    if colon {
        format!("{sign}{:02}:{:02}", a / 3600, (a % 3600) / 60)
    } else {
        format!("{sign}{:02}{:02}", a / 3600, (a % 3600) / 60)
    }
}

pub fn from_millis(millis: i64) -> NaiveDateTime {
    chrono::DateTime::from_timestamp_millis(millis).map(|d| d.naive_utc()).unwrap_or_default()
}

pub fn to_millis(dt: NaiveDateTime) -> i64 {
    dt.and_utc().timestamp_millis()
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Lit(String),
    Field(char, usize),
    Optional(Vec<Tok>),
}

#[derive(Clone, Debug, PartialEq)]
enum Fmt {
    /// `strict_date_optional_time` and friends.
    Iso { strict: bool },
    EpochMillis,
    EpochSecond,
    Pattern(Vec<Tok>),
}

/// A parsed `||` list of date formats.
#[derive(Clone, Debug, PartialEq)]
pub struct DateFormatter {
    pub spec: String,
    fmts: Vec<Fmt>,
    /// French month / day names (`locale: fr`).
    fr: bool,
}

/// Allocation-free equivalent of the default formatter in UTC.
pub fn write_default_utc(millis:i64,out:&mut impl std::fmt::Write)->std::fmt::Result {
    let dt=from_millis(millis);
    write!(out,"{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",dt.year(),dt.month(),dt.day(),dt.hour(),dt.minute(),dt.second(),dt.nanosecond()/1_000_000)
}

impl Default for DateFormatter {
    fn default() -> Self {
        DateFormatter::new(DEFAULT_FORMAT).unwrap()
    }
}

fn named(name: &str) -> Option<Fmt> {
    let n = name.strip_prefix("strict_").unwrap_or(name);
    let strict = name.starts_with("strict_");
    let pat = match n {
        "date_optional_time" | "date_optional_time_nanos" | "date_time_optional" => return Some(Fmt::Iso { strict }),
        "epoch_millis" => return Some(Fmt::EpochMillis),
        "epoch_second" => return Some(Fmt::EpochSecond),
        "basic_date" => "yyyyMMdd",
        "basic_date_time" => "yyyyMMdd'T'HHmmss.SSSZ",
        "basic_date_time_no_millis" => "yyyyMMdd'T'HHmmssZ",
        "basic_time" => "HHmmss.SSSZ",
        "basic_time_no_millis" => "HHmmssZ",
        "basic_t_time" => "'T'HHmmss.SSSZ",
        "date" | "year_month_day" => "yyyy-MM-dd",
        "date_hour" => "yyyy-MM-dd'T'HH",
        "date_hour_minute" => "yyyy-MM-dd'T'HH:mm",
        "date_hour_minute_second" => "yyyy-MM-dd'T'HH:mm:ss",
        "date_hour_minute_second_fraction" | "date_hour_minute_second_millis" => "yyyy-MM-dd'T'HH:mm:ss.SSS",
        "date_time" => "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
        "date_time_no_millis" => "yyyy-MM-dd'T'HH:mm:ssXXX",
        "hour" => "HH",
        "hour_minute" => "HH:mm",
        "hour_minute_second" => "HH:mm:ss",
        "hour_minute_second_fraction" | "hour_minute_second_millis" => "HH:mm:ss.SSS",
        "time" => "HH:mm:ss.SSSXXX",
        "time_no_millis" => "HH:mm:ssXXX",
        "t_time" => "'T'HH:mm:ss.SSSXXX",
        "t_time_no_millis" => "'T'HH:mm:ssXXX",
        "year" => "yyyy",
        "year_month" => "yyyy-MM",
        "ordinal_date" => "yyyy-DDD",
        _ => return None,
    };
    Some(Fmt::Pattern(tokenize(pat).ok()?))
}

fn tokenize(p: &str) -> Result<Vec<Tok>> {
    fn inner(chars: &[char], i: &mut usize, depth: usize) -> Result<Vec<Tok>> {
        let mut out = Vec::new();
        while *i < chars.len() {
            let c = chars[*i];
            if c == '\'' {
                *i += 1;
                let mut lit = String::new();
                while *i < chars.len() {
                    if chars[*i] == '\'' {
                        if *i + 1 < chars.len() && chars[*i + 1] == '\'' {
                            lit.push('\'');
                            *i += 2;
                            continue;
                        }
                        break;
                    }
                    lit.push(chars[*i]);
                    *i += 1;
                }
                *i += 1;
                if lit.is_empty() {
                    lit.push('\'');
                }
                out.push(Tok::Lit(lit));
            } else if c == '[' {
                *i += 1;
                out.push(Tok::Optional(inner(chars, i, depth + 1)?));
            } else if c == ']' {
                *i += 1;
                if depth == 0 {
                    return Err(EsError::illegal_argument("unbalanced ] in date pattern"));
                }
                return Ok(out);
            } else if c.is_ascii_alphabetic() {
                let mut n = 0;
                while *i < chars.len() && chars[*i] == c {
                    n += 1;
                    *i += 1;
                }
                if !"yuMLdDHhkKmsSaEXxZzVGQwWeYn".contains(c) {
                    return Err(EsError::illegal_argument(format!("Unknown pattern letter: {c}")));
                }
                out.push(Tok::Field(c, n));
            } else {
                out.push(Tok::Lit(c.to_string()));
                *i += 1;
            }
        }
        Ok(out)
    }
    let chars: Vec<char> = p.chars().collect();
    let mut i = 0;
    inner(&chars, &mut i, 0)
}

#[derive(Default, Clone, Copy)]
struct Parts {
    year: Option<i32>,
    month: Option<u32>,
    day: Option<u32>,
    doy: Option<u32>,
    hour: Option<u32>,
    pm: Option<bool>,
    minute: Option<u32>,
    second: Option<u32>,
    nanos: Option<u32>,
    offset: Option<i32>,
    zone: Option<Tz>,
}

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const FR_MONTHS: [&str; 12] = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"];
const FR_MONTHS_SHORT: [&str; 12] = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];
const FR_DAYS: [&str; 7] = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"];
const FR_DAYS_SHORT: [&str; 7] = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."];

/// Index and byte length of the leading name of `rest` among `names` (case-insensitive).
fn match_name(rest: &[u8], names: &[&str]) -> Option<(usize, usize)> {
    let r = std::str::from_utf8(rest).ok()?.to_lowercase();
    names.iter().enumerate().find(|(_, n)| r.starts_with(&n.to_lowercase())).map(|(i, n)| (i, n.len()))
}

const DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

fn is_numeric_field(t: Option<&Tok>) -> bool {
    matches!(t, Some(Tok::Field(c, n)) if "yuMdDHhkKmsS".contains(*c) && !(*c == 'M' && *n >= 3))
}

fn take_digits(s: &[u8], pos: usize, min: usize, max: usize) -> Option<(i64, usize)> {
    let mut end = pos;
    while end < s.len() && end - pos < max && s[end].is_ascii_digit() {
        end += 1;
    }
    if end - pos < min {
        return None;
    }
    std::str::from_utf8(&s[pos..end]).ok()?.parse().ok().map(|v| (v, end))
}

fn parse_toks(toks: &[Tok], s: &[u8], mut pos: usize, p: &mut Parts, fr: bool) -> Option<usize> {
    for (ti, t) in toks.iter().enumerate() {
        match t {
            Tok::Lit(l) => {
                let lb = l.as_bytes();
                if s.len() < pos + lb.len() || !s[pos..pos + lb.len()].eq_ignore_ascii_case(lb) {
                    return None;
                }
                pos += lb.len();
            }
            Tok::Optional(inner) => {
                let mut tmp = *p;
                if let Some(np) = parse_toks(inner, s, pos, &mut tmp, fr) {
                    *p = tmp;
                    pos = np;
                }
            }
            Tok::Field(c, n) => {
                let n = *n;
                let adjacent = is_numeric_field(toks.get(ti + 1));
                let (min, max) = if adjacent { (n, n) } else { (1, if n <= 2 { 2 } else { n.max(2) }) };
                match c {
                    'y' | 'u' | 'Y' => {
                        let neg = s.get(pos) == Some(&b'-') && n >= 4;
                        let start = if neg || s.get(pos) == Some(&b'+') { pos + 1 } else { pos };
                        let (v, np) = if n == 2 {
                            let (v, np) = take_digits(s, start, 2, 2)?;
                            (2000 + v, np)
                        } else if adjacent {
                            take_digits(s, start, n, n)?
                        } else {
                            take_digits(s, start, 1, 9)?
                        };
                        p.year = Some(if neg { -(v as i32) } else { v as i32 });
                        pos = np;
                    }
                    'M' | 'L' if n >= 3 && fr => {
                        let (i, len) = match_name(&s[pos..], if n >= 4 { &FR_MONTHS } else { &FR_MONTHS_SHORT })?;
                        p.month = Some(i as u32 + 1);
                        pos += len;
                    }
                    'E' if fr => {
                        let (_, len) = match_name(&s[pos..], if n >= 4 { &FR_DAYS } else { &FR_DAYS_SHORT })?;
                        pos += len;
                    }
                    'M' | 'L' if n >= 3 => {
                        let rest = &s[pos..];
                        let mut found = None;
                        for (i, m) in MONTHS.iter().enumerate() {
                            let full = m.as_bytes();
                            let short = &full[..3];
                            if n >= 4 && rest.len() >= full.len() && rest[..full.len()].eq_ignore_ascii_case(full) {
                                found = Some((i, full.len()));
                                break;
                            }
                            if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case(short) {
                                found = Some((i, 3));
                                if n == 3 {
                                    break;
                                }
                            }
                        }
                        let (i, len) = found?;
                        p.month = Some(i as u32 + 1);
                        pos += len;
                    }
                    'E' => {
                        let rest = &s[pos..];
                        let mut len = None;
                        for d in DAYS {
                            let full = d.as_bytes();
                            if n >= 4 && rest.len() >= full.len() && rest[..full.len()].eq_ignore_ascii_case(full) {
                                len = Some(full.len());
                                break;
                            }
                            if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case(&full[..3]) {
                                len = Some(3);
                                break;
                            }
                        }
                        pos += len?;
                    }
                    'a' => {
                        let rest = &s[pos..];
                        if rest.len() < 2 {
                            return None;
                        }
                        let ap = &rest[..2];
                        if ap.eq_ignore_ascii_case(b"AM") {
                            p.pm = Some(false);
                        } else if ap.eq_ignore_ascii_case(b"PM") {
                            p.pm = Some(true);
                        } else {
                            return None;
                        }
                        pos += 2;
                    }
                    'S' | 'n' => {
                        let max = if adjacent { n } else { 9 };
                        let min = if adjacent { n } else { 1 };
                        let mut end = pos;
                        while end < s.len() && end - pos < max && s[end].is_ascii_digit() {
                            end += 1;
                        }
                        if end - pos < min {
                            return None;
                        }
                        let digits = std::str::from_utf8(&s[pos..end]).ok()?;
                        let mut padded = digits.to_string();
                        while padded.len() < 9 {
                            padded.push('0');
                        }
                        p.nanos = Some(padded.parse().ok()?);
                        pos = end;
                    }
                    'X' | 'x' | 'Z' => {
                        let rest = std::str::from_utf8(&s[pos..]).ok()?;
                        if *c == 'x' && rest.starts_with('Z') {
                            return None;
                        }
                        let (off, len) = parse_offset_tail(rest)?;
                        p.offset = Some(off);
                        pos += len;
                    }
                    'z' | 'V' => {
                        let mut end = pos;
                        while end < s.len() && (s[end].is_ascii_alphanumeric() || b"/_+-:".contains(&s[end])) {
                            end += 1;
                        }
                        let id = std::str::from_utf8(&s[pos..end]).ok()?;
                        p.zone = Some(Tz::parse(id).ok()?);
                        pos = end;
                    }
                    'G' => {
                        let rest = &s[pos..];
                        if rest.len() >= 2 && (rest[..2].eq_ignore_ascii_case(b"AD") || rest[..2].eq_ignore_ascii_case(b"BC")) {
                            pos += 2;
                        } else {
                            return None;
                        }
                    }
                    _ => {
                        let (v, np) = take_digits(s, pos, min, max.max(if *c == 'D' { 3 } else { 2 }))?;
                        let v = v as u32;
                        match c {
                            'M' | 'L' => p.month = Some(v),
                            'd' => p.day = Some(v),
                            'D' => p.doy = Some(v),
                            'H' | 'k' => p.hour = Some(if *c == 'k' && v == 24 { 0 } else { v }),
                            'h' | 'K' => p.hour = Some(v % 12),
                            'm' => p.minute = Some(v),
                            's' => p.second = Some(v),
                            _ => {}
                        }
                        pos = np;
                    }
                }
            }
        }
    }
    Some(pos)
}

fn parts_to_millis(p: &Parts, tz: Tz) -> Option<i64> {
    let year = p.year.unwrap_or(1970);
    let date = if let Some(doy) = p.doy {
        NaiveDate::from_yo_opt(year, doy)?
    } else {
        NaiveDate::from_ymd_opt(year, p.month.unwrap_or(1), p.day.unwrap_or(1))?
    };
    let mut hour = p.hour.unwrap_or(0);
    if p.pm == Some(true) {
        hour = hour % 12 + 12;
    }
    let time = NaiveTime::from_hms_nano_opt(hour, p.minute.unwrap_or(0), p.second.unwrap_or(0), p.nanos.unwrap_or(0))?;
    let local = NaiveDateTime::new(date, time);
    Some(match (p.offset, p.zone) {
        (Some(off), _) => to_millis(local) - off as i64 * 1000,
        (None, Some(z)) => z.local_to_utc(local),
        (None, None) => tz.local_to_utc(local),
    })
}

/// ISO-8601 (`strict_date_optional_time`): `yyyy[-MM[-dd]]['T'HH[:mm[:ss[.S{1,9}]]]][Z|±HH[:]MM]`.
fn parse_iso(s: &str, tz: Tz, strict: bool) -> Option<i64> {
    let b = s.as_bytes();
    let mut p = Parts::default();
    let mut pos = 0;
    let neg = b.first() == Some(&b'-');
    if neg || b.first() == Some(&b'+') {
        pos = 1;
    }
    let (y, np) = if strict { take_digits(b, pos, 4, 4)? } else { take_digits(b, pos, 1, 9)? };
    p.year = Some(if neg { -(y as i32) } else { y as i32 });
    pos = np;
    let two = |pos: usize| -> Option<(i64, usize)> { if strict { take_digits(b, pos, 2, 2) } else { take_digits(b, pos, 1, 2) } };
    if b.get(pos) == Some(&b'-') {
        let (m, np) = two(pos + 1)?;
        p.month = Some(m as u32);
        pos = np;
        if b.get(pos) == Some(&b'-') {
            let (d, np) = two(pos + 1)?;
            p.day = Some(d as u32);
            pos = np;
        }
    }
    if b.get(pos) == Some(&b'T') || b.get(pos) == Some(&b't') || (!strict && b.get(pos) == Some(&b' ')) {
        let (h, np) = two(pos + 1)?;
        p.hour = Some(h as u32);
        pos = np;
        if b.get(pos) == Some(&b':') {
            let (m, np) = two(pos + 1)?;
            p.minute = Some(m as u32);
            pos = np;
            if b.get(pos) == Some(&b':') {
                let (sec, np) = two(pos + 1)?;
                p.second = Some(sec as u32);
                pos = np;
                if b.get(pos) == Some(&b'.') || b.get(pos) == Some(&b',') {
                    let mut end = pos + 1;
                    while end < b.len() && b[end].is_ascii_digit() && end - pos <= 12 {
                        end += 1;
                    }
                    if end == pos + 1 {
                        return None;
                    }
                    let mut digits = s[pos + 1..end].to_string();
                    if digits.len() > 9 {
                        digits.truncate(9);
                    }
                    while digits.len() < 9 {
                        digits.push('0');
                    }
                    p.nanos = Some(digits.parse().ok()?);
                    pos = end;
                }
            }
        }
        if pos < b.len() {
            if b[pos] == b'Z' || b[pos] == b'+' || b[pos] == b'-' {
                let (off, len) = parse_offset_tail(&s[pos..])?;
                p.offset = Some(off);
                pos += len;
            }
        }
    } else if pos < b.len() && (b[pos] == b'Z' || b[pos] == b'+') {
        // Date with zone and no time.
        if b[pos] == b'Z' {
            p.offset = Some(0);
            pos += 1;
        } else {
            p.offset = Some(parse_offset_str(&s[pos..])?);
            pos = b.len();
        }
    }
    if pos != b.len() {
        return None;
    }
    parts_to_millis(&p, tz)
}

fn format_toks(toks: &[Tok], dt: NaiveDateTime, offset: i32, out: &mut String, fr: bool) {
    for t in toks {
        match t {
            Tok::Lit(l) => out.push_str(l),
            Tok::Optional(inner) => format_toks(inner, dt, offset, out, fr),
            Tok::Field(c, n) => {
                let n = *n;
                match c {
                    'y' | 'u' | 'Y' => {
                        if n == 2 {
                            out.push_str(&format!("{:02}", dt.year().rem_euclid(100)));
                        } else {
                            out.push_str(&format!("{:0width$}", dt.year(), width = n.max(4)));
                        }
                    }
                    'M' | 'L' if fr && n >= 3 => {
                        out.push_str(if n >= 4 { FR_MONTHS[dt.month0() as usize] } else { FR_MONTHS_SHORT[dt.month0() as usize] });
                    }
                    'E' if fr => {
                        let d = dt.weekday().num_days_from_monday() as usize;
                        out.push_str(if n >= 4 { FR_DAYS[d] } else { FR_DAYS_SHORT[d] });
                    }
                    'M' | 'L' => {
                        if n >= 4 {
                            out.push_str(MONTHS[dt.month0() as usize]);
                        } else if n == 3 {
                            out.push_str(&MONTHS[dt.month0() as usize][..3]);
                        } else {
                            out.push_str(&format!("{:0n$}", dt.month()));
                        }
                    }
                    'd' => out.push_str(&format!("{:0n$}", dt.day())),
                    'D' => out.push_str(&format!("{:0n$}", dt.ordinal())),
                    'H' => out.push_str(&format!("{:0n$}", dt.hour())),
                    'k' => out.push_str(&format!("{:0n$}", if dt.hour() == 0 { 24 } else { dt.hour() })),
                    'h' => out.push_str(&format!("{:0n$}", if dt.hour() % 12 == 0 { 12 } else { dt.hour() % 12 })),
                    'K' => out.push_str(&format!("{:0n$}", dt.hour() % 12)),
                    'm' => out.push_str(&format!("{:0n$}", dt.minute())),
                    's' => out.push_str(&format!("{:0n$}", dt.second())),
                    'S' | 'n' => {
                        let nanos = format!("{:09}", dt.nanosecond() % 1_000_000_000);
                        out.push_str(&nanos[..n.min(9)]);
                    }
                    'a' => out.push_str(if dt.hour() >= 12 { "PM" } else { "AM" }),
                    'E' => {
                        let d = DAYS[dt.weekday().num_days_from_monday() as usize];
                        out.push_str(if n >= 4 { d } else { &d[..3] });
                    }
                    'X' => out.push_str(&format_offset(offset, n >= 3, true)),
                    'x' => out.push_str(&format_offset(offset, n >= 3, false)),
                    'Z' => {
                        if n >= 5 {
                            out.push_str(&format_offset(offset, true, true))
                        } else {
                            out.push_str(&format_offset(offset, false, false))
                        }
                    }
                    'z' | 'V' => out.push_str(&format_offset(offset, true, true)),
                    'G' => out.push_str(if dt.year() > 0 { "AD" } else { "BC" }),
                    'e' => out.push_str(&dt.weekday().number_from_monday().to_string()),
                    'w' => out.push_str(&format!("{:0n$}", dt.iso_week().week())),
                    'Q' => out.push_str(&((dt.month0() / 3) + 1).to_string()),
                    _ => {}
                }
            }
        }
    }
}

impl DateFormatter {
    /// Apply a mapping `locale` (only `fr` changes names).
    pub fn with_locale(mut self, locale: Option<&str>) -> Self {
        self.fr = locale.is_some_and(|l| l == "fr" || l.starts_with("fr_") || l.starts_with("fr-"));
        self
    }

    pub fn new(spec: &str) -> Result<Self> {
        let mut fmts = Vec::new();
        for part in spec.split("||") {
            let part = part.trim();
            if part.is_empty() {
                if spec.contains("||") {
                    return Err(EsError::illegal_argument(format!("Cannot have empty element in multi date format pattern: {spec}")));
                }
                continue;
            }
            if let Some(f) = named(part) {
                fmts.push(f);
            } else {
                let toks = tokenize(part).map_err(|e| EsError::illegal_argument(format!("Invalid format: [{part}]: {}", e.reason)).caused_by(e))?;
                fmts.push(Fmt::Pattern(toks));
            }
        }
        if fmts.is_empty() {
            return Err(EsError::illegal_argument(format!("Invalid format: [{spec}]")));
        }
        Ok(Self { spec: spec.to_string(), fmts, fr: false })
    }

    /// Parse a date string with the zone used when the value has none.
    pub fn parse_with_tz(&self, s: &str, tz: Tz) -> Result<i64> {
        let s = s.trim();
        for f in &self.fmts {
            let r = match f {
                Fmt::Iso { strict } => parse_iso(s, tz, *strict),
                Fmt::EpochMillis => parse_epoch(s, 1.0),
                Fmt::EpochSecond => parse_epoch(s, 1000.0),
                Fmt::Pattern(toks) => {
                    let mut p = Parts::default();
                    match parse_toks(toks, s.as_bytes(), 0, &mut p, self.fr) {
                        Some(end) if end == s.len() => parts_to_millis(&p, tz),
                        _ => None,
                    }
                }
            };
            if let Some(v) = r {
                return Ok(v);
            }
        }
        Err(EsError::illegal_argument(format!(
            "failed to parse date field [{s}] with format [{}]",
            self.spec
        )))
    }

    pub fn parse(&self, s: &str) -> Result<i64> {
        self.parse_with_tz(s, Tz::UTC)
    }

    /// Format with the first format of the list.
    pub fn format(&self, millis: i64, tz: Tz) -> String {
        let offset = tz.offset_at(millis);
        let dt = from_millis(millis + offset as i64 * 1000);
        match &self.fmts[0] {
            Fmt::Iso { .. } => {
                let mut s = format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}",
                    dt.year(),
                    dt.month(),
                    dt.day(),
                    dt.hour(),
                    dt.minute(),
                    dt.second(),
                    dt.nanosecond() / 1_000_000
                );
                s.push_str(&format_offset(offset, true, true));
                s
            }
            Fmt::EpochMillis => millis.to_string(),
            Fmt::EpochSecond => (millis.div_euclid(1000)).to_string(),
            Fmt::Pattern(toks) => {
                let mut out = String::new();
                format_toks(toks, dt, offset, &mut out, self.fr);
                out
            }
        }
    }

    /// Format epoch nanoseconds with the first format of the list: ISO keeps the
    /// sub-millisecond digits (3 to 9, trailing zeros past 3 dropped), patterns
    /// see the full nanosecond of second.
    pub fn format_nanos(&self, nanos: i64, tz: Tz) -> String {
        let millis = nanos.div_euclid(1_000_000);
        let offset = tz.offset_at(millis);
        let secs = nanos.div_euclid(1_000_000_000) + offset as i64;
        let sub = nanos.rem_euclid(1_000_000_000) as u32;
        let dt = chrono::DateTime::from_timestamp(secs, sub).map(|d| d.naive_utc()).unwrap_or_default();
        match &self.fmts[0] {
            Fmt::Iso { .. } => {
                let mut frac = format!("{sub:09}");
                while frac.len() > 3 && frac.ends_with('0') {
                    frac.pop();
                }
                let mut s = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{frac}", dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), dt.second());
                s.push_str(&format_offset(offset, true, true));
                s
            }
            Fmt::EpochMillis => millis.to_string(),
            Fmt::EpochSecond => millis.div_euclid(1000).to_string(),
            Fmt::Pattern(toks) => {
                let mut out = String::new();
                format_toks(toks, dt, offset, &mut out, self.fr);
                out
            }
        }
    }

    /// True when the first format is epoch based (numbers stay numbers).
    pub fn is_epoch(&self) -> bool {
        matches!(self.fmts[0], Fmt::EpochMillis | Fmt::EpochSecond)
    }

    /// True if any format of the list is `epoch_millis`/`epoch_second`.
    pub fn accepts_numbers(&self) -> bool {
        self.fmts.iter().any(|f| matches!(f, Fmt::EpochMillis | Fmt::EpochSecond))
    }

    /// Parse a JSON number as a date: epoch millis (or seconds) when the
    /// format allows numbers.
    pub fn parse_number(&self, v: f64) -> Result<i64> {
        for f in &self.fmts {
            match f {
                Fmt::EpochMillis => return Ok(v as i64),
                Fmt::EpochSecond => return Ok((v * 1000.0) as i64),
                _ => {}
            }
        }
        // ES parses numbers through the string formats too (e.g. "yyyy").
        self.parse(&format_number(v))
    }
}

fn format_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e18 {
        format!("{}", v as i64)
    } else {
        v.to_string()
    }
}

fn parse_epoch(s: &str, mult: f64) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    let b = s.as_bytes();
    let ok = b.iter().enumerate().all(|(i, c)| c.is_ascii_digit() || (*c == b'-' && i == 0) || *c == b'.');
    if !ok {
        return None;
    }
    if let Ok(i) = s.parse::<i64>() {
        return Some((i as f64 * mult) as i64).map(|v| if mult == 1.0 { i } else { v });
    }
    s.parse::<f64>().ok().map(|f| (f * mult) as i64)
}

// ---------------------------------------------------------------------------
// Calendar units, rounding, date math.

/// A calendar unit (date math rounding and `calendar_interval`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalUnit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

impl CalUnit {
    pub fn parse(s: &str) -> Option<CalUnit> {
        Some(match s {
            "s" | "1s" | "second" => CalUnit::Second,
            "m" | "1m" | "minute" => CalUnit::Minute,
            "h" | "H" | "1h" | "hour" => CalUnit::Hour,
            "d" | "1d" | "day" => CalUnit::Day,
            "w" | "1w" | "week" => CalUnit::Week,
            "M" | "1M" | "month" => CalUnit::Month,
            "q" | "1q" | "quarter" => CalUnit::Quarter,
            "y" | "1y" | "year" => CalUnit::Year,
            _ => return None,
        })
    }

    /// Approximate length in millis (for sizing).
    pub fn approx_millis(&self) -> i64 {
        match self {
            CalUnit::Second => 1000,
            CalUnit::Minute => 60_000,
            CalUnit::Hour => 3_600_000,
            CalUnit::Day => 86_400_000,
            CalUnit::Week => 7 * 86_400_000,
            CalUnit::Month => 30 * 86_400_000,
            CalUnit::Quarter => 91 * 86_400_000,
            CalUnit::Year => 365 * 86_400_000,
        }
    }
}

/// Round `millis` down to the start of the calendar unit in `tz`.
pub fn round_down(millis: i64, unit: CalUnit, tz: Tz) -> i64 {
    let local = tz.to_local(millis);
    let d = local.date();
    let floored = match unit {
        CalUnit::Second => local.with_nanosecond(0).unwrap(),
        CalUnit::Minute => local.with_nanosecond(0).unwrap().with_second(0).unwrap(),
        CalUnit::Hour => d.and_hms_opt(local.hour(), 0, 0).unwrap(),
        CalUnit::Day => d.and_hms_opt(0, 0, 0).unwrap(),
        CalUnit::Week => (d - Duration::days(d.weekday().num_days_from_monday() as i64)).and_hms_opt(0, 0, 0).unwrap(),
        CalUnit::Month => NaiveDate::from_ymd_opt(d.year(), d.month(), 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        CalUnit::Quarter => NaiveDate::from_ymd_opt(d.year(), (d.month0() / 3) * 3 + 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        CalUnit::Year => NaiveDate::from_ymd_opt(d.year(), 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
    };
    match unit {
        // Sub-hour units: offsets are whole minutes (almost always hours),
        // so rounding in UTC after shifting is exact.
        CalUnit::Second | CalUnit::Minute => millis - (to_millis(local) - to_millis(floored)),
        _ => tz.local_to_utc(floored),
    }
}

/// Add `n` calendar units to a UTC instant, in `tz` local time. Saturates
/// at the representable range (see [`try_add_units`]).
pub fn add_units(millis: i64, unit: CalUnit, n: i64, tz: Tz) -> i64 {
    try_add_units(millis, unit, n, tz).unwrap_or(if n < 0 { i64::MIN } else { i64::MAX })
}

/// [`add_units`], or `None` when the result leaves the representable date
/// range (`n` can come straight from a request: `now+1000000000d`).
pub fn try_add_units(millis: i64, unit: CalUnit, n: i64, tz: Tz) -> Option<i64> {
    match unit {
        CalUnit::Second => millis.checked_add(n.checked_mul(1000)?),
        CalUnit::Minute => millis.checked_add(n.checked_mul(60_000)?),
        CalUnit::Hour => millis.checked_add(n.checked_mul(3_600_000)?),
        _ => {
            let local = tz.to_local(millis);
            let shifted = match unit {
                CalUnit::Day => local.checked_add_signed(Duration::try_days(n)?)?,
                CalUnit::Week => local.checked_add_signed(Duration::try_days(n.checked_mul(7)?)?)?,
                CalUnit::Month => add_months(local, n)?,
                CalUnit::Quarter => add_months(local, n.checked_mul(3)?)?,
                CalUnit::Year => add_months(local, n.checked_mul(12)?)?,
                _ => unreachable!(),
            };
            Some(tz.local_to_utc(shifted))
        }
    }
}

/// `dt` plus `n` months, clamping the day to the target month's length;
/// `None` outside chrono's year range.
fn add_months(dt: NaiveDateTime, n: i64) -> Option<NaiveDateTime> {
    let total = (dt.year() as i64 * 12 + dt.month0() as i64).checked_add(n)?;
    let y = i32::try_from(total.div_euclid(12)).ok()?;
    let m = total.rem_euclid(12) as u32 + 1;
    (1..=dt.day()).rev().find_map(|day| NaiveDate::from_ymd_opt(y, m, day)).map(|d| NaiveDateTime::new(d, dt.time()))
}

/// Evaluate ES date math: `now-1d/d`, `2024-01-01||+1M/M`, plain dates.
///
/// `round_up` selects the end of the rounded unit (used for `gt` and
/// `lte`), as ES's `JavaDateMathParser` does.
pub fn parse_date_math(expr: &str, fmt: &DateFormatter, tz: Tz, now: i64, round_up: bool) -> Result<i64> {
    let (base, math) = if let Some(rest) = expr.strip_prefix("now") {
        (now, rest)
    } else if let Some(idx) = expr.find("||") {
        let anchor = &expr[..idx];
        (fmt.parse_with_tz(anchor, tz)?, &expr[idx + 2..])
    } else {
        // Plain date: when rounding up a date without time, ES rounds the
        // missing fields up (e.g. lte "2020-01-01" includes the whole day).
        let v = fmt.parse_with_tz(expr, tz)?;
        if round_up && is_date_only(expr) {
            return Ok(add_units(v, CalUnit::Day, 1, tz) - 1);
        }
        return Ok(v);
    };
    apply_math(base, math, tz, round_up).map_err(|m| EsError::new(400, "parse_exception", m).with("caused_by_expression", expr))
}

fn is_date_only(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10 && b[4] == b'-' && b[7] == b'-'
}

/// ES `JavaDateMathParser.parseMath`: `+N<unit>`, `-N<unit>`, `/<unit>`
/// (rounding takes no count but `1`); errors carry ES's messages.
fn apply_math(mut v: i64, math: &str, tz: Tz, round_up: bool) -> std::result::Result<i64, String> {
    let b = math.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let op = b[i];
        i += 1;
        let round = op == b'/';
        if !round && op != b'+' && op != b'-' {
            return Err(format!("operator not supported for date math [{math}]"));
        }
        if i >= b.len() {
            return Err(format!("truncated date math [{math}]"));
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let n: i64 = if start == i {
            1
        } else {
            if i >= b.len() {
                return Err(format!("truncated date math [{math}]"));
            }
            std::str::from_utf8(&b[start..i]).ok().and_then(|s| s.parse().ok()).ok_or_else(|| format!("truncated date math [{math}]"))?
        };
        if round && n != 1 {
            return Err(format!("rounding `/` can only be used on single unit types [{math}]"));
        }
        let uc = std::str::from_utf8(&b[i..i + 1]).unwrap_or("");
        let unit = CalUnit::parse(uc).ok_or_else(|| format!("unit [{uc}] not supported for date math [{math}]"))?;
        i += 1;
        if round {
            v = round_down(v, unit, tz);
            if round_up {
                v = add_units(v, unit, 1, tz).saturating_sub(1);
            }
        } else {
            v = try_add_units(v, unit, if op == b'-' { -n } else { n }, tz).ok_or_else(|| format!("date math [{math}] is out of the supported date range"))?;
        }
    }
    Ok(v)
}

/// Current time in millis.
pub fn now_millis() -> i64 {
    Utc::now().timestamp_millis()
}

/// Human-readable date for `?human` (e.g. `2024-01-01T00:00:00.000Z`).
pub fn human_date(millis: i64) -> String {
    chrono::DateTime::from_timestamp_millis(millis)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
        .unwrap_or_else(|| millis.to_string())
}

/// Human-readable version string for `?human` (e.g. `8.5.2` for `8521000`).
pub fn human_version(s: &str) -> String {
    let n: u64 = s.parse().unwrap_or(0);
    let major = n / 1_000_000;
    let minor = (n / 1_000) % 1_000;
    let patch = n % 1_000;
    format!("{major}.{minor}.{patch}")
}

/// Is this string a date for dynamic mapping (`dynamic_date_formats`)?
pub fn detect_date(s: &str, formats: &[DateFormatter]) -> bool {
    may_be_date(s) && formats.iter().any(|f| f.parse(s).is_ok())
}

/// Fast reject for [`detect_date`]: dates start with a digit and contain a
/// separator. Checked before the formatters are built, so dynamic mapping
/// of an ordinary string does not compile date patterns.
pub fn may_be_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 8 && b[0].is_ascii_digit() && (s.contains('-') || s.contains('/'))
}

/// Default `dynamic_date_formats`.
pub fn default_dynamic_date_formats() -> Vec<DateFormatter> {
    vec![
        DateFormatter::new("strict_date_optional_time").unwrap(),
        DateFormatter::new("yyyy/MM/dd HH:mm:ss||yyyy/MM/dd").unwrap(),
    ]
}

/// Parse a time value (`30s`, `5m`, `1h`, `-1`, `100ms`) into millis.
pub fn parse_time_value(s: &str) -> Option<i64> {
    let s = s.trim();
    if s == "-1" {
        return Some(-1);
    }
    if s == "0" {
        return Some(0);
    }
    let split = s.find(|c: char| c.is_ascii_alphabetic())?;
    let (n, unit) = s.split_at(split);
    let n: f64 = n.trim().parse().ok()?;
    // ES `TimeValue.parseTimeValue`: units are case-insensitive, except
    // minutes (`M` would be months: no unit).
    let lower = unit.to_ascii_lowercase();
    let unit = if unit == "M" { unit } else { lower.as_str() };
    let mult = match unit {
        "nanos" => 1e-6,
        "micros" => 1e-3,
        "ms" => 1.0,
        "s" => 1000.0,
        "m" => 60_000.0,
        "h" => 3_600_000.0,
        "d" => 86_400_000.0,
        _ => return None,
    };
    Some((n * mult) as i64)
}

/// Fixed interval (`fixed_interval`): `90m`, `1d`, `500ms`.
pub fn parse_fixed_interval(s: &str) -> Option<i64> {
    parse_time_value(s).filter(|v| *v > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(s: &str) -> i64 {
        DateFormatter::default().parse(s).unwrap()
    }

    #[test]
    fn iso_forms() {
        assert_eq!(ms("1970-01-01"), 0);
        assert_eq!(ms("1970-01-01T00:00:01Z"), 1000);
        assert_eq!(ms("1970-01-01T00:00:00.5Z"), 500);
        assert_eq!(ms("1970-01-01T01:00:00+01:00"), 0);
        assert_eq!(ms("1970-01-01T01:00:00+0100"), 0);
        assert_eq!(ms("2015-01-01T12:10:30.123456789Z"), 1420114230123);
        assert_eq!(ms("2015-01"), 1420070400000);
        assert_eq!(ms("1420070400000"), 1420070400000);
        assert!(DateFormatter::default().parse("2015-13-01").is_err());
        assert!(DateFormatter::default().parse("hello").is_err());
    }

    #[test]
    fn offset_parsing() {
        // `Z` / `+HH` / `+HHMM` / `+HH:MM` (java `appendOffsetId` / `appendOffset`).
        assert_eq!(ms("1970-01-01T00:00:00Z"), 0);
        assert_eq!(ms("1970-01-01T01:00:00+01"), 0);
        assert_eq!(ms("1970-01-01T01:00:00+0100"), 0);
        assert_eq!(ms("1970-01-01T01:00:00+01:00"), 0);
        assert_eq!(ms("1970-01-01T01:30:00+01:30"), 0);
        assert_eq!(ms("1970-01-01T23:00:00-01"), 86400000);
        // Custom patterns with `X` / `XX` / `XXX` / `Z` / `ZZZZZ`.
        let f = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ssX").unwrap();
        assert_eq!(f.parse("1970-01-01T01:00:00+01").unwrap(), 0);
        let f = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ssXXX").unwrap();
        assert_eq!(f.parse("1970-01-01T01:00:00+01:00").unwrap(), 0);
        let f = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ssZ").unwrap();
        assert_eq!(f.parse("1970-01-01T01:00:00+0100").unwrap(), 0);
        let f = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ssZZZZZ").unwrap();
        assert_eq!(f.parse("1970-01-01T01:00:00+01:00").unwrap(), 0);
        // `x` never accepts `Z`.
        let f = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ssx").unwrap();
        assert!(f.parse("1970-01-01T01:00:00Z").is_err());
        assert!(f.parse("1970-01-01T01:00:00z").is_err());
        assert_eq!(f.parse("1970-01-01T01:00:00+01").unwrap(), 0);
        // Garbage after the offset is rejected.
        assert!(DateFormatter::default().parse("1970-01-01T00:00:00Zx").is_err());
        assert!(DateFormatter::default().parse("1970-01-01T00:00:00+01:00x").is_err());
    }

    #[test]
    fn named_and_custom_patterns() {
        let f = DateFormatter::new("yyyy-MM-dd HH:mm:ss").unwrap();
        assert_eq!(f.parse("2015-01-01 12:10:30").unwrap(), 1420114230000);
        assert_eq!(f.format(1420114230000, Tz::UTC), "2015-01-01 12:10:30");
        let f = DateFormatter::new("basic_date").unwrap();
        assert_eq!(f.parse("20150101").unwrap(), 1420070400000);
        assert_eq!(f.format(1420070400000, Tz::UTC), "20150101");
        let f = DateFormatter::new("dd/MMM/yyyy:HH:mm:ss Z").unwrap();
        assert_eq!(f.parse("01/Jan/2015:12:10:30 +0000").unwrap(), 1420114230000);
        assert_eq!(f.format(1420114230000, Tz::UTC), "01/Jan/2015:12:10:30 +0000");
        let f = DateFormatter::new("epoch_second").unwrap();
        assert_eq!(f.parse("1420114230").unwrap(), 1420114230000);
        assert_eq!(f.format(1420114230000, Tz::UTC), "1420114230");
        let f = DateFormatter::new("yyyy-MM-dd['T'HH:mm]").unwrap();
        assert_eq!(f.parse("2015-01-01").unwrap(), 1420070400000);
        assert_eq!(f.parse("2015-01-01T01:00").unwrap(), 1420074000000);
        assert!(DateFormatter::new("yyyy-qq").is_err() || DateFormatter::new("yyyy-qq").unwrap().parse("2015-01").is_err());
    }

    #[test]
    fn format_default() {
        let f = DateFormatter::default();
        assert_eq!(f.format(1420114230123, Tz::UTC), "2015-01-01T12:10:30.123Z");
        assert_eq!(f.format(0, Tz::parse("+01:00").unwrap()), "1970-01-01T01:00:00.000+01:00");
    }

    #[test]
    fn date_math() {
        let f = DateFormatter::default();
        let now = ms("2020-03-15T10:20:30Z");
        assert_eq!(parse_date_math("now", &f, Tz::UTC, now, false).unwrap(), now);
        assert_eq!(parse_date_math("now-1d/d", &f, Tz::UTC, now, false).unwrap(), ms("2020-03-14"));
        assert_eq!(parse_date_math("now-1d/d", &f, Tz::UTC, now, true).unwrap(), ms("2020-03-15") - 1);
        assert_eq!(parse_date_math("2020-01-31||+1M", &f, Tz::UTC, now, false).unwrap(), ms("2020-02-29"));
        assert_eq!(parse_date_math("2020-01-01||/y", &f, Tz::UTC, now, true).unwrap(), ms("2021-01-01") - 1);
        assert_eq!(parse_date_math("now/M", &f, Tz::UTC, now, false).unwrap(), ms("2020-03-01"));
        assert_eq!(parse_date_math("2020-01-01", &f, Tz::UTC, now, true).unwrap(), ms("2020-01-02") - 1);
        assert!(parse_date_math("now+1x", &f, Tz::UTC, now, false).is_err());
        // Out-of-range counts are errors: no overflow panic, no endless
        // day-clamping loop.
        for expr in ["now+1000000000d", "now-1000000000d", "now+1000000000w", "now+1000000000y", "now-1000000000y", "now+1000000000M",
                     "now+9223372036854775807s", "now+9223372036854775807h", "2020-01-01||+1000000000y", "now+9223372036854775807y"] {
            let e = parse_date_math(expr, &f, Tz::UTC, now, false).unwrap_err();
            assert_eq!(e.status, 400, "{expr}");
        }
        let tz = Tz::parse("Europe/Paris").unwrap();
        assert!(parse_date_math("now+1000000000y", &f, tz, now, true).is_err());
        assert_eq!(add_units(now, CalUnit::Day, i64::MAX, Tz::UTC), i64::MAX);
        assert_eq!(add_units(now, CalUnit::Year, i64::MIN, Tz::UTC), i64::MIN);
    }

    #[test]
    fn rounding_with_zones() {
        let paris = Tz::parse("Europe/Paris").unwrap();
        // 2020-03-29 is a DST change day in Paris.
        let t = ms("2020-03-29T12:00:00Z");
        assert_eq!(round_down(t, CalUnit::Day, paris), ms("2020-03-28T23:00:00Z"));
        assert_eq!(add_units(ms("2020-03-28T23:00:00Z"), CalUnit::Day, 1, paris), ms("2020-03-29T22:00:00Z"));
        assert_eq!(round_down(t, CalUnit::Month, Tz::UTC), ms("2020-03-01"));
        assert_eq!(round_down(t, CalUnit::Week, Tz::UTC), ms("2020-03-23"));
        assert_eq!(round_down(t, CalUnit::Quarter, Tz::UTC), ms("2020-01-01"));
        assert_eq!(round_down(ms("2020-03-29T12:34:56.789Z"), CalUnit::Minute, Tz::UTC), ms("2020-03-29T12:34:00Z"));
        assert_eq!(Tz::parse("-05:00").unwrap().offset_at(0), -18000);
        assert!(Tz::parse("Mars/Base").is_err());
    }

    #[test]
    fn time_values() {
        assert_eq!(parse_time_value("1s"), Some(1000));
        assert_eq!(parse_time_value("5m"), Some(300_000));
        assert_eq!(parse_time_value("-1"), Some(-1));
        assert_eq!(parse_time_value("100ms"), Some(100));
        assert_eq!(parse_time_value("bogus"), None);
    }
}
