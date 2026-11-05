//! Quartz/ES cron expressions:
//! `<seconds> <minutes> <hours> <day-of-month> <month> <day-of-week> [<year>]`.
//!
//! Supported: `*`, `?`, lists `a,b`, ranges `a-b` (wrapping allowed), steps
//! `a/n`, `*/n`, `a-b/n`, month names `JAN`..`DEC`, day names `SUN`..`SAT`
//! (1 = Sunday), day-of-month `L`, `L-n`, `nW`, `LW`; day-of-week `L`, `nL`,
//! `n#k`. As in ES, exactly one of day-of-month / day-of-week must be `?`.
//! Times are evaluated in UTC.

use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronError(pub String);

impl std::fmt::Display for CronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DomSpec {
    Any,
    Days(BTreeSet<u32>),
    /// `L` / `L-n`
    LastDay(u32),
    /// `nW`
    NearestWeekday(u32),
    /// `LW`
    LastWeekday,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DowSpec {
    Any,
    Days(BTreeSet<u32>),
    /// `nL`: last <day> of the month.
    Last(u32),
    /// `n#k`: k-th <day> of the month.
    Nth(u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cron {
    expr: String,
    seconds: BTreeSet<u32>,
    minutes: BTreeSet<u32>,
    hours: BTreeSet<u32>,
    dom: DomSpec,
    months: BTreeSet<u32>,
    dow: DowSpec,
    years: BTreeSet<u32>,
}

const MONTHS: [&str; 12] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const DAYS: [&str; 7] = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
const MAX_YEAR: u32 = 2099;

fn days_in_month(y: i32, m: u32) -> u32 {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    NaiveDate::from_ymd_opt(ny, nm, 1).unwrap().pred_opt().unwrap().day()
}

/// Quartz day-of-week: 1 = Sunday .. 7 = Saturday.
fn quartz_dow(d: NaiveDate) -> u32 {
    d.weekday().num_days_from_sunday() + 1
}

impl Cron {
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let err = |m: &str| CronError(format!("invalid cron expression [{expr}]: {m}"));
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() < 6 {
            return Err(err("Unexpected end of expression."));
        }
        if fields.len() > 7 {
            return Err(err("too many fields"));
        }
        let num = |s: &str, names: Option<&[&str]>, offset: u32| -> Result<u32, CronError> {
            if let Ok(n) = s.parse::<u32>() {
                return Ok(n);
            }
            if let Some(names) = names {
                let up = s.to_ascii_uppercase();
                if let Some(i) = names.iter().position(|n| *n == up) {
                    return Ok(i as u32 + offset);
                }
            }
            Err(err(&format!("Illegal characters for this position: '{s}'")))
        };
        // Generic field parser.
        let parse_set = |field: &str, min: u32, max: u32, names: Option<&[&str]>, name_offset: u32, range_msg: &str| -> Result<BTreeSet<u32>, CronError> {
            let mut set = BTreeSet::new();
            for tok in field.split(',') {
                if tok.is_empty() {
                    return Err(err("Illegal empty list element"));
                }
                let (range, step) = match tok.split_once('/') {
                    Some((r, s)) => {
                        let s: u32 = s.parse().map_err(|_| err(&format!("Illegal increment '{s}'")))?;
                        if s == 0 || s > max {
                            return Err(err(&format!("Increment > {} or 0 : {s}", max)));
                        }
                        (r, Some(s))
                    }
                    None => (tok, None),
                };
                let (start, end) = if range == "*" || range == "?" {
                    (min, max)
                } else if let Some((a, b)) = range.split_once('-') {
                    (num(a, names, name_offset)?, num(b, names, name_offset)?)
                } else {
                    let a = num(range, names, name_offset)?;
                    (a, if step.is_some() { max } else { a })
                };
                if start < min || start > max || end < min || end > max {
                    return Err(err(range_msg));
                }
                let step = step.unwrap_or(1);
                if start <= end {
                    let mut v = start;
                    while v <= end {
                        set.insert(v);
                        v += step;
                    }
                } else {
                    // wrap-around range, e.g. FRI-MON or 22-2
                    let span = (max - min + 1) as i64;
                    let mut v = start as i64;
                    let stop = end as i64 + span;
                    while v <= stop {
                        let w = ((v - min as i64) % span) + min as i64;
                        set.insert(w as u32);
                        v += step as i64;
                    }
                }
            }
            Ok(set)
        };

        for f in [fields[0], fields[1], fields[2], fields[4]] {
            if f.contains('?') {
                return Err(err("'?' can only be specified for Day-of-Month or Day-of-Week."));
            }
        }
        for f in [fields[0], fields[1], fields[2]] {
            if f.to_ascii_uppercase().chars().any(|c| matches!(c, 'L' | 'W' | '#')) {
                return Err(err("'L', 'W' and '#' can only be used in the day-of-month or day-of-week fields"));
            }
        }
        let seconds = parse_set(fields[0], 0, 59, None, 0, "Minute and Second values must be between 0 and 59")?;
        let minutes = parse_set(fields[1], 0, 59, None, 0, "Minute and Second values must be between 0 and 59")?;
        let hours = parse_set(fields[2], 0, 23, None, 0, "Hour values must be between 0 and 23")?;
        let months = parse_set(fields[4], 1, 12, Some(&MONTHS), 1, "Month values must be between 1 and 12")?;
        let years = match fields.get(6) {
            Some(y) => parse_set(y, 1970, MAX_YEAR, None, 0, "Year values must be between 1970 and 2099")?,
            None => (1970..=MAX_YEAR).collect(),
        };
        // Day of month.
        let dom_f = fields[3].to_ascii_uppercase();
        let dom = if dom_f == "?" {
            DomSpec::Any
        } else if dom_f == "L" {
            DomSpec::LastDay(0)
        } else if dom_f == "LW" {
            DomSpec::LastWeekday
        } else if let Some(off) = dom_f.strip_prefix("L-") {
            let n: u32 = off.parse().map_err(|_| err(&format!("Illegal characters for this position: '{dom_f}'")))?;
            if n > 30 {
                return Err(err("Offset from last day must be <= 30"));
            }
            DomSpec::LastDay(n)
        } else if let Some(d) = dom_f.strip_suffix('W') {
            let n: u32 = d.parse().map_err(|_| err(&format!("Illegal characters for this position: '{dom_f}'")))?;
            if !(1..=31).contains(&n) {
                return Err(err("Day of month values must be between 1 and 31"));
            }
            DomSpec::NearestWeekday(n)
        } else {
            if dom_f.contains('L') || dom_f.contains('W') || dom_f.contains('#') {
                return Err(err("Support for specifying 'L' and 'LW' with other days of the month is not implemented"));
            }
            DomSpec::Days(parse_set(&dom_f, 1, 31, None, 0, "Day of month values must be between 1 and 31")?)
        };

        // Day of week.
        let dow_f = fields[5].to_ascii_uppercase();
        let dow_num = |s: &str| num(s, Some(&DAYS), 1);
        let dow = if dow_f == "?" {
            DowSpec::Any
        } else if dow_f == "L" {
            DowSpec::Days([7].into())
        } else if let Some((d, n)) = dow_f.split_once('#') {
            let d = dow_num(d)?;
            let n: u32 = n.parse().map_err(|_| err("A numeric value between 1 and 5 must follow the '#' option"))?;
            if !(1..=7).contains(&d) {
                return Err(err("Day-of-Week values must be between 1 and 7"));
            }
            if !(1..=5).contains(&n) {
                return Err(err("A numeric value between 1 and 5 must follow the '#' option"));
            }
            DowSpec::Nth(d, n)
        } else if let Some(d) = dow_f.strip_suffix('L') {
            let d = dow_num(d)?;
            if !(1..=7).contains(&d) {
                return Err(err("Day-of-Week values must be between 1 and 7"));
            }
            DowSpec::Last(d)
        } else {
            if dow_f.contains('L') || dow_f.contains('W') {
                return Err(err("Support for specifying 'L' with other days of the week is not implemented"));
            }
            DowSpec::Days(parse_set(&dow_f, 1, 7, Some(&DAYS), 1, "Day-of-Week values must be between 1 and 7")?)
        };

        let dom_spec = dom != DomSpec::Any;
        let dow_spec = dow != DowSpec::Any;
        if dom_spec == dow_spec {
            return Err(err(
                "support for specifying both a day-of-week AND a day-of-month parameter is not implemented.",
            ));
        }
        Ok(Self { expr: expr.to_string(), seconds, minutes, hours, dom, months, dow, years })
    }

    pub fn expression(&self) -> &str {
        &self.expr
    }

    fn day_matches(&self, d: NaiveDate) -> bool {
        let (y, m, day) = (d.year(), d.month(), d.day());
        let last = days_in_month(y, m);
        match &self.dom {
            DomSpec::Any => {}
            DomSpec::Days(s) => return s.contains(&day),
            DomSpec::LastDay(off) => return last >= *off && day == last - off,
            DomSpec::NearestWeekday(n) => {
                let target = (*n).min(last);
                let td = NaiveDate::from_ymd_opt(y, m, target).unwrap();
                let w = quartz_dow(td);
                let actual = match w {
                    7 => {
                        if target == 1 { 3 } else { target - 1 }
                    }
                    1 => {
                        if target == last { target - 2 } else { target + 1 }
                    }
                    _ => target,
                };
                return day == actual;
            }
            DomSpec::LastWeekday => {
                let ld = NaiveDate::from_ymd_opt(y, m, last).unwrap();
                let actual = match quartz_dow(ld) {
                    7 => last - 1,
                    1 => last - 2,
                    _ => last,
                };
                return day == actual;
            }
        }
        let w = quartz_dow(d);
        match &self.dow {
            DowSpec::Any => true,
            DowSpec::Days(s) => s.contains(&w),
            DowSpec::Last(dw) => w == *dw && day + 7 > last,
            DowSpec::Nth(dw, n) => w == *dw && (day - 1) / 7 + 1 == *n,
        }
    }

    /// First fire time strictly after `after_millis` (UTC), or `None`.
    pub fn next_after(&self, after_millis: i64) -> Option<i64> {
        let start = Utc.timestamp_millis_opt(after_millis).single()?.naive_utc();
        // Next whole second strictly after.
        let mut t = start.with_nanosecond(0)? + chrono::Duration::seconds(1);
        let ymd = |y: i32, m: u32, d: u32| NaiveDate::from_ymd_opt(y, m, d).and_then(|d| d.and_hms_opt(0, 0, 0));
        for _ in 0..200_000 {
            let year = t.year() as u32;
            if year > MAX_YEAR {
                return None;
            }
            if !self.years.contains(&year) {
                let ny = self.years.range(year + 1..).next()?;
                t = ymd(*ny as i32, 1, 1)?;
                continue;
            }
            if !self.months.contains(&t.month()) {
                match self.months.range(t.month() + 1..).next() {
                    Some(m) => t = ymd(t.year(), *m, 1)?,
                    None => t = ymd(t.year() + 1, 1, 1)?,
                }
                continue;
            }
            if !self.day_matches(t.date()) {
                t = t.date().succ_opt()?.and_hms_opt(0, 0, 0)?;
                continue;
            }
            if !self.hours.contains(&t.hour()) {
                match self.hours.range(t.hour() + 1..).next() {
                    Some(h) => t = t.date().and_hms_opt(*h, 0, 0)?,
                    None => t = t.date().succ_opt()?.and_hms_opt(0, 0, 0)?,
                }
                continue;
            }
            if !self.minutes.contains(&t.minute()) {
                match self.minutes.range(t.minute() + 1..).next() {
                    Some(mi) => t = t.date().and_hms_opt(t.hour(), *mi, 0)?,
                    None => t = next_hour(t)?,
                }
                continue;
            }
            if !self.seconds.contains(&t.second()) {
                match self.seconds.range(t.second() + 1..).next() {
                    Some(s) => t = t.date().and_hms_opt(t.hour(), t.minute(), *s)?,
                    None => t = next_minute(t)?,
                }
                continue;
            }
            return Some(t.and_utc().timestamp_millis());
        }
        None
    }
}

fn next_hour(t: NaiveDateTime) -> Option<NaiveDateTime> {
    let base = t.date().and_hms_opt(t.hour(), 0, 0)?;
    Some(base + chrono::Duration::hours(1))
}

fn next_minute(t: NaiveDateTime) -> Option<NaiveDateTime> {
    let base = t.date().and_hms_opt(t.hour(), t.minute(), 0)?;
    Some(base + chrono::Duration::minutes(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(s: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis()
    }

    fn next(expr: &str, after: &str) -> String {
        let c = Cron::parse(expr).unwrap();
        crate::util::format_millis(c.next_after(ms(after)).unwrap())
    }

    #[test]
    fn basic_schedules() {
        // ES docs: daily at 1:30am.
        assert_eq!(next("0 30 1 * * ?", "2019-04-23T01:30:00Z"), "2019-04-24T01:30:00.000Z");
        assert_eq!(next("0 30 1 * * ?", "2019-04-23T01:29:59.999Z"), "2019-04-23T01:30:00.000Z");
        assert_eq!(next("0 0/15 * * * ?", "2024-01-01T10:07:00Z"), "2024-01-01T10:15:00.000Z");
        assert_eq!(next("*/10 * * * * ?", "2024-01-01T10:07:55Z"), "2024-01-01T10:08:00.000Z");
        assert_eq!(next("0 0 12 ? * MON-FRI", "2024-01-06T00:00:00Z"), "2024-01-08T12:00:00.000Z"); // Sat → Mon
        assert_eq!(next("0 0 0 1 JAN ?", "2024-03-01T00:00:00Z"), "2025-01-01T00:00:00.000Z");
        assert_eq!(next("0 0 0 29 2 ?", "2024-03-01T00:00:00Z"), "2028-02-29T00:00:00.000Z");
        assert_eq!(next("0 0 22-2 * * ?", "2024-01-01T03:00:00Z"), "2024-01-01T22:00:00.000Z");
        assert_eq!(next("0 0 22-2 * * ?", "2024-01-01T23:30:00Z"), "2024-01-02T00:00:00.000Z");
        assert_eq!(next("0 0 0 * * ? 2030", "2024-01-01T00:00:00Z"), "2030-01-01T00:00:00.000Z");
        assert!(Cron::parse("0 0 0 * * ? 2020").unwrap().next_after(ms("2024-01-01T00:00:00Z")).is_none());
    }

    #[test]
    fn special_days() {
        // Last day of month.
        assert_eq!(next("0 0 0 L * ?", "2024-02-10T00:00:00Z"), "2024-02-29T00:00:00.000Z");
        assert_eq!(next("0 0 0 L-2 * ?", "2024-02-10T00:00:00Z"), "2024-02-27T00:00:00.000Z");
        // 15W: 2024-06-15 is a Saturday → Friday 14th.
        assert_eq!(next("0 0 0 15W * ?", "2024-06-01T00:00:00Z"), "2024-06-14T00:00:00.000Z");
        // 1W: 2024-06-01 is Saturday → Monday 3rd.
        assert_eq!(next("0 0 0 1W * ?", "2024-05-31T12:00:00Z"), "2024-06-03T00:00:00.000Z");
        // LW: 2024-08-31 is Saturday → Friday 30th.
        assert_eq!(next("0 0 0 LW * ?", "2024-08-01T00:00:00Z"), "2024-08-30T00:00:00.000Z");
        // 6#3: third Friday (Jan 2024 → 19th).
        assert_eq!(next("0 0 0 ? * 6#3", "2024-01-01T00:00:00Z"), "2024-01-19T00:00:00.000Z");
        // 6L: last Friday of Jan 2024 → 26th.
        assert_eq!(next("0 0 0 ? * 6L", "2024-01-01T00:00:00Z"), "2024-01-26T00:00:00.000Z");
        assert_eq!(next("0 0 0 ? * FRIL", "2024-01-01T00:00:00Z"), "2024-01-26T00:00:00.000Z");
        // L in dow = Saturday.
        assert_eq!(next("0 0 0 ? * L", "2024-01-01T00:00:00Z"), "2024-01-06T00:00:00.000Z");
        assert_eq!(next("0 0 0 ? * SUN", "2024-01-01T00:00:00Z"), "2024-01-07T00:00:00.000Z");
    }

    #[test]
    fn invalid_expressions() {
        for bad in [
            "0 0 0 * *",
            "0 0 0 * * *",
            "0 0 0 ? * ?",
            "60 0 0 * * ?",
            "0 0 24 * * ?",
            "0 0 0 32 * ?",
            "0 0 0 * 13 ?",
            "0 0 0 ? * 8",
            "0 0 0 ? * 6#6",
            "0 0 0 x * ?",
            "0 0/0 0 * * ?",
            "0 L 0 * * ?",
        ] {
            assert!(Cron::parse(bad).is_err(), "{bad} should fail");
        }
        let e = Cron::parse("0 0 0 * * *").unwrap_err();
        assert_eq!(
            e.0,
            "invalid cron expression [0 0 0 * * *]: support for specifying both a day-of-week AND a day-of-month parameter is not implemented."
        );
    }
}
