//! `java.time.format.DateTimeFormatter` constants and patterns for the
//! (UTC-only) script date values.

use chrono::{Datelike, Timelike};

const DAY_NAMES: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTH_NAMES: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// Marker prefix of a formatter value built by `DateTimeFormatter.ofPattern`.
pub const PATTERN_PREFIX: &str = "pattern:";

pub fn is_formatter(s: &str) -> bool {
    s.starts_with(PATTERN_PREFIX) || matches!(s, "ISO_INSTANT" | "ISO_DATE_TIME" | "ISO_OFFSET_DATE_TIME" | "ISO_ZONED_DATE_TIME" | "ISO_LOCAL_DATE" | "ISO_LOCAL_TIME" | "ISO_LOCAL_DATE_TIME" | "ISO_DATE" | "ISO_TIME" | "ISO_OFFSET_DATE" | "ISO_OFFSET_TIME" | "ISO_ORDINAL_DATE" | "ISO_WEEK_DATE" | "BASIC_ISO_DATE" | "RFC_1123_DATE_TIME")
}

/// ISO local time with a fraction of as many digits as needed (`HH:mm:ss[.SSS]`).
fn local_time(dt: &chrono::NaiveDateTime) -> String {
    let mut s = format!("{:02}:{:02}:{:02}", dt.hour(), dt.minute(), dt.second());
    let ms = dt.nanosecond() / 1_000_000;
    if ms > 0 {
        s.push('.');
        s.push_str(format!("{ms:03}").trim_end_matches('0'));
    }
    s
}

pub fn format(spec: &str, ms: i64) -> Result<String, String> {
    let dt = crate::date::from_millis(ms);
    let date = format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day());
    Ok(match spec {
        "ISO_LOCAL_DATE" => date,
        "ISO_LOCAL_TIME" => local_time(&dt),
        "ISO_LOCAL_DATE_TIME" => format!("{date}T{}", local_time(&dt)),
        "ISO_OFFSET_DATE_TIME" | "ISO_DATE_TIME" | "ISO_ZONED_DATE_TIME" => format!("{date}T{}Z", local_time(&dt)),
        "ISO_INSTANT" => {
            let msp = dt.nanosecond() / 1_000_000;
            if msp > 0 {
                format!("{date}T{:02}:{:02}:{:02}.{msp:03}Z", dt.hour(), dt.minute(), dt.second())
            } else {
                format!("{date}T{:02}:{:02}:{:02}Z", dt.hour(), dt.minute(), dt.second())
            }
        }
        "ISO_DATE" | "ISO_OFFSET_DATE" => format!("{date}Z"),
        "ISO_TIME" | "ISO_OFFSET_TIME" => format!("{}Z", local_time(&dt)),
        "ISO_ORDINAL_DATE" => format!("{:04}-{:03}Z", dt.year(), dt.ordinal()),
        "ISO_WEEK_DATE" => {
            let w = dt.iso_week();
            format!("{:04}-W{:02}-{}Z", w.year(), w.week(), dt.weekday().number_from_monday())
        }
        "BASIC_ISO_DATE" => format!("{:04}{:02}{:02}Z", dt.year(), dt.month(), dt.day()),
        "RFC_1123_DATE_TIME" => format!(
            "{}, {} {} {:04} {:02}:{:02}:{:02} GMT",
            &DAY_NAMES[dt.weekday().num_days_from_monday() as usize][..3],
            dt.day(),
            &MONTH_NAMES[dt.month0() as usize][..3],
            dt.year(),
            dt.hour(),
            dt.minute(),
            dt.second()
        ),
        p if p.starts_with(PATTERN_PREFIX) => pattern(&p[PATTERN_PREFIX.len()..], &dt)?,
        other => return Err(format!("Unknown formatter [{other}]")),
    })
}

/// Java `DateTimeFormatter.ofPattern` letters used in practice.
fn pattern(p: &str, dt: &chrono::NaiveDateTime) -> Result<String, String> {
    let chars: Vec<char> = p.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            i += 1;
            if chars.get(i) == Some(&'\'') {
                out.push('\'');
                i += 1;
                continue;
            }
            while i < chars.len() && chars[i] != '\'' {
                out.push(chars[i]);
                i += 1;
            }
            i += 1;
            continue;
        }
        if !c.is_ascii_alphabetic() {
            out.push(c);
            i += 1;
            continue;
        }
        let mut n = 1;
        while chars.get(i + n) == Some(&c) {
            n += 1;
        }
        i += n;
        let pad = |v: i64, w: usize| format!("{v:0w$}");
        match c {
            'y' | 'u' => out.push_str(&if n == 2 { pad((dt.year() % 100) as i64, 2) } else { pad(dt.year() as i64, n.max(1)) }),
            'M' | 'L' => out.push_str(&match n {
                1 => dt.month().to_string(),
                2 => pad(dt.month() as i64, 2),
                3 => MONTH_NAMES[dt.month0() as usize][..3].to_string(),
                _ => MONTH_NAMES[dt.month0() as usize].to_string(),
            }),
            'd' => out.push_str(&pad(dt.day() as i64, n)),
            'D' => out.push_str(&pad(dt.ordinal() as i64, n)),
            'H' => out.push_str(&pad(dt.hour() as i64, n)),
            'k' => out.push_str(&pad(if dt.hour() == 0 { 24 } else { dt.hour() } as i64, n)),
            'h' => out.push_str(&pad(if dt.hour() % 12 == 0 { 12 } else { dt.hour() % 12 } as i64, n)),
            'K' => out.push_str(&pad((dt.hour() % 12) as i64, n)),
            'm' => out.push_str(&pad(dt.minute() as i64, n)),
            's' => out.push_str(&pad(dt.second() as i64, n)),
            'S' => {
                let f = format!("{:09}", dt.nanosecond());
                out.push_str(&f[..n.min(9)]);
            }
            'n' => out.push_str(&pad(dt.nanosecond() as i64, n)),
            'a' => out.push_str(if dt.hour() < 12 { "AM" } else { "PM" }),
            'E' => {
                let name = DAY_NAMES[dt.weekday().num_days_from_monday() as usize];
                out.push_str(if n >= 4 { name } else { &name[..3] });
            }
            'e' => out.push_str(&pad((dt.weekday().num_days_from_sunday() + 1) as i64, n)),
            'X' | 'x' => out.push_str(if c == 'X' { "Z" } else { "+00" }),
            'Z' => out.push_str("+0000"),
            'z' | 'V' => out.push_str(if n >= 4 { "Coordinated Universal Time" } else { "UTC" }),
            other => return Err(format!("Unknown pattern letter: {other}")),
        }
    }
    Ok(out)
}
