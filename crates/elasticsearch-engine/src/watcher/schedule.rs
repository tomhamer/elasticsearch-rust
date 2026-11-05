//! Watcher trigger schedules: interval, hourly, daily, weekly, monthly,
//! yearly and cron. Everything except `interval` compiles to Quartz cron
//! expressions evaluated by the shared [`crate::ilm::slm::Cron`].

use crate::error::{EsError, Result};
use crate::ilm::slm::Cron;
use serde_json::{json, Value};

pub enum Kind {
    /// Fixed interval in millis, anchored at the watch start time.
    Interval(i64),
    Crons(Vec<Cron>),
}

pub struct Schedule {
    pub kind: Kind,
    /// Schedule type name as it appears in the watch (`interval`, `cron`, ...).
    pub name: &'static str,
    /// The canonical JSON echoed back by GET.
    pub json: Value,
    /// `timezone` of a cron-based schedule (crons evaluate in local time).
    pub tz: Option<crate::date::Tz>,
}

fn bad(msg: impl Into<String>) -> EsError {
    EsError::new(400, "illegal_argument_exception", msg)
}

fn parse_err(msg: impl Into<String>) -> EsError {
    EsError::new(400, "element_parse_exception", msg)
}

/// `10s`, `5m`, `1h`, `2d`, `1w`; a bare number is seconds.
pub fn parse_interval(v: &Value) -> Result<i64> {
    match v {
        Value::Number(n) => {
            let secs = n.as_f64().unwrap_or(0.0);
            let ms = (secs * 1000.0) as i64;
            if ms < 1000 {
                return Err(bad(format!("interval can't be lower than 1000 ms, but [{}s] was", n)));
            }
            Ok(ms)
        }
        Value::String(s) => {
            let s = s.trim();
            let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
            let n: i64 = num.parse().map_err(|_| parse_err(format!("could not parse interval [{s}]")))?;
            let mult = match unit {
                "" | "s" => 1000,
                "m" => 60_000,
                "h" => 3_600_000,
                "d" => 86_400_000,
                "w" => 604_800_000,
                _ => return Err(parse_err(format!("could not parse interval [{s}]. unrecognized unit [{unit}]"))),
            };
            if n * mult < 1000 {
                return Err(bad(format!("interval can't be lower than 1000 ms, but [{s}] was specified")));
            }
            Ok(n * mult)
        }
        _ => Err(parse_err("could not parse interval. expected a string or number")),
    }
}

fn as_list(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.clone(),
        other => vec![other.clone()],
    }
}

fn int_list(v: &Value, what: &str, lo: i64, hi: i64) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for x in as_list(v) {
        let n = match &x {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
        .ok_or_else(|| parse_err(format!("invalid {what} value [{x}]")))?;
        if n < lo || n > hi {
            return Err(parse_err(format!("invalid {what} value [{n}]. {what} must be between {lo} and {hi}")));
        }
        out.push(n);
    }
    Ok(out)
}

/// One `at` entry → (hours, minutes).
fn parse_time(v: &Value) -> Result<(Vec<i64>, Vec<i64>)> {
    match v {
        Value::String(s) => match s.as_str() {
            "midnight" => Ok((vec![0], vec![0])),
            "noon" => Ok((vec![12], vec![0])),
            other => {
                let (h, m) = other.split_once(':').ok_or_else(|| parse_err(format!("could not parse time [{other}]. time format is HH:mm")))?;
                let h: i64 = h.parse().map_err(|_| parse_err(format!("could not parse time [{other}]. time format is HH:mm")))?;
                let m: i64 = m.parse().map_err(|_| parse_err(format!("could not parse time [{other}]. time format is HH:mm")))?;
                if !(0..=23).contains(&h) || !(0..=59).contains(&m) {
                    return Err(parse_err(format!("could not parse time [{other}]. time format is HH:mm")));
                }
                Ok((vec![h], vec![m]))
            }
        },
        Value::Object(o) => {
            let h = match o.get("hour") {
                Some(h) => int_list(h, "hour", 0, 23)?,
                None => return Err(parse_err("could not parse time. missing required [hour] field")),
            };
            let m = match o.get("minute") {
                Some(m) => int_list(m, "minute", 0, 59)?,
                None => vec![0],
            };
            Ok((h, m))
        }
        _ => Err(parse_err("could not parse time. expected a string or object")),
    }
}

fn times(v: Option<&Value>) -> Result<Vec<(Vec<i64>, Vec<i64>)>> {
    match v {
        None => Ok(vec![(vec![0], vec![0])]),
        Some(v) => as_list(v).iter().map(parse_time).collect(),
    }
}

fn join(v: &[i64]) -> String {
    let mut v = v.to_vec();
    v.sort();
    v.dedup();
    v.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
}

const DAYS: [&str; 7] = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
const MONTHS: [&str; 12] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

fn day_name(v: &Value) -> Result<&'static str> {
    let s = match v {
        Value::String(s) => s.to_ascii_lowercase(),
        Value::Number(n) => return DAYS.get((n.as_i64().unwrap_or(0) % 7) as usize).copied().ok_or_else(|| parse_err("invalid day")),
        _ => return Err(parse_err("invalid day of week")),
    };
    for (i, d) in DAYS.iter().enumerate() {
        let full = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"][i];
        if s == full || s == d.to_ascii_lowercase() {
            return Ok(d);
        }
    }
    Err(parse_err(format!("invalid day of week [{s}]")))
}

fn month_name(v: &Value) -> Result<&'static str> {
    match v {
        Value::Number(n) => {
            let i = n.as_i64().unwrap_or(0);
            if (1..=12).contains(&i) {
                Ok(MONTHS[(i - 1) as usize])
            } else {
                Err(parse_err(format!("invalid month [{i}]")))
            }
        }
        Value::String(s) => {
            let l = s.to_ascii_lowercase();
            let full = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
            for (i, m) in MONTHS.iter().enumerate() {
                if l == full[i] || l == m.to_ascii_lowercase() {
                    return Ok(m);
                }
            }
            if let Ok(i) = l.parse::<usize>() {
                if (1..=12).contains(&i) {
                    return Ok(MONTHS[i - 1]);
                }
            }
            Err(parse_err(format!("invalid month [{s}]")))
        }
        _ => Err(parse_err("invalid month")),
    }
}

fn day_of_month(v: &Value) -> Result<String> {
    match v {
        Value::String(s) if s == "last_day" => Ok("L".into()),
        Value::String(s) if s == "first_day" => Ok("1".into()),
        other => Ok(join(&int_list(other, "day", 1, 31)?)),
    }
}

fn cron_from(sec_min_hour: (&str, &str), dom: &str, month: &str, dow: &str) -> String {
    format!("0 {} {} {} {} {}", sec_min_hour.0, sec_min_hour.1, dom, month, dow)
}

impl Schedule {
    pub fn parse(v: &Value) -> Result<Schedule> {
        let Some(o) = v.as_object() else { return Self::parse_one(v) };
        let Some(tzv) = o.get("timezone") else { return Self::parse_one(v) };
        let text = tzv.as_str().ok_or_else(|| {
            EsError::new(400, "parse_exception", format!("could not parse schedule. expected a string value for [timezone], but found [{}] instead", crate::jsonpos::token_name(tzv)))
        })?;
        let tz = crate::date::Tz::parse(text).map_err(|e| {
            EsError::new(400, "parse_exception", format!("could not parse schedule. invalid timezone [{text}]")).caused_by(EsError::new(400, "zone_rules_exception", e.reason))
        })?;
        let mut rest = o.clone();
        rest.remove("timezone");
        let mut sc = Self::parse_one(&Value::Object(rest))?;
        if matches!(sc.kind, Kind::Interval(_)) {
            return Err(EsError::new(400, "parse_exception", format!("could not parse schedule. Timezone is not supported for schedule type [{}]", sc.name)));
        }
        sc.tz = Some(tz);
        Ok(sc)
    }

    fn parse_one(v: &Value) -> Result<Schedule> {
        let o = v.as_object().ok_or_else(|| parse_err("could not parse schedule. expected an object"))?;
        if o.is_empty() {
            return Err(parse_err("could not parse schedule. expected a schedule type field, but no fields were found"));
        }
        if o.len() > 1 {
            return Err(parse_err(format!("could not parse schedule. expected a single schedule type, but found [{}]", o.keys().cloned().collect::<Vec<_>>().join(", "))));
        }
        let (ty, body) = o.iter().next().unwrap();
        match ty.as_str() {
            "interval" => {
                let ms = parse_interval(body)?;
                Ok(Schedule { kind: Kind::Interval(ms), name: "interval", json: json!({"interval": body.clone()}), tz: None })
            }
            "cron" => {
                let exprs: Vec<String> = as_list(body).iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
                if exprs.is_empty() {
                    return Err(parse_err("could not parse cron schedule. expected a string or an array of strings"));
                }
                let mut crons = Vec::new();
                for e in &exprs {
                    crons.push(Cron::parse(e).map_err(|m| EsError::new(400, "illegal_argument_exception", format!("invalid cron expression [{e}]. {m}")))?);
                }
                Ok(Schedule { kind: Kind::Crons(crons), name: "cron", json: json!({"cron": body.clone()}), tz: None })
            }
            "hourly" => {
                let mins = match body.get("minute") {
                    Some(m) => int_list(m, "minute", 0, 59)?,
                    None => vec![0],
                };
                let e = cron_from((join(&mins).as_str(), "*"), "*", "*", "?");
                Self::from_crons("hourly", vec![e], v)
            }
            "daily" => {
                let mut exprs = Vec::new();
                for (h, m) in times(body.get("at"))? {
                    exprs.push(cron_from((join(&m).as_str(), join(&h).as_str()), "*", "*", "?"));
                }
                Self::from_crons("daily", exprs, v)
            }
            "weekly" => {
                let entries = as_list(body);
                let mut exprs = Vec::new();
                for ent in entries {
                    let days: Vec<&str> = as_list(ent.get("on").ok_or_else(|| parse_err("could not parse weekly schedule. missing required [on] field"))?)
                        .iter()
                        .map(day_name)
                        .collect::<Result<_>>()?;
                    for (h, m) in times(ent.get("at"))? {
                        exprs.push(cron_from((join(&m).as_str(), join(&h).as_str()), "?", "*", &days.join(",")));
                    }
                }
                Self::from_crons("weekly", exprs, v)
            }
            "monthly" => {
                let mut exprs = Vec::new();
                for ent in as_list(body) {
                    let dom = day_of_month(ent.get("on").ok_or_else(|| parse_err("could not parse monthly schedule. missing required [on] field"))?)?;
                    for (h, m) in times(ent.get("at"))? {
                        exprs.push(cron_from((join(&m).as_str(), join(&h).as_str()), &dom, "*", "?"));
                    }
                }
                Self::from_crons("monthly", exprs, v)
            }
            "yearly" => {
                let mut exprs = Vec::new();
                for ent in as_list(body) {
                    let months: Vec<&str> = as_list(ent.get("in").ok_or_else(|| parse_err("could not parse yearly schedule. missing required [in] field"))?)
                        .iter()
                        .map(month_name)
                        .collect::<Result<_>>()?;
                    let dom = day_of_month(ent.get("on").ok_or_else(|| parse_err("could not parse yearly schedule. missing required [on] field"))?)?;
                    for (h, m) in times(ent.get("at"))? {
                        exprs.push(cron_from((join(&m).as_str(), join(&h).as_str()), &dom, &months.join(","), "?"));
                    }
                }
                Self::from_crons("yearly", exprs, v)
            }
            other => Err(parse_err(format!("could not parse schedule. unknown schedule type [{other}]"))),
        }
    }

    fn from_crons(name: &'static str, exprs: Vec<String>, original: &Value) -> Result<Schedule> {
        let mut crons = Vec::new();
        for e in &exprs {
            crons.push(Cron::parse(e).map_err(|m| bad(format!("invalid cron expression [{e}]. {m}")))?);
        }
        Ok(Schedule { kind: Kind::Crons(crons), name, json: original.clone(), tz: None })
    }

    /// Next fire time strictly after `after`; intervals are anchored at `start`.
    pub fn next_after(&self, start: i64, after: i64) -> Option<i64> {
        match &self.kind {
            Kind::Interval(ms) => {
                if after < start {
                    return Some(start);
                }
                Some(start + ((after - start) / ms + 1) * ms)
            }
            Kind::Crons(cs) => match self.tz {
                None => cs.iter().filter_map(|c| c.next_after(after)).min(),
                Some(tz) => {
                    // evaluate the crons on the local wall clock
                    let local_after = after + tz.offset_at(after) as i64 * 1000;
                    cs.iter()
                        .filter_map(|c| c.next_after(local_after))
                        .map(|l| tz.local_to_utc(crate::date::from_millis(l)))
                        .filter(|t| *t > after)
                        .min()
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> i64 {
        crate::date::DateFormatter::new("strict_date_optional_time").unwrap().parse(s).unwrap()
    }

    fn fires(s: &Value, start: &str, after: &str, n: usize) -> Vec<i64> {
        let sc = Schedule::parse(s).unwrap();
        let (start, mut t) = (at(start), at(after));
        let mut out = Vec::new();
        for _ in 0..n {
            t = sc.next_after(start, t).unwrap();
            out.push(t);
        }
        out
    }

    #[test]
    fn interval_anchor() {
        let f = fires(&json!({"interval": "10s"}), "2020-01-01T00:00:03Z", "2020-01-01T00:00:15Z", 2);
        assert_eq!(f, vec![at("2020-01-01T00:00:23Z"), at("2020-01-01T00:00:33Z")]);
        assert_eq!(fires(&json!({"interval": "5m"}), "2020-01-01T00:00:00Z", "2019-12-31T00:00:00Z", 1), vec![at("2020-01-01T00:00:00Z")]);
        assert!(Schedule::parse(&json!({"interval": "500ms"})).is_err());
    }

    #[test]
    fn calendar_schedules() {
        let s = "2020-01-01T00:00:00Z";
        assert_eq!(fires(&json!({"hourly": {"minute": [0, 30]}}), s, "2020-01-01T10:10:00Z", 2), vec![at("2020-01-01T10:30:00Z"), at("2020-01-01T11:00:00Z")]);
        assert_eq!(fires(&json!({"daily": {"at": ["noon", "17:30"]}}), s, "2020-01-01T13:00:00Z", 2), vec![at("2020-01-01T17:30:00Z"), at("2020-01-02T12:00:00Z")]);
        // 2020-01-01 is a Wednesday
        assert_eq!(fires(&json!({"weekly": {"on": ["friday", "mon"], "at": "09:00"}}), s, "2020-01-01T10:00:00Z", 2), vec![at("2020-01-03T09:00:00Z"), at("2020-01-06T09:00:00Z")]);
        assert_eq!(fires(&json!({"monthly": {"on": 15, "at": "midnight"}}), s, "2020-01-16T00:00:00Z", 1), vec![at("2020-02-15T00:00:00Z")]);
        assert_eq!(fires(&json!({"monthly": {"on": "last_day", "at": "noon"}}), s, "2020-01-01T00:00:00Z", 2), vec![at("2020-01-31T12:00:00Z"), at("2020-02-29T12:00:00Z")]);
        assert_eq!(fires(&json!({"yearly": {"in": "feb", "on": 29, "at": "06:15"}}), s, "2020-03-01T00:00:00Z", 1), vec![at("2024-02-29T06:15:00Z")]);
        assert_eq!(fires(&json!({"cron": "0 0/5 * * * ?"}), s, "2020-01-01T00:01:00Z", 2), vec![at("2020-01-01T00:05:00Z"), at("2020-01-01T00:10:00Z")]);
        assert_eq!(fires(&json!({"cron": ["0 0 1 * * ?", "0 0 2 * * ?"]}), s, "2020-01-01T01:30:00Z", 1), vec![at("2020-01-01T02:00:00Z")]);
    }
}
