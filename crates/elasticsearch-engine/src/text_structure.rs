//! `_text_structure`: `find_structure` (raw text), `find_message_structure`
//! (a list of messages), `find_field_structure` (values of an index field,
//! fetched by the HTTP layer) and `test_grok_pattern`. Kibana's file upload
//! uses `find_structure`.
//!
//! A port of the essentials of ES's `x-pack/plugin/text-structure`:
//! format detection (NDJSON → delimited `,` `\t` `;` `|` → semi-structured
//! text), timestamp format detection (the ES candidate formats), mapping
//! guesses + field stats (`TextStructureUtils` / `FieldStatsCalculator`),
//! the Grok pattern creator for log lines (value-only candidates, the
//! intermediate-regex and finalisation rules) and the generated ingest
//! pipeline. Not ported: XML, key=value Grok candidates, ICU charset
//! detection (valid UTF-8 → `UTF-8`, otherwise `ISO-8859-1`), timeouts.

use crate::error::{EsError, Result};
use crate::ingest::grok::{self, Grok};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

fn ia(m: impl Into<String>) -> EsError {
    EsError::illegal_argument(m)
}

// ------------------------------------------------------------ overrides

/// Request parameters (`TextStructureOverrides` + sampling options).
#[derive(Clone, Debug, Default)]
pub struct Overrides {
    pub charset: Option<String>,
    pub format: Option<String>,
    pub column_names: Option<Vec<String>>,
    pub has_header_row: Option<bool>,
    pub delimiter: Option<char>,
    pub quote: Option<char>,
    pub should_trim_fields: Option<bool>,
    pub grok_pattern: Option<String>,
    pub timestamp_field: Option<String>,
    pub timestamp_format: Option<String>,
    pub ecs_compatibility: Option<String>,
    pub lines_to_sample: usize,
    pub line_merge_size_limit: usize,
    pub explain: bool,
}

impl Overrides {
    /// From query parameters.
    pub fn from_params(p: &HashMap<String, String>) -> Result<Overrides> {
        let get = |k: &str| p.get(k).filter(|v| !v.is_empty()).cloned();
        let bool_p = |k: &str| -> Result<Option<bool>> {
            match p.get(k).map(String::as_str) {
                None => Ok(None),
                Some("true") | Some("") => Ok(Some(true)),
                Some("false") => Ok(Some(false)),
                Some(v) => Err(ia(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
            }
        };
        let usize_p = |k: &str, d: usize| -> Result<usize> {
            match p.get(k) {
                None => Ok(d),
                Some(v) => v.parse::<usize>().map_err(|_| ia(format!("Failed to parse int parameter [{k}] with value [{v}]"))),
            }
        };
        let ch = |k: &str| -> Result<Option<char>> {
            match get(k) {
                None => Ok(None),
                Some(s) => {
                    let mut it = s.chars();
                    let c = it.next().unwrap();
                    if it.next().is_some() {
                        return Err(ia(format!("[{k}] must be a single character")));
                    }
                    Ok(Some(c))
                }
            }
        };
        let format = get("format").map(|f| f.to_lowercase());
        if let Some(f) = &format {
            let f = if f == "json" { "ndjson" } else { f.as_str() };
            if !matches!(f, "ndjson" | "xml" | "delimited" | "semi_structured_text") {
                return Err(ia(format!("No enum constant org.elasticsearch.xpack.core.textstructure.structurefinder.TextStructure.Format.{}", f.to_uppercase())));
            }
        }
        let lines_to_sample = usize_p("lines_to_sample", 1000)?;
        if lines_to_sample < 2 {
            return Err(EsError::action_validation(&[&format!("[lines_to_sample] must be at least [2] if specified")]));
        }
        Ok(Overrides {
            charset: get("charset"),
            format: format.map(|f| if f == "json" { "ndjson".into() } else { f }),
            column_names: get("column_names").map(|s| s.split(',').map(str::to_string).collect()),
            has_header_row: bool_p("has_header_row")?,
            delimiter: ch("delimiter")?,
            quote: ch("quote")?,
            should_trim_fields: bool_p("should_trim_fields")?,
            grok_pattern: get("grok_pattern"),
            timestamp_field: get("timestamp_field"),
            timestamp_format: get("timestamp_format"),
            ecs_compatibility: get("ecs_compatibility"),
            lines_to_sample,
            line_merge_size_limit: usize_p("line_merge_size_limit", 10_000)?,
            explain: bool_p("explain")?.unwrap_or(false),
        })
    }
}

// ------------------------------------------------------------ timestamp formats

/// One recognised timestamp format.
#[derive(Clone, Debug, PartialEq)]
struct TsFormat {
    /// Java formats (`ISO8601`, `UNIX`, `UNIX_MS`, `TAI64N` or patterns).
    java: Vec<String>,
    /// Grok pattern name (or `CUSTOM_TIMESTAMP` with `custom`).
    grok: String,
    custom: Option<String>,
    /// Simple regex used for the multi-line start pattern.
    simple: String,
    tz_dependent: bool,
    nanos: bool,
}

impl TsFormat {
    fn joda(&self) -> Vec<String> {
        self.java.iter().map(|f| f.replace("yy", "YY").replace("XXX", "ZZ").replace("XX", "Z")).collect()
    }

    fn mapping_with_format(&self) -> Value {
        if self.java.iter().any(|f| f == "TAI64N") {
            return json!({"type": "keyword"});
        }
        let fmts: Vec<String> = self
            .java
            .iter()
            .map(|f| match f.as_str() {
                "ISO8601" => "iso8601".to_string(),
                "UNIX_MS" => "epoch_millis".to_string(),
                "UNIX" => "epoch_second".to_string(),
                other => other.to_string(),
            })
            .collect();
        let mut m = Map::new();
        m.insert("type".into(), json!(if self.nanos { "date_nanos" } else { "date" }));
        if !fmts.is_empty() {
            m.insert("format".into(), json!(fmts.join("||")));
        }
        Value::Object(m)
    }

    fn mapping_without_format(&self) -> Value {
        json!({"type": if self.nanos { "date_nanos" } else { "date" }})
    }

    /// Epoch millis of a value in this format (field stats ordering).
    fn millis(&self, v: &str) -> Option<i64> {
        let f = self.java.first()?;
        match f.as_str() {
            "UNIX" => v.parse::<f64>().ok().map(|x| (x * 1000.0) as i64),
            "UNIX_MS" => v.parse::<i64>().ok(),
            "TAI64N" => None,
            "ISO8601" => crate::date::DateFormatter::new("strict_date_optional_time").ok()?.parse(v).ok(),
            _ => self.java.iter().find_map(|f| crate::date::DateFormatter::new(f).ok()?.parse(v).ok()),
        }
    }
}

struct Candidate {
    grok: &'static str,
    custom: Option<&'static str>,
    re: &'static str,
    simple: &'static str,
    kind: u8,
}

/// ES `TimestampFormatFinder.ORDERED_CANDIDATE_FORMATS` (legacy grok names).
const CANDIDATES: &[Candidate] = &[
    Candidate { grok: "TOMCAT_DATESTAMP", custom: None, re: r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[:.,]\d{3,9} (?:Z|[+-]\d{4})", simple: r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[:.,]\d{3}", kind: 0 },
    Candidate { grok: "TIMESTAMP_ISO8601", custom: None, re: r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?", simple: r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", kind: 1 },
    Candidate { grok: "DATESTAMP_RFC2822", custom: None, re: r"[A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} (?:Z|[+-]\d{2}:?\d{2})", simple: r"\b[A-Z]\S{2}, \d{1,2} [A-Z]\S{2} \d{4} \d{2}:\d{2}:\d{2}\b", kind: 2 },
    Candidate { grok: "HTTPDERROR_DATE", custom: None, re: r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}", simple: r"\b[A-Z]\S{2} [A-Z]\S{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}\b", kind: 3 },
    Candidate { grok: "SYSLOGTIMESTAMP", custom: None, re: r"[A-Z][a-z]{2,8} {1,2}\d{1,2} \d{2}:\d{2}:\d{2}(?:[:.,]\d{3,9})?", simple: r"\b[A-Z]\S{2,8} {1,2}\d{1,2} \d{2}:\d{2}:\d{2}\b", kind: 4 },
    Candidate { grok: "HTTPDATE", custom: None, re: r"\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2} [+-]?\d{4}", simple: r"\b\d{2}/[A-Z]\S{2}/\d{4}:\d{2}:\d{2}:\d{2} ", kind: 5 },
    Candidate { grok: "CATALINA_DATESTAMP", custom: None, re: r"[A-Z][a-z]{2} \d{2}, \d{4} \d{1,2}:\d{2}:\d{2} [AP]M", simple: r"\b[A-Z]\S{2} \d{2}, \d{4} \d{1,2}:\d{2}:\d{2} [AP]M\b", kind: 6 },
    Candidate { grok: "CISCOTIMESTAMP", custom: None, re: r"[A-Z][a-z]{2} {1,2}\d{1,2} \d{4} \d{2}:\d{2}:\d{2}", simple: r"\b[A-Z]\S{2} {1,2}\d{1,2} \d{4} \d{2}:\d{2}:\d{2}\b", kind: 7 },
    Candidate { grok: "DATESTAMP", custom: None, re: r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})[- ]\d{2}:\d{2}:\d{2}", simple: r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{2}){1,2}[- ]\d{2}:\d{2}:\d{2}\b", kind: 8 },
    Candidate { grok: "DATE", custom: None, re: r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})", simple: r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{2}){1,2}\b", kind: 9 },
    Candidate { grok: "POSINT", custom: None, re: r"[12]\d{12}", simple: r"\b\d{13}\b", kind: 10 },
    Candidate { grok: "NUMBER", custom: None, re: r"[12]\d{9}(?:\.\d{3,9})?", simple: r"\b\d{10}\b", kind: 11 },
    Candidate { grok: "BASE16NUM", custom: None, re: r"[0-9A-Fa-f]{24}", simple: r"\b[0-9A-Fa-f]{24}\b", kind: 12 },
    Candidate { grok: "CUSTOM_TIMESTAMP", custom: Some("%{YEAR}-%{MONTHNUM2}-%{MONTHDAY}"), re: r"\d{4}-\d{2}-\d{2}", simple: r"\b\d{4}-\d{2}-\d{2}\b", kind: 13 },
    Candidate {
        grok: "CUSTOM_TIMESTAMP",
        custom: Some("%{MONTH} %{MONTHDAY}, %{YEAR} @ %{HOUR}:%{MINUTE}:%{SECOND}"),
        re: r"[A-Z][a-z]{2} \d{1,2}, \d{4} @ \d{2}:\d{2}:\d{2}\.\d{3}",
        simple: r"\b[A-Z]\S{2} \d{1,2}, \d{4} @ \d{2}:\d{2}:\d{2}\.\d{3}\b",
        kind: 14,
    },
];

struct CompiledCand {
    full: Regex,
    search: Regex,
}

fn compiled() -> &'static Vec<CompiledCand> {
    static C: OnceLock<Vec<CompiledCand>> = OnceLock::new();
    C.get_or_init(|| {
        CANDIDATES
            .iter()
            .map(|c| CompiledCand { full: Regex::new(&format!("^(?:{})$", c.re)).unwrap(), search: Regex::new(&format!(r"\b(?:{})", c.re)).unwrap() })
            .collect()
    })
}

/// ES `iso8601LikeFormatFromExample`.
fn iso_like(example: &str, time_sep: &str, tz_sep: &str) -> String {
    let ex: Vec<char> = example.chars().collect();
    let mut b = format!("yyyy-MM-dd{time_sep}HH:mm");
    let len = |b: &String| b.chars().count();
    if ex.len() > len(&b) && ex[len(&b)] == ':' {
        b.push_str(":ss");
    }
    if ex.len() > len(&b) {
        let next = ex[len(&b)];
        if ".,:".contains(next) {
            b.push(next);
            let mut pos = len(&b);
            while pos < ex.len() && ex[pos].is_ascii_digit() {
                b.push('S');
                pos += 1;
            }
        }
        if ex.len() > len(&b) {
            let rest: String = ex[len(&b)..].iter().collect();
            b.push_str(tz_sep);
            b.push_str(if rest.contains(':') { "XXX" } else { "XX" });
        }
    }
    b
}

fn day_variants(base: &str) -> Vec<String> {
    vec![base.to_string(), base.replacen("dd", " d", 1), base.replacen("dd", "d", 1)]
}

fn indeterminate(example: &str, with_time: bool) -> Vec<String> {
    let re = Regex::new(r"^(\d{1,2})([/.-])(\d{1,2})[/.-](\d{2}|\d{4})").unwrap();
    let Some(c) = re.captures(example) else { return vec![] };
    let a: u32 = c[1].parse().unwrap_or(0);
    let b: u32 = c[3].parse().unwrap_or(0);
    let sep = &c[2];
    let year = if c[4].len() == 4 { "yyyy" } else { "yy" };
    let (f1, f2) = if a > 12 && b <= 12 { ("dd", "MM") } else if b > 12 && a <= 12 { ("MM", "dd") } else { ("dd", "MM") };
    let f1 = if c[1].len() == 1 { &f1[..1] } else { f1 };
    let f2 = if c[3].len() == 1 { &f2[..1] } else { f2 };
    let mut f = format!("{f1}{sep}{f2}{sep}{year}");
    if with_time {
        let tsep = example.chars().nth(c[0].chars().count()).unwrap_or(' ');
        f.push(tsep);
        f.push_str("HH:mm:ss");
    }
    vec![f]
}

fn has_zone(s: &str) -> bool {
    let re = Regex::new(r"(?:Z|[+-]\d{2}:?\d{2})$").unwrap();
    re.is_match(s)
}

fn make_format(c: &Candidate, example: &str) -> TsFormat {
    let (java, tz): (Vec<String>, bool) = match c.kind {
        0 => (vec![iso_like(example, " ", " ")], false),
        1 => {
            if example.contains('T') {
                (vec!["ISO8601".into()], !has_zone(example))
            } else {
                (vec![iso_like(example, " ", "")], !has_zone(example))
            }
        }
        2 => (vec![if Regex::new(r"[+-]\d{4}$").unwrap().is_match(example) { "EEE, dd MMM yyyy HH:mm:ss XX".into() } else { "EEE, dd MMM yyyy HH:mm:ss XXX".into() }], false),
        3 => (vec!["EEE MMM dd HH:mm:ss yyyy".into()], true),
        4 => {
            let frac = Regex::new(r"\d{2}:\d{2}:\d{2}([:.,])(\d+)$").unwrap();
            let base = match frac.captures(example) {
                Some(f) => format!("MMM dd HH:mm:ss{}{}", &f[1], "S".repeat(f[2].len())),
                None => "MMM dd HH:mm:ss".into(),
            };
            (day_variants(&base), true)
        }
        5 => (vec!["dd/MMM/yyyy:HH:mm:ss XX".into()], false),
        6 => (vec!["MMM dd, yyyy h:mm:ss a".into()], true),
        7 => (vec!["MMM dd yyyy HH:mm:ss".into(), "MMM  d yyyy HH:mm:ss".into(), "MMM d yyyy HH:mm:ss".into()], true),
        8 => (indeterminate(example, true), true),
        9 => (indeterminate(example, false), true),
        10 => (vec!["UNIX_MS".into()], false),
        11 => (vec!["UNIX".into()], false),
        12 => (vec!["TAI64N".into()], false),
        13 => (vec!["ISO8601".into()], true),
        _ => (vec!["MMM d, yyyy @ HH:mm:ss.SSS".into()], true),
    };
    let nanos = java.iter().any(|f| f.contains("SSSSSSS"));
    TsFormat { java, grok: c.grok.to_string(), custom: c.custom.map(str::to_string), simple: c.simple.to_string(), tz_dependent: tz, nanos }
}

/// A user-supplied `timestamp_format` turned into a candidate.
fn override_format(fmt: &str) -> std::result::Result<(Regex, Regex, TsFormat), String> {
    match fmt {
        "ISO8601" | "UNIX" | "UNIX_MS" | "TAI64N" => {
            let c = CANDIDATES.iter().find(|c| match fmt {
                "ISO8601" => c.kind == 1,
                "UNIX" => c.kind == 11,
                "UNIX_MS" => c.kind == 10,
                _ => c.kind == 12,
            });
            let c = c.unwrap();
            let re = if fmt == "ISO8601" { r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?)?" } else { c.re };
            let f = TsFormat { java: vec![fmt.into()], grok: c.grok.into(), custom: None, simple: c.simple.into(), tz_dependent: fmt == "ISO8601", nanos: false };
            return Ok((Regex::new(&format!("^(?:{re})$")).unwrap(), Regex::new(&format!(r"\b(?:{re})")).unwrap(), f));
        }
        _ => {}
    }
    let chars: Vec<char> = fmt.chars().collect();
    let mut re = String::new();
    let mut gk = String::new();
    let mut zone = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\'' {
            let mut j = i + 1;
            let mut lit = String::new();
            while j < chars.len() && chars[j] != '\'' {
                lit.push(chars[j]);
                j += 1;
            }
            re.push_str(&regex::escape(&lit));
            gk.push_str(&regex::escape(&lit));
            i = j + 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            let mut n = 1;
            while i + n < chars.len() && chars[i + n] == c {
                n += 1;
            }
            let (r, g): (String, String) = match (c, n) {
                ('y', 2) | ('Y', 2) => (r"\d{2}".into(), "%{YEAR}".into()),
                ('y', _) | ('Y', _) | ('u', _) => (r"\d{4}".into(), "%{YEAR}".into()),
                ('M', 1) => (r"\d{1,2}".into(), "%{MONTHNUM}".into()),
                ('M', 2) => (r"\d{2}".into(), "%{MONTHNUM2}".into()),
                ('M', _) | ('L', _) => (r"[A-Z][a-z]{2,8}".into(), "%{MONTH}".into()),
                ('d', 1) => (r"\d{1,2}".into(), "%{MONTHDAY}".into()),
                ('d', _) => (r"\d{2}".into(), "%{MONTHDAY}".into()),
                ('H', _) | ('h', _) | ('k', _) | ('K', _) => (r"\d{1,2}".into(), "%{HOUR}".into()),
                ('m', _) => (r"\d{2}".into(), "%{MINUTE}".into()),
                ('s', _) => (r"\d{2}".into(), "%{SECOND}".into()),
                ('S', n) => (format!(r"\d{{{n}}}"), format!("[0-9]{{{n}}}")),
                ('E', _) => (r"[A-Z][a-z]{2,8}".into(), "%{DAY}".into()),
                ('a', _) => (r"[AaPp][Mm]".into(), "(?:AM|PM|am|pm)".into()),
                ('X', _) | ('Z', _) | ('x', _) => {
                    zone = true;
                    (r"(?:Z|[+-]\d{2}(?::?\d{2})?)".into(), "(?:Z|[+-]%{HOUR}(?::?%{MINUTE})?)".into())
                }
                ('z', _) | ('V', _) => {
                    zone = true;
                    (r"[A-Za-z][A-Za-z_/+-]+".into(), "%{TZ}".into())
                }
                _ => return Err(format!("Invalid format: [{fmt}]")),
            };
            re.push_str(&r);
            gk.push_str(&g);
            i += n;
            continue;
        }
        re.push_str(&regex::escape(&c.to_string()));
        gk.push_str(&regex::escape(&c.to_string()));
        i += 1;
    }
    let full = Regex::new(&format!("^(?:{re})$")).map_err(|e| e.to_string())?;
    let search = Regex::new(&format!(r"\b(?:{re})")).map_err(|e| e.to_string())?;
    let simple = format!(r"\b{re}");
    let f = TsFormat { java: vec![fmt.to_string()], grok: "CUSTOM_TIMESTAMP".into(), custom: Some(gk), simple, tz_dependent: !zone, nanos: fmt.contains("SSSSSSS") };
    Ok((full, search, f))
}

/// Full-match timestamp format of a field value.
fn value_format(v: &str, ov: &Overrides) -> Result<Option<TsFormat>> {
    if let Some(f) = &ov.timestamp_format {
        let (full, _, tf) = override_format(f).map_err(ia)?;
        return Ok(if full.is_match(v) { Some(tf) } else { None });
    }
    for (c, cc) in CANDIDATES.iter().zip(compiled()) {
        if cc.full.is_match(v) {
            return Ok(Some(make_format(c, v)));
        }
    }
    Ok(None)
}

/// `findTimestampMapping`: all values in one compatible format.
fn timestamp_mapping(values: &[String], ov: &Overrides) -> Result<Option<(TsFormat, Value)>> {
    let mut first: Option<TsFormat> = None;
    for v in values {
        match value_format(v, ov)? {
            None => return Ok(None),
            Some(f) => match &first {
                None => first = Some(f),
                Some(p) if p.grok == f.grok && p.java.len() == f.java.len() => {
                    if p.java != f.java && p.grok != "SYSLOGTIMESTAMP" && p.grok != "CISCOTIMESTAMP" {
                        // e.g. ISO8601 vs explicit pattern: merge by keeping the first
                    }
                    if f.tz_dependent && !p.tz_dependent {
                        first = Some(TsFormat { tz_dependent: true, ..p.clone() });
                    }
                }
                Some(_) => return Ok(None),
            },
        }
    }
    Ok(first.map(|f| {
        let m = f.mapping_with_format();
        (f, m)
    }))
}

// ------------------------------------------------------------ mappings and field stats

fn java_string(v: &Value) -> String {
    crate::ingest::processors::java_string(v)
}

fn number_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-3]?[0-9]{1,2})?$").unwrap())
}

fn ip_grok() -> &'static Grok {
    static G: OnceLock<Grok> = OnceLock::new();
    G.get_or_init(|| Grok::compile("^%{IP}$", grok::bank()).unwrap())
}

fn is_text(s: &str) -> bool {
    if s.chars().count() > 256 {
        return true;
    }
    s.chars().filter(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')).count() > 5
}

/// ES `guessScalarMapping`.
fn guess_scalar(values: &[String], ov: &Overrides) -> Result<Value> {
    if values.iter().all(|v| v == "true" || v == "false") {
        return Ok(json!({"type": "boolean"}));
    }
    if let Some((_, m)) = timestamp_mapping(values, ov)? {
        return Ok(m);
    }
    if values.iter().all(|v| number_re().is_match(v)) {
        if values.iter().all(|v| v.parse::<i64>().is_ok()) {
            return Ok(json!({"type": "long"}));
        }
        if values.iter().all(|v| v.parse::<f64>().is_ok()) {
            return Ok(json!({"type": "double"}));
        }
    } else if values.iter().all(|v| ip_grok().re.is_match(v)) {
        return Ok(json!({"type": "ip"}));
    } else if values.iter().all(|v| Regex::new(r"^POINT \([+-]?[0-9.]+ [+-]?[0-9.]+\)$").unwrap().is_match(v)) {
        return Ok(json!({"type": "geo_point"}));
    } else if values.iter().all(|v| Regex::new(r"^(?:LINESTRING|MULTIPOINT|POLYGON|MULTILINESTRING|MULTIPOLYGON|BBOX|GEOMETRYCOLLECTION) \(.*\)$").unwrap().is_match(v)) {
        return Ok(json!({"type": "geo_shape"}));
    }
    if values.iter().any(|v| is_text(v)) {
        return Ok(json!({"type": "text"}));
    }
    Ok(json!({"type": "keyword"}))
}

/// `FieldStats.toIntegerIfInteger`.
fn num_json(d: f64) -> Value {
    if d.fract() == 0.0 && d.abs() < (1u64 << 53) as f64 {
        json!(d as i64)
    } else {
        json!(d)
    }
}

/// ES `FieldStatsCalculator`.
fn field_stats(mapping: &Value, values: &[String], ov: &Overrides) -> Value {
    let ty = mapping.get("type").and_then(Value::as_str).unwrap_or("keyword");
    let count = values.len();
    let mut m = Map::new();
    m.insert("count".into(), json!(count));
    if matches!(ty, "byte" | "short" | "integer" | "long" | "half_float" | "float" | "double") {
        let mut counts: BTreeMap<OrdF64, usize> = BTreeMap::new();
        for v in values {
            if let Ok(d) = v.parse::<f64>() {
                *counts.entry(OrdF64(d)).or_insert(0) += 1;
            }
        }
        m.insert("cardinality".into(), json!(counts.len()));
        if counts.is_empty() {
            return Value::Object(m);
        }
        m.insert("min_value".into(), num_json(counts.keys().next().unwrap().0));
        m.insert("max_value".into(), num_json(counts.keys().last().unwrap().0));
        // running mean in key order, as ES computes it
        let (mut rc, mut mean) = (0.0f64, f64::NAN);
        for (k, c) in &counts {
            let c = *c as f64;
            let nrc = rc + c;
            if rc > 0.0 {
                mean = mean * (rc / nrc) + k.0 * (c / nrc);
            } else if c > 0.0 {
                mean = k.0;
            }
            rc = nrc;
        }
        m.insert("mean_value".into(), num_json(mean));
        let total = counts.values().sum::<usize>();
        let median = if total % 2 == 1 {
            let target = total / 2 + 1;
            let mut ub = 0;
            counts.iter().find(|(_, c)| {
                ub += **c;
                ub >= target
            }).map(|(k, _)| k.0)
        } else {
            let t1 = total / 2;
            let t2 = t1 + 1;
            let mut t1v = f64::NAN;
            let mut prev: i64 = -1;
            let mut ub = 0usize;
            let mut out = None;
            for (k, c) in &counts {
                ub += c;
                if ub >= t2 {
                    out = Some(if prev < t1 as i64 { k.0 } else { (t1v + k.0) / 2.0 });
                    break;
                }
                if ub >= t1 {
                    t1v = k.0;
                }
                prev = ub as i64;
            }
            out
        };
        if let Some(md) = median {
            m.insert("median_value".into(), num_json(md));
        }
        let mut top: Vec<(&OrdF64, &usize)> = counts.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        m.insert("top_hits".into(), Value::Array(top.into_iter().take(10).map(|(k, c)| json!({"value": num_json(k.0), "count": c})).collect()));
    } else {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for v in values {
            *counts.entry(v.as_str()).or_insert(0) += 1;
        }
        m.insert("cardinality".into(), json!(counts.len()));
        if matches!(ty, "date" | "date_nanos") {
            let fmt = timestamp_mapping(values, ov).ok().flatten().map(|x| x.0);
            let key = |v: &str| fmt.as_ref().and_then(|f| f.millis(v));
            let (mut earliest, mut latest): (Option<(&str, Option<i64>)>, Option<(&str, Option<i64>)>) = (None, None);
            for v in values {
                let k = key(v);
                let lt = |a: &(&str, Option<i64>)| match (k, a.1) {
                    (Some(x), Some(y)) => x < y,
                    _ => v.as_str() < a.0,
                };
                let gt = |a: &(&str, Option<i64>)| match (k, a.1) {
                    (Some(x), Some(y)) => x > y,
                    _ => v.as_str() > a.0,
                };
                if earliest.as_ref().map(|e| lt(e)).unwrap_or(true) {
                    earliest = Some((v, k));
                }
                if latest.as_ref().map(|l| gt(l)).unwrap_or(true) {
                    latest = Some((v, k));
                }
            }
            if let Some(e) = earliest {
                m.insert("earliest".into(), json!(e.0));
            }
            if let Some(l) = latest {
                m.insert("latest".into(), json!(l.0));
            }
        }
        let mut top: Vec<(&&str, &usize)> = counts.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        m.insert("top_hits".into(), Value::Array(top.into_iter().take(10).map(|(k, c)| json!({"value": k, "count": c})).collect()));
    }
    Value::Object(m)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

type Record = Vec<(String, Value)>;

fn flatten_values(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten_values(x, out)),
        Value::Null => {}
        other => out.push(other.clone()),
    }
}

/// ES `guessMappingsAndCalculateFieldStats`.
fn mappings_and_stats(records: &[Record], ov: &Overrides) -> Result<(BTreeMap<String, Value>, BTreeMap<String, Value>)> {
    let mut names: Vec<&String> = records.iter().flat_map(|r| r.iter().map(|(k, _)| k)).collect();
    names.sort();
    names.dedup();
    let mut mappings = BTreeMap::new();
    let mut stats = BTreeMap::new();
    for name in names {
        let vals: Vec<&Value> = records.iter().filter_map(|r| r.iter().find(|(k, _)| k == name).map(|(_, v)| v)).filter(|v| !v.is_null()).collect();
        if vals.is_empty() {
            continue;
        }
        if vals.iter().any(|v| v.is_object()) {
            if vals.iter().all(|v| v.is_object()) {
                mappings.insert(name.clone(), json!({"type": "object"}));
                continue;
            }
            return Err(ia(format!("Field [{name}] has both object and non-object values - this is not supported by Elasticsearch")));
        }
        let mut flat = Vec::new();
        for v in vals {
            flatten_values(v, &mut flat);
        }
        if flat.is_empty() {
            continue;
        }
        let strs: Vec<String> = flat.iter().map(java_string).collect();
        let mapping = guess_scalar(&strs, ov)?;
        stats.insert(name.clone(), field_stats(&mapping, &strs, ov));
        mappings.insert(name.clone(), mapping);
    }
    Ok((mappings, stats))
}

/// ES `guessTimestampField`.
fn guess_timestamp_field(records: &[Record], ov: &Overrides, explanation: &mut Vec<String>) -> Result<Option<(String, TsFormat)>> {
    let Some(first) = records.first() else { return Ok(None) };
    if ov.timestamp_format.as_deref() == Some("null") {
        return Ok(None);
    }
    if let Some(tf) = &ov.timestamp_field {
        if !first.iter().any(|(k, v)| k == tf && !v.is_null()) {
            return Err(ia(format!("Specified timestamp field [{tf}] is not present in record [{}]", record_string(first))));
        }
    }
    let mut candidates = Vec::new();
    for (k, v) in first {
        if ov.timestamp_field.as_ref().map(|t| t != k).unwrap_or(false) || v.is_null() {
            continue;
        }
        if let Some(f) = value_format(&java_string(v), ov)? {
            explanation.push(format!("First sample timestamp match [{}] for field [{k}]", f.java.join(", ")));
            candidates.push((k.clone(), f));
        }
    }
    if candidates.is_empty() {
        if let Some(f) = &ov.timestamp_format {
            return Err(ia(format!("Specified timestamp format [{f}] does not match for record [{}]", record_string(first))));
        }
    }
    'cand: for (k, f) in candidates {
        let mut fmt = f.clone();
        for r in &records[1..] {
            let Some(v) = r.iter().find(|(n, v)| n == &k && !v.is_null()).map(|(_, v)| v) else {
                if ov.timestamp_field.is_some() {
                    return Err(ia(format!("Specified timestamp field [{}] is not present in record [{}]", k, record_string(r))));
                }
                explanation.push(format!("First sample match [{k}] ruled out because record [{}] doesn't have field", record_string(r)));
                continue 'cand;
            };
            match value_format(&java_string(v), ov)? {
                Some(g) if g.grok == fmt.grok => {
                    if g.tz_dependent {
                        fmt.tz_dependent = true;
                    }
                }
                _ => {
                    if let Some(tf) = &ov.timestamp_format {
                        return Err(ia(format!("Specified timestamp format [{tf}] does not match for record [{}] in field [{k}]", record_string(r))));
                    }
                    explanation.push(format!("First sample match [{}] ruled out because record [{}] does not match", fmt.java.join(", "), record_string(r)));
                    continue 'cand;
                }
            }
        }
        explanation.push(format!(
            "{} field is [{k}] with format [{}]",
            if ov.timestamp_field.is_none() { "Guessing timestamp" } else { "Timestamp" },
            fmt.java.join(", ")
        ));
        return Ok(Some((k, fmt)));
    }
    Ok(None)
}

/// Java `Map.toString` of a record.
fn record_string(r: &Record) -> String {
    format!("{{{}}}", r.iter().map(|(k, v)| format!("{k}={}", java_string(v))).collect::<Vec<_>>().join(", "))
}

const CONVERTIBLE: &[&str] = &["integer", "long", "float", "double", "boolean"];

/// ES `makeIngestPipelineDefinition`.
fn ingest_pipeline(
    grok_pattern: Option<&str>,
    custom: &BTreeMap<String, String>,
    csv: Option<&Value>,
    conversions: &BTreeMap<String, Value>,
    ts: Option<&(String, TsFormat)>,
    ecs: Option<&str>,
) -> Option<Value> {
    if grok_pattern.is_none() && csv.is_none() && ts.is_none() {
        return None;
    }
    let mut procs = Vec::new();
    if let Some(g) = grok_pattern {
        let mut m = Map::new();
        m.insert("field".into(), json!("message"));
        m.insert("patterns".into(), json!([g]));
        if !custom.is_empty() {
            m.insert("pattern_definitions".into(), json!(custom));
        }
        m.insert("ecs_compatibility".into(), json!(ecs.filter(|e| !e.is_empty()).unwrap_or("disabled")));
        procs.push(json!({ "grok": m }));
    }
    if let Some(c) = csv {
        procs.push(json!({ "csv": c }));
    }
    if let Some((field, f)) = ts {
        let mut m = Map::new();
        m.insert("field".into(), json!(field));
        if f.tz_dependent {
            m.insert("timezone".into(), json!("{{ event.timezone }}"));
        }
        m.insert("formats".into(), json!(f.java));
        if f.nanos {
            m.insert("output_format".into(), json!("yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSSXXX"));
        }
        procs.push(json!({ "date": m }));
    }
    for (name, mapping) in conversions {
        if let Some(t) = mapping.get("type").and_then(Value::as_str) {
            if CONVERTIBLE.contains(&t) {
                procs.push(json!({"convert": {"field": name, "type": t, "ignore_missing": true}}));
            }
        }
    }
    if let Some(c) = csv {
        let field = c["field"].clone();
        let in_targets = c["target_fields"].as_array().map(|a| a.contains(&field)).unwrap_or(false);
        if !in_targets {
            procs.push(json!({"remove": {"field": field}}));
        }
    }
    if let (Some(_), Some((field, _))) = (grok_pattern, ts) {
        procs.push(json!({"remove": {"field": field}}));
    }
    Some(json!({"description": "Ingest pipeline created by text structure finder", "processors": procs}))
}

// ------------------------------------------------------------ structure assembly

#[derive(Default)]
struct Structure {
    num_lines: usize,
    num_messages: usize,
    sample_start: String,
    charset: String,
    bom: Option<bool>,
    format: String,
    multiline: Option<String>,
    exclude_lines: Option<String>,
    column_names: Option<Vec<String>>,
    has_header_row: Option<bool>,
    delimiter: Option<char>,
    quote: Option<char>,
    should_trim: Option<bool>,
    grok_pattern: Option<String>,
    ecs: String,
    ts: Option<(String, TsFormat)>,
    mappings: BTreeMap<String, Value>,
    pipeline: Option<Value>,
    stats: BTreeMap<String, Value>,
}

impl Structure {
    fn to_json(&self, explanation: &[String], explain: bool) -> Value {
        let mut m = Map::new();
        m.insert("num_lines_analyzed".into(), json!(self.num_lines));
        m.insert("num_messages_analyzed".into(), json!(self.num_messages));
        m.insert("sample_start".into(), json!(self.sample_start));
        m.insert("charset".into(), json!(self.charset));
        if let Some(b) = self.bom {
            m.insert("has_byte_order_marker".into(), json!(b));
        }
        m.insert("format".into(), json!(self.format));
        if let Some(p) = self.multiline.as_ref().filter(|p| !p.is_empty()) {
            m.insert("multiline_start_pattern".into(), json!(p));
        }
        if let Some(p) = self.exclude_lines.as_ref().filter(|p| !p.is_empty()) {
            m.insert("exclude_lines_pattern".into(), json!(p));
        }
        if let Some(c) = self.column_names.as_ref().filter(|c| !c.is_empty()) {
            m.insert("column_names".into(), json!(c));
        }
        if let Some(h) = self.has_header_row {
            m.insert("has_header_row".into(), json!(h));
        }
        if let Some(d) = self.delimiter {
            m.insert("delimiter".into(), json!(d.to_string()));
        }
        if let Some(q) = self.quote {
            m.insert("quote".into(), json!(q.to_string()));
        }
        if let Some(t) = self.should_trim {
            m.insert("should_trim_fields".into(), json!(t));
        }
        if let Some(g) = self.grok_pattern.as_ref().filter(|g| !g.is_empty()) {
            m.insert("grok_pattern".into(), json!(g));
        }
        m.insert("ecs_compatibility".into(), json!(self.ecs));
        let mut need_tz = false;
        if let Some((f, fmt)) = &self.ts {
            m.insert("timestamp_field".into(), json!(f));
            m.insert("joda_timestamp_formats".into(), json!(fmt.joda()));
            m.insert("java_timestamp_formats".into(), json!(fmt.java));
            need_tz = fmt.tz_dependent;
        }
        m.insert("need_client_timezone".into(), json!(need_tz));
        m.insert("mappings".into(), json!({ "properties": self.mappings }));
        if let Some(p) = &self.pipeline {
            m.insert("ingest_pipeline".into(), p.clone());
        }
        if !self.stats.is_empty() {
            m.insert("field_stats".into(), json!(self.stats));
        }
        if explain {
            m.insert("explanation".into(), json!(explanation));
        }
        Value::Object(m)
    }
}

// ------------------------------------------------------------ NDJSON

fn parse_ndjson_line(line: &str) -> Option<Record> {
    let v: Value = serde_json::from_str(line).ok()?;
    match v {
        Value::Object(o) if !o.is_empty() => Some(o.into_iter().collect()),
        _ => None,
    }
}

fn can_ndjson(lines: &[&str], explanation: &mut Vec<String>) -> bool {
    for l in lines {
        match serde_json::from_str::<Value>(l) {
            Ok(Value::Object(o)) if !o.is_empty() => {}
            Ok(Value::Object(_)) => {
                explanation.push(format!("Not NDJSON because an empty object was parsed: [{l}]"));
                return false;
            }
            Ok(_) | Err(_) => {
                explanation.push("Not NDJSON because there was a parsing exception".into());
                return false;
            }
        }
    }
    !lines.is_empty()
}

fn ndjson(lines: &[&str], st: &mut Structure, ov: &Overrides, explanation: &mut Vec<String>) -> Result<()> {
    let records: Vec<Record> = lines.iter().filter_map(|l| parse_ndjson_line(l)).collect();
    st.format = "ndjson".into();
    st.sample_start = lines.iter().take(2).map(|l| format!("{l}\n")).collect();
    st.num_lines = lines.len();
    st.num_messages = records.len();
    let ts = guess_timestamp_field(&records, ov, explanation)?;
    if let Some(t) = &ts {
        st.pipeline = ingest_pipeline(None, &BTreeMap::new(), None, &BTreeMap::new(), Some(t), ov.ecs_compatibility.as_deref());
    }
    let (mut mappings, stats) = mappings_and_stats(&records, ov)?;
    if let Some((_, f)) = &ts {
        mappings.insert("@timestamp".into(), f.mapping_without_format());
    }
    st.ts = ts;
    st.mappings = mappings;
    st.stats = stats;
    Ok(())
}

// ------------------------------------------------------------ delimited

/// RFC-4180-ish row reader (SuperCSV semantics: quoted fields may span lines).
/// Returns rows and the (1-based, exclusive) line number each row ends on.
fn read_rows(sample: &str, delim: char, quote: char) -> (Vec<Vec<Option<String>>>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut line_nos = Vec::new();
    let chars: Vec<char> = sample.chars().collect();
    let mut i = 0;
    let mut line = 0usize;
    while i < chars.len() {
        let mut row: Vec<Option<String>> = Vec::new();
        let mut field = String::new();
        let mut quoted = false;
        let mut was_quoted = false;
        loop {
            if i >= chars.len() {
                row.push(if field.is_empty() && !was_quoted { None } else { Some(std::mem::take(&mut field)) });
                line += 1;
                break;
            }
            let c = chars[i];
            if quoted {
                if c == quote {
                    if i + 1 < chars.len() && chars[i + 1] == quote {
                        field.push(quote);
                        i += 2;
                        continue;
                    }
                    quoted = false;
                    i += 1;
                    continue;
                }
                if c == '\n' {
                    line += 1;
                }
                field.push(c);
                i += 1;
                continue;
            }
            if c == quote && field.is_empty() {
                quoted = true;
                was_quoted = true;
                i += 1;
                continue;
            }
            if c == delim {
                row.push(if field.is_empty() && !was_quoted { None } else { Some(std::mem::take(&mut field)) });
                was_quoted = false;
                i += 1;
                continue;
            }
            if c == '\n' {
                row.push(if field.is_empty() && !was_quoted { None } else { Some(std::mem::take(&mut field)) });
                i += 1;
                line += 1;
                break;
            }
            if c == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                i += 1;
                continue;
            }
            field.push(c);
            i += 1;
        }
        // SuperCSV skips empty lines.
        if row.len() == 1 && row[0].is_none() {
            continue;
        }
        rows.push(row);
        line_nos.push(line);
    }
    (rows, line_nos)
}

fn can_delimited(sample: &str, delim: char, quote: char, min_fields: usize, allowed_bad: f64, explanation: &mut Vec<String>) -> bool {
    let (mut rows, _) = read_rows(sample, delim, quote);
    if rows.len() > 1 && rows[0].len() != rows[rows.len() - 1].len() {
        rows.pop();
    }
    if rows.len() < 2 {
        explanation.push(format!("Not {} because fewer than 2 complete records in sample", delim_name(delim)));
        return false;
    }
    let n = rows[0].len();
    if n < min_fields {
        explanation.push(format!("Not {} because the first row has fewer than [{min_fields}] fields: [{n}]", delim_name(delim)));
        return false;
    }
    let bad = rows.iter().filter(|r| r.len() != n).count();
    if bad as f64 > allowed_bad * rows.len() as f64 {
        explanation.push(format!("Not {} because rows have different numbers of fields", delim_name(delim)));
        return false;
    }
    true
}

fn delim_name(d: char) -> &'static str {
    match d {
        ',' => "CSV",
        '\t' => "TSV",
        ';' => "semicolon delimited values",
        '|' => "pipe delimited values",
        _ => "delimited values",
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 });
        }
        prev = cur;
    }
    prev[b.len()]
}

fn row_str(r: &[Option<String>]) -> String {
    r.iter().map(|f| f.clone().unwrap_or_default()).collect()
}

/// ES `isFirstRowUnusual` (length test + field-wise Levenshtein test on short fields).
fn first_row_unusual(rows: &[Vec<Option<String>>], explanation: &mut Vec<String>) -> bool {
    let first = row_str(&rows[0]).chars().count() as f64;
    let others: Vec<f64> = rows[1..].iter().map(|r| row_str(r).chars().count() as f64).collect();
    let (min, max) = others.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
    let range = max - min;
    if first < min - range / 10.0 || first > max + range / 10.0 {
        explanation.push("First row is unusual based on length test".into());
        return true;
    }
    let ncols = rows[0].len();
    let short: Vec<bool> = (0..ncols).map(|c| rows.iter().all(|r| r.get(c).and_then(|x| x.as_ref()).map(|s| s.chars().count() < 100).unwrap_or(true))).collect();
    let cmp = |a: &[Option<String>], b: &[Option<String>]| -> f64 {
        (0..ncols).filter(|c| short[*c]).map(|c| levenshtein(a.get(c).and_then(|x| x.as_deref()).unwrap_or(""), b.get(c).and_then(|x| x.as_deref()).unwrap_or(""))).sum::<usize>() as f64
    };
    let first_avg = rows[1..].iter().take(10).map(|r| cmp(&rows[0], r)).sum::<f64>() / rows[1..].len().min(10) as f64;
    let mut other_sum = 0.0;
    let mut n = 0;
    'o: for i in 1..rows.len() {
        for j in i + 1..rows.len() {
            other_sum += cmp(&rows[i], &rows[j]);
            n += 1;
            if n >= 10 {
                break 'o;
            }
        }
    }
    if n == 0 {
        return false;
    }
    let other_avg = other_sum / n as f64;
    let unusual = first_avg > other_avg * 1.2;
    explanation.push(format!("First row is {}unusual based on Levenshtein test", if unusual { "" } else { "not " }));
    unusual
}

fn regex_escape_char(c: char) -> String {
    if "\\|()[]{}^$*?+.".contains(c) { format!("\\{c}") } else { c.to_string() }
}

#[allow(clippy::too_many_arguments)]
fn delimited(sample: &str, delim: char, quote: char, trim: bool, st: &mut Structure, ov: &Overrides, explanation: &mut Vec<String>) -> Result<()> {
    let (mut rows, mut line_nos) = read_rows(sample, delim, quote);
    if rows.len() > 1 && rows[0].len() != rows[rows.len() - 1].len() {
        rows.pop();
        line_nos.pop();
    }
    if rows.len() < 2 {
        return Err(ia("Input did not match the specified format [delimited]"));
    }
    let first: Vec<String> = rows[0].iter().map(|f| f.clone().unwrap_or_default()).collect();
    let dup = {
        let mut seen = std::collections::HashSet::new();
        first.iter().filter(|v| !v.is_empty()).find(|v| !seen.insert(v.as_str())).cloned()
    };
    let has_header = match ov.has_header_row {
        Some(h) => {
            if h && ov.column_names.is_none() {
                if let Some(d) = &dup {
                    return Err(ia(format!("Sample specified to contain a header row, but the first row contains duplicate values: [{d}]")));
                }
            }
            explanation.push(format!("Sample specified to {} a header row", if h { "contain" } else { "not contain" }));
            h
        }
        None => {
            if dup.is_some() {
                explanation.push("First row contains duplicate values, so assuming it's not a header".into());
                false
            } else if rows.len() < 3 {
                explanation.push("Too little data to accurately assess whether header is in sample - guessing it is".into());
                true
            } else {
                first_row_unusual(&rows, explanation)
            }
        }
    };
    let header: Vec<String> = if has_header { first.clone() } else { vec![String::new(); first.len()] };
    let columns: Vec<String> = match &ov.column_names {
        Some(c) => {
            if c.len() != header.len() {
                return Err(ia(format!("[{}] column names were specified [{}] but there are [{}] columns in the sample", c.len(), c.join(","), header.len())));
            }
            c.clone()
        }
        None => header.iter().enumerate().map(|(i, h)| {
            let h = if trim { h.trim() } else { h.as_str() };
            if h.is_empty() { format!("column{}", i + 1) } else { h.replace('.', "_") }
        }).collect(),
    };
    let mut records: Vec<Record> = Vec::new();
    for r in rows.iter().skip(if has_header { 1 } else { 0 }) {
        if r.len() != columns.len() {
            continue;
        }
        let rec: Record = columns
            .iter()
            .zip(r)
            .filter_map(|(c, v)| v.as_ref().map(|v| (c.clone(), Value::String(if trim { v.trim().to_string() } else { v.clone() }))))
            .collect();
        records.push(rec);
    }
    let lines: Vec<&str> = sample.split('\n').collect();
    let preamble_end = line_nos.get(1).copied().unwrap_or(1).min(lines.len());
    st.sample_start = lines[..preamble_end].join("\n") + "\n";
    st.format = "delimited".into();
    st.num_lines = *line_nos.last().unwrap_or(&0);
    st.num_messages = records.len();
    st.has_header_row = Some(has_header);
    st.delimiter = Some(delim);
    st.quote = Some(quote);
    st.column_names = Some(columns.clone());
    let qp = regex_escape_char(quote);
    let opt_q = format!("{qp}?");
    let dp = if delim == '\t' { "\\t".to_string() } else { regex_escape_char(delim) };
    if has_header {
        let mut p = String::from("^");
        for (i, h) in header.iter().enumerate() {
            let col = format!("{opt_q}{}{opt_q}", h.replace(quote, &format!("{quote}{quote}")).chars().map(regex_escape_char).collect::<String>());
            if i == 0 {
                p.push_str(&col);
            } else {
                if p.len() + col.len() > 1000 - dp.len() - 2 {
                    p.push_str(".*");
                    break;
                }
                p.push_str(&dp);
                p.push_str(&col);
            }
        }
        st.exclude_lines = Some(p);
    }
    if trim {
        st.should_trim = Some(true);
    }
    let (mut mappings, stats) = mappings_and_stats(&records, ov)?;
    let mut csv = Map::new();
    csv.insert("field".into(), json!("message"));
    csv.insert("target_fields".into(), json!(columns));
    if delim != ',' {
        csv.insert("separator".into(), json!(delim.to_string()));
    }
    if quote != '"' {
        csv.insert("quote".into(), json!(quote.to_string()));
    }
    csv.insert("ignore_missing".into(), json!(false));
    if trim {
        csv.insert("trim".into(), json!(true));
    }
    let csv = Value::Object(csv);
    let ts = guess_timestamp_field(&records, ov, explanation)?;
    st.pipeline = ingest_pipeline(None, &BTreeMap::new(), Some(&csv), &mappings, ts.as_ref(), ov.ecs_compatibility.as_deref());
    // Multi-line start pattern: the timestamp column preceded by its columns.
    if let Some((field, f)) = &ts {
        let idx = columns.iter().position(|c| c == field).unwrap_or(0);
        let mut p = String::from("^");
        for _ in 0..idx {
            p.push_str(&format!("(?:{qp}(?:[^{qp}]|{qp}{qp})*{qp}|[^{qp}{dp}\\n]*){dp}"));
        }
        p.push_str(&opt_q);
        p.push_str(f.simple.trim_start_matches("\\b"));
        st.multiline = Some(p);
        mappings.insert("@timestamp".into(), f.mapping_without_format());
    }
    st.ts = ts;
    st.mappings = mappings;
    st.stats = stats;
    Ok(())
}

// ------------------------------------------------------------ Grok pattern creator

/// Punctuation/space characters ES treats literally (value: needs escaping).
fn punct(c: char) -> Option<bool> {
    if "\"'`‘’“”#@%=\\/|~:;,<>()[]{}«»^$*¿?¡!§¶ \t\n".contains(c) {
        Some("\\|()[]{}^$*?".contains(c))
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Breaks {
    Word,
    None,
    Path,
    Num,
    Hex,
}

struct ValueCand {
    grok: &'static str,
    ty: &'static str,
    field: &'static str,
    breaks: Breaks,
}

const VALUE_CANDS: &[ValueCand] = &[
    ValueCand { grok: "TOMCAT_DATESTAMP", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "TIMESTAMP_ISO8601", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "DATESTAMP_RFC822", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "DATESTAMP_RFC2822", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "DATESTAMP_OTHER", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "DATESTAMP_EVENTLOG", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "HTTPDERROR_DATE", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "SYSLOGTIMESTAMP", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "HTTPDATE", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "CATALINA_DATESTAMP", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "CISCOTIMESTAMP", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "DATESTAMP", ty: "date", field: "extra_timestamp", breaks: Breaks::Word },
    ValueCand { grok: "LOGLEVEL", ty: "keyword", field: "loglevel", breaks: Breaks::Word },
    ValueCand { grok: "URI", ty: "keyword", field: "uri", breaks: Breaks::Word },
    ValueCand { grok: "UUID", ty: "keyword", field: "uuid", breaks: Breaks::Word },
    ValueCand { grok: "MAC", ty: "keyword", field: "macaddress", breaks: Breaks::Word },
    ValueCand { grok: "PATH", ty: "keyword", field: "path", breaks: Breaks::Path },
    ValueCand { grok: "EMAILADDRESS", ty: "keyword", field: "email", breaks: Breaks::Word },
    ValueCand { grok: "IP", ty: "ip", field: "ipaddress", breaks: Breaks::Word },
    ValueCand { grok: "DATE", ty: "date", field: "date", breaks: Breaks::Word },
    ValueCand { grok: "TIME", ty: "keyword", field: "time", breaks: Breaks::Word },
    ValueCand { grok: "QUOTEDSTRING", ty: "keyword", field: "field", breaks: Breaks::None },
    ValueCand { grok: "INT", ty: "long", field: "field", breaks: Breaks::Num },
    ValueCand { grok: "NUMBER", ty: "double", field: "field", breaks: Breaks::Num },
    ValueCand { grok: "BASE16NUM", ty: "keyword", field: "field", breaks: Breaks::Hex },
];

fn value_regex(grok_name: &str, breaks: Breaks, defs: &BTreeMap<String, String>) -> Option<Regex> {
    let g = Grok::compile(&format!("%{{{grok_name}}}"), defs).ok()?;
    let src = g.re.as_str().to_string();
    let src = if breaks == Breaks::Word { format!(r"\b(?:{src})\b") } else { src };
    Regex::new(&src).ok()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// First match (line-wise, as `(?m)%{DATA}<pre>%{X}<post>%{GREEDYDATA}` does)
/// → (preface, value, epilogue).
fn split_on(re: &Regex, breaks: Breaks, snippet: &str) -> Option<(String, String, String)> {
    for line in snippet.split('\n') {
        let mut pos = 0;
        while pos <= line.len() {
            let Some(m) = re.find_at(line, pos) else { break };
            let before = line[..m.start()].chars().last();
            let after: Vec<char> = line[m.end()..].chars().take(2).collect();
            let ok = match breaks {
                Breaks::Word | Breaks::None => true,
                Breaks::Path => before.map(|c| !is_word(c)).unwrap_or(true) && after.first().map(|c| !is_word(*c)).unwrap_or(true),
                Breaks::Num | Breaks::Hex => {
                    let pre_ok = before.map(|c| !(is_word(c) || ".+-".contains(c))).unwrap_or(true);
                    let post_ok = match after.first() {
                        None => true,
                        Some(c) if is_word(*c) || *c == '+' || *c == '-' => false,
                        Some('.') => match after.get(1) {
                            Some(d) if breaks == Breaks::Num => !d.is_ascii_digit(),
                            Some(d) => !is_word(*d),
                            None => true,
                        },
                        _ => true,
                    };
                    pre_ok && post_ok
                }
            };
            if ok && (m.end() > m.start()) {
                return Some((line[..m.start()].to_string(), m.as_str().to_string(), line[m.end()..].to_string()));
            }
            // next char boundary after the match start
            let next = line[m.start()..].chars().next().map(|c| m.start() + c.len_utf8()).unwrap_or(line.len() + 1);
            pos = next;
        }
    }
    None
}

struct GrokCreator<'a> {
    ov: &'a Overrides,
    defs: BTreeMap<String, String>,
    out: String,
    counts: HashMap<String, usize>,
    mappings: &'a mut BTreeMap<String, Value>,
    stats: &'a mut BTreeMap<String, Value>,
    regexes: Vec<Option<Regex>>,
}

impl<'a> GrokCreator<'a> {
    fn new(ov: &'a Overrides, custom: &BTreeMap<String, String>, mappings: &'a mut BTreeMap<String, Value>, stats: &'a mut BTreeMap<String, Value>) -> Self {
        let mut defs = grok::bank().clone();
        defs.extend(custom.iter().map(|(k, v)| (k.clone(), v.clone())));
        let regexes = VALUE_CANDS.iter().map(|c| value_regex(c.grok, c.breaks, &defs)).collect();
        GrokCreator { ov, defs, out: String::new(), counts: HashMap::new(), mappings, stats, regexes }
    }

    fn field_name(&mut self, f: &str) -> String {
        let n = self.counts.entry(f.to_string()).or_insert(0);
        *n += 1;
        if *n > 1 { format!("{f}{n}") } else { f.to_string() }
    }

    fn adjust_for_punctuation(&mut self, snippets: Vec<String>) -> Vec<String> {
        let mut common: Vec<char> = Vec::new();
        let mut started = false;
        for s in &snippets {
            let sc: Vec<char> = s.chars().collect();
            if !started || common.is_empty() {
                common.clear();
                for c in &sc {
                    if punct(*c).is_some() {
                        common.push(*c);
                    } else {
                        break;
                    }
                }
                started = true;
            } else {
                if common.len() > sc.len() {
                    common.truncate(sc.len());
                }
                for i in 0..common.len() {
                    if sc[i] != common[i] {
                        common.truncate(i);
                        break;
                    }
                }
            }
            if common.len() <= 1 {
                return snippets;
            }
        }
        let n = common.len() - 1;
        for c in &common[..n] {
            if punct(*c) == Some(true) {
                self.out.push('\\');
            }
            self.out.push(*c);
        }
        snippets.into_iter().map(|s| s.chars().skip(n).collect()).collect()
    }

    fn append_best(&mut self, is_last: bool, snippets: Vec<String>, ignore_value_only: usize) {
        if snippets.is_empty() {
            if is_last {
                self.finalize(&snippets);
            } else {
                add_intermediate(&mut self.out, &snippets);
            }
            return;
        }
        let snippets = self.adjust_for_punctuation(snippets);
        let mut skipped = ignore_value_only;
        let mut best: Option<usize> = None;
        for i in ignore_value_only..VALUE_CANDS.len() {
            if let Some(re) = &self.regexes[i] {
                if snippets.iter().all(|s| split_on(re, VALUE_CANDS[i].breaks, s).is_some()) {
                    best = Some(i);
                    break;
                }
            }
            skipped += 1;
        }
        match best {
            None => {
                if is_last {
                    self.finalize(&snippets);
                } else {
                    add_intermediate(&mut self.out, &snippets);
                }
            }
            Some(i) => {
                let c = &VALUE_CANDS[i];
                // ES passes ignoreValueOnlyCandidates + 1 (key-value candidate ignored) to the left.
                self.split(c.grok, Some(c.ty), c.field, c.breaks, i, is_last, snippets, skipped + 1, skipped);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn split(&mut self, grok_name: &str, ty: Option<&str>, field: &str, breaks: Breaks, idx: usize, is_last: bool, snippets: Vec<String>, ignore_left: usize, ignore_right: usize) {
        let re = if idx < self.regexes.len() && ty.is_some() { self.regexes[idx].clone() } else { value_regex(grok_name, breaks, &self.defs) };
        let Some(re) = re else { return };
        let mut pre = Vec::new();
        let mut vals = Vec::new();
        let mut post = Vec::new();
        for s in &snippets {
            let (a, b, c) = split_on(&re, breaks, s).unwrap_or_default();
            pre.push(a);
            vals.push(b);
            post.push(c);
        }
        let name = self.field_name(field);
        match ty {
            Some(t) => {
                let mapping = if t == "date" {
                    timestamp_mapping(&vals, &Overrides::default()).ok().flatten().map(|x| x.1).unwrap_or(json!({"type": "date"}))
                } else {
                    json!({ "type": t })
                };
                self.stats.insert(name.clone(), field_stats(&mapping, &vals, self.ov));
                self.mappings.insert(name.clone(), mapping);
            }
            None => {
                // seed (timestamp): stats only
                let mapping = timestamp_mapping(&vals, self.ov).ok().flatten().map(|x| x.1).unwrap_or(json!({"type": "date"}));
                self.stats.insert(name.clone(), field_stats(&mapping, &vals, self.ov));
            }
        }
        self.append_best(false, pre, ignore_left);
        self.out.push_str(&format!("%{{{grok_name}:{name}}}"));
        self.append_best(is_last, post, ignore_right);
    }

    fn finalize(&mut self, snippets: &[String]) {
        if snippets.iter().all(String::is_empty) {
            return;
        }
        let mut others: Vec<Vec<char>> = snippets.iter().map(|s| s.chars().collect()).collect();
        let driver = others.pop().unwrap();
        for (i, ch) in driver.iter().enumerate() {
            match punct(*ch) {
                Some(esc) if others.iter().all(|o| o.len() > i && o[i] == *ch) => {
                    if esc {
                        self.out.push('\\');
                    }
                    self.out.push(*ch);
                    if i == driver.len() - 1 && others.iter().all(|o| *o == driver) {
                        return;
                    }
                }
                _ => break,
            }
        }
        self.out.push_str(".*");
    }
}

/// ES `GrokPatternCreator.addIntermediateRegex`; returns the longest run of
/// identical characters preceding wildcards.
fn add_intermediate(out: &mut String, snippets: &[String]) -> usize {
    if snippets.is_empty() {
        return 0;
    }
    let mut others: Vec<Vec<char>> = snippets.iter().map(|s| s.chars().collect()).collect();
    let driver = others.pop().unwrap();
    let mut last = '\0';
    let mut preceding: Vec<char> = Vec::new();
    let mut wildcard_required = true;
    for ch in &driver {
        let p = punct(*ch);
        if p.is_some() && others.iter().all(|o| o.contains(ch)) {
            if wildcard_required && others.iter().any(|o| o.iter().position(|c| c == ch).map(|x| x > 0).unwrap_or(false)) {
                out.push_str(".*?");
                preceding.push(last);
            }
            if p == Some(true) {
                out.push('\\');
            }
            out.push(*ch);
            wildcard_required = true;
            others = others.into_iter().map(|o| { let i = o.iter().position(|c| c == ch).unwrap(); o[i + 1..].to_vec() }).collect();
            last = *ch;
        } else if wildcard_required {
            out.push_str(".*?");
            preceding.push(last);
            wildcard_required = false;
        }
    }
    if wildcard_required && others.iter().any(|o| !o.is_empty()) {
        out.push_str(".*?");
    }
    let mut longest = 0;
    let mut run = 0;
    for (i, c) in preceding.iter().enumerate() {
        if i > 0 && preceding[i - 1] == *c {
            run += 1;
        } else {
            run = 1;
        }
        longest = longest.max(run);
    }
    longest
}

/// Full-message Grok patterns tried before building one from examples
/// (`(pattern, time field)`).
const FULL_MATCH: &[(&str, &str)] = &[
    ("CATALINALOG", "timestamp"),
    ("COMBINEDAPACHELOG", "timestamp"),
    ("COMMONAPACHELOG", "timestamp"),
    ("HTTPD20_ERRORLOG", "timestamp"),
    ("HTTPD24_ERRORLOG", "timestamp"),
    ("RAILS3", "timestamp"),
    ("RUBY_LOGGER", "timestamp"),
    ("TOMCATLOG", "timestamp"),
];

/// Captured values per field of `pattern` over every message (`None`: some message does not match).
fn grok_values(pattern: &str, defs: &BTreeMap<String, String>, messages: &[String]) -> Result<Option<BTreeMap<String, Vec<String>>>> {
    let g = Grok::compile(pattern, defs)?;
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in messages {
        let Some(c) = g.captures(m) else { return Ok(None) };
        for (k, v) in c {
            out.entry(k).or_default().push(crate::ingest::doc::value_to_string(&v));
        }
    }
    Ok(Some(out))
}

// ------------------------------------------------------------ semi-structured text

fn search_format(line: &str, ov: &Overrides) -> Result<Option<(String, TsFormat, usize)>> {
    if let Some(f) = &ov.timestamp_format {
        let (_, search, tf) = override_format(f).map_err(ia)?;
        return Ok(search.find(line).map(|m| (line[..m.start()].to_string(), tf, 0)));
    }
    let mut best: Option<(usize, String, TsFormat, usize)> = None;
    for (i, (c, cc)) in CANDIDATES.iter().zip(compiled()).enumerate() {
        if let Some(m) = cc.search.find(line) {
            // earliest match wins; ties go to the earlier candidate
            if best.as_ref().map(|b| m.start() < b.0).unwrap_or(true) {
                best = Some((m.start(), line[..m.start()].to_string(), make_format(c, m.as_str()), i));
            }
        }
    }
    Ok(best.map(|b| (b.1, b.2, b.3)))
}

fn log_text(lines: Option<&[&str]>, messages_in: Option<&[String]>, st: &mut Structure, ov: &Overrides, explanation: &mut Vec<String>) -> Result<()> {
    st.format = "semi_structured_text".into();
    let single = ov.timestamp_format.as_deref() == Some("null");
    let source: Vec<String> = match (lines, messages_in) {
        (Some(l), _) => l.iter().map(|s| s.to_string()).collect(),
        (None, Some(m)) => m.to_vec(),
        _ => Vec::new(),
    };
    let mut mappings = BTreeMap::new();
    let mut stats = BTreeMap::new();
    mappings.insert("message".to_string(), json!({"type": "text"}));
    if single {
        explanation.push("Timestamp format is explicitly set to \"null\"".into());
        st.sample_start = source.iter().take(2).map(|l| format!("{l}\n")).collect();
        st.num_lines = source.len();
        st.num_messages = source.len();
        stats.insert("message".to_string(), field_stats(&json!({"type": "text"}), &source, ov));
        let custom = BTreeMap::new();
        let grok_pattern = match &ov.grok_pattern {
            Some(g) => {
                let vals = grok_values(g, grok::bank(), &source)
                    .map_err(|_| ia(format!("Supplied Grok pattern [{g}] cannot be converted to a valid regex")))?
                    .ok_or_else(|| ia(format!("Supplied Grok pattern [{g}] does not match sample messages")))?;
                for (k, v) in vals {
                    let m = guess_scalar(&v, ov)?;
                    stats.insert(k.clone(), field_stats(&m, &v, ov));
                    mappings.insert(k, m);
                }
                g.clone()
            }
            None => {
                let mut gc = GrokCreator::new(ov, &custom, &mut mappings, &mut stats);
                gc.append_best(true, source.clone(), 0);
                gc.out.replace('\t', "\\t").replace('\n', "\\n")
            }
        };
        st.pipeline = ingest_pipeline(Some(&grok_pattern), &custom, None, &mappings, None, ov.ecs_compatibility.as_deref());
        st.grok_pattern = Some(grok_pattern);
        st.mappings = mappings;
        st.stats = stats;
        return Ok(());
    }
    // Timestamp format: the most common among lines (ties → earlier candidate).
    let mut counts: Vec<(TsFormat, usize, usize)> = Vec::new();
    let mut found: Vec<Option<(String, TsFormat)>> = Vec::new();
    for l in &source {
        let f = search_format(l, ov)?;
        if let Some((_, tf, ci)) = &f {
            match counts.iter_mut().find(|c| c.0.grok == tf.grok && c.0.java == tf.java) {
                Some(c) => c.1 += 1,
                None => counts.push((tf.clone(), 1, *ci)),
            }
        }
        found.push(f.map(|x| (x.0, x.1)));
    }
    let Some(best) = counts.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2))).map(|c| c.0.clone()) else {
        return Err(ia(format!("Could not find {} in the sample provided", if ov.timestamp_format.is_none() { "a timestamp" } else { "the specified timestamp format" })));
    };
    explanation.push(format!("{} format is [{}]", if ov.timestamp_format.is_none() { "Most likely timestamp" } else { "Timestamp" }, best.java.join(", ")));
    let mut tz_dep = false;
    let prefaces: Vec<String> = found
        .iter()
        .flatten()
        .filter(|(_, f)| f.grok == best.grok && f.java == best.java)
        .map(|(p, f)| {
            tz_dep |= f.tz_dependent;
            p.clone()
        })
        .collect();
    let mut best = best;
    best.tz_dependent = tz_dep;
    let mut ml = String::from("^");
    let complexity = add_intermediate(&mut ml, &prefaces);
    ml.push_str(&best.simple);
    if ml.starts_with("^\\b") {
        ml.replace_range(1..3, "");
    }
    if complexity >= 8 {
        return Err(ia(format!("Generated multi-line start pattern based on timestamp position [{ml}] is too complex. If your sample is delimited then try overriding the format.")));
    }
    let messages: Vec<String> = match lines {
        Some(lines) => {
            let re = Regex::new(&ml).map_err(|e| ia(e.to_string()))?;
            let mut msgs = Vec::new();
            let mut consumed = 0usize;
            let mut cur: Option<(String, usize)> = None;
            for l in lines {
                if re.is_match(l) {
                    if let Some((m, n)) = cur.take() {
                        msgs.push(m);
                        consumed += n;
                    }
                    cur = Some((l.to_string(), 1));
                } else {
                    match cur.as_mut() {
                        None => consumed += 1,
                        Some((m, n)) => {
                            let after = m.chars().count() + 1 + l.chars().count();
                            if after > ov.line_merge_size_limit {
                                return Err(ia(format!(
                                    "Merging lines into messages resulted in an unacceptably long message. Merged message would have [{}] lines and [{after}] characters (limit [{}]). If you have messages this big please increase the value of [line_merge_size_limit]. Otherwise it probably means the timestamp has been incorrectly detected, so try overriding that.",
                                    *n + 1,
                                    ov.line_merge_size_limit
                                )));
                            }
                            m.push('\n');
                            m.push_str(l);
                            *n += 1;
                        }
                    }
                }
            }
            if msgs.is_empty() {
                return Err(ia("Failed to create more than one message from the sample lines provided. (The last is discarded in case the sample is incomplete.) If your sample does contain multiple messages the problem is probably that the primary timestamp format has been incorrectly detected, so try overriding it."));
            }
            st.num_lines = consumed;
            st.multiline = Some(ml.clone());
            msgs
        }
        None => {
            st.num_lines = source.len();
            source.clone()
        }
    };
    st.num_messages = messages.len();
    st.sample_start = messages.iter().take(2).map(|m| format!("{m}\n")).collect();
    mappings.insert("@timestamp".to_string(), best.mapping_without_format());
    stats.insert("message".to_string(), field_stats(&json!({"type": "text"}), &messages, ov));
    let mut custom: BTreeMap<String, String> = BTreeMap::new();
    if let Some(c) = &best.custom {
        custom.insert("CUSTOM_TIMESTAMP".into(), c.clone());
    }
    let mut defs = grok::bank().clone();
    defs.extend(custom.clone());
    let mut ts_field = ov.timestamp_field.clone();
    let grok_pattern = match &ov.grok_pattern {
        Some(g) => {
            let tf = ts_field.get_or_insert_with(|| "timestamp".into()).clone();
            let vals = grok_values(g, &defs, &messages)
                .map_err(|_| ia(format!("Supplied Grok pattern [{g}] cannot be converted to a valid regex")))?
                .ok_or_else(|| ia(format!("Supplied Grok pattern [{g}] does not match sample messages")))?;
            for (k, v) in vals {
                let m = guess_scalar(&v, ov)?;
                stats.insert(k.clone(), field_stats(&m, &v, ov));
                if k != tf {
                    mappings.insert(k, m);
                }
            }
            g.clone()
        }
        None => {
            let mut full = None;
            for (name, tf) in FULL_MATCH {
                if ts_field.as_ref().map(|t| t == tf).unwrap_or(true) && defs.contains_key(*name) {
                    if let Some(vals) = grok_values(&format!("%{{{name}}}"), &defs, &messages)? {
                        explanation.push(format!("A full message Grok pattern [{name}] looks appropriate"));
                        for (k, v) in vals {
                            let m = guess_scalar(&v, ov)?;
                            stats.insert(k.clone(), field_stats(&m, &v, ov));
                            if k != *tf {
                                mappings.insert(k, m);
                            }
                        }
                        full = Some((format!("%{{{name}}}"), tf.to_string()));
                        break;
                    }
                }
            }
            match full {
                Some((p, tf)) => {
                    ts_field = Some(tf);
                    p
                }
                None => {
                    let tf = ts_field.get_or_insert_with(|| "timestamp".into()).clone();
                    let mut gc = GrokCreator::new(ov, &custom, &mut mappings, &mut stats);
                    gc.split(&best.grok, None, &tf, Breaks::Word, usize::MAX, true, messages.clone(), 0, 0);
                    gc.out.replace('\t', "\\t").replace('\n', "\\n")
                }
            }
        }
    };
    let ts = (ts_field.unwrap_or_else(|| "timestamp".into()), best);
    st.pipeline = ingest_pipeline(Some(&grok_pattern), &custom, None, &mappings, Some(&ts), ov.ecs_compatibility.as_deref());
    st.grok_pattern = Some(grok_pattern);
    st.ts = Some(ts);
    st.mappings = mappings;
    st.stats = stats;
    Ok(())
}

// ------------------------------------------------------------ entry points

fn decode(bytes: &[u8], ov: &Overrides, explanation: &mut Vec<String>) -> Result<(String, String, Option<bool>)> {
    match &ov.charset {
        Some(cs) => {
            let up = cs.to_uppercase();
            let text = match up.as_str() {
                "UTF-8" | "UTF8" => String::from_utf8_lossy(bytes).into_owned(),
                "US-ASCII" | "ASCII" | "ISO-8859-1" | "LATIN1" | "ISO8859_1" | "WINDOWS-1252" | "CP1252" => bytes.iter().map(|b| *b as char).collect(),
                "UTF-16LE" | "UTF-16BE" | "UTF-16" => {
                    let be = up != "UTF-16LE";
                    let u: Vec<u16> = bytes.chunks(2).map(|c| if c.len() < 2 { c[0] as u16 } else if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) }).collect();
                    String::from_utf16_lossy(&u)
                }
                _ => return Err(ia(format!("Supplied character encoding [{cs}] not available"))),
            };
            explanation.push(format!("Using specified character encoding [{cs}]"));
            let bom = if up.starts_with("UTF") { Some(text.starts_with('\u{FEFF}')) } else { None };
            Ok((text.trim_start_matches('\u{FEFF}').to_string(), cs.clone(), bom))
        }
        None => match std::str::from_utf8(bytes) {
            Ok(s) => {
                explanation.push("Using character encoding [UTF-8], which matched the input with [100%] confidence".into());
                Ok((s.trim_start_matches('\u{FEFF}').to_string(), "UTF-8".into(), Some(s.starts_with('\u{FEFF}'))))
            }
            Err(_) => {
                explanation.push("Using character encoding [ISO-8859-1], which matched the input with [50%] confidence".into());
                Ok((bytes.iter().map(|b| *b as char).collect(), "ISO-8859-1".into(), None))
            }
        },
    }
}

fn allowed_bad(ov: &Overrides) -> f64 {
    if ov.delimiter.is_some() {
        0.10
    } else if ov.quote.is_some() || ov.should_trim_fields.is_some() || ov.format.as_deref() == Some("delimited") {
        0.05
    } else {
        0.0
    }
}

fn run_factories(sample: &str, lines: &[&str], messages: Option<&[String]>, st: &mut Structure, ov: &Overrides, explanation: &mut Vec<String>) -> Result<()> {
    let fmt = ov.format.as_deref();
    let bad = allowed_bad(ov);
    if fmt.is_none() || fmt == Some("ndjson") {
        if can_ndjson(lines, explanation) {
            return ndjson(lines, st, ov, explanation);
        }
    }
    if fmt.is_none() || fmt == Some("delimited") {
        let delims: Vec<(char, usize, bool)> = match ov.delimiter {
            Some(d) => vec![(d, 2, ov.should_trim_fields.unwrap_or(false))],
            None => vec![(',', 2, false), ('\t', 2, false), (';', 4, false), ('|', 5, true)],
        };
        if !(fmt.is_none() && messages.is_some() && lines.len() < 2) {
            for (d, min, trim) in delims {
                let quote = ov.quote.unwrap_or('"');
                let trim = ov.should_trim_fields.unwrap_or(trim);
                if can_delimited(sample, d, quote, min, bad, explanation) {
                    return delimited(sample, d, quote, trim, st, ov, explanation);
                }
            }
        }
    }
    if fmt.is_none() || fmt == Some("semi_structured_text") {
        return match messages {
            Some(m) => log_text(None, Some(m), st, ov, explanation),
            None => log_text(Some(lines), None, st, ov, explanation),
        };
    }
    Err(ia(format!("Input did not match {}", match fmt {
        None => "any known formats".to_string(),
        Some(f) => format!("the specified format [{f}]"),
    })))
}

/// `POST /_text_structure/find_structure` over raw bytes.
pub fn find_structure(bytes: &[u8], ov: &Overrides) -> Result<Value> {
    let mut explanation = Vec::new();
    let (text, charset, bom) = decode(bytes, ov, &mut explanation)?;
    let mut sample_lines: Vec<&str> = Vec::new();
    for l in text.split('\n') {
        sample_lines.push(l.strip_suffix('\r').unwrap_or(l));
    }
    if sample_lines.last() == Some(&"") {
        sample_lines.pop();
    }
    sample_lines.truncate(ov.lines_to_sample);
    if sample_lines.len() < 2 {
        return Err(ia(format!("Input contained too few lines [{}] to obtain a meaningful sample", sample_lines.len())));
    }
    let sample: String = sample_lines.iter().map(|l| format!("{l}\n")).collect();
    // Java's split drops trailing empty strings.
    let mut lines: Vec<&str> = sample.split('\n').collect();
    while lines.last() == Some(&"") {
        lines.pop();
    }
    let mut st = Structure { charset, bom, ecs: ov.ecs_compatibility.clone().unwrap_or_else(|| "disabled".into()), ..Default::default() };
    run_factories(&sample, &lines, None, &mut st, ov, &mut explanation)?;
    Ok(st.to_json(&explanation, ov.explain))
}

/// `find_message_structure` / `find_field_structure` over messages.
pub fn find_message_structure(messages: &[String], ov: &Overrides) -> Result<Value> {
    if messages.is_empty() {
        return Err(ia("No messages found"));
    }
    let mut explanation = Vec::new();
    let joined = messages.join("\n");
    let lines: Vec<&str> = joined.split('\n').collect();
    let sample: String = messages.iter().map(|m| format!("{m}\n")).collect();
    let mut st = Structure { charset: "UTF-8".into(), bom: None, ecs: ov.ecs_compatibility.clone().unwrap_or_else(|| "disabled".into()), ..Default::default() };
    // NDJSON: every message must be one object (no embedded newlines).
    let fmt = ov.format.as_deref();
    if (fmt.is_none() || fmt == Some("ndjson")) && messages.iter().all(|m| !m.contains('\n')) && can_ndjson(&lines, &mut explanation) {
        ndjson(&lines, &mut st, ov, &mut explanation)?;
        return Ok(st.to_json(&explanation, ov.explain));
    }
    if fmt.is_none() || fmt == Some("delimited") {
        let bad = allowed_bad(ov);
        let delims: Vec<(char, usize, bool)> = match ov.delimiter {
            Some(d) => vec![(d, 2, ov.should_trim_fields.unwrap_or(false))],
            None => vec![(',', 2, false), ('\t', 2, false), (';', 4, false), ('|', 5, true)],
        };
        for (d, min, trim) in delims {
            let quote = ov.quote.unwrap_or('"');
            if messages.iter().all(|m| !m.contains('\n')) && can_delimited(&sample, d, quote, min, bad, &mut explanation) {
                let mut ov2 = ov.clone();
                if ov2.has_header_row.is_none() {
                    ov2.has_header_row = Some(false);
                }
                delimited(&sample, d, quote, ov.should_trim_fields.unwrap_or(trim), &mut st, &ov2, &mut explanation)?;
                return Ok(st.to_json(&explanation, ov.explain));
            }
        }
    }
    if fmt.is_none() || fmt == Some("semi_structured_text") {
        log_text(None, Some(messages), &mut st, ov, &mut explanation)?;
        return Ok(st.to_json(&explanation, ov.explain));
    }
    Err(ia(format!("Input did not match the specified format [{}]", fmt.unwrap_or(""))))
}

/// UTF-16 length of a string (Java offsets).
fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `POST /_text_structure/test_grok_pattern`.
pub fn test_grok_pattern(body: &Value, ecs: Option<&str>) -> Result<Value> {
    if let Some(e) = ecs {
        if e != "disabled" && e != "v1" {
            return Err(ia(format!("[ecs_compatibility] must be one of [disabled, v1] but was [{e}]")));
        }
    }
    let pattern = body.get("grok_pattern").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["[grok_pattern] is required"]))?;
    let texts = body.get("text").and_then(Value::as_array).ok_or_else(|| EsError::action_validation(&["[text] is required"]))?;
    let g = Grok::compile(pattern, grok::bank())?;
    let names: Vec<(String, String)> = g.re.capture_names().flatten().map(|n| (n.to_string(), g.field_of(n))).collect();
    let mut out = Vec::new();
    for t in texts {
        let t = t.as_str().unwrap_or("");
        match g.re.captures(t) {
            None => out.push(json!({"matched": false})),
            Some(c) => {
                let mut fields: Map<String, Value> = Map::new();
                for (gname, field) in &names {
                    if let Some(m) = c.name(gname) {
                        let entry = json!({"match": m.as_str(), "offset": utf16_len(&t[..m.start()]), "length": utf16_len(m.as_str())});
                        match fields.get_mut(field) {
                            Some(Value::Array(a)) => a.push(entry),
                            _ => {
                                fields.insert(field.clone(), json!([entry]));
                            }
                        }
                    }
                }
                out.push(json!({"matched": true, "fields": fields}));
            }
        }
    }
    Ok(json!({ "matches": out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ov() -> Overrides {
        Overrides { lines_to_sample: 1000, line_merge_size_limit: 10_000, ..Default::default() }
    }

    #[test]
    fn ndjson_structure() {
        let s = "{\"airline\": \"AAL\", \"responsetime\": 132.2046, \"time\": 1403481600}\n{\"airline\": \"JZA\", \"responsetime\": 990.4628, \"time\": 1403481700}\n{\"airline\": \"AAL\", \"responsetime\": 134.2046, \"time\": 1403481800}\n";
        let v = find_structure(s.as_bytes(), &ov()).unwrap();
        assert_eq!(v["format"], json!("ndjson"));
        assert_eq!(v["timestamp_field"], json!("time"));
        assert_eq!(v["java_timestamp_formats"][0], json!("UNIX"));
        assert_eq!(v["mappings"]["properties"]["time"], json!({"type": "date", "format": "epoch_second"}));
        assert_eq!(v["mappings"]["properties"]["responsetime"]["type"], json!("double"));
        assert_eq!(v["field_stats"]["responsetime"]["median_value"], json!(134.2046));
        assert_eq!(v["field_stats"]["airline"]["cardinality"], json!(2));
    }

    #[test]
    fn log_messages() {
        let msgs = vec![
            "2019-05-16 16:56:14 line 1 abcdefghijklmnopqrstuvwxyz".to_string(),
            "2019-05-16 16:56:14 line 2 abcdefghijklmnopqrstuvwxyz\ncontinuation...\ncontinuation...\n".to_string(),
            "2019-05-16 16:56:14 line 3 abcdefghijklmnopqrstuvwxyz".to_string(),
        ];
        let v = find_message_structure(&msgs, &ov()).unwrap();
        assert_eq!(v["format"], json!("semi_structured_text"));
        assert_eq!(v["grok_pattern"], json!("%{TIMESTAMP_ISO8601:timestamp} .*? %{INT:field} .*"));
        assert_eq!(v["java_timestamp_formats"][0], json!("yyyy-MM-dd HH:mm:ss"));
    }

    #[test]
    fn csv_with_header() {
        let s = "name,count,time\nfoo,1,2020-01-01T00:00:00Z\nbar,2,2020-01-02T00:00:00Z\nbaz,3,2020-01-03T00:00:00Z\n";
        let v = find_structure(s.as_bytes(), &ov()).unwrap();
        assert_eq!(v["format"], json!("delimited"));
        assert_eq!(v["has_header_row"], json!(true));
        assert_eq!(v["column_names"], json!(["name", "count", "time"]));
        assert_eq!(v["mappings"]["properties"]["count"]["type"], json!("long"));
        assert_eq!(v["timestamp_field"], json!("time"));
        assert_eq!(v["java_timestamp_formats"], json!(["ISO8601"]));
    }

    #[test]
    fn grok_test() {
        let v = test_grok_pattern(&json!({"grok_pattern": "%{WORD}.*%{WORD:name} %{WORD:name}!", "text": ["Hello   Dave Roberts!", "x"]}), None).unwrap();
        assert_eq!(v["matches"][0]["fields"]["name"][1], json!({"match": "Roberts", "offset": 13, "length": 7}));
        assert_eq!(v["matches"][1], json!({"matched": false}));
    }
}
