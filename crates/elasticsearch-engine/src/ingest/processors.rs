//! Processor implementations and the factory that builds them from config.
//! Names, parameters, defaults and error messages follow the ES
//! `ingest-common` / `ingest-user-agent` / `ingest-geoip` modules; a few
//! Beats-style extras (`copy`, `add_fields`, `number`, `hex_to_dec`,
//! `calculation`) are accepted as well.

use super::doc::{type_name, value_to_string, IngestDocument};
use super::grok::{self, Grok};
use super::{run_node, run_pipeline, Cfg, PResult, PipelineContext, Processor, ProcessorError, ProcessorNode, Script};
use crate::date::{DateFormatter, Tz};
use crate::error::{EsError, Result};
use regex::Regex;
use serde_json::{json, Map, Number, Value};
use std::collections::BTreeMap;

pub(super) fn ia(m: impl Into<String>) -> EsError {
    EsError::illegal_argument(m)
}

pub(super) fn rf(doc: &IngestDocument, f: &str) -> String {
    if f.contains("{{") { doc.render(f) } else { f.to_string() }
}

pub(super) fn missing(field: &str) -> EsError {
    let leaf = field.rsplit('.').next().unwrap_or(field);
    ia(format!("field [{leaf}] not present as part of path [{field}]"))
}

/// Field value for transform processors: `Ok(None)` = skip (ignore_missing).
pub(super) fn input(doc: &IngestDocument, field: &str, ignore_missing: bool, null_msg: &str) -> Result<Option<Value>> {
    match doc.get(field)? {
        None | Some(Value::Null) if ignore_missing => Ok(None),
        None => Err(doc.require(field).err().unwrap_or_else(|| missing(field))),
        Some(Value::Null) => Err(ia(format!("field [{field}] is null, {null_msg}"))),
        Some(v) => Ok(Some(v.clone())),
    }
}

fn f2v(f: f64) -> Value {
    Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
}

/// Java `toString` of a value (doubles keep a `.0`).
pub fn java_string(v: &Value) -> String {
    super::jtext::java_tostring(v)
}

// ------------------------------------------------------------ string processors

struct StrProc {
    field: String,
    target: String,
    ignore_missing: bool,
    f: Box<dyn Fn(&str) -> Result<String> + Send + Sync>,
}

impl Processor for StrProc {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let out = match v {
            Value::String(s) => Value::String((self.f)(&s)?),
            Value::Array(a) => {
                let mut out = Vec::with_capacity(a.len());
                for x in a {
                    match x {
                        Value::String(s) => out.push(Value::String((self.f)(&s)?)),
                        other => {
                            return Err(ia(format!("value [{}] of type [{}] in list field [{}] cannot be cast to [java.lang.String]", java_string(&other), type_name(&other), self.field)).into());
                        }
                    }
                }
                Value::Array(out)
            }
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        doc.set(&self.target, out)?;
        Ok(())
    }
}

fn str_proc(cfg: &mut Cfg, f: Box<dyn Fn(&str) -> Result<String> + Send + Sync>) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    Ok(Box::new(StrProc { field, target, ignore_missing, f }))
}

pub use super::p_data::url_decode;

// ------------------------------------------------------------ set / remove / rename / append / copy

struct SetP {
    field: String,
    value: Option<Value>,
    copy_from: Option<String>,
    overwrite: bool,
    ignore_empty: bool,
    media: super::jtext::MediaType,
}

impl Processor for SetP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let field = rf(doc, &self.field);
        if !self.overwrite {
            if let Ok(Some(v)) = doc.get(&field) {
                if !v.is_null() {
                    return Ok(());
                }
            }
        }
        let v = match &self.copy_from {
            // `getFieldValue(copyFrom, Object.class, ignoreEmptyValue)`: a
            // missing source is null (skipped) with `ignore_empty_value`.
            Some(c) if self.ignore_empty => match doc.get(c) {
                Ok(Some(v)) => v.clone(),
                _ => Value::Null,
            },
            Some(c) => doc.require(c)?.clone(),
            None => doc.render_value_media(self.value.as_ref().unwrap_or(&Value::Null), self.media),
        };
        if self.ignore_empty && (v.is_null() || v.as_str() == Some("")) {
            return Ok(());
        }
        doc.set(&field, v)?;
        Ok(())
    }
}

struct RemoveP {
    fields: Vec<String>,
    keep: Option<Vec<String>>,
    ignore_missing: bool,
}

impl Processor for RemoveP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        if let Some(keep) = &self.keep {
            let mut kept = Map::new();
            for k in keep {
                let k = rf(doc, k);
                if let Some(v) = doc.get(&k)? {
                    let v = v.clone();
                    let mut tmp = IngestDocument::new(std::mem::take(&mut kept), "", None, None, String::new());
                    tmp.ctx_map_mut().shift_remove("_index");
                    tmp.set(&k, v)?;
                    kept = std::mem::take(tmp.ctx_map_mut());
                }
            }
            let meta: Vec<(String, Value)> = doc.ctx_map().iter().filter(|(k, _)| super::doc::METADATA_KEYS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
            let ctx = doc.ctx_map_mut();
            ctx.clear();
            for (k, v) in kept {
                ctx.insert(k, v);
            }
            for (k, v) in meta {
                ctx.insert(k, v);
            }
            return Ok(());
        }
        for f in &self.fields {
            let f = rf(doc, f);
            if self.ignore_missing && !doc.has(&f) {
                continue;
            }
            doc.remove(&f)?;
        }
        Ok(())
    }
}

struct RenameP {
    field: String,
    target: String,
    ignore_missing: bool,
    overwrite: bool,
}

impl Processor for RenameP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let f = rf(doc, &self.field);
        let t = rf(doc, &self.target);
        if !doc.has(&f) {
            if self.ignore_missing {
                return Ok(());
            }
            return Err(ia(format!("field [{f}] doesn't exist")).into());
        }
        if doc.has(&t) && !self.overwrite {
            return Err(ia(format!("field [{t}] already exists")).into());
        }
        let v = doc.remove(&f)?;
        if let Err(e) = doc.set(&t, v.clone()) {
            let _ = doc.set(&f, v);
            return Err(e.into());
        }
        Ok(())
    }
}

struct AppendP {
    field: String,
    value: Value,
    allow_duplicates: bool,
    media: super::jtext::MediaType,
}

impl Processor for AppendP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let f = rf(doc, &self.field);
        let v = doc.render_value_media(&self.value, self.media);
        doc.append(&f, v, self.allow_duplicates)?;
        Ok(())
    }
}

// ------------------------------------------------------------ date

#[derive(Clone)]
enum DateFmt {
    Iso,
    Unix,
    UnixMs,
    Tai64n,
    Java(String),
}

fn parse_date_formats(formats: &[String]) -> Vec<DateFmt> {
    formats
        .iter()
        .map(|f| match f.as_str() {
            "ISO8601" => DateFmt::Iso,
            "UNIX" => DateFmt::Unix,
            "UNIX_MS" => DateFmt::UnixMs,
            "TAI64N" => DateFmt::Tai64n,
            other => DateFmt::Java(other.to_string()),
        })
        .collect()
}

#[cfg(test)]
thread_local! {
    /// Tests: the "current" UTC year yearless dates take.
    pub(crate) static INGEST_YEAR: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

/// The current UTC year (ES ingest `DateFormat` completes a date parsed
/// without a year with it, finding 252).
fn current_year() -> i32 {
    #[cfg(test)]
    if let Some(y) = INGEST_YEAR.with(|y| y.get()) {
        return y;
    }
    chrono::Datelike::year(&chrono::Utc::now())
}

/// Parse an ingest date to epoch nanoseconds (finding 243: the
/// sub-millisecond digits survive to the output format).
fn parse_with(fmts: &[DateFmt], s: &str, tz: Tz) -> Option<i128> {
    for f in fmts {
        let r = match f {
            DateFmt::Iso => DateFormatter::new("date_optional_time").ok().and_then(|d| d.parse_nanos_with(s, tz, None).ok()),
            DateFmt::Unix => s.trim().parse::<f64>().ok().map(|x| (x * 1000.0).round() as i64 as i128 * 1_000_000),
            DateFmt::UnixMs => s.trim().parse::<i64>().ok().map(|m| m as i128 * 1_000_000),
            DateFmt::Tai64n => {
                let t = s.trim().trim_start_matches('@');
                if t.len() >= 24 {
                    let base = i64::from_str_radix(&t[..16], 16).ok();
                    let rest = i64::from_str_radix(&t[16..24], 16).ok();
                    match (base, rest) {
                        (Some(b), Some(r)) => Some((b - 4611686018427387914) as i128 * 1_000_000_000 + r as i128),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            DateFmt::Java(p) => DateFormatter::new(p).ok().and_then(|d| d.parse_nanos_with(s, tz, Some(current_year())).ok()),
        };
        if r.is_some() {
            return r;
        }
    }
    None
}

const EN_MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const EN_DAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/// Rewrite locale month/weekday names to English so the English-only parser can read them;
/// `None` when the text uses English names under a non-English locale (ES fails that parse).
fn localize_input(s: &str, locale: &str, ampm: bool) -> Option<String> {
    let lang = locale.split(['_', '-']).next().unwrap_or("").to_lowercase();
    if lang.is_empty() || lang == "en" {
        return Some(s.to_string());
    }
    let Some((months, days)) = crate::locale_names::month_and_day_names(locale) else { return Some(s.to_string()) };
    let out = s.to_string();
    let mut pairs: Vec<(String, &str)> = Vec::new();
    for (m, i) in months {
        pairs.push((m.to_string(), EN_MONTHS[i]));
    }
    for (d, i) in days {
        pairs.push((d.to_string(), EN_DAYS[i]));
    }
    // The locale's AM / PM markers for a pattern with `a`
    // (`crate::locale_names::am_pm`, CLDR day periods).
    if ampm {
        if let Some([am, pm]) = crate::locale_names::am_pm(locale) {
            pairs.push((am.to_string(), "AM"));
            pairs.push((pm.to_string(), "PM"));
        }
    }
    pairs.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()).then_with(|| a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    pairs.dedup_by(|a, b| a.0.to_lowercase() == b.0.to_lowercase());
    let lower = out.to_lowercase();
    // English names that are not also valid locale names make the parse fail
    let local_set: Vec<String> = pairs.iter().map(|p| p.0.to_lowercase()).collect();
    for en in EN_MONTHS.iter().chain(EN_DAYS.iter()) {
        let e = en.to_lowercase();
        if lower.split(|c: char| !c.is_alphabetic()).any(|w| w == e) && !local_set.iter().any(|l| *l == e) {
            return None;
        }
    }
    // One pass over the original text: at each word start the longest
    // locale name matching a whole word is replaced; replaced text is never
    // scanned again (finding 462: `marzo` -> `March` must not then match
    // `mar` inside `March`).
    let chars: Vec<char> = out.chars().collect();
    let lowered: Vec<Vec<char>> = pairs.iter().map(|(n, _)| n.to_lowercase().chars().collect()).collect();
    let mut res = String::with_capacity(out.len());
    // Letters that form words with their neighbours (ideographic scripts
    // write names without spaces: `2025年3月2日`).
    let word = |c: char| c.is_alphabetic() && !matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF);
    let mut i = 0;
    while i < chars.len() {
        let at_start = i == 0 || !word(chars[i - 1]);
        let hit = at_start
            .then(|| {
                pairs.iter().zip(&lowered).find(|(_, name)| {
                    let n = name.len();
                    i + n <= chars.len()
                        && chars[i..i + n].iter().zip(name.iter()).all(|(c, l)| c.to_lowercase().eq(std::iter::once(*l)))
                        && (!word(name[n - 1]) || chars.get(i + n).is_none_or(|c| !word(*c)))
                })
            })
            .flatten();
        match hit {
            Some(((_, en), name)) => {
                res.push_str(en);
                i += name.len();
            }
            None => {
                res.push(chars[i]);
                i += 1;
            }
        }
    }
    Some(res)
}

/// Whether a Java pattern among `fmts` has an AM/PM field (`a` outside
/// quoted literals).
fn uses_am_pm(fmts: &[DateFmt]) -> bool {
    fmts.iter().any(|f| match f {
        DateFmt::Java(p) => {
            let mut quoted = false;
            p.chars().any(|c| {
                if c == '\'' {
                    quoted = !quoted;
                }
                !quoted && c == 'a'
            })
        }
        _ => false,
    })
}

fn fmt_label(f: &DateFmt) -> String {
    match f {
        DateFmt::Iso => "iso8601".into(),
        DateFmt::Unix => "UNIX".into(),
        DateFmt::UnixMs => "UNIX_MS".into(),
        DateFmt::Tai64n => "TAI64N".into(),
        DateFmt::Java(p) => p.clone(),
    }
}

/// ES `DateProcessor` failure: `unable to parse date [x]` caused by the first
/// format's failure, later formats attached as `suppressed`.
fn date_parse_error(fmts: &[DateFmt], s: &str) -> EsError {
    let one = |f: &DateFmt| {
        let inner = if matches!(f, DateFmt::Iso) {
            "Failed to parse with all enclosed parsers".to_string()
        } else {
            format!("Text '{s}' could not be parsed at index 0")
        };
        ia(format!("failed to parse date field [{s}] with format [{}]", fmt_label(f))).caused_by(EsError::new(400, "date_time_parse_exception", inner))
    };
    let mut first = fmts.first().map(one).unwrap_or_else(|| ia("cannot parse empty datetime"));
    if fmts.len() > 1 {
        let sup: Vec<Value> = fmts[1..].iter().map(|f| one(f).error_object(false)).map(|mut v| {
            if let Some(o) = v.as_object_mut() {
                o.remove("root_cause");
            }
            v
        }).collect();
        first = first.with("suppressed", Value::Array(sup));
    }
    ia(format!("unable to parse date [{s}]")).caused_by(first)
}

fn parse_tz(s: &str) -> Result<Tz> {
    Tz::parse(s).map_err(|_| ia(format!("The datetime zone id '{s}' is not recognised")))
}

struct DateP {
    field: String,
    target: String,
    formats: Vec<DateFmt>,
    timezone: Option<String>,
    output_format: String,
    locale: Option<String>,
}

impl Processor for DateP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = doc.require(&self.field)?.clone();
        let s = match &v {
            Value::Null => return Err(ia("unable to parse date [null]").caused_by(ia("cannot parse empty datetime")).into()),
            other => value_to_string(other),
        };
        let tz = match &self.timezone {
            Some(t) => {
                let t = rf(doc, t);
                Tz::parse(&t).map_err(|_| ia(format!("unable to parse date [{s}]")).caused_by(EsError::new(400, "zone_rules_exception", format!("Unknown time-zone ID: {t}"))))?
            }
            None => Tz::UTC,
        };
        let parse_src = match &self.locale {
            Some(l) => localize_input(&s, &rf(doc, l), uses_am_pm(&self.formats)),
            None => Some(s.clone()),
        };
        let nanos = parse_src.and_then(|p| parse_with(&self.formats, &p, tz)).ok_or_else(|| date_parse_error(&self.formats, &s))?;
        let out = DateFormatter::new(&self.output_format)?.format_nanos(nanos, tz);
        doc.set(&self.target, Value::String(out))?;
        Ok(())
    }
}

/// `date_index_name`: `_index` = `<prefix{date||/rounding{format|tz}}>`.
struct DateIndexNameP {
    field: String,
    prefix: String,
    rounding: String,
    formats: Vec<DateFmt>,
    timezone: String,
    index_format: String,
    /// The `locale` input dates are parsed in (finding 461).
    locale: Option<String>,
}

impl Processor for DateIndexNameP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = doc.require(&self.field)?.clone();
        let s = value_to_string(&v);
        let tz = parse_tz(&rf(doc, &self.timezone))?;
        let parse_src = match &self.locale {
            Some(l) => localize_input(&s, &rf(doc, l), uses_am_pm(&self.formats)),
            None => Some(s.clone()),
        };
        let ms = parse_src.and_then(|p| parse_with(&self.formats, &p, tz)).ok_or_else(|| date_parse_error(&self.formats, &s))?.div_euclid(1_000_000) as i64;
        let fmt = rf(doc, &self.index_format);
        let day = DateFormatter::new(&fmt)?.format(ms, tz);
        let rounding = rf(doc, &self.rounding);
        let name = format!("<{}{{{day}||/{rounding}{{{fmt}|{}}}}}>", rf(doc, &self.prefix), rf(doc, &self.timezone));
        doc.set("_index", Value::String(name))?;
        Ok(())
    }
}

/// Resolve the `_index` an ingest pipeline produced: `date_index_name`
/// expressions (`<p{2016-04-25||/M{yyyy-MM-dd|UTC}}>`) and date math.
pub fn resolve_index_expression(name: &str) -> Result<String> {
    if !(name.starts_with('<') && name.ends_with('>')) {
        return Ok(name.to_string());
    }
    let inner = &name[1..name.len() - 1];
    if let (Some(open), Some(sep)) = (inner.find('{'), inner.find("||/")) {
        if open < sep {
            let prefix = &inner[..open];
            let anchor = &inner[open + 1..sep];
            let rest = &inner[sep + 3..];
            if let Some(brace) = rest.find('{') {
                let unit = &rest[..brace];
                let fmt_tz = rest[brace + 1..].trim_end_matches('}');
                let (fmt, tz) = fmt_tz.split_once('|').unwrap_or((fmt_tz, "UTC"));
                let tz = Tz::parse(tz)?;
                let ms = match DateFormatter::new(fmt).and_then(|f| f.parse_with_tz(anchor, tz)) {
                    Ok(ms) => ms,
                    Err(_) => DateFormatter::new("date_optional_time")?.parse_with_tz(anchor, tz)?,
                };
                let unit = crate::date::CalUnit::parse(unit).ok_or_else(|| ia(format!("unit [{unit}] not supported for date math")))?;
                let rounded = crate::date::round_down(ms, unit, tz);
                return Ok(format!("{prefix}{}", DateFormatter::new(fmt)?.format(rounded, tz)));
            }
        }
    }
    crate::node::resolve_date_math(name)
}

// ------------------------------------------------------------ grok / dissect / kv / csv / gsub / split / join

struct GrokP {
    field: String,
    groks: Vec<Grok>,
    trace_match: bool,
    ignore_missing: bool,
}

impl Processor for GrokP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        for (i, g) in self.groks.iter().enumerate() {
            if let Some(caps) = g.captures_checked(&s)? {
                for (k, v) in caps {
                    doc.set(&k, v)?;
                }
                if self.trace_match {
                    doc.set("_ingest._grok_match_index", Value::String(i.to_string()))?;
                }
                return Ok(());
            }
        }
        Err(ia(format!("Provided Grok expressions do not match field value: [{s}]")).into())
    }
}

struct RedactP {
    field: String,
    groks: Vec<Grok>,
    prefix: String,
    suffix: String,
    ignore_missing: bool,
    trace_redact: bool,
}

impl Processor for RedactP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        // ES `RedactProcessor`: a missing / null field is one error.
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None | Some(Value::Null) => return Err(ia(format!("field [{}] is null or missing", self.field)).into()),
            Some(v) => v.clone(),
        };
        let mut s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        for g in &self.groks {
            let mut out = String::new();
            let mut last = 0;
            for (start, end, label) in g.spans(&s) {
                out.push_str(&s[last..start]);
                out.push_str(&format!("{}{label}{}", self.prefix, self.suffix));
                last = end;
            }
            out.push_str(&s[last..]);
            s = out;
        }
        let redacted = Some(&Value::String(s.clone())) != doc.get(&self.field)?;
        doc.set(&self.field, Value::String(s))?;
        if self.trace_redact {
            doc.set("_ingest._redact._is_redacted", Value::Bool(redacted))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct DKey {
    name: String,
    skip: bool,
    append: bool,
    order: i64,
    is_ref_key: bool,
    is_ref_val: bool,
    right_pad: bool,
}

struct DissectP {
    field: String,
    prefix: String,
    keys: Vec<(DKey, String)>,
    append_separator: String,
    ignore_missing: bool,
    pattern: String,
}

fn dissect_exc(kind: &str, msg: String) -> EsError {
    let class = if kind == "pattern_parse" { "PatternParse" } else { "KeyParse" };
    EsError::new(500, "exception", format!("org.elasticsearch.dissect.DissectException${class}: {msg}")).caused_by(EsError::new(500, kind, msg))
}

fn dissect_no_keys(pattern: &str) -> EsError {
    dissect_exc("pattern_parse", format!("Unable to parse pattern: {pattern} Reason: Unable to find any keys or delimiters."))
}

fn parse_dissect(pattern: &str) -> Result<(String, Vec<(DKey, String)>)> {
    let mut prefix = String::new();
    let mut out: Vec<(DKey, String)> = Vec::new();
    let mut rest = pattern;
    let mut first = true;
    while let Some(st) = rest.find("%{") {
        let lit = &rest[..st];
        if first {
            prefix = lit.to_string();
        } else if let Some(last) = out.last_mut() {
            last.1 = lit.to_string();
        }
        first = false;
        let end = rest[st..].find('}').ok_or_else(|| dissect_no_keys(pattern))? + st;
        let raw_key = &rest[st + 2..end];
        let mut k = raw_key;
        let mut key = DKey { name: String::new(), skip: false, append: false, order: 0, is_ref_key: false, is_ref_val: false, right_pad: false };
        if let Some(x) = k.strip_suffix("->") {
            key.right_pad = true;
            k = x;
        }
        if let Some(x) = k.strip_prefix('?') {
            key.skip = true;
            k = x;
        } else if let Some(x) = k.strip_prefix('+') {
            key.append = true;
            k = x;
        } else if let Some(x) = k.strip_prefix('*') {
            key.is_ref_key = true;
            k = x;
        } else if let Some(x) = k.strip_prefix('&') {
            key.is_ref_val = true;
            k = x;
        }
        let key_err = || dissect_exc("key_parse", format!("Unable to parse key: {raw_key} Reason: The key name could not be determined"));
        if let Some((n, o)) = k.split_once('/') {
            key.order = o.parse().map_err(|_| key_err())?;
            k = n;
        }
        if k.is_empty() {
            if key.append || key.is_ref_key || key.is_ref_val || key.right_pad && false {
                return Err(key_err());
            }
            key.skip = true;
        }
        key.name = k.to_string();
        out.push((key, String::new()));
        rest = &rest[end + 1..];
    }
    if out.is_empty() || out.iter().all(|(k, _)| k.skip && k.name.is_empty() && !k.is_ref_key && !k.is_ref_val) && out.len() == 1 {
        return Err(dissect_no_keys(pattern));
    }
    {
        let ks: Vec<&str> = out.iter().filter(|(k, _)| k.is_ref_key).map(|(k, _)| k.name.as_str()).collect();
        let vs: Vec<&str> = out.iter().filter(|(k, _)| k.is_ref_val).map(|(k, _)| k.name.as_str()).collect();
        let bad: Vec<&str> = ks.iter().filter(|n| !vs.contains(n)).chain(vs.iter().filter(|n| !ks.contains(n))).copied().collect();
        if let Some(b) = bad.first() {
            return Err(dissect_exc("pattern_parse", format!("Unable to parse pattern: {pattern} Reason: Found invalid key/reference associations: '{b}' Please ensure each '*<key>' is matched with a matching '&<key>")));
        }
    }
    if let Some(last) = out.last_mut() {
        last.1 = rest.to_string();
    }
    Ok((prefix, out))
}

impl Processor for DissectP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let fail = || -> ProcessorError { EsError::new(500, "find_match", format!("Unable to find match for dissect pattern: {} against source: {s}", self.pattern)).into() };
        let mut pos = match s.strip_prefix(self.prefix.as_str()) {
            Some(_) => self.prefix.len(),
            None => return Err(fail()),
        };
        let mut vals: Vec<(DKey, String)> = Vec::new();
        for (i, (key, delim)) in self.keys.iter().enumerate() {
            let last = i + 1 == self.keys.len();
            let value_end;
            let next_pos;
            if delim.is_empty() {
                if last {
                    value_end = s.len();
                    next_pos = s.len();
                } else {
                    return Err(fail());
                }
            } else {
                match s[pos..].find(delim.as_str()) {
                    Some(off) => {
                        value_end = pos + off;
                        let mut np = value_end + delim.len();
                        if key.right_pad {
                            while s[np..].starts_with(delim.as_str()) {
                                np += delim.len();
                            }
                        }
                        next_pos = np;
                    }
                    None => return Err(fail()),
                }
            }
            vals.push((key.clone(), s[pos..value_end].to_string()));
            pos = next_pos;
        }
        if pos < s.len() {
            return Err(fail());
        }
        // Assemble: appends (ordered), reference pairs, plain keys.
        let mut appends: BTreeMap<String, Vec<(i64, usize, String)>> = BTreeMap::new();
        let mut refs_k: BTreeMap<String, String> = BTreeMap::new();
        let mut refs_v: BTreeMap<String, String> = BTreeMap::new();
        let mut plain: Vec<(String, usize, String)> = Vec::new();
        for (i, (k, v)) in vals.into_iter().enumerate() {
            if k.skip {
                continue;
            }
            if k.is_ref_key {
                refs_k.insert(k.name.clone(), v);
            } else if k.is_ref_val {
                refs_v.insert(k.name.clone(), v);
            } else if k.append {
                appends.entry(k.name.clone()).or_default().push((k.order, i, v));
            } else {
                plain.push((k.name, i, v));
            }
        }
        for (n, i, v) in plain {
            if appends.contains_key(&n) {
                // A plain capture of an appended key keeps its pattern
                // position (finding 463).
                appends.get_mut(&n).unwrap().push((0, i, v));
            } else {
                doc.set(&n, Value::String(v))?;
            }
        }
        for (n, mut parts) in appends {
            parts.sort_by_key(|p| (p.0, p.1));
            let joined = parts.into_iter().map(|p| p.2).collect::<Vec<_>>().join(&self.append_separator);
            doc.set(&n, Value::String(joined))?;
        }
        for (n, k) in refs_k {
            if let Some(v) = refs_v.get(&n) {
                doc.set(&k, Value::String(v.clone()))?;
            }
        }
        Ok(())
    }
}

impl DissectP {
    fn extract_units(&self,s:&[u16])->std::result::Result<Vec<(String,Vec<u16>)>,ProcessorError>{
        let fail = || -> ProcessorError { EsError::new(500, "find_match", format!("Unable to find match for dissect pattern: {} against source: {}", self.pattern, String::from_utf16_lossy(s))).into() };
        let prefix:Vec<u16>=self.prefix.encode_utf16().collect();
        let mut pos = match s.strip_prefix(prefix.as_slice()) {
            Some(_) => prefix.len(),
            None => return Err(fail()),
        };
        let mut vals: Vec<(DKey, Vec<u16>)> = Vec::new();
        for (i, (key, delim)) in self.keys.iter().enumerate() {
            let delim:Vec<u16>=delim.encode_utf16().collect();
            let last = i + 1 == self.keys.len();
            let value_end;
            let next_pos;
            if delim.is_empty() {
                if last {
                    value_end = s.len();
                    next_pos = s.len();
                } else {
                    return Err(fail());
                }
            } else {
                match s[pos..].windows(delim.len()).position(|window|window==delim.as_slice()) {
                    Some(off) => {
                        value_end = pos + off;
                        let mut np = value_end + delim.len();
                        if key.right_pad {
                            while s[np..].starts_with(delim.as_slice()) {
                                np += delim.len();
                            }
                        }
                        next_pos = np;
                    }
                    None => return Err(fail()),
                }
            }
            vals.push((key.clone(), s[pos..value_end].to_vec()));
            pos = next_pos;
        }
        if pos < s.len() {
            return Err(fail());
        }
        // Assemble: appends (ordered), reference pairs, plain keys.
        let mut appends: BTreeMap<String, Vec<(i64, usize, Vec<u16>)>> = BTreeMap::new();
        let mut refs_k: BTreeMap<String, Vec<u16>> = BTreeMap::new();
        let mut refs_v: BTreeMap<String, Vec<u16>> = BTreeMap::new();
        let mut plain: Vec<(String, usize, Vec<u16>)> = Vec::new();
        for (i, (k, v)) in vals.into_iter().enumerate() {
            if k.skip {
                continue;
            }
            if k.is_ref_key {
                refs_k.insert(k.name.clone(), v);
            } else if k.is_ref_val {
                refs_v.insert(k.name.clone(), v);
            } else if k.append {
                appends.entry(k.name.clone()).or_default().push((k.order, i, v));
            } else {
                plain.push((k.name, i, v));
            }
        }
        let mut output=Vec::new();
        for (n, i, v) in plain {
            if appends.contains_key(&n) {
                appends.get_mut(&n).unwrap().push((0, i, v));
            } else {
                output.push((n,v));
            }
        }
        for (n, mut parts) in appends {
            parts.sort_by_key(|p| (p.0, p.1));
            let joined = parts.into_iter().map(|p| p.2).collect::<Vec<_>>().join(&self.append_separator.encode_utf16().collect::<Vec<_>>()[..]);
            output.push((n,joined));
        }
        for (n, k) in refs_k {
            if let Some(v) = refs_v.get(&n) {
                output.push((String::from_utf16(&k).map_err(|_|ia("invalid UTF16 dissect field name"))?,v.clone()));
            }
        }
        Ok(output)

    }
}
pub(super) fn dissect_source(config:&crate::source_json::Source,doc:&mut super::source_doc::SourceDocument)->PResult {
    let cfg:Value=serde_json::from_str(&config.text()).map_err(EsError::from)?;
    let field=cfg["field"].as_str().ok_or_else(||ia("field required"))?;
    let pattern=cfg["pattern"].as_str().ok_or_else(||ia("pattern required"))?;
    let value=doc.get(field)?.cloned();
    let Some(value)=value.filter(|v|v.text()!="null")else{if cfg["ignore_missing"].as_bool()==Some(true){return Ok(());}return Err(missing(field).into());};
    let value=crate::script::source_from_json(&value);
    if !matches!(&value,crate::script::Val::Str(_)|crate::script::Val::Utf16(_)){return Err(ia("dissect field must be a string").into());}
    let (prefix,keys)=parse_dissect(pattern)?;
    let processor=DissectP{field:field.into(),prefix,keys,append_separator:cfg["append_separator"].as_str().unwrap_or("").into(),ignore_missing:false,pattern:pattern.into()};
    // Dissect scans Java UTF8 bytes: the encoder replaces unpaired units
    // with '?'. Only extracted fields cross this boundary; source stays raw.
    let units=value.string_units();
    let encoded:String=char::decode_utf16(units).map(|c|c.unwrap_or('?')).collect();
    for(key,value)in processor.extract_units(&encoded.encode_utf16().collect::<Vec<_>>())?{doc.set(&key,crate::script::source_to_json(&crate::script::Val::utf16(value))?)?;}
    Ok(())
}

struct GsubP {
    field: String,
    target: String,
    re: Regex,
    /// The replacement in regex crate syntax, or why Java would refuse it
    /// (reported when the processor runs, as Java's `replaceAll` does).
    replacement: std::result::Result<String, String>,
    ignore_missing: bool,
}

impl Processor for GsubP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let replacement = self.replacement.as_ref().map_err(|e| ia(e.clone()))?;
        let f = |s: &str| self.re.replace_all(s, replacement.as_str()).into_owned();
        let out = match v {
            Value::String(s) => Value::String(f(&s)),
            Value::Array(a) => Value::Array(a.iter().map(|x| match x {
                Value::String(s) => Value::String(f(s)),
                o => o.clone(),
            }).collect()),
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        doc.set(&self.target, out)?;
        Ok(())
    }
}

/// Java replacement syntax (`$1`, `${name}`, `\$`) → regex crate syntax,
/// for a pattern with `groups` capturing groups. As Java's
/// `Matcher.appendReplacement`, digits after `$` extend the group number
/// only while that group exists (`$10` with one group is group 1 then `0`,
/// finding 250); a reference to a missing group or a bare `$` is an error.
fn java_replacement(r: &str, groups: usize) -> std::result::Result<String, String> {
    let mut out = String::new();
    let cs: Vec<char> = r.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        match cs[i] {
            '\\' => {
                let Some(&next) = cs.get(i + 1) else { return Err("character to be escaped is missing".into()) };
                if next == '$' {
                    out.push_str("$$");
                } else {
                    out.push(next);
                }
                i += 2;
            }
            '$' => match cs.get(i + 1) {
                Some('{') => {
                    let end = cs[i + 2..].iter().position(|c| *c == '}').ok_or("named capturing group is missing trailing '}'")?;
                    out.extend(&cs[i..i + 2 + end + 1]);
                    i += 2 + end + 1;
                }
                Some(d) if d.is_ascii_digit() => {
                    let mut n = d.to_digit(10).unwrap_or(0) as usize;
                    if n > groups {
                        return Err(format!("No group {n}"));
                    }
                    let mut j = i + 2;
                    while let Some(d) = cs.get(j).and_then(|c| c.to_digit(10)) {
                        let next = n * 10 + d as usize;
                        if next > groups {
                            break;
                        }
                        n = next;
                        j += 1;
                    }
                    out.push_str(&format!("${{{n}}}"));
                    i = j;
                }
                _ => return Err("Illegal group reference".into()),
            },
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Java regex → regex crate (possessive quantifiers become greedy). Syntax
/// errors are reported with `PatternSyntaxException` wording.
pub(super) fn java_regex(p: &str) -> Result<Regex> {
    if let Some(m) = super::jtext::java_pattern_error(p) {
        return Err(ia(format!("Invalid regex pattern. {m}")));
    }
    let fixed = p.replace("*+", "*").replace("++", "+").replace("?+", "?");
    Regex::new(&fixed).map_err(|e| ia(format!("Invalid regex pattern. unsupported regex construct: {e}")))
}

struct SplitP {
    field: String,
    target: String,
    re: Regex,
    ignore_missing: bool,
    preserve_trailing: bool,
}

impl Processor for SplitP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot split.")? else { return Ok(()) };
        let s = match v {
            Value::String(s) => s,
            other => return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.String]", self.field, type_name(&other))).into()),
        };
        let parts = super::p_data::java_split(&self.re, &s, if self.preserve_trailing { usize::MAX } else { 0 });
        doc.set(&self.target, Value::Array(parts.into_iter().map(Value::String).collect()))?;
        Ok(())
    }
}

struct JoinP {
    field: String,
    target: String,
    sep: String,
}

impl Processor for JoinP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let v = doc.require(&self.field)?.clone();
        match v {
            Value::Array(a) => {
                let s = a.iter().map(java_string).collect::<Vec<_>>().join(&self.sep);
                doc.set(&self.target, Value::String(s))?;
                Ok(())
            }
            Value::Null => Err(ia(format!("field [{}] is null, cannot join.", self.field)).into()),
            other => Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.util.List]", self.field, type_name(&other))).into()),
        }
    }
}

// ------------------------------------------------------------ json / convert / bytes / numbers

// ------------------------------------------------------------ control flow

struct DropP;
impl Processor for DropP {
    fn process(&self, _: &mut IngestDocument, _: &PipelineContext) -> PResult {
        Err(ProcessorError::Dropped)
    }
}

struct TerminateP;
impl Processor for TerminateP {
    fn process(&self, _: &mut IngestDocument, _: &PipelineContext) -> PResult {
        Err(ProcessorError::Terminated)
    }
}

struct FailP {
    message: String,
}
impl Processor for FailP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        Err(EsError::new(500, "fail_processor_exception", doc.render(&self.message)).into())
    }
}

struct InferenceP {
    model_id: String,
}
impl Processor for InferenceP {
    fn process(&self, _: &mut IngestDocument, _: &PipelineContext) -> PResult {
        Err(EsError::new(404, "resource_not_found_exception", format!("Could not find trained model [{}]", self.model_id)).into())
    }
}

struct ScriptP {
    script: Script,
}
impl Processor for ScriptP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        self.script.execute(doc)?;
        Ok(())
    }
}

struct PipelineP {
    name: String,
    ignore_missing_pipeline: bool,
}
impl Processor for PipelineP {
    fn process(&self, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
        let name = rf(doc, &self.name);
        match cx.pipeline(&name) {
            Some(p) => run_pipeline(&p, doc, cx),
            None if self.ignore_missing_pipeline => Ok(()),
            None => Err(EsError::new(500, "illegal_state_exception", format!("Pipeline processor configured for non-existent pipeline [{name}]")).into()),
        }
    }
}

struct ForeachP {
    field: String,
    inner: Box<ProcessorNode>,
    ignore_missing: bool,
}

impl Processor for ForeachP {
    fn process(&self, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
        let v = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(missing(&self.field).into()),
            Some(Value::Null) => return Err(ia(format!("field [{}] is null, cannot loop over its elements.", self.field)).into()),
            Some(v) => v.clone(),
        };
        // ES restores the previous value by `put`, so the keys stay behind (as null).
        let prev_value = doc.ingest.get("_value").cloned().unwrap_or(Value::Null);
        let prev_key = doc.ingest.get("_key").cloned().unwrap_or(Value::Null);
        cx.quiet.set(cx.quiet.get() + 1);
        let result = (|| -> PResult {
            match v {
                Value::Array(a) => {
                    let mut out = Vec::with_capacity(a.len());
                    for x in a {
                        doc.ingest_map_mut().insert("_value".into(), x);
                        run_node(&self.inner, doc, cx)?;
                        out.push(doc.ingest_map_mut().insert("_value".into(), prev_value.clone()).unwrap_or(Value::Null));
                    }
                    doc.set(&self.field, Value::Array(out))?;
                    Ok(())
                }
                Value::Object(m) => {
                    let mut out = Map::new();
                    for (k, x) in m {
                        doc.ingest_map_mut().insert("_key".into(), Value::String(k.clone()));
                        doc.ingest_map_mut().insert("_value".into(), x);
                        run_node(&self.inner, doc, cx)?;
                        let nk = doc.ingest_map_mut().insert("_key".into(), prev_key.clone()).filter(|k| !k.is_null()).map(|k| value_to_string(&k));
                        let nv = doc.ingest_map_mut().insert("_value".into(), prev_value.clone()).unwrap_or(Value::Null);
                        // A null or blank new key removes the entry (ES
                        // `Strings.hasText`, finding 476).
                        if let Some(nk) = nk.filter(|k| !k.trim().is_empty()) {
                            out.insert(nk, nv);
                        }
                        let _ = k;
                    }
                    doc.set(&self.field, Value::Object(out))?;
                    Ok(())
                }
                other => Err(ia(format!("field [{}] of type [{}] cannot be cast to a list or map", self.field, type_name(&other))).into()),
            }
        })();
        cx.quiet.set(cx.quiet.get() - 1);
        result
    }
}

// ------------------------------------------------------------ enrichment-ish

struct DotExpanderP {
    field: String,
    path: Option<String>,
    overwrite: bool,
}

impl Processor for DotExpanderP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let base: Map<String, Value> = match &self.path {
            Some(p) => match doc.get(p)? {
                Some(Value::Object(m)) => m.clone(),
                _ => return Ok(()),
            },
            None => doc.ctx_map().clone(),
        };
        let keys: Vec<String> = base
            .keys()
            .filter(|k| k.contains('.') && (self.field == "*" || **k == self.field) && !super::doc::METADATA_KEYS.contains(&k.as_str()))
            .cloned()
            .collect();
        if keys.is_empty() && self.field != "*" {
            let full = match &self.path {
                Some(p) => format!("{p}.{}", self.field),
                None => self.field.clone(),
            };
            if !doc.has(&full) {
                return Ok(());
            }
        }
        for k in keys {
            let target = match &self.path {
                Some(p) => format!("{p}.{k}"),
                None => k.clone(),
            };
            // Every ancestor is checked before the dotted key is removed (ES
            // `DotExpanderProcessor`): a failed expansion leaves it in place,
            // under ignore_failure / on_failure too (finding 249).
            let segs: Vec<&str> = target.split('.').collect();
            for i in 1..segs.len() {
                let pre = segs[..i].join(".");
                if let Ok(Some(pv)) = doc.get(&pre) {
                    if !pv.is_object() && !pv.is_null() && !pv.is_array() {
                        return Err(ia(format!("cannot expand [{k}], because [{pre}] is not an object field, but a value field")).into());
                    }
                }
            }
            let v = match &self.path {
                Some(p) => {
                    let parent = doc.get(p)?.cloned();
                    let mut m = match parent {
                        Some(Value::Object(m)) => m,
                        _ => continue,
                    };
                    let v = m.shift_remove(&k).unwrap_or(Value::Null);
                    doc.set(p, Value::Object(m))?;
                    v
                }
                None => doc.ctx_map_mut().shift_remove(&k).unwrap_or(Value::Null),
            };
            if doc.has(&target) && !self.overwrite {
                doc.append(&target, v, true)?;
            } else {
                doc.set(&target, v)?;
            }
        }
        Ok(())
    }
}

struct RerouteP {
    destination: Option<String>,
    ty: Vec<String>,
    dataset: Vec<String>,
    namespace: Vec<String>,
}

fn is_template(v: &str) -> bool {
    v.contains("{{")
}

/// Runtime sanitising of templated reroute values: lowercase, forbidden characters become `_`.
fn sanitize_ds(s: &str, namespace: bool) -> String {
    let bad = if namespace { "\\/*?\"<>| ,#:" } else { "\\/*?\"<>| ,#:-" };
    let mut out: String = s.to_lowercase().chars().map(|c| if bad.contains(c) { '_' } else { c }).collect();
    out.truncate(100);
    out
}

/// Static reroute values must already be valid: lowercase, at most 100 chars, no
/// separators; `-` is only allowed in a namespace.
fn valid_static(v: &str, namespace: bool) -> bool {
    let bad = if namespace { "\\/*?\"<>| ,#:" } else { "\\/*?\"<>| ,#:-" };
    v.chars().count() <= 100 && v == v.to_lowercase() && !v.chars().any(|c| bad.contains(c))
}

impl RerouteP {
    fn pick(&self, opts: &[String], doc: &IngestDocument, fallback: &str, namespace: bool) -> String {
        for o in opts {
            if is_template(o) {
                let r = sanitize_ds(&doc.render(o), namespace);
                if !r.is_empty() {
                    return r;
                }
            } else {
                return o.clone();
            }
        }
        fallback.to_string()
    }
}

impl Processor for RerouteP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        if let Some(d) = &self.destination {
            let d = rf(doc, d);
            doc.set("_index", Value::String(d))?;
            doc.rerouted = true;
            return Ok(());
        }
        let index = doc.meta_str("_index").unwrap_or_default();
        let bad = || ia(format!("invalid data stream name: [{index}]; must follow naming scheme <type>-<dataset>-<namespace>"));
        let d1 = index.find('-').ok_or_else(bad)?;
        let d2 = index[d1 + 1..].find('-').map(|i| i + d1 + 1).ok_or_else(bad)?;
        let (cur_ty, cur_ds, cur_ns) = (&index[..d1], &index[d1 + 1..d2], &index[d2 + 1..]);
        let dflt = |k: &str| vec![format!("{{{{data_stream.{k}}}}}")];
        let (dt, dd, dn) = (dflt("type"), dflt("dataset"), dflt("namespace"));
        let ty = self.pick(if self.ty.is_empty() { &dt } else { &self.ty }, doc, cur_ty, false);
        let ds = self.pick(if self.dataset.is_empty() { &dd } else { &self.dataset }, doc, cur_ds, false);
        let ns = self.pick(if self.namespace.is_empty() { &dn } else { &self.namespace }, doc, cur_ns, true);
        doc.set("_index", Value::String(format!("{ty}-{ds}-{ns}")))?;
        doc.rerouted = true;
        // Set in dotted or nested notation, keeping the one the document uses.
        fn set_kept(doc: &mut IngestDocument, path: &str, v: &str) -> Result<()> {
            if doc.ctx_map().contains_key(path) {
                doc.ctx_map_mut().insert(path.to_string(), Value::String(v.to_string()));
                Ok(())
            } else {
                doc.set(path, Value::String(v.to_string()))
            }
        }
        set_kept(doc, "data_stream.type", &ty)?;
        set_kept(doc, "data_stream.dataset", &ds)?;
        set_kept(doc, "data_stream.namespace", &ns)?;
        // ECS: `event.dataset` mirrors `data_stream.dataset` when present.
        if doc.ctx_map().contains_key("event.dataset") || doc.has("event.dataset") {
            set_kept(doc, "event.dataset", &ds)?;
        }
        Ok(())
    }
}

struct SecurityUserP {
    field: String,
    properties: Option<Vec<String>>,
}

const SECURITY_PROPS: &[&str] = &["username", "roles", "email", "full_name", "metadata", "api_key", "realm", "authentication_type", "profile_uid"];

impl Processor for SecurityUserP {
    fn process(&self, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
        let Some(u) = &cx.user else {
            return Err(EsError::new(500, "illegal_state_exception", "Security (authentication) is not enabled on this cluster, so there is no active user - the [set_security_user] processor cannot be used without security").into());
        };
        let e = u.effective();
        // ES `SetSecurityUserProcessor`: the selected properties are set on
        // the existing target object (a non-object target is replaced);
        // its other members are kept.
        let mut out = match doc.get(&self.field)? {
            Some(Value::Object(m)) => m.clone(),
            _ => Map::new(),
        };
        let want = |k: &str| self.properties.as_ref().map(|p| p.iter().any(|x| x == k)).unwrap_or(true);
        if want("username") {
            out.insert("username".into(), json!(e.username));
        }
        if want("roles") && !e.roles.is_empty() {
            out.insert("roles".into(), json!(e.roles));
        }
        if want("email") {
            if let Some(m) = &e.email {
                out.insert("email".into(), json!(m));
            }
        }
        if want("full_name") {
            if let Some(m) = &e.full_name {
                out.insert("full_name".into(), json!(m));
            }
        }
        if want("metadata") && e.metadata.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            out.insert("metadata".into(), e.metadata.clone());
        }
        if want("api_key") {
            if let crate::security::Principal::ApiKey { id, name, metadata } = &u.principal {
                let mut k = out.get("api_key").and_then(Value::as_object).cloned().unwrap_or_default();
                k.insert("id".into(), json!(id));
                k.insert("name".into(), json!(name));
                if metadata.as_object().is_some_and(|m| !m.is_empty()) {
                    k.insert("metadata".into(), metadata.clone());
                }
                out.insert("api_key".into(), Value::Object(k));
            }
        }
        if want("realm") {
            let mut r = out.get("realm").and_then(Value::as_object).cloned().unwrap_or_default();
            r.insert("name".into(), json!(u.realm_name));
            r.insert("type".into(), json!(u.realm_type));
            out.insert("realm".into(), Value::Object(r));
        }
        if want("authentication_type") {
            out.insert("authentication_type".into(), json!(u.auth_type.to_uppercase()));
        }
        doc.set(&self.field, Value::Object(out))?;
        Ok(())
    }
}

/// ES geo_grid output: `Envelope` GeoJSON or `BBOX (minLon, maxLon, maxLat, minLat)` WKT.
fn envelope(b: (f64, f64, f64, f64), wkt: bool) -> Value {
    let (x0, y0, x1, y1) = b;
    if wkt {
        use super::jtext::java_double as d;
        return Value::String(format!("BBOX ({}, {}, {}, {})", d(x0), d(x1), d(y1), d(y0)));
    }
    json!({"type": "Envelope", "coordinates": [[x0, y1], [x1, y0]]})
}

/// fdlibm `sinh` (Java `StrictMath.sinh`) for the |x| < 22 range tiles use.
#[allow(dead_code)]
fn fdlibm_sinh(x: f64) -> f64 {
    let h = if x < 0.0 { -0.5 } else { 0.5 };
    let ax = x.abs();
    if ax < 22.0 {
        if ax < 2.0f64.powi(-28) {
            return x;
        }
        let t = ax.exp_m1();
        if ax < 1.0 {
            return h * (2.0 * t - t * t / (t + 1.0));
        }
        return h * (t + t / (t + 1.0));
    }
    x.sinh()
}

/// `GeoTileUtils.toBoundingBox` (min lon, min lat, max lon, max lat).
fn geotile_envelope(id: &str) -> Option<(f64, f64, f64, f64)> {
    let p: Vec<i64> = id.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<_>>>()?;
    if p.len() != 3 || p[0] < 0 || p[0] > 29 {
        return None;
    }
    let tiles = (1i64 << p[0]) as f64;
    let (x, y) = (p[1] as f64, p[2] as f64);
    if x < 0.0 || y < 0.0 || x >= tiles || y >= tiles {
        return None;
    }
    // ES FastMath sinh/atan, bit for bit (`fieldtypes/geo.rs`).
    let (top, left, bottom, right) = crate::fieldtypes::geo::tile_bbox(p[0] as u32, x as u32, y as u32);
    Some((left, bottom, right, top))
}

struct GeoGridP {
    field: String,
    target: String,
    tile_type: String,
    parent_field: Option<String>,
    children_field: Option<String>,
    precision_field: Option<String>,
    non_children_field: Option<String>,
    wkt: bool,
    ignore_missing: bool,
}

/// GeoJSON `Polygon` or WKT `POLYGON` for a closed ring of (lon, lat).
fn ring_shape(ring: &[(f64, f64)], wkt: bool) -> Value {
    let mut pts = ring.to_vec();
    if let Some(first) = pts.first().copied() {
        pts.push(first);
    }
    if wkt {
        use super::jtext::java_double as d;
        return Value::String(format!("POLYGON (({}))", pts.iter().map(|(x, y)| format!("{} {}", d(*x), d(*y))).collect::<Vec<_>>().join(", ")));
    }
    json!({"type": "Polygon", "coordinates": [pts.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>()]})
}

/// The geo_grid processor's WKT shape of a geohex address (tests).
#[cfg(test)]
pub(crate) fn test_geohex_wkt(id: &str) -> Option<String> {
    let mut polys = crate::aggs::geo_hex::cartesian_polys(super::geo_h3::parse_cell_id(id).ok()?);
    for r in &mut polys {
        r.pop();
    }
    let v = if polys.len() == 1 { ring_shape(&polys[0], true) } else { rings_shape(&polys, true) };
    v.as_str().map(str::to_string)
}

/// A multipolygon of open rings (each closed here).
fn rings_shape(rings: &[Vec<(f64, f64)>], wkt: bool) -> Value {
    let closed: Vec<Vec<(f64, f64)>> = rings.iter().map(|r| {
        let mut r = r.clone();
        if let Some(f) = r.first().copied() {
            r.push(f);
        }
        r
    }).collect();
    if wkt {
        use super::jtext::java_double as d;
        let parts: Vec<String> = closed.iter().map(|r| format!("(({}))", r.iter().map(|(x, y)| format!("{} {}", d(*x), d(*y))).collect::<Vec<_>>().join(", "))).collect();
        return Value::String(format!("MULTIPOLYGON ({})", parts.join(",")));
    }
    json!({"type": "MultiPolygon", "coordinates": closed.iter().map(|r| json!([r.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>()])).collect::<Vec<_>>()})
}

impl Processor for GeoGridP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        let id = match v {
            Value::String(s) => s,
            _ => return Err(ia(format!("field [{}] must be a String tile address", self.field)).into()),
        };
        if self.tile_type == "geohex" {
            let h = super::geo_h3::hex_info(&id)?;
            // ES (`H3CartesianUtil.getNormalizeGeometry`): the cell on
            // Lucene's grid, polar cells closed through the pole, cells
            // across the dateline split there.
            let mut polys: Vec<Vec<(f64, f64)>> = crate::aggs::geo_hex::cartesian_polys(super::geo_h3::parse_cell_id(&id)?);
            for r in &mut polys {
                r.pop();
            }
            let shape = if polys.len() == 1 { ring_shape(&polys[0], self.wkt) } else { rings_shape(&polys, self.wkt) };
            doc.set(&self.target, shape)?;
            if let Some(f) = &self.precision_field {
                doc.set(f, Value::from(h.precision))?;
            }
            if let (Some(f), Some(p)) = (&self.parent_field, h.parent) {
                doc.set(f, Value::String(p))?;
            }
            if let Some(f) = &self.children_field {
                doc.set(f, Value::Array(h.children.into_iter().map(Value::String).collect()))?;
            }
            if let Some(f) = &self.non_children_field {
                doc.set(f, Value::Array(h.non_children.into_iter().map(Value::String).collect()))?;
            }
            return Ok(());
        }
        let (bbox, precision, parent, children): (_, usize, Option<String>, Vec<String>) = match self.tile_type.as_str() {
            "geotile" => {
                let b = geotile_envelope(&id).ok_or_else(|| ia("invalid tile definition").caused_by(ia(format!("Invalid geotile_grid hash string of {id}. Must be three integers in a form \"zoom/x/y\"."))))?;
                let p: Vec<i64> = id.split('/').filter_map(|x| x.parse().ok()).collect();
                let parent = if p[0] > 0 { Some(format!("{}/{}/{}", p[0] - 1, p[1] / 2, p[2] / 2)) } else { None };
                let children = (0..4).map(|i| format!("{}/{}/{}", p[0] + 1, p[1] * 2 + (i % 2), p[2] * 2 + i / 2)).collect();
                (b, p[0] as usize, parent, children)
            }
            "geohash" => {
                let b = super::util::geohash_bbox(&id).ok_or_else(|| ia("invalid tile definition").caused_by(ia(format!("unsupported symbol in geohash [{id}]"))))?;
                let parent = if id.len() > 1 { Some(id[..id.len() - 1].to_string()) } else { None };
                let children = "0123456789bcdefghjkmnpqrstuvwxyz".chars().map(|c| format!("{id}{c}")).collect();
                (b, id.len(), parent, children)
            }
            other => return Err(ia(format!("tile type [{other}] is not supported by this server")).into()),
        };
        doc.set(&self.target, envelope(bbox, self.wkt))?;
        if let Some(f) = &self.precision_field {
            doc.set(f, Value::from(precision))?;
        }
        if let (Some(f), Some(p)) = (&self.parent_field, parent) {
            doc.set(f, Value::String(p))?;
        }
        if let Some(f) = &self.children_field {
            doc.set(f, Value::Array(children.into_iter().map(Value::String).collect()))?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------ factory

/// Processor types this server knows.
pub const TYPES: &[&str] = &[
    "append", "attachment", "bytes", "circle", "community_id", "convert", "csv", "date", "date_index_name", "dissect", "dot_expander", "drop", "enrich", "fail", "fingerprint", "foreach", "geo_grid", "geoip",
    "grok", "gsub", "html_strip", "inference", "ip_location", "join", "json", "kv", "lowercase", "network_direction", "pipeline", "redact", "registered_domain", "remove", "rename", "reroute", "script", "set",
    "set_security_user", "sort", "split", "terminate", "trim", "uppercase", "uri_parts", "urldecode", "user_agent",
];

fn grok_defs(cfg: &mut Cfg, ecs_v1: bool) -> Result<BTreeMap<String, String>> {
    let mut defs = grok::bank_es(ecs_v1).clone();
    if let Some(m) = cfg.opt_map("pattern_definitions")? {
        for (k, v) in m {
            let s = match v {
                Value::String(s) => s,
                Value::Array(a) => a.iter().map(value_to_string).collect::<Vec<_>>().join("|"),
                other => value_to_string(&other),
            };
            defs.insert(k, s);
        }
    }
    Ok(defs)
}

fn compile_groks(cfg: &Cfg, patterns: &[String], defs: &BTreeMap<String, String>) -> Result<Vec<Grok>> {
    patterns.iter().map(|p| Grok::compile_es(p, defs).map_err(|e| cfg.err("patterns", format!("Invalid regex pattern found in: [{p}]. {}", e.reason)))).collect()
}

fn media_type(cfg: &mut Cfg) -> Result<super::jtext::MediaType> {
    match cfg.opt_str("media_type")? {
        None => Ok(super::jtext::MediaType::Json),
        Some(m) => super::jtext::MediaType::parse(&m).ok_or_else(|| cfg.err("media_type", format!("property does not contain a supported media type [{m}]"))),
    }
}

/// Build a processor from its config (common options already removed).
pub fn build(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let kind = cfg.kind.clone();
    Ok(match kind.as_str() {
        "inference" => super::inference::build(cfg)?,
        "set" => {
            let field = cfg.req_str("field")?;
            let value = cfg.opt_value("value").filter(|v| !v.is_null());
            let copy_from = cfg.opt_str("copy_from")?;
            if value.is_some() && copy_from.is_some() {
                return Err(cfg.err("copy_from", "cannot set both `copy_from` and `value` in the same processor"));
            }
            if value.is_none() && copy_from.is_none() {
                return Err(cfg.err("value", "required property is missing"));
            }
            let overwrite = cfg.bool_or("override", true)?;
            let ignore_empty = cfg.bool_or("ignore_empty_value", false)?;
            let media = media_type(cfg)?;
            Box::new(SetP { field, value, copy_from, overwrite, ignore_empty, media })
        }
        "remove" => {
            let fields = cfg.str_or_list("field")?;
            let keep = cfg.str_or_list("keep")?;
            if fields.is_some() && keep.is_some() {
                return Err(cfg.err("keep", "Too many fields specified. Please specify either [field] or [keep]."));
            }
            if fields.is_none() && keep.is_none() {
                return Err(cfg.err("keep", "At least one of [field] or [keep] must be specified"));
            }
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            Box::new(RemoveP { fields: fields.unwrap_or_default(), keep, ignore_missing })
        }
        "rename" => {
            let field = cfg.req_str("field")?;
            let target = cfg.req_str("target_field")?;
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let overwrite = cfg.bool_or("override", false)?;
            Box::new(RenameP { field, target, ignore_missing, overwrite })
        }
        "append" => {
            let field = cfg.req_str("field")?;
            let value = cfg.req_value("value")?;
            let allow_duplicates = cfg.bool_or("allow_duplicates", true)?;
            let media = media_type(cfg)?;
            Box::new(AppendP { field, value, allow_duplicates, media })
        }
        "lowercase" => str_proc(cfg, Box::new(|s| Ok(s.to_lowercase())))?,
        "uppercase" => str_proc(cfg, Box::new(|s| Ok(s.to_uppercase())))?,
        "trim" => str_proc(cfg, Box::new(|s| Ok(s.trim().to_string())))?,
        "html_strip" => str_proc(cfg, Box::new(|s| Ok(super::util::html_strip(s))))?,
        "urldecode" => str_proc(cfg, Box::new(url_decode))?,
        "gsub" => {
            let field = cfg.req_str("field")?;
            let pattern = cfg.req_str("pattern")?;
            let replacement = cfg.req_str("replacement")?;
            let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let re = java_regex(&pattern).map_err(|e| cfg.err("pattern", e.reason))?;
            let groups = re.captures_len() - 1;
            Box::new(GsubP { field, target, re, replacement: java_replacement(&replacement, groups), ignore_missing })
        }
        "split" => {
            let field = cfg.req_str("field")?;
            let sep = cfg.req_str("separator")?;
            let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let preserve_trailing = cfg.bool_or("preserve_trailing", false)?;
            let re = java_regex(&sep).map_err(|e| cfg.err("separator", e.reason))?;
            Box::new(SplitP { field, target, re, ignore_missing, preserve_trailing })
        }
        "join" => {
            let field = cfg.req_str("field")?;
            let sep = cfg.req_str("separator")?;
            let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
            Box::new(JoinP { field, target, sep })
        }
        "sort" => {
            let field = cfg.req_str("field")?;
            let order = cfg.str_or("order", "asc")?;
            if order != "asc" && order != "desc" {
                return Err(cfg.err("order", format!("Sort direction [{order}] not recognized. Valid values are: [asc, desc]")));
            }
            let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
            Box::new(super::p_data::SortP { field, target, desc: order == "desc" })
        }
        "json" => {
            let field = cfg.req_str("field")?;
            let target = cfg.opt_str("target_field")?;
            let add_to_root = cfg.bool_or("add_to_root", false)?;
            let strategy = cfg.str_or("add_to_root_conflict_strategy", "replace")?;
            let allow_dup = cfg.bool_or("allow_duplicate_keys", false)?;
            let strict = cfg.bool_or("strict_json_parsing", true)?;
            if add_to_root && target.is_some() {
                return Err(cfg.err("target_field", "Cannot set a target field while also setting `add_to_root` to true"));
            }
            let merge = match strategy.as_str() {
                "replace" => false,
                "merge" => true,
                other => return Err(cfg.err("add_to_root_conflict_strategy", format!("conflict strategy [{other}] not supported, cannot convert field."))),
            };
            Box::new(super::p_data::JsonP { field, target, add_to_root, merge, strict, allow_dup })
        }
        "bytes" => super::p_data::build_bytes(cfg)?,
        "convert" => super::p_data::build_convert(cfg)?,
        "csv" => super::p_data::build_csv(cfg)?,
        "fingerprint" => super::p_data::build_fingerprint(cfg)?,
        "uri_parts" => super::net::build_uri_parts(cfg)?,
        "registered_domain" => super::net::build_registered_domain(cfg)?,
        "network_direction" => super::net::build_network_direction(cfg)?,
        "community_id" => super::net::build_community_id(cfg)?,
        "date" => {
            let field = cfg.req_str("field")?;
            let target = cfg.str_or("target_field", "@timestamp")?;
            let formats = cfg.req_list_str("formats")?;
            let timezone = cfg.opt_str("timezone")?;
            let locale = cfg.opt_str("locale")?;
            let output_format = cfg.str_or("output_format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX")?;
            Box::new(DateP { field, target, formats: parse_date_formats(&formats), timezone, output_format, locale })
        }
        "date_index_name" => {
            let field = cfg.req_str("field")?;
            let prefix = cfg.str_or("index_name_prefix", "")?;
            let rounding = cfg.req_str("date_rounding")?;
            let formats = cfg.opt_list_str("date_formats")?.unwrap_or_else(|| vec!["yyyy-MM-dd'T'HH:mm:ss.SSSXX".into()]);
            let timezone = cfg.str_or("timezone", "UTC")?;
            let locale = cfg.opt_str("locale")?;
            let index_format = cfg.str_or("index_name_format", "yyyy-MM-dd")?;
            Box::new(DateIndexNameP { field, prefix, rounding, formats: parse_date_formats(&formats), timezone, index_format, locale })
        }
        "grok" => {
            let field = cfg.req_str("field")?;
            let patterns = cfg.req_list_str("patterns")?;
            if patterns.is_empty() {
                return Err(cfg.err("patterns", "List of patterns must not be empty"));
            }
            let ecs_v1 = match cfg.opt_str("ecs_compatibility")?.as_deref() {
                None | Some("disabled") => false,
                Some("v1") => true,
                Some(m) => return Err(cfg.err("ecs_compatibility", format!("unsupported mode '{m}'"))),
            };
            let defs = grok_defs(cfg, ecs_v1)?;
            let trace_match = cfg.bool_or("trace_match", false)?;
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let groks = compile_groks(cfg, &patterns, &defs)?;
            Box::new(GrokP { field, groks, trace_match, ignore_missing })
        }
        "redact" => {
            let field = cfg.req_str("field")?;
            let patterns = cfg.req_list_str("patterns")?;
            let defs = grok_defs(cfg, false)?;
            let prefix = cfg.str_or("prefix", "<")?;
            let suffix = cfg.str_or("suffix", ">")?;
            let ignore_missing = cfg.bool_or("ignore_missing", true)?;
            let trace_redact = cfg.bool_or("trace_redact", false)?;
            cfg.ignore(&["skip_if_unlicensed"]);
            let groks = compile_groks(cfg, &patterns, &defs)?;
            Box::new(RedactP { field, groks, prefix, suffix, ignore_missing, trace_redact })
        }
        "dissect" => {
            let field = cfg.req_str("field")?;
            let pattern = cfg.req_str("pattern")?;
            let append_separator = cfg.str_or("append_separator", "")?;
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let (prefix, keys) = parse_dissect(&pattern).map_err(|e| super::with_processor_meta(e, "dissect", cfg.tag.as_deref()))?;
            Box::new(DissectP { field, prefix, keys, append_separator, ignore_missing, pattern })
        }
        "kv" => {
            let field = cfg.req_str("field")?;
            let fs = cfg.req_str("field_split")?;
            let vs = cfg.req_str("value_split")?;
            let target = cfg.opt_str("target_field")?;
            let include = cfg.opt_list_str("include_keys")?;
            let exclude = cfg.opt_list_str("exclude_keys")?;
            let prefix = cfg.str_or("prefix", "")?;
            let trim_key = cfg.opt_str("trim_key")?;
            let trim_value = cfg.opt_str("trim_value")?;
            let strip_brackets = cfg.bool_or("strip_brackets", false)?;
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let field_split = java_regex(&fs).map_err(|e| cfg.err("field_split", e.reason))?;
            let value_split = java_regex(&vs).map_err(|e| cfg.err("value_split", e.reason))?;
            Box::new(super::p_data::KvP { field, field_split, value_split, value_split_src: vs, target, include, exclude, prefix, trim_key, trim_value, strip_brackets, ignore_missing })
        }
        "dot_expander" => {
            let field = cfg.req_str("field")?;
            let path = cfg.opt_str("path")?;
            let overwrite = cfg.bool_or("override", false)?;
            if field != "*" && !field.contains('.') {
                return Err(cfg.err("field", "field does not contain a dot and is not a wildcard"));
            }
            Box::new(DotExpanderP { field, path, overwrite })
        }
        "drop" => Box::new(DropP),
        "terminate" => Box::new(TerminateP),
        "fail" => Box::new(FailP { message: cfg.req_str("message")? }),
        "foreach" => {
            let field = cfg.req_str("field")?;
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let p = cfg.req_value("processor")?;
            let po = p.as_object().ok_or_else(|| cfg.err("processor", "property isn't a map"))?;
            if po.len() != 1 {
                return Err(cfg.err("processor", "Must specify exactly one processor type"));
            }
            let inner = super::compile_processor_source(&p, cfg.raw.as_ref().and_then(|r|r.get("processor")))?;
            Box::new(ForeachP { field, inner: Box::new(inner), ignore_missing })
        }
        "script" => {
            let mut m = Map::new();
            for k in ["source", "inline", "id", "lang", "params"] {
                if let Some(v) = cfg.opt_value(k) {
                    m.insert(k.into(), v);
                }
            }
            if m.contains_key("id") && m.contains_key("source") {
                return Err(config_err(cfg, "Only one of [id] or [source] may be configured"));
            }
            if !m.contains_key("id") && !m.contains_key("source") && !m.contains_key("inline") {
                return Err(config_err(cfg, "Need [id] or [source] parameter to refer to scripts"));
            }
            let script = match &cfg.raw {
                Some(raw) => Script::from_source(&super::source_config::script_spec(raw)),
                None => Script::from_config(&Value::Object(m)),
            }.map_err(|e| super::with_processor_meta(e, "script", cfg.tag.as_deref()))?;
            script.validate(crate::script::ScriptContext::Ingest).map_err(|e| super::with_processor_meta(e, "script", cfg.tag.as_deref()))?;
            Box::new(ScriptP { script })
        }
        "pipeline" => {
            let name = cfg.req_str("name")?;
            let ignore_missing_pipeline = cfg.bool_or("ignore_missing_pipeline", false)?;
            Box::new(PipelineP { name, ignore_missing_pipeline })
        }
        "geoip" | "ip_location" => return super::geoip::build(cfg, &kind),
        "user_agent" => return super::useragent::build(cfg),
        "attachment" => return super::attachment::build(cfg),
        "reroute" => {
            let destination = cfg.opt_str("destination")?;
            let ty = cfg.str_or_list("type")?.unwrap_or_default();
            let dataset = cfg.str_or_list("dataset")?.unwrap_or_default();
            let namespace = cfg.str_or_list("namespace")?.unwrap_or_default();
            if destination.is_some() && (!ty.is_empty() || !dataset.is_empty() || !namespace.is_empty()) {
                return Err(cfg.err("destination", "can only be set if type, dataset, and namespace are not set"));
            }
            for (name, vals, is_ns) in [("type", &ty, false), ("dataset", &dataset, false), ("namespace", &namespace, true)] {
                for v in vals {
                    if !is_template(v) && !valid_static(v, is_ns) {
                        return Err(cfg.err(name, format!("'{v}' contains disallowed characters")));
                    }
                }
            }
            Box::new(RerouteP { destination, ty, dataset, namespace })
        }
        "set_security_user" => {
            let field = cfg.req_str("field")?;
            let properties = cfg.opt_list_str("properties")?;
            if let Some(ps) = &properties {
                for p in ps {
                    if !SECURITY_PROPS.contains(&p.as_str()) {
                        return Err(cfg.err("properties", format!("Property value [{p}] is in valid")));
                    }
                }
            }
            Box::new(SecurityUserP { field, properties })
        }
        "geo_grid" => {
            let field = cfg.req_str("field")?;
            let tile_type = cfg.req_str("tile_type")?.to_lowercase();
            if !matches!(tile_type.as_str(), "geotile" | "geohash" | "geohex") {
                return Err(cfg.err("tile_type", format!("illegal value [{}], valid values are [GEOHASH, GEOTILE, GEOHEX]", tile_type)));
            }
            let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
            let parent_field = cfg.opt_str("parent_field")?;
            let children_field = cfg.opt_str("children_field")?;
            let precision_field = cfg.opt_str("precision_field")?;
            let non_children_field = cfg.opt_str("non_children_field")?;
            let fmt = cfg.str_or("target_format", "GeoJSON")?;
            let wkt = match fmt.to_uppercase().as_str() {
                "WKT" => true,
                "GEOJSON" => false,
                _ => return Err(cfg.err("target_format", format!("illegal value [{fmt}], valid values are [WKT, GEOJSON]"))),
            };
            let ignore_missing = cfg.bool_or("ignore_missing", false)?;
            Box::new(GeoGridP { field, target, tile_type, parent_field, children_field, precision_field, non_children_field, wkt, ignore_missing })
        }
        "enrich" => return crate::enrich::processor::build(cfg),
        "inference" => {
            let model_id = cfg.req_str("model_id")?;
            let _target = cfg.opt_str("target_field")?;
            let _field_map = cfg.opt_map("field_map")?;
            let _inference_config = cfg.opt_map("inference_config")?;
            let _input_output = cfg.opt_value("input_output");
            let _ignore_missing = cfg.bool_or("ignore_missing", false)?;
            let _ignore_missing_inf = cfg.bool_or("ignore_missing_inference", false).ok();
            Box::new(InferenceP { model_id })
        }
        "circle" => super::geo_circle::build_circle(cfg)?,
        other => {
            return Err(super::config_error(other, cfg.tag.as_deref(), None, format!("No processor type exists with name [{other}]")));
        }
    })
}

fn config_err(cfg: &Cfg, msg: &str) -> EsError {
    super::config_error(&cfg.kind, cfg.tag.as_deref(), None, msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_one(kind: &str, cfg: Value, doc: Value) -> std::result::Result<IngestDocument, String> {
        let mut c = Cfg { kind: kind.into(), tag: None, m: cfg.as_object().unwrap().clone(), raw: None };
        let p = build(&mut c).map_err(|e| e.reason)?;
        let store = super::super::IngestService::in_memory();
        let cx = PipelineContext::new(&store);
        let mut d = IngestDocument::new(doc.as_object().unwrap().clone(), "i", None, None, "now".into());
        match p.process(&mut d, &cx) {
            Ok(()) => Ok(d),
            Err(e) => Err(format!("{e:?}")),
        }
    }

    /// Dates outside the 64-bit nanosecond range still parse in the date
    /// processor (it only formats them).
    #[test]
    fn far_dates_parse() {
        for (input, want) in [("9999-12-31T00:00:00Z", "9999-12-31T00:00:00.000Z"), ("1500-01-01T00:00:00Z", "1500-01-01T00:00:00.000Z")] {
            let d = run_one("date", json!({"field": "t", "formats": ["ISO8601"]}), json!({"t": input})).unwrap();
            assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!(want), "{input}");
        }
    }

    /// Finding 243: an explicit nanosecond output_format keeps the
    /// sub-millisecond digits; the default output keeps milliseconds.
    #[test]
    fn date_keeps_requested_nanoseconds() {
        let nanos = json!({"field": "t", "formats": ["ISO8601"], "output_format": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSSXXX"});
        for input in ["2026-11-09T12:00:00.123456789Z", "2026-11-09T12:00:00.123456788Z"] {
            let d = run_one("date", nanos.clone(), json!({"t": input})).unwrap();
            assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!(input));
        }
        let d = run_one("date", json!({"field": "t", "formats": ["ISO8601"]}), json!({"t": "2026-11-09T12:00:00.123456789Z"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2026-11-09T12:00:00.123Z"));
        let d = run_one("date", json!({"field": "t", "formats": ["yyyy-MM-dd HH:mm:ss.SSSSSSSSS"], "output_format": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSSXXX"}), json!({"t": "2026-11-09 12:00:00.000000007"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2026-11-09T12:00:00.000000007Z"));
    }

    /// Finding 252: a yearless pattern takes the current UTC year (ES ingest
    /// `DateFormat`), for date and date_index_name; explicit years stay.
    #[test]
    fn yearless_dates_take_the_current_year() {
        INGEST_YEAR.with(|y| y.set(Some(2026)));
        let d = run_one("date", json!({"field": "t", "formats": ["MMM dd HH:mm:ss"]}), json!({"t": "Nov 09 12:34:56"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2026-11-09T12:34:56.000Z"));
        let d = run_one("date", json!({"field": "t", "formats": ["yyyy MMM dd HH:mm:ss"]}), json!({"t": "2019 Nov 09 12:34:56"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2019-11-09T12:34:56.000Z"));
        // February 29th in a common year does not exist.
        assert!(run_one("date", json!({"field": "t", "formats": ["MMM dd HH:mm:ss"]}), json!({"t": "Feb 29 00:00:00"})).is_err());
        INGEST_YEAR.with(|y| y.set(Some(2024)));
        assert!(run_one("date", json!({"field": "t", "formats": ["MMM dd HH:mm:ss"]}), json!({"t": "Feb 29 00:00:00"})).is_ok());
        INGEST_YEAR.with(|y| y.set(Some(2026)));
        let d = run_one("date_index_name", json!({"field": "t", "date_formats": ["MMM dd HH:mm:ss"], "date_rounding": "d", "index_name_prefix": "logs-", "index_name_format": "yyyy-MM-dd"}), json!({"t": "Nov 09 12:34:56"})).unwrap();
        let index = d.get("_index").unwrap().unwrap().as_str().unwrap().to_string();
        assert_eq!(resolve_index_expression(&index).unwrap(), "logs-2026-11-09");
        INGEST_YEAR.with(|y| y.set(None));
    }

    /// Finding 251: a wall time in a DST gap moves later by the gap (java.time):
    /// a valid Sao Paulo midday routes to its own day, and a Paris spring
    /// 02:30 is 03:30+02:00.
    #[test]
    fn dst_gaps_resolve_forward() {
        let d = run_one("date_index_name", json!({"field": "t", "date_formats": ["ISO8601"], "timezone": "America/Sao_Paulo", "date_rounding": "d", "index_name_prefix": "logs-", "index_name_format": "yyyy-MM-dd"}), json!({"t": "2018-11-04T12:00:00-02:00"})).unwrap();
        let index = d.get("_index").unwrap().unwrap().as_str().unwrap().to_string();
        assert_eq!(resolve_index_expression(&index).unwrap(), "logs-2018-11-04");
        let d = run_one("date", json!({"field": "t", "formats": ["yyyy-MM-dd HH:mm:ss"], "timezone": "Europe/Paris"}), json!({"t": "2024-03-31 02:30:00"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2024-03-31T03:30:00.000+02:00"));
        // An ordinary time and an autumn overlap (earlier offset) are unchanged.
        let d = run_one("date", json!({"field": "t", "formats": ["yyyy-MM-dd HH:mm:ss"], "timezone": "Europe/Paris"}), json!({"t": "2024-10-27 02:30:00"})).unwrap();
        assert_eq!(d.get("@timestamp").unwrap().unwrap(), &json!("2024-10-27T02:30:00.000+02:00"));
    }

    /// Finding 246: set_security_user sets the selected properties on the
    /// existing target object and keeps its other members (also inside
    /// `realm`); an absent or non-object target is created.
    #[test]
    fn set_security_user_merges_into_the_target() {
        let store = super::super::IngestService::in_memory();
        let user = crate::security::Security::system();
        let cx = PipelineContext { user: Some(user.clone()), ..PipelineContext::new(&store) };
        let p = SecurityUserP { field: "user".into(), properties: Some(vec!["username".into(), "realm".into()]) };
        let mut doc = IngestDocument::new(json!({"user": {"id": "u1", "domain": "corp", "username": "old", "realm": {"note": "kept"}}}).as_object().unwrap().clone(), "i", None, None, "now".into());
        p.process(&mut doc, &cx).unwrap();
        let u = doc.get("user").unwrap().unwrap().clone();
        assert_eq!((&u["id"], &u["domain"], &u["realm"]["note"]), (&json!("u1"), &json!("corp"), &json!("kept")), "{u}");
        assert_eq!(u["username"], json!(user.effective().username));
        assert_eq!(u["realm"]["name"], json!(user.realm_name));
        for start in [json!({}), json!({"user": "scalar"})] {
            let mut doc = IngestDocument::new(start.as_object().unwrap().clone(), "i", None, None, "now".into());
            p.process(&mut doc, &cx).unwrap();
            assert_eq!(doc.get("user.username").unwrap().unwrap(), &json!(user.effective().username));
        }
    }

    /// Localized month and weekday names come from the JDK's (CLDR) locale
    /// data for every locale, with regional variants and genitive forms
    /// (findings 461/462).
    #[test]
    fn localized_names_cover_cldr_locales() {
        let l = |s: &str, loc: &str| localize_input(s, loc, false).unwrap();
        // `a`: the locale's day periods.
        let pm = crate::locale_names::am_pm("es").unwrap()[1];
        assert_eq!(localize_input(&format!("02-marzo-2025 3:15 {pm}"), "es", true).unwrap(), "02-March-2025 3:15 PM");
        let am = crate::locale_names::am_pm("de").unwrap()[0];
        assert_eq!(localize_input(&format!("2 März 2025 9:00 {am}"), "de", true).unwrap(), "2 March 2025 9:00 AM");
        assert_eq!(l("02-marzo-2025", "es"), "02-March-2025");
        assert_eq!(l("2 марта 2025", "ru"), "2 March 2025");
        assert_eq!(l("2 de março de 2025", "pt_BR"), "2 de March de 2025");
        assert_eq!(l("lundi 3 février", "fr_CA"), "Monday 3 February");
        assert_eq!(l("3. Mai 2025", "de-CH"), "3. May 2025");
        assert_eq!(l("2025年3月2日", "zh"), "2025年March2日");
        assert_eq!(l("2025年三月2日", "zh"), "2025年March2日");
        assert_eq!(l("2 maart 2025", "nl"), "2 March 2025");
        assert_eq!(l("2 marca 2025", "pl"), "2 March 2025");
        assert_eq!(l("2 Mart 2025", "tr"), "2 March 2025");
        assert_eq!(l("2 ottobre 2025", "it_IT"), "2 October 2025");
        // Legacy Java codes resolve to their CLDR locale.
        assert_eq!(l("2 מרץ 2025", "iw"), "2 March 2025");
        // An unknown locale leaves the text alone.
        assert_eq!(l("2 x 2025", "xx"), "2 x 2025");
    }

    #[test]
    fn helpers() {
        assert_eq!(java_replacement("$1-${x}\\$", 1).unwrap(), "${1}-${x}$$");
        // Finding 250: digits extend a group reference only while that
        // group exists (Java `appendReplacement`).
        assert_eq!(java_replacement("id-$10", 1).unwrap(), "id-${1}0");
        assert_eq!(java_replacement("$21", 2).unwrap(), "${2}1");
        assert_eq!(java_replacement("$10", 10).unwrap(), "${10}");
        assert_eq!(java_replacement("$2", 1).unwrap_err(), "No group 2");
        assert_eq!(java_replacement("a$b", 1).unwrap_err(), "Illegal group reference");
        assert_eq!(java_replacement("\\$5 off", 0).unwrap(), "$$5 off");
        let re = java_regex("^(.+)$").unwrap();
        let rep = java_replacement("id-$10", re.captures_len() - 1).unwrap();
        assert_eq!((re.replace_all("a", rep.as_str()), re.replace_all("b", rep.as_str())), ("id-a0".into(), "id-b0".into()));
        assert_eq!(url_decode("a%20b+c").unwrap(), "a b c");
        let (pre, keys) = parse_dissect("%{a} %{+a} - %{?skip} [%{ts}]").unwrap();
        assert_eq!(pre, "");
        assert_eq!(keys.len(), 4);
        assert_eq!(resolve_index_expression("<my-index-{2016-04-25||/M{yyyy-MM-dd|UTC}}>").unwrap(), "my-index-2016-04-01");
        let m = crate::ingest::net::uri_parts_map("https://user:pw@example.com:8443/a/b.php?x=1#frag", true).unwrap();
        assert_eq!(m["domain"], "example.com");
        assert_eq!(m["port"], 8443);
        assert_eq!(m["extension"], "php");
        assert_eq!(m["username"], "user");
    }
}
