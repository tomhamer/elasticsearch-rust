//! Anomaly detection job configs: parsing and validation of
//! `PUT _ml/anomaly_detectors/{id}` and `_update` bodies with ES's messages
//! (`Job`, `AnalysisConfig`, `Detector`, `DetectionRule`, `AnalysisLimits`,
//! `DataDescription`, `ModelPlotConfig`, `JobUpdate`), and rendering in ES's
//! field order.

use super::xc::{self, bad_request, fmt, unknown, wrap, xparse, Loc, Pos};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

/// `MlConfigVersion.CURRENT` (the `job_version` of new jobs).
pub const CONFIG_VERSION: &str = "12.0.0";
pub const DEFAULT_MML_MB: u64 = 1024;
const MB: u64 = 1 << 20;

/// A parse failure at a body location (the Jackson location ES reports).
pub struct PErr {
    pub e: EsError,
    pub at: Loc,
}
pub type PR<T> = std::result::Result<T, PErr>;

fn at(e: EsError, at: Loc) -> PErr {
    PErr { e, at }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

const FUNCTIONS: &[&str] = &[
    "count", "low_count", "high_count", "non_zero_count", "low_non_zero_count", "high_non_zero_count", "distinct_count", "low_distinct_count", "high_distinct_count", "rare", "freq_rare",
    "info_content", "low_info_content", "high_info_content", "metric", "mean", "low_mean", "high_mean", "avg", "low_avg", "high_avg", "median", "low_median", "high_median", "min", "max", "sum",
    "low_sum", "high_sum", "non_null_sum", "low_non_null_sum", "high_non_null_sum", "varp", "low_varp", "high_varp", "time_of_day", "time_of_week", "lat_long",
];
const COUNT_WITHOUT_FIELD: &[&str] = &["count", "high_count", "low_count", "non_zero_count", "low_non_zero_count", "high_non_zero_count", "time_of_day", "time_of_week"];
const BY_FIELD_FUNCTIONS: &[&str] = &["rare", "freq_rare"];
const OVER_FIELD_FUNCTIONS: &[&str] = &["freq_rare"];
const NO_OVER_FIELD_FUNCTIONS: &[&str] = &["non_zero_count", "low_non_zero_count", "high_non_zero_count"];
const NO_RULE_CONDITION_FUNCTIONS: &[&str] = &["lat_long", "metric", "rare", "freq_rare", "info_content", "low_info_content", "high_info_content", "time_of_day", "time_of_week"];

/// `DetectorFunction.fromString` (shortcuts expanded).
pub fn canonical_function(f: &str) -> Option<&'static str> {
    let full = match f {
        "nzc" => "non_zero_count",
        "low_nzc" => "low_non_zero_count",
        "high_nzc" => "high_non_zero_count",
        "dc" => "distinct_count",
        "low_dc" => "low_distinct_count",
        "high_dc" => "high_distinct_count",
        other => other,
    };
    FUNCTIONS.iter().copied().find(|x| *x == full)
}

fn is_field_function(f: &str) -> bool {
    !COUNT_WITHOUT_FIELD.contains(&f) && !matches!(f, "rare" | "freq_rare")
}

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------

fn obj<'a>(v: &'a Value, p: &Pos, parser: &str) -> PR<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| at(xparse(format!("{} [{parser}] Expected START_OBJECT but was: {}", fmt(p.start), xc::token_name(v))), p.start))
}

fn string(v: &Value, p: &Pos, parser: &str, field: &str) -> PR<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(at(xparse(format!("{} [{parser}] {field} doesn't support values of type: {}", fmt(p.start), xc::token_name(other))), p.start)),
    }
}

fn boolean(v: &Value, p: &Pos, parser: &str, field: &str) -> PR<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" || s == "false" => Ok(s == "true"),
        Value::String(s) => Err(at(wrap(p.start, parser, field, EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed."))), p.start)),
        other => Err(at(xparse(format!("{} [{parser}] {field} doesn't support values of type: {}", fmt(p.start), xc::token_name(other))), p.start)),
    }
}

fn long(v: &Value, p: &Pos, parser: &str, field: &str) -> PR<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).ok_or_else(|| at(wrap(p.start, parser, field, EsError::illegal_argument(format!("Value [{n}] is out of range for a long"))), p.start)),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| at(wrap(p.start, parser, field, EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))), p.start)),
        other => Err(at(xparse(format!("{} [{parser}] {field} doesn't support values of type: {}", fmt(p.start), xc::token_name(other))), p.start)),
    }
}

fn string_list(v: &Value, p: &Pos, parser: &str, field: &str) -> PR<Vec<String>> {
    match v {
        Value::Array(a) => a.iter().enumerate().map(|(i, x)| string(x, p.at(i), parser, field)).collect(),
        Value::String(s) => Ok(vec![s.clone()]),
        other => Err(at(xparse(format!("{} [{parser}] {field} doesn't support values of type: {}", fmt(p.start), xc::token_name(other))), p.start)),
    }
}

/// `TimeValue.parseTimeValue(s, field)` as an ObjectParser field.
fn time_value(v: &Value, p: &Pos, parser: &str, field: &str) -> PR<TimeV> {
    let s = string(v, p, parser, field)?;
    parse_time(&s, field).map_err(|e| at(wrap(p.start, parser, field, e), p.start))
}

/// A parsed `TimeValue`: nanos plus ES's string representation.
#[derive(Clone, Debug)]
pub struct TimeV {
    pub nanos: i128,
    pub rep: String,
}

impl TimeV {
    pub fn seconds(&self) -> i64 {
        (self.nanos / 1_000_000_000) as i64
    }
    pub fn millis(&self) -> i64 {
        (self.nanos / 1_000_000) as i64
    }
}

/// `TimeValue.parseTimeValue` (errors as `illegal_argument_exception`).
pub fn parse_time(s: &str, field: &str) -> Result<TimeV> {
    match crate::ilm::parse_time(s, field) {
        Ok(t) => Ok(TimeV { nanos: t.nanos, rep: t.rep }),
        Err(e) => Err(EsError::illegal_argument(e.reason)),
    }
}

/// `TimeValue.toString()`: the largest unit, one fractional digit
/// (truncated), as ES prints `1.5d` / `19m`.
pub fn time_to_string(nanos: i128) -> String {
    let units: [(&str, i128); 7] = [("d", 86_400_000_000_000), ("h", 3_600_000_000_000), ("m", 60_000_000_000), ("s", 1_000_000_000), ("ms", 1_000_000), ("micros", 1_000), ("nanos", 1)];
    for (u, m) in units {
        if nanos >= m {
            let whole = nanos / m;
            let frac = (nanos % m) * 10 / m;
            return if frac == 0 { format!("{whole}{u}") } else { format!("{whole}.{frac}{u}") };
        }
    }
    format!("{nanos}nanos")
}

/// A duration in the largest unit that represents it exactly (`30d`,
/// `43214m`).
pub fn exact_time_rep(secs: i64) -> String {
    for (u, m) in [("d", 86_400i64), ("h", 3_600), ("m", 60)] {
        if secs % m == 0 && secs != 0 {
            return format!("{}{u}", secs / m);
        }
    }
    format!("{secs}s")
}

/// `ByteSizeValue.parseBytesSizeValue` (bytes).
pub fn parse_bytes(s: &str, field: &str) -> Result<u64> {
    let lower = s.trim().to_ascii_lowercase();
    let bad = || EsError::new(400, "parse_exception", format!("failed to parse setting [{field}] with value [{s}] as a size in bytes: unit is missing or unrecognized"));
    if lower == "0" || lower == "-1" {
        return Ok(0);
    }
    let units: [(&str, u64); 11] = [("pb", 1 << 50), ("p", 1 << 50), ("tb", 1 << 40), ("t", 1 << 40), ("gb", 1 << 30), ("g", 1 << 30), ("mb", 1 << 20), ("m", 1 << 20), ("kb", 1 << 10), ("k", 1 << 10), ("b", 1)];
    for (u, m) in units {
        if let Some(n) = lower.strip_suffix(u) {
            let n = n.trim();
            if let Ok(i) = n.parse::<u64>() {
                return Ok(i * m);
            }
            let f: f64 = n.parse().map_err(|_| bad())?;
            if f < 0.0 {
                return Err(bad());
            }
            return Ok((f * m as f64) as u64);
        }
    }
    Err(bad())
}

/// `ByteSizeValue.toString()` (`9gb`, `1.5gb`, `512mb`).
pub fn bytes_to_string(b: u64) -> String {
    let units: [(&str, u64); 6] = [("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)];
    for (u, m) in units {
        if b >= m {
            let v = b as f64 / m as f64;
            let r = (v * 10.0).round() / 10.0;
            return if r.fract() == 0.0 { format!("{}{u}", r as u64) } else { format!("{r:.1}{u}") };
        }
    }
    "0b".into()
}

/// `MlStrings.doubleQuoteIfNotAlphaNumeric`.
fn quote_field(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        s.to_string()
    } else {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

// ---------------------------------------------------------------------------
// Detection rules
// ---------------------------------------------------------------------------

fn parse_rule(v: &Value, p: &Pos) -> PR<Value> {
    const P: &str = "detection_rule";
    let o = obj(v, p, P)?;
    let mut actions: Option<Vec<String>> = None;
    let mut scope = Map::new();
    let mut conditions: Vec<Value> = Vec::new();
    let mut params: Option<Value> = None;
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "actions" => {
                let list = string_list(val, vp, P, k)?;
                for a in &list {
                    if !matches!(a.as_str(), "skip_result" | "skip_model_update" | "force_time_shift") {
                        return Err(at(wrap(vp.start, P, k, EsError::illegal_argument(format!("Unknown rule action: {a}"))), vp.start));
                    }
                }
                actions = Some(list);
            }
            "scope" => {
                let so = obj(val, vp, "rule_scope")?;
                for (field, fv) in so {
                    let fp = vp.get(field);
                    let fo = obj(fv, fp, "filter_ref")?;
                    let mut filter_id = None;
                    let mut filter_type = "include".to_string();
                    for (fk, x) in fo {
                        match fk.as_str() {
                            "filter_id" => filter_id = Some(string(x, fp.get(fk), "filter_ref", fk)?),
                            "filter_type" => {
                                let t = string(x, fp.get(fk), "filter_ref", fk)?.to_ascii_lowercase();
                                if !matches!(t.as_str(), "include" | "exclude") {
                                    return Err(at(wrap(fp.get(fk).start, "filter_ref", fk, EsError::illegal_argument(format!("Unknown filter_type: {t}"))), fp.get(fk).start));
                                }
                                filter_type = t;
                            }
                            other => return Err(at(unknown(fp.key(other), "filter_ref", other, &["filter_id", "filter_type"]), fp.key(other))),
                        }
                    }
                    let Some(fid) = filter_id else {
                        return Err(at(EsError::illegal_argument("Required [filter_id]"), fp.end));
                    };
                    scope.insert(field.clone(), json!({"filter_id": fid, "filter_type": filter_type}));
                }
            }
            "conditions" => {
                let arr = val.as_array().cloned().unwrap_or_default();
                for (i, c) in arr.iter().enumerate() {
                    let cp = vp.at(i);
                    let co = obj(c, cp, "rule_condition")?;
                    let mut out = Map::new();
                    let (mut applies, mut op, mut value) = (None, None, None);
                    for (ck, cv) in co {
                        let ccp = cp.get(ck);
                        match ck.as_str() {
                            "applies_to" => {
                                let s = string(cv, ccp, "rule_condition", ck)?.to_ascii_lowercase();
                                if !matches!(s.as_str(), "actual" | "typical" | "diff_from_typical" | "time") {
                                    return Err(at(wrap(ccp.start, "rule_condition", ck, EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.ml.job.config.RuleCondition.AppliesTo.{}", s.to_ascii_uppercase()))), ccp.start));
                                }
                                applies = Some(s);
                            }
                            "operator" => {
                                let s = string(cv, ccp, "rule_condition", ck)?.to_ascii_lowercase();
                                if !matches!(s.as_str(), "gt" | "gte" | "lt" | "lte") {
                                    return Err(at(wrap(ccp.start, "rule_condition", ck, EsError::illegal_argument(format!("Unknown operator [{s}]"))), ccp.start));
                                }
                                op = Some(s);
                            }
                            "value" => value = Some(cv.clone()),
                            other => return Err(at(unknown(cp.key(other), "rule_condition", other, &["applies_to", "operator", "value"]), cp.key(other))),
                        }
                    }
                    let missing: Vec<&str> = [("applies_to", applies.is_none()), ("operator", op.is_none()), ("value", value.is_none())].iter().filter(|x| x.1).map(|x| x.0).collect();
                    if !missing.is_empty() {
                        return Err(at(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))), cp.end));
                    }
                    out.insert("applies_to".into(), json!(applies));
                    out.insert("operator".into(), json!(op));
                    out.insert("value".into(), value.unwrap_or(Value::Null));
                    conditions.push(Value::Object(out));
                }
            }
            "params" => params = Some(val.clone()),
            other => return Err(at(unknown(p.key(other), P, other, &["actions", "scope", "conditions", "params"]), p.key(other))),
        }
    }
    if scope.is_empty() && conditions.is_empty() {
        return Err(at(bad_request("Invalid detector rule: at least scope or a condition is required"), p.end));
    }
    let mut r = Map::new();
    r.insert("actions".into(), json!(actions.unwrap_or_else(|| vec!["skip_result".into()])));
    if !scope.is_empty() {
        r.insert("scope".into(), Value::Object(scope));
    }
    if !conditions.is_empty() {
        r.insert("conditions".into(), Value::Array(conditions));
    }
    if let Some(pv) = params {
        r.insert("params".into(), pv);
    }
    Ok(Value::Object(r))
}

/// Rules checks that need the detector (`Detector.Builder.build`).
fn check_rules(function: &str, d: &Map<String, Value>, rules: &[Value]) -> Result<()> {
    if NO_RULE_CONDITION_FUNCTIONS.contains(&function) {
        for r in rules {
            if r["conditions"].as_array().map(|c| c.iter().any(|x| x["applies_to"] != "time")).unwrap_or(false) {
                return Err(bad_request(format!("Invalid detector rule: function {function} only supports conditions that apply to time")));
            }
        }
    }
    let mut fields: Vec<String> = ["by_field_name", "over_field_name", "partition_field_name"].iter().filter_map(|k| d.get(*k).and_then(Value::as_str).map(str::to_string)).collect();
    fields.sort();
    for r in rules {
        if let Some(s) = r["scope"].as_object() {
            for f in s.keys() {
                if !fields.contains(f) {
                    if fields.is_empty() {
                        return Err(bad_request(format!("Invalid detector rule: scope field '{f}' is invalid; detector has no available fields for scoping")));
                    }
                    return Err(bad_request(format!("Invalid detector rule: scope field '{f}' is invalid; select from [{}]", fields.join(", "))));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Detectors
// ---------------------------------------------------------------------------

const DETECTOR_FIELDS: &[&str] = &["detector_description", "function", "field_name", "by_field_name", "over_field_name", "partition_field_name", "use_null", "exclude_frequent", "custom_rules", "detector_index"];

/// Parse + validate one detector (the `detector_index` is set by the caller).
fn parse_detector(v: &Value, p: &Pos) -> PR<Map<String, Value>> {
    const P: &str = "detector";
    let o = obj(v, p, P)?;
    let mut d = Map::new();
    let mut rules: Vec<Value> = Vec::new();
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "detector_description" | "field_name" | "by_field_name" | "over_field_name" | "partition_field_name" => {
                d.insert(k.clone(), json!(string(val, vp, P, k)?));
            }
            "function" => {
                let s = string(val, vp, P, k)?;
                let f = canonical_function(&s).ok_or_else(|| at(wrap(vp.start, P, k, EsError::illegal_argument(format!("Unknown function '{s}'"))), vp.start))?;
                d.insert("function".into(), json!(f));
            }
            "use_null" => {
                d.insert(k.clone(), json!(boolean(val, vp, P, k)?));
            }
            "exclude_frequent" => {
                let s = string(val, vp, P, k)?.to_ascii_lowercase();
                let s = match s.as_str() {
                    "all" | "true" => "all",
                    "none" | "false" => "none",
                    "by" => "by",
                    "over" => "over",
                    _ => return Err(at(wrap(vp.start, P, k, EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.ml.job.config.Detector.ExcludeFrequent.{}", s.to_ascii_uppercase()))), vp.start)),
                };
                d.insert(k.clone(), json!(s));
            }
            "detector_index" => {
                long(val, vp, P, k)?;
            }
            "custom_rules" => {
                let arr = val.as_array().cloned().unwrap_or_default();
                for (i, r) in arr.iter().enumerate() {
                    rules.push(parse_rule(r, vp.at(i)).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?);
                }
            }
            other => return Err(at(unknown(p.key(other), P, other, DETECTOR_FIELDS), p.key(other))),
        }
    }
    let fail = |e: EsError| at(e, p.end);
    let function = d.get("function").and_then(Value::as_str).unwrap_or("metric").to_string();
    d.insert("function".into(), json!(function));
    validate_detector(&function, &d, &rules).map_err(fail)?;
    if !rules.is_empty() {
        d.insert("custom_rules".into(), Value::Array(rules));
    }
    Ok(d)
}

fn validate_detector(function: &str, d: &Map<String, Value>, rules: &[Value]) -> Result<()> {
    let g = |k: &str| d.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
    let (field, by, over, part) = (g("field_name"), g("by_field_name"), g("over_field_name"), g("partition_field_name"));
    if field.is_none() && by.is_none() && over.is_none() && !COUNT_WITHOUT_FIELD.contains(&function) {
        return Err(bad_request("Unless a count or temporal function is used one of field_name, by_field_name or over_field_name must be set"));
    }
    if is_field_function(function) && field.is_none() {
        return Err(bad_request(format!("field_name must be set when the '{function}' function is used")));
    }
    if field.is_some() && !is_field_function(function) {
        return Err(bad_request(format!("field_name cannot be used with function '{function}'")));
    }
    if BY_FIELD_FUNCTIONS.contains(&function) && by.is_none() {
        return Err(bad_request(format!("by_field_name must be set when the '{function}' function is used")));
    }
    if OVER_FIELD_FUNCTIONS.contains(&function) && over.is_none() {
        return Err(bad_request(format!("over_field_name must be set when the '{function}' function is used")));
    }
    if over.is_some() && NO_OVER_FIELD_FUNCTIONS.contains(&function) {
        return Err(bad_request(format!("over_field_name cannot be used with function '{function}'")));
    }
    for f in [field, by, over, part].into_iter().flatten() {
        if f.contains('"') || f.contains('\\') {
            return Err(bad_request(format!("Invalid field name '{f}'. Field names including over, by and partition fields cannot contain any of these characters: [\", \\]")));
        }
    }
    check_rules(function, d, rules)?;
    if let Some(pf) = part {
        if Some(pf) == by {
            return Err(bad_request(format!("partition_field_name and by_field_name cannot be the same: '{pf}'")));
        }
        if Some(pf) == over {
            return Err(bad_request(format!("partition_field_name and over_field_name cannot be the same: '{pf}'")));
        }
    }
    if let Some(b) = by {
        if Some(b) == over {
            return Err(bad_request(format!("by_field_name and over_field_name cannot be the same: '{b}'")));
        }
    }
    for (name, v) in [("by_field_name", by), ("over_field_name", over)] {
        if let Some(x) = v {
            if matches!(x, "count" | "by" | "over") {
                return Err(bad_request(format!("'{x}' is not a permitted value for {name}")));
            }
        }
    }
    Ok(())
}

/// `DefaultDetectorDescription.of`.
pub fn default_description(d: &Map<String, Value>) -> String {
    let g = |k: &str| d.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
    let mut s = d.get("function").and_then(Value::as_str).unwrap_or("metric").to_string();
    if let Some(f) = g("field_name") {
        s.push_str(&format!("({})", quote_field(f)));
    }
    if let Some(b) = g("by_field_name") {
        s.push_str(&format!(" by {}", quote_field(b)));
    }
    if let Some(o) = g("over_field_name") {
        s.push_str(&format!(" over {}", quote_field(o)));
    }
    if d.get("use_null").and_then(Value::as_bool) == Some(true) {
        s.push_str(" usenull=true");
    }
    if let Some(p) = g("partition_field_name") {
        s.push_str(&format!(" partitionfield={}", quote_field(p)));
    }
    if let Some(e) = g("exclude_frequent") {
        s.push_str(&format!(" excludefrequent={e}"));
    }
    s
}

/// Render a detector in ES field order.
fn render_detector(d: &Map<String, Value>, index: usize) -> Value {
    let mut o = Map::new();
    let desc = d.get("detector_description").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| default_description(d));
    o.insert("detector_description".into(), json!(desc));
    for k in ["function", "field_name", "by_field_name", "over_field_name", "partition_field_name", "use_null", "exclude_frequent", "custom_rules"] {
        if let Some(v) = d.get(k) {
            o.insert(k.into(), v.clone());
        }
    }
    o.insert("detector_index".into(), json!(index));
    Value::Object(o)
}

// ---------------------------------------------------------------------------
// Analysis config
// ---------------------------------------------------------------------------

const ANALYSIS_FIELDS: &[&str] = &[
    "bucket_span", "categorization_field_name", "categorization_filters", "categorization_analyzer", "per_partition_categorization", "latency", "summary_count_field_name", "detectors", "influencers",
    "multivariate_by_fields", "model_prune_window",
];

/// The default ML categorization analyzer (`CategorizationAnalyzerConfig.buildStandardCategorizationAnalyzer`).
pub fn standard_categorization_analyzer(filters: &[String]) -> Value {
    let mut char_filter = vec![json!("first_line_with_letters")];
    for f in filters {
        char_filter.push(json!({"pattern": f, "type": "pattern_replace"}));
    }
    json!({"char_filter": char_filter, "tokenizer": "ml_standard", "filter": [
        {"type": "stop", "stopwords": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","Mon","Tue","Wed","Thu","Fri","Sat","Sun","January","February","March","April","May","June","July","August","September","October","November","December","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","GMT","UTC"]},
        {"type": "limit", "max_token_count": "100"}]})
}

/// Parse + validate `analysis_config` into its ES rendering.
fn parse_analysis_config(v: &Value, p: &Pos) -> PR<Value> {
    const P: &str = "analysis_config";
    let o = obj(v, p, P)?;
    let mut bucket_span: Option<TimeV> = None;
    let mut latency: Option<TimeV> = None;
    let mut prune: Option<TimeV> = None;
    let mut cat_field: Option<String> = None;
    let mut cat_filters: Option<Vec<String>> = None;
    let mut cat_analyzer: Option<Value> = None;
    let mut ppc: Option<Value> = None;
    let mut summary: Option<String> = None;
    let mut detectors: Option<Vec<Map<String, Value>>> = None;
    let mut influencers: Vec<String> = Vec::new();
    let mut multivariate: Option<bool> = None;
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "bucket_span" => bucket_span = Some(time_value(val, vp, P, k)?),
            "latency" => latency = Some(time_value(val, vp, P, k)?),
            "model_prune_window" => prune = Some(time_value(val, vp, P, k)?),
            "categorization_field_name" => cat_field = Some(string(val, vp, P, k)?),
            "categorization_filters" => cat_filters = Some(string_list(val, vp, P, k)?),
            "categorization_analyzer" => {
                if !(val.is_string() || val.is_object()) {
                    return Err(at(xparse(format!("{} [{P}] categorization_analyzer doesn't support values of type: {}", fmt(vp.start), xc::token_name(val))), vp.start));
                }
                cat_analyzer = Some(val.clone());
            }
            "per_partition_categorization" => {
                let po = obj(val, vp, "per_partition_categorization")?;
                let mut enabled = false;
                let mut stop = None;
                for (pk, pv) in po {
                    match pk.as_str() {
                        "enabled" => enabled = boolean(pv, vp.get(pk), "per_partition_categorization", pk)?,
                        "stop_on_warn" => stop = Some(boolean(pv, vp.get(pk), "per_partition_categorization", pk)?),
                        other => return Err(at(unknown(vp.key(other), "per_partition_categorization", other, &["enabled", "stop_on_warn"]), vp.key(other))),
                    }
                }
                let mut m = Map::new();
                m.insert("enabled".into(), json!(enabled));
                if enabled {
                    m.insert("stop_on_warn".into(), json!(stop.unwrap_or(false)));
                }
                ppc = Some(Value::Object(m));
            }
            "summary_count_field_name" => summary = Some(string(val, vp, P, k)?),
            "detectors" => {
                let arr = match val {
                    Value::Array(a) => a.clone(),
                    other => return Err(at(xparse(format!("{} [{P}] detectors doesn't support values of type: {}", fmt(vp.start), xc::token_name(other))), vp.start)),
                };
                let mut ds = Vec::new();
                for (i, d) in arr.iter().enumerate() {
                    ds.push(parse_detector(d, vp.at(i)).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?);
                }
                detectors = Some(ds);
            }
            "influencers" => influencers = string_list(val, vp, P, k)?,
            "multivariate_by_fields" => multivariate = Some(boolean(val, vp, P, k)?),
            other => return Err(at(unknown(p.key(other), P, other, ANALYSIS_FIELDS), p.key(other))),
        }
    }
    let end = p.end;
    let fail = |e: EsError| at(e, end);
    let bucket_span = bucket_span.unwrap_or_else(|| TimeV { nanos: 300_000_000_000, rep: "5m".into() });
    check_positive_multiple_secs(&bucket_span, "bucket_span").map_err(fail)?;
    if let Some(l) = &latency {
        if l.nanos < 0 {
            return Err(fail(EsError::illegal_argument(format!("latency cannot be less than 0. Value = {}", time_to_string(l.nanos)))));
        }
        if l.nanos % 1_000_000_000 != 0 {
            return Err(fail(EsError::illegal_argument(format!("latency has to be a multiple of 1s; actual was '{}'", time_to_string(l.nanos)))));
        }
    }
    let detectors = detectors.unwrap_or_default();
    if detectors.is_empty() {
        return Err(fail(bad_request("No detectors configured")));
    }
    for f in [&summary, &cat_field].into_iter().flatten() {
        if f.contains('"') || f.contains('\\') {
            return Err(fail(bad_request(format!("Invalid field name '{f}'. Field names including over, by and partition fields cannot contain any of these characters: [\", \\]"))));
        }
    }
    let uses_mlcategory = detectors.iter().any(|d| ["by_field_name", "over_field_name", "partition_field_name"].iter().any(|k| d.get(*k).and_then(Value::as_str) == Some("mlcategory")));
    if cat_field.is_some() && !uses_mlcategory {
        return Err(fail(bad_request("categorization_field_name is set but mlcategory is not used in any detector by/over/partition field")));
    }
    if cat_field.is_none() && uses_mlcategory {
        return Err(fail(bad_request("categorization_field_name must be set for mlcategory to be available")));
    }
    if cat_analyzer.is_some() {
        if cat_field.is_none() {
            return Err(fail(bad_request("categorization_analyzer requires setting categorization_field_name")));
        }
        if cat_filters.is_some() {
            return Err(fail(bad_request("categorization_filters cannot be used with categorization_analyzer - instead specify them as pattern_replace char_filters in the analyzer")));
        }
    }
    if let Some(fl) = &cat_filters {
        if cat_field.is_none() {
            return Err(fail(bad_request("categorization_filters require setting categorization_field_name")));
        }
        if fl.iter().any(String::is_empty) {
            return Err(fail(bad_request("categorization_filters are not allowed to contain empty strings")));
        }
        let mut sorted = fl.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != fl.len() {
            return Err(fail(bad_request("categorization_filters contain duplicates")));
        }
        for f in fl {
            if regex::Regex::new(f).is_err() {
                return Err(fail(bad_request(format!("categorization_filters contains invalid regular expression '{f}'"))));
            }
        }
    }
    if ppc.as_ref().and_then(|x| x["enabled"].as_bool()) == Some(true) {
        if cat_field.is_none() {
            return Err(fail(bad_request("categorization_field_name must be set for per_partition_categorization to be enabled")));
        }
        let parts: Vec<&str> = detectors.iter().filter_map(|d| d.get("partition_field_name").and_then(Value::as_str)).collect();
        if parts.is_empty() {
            return Err(fail(bad_request("partition_field_name must be set for detectors that reference mlcategory when per-partition categorization is enabled")));
        }
    }
    if summary.is_some() {
        if let Some(d) = detectors.iter().find(|d| d.get("function").and_then(Value::as_str) == Some("metric")) {
            let _ = d;
            return Err(fail(bad_request("The 'metric' function cannot be used in jobs that will take pre-summarized input")));
        }
    }
    let prune_rep = match &prune {
        Some(pw) => {
            check_prune_window(pw, &bucket_span).map_err(fail)?;
            pw.rep.clone()
        }
        None => default_prune_window(bucket_span.seconds()),
    };
    // Render (AnalysisConfig.toXContent order).
    let mut out = Map::new();
    out.insert("bucket_span".into(), json!(bucket_span.rep));
    if let Some(cf) = &cat_field {
        out.insert("categorization_field_name".into(), json!(cf));
        let analyzer = cat_analyzer.clone().unwrap_or_else(|| standard_categorization_analyzer(cat_filters.as_deref().unwrap_or(&[])));
        out.insert("categorization_analyzer".into(), analyzer);
        out.insert("per_partition_categorization".into(), ppc.clone().unwrap_or_else(|| json!({"enabled": false})));
    }
    if let Some(l) = &latency {
        out.insert("latency".into(), json!(l.rep));
    }
    if let Some(s) = &summary {
        out.insert("summary_count_field_name".into(), json!(s));
    }
    out.insert("detectors".into(), Value::Array(detectors.iter().enumerate().map(|(i, d)| render_detector(d, i)).collect()));
    out.insert("influencers".into(), json!(influencers));
    if let Some(m) = multivariate {
        out.insert("multivariate_by_fields".into(), json!(m));
    }
    out.insert("model_prune_window".into(), json!(prune_rep));
    Ok(Value::Object(out))
}

fn check_positive_multiple_secs(t: &TimeV, name: &str) -> Result<()> {
    if t.nanos <= 0 {
        return Err(EsError::illegal_argument(format!("{name} cannot be less or equal than 0. Value = {}", t.rep)));
    }
    if t.nanos % 1_000_000_000 != 0 {
        return Err(EsError::illegal_argument(format!("{name} has to be a multiple of 1s; actual was '{}'", time_to_string(t.nanos))));
    }
    Ok(())
}

/// Default `model_prune_window`: the larger of 30 days and 20 bucket
/// spans, rounded up to a multiple of the bucket span.
pub fn default_prune_window(bucket_secs: i64) -> String {
    let base = (30 * 86_400i64).max(20 * bucket_secs);
    let bs = bucket_secs.max(1);
    let secs = base.div_euclid(bs) * bs + if base % bs == 0 { 0 } else { bs };
    exact_time_rep(secs)
}

fn check_prune_window(pw: &TimeV, bucket_span: &TimeV) -> Result<()> {
    let bs = bucket_span.nanos.max(1);
    if pw.nanos % bs != 0 {
        return Err(bad_request(format!("model_prune_window [{}] must be a multiple of bucket_span [{}]", pw.rep, bucket_span.rep)));
    }
    if pw.nanos < 2 * bs {
        return Err(bad_request(format!("model_prune_window [{}] must be at least 2 times greater than bucket_span [{}]", pw.rep, bucket_span.rep)));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Limits, data description, model plot
// ---------------------------------------------------------------------------

/// `analysis_limits`: (model_memory_limit MB if given, categorization_examples_limit).
fn parse_limits(v: &Value, p: &Pos) -> PR<(Option<u64>, Option<i64>)> {
    const P: &str = "analysis_limits";
    let o = obj(v, p, P)?;
    let mut mml = None;
    let mut cel = None;
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "model_memory_limit" => {
                mml = Some(match val {
                    Value::Number(n) => n.as_i64().unwrap_or(0).max(0) as u64,
                    Value::String(s) => parse_bytes(s, k).map_err(|e| at(wrap(vp.start, P, k, e), vp.start))? / MB,
                    other => return Err(at(xparse(format!("{} [{P}] model_memory_limit doesn't support values of type: {}", fmt(vp.start), xc::token_name(other))), vp.start)),
                });
            }
            "categorization_examples_limit" => cel = Some(long(val, vp, P, k)?),
            other => return Err(at(unknown(p.key(other), P, other, &["model_memory_limit", "categorization_examples_limit"]), p.key(other))),
        }
    }
    let built = |e: EsError| at(xparse("Failed to build [analysis_limits] after last required field arrived").caused_by(e), p.end);
    if let Some(m) = mml {
        if m < 1 {
            return Err(built(bad_request(format!("model_memory_limit must be at least 1 MiB. Value = {m}"))));
        }
    }
    if let Some(c) = cel {
        if c < 0 {
            return Err(built(bad_request(format!("categorization_examples_limit cannot be less than 0. Value = {c}"))));
        }
    }
    Ok((mml, cel))
}

/// Java `DateTimeFormatter` pattern letters.
fn valid_time_format(f: &str) -> bool {
    if matches!(f, "epoch" | "epoch_ms") {
        return true;
    }
    let mut in_quote = false;
    let mut has_letter = false;
    for c in f.chars() {
        if c == '\'' {
            in_quote = !in_quote;
            continue;
        }
        if !in_quote && c.is_ascii_alphabetic() {
            has_letter = true;
            if !"GuyDMLdQqYwWEecFahKkHmsSAnNVzOXxZp".contains(c) {
                return false;
            }
        }
    }
    has_letter && !in_quote
}

fn parse_data_description(v: &Value, p: &Pos) -> PR<Value> {
    const P: &str = "data_description";
    let o = obj(v, p, P)?;
    let mut time_field = "time".to_string();
    let mut time_format = "epoch_ms".to_string();
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "format" => {
                let f = string(val, vp, P, k)?.to_ascii_lowercase();
                if !matches!(f.as_str(), "xcontent" | "delimited" | "json") {
                    return Err(at(wrap(vp.start, P, k, EsError::illegal_argument(format!("Unknown data format: {f}"))), vp.start));
                }
            }
            "time_field" => time_field = string(val, vp, P, k)?,
            "time_format" => {
                let f = string(val, vp, P, k)?;
                if !valid_time_format(&f) {
                    return Err(at(wrap(vp.start, P, k, bad_request(format!("Invalid Time format string '{f}'"))), vp.start));
                }
                time_format = f;
            }
            "field_delimiter" | "quote_character" => {
                string(val, vp, P, k)?;
            }
            other => return Err(at(unknown(p.key(other), P, other, &["format", "time_field", "time_format", "field_delimiter", "quote_character"]), p.key(other))),
        }
    }
    Ok(json!({"time_field": time_field, "time_format": time_format}))
}

fn parse_model_plot(v: &Value, p: &Pos) -> PR<Value> {
    const P: &str = "model_plot_config";
    let o = obj(v, p, P)?;
    let mut enabled = None;
    let mut terms = None;
    let mut annotations = None;
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "enabled" => enabled = Some(boolean(val, vp, P, k)?),
            "terms" => terms = Some(string(val, vp, P, k)?),
            "annotations_enabled" => annotations = Some(boolean(val, vp, P, k)?),
            other => return Err(at(unknown(p.key(other), P, other, &["enabled", "terms", "annotations_enabled"]), p.key(other))),
        }
    }
    let Some(enabled) = enabled else {
        return Err(at(EsError::illegal_argument("Required [enabled]"), p.end));
    };
    let mut m = Map::new();
    m.insert("enabled".into(), json!(enabled));
    if let Some(t) = terms {
        m.insert("terms".into(), json!(t));
    }
    m.insert("annotations_enabled".into(), json!(annotations.unwrap_or(enabled)));
    Ok(Value::Object(m))
}

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

const JOB_FIELDS: &[&str] = &[
    "job_id", "job_type", "groups", "description", "analysis_config", "analysis_limits", "data_description", "model_plot_config", "renormalization_window_days", "background_persist_interval",
    "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days", "custom_settings", "results_index_name", "allow_lazy_open", "datafeed_config",
];

/// A parsed `PUT` job body before the cluster checks.
pub struct NewJob {
    /// Rendered config (without `create_time` etc. which [`finish`] adds).
    pub parts: Map<String, Value>,
    /// The explicit `analysis_limits.model_memory_limit` (MB).
    pub explicit_mml: Option<u64>,
    pub cel: Option<i64>,
    pub datafeed: Option<Value>,
    pub warnings: Vec<String>,
}

fn id_error(kind: &str, id: &str) -> EsError {
    EsError::illegal_argument(format!("Invalid {kind}; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"))
}

/// Parse a `PUT _ml/anomaly_detectors/{id}` body (REST validation: parsing,
/// `Job.Builder.validateInputFields`, analysis config rules).
pub fn parse_put(id: &str, body: &Value, raw: &str) -> Result<NewJob> {
    let pos = xc::positions(raw);
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        other => return Err(xparse(format!("[1:1] [job_details] Expected START_OBJECT but was: {}", xc::token_name(other)))),
    };
    parse_job_body(id, o, &pos).map_err(|pe| pe.e)
}

fn parse_job_body(id: &str, o: &Map<String, Value>, p: &Pos) -> PR<NewJob> {
    const P: &str = "job_details";
    let mut body_id: Option<String> = None;
    let mut parts = Map::new();
    let mut ac: Option<Value> = None;
    let mut dd: Option<Value> = None;
    let mut mml = None;
    let mut cel = None;
    let mut datafeed = None;
    let mut groups: Vec<String> = Vec::new();
    let mut nums: Map<String, Value> = Map::new();
    for (k, val) in o {
        let vp = p.get(k);
        match k.as_str() {
            "job_id" => body_id = Some(string(val, vp, P, k)?),
            "job_type" => {
                string(val, vp, P, k)?;
            }
            "description" => {
                parts.insert("description".into(), json!(string(val, vp, P, k)?));
            }
            "groups" => groups = string_list(val, vp, P, k)?,
            "analysis_config" => ac = Some(parse_analysis_config(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?),
            "analysis_limits" => {
                let (m, c) = parse_limits(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?;
                mml = m;
                cel = c;
            }
            "data_description" => dd = Some(parse_data_description(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?),
            "model_plot_config" => {
                parts.insert("model_plot_config".into(), parse_model_plot(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?);
            }
            "renormalization_window_days" | "model_snapshot_retention_days" | "daily_model_snapshot_retention_after_days" | "results_retention_days" => {
                nums.insert(k.clone(), json!(long(val, vp, P, k)?));
            }
            "background_persist_interval" => {
                let t = time_value(val, vp, P, k)?;
                parts.insert(k.clone(), json!(t.rep));
            }
            "custom_settings" => {
                obj(val, vp, P)?;
                parts.insert(k.clone(), val.clone());
            }
            "results_index_name" => {
                parts.insert(k.clone(), json!(string(val, vp, P, k)?));
            }
            "allow_lazy_open" => {
                parts.insert(k.clone(), json!(boolean(val, vp, P, k)?));
            }
            "datafeed_config" => {
                obj(val, vp, P)?;
                datafeed = Some(val.clone());
            }
            other => return Err(at(unknown(p.key(other), P, other, JOB_FIELDS), p.key(other))),
        }
    }
    let plain = |e: EsError| at(e, p.end);
    if let Some(b) = &body_id {
        if !id.is_empty() && b != id {
            return Err(plain(EsError::illegal_argument(format!("Inconsistent job_id; '{b}' specified in the body differs from '{id}' specified as a URL argument"))));
        }
    }
    let Some(ac) = ac else { return Err(plain(EsError::illegal_argument("An analysis_config must be set"))) };
    let Some(dd) = dd else { return Err(plain(EsError::illegal_argument("A data_description must be set"))) };
    validate_id("job_id", id).map_err(plain)?;
    for g in &groups {
        if !super::is_valid_id(g) {
            return Err(plain(EsError::illegal_argument(format!(
                "Invalid group id '{g}'; must be non-empty string and may contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"
            ))));
        }
    }
    if let Some(r) = parts.get("results_index_name").and_then(Value::as_str) {
        if !r.is_empty() && !super::is_valid_id(r) {
            return Err(plain(id_error("results_index_name", r)));
        }
    }
    for k in ["renormalization_window_days", "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days"] {
        if let Some(n) = nums.get(k).and_then(Value::as_i64) {
            if n < 0 {
                return Err(plain(EsError::illegal_argument(format!("{k} cannot be less than 0. Value = {n}"))));
            }
        }
    }
    let msrd = nums.get("model_snapshot_retention_days").and_then(Value::as_i64).unwrap_or(10);
    let daily = nums.get("daily_model_snapshot_retention_after_days").and_then(Value::as_i64).unwrap_or_else(|| msrd.min(1));
    check_snapshot_retention(msrd, daily).map_err(plain)?;
    check_time_field(&ac, &dd).map_err(plain)?;
    check_duplicate_detectors(&ac).map_err(plain)?;
    let mut warnings = Vec::new();
    if let Some(bs) = ac["bucket_span"].as_str() {
        if let Ok(t) = parse_time(bs, "bucket_span") {
            let secs = t.seconds();
            if secs > 86_400 && secs % 86_400 != 0 {
                warnings.push(format!("bucket_span {} [{secs}s] is not an integral multiple of the number of seconds in 1d [86400s]. This is now deprecated.", time_to_string(t.nanos)));
            } else if secs < 86_400 && 86_400 % secs != 0 {
                warnings.push(format!("bucket_span {} [{secs}s] is not an integral divisor of the number of seconds in 1d [86400s]. This is now deprecated.", time_to_string(t.nanos)));
            }
        }
    }
    if !groups.is_empty() {
        parts.insert("groups".into(), json!(groups));
    }
    parts.insert("analysis_config".into(), ac);
    parts.insert("data_description".into(), dd);
    for (k, v) in nums {
        parts.insert(k, v);
    }
    parts.insert("model_snapshot_retention_days".into(), json!(msrd));
    parts.insert("daily_model_snapshot_retention_after_days".into(), json!(daily));
    Ok(NewJob { parts, explicit_mml: mml, cel, datafeed, warnings })
}

pub fn validate_id(kind: &str, id: &str) -> Result<()> {
    if !super::is_valid_id(id) {
        return Err(id_error(kind, id));
    }
    if id.len() > super::ID_LENGTH_LIMIT {
        return Err(EsError::illegal_argument(format!("The {} cannot contain more than {} characters.", kind.replace('_', " "), super::ID_LENGTH_LIMIT)));
    }
    Ok(())
}

fn check_snapshot_retention(msrd: i64, daily: i64) -> Result<()> {
    if daily > msrd {
        return Err(EsError::illegal_argument(format!(
            "The value of daily_model_snapshot_retention_after_days [{daily}] cannot be greater than model_snapshot_retention_days [{msrd}]"
        )));
    }
    Ok(())
}

fn check_time_field(ac: &Value, dd: &Value) -> Result<()> {
    let tf = dd["time_field"].as_str().unwrap_or("time");
    let mut used: Vec<&str> = Vec::new();
    for d in ac["detectors"].as_array().into_iter().flatten() {
        for k in ["field_name", "by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(s) = d[k].as_str() {
                used.push(s);
            }
        }
    }
    for k in ["summary_count_field_name", "categorization_field_name"] {
        if let Some(s) = ac[k].as_str() {
            used.push(s);
        }
    }
    for i in ac["influencers"].as_array().into_iter().flatten() {
        if let Some(s) = i.as_str() {
            used.push(s);
        }
    }
    if used.contains(&tf) {
        return Err(EsError::illegal_argument("data_description.time_field may not be used in the analysis_config"));
    }
    Ok(())
}

fn check_duplicate_detectors(ac: &Value) -> Result<()> {
    let ds = ac["detectors"].as_array().cloned().unwrap_or_default();
    let key = |d: &Value| {
        let mut m = d.as_object().cloned().unwrap_or_default();
        m.remove("detector_index");
        m.remove("detector_description");
        m
    };
    let mut dups: Vec<String> = Vec::new();
    for i in 0..ds.len() {
        for j in 0..i {
            if key(&ds[i]) == key(&ds[j]) {
                dups.push(ds[i]["detector_description"].as_str().unwrap_or("").to_string());
            }
        }
    }
    if !dups.is_empty() {
        return Err(EsError::illegal_argument(format!("Duplicate detectors are not allowed: [{}]", dups.join(", "))));
    }
    Ok(())
}

/// Assemble the stored job (ES `Job.toXContent` order) from parsed parts.
pub fn finish(id: &str, nj: &NewJob, default_mml_mb: u64, create_time: i64) -> Value {
    let p = &nj.parts;
    let mut o = Map::new();
    o.insert("job_id".into(), json!(id));
    o.insert("job_type".into(), json!("anomaly_detector"));
    o.insert("job_version".into(), json!(CONFIG_VERSION));
    o.insert("create_time".into(), json!(create_time));
    let mml = nj.explicit_mml.unwrap_or(default_mml_mb);
    let limits = json!({"model_memory_limit": format!("{mml}mb"), "categorization_examples_limit": nj.cel.unwrap_or(4)});
    let results_index = match p.get("results_index_name").and_then(Value::as_str) {
        Some(r) if !r.is_empty() && r != "shared" => format!("custom-{r}"),
        _ => "shared".into(),
    };
    let mut rest = Map::new();
    for k in ["custom_settings", "groups", "description"] {
        if let Some(v) = p.get(k) {
            rest.insert(k.into(), v.clone());
        }
    }
    rest.insert("analysis_config".into(), p["analysis_config"].clone());
    rest.insert("analysis_limits".into(), limits);
    rest.insert("data_description".into(), p["data_description"].clone());
    for k in ["model_plot_config", "renormalization_window_days", "background_persist_interval", "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days"] {
        if let Some(v) = p.get(k) {
            rest.insert(k.into(), v.clone());
        }
    }
    rest.insert("results_index_name".into(), json!(results_index));
    rest.insert("allow_lazy_open".into(), p.get("allow_lazy_open").cloned().unwrap_or(json!(false)));
    o.extend(rest);
    Value::Object(o)
}

/// The job as ES renders it: fields in `Job.toXContent` order with the
/// datafeed spliced in; `exclude_generated` drops generated fields.
pub fn render(job: &Value, datafeed: Option<&Value>, exclude_generated: bool) -> Value {
    let src = job.as_object().cloned().unwrap_or_default();
    let mut o = Map::new();
    o.insert("job_id".into(), src.get("job_id").cloned().unwrap_or(Value::Null));
    if !exclude_generated {
        for k in ["job_type", "job_version", "create_time", "finished_time", "model_snapshot_id", "deleting", "model_snapshot_min_version"] {
            if let Some(v) = src.get(k) {
                o.insert(k.into(), v.clone());
            }
        }
        if let Some(v) = src.get("custom_settings") {
            o.insert("custom_settings".into(), v.clone());
        }
        if let Some(d) = datafeed {
            o.insert("datafeed_config".into(), d.clone());
        }
    } else if let Some(Value::Object(cs)) = src.get("custom_settings") {
        let mut cs = cs.clone();
        cs.remove("created_by");
        o.insert("custom_settings".into(), Value::Object(cs));
    }
    for k in [
        "groups", "description", "analysis_config", "analysis_limits", "data_description", "model_plot_config", "renormalization_window_days", "background_persist_interval",
        "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days", "results_index_name", "allow_lazy_open", "blocked",
    ] {
        if let Some(v) = src.get(k) {
            if exclude_generated && k == "results_index_name" {
                // `custom-` prefix is generated; the export keeps the user's name.
                let r = v.as_str().unwrap_or("shared");
                o.insert(k.into(), json!(r.strip_prefix("custom-").unwrap_or(r)));
                continue;
            }
            o.insert(k.into(), v.clone());
        }
    }
    Value::Object(o)
}

/// Model memory limit of a stored job in bytes.
pub fn mml_bytes(job: &Value) -> u64 {
    job["analysis_limits"]["model_memory_limit"].as_str().and_then(|s| parse_bytes(s, "model_memory_limit").ok()).unwrap_or(DEFAULT_MML_MB * MB)
}

/// Filter ids referenced by the job's detector rules.
pub fn referenced_filters(job: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for d in job["analysis_config"]["detectors"].as_array().into_iter().flatten() {
        for r in d["custom_rules"].as_array().into_iter().flatten() {
            if let Some(s) = r["scope"].as_object() {
                for f in s.values() {
                    if let Some(id) = f["filter_id"].as_str() {
                        if !out.iter().any(|x| x == id) {
                            out.push(id.to_string());
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

const UPDATE_FIELDS: &[&str] = &[
    "job_id", "groups", "description", "detectors", "model_plot_config", "analysis_limits", "renormalization_window_days", "background_persist_interval", "model_snapshot_retention_days",
    "daily_model_snapshot_retention_after_days", "results_retention_days", "custom_settings", "categorization_filters", "allow_lazy_open", "model_prune_window", "per_partition_categorization",
];

/// A parsed `_update` body.
pub struct JobUpdate {
    pub fields: Map<String, Value>,
    /// (detector_index, description, custom_rules)
    pub detectors: Vec<(i64, Option<String>, Option<Vec<Value>>)>,
    pub mml: Option<u64>,
    pub cel: Option<i64>,
    pub prune: Option<TimeV>,
}

/// Parse `POST _ml/anomaly_detectors/{id}/_update` (`JobUpdate.PARSER`).
pub fn parse_update(body: &Value, raw: &str) -> Result<JobUpdate> {
    const P: &str = "job_update";
    let pos = xc::positions(raw);
    let empty = Map::new();
    let o = match body {
        Value::Object(o) => o,
        Value::Null => &empty,
        other => return Err(xparse(format!("[1:1] [{P}] Expected START_OBJECT but was: {}", xc::token_name(other)))),
    };
    // ConstructingObjectParser: field failures are wrapped once more.
    let ctor = |pe: PErr| xparse(format!("{} failed to build [{P}] after last required field arrived", fmt(pe.at))).caused_by(pe.e);
    let mut u = JobUpdate { fields: Map::new(), detectors: Vec::new(), mml: None, cel: None, prune: None };
    for (k, val) in o {
        let vp = pos.get(k);
        let r: PR<()> = (|| {
            match k.as_str() {
                "job_id" | "description" => {
                    u.fields.insert(k.clone(), json!(string(val, vp, P, k)?));
                }
                "groups" | "categorization_filters" => {
                    u.fields.insert(k.clone(), json!(string_list(val, vp, P, k)?));
                }
                "detectors" => {
                    for (i, d) in val.as_array().cloned().unwrap_or_default().iter().enumerate() {
                        let dp = vp.at(i);
                        let dobj = obj(d, dp, "detector_update")?;
                        let mut idx = None;
                        let mut desc = None;
                        let mut rules = None;
                        for (dk, dv) in dobj {
                            match dk.as_str() {
                                "detector_index" => idx = Some(long(dv, dp.get(dk), "detector_update", dk)?),
                                "description" => desc = Some(string(dv, dp.get(dk), "detector_update", dk)?),
                                "custom_rules" => {
                                    let mut rs = Vec::new();
                                    for (ri, rv) in dv.as_array().cloned().unwrap_or_default().iter().enumerate() {
                                        rs.push(parse_rule(rv, dp.get(dk).at(ri)).map_err(|pe| at(wrap(pe.at, "detector_update", dk, pe.e), pe.at))?);
                                    }
                                    rules = Some(rs);
                                }
                                other => return Err(at(unknown(dp.key(other), "detector_update", other, &["detector_index", "description", "custom_rules"]), dp.key(other))),
                            }
                        }
                        let Some(idx) = idx else { return Err(at(EsError::illegal_argument("Required [detector_index]"), dp.end)) };
                        u.detectors.push((idx, desc, rules));
                    }
                }
                "model_plot_config" => {
                    u.fields.insert(k.clone(), parse_model_plot(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?);
                }
                "analysis_limits" => {
                    let (m, c) = parse_limits(val, vp).map_err(|pe| at(wrap(pe.at, P, k, pe.e), pe.at))?;
                    u.mml = m;
                    u.cel = c;
                    u.fields.insert(k.clone(), Value::Bool(true));
                }
                "renormalization_window_days" | "model_snapshot_retention_days" | "daily_model_snapshot_retention_after_days" | "results_retention_days" => {
                    u.fields.insert(k.clone(), json!(long(val, vp, P, k)?));
                }
                "background_persist_interval" => {
                    let t = time_value(val, vp, P, k)?;
                    u.fields.insert(k.clone(), json!(t.rep));
                }
                "model_prune_window" => {
                    u.prune = Some(time_value(val, vp, P, k)?);
                }
                "custom_settings" => {
                    obj(val, vp, P)?;
                    u.fields.insert(k.clone(), val.clone());
                }
                "allow_lazy_open" => {
                    u.fields.insert(k.clone(), json!(boolean(val, vp, P, k)?));
                }
                "per_partition_categorization" => {
                    obj(val, vp, P)?;
                    u.fields.insert(k.clone(), val.clone());
                }
                other => return Err(at(unknown(pos.key(other), P, other, UPDATE_FIELDS), pos.key(other))),
            }
            Ok(())
        })();
        if let Err(pe) = r {
            // unknown fields are reported directly; value failures through the ctor wrapper
            if pe.e.reason.contains("unknown field") {
                return Err(pe.e);
            }
            return Err(ctor(pe));
        }
    }
    Ok(u)
}

/// Apply an update to a stored job (`JobUpdate.mergeWithJob`); `open` is
/// whether the job is opened. Group-name clashes are checked by the caller.
pub fn apply_update(job: &Value, u: &JobUpdate, open: bool, max_mml_mb: Option<u64>) -> Result<Value> {
    let mut j = job.as_object().cloned().unwrap_or_default();
    if let Some(g) = u.fields.get("groups") {
        for x in g.as_array().into_iter().flatten().filter_map(Value::as_str) {
            if !super::is_valid_id(x) {
                return Err(EsError::illegal_argument(format!(
                    "Invalid group id '{x}'; must be non-empty string and may contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"
                )));
            }
        }
        if g.as_array().map(|a| a.is_empty()).unwrap_or(true) {
            j.remove("groups");
        } else {
            j.insert("groups".into(), g.clone());
        }
    }
    if let Some(d) = u.fields.get("description") {
        j.insert("description".into(), d.clone());
    }
    let mut ac = j.get("analysis_config").and_then(Value::as_object).cloned().unwrap_or_default();
    if !u.detectors.is_empty() {
        let mut ds: Vec<Value> = ac.get("detectors").and_then(Value::as_array).cloned().unwrap_or_default();
        for (idx, desc, rules) in &u.detectors {
            if *idx < 0 || *idx as usize >= ds.len() {
                return Err(bad_request(format!("Supplied detector_index [{idx}] is >= the number of detectors [{}]", ds.len())));
            }
            let d = ds[*idx as usize].as_object_mut().expect("detector object");
            if let Some(desc) = desc {
                d.insert("detector_description".into(), json!(desc));
            }
            if let Some(rs) = rules {
                let function = d.get("function").and_then(Value::as_str).unwrap_or("metric").to_string();
                check_rules(&function, d, rs)?;
                // keep ES field order: custom_rules before detector_index
                let index = d.remove("detector_index");
                d.remove("custom_rules");
                if !rs.is_empty() {
                    d.insert("custom_rules".into(), Value::Array(rs.clone()));
                }
                if let Some(i) = index {
                    d.insert("detector_index".into(), i);
                }
            }
        }
        ac.insert("detectors".into(), Value::Array(ds));
    }
    if let Some(pw) = &u.prune {
        let bs = parse_time(ac.get("bucket_span").and_then(Value::as_str).unwrap_or("5m"), "bucket_span")?;
        check_prune_window(pw, &bs)?;
        ac.insert("model_prune_window".into(), json!(pw.rep));
    }
    if let Some(cf) = u.fields.get("categorization_filters") {
        let filters: Vec<String> = cf.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
        if ac.contains_key("categorization_field_name") {
            ac.insert("categorization_analyzer".into(), standard_categorization_analyzer(&filters));
        }
    }
    if let Some(ppc) = u.fields.get("per_partition_categorization") {
        if let Some(cur) = ac.get_mut("per_partition_categorization").and_then(Value::as_object_mut) {
            if let Some(s) = ppc.get("stop_on_warn") {
                cur.insert("stop_on_warn".into(), s.clone());
            }
        }
    }
    j.insert("analysis_config".into(), Value::Object(ac));
    if u.fields.contains_key("analysis_limits") {
        if open {
            return Err(bad_request("Cannot update analysis_limits while the job is open"));
        }
        let mut lim = j.get("analysis_limits").and_then(Value::as_object).cloned().unwrap_or_default();
        if let Some(m) = u.mml {
            if let Some(max) = max_mml_mb {
                if m > max {
                    return Err(EsError::illegal_argument(format!(
                        "model_memory_limit [{}] must be less than the value of the xpack.ml.max_model_memory_limit setting [{}]",
                        bytes_to_string(m * MB),
                        bytes_to_string(max * MB)
                    )));
                }
            }
            lim.insert("model_memory_limit".into(), json!(format!("{m}mb")));
        }
        if let Some(c) = u.cel {
            lim.insert("categorization_examples_limit".into(), json!(c));
        }
        j.insert("analysis_limits".into(), Value::Object(lim));
    }
    for k in ["renormalization_window_days", "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days"] {
        if let Some(n) = u.fields.get(k).and_then(Value::as_i64) {
            if n < 0 {
                return Err(EsError::illegal_argument(format!("{k} cannot be less than 0. Value = {n}")));
            }
        }
    }
    for k in ["model_plot_config", "renormalization_window_days", "background_persist_interval", "model_snapshot_retention_days", "daily_model_snapshot_retention_after_days", "results_retention_days", "custom_settings", "allow_lazy_open"] {
        if let Some(v) = u.fields.get(k) {
            j.insert(k.into(), v.clone());
        }
    }
    let msrd = j.get("model_snapshot_retention_days").and_then(Value::as_i64).unwrap_or(10);
    let daily = j.get("daily_model_snapshot_retention_after_days").and_then(Value::as_i64).unwrap_or(1);
    check_snapshot_retention(msrd, daily)?;
    // Re-render in canonical order.
    Ok(render(&Value::Object(j), None, false))
}

// ---------------------------------------------------------------------------
// Validate APIs
// ---------------------------------------------------------------------------

/// Parse + validate an `analysis_config` object (estimate API).
pub fn parse_analysis_config_only(v: &Value, p: &Pos) -> PR<Value> {
    parse_analysis_config(v, p)
}

/// `POST _ml/anomaly_detectors/_validate/detector`.
pub fn validate_detector_body(body: &Value, raw: &str) -> Result<()> {
    let pos = xc::positions(raw);
    if body.is_null() {
        return Err(EsError::new(400, "parse_exception", "request body is required"));
    }
    parse_detector(body, &pos).map(|_| ()).map_err(|pe| pe.e)
}

/// `POST _ml/anomaly_detectors/_validate`: the job checks of a put
/// (without the cluster-state ones).
pub fn validate_job_body(body: &Value, raw: &str) -> Result<()> {
    let pos = xc::positions(raw);
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        other => return Err(xparse(format!("[1:1] [job_details] Expected START_OBJECT but was: {}", xc::token_name(other)))),
    };
    let id = o.get("job_id").and_then(Value::as_str).unwrap_or("").to_string();
    let mut copy = o.clone();
    if id.is_empty() {
        // the id is optional here; validate everything else
        copy.remove("job_id");
        return parse_job_body("ok", &copy, &pos).map(|_| ()).map_err(|pe| pe.e);
    }
    parse_job_body(&id, &copy, &pos).map(|_| ()).map_err(|pe| pe.e)
}
