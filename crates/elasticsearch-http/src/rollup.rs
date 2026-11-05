//! Rollup (`_rollup/*`, `_rollup_search`) and searchable snapshots
//! (`_searchable_snapshots/*`) APIs.
//!
//! Rollup is deprecated in 8.x but still works: a cluster accepts new jobs
//! once it has rollup usage, i.e. an index whose mapping `_meta` carries a
//! `_rollup` key (every job's rollup index does, so the first job needs a
//! pre-existing one). Job configs live in the extension state (ES: persistent
//! tasks) and are mirrored into the rollup index's `_meta._rollup`, which is
//! what the caps APIs and `_rollup_search` read, exactly as ES does. A job's
//! indexer is started and stopped by the API.
//!
//! `_rollup_search` is `TransportRollupSearchAction` in Rust: the request's
//! aggregations pick the best job per `RollupJobIdentifierUtils`, are
//! translated onto the rollup document fields (`RollupRequestTranslator`),
//! run against the rollup index filtered to that job, and the result is
//! unrolled back into the requested shape (`RollupResponseTranslator`).
//!
//! The index-level searchable snapshot APIs resolve their targets.

use std::collections::HashMap;
use crate::{method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::date::parse_time_value;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::field_caps::{field_caps, FieldCapsOpts};
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::node::{Node, ResolveOpts, ES_VERSION, NODE_ID};
use elasticsearch_engine::search_ext::ext;
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

/// URL parameters of a rollup search (it parses the request as `_search` does).
const SEARCH_PARAMS: &[&str] = &[
    "analyzer", "analyze_wildcard", "ccs_minimize_roundtrips", "default_operator", "df", "explain", "stored_fields", "docvalue_fields", "from",
    "force_synthetic_source", "ignore_unavailable", "ignore_throttled", "allow_no_indices", "expand_wildcards", "lenient", "preference", "q", "routing",
    "scroll", "search_type", "size", "sort", "_source", "_source_excludes", "_source_includes", "terminate_after", "stats", "suggest_field",
    "suggest_mode", "suggest_size", "suggest_text", "timeout", "track_scores", "track_total_hits", "allow_partial_search_results", "typed_keys",
    "version", "seq_no_primary_term", "request_cache", "batched_reduce_size", "max_concurrent_shard_requests", "pre_filter_shard_size",
    "rest_total_hits_as_int", "min_compatible_shard_node", "include_named_queries_score",
];

const SUPPORTED_METRICS: &[&str] = &["max", "min", "sum", "avg", "value_count"];
const SUPPORTED_DATE_METRICS: &[&str] = &["max", "min", "value_count"];
const DATE_TYPES: &[&str] = &["date", "date_nanos"];
const NUMERIC_TYPES: &[&str] = &["long", "integer", "short", "byte", "double", "float", "half_float", "scaled_float", "unsigned_long"];
const PARSER_NAME: &str = "xpack/rollup/jobconfig";

/// IANA names ES's `DateUtils.DEPRECATED_LONG_TIMEZONES` maps to their
/// modern equivalents (a job created with one gets a deprecation warning).
const DEPRECATED_TIMEZONES: &[(&str, &str)] = &[
    ("Australia/ACT", "Australia/Sydney"), ("Australia/LHI", "Australia/Lord_Howe"), ("Australia/NSW", "Australia/Sydney"),
    ("Australia/North", "Australia/Darwin"), ("Australia/Queensland", "Australia/Brisbane"), ("Australia/South", "Australia/Adelaide"),
    ("Australia/Tasmania", "Australia/Hobart"), ("Australia/Victoria", "Australia/Melbourne"), ("Australia/West", "Australia/Perth"),
    ("Brazil/Acre", "America/Rio_Branco"), ("Brazil/DeNoronha", "America/Noronha"), ("Brazil/East", "America/Sao_Paulo"),
    ("Brazil/West", "America/Manaus"), ("Canada/Atlantic", "America/Halifax"), ("Canada/Central", "America/Winnipeg"),
    ("Canada/Eastern", "America/Toronto"), ("Canada/Mountain", "America/Edmonton"), ("Canada/Newfoundland", "America/St_Johns"),
    ("Canada/Pacific", "America/Vancouver"), ("Canada/Saskatchewan", "America/Regina"), ("Canada/Yukon", "America/Whitehorse"),
    ("Chile/Continental", "America/Santiago"), ("Chile/EasterIsland", "Pacific/Easter"), ("Cuba", "America/Havana"), ("Egypt", "Africa/Cairo"),
    ("Eire", "Europe/Dublin"), ("GB", "Europe/London"), ("GB-Eire", "Europe/London"), ("Hongkong", "Asia/Hong_Kong"),
    ("Iceland", "Atlantic/Reykjavik"), ("Iran", "Asia/Tehran"), ("Israel", "Asia/Jerusalem"), ("Jamaica", "America/Jamaica"),
    ("Japan", "Asia/Tokyo"), ("Kwajalein", "Pacific/Kwajalein"), ("Libya", "Africa/Tripoli"), ("Mexico/BajaNorte", "America/Tijuana"),
    ("Mexico/BajaSur", "America/Mazatlan"), ("Mexico/General", "America/Mexico_City"), ("NZ", "Pacific/Auckland"), ("NZ-CHAT", "Pacific/Chatham"),
    ("Navajo", "America/Denver"), ("PRC", "Asia/Shanghai"), ("Poland", "Europe/Warsaw"), ("Portugal", "Europe/Lisbon"), ("ROK", "Asia/Seoul"),
    ("Singapore", "Asia/Singapore"), ("Turkey", "Europe/Istanbul"), ("US/Alaska", "America/Anchorage"), ("US/Aleutian", "America/Adak"),
    ("US/Arizona", "America/Phoenix"), ("US/Central", "America/Chicago"), ("US/East-Indiana", "America/Indiana/Indianapolis"),
    ("US/Eastern", "America/New_York"), ("US/Hawaii", "Pacific/Honolulu"), ("US/Indiana-Starke", "America/Indiana/Knox"),
    ("US/Michigan", "America/Detroit"), ("US/Mountain", "America/Denver"), ("US/Pacific", "America/Los_Angeles"), ("US/Samoa", "Pacific/Pago_Pago"),
    ("W-SU", "Europe/Moscow"),
];

fn modern_timezone(tz: &str) -> Option<&'static str> {
    DEPRECATED_TIMEZONES.iter().find(|(old, _)| *old == tz).map(|(_, new)| *new)
}

/// Time zones with the same rules compare equal (ES compares `ZoneRules`).
fn canonical_timezone(tz: &str) -> String {
    match modern_timezone(tz) {
        Some(m) => m.to_string(),
        None => match tz {
            "Z" | "UTC" | "Etc/UTC" | "Etc/GMT" | "GMT" | "+00:00" | "-00:00" | "+0000" | "Zulu" | "Universal" | "Etc/Universal" | "Etc/Zulu" | "UCT" | "Etc/UCT" => "UTC".to_string(),
            other => other.to_string(),
        },
    }
}

// ------------------------------------------------------------------ helpers

/// The XContent token name of a JSON value (ES parser error wording).
pub fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn ia(msg: impl Into<String>) -> EsError {
    EsError::illegal_argument(msg)
}

fn parse_err(msg: impl Into<String>) -> EsError {
    EsError::x_content_parse(format!("[1:2] [{PARSER_NAME}] {}", msg.into()))
}

/// Java's `Collection.toString()` of a list of strings.
fn java_list(items: &[&str]) -> String {
    format!("[{}]", items.join(", "))
}

/// Base duration (millis) of a calendar interval expression, per
/// `DateHistogramAggregationBuilder.DATE_FIELD_UNITS` (`ChronoUnit` durations).
fn calendar_unit_millis(s: &str) -> Option<i64> {
    Some(match s {
        "year" | "1y" => 31_556_952_000,
        "quarter" | "1q" => 7_889_238_000,
        "month" | "1M" => 2_629_746_000,
        "week" | "1w" => 604_800_000,
        "day" | "1d" => 86_400_000,
        "hour" | "1h" => 3_600_000,
        "minute" | "1m" => 60_000,
        "second" | "1s" => 1000,
        _ => return None,
    })
}

/// `Rounding.DateTimeUnit.estimateMillis` of a calendar unit.
fn calendar_unit_estimate(s: &str) -> Option<i64> {
    Some(match s {
        "year" | "1y" => 365 * 86_400_000,
        "quarter" | "1q" => 90 * 86_400_000,
        "month" | "1M" => 30 * 86_400_000,
        "week" | "1w" => 7 * 86_400_000,
        "day" | "1d" => 86_400_000,
        "hour" | "1h" => 3_600_000,
        "minute" | "1m" => 60_000,
        "second" | "1s" => 1000,
        _ => return None,
    })
}

fn fixed_millis(s: &str) -> Option<i64> {
    parse_time_value(s).filter(|v| *v > 0)
}

/// `DateHistogramInterval.estimateMillis()`.
fn estimate_millis(s: &str) -> i64 {
    calendar_unit_estimate(s).or_else(|| fixed_millis(s)).unwrap_or(i64::MAX)
}

fn validate_calendar_interval(request: &str, config: Option<&str>) -> bool {
    let Some(config) = config else { return false };
    match (calendar_unit_millis(request), calendar_unit_millis(config)) {
        (Some(r), Some(c)) => r >= c,
        _ => false,
    }
}

fn validate_fixed_interval(request: &str, config: Option<&str>) -> bool {
    let Some(config) = config else { return false };
    match (fixed_millis(request), fixed_millis(config)) {
        (Some(r), Some(c)) => r >= c && r % c == 0,
        _ => false,
    }
}

fn str_of<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

// --------------------------------------------------------------- job config

/// Parse and validate a `PUT _rollup/job/{id}` body into the normalised
/// config ES stores (`RollupJobConfig.toXContent`).
fn parse_config(path_id: &str, body: &Value) -> Result<Value> {
    let o = body.as_object().ok_or_else(|| EsError::x_content_parse(format!("[1:1] [{PARSER_NAME}] Expected [START_OBJECT] but found [{}]", token_name(body))))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "id" | "index_pattern" | "rollup_index" | "groups" | "metrics" | "timeout" | "cron" | "page_size") {
            return Err(parse_err(format!("unknown field [{k}]")));
        }
    }
    let text = |k: &str| -> Result<Option<String>> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(parse_err(format!("[{k}] doesn't support values of type: {}", token_name(other)))),
        }
    };
    let required = |k: &str| -> Result<String> { text(k)?.ok_or_else(|| ia(format!("Required [{k}]"))) };
    let id = text("id")?.unwrap_or_else(|| path_id.to_string());
    let index_pattern = required("index_pattern")?;
    let rollup_index = required("rollup_index")?;
    let cron = required("cron")?;
    let page_size = match o.get("page_size") {
        Some(Value::Number(n)) => n.as_i64().ok_or_else(|| parse_err("[page_size] out of range of int"))?,
        Some(Value::String(s)) => s.parse::<i64>().map_err(|_| parse_err(format!("failed to parse field [page_size]: For input string: \"{s}\"")))?,
        None => return Err(ia("Required [page_size]")),
        Some(other) => return Err(parse_err(format!("[page_size] doesn't support values of type: {}", token_name(other)))),
    };
    let groups = match o.get("groups") {
        None | Some(Value::Null) => None,
        Some(g) => Some(parse_groups(g)?),
    };
    let metrics = match o.get("metrics") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().map(parse_metric).collect::<Result<Vec<_>>>()?,
        Some(other) => return Err(parse_err(format!("[metrics] doesn't support values of type: {}", token_name(other)))),
    };
    let timeout = match text("timeout")? {
        Some(t) => {
            parse_time_value(&t).ok_or_else(|| ia(format!("failed to parse setting [timeout] with value [{t}] as a time value: unit is missing or unrecognized")))?;
            t
        }
        None => "20s".to_string(),
    };

    // RollupJobConfig's constructor checks.
    if id.is_empty() {
        return Err(ia("Id must be a non-null, non-empty string"));
    }
    if index_pattern.is_empty() {
        return Err(ia("Index pattern must be a non-null, non-empty string"));
    }
    for index in index_pattern.split(',').map(str::trim) {
        if index == "*" || index == "_all" {
            return Err(ia("Index pattern must not match all indices (as it would match it's own rollup index"));
        }
        if index.contains('*') && elasticsearch_engine::mapping::wildcard_match(index, &rollup_index) {
            return Err(ia("Index pattern would match rollup index name which is not allowed"));
        }
        if index == rollup_index {
            return Err(ia("Rollup index may not be the same as the index pattern"));
        }
    }
    if rollup_index.is_empty() {
        return Err(ia("Rollup index must be a non-null, non-empty string"));
    }
    if cron.is_empty() {
        return Err(ia("Cron schedule must be a non-null, non-empty string"));
    }
    if page_size <= 0 {
        return Err(ia("Page size is mandatory and  must be a positive long"));
    }
    if groups.is_none() && metrics.is_empty() {
        return Err(ia("At least one grouping or metric must be configured"));
    }
    let mut cfg = Map::new();
    cfg.insert("id".into(), json!(id));
    cfg.insert("index_pattern".into(), json!(index_pattern));
    cfg.insert("rollup_index".into(), json!(rollup_index));
    cfg.insert("cron".into(), json!(cron));
    if let Some(g) = groups {
        cfg.insert("groups".into(), g);
    }
    cfg.insert("metrics".into(), Value::Array(metrics));
    cfg.insert("timeout".into(), json!(timeout));
    cfg.insert("page_size".into(), json!(page_size));
    Ok(Value::Object(cfg))
}

fn parse_groups(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| parse_err("[groups] doesn't support values of type: ".to_string() + token_name(v)))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "date_histogram" | "histogram" | "terms") {
            return Err(EsError::x_content_parse(format!("[1:2] [groups] unknown field [{k}]")));
        }
    }
    let dh = o.get("date_histogram").filter(|v| !v.is_null()).ok_or_else(|| ia("Required [date_histogram]"))?;
    let mut out = Map::new();
    out.insert("date_histogram".into(), parse_date_histogram(dh)?);
    if let Some(h) = o.get("histogram").filter(|v| !v.is_null()) {
        let ho = h.as_object().ok_or_else(|| parse_err("[histogram] must be an object"))?;
        for k in ho.keys() {
            if !matches!(k.as_str(), "interval" | "fields") {
                return Err(EsError::x_content_parse(format!("[1:2] [histogram] unknown field [{k}]")));
            }
        }
        let interval = ho.get("interval").and_then(Value::as_i64).ok_or_else(|| ia("Required [interval]"))?;
        if interval <= 0 {
            return Err(ia("Interval must be a positive long"));
        }
        let fields = string_list(ho.get("fields"), "fields")?;
        if fields.is_empty() {
            return Err(ia("Fields must have at least one value"));
        }
        out.insert("histogram".into(), json!({"interval": interval, "fields": fields}));
    }
    if let Some(t) = o.get("terms").filter(|v| !v.is_null()) {
        let to = t.as_object().ok_or_else(|| parse_err("[terms] must be an object"))?;
        for k in to.keys() {
            if k != "fields" {
                return Err(EsError::x_content_parse(format!("[1:2] [terms] unknown field [{k}]")));
            }
        }
        let fields = string_list(to.get("fields"), "fields")?;
        if fields.is_empty() {
            return Err(ia("Fields must have at least one value"));
        }
        out.insert("terms".into(), json!({"fields": fields}));
    }
    Ok(Value::Object(out))
}

fn string_list(v: Option<&Value>, what: &str) -> Result<Vec<String>> {
    match v {
        None | Some(Value::Null) => Err(ia(format!("Required [{what}]"))),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| parse_err(format!("[{what}] must be a list of strings")))).collect(),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(other) => Err(parse_err(format!("[{what}] doesn't support values of type: {}", token_name(other)))),
    }
}

fn parse_date_histogram(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| parse_err("[date_histogram] must be an object"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "field" | "interval" | "calendar_interval" | "fixed_interval" | "delay" | "time_zone") {
            return Err(EsError::x_content_parse(format!("[1:2] [date_histogram] unknown field [{k}]")));
        }
    }
    let field = str_of(v, "field").ok_or_else(|| ia("Required [field]"))?;
    let old = str_of(v, "interval");
    let cal = str_of(v, "calendar_interval");
    let fix = str_of(v, "fixed_interval");
    if old.is_some() && (cal.is_some() || fix.is_some()) {
        return Err(ia("Cannot use [interval] with [fixed_interval] or [calendar_interval] configuration options."));
    }
    if cal.is_some() && fix.is_some() {
        return Err(ia("Cannot set both [fixed_interval] and [calendar_interval] configuration options."));
    }
    let (kind, interval) = if let Some(c) = cal {
        if calendar_unit_millis(c).is_none() {
            return Err(ia(format!("The supplied interval [{c}] could not be parsed as a calendar interval.")));
        }
        ("calendar_interval", c)
    } else if let Some(f) = fix {
        if fixed_millis(f).is_none() {
            return Err(ia(format!("failed to parse setting [interval] with value [{f}] as a time value: unit is missing or unrecognized")));
        }
        ("fixed_interval", f)
    } else if let Some(i) = old {
        ("interval", i)
    } else {
        return Err(ia("An interval is required.  Use [fixed_interval] or [calendar_interval]."));
    };
    let mut out = Map::new();
    out.insert(kind.into(), json!(interval));
    out.insert("field".into(), json!(field));
    if let Some(d) = str_of(v, "delay") {
        out.insert("delay".into(), json!(d));
    }
    out.insert("time_zone".into(), json!(str_of(v, "time_zone").unwrap_or("UTC")));
    Ok(Value::Object(out))
}

fn parse_metric(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| parse_err("[metrics] must be a list of objects"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "field" | "metrics") {
            return Err(EsError::x_content_parse(format!("[1:2] [metrics] unknown field [{k}]")));
        }
    }
    let field = str_of(v, "field").ok_or_else(|| ia("Required [field]"))?;
    if field.is_empty() {
        return Err(ia("Field must be a non-null, non-empty string"));
    }
    let metrics = string_list(o.get("metrics"), "metrics")?;
    if metrics.is_empty() {
        return Err(ia("Metrics must be a non-null, non-empty array of strings"));
    }
    for m in &metrics {
        if !SUPPORTED_METRICS.contains(&m.as_str()) {
            return Err(ia(format!("Unsupported metric [{m}]. Supported metrics include: {}", java_list(SUPPORTED_METRICS))));
        }
    }
    Ok(json!({"field": field, "metrics": metrics}))
}

/// Every field a config reads (groups and metrics), for the field caps lookup.
fn config_fields(cfg: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |f: &str| {
        if !out.iter().any(|x| x == f) {
            out.push(f.to_string());
        }
    };
    if let Some(dh) = cfg.pointer("/groups/date_histogram/field").and_then(Value::as_str) {
        push(dh);
    }
    for path in ["/groups/histogram/fields", "/groups/terms/fields"] {
        if let Some(a) = cfg.pointer(path).and_then(Value::as_array) {
            a.iter().filter_map(Value::as_str).for_each(&mut push);
        }
    }
    if let Some(a) = cfg.get("metrics").and_then(Value::as_array) {
        a.iter().filter_map(|m| m["field"].as_str()).for_each(&mut push);
    }
    out
}

/// `RollupJobConfig.validateMappings`: the configured fields against the
/// field caps of the indices the pattern matches (one error per problem).
fn validate_mappings(cfg: &Value, caps: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let caps_of = |field: &str| -> Option<&Map<String, Value>> { caps.get(field).and_then(Value::as_object).filter(|m| !m.is_empty()) };
    let types_of = |m: &Map<String, Value>| -> String { format!("[{}]", m.keys().map(String::as_str).collect::<Vec<_>>().join(", ")) };
    let aggregatable = |c: &Value| c["aggregatable"].as_bool().unwrap_or(false);
    let date_types = DATE_TYPES.join(",");

    if let Some(field) = cfg.pointer("/groups/date_histogram/field").and_then(Value::as_str) {
        match caps_of(field) {
            Some(m) => {
                let mut matched = false;
                for dt in DATE_TYPES {
                    if m.contains_key(*dt) && m.len() == 1 {
                        matched = true;
                        if !aggregatable(&m[*dt]) {
                            errs.push(format!("The field [{field}] must be aggregatable across all indices, but is not."));
                        }
                    }
                }
                if !matched {
                    errs.push(format!(
                        "The field referenced by a date_histo group must be one of type [{date_types}] across all indices in the index pattern.  Found: {} for field [{field}]",
                        types_of(m)
                    ));
                }
            }
            None => errs.push(format!("Could not find one of [{date_types}] fields with name [{field}] in any of the indices matching the index pattern.")),
        }
    }
    if let Some(fields) = cfg.pointer("/groups/histogram/fields").and_then(Value::as_array) {
        for field in fields.iter().filter_map(Value::as_str) {
            match caps_of(field) {
                Some(m) => {
                    for (ty, c) in m {
                        if NUMERIC_TYPES.contains(&ty.as_str()) {
                            if !aggregatable(c) {
                                errs.push(format!("The field [{field}] must be aggregatable across all indices, but is not."));
                            }
                        } else {
                            errs.push(format!("The field referenced by a histo group must be a [numeric] type, but found {} for field [{field}]", types_of(m)));
                        }
                    }
                }
                None => errs.push(format!("Could not find a [numeric] field with name [{field}] in any of the indices matching the index pattern.")),
            }
        }
    }
    if let Some(fields) = cfg.pointer("/groups/terms/fields").and_then(Value::as_array) {
        for field in fields.iter().filter_map(Value::as_str) {
            match caps_of(field) {
                Some(m) => {
                    for (ty, c) in m {
                        if ty == "keyword" || ty == "text" || NUMERIC_TYPES.contains(&ty.as_str()) {
                            if !aggregatable(c) {
                                errs.push(format!("The field [{field}] must be aggregatable across all indices, but is not."));
                            }
                        } else {
                            errs.push(format!("The field referenced by a terms group must be a [numeric] or [keyword/text] type, but found {} for field [{field}]", types_of(m)));
                        }
                    }
                }
                None => errs.push(format!("Could not find a [numeric] or [keyword/text] field with name [{field}] in any of the indices matching the index pattern.")),
            }
        }
    }
    if let Some(metrics) = cfg.get("metrics").and_then(Value::as_array) {
        for mc in metrics {
            let Some(field) = mc["field"].as_str() else { continue };
            let wanted: Vec<&str> = mc["metrics"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            match caps_of(field) {
                Some(m) => {
                    for (ty, c) in m {
                        if NUMERIC_TYPES.contains(&ty.as_str()) {
                            if !aggregatable(c) {
                                errs.push(format!("The field [{field}] must be aggregatable across all indices, but is not."));
                            }
                        } else if DATE_TYPES.contains(&ty.as_str()) {
                            if !aggregatable(c) {
                                errs.push(format!("The field [{field}] must be aggregatable across all indices, but is not."));
                            } else if !wanted.iter().all(|w| SUPPORTED_DATE_METRICS.contains(w)) {
                                let unsupported: Vec<&str> = wanted.iter().copied().filter(|w| !SUPPORTED_DATE_METRICS.contains(w)).collect();
                                errs.push(format!(
                                    "Only the metrics {} are supported for [{ty}] types, but unsupported metrics {} supplied for field [{field}]",
                                    java_list(SUPPORTED_DATE_METRICS),
                                    java_list(&unsupported)
                                ));
                            }
                        } else {
                            errs.push(format!("The field referenced by a metric group must be a [numeric] or [{date_types}] type, but found {} for field [{field}]", types_of(m)));
                        }
                    }
                }
                None => errs.push(format!("Could not find a [numeric] or [{date_types}] field with name [{field}] in any of the indices matching the index pattern.")),
            }
        }
    }
    errs
}

// ---------------------------------------------------------------- job caps

/// `RollupJobCaps`: what one job rolled up, per source field.
#[derive(Clone, Debug)]
struct JobCaps {
    job_id: String,
    rollup_index: String,
    index_pattern: String,
    /// source field → the agg caps on it (`{"agg": .., ...}`), in config order.
    fields: BTreeMap<String, Vec<Map<String, Value>>>,
}

impl JobCaps {
    fn from_config(cfg: &Value) -> Option<JobCaps> {
        let job_id = str_of(cfg, "id")?.to_string();
        let rollup_index = str_of(cfg, "rollup_index")?.to_string();
        let index_pattern = str_of(cfg, "index_pattern")?.to_string();
        let mut fields: BTreeMap<String, Vec<Map<String, Value>>> = BTreeMap::new();
        if let Some(dh) = cfg.pointer("/groups/date_histogram").and_then(Value::as_object) {
            let field = dh.get("field").and_then(Value::as_str)?;
            let mut cap = Map::new();
            cap.insert("agg".into(), json!("date_histogram"));
            for k in ["interval", "calendar_interval", "fixed_interval"] {
                if let Some(v) = dh.get(k) {
                    cap.insert(k.into(), v.clone());
                }
            }
            if let Some(d) = dh.get("delay") {
                cap.insert("delay".into(), d.clone());
            }
            cap.insert("time_zone".into(), dh.get("time_zone").cloned().unwrap_or(json!("UTC")));
            fields.entry(field.to_string()).or_default().push(cap);
        }
        if let Some(h) = cfg.pointer("/groups/histogram").and_then(Value::as_object) {
            let mut cap = Map::new();
            cap.insert("agg".into(), json!("histogram"));
            cap.insert("interval".into(), h.get("interval").cloned().unwrap_or(Value::Null));
            for f in h.get("fields").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                fields.entry(f.to_string()).or_default().push(cap.clone());
            }
        }
        if let Some(t) = cfg.pointer("/groups/terms/fields").and_then(Value::as_array) {
            let mut cap = Map::new();
            cap.insert("agg".into(), json!("terms"));
            for f in t.iter().filter_map(Value::as_str) {
                fields.entry(f.to_string()).or_default().push(cap.clone());
            }
        }
        for mc in cfg.get("metrics").and_then(Value::as_array).into_iter().flatten() {
            let Some(field) = mc["field"].as_str() else { continue };
            for m in mc["metrics"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                let mut cap = Map::new();
                cap.insert("agg".into(), json!(m));
                fields.entry(field.to_string()).or_default().push(cap);
            }
        }
        Some(JobCaps { job_id, rollup_index, index_pattern, fields })
    }

    fn to_json(&self) -> Value {
        let fields: Map<String, Value> = self.fields.iter().map(|(f, aggs)| (f.clone(), Value::Array(aggs.iter().cloned().map(Value::Object).collect()))).collect();
        json!({"job_id": self.job_id, "rollup_index": self.rollup_index, "index_pattern": self.index_pattern, "fields": fields})
    }

    /// The caps of `field` whose `agg` is `agg`.
    fn aggs_of<'a>(&'a self, field: &str, agg: &str) -> impl Iterator<Item = &'a Map<String, Value>> + 'a {
        let agg = agg.to_string();
        self.fields.get(field).into_iter().flatten().filter(move |c| c.get("agg").and_then(Value::as_str) == Some(agg.as_str()))
    }
}

/// The jobs recorded in an index's mapping `_meta._rollup` (none for a live index).
fn index_job_caps(svc: &IndexService) -> Vec<JobCaps> {
    let meta = svc.meta().mappings.meta;
    let Some(rollup) = meta.as_ref().and_then(|m| m.get("_rollup")).and_then(Value::as_object) else { return Vec::new() };
    rollup.values().filter_map(JobCaps::from_config).collect()
}

fn has_rollup_indices(node: &Node) -> bool {
    node.all_indices().iter().any(|svc| svc.meta().mappings.meta.as_ref().and_then(|m| m.get("_rollup")).is_some())
}

fn rollable_caps_json(by_key: BTreeMap<String, Vec<JobCaps>>) -> Value {
    let mut out = Map::new();
    for (k, mut caps) in by_key {
        caps.sort_by(|a, b| a.job_id.cmp(&b.job_id));
        out.insert(k, json!({"rollup_jobs": caps.iter().map(JobCaps::to_json).collect::<Vec<_>>()}));
    }
    Value::Object(out)
}

// ---------------------------------------------------------------- job APIs

fn job_json(id: &str, job: &Value) -> Value {
    json!({
        "config": job["config"],
        "status": {"job_state": job["state"].as_str().unwrap_or("stopped")},
        "stats": {
            "pages_processed": 0, "documents_processed": 0, "rollups_indexed": 0, "trigger_count": 0,
            "index_time_in_ms": 0, "index_total": 0, "index_failures": 0,
            "search_time_in_ms": 0, "search_total": 0, "search_failures": 0,
            "processing_time_in_ms": 0, "processing_total": 0
        },
        "_id": id,
    })
}

fn put_job(node: &Node, req: &Req, id: &str) -> R {
    let body = req.json()?;
    if body.is_null() {
        return Err(EsError::new(400, "parse_exception", "request body is required"));
    }
    let cfg = parse_config(id, &body)?;
    let id = cfg["id"].as_str().unwrap_or(id).to_string();
    let st = ext(node);
    let mut warnings = Vec::new();
    let tz = cfg.pointer("/groups/date_histogram/time_zone").and_then(Value::as_str).unwrap_or("UTC");
    if let Some(modern) = modern_timezone(tz) {
        warnings.push(format!("Creating Rollup job [{id}] with timezone [{tz}], but [{tz}] has been deprecated by the IANA.  Use [{modern}] instead."));
    }
    if st.rollup_jobs.read().is_empty() && !has_rollup_indices(node) {
        return Err(ia("new rollup jobs are not allowed in clusters that don't have any rollup usage, since rollup has been deprecated"));
    }
    // Field caps of the indices the pattern matches (ES: strict expand open).
    let index_pattern = cfg["index_pattern"].as_str().unwrap_or_default().to_string();
    let targets = node.resolve(&index_pattern, ResolveOpts::default())?;
    let fields = config_fields(&cfg);
    let caps = field_caps(&targets, &FieldCapsOpts { fields: &fields, include_unmapped: false, filters: &[], types: &[], runtime_mappings: None, include_empty_fields: true })?;
    let errs = validate_mappings(&cfg, &caps["fields"]);
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let rollup_index = cfg["rollup_index"].as_str().unwrap_or_default().to_string();
    match node.get_index(&rollup_index) {
        None => {
            let mappings = json!({
                "_meta": {"rollup-version": "", "_rollup": {id.clone(): cfg.clone()}},
                "dynamic_templates": [
                    {"strings": {"match_mapping_type": "string", "mapping": {"type": "keyword"}}},
                    {"date_histograms": {"path_match": "*.date_histogram.timestamp", "mapping": {"type": "date"}}}
                ]
            });
            if let Err(e) = node.create_index(&rollup_index, &json!({"mappings": mappings})) {
                if e.kind == "resource_already_exists_exception" {
                    update_rollup_meta(node, &rollup_index, &id, &cfg)?;
                } else {
                    return Err(EsError::new(500, "runtime_exception", format!("Could not create index for rollup job [{id}]")).caused_by(e));
                }
            }
        }
        Some(_) => update_rollup_meta(node, &rollup_index, &id, &cfg)?,
    }
    {
        let mut jobs = st.rollup_jobs.write();
        if jobs.contains_key(&id) {
            return Err(EsError::new(409, "status_exception", format!("Cannot create job [{id}] because it has already been created (task exists)")));
        }
        jobs.insert(id.clone(), json!({"config": cfg, "state": "stopped"}));
    }
    st.persist()?;
    let mut r = Resp::ack();
    for w in warnings {
        r.headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"{w}\"", crate::cluster::BUILD_HASH)));
    }
    Ok(r)
}

/// Record a job in an existing rollup index's `_meta._rollup`
/// (`TransportPutRollupJobAction.updateMapping`).
fn update_rollup_meta(node: &Node, rollup_index: &str, id: &str, cfg: &Value) -> Result<()> {
    let svc = node.get_index(rollup_index).ok_or_else(|| EsError::new(500, "runtime_exception", format!("Could not update mappings for rollup job [{id}]")))?;
    let Some(mut meta) = svc.meta().mappings.meta else {
        return Err(EsError::new(
            500,
            "runtime_exception",
            format!("Rollup data cannot be added to existing indices that contain non-rollup data (expected to find _meta key in mapping of rollup index [{rollup_index}] but not found)."),
        ));
    };
    let Some(rollup) = meta.get_mut("_rollup").and_then(Value::as_object_mut) else {
        return Err(EsError::new(
            500,
            "runtime_exception",
            format!("Rollup data cannot be added to existing indices that contain non-rollup data (expected to find rollup meta key [_rollup] in mapping of rollup index [{rollup_index}] but not found)."),
        ));
    };
    if rollup.contains_key(id) {
        return Err(EsError::new(409, "status_exception", format!("Cannot create rollup job [{id}] because job was previously created (existing metadata).")));
    }
    rollup.insert(id.to_string(), cfg.clone());
    svc.put_mapping(&json!({"_meta": meta}))
}

fn get_jobs(node: &Node, id: &str) -> R {
    let st = ext(node);
    let jobs = st.rollup_jobs.read();
    let out: Vec<Value> = jobs
        .iter()
        .filter(|(k, _)| id == "_all" || (id.contains('*') && elasticsearch_engine::mapping::wildcard_match(id, k)) || id.split(',').any(|p| p == k.as_str()))
        .map(|(k, j)| {
            let mut v = job_json(k, j);
            v.as_object_mut().map(|m| m.remove("_id"));
            v
        })
        .collect();
    Ok(Resp::ok(json!({"jobs": out})))
}

fn delete_job(node: &Node, id: &str) -> R {
    let st = ext(node);
    let state = st.rollup_jobs.read().get(id).map(|j| j["state"].as_str().unwrap_or("stopped").to_string());
    let Some(state) = state else {
        return Err(EsError::resource_not_found(format!("the task with id [{id}] doesn't exist")));
    };
    if state != "stopped" {
        let reason = format!("Could not delete job [{id}] because indexer state is [{}].  Job must be [STOPPED] before deletion.", state.to_uppercase());
        return Ok(Resp::json(
            500,
            json!({
                "task_failures": [{"task_id": 0, "node_id": NODE_ID, "status": "INTERNAL_SERVER_ERROR", "reason": {"type": "illegal_state_exception", "reason": reason}}],
                "acknowledged": false
            }),
        ));
    }
    st.rollup_jobs.write().remove(id);
    st.persist()?;
    Ok(Resp::ack())
}

fn set_job_state(node: &Node, id: &str, start: bool) -> R {
    let st = ext(node);
    {
        let mut jobs = st.rollup_jobs.write();
        let Some(job) = jobs.get_mut(id) else {
            return Err(EsError::resource_not_found(format!("Task for Rollup Job [{id}] not found")));
        };
        job["state"] = json!(if start { "started" } else { "stopped" });
    }
    st.persist()?;
    Ok(Resp::ok(if start { json!({"started": true}) } else { json!({"stopped": true}) }))
}

/// Number of rollup jobs (xpack usage).
pub fn job_count(node: &Node) -> usize {
    ext(node).rollup_jobs.read().len()
}

/// `/_rollup/...` (no index).
pub async fn handle_global(node: Arc<Node>, req: Req) -> R {
    let m = req.method.clone();
    match (req.seg(1), req.segs.len()) {
        // ES answers a bare `_rollup/job` / `_rollup/data` GET with a 405
        // naming POST, and has no POST handler there.
        ("job" | "data", 2) if m == Method::POST => crate::not_found_handler(&req),
        ("job" | "data", 2) => method_not_allowed(&req, &["POST"]),
        ("job", 3) => {
            crate::util::check_params(&req, &[])?;
            let id = req.seg(2).to_string();
            match m {
                Method::GET => crate::blocking(move || get_jobs(&node, &id)).await,
                Method::PUT => crate::blocking(move || put_job(&node, &req, &id)).await,
                Method::DELETE => crate::blocking(move || delete_job(&node, &id)).await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("job", 4) if matches!(req.seg(3), "_start" | "_stop") => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let start = req.seg(3) == "_start";
            crate::util::check_params(&req, if start { &[] } else { &["wait_for_completion", "timeout"] })?;
            let id = req.seg(2).to_string();
            crate::blocking(move || set_job_state(&node, &id, start)).await
        }
        // GET _rollup/data/{id}: caps of the jobs whose index pattern is `id`
        // (`_all`: every job, keyed by its index pattern).
        ("data", 3) if m == Method::GET => {
            crate::util::check_params(&req, &[])?;
            let id = req.seg(2).to_string();
            crate::blocking(move || {
                let mut by_pattern: BTreeMap<String, Vec<JobCaps>> = BTreeMap::new();
                for svc in node.all_indices() {
                    for cap in index_job_caps(&svc) {
                        if id == "_all" {
                            by_pattern.entry(cap.index_pattern.clone()).or_default().push(cap);
                        } else if cap.index_pattern == id {
                            by_pattern.entry(id.clone()).or_default().push(cap);
                        }
                    }
                }
                Ok(Resp::ok(rollable_caps_json(by_pattern)))
            })
            .await
        }
        ("data", 3) => method_not_allowed(&req, &["GET"]),
        _ => crate::not_found_handler(&req),
    }
}

/// `/{index}/_rollup/data`: rollup caps of the rollup indices in `index`.
pub async fn index_caps(node: Arc<Node>, req: Req, index: String) -> R {
    if req.seg(2) != "data" || req.segs.len() != 3 {
        return crate::not_found_handler(&req);
    }
    if req.method != Method::GET {
        return method_not_allowed(&req, &["GET"]);
    }
    crate::util::check_params(&req, &["ignore_unavailable", "allow_no_indices", "expand_wildcards"])?;
    crate::blocking(move || {
        let targets = node.resolve(&index, ResolveOpts::from_params(&req.params))?;
        let mut by_index: BTreeMap<String, Vec<JobCaps>> = BTreeMap::new();
        let mut seen = std::collections::HashSet::new();
        for t in &targets {
            if !seen.insert(t.svc.name.clone()) {
                continue;
            }
            for cap in index_job_caps(&t.svc) {
                by_index.entry(cap.rollup_index.clone()).or_default().push(cap);
            }
        }
        Ok(Resp::ok(rollable_caps_json(by_index)))
    })
    .await
}

// ------------------------------------------------------------ rollup search

/// `/{index}/_rollup_search` (and `/_rollup_search`).
pub async fn search(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    if !matches!(req.method, Method::GET | Method::POST) {
        return method_not_allowed(&req, &["GET", "POST"]);
    }
    crate::util::check_params(&req, SEARCH_PARAMS)?;
    crate::blocking(move || rollup_search(&node, &req, index.as_deref())).await
}

fn agg_class_name(ty: &str) -> String {
    let mut s = String::new();
    for part in ty.split('_') {
        let mut cs = part.chars();
        if let Some(c) = cs.next() {
            s.push(c.to_ascii_uppercase());
            s.extend(cs);
        }
    }
    s + "AggregationBuilder"
}

/// (type, definition, sub-aggregations) of one aggregation request.
fn agg_parts(name: &str, agg: &Value) -> Result<(String, Value, Option<Map<String, Value>>)> {
    let o = agg.as_object().ok_or_else(|| ia(format!("Aggregation [{name}] must be an object")))?;
    let subs = o.get("aggs").or_else(|| o.get("aggregations")).and_then(Value::as_object).cloned();
    let ty = o.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).ok_or_else(|| ia(format!("Aggregation [{name}] has no type")))?;
    Ok((ty.clone(), o[ty].clone(), subs))
}

fn field_of(ty: &str, def: &Value) -> Result<String> {
    def.get("field").and_then(Value::as_str).map(str::to_string).ok_or_else(|| ia(format!("Rollup requires [{ty}] aggregations to name a [field]")))
}

/// The date histogram's interval as `(calendar, fixed)`, validated.
fn date_histo_intervals(def: &Value) -> Result<(Option<String>, Option<String>)> {
    let cal = str_of(def, "calendar_interval").map(str::to_string);
    let fix = str_of(def, "fixed_interval").map(str::to_string);
    if let Some(c) = &cal {
        if calendar_unit_millis(c).is_none() {
            return Err(EsError::x_content_parse("[1:2] [date_histogram] failed to parse field [calendar_interval]")
                .caused_by(ia(format!("The supplied interval [{c}] could not be parsed as a calendar interval."))));
        }
    }
    if let Some(f) = &fix {
        if fixed_millis(f).is_none() {
            return Err(EsError::x_content_parse("[1:2] [date_histogram] failed to parse field [fixed_interval]")
                .caused_by(ia(format!("failed to parse setting [date_histogram.fixedInterval] with value [{f}] as a time value: unit is missing or unrecognized"))));
        }
    }
    if cal.is_none() && fix.is_none() {
        if def.get("interval").is_some() {
            return Err(EsError::x_content_parse("[1:2] [date_histogram] unknown field [interval]"));
        }
        return Err(ia("An interval of some variety must be configured on the date_histogram aggregation."));
    }
    Ok((cal, fix))
}

/// `RollupJobIdentifierUtils.COMPARATOR`: coarser jobs first.
fn caps_order(a: &JobCaps, b: &JobCaps) -> Ordering {
    if a.job_id == b.job_id {
        return Ordering::Equal;
    }
    let weigh = |c: &JobCaps| -> (i64, f32, i64) {
        let mut time = i64::MAX;
        let mut histo = 0f32;
        let mut n = 0;
        let mut terms = 0i64;
        for aggs in c.fields.values() {
            for agg in aggs {
                match agg.get("agg").and_then(Value::as_str) {
                    Some("date_histogram") => {
                        let iv = ["interval", "calendar_interval", "fixed_interval"].iter().find_map(|k| agg.get(*k).and_then(Value::as_str)).unwrap_or("");
                        time = estimate_millis(iv);
                    }
                    Some("histogram") => {
                        histo += agg.get("interval").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                        n += 1;
                    }
                    Some("terms") => terms += 1,
                    _ => {}
                }
            }
        }
        (time, if n == 0 { 0.0 } else { histo / n as f32 }, terms)
    };
    let (at, ah, atm) = weigh(a);
    let (bt, bh, btm) = weigh(b);
    match at.cmp(&bt) {
        Ordering::Equal => {}
        o => return o.reverse(),
    }
    match ah.partial_cmp(&bh).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        o => {
            if ah == 0.0 {
                return Ordering::Less;
            } else if bh == 0.0 {
                return Ordering::Greater;
            }
            return o.reverse();
        }
    }
    atm.cmp(&btm)
}

fn top_caps<'a>(mut local: Vec<&'a JobCaps>) -> &'a JobCaps {
    local.sort_by(|a, b| caps_order(a, b));
    local[0]
}

/// `RollupJobIdentifierUtils.findBestJobs` for one aggregation tree.
fn find_best_jobs<'a>(name: &str, agg: &Value, caps: Vec<&'a JobCaps>, best: &mut Vec<&'a JobCaps>) -> Result<()> {
    let (ty, def, subs) = agg_parts(name, agg)?;
    let unsupported = |field: &str| ia(format!("There is not a rollup job that has a [{ty}] agg on field [{field}] which also satisfies all requirements of query."));
    let local: Vec<&JobCaps> = match ty.as_str() {
        "date_histogram" => {
            let field = field_of(&ty, &def)?;
            let (cal, fix) = date_histo_intervals(&def)?;
            let tz = canonical_timezone(str_of(&def, "time_zone").unwrap_or("UTC"));
            let local: Vec<&JobCaps> = caps
                .into_iter()
                .filter(|c| {
                    // Only the first date_histogram cap of the field counts (ES breaks out of the loop).
                    let Some(a) = c.aggs_of(&field, "date_histogram").next() else { return false };
                    if canonical_timezone(a.get("time_zone").and_then(Value::as_str).unwrap_or("UTC")) != tz {
                        return false;
                    }
                    let legacy = a.get("interval").and_then(Value::as_str);
                    if let Some(c) = &cal {
                        validate_calendar_interval(c, a.get("calendar_interval").and_then(Value::as_str)) || validate_calendar_interval(c, legacy)
                    } else if let Some(f) = &fix {
                        validate_fixed_interval(f, a.get("fixed_interval").and_then(Value::as_str)) || validate_fixed_interval(f, legacy)
                    } else {
                        false
                    }
                })
                .collect();
            if local.is_empty() {
                return Err(unsupported(&field));
            }
            local
        }
        "histogram" => {
            let field = field_of(&ty, &def)?;
            let interval = def.get("interval").and_then(Value::as_f64).ok_or_else(|| ia("[interval] must be 1 or greater for aggregation [histogram]"))?;
            let local: Vec<&JobCaps> = caps
                .into_iter()
                .filter(|c| {
                    let Some(a) = c.aggs_of(&field, "histogram").next() else { return false };
                    let ci = a.get("interval").and_then(Value::as_f64).unwrap_or(f64::MAX);
                    ci <= interval && interval % ci == 0.0
                })
                .collect();
            if local.is_empty() {
                return Err(unsupported(&field));
            }
            local
        }
        "terms" => {
            let field = field_of(&ty, &def)?;
            let local: Vec<&JobCaps> = caps.into_iter().filter(|c| c.aggs_of(&field, "terms").next().is_some()).collect();
            if local.is_empty() {
                return Err(unsupported(&field));
            }
            local
        }
        t if SUPPORTED_METRICS.contains(&t) => {
            let field = field_of(&ty, &def)?;
            let local: Vec<&JobCaps> = caps.into_iter().filter(|c| c.aggs_of(&field, t).next().is_some()).collect();
            if local.is_empty() {
                return Err(ia(format!("There is not a rollup job that has a [{ty}] agg with name [{name}] which also satisfies all requirements of query.")));
            }
            best.push(top_caps(local));
            return Ok(());
        }
        other => {
            return Err(ia(format!(
                "Unable to translate aggregation tree into Rollup.  Aggregation [{name}] is of type [{}] which is currently unsupported.",
                agg_class_name(other)
            )))
        }
    };
    match subs {
        Some(subs) if !subs.is_empty() => {
            for (n, a) in &subs {
                find_best_jobs(n, a, local.clone(), best)?;
            }
        }
        _ => best.push(top_caps(local)),
    }
    Ok(())
}

/// `RollupRequestTranslator.translateAggregation`: the request aggregation
/// rewritten onto the rollup document fields, plus the `_count` sums.
fn translate_agg(name: &str, agg: &Value, out: &mut Map<String, Value>) -> Result<()> {
    let (ty, def, subs) = agg_parts(name, agg)?;
    let field = field_of(&ty, &def)?;
    match ty.as_str() {
        "date_histogram" | "histogram" | "terms" => {
            let mut d = def.clone();
            let value = if ty == "date_histogram" { "timestamp" } else { "value" };
            d["field"] = json!(format!("{field}.{ty}.{value}"));
            if ty == "date_histogram" && d.get("time_zone").is_none() {
                d["time_zone"] = json!("UTC");
            }
            let mut sub_out = Map::new();
            for (n, a) in subs.iter().flatten() {
                translate_agg(n, a, &mut sub_out)?;
            }
            sub_out.insert(format!("{name}._count"), json!({"sum": {"field": format!("{field}.{ty}._count")}}));
            out.insert(name.to_string(), json!({ ty: d, "aggs": sub_out }));
        }
        "avg" => {
            out.insert(format!("{name}.value"), json!({"sum": {"field": format!("{field}.avg.value")}}));
            out.insert(format!("{name}._count"), json!({"sum": {"field": format!("{field}.avg._count")}}));
        }
        "max" | "min" | "sum" | "value_count" => {
            let mut d = def.clone();
            d["field"] = json!(format!("{field}.{ty}.value"));
            out.insert(name.to_string(), json!({ ty: d }));
        }
        other => {
            return Err(ia(format!(
                "Unable to translate aggregation tree into Rollup.  Aggregation [{name}] is of type [{}] which is currently unsupported.",
                agg_class_name(other)
            )))
        }
    }
    Ok(())
}

/// Merge the translated aggregation results of a second job's search into
/// the first's (guided by the translated request `tree`).
fn merge_translated(tree: &Map<String, Value>, into: &mut Map<String, Value>, other: &Map<String, Value>) {
    for (name, def) in tree {
        let Some(o) = other.get(name) else { continue };
        let Some(ty) = def.as_object().and_then(|m| m.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta"))) else { continue };
        if !into.contains_key(name) {
            into.insert(name.clone(), o.clone());
            continue;
        }
        let Some(cur) = into.get_mut(name) else { continue };
        match ty.as_str() {
            "date_histogram" | "histogram" | "terms" => {
                let subs = def.get("aggs").and_then(Value::as_object).cloned().unwrap_or_default();
                let (Some(cb), Some(ob)) = (cur.get_mut("buckets").and_then(Value::as_array_mut), o.get("buckets").and_then(Value::as_array)) else { continue };
                for b in ob {
                    if let Some(existing) = cb.iter_mut().find(|x| x.get("key") == b.get("key")) {
                        let dc = existing["doc_count"].as_i64().unwrap_or(0) + b["doc_count"].as_i64().unwrap_or(0);
                        existing["doc_count"] = json!(dc);
                        if let (Some(em), Some(bm)) = (existing.as_object_mut(), b.as_object()) {
                            merge_translated(&subs, em, bm);
                        }
                    } else {
                        cb.push(b.clone());
                    }
                }
                if ty == "terms" {
                    cb.sort_by(|a, b| b["doc_count"].as_i64().cmp(&a["doc_count"].as_i64()).then_with(|| a["key"].to_string().cmp(&b["key"].to_string())));
                } else {
                    cb.sort_by(|a, b| a["key"].as_f64().partial_cmp(&b["key"].as_f64()).unwrap_or(Ordering::Equal));
                }
            }
            "sum" | "value_count" => {
                let v = cur["value"].as_f64().unwrap_or(0.0) + o["value"].as_f64().unwrap_or(0.0);
                cur["value"] = json!(v);
            }
            "max" | "min" => match (cur["value"].as_f64(), o["value"].as_f64()) {
                (Some(a), Some(b)) => cur["value"] = json!(if ty == "max" { a.max(b) } else { a.min(b) }),
                (None, Some(b)) => cur["value"] = json!(b),
                _ => {}
            },
            _ => {}
        }
    }
}

fn typed_name(ty: &str, name: &str, typed: bool, res: &Value) -> String {
    if !typed {
        return name.to_string();
    }
    let prefix = if ty == "terms" {
        let keys: Vec<&Value> = res["buckets"].as_array().into_iter().flatten().map(|b| &b["key"]).collect();
        if keys.iter().any(|k| k.is_string()) {
            "sterms"
        } else if keys.iter().any(|k| k.is_f64()) {
            "dterms"
        } else {
            "lterms"
        }
    } else {
        ty
    };
    format!("{prefix}#{name}")
}

/// `RollupResponseTranslator.unrollAgg`: the translated aggregation results
/// back in the shape of the request aggregations.
fn unroll_aggs(orig: &Map<String, Value>, inner: &Map<String, Value>, typed: bool) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    for (name, agg) in orig {
        let (ty, def, subs) = agg_parts(name, agg)?;
        match ty.as_str() {
            "date_histogram" | "histogram" | "terms" => {
                let Some(ia_) = inner.get(name) else { continue };
                let mut res = ia_.clone();
                let min_doc_count = def.get("min_doc_count").and_then(Value::as_i64).unwrap_or(if ty == "terms" { 1 } else { 0 });
                let count_key = format!("{name}._count");
                let subs = subs.unwrap_or_default();
                let unroll_bucket = |b: &Value| -> Result<Option<Value>> {
                    let bo = b.as_object().cloned().unwrap_or_default();
                    let count = bo.get(&count_key).and_then(|c| c["value"].as_f64()).map(|v| v.round() as i64).unwrap_or_else(|| bo.get("doc_count").and_then(Value::as_i64).unwrap_or(0));
                    if count < min_doc_count || (ty == "terms" && count == 0) {
                        return Ok(None);
                    }
                    let mut nb = Map::new();
                    for k in ["key_as_string", "key", "from", "from_as_string", "to", "to_as_string"] {
                        if let Some(v) = bo.get(k) {
                            nb.insert(k.into(), v.clone());
                        }
                    }
                    nb.insert("doc_count".into(), json!(count));
                    nb.extend(unroll_aggs(&subs, &bo, typed)?);
                    Ok(Some(Value::Object(nb)))
                };
                match res.get_mut("buckets") {
                    Some(Value::Array(a)) => {
                        let mut nb = Vec::with_capacity(a.len());
                        for b in a.iter() {
                            if let Some(v) = unroll_bucket(b)? {
                                nb.push(v);
                            }
                        }
                        if ty == "terms" && def.get("order").is_none() {
                            nb.sort_by(|a, b| b["doc_count"].as_i64().cmp(&a["doc_count"].as_i64()).then_with(|| a["key"].to_string().cmp(&b["key"].to_string())));
                        }
                        *a = nb;
                    }
                    Some(Value::Object(m)) => {
                        let mut nm = Map::new();
                        for (k, b) in m.iter() {
                            if let Some(v) = unroll_bucket(b)? {
                                nm.insert(k.clone(), v);
                            }
                        }
                        *m = nm;
                    }
                    _ => {}
                }
                let key = typed_name(&ty, name, typed, &res);
                out.insert(key, res);
            }
            "avg" => {
                let sum = inner.get(&format!("{name}.value")).and_then(|v| v["value"].as_f64()).unwrap_or(0.0);
                let count = inner.get(&format!("{name}._count")).and_then(|v| v["value"].as_f64()).unwrap_or(0.0);
                let value = if count > 0.0 { json!(sum / count) } else { Value::Null };
                out.insert(typed_name("avg", name, typed, &Value::Null), json!({"value": value}));
            }
            "max" | "min" | "sum" | "value_count" => {
                if let Some(v) = inner.get(name) {
                    out.insert(typed_name(&ty, name, typed, v), v.clone());
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// `TransportRollupSearchAction.rewriteQuery`: the query on rolled-up fields.
fn rewrite_query(q: Option<&Value>, caps: &[&JobCaps]) -> Result<Value> {
    let Some(q) = q.filter(|q| !q.is_null()) else { return Ok(json!({"match_all": {}})) };
    let o = q.as_object().filter(|m| m.len() == 1).ok_or_else(|| ia("Unsupported Query in search request: [unknown]"))?;
    let (ty, body) = o.iter().next().unwrap();
    let clauses = |v: Option<&Value>| -> Result<Vec<Value>> {
        match v {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(a)) => a.iter().map(|c| rewrite_query(Some(c), caps)).collect(),
            Some(single) => Ok(vec![rewrite_query(Some(single), caps)?]),
        }
    };
    let field_entry = |body: &Value| -> Result<(String, Value)> {
        let m = body.as_object().ok_or_else(|| ia(format!("[{ty}] query malformed, no start_object after query name")))?;
        let (f, v) = m.iter().find(|(k, _)| !matches!(k.as_str(), "boost" | "_name")).ok_or_else(|| ia(format!("[{ty}] query malformed, no field")))?;
        Ok((f.clone(), v.clone()))
    };
    Ok(match ty.as_str() {
        "bool" => {
            let mut b = Map::new();
            for k in ["must", "must_not", "should", "filter"] {
                let cs = clauses(body.get(k))?;
                if !cs.is_empty() {
                    b.insert(k.into(), Value::Array(cs));
                }
            }
            json!({"bool": b})
        }
        "constant_score" => json!({"constant_score": {"filter": rewrite_query(body.get("filter"), caps)?}}),
        "boosting" => json!({"boosting": {"positive": rewrite_query(body.get("positive"), caps)?, "negative": rewrite_query(body.get("negative"), caps)?, "negative_boost": body.get("negative_boost").cloned().unwrap_or(json!(0.0))}}),
        "dis_max" => json!({"dis_max": {"queries": clauses(body.get("queries"))?}}),
        "range" => {
            let (f, v) = field_entry(body)?;
            let mut r = v.as_object().cloned().unwrap_or_default();
            r.remove("boost");
            json!({"range": { rewrite_field_name(caps, "range", &f)?: r }})
        }
        "term" => {
            let (f, v) = field_entry(body)?;
            let value = v.get("value").cloned().unwrap_or(v);
            json!({"term": { rewrite_field_name(caps, "term", &f)?: value }})
        }
        "terms" => {
            let (f, v) = field_entry(body)?;
            json!({"terms": { rewrite_field_name(caps, "term", &f)?: v }})
        }
        "match_all" => q.clone(),
        other => return Err(ia(format!("Unsupported Query in search request: [{other}]"))),
    })
}

fn rewrite_field_name(caps: &[&JobCaps], builder: &str, field: &str) -> Result<String> {
    let mut names: Vec<String> = Vec::new();
    for c in caps {
        for a in c.fields.get(field).into_iter().flatten() {
            let n = match a.get("agg").and_then(Value::as_str) {
                Some("date_histogram") => format!("{field}.date_histogram.timestamp"),
                Some(t @ ("terms" | "histogram")) => format!("{field}.{t}.value"),
                _ => continue,
            };
            if !names.contains(&n) {
                names.push(n);
            }
        }
    }
    match names.len() {
        0 => Err(ia(format!("Field [{field}] in [{builder}] query is not available in selected rollup indices, cannot query."))),
        1 => Ok(names.remove(0)),
        _ => Err(ia(format!("Ambiguous field name resolution when mapping to rolled fields.  Field name [{field}] was mapped to: [{}].", names.join(",")))),
    }
}

fn validate_search_request(body: &Value) -> Result<()> {
    if body.get("size").and_then(Value::as_i64) != Some(0) {
        return Err(ia("Rollup does not support returning search hits, please try again with [size: 0]."));
    }
    if body.get("post_filter").is_some() {
        return Err(ia("Rollup search does not support post filtering."));
    }
    if body.get("suggest").is_some() {
        return Err(ia("Rollup search does not support suggestors."));
    }
    if body.get("highlight").is_some() {
        return Err(ia("Rollup search does not support highlighting."));
    }
    if body.get("profile").and_then(Value::as_bool) == Some(true) {
        return Err(ia("Rollup search does not support profiling at the moment."));
    }
    if body.get("explain").and_then(Value::as_bool) == Some(true) {
        return Err(ia("Rollup search does not support explaining."));
    }
    Ok(())
}

fn parse_resp(s: String) -> Result<Value> {
    serde_json::from_str(&s).map_err(|e| EsError::internal(e.to_string()))
}

/// `RollupResponseTranslator.mergeFinalResponse`: no hits, the shard counts
/// and took of every search involved, and the unrolled aggregations.
fn final_response(inners: &[Value], live: Option<&Value>, aggs: Option<Map<String, Value>>, total_as_int: bool) -> Value {
    let sum = |k: &str| -> i64 { inners.iter().map(|r| r["_shards"][k].as_i64().unwrap_or(0)).sum::<i64>() + live.map(|l| l["_shards"][k].as_i64().unwrap_or(0)).unwrap_or(0) };
    let mut took: i64 = inners.iter().map(|r| r["took"].as_i64().unwrap_or(0)).sum();
    if let Some(l) = live {
        took = took.max(l["took"].as_i64().unwrap_or(0));
    }
    let timed_out = inners.iter().any(|r| r["timed_out"].as_bool() == Some(true));
    let mut out = Map::new();
    out.insert("took".into(), json!(took));
    out.insert("timed_out".into(), json!(timed_out));
    out.insert("_shards".into(), json!({"total": sum("total"), "successful": sum("successful"), "skipped": sum("skipped"), "failed": 0}));
    let total = if total_as_int { json!(0) } else { json!({"value": 0, "relation": "eq"}) };
    out.insert("hits".into(), json!({"total": total, "max_score": Value::Null, "hits": []}));
    if let Some(a) = aggs.filter(|a| !a.is_empty()) {
        out.insert("aggregations".into(), Value::Object(a));
    }
    Value::Object(out)
}

/// Combine the live indices' aggregations with the unrolled ones: a bucket or
/// metric the live search has wins (`RollupResponseTranslator.combineResponses`).
fn combine_with_live(live: &Map<String, Value>, rolled: Map<String, Value>) -> Map<String, Value> {
    let mut out = live.clone();
    for (k, v) in rolled {
        match out.get_mut(&k) {
            None => {
                out.insert(k, v);
            }
            Some(cur) => {
                if let (Some(cb), Some(rb)) = (cur.get_mut("buckets").and_then(Value::as_array_mut), v.get("buckets").and_then(Value::as_array)) {
                    for b in rb {
                        if !cb.iter().any(|x| x.get("key") == b.get("key")) {
                            cb.push(b.clone());
                        }
                    }
                    cb.sort_by(|a, b| a["key"].as_f64().partial_cmp(&b["key"].as_f64()).unwrap_or(Ordering::Equal));
                }
            }
        }
    }
    out
}

/// A `_rollup_search` sub-search: on a cluster through the coordinator
/// (every shard of the targets), else on this node's engine.
fn search_on(req: &Req, node: &Node, body: &Value, params: &HashMap<String, String>, index: Option<&str>, opts: ResolveOpts, access: Option<&crate::security::access::ReadAccess>) -> Result<String> {
    crate::search_api::coordinated_search(node, req, body, params, index.unwrap_or("_all"), opts, access)
}

fn rollup_search(node: &Node, req: &Req, index: Option<&str>) -> R {
    let opts = ResolveOpts::from_params(&req.params);
    let targets = node.resolve(index.unwrap_or("_all"), opts.clone())?;
    if targets.is_empty() {
        return Err(ia("Must specify at least one concrete index."));
    }
    let mut live: Vec<String> = Vec::new();
    let mut rollup: Vec<(String, Vec<JobCaps>)> = Vec::new();
    for t in &targets {
        let name = t.svc.name.clone();
        if live.contains(&name) || rollup.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let caps = index_job_caps(&t.svc);
        if caps.is_empty() {
            live.push(name);
        } else {
            rollup.push((name, caps));
        }
    }
    if rollup.len() > 1 {
        let names: Vec<&str> = rollup.iter().map(|(n, _)| n.as_str()).collect();
        return Err(ia(format!("RollupSearch currently only supports searching one rollup index at a time. Found the following rollup indices: {}", java_list(&names))));
    }
    let body = req.json()?;
    let body = if body.is_null() { json!({}) } else { body };
    let access = req.access.as_deref();
    if rollup.is_empty() {
        if live.is_empty() {
            return Err(ia("Must specify at least one rollup index in _rollup_search API"));
        }
        let json = search_on(req, node, &body, &req.params, Some(&live.join(",")), opts, access)?;
        return Ok(Resp::raw(200, json));
    }
    validate_search_request(&body)?;
    let (rollup_index, caps) = &rollup[0];
    let typed = req.flag("typed_keys")?;
    let total_as_int = req.flag("rest_total_hits_as_int")?;
    let mut inner_params = req.params.clone();
    inner_params.remove("typed_keys");
    inner_params.remove("rest_total_hits_as_int");
    let live_resp = if live.is_empty() { None } else { Some(parse_resp(search_on(req, node, &body, &inner_params, Some(&live.join(",")), opts.clone(), access)?)?) };
    let aggs = body.get("aggs").or_else(|| body.get("aggregations")).and_then(Value::as_object).filter(|m| !m.is_empty());
    let Some(aggs) = aggs else {
        let inner = parse_resp(search_on(req, node, &body, &inner_params, Some(rollup_index.as_str()), opts, access)?)?;
        let live_aggs = live_resp.as_ref().and_then(|l| l["aggregations"].as_object()).cloned();
        return Ok(Resp::ok(final_response(&[inner], live_resp.as_ref(), live_aggs, total_as_int)));
    };
    // Pick the jobs (one per aggregation tree), translate and filter per job.
    let mut chosen: Vec<&JobCaps> = Vec::new();
    for (name, agg) in aggs {
        let mut best = Vec::new();
        find_best_jobs(name, agg, caps.iter().collect(), &mut best)?;
        for b in best {
            if !chosen.iter().any(|c| c.job_id == b.job_id) {
                chosen.push(b);
            }
        }
    }
    let mut translated = Map::new();
    for (name, agg) in aggs {
        translate_agg(name, agg, &mut translated)?;
    }
    let query = rewrite_query(body.get("query"), &chosen)?;
    let mut inners = Vec::new();
    let mut merged: Option<Map<String, Value>> = None;
    for job in &chosen {
        let rolled = json!({
            "size": 0,
            "query": {"bool": {"must": [query.clone()], "filter": [{"term": {"_rollup.id": job.job_id}}, {"terms": {"_rollup.version": [1, 2]}}]}},
            "aggs": translated,
        });
        let r = parse_resp(search_on(req, node, &rolled, &inner_params, Some(rollup_index.as_str()), opts.clone(), access)?)?;
        let aggs_r = r["aggregations"].as_object().cloned().unwrap_or_default();
        match &mut merged {
            None => merged = Some(aggs_r),
            Some(m) => merge_translated(&translated, m, &aggs_r),
        }
        inners.push(r);
    }
    let mut unrolled = unroll_aggs(aggs, &merged.unwrap_or_default(), typed)?;
    if let Some(l) = live_resp.as_ref().and_then(|l| l["aggregations"].as_object()) {
        unrolled = combine_with_live(l, unrolled);
    }
    Ok(Resp::ok(final_response(&inners, live_resp.as_ref(), Some(unrolled), total_as_int)))
}

// ------------------------------------------------------ searchable snapshots

/// `_searchable_snapshots/*`, with or without an index.
pub async fn searchable_snapshots(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let off = usize::from(index.is_some());
    let rest: Vec<&str> = req.segs.iter().skip(off + 1).map(String::as_str).collect();
    let m = req.method.clone();
    match rest.as_slice() {
        ["cache", "stats"] | [_, "cache", "stats"] if index.is_none() => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            crate::util::check_params(&req, &[])?;
            let nodes_expr = if rest.len() == 3 { rest[0].to_string() } else { "_all".to_string() };
            let human = req.flag("human")?;
            let mut nodes = serde_json::Map::new();
            if crate::observability::node_matches(&node, &nodes_expr) {
                nodes.insert(NODE_ID.to_string(), json!({"shared_cache": shared_cache_stats(&node, human)}));
            }
            Ok(Resp::json(200, json!({"nodes": nodes})))
        }
        ["cache", "clear"] | ["stats"] => {
            let stats = rest[0] == "stats";
            let want = if stats { Method::GET } else { Method::POST };
            if m != want {
                return method_not_allowed(&req, &[want.as_str()]);
            }
            crate::util::check_params(&req, if stats { &["level"] } else { &["index", "ignore_unavailable", "allow_no_indices", "expand_wildcards"] })?;
            let level = req.param("level").unwrap_or("indices").to_string();
            if stats && !matches!(level.as_str(), "cluster" | "indices" | "shards") {
                return Err(EsError::illegal_argument(format!("level parameter must be one of [cluster] or [indices] or [shards] but was [{level}]")));
            }
            let human = req.flag("human")?;
            crate::blocking(move || {
                let target = index.as_deref().or(req.param("index")).unwrap_or("_all").to_string();
                let targets = node.resolve(&target, ResolveOpts::from_params(&req.params))?;
                let mut seen = std::collections::HashSet::new();
                let mounted: Vec<Arc<IndexService>> = targets
                    .into_iter()
                    .map(|t| t.svc)
                    .filter(|svc| elasticsearch_engine::snapshot::is_searchable_snapshot(svc) && seen.insert(svc.name.clone()))
                    .collect();
                if mounted.is_empty() {
                    return Err(EsError::resource_not_found("No searchable snapshots indices found"));
                }
                let shards: usize = mounted.iter().map(|svc| svc.local_shards().len().max(1)).sum();
                let header = json!({"total": shards, "successful": shards, "failed": 0});
                if !stats {
                    return Ok(Resp::ok(json!({"_shards": header})));
                }
                Ok(Resp::ok(snapshot_stats_json(&mounted, &level, human, header)))
            })
            .await
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `shared_cache` of `GET /_searchable_snapshots/cache/stats` (the node's
/// `xpack.searchable.snapshot.shared_cache.*` settings).
fn shared_cache_stats(node: &Node, human: bool) -> Value {
    let setting = |k: &str, d: i64| -> i64 {
        node.node_settings.read().get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_i64().map(|n| n.to_string()))).and_then(|s| elasticsearch_engine::ingest::util::parse_bytes(&s)).unwrap_or(d)
    };
    let size = setting("xpack.searchable.snapshot.shared_cache.size", 0).max(0) as u64;
    let region = setting("xpack.searchable.snapshot.shared_cache.region_size", 16 * 1024 * 1024).max(1) as u64;
    let mut o = serde_json::Map::new();
    o.insert("reads".into(), json!(0));
    let mut hb = |o: &mut serde_json::Map<String, Value>, raw: &str, h: &str, n: u64| {
        if human {
            o.insert(h.into(), json!(crate::util::human_bytes(n)));
        }
        o.insert(raw.into(), json!(n));
    };
    hb(&mut o, "bytes_read_in_bytes", "bytes_read", 0);
    o.insert("writes".into(), json!(0));
    hb(&mut o, "bytes_written_in_bytes", "bytes_written", 0);
    o.insert("evictions".into(), json!(0));
    o.insert("num_regions".into(), json!(size / region));
    hb(&mut o, "size_in_bytes", "size", size);
    hb(&mut o, "region_size_in_bytes", "region_size", region);
    Value::Object(o)
}

/// Per-file-extension input stats of one shard directory: (ext, sizes).
fn shard_files(dir: &std::path::Path, out: &mut std::collections::BTreeMap<String, Vec<u64>>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n != "translog") {
                    shard_files(&path, out);
                }
            } else if let Ok(md) = e.metadata() {
                let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("other").to_string();
                out.entry(ext).or_default().push(md.len());
            }
        }
    }
}

/// `CacheIndexInputStats` objects (sorted by extension) for files grouped by extension.
fn input_stats(files: &std::collections::BTreeMap<String, Vec<u64>>, human: bool) -> Vec<Value> {
    let counter = || json!({"count": 0, "sum": 0, "min": 0, "max": 0});
    let timed = || {
        let mut c = json!({"count": 0, "sum": 0, "min": 0, "max": 0});
        if human {
            c["time"] = json!(crate::util::human_time_nanos(0));
        }
        c["time_in_nanos"] = json!(0);
        c
    };
    files
        .iter()
        .map(|(ext, sizes)| {
            let n = sizes.len() as u64;
            let total: u64 = sizes.iter().sum();
            let (min, max) = (sizes.iter().copied().min().unwrap_or(0), sizes.iter().copied().max().unwrap_or(0));
            let avg = if n > 0 { total / n } else { 0 };
            let mut size = serde_json::Map::new();
            for (h, raw, v) in [("total", "total_in_bytes", total), ("min", "min_in_bytes", min), ("max", "max_in_bytes", max), ("average", "average_in_bytes", avg)] {
                if human {
                    size.insert(h.into(), json!(crate::util::human_bytes(v)));
                }
                size.insert(raw.into(), json!(v));
            }
            json!({
                "file_ext": ext, "num_files": n, "open_count": n, "close_count": n, "size": size,
                "contiguous_bytes_read": counter(), "non_contiguous_bytes_read": counter(),
                "cached_bytes_read": counter(), "index_cache_bytes_read": counter(),
                "cached_bytes_written": timed(), "direct_bytes_read": timed(), "optimized_bytes_read": timed(),
                "forward_seeks": {"small": counter(), "large": counter()},
                "backward_seeks": {"small": counter(), "large": counter()},
                "blob_store_bytes_requested": counter(), "lucene_bytes_read": counter(),
                "current_index_cache_fills": 0,
            })
        })
        .collect()
}

/// `GET /_searchable_snapshots/stats` (`SearchableSnapshotsStatsResponse`).
fn snapshot_stats_json(mounted: &[Arc<IndexService>], level: &str, human: bool, header: Value) -> Value {
    use std::collections::BTreeMap;
    let mut all: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut indices = serde_json::Map::new();
    let mut sorted: Vec<&Arc<IndexService>> = mounted.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for svc in sorted {
        let mut idx_files: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut shards = serde_json::Map::new();
        let mut local = svc.local_shards();
        local.sort_by_key(|s| s.id);
        for sh in local {
            let mut files = BTreeMap::new();
            shard_files(sh.shard.path(), &mut files);
            for (k, v) in &files {
                idx_files.entry(k.clone()).or_default().extend(v.iter().copied());
            }
            shards.insert(
                sh.id.to_string(),
                json!([{
                    "snapshot_uuid": svc.setting("index.store.snapshot.snapshot_uuid").unwrap_or_default(),
                    "index_uuid": svc.setting("index.store.snapshot.index_uuid").unwrap_or_default(),
                    "shard": {"id": sh.id, "state": "STARTED", "primary": true, "node": NODE_ID},
                    "files": input_stats(&files, human),
                }]),
            );
        }
        for (k, v) in &idx_files {
            all.entry(k.clone()).or_default().extend(v.iter().copied());
        }
        let mut entry = json!({"total": input_stats(&idx_files, human)});
        if level == "shards" {
            entry["shards"] = Value::Object(shards);
        }
        indices.insert(svc.name.clone(), entry);
    }
    let mut out = json!({"_shards": header, "total": input_stats(&all, human)});
    if level != "cluster" {
        out["indices"] = Value::Object(indices);
    }
    out
}
