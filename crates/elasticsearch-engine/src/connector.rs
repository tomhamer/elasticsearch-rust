//! Connectors (`_connector`): connector documents, sync jobs and connector
//! secrets, as ES's `x-pack/plugin/ent-search` connector module keeps them.
//!
//! Storage mirrors ES: connectors live in `.elastic-connectors-v1`, sync jobs
//! in `.elastic-connectors-sync-jobs-v1` and secrets in `.connector-secrets`
//! (hidden indices created on first write). Partial updates use ES's update
//! `doc` semantics (objects merged recursively, new keys appended, `noop`
//! when nothing changed); objects ES rebuilds from a Java `HashMap` (the
//! filtering block after an edit, configuration entries, update docs) are
//! rendered in that map's iteration order so `GET` bodies match byte for
//! byte. Request parsing reproduces ES's `ObjectParser` errors, including the
//! `[line:col]` token positions.

use crate::error::{EsError, Result};
use crate::index::{IndexService, WriteReq};
use crate::node::{Node, ResolveOpts};
use crate::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const CONNECTOR_INDEX: &str = ".elastic-connectors-v1";
pub const SYNC_JOB_INDEX: &str = ".elastic-connectors-sync-jobs-v1";
pub const SECRETS_INDEX: &str = ".connector-secrets";

/// `(status, body)` of a REST response.
pub type Out = Result<(u16, Value)>;

// ------------------------------------------------------------------ positions

/// The request body text plus the token positions of every member, so
/// errors carry ES's `[line:col]` prefix.
pub struct Src {
    text: String,
    /// JSON pointer -> (key start, value start, value end) byte offsets.
    pos: HashMap<String, (usize, usize, usize)>,
}

impl Src {
    pub fn new(body: &[u8]) -> Src {
        let text = String::from_utf8_lossy(body).into_owned();
        let mut pos = HashMap::new();
        let b = text.as_bytes();
        let mut i = 0;
        skip_ws(b, &mut i);
        if i < b.len() {
            let start = i;
            let end = scan(b, &mut i, "", &mut pos);
            pos.insert(String::new(), (start, start, end));
        }
        Src { text, pos }
    }

    fn lc(&self, off: usize) -> String {
        let off = off.min(self.text.len());
        let before = &self.text[..off];
        let line = before.matches('\n').count() + 1;
        let ls = before.rfind('\n').map(|p| p + 1).unwrap_or(0);
        let col = self.text[ls..off].chars().count() + 1;
        format!("[{line}:{col}]")
    }
    /// Position of the value at `ptr`.
    pub fn val(&self, ptr: &str) -> String {
        self.lc(self.pos.get(ptr).map(|p| p.1).unwrap_or(0))
    }
    /// Position of the key of the member at `ptr`.
    pub fn key(&self, ptr: &str) -> String {
        self.lc(self.pos.get(ptr).map(|p| p.0).unwrap_or(0))
    }
    /// Position of the last token of the value at `ptr` (an object's `}`).
    pub fn end(&self, ptr: &str) -> String {
        self.lc(self.pos.get(ptr).map(|p| p.2).unwrap_or(0))
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

fn scan_string(b: &[u8], i: &mut usize) -> String {
    let start = *i + 1;
    *i += 1;
    while *i < b.len() && b[*i] != b'"' {
        if b[*i] == b'\\' {
            *i += 1;
        }
        *i += 1;
    }
    let s = String::from_utf8_lossy(&b[start.min(b.len())..(*i).min(b.len())]).into_owned();
    *i += 1;
    s
}

/// Scan one value at `*i`; returns the offset of its last byte.
fn scan(b: &[u8], i: &mut usize, ptr: &str, out: &mut HashMap<String, (usize, usize, usize)>) -> usize {
    if *i >= b.len() {
        return b.len().saturating_sub(1);
    }
    match b[*i] {
        b'{' => {
            *i += 1;
            loop {
                skip_ws(b, i);
                if *i >= b.len() {
                    return b.len().saturating_sub(1);
                }
                if b[*i] == b'}' {
                    *i += 1;
                    return *i - 1;
                }
                if b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                let ks = *i;
                let k = if b[*i] == b'"' { scan_string(b, i) } else { String::new() };
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b':' {
                    *i += 1;
                }
                skip_ws(b, i);
                let p = format!("{ptr}/{}", k.replace('~', "~0").replace('/', "~1"));
                let vs = *i;
                let ve = scan(b, i, &p, out);
                out.insert(p, (ks, vs, ve));
            }
        }
        b'[' => {
            *i += 1;
            let mut n = 0;
            loop {
                skip_ws(b, i);
                if *i >= b.len() {
                    return b.len().saturating_sub(1);
                }
                if b[*i] == b']' {
                    *i += 1;
                    return *i - 1;
                }
                if b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                let p = format!("{ptr}/{n}");
                let vs = *i;
                let ve = scan(b, i, &p, out);
                out.insert(p, (vs, vs, ve));
                n += 1;
            }
        }
        b'"' => {
            scan_string(b, i);
            *i - 1
        }
        _ => {
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
                *i += 1;
            }
            *i - 1
        }
    }
}

fn child(ptr: &str, k: &str) -> String {
    format!("{ptr}/{}", k.replace('~', "~0").replace('/', "~1"))
}

// ------------------------------------------------------------------ errors

fn token(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn xparse(loc: &str, obj: &str, msg: impl AsRef<str>) -> EsError {
    EsError::new(400, "x_content_parse_exception", format!("{loc} [{obj}] {}", msg.as_ref()))
}

fn wrong_type(src: &Src, ptr: &str, obj: &str, field: &str, v: &Value) -> EsError {
    xparse(&src.val(ptr), obj, format!("{field} doesn't support values of type: {}", token(v)))
}

fn failed_field(loc: &str, obj: &str, field: &str, cause: EsError) -> EsError {
    xparse(loc, obj, format!("failed to parse field [{field}]")).caused_by(cause)
}

fn iae(msg: impl Into<String>) -> EsError {
    EsError::illegal_argument(msg)
}

fn status_ex(msg: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", msg)
}

/// Lucene `LevenshteinDistance` similarity.
fn similarity(a: &str, b: &str) -> f32 {
    let sa: Vec<char> = a.chars().collect();
    let ot: Vec<char> = b.chars().collect();
    let (n, m) = (sa.len(), ot.len());
    if n == 0 || m == 0 {
        return if n == m { 1.0 } else { 0.0 };
    }
    let mut p: Vec<usize> = (0..=n).collect();
    let mut d = vec![0usize; n + 1];
    for j in 1..=m {
        d[0] = j;
        for i in 1..=n {
            let cost = usize::from(sa[i - 1] != ot[j - 1]);
            d[i] = (d[i - 1] + 1).min(p[i] + 1).min(p[i - 1] + cost);
        }
        std::mem::swap(&mut p, &mut d);
    }
    1.0 - p[n] as f32 / n.max(m) as f32
}

fn unknown_field(src: &Src, ptr: &str, obj: &str, field: &str, known: &[&str]) -> EsError {
    let mut cands: Vec<(f32, &str)> = known.iter().map(|k| (similarity(field, k), *k)).filter(|(s, _)| *s > 0.7).collect();
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(b.1)));
    let hint = match cands.len() {
        0 => String::new(),
        1 => format!(" did you mean [{}]?", cands[0].1),
        _ => format!(" did you mean any of {}?", format!("[{}]", cands.iter().map(|c| c.1).collect::<Vec<_>>().join(", "))),
    };
    xparse(&src.key(ptr), obj, format!("unknown field [{field}]{hint}"))
}

fn not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("connector [{id}] not found"))
}

fn empty_id() -> &'static str {
    "[connector_id] cannot be [null] or [\"\"]."
}

fn empty_job_id() -> &'static str {
    "[connector_sync_job_id] of the connector sync job cannot be null or empty."
}

fn validation(errs: &[String]) -> Result<()> {
    if errs.is_empty() {
        return Ok(());
    }
    let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
    Err(EsError::action_validation(&refs))
}

// ------------------------------------------------------------------ object parser

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Str,
    StrOrNull,
    Bool,
    Long,
    Obj,
    ObjOrNull,
    Arr,
    /// Object or array (`advanced_snippet.value`).
    ObjOrArr,
    Any,
}

fn kind_ok(k: Kind, v: &Value) -> bool {
    match k {
        Kind::Str => v.is_string(),
        Kind::StrOrNull => v.is_string() || v.is_null(),
        Kind::Bool => v.is_boolean() || matches!(v.as_str(), Some("true" | "false")),
        Kind::Long => v.is_i64() || v.is_u64() || v.as_str().map(|s| s.parse::<i64>().is_ok()).unwrap_or(false),
        Kind::Obj => v.is_object(),
        Kind::ObjOrNull => v.is_object() || v.is_null(),
        Kind::Arr => v.is_array(),
        Kind::ObjOrArr => v.is_object() || v.is_array(),
        Kind::Any => true,
    }
}

/// One `ObjectParser` pass over an object: rejects unknown fields and
/// mistyped values, in document order, and reports missing required fields
/// (`Required [a, b]`, declaration order).
fn check_obj(src: &Src, ptr: &str, obj: &str, v: &Value, fields: &[(&str, Kind)], required: &[&str]) -> Result<()> {
    let Some(o) = v.as_object() else { return Ok(()) };
    let names: Vec<&str> = fields.iter().map(|f| f.0).collect();
    for (k, val) in o {
        let p = child(ptr, k);
        match fields.iter().find(|f| f.0 == k) {
            None => return Err(unknown_field(src, &p, obj, k, &names)),
            Some((_, kind)) if !kind_ok(*kind, val) => return Err(wrong_type(src, &p, obj, k, val)),
            _ => {}
        }
    }
    let missing: Vec<&str> = required.iter().copied().filter(|r| !o.contains_key(*r)).collect();
    if !missing.is_empty() {
        return Err(iae(format!("Required [{}]", missing.join(", "))));
    }
    Ok(())
}

fn as_bool(v: &Value) -> bool {
    v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))
}

fn as_long(v: &Value) -> i64 {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(0)
}

// ------------------------------------------------------------------ Java HashMap order

fn java_hash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for u in s.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(u as u32);
    }
    h
}

/// Keys in the iteration order of a `java.util.HashMap` filled with them
/// in the given order (default capacity, 0.75 load factor).
fn java_order(keys: &[String]) -> Vec<String> {
    let mut cap = 16usize;
    while keys.len() as f64 > cap as f64 * 0.75 {
        cap *= 2;
    }
    let idx = |k: &str| {
        let h = java_hash(k);
        ((h ^ (h >> 16)) as usize) & (cap - 1)
    };
    let mut v: Vec<(usize, usize, String)> = keys.iter().enumerate().map(|(i, k)| (idx(k), i, k.clone())).collect();
    v.sort();
    v.into_iter().map(|x| x.2).collect()
}

fn hash_map(o: Map<String, Value>) -> Map<String, Value> {
    let keys: Vec<String> = o.keys().cloned().collect();
    let mut o = o;
    let mut out = Map::new();
    for k in java_order(&keys) {
        if let Some(v) = o.shift_remove(&k) {
            out.insert(k, v);
        }
    }
    out
}

/// Every object in `v` rendered in `HashMap` order.
fn hash_deep(v: Value) -> Value {
    match v {
        Value::Object(o) => {
            let o: Map<String, Value> = o.into_iter().map(|(k, v)| (k, hash_deep(v))).collect();
            Value::Object(hash_map(o))
        }
        Value::Array(a) => Value::Array(a.into_iter().map(hash_deep).collect()),
        other => other,
    }
}

// ------------------------------------------------------------------ dates

/// `Instant.toString`-like rendering (fraction trimmed, at least millis).
fn fmt_nanos(nanos: i128) -> String {
    let secs = nanos.div_euclid(1_000_000_000) as i64;
    let frac = nanos.rem_euclid(1_000_000_000) as u32;
    let dt = chrono::DateTime::from_timestamp(secs, 0).unwrap_or_default();
    let mut f = format!("{frac:09}");
    while f.len() > 3 && f.ends_with('0') {
        f.pop();
    }
    format!("{}.{f}Z", dt.format("%Y-%m-%dT%H:%M:%S"))
}

pub fn now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    fmt_nanos(d.as_nanos() as i128)
}

/// Parse `strict_date_optional_time||epoch_millis`; `None` when invalid.
fn parse_date(v: &Value) -> Option<String> {
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return None,
    };
    if !s.is_empty() && s.trim_start_matches('-').chars().all(|c| c.is_ascii_digit()) {
        let ms: i128 = s.parse().ok()?;
        return Some(fmt_nanos(ms * 1_000_000));
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
        return Some(fmt_nanos(dt.timestamp() as i128 * 1_000_000_000 + dt.timestamp_subsec_nanos() as i128));
    }
    let fmt = crate::date::DateFormatter::new("strict_date_optional_time").ok()?;
    let ms = fmt.parse(&s).ok()?;
    Some(fmt_nanos(ms as i128 * 1_000_000))
}

fn sort_nanos(s: &str) -> Option<i128> {
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.timestamp() as i128 * 1_000_000_000 + dt.timestamp_subsec_nanos() as i128)
}

fn date_err(src: &Src, ptr: &str, obj: &str, field: &str, v: &Value) -> EsError {
    let shown = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    failed_field(&src.val(ptr), obj, field, iae(format!("failed to parse date field [{shown}] with format [strict_date_optional_time||epoch_millis]"))
        .caused_by(EsError::new(400, "date_time_parse_exception", "Failed to parse with all enclosed parsers")))
}

// ------------------------------------------------------------------ enums

const CONNECTOR_STATUS: &[&str] = &["created", "needs_configuration", "configured", "connected", "error"];
const SYNC_STATUS: &[&str] = &["canceling", "canceled", "completed", "error", "in_progress", "pending", "suspended"];
const JOB_TYPES: &[&str] = &["full", "incremental", "access_control"];
const TRIGGERS: &[&str] = &["on_demand", "scheduled"];

fn enum_value(v: &str, allowed: &[&str], java: &str) -> Result<String> {
    if allowed.contains(&v) {
        Ok(v.to_string())
    } else {
        Err(iae(format!("Unknown {java} [{v}].")))
    }
}

/// `ConnectorStateMachine`.
fn connector_transition_ok(from: &str, to: &str) -> bool {
    let next: &[&str] = match from {
        "created" => &["needs_configuration", "error"],
        "needs_configuration" => &["configured", "error"],
        "configured" => &["needs_configuration", "connected", "error"],
        "connected" => &["configured", "error"],
        "error" => &["connected", "configured", "error"],
        _ => &[],
    };
    next.contains(&to)
}

/// `ConnectorSyncStatusStateMachine`.
fn sync_transition_ok(from: &str, to: &str) -> bool {
    let next: &[&str] = match from {
        "pending" => &["in_progress", "canceled"],
        "in_progress" => &["canceling", "completed", "suspended", "error"],
        "suspended" => &["in_progress", "canceled"],
        "canceling" => &["canceled", "error"],
        _ => &[],
    };
    next.contains(&to)
}

fn sync_transition_err(from: &str, to: &str) -> EsError {
    let msg = format!("Invalid transition attempt from [{from}] to [{to}]. Such a ConnectorSyncStatus transition is not supported by the Connector Protocol.");
    status_ex(msg.clone()).caused_by(EsError::new(400, "connector_sync_job_invalid_status_transition_exception", msg))
}

// ------------------------------------------------------------------ validation helpers

/// `MetadataCreateIndexService.validateIndexOrAliasName` + lowercase rule.
fn index_name_problem(name: &str) -> Option<String> {
    let why = if name.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ',')) {
        "must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]".to_string()
    } else if name.is_empty() {
        "must not be empty".into()
    } else if name.contains('#') {
        "must not contain '#'".into()
    } else if name.contains(':') {
        "must not contain ':'".into()
    } else if name.starts_with(['_', '-', '+']) {
        "must not start with '_', '-', or '+'".into()
    } else if name.len() > 255 {
        format!("index name is too long, ({} > 255)", name.len())
    } else if name == "." || name == ".." {
        "must not be '.' or '..'".into()
    } else if name.to_lowercase() != name {
        "must be lowercase".into()
    } else {
        return None;
    };
    Some(format!("[{name}] org.elasticsearch.indices.InvalidIndexNameException: Invalid index name [{name}], {why}"))
}

/// ES `Cron` (Quartz syntax) validation; the error is the message ES wraps
/// as `illegal cron expression format [...]`.
fn cron_problem(expr: &str) -> Option<String> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() < 6 {
        return Some("java.lang.IllegalArgumentException: Unexpected end of expression.".into());
    }
    if parts.len() > 7 {
        return Some("java.lang.IllegalArgumentException: Unexpected character in expression.".into());
    }
    const MONTHS: &[&str] = &["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
    const DAYS: &[&str] = &["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
    let ranges: [(u32, u32, &str); 7] = [
        (0, 59, "Minute and Second values must be between 0 and 59"),
        (0, 59, "Minute and Second values must be between 0 and 59"),
        (0, 23, "Hour values must be between 0 and 23"),
        (1, 31, "Day of month values must be between 1 and 31"),
        (1, 12, "Month values must be between 1 and 12"),
        (1, 7, "Day-of-Week values must be between 1 and 7"),
        (1970, 2199, "Year values must be between 1970 and 2199"),
    ];
    for (i, p) in parts.iter().enumerate() {
        let (lo, hi, msg) = ranges[i];
        if *p == "?" {
            if i != 3 && i != 5 {
                return Some("java.lang.IllegalArgumentException: '?' can only be specified for Day-of-Month or Day-of-Week.".into());
            }
            continue;
        }
        for item in p.split(',') {
            for tok in item.split(['-', '/', '#']) {
                let t = tok.trim_end_matches(['L', 'W']).to_uppercase();
                if t.is_empty() || t == "*" || t == "L" || t == "LW" {
                    continue;
                }
                if let Ok(n) = t.parse::<u32>() {
                    if n < lo || n > hi {
                        return Some(format!("java.lang.IllegalArgumentException: {msg}"));
                    }
                } else if !((i == 4 && MONTHS.contains(&t.as_str())) || (i == 5 && DAYS.contains(&t.as_str()))) {
                    return Some(format!("java.lang.IllegalArgumentException: Illegal characters for this position: '{t}'"));
                }
            }
        }
    }
    let (dom, dow) = (parts[3], parts[5]);
    if dom != "?" && dow != "?" {
        return Some("java.lang.IllegalArgumentException: Support for specifying both a day-of-week AND a day-of-month parameter is not implemented.".into());
    }
    None
}

// ------------------------------------------------------------------ storage

fn system_settings() -> Value {
    json!({"index": {"number_of_shards": 1, "auto_expand_replicas": "0-1", "hidden": true}})
}

fn ensure_index(node: &Node, name: &str) -> Result<Arc<IndexService>> {
    if let Some(s) = node.get_index(name) {
        return Ok(s);
    }
    match node.create_index(name, &json!({"settings": system_settings(), "mappings": {"dynamic": false}})) {
        Ok(s) => Ok(s),
        Err(e) => node.get_index(name).ok_or(e),
    }
}

fn read(node: &Node, index: &str, id: &str) -> Result<Option<Value>> {
    let Some(svc) = node.get_index(index) else { return Ok(None) };
    match svc.get(id)? {
        Some(g) => Ok(g.source.and_then(|s| serde_json::from_str(&s).ok())),
        None => Ok(None),
    }
}

/// Index a document; `true` when it was created.
fn write(node: &Node, index: &str, id: &str, src: &Value) -> Result<bool> {
    let svc = ensure_index(node, index)?;
    let mut r = svc.write(vec![WriteReq::Index { id: Some(id.to_string()), source: src.to_string(), routing: None, cond: WriteCond::default() }], false);
    let w = r.pop().ok_or_else(|| EsError::internal("no write result"))??;
    Ok(w.result == "created")
}

fn delete(node: &Node, index: &str, id: &str) -> Result<bool> {
    let Some(svc) = node.get_index(index) else { return Ok(false) };
    let mut r = svc.write(vec![WriteReq::Delete { id: id.to_string(), routing: None, cond: WriteCond::default() }], false);
    let w = r.pop().ok_or_else(|| EsError::internal("no write result"))??;
    Ok(w.result == "deleted")
}

/// Every document of `index` as `(id, source)`, in indexing order.
fn scan_all(node: &Node, index: &str) -> Result<Vec<(String, Value)>> {
    let Some(svc) = node.get_index(index) else { return Ok(Vec::new()) };
    let _ = svc.refresh();
    let targets = node.resolve(index, ResolveOpts { expand_hidden: true, ..Default::default() })?;
    let req = crate::search::parse_request(&json!({"size": 10000, "query": {"match_all": {}}, "seq_no_primary_term": true, "sort": [{"_seq_no": "asc"}]}), &HashMap::new())?;
    let o = crate::search::execute(&targets, &req, &crate::watcher::search_opts(node))?;
    let v: Value = serde_json::from_str(&o.json).map_err(|e| EsError::internal(e.to_string()))?;
    Ok(v["hits"]["hits"].as_array().cloned().unwrap_or_default().into_iter().map(|h| (h["_id"].as_str().unwrap_or("").to_string(), h["_source"].clone())).collect())
}

fn with_id(id: &str, src: &Value) -> Value {
    let mut o = Map::new();
    o.insert("id".into(), json!(id));
    if let Some(s) = src.as_object() {
        for (k, v) in s {
            o.insert(k.clone(), v.clone());
        }
    }
    Value::Object(o)
}

/// ES update `doc` merge: objects merge recursively, other values replace
/// in place, new keys are appended.
fn merge(dst: &mut Value, upd: &Value) {
    match (dst, upd) {
        (Value::Object(d), Value::Object(u)) => {
            for (k, v) in u {
                match d.get_mut(k) {
                    Some(existing) if existing.is_object() && v.is_object() => merge(existing, v),
                    Some(existing) => *existing = v.clone(),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, u) => *d = u.clone(),
    }
}

fn result(r: &str) -> Value {
    json!({"result": r})
}

/// Apply an update doc (`HashMap` ordered) to a stored connector.
fn update_connector_doc(node: &Node, id: &str, upd: Map<String, Value>) -> Out {
    let svc = node.get_index(CONNECTOR_INDEX);
    let cur = read(node, CONNECTOR_INDEX, id)?;
    let Some(mut doc) = cur else {
        return Err(match svc {
            Some(s) => EsError::document_missing(CONNECTOR_INDEX, &s.uuid, id),
            None => EsError::index_not_found(CONNECTOR_INDEX),
        });
    };
    let before = doc.clone();
    merge(&mut doc, &Value::Object(hash_map(upd)));
    if doc == before {
        return Ok((200, result("noop")));
    }
    write(node, CONNECTOR_INDEX, id, &doc)?;
    Ok((200, result("updated")))
}

/// A connector that is not soft-deleted (`getConnector(id, false)`).
fn load_connector(node: &Node, id: &str) -> Result<Value> {
    load_connector_incl(node, id, false)
}

fn load_connector_incl(node: &Node, id: &str, include_deleted: bool) -> Result<Value> {
    match read(node, CONNECTOR_INDEX, id)? {
        Some(d) if include_deleted || !is_deleted(&d) => Ok(d),
        _ => Err(not_found(id)),
    }
}

/// `Connector.IS_DELETED_FIELD` (soft delete, ES 9).
fn is_deleted(doc: &Value) -> bool {
    doc.get("deleted").and_then(Value::as_bool).unwrap_or(false)
}

/// Index names of Elastic-managed (native) connectors must start with this.
const MANAGED_CONNECTOR_INDEX_PREFIX: &str = "content-";

fn store_connector(node: &Node, id: &str, before: &Value, doc: &Value) -> Out {
    if doc == before {
        return Ok((200, result("noop")));
    }
    write(node, CONNECTOR_INDEX, id, doc)?;
    Ok((200, result("updated")))
}

// ------------------------------------------------------------------ defaults

fn default_rule(ts: &str, order: i64) -> Value {
    json!({"created_at": ts, "field": "_", "id": "DEFAULT", "order": order, "policy": "include", "rule": "regex", "updated_at": ts, "value": ".*"})
}

fn default_filtering(ts: &str) -> Value {
    let rules = json!({
        "advanced_snippet": {"created_at": ts, "updated_at": ts, "value": {}},
        "rules": [default_rule(ts, 0)],
        "validation": {"errors": [], "state": "valid"}
    });
    json!([{"active": rules.clone(), "domain": "DEFAULT", "draft": rules}])
}

fn default_scheduling() -> Value {
    let s = json!({"enabled": false, "interval": "0 0 0 * * ?"});
    json!({"access_control": s.clone(), "full": s.clone(), "incremental": s})
}

// ------------------------------------------------------------------ connectors

const PUT_FIELDS: &[(&str, Kind)] = &[
    ("description", Kind::StrOrNull),
    ("index_name", Kind::StrOrNull),
    ("is_native", Kind::Bool),
    ("language", Kind::StrOrNull),
    ("name", Kind::StrOrNull),
    ("service_type", Kind::StrOrNull),
];

fn index_in_use(node: &Node, index: &str, except: Option<&str>) -> Result<bool> {
    Ok(scan_all(node, CONNECTOR_INDEX)?.iter().any(|(id, s)| Some(id.as_str()) != except && !is_deleted(s) && s["index_name"].as_str() == Some(index)))
}

/// `PUT /_connector[/{id}]`, `POST /_connector`.
pub fn put(node: &Node, id: Option<&str>, body: &Value, src: &Src) -> Out {
    let empty = json!({});
    let body = if body.is_null() { &empty } else { body };
    if !body.is_object() {
        return Err(EsError::parsing("request body must be an object"));
    }
    check_obj(src, "", "connector_put_request", body, PUT_FIELDS, &[])?;
    let s = |k: &str| body.get(k).and_then(Value::as_str).map(str::to_string);
    let index_name = s("index_name");
    let mut errs = Vec::new();
    if id == Some("") {
        errs.push(empty_id().to_string());
    }
    if let Some(p) = index_name.as_deref().and_then(index_name_problem) {
        errs.push(p);
    }
    // `ConnectorActionRequest.validateManagedConnectorIndexPrefix`.
    if body.get("is_native").is_some_and(as_bool) {
        if let Some(ix) = index_name.as_deref().filter(|ix| !ix.starts_with(MANAGED_CONNECTOR_INDEX_PREFIX)) {
            errs.push(format!("Index [{ix}] is invalid. Index attached to an Elastic-managed connector must start with the prefix: [{MANAGED_CONNECTOR_INDEX_PREFIX}]"));
        }
    }
    validation(&errs)?;
    if let Some(ix) = &index_name {
        if index_in_use(node, ix, id)? {
            return Err(status_ex(format!("Index name [{ix}] is used by another connector.")));
        }
    }
    let native = body.get("is_native").map(as_bool).unwrap_or(false);
    let ts = now();
    let mut doc = Map::new();
    doc.insert("configuration".into(), json!({}));
    doc.insert("custom_scheduling".into(), json!({}));
    if let Some(d) = s("description") {
        doc.insert("description".into(), json!(d));
    }
    doc.insert("filtering".into(), default_filtering(&ts));
    if let Some(ix) = &index_name {
        doc.insert("index_name".into(), json!(ix));
    }
    doc.insert("is_native".into(), json!(native));
    if let Some(l) = s("language") {
        doc.insert("language".into(), json!(l));
    }
    doc.insert("name".into(), json!(s("name").unwrap_or_default()));
    doc.insert("scheduling".into(), default_scheduling());
    if let Some(st) = s("service_type") {
        doc.insert("service_type".into(), json!(st));
    }
    doc.insert("status".into(), json!(if native { "needs_configuration" } else { "created" }));
    doc.insert("sync_now".into(), json!(false));
    doc.insert("deleted".into(), json!(false));
    let id = match id {
        Some(i) => i.to_string(),
        None => crate::index::auto_id(),
    };
    let created = write(node, CONNECTOR_INDEX, &id, &Value::Object(doc))?;
    Ok((if created { 201 } else { 200 }, json!({"id": id, "result": if created { "created" } else { "updated" }})))
}

/// `GET /_connector/{id}[?include_deleted=true]`.
pub fn get(node: &Node, id: &str, include_deleted: bool) -> Out {
    Ok((200, with_id(id, &load_connector_incl(node, id, include_deleted)?)))
}

/// `DELETE /_connector/{id}[?hard=true]`: a soft delete (`deleted: true`)
/// unless `hard`.
pub fn delete_connector(node: &Node, id: &str, delete_sync_jobs: bool, hard: bool) -> Out {
    if id.is_empty() {
        return Err(EsError::action_validation(&[empty_id()]));
    }
    if hard {
        if !delete(node, CONNECTOR_INDEX, id)? {
            return Err(not_found(id));
        }
    } else {
        let mut doc = load_connector(node, id)?;
        doc["deleted"] = json!(true);
        write(node, CONNECTOR_INDEX, id, &doc)?;
    }
    if delete_sync_jobs {
        for (jid, s) in scan_all(node, SYNC_JOB_INDEX)? {
            if s["connector"]["id"].as_str() == Some(id) {
                delete(node, SYNC_JOB_INDEX, &jid)?;
            }
        }
    }
    Ok((200, json!({"acknowledged": true})))
}

fn csv(v: Option<&str>) -> Vec<String> {
    v.map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
}

fn page(from: i64, size: i64) -> Result<(usize, usize)> {
    if from < 0 {
        return Err(iae("Parameter [from] cannot be < 0"));
    }
    if size < 0 {
        return Err(iae("Parameter [size] cannot be < 0"));
    }
    if from + size > 10000 {
        let msg = format!("Result window is too large, from + size must be less than or equal to: [10000] but was [{}]. See the scroll api for a more efficient way to request large data sets. This limit can be set by changing the [index.max_result_window] index level setting.", from + size);
        return Err(EsError::new(400, "search_phase_execution_exception", "all shards failed").caused_by(iae(msg)));
    }
    Ok((from as usize, size as usize))
}

/// Parameters of `GET /_connector`.
pub struct ListParams<'a> {
    pub from: i64,
    pub size: i64,
    pub index_name: Option<&'a str>,
    pub connector_name: Option<&'a str>,
    pub service_type: Option<&'a str>,
    pub query: Option<&'a str>,
    pub include_deleted: bool,
}

/// `GET /_connector`: sorted by `index_name`.
pub fn list(node: &Node, p: &ListParams) -> Out {
    let indices = csv(p.index_name);
    let errs: Vec<String> = indices.iter().filter_map(|i| index_name_problem(i)).collect();
    validation(&errs)?;
    let (from, size) = page(p.from, p.size)?;
    let names = csv(p.connector_name);
    let types = csv(p.service_type);
    let q = p.query.map(|q| q.to_lowercase());
    let mut all: Vec<(String, Value)> = scan_all(node, CONNECTOR_INDEX)?
        .into_iter()
        .filter(|(_, s)| {
            let f = |k: &str| s[k].as_str().unwrap_or("").to_string();
            (p.include_deleted || !is_deleted(s))
                && (indices.is_empty() || indices.contains(&f("index_name")))
                && (names.is_empty() || names.contains(&f("name")))
                && (types.is_empty() || types.contains(&f("service_type")))
                && q.as_ref().is_none_or(|q| ["name", "description", "index_name"].iter().any(|k| f(k).to_lowercase().contains(q.as_str())))
        })
        .collect();
    all.sort_by(|a, b| match (a.1["index_name"].as_str(), b.1["index_name"].as_str()) {
        (Some(x), Some(y)) => x.cmp(y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    let count = all.len();
    let results: Vec<Value> = all.iter().skip(from).take(size).map(|(id, s)| with_id(id, s)).collect();
    Ok((200, json!({"count": count, "results": results})))
}

/// Parse a single-field update body (`{"<field>": <value>}`).
fn single(src: &Src, obj: &str, body: &Value, fields: &[(&str, Kind)]) -> Result<()> {
    if body.is_null() {
        return Ok(());
    }
    if !body.is_object() {
        return Err(EsError::parsing("request body must be an object"));
    }
    check_obj(src, "", obj, body, fields, &[])
}

fn need_id(id: &str, extra: Vec<String>) -> Result<()> {
    let mut errs = Vec::new();
    if id.is_empty() {
        errs.push(empty_id().to_string());
    }
    errs.extend(extra);
    validation(&errs)
}

/// `PUT /_connector/{id}/_check_in`.
pub fn check_in(node: &Node, id: &str) -> Out {
    need_id(id, vec![])?;
    let mut m = Map::new();
    m.insert("last_seen".into(), json!(now()));
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_error`.
pub fn update_error(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_error_request", body, &[("error", Kind::StrOrNull)])?;
    need_id(id, vec![])?;
    let err = body.get("error").cloned().unwrap_or(Value::Null);
    let mut m = Map::new();
    m.insert("status".into(), json!(if err.is_null() { "connected" } else { "error" }));
    m.insert("error".into(), err);
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_name`.
pub fn update_name(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_name_request", body, &[("name", Kind::StrOrNull), ("description", Kind::StrOrNull)])?;
    let name = body.get("name").filter(|v| !v.is_null());
    let desc = body.get("description").filter(|v| !v.is_null());
    let mut errs = Vec::new();
    if name.is_none() && desc.is_none() {
        errs.push("[name] and [description] cannot both be [null]. Please provide a value for at least one of them.".to_string());
    }
    need_id(id, errs)?;
    let mut m = Map::new();
    if let Some(n) = name {
        m.insert("name".into(), n.clone());
    }
    if let Some(d) = desc {
        m.insert("description".into(), d.clone());
    }
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_service_type`.
pub fn update_service_type(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_service_type_request", body, &[("service_type", Kind::Str)])?;
    let st = body.get("service_type").and_then(Value::as_str);
    let mut errs = Vec::new();
    if st.is_none_or(str::is_empty) {
        errs.push("[service_type] cannot be [null] or [\"\"].".to_string());
    }
    need_id(id, errs)?;
    let mut doc = load_connector(node, id)?;
    let before = doc.clone();
    let status = if doc["status"].as_str() == Some("created") { "created" } else { "needs_configuration" };
    let mut m = Map::new();
    m.insert("service_type".into(), json!(st));
    m.insert("status".into(), json!(status));
    merge(&mut doc, &Value::Object(hash_map(m)));
    store_connector(node, id, &before, &doc)
}

/// `PUT /_connector/{id}/_index_name`.
pub fn update_index_name(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_index_name_request", body, &[("index_name", Kind::StrOrNull)])?;
    let ix = body.get("index_name").and_then(Value::as_str).map(str::to_string);
    let errs: Vec<String> = ix.as_deref().and_then(index_name_problem).into_iter().collect();
    need_id(id, errs)?;
    if read(node, CONNECTOR_INDEX, id)?.is_none() {
        return update_connector_doc(node, id, Map::new());
    }
    if let Some(i) = &ix {
        if index_in_use(node, i, Some(id))? {
            return Err(status_ex(format!("Index name [{i}] is used by another connector.")));
        }
    }
    let cur = load_connector(node, id)?;
    if let Some(i) = ix.as_deref().filter(|i| cur["is_native"].as_bool() == Some(true) && !i.starts_with(MANAGED_CONNECTOR_INDEX_PREFIX)) {
        return Err(status_ex(format!(
            "Index attached to an Elastic-managed connector must start with the prefix: [{MANAGED_CONNECTOR_INDEX_PREFIX}]. The index name in the payload [{i}] doesn't comply with this requirement."
        )));
    }
    let mut m = Map::new();
    m.insert("index_name".into(), json!(ix));
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_native`.
pub fn update_native(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_native_request", body, &[("is_native", Kind::Bool)])?;
    need_id(id, vec![])?;
    let native = body.get("is_native").map(as_bool).unwrap_or(false);
    let cur = load_connector(node, id)?;
    if let Some(ix) = cur["index_name"].as_str().filter(|ix| native && !ix.starts_with(MANAGED_CONNECTOR_INDEX_PREFIX)) {
        return Err(status_ex(format!(
            "The index name [{ix}] attached to the connector [{id}] must start with the required prefix: [{MANAGED_CONNECTOR_INDEX_PREFIX}] to be Elastic-managed. Please update the attached index first to comply with this requirement."
        )));
    }
    // A connected connector has to reconnect (ES 9: only `connected` moves to `configured`).
    let status = cur["status"].as_str().unwrap_or("created");
    let status = if status == "connected" && connector_transition_ok(status, "configured") { "configured" } else { status };
    let mut m = Map::new();
    m.insert("is_native".into(), json!(native));
    m.insert("status".into(), json!(status));
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_status`.
pub fn update_status(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_status_request", body, &[("status", Kind::Str)])?;
    let status = match body.get("status").and_then(Value::as_str) {
        Some(s) => Some(enum_value(s, CONNECTOR_STATUS, "ConnectorStatus").map_err(|e| failed_field(&src.val("/status"), "connector_update_status_request", "status", e))?),
        None => None,
    };
    let mut errs = Vec::new();
    if status.is_none() {
        errs.push("[status] cannot be [null].".to_string());
    }
    need_id(id, errs)?;
    let status = status.unwrap_or_default();
    let mut doc = load_connector(node, id)?;
    let cur = doc["status"].as_str().unwrap_or("created").to_string();
    if !connector_transition_ok(&cur, &status) {
        let msg = format!("Invalid transition attempt from [{cur}] to [{status}]. Such a status transition is not supported by the Connector Protocol.");
        return Err(status_ex(msg.clone()).caused_by(EsError::new(400, "connector_invalid_status_transition_exception", msg)));
    }
    let before = doc.clone();
    merge(&mut doc, &json!({"status": status}));
    store_connector(node, id, &before, &doc)
}

/// `PUT /_connector/{id}/_api_key_id`.
pub fn update_api_key_id(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    single(src, "connector_update_api_key_id_request", body, &[("api_key_id", Kind::StrOrNull), ("api_key_secret_id", Kind::StrOrNull)])?;
    let a = body.get("api_key_id").filter(|v| !v.is_null());
    let b = body.get("api_key_secret_id").filter(|v| !v.is_null());
    let mut errs = Vec::new();
    if a.is_none() && b.is_none() {
        errs.push("[api_key_id] and [api_key_secret_id] cannot both be [null]. Please provide a value for at least one of them.".to_string());
    }
    need_id(id, errs)?;
    let mut m = Map::new();
    if let Some(a) = a {
        m.insert("api_key_id".into(), a.clone());
    }
    if let Some(b) = b {
        m.insert("api_key_secret_id".into(), b.clone());
    }
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_features`.
pub fn update_features(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_features_request";
    single(src, OBJ, body, &[("features", Kind::Obj)])?;
    let enabled = &[("enabled", Kind::Bool)];
    let mut feats = Map::new();
    if let Some(f) = body.get("features") {
        let fail = |e: EsError| failed_field(&src.val("/features"), OBJ, "features", e);
        check_obj(src, "/features", "connector_features", f, &[
            ("document_level_security", Kind::Obj),
            ("incremental_sync", Kind::Obj),
            ("native_connector_api_keys", Kind::Obj),
            ("sync_rules", Kind::Obj),
        ], &[]).map_err(fail)?;
        for (k, v) in f.as_object().into_iter().flatten() {
            let p = child("/features", k);
            if k == "sync_rules" {
                check_obj(src, &p, "sync_rules_features", v, &[("advanced", Kind::Obj), ("basic", Kind::Obj)], &[]).map_err(fail)?;
                let mut sr = Map::new();
                for (sk, sv) in v.as_object().into_iter().flatten() {
                    check_obj(src, &child(&p, sk), "feature_enabled", sv, enabled, &[]).map_err(fail)?;
                    sr.insert(sk.clone(), json!({"enabled": sv.get("enabled").map(as_bool)}));
                }
                feats.insert(k.clone(), Value::Object(sr));
            } else {
                check_obj(src, &p, "feature_enabled", v, enabled, &[]).map_err(fail)?;
                feats.insert(k.clone(), json!({"enabled": v.get("enabled").map(as_bool)}));
            }
        }
    }
    let mut errs = Vec::new();
    if body.get("features").is_none() {
        errs.push("[features] cannot be [null].".to_string());
    }
    need_id(id, errs)?;
    let mut m = Map::new();
    m.insert("features".into(), Value::Object(feats));
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_pipeline`.
pub fn update_pipeline(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_pipeline_request";
    single(src, OBJ, body, &[("pipeline", Kind::Obj)])?;
    let mut errs = Vec::new();
    match body.get("pipeline") {
        Some(p) => {
            let fields = &[("extract_binary_content", Kind::Bool), ("name", Kind::Str), ("reduce_whitespace", Kind::Bool), ("run_ml_inference", Kind::Bool)];
            check_obj(src, "/pipeline", "ingest_pipeline_params", p, fields, &["extract_binary_content", "name", "reduce_whitespace", "run_ml_inference"])
                .map_err(|e| if e.kind == "illegal_argument_exception" { failed_field(&src.end("/pipeline"), OBJ, "pipeline", e) } else { failed_field(&src.val("/pipeline"), OBJ, "pipeline", e) })?;
        }
        None => errs.push("[pipeline] cannot be [null].".to_string()),
    }
    need_id(id, errs)?;
    let p = &body["pipeline"];
    let pipeline = json!({
        "extract_binary_content": p.get("extract_binary_content").map(as_bool),
        "name": p["name"],
        "reduce_whitespace": p.get("reduce_whitespace").map(as_bool),
        "run_ml_inference": p.get("run_ml_inference").map(as_bool),
    });
    let mut m = Map::new();
    m.insert("pipeline".into(), pipeline);
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_scheduling`.
pub fn update_scheduling(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_scheduling_request";
    single(src, OBJ, body, &[("scheduling", Kind::Obj)])?;
    let mut sched = Map::new();
    let mut errs = Vec::new();
    match body.get("scheduling") {
        Some(s) => {
            let outer = |loc: &str, e: EsError| failed_field(loc, OBJ, "scheduling", e);
            check_obj(src, "/scheduling", "connector_scheduling", s, &[("access_control", Kind::Obj), ("full", Kind::Obj), ("incremental", Kind::Obj)], &[])
                .map_err(|e| outer(&src.val("/scheduling"), e))?;
            for (k, v) in s.as_object().into_iter().flatten() {
                let p = child("/scheduling", k);
                let mid = |loc: &str, e: EsError| outer(loc, xparse(loc, "connector_scheduling", format!("failed to parse field [{k}]")).caused_by(e));
                check_obj(src, &p, "schedule_config", v, &[("enabled", Kind::Bool), ("interval", Kind::Str)], &["enabled", "interval"]).map_err(|e| {
                    if e.kind == "illegal_argument_exception" { mid(&src.end(&p), e) } else { mid(&src.val(&p), e) }
                })?;
                let interval = v["interval"].as_str().unwrap_or("");
                if let Some(why) = cron_problem(interval) {
                    let ip = child(&p, "interval");
                    let loc = src.val(&ip);
                    let cause = xparse(&loc, "schedule_config", "failed to parse field [interval]").caused_by(
                        EsError::new(400, "x_content_parse_exception", "Failed to build [schedule_config] after last required field arrived")
                            .caused_by(iae(format!("invalid cron expression [{interval}]")).caused_by(iae(format!("illegal cron expression format [{why}]")))),
                    );
                    return Err(mid(&loc, cause));
                }
                sched.insert(k.clone(), json!({"enabled": as_bool(&v["enabled"]), "interval": interval}));
            }
            if sched.is_empty() {
                errs.push("[scheduling] object needs to define at least one schedule type: [full | incremental | access_control]".to_string());
            }
        }
        None => errs.push("[scheduling] cannot be [null].".to_string()),
    }
    need_id(id, errs)?;
    let mut m = Map::new();
    m.insert("scheduling".into(), Value::Object(sched));
    update_connector_doc(node, id, m)
}

/// `PUT /_connector/{id}/_last_sync`.
pub fn last_sync(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_last_sync_stats_request";
    const DATES: &[&str] = &["last_access_control_sync_scheduled_at", "last_incremental_sync_scheduled_at", "last_seen", "last_sync_scheduled_at", "last_synced"];
    const STATUSES: &[&str] = &["last_access_control_sync_status", "last_sync_status"];
    let fields: &[(&str, Kind)] = &[
        ("last_access_control_sync_error", Kind::StrOrNull),
        ("last_access_control_sync_scheduled_at", Kind::Any),
        ("last_access_control_sync_status", Kind::StrOrNull),
        ("last_deleted_document_count", Kind::Long),
        ("last_incremental_sync_scheduled_at", Kind::Any),
        ("last_indexed_document_count", Kind::Long),
        ("last_seen", Kind::Any),
        ("last_sync_error", Kind::StrOrNull),
        ("last_sync_scheduled_at", Kind::Any),
        ("last_sync_status", Kind::StrOrNull),
        ("last_synced", Kind::Any),
        ("sync_cursor", Kind::Any),
    ];
    single(src, OBJ, body, fields)?;
    let mut m = Map::new();
    for (k, v) in body.as_object().into_iter().flatten() {
        let p = child("", k);
        if v.is_null() {
            continue;
        }
        let out = if DATES.contains(&k.as_str()) {
            json!(parse_date(v).ok_or_else(|| date_err(src, &p, OBJ, k, v))?)
        } else if STATUSES.contains(&k.as_str()) {
            json!(enum_value(v.as_str().unwrap_or(""), SYNC_STATUS, "ConnectorSyncStatus").map_err(|e| failed_field(&src.val(&p), OBJ, k, e))?)
        } else if k.ends_with("_count") {
            json!(as_long(v))
        } else {
            v.clone()
        };
        m.insert(k.clone(), out);
    }
    need_id(id, vec![])?;
    update_connector_doc(node, id, m)
}

// ---- configuration

const CONFIG_FIELDS: &[(&str, Kind)] = &[
    ("category", Kind::StrOrNull),
    ("default_value", Kind::Any),
    ("depends_on", Kind::Arr),
    ("display", Kind::StrOrNull),
    ("label", Kind::Str),
    ("options", Kind::Arr),
    ("order", Kind::Long),
    ("placeholder", Kind::StrOrNull),
    ("required", Kind::Bool),
    ("sensitive", Kind::Bool),
    ("tooltip", Kind::StrOrNull),
    ("type", Kind::StrOrNull),
    ("ui_restrictions", Kind::Arr),
    ("validations", Kind::Arr),
    ("value", Kind::Any),
];

/// One `connector_configuration` entry; returns it as ES stores it.
fn parse_config_field(src: &Src, ptr: &str, v: &Value) -> Result<Value> {
    const C: &str = "connector_configuration";
    if !v.is_object() {
        return Err(xparse(&src.val(ptr), C, format!("{} doesn't support values of type: {}", ptr.rsplit('/').next().unwrap_or(""), token(v))));
    }
    check_obj(src, ptr, C, v, CONFIG_FIELDS, &[])?;
    let o = v.as_object().unwrap();
    for (k, en, java) in [("display", &["textbox", "textarea", "numeric", "toggle", "dropdown"][..], "ConfigurationDisplayType"), ("type", &["str", "int", "list", "bool"][..], "ConfigurationFieldType")] {
        if let Some(s) = o.get(k).and_then(Value::as_str) {
            enum_value(s, en, java).map_err(|e| failed_field(&src.val(&child(ptr, k)), C, k, e))?;
        }
    }
    if let Some(vals) = o.get("validations").and_then(Value::as_array) {
        for (i, val) in vals.iter().enumerate() {
            let vp = format!("{}/{i}", child(ptr, "validations"));
            let wrap = |e: EsError| failed_field(&src.val(&vp), C, "validations", e);
            check_obj(src, &vp, "connector_configuration_validation", val, &[("constraint", Kind::Any), ("type", Kind::Str)], &[]).map_err(wrap)?;
            if let Some(t) = val.get("type").and_then(Value::as_str) {
                let tp = child(&vp, "type");
                if let Err(e) = enum_value(t, &["less_than", "greater_than", "list_type", "included_in", "regex"], "ConfigurationValidationType") {
                    let loc = src.val(&tp);
                    return Err(failed_field(&loc, C, "validations", failed_field(&loc, "connector_configuration_validation", "type", e)));
                }
            }
        }
    }
    if !o.contains_key("label") {
        return Err(iae("Required [label]"));
    }
    let mut out = Map::new();
    for (k, val) in o {
        let stored = match k.as_str() {
            "validations" => Value::Array(val.as_array().cloned().unwrap_or_default().into_iter().map(|x| json!({"type": x["type"], "constraint": x["constraint"]})).collect()),
            "depends_on" | "options" => hash_deep(val.clone()),
            _ => val.clone(),
        };
        out.insert(k.clone(), stored);
    }
    out.entry("default_value").or_insert(Value::Null);
    out.entry("sensitive").or_insert(json!(false));
    out.entry("required").or_insert(json!(false));
    Ok(Value::Object(hash_map(out)))
}

/// `PUT /_connector/{id}/_configuration`.
pub fn update_configuration(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_configuration_request";
    single(src, OBJ, body, &[("configuration", Kind::ObjOrNull), ("values", Kind::ObjOrNull)])?;
    let config = body.get("configuration").filter(|v| !v.is_null());
    let values = body.get("values").filter(|v| !v.is_null());
    let mut parsed = Map::new();
    if let Some(c) = config {
        for (k, v) in c.as_object().into_iter().flatten() {
            let p = child("/configuration", k);
            let f = parse_config_field(src, &p, v).map_err(|e| {
                let loc = if e.kind == "illegal_argument_exception" { src.end(&p) } else { e.reason.split(' ').next().unwrap_or("").to_string() };
                failed_field(&loc, OBJ, "configuration", e)
            })?;
            parsed.insert(k.clone(), f);
        }
    }
    let mut errs = Vec::new();
    match (config.is_some(), values.is_some()) {
        (false, false) => errs.push("[configuration] and [values] cannot both be null.".to_string()),
        (true, true) => errs.push("[configuration] and [values] cannot both be provided in the same request.".to_string()),
        _ => {}
    }
    need_id(id, errs)?;
    let mut doc = load_connector(node, id)?;
    let before = doc.clone();
    if config.is_some() {
        doc["configuration"] = Value::Object(hash_map(parsed));
    } else if let Some(vals) = values.and_then(Value::as_object) {
        let unknown: Vec<&str> = vals.keys().filter(|k| doc["configuration"].get(k.as_str()).is_none()).map(String::as_str).collect();
        if !unknown.is_empty() {
            return Err(status_ex(format!(
                "Unknown [configuration] fields in the request payload: [{}]. Remove them from request or register their schema first.",
                unknown.join(", ")
            )));
        }
        for (k, v) in vals {
            doc["configuration"][k]["value"] = v.clone();
        }
    }
    merge(&mut doc, &json!({"status": "configured"}));
    store_connector(node, id, &before, &doc)
}

// ---- filtering

const RULE_FIELDS: &[(&str, Kind)] = &[
    ("created_at", Kind::Any),
    ("field", Kind::Str),
    ("id", Kind::Str),
    ("order", Kind::Long),
    ("policy", Kind::Str),
    ("rule", Kind::Str),
    ("updated_at", Kind::Any),
    ("value", Kind::Str),
];

/// Validate and normalise one filtering rule (dates re-rendered).
fn parse_rule(src: &Src, ptr: &str, v: &Value, wrap: &dyn Fn(&str, EsError) -> EsError) -> Result<Value> {
    const R: &str = "connector_filtering_rule";
    check_obj(src, ptr, R, v, RULE_FIELDS, &["field", "id", "order", "policy", "rule", "value"])
        .map_err(|e| if e.kind == "illegal_argument_exception" { wrap(&src.end(ptr), e) } else { wrap(&src.val(ptr), e) })?;
    let mut o = v.as_object().cloned().unwrap_or_default();
    for k in ["created_at", "updated_at"] {
        match o.get(k) {
            Some(d) if !d.is_null() => {
                let p = child(ptr, k);
                let s = parse_date(d).ok_or_else(|| wrap(&src.val(&p), date_err(src, &p, R, k, d)))?;
                o.insert(k.into(), json!(s));
            }
            _ => {
                o.insert(k.into(), json!(now()));
            }
        }
    }
    let pol = o["policy"].as_str().unwrap_or("").to_string();
    enum_value(&pol, &["exclude", "include"], "FilteringPolicy").map_err(|e| wrap(&src.val(&child(ptr, "policy")), failed_field(&src.val(&child(ptr, "policy")), R, "policy", e)))?;
    let rule = o["rule"].as_str().unwrap_or("").to_string();
    enum_value(&rule, &["contains", "ends_with", "equals", "regex", "starts_with", ">", "<"], "FilteringRuleCondition")
        .map_err(|e| wrap(&src.val(&child(ptr, "rule")), failed_field(&src.val(&child(ptr, "rule")), R, "rule", e)))?;
    o.insert("order".into(), json!(as_long(&o["order"])));
    Ok(Value::Object(o))
}

fn parse_snippet(src: &Src, ptr: &str, v: &Value, wrap: &dyn Fn(&str, EsError) -> EsError) -> Result<Value> {
    const A: &str = "connector_filtering_advanced_snippet";
    check_obj(src, ptr, A, v, &[("created_at", Kind::Any), ("updated_at", Kind::Any), ("value", Kind::ObjOrArr)], &["value"])
        .map_err(|e| if e.kind == "illegal_argument_exception" { wrap(&src.end(ptr), e) } else { wrap(&src.val(&first_bad(ptr, v)), e) })?;
    let mut o = Map::new();
    for k in ["created_at", "updated_at"] {
        let s = match v.get(k) {
            Some(d) if !d.is_null() => {
                let p = child(ptr, k);
                parse_date(d).ok_or_else(|| wrap(&src.val(&p), date_err(src, &p, A, k, d)))?
            }
            _ => now(),
        };
        o.insert(k.into(), json!(s));
    }
    o.insert("value".into(), v["value"].clone());
    Ok(Value::Object(o))
}

/// The member of `v` whose error is reported (first mistyped one).
fn first_bad(ptr: &str, v: &Value) -> String {
    if let Some(val) = v.get("value") {
        if !val.is_object() && !val.is_array() {
            return child(ptr, "value");
        }
    }
    ptr.to_string()
}

/// Sort rules by `order`; the `DEFAULT` rule always goes last.
fn order_rules(rules: Vec<Value>) -> Vec<Value> {
    let (mut def, mut rest): (Vec<Value>, Vec<Value>) = rules.into_iter().partition(|r| r["id"] == "DEFAULT");
    rest.sort_by_key(|r| r["order"].as_i64().unwrap_or(0));
    let order = rest.last().map(|r| r["order"].as_i64().unwrap_or(0) + 1).unwrap_or(0);
    let mut d = def.pop().unwrap_or_else(|| default_rule(&now(), order));
    d["order"] = json!(order);
    rest.push(d);
    rest
}

/// `PUT /_connector/{id}/_filtering`.
pub fn update_filtering(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_filtering_request";
    single(src, OBJ, body, &[("filtering", Kind::Arr), ("advanced_snippet", Kind::Obj), ("rules", Kind::Arr)])?;
    let filtering = body.get("filtering").filter(|v| !v.is_null());
    let snippet_in = body.get("advanced_snippet").filter(|v| !v.is_null());
    let rules_in = body.get("rules").filter(|v| !v.is_null());
    let mut full = None;
    if let Some(f) = filtering {
        let mut list = Vec::new();
        for (i, entry) in f.as_array().into_iter().flatten().enumerate() {
            let ep = format!("/filtering/{i}");
            let wrap = |loc: &str, e: EsError| failed_field(loc, OBJ, "filtering", e);
            check_obj(src, &ep, "connector_filtering", entry, &[("active", Kind::Obj), ("domain", Kind::Str), ("draft", Kind::Obj)], &["active", "draft"])
                .map_err(|e| if e.kind == "illegal_argument_exception" { wrap(&src.end(&ep), e) } else { wrap(&src.val(&ep), e) })?;
            let mut out = Map::new();
            for side in ["active", "draft"] {
                let sp = child(&ep, side);
                let s = &entry[side];
                check_obj(src, &sp, "connector_filtering_rules", s, &[("advanced_snippet", Kind::Obj), ("rules", Kind::Arr), ("validation", Kind::Obj)], &["advanced_snippet", "rules", "validation"])
                    .map_err(|e| if e.kind == "illegal_argument_exception" { wrap(&src.end(&sp), e) } else { wrap(&src.val(&sp), e) })?;
                let snip = parse_snippet(src, &child(&sp, "advanced_snippet"), &s["advanced_snippet"], &wrap)?;
                let mut rules = Vec::new();
                for (j, r) in s["rules"].as_array().into_iter().flatten().enumerate() {
                    rules.push(parse_rule(src, &format!("{}/{j}", child(&sp, "rules")), r, &wrap)?);
                }
                let val = parse_validation(src, &child(&sp, "validation"), &s["validation"], &wrap)?;
                out.insert(side.into(), json!({"advanced_snippet": snip, "rules": rules, "validation": val}));
            }
            out.insert("domain".into(), entry.get("domain").cloned().unwrap_or(json!("DEFAULT")));
            list.push(Value::Object(out));
        }
        full = Some(list);
    }
    let wrap_s = |loc: &str, e: EsError| failed_field(loc, OBJ, "advanced_snippet", e);
    let snippet = match snippet_in {
        Some(s) => Some(parse_snippet(src, "/advanced_snippet", s, &wrap_s)?),
        None => None,
    };
    let wrap_r = |loc: &str, e: EsError| failed_field(loc, OBJ, "rules", e);
    let rules = match rules_in {
        Some(r) => {
            let mut v = Vec::new();
            for (j, x) in r.as_array().into_iter().flatten().enumerate() {
                v.push(parse_rule(src, &format!("/rules/{j}"), x, &wrap_r)?);
            }
            Some(v)
        }
        None => None,
    };
    let mut errs = Vec::new();
    if full.is_none() && snippet.is_none() && rules.is_none() {
        errs.push("[advanced_snippet] and [rules] cannot be both [null].".to_string());
    }
    need_id(id, errs)?;
    let mut doc = load_connector(node, id)?;
    let before = doc.clone();
    let new_filtering = match full {
        Some(list) => Value::Array(list),
        None => {
            let mut list = doc["filtering"].as_array().cloned().unwrap_or_default();
            if list.is_empty() {
                list = default_filtering(&now()).as_array().cloned().unwrap_or_default();
            }
            let idx = list.iter().position(|f| f["domain"] == "DEFAULT").unwrap_or(0);
            let draft = &mut list[idx]["draft"];
            if let Some(s) = snippet {
                draft["advanced_snippet"] = s;
            }
            if let Some(r) = rules {
                draft["rules"] = Value::Array(order_rules(r));
            }
            draft["validation"] = json!({"errors": [], "state": "edited"});
            Value::Array(list)
        }
    };
    doc["filtering"] = hash_deep(new_filtering);
    store_connector(node, id, &before, &doc)
}

fn parse_validation(src: &Src, ptr: &str, v: &Value, wrap: &dyn Fn(&str, EsError) -> EsError) -> Result<Value> {
    const V: &str = "filtering_validation_info";
    check_obj(src, ptr, V, v, &[("errors", Kind::Arr), ("state", Kind::Str)], &["errors", "state"])
        .map_err(|e| if e.kind == "illegal_argument_exception" { wrap(&src.end(ptr), e) } else { wrap(&src.val(ptr), e) })?;
    let state = v["state"].as_str().unwrap_or("");
    enum_value(state, &["edited", "invalid", "valid"], "FilteringValidationState").map_err(|e| wrap(&src.val(&child(ptr, "state")), failed_field(&src.val(&child(ptr, "state")), V, "state", e)))?;
    let mut errors = Vec::new();
    for (i, e) in v["errors"].as_array().into_iter().flatten().enumerate() {
        let p = format!("{}/{i}", child(ptr, "errors"));
        check_obj(src, &p, "filtering_validation", e, &[("ids", Kind::Arr), ("messages", Kind::Arr)], &["ids", "messages"])
            .map_err(|er| if er.kind == "illegal_argument_exception" { wrap(&src.end(&p), er) } else { wrap(&src.val(&p), er) })?;
        errors.push(json!({"ids": e["ids"], "messages": e["messages"]}));
    }
    Ok(json!({"errors": errors, "state": state}))
}

/// `PUT /_connector/{id}/_filtering/_validation`.
pub fn update_filtering_validation(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_update_filtering_validation";
    single(src, OBJ, body, &[("validation", Kind::Obj)])?;
    let wrap = |loc: &str, e: EsError| failed_field(loc, OBJ, "validation", e);
    let val = match body.get("validation") {
        Some(v) => Some(parse_validation(src, "/validation", v, &wrap)?),
        None => None,
    };
    let mut errs = Vec::new();
    if val.is_none() {
        errs.push("[validation] cannot be [null].".to_string());
    }
    need_id(id, errs)?;
    let mut doc = load_connector(node, id)?;
    let before = doc.clone();
    let mut list = doc["filtering"].as_array().cloned().unwrap_or_default();
    let idx = list.iter().position(|f| f["domain"] == "DEFAULT").unwrap_or(0);
    if let Some(f) = list.get_mut(idx) {
        f["draft"]["validation"] = val.unwrap_or(Value::Null);
    }
    doc["filtering"] = hash_deep(Value::Array(list));
    store_connector(node, id, &before, &doc)
}

/// `PUT /_connector/{id}/_filtering/_activate`.
pub fn activate_filtering(node: &Node, id: &str) -> Out {
    need_id(id, vec![])?;
    let mut doc = load_connector(node, id)?;
    let before = doc.clone();
    let mut list = doc["filtering"].as_array().cloned().unwrap_or_default();
    let idx = list.iter().position(|f| f["domain"] == "DEFAULT").unwrap_or(0);
    if let Some(f) = list.get_mut(idx) {
        let state = f["draft"]["validation"]["state"].as_str().unwrap_or("").to_string();
        if state != "valid" {
            return Err(status_ex(format!(
                "Filtering draft needs to be validated by the connector service before activation. Current filtering draft validation state [{state}] is not equal to [valid]."
            )));
        }
        f["active"] = f["draft"].clone();
    }
    doc["filtering"] = hash_deep(Value::Array(list));
    store_connector(node, id, &before, &doc)
}

// ------------------------------------------------------------------ secrets

fn secret_value(src: &Src, obj: &str, body: &Value) -> Result<String> {
    single(src, obj, body, &[("value", Kind::Str)])?;
    match body.get("value").and_then(Value::as_str) {
        Some(v) if !v.is_empty() => Ok(v.to_string()),
        _ => Err(EsError::action_validation(&["[value] of the connector secret cannot be [null] or [\"\"]"])),
    }
}

fn no_secret(id: &str) -> EsError {
    EsError::resource_not_found(format!("No secret with id [{id}]"))
}

/// `POST /_connector/_secret`.
pub fn secret_post(node: &Node, body: &Value, src: &Src) -> Out {
    let v = secret_value(src, "post_secret_request", body)?;
    let id = crate::index::auto_id();
    write(node, SECRETS_INDEX, &id, &json!({"value": v}))?;
    Ok((200, json!({"id": id})))
}

/// `PUT /_connector/_secret/{id}`.
pub fn secret_put(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    let v = secret_value(src, "connector_secret_put_request", body)?;
    let created = write(node, SECRETS_INDEX, id, &json!({"value": v}))?;
    Ok((if created { 201 } else { 200 }, json!({"result": if created { "created" } else { "updated" }})))
}

/// `GET /_connector/_secret/{id}`.
pub fn secret_get(node: &Node, id: &str) -> Out {
    let doc = read(node, SECRETS_INDEX, id)?.ok_or_else(|| no_secret(id))?;
    Ok((200, json!({"id": id, "value": doc["value"]})))
}

/// `DELETE /_connector/_secret/{id}`.
pub fn secret_delete(node: &Node, id: &str) -> Out {
    if !delete(node, SECRETS_INDEX, id)? {
        return Err(no_secret(id));
    }
    Ok((200, json!({"deleted": true})))
}

// ------------------------------------------------------------------ sync jobs

fn job_missing(id: &str) -> EsError {
    EsError::resource_not_found(id.to_string())
}

fn job_not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("connector sync job [{id}] not found"))
}

fn need_job_id(id: &str, extra: Vec<String>) -> Result<()> {
    let mut errs = Vec::new();
    if id.is_empty() {
        errs.push(empty_job_id().to_string());
    }
    errs.extend(extra);
    validation(&errs)
}

/// The connector snapshot a sync job carries (`toXContent` key order).
fn canonical_filtering(f: &Value) -> Value {
    let rules: Vec<Value> = f["rules"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| {
            let mut o = Map::new();
            for k in ["created_at", "field", "id", "order", "policy", "rule", "updated_at", "value"] {
                if let Some(v) = r.get(k) {
                    o.insert(k.into(), v.clone());
                }
            }
            Value::Object(o)
        })
        .collect();
    let a = &f["advanced_snippet"];
    json!({
        "advanced_snippet": {"created_at": a["created_at"], "updated_at": a["updated_at"], "value": a["value"]},
        "rules": rules,
        "validation": {"errors": f["validation"]["errors"], "state": f["validation"]["state"]},
    })
}

/// `POST /_connector/_sync_job`.
pub fn sync_job_post(node: &Node, body: &Value, src: &Src) -> Out {
    sync_job_post_with_reader(node, body, src, &|id| read(node, CONNECTOR_INDEX, id))
}

/// The cluster supplies a primary-routed connector lookup while the sync job
/// itself is written under its own primary's replication guard.
pub fn sync_job_post_with_reader(node: &Node, body: &Value, src: &Src, reader: &dyn Fn(&str) -> Result<Option<Value>>) -> Out {
    const OBJ: &str = "connector_sync_job_post_request";
    if !body.is_object() {
        return Err(EsError::parsing("request body must be an object"));
    }
    check_obj(src, "", OBJ, body, &[("id", Kind::Str), ("job_type", Kind::StrOrNull), ("trigger_method", Kind::StrOrNull)], &["id"])?;
    let build = |e: EsError| EsError::new(400, "x_content_parse_exception", format!("Failed to build [{OBJ}] after last required field arrived")).caused_by(e);
    let job_type = match body.get("job_type").and_then(Value::as_str) {
        Some(t) => enum_value(t, JOB_TYPES, "ConnectorSyncJobType").map_err(build)?,
        None => "full".into(),
    };
    let trigger = match body.get("trigger_method").and_then(Value::as_str) {
        Some(t) => enum_value(t, TRIGGERS, "ConnectorSyncJobTriggerMethod").map_err(build)?,
        None => "on_demand".into(),
    };
    let cid = body["id"].as_str().unwrap_or("").to_string();
    if cid.is_empty() {
        return Err(EsError::action_validation(&["[id] of the connector cannot be null or empty"]));
    }
    let conn = reader(&cid)?.ok_or_else(|| EsError::resource_not_found(format!("Connector with id '{cid}' does not exist.")))?;
    let Some(index) = conn["index_name"].as_str() else {
        return Err(status_ex(format!(
            "Cannot start a sync for connector [{cid}] with no index attached. Set the [index_name] property for the connector to enable syncing data."
        )));
    };
    if conn["service_type"].as_str().is_none() {
        return Err(status_ex(format!(
            "Cannot start a sync for connector [{cid}] with [service_type] not defined. Set the service type of your connector before starting the sync."
        )));
    }
    let index = if job_type == "access_control" { format!(".search-acl-filter-{index}") } else { index.to_string() };
    let mut c = Map::new();
    c.insert("id".into(), json!(cid));
    let active = conn["filtering"].as_array().and_then(|l| l.first()).map(|f| canonical_filtering(&f["active"])).unwrap_or(Value::Null);
    c.insert("filtering".into(), active);
    c.insert("index_name".into(), json!(index));
    for k in ["language", "pipeline", "service_type"] {
        if let Some(v) = conn.get(k).filter(|v| !v.is_null()) {
            c.insert(k.into(), v.clone());
        }
    }
    c.insert("configuration".into(), conn.get("configuration").cloned().unwrap_or(json!({})));
    let ts = now();
    let doc = json!({
        "connector": c,
        "created_at": ts,
        "deleted_document_count": 0,
        "indexed_document_count": 0,
        "indexed_document_volume": 0,
        "job_type": job_type,
        "last_seen": ts,
        "metadata": {},
        "status": "pending",
        "total_document_count": 0,
        "trigger_method": trigger,
    });
    let id = crate::index::auto_id();
    write(node, SYNC_JOB_INDEX, &id, &doc)?;
    Ok((200, json!({"id": id})))
}

/// `GET /_connector/_sync_job/{id}`.
pub fn sync_job_get(node: &Node, id: &str) -> Out {
    need_job_id(id, vec![])?;
    let doc = read(node, SYNC_JOB_INDEX, id)?.ok_or_else(|| job_missing(id))?;
    Ok((200, with_id(id, &doc)))
}

/// `DELETE /_connector/_sync_job/{id}`.
pub fn sync_job_delete(node: &Node, id: &str) -> Out {
    need_job_id(id, vec![])?;
    if !delete(node, SYNC_JOB_INDEX, id)? {
        return Err(job_missing(id));
    }
    Ok((200, json!({"acknowledged": true})))
}

fn update_job(node: &Node, id: &str, missing: fn(&str) -> EsError, f: impl FnOnce(&Value) -> Result<Map<String, Value>>) -> Out {
    let mut doc = read(node, SYNC_JOB_INDEX, id)?.ok_or_else(|| missing(id))?;
    let upd = f(&doc)?;
    let before = doc.clone();
    merge(&mut doc, &Value::Object(hash_map(upd)));
    if doc == before {
        return Ok((200, result("noop")));
    }
    write(node, SYNC_JOB_INDEX, id, &doc)?;
    Ok((200, result("updated")))
}

/// `PUT /_connector/_sync_job/{id}/_check_in`.
pub fn sync_job_check_in(node: &Node, id: &str) -> Out {
    need_job_id(id, vec![])?;
    update_job(node, id, job_not_found, |_| {
        let mut m = Map::new();
        m.insert("last_seen".into(), json!(now()));
        Ok(m)
    })
}

/// `PUT /_connector/_sync_job/{id}/_cancel`.
pub fn sync_job_cancel(node: &Node, id: &str) -> Out {
    need_job_id(id, vec![])?;
    update_job(node, id, job_missing, |doc| {
        let cur = doc["status"].as_str().unwrap_or("").to_string();
        let ts = now();
        let mut m = Map::new();
        if cur == "pending" || cur == "suspended" {
            m.insert("status".into(), json!("canceled"));
            m.insert("completed_at".into(), json!(ts));
            m.insert("cancelation_requested_at".into(), json!(ts));
            m.insert("canceled_at".into(), json!(ts));
            return Ok(m);
        }
        if !sync_transition_ok(&cur, "canceling") {
            return Err(sync_transition_err(&cur, "canceling"));
        }
        m.insert("status".into(), json!("canceling"));
        m.insert("cancelation_requested_at".into(), json!(ts));
        Ok(m)
    })
}

/// `PUT /_connector/_sync_job/{id}/_error`.
pub fn sync_job_error(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_sync_job_error";
    single(src, OBJ, body, &[("error", Kind::Str)])?;
    if body.get("error").is_none() {
        return Err(iae("Required [error]"));
    }
    need_job_id(id, vec![])?;
    let err = body["error"].clone();
    update_job(node, id, job_missing, |doc| {
        let cur = doc["status"].as_str().unwrap_or("");
        if !sync_transition_ok(cur, "error") {
            return Err(sync_transition_err(cur, "error"));
        }
        let mut m = Map::new();
        m.insert("error".into(), err);
        m.insert("status".into(), json!("error"));
        Ok(m)
    })
}

/// `PUT /_connector/_sync_job/{id}/_stats`.
pub fn sync_job_stats(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "connector_sync_job_update_ingestion_stats";
    let fields: &[(&str, Kind)] = &[
        ("deleted_document_count", Kind::Long),
        ("indexed_document_count", Kind::Long),
        ("indexed_document_volume", Kind::Long),
        ("total_document_count", Kind::Long),
        ("last_seen", Kind::Any),
        ("metadata", Kind::Obj),
    ];
    if !body.is_object() {
        return Err(EsError::parsing("request body must be an object"));
    }
    check_obj(src, "", OBJ, body, fields, &["deleted_document_count", "indexed_document_count", "indexed_document_volume"])?;
    let last_seen = match body.get("last_seen").filter(|v| !v.is_null()) {
        Some(v) => Some(parse_date(v).ok_or_else(|| date_err(src, "/last_seen", OBJ, "last_seen", v))?),
        None => None,
    };
    let mut errs = Vec::new();
    for k in ["deleted_document_count", "indexed_document_count", "indexed_document_volume", "total_document_count"] {
        if body.get(k).map(as_long).is_some_and(|n| n < 0) {
            errs.push(format!("[{k}] cannot be negative."));
        }
    }
    need_job_id(id, errs)?;
    update_job(node, id, job_not_found, |_| {
        let mut m = Map::new();
        for k in ["deleted_document_count", "indexed_document_count", "indexed_document_volume", "total_document_count"] {
            if let Some(v) = body.get(k).filter(|v| !v.is_null()) {
                m.insert(k.into(), json!(as_long(v)));
            }
        }
        m.insert("last_seen".into(), json!(last_seen.clone().unwrap_or_else(now)));
        if let Some(md) = body.get("metadata").filter(|v| !v.is_null()) {
            m.insert("metadata".into(), md.clone());
        }
        Ok(m)
    })
}

/// `PUT /_connector/_sync_job/{id}/_claim`.
pub fn sync_job_claim(node: &Node, id: &str, body: &Value, src: &Src) -> Out {
    const OBJ: &str = "claim_connector_sync_job";
    if !body.is_object() {
        return Err(EsError::parsing("request body must be an object"));
    }
    check_obj(src, "", OBJ, body, &[("worker_hostname", Kind::Str), ("sync_cursor", Kind::Any)], &["worker_hostname"])?;
    need_job_id(id, vec![])?;
    let host = body["worker_hostname"].clone();
    let cursor = body.get("sync_cursor").filter(|v| !v.is_null()).cloned();
    update_job(node, id, job_missing, |_| {
        let ts = now();
        let mut m = Map::new();
        m.insert("last_seen".into(), json!(ts));
        m.insert("started_at".into(), json!(ts));
        m.insert("status".into(), json!("in_progress"));
        m.insert("worker_hostname".into(), host);
        if let Some(c) = cursor {
            m.insert("connector".into(), json!({"sync_cursor": c}));
        }
        Ok(m)
    })
}

/// Parameters of `GET /_connector/_sync_job`.
pub struct JobListParams<'a> {
    pub from: i64,
    pub size: i64,
    pub connector_id: Option<&'a str>,
    pub status: Option<&'a str>,
    pub job_type: Option<&'a str>,
}

/// `GET /_connector/_sync_job`: newest first.
pub fn sync_job_list(node: &Node, p: &JobListParams) -> Out {
    let status = p.status.map(|s| enum_value(s, SYNC_STATUS, "ConnectorSyncStatus")).transpose()?;
    let types = csv(p.job_type);
    for t in &types {
        enum_value(t, JOB_TYPES, "ConnectorSyncJobType")?;
    }
    let (from, size) = page(p.from, p.size)?;
    let mut all: Vec<(usize, String, Value)> = scan_all(node, SYNC_JOB_INDEX)?
        .into_iter()
        .enumerate()
        .filter(|(_, (_, s))| {
            p.connector_id.is_none_or(|c| s["connector"]["id"].as_str() == Some(c))
                && status.as_ref().is_none_or(|st| s["status"].as_str() == Some(st.as_str()))
                && (types.is_empty() || types.iter().any(|t| s["job_type"].as_str() == Some(t.as_str())))
        })
        .map(|(i, (id, s))| (i, id, s))
        .collect();
    all.sort_by(|a, b| {
        let ka = a.2["created_at"].as_str().and_then(sort_nanos).unwrap_or(0);
        let kb = b.2["created_at"].as_str().and_then(sort_nanos).unwrap_or(0);
        kb.cmp(&ka).then(b.0.cmp(&a.0))
    });
    let count = all.len();
    let results: Vec<Value> = all.iter().skip(from).take(size).map(|(_, id, s)| with_id(id, s)).collect();
    Ok((200, json!({"count": count, "results": results})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions() {
        let s = Src::new(br#"{"pipeline":{"extract_binary_content":true,"name":"p","run_ml_inference":false}}"#);
        assert_eq!(s.end("/pipeline"), "[1:79]");
        let s = Src::new(br#"{"error":{"a":1}}"#);
        assert_eq!(s.val("/error"), "[1:10]");
        let s = Src::new(br#"{"foo":1}"#);
        assert_eq!(s.key("/foo"), "[1:2]");
    }

    #[test]
    fn hash_order() {
        let keys: Vec<String> = ["created_at", "field", "id", "order", "policy", "rule", "updated_at", "value"].iter().map(|s| s.to_string()).collect();
        assert_eq!(java_order(&keys), vec!["field", "updated_at", "created_at", "rule", "id", "value", "order", "policy"]);
    }

    #[test]
    fn cron() {
        assert!(cron_problem("0 0 0 * * ?").is_none());
        assert!(cron_problem("61 0 0 * * ?").unwrap().contains("between 0 and 59"));
    }

    #[test]
    fn nanos() {
        assert_eq!(parse_date(&json!("2023-05-25T12:30:00.000Z")).unwrap(), "2023-05-25T12:30:00.000Z");
        assert_eq!(parse_date(&json!("2023-12-04T08:45:50.567149Z")).unwrap(), "2023-12-04T08:45:50.567149Z");
    }
}
