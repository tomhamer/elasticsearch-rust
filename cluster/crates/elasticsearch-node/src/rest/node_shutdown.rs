//! Node shutdown API (x-pack `shutdown` plugin), ES 8.19 semantics.
//!
//! * `PUT /_nodes/{node_id}/shutdown`
//! * `GET /_nodes/shutdown`, `GET /_nodes/{node_id}/shutdown`
//! * `DELETE /_nodes/{node_id}/shutdown`
//!
//! Records live in the `node_shutdown` metadata custom
//! (`{"nodes": {id: record}}`, ES `NodesShutdownMetadata`). The allocator
//! reads them (`allocation::context::all_shutdowns`): `remove` / `replace`
//! nodes are drained, `restart` nodes keep their replicas unassigned for
//! `allocation_delay` (default 5m) instead of reallocating them. The status
//! reported by `GET` is computed on the master from the live routing table
//! (ES `TransportGetShutdownStatusAction`).

use crate::front::{RestReq, RestResp};
use crate::master::now_millis;
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// Metadata custom name (ES `NodesShutdownMetadata.TYPE`).
pub const CUSTOM: &str = "node_shutdown";

const PARSER: &str = "put_node_shutdown_request";
pub(super) const COMMON_PARAMS: &[&str] = &["pretty", "human", "error_trace", "filter_path", "format"];
/// ES `SingleNodeShutdownMetadata.DEFAULT_RESTART_SHARD_ALLOCATION_DELAY`.
const DEFAULT_RESTART_DELAY: &str = "5m";

// ------------------------------------------------------------ errors

#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub error: Value,
}

impl ApiError {
    pub(super) fn simple(status: u16, kind: &str, reason: &str) -> ApiError {
        ApiError { status, error: json!({"root_cause": [{"type": kind, "reason": reason}], "type": kind, "reason": reason}) }
    }

    /// An exception with a cause chain; `root_cause` is the outermost
    /// exception's type and reason (as ES reports for these parse errors).
    pub(super) fn chained(status: u16, outer: Value) -> ApiError {
        let mut e = outer.clone();
        let root = json!({"type": outer["type"], "reason": outer["reason"]});
        if let Some(m) = e.as_object_mut() {
            let mut out = Map::new();
            out.insert("root_cause".into(), json!([root]));
            for (k, v) in std::mem::take(m) {
                out.insert(k, v);
            }
            *m = out;
        }
        ApiError { status, error: e }
    }

    pub fn resp(&self) -> RestResp {
        RestResp::json(self.status, &json!({"error": self.error, "status": self.status}))
    }

    pub(super) fn encode(&self) -> String {
        format!("node_shutdown|{}|{}", self.status, self.error)
    }

    fn decode(s: &str) -> Option<ApiError> {
        let rest = s.strip_prefix("node_shutdown|")?;
        let (status, err) = rest.split_once('|')?;
        Some(ApiError { status: status.parse().ok()?, error: serde_json::from_str(err).ok()? })
    }
}

pub(super) fn exc(kind: &str, reason: impl Into<String>) -> Value {
    json!({"type": kind, "reason": reason.into()})
}

pub(super) fn task_error(e: String) -> RestResp {
    match ApiError::decode(&e) {
        Some(a) => a.resp(),
        None => RestResp::error(500, "illegal_state_exception", &e),
    }
}

// ------------------------------------------------------------ TimeValue

/// ES `TimeValue` (duration in its own unit + the unit suffix).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeValue {
    pub duration: i64,
    /// `nanos`, `micros`, `ms`, `s`, `m`, `h`, `d`.
    pub unit: &'static str,
}

impl TimeValue {
    fn nanos_per(unit: &str) -> i128 {
        match unit {
            "nanos" => 1,
            "micros" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60_000_000_000,
            "h" => 3_600_000_000_000,
            _ => 86_400_000_000_000,
        }
    }

    pub fn nanos(&self) -> i128 {
        self.duration as i128 * Self::nanos_per(self.unit)
    }

    /// `TimeValue.getStringRep`.
    pub fn string_rep(&self) -> String {
        if self.duration < 0 {
            return self.duration.to_string();
        }
        format!("{}{}", self.duration, self.unit)
    }

    /// `TimeValue.parseTimeValue(sValue, settingName)`; the error is the
    /// exception JSON (`illegal_argument_exception`, maybe with a
    /// `number_format_exception` cause).
    pub fn parse(s: &str, setting: &str) -> Result<TimeValue, Value> {
        let normalized = s.to_lowercase();
        let normalized = normalized.trim();
        let suffixes: [(&str, &'static str); 7] = [("nanos", "nanos"), ("micros", "micros"), ("ms", "ms"), ("s", "s"), ("m", "m"), ("h", "h"), ("d", "d")];
        for (suffix, unit) in suffixes {
            // Minutes are case-sensitive ("M" would be months).
            let hit = if suffix == "m" { s.ends_with('m') } else { normalized.ends_with(suffix) };
            if hit {
                let num = normalized[..normalized.len() - suffix.len()].trim();
                return parse_long(s, num, setting).map(|duration| TimeValue { duration, unit });
            }
        }
        let n = normalized;
        if n.starts_with('-') && n.len() >= 2 && n[1..n.len() - 1].chars().all(|c| c == '0') && n.ends_with('1') {
            return Ok(TimeValue { duration: -1, unit: "ms" });
        }
        if !n.is_empty() && n.chars().all(|c| c == '0') {
            return Ok(TimeValue { duration: 0, unit: "ms" });
        }
        Err(exc("illegal_argument_exception", format!("failed to parse setting [{setting}] with value [{s}] as a time value: unit is missing or unrecognized")))
    }
}

/// Java `Long.parseLong` acceptance: optional sign, ASCII digits, in range.
fn java_long(s: &str) -> Option<i64> {
    let digits = s.strip_prefix(['-', '+']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<i64>().ok()
}

/// Java `Double.parseDouble` acceptance (decimal forms; good enough for the
/// "fractional" distinction).
fn java_double(s: &str) -> bool {
    let t = s.trim();
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    !t.is_empty() && t.parse::<f64>().is_ok() && !t.eq_ignore_ascii_case("inf") && !t.eq_ignore_ascii_case("-inf") && !t.eq_ignore_ascii_case("+inf")
}

fn parse_long(initial: &str, num: &str, setting: &str) -> Result<i64, Value> {
    match java_long(num) {
        Some(v) if v < -1 => Err(exc(
            "illegal_argument_exception",
            format!("failed to parse setting [{setting}] with value [{initial}] as a time value: negative durations are not supported"),
        )),
        Some(v) => Ok(v),
        None => {
            let nfe = exc("number_format_exception", format!("For input string: \"{num}\""));
            let reason = if java_double(num) {
                format!("failed to parse [{initial}], fractional time values are not supported")
            } else {
                format!("failed to parse [{initial}]")
            };
            let mut e = exc("illegal_argument_exception", reason);
            e["caused_by"] = nfe;
            Err(e)
        }
    }
}

// ------------------------------------------------------------ request body

/// A put request after parsing (ES `PutShutdownNodeAction.Request`).
#[derive(Debug, Clone, PartialEq)]
pub struct PutRequest {
    /// `REMOVE`, `RESTART`, `REPLACE`, `SIGTERM`.
    pub kind: &'static str,
    pub reason: String,
    pub allocation_delay: Option<TimeValue>,
    pub target_node_name: Option<String>,
    pub grace_period: Option<TimeValue>,
}

/// JSON token kinds, named as ES `XContentParser.Token`.
pub(super) fn token_name(b: u8) -> &'static str {
    match b {
        b'{' => "START_OBJECT",
        b'[' => "START_ARRAY",
        b'"' => "VALUE_STRING",
        b't' | b'f' => "VALUE_BOOLEAN",
        b'n' => "VALUE_NULL",
        _ => "VALUE_NUMBER",
    }
}

/// Minimal scanner over a (serde-validated) JSON document that yields the
/// top-level object's fields with Jackson token locations (1-based line and
/// byte column).
struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
    line: usize,
    line_start: usize,
}

pub(super) struct Field {
    pub(super) name: String,
    pub(super) name_loc: (usize, usize),
    pub(super) value_loc: (usize, usize),
    pub(super) value_tok: u8,
    pub(super) value: Value,
}

impl<'a> Scanner<'a> {
    fn new(b: &'a [u8]) -> Self {
        Scanner { b, i: 0, line: 1, line_start: 0 }
    }

    fn loc(&self) -> (usize, usize) {
        (self.line, self.i - self.line_start + 1)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\n' => {
                    self.i += 1;
                    self.line += 1;
                    self.line_start = self.i;
                }
                b'\r' => {
                    self.i += 1;
                    if self.b.get(self.i) != Some(&b'\n') {
                        self.line += 1;
                        self.line_start = self.i;
                    }
                }
                b' ' | b'\t' => self.i += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> u8 {
        self.b.get(self.i).copied().unwrap_or(0)
    }

    /// Skips a string starting at the opening quote.
    fn skip_string(&mut self) {
        self.i += 1;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\\' => self.i += 2,
                b'"' => {
                    self.i += 1;
                    return;
                }
                _ => self.i += 1,
            }
        }
    }

    /// Skips any value, tracking newlines inside containers.
    fn skip_value(&mut self) {
        match self.peek() {
            b'"' => self.skip_string(),
            b'{' | b'[' => {
                let mut depth = 0usize;
                loop {
                    self.ws();
                    if self.i >= self.b.len() {
                        return;
                    }
                    match self.b[self.i] {
                        b'{' | b'[' => {
                            depth += 1;
                            self.i += 1;
                        }
                        b'}' | b']' => {
                            depth -= 1;
                            self.i += 1;
                            if depth == 0 {
                                return;
                            }
                        }
                        b'"' => self.skip_string(),
                        _ => self.i += 1,
                    }
                }
            }
            _ => {
                while self.i < self.b.len() && !matches!(self.b[self.i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    self.i += 1;
                }
            }
        }
    }
}

/// Top-level fields of `body` in order, or the first token's kind and
/// location when it is not an object.
pub(super) fn scan_fields(body: &[u8]) -> Result<Vec<Field>, (&'static str, (usize, usize))> {
    let mut sc = Scanner::new(body);
    sc.ws();
    if sc.peek() != b'{' {
        return Err((token_name(sc.peek()), sc.loc()));
    }
    sc.i += 1;
    let mut out = Vec::new();
    loop {
        sc.ws();
        match sc.peek() {
            b'}' | 0 => break,
            b',' => {
                sc.i += 1;
                continue;
            }
            _ => {}
        }
        let name_loc = sc.loc();
        let start = sc.i;
        sc.skip_string();
        let name: String = serde_json::from_slice(&body[start..sc.i]).unwrap_or_default();
        sc.ws();
        sc.i += 1; // ':'
        sc.ws();
        let value_loc = sc.loc();
        let value_tok = sc.peek();
        let vstart = sc.i;
        sc.skip_value();
        let value: Value = serde_json::from_slice(&body[vstart..sc.i]).unwrap_or(Value::Null);
        out.push(Field { name, name_loc, value_loc, value_tok, value });
    }
    Ok(out)
}

pub(super) fn loc_str((l, c): (usize, usize)) -> String {
    format!("[{l}:{c}]")
}

/// `SingleNodeShutdownMetadata.Type.parse`.
fn parse_type(s: &str) -> Result<&'static str, Value> {
    match s.to_lowercase().as_str() {
        "remove" => Ok("REMOVE"),
        "restart" => Ok("RESTART"),
        "replace" => Ok("REPLACE"),
        "sigterm" => Ok("SIGTERM"),
        _ => Err(exc("illegal_argument_exception", format!("unknown shutdown type: {s}"))),
    }
}

/// The parser's builder (`RestPutShutdownNodeAction` constructor function).
fn build(args: &[Option<String>; 5], node_id: &str) -> Result<PutRequest, Value> {
    let kind = parse_type(args[0].as_deref().unwrap_or(""))?;
    let setting = format!("put-shutdown-node-request-{node_id}");
    let allocation_delay = args[2].as_deref().map(|s| TimeValue::parse(s, &setting)).transpose()?;
    let grace_period = args[4].as_deref().map(|s| TimeValue::parse(s, &setting)).transpose()?;
    Ok(PutRequest { kind, reason: args[1].clone().unwrap_or_default(), allocation_delay, target_node_name: args[3].clone(), grace_period })
}

const FIELDS: [&str; 5] = ["type", "reason", "allocation_delay", "target_node_name", "grace_period"];

fn build_failed() -> String {
    format!("Failed to build [{PARSER}] after last required field arrived")
}

/// ES `ConstructingObjectParser` semantics over the request body.
pub fn parse_body(body: &[u8], node_id: &str) -> Result<PutRequest, ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Err(ApiError::simple(400, "parse_exception", "request body is required"));
    }
    if serde_json::from_slice::<Value>(body).is_err() {
        return Err(ApiError::simple(400, "x_content_parse_exception", &format!("[1:1] [{PARSER}] failed to parse JSON body")));
    }
    let fields = match scan_fields(body) {
        Ok(f) => f,
        Err((tok, loc)) => {
            return Err(ApiError::simple(400, "x_content_parse_exception", &format!("{} [{PARSER}] Expected START_OBJECT but was: {tok}", loc_str(loc))));
        }
    };
    let mut args: [Option<String>; 5] = Default::default();
    let mut collected = 0usize;
    let mut built: Option<PutRequest> = None;
    for f in &fields {
        let Some(pos) = FIELDS.iter().position(|n| *n == f.name) else {
            return Err(ApiError::simple(400, "x_content_parse_exception", &format!("{} [{PARSER}] unknown field [{}]", loc_str(f.name_loc), f.name)));
        };
        let Value::String(s) = &f.value else {
            return Err(ApiError::simple(
                400,
                "x_content_parse_exception",
                &format!("{} [{PARSER}] {} doesn't support values of type: {}", loc_str(f.value_loc), f.name, token_name(f.value_tok)),
            ));
        };
        if built.is_some() {
            // Constructor args arriving after the build would be a
            // duplicate; ES rejects duplicate keys at the JSON layer.
            return Err(ApiError::simple(400, "x_content_parse_exception", &format!("{} Duplicate field '{}'", loc_str(f.name_loc), f.name)));
        }
        if args[pos].is_none() {
            collected += 1;
        }
        args[pos] = Some(s.clone());
        if collected == FIELDS.len() {
            match build(&args, node_id) {
                Ok(r) => built = Some(r),
                Err(cause) => {
                    let mut mid = exc("x_content_parse_exception", build_failed());
                    mid["caused_by"] = cause;
                    let mut outer = exc("x_content_parse_exception", format!("{} [{PARSER}] failed to parse field [{}]", loc_str(f.value_loc), f.name));
                    outer["caused_by"] = mid;
                    return Err(ApiError::chained(400, outer));
                }
            }
        }
    }
    if let Some(r) = built {
        return Ok(r);
    }
    let missing: Vec<&str> = [0usize, 1].iter().filter(|&&i| args[i].is_none()).map(|&i| FIELDS[i]).collect();
    if !missing.is_empty() {
        return Err(ApiError::simple(400, "illegal_argument_exception", &format!("Required [{}]", missing.join(", "))));
    }
    build(&args, node_id).map_err(|cause| {
        let mut outer = exc("x_content_parse_exception", build_failed());
        outer["caused_by"] = cause;
        ApiError::chained(400, outer)
    })
}

/// `PutShutdownNodeAction.Request.validate`.
pub fn validate(r: &PutRequest) -> Result<(), ApiError> {
    let mut errs: Vec<String> = Vec::new();
    if r.allocation_delay.is_some() && r.kind != "RESTART" {
        errs.push("allocation_delay is only allowed for RESTART-type shutdown requests".into());
    }
    match (&r.target_node_name, r.kind) {
        (Some(t), k) if k != "REPLACE" => {
            errs.push(format!("target node name is only valid for REPLACE type shutdowns, but was given type [{k}] and target node name [{t}]"))
        }
        (None, "REPLACE") => errs.push("target node name is required for REPLACE type shutdowns".into()),
        _ => {}
    }
    if r.kind == "SIGTERM" && r.grace_period.is_none() {
        errs.push("grace period is required for SIGTERM shutdowns".into());
    } else if r.kind != "SIGTERM" && r.grace_period.is_some() {
        let t = r.target_node_name.as_deref().unwrap_or("null");
        errs.push(format!("grace period is only valid for SIGTERM type shutdowns, but was given type [{}] and target node name [{t}]", r.kind));
    }
    if errs.is_empty() {
        return Ok(());
    }
    let mut msg = String::from("Validation Failed: ");
    for (i, e) in errs.iter().enumerate() {
        msg.push_str(&format!("{}: {e};", i + 1));
    }
    Err(ApiError::simple(400, "action_request_validation_exception", &msg))
}

// ------------------------------------------------------------ metadata

fn records(state: &ClusterState) -> Map<String, Value> {
    state.metadata.customs.get(CUSTOM).and_then(|v| v["nodes"].as_object().cloned()).unwrap_or_default()
}

/// Effective allocation delay of a stored record
/// (`SingleNodeShutdownMetadata.getAllocationDelay`).
fn effective_delay(rec: &Value) -> Option<String> {
    match rec["allocation_delay"].as_str() {
        Some(d) => Some(d.to_string()),
        None if rec["type"].as_str() == Some("RESTART") => Some(DEFAULT_RESTART_DELAY.into()),
        None => None,
    }
}

/// `TransportPutShutdownNodeAction.isNoop`.
fn is_noop(existing: &Value, r: &PutRequest) -> bool {
    if existing["type"].as_str() != Some(r.kind) || existing["reason"].as_str() != Some(r.reason.as_str()) {
        return false;
    }
    let ex_delay = effective_delay(existing).and_then(|d| TimeValue::parse(&d, "x").ok()).map(|t| t.nanos());
    if ex_delay != r.allocation_delay.as_ref().map(TimeValue::nanos) {
        return false;
    }
    existing["target_node_name"].as_str() == r.target_node_name.as_deref()
}

fn record(state: &ClusterState, node_id: &str, r: &PutRequest, now: i64) -> Value {
    let dn = state.nodes.get(node_id);
    let mut rec = json!({
        "node_id": node_id,
        "node_ephemeral_id": dn.map(|n| json!(n.ephemeral_id)).unwrap_or(Value::Null),
        "type": r.kind,
        "reason": r.reason,
        "shutdown_started_millis": now,
        "node_seen": dn.is_some(),
    });
    if let Some(d) = &r.allocation_delay {
        rec["allocation_delay"] = json!(d.string_rep());
    }
    if let Some(t) = &r.target_node_name {
        rec["target_node_name"] = json!(t);
    }
    if let Some(g) = &r.grace_period {
        rec["grace_period"] = json!(g.string_rep());
    }
    rec
}

/// Flags records whose node has joined as seen (ES
/// `NodeSeenService`); run on every membership change. Returns whether the
/// state changed.
pub fn mark_seen(state: &mut ClusterState) -> bool {
    let Some(custom) = state.metadata.customs.get(CUSTOM) else { return false };
    let mut custom = custom.clone();
    let Some(nodes) = custom.get_mut("nodes").and_then(Value::as_object_mut) else { return false };
    let mut changed = false;
    for (id, rec) in nodes.iter_mut() {
        if state.nodes.contains_key(id) && rec["node_seen"] != json!(true) {
            rec["node_seen"] = json!(true);
            changed = true;
        }
    }
    if changed {
        state.metadata.customs.insert(CUSTOM.into(), custom);
        state.metadata.version += 1;
    }
    changed
}

// ------------------------------------------------------------ handlers

pub(super) fn check_params(req: &RestReq, allowed: &[&str]) -> Option<RestResp> {
    let mut bad: Vec<String> = req.params().into_keys().filter(|k| !allowed.contains(&k.as_str()) && !COMMON_PARAMS.contains(&k.as_str())).collect();
    if bad.is_empty() {
        return None;
    }
    bad.sort();
    let msg = super::util::unrecognized(req.path(), &bad, allowed, "parameter");
    Some(RestResp::error(400, "illegal_argument_exception", &msg))
}

pub(super) fn method_not_allowed(req: &RestReq, allowed: &str) -> RestResp {
    RestResp::json(
        405,
        &json!({"error": format!("Incorrect HTTP method for uri [{}] and method [{}], allowed: [{allowed}]", req.uri, req.method), "status": 405}),
    )
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    // `/_nodes/shutdown` or `/_nodes/{id}/shutdown`.
    let id = if segs.len() >= 3 { Some(segs[1].clone()) } else { None };
    let m = req.method();
    match (m.as_str(), id) {
        ("GET", id) => get(node, req, id).await,
        ("PUT", Some(id)) => put(node, req, id).await,
        ("DELETE", Some(id)) => delete(node, req, id).await,
        (_, Some(_)) => method_not_allowed(req, "GET, PUT, DELETE"),
        (_, None) => method_not_allowed(req, "GET"),
    }
}

async fn put(node: &Arc<ClusterNode>, req: &RestReq, id: String) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout", "timeout"]) {
        return r;
    }
    let r = match parse_body(&req.body, &id) {
        Ok(r) => r,
        Err(e) => return e.resp(),
    };
    if let Err(e) = validate(&r) {
        return e.resp();
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let ctx = node.master.clone();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        let existing = records(s);
        if existing.get(&id).is_some_and(|e| is_noop(e, &r)) {
            return Ok(s.clone());
        }
        let mut next = s.clone();
        let rec = record(&next, &id, &r, now_millis());
        let custom = next.metadata.customs.entry(CUSTOM.into()).or_insert_with(|| json!({"nodes": {}}));
        if !custom["nodes"].is_object() {
            custom["nodes"] = json!({});
        }
        custom["nodes"].as_object_mut().map(|n| n.insert(id.clone(), rec));
        next.metadata.version += 1;
        Ok(ctx.reroute(&next))
    });
    match node.coord().submit(task).await {
        Ok(_) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => task_error(e),
    }
}

async fn delete(node: &Arc<ClusterNode>, req: &RestReq, id: String) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout", "timeout"]) {
        return r;
    }
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let ctx = node.master.clone();
    let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
        if !records(s).contains_key(&id) {
            return Err(ApiError::simple(404, "resource_not_found_exception", &format!("node [{id}] is not currently shutting down")).encode());
        }
        let mut next = s.clone();
        if let Some(n) = next.metadata.customs.get_mut(CUSTOM).and_then(|c| c["nodes"].as_object_mut()) {
            n.remove(&id);
        }
        next.metadata.version += 1;
        Ok(ctx.reroute(&next))
    });
    match node.coord().submit(task).await {
        Ok(_) => RestResp::json(200, &json!({"acknowledged": true})),
        Err(e) => task_error(e),
    }
}

/// `SingleNodeShutdownMetadata.Status.combine`.
fn combine(statuses: &[&str]) -> &'static str {
    let rank = |s: &str| match s {
        "NOT_STARTED" => 0,
        "IN_PROGRESS" => 1,
        "STALLED" => 2,
        _ => -1,
    };
    let worst = statuses.iter().filter(|s| **s != "COMPLETE").max_by_key(|s| rank(s));
    match worst.copied() {
        Some("NOT_STARTED") => "NOT_STARTED",
        Some("IN_PROGRESS") => "IN_PROGRESS",
        Some("STALLED") => "STALLED",
        _ => "COMPLETE",
    }
}

/// One `GET` entry (ES `SingleNodeShutdownStatus.toXContent`).
pub fn render_status(rec: &Value, migration: Value, human: bool) -> Value {
    let mut m = Map::new();
    m.insert("node_id".into(), rec["node_id"].clone());
    m.insert("node_ephemeral_id".into(), rec["node_ephemeral_id"].clone());
    m.insert("type".into(), rec["type"].clone());
    m.insert("reason".into(), rec["reason"].clone());
    if let Some(d) = effective_delay(rec) {
        m.insert("allocation_delay".into(), json!(d));
    }
    let started = rec["shutdown_started_millis"].as_i64().unwrap_or(0);
    if human {
        m.insert("shutdown_started".into(), json!(super::util::iso_millis(started)));
    }
    m.insert("shutdown_started_millis".into(), json!(started));
    let ms = migration["status"].as_str().unwrap_or("COMPLETE").to_string();
    m.insert("status".into(), json!(combine(&[&ms, "COMPLETE", "COMPLETE"])));
    m.insert("shard_migration".into(), migration);
    m.insert("persistent_tasks".into(), json!({"status": "COMPLETE"}));
    m.insert("plugins".into(), json!({"status": "COMPLETE"}));
    if let Some(t) = rec.get("target_node_name").filter(|v| !v.is_null()) {
        m.insert("target_node_name".into(), t.clone());
    }
    if let Some(g) = rec.get("grace_period").filter(|v| !v.is_null()) {
        m.insert("grace_period".into(), g.clone());
    }
    Value::Object(m)
}

async fn get(node: &Arc<ClusterNode>, req: &RestReq, ids: Option<String>) -> RestResp {
    if let Some(r) = check_params(req, &["master_timeout"]) {
        return r;
    }
    // The migration status needs the master's routing decisions and
    // cluster info (ES runs this as a master-node action).
    if !node.is_master() {
        return super::forward_to_master(node, req).await;
    }
    let st = node.state();
    let recs = records(&st);
    let wanted: Option<Vec<String>> = ids.map(|s| s.split(',').map(str::to_string).collect());
    let human = matches!(req.params().get("human").map(String::as_str), Some("") | Some("true"));
    let info = node.master.cluster_info.lock().clone();
    let now = now_millis();
    let mut out = Vec::new();
    for (nid, rec) in &recs {
        if wanted.as_ref().is_some_and(|w| !w.contains(nid)) {
            continue;
        }
        let ty = rec["type"].as_str().unwrap_or("REMOVE");
        let seen = rec["node_seen"].as_bool().unwrap_or(false);
        let mig = node.master.allocation.shutdown_migration_status(&st, nid, ty, seen, now, &info);
        out.push(render_status(rec, mig.to_json(), human));
    }
    RestResp::json(200, &json!({"nodes": out}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(body: &str) -> Value {
        parse_body(body.as_bytes(), "n1").unwrap_err().error
    }

    #[test]
    fn parses_minimal_restart() {
        let r = parse_body(br#"{"type":"restart","reason":"x"}"#, "n1").unwrap();
        assert_eq!(r.kind, "RESTART");
        assert!(validate(&r).is_ok());
    }

    #[test]
    fn empty_body() {
        assert_eq!(err("")["type"], "parse_exception");
        assert_eq!(err("")["reason"], "request body is required");
    }

    #[test]
    fn not_an_object() {
        assert_eq!(err("[]")["reason"], "[1:1] [put_node_shutdown_request] Expected START_OBJECT but was: START_ARRAY");
    }

    #[test]
    fn unknown_field_location() {
        let e = err("{\n  \"type\": \"restart\",\n  \"foo\": \"bar\"\n}");
        assert_eq!(e["type"], "x_content_parse_exception");
        assert_eq!(e["reason"], "[3:3] [put_node_shutdown_request] unknown field [foo]");
    }

    #[test]
    fn wrong_value_type() {
        let e = err(r#"{"type": 5, "reason": "x"}"#);
        assert_eq!(e["reason"], "[1:10] [put_node_shutdown_request] type doesn't support values of type: VALUE_NUMBER");
        let e = err(r#"{"type":"restart","reason":null}"#);
        assert_eq!(e["reason"], "[1:28] [put_node_shutdown_request] reason doesn't support values of type: VALUE_NULL");
    }

    #[test]
    fn required_fields() {
        assert_eq!(err(r#"{"type":"restart"}"#)["reason"], "Required [reason]");
        assert_eq!(err("{}")["reason"], "Required [type, reason]");
    }

    #[test]
    fn bad_type_at_finish() {
        let e = err(r#"{"type":"bogus","reason":"x"}"#);
        assert_eq!(e["type"], "x_content_parse_exception");
        assert_eq!(e["reason"], "Failed to build [put_node_shutdown_request] after last required field arrived");
        assert_eq!(e["caused_by"]["reason"], "unknown shutdown type: bogus");
        assert_eq!(e["root_cause"][0]["reason"], e["reason"]);
    }

    #[test]
    fn build_mid_object() {
        let body = r#"{"type":"restart","reason":"x","allocation_delay":"10x","target_node_name":"a","grace_period":"1s"}"#;
        let e = err(body);
        let col = body.find("\"1s\"").unwrap() + 1;
        assert_eq!(e["reason"], format!("[1:{col}] [put_node_shutdown_request] failed to parse field [grace_period]"));
        assert_eq!(e["caused_by"]["reason"], "Failed to build [put_node_shutdown_request] after last required field arrived");
        assert_eq!(
            e["caused_by"]["caused_by"]["reason"],
            "failed to parse setting [put-shutdown-node-request-n1] with value [10x] as a time value: unit is missing or unrecognized"
        );
    }

    #[test]
    fn time_values() {
        let p = |s| TimeValue::parse(s, "s");
        assert_eq!(p("10m").unwrap().string_rep(), "10m");
        assert_eq!(p("10M").unwrap_err()["reason"], "failed to parse setting [s] with value [10M] as a time value: unit is missing or unrecognized");
        assert_eq!(p("0").unwrap().string_rep(), "0ms");
        assert_eq!(p("-1").unwrap().string_rep(), "-1");
        assert_eq!(p("1.5s").unwrap_err()["reason"], "failed to parse [1.5s], fractional time values are not supported");
        assert_eq!(p("1.5s").unwrap_err()["caused_by"]["reason"], "For input string: \"1.5\"");
        assert_eq!(p("xs").unwrap_err()["reason"], "failed to parse [xs]");
        assert_eq!(p("-5s").unwrap_err()["reason"], "failed to parse setting [s] with value [-5s] as a time value: negative durations are not supported");
        assert_eq!(p("300s").unwrap().nanos(), p("5m").unwrap().nanos());
    }

    #[test]
    fn validation_messages() {
        let r = parse_body(br#"{"type":"remove","reason":"x","allocation_delay":"1m","target_node_name":"t","grace_period":"1s"}"#, "n").unwrap();
        let e = validate(&r).unwrap_err().error;
        assert_eq!(
            e["reason"],
            "Validation Failed: 1: allocation_delay is only allowed for RESTART-type shutdown requests;2: target node name is only valid for REPLACE type shutdowns, but was given type [REMOVE] and target node name [t];3: grace period is only valid for SIGTERM type shutdowns, but was given type [REMOVE] and target node name [t];"
        );
        let r = parse_body(br#"{"type":"replace","reason":"x"}"#, "n").unwrap();
        assert_eq!(validate(&r).unwrap_err().error["reason"], "Validation Failed: 1: target node name is required for REPLACE type shutdowns;");
        let r = parse_body(br#"{"type":"sigterm","reason":"x"}"#, "n").unwrap();
        assert_eq!(validate(&r).unwrap_err().error["reason"], "Validation Failed: 1: grace period is required for SIGTERM shutdowns;");
    }

    #[test]
    fn noop_detection() {
        let r = parse_body(br#"{"type":"restart","reason":"x","allocation_delay":"300s"}"#, "n").unwrap();
        let rec = json!({"type": "RESTART", "reason": "x"});
        assert!(is_noop(&rec, &r));
        let r2 = parse_body(br#"{"type":"restart","reason":"x"}"#, "n").unwrap();
        assert!(!is_noop(&rec, &r2));
        let rec2 = json!({"type": "REMOVE", "reason": "x"});
        let r3 = parse_body(br#"{"type":"remove","reason":"x"}"#, "n").unwrap();
        assert!(is_noop(&rec2, &r3));
    }

    #[test]
    fn render_order_and_combine() {
        let rec = json!({"node_id": "a", "node_ephemeral_id": "e", "type": "RESTART", "reason": "r", "shutdown_started_millis": 5, "node_seen": true});
        let v = render_status(&rec, json!({"status": "COMPLETE"}), false);
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["node_id", "node_ephemeral_id", "type", "reason", "allocation_delay", "shutdown_started_millis", "status", "shard_migration", "persistent_tasks", "plugins"]);
        assert_eq!(v["allocation_delay"], "5m");
        assert_eq!(combine(&["IN_PROGRESS", "COMPLETE", "STALLED"]), "STALLED");
        assert_eq!(combine(&["COMPLETE"]), "COMPLETE");
    }

    #[test]
    fn task_error_roundtrip() {
        let e = ApiError::simple(404, "resource_not_found_exception", "node [x] is not currently shutting down");
        assert_eq!(ApiError::decode(&e.encode()), Some(e));
    }
}
