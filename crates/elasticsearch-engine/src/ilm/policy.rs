//! ILM policy parsing, validation (`TimeseriesLifecycleType`) and step
//! compilation (`LifecyclePolicy.toSteps`).

use super::{parse_size, parse_time, TimeVal};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

pub const PHASES: [&str; 5] = ["hot", "warm", "cold", "frozen", "delete"];

const HOT_ACTIONS: &[&str] = &["set_priority", "unfollow", "rollover", "readonly", "downsample", "shrink", "forcemerge", "searchable_snapshot"];
const WARM_ACTIONS: &[&str] = &["set_priority", "unfollow", "readonly", "downsample", "allocate", "migrate", "shrink", "forcemerge"];
const COLD_ACTIONS: &[&str] = &["set_priority", "unfollow", "readonly", "downsample", "searchable_snapshot", "allocate", "migrate", "freeze"];
const FROZEN_ACTIONS: &[&str] = &["unfollow", "searchable_snapshot"];
const DELETE_ACTIONS: &[&str] = &["wait_for_snapshot", "delete"];
const ALL_ACTIONS: &[&str] = &[
    "set_priority", "unfollow", "rollover", "readonly", "downsample", "shrink", "forcemerge", "searchable_snapshot", "allocate", "migrate", "freeze",
    "wait_for_snapshot", "delete",
];

pub fn ordered_actions(phase: &str) -> &'static [&'static str] {
    match phase {
        "hot" => HOT_ACTIONS,
        "warm" => WARM_ACTIONS,
        "cold" => COLD_ACTIONS,
        "frozen" => FROZEN_ACTIONS,
        "delete" => DELETE_ACTIONS,
        _ => &[],
    }
}

pub(super) fn xparse(reason: impl Into<String>) -> EsError {
    EsError::x_content_parse(reason)
}

// ---------------------------------------------------------------------------
// Source locations for parse errors (`[line:col]`), like XContentParseException
// ---------------------------------------------------------------------------

pub(super) type Pos = (u32, u32);

/// A JSON value with the positions ES's parser reports for it.
enum LNode {
    Obj { start: Pos, end: Pos, fields: Vec<(String, Pos, LNode)> },
    Other { start: Pos },
}

impl LNode {
    fn start(&self) -> Pos {
        match self {
            LNode::Obj { start, .. } | LNode::Other { start } => *start,
        }
    }
}

struct LParser<'a> {
    c: std::iter::Peekable<std::str::Chars<'a>>,
    pos: Pos,
}

impl LParser<'_> {
    fn bump(&mut self) -> Option<char> {
        let ch = self.c.next()?;
        if ch == '\n' {
            self.pos = (self.pos.0 + 1, 1);
        } else {
            self.pos.1 += 1;
        }
        Some(ch)
    }
    fn ws(&mut self) {
        while matches!(self.c.peek(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }
    fn string(&mut self) -> Option<String> {
        self.bump()?; // opening quote
        let mut out = String::new();
        loop {
            match self.bump()? {
                '"' => return Some(out),
                '\\' => {
                    let e = self.bump()?;
                    match e {
                        'u' => {
                            let mut h = String::new();
                            for _ in 0..4 {
                                h.push(self.bump()?);
                            }
                            out.push(char::from_u32(u32::from_str_radix(&h, 16).ok()?).unwrap_or('\u{fffd}'));
                        }
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        other => out.push(other),
                    }
                }
                ch => out.push(ch),
            }
        }
    }
    fn value(&mut self) -> Option<LNode> {
        self.ws();
        let start = self.pos;
        match *self.c.peek()? {
            '{' => {
                self.bump();
                let mut fields = Vec::new();
                loop {
                    self.ws();
                    let end = self.pos;
                    match *self.c.peek()? {
                        '}' => {
                            self.bump();
                            return Some(LNode::Obj { start, end, fields });
                        }
                        ',' => {
                            self.bump();
                        }
                        '"' => {
                            let kp = self.pos;
                            let k = self.string()?;
                            self.ws();
                            if self.bump()? != ':' {
                                return None;
                            }
                            let v = self.value()?;
                            fields.push((k, kp, v));
                        }
                        _ => return None,
                    }
                }
            }
            '[' => {
                self.bump();
                loop {
                    self.ws();
                    match *self.c.peek()? {
                        ']' => {
                            self.bump();
                            return Some(LNode::Other { start });
                        }
                        ',' => {
                            self.bump();
                        }
                        _ => {
                            self.value()?;
                        }
                    }
                }
            }
            '"' => {
                self.string()?;
                Some(LNode::Other { start })
            }
            _ => {
                while matches!(self.c.peek(), Some(c) if !matches!(c, ',' | '}' | ']') && !c.is_whitespace()) {
                    self.bump();
                }
                Some(LNode::Other { start })
            }
        }
    }
}

thread_local! {
    static RAW: std::cell::RefCell<Option<LNode>> = const { std::cell::RefCell::new(None) };
    static PATH: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    static CUR: std::cell::Cell<Pos> = const { std::cell::Cell::new((1, 1)) };
}

/// Keeps a path segment pushed for the duration of a scope.
pub(super) struct PathGuard;
impl PathGuard {
    pub(super) fn push(seg: &str) -> PathGuard {
        PATH.with(|p| p.borrow_mut().push(seg.to_string()));
        PathGuard
    }
}
impl Drop for PathGuard {
    fn drop(&mut self) {
        PATH.with(|p| {
            p.borrow_mut().pop();
        });
    }
}

#[derive(Clone, Copy)]
pub(super) enum At {
    /// Start of the value at `path/extra`.
    Val,
    /// Start of the key `extra` under `path`.
    Key,
    /// Closing brace of the object at `path`.
    End,
}

/// Position of a node relative to the current path, if the raw body is known.
pub(super) fn locate(at: At, extra: Option<&str>) -> Option<Pos> {
    RAW.with(|r| {
        let r = r.borrow();
        let mut node = r.as_ref()?;
        let path = PATH.with(|p| p.borrow().clone());
        let mut key_pos = None;
        for seg in path.iter().map(String::as_str).chain(extra) {
            match node {
                LNode::Obj { fields, .. } => {
                    let (_, kp, n) = fields.iter().find(|(k, _, _)| k == seg)?;
                    key_pos = Some(*kp);
                    node = n;
                }
                LNode::Other { .. } => return None,
            }
        }
        match at {
            At::Val => Some(node.start()),
            At::Key => key_pos,
            At::End => match node {
                LNode::Obj { end, .. } => Some(*end),
                LNode::Other { start } => Some(*start),
            },
        }
    })
}

pub(super) fn set_cur(at: At, extra: Option<&str>) -> Pos {
    let p = locate(at, extra).unwrap_or((1, 1));
    CUR.with(|c| c.set(p));
    p
}

pub(super) fn fmt_pos(p: Pos) -> String {
    format!("{}:{}", p.0, p.1)
}

fn cur() -> String {
    fmt_pos(CUR.with(|c| c.get()))
}

/// Run `f` with source locations resolved against `raw` (the request body).
pub fn with_source<R>(raw: &str, f: impl FnOnce() -> R) -> R {
    let mut p = LParser { c: raw.chars().peekable(), pos: (1, 1) };
    let tree = p.value();
    RAW.with(|r| *r.borrow_mut() = tree);
    PATH.with(|p| p.borrow_mut().clear());
    CUR.with(|c| c.set((1, 1)));
    let out = f();
    RAW.with(|r| *r.borrow_mut() = None);
    out
}

/// `XContentParseException` for a field whose value failed to parse (the
/// parser sits on the value token).
fn field_err(parser: &str, field: &str, cause: EsError) -> EsError {
    set_cur(At::Val, Some(field));
    wrap(parser, field, cause)
}

/// An outer `[parser] failed to parse field [field]` layer at the current location.
pub(super) fn wrap(parser: &str, field: &str, cause: EsError) -> EsError {
    xparse(format!("[{}] [{parser}] failed to parse field [{field}]", cur())).caused_by(cause)
}

fn unknown(parser: &str, k: &str) -> EsError {
    let kp = locate(At::Key, Some(k)).unwrap_or((1, 2));
    set_cur(At::Val, Some(k));
    xparse(format!("[{}] [{parser}] unknown field [{k}]", fmt_pos(kp)))
}

fn tname(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Array(_) => "START_ARRAY",
        Value::Null => "VALUE_NULL",
        Value::Object(_) => "START_OBJECT",
    }
}

/// `field doesn't support values of type: T` (ObjectParser, at the value token).
fn unsupported(parser: &str, field: &str, v: &Value) -> EsError {
    let p = set_cur(At::Val, None);
    xparse(format!("[{}] [{parser}] {field} doesn't support values of type: {}", fmt_pos(p), tname(v)))
}

/// `TimeValue.parseTimeValue` failures as ILM's parsers surface them.
pub(super) fn tv(s: &str, setting: &str) -> Result<TimeVal> {
    parse_time(s, setting).map_err(|e| {
        if e.reason.starts_with("failed to parse setting") {
            EsError::illegal_argument(e.reason)
        } else if let Some(v) = e.reason.strip_prefix("failed to parse [").and_then(|r| r.strip_suffix(']')) {
            let lower = v.trim().to_ascii_lowercase();
            let unit = ["nanos", "micros", "ms", "s", "m", "h", "d"].into_iter().find(|u| lower.ends_with(u)).unwrap_or("");
            let digits = lower[..lower.len() - unit.len()].trim().to_string();
            EsError::illegal_argument(format!("failed to parse [{v}]")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{digits}\"")))
        } else {
            EsError::illegal_argument(e.reason)
        }
    })
}

fn as_obj<'a>(v: &'a Value, parser: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| {
        let p = set_cur(At::Val, None);
        xparse(format!("[{}] [{parser}] expected an object", fmt_pos(p)))
    })
}

fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn int_of(v: &Value, parser: &str, field: &str) -> Result<i64> {
    match v {
        Value::Number(n) if n.as_i64().is_some() => Ok(n.as_i64().unwrap()),
        Value::String(s) => s.trim().parse().map_err(|_| field_err(parser, field, EsError::illegal_argument(format!("For input string: \"{s}\"")))),
        _ => Err(field_err(parser, field, EsError::illegal_argument(format!("[{field}] must be a number")))),
    }
}

fn bool_of(v: &Value, parser: &str, field: &str) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" || s == "false" => Ok(s == "true"),
        _ => Err(field_err(parser, field, EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", str_of(v))))),
    }
}

/// Parse a `PUT /_ilm/policy/{name}` body into the normalized policy.
pub fn parse_put_body(body: &Value) -> Result<Value> {
    let o = match body {
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        v => as_obj(v, "put_lifecycle_request")?,
    };
    let _g0 = PathGuard::push("policy");
    if let Some(p) = o.get("policy").filter(|p| !p.is_object()) {
        return Err(unsupported("put_lifecycle_request", "policy", p));
    }
    drop(_g0);
    for k in o.keys() {
        if k != "policy" {
            return Err(unknown("put_lifecycle_request", k));
        }
    }
    let p = o.get("policy").ok_or_else(|| EsError::illegal_argument("Required [policy]"))?;
    let _g = PathGuard::push("policy");
    parse_policy(p).map_err(|e| wrap("put_lifecycle_request", "policy", e))
}

pub fn parse_policy(p: &Value) -> Result<Value> {
    let o = as_obj(p, "lifecycle_policy")?;
    let mut phases_out = Map::new();
    let mut meta = None;
    let mut deprecated = None;
    let mut saw_phases = false;
    for (k, v) in o {
        match k.as_str() {
            "phases" => {
                saw_phases = true;
                let _g = PathGuard::push("phases");
                let ph = match v {
                    Value::Object(m) => m,
                    other => return Err(unsupported("lifecycle_policy", "phases", other)),
                };
                for (name, def) in ph {
                    let _g = PathGuard::push(name);
                    let parsed = parse_phase(name, def).map_err(|e| wrap("lifecycle_policy", "phases", wrap("phases", name, e)))?;
                    phases_out.insert(name.clone(), parsed);
                }
            }
            "_meta" => {
                if !v.is_object() {
                    return Err(field_err("lifecycle_policy", "_meta", EsError::illegal_argument("[_meta] must be an object")));
                }
                meta = Some(v.clone());
            }
            "deprecated" => deprecated = Some(bool_of(v, "lifecycle_policy", "deprecated")?),
            other => return Err(unknown("lifecycle_policy", other)),
        }
    }
    if !saw_phases {
        set_cur(At::End, None);
        return Err(EsError::illegal_argument("Required [phases]"));
    }
    let mut out = Map::new();
    out.insert("phases".into(), Value::Object(phases_out));
    if let Some(m) = meta {
        out.insert("_meta".into(), m);
    }
    if let Some(d) = deprecated {
        out.insert("deprecated".into(), json!(d));
    }
    Ok(Value::Object(out))
}

fn parse_phase(name: &str, def: &Value) -> Result<Value> {
    let o = match def {
        Value::Object(m) => m,
        other => {
            let p = set_cur(At::Val, None);
            return Err(xparse(format!("[{}] [phase] Expected START_OBJECT but was: {}", fmt_pos(p), tname(other))));
        }
    };
    let mut min_age = TimeVal::zero();
    let mut actions = None;
    for (k, v) in o {
        match k.as_str() {
            "min_age" => {
                let s = str_of(v);
                min_age = if s.starts_with('-') { TimeVal::zero() } else { tv(&s, "min_age").map_err(|e| field_err("phase", "min_age", e))? };
            }
            "actions" => {
                let _g = PathGuard::push("actions");
                let a = match v {
                    Value::Object(m) => m,
                    other => return Err(unsupported("phase", "actions", other)),
                };
                let mut out = Map::new();
                for (an, av) in a {
                    if !ALL_ACTIONS.contains(&an.as_str()) {
                        let kp = set_cur(At::Key, Some(an));
                        let e = EsError::new(400, "named_object_not_found_exception", format!("[{}] unknown field [{an}]", fmt_pos(kp)));
                        return Err(wrap("phase", "actions", wrap("actions", an, e)));
                    }
                    let _g = PathGuard::push(an);
                    let parsed = parse_action(an, av).map_err(|e| wrap("phase", "actions", wrap("actions", an, e)))?;
                    out.insert(an.clone(), parsed);
                }
                actions = Some(out);
            }
            other => return Err(unknown("phase", other)),
        }
    }
    let _ = name;
    let actions = actions.ok_or_else(|| {
        set_cur(At::End, None);
        EsError::illegal_argument("Required [actions]")
    })?;
    Ok(json!({"min_age": min_age.rep, "actions": Value::Object(actions)}))
}

fn check_keys(o: &Map<String, Value>, parser: &str, allowed: &[&str]) -> Result<()> {
    for k in o.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(unknown(parser, k));
        }
    }
    Ok(())
}

fn string_map(v: &Value, parser: &str, field: &str) -> Result<Map<String, Value>> {
    match v {
        Value::Object(m) => Ok(m.iter().map(|(k, x)| (k.clone(), Value::String(str_of(x)))).collect()),
        Value::Null => Ok(Map::new()),
        _ => Err(field_err(parser, field, EsError::illegal_argument(format!("[{field}] must be an object")))),
    }
}

/// Parse + normalize one action body (each action's `toXContent`). A failure
/// raised while building the action (not while parsing a field) is layered
/// the way the action's ES parser reports it, at the object's closing brace.
pub fn parse_action(name: &str, v: &Value) -> Result<Value> {
    match parse_action_inner(name, v) {
        Err(e) if !matches!(e.kind.as_str(), "x_content_parse_exception" | "named_object_not_found_exception") => {
            let failed = |leaf| xparse(format!("Failed to build [{name}] after last required field arrived")).caused_by(leaf);
            // a ConstructingObjectParser builds when its last declared
            // constructor argument arrives, else when the object ends
            let ctor: &[&str] = match name {
                "set_priority" => &["priority"],
                "forcemerge" => &["max_num_segments", "index_codec"],
                "shrink" => &["number_of_shards", "max_primary_shard_size", "allow_write_after_shrink"],
                "allocate" => &["number_of_replicas", "total_shards_per_node", "include", "exclude", "require"],
                _ => &[],
            };
            if ctor.is_empty() || e.reason.starts_with("Required [") {
                set_cur(At::End, None);
                return Err(e);
            }
            let obj = v.as_object();
            let present = |f: &&str| obj.map(|o| o.contains_key(*f)).unwrap_or(false);
            if ctor.iter().all(|f| present(&f)) {
                let last = ctor.iter().max_by_key(|f| locate(At::Val, Some(f))).unwrap();
                set_cur(At::Val, Some(last));
                Err(wrap(name, last, failed(e)))
            } else {
                set_cur(At::End, None);
                Err(failed(e))
            }
        }
        other => other,
    }
}

fn parse_action_inner(name: &str, v: &Value) -> Result<Value> {
    let empty = Map::new();
    let o = match v {
        Value::Null => &empty,
        other => as_obj(other, name)?,
    };
    match name {
        "set_priority" => {
            check_keys(o, name, &["priority"])?;
            let p = match o.get("priority") {
                None | Some(Value::Null) => Value::Null,
                Some(x) => {
                    let n = int_of(x, name, "priority")?;
                    if n < 0 {
                        return Err(EsError::illegal_argument("[priority] must be 0 or greater"));
                    }
                    json!(n)
                }
            };
            Ok(json!({"priority": p}))
        }
        "unfollow" | "readonly" | "freeze" => {
            check_keys(o, name, &[])?;
            Ok(json!({}))
        }
        "rollover" => {
            const KEYS: [&str; 10] = [
                "max_size", "max_primary_shard_size", "max_age", "max_docs", "max_primary_shard_docs", "min_size", "min_primary_shard_size", "min_age", "min_docs",
                "min_primary_shard_docs",
            ];
            check_keys(o, name, &KEYS)?;
            let mut out = Map::new();
            for k in KEYS {
                let x = match o.get(k) {
                    Some(x) if !x.is_null() => x,
                    _ => continue,
                };
                let val = if k.ends_with("size") {
                    json!(parse_size(&str_of(x), k).map_err(|e| field_err("rollover_conditions", k, e))?.1)
                } else if k.ends_with("age") {
                    json!(tv(&str_of(x), k).map_err(|e| field_err("rollover_conditions", k, e))?.rep)
                } else {
                    json!(int_of(x, "rollover_conditions", k)?)
                };
                out.insert(k.to_string(), val);
            }
            if !out.keys().any(|k| k.starts_with("max_")) {
                return Err(EsError::illegal_argument("At least one max_* rollover condition must be set."));
            }
            Ok(Value::Object(out))
        }
        "downsample" => {
            check_keys(o, name, &["fixed_interval", "wait_timeout", "force_merge_index"])?;
            let fi = match o.get("fixed_interval") {
                Some(x) if !x.is_null() => str_of(x),
                _ => return Err(EsError::illegal_argument("Required [fixed_interval]")),
            };
            if crate::date::parse_fixed_interval(&fi).is_none() {
                return Err(field_err(name, "fixed_interval", EsError::illegal_argument(format!("failed to parse value [{fi}] as a fixed interval"))));
            }
            let wt = match o.get("wait_timeout") {
                Some(x) if !x.is_null() => tv(&str_of(x), "wait_timeout").map_err(|e| field_err(name, "wait_timeout", e))?.rep,
                _ => "1d".to_string(),
            };
            let mut out = json!({"fixed_interval": fi, "wait_timeout": wt});
            if let Some(f) = o.get("force_merge_index") {
                out["force_merge_index"] = json!(bool_of(f, name, "force_merge_index")?);
            }
            Ok(out)
        }
        "shrink" => {
            check_keys(o, name, &["number_of_shards", "max_primary_shard_size", "allow_write_after_shrink"])?;
            let n = match o.get("number_of_shards") {
                Some(x) if !x.is_null() => Some(int_of(x, name, "number_of_shards")?),
                _ => None,
            };
            let size = match o.get("max_primary_shard_size") {
                Some(x) if !x.is_null() => Some(parse_size(&str_of(x), "max_primary_shard_size").map_err(|e| field_err(name, "max_primary_shard_size", e))?),
                _ => None,
            };
            if n.is_some() && size.is_some() {
                return Err(EsError::illegal_argument("Cannot set both [number_of_shards] and [max_primary_shard_size]"));
            }
            if n.is_none() && size.is_none() {
                return Err(EsError::illegal_argument("Either [number_of_shards] or [max_primary_shard_size] must be set"));
            }
            if let Some((b, _)) = &size {
                if *b == 0 {
                    return Err(EsError::illegal_argument("[max_primary_shard_size] must be greater than 0"));
                }
            }
            if let Some(n) = n {
                if n <= 0 {
                    return Err(EsError::illegal_argument("[number_of_shards] must be greater than 0"));
                }
            }
            let allow = match o.get("allow_write_after_shrink") {
                Some(x) => bool_of(x, name, "allow_write_after_shrink")?,
                None => false,
            };
            let mut out = Map::new();
            if let Some(n) = n {
                out.insert("number_of_shards".into(), json!(n));
            }
            if let Some((_, rep)) = size {
                out.insert("max_primary_shard_size".into(), json!(rep));
            }
            out.insert("allow_write_after_shrink".into(), json!(allow));
            Ok(Value::Object(out))
        }
        "forcemerge" => {
            check_keys(o, name, &["max_num_segments", "index_codec"])?;
            let n = match o.get("max_num_segments") {
                Some(x) => int_of(x, name, "max_num_segments")?,
                None => return Err(EsError::illegal_argument("Required [max_num_segments]")),
            };
            if n <= 0 {
                return Err(EsError::illegal_argument("[max_num_segments] must be a positive integer"));
            }
            let mut out = json!({"max_num_segments": n});
            if let Some(c) = o.get("index_codec").filter(|c| !c.is_null()) {
                let c = str_of(c);
                if c != "best_compression" {
                    return Err(EsError::illegal_argument(format!("unknown index codec: [{c}]")));
                }
                out["index_codec"] = json!(c);
            }
            Ok(out)
        }
        "searchable_snapshot" => {
            check_keys(o, name, &["snapshot_repository", "force_merge_index", "total_shards_per_node", "replicate_for", "force_merge_on_clone"])?;
            let repo = match o.get("snapshot_repository") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                _ => return Err(EsError::illegal_argument("Required [snapshot_repository]")),
            };
            let fm = match o.get("force_merge_index") {
                Some(x) => bool_of(x, name, "force_merge_index")?,
                None => true,
            };
            let mut out = json!({"snapshot_repository": repo, "force_merge_index": fm});
            if let Some(t) = o.get("total_shards_per_node") {
                let t = int_of(t, name, "total_shards_per_node")?;
                if t < 1 {
                    return Err(EsError::illegal_argument("[total_shards_per_node] must be >= 1"));
                }
                out["total_shards_per_node"] = json!(t);
            }
            if let Some(r) = o.get("replicate_for") {
                out["replicate_for"] = json!(tv(&str_of(r), "replicate_for").map_err(|e| field_err(name, "replicate_for", e))?.rep);
            }
            if let Some(c) = o.get("force_merge_on_clone").filter(|v| !v.is_null()) {
                let c = bool_of(c, name, "force_merge_on_clone")?;
                if !fm {
                    return Err(EsError::illegal_argument("[force_merge_on_clone] is not allowed when [force_merge_index] is [false]"));
                }
                out["force_merge_on_clone"] = json!(c);
            }
            Ok(out)
        }
        "allocate" => {
            check_keys(o, name, &["number_of_replicas", "total_shards_per_node", "include", "exclude", "require"])?;
            let replicas = match o.get("number_of_replicas") {
                Some(x) if !x.is_null() => Some(int_of(x, name, "number_of_replicas")?),
                _ => None,
            };
            let tspn = match o.get("total_shards_per_node") {
                Some(x) if !x.is_null() => Some(int_of(x, name, "total_shards_per_node")?),
                _ => None,
            };
            let include = string_map(o.get("include").unwrap_or(&Value::Null), name, "include")?;
            let exclude = string_map(o.get("exclude").unwrap_or(&Value::Null), name, "exclude")?;
            let require = string_map(o.get("require").unwrap_or(&Value::Null), name, "require")?;
            if include.is_empty() && exclude.is_empty() && require.is_empty() && replicas.is_none() && tspn.is_none() {
                return Err(EsError::illegal_argument(
                    "At least one of include, exclude or require must contain attributes for action allocate. Otherwise the number_of_replicas or the total_shards_per_node options must be configured.",
                ));
            }
            if let Some(r) = replicas {
                if r < 0 {
                    return Err(EsError::illegal_argument("[number_of_replicas] must be >= 0"));
                }
            }
            if let Some(t) = tspn {
                if t < -1 {
                    return Err(EsError::illegal_argument("[total_shards_per_node] must be >= -1"));
                }
            }
            let mut out = Map::new();
            if let Some(r) = replicas {
                out.insert("number_of_replicas".into(), json!(r));
            }
            if let Some(t) = tspn {
                out.insert("total_shards_per_node".into(), json!(t));
            }
            out.insert("include".into(), Value::Object(include));
            out.insert("exclude".into(), Value::Object(exclude));
            out.insert("require".into(), Value::Object(require));
            Ok(Value::Object(out))
        }
        "migrate" => {
            check_keys(o, name, &["enabled"])?;
            let en = match o.get("enabled") {
                Some(x) => bool_of(x, name, "enabled")?,
                None => true,
            };
            Ok(json!({"enabled": en}))
        }
        "wait_for_snapshot" => {
            check_keys(o, name, &["policy"])?;
            match o.get("policy") {
                Some(Value::String(s)) if !s.is_empty() => Ok(json!({"policy": s})),
                None | Some(Value::Null) => Err(EsError::illegal_argument("Required [policy]")),
                _ => Err(EsError::illegal_argument("policy name must be specified")),
            }
        }
        "delete" => {
            check_keys(o, name, &["delete_searchable_snapshot"])?;
            let d = match o.get("delete_searchable_snapshot") {
                Some(x) => bool_of(x, name, "delete_searchable_snapshot")?,
                None => true,
            };
            Ok(json!({"delete_searchable_snapshot": d}))
        }
        other => Err(EsError::illegal_argument(format!("unknown action [{other}]"))),
    }
}

/// `LifecyclePolicy.validatePolicyName`.
pub fn validate_name(name: &str) -> Result<()> {
    let bad = |why: &str| Err(EsError::illegal_argument(format!("invalid policy name [{name}]: {why}")));
    if name.is_empty() {
        return bad("must not be null or empty");
    }
    if name.contains(',') {
        return bad("must not contain ','");
    }
    if name.contains(' ') {
        return bad("must not contain spaces");
    }
    if name.starts_with('_') {
        return bad("must not start with '_'");
    }
    if name.len() > 255 {
        return bad(&format!("name is too long, ({} > 255)", name.len()));
    }
    Ok(())
}

fn phases(policy: &Value) -> Vec<(String, Value)> {
    policy["phases"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default()
}

pub fn phase_min_age_millis(phase: &Value) -> i64 {
    phase["min_age"].as_str().and_then(|s| parse_time(s, "min_age").ok()).map(|t| t.millis()).unwrap_or(0)
}

fn interval_millis(s: &str) -> i64 {
    crate::date::parse_fixed_interval(s).unwrap_or(0)
}

/// `TimeseriesLifecycleType.validate` + `validateMonotonicallyIncreasingPhaseTimings`
/// (`PutLifecycleRequest.validate`): every message, in ES's order.
pub fn validation_errors(policy: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    if let Err(e) = validate_type(policy) {
        errs.push(e);
    }
    let timing = validate_timings(policy);
    if !timing.is_empty() {
        errs.push(timing);
    }
    errs
}

fn validate_type(policy: &Value) -> std::result::Result<(), String> {
    let ph = phases(policy);
    for (name, def) in &ph {
        if !PHASES.contains(&name.as_str()) {
            return Err(format!("Timeseries lifecycle does not support phase [{name}]"));
        }
        if let Some(a) = def["actions"].as_object() {
            for an in a.keys() {
                if !ordered_actions(name).contains(&an.as_str()) {
                    return Err(format!("invalid action [{an}] defined in phase [{name}]"));
                }
            }
        }
    }
    let has = |p: &str, a: &str| ph.iter().any(|(n, d)| n == p && d["actions"].get(a).is_some());
    let get = |p: &str| ph.iter().find(|(n, _)| n == p).map(|(_, d)| d.clone());
    if let Some(hot) = get("hot") {
        if hot["actions"].get("rollover").is_none() {
            let bad: Vec<&str> = ["readonly", "shrink", "forcemerge", "downsample", "searchable_snapshot"]
                .iter()
                .copied()
                .filter(|a| hot["actions"].get(*a).is_some())
                .collect();
            if !bad.is_empty() {
                return Err(format!("the [{}] action(s) may not be used in the [hot] phase without an accompanying [rollover] action", bad.join(", ")));
            }
        }
    }
    let conflicting: Vec<String> = PHASES
        .iter()
        .filter_map(|p| get(p).map(|d| (p, d)))
        .filter(|(_, d)| {
            d["actions"]["migrate"]["enabled"].as_bool() == Some(true)
                && d["actions"].get("allocate").map(|a| ["include", "exclude", "require"].iter().any(|k| a[*k].as_object().map(|m| !m.is_empty()).unwrap_or(false))).unwrap_or(false)
        })
        .map(|(p, _)| p.to_string())
        .collect();
    if !conflicting.is_empty() {
        return Err(format!(
            "phases [{}] specify an enabled migrate action and an allocate action with allocation rules. specify only a single data migration in each phase",
            conflicting.join(",")
        ));
    }
    // searchable_snapshot followed by forcemerge/freeze/shrink/downsample
    let mut following: Vec<&str> = Vec::new();
    if has("hot", "searchable_snapshot") {
        following.extend(PHASES.iter().copied().filter(|p| *p != "hot"));
    } else if has("cold", "searchable_snapshot") {
        following.push("frozen");
    }
    let illegal: Vec<&str> = following
        .into_iter()
        .filter(|p| ["forcemerge", "freeze", "shrink", "downsample"].iter().any(|a| has(p, a)))
        .collect();
    if !illegal.is_empty() {
        return Err(format!(
            "phases [{}] define one or more of [forcemerge, freeze, shrink, downsample] actions which are not allowed after a managed index is mounted as a searchable snapshot",
            illegal.join(",")
        ));
    }
    let mut repos: Vec<String> = ph.iter().filter_map(|(_, d)| d["actions"]["searchable_snapshot"]["snapshot_repository"].as_str().map(str::to_string)).collect();
    repos.sort();
    repos.dedup();
    if repos.len() > 1 {
        return Err(format!(
            "policy specifies [searchable_snapshot] action multiple times with differing repositories [{}], the same repository must be used for all searchable snapshot actions",
            repos.join(", ")
        ));
    }
    if get("frozen").is_some() && !has("frozen", "searchable_snapshot") {
        return Err("policy specifies the [frozen] phase without a corresponding [searchable_snapshot] action, but a searchable snapshot action is required in the frozen phase".into());
    }
    let ds: Vec<(&str, String)> = PHASES.iter().copied().filter_map(|p| get(p).and_then(|d| d["actions"]["downsample"]["fixed_interval"].as_str().map(|s| (p, s.to_string())))).collect();
    for w in ds.windows(2) {
        let (p1, i1) = &w[0];
        let (p2, i2) = &w[1];
        let (a, b) = (interval_millis(i1), interval_millis(i2));
        if a >= b {
            return Err(format!("Downsampling interval [{i2}] for phase [{p2}] must be greater than the interval [{i1}] for phase [{p1}]"));
        } else if a > 0 && b % a != 0 {
            return Err(format!("Downsampling interval [{i2}] for phase [{p2}] must be a multiple of the interval [{i1}] for phase [{p1}]"));
        }
    }
    Ok(())
}

fn validate_timings(policy: &Value) -> String {
    let ph = phases(policy);
    let age = |p: &str| -> Option<(i64, String)> {
        ph.iter().find(|(n, _)| n == p).and_then(|(_, d)| {
            let rep = d["min_age"].as_str().unwrap_or("0ms").to_string();
            let ms = phase_min_age_millis(d);
            if ms == 0 {
                None
            } else {
                Some((ms, rep))
            }
        })
    };
    let mut errors = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    for (i, p) in PHASES.iter().enumerate() {
        let (ms, rep) = match age(p) {
            Some(x) => x,
            None => continue,
        };
        if invalid.iter().any(|x| x == p) {
            continue;
        }
        let bad: Vec<(&str, String)> = PHASES[i + 1..].iter().filter_map(|q| age(q).filter(|(m, _)| *m < ms).map(|(_, r)| (*q, r))).collect();
        if bad.is_empty() {
            continue;
        }
        for (q, _) in &bad {
            invalid.push(q.to_string());
        }
        let mut err = format!("Your policy is configured to run the {} phase (min_age: {})", bad[0].0, bad[0].1);
        if bad.len() > 1 {
            for (q, r) in &bad[1..] {
                err.push_str(&format!(", the {q} phase (min_age: {r})"));
            }
            if let Some(idx) = err.rfind(',') {
                err = format!("{} and{}", &err[..idx], &err[idx + 1..]);
            }
        }
        err.push_str(&format!(
            " before the {p} phase (min_age: {rep}). You should change the phase timing so that the phases will execute in the order of hot, warm, then cold."
        ));
        errors.push(err);
    }
    errors.join(",")
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StepKey {
    pub phase: String,
    pub action: String,
    pub name: String,
}

impl StepKey {
    pub fn new(p: &str, a: &str, n: &str) -> StepKey {
        StepKey { phase: p.into(), action: a.into(), name: n.into() }
    }
    /// `Step.StepKey.toString()`.
    pub fn render(&self) -> String {
        format!("{{\"phase\":\"{}\",\"action\":\"{}\",\"name\":\"{}\"}}", self.phase, self.action, self.name)
    }
}

/// What a step does when the runner reaches it.
#[derive(Clone, Debug, PartialEq)]
pub enum StepKind {
    Init,
    /// End of a phase: waits for the next phase's `min_age`.
    PhaseComplete,
    /// Async wait step (evaluated on the periodic run only).
    Wait,
    /// Cluster-state / async action step (runs as soon as it is reached).
    Act,
    /// Branch: skips to the step after the action when the predicate says so.
    Branch,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub key: StepKey,
    pub kind: StepKind,
    /// Action config (`phase_definition.actions[action]`).
    pub config: Value,
}

fn steps_for_action(phase: &str, action: &str, cfg: &Value) -> Vec<(String, StepKind)> {
    use StepKind::*;
    let s = |n: &str, k: StepKind| (n.to_string(), k);
    match action {
        "set_priority" => vec![s("set_priority", Act)],
        "unfollow" => vec![s("branch-check-unfollow-prerequisites", Branch)],
        "rollover" => vec![
            s("check-rollover-ready", Wait),
            s("attempt-rollover", Act),
            s("wait-for-active-shards", Act),
            s("update-rollover-lifecycle-date", Act),
            s("set-indexing-complete", Act),
        ],
        "readonly" => vec![s("check-not-write-index", Act), s("check-ts-end-time-passed", Act), s("readonly", Act)],
        "forcemerge" => {
            let mut v = vec![
                s("branch-forcemerge-check-prerequisites", Branch),
                s("check-not-write-index", Act),
                s("check-ts-end-time-passed", Act),
                s("readonly", Act),
            ];
            if cfg.get("index_codec").is_some() {
                v.extend([s("close-index", Act), s("update-settings", Act), s("open-index", Act), s("wait-for-index-color", Act)]);
            }
            v.extend([s("forcemerge", Act), s("segment-count", Act)]);
            v
        }
        "shrink" => {
            let mut v = vec![
                s("branch-check-prerequisites", Branch),
                s("check-not-write-index", Act),
                s("wait-for-shard-history-leases", Act),
                s("check-ts-end-time-passed", Act),
                s("readonly", Act),
                s("check-target-shards-count", Act),
                s("cleanup-shrink-index", Act),
                s("generate-index-name", Act),
                s("set-single-node-allocation", Act),
                s("check-shrink-allocation", Act),
                s("shrink", Act),
                s("shrunk-shards-allocated", Act),
                s("copy-execution-state", Act),
                s("branch-on-datastream-check", Act),
                s("aliases", Act),
                s("is-shrunken-index", Act),
            ];
            if cfg["allow_write_after_shrink"].as_bool() == Some(true) {
                v.push(s("update-settings", Act));
            }
            v
        }
        "allocate" => vec![s("allocate", Act), s("check-allocation", Act)],
        "migrate" => {
            if cfg["enabled"].as_bool() == Some(false) {
                vec![]
            } else {
                vec![s("branch-check-skip-action", Branch), s("migrate", Act), s("check-migration", Act)]
            }
        }
        "freeze" => vec![s("branch-freeze-check-prerequisites", Branch), s("check-not-write-index", Act), s("freeze", Act)],
        "downsample" => vec![
            s("branch-on-timeseries-check", Branch),
            s("check-not-write-index", Act),
            s("wait-for-shard-history-leases", Act),
            s("check-ts-end-time-passed", Act),
            s("readonly", Act),
            s("cleanup-target-index", Act),
            s("generate-downsample-index-name", Act),
            s("rollup-index", Act),
            s("wait-for-index-color", Act),
            s("copy-execution-state", Act),
            s("copy_settings", Act),
            s("branch-on-datastream-check", Act),
            s("replace-datastream-backing-index", Act),
            s("delete", Act),
        ],
        "searchable_snapshot" => vec![s("branch-check-prerequisites", Branch)],
        "wait_for_snapshot" => vec![s("wait-for-snapshot", Wait)],
        "delete" => vec![s("wait-for-shard-history-leases", Act), s("check-ts-end-time-passed", Act), s("delete", Act)],
        _ => {
            let _ = phase;
            vec![]
        }
    }
}

/// Actions of a phase in execution order, with ES's injected `migrate`
/// for warm/cold phases that neither migrate nor allocate with rules.
pub fn phase_actions(phase: &str, def: &Value) -> Vec<(String, Value)> {
    let acts = def["actions"].as_object().cloned().unwrap_or_default();
    let mut out: Vec<(String, Value)> = Vec::new();
    let inject_migrate = matches!(phase, "warm" | "cold")
        && !acts.contains_key("migrate")
        && !acts.contains_key("searchable_snapshot")
        && !acts.get("allocate").map(|a| ["include", "exclude", "require"].iter().any(|k| a[*k].as_object().map(|m| !m.is_empty()).unwrap_or(false))).unwrap_or(false);
    for a in ordered_actions(phase) {
        if let Some(cfg) = acts.get(*a) {
            out.push((a.to_string(), cfg.clone()));
        } else if *a == "migrate" && inject_migrate {
            out.push(("migrate".into(), json!({"enabled": true})));
        }
    }
    out
}

/// The steps of one phase (actions then the phase-complete step).
pub fn phase_steps(phase: &str, def: &Value) -> Vec<Step> {
    let mut steps = Vec::new();
    for (a, cfg) in phase_actions(phase, def) {
        for (n, k) in steps_for_action(phase, &a, &cfg) {
            steps.push(Step { key: StepKey::new(phase, &a, &n), kind: k, config: cfg.clone() });
        }
    }
    steps.push(Step { key: StepKey::new(phase, "complete", "complete"), kind: StepKind::PhaseComplete, config: Value::Null });
    steps
}

/// Ordered phase names present in a policy.
pub fn ordered_phases(policy: &Value) -> Vec<String> {
    PHASES.iter().filter(|p| policy["phases"].get(**p).is_some()).map(|p| p.to_string()).collect()
}

/// Every step key of a policy (`PolicyStepsRegistry`), including
/// `new/init`, `new/complete` and the terminal step.
pub fn all_step_keys(policy: &Value) -> Vec<StepKey> {
    let mut out = vec![StepKey::new("new", "init", "init"), StepKey::new("new", "complete", "complete")];
    for p in ordered_phases(policy) {
        for s in phase_steps(&p, &policy["phases"][&p]) {
            out.push(s.key);
        }
    }
    out.push(StepKey::new("completed", "completed", "completed"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_validate() {
        let p = parse_put_body(&json!({"policy": {"phases": {"warm": {"min_age": "10s", "actions": {"forcemerge": {"max_num_segments": 10000}}}, "delete": {"min_age": "30s", "actions": {"delete": {}}}}}})).unwrap();
        assert_eq!(p["phases"]["delete"]["actions"]["delete"]["delete_searchable_snapshot"], true);
        assert!(validation_errors(&p).is_empty());
        let bad = parse_policy(&json!({"phases": {"hot": {"actions": {"readonly": {}}}}})).unwrap();
        assert_eq!(validation_errors(&bad), vec!["the [readonly] action(s) may not be used in the [hot] phase without an accompanying [rollover] action".to_string()]);
        let order = parse_policy(&json!({"phases": {"warm": {"min_age": "10s", "actions": {}}, "delete": {"min_age": "5s", "actions": {"delete": {}}}}})).unwrap();
        assert_eq!(
            validation_errors(&order)[0],
            "Your policy is configured to run the delete phase (min_age: 5s) before the warm phase (min_age: 10s). You should change the phase timing so that the phases will execute in the order of hot, warm, then cold."
        );
        let e = parse_policy(&json!({"phases": {"hot": {"actions": {"foo": {}}}}})).unwrap_err();
        assert_eq!(e.kind, "x_content_parse_exception");
        let keys = all_step_keys(&p);
        assert!(keys.contains(&StepKey::new("warm", "migrate", "migrate")), "migrate injected in warm");
        assert!(keys.contains(&StepKey::new("delete", "delete", "delete")));
    }
}
