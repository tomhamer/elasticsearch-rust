//! Watch execution: input -> condition -> transform -> actions, with
//! throttling, acking, status update and history.

use super::{actions, httpc, iso, parse_iso, service, CurrentExec, WatcherService};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts, NODE_ID};
use crate::script::{self, ScriptContext, Val};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Execution context (`ctx`).
#[derive(Clone)]
pub struct Ctx {
    pub id: String,
    pub watch_id: String,
    pub execution_time: i64,
    pub triggered: String,
    pub scheduled: String,
    pub metadata: Value,
    pub payload: Value,
    pub vars: Value,
    /// Node data directory (stored scripts).
    pub data: std::path::PathBuf,
    /// Security: the user the watch runs as (`None`: unrestricted).
    pub user: Option<String>,
}

impl Ctx {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "watch_id": self.watch_id, "execution_time": java_zdt(self.execution_time),
            "trigger": {"triggered_time": java_zdt_str(&self.triggered), "scheduled_time": java_zdt_str(&self.scheduled)},
            "metadata": self.metadata, "payload": self.payload, "vars": self.vars, "node": NODE_ID
        })
    }
}

/// `ZonedDateTime.toString()` of a UTC instant (`2015-01-04T00:00Z`,
/// `2015-05-05T20:58:02.443Z`): seconds only when non-zero, fraction only
/// when non-zero. This is how dates render in watcher templates.
pub fn java_zdt(ms: i64) -> String {
    let dt = crate::date::from_millis(ms);
    let mut out = dt.format("%Y-%m-%dT%H:%M").to_string();
    let milli = ms.rem_euclid(1000);
    let sec = dt.format("%S").to_string();
    if sec != "00" || milli != 0 {
        out.push(':');
        out.push_str(&sec);
        if milli != 0 {
            out.push_str(&format!(".{milli:03}"));
        }
    }
    out.push('Z');
    out
}

fn java_zdt_str(s: &str) -> String {
    parse_iso(s).map(java_zdt).unwrap_or_else(|| s.to_string())
}

#[derive(Clone, Default)]
pub struct ExecOptions {
    pub ignore_condition: bool,
    pub alternative_input: Option<Value>,
    pub action_modes: HashMap<String, String>,
    pub record: bool,
    pub manual: bool,
    pub debug: bool,
    /// Manual trigger data (`triggered_time` / `scheduled_time`).
    pub trigger_data: Option<Value>,
    /// Security: the user the execution runs as (recorded as `user`).
    pub user: Option<String>,
}

/// Security check of `action` on index `name` for the watch's user.
pub fn user_allowed(node: &Node, user: Option<&str>, action: &str, name: &str) -> bool {
    let Some(u) = user else { return true };
    let Some(a) = node.security.lookup_user(u) else { return false };
    let perm = node.security.permission(&a);
    perm.is_superuser || perm.check_index(action, name)
}

// ---------------------------------------------------------------------------
// Templates and paths
// ---------------------------------------------------------------------------

/// Turn `{{x}}` into `{{{x}}}` (watcher templates are plain text, no escaping).
fn unescape_mustache(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 8);
    let mut i = 0;
    while i < b.len() {
        if src[i..].starts_with("{{{") {
            match src[i..].find("}}}") {
                Some(e) => {
                    out.push_str(&src[i..i + e + 3]);
                    i += e + 3;
                }
                None => {
                    out.push_str(&src[i..]);
                    break;
                }
            }
        } else if src[i..].starts_with("{{") {
            let Some(e) = src[i + 2..].find("}}") else {
                out.push_str(&src[i..]);
                break;
            };
            let inner = &src[i + 2..i + 2 + e];
            let t = inner.trim_start();
            if t.starts_with(['#', '/', '^', '!', '>', '&', '=']) || t.is_empty() {
                out.push_str(&src[i..i + e + 4]);
            } else {
                out.push_str("{{{");
                out.push_str(inner.trim());
                out.push_str("}}}");
            }
            i += e + 4;
        } else {
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Render a watcher text template against `{ctx: ...}`.
pub fn render(src: &str, ctx: &Value) -> String {
    if !src.contains("{{") {
        return src.to_string();
    }
    let params = json!({ "ctx": ctx });
    crate::template::render(&unescape_mustache(src), &params).unwrap_or_else(|_| src.to_string())
}

/// Render every string inside a JSON value.
pub fn render_value(v: &Value, ctx: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(render(s, ctx)),
        Value::Array(a) => Value::Array(a.iter().map(|x| render_value(x, ctx)).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), render_value(x, ctx))).collect()),
        other => other.clone(),
    }
}

/// `a.b.0.c` lookup (array indices are numeric segments).
pub fn path_get<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = match cur {
            Value::Object(o) => o.get(seg)?,
            Value::Array(a) => a.get(seg.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Run a search request definition; returns `(response, rendered request)`.
pub fn run_search(node: &Node, req: &Value, ctx: &Ctx) -> Result<(Value, Value)> {
    let cj = ctx.to_json();
    let indices: Vec<String> = match req.get("indices") {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(|s| render(s, &cj))).collect(),
        Some(Value::String(s)) => vec![render(s, &cj)],
        _ => Vec::new(),
    };
    let mut body = match (req.get("body"), req.get("template")) {
        (Some(b), _) => {
            let text = crate::template::render(&b.to_string(), &json!({"ctx": cj}))?;
            serde_json::from_str::<Value>(&text).map_err(|e| EsError::json_parse(&e))?
        }
        (None, Some(t)) => {
            let src = match t.get("source").or(t.get("inline")) {
                Some(x) => crate::template::source_text(x),
                None => match t.get("id").and_then(Value::as_str) {
                    Some(id) => script::stored::for_data_dir(&ctx.data)
                        .get(id)
                        .map(|st| st.source_json.as_ref().map(crate::template::source_text).unwrap_or(st.source))
                        .ok_or_else(|| EsError::resource_not_found(format!("unable to find script [{id}] in cluster state")))?,
                    None => String::new(),
                },
            };
            let mut params = t.get("params").cloned().unwrap_or(json!({}));
            if let Some(o) = params.as_object_mut() {
                o.insert("ctx".into(), cj.clone());
            }
            crate::template::render_to_json(&src, &params)?
        }
        _ => json!({"query": {"match_all": {}}}),
    };
    if body.is_null() {
        body = json!({});
    }
    let mut p: HashMap<String, String> = HashMap::new();
    p.insert("ignore_unavailable".into(), "true".into());
    p.insert("allow_no_indices".into(), "true".into());
    if let Some(io) = req.get("indices_options").and_then(Value::as_object) {
        for (k, v) in io {
            let val = match v {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
                other => other.to_string(),
            };
            p.insert(k.clone(), val);
        }
    }
    let rest_total = req.get("rest_total_hits_as_int").and_then(Value::as_bool).unwrap_or(true);
    let expr = if indices.is_empty() { "_all".to_string() } else { indices.join(",") };
    let mut targets = node.resolve(&expr, ResolveOpts::from_params(&p))?;
    if ctx.user.is_some() {
        // run as the watch's user: indices it may not read are skipped
        // (the default indices options ignore unavailable indices)
        targets.retain(|t| user_allowed(node, ctx.user.as_deref(), "indices:data/read/search", &t.svc.name));
    }
    let mut sr = crate::search::parse_request(&body, &HashMap::new())?;
    sr.rest_total_hits_as_int = rest_total;
    let resp: Value = match super::remote().filter(|_| !targets.is_empty()) {
        // On a cluster: over every shard of the (permitted) targets.
        Some(rest) => {
            let mut names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
            names.sort();
            names.dedup();
            let uri = format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true&rest_total_hits_as_int={rest_total}", super::path_expr(&names.join(",")));
            rest("POST", &uri, Some(body.clone()))?
        }
        None => {
            let out = crate::search::execute(&targets, &sr, &super::search_opts(node))?;
            serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))?
        }
    };
    let mut rendered = json!({"search_type": req.get("search_type").cloned().unwrap_or(json!("query_then_fetch")), "indices": indices, "rest_total_hits_as_int": rest_total, "body": body});
    if let Some(io) = req.get("indices_options") {
        rendered["indices_options"] = io.clone();
    }
    if let Some(t) = req.get("template").and_then(Value::as_object) {
        // Script.toXContent of the template (default lang mustache)
        let mut tm = Map::new();
        if let Some(x) = t.get("source").or(t.get("inline")) {
            tm.insert("source".into(), x.clone());
            tm.insert("lang".into(), t.get("lang").cloned().unwrap_or(json!("mustache")));
        } else if let Some(id) = t.get("id") {
            tm.insert("id".into(), id.clone());
        }
        if let Some(o) = t.get("options").filter(|o| o.as_object().map(|m| !m.is_empty()).unwrap_or(false)) {
            tm.insert("options".into(), o.clone());
        }
        if let Some(p) = t.get("params").filter(|o| o.as_object().map(|m| !m.is_empty()).unwrap_or(false)) {
            tm.insert("params".into(), p.clone());
        }
        rendered["template"] = Value::Object(tm);
    }
    Ok((resp, rendered))
}

fn apply_extract(payload: &mut Value, def: &Value) {
    if let Some(keys) = def.get("extract").and_then(Value::as_array) {
        let mut m = Map::new();
        for k in keys.iter().filter_map(Value::as_str) {
            if let Some(v) = path_get(payload, k) {
                m.insert(k.to_string(), v.clone());
            }
        }
        *payload = Value::Object(m);
    }
}

/// Run one input definition; returns the input result object (with `payload`
/// on success).
pub fn run_input(node: &Node, input: &Value, ctx: &Ctx) -> Value {
    let (ty, def) = match input.as_object().and_then(|o| o.iter().next()) {
        Some((k, v)) => (k.as_str(), v),
        None => return json!({"type": "none", "status": "success", "payload": {}}),
    };
    let fail = |reason: String| json!({"type": ty, "status": "failure", "error": {"root_cause": [{"type": "exception", "reason": reason}], "type": "exception", "reason": reason}});
    match ty {
        "none" => json!({"type": "none", "status": "success", "payload": {}}),
        "simple" => json!({"type": "simple", "status": "success", "payload": def}),
        "search" => {
            let req = &def["request"];
            match run_search(node, req, ctx) {
                Ok((mut resp, rendered)) => {
                    apply_extract(&mut resp, def);
                    json!({"type": "search", "status": "success", "payload": resp, "search": {"request": rendered}})
                }
                Err(e) => json!({"type": "search", "status": "failure", "error": e.error_object(false)}),
            }
        }
        "http" => {
            let cj = ctx.to_json();
            let req = match actions::render_request_templates(&def["request"], &cj, &ctx.data) {
                Ok(r) => r,
                Err(e) => return fail(e),
            };
            let r = match httpc::HttpRequest::from_json(&req, &|s| render(s, &cj)) {
                Ok(r) => r,
                Err(e) => return fail(e),
            };
            if let Err(e) = actions::check_whitelist(node, &r) {
                return fail(e);
            }
            match httpc::send(&r) {
                Err(e) => json!({"type": "http", "status": "failure", "error": actions::http_error(&r, &e), "http": {"request": r.to_json_sim()}}),
                Ok(resp) => {
                    let body = resp.body_str();
                    let ct = def.get("response_content_type").and_then(Value::as_str).map(str::to_string).or_else(|| {
                        resp.header("content-type").map(|c| if c.contains("yaml") { "yaml".to_string() } else if c.contains("json") { "json".to_string() } else { "text".to_string() })
                    });
                    let mut payload = match ct.as_deref() {
                        Some("json") => serde_json::from_str::<Value>(&body).unwrap_or_else(|_| json!({"_value": body})),
                        Some("yaml") => serde_yaml::from_str::<Value>(&body).unwrap_or_else(|_| json!({"_value": body})),
                        _ => json!({"_value": body}),
                    };
                    apply_extract(&mut payload, def);
                    let mut headers = Map::new();
                    for (k, v) in &resp.headers {
                        headers.entry(k.to_ascii_lowercase()).or_insert_with(|| json!([])).as_array_mut().unwrap().push(json!(v));
                    }
                    if let Some(o) = payload.as_object_mut() {
                        o.insert("_status_code".into(), json!(resp.status));
                        o.insert("_headers".into(), Value::Object(headers));
                    }
                    json!({"type": "http", "status": "success", "payload": payload, "http": {"request": r.to_json_sim(), "status_code": resp.status}})
                }
            }
        }
        "chain" => {
            let mut c = ctx.clone();
            let mut acc = Map::new();
            let mut results = Vec::new();
            for sub in def["inputs"].as_array().cloned().unwrap_or_default() {
                let Some((name, d)) = sub.as_object().and_then(|o| o.iter().next().map(|(k, v)| (k.clone(), v.clone()))) else { continue };
                let r = run_input(node, &d, &c);
                if r["status"] != "success" {
                    return json!({"type": "chain", "status": "failure", "reason": r["reason"].clone()});
                }
                acc.insert(name.clone(), r["payload"].clone());
                c.payload = Value::Object(acc.clone());
                results.push(json!({name: r}));
            }
            json!({"type": "chain", "status": "success", "payload": Value::Object(acc), "chain": {"steps": results}})
        }
        "transform" => {
            let r = run_transform(node, def, ctx);
            if r["status"] == "success" {
                json!({"type": "transform", "status": "success", "payload": r["payload"]})
            } else {
                json!({"type": "transform", "status": "failure", "reason": r["reason"].clone()})
            }
        }
        _ => fail(format!("unknown input type [{ty}]")),
    }
}

// ---------------------------------------------------------------------------
// Scripts
// ---------------------------------------------------------------------------

/// A script definition resolved to `(lang, source, params)` (stored scripts
/// are looked up on every execution, so updates take effect).
pub fn resolve_script(def: &Value, data: &std::path::Path, default_lang: &str) -> Result<(String, String, Value)> {
    let spec = script::parse_spec(def)?;
    match (&spec.source, &spec.id) {
        (Some(s), _) => Ok((spec.lang.clone().unwrap_or_else(|| default_lang.to_string()), s.clone(), spec.params.clone())),
        (None, Some(id)) => {
            let st = script::stored::for_data_dir(data).get(id).ok_or_else(|| EsError::resource_not_found(format!("unable to find script [{id}] in cluster state")))?;
            Ok((st.lang.clone(), st.source.clone(), spec.params.clone()))
        }
        _ => Err(EsError::parsing("script requires a source")),
    }
}

pub fn run_script(def: &Value, ctx: &Value, data: &std::path::Path) -> Result<Value> {
    run_script_context(def, ctx, data, ScriptContext::Watcher)
}
fn run_script_context(def: &Value, ctx: &Value, data: &std::path::Path, context: ScriptContext) -> Result<Value> {
    let (_, src, params) = resolve_script(def, data, "painless")?;
    let cs = script::compile(&src, &params, context)?;
    let mut ex = cs.executor(None, None);
    ex.set_ctx(Val::from_json(ctx));
    let out = ex.run().map_err(|e| cs.error(&e))?;
    // Payload.Simple: CollectionUtils.ensureNoSelfReferences
    out.try_to_json().map_err(|_| EsError::illegal_argument("Iterable object is self-referencing itself (watcher action payload)"))
}

/// Render a `TextTemplate`: a mustache string, or a script object (inline
/// or stored, mustache or painless) whose params are merged with `ctx`.
pub fn render_text(v: &Value, ctx: &Value, data: &std::path::Path) -> Result<String> {
    match v {
        Value::String(s) => Ok(render(s, ctx)),
        Value::Object(_) => {
            let (lang, src, params) = resolve_script(v, data, "mustache")?;
            let mut model = params.as_object().cloned().unwrap_or_default();
            model.insert("ctx".into(), ctx.clone());
            if lang == "mustache" {
                crate::template::render(&unescape_mustache(&src), &Value::Object(model))
            } else {
                let cs = script::compile(&src, &Value::Object(model), ScriptContext::WatcherTemplate)?;
                let mut ex = cs.executor(None, None);
                ex.set_ctx(Val::from_json(ctx));
                let out = ex.run().map_err(|e| cs.error(&e))?.to_json();
                Ok(match out {
                    Value::String(s) => s,
                    Value::Null => String::new(),
                    other => other.to_string(),
                })
            }
        }
        Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// A comparison operand: rendered, with date math resolved to epoch millis.
fn operand(v: &Value, ctx: &Value, now: i64) -> (Value, bool) {
    if let Value::String(s) = v {
        let r = render(s, ctx);
        let inner = r.strip_prefix("<{").and_then(|x| x.strip_suffix("}>")).unwrap_or(&r).to_string();
        if inner.starts_with("now") || inner.contains("||") {
            if let Ok(f) = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis") {
                if let Ok(ms) = crate::date::parse_date_math(&inner, &f, crate::date::Tz::UTC, now, false) {
                    return (json!(ms), true);
                }
            }
        }
        return (Value::String(r), false);
    }
    (v.clone(), false)
}

fn compare_values(l: &Value, op: &str, r: &Value, date: bool) -> std::result::Result<bool, String> {
    use std::cmp::Ordering;
    let ord: Option<Ordering> = if date {
        let lm = match l {
            Value::String(s) => parse_iso(s),
            other => as_num(other).map(|x| x as i64),
        };
        match (lm, as_num(r)) {
            (Some(a), Some(b)) => (a as f64).partial_cmp(&b),
            _ => None,
        }
    } else if let (Some(a), Some(b)) = (as_num(l), as_num(r)) {
        a.partial_cmp(&b)
    } else if let (Value::String(a), Value::String(b)) = (l, r) {
        Some(a.cmp(b))
    } else {
        None
    };
    match op {
        "eq" => Ok(ord.map(|o| o == Ordering::Equal).unwrap_or(l == r)),
        "not_eq" => Ok(ord.map(|o| o != Ordering::Equal).unwrap_or(l != r)),
        "gt" | "gte" | "lt" | "lte" => {
            let o = ord.ok_or_else(|| format!("unable to compare values [{}] and [{}]", l, r))?;
            Ok(match op {
                "gt" => o == Ordering::Greater,
                "gte" => o != Ordering::Less,
                "lt" => o == Ordering::Less,
                _ => o != Ordering::Greater,
            })
        }
        other => Err(format!("unknown comparison operator [{other}]")),
    }
}

pub fn run_condition(cond: &Value, ctx: &Ctx) -> Value {
    let (ty, def) = match cond.as_object().and_then(|o| o.iter().next()) {
        Some((k, v)) => (k.as_str(), v),
        None => return json!({"type": "always", "status": "success", "met": true}),
    };
    let cj = ctx.to_json();
    let fail = |reason: String| json!({"type": ty, "status": "failure", "reason": reason});
    match ty {
        "always" => json!({"type": "always", "status": "success", "met": true}),
        "never" => json!({"type": "never", "status": "success", "met": false}),
        "compare" => {
            let Some((path, ops)) = def.as_object().and_then(|o| o.iter().next()) else { return fail("compare condition requires a path".into()) };
            let Some((op, rhs)) = ops.as_object().and_then(|o| o.iter().next()) else { return fail("compare condition requires an operator".into()) };
            let key = path.strip_prefix("ctx.").unwrap_or(path);
            let lhs = path_get(&cj, key).cloned().unwrap_or(Value::Null);
            let (r, date) = operand(rhs, &cj, ctx.execution_time);
            match compare_values(&lhs, op, &r, date) {
                Ok(met) => json!({"type": "compare", "status": "success", "met": met, "compare": {"resolved_values": {path.clone(): lhs}}}),
                Err(e) => fail(e),
            }
        }
        "array_compare" => {
            let Some((path, spec)) = def.as_object().and_then(|o| o.iter().next()) else { return fail("array_compare condition requires a path".into()) };
            let key = path.strip_prefix("ctx.").unwrap_or(path);
            let arr = path_get(&cj, key).and_then(Value::as_array).cloned().unwrap_or_default();
            let sub = spec.get("path").and_then(Value::as_str);
            let Some((op, o2)) = spec.as_object().and_then(|o| o.iter().find(|(k, _)| k.as_str() != "path")) else { return fail("array_compare requires an operator".into()) };
            let quant = o2.get("quantifier").and_then(Value::as_str).unwrap_or("some");
            let (r, date) = operand(&o2["value"], &cj, ctx.execution_time);
            let mut matches = 0usize;
            for el in &arr {
                let v = match sub {
                    Some(p) => path_get(el, p).cloned().unwrap_or(Value::Null),
                    None => el.clone(),
                };
                match compare_values(&v, op, &r, date) {
                    Ok(true) => matches += 1,
                    Ok(false) => {}
                    Err(e) => return fail(e),
                }
            }
            let met = if quant == "all" { matches == arr.len() } else { matches > 0 };
            json!({"type": "array_compare", "status": "success", "met": met, "array_compare": {"resolved_values": {path.clone(): arr}}})
        }
        "script" => match run_script_context(def, &cj, &ctx.data, ScriptContext::WatcherCondition) {
            Ok(Value::Bool(b)) => json!({"type": "script", "status": "success", "met": b}),
            Ok(other) => fail(format!("condition script must return a boolean, but returned [{other}]")),
            Err(e) => fail(e.reason.clone()),
        },
        _ => fail(format!("unknown condition type [{ty}]")),
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

pub fn run_transform(node: &Node, t: &Value, ctx: &Ctx) -> Value {
    let (ty, def) = match t.as_object().and_then(|o| o.iter().next()) {
        Some((k, v)) => (k.as_str(), v),
        None => return json!({"type": "none", "status": "success", "payload": ctx.payload}),
    };
    let fail = |reason: String| json!({"type": ty, "status": "failure", "reason": reason});
    match ty {
        "search" => match run_search(node, &def["request"], ctx) {
            Ok((resp, rendered)) => json!({"type": "search", "status": "success", "payload": resp, "search": {"request": rendered}}),
            Err(e) => json!({"type": ty, "status": "failure", "reason": e.reason.clone(), "error": e.error_object(false)}),
        },
        "script" => match run_script(def, &ctx.to_json(), &ctx.data) {
            Ok(v) => {
                let payload = if v.is_object() { v } else { json!({"_value": v}) };
                json!({"type": "script", "status": "success", "payload": payload})
            }
            Err(e) => json!({"type": ty, "status": "failure", "reason": e.reason.clone(), "error": e.error_object(false)}),
        },
        "chain" => {
            let mut c = ctx.clone();
            for sub in def.as_array().cloned().unwrap_or_default() {
                let r = run_transform(node, &sub, &c);
                if r["status"] != "success" {
                    return json!({"type": "chain", "status": "failure", "reason": r["reason"].clone()});
                }
                c.payload = r["payload"].clone();
            }
            json!({"type": "chain", "status": "success", "payload": c.payload})
        }
        _ => fail(format!("unknown transform type [{ty}]")),
    }
}

// ---------------------------------------------------------------------------
// Throttling / ack
// ---------------------------------------------------------------------------

fn plural(n: i64, unit: &str) -> String {
    if n == 1 { format!("{n} {unit}") } else { format!("{n} {unit}s") }
}

/// `TimeValue`-style rendering (`5s`, `59ms`, `1h`).
pub fn fmt_tv(ms: i64) -> String {
    if ms > 0 {
        for (u, s) in [("d", 86_400_000i64), ("h", 3_600_000), ("m", 60_000), ("s", 1000)] {
            if ms % s == 0 {
                return format!("{}{}", ms / s, u);
            }
        }
    }
    format!("{ms}ms")
}

/// Joda `PeriodFormat.getDefault()` rendering of a duration.
pub fn fmt_period(ms: i64) -> String {
    let mut rest = ms;
    let mut parts = Vec::new();
    for (unit, size) in [("week", 604_800_000i64), ("day", 86_400_000), ("hour", 3_600_000), ("minute", 60_000), ("second", 1000), ("millisecond", 1)] {
        let n = rest / size;
        rest %= size;
        if n > 0 {
            parts.push(plural(n, unit));
        }
    }
    if parts.is_empty() {
        return "0 milliseconds".into();
    }
    if parts.len() == 1 {
        return parts.remove(0);
    }
    let last = parts.pop().unwrap();
    format!("{} and {}", parts.join(", "), last)
}

/// `Some((acked, reason))` when the action is throttled (period or ack).
fn throttle_reason(aid: &str, def: &Value, astatus: &Value, watch_throttle: Option<i64>, default_ms: i64, now: i64) -> Option<(bool, String)> {
    let period = def["throttle_period_in_millis"].as_i64().or(watch_throttle).unwrap_or(default_ms);
    if let Some(ts) = astatus["last_successful_execution"]["timestamp"].as_str().and_then(parse_iso) {
        let elapsed = now - ts;
        if elapsed < period {
            return Some((false, format!("throttling interval is set to [{}] but time elapsed since last execution is [{}]", fmt_tv(period), fmt_tv(elapsed))));
        }
    }
    if astatus["ack"]["state"] == "acked" {
        let ts = astatus["ack"]["timestamp"].as_str().unwrap_or("");
        return Some((true, format!("action [{aid}] was acked at [{ts}]")));
    }
    None
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Execute a watch. Returns `(record id, watch record)`.
pub fn execute(node: &Node, svc: &WatcherService, watch_id: &str, src: &Value, doc_version: i64, scheduled: Option<i64>, triggered: i64, opts: &ExecOptions) -> Result<(String, Value)> {
    let start = crate::date::now_millis();
    let nanos = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string();
    let rid = format!("{}_{}-{}", watch_id, uuid::Uuid::new_v4(), nanos);
    let mut status = src["status"].clone();
    if !status.is_object() {
        status = json!({});
    }
    status["version"] = json!(doc_version);
    let triggered_s = iso(triggered);
    let sched_s = iso(scheduled.unwrap_or(triggered));
    let mut ctx = Ctx {
        id: rid.clone(),
        watch_id: watch_id.to_string(),
        execution_time: start,
        triggered: triggered_s.clone(),
        scheduled: sched_s.clone(),
        metadata: src.get("metadata").cloned().unwrap_or(json!({})),
        payload: json!({}),
        vars: json!({}),
        data: node.data.clone(),
        user: opts.user.clone().filter(|_| node.security.enabled()),
    };
    if let Some(td) = &opts.trigger_data {
        // ScheduleTriggerEngine.simulateEvent: scheduled defaults to triggered
        let conv = |k: &str| -> Result<Option<i64>> {
            let Some(v) = td.get(k) else { return Ok(None) };
            let ms = match v {
                Value::String(s) => crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis")
                    .ok()
                    .and_then(|f| crate::date::parse_date_math(s, &f, crate::date::Tz::UTC, start, false).ok()),
                Value::Number(n) => n.as_i64(),
                _ => None,
            };
            let what = if k == "triggered_time" { "triggered" } else { "scheduled" };
            ms.map(Some).ok_or_else(|| EsError::illegal_argument(format!("could not simulate schedule event. could not convert provided {what} time [{}] to date/time", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))))
        };
        if let Some(t) = conv("triggered_time")? {
            ctx.triggered = iso(t);
        }
        ctx.scheduled = match conv("scheduled_time")? {
            Some(t) => iso(t),
            None => ctx.triggered.clone(),
        };
    }
    svc.current.lock().insert(rid.clone(), CurrentExec { watch_id: watch_id.to_string(), record_id: rid.clone(), triggered_time: triggered, execution_time: start, phase: "input" });
    let triggered_doc = if !opts.manual {
        super::triggered_index(node).ok().map(|s| {
            let _ = super::put_doc(&s, &rid, &json!({"trigger_event": {"type": "schedule", "triggered_time": triggered_s, "schedule": {"scheduled_time": sched_s}}, "state": "waiting"}), Default::default());
            s
        })
    } else {
        None
    };

    // input
    let input_def = match &opts.alternative_input {
        Some(p) => json!({"simple": p}),
        None => src["input"].clone(),
    };
    let input_res = run_input(node, &input_def, &ctx);
    let mut state = "executed";
    let mut messages: Vec<&str> = Vec::new();
    let mut actions_res: Vec<Value> = Vec::new();
    let mut cond_res = Value::Null;
    let mut transform_res = Value::Null;
    let mut met = false;
    let mut transform_failed = false;
    let mut ack_resets: Vec<String> = Vec::new();
    let mut index_time_field: Option<(String, String)> = None;
    let now_ck = crate::date::now_millis();
    let watch_throttle = src["throttle_period_in_millis"].as_i64();
    let default_throttle = super::default_throttle_ms(node);
    if input_res["status"] != "success" {
        state = "failed";
        messages.push("failed to execute watch input");
    } else {
        ctx.payload = input_res["payload"].clone();
        if let Some(c) = svc.current.lock().get_mut(&rid) {
            c.phase = "condition";
        }
        cond_res = if opts.ignore_condition { json!({"type": "always", "status": "success", "met": true}) } else { run_condition(&src["condition"], &ctx) };
        if cond_res["status"] != "success" {
            state = "failed";
            messages.push("failed to execute watch condition");
        } else {
            met = cond_res["met"].as_bool().unwrap_or(false) || opts.ignore_condition;
            if !met {
                state = "execution_not_needed";
            } else {
                if let Some(t) = src.get("transform").filter(|_| src["actions"].as_object().map(|a| !a.is_empty()).unwrap_or(false)) {
                    transform_res = run_transform(node, t, &ctx);
                    if transform_res["status"] == "success" {
                        ctx.payload = transform_res["payload"].clone();
                    } else {
                        // ES derives the state from the condition result: still `executed`
                        transform_failed = true;
                        messages.push("failed to execute watch transform");
                    }
                }
                if !transform_failed {
                    if let Some(c) = svc.current.lock().get_mut(&rid) {
                        c.phase = "actions";
                    }
                    let default_mode = "execute";
                    let acts = src["actions"].as_object().cloned().unwrap_or_default();
                    for (aid, def) in &acts {
                        let ty = super::model::action_type(def);
                        let mode = opts.action_modes.get(aid).or_else(|| opts.action_modes.get("_all")).map(String::as_str).unwrap_or(default_mode).to_string();
                        if mode == "skip" {
                            actions_res.push(json!({"id": aid, "type": ty, "status": "throttled", "reason": "manually skipped"}));
                            continue;
                        }
                        let force = mode.starts_with("force_");
                        let astatus = status["actions"][aid].clone();
                        if !force {
                            if let Some((acked, reason)) = throttle_reason(aid, def, &astatus, watch_throttle, default_throttle, now_ck) {
                                actions_res.push(json!({"id": aid, "type": ty, "status": if acked { "acknowledged" } else { "throttled" }, "reason": reason}));
                                continue;
                            }
                        }
                        let mut actx = ctx.clone();
                        if let Some(c) = def.get("condition") {
                            let r = run_condition(c, &actx);
                            if r["met"] != true {
                                ack_resets.push(aid.clone());
                                let ctype = r["type"].clone();
                                actions_res.push(json!({"id": aid, "type": ty, "status": "condition_failed", "reason": "condition not met. skipping", ty: {}, "condition": {"type": ctype, "status": "success", "met": false}}));
                                continue;
                            }
                        }
                        if let Some(t) = def.get("transform") {
                            let r = run_transform(node, t, &actx);
                            if r["status"] == "success" {
                                actx.payload = r["payload"].clone();
                            } else {
                                actions_res.push(json!({"id": aid, "type": ty, "status": "failure", "transform": r, "reason": "Failed to transform payload"}));
                                continue;
                            }
                        }
                        let simulate = mode.contains("simulate");
                        let t0 = std::time::Instant::now();
                        let r = actions::run_action(node, aid, def, &actx, simulate);
                        svc.record_action(ty, t0.elapsed().as_millis() as u64);
                        if ty == "index" && r["status"] != "failure" {
                            if let Some(f) = def["index"]["execution_time_field"].as_str() {
                                index_time_field = Some((f.to_string(), iso(start)));
                            }
                        }
                        actions_res.push(r);
                    }
                    if actions_res.iter().any(|a| a["status"] == "acknowledged") {
                        state = "acknowledged";
                    } else if actions_res.iter().any(|a| a["status"] == "throttled") {
                        state = "throttled";
                    }
                }
            }
        }
    }

    // status update
    let ts = iso(now_ck);
    if input_res["status"] == "success" && cond_res["status"] == "success" {
        status["last_checked"] = json!(ts);
    }
    if met {
        status["last_met_condition"] = json!(ts);
    }
    if !status["actions"].is_object() {
        status["actions"] = json!({});
    }
    if input_res["status"] == "success" && cond_res["status"] == "success" && !met {
        let ids: Vec<String> = status["actions"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        for id in ids {
            let st = status["actions"][&id]["ack"]["state"].as_str().unwrap_or("").to_string();
            if st == "acked" || st == "ackable" {
                status["actions"][&id]["ack"] = json!({"timestamp": ts, "state": "awaits_successful_execution"});
            }
        }
    }
    for id in &ack_resets {
        if status["actions"][id].is_object() {
            status["actions"][id]["ack"] = json!({"timestamp": ts, "state": "awaits_successful_execution"});
        }
    }
    for a in &actions_res {
        let id = a["id"].as_str().unwrap_or("");
        let entry = &mut status["actions"][id];
        if !entry.is_object() {
            *entry = json!({"ack": {"timestamp": ts, "state": "awaits_successful_execution"}});
        }
        match a["status"].as_str().unwrap_or("") {
            "success" | "simulated" => {
                entry["last_execution"] = json!({"timestamp": ts, "successful": true});
                entry["last_successful_execution"] = json!({"timestamp": ts, "successful": true});
                if entry["ack"]["state"] == "awaits_successful_execution" {
                    entry["ack"] = json!({"timestamp": ts, "state": "ackable"});
                }
            }
            "failure" => {
                entry["last_execution"] = json!({"timestamp": ts, "successful": false, "reason": a["reason"].as_str().unwrap_or("")});
            }
            "throttled" | "acknowledged" => {
                entry["last_throttle"] = json!({"timestamp": ts, "reason": a["reason"].clone()});
            }
            _ => {}
        }
    }
    status["execution_state"] = json!(state);
    let status = super::api::canon_status(&status);

    let end = crate::date::now_millis();
    let mut result = Map::new();
    result.insert("execution_time".into(), json!(iso(start)));
    result.insert("execution_duration".into(), json!(end - start));
    // ES quirk: the index action's execution_time_field is written into the shared payload map.
    let mut input_res = input_res;
    if let Some((f, t)) = &index_time_field {
        let target = if transform_res.is_null() { &mut input_res } else { &mut transform_res };
        if let Some(p) = target.get_mut("payload").and_then(Value::as_object_mut) {
            p.insert(f.clone(), json!(t));
        }
    }
    result.insert("input".into(), input_res);
    if !cond_res.is_null() {
        result.insert("condition".into(), cond_res);
    }
    if !transform_res.is_null() {
        result.insert("transform".into(), transform_res);
    }
    result.insert("actions".into(), Value::Array(actions_res));
    let trigger_event = if opts.manual {
        json!({"type": "manual", "triggered_time": ctx.triggered, "manual": {"schedule": {"scheduled_time": ctx.scheduled}}})
    } else {
        json!({"type": "schedule", "triggered_time": triggered_s, "schedule": {"scheduled_time": sched_s}})
    };
    let mut record = Map::new();
    record.insert("watch_id".into(), json!(watch_id));
    record.insert("node".into(), json!(NODE_ID));
    record.insert("state".into(), json!(state));
    if let Some(u) = &ctx.user {
        record.insert("user".into(), json!(u));
    }
    let stamp = match &opts.trigger_data {
        Some(_) => ctx.triggered.clone(),
        None => nanos.clone(),
    };
    record.insert("@timestamp".into(), json!(stamp));
    record.insert("status".into(), status.clone());
    record.insert("trigger_event".into(), trigger_event);
    record.insert("input".into(), super::model::redact(&src["input"]));
    record.insert("condition".into(), src["condition"].clone());
    if let Some(m) = src.get("metadata") {
        record.insert("metadata".into(), m.clone());
    }
    record.insert("result".into(), Value::Object(result));
    record.insert("messages".into(), json!(messages));
    let record = Value::Object(record);

    if opts.record {
        super::api::store_status(node, svc, watch_id, &status);
        super::write_history(node, &rid, &record, start);
    }
    if let Some(s) = triggered_doc {
        let _ = super::delete_doc(&s, &rid);
    }
    svc.current.lock().remove(&rid);
    svc.counters.executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((rid, record))
}

/// Scheduler entry: execute the stored watch `id` because its trigger fired.
pub fn run_scheduled(node: &Node, svc: &WatcherService, id: &str, scheduled: i64, triggered: i64) -> Result<()> {
    let src = match svc.cache.read().get(id) {
        Some(w) => w.src.clone(),
        None => return Ok(()),
    };
    let user = src["status"]["headers"]["_xpack_security_authentication"].as_str().map(str::to_string);
    let opts = ExecOptions { record: true, user, ..Default::default() };
    let version = svc.cache.read().get(id).map(|w| w.version as i64).unwrap_or(-1);
    execute(node, svc, id, &src, version, Some(scheduled), triggered, &opts)?;
    let _ = service(node);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_referencing_payload() {
        let _compilation_budget = crate::script::rate_limit::fixture_budget("10000/1m");
        let e = run_script(&json!("def x = [:] ; def y = [:] ; x.a = y ; y.a = x ; return x"), &json!({}), std::path::Path::new("/nonexistent")).unwrap_err();
        assert_eq!(e.reason, "Iterable object is self-referencing itself (watcher action payload)");
        assert_eq!(java_zdt(1_420_329_600_000), "2015-01-04T00:00Z");
        assert_eq!(java_zdt(1_430_859_482_443), "2015-05-05T20:58:02.443Z");
    }

    #[test]
    fn templates() {
        let ctx = json!({"payload": {"a": 1, "hits": {"total": 5, "hits": [{"n": "x"}, {"n": "y"}]}}, "watch_id": "w\"1"});
        assert_eq!(render("hi {{ctx.payload.a}} {{ctx.watch_id}}", &ctx), "hi 1 w\"1");
        assert_eq!(render("{{#ctx.payload.hits.hits}}[{{n}}]{{/ctx.payload.hits.hits}}", &ctx), "[x][y]");
        assert_eq!(unescape_mustache("{{{a}}} {{b}} {{#c}}"), "{{{a}}} {{{b}}} {{#c}}");
    }

    #[test]
    fn compare_and_period() {
        let ctx = Ctx { id: "i".into(), watch_id: "w".into(), execution_time: 1_000_000, triggered: iso(0), scheduled: iso(0), metadata: json!({}), payload: json!({"hits": {"total": 7}, "arr": [{"v": 1}, {"v": 5}]}), vars: json!({}), data: Default::default(), user: None };
        assert_eq!(run_condition(&json!({"compare": {"ctx.payload.hits.total": {"gte": 5}}}), &ctx)["met"], true);
        assert_eq!(run_condition(&json!({"compare": {"ctx.payload.hits.total": {"lt": 5}}}), &ctx)["met"], false);
        assert_eq!(run_condition(&json!({"array_compare": {"ctx.payload.arr": {"path": "v", "gte": {"value": 5, "quantifier": "some"}}}}), &ctx)["met"], true);
        assert_eq!(run_condition(&json!({"array_compare": {"ctx.payload.arr": {"path": "v", "gte": {"value": 5, "quantifier": "all"}}}}), &ctx)["met"], false);
        assert_eq!(fmt_period(5000), "5 seconds");
        assert_eq!(fmt_period(3020), "3 seconds and 20 milliseconds");
        assert_eq!(fmt_period(3_723_000), "1 hour, 2 minutes and 3 seconds");
    }
}
