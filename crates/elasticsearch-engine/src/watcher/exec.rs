//! Watch execution: input -> condition -> transform -> actions, with
//! throttling, acking, status update and history.

use super::{actions, httpc, iso, parse_iso, service, CurrentExec, WatcherService};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts, NODE_ID};
use crate::script::{self, ScriptContext, Val};
use serde_json::{json, Map, Value};
use crate::security::owner::StoredAuthn;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// Security: the subject the watch runs as (`None`: security off, or a
    /// watch stored with security off, which runs as the internal origin).
    pub identity: Option<Arc<StoredAuthn>>,
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
    /// Security: the subject the execution runs as (its principal is
    /// recorded as `user`).
    pub identity: Option<StoredAuthn>,
    /// The `.triggered_watches` record this execution runs for (a recovered
    /// trigger keeps its record id; `None`: a new one).
    pub record_id: Option<String>,
}

/// Security check of `action` on index `name` for the watch's subject.
pub fn user_allowed(node: &Node, identity: Option<&StoredAuthn>, action: &str, name: &str) -> bool {
    let Some(id) = identity else { return true };
    let perm = node.security.permission(&id.to_authn());
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
/// The body (and routing) a clustered search input sends for `targets`
/// resolved here: the coordinator searches concrete indices, so each
/// target's alias filter is carried as a filter on its `_index` (finding
/// 204: a filtered alias keeps selecting only its documents), and the
/// routing is kept when every target routes the same way.
fn remote_search_body(body: &Value, targets: &[crate::search::Target]) -> (Value, Option<String>) {
    let mut send = body.clone();
    if targets.iter().any(|t| t.alias_filter.is_some()) {
        let per_target: Vec<Value> = targets
            .iter()
            .map(|t| match &t.alias_filter {
                Some(f) => json!({"bool": {"filter": [{"term": {"_index": t.svc.name}}, f]}}),
                None => json!({"term": {"_index": t.svc.name}}),
            })
            .collect();
        let query = send.get("query").cloned().unwrap_or_else(|| json!({"match_all": {}}));
        send["query"] = json!({"bool": {"must": [query], "filter": [{"bool": {"should": per_target, "minimum_should_match": 1}}]}});
    }
    let first = targets.first().and_then(|t| t.routing.clone());
    let routing = first.filter(|r| targets.iter().all(|t| t.routing.as_deref() == Some(r.as_str())));
    (send, routing)
}

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
    if let Some(id) = ctx.identity.as_deref() {
        // Run as the watch's subject: indices it may not read are skipped
        // (the default indices options ignore unavailable indices), and its
        // document / field level security applies to the rest.
        const SEARCH: &str = "indices:data/read/search";
        let authn = id.to_authn();
        let perm = node.security.permission(&authn);
        targets.retain(|t| perm.is_superuser || perm.check_index(SEARCH, &t.svc.name));
        if perm.has_dls_fls() {
            for t in targets.iter_mut() {
                let a = perm.index_access(SEARCH, &[t.svc.name.as_str()], &authn)?;
                t.access = Some(Arc::new(a)).filter(|a| a.is_restricted());
            }
        }
    }
    let mut sr = crate::search::parse_request(&body, &HashMap::new())?;
    sr.rest_total_hits_as_int = rest_total;
    let resp: Value = match super::remote_as(ctx.identity.as_deref())?.filter(|_| !targets.is_empty()) {
        // On a cluster: over every shard of the (permitted) targets.
        Some(rest) => {
            let mut names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
            names.sort();
            names.dedup();
            let (send, routing) = remote_search_body(&body, &targets);
            let mut uri = format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true&rest_total_hits_as_int={rest_total}", super::path_expr(&names.join(",")));
            if let Some(r) = routing {
                uri.push_str(&format!("&routing={}", super::path_expr(&r)));
            }
            rest("POST", &uri, Some(send))?
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

/// `extract`: keep only the selected paths, structurally (ES
/// `XContentFilterKeysUtils.filterMapOrdered`): parents stay objects, and a
/// path through an array selects in every element (#585).
fn apply_extract(payload: &mut Value, def: &Value) {
    if let Some(keys) = def.get("extract").and_then(Value::as_array) {
        let keys: Vec<&str> = keys.iter().filter_map(Value::as_str).collect();
        *payload = match filter_keys(payload, "", &keys) {
            Some(v @ Value::Object(_)) => v,
            _ => json!({}),
        };
    }
}

fn filter_keys(v: &Value, path: &str, keys: &[&str]) -> Option<Value> {
    if !path.is_empty() && keys.contains(&path) {
        return Some(v.clone());
    }
    let under = |p: &str| keys.iter().any(|k| *k == p || (k.len() > p.len() && k.starts_with(p) && k.as_bytes()[p.len()] == b'.'));
    match v {
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, child) in o {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                if under(&p) {
                    if let Some(c) = filter_keys(child, &p, keys) {
                        out.insert(k.clone(), c);
                    }
                }
            }
            (!out.is_empty() || path.is_empty()).then_some(Value::Object(out))
        }
        Value::Array(a) => {
            let out: Vec<Value> = a.iter().filter_map(|e| filter_keys(e, path, keys)).collect();
            (!out.is_empty()).then_some(Value::Array(out))
        }
        _ => None,
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
            match httpc::send(node, &r) {
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

/// A comparison operand (ES `AbstractCompareCondition.resolveConfiguredValue`):
/// `<{date math}>` resolves to epoch millis; `{{path}}` to the typed value
/// at that execution-context path (objects, arrays, booleans and null
/// keep their type, #582); any other string is literal.
fn operand(v: &Value, ctx: &Value, now: i64) -> (Value, bool) {
    if let Value::String(s) = v {
        if let Some(inner) = s.strip_prefix("<{").and_then(|x| x.strip_suffix("}>")) {
            if let Ok(f) = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis") {
                if let Ok(ms) = crate::date::parse_date_math(inner, &f, crate::date::Tz::UTC, now, false) {
                    return (json!(ms), true);
                }
            }
            return (v.clone(), false);
        }
        if let Some(path) = s.strip_prefix("{{").and_then(|x| x.strip_suffix("}}")).filter(|p| !p.is_empty()) {
            let path = path.trim();
            let found = match path {
                "ctx" => Some(ctx.clone()),
                p => path_get(ctx, p.strip_prefix("ctx.").unwrap_or(p)).cloned(),
            };
            return (found.unwrap_or(Value::Null), false);
        }
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
    let rid = opts.record_id.clone().unwrap_or_else(|| format!("{}_{}-{}", watch_id, uuid::Uuid::new_v4(), nanos));
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
        identity: opts.identity.clone().filter(|_| node.security.enabled()).map(Arc::new),
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
    // One execution per watch at a time (ES `ExecutionService.execute`:
    // `currentExecutions.put` refuses a second one), from admission through
    // the status update: an overlapping run, scheduled or manual, records
    // NOT_EXECUTED_ALREADY_QUEUED and runs no input, condition or action
    // (#583).
    {
        let mut cur = svc.current.lock();
        if watch_id != "_inlined_" && cur.values().any(|c| c.watch_id == watch_id) {
            drop(cur);
            let trigger_event = if opts.manual {
                json!({"type": "manual", "triggered_time": ctx.triggered, "manual": {"schedule": {"scheduled_time": ctx.scheduled}}})
            } else {
                json!({"type": "schedule", "triggered_time": triggered_s, "schedule": {"scheduled_time": sched_s}})
            };
            let mut record = Map::new();
            record.insert("watch_id".into(), json!(watch_id));
            record.insert("node".into(), json!(NODE_ID));
            record.insert("state".into(), json!("not_executed_already_queued"));
            record.insert("@timestamp".into(), json!(nanos));
            record.insert("status".into(), status.clone());
            record.insert("trigger_event".into(), trigger_event);
            record.insert("input".into(), super::model::redact(&src["input"]));
            record.insert("condition".into(), src["condition"].clone());
            if let Some(m) = src.get("metadata") {
                record.insert("metadata".into(), m.clone());
            }
            record.insert("messages".into(), json!(["Watch is already queued in thread pool"]));
            let record = Value::Object(record);
            if opts.record {
                super::write_history(node, &rid, &record, start);
            }
            return Ok((rid, record));
        }
        cur.insert(rid.clone(), CurrentExec { watch_id: watch_id.to_string(), record_id: rid.clone(), triggered_time: triggered, execution_time: start, phase: "input" });
    }
    struct Release<'a>(&'a WatcherService, String);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.current.lock().remove(&self.1);
        }
    }
    let _release = Release(svc, rid.clone());
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
                let reason = a["reason"].as_str().or_else(|| a["error"]["reason"].as_str()).unwrap_or("");
                entry["last_execution"] = json!({"timestamp": ts, "successful": false, "reason": reason});
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
    if let Some(id) = &ctx.identity {
        record.insert("user".into(), json!(id.principal()));
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
        super::api::store_status(node, svc, watch_id, &src["status"], &status);
        super::write_history(node, &rid, &record, start);
    }
    if let Some(s) = triggered_doc {
        let _ = super::delete_doc(&s, &rid);
    }
    svc.current.lock().remove(&rid);
    svc.counters.executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((rid, record))
}

/// The subject a stored watch runs as: the authentication stored with it
/// by the put (ES's `_xpack_security_authentication` header). A watch from
/// an earlier version recorded only the user name: that user, as it is now.
/// `None` for a watch stored with security off (ES: the watcher origin).
pub fn stored_identity(node: &Node, watch_id: &str, src: &Value) -> Result<Option<StoredAuthn>> {
    let Some(h) = src["status"]["headers"]["_xpack_security_authentication"].as_str() else { return Ok(None) };
    if !node.security.enabled() {
        return Ok(None);
    }
    if let Some(id) = StoredAuthn::decode(h) {
        return Ok(Some(id));
    }
    match node.security.lookup_user(h) {
        Some(a) => Ok(Some(StoredAuthn::of(&a))),
        None => Err(EsError::new(403, "security_exception", format!("watch [{watch_id}] runs as user [{h}], which no longer exists"))),
    }
}

/// Scheduler entry: execute the stored watch `id` because its trigger fired.
/// Execute a recovered trigger (a `.triggered_watches` record left by an
/// execution that never finished, e.g. across a restart) with its original
/// scheduled and triggered times; the record is retired when it completes.
pub fn run_recovered(node: &Node, svc: &WatcherService, id: &str, record_id: &str, scheduled: i64, triggered: i64) -> Result<()> {
    let src = match svc.cache.read().get(id) {
        Some(w) => w.src.clone(),
        None => return Ok(()),
    };
    let identity = stored_identity(node, id, &src)?;
    let opts = ExecOptions { record: true, identity, record_id: Some(record_id.to_string()), ..Default::default() };
    let version = svc.cache.read().get(id).map(|w| w.version as i64).unwrap_or(-1);
    execute(node, svc, id, &src, version, Some(scheduled), triggered, &opts)?;
    Ok(())
}

pub fn run_scheduled(node: &Node, svc: &WatcherService, id: &str, scheduled: i64, triggered: i64) -> Result<()> {
    let src = match svc.cache.read().get(id) {
        Some(w) => w.src.clone(),
        None => return Ok(()),
    };
    let identity = stored_identity(node, id, &src)?;
    let opts = ExecOptions { record: true, identity, ..Default::default() };
    let version = svc.cache.read().get(id).map(|w| w.version as i64).unwrap_or(-1);
    execute(node, svc, id, &src, version, Some(scheduled), triggered, &opts)?;
    let _ = service(node);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finding 204: the body a clustered search input sends for a filtered
    /// alias selects exactly the alias's documents from the concrete index.
    #[test]
    fn clustered_search_keeps_alias_filters() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("orders", &json!({"mappings": {"properties": {"status": {"type": "keyword"}}}})).unwrap();
        let reqs = [("1", "failed"), ("2", "ok"), ("3", "failed")]
            .iter()
            .map(|(id, st)| crate::index::WriteReq::Index { id: Some(id.to_string()), source: json!({"status": st}).to_string(), routing: None, cond: Default::default() })
            .collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        node.update_aliases(&[json!({"add": {"index": "orders", "alias": "failed-orders", "filter": {"term": {"status": "failed"}}}})]).unwrap();
        let ids = |targets: &[crate::search::Target], body: &Value| -> Vec<String> {
            let req = crate::search::parse_request(body, &HashMap::new()).unwrap();
            let out = crate::search::execute(targets, &req, &super::super::search_opts(&node)).unwrap();
            let v: Value = serde_json::from_str(&out.json).unwrap();
            let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
            ids.sort();
            ids
        };
        let body = json!({"query": {"match_all": {}}});
        let alias_targets = node.resolve("failed-orders", ResolveOpts::default()).unwrap();
        let want = ids(&alias_targets, &body);
        assert_eq!(want, vec!["1", "3"]);
        // What the coordinator receives: the concrete index and this body.
        let (send, routing) = remote_search_body(&body, &alias_targets);
        let concrete = node.resolve("orders", ResolveOpts::default()).unwrap();
        assert_eq!(ids(&concrete, &send), want);
        assert_eq!(routing, None);
        // An unfiltered target is sent unchanged.
        assert_eq!(remote_search_body(&body, &concrete).0, body);
    }

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

    fn exec_ctx(payload: Value) -> Ctx {
        Ctx { id: "i".into(), watch_id: "w".into(), execution_time: 1_000_000, triggered: iso(0), scheduled: iso(0), metadata: json!({}), payload, vars: json!({}), data: Default::default(), identity: None }
    }

    fn put(node: &Node, id: &str, w: Value) {
        super::super::api::put_watch(node, id, &w, &super::super::api::PutParams { active: true, version: None, if_seq_no: None, if_primary_term: None, identity: None }).unwrap();
    }

    fn run(node: &Node, id: &str, body: Value) -> Value {
        super::super::api::execute_watch(node, Some(id), &body, false, None).unwrap()["watch_record"].clone()
    }

    /// #581: an empty, missing or non-array foreach object fails the
    /// action (ES `ActionWrapper`); it never records a successful execution
    /// that throttles the next real alert.
    #[test]
    fn empty_foreach_fails_and_never_throttles() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        put(&node, "fe", json!({"trigger": {"schedule": {"interval": "1h"}}, "input": {"simple": {}}, "condition": {"always": {}}, "throttle_period": "1h",
            "actions": {"log": {"foreach": "ctx.payload.alerts", "logging": {"text": "alert {{ctx.payload._value}}"}}}}));
        for (payload, reason) in [(json!({"alerts": []}), "was an empty list"), (json!({}), "was null"), (json!({"alerts": "x"}), "was not a an array")] {
            let r = run(&node, "fe", json!({"alternative_input": payload, "record_execution": true}));
            let a = &r["result"]["actions"][0];
            assert_eq!(a["status"], "failure", "{r}");
            assert!(a["error"]["reason"].as_str().unwrap().contains(reason), "{a}");
        }
        let (_, st) = super::super::api::get_watch(&node, "fe").unwrap();
        assert!(st["status"]["actions"]["log"].get("last_successful_execution").is_none(), "{st}");
        // The real alert right after is executed, not throttled.
        let r = run(&node, "fe", json!({"alternative_input": {"alerts": ["disk full"]}, "record_execution": true}));
        assert_eq!(r["result"]["actions"][0]["status"], "success", "{r}");
        assert_eq!(r["result"]["actions"][0]["number_of_actions_executed"], 1, "{r}");
    }

    /// #582: `{{path}}` references in compare conditions resolve to typed
    /// context values (objects, arrays, booleans, null); other strings are
    /// literal; date math still resolves.
    #[test]
    fn compare_references_keep_their_type() {
        let ctx = exec_ctx(json!({"observed": {"healthy": true}, "expected": {"healthy": true}, "other": {"healthy": false},
            "list": [1, 2], "same": [1, 2], "flag": true, "n": 7, "limit": 5, "s": "{{x}}"}));
        let c = |k: &str, op: &str, v: &str| run_condition(&json!({"compare": {format!("ctx.payload.{k}"): {op: v}}}), &ctx)["met"].clone();
        assert_eq!(c("observed", "not_eq", "{{ctx.payload.expected}}"), false);
        assert_eq!(c("observed", "eq", "{{ctx.payload.expected}}"), true);
        assert_eq!(c("observed", "eq", "{{ctx.payload.other}}"), false);
        assert_eq!(c("observed", "not_eq", "{{ctx.payload.other}}"), true);
        assert_eq!(c("list", "eq", "{{ctx.payload.same}}"), true);
        assert_eq!(c("flag", "eq", "{{ctx.payload.flag}}"), true);
        assert_eq!(c("n", "gt", "{{ctx.payload.limit}}"), true);
        assert_eq!(c("missing", "eq", "{{ctx.payload.absent}}"), true, "null equals null");
        assert_eq!(c("s", "eq", "{{x}}"), false, "x is not in the context: null");
        assert_eq!(run_condition(&json!({"compare": {"ctx.execution_time": {"gte": "<{now-1h}>"}}}), &ctx)["status"], "success");
    }

    /// #583: a second execution of a watch while one is in flight records
    /// `not_executed_already_queued` and runs nothing; once the first is
    /// done the watch executes again.
    #[test]
    fn overlapping_executions_of_one_watch_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("out", &json!({})).unwrap();
        put(&node, "slow", json!({"trigger": {"schedule": {"interval": "1h"}}, "input": {"simple": {"a": 1}}, "condition": {"always": {}},
            "actions": {"idx": {"index": {"index": "out"}}}}));
        let svc = super::super::service(&node);
        svc.current.lock().insert("other-run".into(), CurrentExec { watch_id: "slow".into(), record_id: "other-run".into(), triggered_time: 0, execution_time: 0, phase: "actions" });
        let r = run(&node, "slow", json!({"record_execution": true}));
        assert_eq!(r["state"], "not_executed_already_queued", "{r}");
        assert!(r.get("result").is_none(), "{r}");
        node.get_index("out").unwrap().refresh().unwrap();
        assert_eq!(node.get_index("out").unwrap().doc_count_realtime().0, 0);
        svc.current.lock().remove("other-run");
        let r = run(&node, "slow", json!({"record_execution": true}));
        assert_eq!(r["state"], "executed", "{r}");
        assert!(svc.current.lock().is_empty(), "the execution released its slot");
    }

    /// #584: a standalone index action runs the destination's default and
    /// final pipelines, as the index API does (transform, drop, failure).
    #[test]
    fn index_action_runs_destination_pipelines() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ingest().put("strip", json!({"processors": [{"remove": {"field": "raw", "ignore_missing": true}}, {"set": {"field": "seen", "value": true}},
            {"drop": {"if": "ctx.drop == true"}}]}), None).unwrap();
        node.ingest().put("bad", json!({"processors": [{"fail": {"message": "rejected by pipeline"}}]}), None).unwrap();
        node.create_index("dest", &json!({"settings": {"index.final_pipeline": "strip"}})).unwrap();
        node.create_index("strict", &json!({"settings": {"index.default_pipeline": "bad"}})).unwrap();
        let watch = |index: &str, payload: Value| json!({"trigger": {"schedule": {"interval": "1h"}}, "input": {"simple": payload}, "condition": {"always": {}},
            "actions": {"idx": {"index": {"index": index, "doc_id": "d1", "refresh": "true"}}}});
        put(&node, "w1", watch("dest", json!({"raw": "secret", "v": 1})));
        let r = run(&node, "w1", json!({}));
        assert_eq!(r["result"]["actions"][0]["status"], "success", "{r}");
        let g = node.get_index("dest").unwrap().get("d1").unwrap().unwrap();
        let src: Value = serde_json::from_str(g.source.as_deref().unwrap()).unwrap();
        assert_eq!(src, json!({"v": 1, "seen": true}), "the final pipeline ran");
        put(&node, "w2", watch("dest", json!({"drop": true, "v": 2})));
        let r = run(&node, "w2", json!({}));
        assert_eq!(r["result"]["actions"][0]["index"]["response"]["result"], "noop", "{r}");
        put(&node, "w3", watch("strict", json!({"v": 3})));
        let r = run(&node, "w3", json!({}));
        assert_eq!(r["result"]["actions"][0]["status"], "failure", "{r}");
        assert!(r["result"]["actions"][0]["reason"].as_str().unwrap_or("").contains("rejected by pipeline"), "{r}");
    }

    /// #585: `extract` keeps the selected paths nested (and through
    /// arrays), so conditions and templates read them as without it.
    #[test]
    fn extract_keeps_nested_paths() {
        let mut p = json!({"hits": {"total": 10, "max_score": 1.0, "hits": [{"_id": "a", "_source": {"name": "x", "big": 1}}, {"_id": "b", "_source": {"name": "y"}}]}, "took": 3});
        apply_extract(&mut p, &json!({"extract": ["hits.total", "hits.hits._source.name"]}));
        assert_eq!(p, json!({"hits": {"total": 10, "hits": [{"_source": {"name": "x"}}, {"_source": {"name": "y"}}]}}));
        let ctx = exec_ctx(p.clone());
        assert_eq!(run_condition(&json!({"compare": {"ctx.payload.hits.total": {"gte": 5}}}), &ctx)["met"], true);
        assert_eq!(render("{{#ctx.payload.hits.hits}}[{{_source.name}}]{{/ctx.payload.hits.hits}}", &json!({"payload": p})), "[x][y]");
        let mut q = json!({"aggregations": {"by": {"buckets": [{"key": "k", "doc_count": 2}]}}});
        apply_extract(&mut q, &json!({"extract": ["aggregations.by"]}));
        assert_eq!(q["aggregations"]["by"]["buckets"][0]["key"], "k");
        let mut r = json!({"a": 1});
        apply_extract(&mut r, &json!({"extract": ["missing"]}));
        assert_eq!(r, json!({}));
    }

    #[test]
    fn compare_and_period() {
        let ctx = Ctx { id: "i".into(), watch_id: "w".into(), execution_time: 1_000_000, triggered: iso(0), scheduled: iso(0), metadata: json!({}), payload: json!({"hits": {"total": 7}, "arr": [{"v": 1}, {"v": 5}]}), vars: json!({}), data: Default::default(), identity: None };
        assert_eq!(run_condition(&json!({"compare": {"ctx.payload.hits.total": {"gte": 5}}}), &ctx)["met"], true);
        assert_eq!(run_condition(&json!({"compare": {"ctx.payload.hits.total": {"lt": 5}}}), &ctx)["met"], false);
        assert_eq!(run_condition(&json!({"array_compare": {"ctx.payload.arr": {"path": "v", "gte": {"value": 5, "quantifier": "some"}}}}), &ctx)["met"], true);
        assert_eq!(run_condition(&json!({"array_compare": {"ctx.payload.arr": {"path": "v", "gte": {"value": 5, "quantifier": "all"}}}}), &ctx)["met"], false);
        assert_eq!(fmt_period(5000), "5 seconds");
        assert_eq!(fmt_period(3020), "3 seconds and 20 milliseconds");
        assert_eq!(fmt_period(3_723_000), "1 hour, 2 minutes and 3 seconds");
    }
}
