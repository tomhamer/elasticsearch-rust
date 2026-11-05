//! Watch definition parsing / normalization and watch status helpers.

use super::normalize;
use super::schedule::Schedule;
use super::{iso, CachedWatch};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub const INPUTS: &[&str] = &["none", "simple", "search", "http", "chain", "transform"];
pub const CONDITIONS: &[&str] = &["always", "never", "compare", "array_compare", "script"];
pub const TRANSFORMS: &[&str] = &["search", "script", "chain"];
pub const ACTION_TYPES: &[&str] = &["logging", "index", "webhook", "email", "slack", "pagerduty", "jira"];

pub fn parse_exc(msg: impl Into<String>) -> EsError {
    EsError::new(400, "parse_exception", msg)
}

fn single_key<'a>(v: &'a Value, what: &str, watch: &str, known: &[&str]) -> Result<(&'a String, &'a Value)> {
    let o = v.as_object().ok_or_else(|| parse_exc(format!("could not parse {what} for watch [{watch}]. expected an object")))?;
    let (k, val) = o.iter().next().ok_or_else(|| parse_exc(format!("could not parse {what} for watch [{watch}]. expected a {what} type field, but no fields were found")))?;
    if o.len() > 1 {
        return Err(parse_exc(format!("could not parse {what} for watch [{watch}]. expected a single {what} type, but found [{}]", o.keys().cloned().collect::<Vec<_>>().join(", "))));
    }
    if !known.contains(&k.as_str()) {
        if what == "trigger" {
            return Err(parse_exc(format!("could not parse trigger [{k}] for [{watch}]. unknown trigger type [{k}]")));
        }
        return Err(parse_exc(format!("could not parse {what} for watch [{watch}]. unknown {what} type [{k}]")));
    }
    Ok((k, val))
}

/// Initial `status` of a fresh watch.
pub fn new_status(actions: &Map<String, Value>, now: i64, active: bool) -> Value {
    let ts = iso(now);
    let mut acts = Map::new();
    for id in actions.keys() {
        acts.insert(id.clone(), json!({"ack": {"timestamp": ts, "state": "awaits_successful_execution"}}));
    }
    json!({"state": {"active": active, "timestamp": ts}, "actions": acts, "version": -1})
}

fn validate_transform(t: &Value, what: &str, watch: &str) -> Result<()> {
    let (k, v) = single_key(t, what, watch, TRANSFORMS)?;
    if k == "chain" {
        for sub in v.as_array().ok_or_else(|| parse_exc(format!("could not parse transform for watch [{watch}]. chain transform expects an array")))? {
            validate_transform(sub, what, watch)?;
        }
    }
    Ok(())
}

/// `HttpRequestTemplate.Builder.fromUrl` validation of a request's `url`.
fn check_url(req: &Value) -> Result<()> {
    let Some(u) = req.get("url") else { return Ok(()) };
    let Some(u) = u.as_str() else { return Ok(()) };
    if u.is_empty() {
        return Err(parse_exc("Configured URL is empty, please configure a valid URL"));
    }
    match u.split_once(':') {
        Some((scheme, rest)) if !scheme.is_empty() && scheme.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) => {
            if let Some(auth) = rest.strip_prefix("//") {
                let host = auth.split(['/', '?', '#']).next().unwrap_or("");
                if host.is_empty() && auth.len() == host.len() {
                    return Err(parse_exc(format!("Malformed URL [{u}]")));
                }
            }
            Ok(())
        }
        _ => Err(parse_exc(format!("URL [{u}] does not contain a scheme"))),
    }
}

/// URL checks of every http request template of a watch (http input,
/// webhook actions, email http attachments).
fn check_urls(input: Option<&Value>, actions: &Map<String, Value>) -> Result<()> {
    fn input_urls(i: &Value) -> Result<()> {
        if let Some(r) = i.get("http").and_then(|h| h.get("request")) {
            check_url(r)?;
        }
        if let Some(a) = i.get("chain").and_then(|c| c.get("inputs")).and_then(Value::as_array) {
            for sub in a {
                for x in sub.as_object().into_iter().flat_map(|o| o.values()) {
                    input_urls(x)?;
                }
            }
        }
        Ok(())
    }
    if let Some(i) = input {
        input_urls(i)?;
    }
    for def in actions.values() {
        if let Some(w) = def.get("webhook") {
            check_url(w)?;
        }
        if let Some(Value::Object(at)) = def.get("email").and_then(|e| e.get("attachments")) {
            for a in at.values() {
                if let Some(r) = a.get("http").and_then(|h| h.get("request")) {
                    check_url(r)?;
                }
            }
        }
    }
    Ok(())
}

/// Parse a `PUT _watcher/watch/{id}` body into the stored document source.
pub fn parse_watch(id: &str, body: &Value, now: i64, active: bool) -> Result<Value> {
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::action_validation(&["watch source is missing"])),
        _ => return Err(parse_exc(format!("could not parse watch [{id}]. expected an object"))),
    };
    for k in o.keys() {
        if !matches!(k.as_str(), "trigger" | "input" | "condition" | "transform" | "actions" | "metadata" | "throttle_period" | "throttle_period_in_millis" | "status" | "headers") {
            return Err(parse_exc(format!("could not parse watch [{id}]. unexpected field [{k}]")));
        }
    }
    let trigger = o.get("trigger").ok_or_else(|| parse_exc(format!("could not parse watch [{id}]. missing required field [trigger]")))?;
    let (tk, tv) = single_key(trigger, "trigger", id, &["schedule"])?;
    let _ = tk;
    Schedule::parse(tv).map_err(|e| {
        if e.kind == "illegal_argument_exception" {
            EsError::new(400, "parse_exception", format!("could not parse schedule: {}", e.reason)).caused_by(e)
        } else {
            e
        }
    })?;
    let input = match o.get("input") {
        None => json!({"none": {}}),
        Some(i) => {
            let (k, v) = single_key(i, "input", id, INPUTS)?;
            if k == "chain" {
                let inputs = v.get("inputs").and_then(Value::as_array).ok_or_else(|| parse_exc(format!("could not parse [chain] input for watch [{id}]. expected an array of [inputs]")))?;
                for sub in inputs {
                    let so = sub.as_object().filter(|m| m.len() == 1).ok_or_else(|| parse_exc(format!("could not parse [chain] input for watch [{id}]. expected a named input")))?;
                    let (sk, sv) = single_key(so.values().next().unwrap(), "input", id, INPUTS)?;
                    if sk == "transform" {
                        validate_transform(sv, "transform", id)?;
                    }
                }
            }
            if k == "transform" {
                validate_transform(v, "transform", id)?;
            }
            normalize::input(i)
        }
    };
    let condition = match o.get("condition") {
        None => json!({"always": {}}),
        Some(c) => {
            single_key(c, "condition", id, CONDITIONS)?;
            normalize::condition(c)
        }
    };
    if let Some(t) = o.get("transform") {
        validate_transform(t, "transform", id)?;
    }
    let mut actions = Map::new();
    if let Some(a) = o.get("actions") {
        let ao = a.as_object().ok_or_else(|| parse_exc(format!("could not parse watch [{id}]. [actions] must be an object")))?;
        for (aid, def) in ao {
            actions.insert(aid.clone(), normalize::action(&parse_action(id, aid, def)?));
        }
    }
    check_urls(o.get("input"), &actions)?;
    let actions = normalize::java_ordered(&actions);
    let mut out = Map::new();
    out.insert("trigger".into(), trigger.clone());
    out.insert("input".into(), input);
    out.insert("condition".into(), condition);
    if let Some(t) = o.get("transform") {
        out.insert("transform".into(), normalize::transform(t));
    }
    out.insert("actions".into(), Value::Object(actions.clone()));
    if let Some(m) = o.get("metadata") {
        if !m.is_object() {
            return Err(parse_exc(format!("could not parse watch [{id}]. [metadata] must be an object")));
        }
        out.insert("metadata".into(), m.clone());
    }
    if let Some(ms) = throttle_of(o, &format!("watch [{id}]"))? {
        out.insert("throttle_period_in_millis".into(), json!(ms));
    }
    out.insert("status".into(), new_status(&actions, now, active));
    Ok(Value::Object(out))
}

fn throttle_of(o: &Map<String, Value>, what: &str) -> Result<Option<i64>> {
    if let Some(v) = o.get("throttle_period_in_millis") {
        return v.as_i64().map(Some).ok_or_else(|| parse_exc(format!("could not parse {what}. [throttle_period_in_millis] must be a number")));
    }
    match o.get("throttle_period") {
        None => Ok(None),
        Some(Value::String(s)) => crate::ilm::parse_time_millis(s)
            .map(Some)
            .ok_or_else(|| parse_exc(format!("failed to parse time value [{s}] for [throttle_period]"))),
        Some(Value::Number(n)) => Ok(n.as_i64()),
        Some(_) => Err(parse_exc(format!("could not parse {what}. [throttle_period] must be a time value"))),
    }
}

fn parse_action(watch: &str, aid: &str, def: &Value) -> Result<Value> {
    let o = def.as_object().ok_or_else(|| parse_exc(format!("could not parse action [{aid}] for watch [{watch}]. expected an object")))?;
    let mut out = Map::new();
    let mut types = 0;
    for (k, v) in o {
        match k.as_str() {
            "throttle_period" | "throttle_period_in_millis" => {}
            "condition" => {
                single_key(v, "condition", watch, CONDITIONS)?;
                out.insert(k.clone(), v.clone());
            }
            "transform" => {
                validate_transform(v, "transform", watch)?;
                out.insert(k.clone(), v.clone());
            }
            "foreach" | "max_iterations" => {
                out.insert(k.clone(), v.clone());
            }
            t if ACTION_TYPES.contains(&t) => {
                types += 1;
                if !v.is_object() {
                    return Err(parse_exc(format!("could not parse action [{aid}] for watch [{watch}]. expected an object for [{t}]")));
                }
                out.insert(k.clone(), v.clone());
            }
            other => return Err(parse_exc(format!("could not parse action [{aid}] for watch [{watch}]. unknown action type [{other}]", ).replace(&format!("action [{aid}] for watch [{watch}]"), &format!("action [{watch}/{aid}]")))),
        }
    }
    if types == 0 {
        return Err(parse_exc(format!("could not parse action [{watch}/{aid}]. missing action type")));
    }
    if types > 1 {
        return Err(parse_exc(format!("could not parse action [{aid}] for watch [{watch}]. multiple action types")));
    }
    if let Some(ms) = throttle_of(o, &format!("action [{aid}] for watch [{watch}]"))? {
        out.insert("throttle_period_in_millis".into(), json!(ms));
    }
    Ok(Value::Object(out))
}

pub fn build_cached(id: &str, src: Value, version: u64, seq_no: i64, primary_term: u64) -> Result<CachedWatch> {
    let sched = Schedule::parse(&src["trigger"]["schedule"])?;
    Ok(CachedWatch { id: id.to_string(), src, schedule: Arc::new(sched), version, seq_no, primary_term, next_fire: None })
}

/// Action type of an action definition.
pub fn action_type(def: &Value) -> &'static str {
    ACTION_TYPES.iter().copied().find(|t| def.get(*t).is_some()).unwrap_or("unknown")
}

/// `WatcherXContentParser.REDACTED_PASSWORD`.
pub const REDACTED: &str = "::es_redacted::";

/// Hide secrets in a watch definition as `toXContent(HIDE_SECRETS)` does:
/// basic-auth passwords, `Authorization` headers of http request templates,
/// the email action `password`.
pub fn redact(v: &Value) -> Value {
    fn walk(v: &mut Value, parent: &str) {
        match v {
            Value::Object(o) => {
                if parent == "basic" && o.contains_key("password") {
                    o.insert("password".into(), json!(REDACTED));
                }
                if parent == "email" && o.get("password").map(Value::is_string).unwrap_or(false) {
                    o.insert("password".into(), json!(REDACTED));
                }
                if parent == "headers" {
                    let key = ["Authorization", "authorization"].into_iter().find(|k| o.contains_key(*k));
                    if let Some(k) = key {
                        o.insert(k.into(), json!(REDACTED));
                    }
                    return;
                }
                for (k, x) in o.iter_mut() {
                    if !matches!(k.as_str(), "metadata" | "simple" | "body" | "payload") {
                        walk(x, k);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| walk(x, parent)),
            _ => {}
        }
    }
    let mut out = v.clone();
    walk(&mut out, "");
    out
}

/// Replace redacted secrets in `new` by the values stored at the same place
/// in `old` (an update keeps the stored secret); a redacted secret with no
/// stored counterpart is dropped.
pub fn restore_redacted(new: &mut Value, old: Option<&Value>) {
    if let Value::Object(o) = new {
        let keys: Vec<String> = o.keys().cloned().collect();
        for k in keys {
            let prev = old.and_then(|x| x.get(&k));
            let x = o.get_mut(&k).unwrap();
            if k == "password" && x.as_str() == Some(REDACTED) {
                match prev {
                    Some(p) if p.as_str() != Some(REDACTED) => *x = p.clone(),
                    _ => {
                        o.remove(&k);
                    }
                }
            } else {
                restore_redacted(x, prev);
            }
        }
    } else if let Value::Array(a) = new {
        for (i, x) in a.iter_mut().enumerate() {
            restore_redacted(x, old.and_then(|o| o.get(i)));
        }
    }
}

/// Field name of the first redacted secret in a watch body, if any.
pub fn find_redacted(v: &Value) -> Option<String> {
    match v {
        Value::Object(o) => o.iter().find_map(|(k, x)| if k == "password" && x.as_str() == Some(REDACTED) { Some(k.clone()) } else { find_redacted(x) }),
        Value::Array(a) => a.iter().find_map(find_redacted),
        _ => None,
    }
}

pub const IGNORE_THROTTLED_WARNING: &str = "Deprecated field [ignore_throttled] used, this field is unused and will be removed entirely";

/// Deprecation warnings parsing watch source `v` emits (`ignore_throttled`
/// inside any search request's `indices_options`).
pub fn parse_warnings(v: &Value) -> Vec<String> {
    fn walk(v: &Value, found: &mut bool) {
        match v {
            Value::Object(o) => {
                for (k, x) in o {
                    if k == "indices_options" && x.get("ignore_throttled").is_some() {
                        *found = true;
                    } else if !matches!(k.as_str(), "body" | "metadata" | "payload" | "simple" | "status") {
                        walk(x, found);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, found)),
            _ => {}
        }
    }
    let mut found = false;
    walk(v, &mut found);
    if found { vec![IGNORE_THROTTLED_WARNING.to_string()] } else { Vec::new() }
}

/// The watch as GET returns it: the stored source without `status`.
pub fn watch_view(src: &Value) -> Value {
    let mut m = src.as_object().cloned().unwrap_or_default();
    m.remove("status");
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let b = json!({"trigger": {"schedule": {"interval": "1h"}}, "actions": {"a": {"logging": {"text": "x"}, "throttle_period": "10s"}}});
        let s = parse_watch("w", &b, 0, true).unwrap();
        assert_eq!(s["input"], json!({"none": {}}));
        assert_eq!(s["actions"]["a"]["throttle_period_in_millis"], 10000);
        assert_eq!(s["status"]["actions"]["a"]["ack"]["state"], "awaits_successful_execution");
        assert!(parse_watch("w", &json!({}), 0, true).unwrap_err().reason.contains("missing required field [trigger]"));
        assert!(parse_watch("w", &json!({"trigger": {"schedule": {"interval": "1h"}}, "input": {"foo": {}}}), 0, true).is_err());
    }
}
