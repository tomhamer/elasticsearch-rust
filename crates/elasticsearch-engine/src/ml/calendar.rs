//! ML calendars, scheduled events and filters.

use super::store::{service, State};
use super::xc::bad_request;
use crate::error::{EsError, Result};
use crate::node::Node;
use serde_json::{json, Map, Value};

fn no_calendar(id: &str) -> EsError {
    EsError::resource_not_found(format!("No calendar with id [{id}]"))
}

fn validation(errors: &[String]) -> EsError {
    let refs: Vec<&str> = errors.iter().map(String::as_str).collect();
    EsError::action_validation(&refs)
}

fn invalid_id(kind: &str, id: &str) -> String {
    format!("Invalid {kind}; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric")
}

fn strings(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        _ => vec![],
    }
}

fn render_calendar(c: &Value) -> Value {
    let mut o = Map::new();
    o.insert("calendar_id".into(), c["calendar_id"].clone());
    o.insert("job_ids".into(), c.get("job_ids").cloned().unwrap_or(json!([])));
    if let Some(d) = c.get("description").filter(|d| !d.is_null()) {
        o.insert("description".into(), d.clone());
    }
    Value::Object(o)
}

/// `PUT _ml/calendars/{id}`.
pub fn put_calendar(node: &Node, id: &str, body: &Value) -> Result<Value> {
    let mut errs = Vec::new();
    if id == "_all" {
        errs.push("Cannot create a Calendar with the reserved name [_all]".to_string());
    }
    if !super::is_valid_id(id) {
        errs.push(invalid_id("calendar_id", id));
    } else if id.len() > super::ID_LENGTH_LIMIT {
        errs.push(format!("The calendar_id cannot contain more than {} characters.", super::ID_LENGTH_LIMIT));
    }
    if !errs.is_empty() {
        return Err(validation(&errs));
    }
    let o = body.as_object().cloned().unwrap_or_default();
    for k in o.keys() {
        if !matches!(k.as_str(), "calendar_id" | "job_ids" | "description") {
            return Err(super::xc::unknown((1, 2), "calendar", k, &["calendar_id", "job_ids", "description"]));
        }
    }
    if let Some(b) = o.get("calendar_id").and_then(Value::as_str) {
        if b != id {
            return Err(EsError::illegal_argument(format!("Inconsistent calendar_id; '{b}' specified in the body differs from '{id}' specified as a URL argument")));
        }
    }
    let job_ids = o.get("job_ids").map(strings).unwrap_or_default();
    let mut c = Map::new();
    c.insert("calendar_id".into(), json!(id));
    c.insert("job_ids".into(), json!(job_ids));
    if let Some(d) = o.get("description").filter(|d| !d.is_null()) {
        c.insert("description".into(), d.clone());
    }
    let c = Value::Object(c);
    service(node).mutate(|st| {
        if st.calendars.contains_key(id) {
            return Err(bad_request(format!("Cannot create calendar with id [{id}] as it already exists")));
        }
        st.calendars.insert(id.to_string(), c.clone());
        Ok(())
    })?;
    Ok(render_calendar(&c))
}

fn expand_calendars(st: &State, expr: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for tok in expr.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let wild = tok == "_all" || tok.contains('*');
        let m: Vec<String> = st.calendars.keys().filter(|id| if wild { tok == "_all" || crate::mapping::wildcard_match(tok, id) } else { *id == tok }).cloned().collect();
        if m.is_empty() && !wild {
            return Err(no_calendar(tok));
        }
        out.extend(m);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// `GET _ml/calendars[/{id}]`.
pub fn get_calendars(node: &Node, id: Option<&str>, from: Option<i64>, size: Option<i64>) -> Result<Value> {
    let concrete = id.filter(|i| !i.is_empty() && *i != "_all" && *i != "*");
    if concrete.is_some() && (from.is_some() || size.is_some()) {
        return Err(validation(&["Params [from, size] are incompatible with [calendar_id].".to_string()]));
    }
    let st = service(node).read();
    let ids = expand_calendars(&st, id.unwrap_or("_all"))?;
    let total = ids.len();
    let from = from.unwrap_or(0).max(0) as usize;
    let size = size.unwrap_or(100).max(0) as usize;
    let cals: Vec<Value> = ids.iter().skip(from).take(size).map(|i| render_calendar(&st.calendars[i])).collect();
    Ok(json!({"count": total, "calendars": cals}))
}

/// `DELETE _ml/calendars/{id}`.
pub fn delete_calendar(node: &Node, id: &str) -> Result<Value> {
    service(node).mutate(|st| {
        if st.calendars.remove(id).is_none() {
            return Err(no_calendar(id));
        }
        st.events.retain(|e| e["calendar_id"].as_str() != Some(id));
        Ok(())
    })?;
    Ok(json!({"acknowledged": true}))
}

/// `PUT _ml/calendars/{id}/jobs/{job_ids}`.
pub fn put_calendar_job(node: &Node, id: &str, jobs: &str) -> Result<Value> {
    service(node).mutate(|st| {
        let c = st.calendars.get_mut(id).ok_or_else(|| no_calendar(id))?;
        let mut ids = strings(&c["job_ids"]);
        for j in jobs.split(',').map(str::trim).filter(|j| !j.is_empty()) {
            if !ids.iter().any(|x| x == j) {
                ids.push(j.to_string());
            }
        }
        c["job_ids"] = json!(ids);
        Ok(render_calendar(c))
    })
}

/// `DELETE _ml/calendars/{id}/jobs/{job_ids}`.
pub fn delete_calendar_job(node: &Node, id: &str, jobs: &str) -> Result<Value> {
    service(node).mutate(|st| {
        let c = st.calendars.get_mut(id).ok_or_else(|| no_calendar(id))?;
        let mut ids = strings(&c["job_ids"]);
        for j in jobs.split(',').map(str::trim).filter(|j| !j.is_empty()) {
            let before = ids.len();
            ids.retain(|x| x != j);
            if ids.len() == before {
                return Err(bad_request(format!("Cannot remove [{j}] as it is not present in calendar [{id}]")));
            }
        }
        c["job_ids"] = json!(ids);
        Ok(render_calendar(c))
    })
}

/// `Instant.toString()` of epoch millis (`2017-12-03T00:00:00Z`).
fn instant(ms: i64) -> String {
    let d = crate::date::from_millis(ms);
    if ms % 1000 == 0 {
        d.format("%Y-%m-%dT%H:%M:%SZ").to_string()
    } else {
        d.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
    }
}

fn parse_event_time(v: &Value, field: &str) -> Result<i64> {
    match v {
        Value::Number(n) => n.as_i64().ok_or_else(|| EsError::illegal_argument(format!("[{field}] must be a long"))),
        Value::String(s) => {
            if let Ok(n) = s.parse::<i64>() {
                return Ok(n);
            }
            crate::date::DateFormatter::new("strict_date_optional_time")
                .and_then(|f| f.parse(s))
                .map_err(|_| EsError::illegal_argument(format!("unexpected value [{s}] for [{field}]")))
        }
        _ => Err(EsError::illegal_argument(format!("unexpected token for [{field}]"))),
    }
}

/// `POST _ml/calendars/{id}/events`.
pub fn post_events(node: &Node, id: &str, body: &Value) -> Result<Value> {
    let events = body.get("events").and_then(Value::as_array).cloned().unwrap_or_default();
    if events.is_empty() {
        return Err(bad_request("At least 1 event is required"));
    }
    let mut parsed = Vec::new();
    for e in &events {
        let o = e.as_object().cloned().unwrap_or_default();
        let desc = o.get("description").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [description, start_time, end_time]"))?;
        let start = parse_event_time(o.get("start_time").ok_or_else(|| EsError::illegal_argument("Required [start_time]"))?, "start_time")?;
        let end = parse_event_time(o.get("end_time").ok_or_else(|| EsError::illegal_argument("Required [end_time]"))?, "end_time")?;
        if let Some(c) = o.get("calendar_id").and_then(Value::as_str) {
            if c != id {
                return Err(bad_request(format!("Inconsistent calendar_id; '{c}' specified in the body differs from '{id}' specified as a URL argument")));
            }
        }
        if start >= end {
            return Err(bad_request(format!("Event start time [{}] must come before end time [{}]", instant(start), instant(end))));
        }
        let mut m = Map::new();
        m.insert("description".into(), json!(desc));
        m.insert("start_time".into(), json!(start));
        m.insert("end_time".into(), json!(end));
        m.insert("skip_result".into(), o.get("skip_result").cloned().unwrap_or(json!(true)));
        m.insert("skip_model_update".into(), o.get("skip_model_update").cloned().unwrap_or(json!(true)));
        if let Some(f) = o.get("force_time_shift") {
            m.insert("force_time_shift".into(), f.clone());
        }
        m.insert("calendar_id".into(), json!(id));
        parsed.push(Value::Object(m));
    }
    let out = parsed.clone();
    service(node).mutate(|st| {
        if !st.calendars.contains_key(id) {
            return Err(no_calendar(id));
        }
        for mut e in parsed {
            e["event_id"] = json!(crate::index::new_uuid().chars().take(20).collect::<String>());
            st.events.push(e);
        }
        Ok(())
    })?;
    Ok(json!({"events": out}))
}

/// `GET _ml/calendars/{id}/events`.
pub fn get_events(node: &Node, id: &str, job_id: Option<&str>, start: Option<&str>, end: Option<&str>, from: Option<i64>, size: Option<i64>) -> Result<Value> {
    let any = matches!(id, "_all" | "*");
    if job_id.is_some() && !any {
        return Err(validation(&["If job_id is used calendar_id must be '_all' or '*'".to_string()]));
    }
    let st = service(node).read();
    let cals: Vec<String> = if any { st.calendars.keys().cloned().collect() } else { expand_calendars(&st, id)? };
    let cals: Vec<String> = match job_id {
        Some(j) => {
            let mut names = vec![j.to_string()];
            if let Some(job) = st.jobs.get(j) {
                names.extend(job["groups"].as_array().into_iter().flatten().filter_map(|g| g.as_str().map(str::to_string)));
            }
            cals.into_iter().filter(|c| strings(&st.calendars[c]["job_ids"]).iter().any(|x| names.contains(x))).collect()
        }
        None => cals,
    };
    let start = start.map(|s| super::anomaly::parse_datafeed_time(s, "start")).transpose()?;
    let end = end.map(|s| super::anomaly::parse_datafeed_time(s, "end")).transpose()?;
    let mut evs: Vec<Value> = st
        .events
        .iter()
        .filter(|e| e["calendar_id"].as_str().map(|c| cals.iter().any(|x| x == c)).unwrap_or(false))
        .filter(|e| start.map(|s| e["end_time"].as_i64().unwrap_or(0) > s).unwrap_or(true))
        .filter(|e| end.map(|x| e["start_time"].as_i64().unwrap_or(0) < x).unwrap_or(true))
        .cloned()
        .collect();
    evs.sort_by_key(|e| e["start_time"].as_i64().unwrap_or(0));
    let total = evs.len();
    let from = from.unwrap_or(0).max(0) as usize;
    let size = size.unwrap_or(100).max(0) as usize;
    let page: Vec<Value> = evs.into_iter().skip(from).take(size).collect();
    Ok(json!({"count": total, "events": page}))
}

/// `DELETE _ml/calendars/{id}/events/{event_id}`.
pub fn delete_event(node: &Node, id: &str, event_id: &str) -> Result<Value> {
    service(node).mutate(|st| {
        if !st.calendars.contains_key(id) {
            return Err(no_calendar(id));
        }
        let pos = st.events.iter().position(|e| e["event_id"].as_str() == Some(event_id)).ok_or_else(|| EsError::resource_not_found(format!("No event with id [{event_id}]")))?;
        let cal = st.events[pos]["calendar_id"].as_str().unwrap_or("").to_string();
        if cal != id {
            return Err(bad_request(format!("Event [{event_id}] has calendar_id [{cal}] which does not match the calendar_id [{id}] specified in the request")));
        }
        st.events.remove(pos);
        Ok(())
    })?;
    Ok(json!({"acknowledged": true}))
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

const MAX_ITEMS: usize = 10_000;

fn render_filter(f: &Value) -> Value {
    let mut o = Map::new();
    o.insert("filter_id".into(), f["filter_id"].clone());
    if let Some(d) = f.get("description").filter(|d| !d.is_null()) {
        o.insert("description".into(), d.clone());
    }
    o.insert("items".into(), f.get("items").cloned().unwrap_or(json!([])));
    Value::Object(o)
}

fn sorted_items(items: Vec<String>) -> Vec<String> {
    let mut v = items;
    v.sort();
    v.dedup();
    v
}

/// `PUT _ml/filters/{id}`.
pub fn put_filter(node: &Node, id: &str, body: &Value) -> Result<Value> {
    let o = body.as_object().cloned().unwrap_or_default();
    for k in o.keys() {
        if !matches!(k.as_str(), "filter_id" | "description" | "items") {
            return Err(super::xc::unknown((1, 2), "filter", k, &["filter_id", "description", "items"]));
        }
    }
    if let Some(b) = o.get("filter_id").and_then(Value::as_str) {
        if b != id {
            return Err(EsError::illegal_argument(format!("Inconsistent filter_id; '{b}' specified in the body differs from '{id}' specified as a URL argument")));
        }
    }
    if !super::is_valid_id(id) {
        return Err(bad_request(invalid_id("filter_id", id)));
    }
    let items = sorted_items(o.get("items").map(strings).unwrap_or_default());
    if items.len() > MAX_ITEMS {
        return Err(bad_request(format!("Filter [{id}] contains too many items; up to [{MAX_ITEMS}] items are allowed")));
    }
    let mut f = Map::new();
    f.insert("filter_id".into(), json!(id));
    if let Some(d) = o.get("description").filter(|d| !d.is_null()) {
        f.insert("description".into(), d.clone());
    }
    f.insert("items".into(), json!(items));
    let f = Value::Object(f);
    service(node).mutate(|st| {
        if st.filters.contains_key(id) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("A filter with id [{id}] already exists")));
        }
        st.filters.insert(id.to_string(), f.clone());
        Ok(())
    })?;
    Ok(render_filter(&f))
}

/// `GET _ml/filters[/{id}]`.
pub fn get_filters(node: &Node, id: Option<&str>, from: Option<i64>, size: Option<i64>) -> Result<Value> {
    let st = service(node).read();
    let mut ids: Vec<String> = Vec::new();
    for tok in id.unwrap_or("_all").split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let wild = tok == "_all" || tok.contains('*');
        let m: Vec<String> = st.filters.keys().filter(|f| if wild { tok == "_all" || crate::mapping::wildcard_match(tok, f) } else { *f == tok }).cloned().collect();
        if m.is_empty() && !wild {
            return Err(EsError::resource_not_found(format!("Unable to find filter [{tok}]")));
        }
        ids.extend(m);
    }
    ids.sort();
    ids.dedup();
    let total = ids.len();
    let from = from.unwrap_or(0).max(0) as usize;
    let size = size.unwrap_or(100).max(0) as usize;
    let filters: Vec<Value> = ids.iter().skip(from).take(size).map(|i| render_filter(&st.filters[i])).collect();
    Ok(json!({"count": total, "filters": filters}))
}

/// `POST _ml/filters/{id}/_update`.
pub fn update_filter(node: &Node, id: &str, body: &Value) -> Result<Value> {
    let o = body.as_object().cloned().unwrap_or_default();
    service(node).mutate(|st| {
        let f = st.filters.get_mut(id).ok_or_else(|| EsError::resource_not_found(format!("No filter with id [{id}] exists")))?;
        let mut items = strings(&f["items"]);
        for a in o.get("add_items").map(strings).unwrap_or_default() {
            items.push(a);
        }
        for r in o.get("remove_items").map(strings).unwrap_or_default() {
            let before = items.len();
            items.retain(|x| *x != r);
            if items.len() == before {
                return Err(bad_request(format!("Cannot remove item [{r}] as it is not present in filter [{id}]")));
            }
        }
        let items = sorted_items(items);
        if items.len() > MAX_ITEMS {
            return Err(bad_request(format!("Filter [{id}] contains too many items; up to [{MAX_ITEMS}] items are allowed")));
        }
        let mut nf = Map::new();
        nf.insert("filter_id".into(), json!(id));
        let desc = o.get("description").cloned().or_else(|| f.get("description").cloned());
        if let Some(d) = desc.filter(|d| !d.is_null()) {
            nf.insert("description".into(), d);
        }
        nf.insert("items".into(), json!(items));
        *f = Value::Object(nf);
        Ok(render_filter(f))
    })
}

/// `DELETE _ml/filters/{id}`.
pub fn delete_filter(node: &Node, id: &str) -> Result<Value> {
    service(node).mutate(|st| {
        if !st.filters.contains_key(id) {
            return Err(EsError::resource_not_found(format!("Could not delete filter with ID [{id}] because it does not exist")));
        }
        let users: Vec<String> = st.jobs.iter().filter(|(_, j)| super::job::referenced_filters(j).iter().any(|f| f == id)).map(|(k, _)| k.clone()).collect();
        if !users.is_empty() {
            return Err(EsError::new(409, "status_exception", format!("Cannot delete filter [{id}] currently used by jobs [{}]", users.join(", "))));
        }
        st.filters.remove(id);
        Ok(())
    })?;
    Ok(json!({"acknowledged": true}))
}
