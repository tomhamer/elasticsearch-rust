//! REST-level watcher operations (the HTTP layer only routes and formats).

use super::exec::{self, ExecOptions};
use super::model;
use super::{iso, put_doc, service, watches_index, CachedWatch, WatcherService};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts, NODE_ID};
use crate::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn not_exist(id: &str) -> EsError {
    EsError::resource_not_found(format!("Watch with id [{id}] does not exist"))
}

/// Canonical key order of a watch status (state, last_checked,
/// last_met_condition, actions, execution_state, version).
pub fn canon_status(s: &Value) -> Value {
    let mut m = Map::new();
    for k in ["state", "last_checked", "last_met_condition"] {
        if let Some(v) = s.get(k) {
            m.insert(k.into(), v.clone());
        }
    }
    if let Some(Value::Object(acts)) = s.get("actions") {
        let mut am = Map::new();
        for (id, a) in acts {
            let mut one = Map::new();
            for k in ["ack", "last_execution", "last_successful_execution", "last_throttle"] {
                if let Some(v) = a.get(k) {
                    one.insert(k.into(), v.clone());
                }
            }
            am.insert(id.clone(), Value::Object(one));
        }
        m.insert("actions".into(), Value::Object(am));
    } else {
        m.insert("actions".into(), json!({}));
    }
    for k in ["execution_state", "version"] {
        if let Some(v) = s.get(k) {
            m.insert(k.into(), v.clone());
        }
    }
    Value::Object(m)
}

fn validate_id(id: &str) -> Result<()> {
    if id.is_empty() || id.contains(' ') || id.contains('/') {
        return Err(EsError::action_validation(&["watch id contains whitespace"]));
    }
    Ok(())
}

fn compile_scripts_in(v: &Value, context: crate::script::ScriptContext) -> Result<()> {
    if let Some(sc) = v.get("script") {
        let spec = crate::script::parse_spec(sc)?;
        if let Some(src) = spec.source {
            crate::script::compile(&src, &spec.params, context)?;
        }
    }
    if let Some(chain) = v.get("chain").and_then(Value::as_array) {
        for c in chain {
            compile_scripts_in(c, context)?;
        }
    }
    Ok(())
}

/// Compile every script of the watch (compile errors fail the PUT / inline execute).
pub fn validate_scripts(src: &Value) -> Result<()> {
    fn input_scripts(i: &Value) -> Result<()> {
        if let Some(t) = i.get("transform") {
            compile_scripts_in(t, crate::script::ScriptContext::Watcher)?;
        }
        for sub in i.get("chain").and_then(|c| c.get("inputs")).and_then(Value::as_array).into_iter().flatten() {
            for x in sub.as_object().into_iter().flat_map(|o| o.values()) {
                input_scripts(x)?;
            }
        }
        Ok(())
    }
    input_scripts(&src["input"])?;
    compile_scripts_in(&src["condition"], crate::script::ScriptContext::WatcherCondition)?;
    if let Some(t) = src.get("transform") {
        compile_scripts_in(t, crate::script::ScriptContext::Watcher)?;
    }
    if let Some(acts) = src["actions"].as_object() {
        for a in acts.values() {
            if let Some(c) = a.get("condition") {
                compile_scripts_in(c, crate::script::ScriptContext::WatcherCondition)?;
            }
            if let Some(t) = a.get("transform") {
                compile_scripts_in(t, crate::script::ScriptContext::Watcher)?;
            }
        }
    }
    Ok(())
}

fn validate_accounts(node: &Node, src: &Value) -> Result<()> {
    let has_account = |kind: &str| -> bool {
        let prefix = format!("xpack.notification.{kind}.account.");
        let cs = node.cluster_settings.read();
        [&cs.0, &cs.1].iter().any(|m| {
            m.keys().any(|k| k.starts_with(&prefix))
                || m.get("xpack").and_then(|x| x.get("notification")).and_then(|x| x.get(kind)).and_then(|x| x.get("account")).map(|a| a.as_object().map(|o| !o.is_empty()).unwrap_or(false)).unwrap_or(false)
        })
    };
    if let Some(acts) = src["actions"].as_object() {
        for def in acts.values() {
            let ty = model::action_type(def);
            if matches!(ty, "slack" | "pagerduty" | "jira") && !has_account(ty) {
                return Err(EsError::illegal_argument(format!(
                    "no accounts of type [{ty}] configured. Please set up an account using the [xpack.notification.{ty}] settings"
                )));
            }
        }
    }
    Ok(())
}

pub struct PutParams {
    pub active: bool,
    pub version: Option<i64>,
    pub if_seq_no: Option<i64>,
    pub if_primary_term: Option<u64>,
    /// Security: the (effective) user storing the watch; scheduled
    /// executions run as this user.
    pub user: Option<String>,
}

/// `PUT _watcher/watch/{id}`.
pub fn put_watch(node: &Node, id: &str, body: &Value, p: &PutParams) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    validate_id(id)?;
    if p.version.is_some() {
        return Err(EsError::new(500, "unsupported_operation_exception", "update requests do not support versioning"));
    }
    let now = crate::date::now_millis();
    let mut body = body.clone();
    if let Some(field) = model::find_redacted(&body) {
        if p.if_seq_no.is_none() {
            return Err(model::parse_exc(format!("found redacted password in field [{field}]")));
        }
        let stored = svc.cache.read().get(id).map(|w| w.src.clone());
        model::restore_redacted(&mut body, stored.as_ref());
    }
    let mut src = model::parse_watch(id, &body, now, p.active)?;
    if let Some(u) = p.user.as_ref().filter(|_| node.security.enabled()) {
        src["status"]["headers"] = json!({"_xpack_security_authentication": u});
    }
    validate_accounts(node, &src)?;
    validate_scripts(&src)?;
    let idx = watches_index(node)?;
    let _g = svc.load_lock.lock();
    let cond = WriteCond { if_seq_no: p.if_seq_no, if_primary_term: p.if_primary_term, ..Default::default() };
    cond.validate(false)?;
    let (version, seq_no, pt, created) = put_doc(&idx, id, &src, cond)?;
    let cw = model::build_cached(id, src, version, seq_no, pt)?;
    svc.cache.write().insert(id.to_string(), cw);
    Ok(json!({"_id": id, "_version": version, "_seq_no": seq_no, "_primary_term": pt, "created": created}))
}

/// `GET _watcher/watch/{id}` -> `(status code, body)`.
pub fn get_watch(node: &Node, id: &str) -> Result<(u16, Value)> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let cache = svc.cache.read();
    match cache.get(id) {
        None => Ok((404, json!({"found": false, "_id": id}))),
        Some(w) => {
            let mut status = w.src["status"].clone();
            status["version"] = json!(w.version);
            Ok((200, json!({
                "found": true, "_id": id, "_version": w.version, "_seq_no": w.seq_no, "_primary_term": w.primary_term,
                "status": canon_status(&status), "watch": model::redact(&model::watch_view(&w.src))
            })))
        }
    }
}

/// `DELETE _watcher/watch/{id}`.
pub fn delete_watch(node: &Node, id: &str) -> Result<(u16, Value)> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let _g = svc.load_lock.lock();
    let Some(idx) = node.get_index(super::WATCHES_INDEX) else {
        return Err(EsError::index_not_found(super::WATCHES_INDEX));
    };
    svc.cache.write().remove(id);
    match super::delete_doc(&idx, id)? {
        Some(v) => Ok((200, json!({"_id": id, "_version": v, "found": true}))),
        None => Ok((404, json!({"_id": id, "_version": 1, "found": false}))),
    }
}

/// Persist `status` of watch `id` (called by the executor).
pub fn store_status(node: &Node, svc: &WatcherService, id: &str, status: &Value) {
    let _g = svc.load_lock.lock();
    let Some(idx) = node.get_index(super::WATCHES_INDEX) else { return };
    let mut cache = svc.cache.write();
    let Some(cw) = cache.get_mut(id) else { return };
    let mut src = cw.src.clone();
    let headers = src["status"].get("headers").cloned();
    src["status"] = status.clone();
    if let Some(h) = headers {
        src["status"]["headers"] = h;
    }
    match put_doc(&idx, id, &src, WriteCond::default()) {
        Ok((v, s, pt, _)) => {
            cw.src = src;
            cw.version = v;
            cw.seq_no = s;
            cw.primary_term = pt;
        }
        Err(e) => tracing::warn!("failed to store status of watch [{id}]: {e}"),
    }
}

fn update_status(node: &Node, svc: &WatcherService, id: &str, new_status: Value) -> Result<(u64, i64, u64)> {
    let idx = watches_index(node)?;
    let mut cache = svc.cache.write();
    let cw = cache.get_mut(id).ok_or_else(|| not_exist(id))?;
    let mut src = cw.src.clone();
    let headers = src["status"].get("headers").cloned();
    src["status"] = new_status;
    if let Some(h) = headers {
        src["status"]["headers"] = h;
    }
    let (v, s, pt, _) = put_doc(&idx, id, &src, WriteCond::default())?;
    cw.src = src;
    cw.version = v;
    cw.seq_no = s;
    cw.primary_term = pt;
    cw.next_fire = None;
    Ok((v, s, pt))
}

/// `POST _watcher/watch/{id}/_ack[/{actions}]`.
pub fn ack_watch(node: &Node, id: &str, actions: Option<&str>) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let _g = svc.load_lock.lock();
    let (mut status, loaded_version) = {
        let cache = svc.cache.read();
        let cw = cache.get(id).ok_or_else(|| not_exist(id))?;
        (cw.src["status"].clone(), cw.version)
    };
    let wanted: Vec<String> = actions.map(|a| a.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
    let all = wanted.is_empty() || wanted.iter().any(|w| w == "_all");
    let now = iso(crate::date::now_millis());
    let mut changed = false;
    let ids: Vec<String> = status["actions"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    for aid in ids {
        if !all && !wanted.contains(&aid) {
            continue;
        }
        if status["actions"][&aid]["ack"]["state"] == "ackable" {
            status["actions"][&aid]["ack"] = json!({"timestamp": now, "state": "acked"});
            changed = true;
        }
    }
    if changed {
        update_status(node, &svc, id, status.clone())?;
    }
    status["version"] = json!(loaded_version);
    Ok(json!({"status": canon_status(&status)}))
}

/// `PUT _watcher/watch/{id}/_activate|_deactivate`.
pub fn set_active(node: &Node, id: &str, active: bool) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let _g = svc.load_lock.lock();
    let (mut status, loaded_version) = {
        let cache = svc.cache.read();
        match cache.get(id) {
            Some(cw) => (cw.src["status"].clone(), cw.version),
            None => {
                let uuid = node.get_index(super::WATCHES_INDEX).map(|i| i.uuid.clone()).unwrap_or_else(|| "_na_".into());
                return Err(EsError::document_missing(super::WATCHES_INDEX, &uuid, id));
            }
        }
    };
    let _ = loaded_version;
    if status["state"]["active"].as_bool() != Some(active) {
        status["state"] = json!({"active": active, "timestamp": iso(crate::date::now_millis())});
    }
    let (version, _, _) = update_status(node, &svc, id, status.clone())?;
    status["version"] = json!(version);
    Ok(json!({"status": canon_status(&status)}))
}

/// Request-level parameters of `_execute`.
pub fn execute_watch(node: &Node, id: Option<&str>, body: &Value, _debug: bool, user: Option<String>) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let mut opts = ExecOptions { manual: true, user, ..Default::default() };
    let mut inline: Option<Value> = None;
    if let Some(o) = body.as_object() {
        let pe = |m: String| model::parse_exc(format!("could not parse watch execution request. {m}"));
        for (k, v) in o {
            match v {
                Value::Bool(b) => match k.as_str() {
                    "ignore_condition" => opts.ignore_condition = *b,
                    "record_execution" => opts.record = *b,
                    _ => return Err(pe(format!("unexpected boolean field [{k}]"))),
                },
                Value::Object(m) => match k.as_str() {
                    "alternative_input" => opts.alternative_input = Some(v.clone()),
                    "trigger_data" => opts.trigger_data = Some(v.clone()),
                    "watch" => inline = Some(v.clone()),
                    "action_modes" => {
                        for (a, mv) in m {
                            let Some(ms) = mv.as_str() else { return Err(pe(format!("unexpected array field [{a}]"))) };
                            if !matches!(ms, "simulate" | "force_simulate" | "execute" | "force_execute" | "skip") {
                                return Err(model::parse_exc("could not parse watch execution request").caused_by(EsError::illegal_argument(format!("unknown action execution mode [{ms}]"))));
                            }
                            opts.action_modes.insert(a.clone(), ms.to_string());
                        }
                    }
                    "trigger" | "input" | "condition" | "actions" | "transform" | "throttle_period" | "throttle_period_in_millis" | "metadata" | "status" => {
                        return Err(model::parse_exc(format!("please wrap watch including field [{k}] inside a \"watch\" field")));
                    }
                    _ => return Err(pe(format!("unexpected object field [{k}]"))),
                },
                other => return Err(pe(format!("unexpected token [{}]", crate::jsonpos::token_name(other)))),
            }
        }
    }
    let now = crate::date::now_millis();
    let (watch_id, src, version) = match (id, inline) {
        (Some(id), None) => {
            let cache = svc.cache.read();
            let cw = cache.get(id).ok_or_else(|| {
                // Without any watch stored there is no `.watches` index to read from.
                if node.get_index(super::WATCHES_INDEX).is_none() {
                    EsError::index_not_found(super::WATCHES_INDEX)
                } else {
                    not_exist(id)
                }
            })?;
            (id.to_string(), cw.src.clone(), cw.version as i64)
        }
        (Some(_), Some(_)) => return Err(EsError::action_validation(&["a watch cannot be both stored and inlined"])),
        (None, Some(w)) => {
            opts.record = false;
            {
                let src = model::parse_watch("_inlined_", &w, now, true)?;
                validate_scripts(&src)?;
                ("_inlined_".to_string(), src, -1)
            }
        }
        (None, None) => return Err(EsError::action_validation(&["a watch execution request must either have a watch id or an inline watch source, but both are missing"])),
    };
    let (rid, record) = exec::execute(node, &svc, &watch_id, &src, version, None, now, &opts)?;
    Ok(json!({"_id": rid, "watch_record": record}))
}

pub fn start(node: &Node) -> Value {
    service(node).set_stopped(false);
    json!({"acknowledged": true})
}

pub fn stop(node: &Node) -> Value {
    service(node).set_stopped(true);
    json!({"acknowledged": true})
}

/// `GET _watcher/stats[/{metric}]`.
pub fn stats(node: &Node, metric: Option<&str>) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let metrics: Vec<String> = metric.map(|m| m.split(',').map(str::to_string).collect()).unwrap_or_default();
    let want = |n: &str| metrics.iter().any(|m| m == "_all" || m == "all" || m == n);
    let stopped = svc.manually_stopped();
    let max = if svc.counters.executed.load(Ordering::Relaxed) > 0 { std::thread::available_parallelism().map(|n| (n.get() * 5).min(50)).unwrap_or(2) } else { 0 };
    let mut n = json!({
        "node_id": NODE_ID,
        "watcher_state": svc.state_name(),
        "watch_count": if stopped { 0 } else { svc.cache.read().values().filter(|w| w.active()).count() },
        "execution_thread_pool": {"queue_size": 0, "max_size": max}
    });
    if want("current_watches") {
        let cur: Vec<Value> = svc
            .current
            .lock()
            .values()
            .map(|c| json!({"watch_id": c.watch_id, "watch_record_id": c.record_id, "triggered_time": iso(c.triggered_time), "execution_time": iso(c.execution_time), "execution_phase": c.phase}))
            .collect();
        n["current_watches"] = json!(cur);
    }
    if want("queued_watches") || want("pending_watches") {
        n["queued_watches"] = json!([]);
    }
    Ok(json!({
        "_nodes": {"total": 1, "successful": 1, "failed": 0},
        "cluster_name": node.cluster_name,
        "manually_stopped": stopped,
        "stats": [n]
    }))
}

/// Settings `_watcher/settings` may read / write (ES `UpdateWatcherSettingsAction`).
fn settable_setting(k: &str) -> bool {
    (matches!(k, "index.auto_expand_replicas" | "index.number_of_replicas")
        || ["index.routing.allocation.exclude.", "index.routing.allocation.include.", "index.routing.allocation.require."].iter().any(|p| k.starts_with(p)))
        && k != "index.routing.allocation.include._tier_preference"
}

/// `GET _watcher/settings`.
pub fn get_settings(node: &Node) -> Value {
    let Some(idx) = node.get_index(super::WATCHES_INDEX) else { return json!({}) };
    let mut flat = crate::settings::FlatSettings::new();
    for (k, v) in idx.meta().settings.iter() {
        if settable_setting(k) && !v.is_null() {
            flat.insert(k.clone(), v.clone());
        }
    }
    if !flat.contains_key("index.number_of_replicas") {
        let reps = if flat.contains_key("index.auto_expand_replicas") { "0" } else { "1" };
        flat.insert("index.number_of_replicas".into(), json!(reps));
    }
    crate::settings::unflatten(&flat)
}

/// `PUT _watcher/settings`.
pub fn put_settings(node: &Node, body: &Value) -> Result<Value> {
    let flat = crate::settings::normalize_index_settings(body);
    let mut bad: Vec<&String> = flat.keys().filter(|k| !settable_setting(k)).collect();
    if !bad.is_empty() {
        bad.sort();
        let bad = bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        return Err(EsError::action_validation(&[&format!("illegal settings: [{bad}], these settings may not be configured. Only the following settings may be configured: [index.number_of_replicas, index.auto_expand_replicas], [index.routing.allocation.require.*, index.routing.allocation.include.*, index.routing.allocation.exclude.*] excluding the following explicitly denied settings: [index.routing.allocation.include._tier_preference]")]));
    }
    let idx = watches_index(node)?;
    idx.put_settings(flat, false)?;
    Ok(json!({"acknowledged": true}))
}

/// `POST _watcher/_query/watches`.
pub fn query_watches(node: &Node, body: &Value) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let Some(idx) = node.get_index(super::WATCHES_INDEX) else { return Ok(json!({"count": 0, "watches": []})) };
    let _ = idx.refresh();
    let mut sb = Map::new();
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            match k.as_str() {
                "from" | "size" | "query" | "sort" | "search_after" => {
                    sb.insert(k.clone(), v.clone());
                }
                other => {
                    return Err(EsError::x_content_parse(format!("[1:2] [query_watches_request] unknown field [{other}]")));
                }
            }
        }
    }
    sb.insert("track_total_hits".into(), json!(true));
    sb.insert("seq_no_primary_term".into(), json!(true));
    let v: Value = match super::remote() {
        // On a cluster: `.watches` is read through the coordinator.
        Some(rest) => {
            crate::search::parse_request(&Value::Object(sb.clone()), &HashMap::new())?;
            let _ = rest("POST", &format!("/{}/_refresh", super::WATCHES_INDEX), None);
            rest("POST", &format!("/{}/_search?expand_wildcards=all&rest_total_hits_as_int=true", super::WATCHES_INDEX), Some(Value::Object(sb)))?
        }
        None => {
            let targets = node.resolve(super::WATCHES_INDEX, ResolveOpts { expand_hidden: true, ..Default::default() })?;
            let mut req = crate::search::parse_request(&Value::Object(sb), &HashMap::new())?;
            req.rest_total_hits_as_int = true;
            let out = crate::search::execute(&targets, &req, &super::search_opts(node))?;
            serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))?
        }
    };
    let watches: Vec<Value> = v["hits"]["hits"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|h| {
            let src = &h["_source"];
            json!({"_id": h["_id"], "watch": model::redact(&model::watch_view(src)), "status": canon_status(&src["status"]), "_seq_no": h["_seq_no"], "_primary_term": h["_primary_term"]})
        })
        .collect();
    Ok(json!({"count": v["hits"]["total"], "watches": watches}))
}

fn bump(m: &mut Map<String, Value>, key: &str, active: bool) {
    let e = m.entry(key.to_string()).or_insert_with(|| json!({"total": 0, "active": 0}));
    e["total"] = json!(e["total"].as_u64().unwrap_or(0) + 1);
    if active {
        e["active"] = json!(e["active"].as_u64().unwrap_or(0) + 1);
    }
}

/// `_xpack/usage` watcher section.
pub fn usage(node: &Node) -> Value {
    let svc = service(node);
    svc.ensure_loaded(node);
    let cache = svc.cache.read();
    let (mut input, mut cond, mut trans, mut action) = (Map::new(), Map::new(), Map::new(), Map::new());
    let mut sched = Map::new();
    let mut trig_all = json!({"total": 0, "active": 0});
    let mut meta = json!({"total": 0, "active": 0});
    let (mut total, mut active_n) = (0u64, 0u64);
    let first_key = |v: &Value| v.as_object().and_then(|o| o.keys().next().cloned());
    for w in cache.values() {
        let active = w.active();
        if !active {
            // the trigger service only tracks active watches
            continue;
        }
        total += 1;
        active_n += 1;
        if let Some(k) = first_key(&w.src["input"]) {
            bump(&mut input, &k, active);
            bump(&mut input, "_all", active);
        }
        if let Some(k) = first_key(&w.src["condition"]) {
            bump(&mut cond, &k, active);
            bump(&mut cond, "_all", active);
        }
        if let Some(k) = first_key(&w.src["transform"]) {
            bump(&mut trans, &k, active);
            bump(&mut trans, "_all", active);
        }
        if let Some(Value::Object(acts)) = w.src.get("actions") {
            for a in acts.values() {
                bump(&mut action, model::action_type(a), active);
                bump(&mut action, "_all", active);
            }
        }
        bump(&mut sched, w.schedule.name, active);
        bump(&mut sched, "_all", active);
        trig_all["total"] = json!(trig_all["total"].as_u64().unwrap_or(0) + 1);
        if active {
            trig_all["active"] = json!(trig_all["active"].as_u64().unwrap_or(0) + 1);
        }
        if w.src.get("metadata").map(|m| m.as_object().map(|o| !o.is_empty()).unwrap_or(false)).unwrap_or(false) {
            meta["total"] = json!(meta["total"].as_u64().unwrap_or(0) + 1);
            if active {
                meta["active"] = json!(meta["active"].as_u64().unwrap_or(0) + 1);
            }
        }
    }
    let mut exec_actions = Map::new();
    let (mut all_n, mut all_t) = (0u64, 0u64);
    for (ty, (n, t)) in svc.action_stats.lock().iter() {
        exec_actions.insert(ty.clone(), json!({"total": n, "total_time_in_ms": t}));
        all_n += n;
        all_t += t;
    }
    exec_actions.insert("_all".into(), json!({"total": all_n, "total_time_in_ms": all_t}));
    let mut watch = Map::new();
    sched.insert("total".into(), json!(total));
    sched.insert("active".into(), json!(active_n));
    watch.insert("input".into(), Value::Object(input));
    if meta["total"].as_u64().unwrap_or(0) > 0 {
        watch.insert("metadata".into(), meta);
    }
    if !trans.is_empty() {
        watch.insert("transform".into(), Value::Object(trans));
    }
    watch.insert("condition".into(), Value::Object(cond));
    watch.insert("action".into(), Value::Object(action));
    let mut trigger = Map::new();
    trigger.insert("schedule".into(), Value::Object(sched));
    trigger.insert("_all".into(), trig_all);
    watch.insert("trigger".into(), Value::Object(trigger));
    json!({
        "available": true,
        "enabled": super::watcher_enabled(node),
        "execution": {"actions": exec_actions},
        "watch": watch,
        "count": {"total": total, "active": active_n}
    })
}

/// Cached watch accessor for tests and other modules.
pub fn cached(node: &Node, id: &str) -> Option<Arc<Value>> {
    let svc = service(node);
    let cache = svc.cache.read();
    cache.get(id).map(|w: &CachedWatch| Arc::new(w.src.clone()))
}
