//! REST-level watcher operations (the HTTP layer only routes and formats).

use super::exec::{self, ExecOptions};
use super::model;
use super::{iso, put_doc, service, watches_index, CachedWatch, WatcherService};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts, NODE_ID};
use crate::security::owner::StoredAuthn;
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
    /// Security: the subject storing the watch; scheduled executions run
    /// as it (stored with the watch, as ES stores the security headers).
    pub identity: Option<StoredAuthn>,
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
    if let Some(id) = p.identity.as_ref().filter(|_| node.security.enabled()) {
        src["status"]["headers"] = json!({"_xpack_security_authentication": id.encode()});
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

/// On a cluster, make the cached copy of watch `id` the one in `.watches`
/// (read through the coordinator, real time, as ES's `TransportGetWatchAction`
/// reads the index): executions on other nodes update a watch's status, and
/// another master may have stored it. Off a cluster the cache is the index.
pub fn sync_from_index(node: &Node, svc: &WatcherService, id: &str) -> Result<()> {
    if super::remote().is_none() {
        return Ok(());
    }
    reload_one(node, svc, id)
}

/// Make the cached copy of watch `id` the document in `.watches` (on a
/// cluster through the coordinator, else this node's index). A watch whose
/// trigger, start time and activity did not change keeps its next fire time.
fn reload_one(node: &Node, svc: &WatcherService, id: &str) -> Result<()> {
    let current = match node.get_index(super::WATCHES_INDEX) {
        Some(idx) => super::get_doc(&idx, id)?,
        None if super::remote().is_some() => {
            // Not known here yet: the coordinator still knows the index.
            let rest = super::remote().ok_or_else(|| EsError::internal("no coordinator"))?;
            match rest("GET", &format!("/{}/_doc/{}", super::WATCHES_INDEX, super::path_expr(id)), None) {
                Ok(v) if v["found"] == true => Some((v["_source"].clone(), v["_version"].as_u64().unwrap_or(1), v["_seq_no"].as_i64().unwrap_or(0), v["_primary_term"].as_u64().unwrap_or(1))),
                Ok(_) => None,
                Err(e) if e.kind == "index_not_found_exception" => None,
                Err(e) => return Err(e),
            }
        }
        None => None,
    };
    let _g = svc.load_lock.lock();
    let mut cache = svc.cache.write();
    let Some((src, version, seq_no, pt)) = current else {
        cache.remove(id);
        return Ok(());
    };
    let mut cw = model::build_cached(id, src, version, seq_no, pt)?;
    if let Some(old) = cache.get(id) {
        if old.src["trigger"] == cw.src["trigger"] && old.start_time() == cw.start_time() && old.active() == cw.active() {
            cw.next_fire = old.next_fire;
        }
    }
    cache.insert(id.to_string(), cw);
    Ok(())
}

/// `GET _watcher/watch/{id}` -> `(status code, body)`.
pub fn get_watch(node: &Node, id: &str) -> Result<(u16, Value)> {
    let svc = service(node);
    svc.ensure_loaded(node);
    sync_from_index(node, &svc, id)?;
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

/// How often a status write retries after a concurrent change of the watch.
const STATUS_RETRIES: usize = 16;

fn is_conflict(e: &EsError) -> bool {
    e.kind == "version_conflict_engine_exception"
}

/// The status an execution stores, merged into the watch's `current` status
/// (ES `ExecutionService.updateWatchStatus` writes the status without its
/// `state`, conditioned on the watch's seq_no/primary_term). `base` is the
/// status the execution started from and `exec` the one it computed. Control
/// state accepted meanwhile wins: activation is never the execution's to
/// write, and an action whose ack changed since `base` (acknowledged, or
/// reset by another execution) keeps the current ack. Actions the current
/// watch no longer has (it was replaced) are dropped.
pub fn merge_exec_status(current: &Value, base: &Value, exec: &Value) -> Value {
    let mut out = exec.clone();
    match current.get("state") {
        Some(st) => out["state"] = st.clone(),
        None => {
            if let Some(o) = out.as_object_mut() {
                o.remove("state");
            }
        }
    }
    if let Some(h) = current.get("headers") {
        out["headers"] = h.clone();
    }
    let cur_actions = current.get("actions").and_then(Value::as_object);
    if let Some(acts) = out.get_mut("actions").and_then(Value::as_object_mut) {
        if let Some(cur) = cur_actions {
            acts.retain(|aid, _| cur.contains_key(aid));
        }
        for (aid, a) in acts.iter_mut() {
            let Some(cur_ack) = cur_actions.and_then(|c| c.get(aid)).and_then(|c| c.get("ack")) else { continue };
            if base["actions"][aid]["ack"] != *cur_ack {
                a["ack"] = cur_ack.clone();
            }
        }
        if let Some(cur) = cur_actions {
            for (aid, c) in cur {
                if !acts.contains_key(aid) {
                    acts.insert(aid.clone(), c.clone());
                }
            }
        }
    }
    canon_with_headers(&out)
}

/// [`canon_status`], keeping the stored security `headers`.
fn canon_with_headers(s: &Value) -> Value {
    let mut c = canon_status(s);
    if let Some(h) = s.get("headers") {
        c["headers"] = h.clone();
    }
    c
}

/// Persist the status an execution of watch `id` computed (`exec`, from the
/// status `base` it started with). The watch is re-read, the status merged
/// into it ([`merge_exec_status`]) and written conditionally on the read's
/// seq_no/primary_term; a concurrent change (ack, (de)activation, another
/// execution) makes the write retry from a fresh read, so an accepted
/// control change is never undone. A watch deleted meanwhile is left alone.
pub fn store_status(node: &Node, svc: &WatcherService, id: &str, base: &Value, exec: &Value) {
    let Some(idx) = node.get_index(super::WATCHES_INDEX) else { return };
    let _g = svc.load_lock.lock();
    for _ in 0..STATUS_RETRIES {
        let (mut src, _, seq_no, pt) = match super::get_doc(&idx, id) {
            Ok(Some(c)) => c,
            Ok(None) => {
                svc.cache.write().remove(id);
                return;
            }
            Err(e) => {
                tracing::warn!("failed to read watch [{id}] to store its status: {e}");
                return;
            }
        };
        src["status"] = merge_exec_status(&src["status"], base, exec);
        let cond = WriteCond { if_seq_no: Some(seq_no), if_primary_term: Some(pt), ..Default::default() };
        match put_doc(&idx, id, &src, cond) {
            Ok((v, s, pt, _)) => {
                let mut cache = svc.cache.write();
                if cache.get(id).is_some_and(|old| old.seq_no > s) {
                    return;
                }
                match model::build_cached(id, src, v, s, pt) {
                    Ok(mut cw) => {
                        if let Some(old) = cache.get(id) {
                            if old.src["trigger"] == cw.src["trigger"] && old.start_time() == cw.start_time() && old.active() == cw.active() {
                                cw.next_fire = old.next_fire;
                            }
                        }
                        cache.insert(id.to_string(), cw);
                    }
                    Err(e) => tracing::warn!("failed to cache watch [{id}]: {e}"),
                }
                return;
            }
            Err(e) if is_conflict(&e) => continue,
            Err(e) => {
                tracing::warn!("failed to store status of watch [{id}]: {e}");
                return;
            }
        }
    }
    tracing::warn!("failed to store status of watch [{id}]: it kept changing concurrently");
}

/// Write `new_status` of the cached watch `id`, conditioned on the cached
/// copy's seq_no/primary_term (ES's ack and (de)activate actions update the
/// watch with `if_seq_no`); a conflict surfaces to the caller, which re-reads.
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
    let cond = WriteCond { if_seq_no: Some(cw.seq_no), if_primary_term: Some(cw.primary_term), ..Default::default() };
    let (v, s, pt, _) = put_doc(&idx, id, &src, cond)?;
    cw.src = src;
    cw.version = v;
    cw.seq_no = s;
    cw.primary_term = pt;
    cw.next_fire = None;
    Ok((v, s, pt))
}

/// Run a status change of watch `id` against a fresh read of it, retrying
/// from a new read when the watch changed concurrently.
fn with_retry<T>(node: &Node, svc: &WatcherService, id: &str, mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let mut attempt = 0;
    loop {
        match f() {
            Err(e) if is_conflict(&e) && attempt + 1 < STATUS_RETRIES => {
                attempt += 1;
                reload_one(node, svc, id)?;
            }
            r => return r,
        }
    }
}

/// `POST _watcher/watch/{id}/_ack[/{actions}]`.
pub fn ack_watch(node: &Node, id: &str, actions: Option<&str>) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    sync_from_index(node, &svc, id)?;
    with_retry(node, &svc, id, || ack_once(node, &svc, id, actions))
}

fn ack_once(node: &Node, svc: &WatcherService, id: &str, actions: Option<&str>) -> Result<Value> {
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
        update_status(node, svc, id, status.clone())?;
    }
    status["version"] = json!(loaded_version);
    Ok(json!({"status": canon_status(&status)}))
}

/// `PUT _watcher/watch/{id}/_activate|_deactivate`.
pub fn set_active(node: &Node, id: &str, active: bool) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    sync_from_index(node, &svc, id)?;
    with_retry(node, &svc, id, || set_active_once(node, &svc, id, active))
}

fn set_active_once(node: &Node, svc: &WatcherService, id: &str, active: bool) -> Result<Value> {
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
    let (version, _, _) = update_status(node, svc, id, status.clone())?;
    status["version"] = json!(version);
    Ok(json!({"status": canon_status(&status)}))
}

/// Request-level parameters of `_execute`.
/// `caller`: the requesting subject. `_execute` runs the watch, inline or
/// stored, as it (ES `TransportExecuteWatchAction` replaces the stored
/// watch's security headers with the request's, so a run-as request runs
/// and records as the run-as user); scheduled runs use the stored subject.
pub fn execute_watch(node: &Node, id: Option<&str>, body: &Value, _debug: bool, caller: Option<StoredAuthn>) -> Result<Value> {
    let svc = service(node);
    svc.ensure_loaded(node);
    let mut opts = ExecOptions { manual: true, identity: caller, ..Default::default() };
    if let Some(id) = id {
        sync_from_index(node, &svc, id)?;
    }
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
            if opts.identity.is_none() {
                opts.identity = exec::stored_identity(node, id, &cw.src)?;
            }
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

#[cfg(test)]
mod inflight_tests {
    //! An execution that started before an ack or a deactivation stores its
    //! status after them: the accepted control change must stay effective
    //! (ES `ExecutionService.updateWatchStatus` never writes `state` and
    //! writes conditionally on the watch's seq_no/primary_term).
    use super::*;
    use crate::watcher::{get_doc, put_doc_local, WATCHES_INDEX};

    fn watch() -> Value {
        json!({
            "trigger": {"schedule": {"interval": "1h"}},
            "input": {"simple": {"n": 1}},
            "condition": {"always": {}},
            "actions": {"log": {"logging": {"text": "watch {{ctx.watch_id}}"}}}
        })
    }

    fn setup(id: &str) -> (tempfile::TempDir, Arc<Node>) {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let p = PutParams { active: true, version: None, if_seq_no: None, if_primary_term: None, identity: None };
        put_watch(&node, id, &watch(), &p).unwrap();
        (tmp, node)
    }

    /// The `.watches` document as stored (not the cache).
    fn stored(node: &Node, id: &str) -> Value {
        let idx = node.get_index(WATCHES_INDEX).unwrap();
        get_doc(&idx, id).unwrap().unwrap().0
    }

    /// Run the watch from `src` (the copy an execution started with) and
    /// record it, as a scheduled or `record_execution: true` run does.
    fn run(node: &Node, id: &str, src: &Value) {
        let svc = service(node);
        let opts = ExecOptions { record: true, manual: true, ..Default::default() };
        exec::execute(node, &svc, id, src, 1, None, crate::date::now_millis(), &opts).unwrap();
    }

    fn snapshot(node: &Node, id: &str) -> Value {
        (*cached(node, id).unwrap()).clone()
    }

    #[test]
    fn ack_during_execution_is_not_undone() {
        let (_tmp, node) = setup("w");
        run(&node, "w", &snapshot(&node, "w"));
        assert_eq!(stored(&node, "w")["status"]["actions"]["log"]["ack"]["state"], "ackable");
        // An execution starts from this copy, the watch is acknowledged
        // meanwhile, then the execution stores its status.
        let inflight = snapshot(&node, "w");
        let before = stored(&node, "w")["status"]["last_checked"].clone();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let r = ack_watch(&node, "w", None).unwrap();
        assert_eq!(r["status"]["actions"]["log"]["ack"]["state"], "acked");
        run(&node, "w", &inflight);
        let st = stored(&node, "w")["status"].clone();
        assert_eq!(st["actions"]["log"]["ack"]["state"], "acked", "ack reverted by the in-flight execution: {st}");
        let (_, got) = get_watch(&node, "w").unwrap();
        assert_eq!(got["status"]["actions"]["log"]["ack"]["state"], "acked", "{got}");
        // The execution's own fields still landed.
        assert!(st["last_checked"].is_string() && st["last_checked"] != before, "{st}");
        assert!(st["execution_state"].is_string(), "{st}");
        assert_eq!(st["state"]["active"], true);
    }

    #[test]
    fn deactivate_during_execution_is_not_undone() {
        let (_tmp, node) = setup("w");
        let inflight = snapshot(&node, "w");
        assert_eq!(inflight["status"]["state"]["active"], true);
        let r = set_active(&node, "w", false).unwrap();
        assert_eq!(r["status"]["state"]["active"], false);
        run(&node, "w", &inflight);
        let st = stored(&node, "w")["status"].clone();
        assert_eq!(st["state"]["active"], false, "deactivation reverted by the in-flight execution: {st}");
        assert!(st["last_checked"].is_string(), "the execution's status was stored: {st}");
        assert_eq!(cached(&node, "w").unwrap()["status"]["state"]["active"], false);
        let (_, got) = get_watch(&node, "w").unwrap();
        assert_eq!(got["status"]["state"]["active"], false, "{got}");
        // A deactivated watch is not scheduled.
        let svc = service(&node);
        assert!(crate::watcher::due_watches(&svc, crate::date::now_millis() + 10 * 3_600_000).iter().all(|(id, _)| id != "w"));
        // Re-activation still works and is likewise kept.
        let inflight = snapshot(&node, "w");
        set_active(&node, "w", true).unwrap();
        run(&node, "w", &inflight);
        assert_eq!(stored(&node, "w")["status"]["state"]["active"], true);
    }

    #[test]
    fn ack_retries_after_a_concurrent_status_write() {
        let (_tmp, node) = setup("w");
        run(&node, "w", &snapshot(&node, "w"));
        // Another writer (an execution on another node) stores a status the
        // cache has not seen: the conditional ack conflicts, re-reads and
        // keeps that write's fields.
        let idx = node.get_index(WATCHES_INDEX).unwrap();
        let mut src = stored(&node, "w");
        src["status"]["last_checked"] = json!("2030-01-01T00:00:00.000Z");
        put_doc_local(&idx, "w", &src, WriteCond::default()).unwrap();
        let r = ack_watch(&node, "w", None).unwrap();
        assert_eq!(r["status"]["actions"]["log"]["ack"]["state"], "acked", "{r}");
        let st = stored(&node, "w")["status"].clone();
        assert_eq!(st["actions"]["log"]["ack"]["state"], "acked", "{st}");
        assert_eq!(st["last_checked"], "2030-01-01T00:00:00.000Z", "the concurrent write survived: {st}");
    }

    #[test]
    fn merge_keeps_control_state_and_execution_fields() {
        let base = json!({"state": {"active": true, "timestamp": "t0"}, "actions": {"a": {"ack": {"state": "ackable", "timestamp": "t0"}}, "gone": {"ack": {"state": "ackable"}}}});
        let current = json!({"state": {"active": false, "timestamp": "t1"}, "headers": {"h": "x"},
            "actions": {"a": {"ack": {"state": "acked", "timestamp": "t1"}}, "b": {"ack": {"state": "awaits_successful_execution"}}}});
        let exec = json!({"state": {"active": true, "timestamp": "t0"}, "last_checked": "t2", "execution_state": "executed",
            "actions": {"a": {"ack": {"state": "ackable", "timestamp": "t0"}, "last_execution": {"timestamp": "t2", "successful": true}}, "gone": {"ack": {"state": "ackable"}}}});
        let m = merge_exec_status(&current, &base, &exec);
        assert_eq!(m["state"], current["state"]);
        assert_eq!(m["headers"], current["headers"]);
        assert_eq!(m["actions"]["a"]["ack"]["state"], "acked");
        assert_eq!(m["actions"]["a"]["last_execution"]["timestamp"], "t2");
        assert_eq!(m["last_checked"], "t2");
        assert_eq!(m["execution_state"], "executed");
        assert!(m["actions"].get("gone").is_none(), "{m}");
        assert_eq!(m["actions"]["b"], current["actions"]["b"]);
        // An ack the execution itself changed (base == current) is the execution's.
        let current2 = json!({"state": {"active": true}, "actions": {"a": {"ack": {"state": "ackable", "timestamp": "t0"}}}});
        let exec2 = json!({"actions": {"a": {"ack": {"state": "awaits_successful_execution", "timestamp": "t2"}}}});
        assert_eq!(merge_exec_status(&current2, &base, &exec2)["actions"]["a"]["ack"]["state"], "awaits_successful_execution");
    }
}
