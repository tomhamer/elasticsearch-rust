//! Watch actions: logging, index, webhook, email, slack, pagerduty, jira.

use super::exec::{path_get, render, render_value, Ctx};
use super::httpc::{self, HttpRequest};
use super::{iso, setting};
use crate::node::Node;
use crate::shard::WriteCond;
use base64::Engine;
use serde_json::{json, Map, Value};
use elasticsearch_ssl::rustls;
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

pub const DEFAULT_MAX_ITERATIONS: usize = 100;

/// `xpack.http.whitelist` check (default `*`).
pub fn check_whitelist(node: &Node, r: &HttpRequest) -> Result<(), String> {
    let wl = setting(node, "xpack.http.whitelist").unwrap_or_else(|| "*".into());
    let target = format!("{}:{}", r.host, r.port);
    if wl.split(',').map(str::trim).filter(|p| !p.is_empty()).any(|p| crate::mapping::wildcard_match(p, &r.host) || crate::mapping::wildcard_match(p, &target)) {
        Ok(())
    } else {
        Err(format!("host [{target}] is not whitelisted in setting [xpack.http.whitelist], will not connect"))
    }
}

type Single = (&'static str, Value, Option<String>);

fn ok(simulate: bool, body: Value) -> Single {
    (if simulate { "simulated" } else { "success" }, body, None)
}

fn fail(reason: impl Into<String>, body: Value) -> Single {
    ("failure", body, Some(reason.into()))
}

/// Run one action (with `foreach` handling); returns the action result object.
pub fn run_action(node: &Node, aid: &str, def: &Value, ctx: &Ctx, simulate: bool) -> Value {
    let ty = super::model::action_type(def);
    if let Some(fe) = def.get("foreach").and_then(Value::as_str) {
        let cj = ctx.to_json();
        let key = fe.strip_prefix("ctx.").unwrap_or(fe);
        // ES `ActionWrapper.execute`: a missing, non-collection or empty
        // foreach object fails the action (it never counts as a success
        // that would arm the throttle, #581).
        let fail = |msg: String| json!({"id": aid, "type": ty, "status": "failure",
            "error": {"root_cause": [{"type": "exception", "reason": msg}], "type": "exception", "reason": msg}});
        let list = match path_get(&cj, key) {
            Some(Value::Array(a)) if a.is_empty() => return fail(format!("foreach object [{fe}] was an empty list, could not run any action")),
            Some(Value::Array(a)) => a.clone(),
            None | Some(Value::Null) => return fail(format!("specified foreach object was null: [{fe}]")),
            Some(_) => return fail(format!("specified foreach object was not a an array/collection: [{fe}]")),
        };
        let max = def.get("max_iterations").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(DEFAULT_MAX_ITERATIONS);
        let mut results = Vec::new();
        let mut all_ok = true;
        let mut limited = false;
        for (i, item) in list.iter().enumerate() {
            if i >= max {
                limited = true;
                break;
            }
            let mut c = ctx.clone();
            c.payload = if item.is_object() { item.clone() } else { json!({"_value": item}) };
            let (st, mut body, _reason) = run_single(node, ty, def, &c, simulate);
            if st == "failure" {
                all_ok = false;
            }
            if let Some(o) = body.as_object_mut() {
                o.remove("__error");
            }
            let mut r = Map::new();
            r.insert(ty.to_string(), body);
            results.push(Value::Object(r));
        }
        let mut out = json!({"id": aid, "type": ty, "status": if !all_ok { "failure" } else if simulate { "simulated" } else { "success" }, "number_of_actions_executed": results.len(), "foreach": results});
        if let Some(m) = def.get("max_iterations") {
            out["max_iterations"] = m.clone();
        }
        let _ = limited;
        return out;
    }
    let (st, mut body, reason) = run_single(node, ty, def, ctx, simulate);
    let error = body.as_object_mut().and_then(|o| o.remove("__error"));
    let mut m = Map::new();
    m.insert("id".into(), json!(aid));
    m.insert("type".into(), json!(ty));
    m.insert("status".into(), json!(st));
    if let Some(e) = error {
        m.insert("error".into(), e);
    } else if let Some(r) = reason {
        m.insert("reason".into(), json!(r));
    }
    if st != "failure" || body.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
        m.insert(ty.to_string(), body);
    }
    Value::Object(m)
}

fn run_single(node: &Node, ty: &str, def: &Value, ctx: &Ctx, simulate: bool) -> Single {
    let cj = ctx.to_json();
    let d = &def[ty];
    match ty {
        "logging" => logging(d, &cj, &ctx.data, simulate),
        "index" => index(node, d, ctx, &cj, simulate),
        "webhook" => webhook(node, d, &cj, &ctx.data, simulate),
        "email" => email(node, d, ctx, &cj, simulate),
        "slack" => slack(node, d, &cj, simulate),
        "pagerduty" => pagerduty(node, d, &cj, simulate),
        "jira" => jira(node, d, &cj, simulate),
        other => fail(format!("unknown action type [{other}]"), json!({})),
    }
}

fn logging(d: &Value, cj: &Value, data: &std::path::Path, simulate: bool) -> Single {
    let text = match super::exec::render_text(&d["text"], cj, data) {
        Ok(t) => t,
        Err(e) => return fail(e.reason.clone(), json!({})),
    };
    let category = d["category"].as_str().unwrap_or("xpack.watcher.actions.logging");
    match d["level"].as_str().unwrap_or("info") {
        "error" => tracing::error!(target: "watcher", "[{category}] {text}"),
        "warn" => tracing::warn!(target: "watcher", "[{category}] {text}"),
        "debug" => tracing::debug!(target: "watcher", "[{category}] {text}"),
        "trace" => tracing::trace!(target: "watcher", "[{category}] {text}"),
        _ => tracing::info!(target: "watcher", "[{category}] {text}"),
    }
    ok(simulate, json!({"logged_text": text}))
}

fn index(node: &Node, d: &Value, ctx: &Ctx, cj: &Value, simulate: bool) -> Single {
    let target = d["index"].as_str().map(|s| render(s, cj));
    let doc_id = d["doc_id"].as_str().map(|s| render(s, cj));
    let time_field = d["execution_time_field"].as_str();
    let mut docs: Vec<Value> = match ctx.payload.get("_doc") {
        Some(Value::Array(a)) => a.clone(),
        Some(o) if o.is_object() => vec![o.clone()],
        _ => vec![ctx.payload.clone()],
    };
    let multi = matches!(ctx.payload.get("_doc"), Some(Value::Array(_)));
    for doc in docs.iter_mut() {
        if let (Some(f), Some(o)) = (time_field, doc.as_object_mut()) {
            o.insert(f.to_string(), json!(iso(ctx.execution_time)));
        }
    }
    // Each document's destination and id (ES `ExecutableIndexAction`
    // `getField`): its own `_index` / `_id`, else the action's; both set is
    // an error (finding 205: documents never collapse into one index).
    let watch_id = ctx.watch_id.clone();
    let mut items: Vec<(String, Option<String>, Value)> = Vec::with_capacity(docs.len());
    for mut doc in docs {
        let (own_index, own_id) = match doc.as_object_mut() {
            Some(o) => (o.remove("_index").map(|v| scalar_text(&v)), o.remove("_id").map(|v| scalar_text(&v))),
            None => (None, None),
        };
        let index = match (own_index, &target) {
            (Some(_), Some(_)) => return fail(both_set(&watch_id, "index", "_index"), json!({})),
            (Some(i), None) => i,
            (None, Some(t)) => t.clone(),
            (None, None) => return fail("unable to index document: no index specified", json!({})),
        };
        let id = match (own_id, &doc_id) {
            (Some(_), Some(_)) => return fail(both_set(&watch_id, "id", "_id"), json!({})),
            (Some(i), None) => Some(i),
            (None, d) => d.clone(),
        };
        items.push((index, id, doc));
    }
    if simulate {
        // ES `IndexAction.Simulated`: one document reports its request; a
        // `_doc` list reports each request's `_id` / `_index` (index and
        // doc_id empty), as `ExecutableIndexAction.indexBulk` does.
        let mut rq = Map::new();
        if multi {
            rq.insert("index".into(), json!(""));
            rq.insert("doc_id".into(), json!(""));
        } else {
            rq.insert("index".into(), json!(items[0].0));
            if let Some(i) = &items[0].1 {
                rq.insert("doc_id".into(), json!(i));
            }
        }
        let refresh = match &d["refresh"] {
            Value::Bool(b) => Some(if *b { "true" } else { "false" }),
            Value::String(r) if r.is_empty() || r == "true" => Some("true"),
            Value::String(r) if r == "false" => Some("false"),
            Value::String(r) if r == "wait_for" => Some("wait_for"),
            _ => None,
        };
        if let Some(r) = refresh {
            rq.insert("refresh".into(), json!(r));
        }
        let src = if multi { json!(items.iter().map(|(i, id, _)| json!({"_id": id, "_index": i})).collect::<Vec<_>>()) } else { items[0].2.clone() };
        rq.insert("source".into(), src);
        return ok(true, json!({"request": rq}));
    }
    let mut indices: Vec<&str> = items.iter().map(|(i, _, _)| i.as_str()).collect();
    indices.sort();
    indices.dedup();
    for index_name in indices {
        if !super::exec::user_allowed(node, ctx.identity.as_deref(), "indices:data/write/bulk[s]", index_name) {
            let id = ctx.identity.as_deref();
            let u = id.map(|i| i.principal().to_string()).unwrap_or_default();
            let roles = id.map(|i| node.security.permission(&i.to_authn()).effective_roles.join(",")).unwrap_or_default();
            return fail(format!("action [indices:data/write/bulk[s]] is unauthorized for user [{u}] with effective roles [{roles}] on indices [{index_name}], this action is granted by the index privileges [create_doc,create,delete,index,write,all]"), json!({}));
        }
    }
    let create = d["op_type"].as_str() == Some("create");
    let refresh = matches!(d["refresh"].as_str(), Some("true" | "wait_for")) || d["refresh"] == true;
    // On a cluster the write runs as the watch's subject.
    match super::remote_as(ctx.identity.as_deref()) {
        Ok(Some(rest)) => return index_remote(&rest, items, create, refresh, multi),
        Ok(None) => {}
        Err(e) => return fail(e.reason.clone(), json!({})),
    }
    let mut successful = Vec::new();
    let mut failed = Vec::new();
    let mut touched: Vec<Arc<crate::index::IndexService>> = Vec::new();
    for (index_name, id, doc) in items {
        let id = id.unwrap_or_else(crate::index::auto_id);
        // The destination's default and final pipelines run, as for any
        // index request (#584): the document is written as they leave it,
        // dropped, or failed with their error.
        let mut cond = WriteCond { create, ..Default::default() };
        let source = doc.to_string();
        let req = crate::ingest::IngestRequest { index: &index_name, id: Some(&id), routing: None, source: &source, pipeline: None, cond: crate::ingest::WriteMeta::of(&cond) };
        let (index_name, id, routing, source, dynamic_templates) = match node.run_ingest_ext(req) {
            Ok(crate::ingest::IngestOutcome::Unchanged) => (index_name, id, None, source, None),
            Ok(crate::ingest::IngestOutcome::Dropped { index }) => {
                successful.push(json!({"created": false, "result": "noop", "id": id, "version": -3, "index": index}));
                continue;
            }
            Ok(crate::ingest::IngestOutcome::Changed { index, id: new_id, routing, source, dynamic_templates, cond: after }) => {
                after.apply(&mut cond);
                (index, new_id.unwrap_or(id), routing, source, dynamic_templates)
            }
            Err(f) => {
                failed.push(json!({"id": id, "type": f.error.kind, "reason": f.error.reason}));
                continue;
            }
        };
        let svc = match node.write_index(&index_name, true) {
            Ok(s) => s,
            Err(e) => {
                failed.push(json!({"id": id, "type": e.kind, "reason": e.reason}));
                continue;
            }
        };
        let write = || svc.write(vec![crate::index::WriteReq::Index { id: Some(id.clone()), source, routing, cond }], false).pop().unwrap_or_else(|| Err(crate::error::EsError::internal("no write result")));
        let res = match dynamic_templates {
            Some(dt) => crate::document::with_dynamic_templates(dt, write),
            None => write(),
        };
        match res {
            Ok(w) => {
                let created = w.result == "created";
                successful.push(json!({"created": created, "result": if created { "created" } else { "updated" }, "id": id, "version": w.version, "index": svc.name}))
            }
            Err(e) => failed.push(json!({"id": id, "type": e.kind, "reason": e.reason})),
        }
        if !touched.iter().any(|t| Arc::ptr_eq(t, &svc)) {
            touched.push(svc);
        }
    }
    if refresh {
        for svc in &touched {
            let _ = svc.refresh();
        }
    }
    if multi {
        let status_ok = failed.is_empty();
        let body = json!({"response": successful.iter().chain(failed.iter()).cloned().collect::<Vec<_>>()});
        if status_ok { ok(false, body) } else { fail("failed to index one or more documents", body) }
    } else if let Some(f) = failed.first() {
        fail(f["reason"].as_str().unwrap_or("index failed"), json!({}))
    } else {
        ok(false, json!({"response": successful[0]}))
    }
}

fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// ES `ExecutableIndexAction.getField`'s conflict error.
fn both_set(watch_id: &str, field: &str, doc_field: &str) -> String {
    format!("could not execute action [index] of watch [{watch_id}]. [ctx.payload._doc.{doc_field}] or [ctx.payload.{doc_field}] were set together with action [{field}] field. Only set one of them")
}

/// The `_bulk` body of an index action's documents: `(index, id, source)` each.
fn index_bulk_body(docs: &[(String, String, Value)], create: bool) -> String {
    let op = if create { "create" } else { "index" };
    let mut out = String::new();
    for (index, id, doc) in docs {
        out.push_str(&json!({ op: {"_index": index, "_id": id} }).to_string());
        out.push('\n');
        out.push_str(&doc.to_string());
        out.push('\n');
    }
    out
}

/// The index action on a cluster: one `_bulk` through the coordinator,
/// each document to its own index.
fn index_remote(rest: &super::RestCall, items: Vec<(String, Option<String>, Value)>, create: bool, refresh: bool, multi: bool) -> Single {
    let items: Vec<(String, String, Value)> = items.into_iter().map(|(index, id, doc)| (index, id.unwrap_or_else(crate::index::auto_id), doc)).collect();
    let uri = if refresh { "/_bulk?refresh=true" } else { "/_bulk" };
    let resp = match rest("POST", uri, Some(Value::String(index_bulk_body(&items, create)))) {
        Ok(r) => r,
        Err(e) => return fail(e.reason.clone(), json!({})),
    };
    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for item in resp["items"].as_array().into_iter().flatten() {
        let Some(r) = item.as_object().and_then(|o| o.values().next()) else { continue };
        let id = r["_id"].clone();
        match r.get("error") {
            Some(e) => failed.push(json!({"id": id, "type": e["type"], "reason": e["reason"]})),
            None => {
                let created = r["result"].as_str() == Some("created");
                successful.push(json!({"created": created, "result": if created { "created" } else { "updated" }, "id": id, "version": r["_version"], "index": r["_index"]}));
            }
        }
    }
    if multi {
        let status_ok = failed.is_empty();
        let body = json!({"response": successful.iter().chain(failed.iter()).cloned().collect::<Vec<_>>()});
        if status_ok { ok(false, body) } else { fail("failed to index one or more documents", body) }
    } else if let Some(f) = failed.first() {
        fail(f["reason"].as_str().unwrap_or("index failed"), json!({}))
    } else if let Some(s) = successful.first() {
        ok(false, json!({"response": s}))
    } else {
        fail("index failed", json!({}))
    }
}

/// Render object-form `TextTemplate`s (`{"source": .., "params": ..}`) of a
/// request template (`body`) to strings up front.
pub(crate) fn render_request_templates(d: &Value, cj: &Value, data: &std::path::Path) -> Result<Value, String> {
    let mut d = d.clone();
    if let Some(b) = d.get("body").filter(|b| b.is_object()).cloned() {
        let text = super::exec::render_text(&b, cj, data).map_err(|e| e.reason)?;
        d["body"] = Value::String(text);
        // the rendered text is final: `HttpRequest::from_json` must not render it again
        d[httpc::BODY_RENDERED] = Value::Bool(true);
    }
    Ok(d)
}

fn webhook(node: &Node, d: &Value, cj: &Value, data: &std::path::Path, simulate: bool) -> Single {
    let d = &match render_request_templates(d, cj, data) {
        Ok(d) => d,
        Err(e) => return fail(e, json!({})),
    };
    let r = match HttpRequest::from_json(d, &|s| render(s, cj)) {
        Ok(r) => r,
        Err(e) => return fail(e, json!({})),
    };
    if simulate {
        return ok(true, json!({"request": r.to_json_sim()}));
    }
    if let Err(e) = check_whitelist(node, &r) {
        return fail(e, json!({"request": r.to_json()}));
    }
    match httpc::send(node, &r) {
        Ok(resp) => {
            let mut headers = Map::new();
            for (k, v) in &resp.headers {
                headers.entry(k.clone()).or_insert_with(|| json!([])).as_array_mut().unwrap().push(json!(v));
            }
            let body = json!({"request": r.to_json(), "response": {"status": resp.status, "headers": headers, "body": resp.body_str()}});
            if resp.status >= 400 {
                ("failure", body, Some(format!("received [{}] status code", resp.status)))
            } else {
                ok(false, body)
            }
        }
        Err(e) => ("failure", json!({"__error": http_error(&r, &e)}), None),
    }
}

/// ES-shaped exception object for a failed HTTP call.
pub(crate) fn http_error(r: &HttpRequest, e: &str) -> Value {
    let (ty, reason, cause) = if e.to_ascii_lowercase().contains("refused") {
        ("http_host_connect_exception", format!("Connect to {}:{} [/{}] failed: Connection refused (connect failed)", r.host, r.port, r.host), Some(("connect_exception", "Connection refused (connect failed)".to_string())))
    } else {
        ("exception", e.to_string(), None)
    };
    let mut o = json!({"root_cause": [{"type": ty, "reason": reason}], "type": ty, "reason": reason});
    if let Some((ct, cr)) = cause {
        o["caused_by"] = json!({"type": ct, "reason": cr});
    }
    o
}

// ---------------------------------------------------------------------------
// Email
// ---------------------------------------------------------------------------

fn addr_list(v: Option<&Value>, cj: &Value) -> Vec<String> {
    match v {
        Some(Value::String(s)) => s.split(',').map(|x| render(x.trim(), cj)).filter(|x| !x.is_empty()).collect(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(|x| render(x, cj)).collect(),
        _ => Vec::new(),
    }
}

fn account_setting(node: &Node, kind: &str, account: &str, key: &str) -> Option<String> {
    setting(node, &format!("xpack.notification.{kind}.account.{account}.{key}"))
}

/// The named (or first) account of a notification service, found by scanning
/// the cluster settings for `xpack.notification.<kind>.account.<name>.*`.
fn find_account(node: &Node, kind: &str, wanted: Option<&str>) -> Option<String> {
    if let Some(w) = wanted {
        return Some(w.to_string());
    }
    if let Some(d) = setting(node, &format!("xpack.notification.{kind}.default_account")) {
        return Some(d);
    }
    let prefix = format!("xpack.notification.{kind}.account.");
    let cs = node.cluster_settings.read();
    for m in [&cs.1, &cs.0] {
        for k in m.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                if let Some((name, _)) = rest.split_once('.') {
                    return Some(name.to_string());
                }
            }
        }
        if let Some(Value::Object(n)) = m.get("xpack").and_then(|x| x.get("notification")).and_then(|x| x.get(kind)).and_then(|x| x.get("account")) {
            if let Some(name) = n.keys().next() {
                return Some(name.clone());
            }
        }
    }
    None
}

fn email(node: &Node, d: &Value, ctx: &Ctx, cj: &Value, simulate: bool) -> Single {
    let account = find_account(node, "email", d["account"].as_str());
    let from = d["from"].as_str().map(|s| render(s, cj)).or_else(|| account.as_ref().and_then(|a| account_setting(node, "email", a, "email_defaults.from")));
    let to = addr_list(d.get("to"), cj);
    let cc = addr_list(d.get("cc"), cj);
    let bcc = addr_list(d.get("bcc"), cj);
    let reply_to = addr_list(d.get("reply_to"), cj);
    let subject = render(d["subject"].as_str().unwrap_or(""), cj);
    let (text, html) = match &d["body"] {
        Value::String(s) => (Some(render(s, cj)), None),
        Value::Object(o) => (o.get("text").and_then(Value::as_str).map(|s| render(s, cj)), o.get("html").and_then(Value::as_str).map(|s| render(s, cj))),
        _ => (None, None),
    };
    let mut msg = Map::new();
    msg.insert("id".into(), json!(ctx.id));
    if let Some(f) = &from {
        msg.insert("from".into(), json!(f));
    }
    if !reply_to.is_empty() {
        msg.insert("reply_to".into(), json!(reply_to));
    }
    if let Some(p) = d["priority"].as_str() {
        msg.insert("priority".into(), json!(p));
    }
    msg.insert("to".into(), json!(to));
    if !cc.is_empty() {
        msg.insert("cc".into(), json!(cc));
    }
    if !bcc.is_empty() {
        msg.insert("bcc".into(), json!(bcc));
    }
    msg.insert("subject".into(), json!(subject));
    let mut body = Map::new();
    if let Some(t) = &text {
        body.insert("text".into(), json!(t));
    }
    if let Some(h) = &html {
        body.insert("html".into(), json!(h));
    }
    if !body.is_empty() {
        msg.insert("body".into(), Value::Object(body));
    }
    let mut attachments = Map::new();
    if let Some(Value::Object(at)) = d.get("attachments") {
        for (name, a) in at {
            let (ty, spec) = a.as_object().and_then(|o| o.iter().next()).map(|(k, v)| (k.as_str(), v.clone())).unwrap_or(("data", json!({})));
            attachments.insert(name.clone(), json!({"type": ty, ty: render_value(&spec, cj)}));
        }
    }
    let attach_json = if attachments.is_empty() { None } else { Some(Value::Object(attachments)) };
    let mut result = json!({"message": Value::Object(msg.clone())});
    if simulate {
        if let Some(a) = &account {
            result["account"] = json!(a);
        }
        return ok(true, result);
    }
    let Some(account) = account else {
        return fail(format!("failed to send email with subject [{subject}] via account [default]: no email accounts are configured"), result);
    };
    result["account"] = json!(account);
    let Some(host) = account_setting(node, "email", &account, "smtp.host") else {
        return fail(format!("failed to send email with subject [{subject}] via account [{account}]: account [{account}] is not configured"), result);
    };
    let cfg = match smtp_config(node, &account, host) {
        Ok(c) => c,
        Err(e) => return fail(format!("failed to send email with subject [{subject}] via account [{account}]: {e}"), result),
    };
    let Some(from) = from else {
        return fail(format!("failed to send email with subject [{subject}] via account [{account}]: no from address"), result);
    };
    let mut all_rcpt = to.clone();
    all_rcpt.extend(cc.clone());
    all_rcpt.extend(bcc.clone());
    let raw = build_mime(&from, &to, &cc, &reply_to, &subject, text.as_deref(), html.as_deref(), &ctx.id, attach_json.as_ref());
    let tls = || -> Result<Arc<rustls::ClientConfig>, String> {
        crate::remote::client_tls(&node.node_settings.read(), EMAIL_SSL, cfg.trusted_without_verification)
    };
    match smtp_send(&cfg, &tls, &from, &all_rcpt, &raw) {
        Ok(()) => ok(false, result),
        Err(e) => fail(format!("failed to send email with subject [{subject}] via account [{account}]: {e}"), result),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_mime(from: &str, to: &[String], cc: &[String], reply_to: &[String], subject: &str, text: Option<&str>, html: Option<&str>, id: &str, attachments: Option<&Value>) -> String {
    let mut h = String::new();
    h.push_str(&format!("From: {from}\r\nTo: {}\r\n", to.join(", ")));
    if !cc.is_empty() {
        h.push_str(&format!("Cc: {}\r\n", cc.join(", ")));
    }
    if !reply_to.is_empty() {
        h.push_str(&format!("Reply-To: {}\r\n", reply_to.join(", ")));
    }
    h.push_str(&format!("Subject: {subject}\r\nMessage-ID: <{id}@watcher>\r\nMIME-Version: 1.0\r\n"));
    let boundary = format!("----=_watcher_{}", uuid::Uuid::new_v4().simple());
    let mut body = String::new();
    match (text, html) {
        (Some(t), Some(hh)) => {
            h.push_str(&format!("Content-Type: multipart/alternative; boundary=\"{boundary}\"\r\n\r\n"));
            body.push_str(&format!("--{boundary}\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n{t}\r\n--{boundary}\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n{hh}\r\n--{boundary}--\r\n"));
        }
        (None, Some(hh)) => {
            h.push_str("Content-Type: text/html; charset=UTF-8\r\n\r\n");
            body.push_str(hh);
        }
        (t, None) => {
            h.push_str("Content-Type: text/plain; charset=UTF-8\r\n\r\n");
            body.push_str(t.unwrap_or(""));
        }
    }
    let _ = attachments;
    format!("{h}{body}")
}

/// The email notification's TLS settings (`xpack.notification.email.ssl.*`).
pub const EMAIL_SSL: &str = "xpack.notification.email.ssl.";

/// How an email account talks to its SMTP server (ES's
/// `xpack.notification.email.account.<name>.smtp.*`, JavaMail semantics).
#[derive(Clone, Debug)]
pub(crate) struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub user: Option<String>,
    pub password: Option<String>,
    /// `smtp.ssl.enable` (implicit TLS from the first byte; port 465 when unset).
    pub implicit_tls: bool,
    /// `smtp.starttls.enable`: upgrade when the server offers STARTTLS.
    pub starttls_enable: bool,
    /// `smtp.starttls.required`: refuse to continue without STARTTLS.
    pub starttls_required: bool,
    /// The host is listed in `smtp.ssl.trust` (its certificate is not verified).
    pub trusted_without_verification: bool,
    pub connection_timeout: Duration,
    pub timeout: Duration,
}

fn bool_setting(v: Option<String>, key: &str) -> Result<Option<bool>, String> {
    match v.as_deref() {
        None => Ok(None),
        Some("true") => Ok(Some(true)),
        Some("false") => Ok(Some(false)),
        Some(other) => Err(format!("Failed to parse value [{other}] for setting [{key}] as only [true] or [false] are allowed.")),
    }
}

fn smtp_config(node: &Node, account: &str, host: String) -> Result<SmtpConfig, String> {
    let get = |k: &str| account_setting(node, "email", account, k);
    let flag = |k: &str| bool_setting(get(k), &format!("xpack.notification.email.account.{account}.{k}"));
    let port: u16 = match get("smtp.port") {
        Some(p) => p.parse().map_err(|_| format!("invalid smtp port [{p}]"))?,
        None => {
            if flag("smtp.ssl.enable")? == Some(true) { 465 } else { 25 }
        }
    };
    let implicit_tls = flag("smtp.ssl.enable")?.unwrap_or(port == 465);
    let starttls_required = flag("smtp.starttls.required")?.unwrap_or(false);
    let starttls_enable = flag("smtp.starttls.enable")?.unwrap_or(false) || starttls_required;
    // JavaMail `mail.smtp.ssl.trust`: `*` or a whitespace / comma separated host list.
    let trusted_without_verification = get("smtp.ssl.trust").is_some_and(|t| t.split(|c: char| c == ',' || c.is_whitespace()).any(|h| h == "*" || h.eq_ignore_ascii_case(&host)));
    let time = |k: &str| -> Result<Duration, String> {
        match get(k) {
            None => Ok(Duration::from_secs(120)),
            Some(v) => crate::ilm::parse_time_millis(&v).map(|ms| Duration::from_millis(ms.max(1) as u64)).ok_or_else(|| format!("invalid time value [{v}] for [{k}]")),
        }
    };
    Ok(SmtpConfig {
        host,
        port,
        user: get("smtp.user"),
        password: get("smtp.password").or_else(|| get("smtp.secure_password")),
        implicit_tls,
        starttls_enable,
        starttls_required,
        trusted_without_verification,
        connection_timeout: time("smtp.connection_timeout")?,
        timeout: time("smtp.timeout")?,
    })
}

/// An SMTP connection: plain until STARTTLS, or TLS from the start.
enum SmtpStream {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
    Upgrading,
}

impl SmtpStream {
    fn is_tls(&self) -> bool {
        matches!(self, SmtpStream::Tls(_))
    }
}

impl std::io::Read for SmtpStream {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            SmtpStream::Plain(s) => s.read(b),
            SmtpStream::Tls(s) => s.read(b),
            SmtpStream::Upgrading => Err(std::io::Error::other("connection is being upgraded")),
        }
    }
}

impl Write for SmtpStream {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            SmtpStream::Plain(s) => s.write(b),
            SmtpStream::Tls(s) => s.write(b),
            SmtpStream::Upgrading => Err(std::io::Error::other("connection is being upgraded")),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            SmtpStream::Plain(s) => s.flush(),
            SmtpStream::Tls(s) => s.flush(),
            SmtpStream::Upgrading => Ok(()),
        }
    }
}

struct Smtp {
    s: SmtpStream,
    /// Bytes read but not yet consumed.
    buf: Vec<u8>,
}

impl Smtp {
    fn line(&mut self) -> Result<String, String> {
        loop {
            if let Some(i) = self.buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=i).collect();
                return Ok(String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string());
            }
            if self.buf.len() > 64 * 1024 {
                return Err("smtp reply line too long".into());
            }
            let mut chunk = [0u8; 4096];
            let n = std::io::Read::read(&mut self.s, &mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("connection closed".into());
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    /// One (possibly multi-line) reply: its code and lines.
    fn reply(&mut self) -> Result<(u16, Vec<String>), String> {
        let mut lines = Vec::new();
        loop {
            let line = self.line()?;
            let code: u16 = line.get(..3).and_then(|c| c.parse().ok()).ok_or_else(|| format!("bad smtp reply [{line}]"))?;
            let more = line.as_bytes().get(3) == Some(&b'-');
            lines.push(line);
            if !more {
                return Ok((code, lines));
            }
        }
    }

    fn expect(&mut self, codes: &[u16]) -> Result<Vec<String>, String> {
        let (c, lines) = self.reply()?;
        if codes.contains(&c) { Ok(lines) } else { Err(lines.join(" ")) }
    }

    fn send(&mut self, cmd: &str) -> Result<(), String> {
        self.s.write_all(cmd.as_bytes()).and_then(|_| self.s.write_all(b"\r\n")).and_then(|_| self.s.flush()).map_err(|e| e.to_string())
    }

    /// EHLO; the extension keywords the server advertises (upper case).
    fn ehlo(&mut self) -> Result<Vec<String>, String> {
        self.send("EHLO watcher")?;
        let lines = self.expect(&[250])?;
        Ok(lines.iter().skip(1).map(|l| l.get(4..).unwrap_or("").to_ascii_uppercase()).collect())
    }

    /// Switch the connection to TLS (verified per `tls`).
    fn start_tls(&mut self, host: &str, tls: Arc<rustls::ClientConfig>) -> Result<(), String> {
        // Nothing the server sent before the handshake may be taken as
        // coming over TLS (STARTTLS response injection).
        if !self.buf.is_empty() {
            return Err("unexpected data before the TLS handshake".into());
        }
        let SmtpStream::Plain(tcp) = std::mem::replace(&mut self.s, SmtpStream::Upgrading) else {
            return Err("connection is already encrypted".into());
        };
        let name = rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|e| format!("invalid TLS server name [{host}]: {e}"))?;
        let conn = rustls::ClientConnection::new(tls, name).map_err(|e| e.to_string())?;
        let mut stream = rustls::StreamOwned::new(conn, tcp);
        while stream.conn.is_handshaking() {
            stream.conn.complete_io(&mut stream.sock).map_err(|e| format!("TLS handshake with [{host}] failed: {e}"))?;
        }
        self.s = SmtpStream::Tls(Box::new(stream));
        Ok(())
    }
}

/// Send one message. TLS (implicit or STARTTLS) is established and its
/// certificate verified before any credential is sent; with
/// `smtp.starttls.required` a server that does not offer STARTTLS fails the
/// send before authentication.
fn smtp_send(cfg: &SmtpConfig, tls: &dyn Fn() -> Result<Arc<rustls::ClientConfig>, String>, from: &str, rcpt: &[String], data: &str) -> Result<(), String> {
    use std::net::ToSocketAddrs;
    let addr = (cfg.host.as_str(), cfg.port).to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("unknown smtp host")?;
    let tcp = TcpStream::connect_timeout(&addr, cfg.connection_timeout).map_err(|e| e.to_string())?;
    tcp.set_read_timeout(Some(cfg.timeout)).ok();
    tcp.set_write_timeout(Some(cfg.timeout)).ok();
    let mut c = Smtp { s: SmtpStream::Plain(tcp), buf: Vec::new() };
    if cfg.implicit_tls {
        c.start_tls(&cfg.host, tls()?)?;
    }
    c.expect(&[220])?;
    let mut ext = c.ehlo()?;
    if !c.s.is_tls() && cfg.starttls_enable {
        if ext.iter().any(|e| e == "STARTTLS") {
            c.send("STARTTLS")?;
            c.expect(&[220])?;
            c.start_tls(&cfg.host, tls()?)?;
            ext = c.ehlo()?;
        } else if cfg.starttls_required {
            let _ = c.send("QUIT");
            return Err(format!("STARTTLS is required but host [{}] does not support STARTTLS", cfg.host));
        }
    }
    if cfg.starttls_required && !c.s.is_tls() {
        let _ = c.send("QUIT");
        return Err("STARTTLS is required but the connection is not encrypted".into());
    }
    if let (Some(u), Some(p)) = (&cfg.user, &cfg.password) {
        if !ext.iter().any(|e| e.starts_with("AUTH") && e.split_whitespace().any(|m| m == "PLAIN")) && ext.iter().any(|e| e.starts_with("AUTH")) {
            let _ = c.send("QUIT");
            return Err("the server does not support the PLAIN authentication mechanism".into());
        }
        let tok = base64::engine::general_purpose::STANDARD.encode(format!("\0{u}\0{p}"));
        c.send(&format!("AUTH PLAIN {tok}"))?;
        c.expect(&[235])?;
    }
    c.send(&format!("MAIL FROM:<{from}>"))?;
    c.expect(&[250])?;
    for t in rcpt {
        c.send(&format!("RCPT TO:<{t}>"))?;
        c.expect(&[250, 251])?;
    }
    c.send("DATA")?;
    c.expect(&[354])?;
    let stuffed = data.replace("\r\n.", "\r\n..");
    c.send(&format!("{stuffed}\r\n."))?;
    c.expect(&[250])?;
    let _ = c.send("QUIT");
    Ok(())
}

// ---------------------------------------------------------------------------
// slack / pagerduty / jira
// ---------------------------------------------------------------------------

fn post_json(node: &Node, url: &str, body: &Value, auth: Option<(String, String)>) -> Result<(Value, Value), String> {
    let mut req = HttpRequest::from_json(&json!({"url": url, "method": "post", "headers": {"Content-Type": "application/json"}, "body": body.to_string()}), &|s| s.to_string())?;
    req.auth = auth;
    check_whitelist(node, &req)?;
    let resp = httpc::send(node, &req)?;
    let rj = json!({"status": resp.status, "body": resp.body_str()});
    if resp.status >= 400 {
        return Err(format!("received [{}] status code: {}", resp.status, resp.body_str()));
    }
    Ok((req.to_json(), rj))
}

fn slack(node: &Node, d: &Value, cj: &Value, simulate: bool) -> Single {
    let account = find_account(node, "slack", d["account"].as_str());
    let m = &d["message"];
    let tos: Vec<String> = match m.get("to") {
        Some(Value::String(s)) => s.split(',').map(|x| x.trim().to_string()).collect(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        _ => Vec::new(),
    };
    let mut base = Map::new();
    if let Some(f) = m["from"].as_str() {
        base.insert("username".into(), json!(render(f, cj)));
    }
    if let Some(t) = m["text"].as_str() {
        base.insert("text".into(), json!(render(t, cj)));
    }
    if let Some(i) = m["icon"].as_str() {
        base.insert(if i.starts_with("http") { "icon_url" } else { "icon_emoji" }.into(), json!(i));
    }
    if let Some(a) = m.get("attachments") {
        base.insert("attachments".into(), render_value(a, cj));
    }
    let channels: Vec<Option<String>> = if tos.is_empty() { vec![None] } else { tos.into_iter().map(Some).collect() };
    let mut sent = Vec::new();
    let mut failure: Option<String> = None;
    for ch in channels {
        let mut payload = base.clone();
        if let Some(c) = &ch {
            payload.insert("channel".into(), json!(c));
        }
        let payload = Value::Object(payload);
        let mut entry = json!({"message": payload});
        if let Some(c) = &ch {
            entry["to"] = json!(c);
        }
        if simulate {
            entry["status"] = json!("simulated");
        } else {
            let url = account.as_ref().and_then(|a| account_setting(node, "slack", a, "url").or_else(|| account_setting(node, "slack", a, "secure_url")));
            match url {
                None => {
                    failure = Some(format!("failed to send slack message: account [{}] not found", account.clone().unwrap_or_else(|| "default".into())));
                    entry["status"] = json!("failure");
                }
                Some(u) => match post_json(node, &u, &payload, None) {
                    Ok((rq, rs)) => {
                        entry["status"] = json!("success");
                        entry["request"] = rq;
                        entry["response"] = rs;
                    }
                    Err(e) => {
                        failure = Some(e.clone());
                        entry["status"] = json!("failure");
                        entry["reason"] = json!(e);
                    }
                },
            }
        }
        sent.push(entry);
    }
    let mut body = json!({"sent_messages": sent});
    if let Some(a) = account {
        body["account"] = json!(a);
    }
    match failure {
        Some(f) => fail(f, body),
        None => ok(simulate, body),
    }
}

fn pagerduty(node: &Node, d: &Value, cj: &Value, simulate: bool) -> Single {
    let account = find_account(node, "pagerduty", d["account"].as_str());
    let mut ev = Map::new();
    let key = account.as_ref().and_then(|a| account_setting(node, "pagerduty", a, "service_api_key").or_else(|| account_setting(node, "pagerduty", a, "secure_service_api_key")));
    if let Some(k) = &key {
        ev.insert("service_key".into(), json!(k));
    }
    ev.insert("event_type".into(), json!(d["event_type"].as_str().unwrap_or("trigger")));
    ev.insert("description".into(), json!(render(d["description"].as_str().unwrap_or(""), cj)));
    for k in ["incident_key", "client", "client_url"] {
        if let Some(s) = d[k].as_str() {
            ev.insert(k.into(), json!(render(s, cj)));
        }
    }
    if d["attach_payload"] == true {
        ev.insert("details".into(), json!({"watch_id": cj["watch_id"], "payload": cj["payload"]}));
    }
    if let Some(c) = d.get("contexts") {
        ev.insert("contexts".into(), render_value(c, cj));
    }
    let event = Value::Object(ev);
    let mut body = json!({"event": event, "account": account});
    if simulate {
        return ok(true, body);
    }
    let url = account.as_ref().and_then(|a| account_setting(node, "pagerduty", a, "url")).unwrap_or_else(|| "http://events.pagerduty.com/generic/2010-04-15/create_event.json".into());
    match post_json(node, &url, &event, None) {
        Ok((rq, rs)) => {
            body["sent_event"] = json!({"event": event, "request": rq, "response": rs});
            ok(false, body)
        }
        Err(e) => fail(e, body),
    }
}

fn jira(node: &Node, d: &Value, cj: &Value, simulate: bool) -> Single {
    let account = find_account(node, "jira", d["account"].as_str());
    let fields = render_value(&d["fields"], cj);
    let payload = json!({"fields": fields});
    let mut body = json!({"account": account, "fields": fields});
    if simulate {
        return ok(true, body);
    }
    let Some(a) = account.as_ref() else {
        return fail("failed to create jira issue: no jira account configured", body);
    };
    let Some(url) = account_setting(node, "jira", a, "url") else {
        return fail(format!("failed to create jira issue: account [{a}] not found"), body);
    };
    let auth = account_setting(node, "jira", a, "user").map(|u| (u, account_setting(node, "jira", a, "password").or_else(|| account_setting(node, "jira", a, "secure_password")).unwrap_or_default()));
    let url = format!("{}/rest/api/2/issue", url.trim_end_matches('/'));
    match post_json(node, &url, &payload, auth) {
        Ok((rq, rs)) => {
            body["result"] = json!({"request": rq, "response": rs});
            ok(false, body)
        }
        Err(e) => fail(e, body),
    }
}

#[cfg(test)]
mod remote_index_tests {
    use super::*;

    #[test]
    fn index_bulk_body_lines() {
        let b = index_bulk_body(&[("i".into(), "1".into(), json!({"a": 1})), ("j".into(), "1".into(), json!({"b": 2}))], true);
        let lines: Vec<Value> = b.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines, vec![json!({"create": {"_index": "i", "_id": "1"}}), json!({"a": 1}), json!({"create": {"_index": "j", "_id": "1"}}), json!({"b": 2})]);
    }

    fn ctx(payload: Value) -> Ctx {
        Ctx { id: "i".into(), watch_id: "w".into(), execution_time: 1_000_000, triggered: super::super::iso(0), scheduled: super::super::iso(0), metadata: json!({}), payload, vars: json!({}), data: Default::default(), identity: None }
    }

    /// Finding 205: each `_doc` element goes to its own `_index`, so two
    /// documents sharing an `_id` in different indices both survive; an
    /// element `_index` together with the action's `index` is ES's error.
    #[test]
    fn multi_document_index_action_routes_each_document() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let payload = json!({"_doc": [{"_index": "a", "_id": "same", "v": 1}, {"_index": "b", "_id": "same", "v": 2}]});
        let r = index(&node, &json!({"refresh": "true"}), &ctx(payload.clone()), &json!({}), false);
        assert_eq!(r.0, "success", "{:?}", r);
        for (idx, v) in [("a", 1), ("b", 2)] {
            let g = node.get_index(idx).unwrap_or_else(|| panic!("{idx} missing")).get("same").unwrap().unwrap();
            let src: Value = serde_json::from_str(g.source.as_deref().unwrap()).unwrap();
            assert_eq!(src, json!({"v": v}), "{idx}");
        }
        let r = index(&node, &json!({"index": "c"}), &ctx(payload), &json!({}), false);
        assert_eq!(r.0, "failure");
        assert!(r.2.as_deref().is_some_and(|m| m.contains("were set together with action [index] field")), "{:?}", r);
    }

    /// ES `ExecutableIndexAction` simulate: a `_doc` list reports each
    /// request's `_id` / `_index`; one document reports its own request.
    #[test]
    fn simulated_index_action_reports_requests() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let r = index(&node, &json!({}), &ctx(json!({"_doc": [{"_id": "the-id", "_index": "the-index", "a": "b"}]})), &json!({}), true);
        assert_eq!(r.0, "simulated", "{r:?}");
        assert_eq!(r.1, json!({"request": {"index": "", "doc_id": "", "source": [{"_id": "the-id", "_index": "the-index"}]}}));
        let r = index(&node, &json!({"index": "i", "refresh": "wait_for"}), &ctx(json!({"_id": "x", "a": 1})), &json!({}), true);
        assert_eq!(r.1, json!({"request": {"index": "i", "doc_id": "x", "refresh": "wait_for", "source": {"a": 1}}}));
        assert!(node.get_index("the-index").is_none() && node.get_index("i").is_none());
    }
}

#[cfg(test)]
mod smtp_tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::Mutex;

    fn data(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../elasticsearch-ssl/tests/data").join(name)
    }

    fn server_tls() -> Arc<rustls::ServerConfig> {
        let settings = [
            ("xpack.security.http.ssl.enabled", "true".to_string()),
            ("xpack.security.http.ssl.certificate", data("node.crt").display().to_string()),
            ("xpack.security.http.ssl.key", data("node.key").display().to_string()),
        ];
        let get = |k: &str| settings.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
        elasticsearch_ssl::http_server(&get, &|_| Vec::new(), &data("")).unwrap().unwrap()
    }

    /// A client config trusting the fixture CA (`ca`) or the default roots.
    fn client_tls(ca: bool) -> Arc<rustls::ClientConfig> {
        let mut settings = Map::new();
        if ca {
            settings.insert(format!("{EMAIL_SSL}certificate_authorities"), json!([data("ca.crt").display().to_string()]));
        }
        crate::remote::client_tls(&settings, EMAIL_SSL, false).unwrap()
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        /// Plain SMTP that never offers STARTTLS.
        Plain,
        /// Plain SMTP offering STARTTLS.
        StartTls,
        /// TLS from the first byte.
        Implicit,
    }

    trait Io: Read + Write {}
    impl<T: Read + Write> Io for T {}

    /// What the server received: `(over TLS, line)`.
    type Log = Arc<Mutex<Vec<(bool, String)>>>;

    fn read_line(s: &mut dyn Io) -> Option<String> {
        let mut out = Vec::new();
        let mut b = [0u8; 1];
        loop {
            match s.read(&mut b) {
                Ok(1) => {
                    out.push(b[0]);
                    if b[0] == b'\n' {
                        return Some(String::from_utf8_lossy(&out).trim_end().to_string());
                    }
                }
                _ => return None,
            }
        }
    }

    /// Serve one session on `s`; `Some(())` when the client asked for STARTTLS.
    fn session(s: &mut dyn Io, log: &Log, tls: bool, offer_starttls: bool, greet: bool) -> Option<()> {
        if greet {
            s.write_all(b"220 fake ESMTP\r\n").ok()?;
        }
        let mut in_data = false;
        loop {
            let line = read_line(s)?;
            log.lock().unwrap().push((tls, line.clone()));
            if in_data {
                if line == "." {
                    in_data = false;
                    s.write_all(b"250 queued\r\n").ok()?;
                }
                continue;
            }
            let cmd = line.to_ascii_uppercase();
            let reply: &[u8] = if cmd.starts_with("EHLO") {
                if offer_starttls && !tls { b"250-fake\r\n250-STARTTLS\r\n250 AUTH PLAIN\r\n" } else { b"250-fake\r\n250 AUTH PLAIN\r\n" }
            } else if cmd == "STARTTLS" {
                s.write_all(b"220 go ahead\r\n").ok()?;
                return Some(());
            } else if cmd.starts_with("AUTH") {
                b"235 ok\r\n"
            } else if cmd == "DATA" {
                in_data = true;
                b"354 go\r\n"
            } else if cmd == "QUIT" {
                let _ = s.write_all(b"221 bye\r\n");
                return None;
            } else {
                b"250 ok\r\n"
            };
            s.write_all(reply).ok()?;
            s.flush().ok()?;
        }
    }

    fn fake_smtp(mode: Mode) -> (u16, Log) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let log: Log = Arc::default();
        let log2 = log.clone();
        std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            c.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let upgrade = |c: TcpStream| rustls::StreamOwned::new(rustls::ServerConnection::new(server_tls()).unwrap(), c);
            match mode {
                Mode::Implicit => {
                    let mut t = upgrade(c);
                    session(&mut t, &log2, true, false, true);
                }
                Mode::Plain => {
                    session(&mut c, &log2, false, false, true);
                }
                Mode::StartTls => {
                    if session(&mut c, &log2, false, true, true).is_some() {
                        let mut t = upgrade(c);
                        session(&mut t, &log2, true, false, false);
                    }
                }
            }
        });
        (port, log)
    }

    fn cfg(port: u16, implicit_tls: bool, starttls_enable: bool, starttls_required: bool) -> SmtpConfig {
        SmtpConfig {
            host: "localhost".into(),
            port,
            user: Some("watcher".into()),
            password: Some("s3cret-password".into()),
            implicit_tls,
            starttls_enable,
            starttls_required,
            trusted_without_verification: false,
            connection_timeout: Duration::from_secs(5),
            timeout: Duration::from_secs(5),
        }
    }

    const MAIL: &str = "Subject: hi\r\n\r\nbody line";

    fn leaked(log: &Log) -> bool {
        let tok = base64::engine::general_purpose::STANDARD.encode("\0watcher\0s3cret-password");
        log.lock().unwrap().iter().any(|(_, l)| l.to_ascii_uppercase().starts_with("AUTH") || l.contains(&tok))
    }

    #[test]
    fn required_starttls_without_server_support_sends_no_credentials() {
        let (port, log) = fake_smtp(Mode::Plain);
        let e = smtp_send(&cfg(port, false, true, true), &|| Ok(client_tls(true)), "a@x", &["b@x".into()], MAIL).unwrap_err();
        assert!(e.contains("STARTTLS is required"), "{e}");
        std::thread::sleep(Duration::from_millis(100));
        assert!(!leaked(&log), "{:?}", log.lock().unwrap());
    }

    #[test]
    fn starttls_upgrades_before_authenticating() {
        let (port, log) = fake_smtp(Mode::StartTls);
        smtp_send(&cfg(port, false, true, true), &|| Ok(client_tls(true)), "a@x", &["b@x".into()], MAIL).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let log = log.lock().unwrap();
        let auth = log.iter().find(|(_, l)| l.starts_with("AUTH")).expect("authenticated");
        assert!(auth.0, "AUTH must travel over TLS: {log:?}");
        assert!(log.iter().any(|(tls, l)| *tls && l == "body line"), "{log:?}");
        assert!(log.iter().filter(|(tls, _)| !tls).all(|(_, l)| !l.starts_with("AUTH") && !l.starts_with("MAIL")), "{log:?}");
    }

    #[test]
    fn implicit_tls_delivers_to_a_trusted_server() {
        let (port, log) = fake_smtp(Mode::Implicit);
        smtp_send(&cfg(port, true, false, false), &|| Ok(client_tls(true)), "a@x", &["b@x".into()], MAIL).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let log = log.lock().unwrap();
        assert!(log.iter().all(|(tls, _)| *tls), "{log:?}");
        assert!(log.iter().any(|(_, l)| l == "Subject: hi"), "{log:?}");
    }

    #[test]
    fn untrusted_certificates_are_refused_before_authentication() {
        let (port, log) = fake_smtp(Mode::Implicit);
        let e = smtp_send(&cfg(port, true, false, false), &|| Ok(client_tls(false)), "a@x", &["b@x".into()], MAIL).unwrap_err();
        assert!(e.contains("TLS handshake"), "{e}");
        let (port, log2) = fake_smtp(Mode::StartTls);
        let e = smtp_send(&cfg(port, false, true, false), &|| Ok(client_tls(false)), "a@x", &["b@x".into()], MAIL).unwrap_err();
        assert!(e.contains("TLS handshake"), "{e}");
        std::thread::sleep(Duration::from_millis(100));
        assert!(!leaked(&log) && !leaked(&log2));
    }
}
