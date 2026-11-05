//! Watch actions: logging, index, webhook, email, slack, pagerduty, jira.

use super::exec::{path_get, render, render_value, Ctx};
use super::httpc::{self, HttpRequest};
use super::{iso, put_doc, setting};
use crate::node::Node;
use crate::shard::WriteCond;
use base64::Engine;
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
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
        let list = path_get(&cj, key).and_then(Value::as_array).cloned().unwrap_or_default();
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
    let Some(index_name) = target.or_else(|| docs.first().and_then(|d| d["_index"].as_str().map(str::to_string))) else {
        return fail("unable to index document: no index specified", json!({}));
    };
    if simulate {
        let src = if multi { json!(docs) } else { docs[0].clone() };
        let mut rq = Map::new();
        rq.insert("index".into(), json!(index_name));
        if let Some(i) = &doc_id {
            rq.insert("doc_id".into(), json!(i));
        }
        rq.insert("source".into(), src);
        return ok(true, json!({"request": rq}));
    }
    if !super::exec::user_allowed(node, ctx.user.as_deref(), "indices:data/write/bulk[s]", &index_name) {
        let u = ctx.user.clone().unwrap_or_default();
        let roles = node.security.lookup_user(&u).map(|a| node.security.permission(&a).effective_roles.join(",")).unwrap_or_default();
        return fail(format!("action [indices:data/write/bulk[s]] is unauthorized for user [{u}] with effective roles [{roles}] on indices [{index_name}], this action is granted by the index privileges [create_doc,create,delete,index,write,all]"), json!({}));
    }
    let create = d["op_type"].as_str() == Some("create");
    let refresh = matches!(d["refresh"].as_str(), Some("true" | "wait_for")) || d["refresh"] == true;
    if let Some(rest) = super::remote() {
        return index_remote(&rest, &index_name, doc_id, docs, create, refresh, multi);
    }
    let svc = match node.write_index(&index_name, true) {
        Ok(s) => s,
        Err(e) => return fail(e.reason.clone(), json!({})),
    };
    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for mut doc in docs {
        let mut id = doc_id.clone();
        if let Some(o) = doc.as_object_mut() {
            if let Some(Value::String(i)) = o.remove("_id") {
                id.get_or_insert(i);
            }
            o.remove("_index");
        }
        let id = id.unwrap_or_else(crate::index::auto_id);
        match put_doc(&svc, &id, &doc, WriteCond { create, ..Default::default() }) {
            Ok((version, _, _, created)) => successful.push(json!({"created": created, "result": if created { "created" } else { "updated" }, "id": id, "version": version, "index": svc.name})),
            Err(e) => failed.push(json!({"id": id, "type": e.kind, "reason": e.reason})),
        }
    }
    if refresh {
        let _ = svc.refresh();
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

/// The `_bulk` body of an index action's documents: `(id, source)` each.
fn index_bulk_body(index: &str, docs: &[(String, Value)], create: bool) -> String {
    let op = if create { "create" } else { "index" };
    let mut out = String::new();
    for (id, doc) in docs {
        out.push_str(&json!({ op: {"_index": index, "_id": id} }).to_string());
        out.push('\n');
        out.push_str(&doc.to_string());
        out.push('\n');
    }
    out
}

/// The index action on a cluster: one `_bulk` through the coordinator.
fn index_remote(rest: &super::RestCall, index_name: &str, doc_id: Option<String>, docs: Vec<Value>, create: bool, refresh: bool, multi: bool) -> Single {
    let mut items: Vec<(String, Value)> = Vec::new();
    for mut doc in docs {
        let mut id = doc_id.clone();
        if let Some(o) = doc.as_object_mut() {
            if let Some(Value::String(i)) = o.remove("_id") {
                id.get_or_insert(i);
            }
            o.remove("_index");
        }
        items.push((id.unwrap_or_else(crate::index::auto_id), doc));
    }
    let uri = if refresh { "/_bulk?refresh=true" } else { "/_bulk" };
    let resp = match rest("POST", uri, Some(Value::String(index_bulk_body(index_name, &items, create)))) {
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
    match httpc::send(&r) {
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
    let port: u16 = account_setting(node, "email", &account, "smtp.port").and_then(|p| p.parse().ok()).unwrap_or(25);
    let user = account_setting(node, "email", &account, "smtp.user");
    let pass = account_setting(node, "email", &account, "smtp.password").or_else(|| account_setting(node, "email", &account, "smtp.secure_password"));
    let Some(from) = from else {
        return fail(format!("failed to send email with subject [{subject}] via account [{account}]: no from address"), result);
    };
    let mut all_rcpt = to.clone();
    all_rcpt.extend(cc.clone());
    all_rcpt.extend(bcc.clone());
    let raw = build_mime(&from, &to, &cc, &reply_to, &subject, text.as_deref(), html.as_deref(), &ctx.id, attach_json.as_ref());
    match smtp_send(&host, port, user.as_deref(), pass.as_deref(), &from, &all_rcpt, &raw) {
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

fn smtp_send(host: &str, port: u16, user: Option<&str>, pass: Option<&str>, from: &str, rcpt: &[String], data: &str) -> Result<(), String> {
    use std::net::ToSocketAddrs;
    let addr = (host, port).to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("unknown smtp host")?;
    let s = TcpStream::connect_timeout(&addr, Duration::from_secs(10)).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut w = s.try_clone().map_err(|e| e.to_string())?;
    let mut r = BufReader::new(s);
    fn read_reply(r: &mut BufReader<TcpStream>) -> Result<(u16, String), String> {
        let mut full = String::new();
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("connection closed".into());
            }
            full.push_str(&line);
            if line.len() < 4 || line.as_bytes()[3] != b'-' {
                let code = line.get(..3).and_then(|c| c.parse().ok()).ok_or("bad smtp reply")?;
                return Ok((code, full));
            }
        }
    }
    let expect = |r: &mut BufReader<TcpStream>, codes: &[u16]| -> Result<(), String> {
        let (c, t) = read_reply(r)?;
        if codes.contains(&c) { Ok(()) } else { Err(t.trim().to_string()) }
    };
    expect(&mut r, &[220])?;
    w.write_all(b"EHLO watcher\r\n").map_err(|e| e.to_string())?;
    expect(&mut r, &[250])?;
    if let (Some(u), Some(p)) = (user, pass) {
        let tok = base64::engine::general_purpose::STANDARD.encode(format!("\0{u}\0{p}"));
        w.write_all(format!("AUTH PLAIN {tok}\r\n").as_bytes()).map_err(|e| e.to_string())?;
        expect(&mut r, &[235])?;
    }
    w.write_all(format!("MAIL FROM:<{from}>\r\n").as_bytes()).map_err(|e| e.to_string())?;
    expect(&mut r, &[250])?;
    for t in rcpt {
        w.write_all(format!("RCPT TO:<{t}>\r\n").as_bytes()).map_err(|e| e.to_string())?;
        expect(&mut r, &[250, 251])?;
    }
    w.write_all(b"DATA\r\n").map_err(|e| e.to_string())?;
    expect(&mut r, &[354])?;
    let stuffed = data.replace("\r\n.", "\r\n..");
    w.write_all(format!("{stuffed}\r\n.\r\n").as_bytes()).map_err(|e| e.to_string())?;
    expect(&mut r, &[250])?;
    let _ = w.write_all(b"QUIT\r\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// slack / pagerduty / jira
// ---------------------------------------------------------------------------

fn post_json(node: &Node, url: &str, body: &Value, auth: Option<(String, String)>) -> Result<(Value, Value), String> {
    let mut req = HttpRequest::from_json(&json!({"url": url, "method": "post", "headers": {"Content-Type": "application/json"}, "body": body.to_string()}), &|s| s.to_string())?;
    req.auth = auth;
    check_whitelist(node, &req)?;
    let resp = httpc::send(&req)?;
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
        let b = index_bulk_body("i", &[("1".into(), json!({"a": 1}))], true);
        let lines: Vec<Value> = b.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines, vec![json!({"create": {"_index": "i", "_id": "1"}}), json!({"a": 1})]);
    }
}
