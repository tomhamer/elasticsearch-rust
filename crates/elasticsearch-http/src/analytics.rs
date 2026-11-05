//! Behavioral analytics (`/_application/analytics`, x-pack ent-search):
//! collection CRUD and event ingestion.
//!
//! A collection is its event data stream `behavioral_analytics-events-{name}`
//! (it exists exactly when the data stream does), created from the built-in
//! `behavioral_analytics-events-default` template, which is installed on
//! first use. Posted events are validated like ES's event parsers, enriched
//! with the session's user agent / client address, and appended to the data
//! stream (ES does this asynchronously; failures never reach the caller).

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::license::{self, Mode};
use elasticsearch_engine::node::Node;
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::sync::Arc;

const DS_PREFIX: &str = "behavioral_analytics-events-";
const TEMPLATE: &str = "behavioral_analytics-events-default";
const TEMPLATE_VERSION: i64 = 3;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let m = req.method.clone();
    let segs: Vec<String> = req.segs.clone();
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    match s.as_slice() {
        [_, _] => match m {
            Method::GET => blocking(move || get(&node, None)).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        [_, _, name] => {
            let name = name.to_string();
            match m {
                Method::GET => blocking(move || get(&node, Some(&name))).await,
                Method::PUT => blocking(move || put(&node, &req, &name)).await,
                Method::DELETE => blocking(move || delete(&node, &name)).await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        [_, _, name, "event", ty] => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let (name, ty) = (name.to_string(), ty.to_string());
            blocking(move || post_event(&node, &req, &name, &ty)).await
        }
        _ => crate::not_found_handler(&req),
    }
}

fn check_license(node: &Node) -> Result<(), EsError> {
    license::check_ent_search(node, "behavioral analytics", Mode::Platinum)
}

pub(crate) fn collections(node: &Node, pattern: &str) -> Vec<String> {
    node.data_streams_matching(&format!("{DS_PREFIX}{pattern}"), true).into_iter().filter_map(|d| d.name.strip_prefix(DS_PREFIX).map(str::to_string)).collect()
}

fn get(node: &Node, name: Option<&str>) -> R {
    check_license(node)?;
    let mut names: Vec<String> = Vec::new();
    match name {
        None | Some("_all") | Some("*") => names = collections(node, "*"),
        Some(expr) => {
            for part in expr.split(',').filter(|p| !p.is_empty()) {
                if part.contains('*') {
                    names.extend(collections(node, part));
                } else if node.data_stream(&format!("{DS_PREFIX}{part}")).is_some() {
                    names.push(part.to_string());
                } else {
                    return Err(EsError::resource_not_found(format!("no such analytics collection [{part}] ")));
                }
            }
        }
    }
    let mut out = Map::new();
    for n in names {
        out.insert(n.clone(), json!({"event_data_stream": {"name": format!("{DS_PREFIX}{n}")}}));
    }
    Ok(Resp::ok(Value::Object(out)))
}

/// `MetadataCreateDataStreamService.validateDataStreamName`.
fn validate_ds_name(ds: &str) -> Result<(), EsError> {
    if ds.chars().any(|c| matches!(c, ' ' | '"' | '*' | ',' | '/' | '<' | '>' | '?' | '\\' | '|')) {
        return Err(EsError::illegal_argument(format!("data_stream [{ds}] must not contain the following characters [' ','\"','*',',','/','<','>','?','\\','|']")));
    }
    if ds.contains('#') {
        return Err(EsError::illegal_argument(format!("data_stream [{ds}] must not contain '#'")));
    }
    if ds.contains(':') {
        return Err(EsError::illegal_argument(format!("data_stream [{ds}] must not contain ':'")));
    }
    if ds.to_lowercase() != ds {
        return Err(EsError::illegal_argument(format!("data_stream [{ds}] must be lowercase")));
    }
    Ok(())
}

fn put(node: &Node, req: &Req, name: &str) -> R {
    if !req.body.is_empty() {
        return Err(EsError::illegal_argument(format!("request [{} {}] does not support having a body", req.method, req.path)));
    }
    check_license(node)?;
    let ds = format!("{DS_PREFIX}{name}");
    let wrap = |e: EsError| EsError::new(400, "status_exception", format!("error while creating analytics collection [{name}]")).caused_by(e);
    validate_ds_name(&ds).map_err(wrap)?;
    if node.data_stream(&ds).is_some() {
        return Err(EsError::new(400, "resource_already_exists_exception", format!("analytics collection [{name}] already exists")));
    }
    install_templates(node)?;
    match node.create_data_stream(&ds) {
        Ok(_) => {}
        Err(e) if e.kind == "resource_already_exists_exception" => {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("analytics collection [{name}] already exists")));
        }
        Err(e) => return Err(wrap(e)),
    }
    Ok(Resp::json(201, json!({"acknowledged": true, "name": name})))
}

fn delete(node: &Node, name: &str) -> R {
    check_license(node)?;
    let missing = || EsError::resource_not_found(format!("analytics collection [{name}] does not exists"));
    let found = if name.contains('*') { !collections(node, name).is_empty() } else { !name.contains(',') && node.data_stream(&format!("{DS_PREFIX}{name}")).is_some() };
    if !found {
        return Err(missing());
    }
    node.delete_data_streams(&format!("{DS_PREFIX}{name}"))?;
    Ok(Resp::ack())
}

/// The built-in templates ES's `AnalyticsTemplateRegistry` installs. The
/// final pipeline (user agent / geoip enrichment) is applied by
/// [`enrich`] instead, so the settings omit `index.final_pipeline`.
fn install_templates(node: &Node) -> Result<(), EsError> {
    if node.templates.read().index.contains_key(TEMPLATE) {
        return Ok(());
    }
    let kw = json!({"ignore_above": 1024, "type": "keyword"});
    let url = json!({"properties": {"path": kw, "extension": kw, "fragment": kw, "original": kw, "scheme": kw, "port": {"type": "long"}, "domain": kw, "query": kw}});
    let mappings = json!({"dynamic": "strict", "properties": {
        "search": {"properties": {"query": kw, "filters": {"type": "flattened"}, "page": {"properties": {"current": {"type": "integer"}, "size": {"type": "integer"}}},
            "sort": {"properties": {"name": kw, "direction": kw}}, "results": {"properties": {"items": {"type": "object", "enabled": false}, "total_results": {"type": "integer"}}},
            "search_application": kw}},
        "@timestamp": {"type": "date"},
        "data_stream": {"properties": {"namespace": {"type": "keyword"}, "type": {"type": "keyword"}, "dataset": {"type": "keyword"}}},
        "session": {"properties": {
            "location": {"properties": {"continent_name": kw, "region_iso_code": kw, "city_name": kw, "country_iso_code": kw, "country_name": kw, "region_name": kw, "location": {"type": "geo_point"}}},
            "id": kw,
            "user_agent": {"properties": {"os": {"properties": {"name": kw, "version": kw, "full": kw}}, "browser": {"properties": {"name": kw, "version": kw}}, "device": {"properties": {"name": kw, "type": kw}}}}}},
        "document": {"properties": {"index": kw, "id": kw}},
        "page": {"properties": {"referrer": url, "title": kw, "url": url}},
        "event": {"properties": {"action": kw, "source": kw}},
        "user": {"properties": {"id": kw}},
        "tags": kw,
    }});
    let settings = json!({
        "template": {"settings": {"index": {"codec": "best_compression", "sort": {"field": ["session.id", "@timestamp"], "order": ["asc", "asc"]}, "number_of_shards": "1", "auto_expand_replicas": "0-1"}},
                     "lifecycle": {"enabled": true, "data_retention": "180d"}},
        "version": TEMPLATE_VERSION,
        "_meta": {"managed": true, "description": "Built-in settings applied by default to behavioral analytics event data streams."},
    });
    let mappings = json!({"template": {"mappings": mappings}, "version": TEMPLATE_VERSION,
        "_meta": {"managed": true, "description": "Built-in mapping applied by default to behavioral analytics event data streams."}});
    let index = json!({
        "index_patterns": [format!("{DS_PREFIX}*")],
        "composed_of": ["behavioral_analytics-events-settings", "behavioral_analytics-events-mappings"],
        "priority": 100, "version": TEMPLATE_VERSION,
        "_meta": {"managed": true, "description": "Built-in template applied by default to behavioral analytics event data streams."},
        "data_stream": {"hidden": false, "allow_custom_routing": true},
        "allow_auto_create": false,
    });
    {
        let mut t = node.templates.write();
        t.component.entry("behavioral_analytics-events-settings".into()).or_insert(settings);
        t.component.entry("behavioral_analytics-events-mappings".into()).or_insert(mappings);
        t.index.entry(TEMPLATE.into()).or_insert(index);
    }
    node.persist_state()
}

// ------------------------------------------------------------ events

/// A JSON value with the 1-based `(line, col)` of its first token and of
/// its last token (the closing bracket of an object / array).
struct PV {
    v: Value,
    start: (usize, usize),
    end: (usize, usize),
    /// Object members: key, key position, value.
    members: Vec<(String, (usize, usize), PV)>,
}

/// A small JSON parser that keeps token positions (Jackson's `[line:col]`).
struct Pos<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl<'a> Pos<'a> {
    fn lc(&self, at: usize) -> (usize, usize) {
        let before = &self.text[..at.min(self.text.len())];
        let line = before.matches('\n').count() + 1;
        let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
        (line, col)
    }
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Option<PV> {
        self.ws();
        let at = self.i;
        match *self.s.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut members = Vec::new();
                let mut obj = Map::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b'}') {
                        let end = self.lc(self.i);
                        self.i += 1;
                        return Some(PV { v: Value::Object(obj), start: self.lc(at), end, members });
                    }
                    let kpos = self.lc(self.i);
                    let k = match self.value()?.v {
                        Value::String(k) => k,
                        _ => return None,
                    };
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return None;
                    }
                    self.i += 1;
                    let v = self.value()?;
                    obj.insert(k.clone(), v.v.clone());
                    members.push((k, kpos, v));
                    self.ws();
                    if self.s.get(self.i) == Some(&b',') {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut arr = Vec::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b']') {
                        let end = self.lc(self.i);
                        self.i += 1;
                        return Some(PV { v: Value::Array(arr), start: self.lc(at), end, members: Vec::new() });
                    }
                    arr.push(self.value()?.v);
                    self.ws();
                    if self.s.get(self.i) == Some(&b',') {
                        self.i += 1;
                    }
                }
            }
            _ => {
                // Scalars: let serde parse the longest prefix that is a value.
                let mut de = serde_json::Deserializer::from_str(&self.text[self.i..]).into_iter::<Value>();
                let v = de.next()?.ok()?;
                self.i += de.byte_offset();
                let p = self.lc(at);
                Some(PV { v, start: p, end: p, members: Vec::new() })
            }
        }
    }
}

fn xparse(pos: (usize, usize), msg: String) -> EsError {
    EsError::x_content_parse(format!("[{}:{}] {msg}", pos.0, pos.1))
}

fn required_one_of(fields: &[&str]) -> EsError {
    EsError::illegal_argument(format!("Required one of fields [{}], but none were specified.", fields.join(", ")))
}

/// Field rules of one event object: allowed keys, nested object parsers,
/// required field sets.
struct ObjSpec {
    name: &'static str,
    fields: &'static [(&'static str, Kind)],
    required: &'static [&'static [&'static str]],
}

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Int,
    Obj(&'static ObjSpec),
    /// `search.filters`: string or list of strings per key.
    Filters,
    /// `search.results.items`: list of `{document, page}`.
    Items,
}

static SESSION: ObjSpec = ObjSpec { name: "session", fields: &[("id", Kind::Str)], required: &[&["id"]] };
static USER: ObjSpec = ObjSpec { name: "user", fields: &[("id", Kind::Str)], required: &[&["id"]] };
static PAGE: ObjSpec = ObjSpec { name: "page", fields: &[("url", Kind::Str), ("title", Kind::Str), ("referrer", Kind::Str)], required: &[&["url"]] };
static DOCUMENT: ObjSpec = ObjSpec { name: "document", fields: &[("id", Kind::Str), ("index", Kind::Str)], required: &[&["id"]] };
static SORT: ObjSpec = ObjSpec { name: "sort", fields: &[("name", Kind::Str), ("direction", Kind::Str)], required: &[&["name"]] };
static PAGINATION: ObjSpec = ObjSpec { name: "page", fields: &[("current", Kind::Int), ("size", Kind::Int)], required: &[&["current"], &["size"]] };
static RESULTS: ObjSpec = ObjSpec { name: "results", fields: &[("items", Kind::Items), ("total_results", Kind::Int)], required: &[] };
static SEARCH: ObjSpec = ObjSpec {
    name: "search",
    fields: &[("query", Kind::Str), ("search_application", Kind::Str), ("sort", Kind::Obj(&SORT)), ("filters", Kind::Filters), ("page", Kind::Obj(&PAGINATION)), ("results", Kind::Obj(&RESULTS))],
    required: &[&["query"]],
};
static RESULT_ITEM: ObjSpec = ObjSpec { name: "items", fields: &[("document", Kind::Obj(&DOCUMENT)), ("page", Kind::Obj(&PAGE))], required: &[] };
static PAGE_VIEW: ObjSpec = ObjSpec {
    name: "page_view_event",
    fields: &[("session", Kind::Obj(&SESSION)), ("user", Kind::Obj(&USER)), ("page", Kind::Obj(&PAGE)), ("document", Kind::Obj(&DOCUMENT))],
    required: &[&["session"], &["user"], &["page"]],
};
static SEARCH_EVENT: ObjSpec = ObjSpec { name: "search_event", fields: &[("session", Kind::Obj(&SESSION)), ("user", Kind::Obj(&USER)), ("search", Kind::Obj(&SEARCH))], required: &[&["session"], &["user"], &["search"]] };
static SEARCH_CLICK: ObjSpec = ObjSpec {
    name: "search_click_event",
    fields: &[("session", Kind::Obj(&SESSION)), ("user", Kind::Obj(&USER)), ("search", Kind::Obj(&SEARCH)), ("page", Kind::Obj(&PAGE)), ("document", Kind::Obj(&DOCUMENT))],
    required: &[&["session"], &["user"], &["search"], &["document", "page"]],
};

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
    }
}

/// Validate `pv` against `spec`; returns the parsed object (normalized:
/// filters become lists).
fn parse_obj(pv: &PV, spec: &ObjSpec) -> Result<Map<String, Value>, EsError> {
    parse_obj_at(pv, spec).map_err(|(e, _)| e)
}

/// [`parse_obj`], also reporting where the parser stood when it failed
/// (the location an enclosing "failed to parse field" error reports).
fn parse_obj_at(pv: &PV, spec: &ObjSpec) -> Result<Map<String, Value>, (EsError, (usize, usize))> {
    if !pv.v.is_object() {
        return Err((xparse(pv.start, format!("[{}] Expected START_OBJECT but was: {}", spec.name, token_name(&pv.v))), pv.start));
    }
    let mut out = Map::new();
    for (k, kpos, v) in &pv.members {
        let Some((_, kind)) = spec.fields.iter().find(|(f, _)| f == k) else {
            return Err((xparse(*kpos, format!("[{}] unknown field [{k}]", spec.name)), v.start));
        };
        let fail = |cause: EsError, at: (usize, usize)| (xparse(at, format!("[{}] failed to parse field [{k}]", spec.name)).caused_by(cause), at);
        let parsed = match kind {
            Kind::Str => match &v.v {
                Value::String(_) | Value::Number(_) | Value::Bool(_) => json!(elasticsearch_engine::query::dsl::scalar_string(&v.v)),
                other => return Err((xparse(v.start, format!("[{}] {k} doesn't support values of type: {}", spec.name, token_name(other))), v.start)),
            },
            Kind::Int => match &v.v {
                Value::Number(n) if n.is_i64() => v.v.clone(),
                Value::String(s) if s.trim().parse::<i64>().is_ok() => json!(s.trim().parse::<i64>().unwrap_or(0)),
                other => return Err((xparse(v.start, format!("[{}] {k} doesn't support values of type: {}", spec.name, token_name(other))), v.start)),
            },
            Kind::Obj(sub) => Value::Object(parse_obj_at(v, sub).map_err(|(e, at)| fail(e, at))?),
            Kind::Filters => {
                let Some(o) = v.v.as_object() else { return Err((xparse(v.start, format!("[{}] {k} doesn't support values of type: {}", spec.name, token_name(&v.v))), v.start)) };
                let mut m = Map::new();
                for (fk, fv) in o {
                    let list = match fv {
                        Value::Array(a) => a.iter().map(|x| json!(elasticsearch_engine::query::dsl::scalar_string(x))).collect(),
                        other => vec![json!(elasticsearch_engine::query::dsl::scalar_string(other))],
                    };
                    m.insert(fk.clone(), Value::Array(list));
                }
                Value::Object(m)
            }
            Kind::Items => {
                let Some(a) = v.v.as_array() else { return Err((xparse(v.start, format!("[{}] {k} doesn't support values of type: {}", spec.name, token_name(&v.v))), v.start)) };
                let mut items = Vec::new();
                for item in a {
                    let text = item.to_string();
                    let mut p = Pos { s: text.as_bytes(), text: &text, i: 0 };
                    let Some(ipv) = p.value() else { continue };
                    items.push(Value::Object(parse_obj(&ipv, &RESULT_ITEM).map_err(|e| fail(e, v.start))?));
                }
                Value::Array(items)
            }
        };
        out.insert(k.clone(), parsed);
    }
    for set in spec.required {
        if !set.iter().any(|f| out.contains_key(*f)) {
            return Err((required_one_of(set), pv.end));
        }
    }
    Ok(out)
}

fn post_event(node: &Node, req: &Req, name: &str, ty: &str) -> R {
    let spec: &ObjSpec = match ty {
        "page_view" => &PAGE_VIEW,
        "search" => &SEARCH_EVENT,
        "search_click" => &SEARCH_CLICK,
        other => return Err(EsError::action_validation(&[&format!("invalid event type: [{other}]")])),
    };
    check_license(node)?;
    let ds = format!("{DS_PREFIX}{name}");
    if node.data_stream(&ds).is_none() {
        return Err(EsError::resource_not_found(format!("no such analytics collection [{name}]")));
    }
    let text = req.body_str()?;
    if text.trim().is_empty() {
        return Err(EsError::new(400, "parse_exception", "request body or source parameter is required"));
    }
    req.json()?;
    let mut p = Pos { s: text.as_bytes(), text, i: 0 };
    let pv = p.value().ok_or_else(|| EsError::parsing("request body is not valid JSON"))?;
    let mut parsed = parse_obj(&pv, spec)?;
    // Session enrichment: client user agent and address.
    if let Some(Value::Object(sess)) = parsed.get_mut("session") {
        let mut s = Map::new();
        if let Some(ua) = req.headers.get("user-agent").and_then(|v| v.to_str().ok()) {
            s.insert("user_agent".into(), json!(ua));
        }
        s.insert("id".into(), sess.get("id").cloned().unwrap_or(Value::Null));
        if let Some(ip) = req.headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()).and_then(|v| v.split(',').next()).map(str::trim) {
            s.insert("ip".into(), json!(ip));
        }
        *sess = s;
    }
    let mut event = Map::new();
    event.insert("@timestamp".into(), json!(elasticsearch_engine::date::now_millis()));
    event.insert("event".into(), json!({"action": ty}));
    event.insert("data_stream".into(), json!({"type": "behavioral_analytics", "dataset": "events", "namespace": name}));
    for k in ["search", "session", "document", "page", "user"] {
        if let Some(v) = parsed.remove(k) {
            event.insert(k.into(), if k == "search" { order_search(v) } else if k == "page" { order_keys(v, &["url", "referrer", "title"]) } else { v });
        }
    }
    let event = Value::Object(event);
    emit(node, &ds, &event);
    let mut out = json!({"accepted": true});
    if req.flag("debug")? {
        out["event"] = event;
    }
    Ok(Resp::json(202, out))
}

fn order_keys(v: Value, order: &[&str]) -> Value {
    let Value::Object(mut o) = v else { return v };
    let mut out = Map::new();
    for k in order {
        if let Some(x) = o.remove(*k) {
            out.insert((*k).to_string(), x);
        }
    }
    out.extend(o);
    Value::Object(out)
}

fn order_search(v: Value) -> Value {
    let Value::Object(mut o) = v else { return v };
    if let Some(p) = o.remove("page") {
        o.insert("page".into(), order_keys(p, &["size", "current"]));
    }
    if let Some(r) = o.remove("results") {
        o.insert("results".into(), order_keys(r, &["total_results", "items"]));
    }
    order_keys(Value::Object(o), &["page", "results", "query", "filters", "sort", "search_application"])
}

/// Append the event to the collection's data stream, shaped as ES's final
/// pipeline leaves it (URLs split into parts, the client address dropped).
fn emit(node: &Node, ds: &str, event: &Value) {
    let mut doc = event.clone();
    let uri = |v: &Value| -> Value {
        let s = v.as_str().unwrap_or("");
        let mut o = Map::new();
        o.insert("original".into(), json!(s));
        if let Some((scheme, rest)) = s.split_once("://") {
            o.insert("scheme".into(), json!(scheme));
            let (hostport, path) = rest.split_once('/').map(|(h, p)| (h, format!("/{p}"))).unwrap_or((rest, String::new()));
            let (host, port) = hostport.split_once(':').map(|(h, p)| (h, p.parse::<i64>().ok())).unwrap_or((hostport, None));
            o.insert("domain".into(), json!(host));
            if let Some(p) = port {
                o.insert("port".into(), json!(p));
            }
            let (path, query) = path.split_once('?').map(|(a, b)| (a.to_string(), Some(b.to_string()))).unwrap_or((path, None));
            if !path.is_empty() {
                o.insert("path".into(), json!(path));
            }
            if let Some(q) = query {
                o.insert("query".into(), json!(q));
            }
        }
        Value::Object(o)
    };
    if let Some(p) = doc.get_mut("page").and_then(Value::as_object_mut) {
        for k in ["url", "referrer"] {
            if let Some(v) = p.get(k).cloned() {
                p.insert(k.into(), uri(&v));
            }
        }
    }
    if let Some(s) = doc.get_mut("session").and_then(Value::as_object_mut) {
        s.remove("ip");
        s.remove("user_agent");
    }
    if let Ok(svc) = node.write_index(ds, false) {
        let _ = svc.write(vec![WriteReq::Index { id: None, source: doc.to_string(), routing: None, cond: WriteCond::default() }], false);
    }
}
