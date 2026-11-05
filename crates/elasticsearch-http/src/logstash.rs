//! `x-pack-logstash`: `PUT|GET|DELETE /_logstash/pipeline/{id}` (Logstash
//! central pipeline management). Pipelines are documents of the hidden
//! `.logstash` system index, stored with the request body as `_source`
//! exactly as Elasticsearch does.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::{IndexService, WriteReq};
use elasticsearch_engine::mapping::wildcard_match;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{self, ExecOpts, SearchRequest, Target};
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

const INDEX: &str = ".logstash";
const VERSION: &str = elasticsearch_engine::version::ES_VERSION;
/// `Required [...]` reports missing fields in this order.
const REQUIRED: &[&str] = &["last_modified", "pipeline_metadata", "username", "pipeline", "pipeline_settings"];
/// Top-level properties of the `.logstash` mapping.
const PROPS: &[&str] = &["description", "last_modified", "metadata", "pipeline", "pipeline_metadata", "pipeline_settings", "username"];

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    if req.seg(1) != "pipeline" || req.segs.len() > 3 {
        return crate::not_found_handler(&req);
    }
    let has_id = req.segs.len() == 3;
    match (&req.method, has_id) {
        (&Method::GET, _) => blocking(move || get(&node, &req)).await,
        (&Method::PUT, true) => blocking(move || put(&node, &req)).await,
        (&Method::DELETE, true) => blocking(move || delete(&node, &req)).await,
        (_, true) => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
        _ => method_not_allowed(&req, &["GET"]),
    }
}

fn text_empty(status: u16) -> Resp {
    let mut r = Resp::empty(status);
    r.headers.push(("content-type", "text/plain; charset=UTF-8".to_string()));
    r
}

/// The `.logstash` index (created on first write, with ES's settings/mapping).
fn ensure_index(node: &Node) -> Result<Arc<IndexService>, EsError> {
    if let Some(s) = node.get_index(INDEX) {
        return Ok(s);
    }
    let body = json!({
        "settings": {"index": {
            "codec": "best_compression",
            "routing": {"allocation": {"include": {"_tier_preference": "data_content"}}},
            "hidden": true,
            "number_of_shards": 1,
            "auto_expand_replicas": "0-1"
        }},
        "mappings": {
            "dynamic": "strict",
            "_meta": {"logstash-version": VERSION, "managed_index_mappings_version": 1},
            "properties": {
                "description": {"type": "text"},
                "last_modified": {"type": "date"},
                "metadata": {"type": "object", "dynamic": "false"},
                "pipeline": {"type": "text"},
                "pipeline_metadata": {"properties": {"type": {"type": "keyword"}, "version": {"type": "short"}}},
                "pipeline_settings": {"type": "object", "dynamic": "false"},
                "username": {"type": "keyword"}
            }
        }
    });
    match node.create_index(INDEX, &body) {
        Ok(s) => Ok(s),
        Err(e) if e.kind == "resource_already_exists_exception" => node.get_index(INDEX).ok_or(e),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------- JSON positions

/// 1-based `(line, column)` of the first character of every object member
/// value, keyed by dotted path (what Jackson reports as the token location).
fn positions(s: &str) -> HashMap<String, (usize, usize)> {
    struct P<'a> {
        b: &'a [u8],
        i: usize,
        line: usize,
        col: usize,
        out: HashMap<String, (usize, usize)>,
    }
    impl P<'_> {
        fn bump(&mut self) {
            if let Some(&c) = self.b.get(self.i) {
                self.i += 1;
                if c == b'\n' {
                    self.line += 1;
                    self.col = 1;
                } else if c & 0xC0 != 0x80 {
                    self.col += 1;
                }
            }
        }
        fn ws(&mut self) {
            while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.bump();
            }
        }
        fn string(&mut self) -> String {
            let start = self.i;
            self.bump();
            while let Some(&c) = self.b.get(self.i) {
                if c == b'\\' {
                    self.bump();
                    self.bump();
                    continue;
                }
                self.bump();
                if c == b'"' {
                    break;
                }
            }
            serde_json::from_slice(&self.b[start..self.i]).unwrap_or_default()
        }
        fn value(&mut self, path: &str) {
            self.ws();
            match self.b.get(self.i) {
                Some(b'{') => {
                    self.bump();
                    loop {
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b'"') => {
                                let k = self.string();
                                self.ws();
                                self.bump(); // ':'
                                self.ws();
                                let p = if path.is_empty() { k } else { format!("{path}.{k}") };
                                self.out.insert(p.clone(), (self.line, self.col));
                                self.value(&p);
                            }
                            Some(b',') => self.bump(),
                            _ => {
                                self.bump();
                                break;
                            }
                        }
                        if self.i >= self.b.len() {
                            break;
                        }
                    }
                }
                Some(b'[') => {
                    self.bump();
                    loop {
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b']') | None => {
                                self.bump();
                                break;
                            }
                            Some(b',') => self.bump(),
                            _ => self.value(""),
                        }
                    }
                }
                Some(b'"') => {
                    self.string();
                }
                _ => {
                    while let Some(&c) = self.b.get(self.i) {
                        if matches!(c, b',' | b'}' | b']' | b' ' | b'\n' | b'\t' | b'\r') {
                            break;
                        }
                        self.bump();
                    }
                }
            }
        }
    }
    let mut p = P { b: s.as_bytes(), i: 0, line: 1, col: 1, out: HashMap::new() };
    p.value("");
    p.out
}

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(true) => "VALUE_TRUE",
        Value::Bool(false) => "VALUE_FALSE",
        Value::Number(n) if n.is_f64() => "VALUE_NUMBER_FLOAT",
        Value::Number(_) => "VALUE_NUMBER_INT",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn valid_instant(s: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(s).is_ok() || s.parse::<i64>().is_ok()
}

/// Request body checks in the order Elasticsearch performs them: the
/// `PipelineConfig` object parser, the required-field check, then the
/// strict `.logstash` mapping.
fn validate(body: &str, v: &Value) -> Result<(), EsError> {
    let pos = positions(body);
    let at = |p: &str| pos.get(p).copied().unwrap_or((1, 1));
    let Some(o) = v.as_object() else {
        return Err(EsError::new(400, "x_content_parse_exception", format!("[1:1] [pipeline] Expected START_OBJECT but was: {}", token_name(v))));
    };
    for (k, x) in o {
        let (l, c) = at(k);
        match k.as_str() {
            "last_modified" => match x {
                Value::String(s) if !valid_instant(s) => {
                    return Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [pipeline] failed to parse field [last_modified]"))
                        .caused_by(EsError::new(400, "date_time_parse_exception", format!("Text '{s}' could not be parsed at index 0"))));
                }
                Value::String(_) | Value::Null => {}
                other => {
                    return Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [pipeline] last_modified doesn't support values of type: {}", token_name(other))));
                }
            },
            "pipeline_metadata" | "pipeline_settings" | "metadata" if k != "metadata" => {
                if !x.is_object() && !x.is_null() {
                    return Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [pipeline] {k} doesn't support values of type: {}", token_name(x))));
                }
            }
            "description" | "pipeline" | "username" => {
                if x.is_array() || x.is_object() {
                    return Err(EsError::new(400, "x_content_parse_exception", format!("[{l}:{c}] [pipeline] {k} doesn't support values of type: {}", token_name(x))));
                }
            }
            _ => {}
        }
    }
    let missing: Vec<&str> = REQUIRED.iter().copied().filter(|k| o.get(*k).map(Value::is_null).unwrap_or(true)).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    for (k, x) in o {
        let (l, c) = at(k);
        if !PROPS.contains(&k.as_str()) {
            return Err(EsError::new(400, "strict_dynamic_mapping_exception", format!("[{l}:{c}] mapping set to strict, dynamic introduction of [{k}] within [_doc] is not allowed")));
        }
        if k == "pipeline_metadata" {
            if let Some(m) = x.as_object() {
                for (mk, mv) in m {
                    let (l, c) = at(&format!("pipeline_metadata.{mk}"));
                    match mk.as_str() {
                        "type" | "version" => {
                            if mk == "version" {
                                let bad = match mv {
                                    Value::String(s) => s.trim().parse::<f64>().is_err().then(|| s.clone()),
                                    _ => None,
                                };
                                if let Some(s) = bad {
                                    return Err(EsError::document_parsing(format!(
                                        "[{l}:{c}] failed to parse field [pipeline_metadata.version] of type [short] in document with id 'ID'. Preview of field's value: '{s}'"
                                    ))
                                    .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))));
                                }
                            }
                        }
                        _ => {
                            return Err(EsError::new(400, "strict_dynamic_mapping_exception", format!("[{l}:{c}] mapping set to strict, dynamic introduction of [{mk}] within [pipeline_metadata] is not allowed")));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// ES pipeline ids: begin with a letter or underscore, then letters,
/// digits, underscores and dashes.
fn valid_id(id: &str) -> bool {
    let mut ch = id.chars();
    matches!(ch.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && ch.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn put(node: &Node, req: &Req) -> R {
    let id = req.seg(2).to_string();
    if !valid_id(&id) {
        return Err(EsError::illegal_argument(format!(
            "Invalid pipeline [{id}] ID received. Pipeline ID must begin with a letter or underscore and can contain only letters, underscores, dashes, hyphens, and numbers"
        )));
    }
    let body = req.body_str()?;
    if body.trim().is_empty() {
        return Err(EsError::new(400, "parse_exception", "request body is required"));
    }
    let v: Value = serde_json::from_str(body).map_err(|e| {
        EsError::new(400, "x_content_parse_exception", format!("[{}:{}] {e}", e.line(), e.column())).caused_by(EsError::new(400, "json_parse_exception", e.to_string()))
    })?;
    validate(body, &v).map_err(|mut e| {
        e.reason = e.reason.replace("document with id 'ID'", &format!("document with id '{id}'"));
        e
    })?;
    let svc = ensure_index(node)?;
    let r = svc.write(vec![WriteReq::Index { id: Some(id), source: body.to_string(), routing: None, cond: WriteCond::default() }], true).remove(0)?;
    svc.refresh()?;
    Ok(text_empty(if r.result == "created" { 201 } else { 200 }))
}

fn delete(node: &Node, req: &Req) -> R {
    let id = req.seg(2).to_string();
    let Some(svc) = node.get_index(INDEX) else { return Ok(text_empty(404)) };
    if svc.get(&id)?.is_none() {
        return Ok(text_empty(404));
    }
    svc.write(vec![WriteReq::Delete { id, routing: None, cond: WriteCond::default() }], true).remove(0)?;
    svc.refresh()?;
    Ok(text_empty(200))
}

fn get(node: &Node, req: &Req) -> R {
    let ids: Vec<String> = if req.segs.len() == 3 { req.seg(2).split(',').filter(|s| !s.is_empty()).map(str::to_string).collect() } else { Vec::new() };
    let Some(svc) = node.get_index(INDEX) else {
        return Ok(Resp::json(if ids.is_empty() { 200 } else { 404 }, json!({})));
    };
    // The reference answers from a search: hits come in index order.
    let all = list_all(&svc)?;
    let found: Vec<(String, String)> = all.into_iter().filter(|(k, _)| ids.is_empty() || ids.iter().any(|p| if p.contains('*') { wildcard_match(p, k) } else { p == k })).collect();
    if found.is_empty() && !ids.is_empty() {
        return Ok(Resp::json(404, json!({})));
    }
    let mut out = Map::new();
    for (id, src) in found {
        out.insert(id, serde_json::from_str(&src).unwrap_or(Value::Null));
    }
    Ok(Resp::ok(Value::Object(out)))
}

/// Every pipeline `(id, raw source)`, in search order.
fn list_all(svc: &Arc<IndexService>) -> Result<Vec<(String, String)>, EsError> {
    svc.refresh()?;
    let t = Target::new(svc.clone());
    let sreq = SearchRequest { query: Some(json!({"match_all": {}})), size: 10_000, ..Default::default() };
    let out = search::execute(&[t], &sreq, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None })?;
    let v: Value = serde_json::from_str(&out.json)?;
    let mut rows: Vec<(String, String)> = v["hits"]["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| (h["_id"].as_str().unwrap_or("").to_string(), h["_source"].to_string()))
        .collect();
    Ok(rows)
}
