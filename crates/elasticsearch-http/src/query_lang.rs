//! Query languages and search extensions:
//! search templates (`_search/template`, `_msearch/template`,
//! `_render/template`), ES|QL (`_query`, `_query/async`), SQL (`_sql`),
//! EQL (`_eql/search`), `_rank_eval`, search applications
//! (`_application/search_application`), query rules (`_query_rules`) and
//! the synonyms API (`_synonyms`).

use crate::search_api::run_search;
use crate::{blocking, method_not_allowed, Req, Resp, RespBody, R};
use axum::http::Method;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::esql::exec::{Column, EsqlOptions, Table};
use elasticsearch_engine::esql::format as fmt;
use elasticsearch_engine::esql::parse::{json_to_val, Params};
use elasticsearch_engine::esql::{self, Val};
use elasticsearch_engine::node::{Node, ResolveOpts, NODE_ID};
use elasticsearch_engine::search_ext::{self, ext};
use elasticsearch_engine::{eql, rank_eval, sql, template};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Entry point. `index` is set for `/{index}/...` paths.
pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    blocking(move || route(&node, &req, index.as_deref())).await
}

fn route(node: &Node, req: &Req, index: Option<&str>) -> R {
    let off = if index.is_some() { 1 } else { 0 };
    let api = req.seg(off).to_string();
    let a = req.seg(off + 1).to_string();
    let b = req.seg(off + 2).to_string();
    let c = req.seg(off + 3).to_string();
    let m = &req.method;
    let get_post = *m == Method::GET || *m == Method::POST;
    match api.as_str() {
        "_search" | "_msearch" if a == "template" => {
            if !get_post {
                return method_not_allowed(req, &["GET", "POST"]);
            }
            if api == "_search" { search_template(node, req, index) } else { msearch_template(node, req, index) }
        }
        "_render" if a == "template" => {
            if !get_post {
                return method_not_allowed(req, &["GET", "POST"]);
            }
            render_template(node, req, if b.is_empty() { None } else { Some(b.as_str()) })
        }
        "_query" if index.is_none() => esql_route(node, req, &a, &b, &c),
        "_sql" if index.is_none() => sql_route(node, req, &a, &b, &c),
        "_eql" if a == "search" => eql_route(node, req, index, &b, &c),
        "_rank_eval" => {
            if !get_post {
                return method_not_allowed(req, &["GET", "POST"]);
            }
            rank_eval_api(node, req, index)
        }
        "_application" if index.is_none() && a == "search_application" => {
            elasticsearch_engine::license::check_ent_search(node, "search application", elasticsearch_engine::license::Mode::Platinum)?;
            search_app_route(node, req, &b, &c)
        }
        "_query_rules" if index.is_none() => {
            elasticsearch_engine::license::check_ent_search(node, "query rules", elasticsearch_engine::license::Mode::Enterprise)?;
            query_rules_route(node, req, &a, &b, &c)
        }
        "_synonyms" if index.is_none() => synonyms_route(node, req, &a, &b),
        _ => crate::not_found_handler(req),
    }
}

fn typed(status: u16, body: Vec<u8>, ct: &str) -> Resp {
    Resp { status, body: RespBody::Bytes(body), headers: vec![("content-type", ct.to_string())] }
}

fn body_obj(req: &Req) -> Result<Value> {
    let v = req.json()?;
    Ok(if v.is_null() { json!({}) } else { v })
}

/// `[line:col]` (1-based) of a top-level key (`value == false`) or of its
/// value (`value == true`) in a JSON object text, as Jackson reports token
/// locations.
fn top_level_location(text: &str, key: &str, value: bool) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let (mut depth, mut i) = (0i32, 0usize);
    while i < b.len() {
        match b[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                let s = text.get(start + 1..i).unwrap_or("");
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if depth == 1 && j < b.len() && b[j] == b':' && serde_json::from_str::<String>(&format!("\"{s}\"")).ok().as_deref() == Some(key) {
                    let mut at = start;
                    if value {
                        at = j + 1;
                        while at < b.len() && b[at].is_ascii_whitespace() {
                            at += 1;
                        }
                    }
                    let before = &text[..at];
                    let line = before.matches('\n').count() + 1;
                    let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
                    return Some((line, col));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// `SuggestingErrorOnUnknown.suggest`: candidates with a Lucene Levenshtein
/// similarity above 0.5, best first.
fn did_you_mean(field: &str, candidates: &[&str]) -> String {
    let mut scored: Vec<(f32, &str)> = candidates
        .iter()
        .map(|c| {
            let max = field.chars().count().max(c.chars().count()).max(1);
            (1.0 - esql::levenshtein(field, c) as f32 / max as f32, *c)
        })
        .filter(|(d, _)| *d > 0.5)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(b.1)));
    match scored.len() {
        0 => String::new(),
        1 => format!(" did you mean [{}]?", scored[0].1),
        _ => format!(" did you mean any of [{}]?", scored.iter().map(|s| s.1).collect::<Vec<_>>().join(", ")),
    }
}

/// Strict `ObjectParser` checks on a request body: unknown top-level fields
/// and non-boolean values of boolean fields fail like ES
/// (`x_content_parse_exception` with the token location).
fn check_body_fields(req: &Req, body: &Value, parser: &str, allowed: &[&str], bools: &[&str]) -> Result<()> {
    let Some(obj) = body.as_object() else { return Ok(()) };
    let text = req.body_str().unwrap_or("");
    let loc = |k: &str, value: bool| top_level_location(text, k, value).map(|(l, c)| format!("[{l}:{c}] ")).unwrap_or_default();
    for (k, v) in obj {
        if !allowed.contains(&k.as_str()) {
            return Err(EsError::x_content_parse(format!("{}[{parser}] unknown field [{k}]{}", loc(k, false), did_you_mean(k, allowed))));
        }
        if bools.contains(&k.as_str()) {
            let bad = match v {
                Value::Bool(_) | Value::Null => None,
                Value::String(s) if s == "true" || s == "false" => None,
                Value::String(s) => Some(s.clone()),
                other => Some(other.to_string()),
            };
            if let Some(s) = bad {
                return Err(EsError::x_content_parse(format!("{}[{parser}] failed to parse field [{k}]", loc(k, true)))
                    .caused_by(EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed."))));
            }
        }
    }
    Ok(())
}

const EQL_BODY_FIELDS: &[&str] = &[
    "query",
    "filter",
    "timestamp_field",
    "tiebreaker_field",
    "event_category_field",
    "size",
    "fetch_size",
    "wait_for_completion_timeout",
    "keep_alive",
    "keep_on_completion",
    "result_position",
    "fields",
    "runtime_mappings",
    "max_samples_per_key",
    "allow_partial_search_results",
    "allow_partial_sequence_results",
];
const EQL_BOOL_FIELDS: &[&str] = &["keep_on_completion", "allow_partial_search_results", "allow_partial_sequence_results"];

/// `RequestXContent` (ES 9.5.5): the declared fields, the query-settings
/// root aliases (`approximation`, `project_routing`, `time_zone`) and `settings`.
const ESQL_BODY_FIELDS: &[&str] = &[
    "query",
    "columnar",
    "filter",
    "accept_pragma_risks",
    "include_ccs_metadata",
    "include_execution_metadata",
    "pragma",
    "params",
    "locale",
    "profile",
    "tables",
    "approximation",
    "project_routing",
    "time_zone",
    "settings",
];
const ESQL_ASYNC_BODY_FIELDS: &[&str] = &[
    "query",
    "columnar",
    "filter",
    "accept_pragma_risks",
    "include_ccs_metadata",
    "include_execution_metadata",
    "pragma",
    "params",
    "locale",
    "profile",
    "tables",
    "approximation",
    "project_routing",
    "time_zone",
    "settings",
    "keep_on_completion",
    "wait_for_completion_timeout",
    "keep_alive",
];
const ESQL_BOOL_FIELDS: &[&str] = &["columnar", "profile", "include_ccs_metadata", "include_execution_metadata", "accept_pragma_risks", "keep_on_completion"];

/// `[line:col]` (1-based) of the key at `path` (or of its value) in a JSON
/// text, as Jackson reports token locations.
fn json_location(text: &str, path: &[&str], value: bool) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    // Per open container: Some(current key) for objects, None for arrays.
    let mut stack: Vec<Option<String>> = Vec::new();
    let mut i = 0usize;
    let at = |pos: usize| {
        let before = &text[..pos];
        let line = before.matches('\n').count() + 1;
        let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
        (line, col)
    };
    while i < b.len() {
        match b[i] {
            b'{' => stack.push(Some(String::new())),
            b'[' => stack.push(None),
            b'}' | b']' => {
                stack.pop();
            }
            b'"' => {
                let start = i;
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                let raw = text.get(start + 1..i).unwrap_or("");
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < b.len() && b[j] == b':' && matches!(stack.last(), Some(Some(_))) {
                    let key = serde_json::from_str::<String>(&format!("\"{raw}\"")).unwrap_or_else(|_| raw.to_string());
                    if let Some(Some(k)) = stack.last_mut() {
                        *k = key;
                    }
                    let keys: Vec<&str> = stack.iter().map(|k| k.as_deref().unwrap_or("")).collect();
                    if keys == path {
                        if !value {
                            return Some(at(start));
                        }
                        let mut v = j + 1;
                        while v < b.len() && b[v].is_ascii_whitespace() {
                            v += 1;
                        }
                        return Some(at(v));
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// A fresh async execution id: `(id, doc_id)`; stored results are keyed by doc id.
fn new_id() -> (String, String) {
    esql::async_id::new_id(NODE_ID)
}

/// The storage key (doc id) of an async id; malformed ids fail like ES
/// (`invalid id: [..]` 400s).
fn async_key(id: &str) -> Result<String> {
    Ok(esql::async_id::decode(id)?.doc_id)
}

// ------------------------------------------------------------- templates

/// `(source, params)` of a template request.
fn template_of(node: &Node, body: &Value, path_id: Option<&str>) -> Result<(String, Value)> {
    let params = body.get("params").cloned().unwrap_or_else(|| json!({}));
    if let Some(src) = body.get("source").or_else(|| body.get("inline")) {
        return Ok((template::source_text(src), params));
    }
    let id = body.get("id").and_then(Value::as_str).or(path_id).ok_or_else(|| EsError::action_validation(&["template is missing"]))?;
    Ok((search_ext::stored_template(node, id)?, params))
}

fn rendered_body(node: &Node, body: &Value, path_id: Option<&str>) -> Result<Value> {
    let (src, params) = template_of(node, body, path_id)?;
    // ES compiles the template through the script cache (`template` context).
    template::admit(&src)?;
    let mut out = template::render_to_json(&src, &params)?;
    for k in ["explain", "profile"] {
        if body.get(k).and_then(Value::as_bool) == Some(true) {
            if let Some(o) = out.as_object_mut() {
                o.insert(k.into(), json!(true));
            }
        }
    }
    Ok(out)
}

fn search_template(node: &Node, req: &Req, index: Option<&str>) -> R {
    let body = body_obj(req)?;
    let rendered = rendered_body(node, &body, None)?;
    let mut params = req.params.clone();
    for k in ["explain", "profile"] {
        params.remove(k);
    }
    let json = run_search(node, &rendered, &params, index, ResolveOpts::from_params(&req.params), req.access.as_deref())?;
    Ok(Resp::raw(200, json))
}

fn msearch_template(node: &Node, req: &Req, index: Option<&str>) -> R {
    let start = Instant::now();
    let text = req.body_str()?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut responses: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let header: Value = serde_json::from_str(lines[i]).map_err(|e| EsError::json_parse(&e))?;
        let tbody: Value = match lines.get(i + 1) {
            Some(l) => serde_json::from_str(l).map_err(|e| EsError::json_parse(&e))?,
            None => return Err(EsError::illegal_argument("The msearch request must be terminated by a newline [\n]")),
        };
        i += 2;
        let idx: Option<String> = match header.get("index") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => Some(a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(",")),
            _ => index.map(str::to_string),
        };
        let mut params: HashMap<String, String> = HashMap::new();
        for k in ["rest_total_hits_as_int", "typed_keys"] {
            if let Some(v) = req.param(k) {
                params.insert(k.into(), v.into());
            }
        }
        let r = rendered_body(node, &tbody, None).and_then(|b| run_search(node, &b, &params, idx.as_deref(), ResolveOpts::from_params(&params), req.access.as_deref()));
        match r {
            Ok(mut s) => {
                s.pop();
                s.push_str(",\"status\":200}");
                responses.push(s);
            }
            Err(e) => responses.push(json!({"error": e.error_object(false), "status": e.status}).to_string()),
        }
    }
    Ok(Resp::raw(200, format!("{{\"took\":{},\"responses\":[{}]}}", start.elapsed().as_millis(), responses.join(","))))
}

fn render_template(node: &Node, req: &Req, id: Option<&str>) -> R {
    let body = body_obj(req)?;
    let rendered = rendered_body(node, &body, id)?;
    Ok(Resp::ok(json!({ "template_output": rendered })))
}

// ------------------------------------------------------------------- ES|QL

fn table_from_json(v: &Value) -> Table {
    let cols: Vec<Column> = v["columns"]
        .as_array()
        .map(|a| a.iter().map(|c| Column { name: c["name"].as_str().unwrap_or("").into(), ty: c["type"].as_str().unwrap_or("").into() }).collect())
        .unwrap_or_default();
    let rows = v["values"].as_array().map(|a| a.iter().map(|r| r.as_array().map(|x| x.iter().map(json_to_val).collect()).unwrap_or_default()).collect()).unwrap_or_default();
    Table { cols, rows }
}

/// Render a table response in the requested format.
/// `all`: every column when `t` holds only the non-null ones
/// ([`esql::EsqlResult::all_columns`], with `drop_null`).
#[allow(clippy::too_many_arguments)]
fn render_table(req: &Req, t: &Table, all: Option<&[Column]>, mut meta: Map<String, Value>, columnar: bool, drop_null: bool, row_key: &str, extra_headers: Vec<(&'static str, String)>) -> R {
    let format = fmt::pick_format(req.param("format"), req.headers.get("accept").and_then(|v| v.to_str().ok()))?;
    // A table of the non-null columns with nulls kept (`sparse_nulls`): the
    // streamed JSON writes the missing columns' nulls; other formats render
    // the full table.
    let expanded;
    let (t, all) = match all {
        Some(cols) if !drop_null && !(format == "json" && streamed_json(req)) => {
            expanded = fmt::expanded_table(t, cols);
            (&expanded, None)
        }
        _ => (t, all),
    };
    let mut resp = match format.as_str() {
        "json" if streamed_json(req) => {
            let col_meta = meta.remove("__col_meta");
            let keep = match all {
                Some(cols) if !drop_null => fmt::columns_expanded(t, cols, &mut meta),
                _ => fmt::columns_only_all(t, all, drop_null, &mut meta),
            };
            if let Some(Value::Object(cm)) = col_meta {
                for key in ["columns", "all_columns"] {
                    if let Some(Value::Array(cols)) = meta.get_mut(key) {
                        for c in cols.iter_mut() {
                            if let Some(m) = c["name"].as_str().and_then(|n| cm.get(n)) {
                                c["_meta"] = m.clone();
                            }
                        }
                    }
                }
            }
            let values_key = if row_key != "values" && !columnar { row_key } else { "values" };
            // Where the tree would hold the values (its key order).
            meta.insert(values_key.into(), Value::Null);
            let body = fmt::StreamedResponse { meta: &meta, values_key, table: t, keep: &keep, columnar };
            // The serializer's bytes, null runs copied (see `write_json`).
            Resp { status: 200, body: RespBody::Owned { bytes: crate::wire::write_admitted(|w| body.write_json(w))?, json: true }, headers: Vec::new() }
        }
        "json" | "yaml" | "cbor" => {
            let col_meta = meta.remove("__col_meta");
            fmt::columns_values_all(t, all, columnar, drop_null, &mut meta);
            // `column_metadata`: `_meta` on the column descriptors.
            if let Some(Value::Object(cm)) = col_meta {
                for key in ["columns", "all_columns"] {
                    if let Some(Value::Array(cols)) = meta.get_mut(key) {
                        for c in cols.iter_mut() {
                            if let Some(m) = c["name"].as_str().and_then(|n| cm.get(n)) {
                                c["_meta"] = m.clone();
                            }
                        }
                    }
                }
            }
            if row_key != "values" && !columnar {
                if let Some(v) = meta.remove("values") {
                    meta.insert(row_key.into(), v);
                }
            }
            let v = Value::Object(meta);
            match format.as_str() {
                "json" => Resp::ok(v),
                "yaml" => typed(200, fmt::to_yaml(&v).into_bytes(), fmt::content_type("yaml").unwrap()),
                _ => typed(200, fmt::to_cbor(&v), fmt::content_type("cbor").unwrap()),
            }
        }
        "smile" | "arrow" => return Err(EsError::illegal_argument(format!("format [{format}] is not supported by this node"))),
        f => {
            meta.remove("__col_meta");
            let header = req.param("header").map(|h| h != "absent").unwrap_or(true);
            let s = fmt::text_table(t, f, req.param("delimiter"), header, drop_null)?;
            typed(200, s.into_bytes(), fmt::content_type(f).unwrap())
        }
    };
    resp.headers.extend(extra_headers);
    Ok(resp)
}

/// A plain JSON table response is serialized from the rows directly
/// ([`fmt::StreamedResponse`]): no `Value` per cell. Responses the HTTP
/// layer rewrites afterwards (pretty, `filter_path`, `human`, another wire
/// format) keep the tree, so they are exactly as before.
fn streamed_json(req: &Req) -> bool {
    let wire = |name: &str| req.headers.get(name).and_then(|v| v.to_str().ok()).is_some_and(|v| {
        let v = v.to_ascii_lowercase();
        v.contains("cbor") || v.contains("smile") || v.contains("yaml")
    });
    ["pretty", "filter_path", "human"].iter().all(|p| req.param(p).is_none()) && !wire("accept") && !wire("content-type")
}

/// Java `toString` of a JSON value (maps `{k=v}`, lists `[a, b]`).
fn java_str(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(java_str).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, v)| format!("{k}={}", java_str(v))).collect::<Vec<_>>().join(", ")),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => i.to_string(),
            (None, Some(d)) => esql::fmt_double(d),
            _ => n.to_string(),
        },
        Value::Bool(b) => b.to_string(),
    }
}

/// `DataType.fromJava` of a parameter value (as a type tag).
fn param_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::String(_) => "keyword",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.as_i64().is_some_and(|i| i >= i32::MIN as i64 && i <= i32::MAX as i64) => "integer",
        Value::Number(n) if n.as_i64().is_some() => "long",
        Value::Number(_) => "double",
        Value::Array(_) | Value::Object(_) => "unsupported",
    }
}

/// `[line:col]` positions of the elements of the JSON array starting at
/// byte `start` of `text`, and of its closing bracket.
fn array_element_locations(text: &str, start: usize) -> (Vec<String>, String) {
    let b = text.as_bytes();
    let at = |pos: usize| {
        let before = &text[..pos.min(text.len())];
        let line = before.matches('\n').count() + 1;
        let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
        format!("[{line}:{col}]")
    };
    let mut out = Vec::new();
    let (mut depth, mut i, mut expecting) = (0i32, start, true);
    while i < b.len() {
        let c = b[i];
        match c {
            b'"' => {
                if depth == 1 && expecting {
                    out.push(at(i));
                    expecting = false;
                }
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'[' | b'{' => {
                if depth == 1 && expecting {
                    out.push(at(i));
                    expecting = false;
                }
                depth += 1;
            }
            b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return (out, at(i));
                }
            }
            b',' if depth == 1 => expecting = true,
            c if depth == 1 && expecting && !c.is_ascii_whitespace() => {
                out.push(at(i));
                expecting = false;
            }
            _ => {}
        }
        i += 1;
    }
    (out, at(i))
}

/// `RequestXContent.parseParams` validation of `params`.
fn validate_params(text: &str, body: &Value) -> Result<()> {
    let Some(params) = body.get("params") else { return Ok(()) };
    let start = json_location(text, &["params"], true).map(|(l, c)| {
        // Byte offset of (line, col).
        let mut off = 0;
        for (k, line) in text.split('\n').enumerate() {
            if k + 1 == l {
                off += line.char_indices().nth(c - 1).map(|(i, _)| i).unwrap_or(0);
                break;
            }
            off += line.len() + 1;
        }
        off
    });
    let Value::Array(items) = params else {
        let loc = start.map(|_| String::new()).unwrap_or_default();
        let _ = loc;
        return Ok(());
    };
    let (locs, end) = start.map(|s| array_element_locations(text, s)).unwrap_or_default();
    let loc = |i: usize| locs.get(i).cloned().unwrap_or_else(|| "[1:1]".into());
    let mut errors: Vec<String> = Vec::new();
    let (mut named, mut unnamed): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    let classes = ["VALUE", "IDENTIFIER", "PATTERN"];
    for (i, item) in items.iter().enumerate() {
        let l = loc(i);
        match item {
            Value::Object(m) => {
                if m.len() > 1 {
                    errors.push(format!("{l} Cannot parse more than one key:value pair as parameter, found [{}]", m.iter().map(|(k, v)| format!("{{{k}:{}}}", java_str(v))).collect::<Vec<_>>().join(", ")));
                }
                for (name, v) in m {
                    let valid = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                    if !valid {
                        errors.push(format!("{l} [{name}] is not a valid parameter name, a valid parameter name starts with a letter or underscore, and contains letters, digits and underscores only"));
                    }
                    let entry = format!("{name}={}", java_str(v));
                    let mut check_value = |class: &str, value: &Value, errors: &mut Vec<String>| {
                        if let Value::Array(list) = value {
                            if class != "VALUE" {
                                errors.push(format!("{l} {entry} parameter is multivalued, only VALUE parameters can be multivalued"));
                                return;
                            }
                            let mut ty: Option<&str> = None;
                            for x in list {
                                let t = param_type(x);
                                if t == "null" {
                                    errors.push(format!("{l} Parameter [{name}] contains a null entry: {}. Null values are not allowed in multivalued params", java_str(value)));
                                    break;
                                } else if ty.is_some_and(|p| p != t) {
                                    errors.push(format!("{l} Parameter [{name}] contains mixed data types: {}. Mixed data types are not allowed in multivalued params.", java_str(value)));
                                    break;
                                }
                                ty = Some(t);
                            }
                            return;
                        }
                        if class == "IDENTIFIER" || class == "PATTERN" {
                            let bad = !value.is_string() || (class == "PATTERN" && !value.as_str().unwrap_or("").contains('*'));
                            if bad {
                                errors.push(format!(
                                    "{l} [{}] is not a valid value for {class} parameter, a valid value for {class} parameter is a string{}",
                                    java_str(value),
                                    if class == "PATTERN" { " and contains *" } else { "" }
                                ));
                            }
                        }
                    };
                    match v {
                        Value::Object(attrs) => {
                            if attrs.len() != 1 {
                                let what = if attrs.is_empty() { " has no valid param attribute".to_string() } else { format!(" has multiple param attributes [{}]", attrs.keys().cloned().collect::<Vec<_>>().join(", ")) };
                                errors.push(format!("{l} [{name}]{what}, only one of VALUE, IDENTIFIER, PATTERN can be defined in a param"));
                            }
                            for (k, x) in attrs {
                                let up = k.to_ascii_uppercase();
                                if !classes.contains(&up.as_str()) {
                                    errors.push(format!("{l} [{k}] is not a valid param attribute, a valid attribute is any of VALUE, IDENTIFIER, PATTERN"));
                                    continue;
                                }
                                check_value(&up, x, &mut errors);
                            }
                        }
                        other => check_value("VALUE", other, &mut errors),
                    }
                    named.push(format!("{name}={}", java_str(v)));
                }
            }
            Value::Array(list) => {
                let (mut ty, mut null, mut mixed): (Option<&str>, bool, bool) = (None, false, false);
                for x in list {
                    let t = param_type(x);
                    null |= t == "null";
                    mixed |= ty.is_some_and(|p| p != t) && t != "null";
                    if t != "null" {
                        ty = Some(t);
                    }
                }
                if null {
                    errors.push(format!("{l} Parameter contains a null entry: {}. Null values are not allowed in multivalued params", java_str(item)));
                } else if mixed {
                    errors.push(format!("{l} Parameter contains mixed data types: {}. Mixed data types are not allowed in multivalued params.", java_str(item)));
                }
                unnamed.push(java_str(item));
            }
            other => unnamed.push(java_str(other)),
        }
    }
    if !named.is_empty() && !unnamed.is_empty() {
        errors.push(format!("Params cannot contain both named and unnamed parameters; got [{}] and [{}]", named.join(", "), unnamed.join(", ")));
    }
    if errors.is_empty() {
        return Ok(());
    }
    let inner = EsError::x_content_parse(format!("Failed to parse params: {}", errors.join("; ")));
    Err(EsError::x_content_parse(format!("{end} [esql/query] failed to parse field [params]")).caused_by(inner))
}

fn esql_opts<'a>(req: &Req, body: &'a Value) -> Result<EsqlOptions<'a>> {
    let text = req.body_str().unwrap_or("");
    validate_params(text, body)?;
    let loc = |path: &[&str], value: bool| json_location(text, path, value).map(|(l, c)| format!("[{l}:{c}] ")).unwrap_or_default();
    let body_settings = esql::settings::BodySettings::from_body(body, &loc)?;
    let access = req.access.clone().filter(|a| a.is_restricted());
    let expander = req.extensions.get::<crate::QueryTargetExpander>().cloned();
    let restrict: Option<esql::exec::Restrict> = if access.is_some() || expander.is_some() {
        Some(std::sync::Arc::new(move |targets| {
            let targets = match &expander { Some(expander) => (expander.0)(targets)?, None => targets };
            match &access { Some(access) => access.apply(targets), None => Ok(targets) }
        }))
    } else { None };
    let shards = req.extensions.get::<esql::remote::ShardSourceExt>().map(|s| s.0.clone());
    Ok(EsqlOptions { params: Params::from_json(body.get("params"))?, filter: body.get("filter"), locale: body.get("locale").and_then(Value::as_str).map(str::to_string), body_settings, restrict, shards, ..Default::default() })
}

/// A cluster node's read of this node's shards for a query it runs
/// ([`esql::remote::FetchRequest`]): the per-shard results, as JSON.
fn serve_esql_fetch(node: &Node, req: &Req, fetch: &esql::remote::FetchRequest) -> R {
    let access = req.access.clone().filter(|a| a.is_restricted());
    let restrict: Option<esql::exec::Restrict> = access.map(|access| std::sync::Arc::new(move |targets| access.apply(targets)) as esql::exec::Restrict);
    let shards = req.extensions.get::<esql::remote::ShardSourceExt>().map(|s| s.0.clone());
    let results = esql::remote::serve(node, &fetch.0, shards.as_ref(), restrict.as_ref())?;
    Ok(Resp::json(200, serde_json::to_value(results).map_err(|e| EsError::internal(e.to_string()))?))
}

/// `Warning` headers of an ES|QL response.
fn esql_warning_headers(warnings: &[String]) -> Vec<(&'static str, String)> {
    warnings
        .iter()
        .map(|w| ("Warning", format!("299 Elasticsearch-{}-{} \"{}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH, w.replace('\\', "\\\\").replace('"', "\\\""))))
        .collect()
}

/// A finished ES|QL query: its table, response metadata, warnings and the
/// time zone its dates render in.
struct EsqlRun {
    table: Table,
    /// Every column, when `table` dropped its all-null ones.
    all_columns: Option<Vec<Column>>,
    meta: Map<String, Value>,
    warnings: Vec<String>,
    zone: elasticsearch_engine::date::Tz,
}

fn run_esql(node: &Node, req: &Req, body: &Value, is_async: bool) -> Result<EsqlRun> {
    check_body_fields(req, body, if is_async { "esql/async_query" } else { "esql/query" }, if is_async { ESQL_ASYNC_BODY_FIELDS } else { ESQL_BODY_FIELDS }, ESQL_BOOL_FIELDS)?;
    // `allow_partial_results` (URL only, default true since 9.0): a single
    // node never has failed shards, so the result is never partial; the
    // flag is validated only.
    req.flag("allow_partial_results")?;
    let q = body.get("query").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["[query] is required"]))?;
    if q.trim().is_empty() {
        return Err(EsError::action_validation(&["[query] is required"]));
    }
    let stat = |k: &str| *ext(node).esql_stats.lock().entry(k.to_string()).or_insert(0) += 1;
    stat("queries.rest.total");
    let start_ms = elasticsearch_engine::date::now_millis();
    // Registered as a running query (`GET /_query/queries`) while it executes.
    let (task_id, task_key) = {
        let (id, _) = new_id();
        let key = esql::async_id::decode(&id).map(|d| format!("{}:{}", d.node_id, d.task)).unwrap_or_default();
        (id, key)
    };
    ext(node).esql_running.lock().insert(task_key.clone(), (task_id, start_ms, Instant::now(), q.to_string()));
    // A synchronous response may build only its non-null columns.
    let drop_null = !is_async && req.flag("drop_null_columns")?;
    // The sync response renders a result narrowed to its non-null columns
    // (`render_table`); the async one stores the table as it is.
    let res = esql_opts(req, body).and_then(|o| esql::run(node, q, EsqlOptions { drop_null_columns: drop_null, sparse_nulls: !is_async, ..o }));
    ext(node).esql_running.lock().remove(&task_key);
    let res = match res {
        Ok(r) => r,
        // Forwarded to the node holding every shard: that node runs and
        // counts the query (ES counts each query once, where it runs).
        Err(e) if e.kind == esql::remote::FORWARD_ERROR_TYPE => {
            if let Some(n) = ext(node).esql_stats.lock().get_mut("queries.rest.total") {
                *n = n.saturating_sub(1);
            }
            return Err(e);
        }
        Err(e) => {
            stat("queries.rest.failed");
            return Err(e);
        }
    };
    for f in &res.features {
        stat(&format!("features.{f}"));
    }
    for f in &res.functions {
        stat(&format!("functions.{f}"));
    }
    for (name, used) in [("time_zone", res.settings.time_zone_set), ("unmapped_fields", res.settings.unmapped != esql::settings::Unmapped::Default), ("column_metadata", res.settings.column_metadata), ("approximation", res.settings.approximation.is_some())] {
        if used {
            stat(&format!("settings.{name}"));
        }
    }
    if let Some((b, _)) = esql::telemetry::TOOK.iter().find(|(_, lim)| res.took_ms < *lim) {
        stat(&format!("took.{b}"));
    }
    let end_ms = elasticsearch_engine::date::now_millis();
    let mut meta = Map::new();
    meta.insert("took".into(), json!(res.took_ms));
    meta.insert("is_partial".into(), json!(false));
    meta.insert("completion_time_in_millis".into(), json!(end_ms));
    meta.insert("documents_found".into(), json!(res.documents_found));
    meta.insert("values_loaded".into(), json!(res.values_loaded));
    meta.insert("rows_emitted".into(), json!(res.rows_emitted));
    meta.insert("bytes_read".into(), json!(0));
    meta.insert("read_nanos".into(), json!(0));
    meta.insert("cpu_nanos".into(), json!(res.took_ms * 1_000_000));
    meta.insert("start_time_in_millis".into(), json!(start_ms));
    // Results are kept (async) for the default 5 days.
    meta.insert("expiration_time_in_millis".into(), json!(start_ms + 5 * 86_400_000));
    if body.get("profile").and_then(Value::as_bool) == Some(true) {
        let mut p = esql_profile(q, &res);
        if res.settings.unmapped != esql::settings::Unmapped::Default {
            p["unmapped_fields"] = json!(res.settings.unmapped.name());
        }
        meta.insert("profile".into(), p);
    }
    if !res.col_meta.is_empty() {
        meta.insert("__col_meta".into(), Value::Object(res.col_meta.clone().into_iter().collect()));
    }
    if body.get("include_ccs_metadata").and_then(Value::as_bool) == Some(true) {
        meta.insert("_clusters".into(), json!({"total": 1, "successful": 1, "running": 0, "skipped": 0, "partial": 0, "failed": 0, "details": {"(local)": {"status": "successful", "indices": "*", "took": res.took_ms, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}}}}));
    }
    Ok(EsqlRun { table: res.table, all_columns: res.all_columns, meta, warnings: res.warnings, zone: res.settings.time_zone })
}

/// The `profile` section of an ES|QL response, shaped like ES's (query and
/// planning timings plus the `final` / `node_reduce` / `data` drivers and
/// their operators). This engine runs one pipeline, so the per-driver
/// numbers are the whole query's.
fn esql_profile(q: &str, res: &esql::EsqlResult) -> Value {
    let now = elasticsearch_engine::date::now_millis() as u64;
    let took_ms = res.took_ms;
    let start = now.saturating_sub(took_ms);
    let nanos = took_ms * 1_000_000;
    let cols: Vec<String> = res.all_columns.as_ref().unwrap_or(&res.table.cols).iter().map(|c| c.name.clone()).collect();
    let rows = res.table.rows.len();
    let driver = |desc: &str, ops: Value, docs: u64, vals: u64| {
        json!({
            "task_description": desc, "start_millis": start, "stop_millis": now, "took_nanos": nanos, "cpu_nanos": nanos,
            "documents_found": docs, "values_loaded": vals, "iterations": 1, "operators": ops,
            "sleeps": {"counts": {}, "first": [], "last": []}
        })
    };
    let output = json!({"operator": format!("OutputOperator[columns = [{}]]", cols.join(", "))});
    let from_index = matches!(q.split_whitespace().next().map(str::to_ascii_lowercase).as_deref(), Some("from" | "ts" | "metrics"));
    let drivers = if from_index {
        vec![
            driver("final", json!([{"operator": "ExchangeSourceOperator", "status": {"pages_waiting": 0, "pages_emitted": 1, "rows_emitted": rows}}, output]), 0, 0),
            driver("node_reduce", json!([{"operator": "ExchangeSourceOperator", "status": {"pages_waiting": 0, "pages_emitted": 1, "rows_emitted": rows}}, {"operator": "ExchangeSinkOperator", "status": {"pages_received": 1, "rows_received": rows}}]), 0, 0),
            driver(
                "data",
                json!([
                    {"operator": "LuceneSourceOperator[maxPageSize = 21845, remainingDocs = 0]", "status": {"processed_slices": 1, "processed_queries": ["*:*"], "processed_shards": [], "process_nanos": nanos, "slice_index": 0, "total_slices": 1, "pages_emitted": 1, "slice_min": 0, "slice_max": 0, "current": 2147483647, "rows_emitted": res.documents_found}},
                    {"operator": "ValuesSourceReaderOperator", "status": {"pages_processed": 1, "rows_received": res.documents_found, "rows_emitted": res.documents_found, "values_loaded": res.values_loaded}},
                    {"operator": "ExchangeSinkOperator", "status": {"pages_received": 1, "rows_received": rows}}
                ]),
                res.documents_found,
                res.values_loaded,
            ),
        ]
    } else {
        vec![driver("final", json!([{"operator": "LocalSourceOperator"}, output]), 0, 0)]
    };
    json!({
        "query": {"start_millis": start, "stop_millis": now, "took_millis": took_ms, "took_nanos": nanos},
        "planning": {"start_millis": start, "stop_millis": start, "took_millis": 0, "took_nanos": 0},
        "drivers": drivers
    })
}

fn esql_route(node: &Node, req: &Req, a: &str, b: &str, c: &str) -> R {
    let drop_null = req.flag("drop_null_columns")?;
    match (a, b, c, &req.method) {
        ("", "", "", &Method::POST) => {
            if let Some(fetch) = req.extensions.get::<esql::remote::FetchRequest>() {
                return serve_esql_fetch(node, req, fetch);
            }
            let body = body_obj(req)?;
            let EsqlRun { table: t, all_columns, meta, warnings, zone } = run_esql(node, req, &body, false)?;
            let _zone = esql::set_zone(zone);
            let columnar = body.get("columnar").and_then(Value::as_bool).unwrap_or(false);
            render_table(req, &t, all_columns.as_deref(), meta, columnar, drop_null, "values", esql_warning_headers(&warnings))
        }
        ("", "", "", _) => method_not_allowed(req, &["POST"]),
        ("async", "", "", &Method::POST) => {
            let body = body_obj(req)?;
            let EsqlRun { table: t, mut meta, warnings, zone, .. } = run_esql(node, req, &body, true)?;
            let _zone = esql::set_zone(zone);
            let columnar = body.get("columnar").and_then(Value::as_bool).unwrap_or(false);
            let keep = body.get("keep_on_completion").and_then(Value::as_bool).unwrap_or(false);
            meta.insert("is_running".into(), json!(false));
            let mut headers = vec![("X-Elasticsearch-Async-Is-Running", "?0".to_string())];
            headers.extend(esql_warning_headers(&warnings));
            if keep {
                let (id, doc) = new_id();
                // Stored for GET (row-major JSON).
                let mut stored = meta.clone();
                stored.insert("__id".into(), json!(id));
                if !warnings.is_empty() {
                    stored.insert("__warnings".into(), json!(warnings));
                }
                fmt::columns_values(&t, false, false, &mut stored);
                ext(node).esql_async.lock().insert(doc, Value::Object(stored));
                let mut m2 = Map::new();
                m2.insert("id".into(), json!(id));
                m2.extend(meta);
                meta = m2;
                headers.push(("X-Elasticsearch-Async-Id", id));
            }
            render_table(req, &t, None, meta, columnar, drop_null, "values", headers)
        }
        ("async", id, "", &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let v = ext(node).esql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            let t = table_from_json(&v);
            let mut meta = v.as_object().cloned().unwrap_or_default();
            let sid = meta.remove("__id").and_then(|x| x.as_str().map(str::to_string)).unwrap_or_else(|| id.to_string());
            let warnings: Vec<String> = meta.remove("__warnings").and_then(|w| serde_json::from_value(w).ok()).unwrap_or_default();
            meta.remove("columns");
            meta.remove("values");
            let mut m2 = Map::new();
            m2.insert("id".into(), json!(sid));
            m2.extend(meta);
            let mut headers = vec![("X-Elasticsearch-Async-Id", sid.clone()), ("X-Elasticsearch-Async-Is-Running", "?0".into())];
            headers.extend(esql_warning_headers(&warnings));
            render_table(req, &t, None, m2, false, drop_null, "values", headers)
        }
        ("async", id, "stop", &Method::POST) | ("async", id, "stop", &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let v = ext(node).esql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            let t = table_from_json(&v);
            let mut meta = v.as_object().cloned().unwrap_or_default();
            meta.remove("__id");
            meta.remove("__warnings");
            meta.remove("columns");
            meta.remove("values");
            render_table(req, &t, None, meta, false, drop_null, "values", vec![])
        }
        ("async", id, "", &Method::DELETE) if !id.is_empty() => {
            let key = async_key(id)?;
            if ext(node).esql_async.lock().remove(&key).is_none() {
                return Err(EsError::resource_not_found(id.to_string()));
            }
            Ok(Resp::ack())
        }
        ("view", name, "", &Method::PUT) if !name.is_empty() => put_view(node, req, name),
        ("view", names, "", &Method::GET) => {
            let all = esql::views::store(node).all();
            let mut out = Vec::new();
            if names.is_empty() || names == "_all" || names == "*" {
                out.extend(all.iter().map(|(n, q)| json!({"name": n, "query": q})));
            } else {
                for n in names.split(',') {
                    if n.contains('*') {
                        out.extend(all.iter().filter(|(k, _)| elasticsearch_engine::mapping::wildcard_match(n, k)).map(|(k, q)| json!({"name": k, "query": q})));
                    } else {
                        let q = all.get(n).ok_or_else(|| EsError::resource_not_found(format!("view [{n}] not found")))?;
                        out.push(json!({"name": n, "query": q}));
                    }
                }
            }
            Ok(Resp::ok(json!({ "views": out })))
        }
        ("view", names, "", &Method::DELETE) if !names.is_empty() => {
            let list: Vec<String> = names.split(',').map(str::to_string).collect();
            esql::views::store(node).delete(&list)?;
            Ok(Resp::ack())
        }
        ("queries", "", "", &Method::GET) => {
            let running = ext(node).esql_running.lock().clone();
            let mut qs = Map::new();
            for (id, start, inst, query) in running.values() {
                qs.insert(id.clone(), json!({"start_time_millis": start, "running_time_nanos": inst.elapsed().as_nanos() as u64, "query": query}));
            }
            Ok(Resp::ok(json!({ "queries": qs })))
        }
        ("queries", id, "", &Method::GET) => {
            let d = esql::async_id::decode(id)?;
            let task = format!("{}:{}", d.node_id, d.task);
            if d.node_id != NODE_ID {
                return Err(EsError::new(500, "exception", format!("task [{task}] belongs to the node [{}] which isn't part of the cluster and there is no record of the task", d.node_id)));
            }
            let running = ext(node).esql_running.lock().get(&task).cloned();
            match running {
                Some((_, start, inst, query)) => Ok(Resp::ok(json!({"start_time_millis": start, "running_time_nanos": inst.elapsed().as_nanos() as u64, "documents_found": 0, "values_loaded": 0, "query": query}))),
                None => Err(EsError::new(500, "exception", format!("task [{task}] isn't running and hasn't stored its results"))),
            }
        }
        _ => crate::not_found_handler(req),
    }
}

/// `PUT /_query/view/{name}`.
fn put_view(node: &Node, req: &Req, name: &str) -> R {
    let body = body_obj(req)?;
    check_body_fields(req, &body, "view", &["query"], &[])?;
    let Some(q) = body.get("query") else { return Err(EsError::illegal_argument("Required [query]")) };
    let q = q.as_str().unwrap_or("");
    let mut problems: Vec<String> = Vec::new();
    if let Some(m) = esql::views::validate_name(name) {
        problems.push(m);
    }
    if q.trim().is_empty() {
        problems.push("view query is missing or empty".into());
    }
    if !problems.is_empty() {
        return Err(EsError::action_validation(&problems.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    // Names are shared with indices, aliases and data streams.
    let resolves = {
        let mut o = ResolveOpts::from_params(&HashMap::new());
        o.allow_no_indices = true;
        node.resolve(name, o).map(|t| !t.is_empty()).unwrap_or(false)
    };
    let what = if node.get_index(name).is_some() {
        Some("concrete index")
    } else if node.data_stream(name).is_some() {
        Some("data_stream")
    } else if resolves {
        Some("alias")
    } else {
        None
    };
    if let Some(w) = what {
        return Err(EsError::illegal_argument(format!("view [{name}] cannot be created, an existing {w} with that name is present")));
    }
    // The query must parse (with no parameters).
    esql::parse::parse_statement(q, &Params::default())?;
    esql::views::store(node).put(name, q)?;
    Ok(Resp::ack())
}

// --------------------------------------------------------------------- SQL

/// The `esql` section of `GET /_xpack/usage`.
pub fn esql_usage(node: &Node) -> Value {
    let s = ext(node).esql_stats.lock().clone();
    let g = |k: String| s.get(&k).copied().unwrap_or(0);
    let features: Map<String, Value> = esql::telemetry::FEATURES.iter().map(|f| (f.to_string(), json!(g(format!("features.{f}"))))).collect();
    let functions: Map<String, Value> = esql::telemetry::FUNCTIONS.iter().map(|f| (f.to_string(), json!(g(format!("functions.{f}"))))).collect();
    let settings: Map<String, Value> = esql::telemetry::SETTINGS.iter().map(|f| (f.to_string(), json!(g(format!("settings.{f}"))))).collect();
    let took: Map<String, Value> = esql::telemetry::TOOK.iter().map(|(f, _)| (f.to_string(), json!(g(format!("took.{f}"))))).collect();
    let (rt, rf) = (g("queries.rest.total".into()), g("queries.rest.failed".into()));
    json!({
        "available": true,
        "enabled": true,
        "features": features,
        "settings": settings,
        "took": took,
        "functions": functions,
        "queries": {"rest": {"total": rt, "failed": rf}, "kibana": {"total": 0, "failed": 0}, "_all": {"total": rt, "failed": rf}},
    })
}

/// The caller's DLS/FLS over the targets a SQL query reads.
fn access_restrict(req: &Req) -> Option<esql::exec::Restrict> {
    let access = req.access.clone().filter(|a| a.is_restricted())?;
    Some(std::sync::Arc::new(move |targets| access.apply(targets)))
}

fn sql_stat(node: &Node, k: &str) {
    *ext(node).sql_stats.lock().entry(k.to_string()).or_insert(0) += 1;
}

fn sql_route(node: &Node, req: &Req, a: &str, b: &str, c: &str) -> R {
    match (a, b, c, &req.method) {
        ("", "", "", &Method::POST) | ("", "", "", &Method::GET) => sql_query(node, req),
        ("translate", "", "", &Method::POST) | ("translate", "", "", &Method::GET) => {
            let body = body_obj(req)?;
            let q = body.get("query").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["query is required"]))?;
            let fetch = body.get("fetch_size").and_then(Value::as_u64).unwrap_or(1000) as usize;
            sql_stat(node, "translate");
            Ok(Resp::ok(sql::translate(node, q, body.get("params"), fetch)?))
        }
        ("close", "", "", &Method::POST) => {
            let body = body_obj(req)?;
            let cur = body.get("cursor").and_then(Value::as_str).filter(|c| !c.is_empty()).ok_or_else(|| EsError::action_validation(&["[cursor] is required"]))?;
            esql::async_id::check_sql_cursor(cur)?;
            let removed = ext(node).sql_cursors.lock().remove(cur);
            let ok = removed.is_some();
            drop(removed);
            Ok(Resp::ok(json!({ "succeeded": ok })))
        }
        ("stats", "", "", &Method::GET) => {
            let s = ext(node).sql_stats.lock().clone();
            let g = |k: &str| s.get(k).copied().unwrap_or(0);
            Ok(Resp::ok(json!({
                "_nodes": {"total": 1, "successful": 1, "failed": 0},
                "cluster_name": node.cluster_name,
                "stats": [{
                    "node_id": NODE_ID,
                    "stats": {
                        "queries": {
                            "_all": {"total": g("total"), "paging": g("paging"), "failed": g("failed")},
                            "rest": {"total": g("total"), "paging": g("paging"), "failed": g("failed")},
                            "translate": {"total": g("translate"), "paging": 0, "failed": 0},
                        },
                        "features": {"command": 0, "groupby": g("groupby"), "having": g("having"), "join": 0, "limit": g("limit"), "local": 0, "orderby": g("orderby"), "subselect": 0, "where": g("where")}
                    }
                }]
            })))
        }
        ("async", "status", id, &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let v = ext(node).sql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            Ok(Resp::ok(json!({"id": v["id"], "is_running": false, "is_partial": false, "start_time_in_millis": v["start"], "expiration_time_in_millis": v["expires"], "completion_status": 200})))
        }
        ("async", "delete", id, &Method::DELETE) if !id.is_empty() => {
            let key = async_key(id)?;
            if ext(node).sql_async.lock().remove(&key).is_none() {
                return Err(EsError::resource_not_found(id.to_string()));
            }
            Ok(Resp::ack())
        }
        ("async", id, "", &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let v = ext(node).sql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            let t = table_from_json(&json!({"columns": v["columns"], "values": v["rows"]}));
            let mut meta = Map::new();
            meta.insert("id".into(), v["id"].clone());
            meta.insert("is_partial".into(), json!(false));
            meta.insert("is_running".into(), json!(false));
            render_table(req, &t, None, meta, false, false, "rows", vec![])
        }
        ("", ..) => method_not_allowed(req, &["GET", "POST"]),
        _ => crate::not_found_handler(req),
    }
}

fn sql_query(node: &Node, req: &Req) -> R {
    let body = body_obj(req)?;
    sql_stat(node, "total");
    let columnar = body.get("columnar").and_then(Value::as_bool).unwrap_or(false);
    let page_timeout = sql_page_timeout(req, &body)?;
    // This handler already runs on a blocking worker. Opportunistic cleanup
    // bounds abandoned cursor retention even while maintenance is busy.
    drop(search_ext::take_expired_sql_cursors(node, Instant::now()));
    // Cursor continuation.
    if let Some(cur) = body.get("cursor").and_then(Value::as_str).filter(|c| !c.is_empty()) {
        esql::async_id::check_sql_cursor(cur)?;
        sql_stat(node, "paging");
        let removed = ext(node).sql_cursors.lock().remove(cur);
        let cursor = removed.filter(|cursor| cursor.expires > Instant::now())
            .ok_or_else(|| EsError::new(404, "search_phase_execution_exception", "all shards failed")
                .with("phase", json!("query")).with("grouped", json!(true))
                .caused_by(EsError::new(404, "search_context_missing_exception", format!("No search context found for SQL cursor [{cur}]"))))?;
        return sql_page(node, req, cursor.columns, cursor.rows, cursor.fetch, columnar, false, page_timeout);
    }
    let q = body.get("query").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["one of [query] or [cursor] is required"]))?;
    let fetch = body.get("fetch_size").and_then(Value::as_u64).unwrap_or(1000).max(1) as usize;
    let upper = q.to_ascii_uppercase();
    for (kw, stat) in [(" WHERE ", "where"), (" GROUP BY ", "groupby"), (" HAVING ", "having"), (" ORDER BY ", "orderby"), (" LIMIT ", "limit")] {
        if upper.contains(kw) {
            sql_stat(node, stat);
        }
    }
    let shards = req.extensions.get::<esql::remote::ShardSourceExt>().map(|s| s.0.clone());
    if !body.get("wait_for_completion_timeout").is_some_and(|v| !v.is_null()) {
        if let Some(scan) = sql::scan_with(node, q, body.get("params"), body.get("filter"), access_restrict(req), shards.clone())? {
            let cols = scan.columns.clone();
            return sql_page(node, req, cols, search_ext::SqlRows::Scan(scan), fetch, columnar, true, page_timeout);
        }
    }
    let t = match sql::execute_with(node, q, body.get("params"), body.get("filter"), access_restrict(req), shards) {
        Ok(t) => t,
        Err(e) => {
            sql_stat(node, "failed");
            return Err(e);
        }
    };
    let cols: Vec<Value> = t.cols.iter().map(|c| json!({"name": c.name, "type": c.ty})).collect();
    // ES runs a SQL query asynchronously only when `wait_for_completion_timeout` is set;
    // the id is always returned, the response is kept only with `keep_on_completion`.
    if body.get("wait_for_completion_timeout").is_some_and(|v| !v.is_null()) {
        // Release each materialized row as it is converted, rather than keeping
        // both complete tables alive until the response has been built.
        let rows: Vec<Value> = t.rows.into_iter().map(|r| Value::Array(r.into_iter().map(|v| v.to_json()).collect())).collect();
        let (id, doc) = new_id();
        if body.get("keep_on_completion").and_then(Value::as_bool).unwrap_or(false) {
            let now = elasticsearch_engine::date::now_millis();
            let ka = body.get("keep_alive").and_then(Value::as_str).and_then(elasticsearch_engine::date::parse_time_value).unwrap_or(5 * 86_400_000);
            ext(node).sql_async.lock().insert(doc, json!({"id": id, "columns": cols, "rows": rows, "start": now, "expires": now + ka}));
        }
        let t2 = table_from_json(&json!({"columns": cols, "values": rows}));
        let mut meta = Map::new();
        meta.insert("id".into(), json!(id));
        meta.insert("is_partial".into(), json!(false));
        meta.insert("is_running".into(), json!(false));
        let format = fmt::pick_format(req.param("format"), req.headers.get("accept").and_then(|v| v.to_str().ok()))?;
        // The async status travels in headers for the text formats only.
        let headers = if matches!(format.as_str(), "json" | "yaml" | "cbor" | "smile") { vec![] } else { vec![("Async-ID", id.clone()), ("Async-partial", "false".into()), ("Async-running", "false".into())] };
        return render_table(req, &t2, None, meta, columnar, false, "rows", headers);
    }
    // Rows a page does not return stay executor values until their page.
    sql_page(node, req, cols, search_ext::SqlRows::Table(t.rows.into_iter()), fetch, columnar, true, page_timeout)
}

fn sql_page_timeout(req: &Req, body: &Value) -> Result<Duration> {
    let loc = top_level_location(req.body_str().unwrap_or(""), "page_timeout", true)
        .map(|(line, column)| format!("[{line}:{column}] ")).unwrap_or_default();
    let millis = match body.get("page_timeout") {
        None => 45_000,
        Some(Value::String(value)) => sql_timeout_millis(value).map_err(|cause|
            EsError::x_content_parse(format!("{loc}[sql/query] failed to parse field [page_timeout]")).caused_by(cause))?,
        Some(value) => {
            let token = match value {
                Value::Null => "VALUE_NULL", Value::Bool(true) => "VALUE_TRUE",
                Value::Bool(false) => "VALUE_FALSE", Value::Number(_) => "VALUE_NUMBER",
                Value::Array(_) => "START_ARRAY", Value::Object(_) => "START_OBJECT",
                Value::String(_) => unreachable!(),
            };
            return Err(EsError::x_content_parse(format!("{loc}[sql/query] page_timeout doesn't support values of type: {token}")));
        }
    };
    Ok(Duration::from_millis(millis.max(0) as u64))
}

fn sql_timeout_millis(value: &str) -> Result<i64> {
    let text = value.trim();
    if text == "0" { return Ok(0); }
    if text == "-1" { return Ok(-1); }
    let units = [("nanos", 1i128, 1_000_000i128), ("micros", 1, 1000),
        ("ms", 1, 1), ("s", 1000, 1), ("m", 60_000, 1),
        ("h", 3_600_000, 1), ("d", 86_400_000, 1)];
    let Some((number, multiplier, divisor)) = units.iter().find_map(|(suffix, mul, div)|
        text.strip_suffix(suffix).map(|number| (number.trim(), *mul, *div))) else {
        return Err(EsError::illegal_argument(format!("failed to parse setting [page_timeout] with value [{value}] as a time value: unit is missing or unrecognized")));
    };
    let number = number.parse::<i64>().map_err(|_| {
        let fractional = if number.contains('.') { ", fractional time values are not supported" } else { "" };
        EsError::illegal_argument(format!("failed to parse [{value}]{fractional}"))
            .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{number}\"")))
    })?;
    if number < -1 {
        return Err(EsError::illegal_argument(format!("failed to parse setting [page_timeout] with value [{value}] as a time value: negative durations are not supported")));
    }
    Ok(((number as i128 * multiplier) / divisor).clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

fn sql_page(node: &Node, req: &Req, cols: Vec<Value>, mut remaining: search_ext::SqlRows, fetch: usize, columnar: bool, first: bool, page_timeout: Duration) -> R {
    let rows = remaining.page(node, fetch)?;
    let cursor = if remaining.len() == 0 {
        None
    } else {
        let id = esql::async_id::new_sql_cursor();
        let expires = Instant::now().checked_add(page_timeout).ok_or_else(|| EsError::illegal_argument("page_timeout is too large"))?;
        ext(node).sql_cursors.lock().insert(id.clone(), search_ext::SqlCursor { columns: cols.clone(), rows: remaining, fetch, expires });
        Some(id)
    };
    let t = table_from_json(&json!({"columns": cols, "values": rows}));
    let format = fmt::pick_format(req.param("format"), req.headers.get("accept").and_then(|v| v.to_str().ok()))?;
    let mut headers = vec![];
    if let Some(c) = &cursor {
        headers.push(("Cursor", c.clone()));
    }
    if format == "json" || format == "yaml" || format == "cbor" {
        let mut meta = Map::new();
        if first {
            meta.insert("columns".into(), Value::Array(cols));
        }
        if columnar {
            let ncol = t.cols.len();
            let values: Vec<Value> = (0..ncol).map(|i| Value::Array(t.rows.iter().map(|r| r[i].to_json()).collect())).collect();
            meta.insert("values".into(), Value::Array(values));
        } else {
            meta.insert("rows".into(), Value::Array(t.rows.iter().map(|r| Value::Array(r.iter().map(Val::to_json).collect())).collect()));
        }
        if let Some(c) = cursor {
            meta.insert("cursor".into(), json!(c));
        }
        let v = Value::Object(meta);
        let mut resp = match format.as_str() {
            "json" => Resp::ok(v),
            "yaml" => typed(200, fmt::to_yaml(&v).into_bytes(), fmt::content_type("yaml").unwrap()),
            _ => typed(200, fmt::to_cbor(&v), fmt::content_type("cbor").unwrap()),
        };
        resp.headers.extend(headers);
        return Ok(resp);
    }
    if format == "smile" || format == "arrow" {
        return Err(EsError::illegal_argument(format!("format [{format}] is not supported by this node")));
    }
    let header = first && req.param("header").map(|h| h != "absent").unwrap_or(true);
    let s = fmt::text_table(&t, &format, req.param("delimiter"), header, false)?;
    let mut resp = typed(200, s.into_bytes(), fmt::content_type(&format).unwrap());
    resp.headers.extend(headers);
    Ok(resp)
}

// --------------------------------------------------------------------- EQL

/// A concrete index name in a REST path.
fn eql_path(index: &str) -> String {
    index.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn eql_route(node: &Node, req: &Req, index: Option<&str>, b: &str, c: &str) -> R {
    match (index, b, c, &req.method) {
        (Some(ix), "", "", &Method::GET) | (Some(ix), "", "", &Method::POST) => {
            let body = body_obj(req)?;
            check_body_fields(req, &body, "eql/search", EQL_BODY_FIELDS, EQL_BOOL_FIELDS)?;
            // Partial-result flags (URL parameters override the body); with a
            // single node there are no shard failures, so they only validate.
            for k in ["allow_partial_search_results", "allow_partial_sequence_results"] {
                req.flag(k)?;
            }
            // On a cluster every index search runs over all its shards: through
            // the coordinator, unless this node holds every shard of every
            // target and the caller reads them unrestricted (no document or
            // field security to apply), when the stage searches read the
            // local shards directly (no REST round trip, JSON response and
            // parse per stage search).
            let all_local = req.access.is_none()
                && !elasticsearch_engine::ccr::remote::has_remote_expression(ix)
                && node.resolve(ix, {
                    // As the EQL search resolves its targets (`eql_resolve_opts`).
                    let mut o = ResolveOpts::from_params(&req.params);
                    if !req.params.contains_key("ignore_unavailable") {
                        o.ignore_unavailable = true;
                    }
                    o
                }).is_ok_and(|ts| !ts.is_empty() && ts.iter().all(|t| !t.svc.is_sparse()));
            let remote = req.extensions.get::<crate::ClusterRest>().filter(|_| !all_local).cloned().map(|rest| -> eql::RemoteSearch {
                Arc::new(move |index: &str, routing: Option<&str>, body: &Value| {
                    let mut path = format!("/{}/_search?ignore_unavailable=true", eql_path(index));
                    if let Some(routing) = routing { path.push_str(&format!("&routing={}", eql_path(routing))); }
                    (rest.0)("POST", &path, Some(body.clone()))
                })
            });
            let scope = eql::remote_scope(remote);
            let mut resp = eql::search(node, ix, &req.params, &body)?;
            drop(scope);
            let flag = |k: &str| -> Result<bool> {
                match req.param(k) {
                    Some(_) => req.flag(k),
                    None => Ok(body.get(k).and_then(Value::as_bool).unwrap_or(false)),
                }
            };
            let keep = flag("keep_on_completion")?;
            // ES runs the search asynchronously (and returns an id) only when
            // `wait_for_completion_timeout` is set; the response is kept only
            // with `keep_on_completion`.
            let wfct = req.param("wait_for_completion_timeout").map(str::to_string).or_else(|| body.get("wait_for_completion_timeout").and_then(Value::as_str).map(str::to_string));
            if wfct.is_some() {
                let (id, doc) = new_id();
                let mut m = Map::new();
                m.insert("id".into(), json!(id));
                m.extend(resp.as_object().cloned().unwrap_or_default());
                resp = Value::Object(m);
                if keep {
                    let now = elasticsearch_engine::date::now_millis();
                    let ka = req.param("keep_alive").or_else(|| body.get("keep_alive").and_then(Value::as_str)).and_then(elasticsearch_engine::date::parse_time_value).unwrap_or(5 * 86_400_000);
                    let mut stored = resp.clone();
                    stored["__expires"] = json!(now + ka);
                    ext(node).eql_async.lock().insert(doc, stored);
                }
            }
            Ok(Resp::ok(resp))
        }
        (None, "status", id, &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let v = ext(node).eql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            Ok(Resp::ok(json!({"id": v["id"], "is_running": false, "is_partial": false, "expiration_time_in_millis": v["__expires"], "completion_status": 200})))
        }
        (None, id, "", &Method::GET) if !id.is_empty() => {
            let key = async_key(id)?;
            let mut v = ext(node).eql_async.lock().get(&key).cloned().ok_or_else(|| EsError::resource_not_found(id.to_string()))?;
            if let Some(o) = v.as_object_mut() {
                o.remove("__expires");
            }
            Ok(Resp::ok(v))
        }
        (None, id, "", &Method::DELETE) if !id.is_empty() => {
            let key = async_key(id)?;
            if ext(node).eql_async.lock().remove(&key).is_none() {
                return Err(EsError::resource_not_found(id.to_string()));
            }
            Ok(Resp::ack())
        }
        _ => crate::not_found_handler(req),
    }
}

// --------------------------------------------------------------- rank_eval

fn rank_eval_api(node: &Node, req: &Req, index: Option<&str>) -> R {
    let body = body_obj(req)?;
    let resolve = ResolveOpts::from_params(&req.params);
    let params: HashMap<String, String> = HashMap::new();
    let cluster = req.extensions.get::<crate::ClusterSearchExecutor>();
    let search = |b: &Value| -> Result<Value> {
        let s = match cluster {
            Some(executor) => (executor.0)(index, b)?,
            None => run_search(node, b, &params, index, resolve.clone(), req.access.as_deref())?,
        };
        serde_json::from_str(&s).map_err(|e| EsError::internal(e.to_string()))
    };
    let render = |src: &str, p: &Value| template::render_to_json(src, p);
    Ok(Resp::ok(rank_eval::run(&body, &search, &render)?))
}

// ------------------------------------------------------- search applications

/// The template a search application without one uses (ES 8.19's
/// `SearchApplicationTemplate.DEFAULT_TEMPLATE`).
fn default_app_template() -> Value {
    json!({"script": {
        "source": "{\n  \"query\": {\n    \"query_string\": {\n        \"query\": \"{{query_string}}\",\n        \"default_field\": \"{{default_field}}\"\n        }\n    }\n}\n",
        "lang": "mustache",
        "params": {"default_field": "*", "query_string": "*"}
    }})
}

const DEFAULT_TEMPLATE_WARNING: &str = "Using default search application template which is subject to change. We recommend storing a template to avoid breaking changes.";

fn default_template_warning() -> (&'static str, String) {
    ("Warning", format!("299 Elasticsearch-{}-{} \"{DEFAULT_TEMPLATE_WARNING}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH))
}

/// Render the application's template (the default one when it stores none)
/// with the stored params overridden by `req_params`, as the search request
/// body ES would run: the rendered text may carry trailing commas, and the
/// query is re-serialised the way ES's query builders print it.
fn app_render(app: &Value, req_params: Option<&Value>) -> Result<Value> {
    let tpl = app.get("template").filter(|t| !t.is_null()).cloned().unwrap_or_else(default_app_template);
    let script = tpl.get("script").cloned().unwrap_or_default();
    let src = template::source_text(script.get("source").unwrap_or(&Value::Null));
    // The dictionary validates the request's params alone, before the
    // template's defaults are merged in (`SearchApplicationTemplateService.renderQuery`).
    let given = match req_params {
        Some(Value::Object(p)) => p.clone(),
        _ => Map::new(),
    };
    if let Some(dict) = tpl.get("dictionary").filter(|d| d.is_object()) {
        validate_app_params(dict, &given)?;
    }
    let mut params = script.get("params").and_then(Value::as_object).cloned().unwrap_or_default();
    params.extend(given);
    let mut rendered = template::render_to_json_lenient(&src, &Value::Object(params))?;
    normalize_rendered_query(&mut rendered);
    Ok(rendered)
}

/// Validate the render params against the template's `dictionary` (a JSON
/// schema: `required`, `additionalProperties`, `properties.<p>.type`).
fn validate_app_params(dict: &Value, params: &Map<String, Value>) -> Result<()> {
    let mut errs: Vec<String> = Vec::new();
    let props = dict.get("properties").and_then(Value::as_object);
    for r in dict.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
        if !params.contains_key(r) {
            errs.push(format!("$.{r}: is missing but it is required"));
        }
    }
    if dict.get("additionalProperties") == Some(&Value::Bool(false)) {
        for k in params.keys() {
            if !props.is_some_and(|p| p.contains_key(k)) {
                errs.push(format!("$.{k}: is not defined in the schema and the schema does not allow additional properties"));
            }
        }
    }
    let kind = |v: &Value| match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => if n.is_f64() { "number" } else { "integer" },
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    for (k, def) in props.into_iter().flatten() {
        let (Some(v), Some(t)) = (params.get(k), def.get("type").and_then(Value::as_str)) else { continue };
        let actual = kind(v);
        let ok = actual == t || (t == "number" && actual == "integer");
        if !ok {
            errs.push(format!("$.{k}: {actual} found, {t} expected"));
        }
    }
    if errs.is_empty() {
        return Ok(());
    }
    let msg = errs.iter().enumerate().map(|(i, e)| format!("{}: {e};", i + 1)).collect::<Vec<_>>().join(" ");
    Err(EsError::new(400, "validation_exception", format!("Validation Failed: {msg}")))
}

/// `Float.toString` of a boost parsed from `field^boost`.
fn java_float(s: &str) -> Option<String> {
    let f: f32 = s.parse().ok()?;
    if !f.is_finite() {
        return None;
    }
    Some(if f.fract() == 0.0 && f.abs() < 1e7 { format!("{f:.1}") } else { f.to_string() })
}

/// Print the rendered query the way ES's query builders serialise a parsed
/// query: `term`/`match` short forms become their object forms and
/// `multi_match` field boosts are floats.
fn normalize_rendered_query(v: &mut Value) {
    match v {
        Value::Array(a) => a.iter_mut().for_each(normalize_rendered_query),
        Value::Object(m) => {
            for (k, val) in m.iter_mut() {
                match (k.as_str(), &mut *val) {
                    ("term" | "match", Value::Object(inner)) if inner.len() == 1 => {
                        let key = if k == "term" { "value" } else { "query" };
                        for (_, fv) in inner.iter_mut() {
                            if !fv.is_object() {
                                *fv = json!({ key: fv.take() });
                            }
                        }
                    }
                    ("multi_match", Value::Object(inner)) => {
                        if let Some(Value::Array(fields)) = inner.get_mut("fields") {
                            for f in fields.iter_mut() {
                                if let Some(s) = f.as_str() {
                                    if let Some((name, boost)) = s.rsplit_once('^') {
                                        if let Some(b) = java_float(boost) {
                                            *f = Value::String(format!("{name}^{b}"));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => normalize_rendered_query(val),
                }
            }
        }
        _ => {}
    }
}

/// Validate a search application `template` (`SearchApplicationTemplate`
/// parsing): a mustache script with an inline source, and a dictionary
/// that is a JSON schema.
fn validate_app_template(tpl: &Value) -> Result<Value> {
    let failed = |col: usize| EsError::x_content_parse(format!("[1:{col}] [search_application] failed to parse field [template]"));
    let o = tpl.as_object().ok_or_else(|| failed(2))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "script" | "dictionary") {
            return Err(failed(2).caused_by(EsError::x_content_parse(format!("[1:2] [search_template] unknown field [{k}]"))));
        }
    }
    let Some(script) = o.get("script").and_then(Value::as_object) else {
        return Err(EsError::action_validation(&["script required for template"]));
    };
    let build_failed = |cause: EsError| failed(2).caused_by(EsError::x_content_parse("Failed to build [search_template] after last required field arrived").caused_by(cause));
    if script.contains_key("id") || !script.contains_key("source") {
        return Err(build_failed(EsError::illegal_argument("Search Application templates must specify an inline [source] script")));
    }
    let lang = script.get("lang").and_then(Value::as_str).unwrap_or("mustache");
    if lang != "mustache" {
        return Err(build_failed(EsError::illegal_argument(format!("Search Application templates must be mustache scripts, but [{lang}] was given"))));
    }
    if let Some(dict) = o.get("dictionary") {
        const TYPES: &[&str] = &["string", "number", "integer", "boolean", "object", "array", "null"];
        let bad = dict.get("properties").and_then(Value::as_object).into_iter().flatten().find_map(|(name, def)| match def.get("type") {
            Some(Value::String(t)) if !TYPES.contains(&t.as_str()) => Some(format!("$.properties.{name}.type: does not have a value in the enumeration [array, boolean, integer, null, number, object, string]")),
            Some(Value::Array(ts)) => ts.iter().find(|t| !t.as_str().is_some_and(|t| TYPES.contains(&t))).map(|t| format!("$.properties.{name}.type: {t} is not a valid type")),
            _ => None,
        });
        if let Some(msg) = bad {
            return Err(failed(2).caused_by(EsError::x_content_parse("[1:2] [search_template] failed to parse field [dictionary]").caused_by(EsError::new(400, "validation_exception", format!("Validation Failed: 1: {msg};")))));
        }
    }
    let mut out = tpl.clone();
    if let Some(sm) = out.get_mut("script").and_then(Value::as_object_mut) {
        sm.entry("lang").or_insert(json!("mustache"));
        if sm.get("source").is_some_and(Value::is_object) {
            sm.entry("options").or_insert(json!({"content_type": "application/json;charset=utf-8"}));
        }
    }
    Ok(out)
}

fn app_not_found(name: &str) -> EsError {
    EsError::resource_not_found(format!("Search Application [{name}] not found"))
}

fn search_app_route(node: &Node, req: &Req, name: &str, sub: &str) -> R {
    let st = ext(node);
    match (name, sub, &req.method) {
        ("", "", &Method::GET) => {
            let from = req.param("from").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let size = req.param("size").and_then(|v| v.parse().ok()).unwrap_or(100usize);
            let q = req.param("q").unwrap_or("*");
            let g = st.search_apps.read();
            let matched: Vec<&Value> = g.iter().filter(|(n, _)| q == "*" || n.as_str() == q || (q.ends_with('*') && n.starts_with(q.trim_end_matches('*')))).map(|(_, v)| v).collect();
            let results: Vec<Value> = matched
                .iter()
                .skip(from)
                .take(size)
                .map(|a| {
                    let mut r = json!({"name": a["name"], "updated_at_millis": a["updated_at_millis"]});
                    if let Some(c) = a.get("analytics_collection_name") {
                        r["analytics_collection_name"] = c.clone();
                    }
                    r
                })
                .collect();
            Ok(Resp::ok(json!({"count": matched.len(), "results": results})))
        }
        (n, "", &Method::PUT) if !n.is_empty() => {
            let body = body_obj(req)?;
            for k in body.as_object().map(|o| o.keys()).into_iter().flatten() {
                if !matches!(k.as_str(), "indices" | "template" | "analytics_collection_name" | "name" | "updated_at_millis") {
                    return Err(EsError::x_content_parse(format!("[1:2] [search_application] unknown field [{k}]")));
                }
            }
            let indices: Vec<String> = match body.get("indices") {
                Some(Value::Array(a)) if !a.is_empty() => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                _ => return Err(EsError::action_validation(&["indices are missing"])),
            };
            // The alias update runs as the user (`IndicesAliasesRequest` through
            // the security filter): the indices and the alias need the privilege.
            if let Some(acc) = req.access.as_deref() {
                let mut names = indices.clone();
                names.push(n.to_string());
                acc.authorize_names(node, "indices:admin/aliases", &names)?;
            }
            for ix in &indices {
                if let Err(e) = node.resolve(ix, ResolveOpts::from_params(&HashMap::new())) {
                    return Err(EsError::illegal_argument(format!("no such index [{ix}]")).caused_by(e));
                }
            }
            let exists = st.search_apps.read().contains_key(n);
            if exists && req.flag("create")? {
                return Err(EsError::new(409, "version_conflict_engine_exception", format!("[{n}]: version conflict, document already exists (current version [1])")));
            }
            let mut app = json!({"name": n, "indices": indices, "updated_at_millis": elasticsearch_engine::date::now_millis()});
            if let Some(c) = body.get("analytics_collection_name") {
                app["analytics_collection_name"] = c.clone();
            }
            // Without a template the application renders the default one (and
            // says so, as the default is subject to change). A stored template
            // is validated for shape only; it renders at search time.
            let mut warn = false;
            match body.get("template") {
                Some(tpl) if !tpl.is_null() => {
                    app["template"] = validate_app_template(tpl)?;
                }
                _ => warn = true,
            }
            // The application is an alias over its indices.
            let previous: Vec<String> = st.search_apps.read().get(n).and_then(|a| a["indices"].as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let mut actions: Vec<Value> = previous.iter().filter(|p| !indices.contains(p)).filter(|p| node.get_index(p).is_some()).map(|p| json!({"remove": {"index": p, "alias": n}})).collect();
            actions.extend(indices.iter().map(|ix| json!({"add": {"index": ix, "alias": n}})));
            node.update_aliases(&actions)?;
            st.search_apps.write().insert(n.to_string(), app);
            st.persist()?;
            let mut r = Resp::ok(json!({"result": if exists { "updated" } else { "created" }}));
            if warn {
                r.headers.push(default_template_warning());
            }
            Ok(r)
        }
        (n, "", &Method::GET) if !n.is_empty() => {
            let mut app = st.search_apps.read().get(n).cloned().ok_or_else(|| app_not_found(n))?;
            let mut r = Resp::ok(Value::Null);
            if app.get("template").is_none_or(Value::is_null) {
                app["template"] = default_app_template();
                r.headers.push(default_template_warning());
            }
            r.body = RespBody::Json(app);
            Ok(r)
        }
        (n, "", &Method::DELETE) if !n.is_empty() => {
            let Some(app) = st.search_apps.write().remove(n) else {
                return Err(app_not_found(n));
            };
            st.persist()?;
            let actions: Vec<Value> = app["indices"].as_array().into_iter().flatten().filter_map(Value::as_str).filter(|ix| node.get_index(ix).is_some_and(|s| s.aliases().contains_key(n))).map(|ix| json!({"remove": {"index": ix, "alias": n}})).collect();
            if !actions.is_empty() {
                let _ = node.update_aliases(&actions);
            }
            Ok(Resp::ack())
        }
        (n, "_search", &Method::POST) | (n, "_search", &Method::GET) => {
            let app = st.search_apps.read().get(n).cloned().ok_or_else(|| app_not_found(n))?;
            let body = body_obj(req)?;
            let rendered = app_render(&app, body.get("params"))?;
            let rendered = search_ext::rewrite_query(node, &rendered)?;
            let indices: Vec<String> = app["indices"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let json = if let Some(plan) = req.extensions.get::<crate::PreparedSearchApplication>() {
                if plan.indices.iter().any(|index| !indices.contains(index)) {
                    return Err(EsError::illegal_argument("application query targets changed during execution"));
                }
                run_search(node, &plan.query, &req.params, Some(&plan.indices.join(",")), ResolveOpts::from_params(&req.params), req.access.as_deref())?
            } else if let Some(executor) = req.extensions.get::<crate::SearchApplicationExecutor>() {
                (executor.0)(&indices, &rendered, &req.params)?
            } else {
                run_search(node, &rendered, &req.params, Some(&indices.join(",")), ResolveOpts::from_params(&req.params), req.access.as_deref())?
            };
            Ok(Resp::raw(200, json))
        }
        (n, "_render_query", &Method::POST) => {
            let app = st.search_apps.read().get(n).cloned().ok_or_else(|| app_not_found(n))?;
            let body = body_obj(req)?;
            Ok(Resp::ok(app_render(&app, body.get("params"))?))
        }
        _ => crate::not_found_handler(req),
    }
}

// --------------------------------------------------------------- query rules

fn max_rules_per_ruleset(node: &Node) -> usize {
    let settings = node.cluster_settings.read();
    let key = "xpack.applications.rules.max_rules_per_ruleset";
    settings.1.get(key).or_else(|| settings.0.get(key))
        .and_then(|value| value.as_u64().or_else(|| value.as_str().and_then(|s| s.parse().ok())))
        .and_then(|value| usize::try_from(value).ok()).unwrap_or(100)
}

fn normalize_rule(rule_id: &str, body: &Value) -> Result<Value> {
    search_ext::validate_rule_def(body)?;
    let mut r = Map::new();
    r.insert("rule_id".into(), json!(rule_id));
    r.insert("type".into(), body["type"].clone());
    let crit = match &body["criteria"] {
        Value::Array(a) => Value::Array(a.clone()),
        o => Value::Array(vec![o.clone()]),
    };
    r.insert("criteria".into(), crit);
    r.insert("actions".into(), body["actions"].clone());
    if let Some(p) = body.get("priority") {
        r.insert("priority".into(), p.clone());
    }
    Ok(Value::Object(r))
}

/// Rules with a priority first (ascending), then the rest in their order.
fn sort_rules(rules: &mut [Value]) {
    rules.sort_by_key(|r| match r.get("priority").and_then(Value::as_i64) {
        Some(p) => (0, p),
        None => (1, 0),
    });
}

fn ruleset_not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("Query ruleset [{id}] not found"))
}

/// Tally criteria types and rule types of `rules` into the two maps.
fn count_rule_types(rules: &[Value], crit: &mut Map<String, Value>, types: &mut Map<String, Value>) {
    let mut bump = |m: &mut Map<String, Value>, k: &str| {
        let n = m.get(k).and_then(Value::as_u64).unwrap_or(0);
        m.insert(k.to_string(), json!(n + 1));
    };
    for r in rules {
        for c in r["criteria"].as_array().into_iter().flatten() {
            bump(crit, c["type"].as_str().unwrap_or(""));
        }
        bump(types, r["type"].as_str().unwrap_or(""));
    }
}

/// `enterprise_search.query_rulesets` of `GET /_xpack/usage`
/// (`EnterpriseSearchUsageTransportAction.addQueryRulesetUsage`).
pub fn query_rules_usage(node: &Node) -> Value {
    let st = ext(node);
    let g = st.rulesets.read();
    let (mut crit, mut types) = (Map::new(), Map::new());
    let counts: Vec<usize> = g.values().map(Vec::len).collect();
    for rules in g.values() {
        count_rule_types(rules, &mut crit, &mut types);
    }
    let mut out = json!({
        "total_count": g.len(),
        "total_rule_count": counts.iter().sum::<usize>(),
        "min_rule_count": counts.iter().min().copied().unwrap_or(0),
        "max_rule_count": counts.iter().max().copied().unwrap_or(0),
    });
    if !crit.is_empty() {
        out["rule_criteria_total_counts"] = Value::Object(crit);
    }
    if !types.is_empty() {
        out["rule_type_total_counts"] = Value::Object(types);
    }
    out
}

fn query_rules_route(node: &Node, req: &Req, rs: &str, sub: &str, rule: &str) -> R {
    let st = ext(node);
    let max_rules = max_rules_per_ruleset(node);
    match (rs, sub, rule, &req.method) {
        ("", "", "", &Method::GET) => {
            let from = req.param("from").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let size = req.param("size").and_then(|v| v.parse().ok()).unwrap_or(100usize);
            let g = st.rulesets.read();
            let results: Vec<Value> = g
                .iter()
                .skip(from)
                .take(size)
                .map(|(id, rules)| {
                    let (mut crit, mut types) = (Map::new(), Map::new());
                    count_rule_types(rules, &mut crit, &mut types);
                    json!({"ruleset_id": id, "rule_total_count": rules.len(), "rule_criteria_types_counts": crit, "rule_type_counts": types})
                })
                .collect();
            Ok(Resp::ok(json!({"count": g.len(), "results": results})))
        }
        (id, "", "", &Method::PUT) if !id.is_empty() => {
            let body = body_obj(req)?;
            // `rules` is a list, or a single rule object.
            let rules_v: Vec<Value> = match body.get("rules") {
                Some(Value::Array(a)) if !a.is_empty() => a.clone(),
                Some(o @ Value::Object(_)) => vec![o.clone()],
                _ => return Err(EsError::action_validation(&["rules cannot be empty"])),
            };
            if rules_v.len() > max_rules {
                return Err(EsError::action_validation(&[&format!("The number of rules in a ruleset cannot exceed [{max_rules}]")]));
            }
            let mut rules = Vec::new();
            for r in &rules_v {
                let rid = r.get("rule_id").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["rule_id cannot be null or empty"]))?;
                rules.push(normalize_rule(rid, r)?);
            }
            sort_rules(&mut rules);
            let created = st.rulesets.write().insert(id.to_string(), rules).is_none();
            st.persist()?;
            Ok(Resp::ok(json!({"result": if created { "created" } else { "updated" }})))
        }
        (id, "", "", &Method::GET) if !id.is_empty() => {
            let rules = st.rulesets.read().get(id).cloned().ok_or_else(|| ruleset_not_found(id))?;
            Ok(Resp::ok(json!({"ruleset_id": id, "rules": rules})))
        }
        (id, "", "", &Method::DELETE) if !id.is_empty() => {
            if st.rulesets.write().remove(id).is_none() {
                return Err(ruleset_not_found(id));
            }
            st.persist()?;
            Ok(Resp::ack())
        }
        (id, "_test", "", &Method::POST) => {
            let body = body_obj(req)?;
            // `TestQueryRulesetAction.Request` (ObjectParser `test_query_ruleset_request`).
            if let Some(o) = body.as_object() {
                for (k, v) in o {
                    if k != "match_criteria" {
                        return Err(EsError::unknown_field_xc("test_query_ruleset_request", k, None, &["match_criteria"]));
                    }
                    if !v.is_object() {
                        return Err(EsError::unsupported_type_xc("test_query_ruleset_request", "match_criteria", v, None));
                    }
                }
            }
            let criteria = body.get("match_criteria").and_then(Value::as_object).cloned().ok_or_else(|| EsError::illegal_argument("Required [match_criteria]"))?;
            if !st.rulesets.read().contains_key(id) {
                return Err(EsError::resource_not_found(id.to_string()));
            }
            let matched = search_ext::matching_rules(node, &[id.to_string()], &criteria)?;
            let list: Vec<Value> = matched.iter().map(|(rs, r)| json!({"ruleset_id": rs, "rule_id": r["rule_id"]})).collect();
            Ok(Resp::ok(json!({"total_matched_rules": list.len(), "matched_rules": list})))
        }
        (id, "_rule", rid, &Method::PUT) if !rid.is_empty() => {
            let body = body_obj(req)?;
            let rule = normalize_rule(rid, &body)?;
            let created = {
                let mut g = st.rulesets.write();
                let rules = g.entry(id.to_string()).or_default();
                let created = match rules.iter_mut().find(|r| r["rule_id"] == json!(rid)) {
                    Some(r) => {
                        *r = rule;
                        false
                    }
                    None => {
                        if rules.len() >= max_rules {
                            return Err(EsError::illegal_argument(format!("The number of rules in a ruleset cannot exceed [{max_rules}]")));
                        }
                        rules.push(rule);
                        true
                    }
                };
                sort_rules(rules);
                created
            };
            st.persist()?;
            Ok(Resp::ok(json!({"result": if created { "created" } else { "updated" }})))
        }
        (id, "_rule", rid, &Method::GET) if !rid.is_empty() => {
            let g = st.rulesets.read();
            let rules = g.get(id).ok_or_else(|| ruleset_not_found(id))?;
            let r = rules.iter().find(|r| r["rule_id"] == json!(rid)).cloned().ok_or_else(|| EsError::resource_not_found(format!("rule [{rid}] in ruleset [{id}] not found")))?;
            Ok(Resp::ok(r))
        }
        (id, "_rule", rid, &Method::DELETE) if !rid.is_empty() => {
            {
                let mut g = st.rulesets.write();
                let rules = g.get_mut(id).ok_or_else(|| ruleset_not_found(id))?;
                let before = rules.len();
                rules.retain(|r| r["rule_id"] != json!(rid));
                if rules.len() == before {
                    return Err(EsError::resource_not_found(format!("rule [{rid}] in ruleset [{id}] not found")));
                }
                // A ruleset without rules is gone.
                if rules.is_empty() {
                    g.remove(id);
                }
            }
            st.persist()?;
            Ok(Resp::ack())
        }
        _ => crate::not_found_handler(req),
    }
}

// ----------------------------------------------------------------- synonyms

fn page_params(req: &Req) -> Result<(usize, usize)> {
    let p = |k: &str, d: i64| -> Result<i64> {
        match req.param(k) {
            None => Ok(d),
            Some(v) => v.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("[{k}] must be a number"))),
        }
    };
    let from = p("from", 0)?;
    let size = p("size", 10)?;
    if from < 0 {
        return Err(EsError::action_validation(&["[from] must be a positive integer"]));
    }
    if size < 0 {
        return Err(EsError::action_validation(&["[size] must be a positive integer"]));
    }
    if from > 100_000 || size > 100_000 {
        return Err(EsError::action_validation(&["[from] and [size] must be less than or equal to [100000]"]));
    }
    Ok((from as usize, size as usize))
}

/// `SynonymUpdateResponse` (201 when the set / rule was created).
fn synonym_update(created: bool, reload: Option<Value>) -> Resp {
    let mut out = json!({"result": if created { "created" } else { "updated" }});
    if let Some(r) = reload {
        out["reload_analyzers_details"] = r;
    }
    Resp::json(if created { 201 } else { 200 }, out)
}

fn synonyms_route(node: &Node, req: &Req, set: &str, rule: &str) -> R {
    let st = ext(node);
    let not_found = |s: &str| EsError::resource_not_found(format!("synonyms set [{s}] not found"));
    match (set, rule, &req.method) {
        ("", "", &Method::GET) => {
            let (from, size) = page_params(req)?;
            let g = st.synonyms.read();
            let results: Vec<Value> = g.iter().skip(from).take(size).map(|(k, rs)| json!({"synonyms_set": k, "count": rs.len()})).collect();
            Ok(Resp::ok(json!({"count": g.len(), "results": results})))
        }
        (s, "", &Method::PUT) if !s.is_empty() => {
            let body = body_obj(req)?;
            let (created, reload) = search_ext::put_synonyms(node, s, &body, req.bool_param("refresh")?.unwrap_or(true), req.bool_param("append")?.unwrap_or(false))?;
            Ok(synonym_update(created, reload))
        }
        (s, "", &Method::GET) if !s.is_empty() => {
            // `RestGetSynonymsAction`: offset paging, or `search_after` (rule id) cursor paging.
            if req.param("from").is_some() && req.param("search_after").is_some() {
                return Err(EsError::illegal_argument("[from] and [search_after] cannot be used together"));
            }
            let (from, size) = page_params(req)?;
            let after = req.param("search_after").filter(|a| !a.trim().is_empty());
            let g = st.synonyms.read();
            let rs = g.get(s).ok_or_else(|| not_found(s))?;
            let mut sorted: Vec<&(String, String)> = rs.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let page = sorted.iter().filter(|(id, _)| after.is_none_or(|a| id.as_str() > a)).skip(if after.is_some() { 0 } else { from });
            let items: Vec<Value> = page.take(size).map(|(id, syn)| json!({"id": id, "synonyms": syn})).collect();
            Ok(Resp::ok(json!({"count": rs.len(), "synonyms_set": items})))
        }
        (s, "", &Method::DELETE) if !s.is_empty() => {
            search_ext::delete_synonyms(node, s)?;
            Ok(Resp::ack())
        }
        (s, r, &Method::PUT) => {
            let body = body_obj(req)?;
            let (created, reload) = search_ext::put_synonym_rule(node, s, r, &body, req.bool_param("refresh")?.unwrap_or(true))?;
            Ok(synonym_update(created, reload))
        }
        (s, r, &Method::GET) => {
            let g = st.synonyms.read();
            let rs = g.get(s).ok_or_else(|| not_found(s))?;
            let (id, syn) = rs.iter().find(|(id, _)| id == r).ok_or_else(|| EsError::resource_not_found(format!("synonym rule [{r}] not found")))?;
            Ok(Resp::ok(json!({"id": id, "synonyms": syn})))
        }
        (s, r, &Method::DELETE) => {
            let reload = search_ext::delete_synonym_rule(node, s, r, req.bool_param("refresh")?.unwrap_or(true))?;
            let mut out = json!({"result": "deleted"});
            if let Some(r) = reload {
                out["reload_analyzers_details"] = r;
            }
            Ok(Resp::ok(out))
        }
        _ => crate::not_found_handler(req),
    }
}
