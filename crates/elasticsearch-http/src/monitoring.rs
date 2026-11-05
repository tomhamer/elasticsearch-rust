//! Legacy monitoring bulk (`POST|PUT /_monitoring/bulk`): Beats, Logstash and
//! Kibana internal collection. Each `index` action becomes one document in
//! `.monitoring-{system}-7-{yyyy.MM.dd}`, wrapped in the monitoring envelope
//! (`cluster_uuid`, `timestamp`, `interval_ms`, `type`, `source_node`, and the
//! payload under its type name). Documents are only written while
//! `xpack.monitoring.collection.enabled` is true; otherwise the request is
//! answered with `ignored: true`.

use crate::observability::TRANSPORT_ADDRESS;
use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::{IndexService, WriteReq};
use elasticsearch_engine::node::{Node, NODE_ID};
use elasticsearch_engine::shard::WriteCond;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

/// `MonitoredSystem`s accepted by the bulk API and their API versions.
const SYSTEMS: &[&str] = &["kibana", "logstash", "beats"];
const API_VERSIONS: &[&str] = &["7", "6"];
const TEMPLATE_VERSION: &str = "7";

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    if req.segs.len() != 2 || req.seg(1) != "bulk" {
        return crate::not_found_handler(&req);
    }
    if !matches!(req.method, Method::POST | Method::PUT) {
        return method_not_allowed(&req, &["POST", "PUT"]);
    }
    blocking(move || bulk(&node, &req)).await
}

struct Doc {
    ty: String,
    id: Option<String>,
    source: Value,
}

fn required(req: &Req, k: &str) -> Result<String, EsError> {
    match req.param(k) {
        Some(v) if !v.is_empty() => Ok(v.to_string()),
        _ => Err(EsError::illegal_argument(format!("no [{k}] for monitoring bulk request"))),
    }
}

/// The `null_pointer_exception` ES answers for an action without `_type`.
fn npe() -> Resp {
    let e = json!({"type": "null_pointer_exception", "reason": null});
    Resp::json(500, json!({"error": {"root_cause": [e.clone()], "type": "null_pointer_exception", "reason": null}, "status": 500}))
}

/// Parse the bulk body: `index` / `create` actions only, `_index` absent or `_data`
/// (the latter dropped, as ES has done since 5.5).
fn parse(text: &str) -> Result<Result<Vec<Doc>, Resp>, EsError> {
    if !text.ends_with('\n') {
        return Err(EsError::illegal_argument("The bulk request must be terminated by a newline [\\n]"));
    }
    let lines: Vec<&str> = text.split('\n').filter(|l| !l.trim().is_empty()).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let action: Value = serde_json::from_str(lines[i]).map_err(|e| EsError::json_parse(&e))?;
        let o = action.as_object().ok_or_else(|| EsError::illegal_argument("Malformed action/metadata line [1], expected START_OBJECT"))?;
        let (op, meta) = o.iter().next().ok_or_else(|| EsError::illegal_argument("Malformed action/metadata line [1], expected FIELD_NAME but found [END_OBJECT]"))?;
        if op != "index" && op != "create" {
            return Err(EsError::illegal_argument("monitoring bulk requests should only contain index requests"));
        }
        let meta = meta.as_object().cloned().unwrap_or_default();
        let source: Value = match lines.get(i + 1) {
            Some(l) => serde_json::from_str(l).map_err(|e| EsError::json_parse(&e))?,
            None => return Err(EsError::illegal_argument("The bulk request must be terminated by a newline [\\n]")),
        };
        i += 2;
        let skip = match meta.get("_index").and_then(Value::as_str) {
            None => false,
            Some("_data") => true,
            Some(other) => return Err(EsError::illegal_argument(format!("unrecognized index name [{other}]"))),
        };
        let Some(ty) = meta.get("_type").and_then(Value::as_str) else { return Ok(Err(npe())) };
        if skip {
            continue;
        }
        out.push(Doc { ty: ty.to_string(), id: meta.get("_id").and_then(Value::as_str).map(str::to_string), source });
    }
    Ok(Ok(out))
}

fn bulk(node: &Node, req: &Req) -> R {
    let start = Instant::now();
    let system_id = required(req, "system_id")?;
    let version = required(req, "system_api_version")?;
    let interval = required(req, "interval")?;
    let text = req.body_str()?;
    if text.trim().is_empty() {
        return Err(EsError::new(400, "parse_exception", "no body content for monitoring bulk request"));
    }
    let system = system_id.to_ascii_lowercase();
    if !SYSTEMS.contains(&system.as_str()) || !API_VERSIONS.contains(&version.as_str()) {
        return Err(EsError::illegal_argument(format!("system_api_version [{version}] is not supported by system_id [{system_id}]")));
    }
    let interval_ms = elasticsearch_engine::date::parse_time_value(&interval)
        .filter(|_| interval.chars().last().is_some_and(|c| c.is_ascii_alphabetic()) || interval == "-1" || interval == "0")
        .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [interval] with value [{interval}] as a time value: unit is missing or unrecognized")))?;
    let docs = match parse(text)? {
        Ok(d) => d,
        Err(resp) => return Ok(resp),
    };
    let enabled = elasticsearch_engine::ilm::cluster_setting(node, "xpack.monitoring.collection.enabled").as_deref() == Some("true");
    if !enabled {
        return Ok(Resp::ok(json!({"took": start.elapsed().as_millis() as u64, "ignored": true, "errors": false})));
    }
    let now = elasticsearch_engine::date::now_millis();
    let ts = elasticsearch_engine::watcher::iso(now);
    let index = format!(".monitoring-{system}-{TEMPLATE_VERSION}-{}", ts[..10].replace('-', "."));
    let source_node = json!({"uuid": NODE_ID, "host": "127.0.0.1", "transport_address": TRANSPORT_ADDRESS, "ip": "127.0.0.1", "name": node.name, "timestamp": ts});
    let envelope = |d: Doc| {
        let mut src = serde_json::Map::new();
        src.insert("cluster_uuid".into(), json!(node.cluster_uuid));
        src.insert("timestamp".into(), json!(ts));
        src.insert("interval_ms".into(), json!(interval_ms));
        src.insert("type".into(), json!(d.ty));
        src.insert("source_node".into(), source_node.clone());
        src.insert(d.ty.clone(), d.source);
        (d.id, Value::Object(src))
    };
    // On a cluster the exporter writes through the coordinator (as the
    // monitoring origin): the index and its documents reach every copy.
    if let Some(rest) = req.internal_cluster_rest() {
        if let Err(e) = (rest.0)("PUT", &format!("/{index}"), Some(monitoring_index_body())) {
            if e.kind != "resource_already_exists_exception" {
                return Ok(export_error(start, vec![export_exception(&index, &e.reason, e.error_object(false))]));
            }
        }
        let body = cluster_bulk_body(docs.into_iter().map(envelope));
        return Ok(match (rest.0)("POST", &format!("/{index}/_bulk"), Some(Value::String(body))) {
            Err(e) => export_error(start, vec![export_exception(&index, &e.reason, e.error_object(false))]),
            Ok(v) => {
                let failed: Vec<Value> = v["items"].as_array().into_iter().flatten().filter_map(|item| item.as_object()?.values().next()?.get("error").cloned())
                    .map(|error| export_exception(&index, &error["reason"].as_str().unwrap_or("").to_string(), error.clone())).collect();
                if failed.is_empty() {
                    Resp::ok(json!({"took": start.elapsed().as_millis() as u64, "ignored": false, "errors": false}))
                } else {
                    export_error(start, failed)
                }
            }
        });
    }
    let svc = monitoring_index(node, &index)?;
    if !svc.is_open() {
        return Ok(export_failure(start, &svc));
    }
    let reqs: Vec<WriteReq> = docs
        .into_iter()
        .map(|d| {
            let mut src = serde_json::Map::new();
            src.insert("cluster_uuid".into(), json!(node.cluster_uuid));
            src.insert("timestamp".into(), json!(ts));
            src.insert("interval_ms".into(), json!(interval_ms));
            src.insert("type".into(), json!(d.ty));
            src.insert("source_node".into(), source_node.clone());
            src.insert(d.ty.clone(), d.source);
            WriteReq::Index { id: d.id, source: Value::Object(src).to_string(), routing: None, cond: WriteCond::default() }
        })
        .collect();
    let failed: Vec<EsError> = svc.write(reqs, false).into_iter().filter_map(Result::err).collect();
    if !failed.is_empty() {
        let exceptions: Vec<Value> = failed.iter().map(|e| json!({"type": "export_exception", "reason": format!("[{}] {}", svc.name, e.reason), "caused_by": e.error_object(false)})).collect();
        return Ok(export_error(start, exceptions));
    }
    Ok(Resp::ok(json!({"took": start.elapsed().as_millis() as u64, "ignored": false, "errors": false})))
}

/// The monitoring index of today, created with the envelope mappings of
/// ES's `.monitoring-*` templates on first use.
fn monitoring_index(node: &Node, name: &str) -> Result<Arc<IndexService>, EsError> {
    if let Some(s) = node.get_index(name) {
        return Ok(s);
    }
    match node.create_index(name, &monitoring_index_body()) {
        Ok(s) => Ok(s),
        Err(e) if e.kind == "resource_already_exists_exception" => node.get_index(name).ok_or(e),
        Err(e) => Err(e),
    }
}

/// Settings and envelope mappings of a `.monitoring-*` index.
fn monitoring_index_body() -> Value {
    let kw = json!({"type": "keyword"});
    let date = json!({"type": "date", "format": "date_time"});
    json!({
        "settings": {"index": {"number_of_shards": 1, "auto_expand_replicas": "0-1", "format": TEMPLATE_VERSION}},
        "mappings": {"dynamic": false, "properties": {
            "cluster_uuid": kw, "timestamp": date, "interval_ms": {"type": "long"}, "type": kw,
            "source_node": {"properties": {"uuid": kw, "host": kw, "transport_address": kw, "ip": kw, "name": kw, "timestamp": date}},
        }},
    })
}

/// The NDJSON `_bulk` body of exported documents (the index is in the path).
fn cluster_bulk_body(docs: impl Iterator<Item = (Option<String>, Value)>) -> String {
    let mut out = String::new();
    for (id, source) in docs {
        let action = match id {
            Some(id) => json!({"index": {"_id": id}}),
            None => json!({"index": {}}),
        };
        out.push_str(&action.to_string());
        out.push('\n');
        out.push_str(&source.to_string());
        out.push('\n');
    }
    out
}

/// One exporter failure (`ExportException`) for a document of `index`.
fn export_exception(index: &str, reason: &str, cause: Value) -> Value {
    json!({"type": "export_exception", "reason": format!("[{index}] {reason}"), "caused_by": cause})
}

fn export_error(start: Instant, exceptions: Vec<Value>) -> Resp {
    Resp::json(500, json!({
        "took": start.elapsed().as_millis() as u64, "ignored": false, "errors": true,
        "error": {"type": "export_exception", "reason": "failed to flush export bulks", "caused_by": {
            "type": "export_exception", "reason": "bulk [default_local] reports failures when exporting documents", "exceptions": exceptions}},
    }))
}

/// Writing to a closed monitoring index.
fn export_failure(start: Instant, svc: &IndexService) -> Resp {
    let closed = json!({"type": "index_closed_exception", "reason": "closed", "index_uuid": svc.uuid, "index": svc.name});
    export_error(start, vec![json!({"type": "export_exception", "reason": format!("[{}/{}] org.elasticsearch.indices.IndexClosedException: closed", svc.name, svc.uuid), "caused_by": closed})])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_documents_form_an_index_only_bulk_body() {
        let body = cluster_bulk_body(vec![(None, json!({"type": "a"})), (Some("7".to_string()), json!({"type": "b"}))].into_iter());
        assert_eq!(body, "{\"index\":{}}\n{\"type\":\"a\"}\n{\"index\":{\"_id\":\"7\"}}\n{\"type\":\"b\"}\n");
        let e = export_exception(".monitoring-beats-7-2026.10.14", "closed", json!({"type": "index_closed_exception"}));
        assert_eq!(e["type"], "export_exception");
        assert_eq!(e["reason"], "[.monitoring-beats-7-2026.10.14] closed");
    }
}
