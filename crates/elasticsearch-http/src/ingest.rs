//! Ingest APIs: pipeline CRUD, `_simulate` (pipeline and ingest variants),
//! grok patterns, geoip; plus [`prepare_ext`], which the document APIs
//! call to run pipelines before a write.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::ingest::{grok, IngestOutcome, IngestRequest};
use elasticsearch_engine::node::Node;
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let segs: Vec<String> = req.segs.clone();
    let s = |i: usize| segs.get(i).map(String::as_str).unwrap_or("");
    let m = req.method.clone();
    let n = segs.len();
    match (s(1), n) {
        ("pipeline", 2) => match m {
            Method::GET => blocking(move || get_pipelines(&node, &req, None)).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("pipeline", 3) if s(2) == "_simulate" => match m {
            Method::GET | Method::POST => blocking(move || simulate(&node, &req, None)).await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        ("pipeline", 3) => {
            let id = s(2).to_string();
            match m {
                Method::GET => blocking(move || get_pipelines(&node, &req, Some(&id))).await,
                Method::PUT => blocking(move || put_pipeline(&node, &req, &id)).await,
                Method::DELETE => blocking(move || {
                    node.delete_pipeline(&id)?;
                    // The template registries reinstall a deleted built-in pipeline.
                    elasticsearch_engine::registry::ensure(&node);
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("pipeline", 4) if s(3) == "_simulate" => {
            let id = s(2).to_string();
            match m {
                Method::GET | Method::POST => blocking(move || simulate(&node, &req, Some(&id))).await,
                _ => method_not_allowed(&req, &["GET", "POST"]),
            }
        }
        ("processor", 3) if s(2) == "grok" && m == Method::GET => {
            let ecs_v1 = match req.param("ecs_compatibility") {
                None | Some("disabled") => false,
                Some("v1") => true,
                Some(o) => return Err(EsError::illegal_argument(format!("unsupported ECS compatibility mode [{o}]"))),
            };
            req.flag("s")?;
            let mut pats = Map::new();
            for (k, v) in grok::bank_es(ecs_v1) {
                pats.insert(k.clone(), Value::String(v.clone()));
            }
            Ok(Resp::ok(json!({"patterns": pats})))
        }
        ("_simulate", 2) => match m {
            Method::GET | Method::POST => blocking(move || simulate_ingest(&node, &req, None)).await,
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        ("geoip", 3) if s(2) == "stats" => match m {
            Method::GET => Ok(Resp::ok(elasticsearch_engine::ingest::geoip::stats_json(Some(&node.data), elasticsearch_engine::node::NODE_ID))),
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("geoip", 3) | ("geoip", 4) | ("ip_location", 3) | ("ip_location", 4) if s(2) == "database" => {
            let ip_location = s(1) == "ip_location";
            let id = s(3).to_string();
            match (m, n) {
                (Method::GET, _) => blocking(move || {
                    let ids = if id.is_empty() { None } else { Some(id.as_str()) };
                    Ok(Resp::ok(elasticsearch_engine::ingest::geoip::get_configs(&node.data, ids)?))
                })
                .await,
                (Method::PUT, 4) => blocking(move || {
                    let body = req.json()?;
                    let raw = req.body_str()?.to_string();
                    elasticsearch_engine::ingest::geoip::put_config(&node.data, &id, &body, &raw, ip_location)?;
                    Ok(Resp::ack())
                })
                .await,
                (Method::DELETE, 4) => blocking(move || {
                    elasticsearch_engine::ingest::geoip::delete_config(&node.data, &id)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        (_, 3) if s(2) == "_simulate" => {
            let index = s(1).to_string();
            match m {
                Method::GET | Method::POST => blocking(move || simulate_ingest(&node, &req, Some(&index))).await,
                _ => method_not_allowed(&req, &["GET", "POST"]),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

fn get_pipelines(node: &Node, req: &Req, ids: Option<&str>) -> R {
    let store = node.ingest();
    let summary = req.flag("summary")?;
    let found = store.get(ids);
    let mut out = Map::new();
    let human = req.flag("human")?;
    for (id, cfg) in found {
        let cfg = match cfg {
            // `GetPipelineResponse`: `timestampFieldsFromUnixEpochMillis`.
            Value::Object(m) if human => {
                let mut o = Map::new();
                for (k, v) in m {
                    if let (Some(h), Some(ms)) = (k.strip_suffix("_millis").filter(|h| *h == "created_date" || *h == "modified_date"), v.as_i64()) {
                        o.insert(h.to_string(), json!(elasticsearch_engine::date::human_date(ms)));
                    }
                    o.insert(k, v);
                }
                Value::Object(o)
            }
            other => other,
        };
        out.insert(id, if summary { json!({}) } else { cfg });
    }
    if out.is_empty() {
        // ES: `found == !pipelines.isEmpty()`, 404 otherwise (also for get-all).
        return Ok(Resp::json(404, json!({})));
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn put_pipeline(node: &Node, req: &Req, id: &str) -> R {
    let has_body = !req.body.is_empty() && !req.body_str()?.trim().is_empty();
    let source = req.param("source");
    let sct = req.param("source_content_type");
    let body: Value = if has_body {
        let mut extra: Vec<&str> = Vec::new();
        if source.is_some() {
            extra.push("source");
        }
        if sct.is_some() {
            extra.push("source_content_type");
        }
        if !extra.is_empty() {
            let list = extra.iter().map(|p| format!("[{p}]")).collect::<Vec<_>>().join(", ");
            let noun = if extra.len() == 1 { "parameter" } else { "parameters" };
            return Err(EsError::illegal_argument(format!("request [{}] contains unrecognized {noun}: {list}", req.path)));
        }
        req.json()?
    } else if let Some(src) = source {
        match sct {
            None => return Err(EsError::new(400, "validation_exception", "Validation Failed: 1: source and source_content_type parameters are required;")),
            Some("application/json") => serde_json::from_str(src).map_err(|e| EsError::json_parse(&e))?,
            Some(other) => return Err(EsError::new(400, "validation_exception", format!("Validation Failed: 1: Unknown value for source_content_type [{other}];"))),
        }
    } else {
        return Err(EsError::new(400, "parse_exception", "request body or source parameter is required"));
    };
    let if_version = match req.param("if_version") {
        Some(v) => Some(v.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("invalid value [{v}] specified for [if_version]. must be an integer value")))?),
        None => None,
    };
    let _enrich = elasticsearch_engine::enrich::NodeScope::enter(node);
    node.ingest().put(id, body, if_version)?;
    Ok(Resp::ack())
}

fn simulate(node: &Node, req: &Req, id: Option<&str>) -> R {
    let _ingest_user = req.access.as_deref().map(|a| elasticsearch_engine::ingest::IngestContext::enter(elasticsearch_engine::ingest::IngestContext { user: Some(a.user().clone()) }));
    let raw=req.body_str()?;
    let body=elasticsearch_engine::source_json::Source::parse(raw)?;
    let verbose=req.flag("verbose")?;
    Ok(Resp::raw(200,node.simulate_pipeline_source(id,&body,verbose)?.text()))
}

fn simulate_ingest(node: &Node, req: &Req, index: Option<&str>) -> R {
    let _ingest_user = req.access.as_deref().map(|a| elasticsearch_engine::ingest::IngestContext::enter(elasticsearch_engine::ingest::IngestContext { user: Some(a.user().clone()) }));
    let body=elasticsearch_engine::source_json::Source::parse(req.body_str()?)?;
    Ok(Resp::raw(200,node.simulate_ingest_source(index,req.param("pipeline"),&body)?.text()))
}

/// What to write after ingest.
pub(crate) enum Prepared {
    Write { index: String, id: Option<String>, routing: Option<String>, source: String, dynamic_templates: Option<std::collections::HashMap<String, String>> },
    Dropped { index: String, id: Option<String> },
}

/// Run the pipelines that apply to an index/create request.
/// On failure, keeps the failure store details of the ingest failure (and
/// hands the original source back for the failure document).
pub(crate) fn prepare_ext(
    node: &Node,
    index: &str,
    id: Option<String>,
    routing: Option<String>,
    source: String,
    pipeline: Option<&str>,
) -> Result<Prepared, Box<(elasticsearch_engine::ingest::IngestFailure, String)>> {
    let out = match node.run_ingest_ext(IngestRequest { index, id: id.as_deref(), routing: routing.as_deref(), source: &source, pipeline }) {
        Ok(o) => o,
        Err(f) => return Err(Box::new((f, source))),
    };
    Ok(match out {
        IngestOutcome::Unchanged => Prepared::Write { index: index.to_string(), id, routing, source, dynamic_templates: None },
        IngestOutcome::Dropped { index } => Prepared::Dropped { index, id },
        IngestOutcome::Changed { index, id, routing, source, dynamic_templates } => Prepared::Write { index, id, routing, source, dynamic_templates },
    })
}

/// Response body of a write dropped by a pipeline.
pub(crate) fn dropped_json(index: &str, id: Option<&str>) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("_index".into(), json!(index));
    m.insert("_id".into(), json!(id.unwrap_or("auto-generated")));
    m.insert("_version".into(), json!(-3));
    m.insert("result".into(), json!("noop"));
    m.insert("_shards".into(), json!({"total": 0, "successful": 0, "failed": 0}));
    m
}
