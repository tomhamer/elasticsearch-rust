//! `/_ml/...` REST handlers: anomaly detection jobs, datafeeds, calendars,
//! filters (`elasticsearch_engine::ml`), data frame analytics ([`dfa`]),
//! `_evaluate` ([`evaluate`]) and trained models
//! (`elasticsearch_engine::inference::models`).

pub(crate) mod dfa;
mod evaluate;

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::inference::models;
use elasticsearch_engine::ml::{anomaly, calendar, datafeed, memory, results, validate};
use elasticsearch_engine::node::{Node, NODE_ID};
use serde_json::{json, Map, Value};
use std::sync::Arc;

fn body_obj(req: &Req) -> Result<Value, EsError> {
    let b = req.json()?;
    Ok(if b.is_null() { json!({}) } else { b })
}

fn raw(req: &Req) -> String {
    String::from_utf8_lossy(&req.body).into_owned()
}

/// A request option given as a URL param or in the body (ES lets ML
/// start/stop/close/open APIs take them either way).
fn opt(req: &Req, body: &Value, k: &str) -> Option<String> {
    req.param(k).map(str::to_string).or_else(|| {
        body.get(k).map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    })
}

fn opt_bool(req: &Req, body: &Value, k: &str, default: bool) -> Result<bool, EsError> {
    match opt(req, body, k).as_deref() {
        None => Ok(default),
        Some("true") | Some("") => Ok(true),
        Some("false") => Ok(false),
        Some(v) => Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
    }
}

fn int_param(req: &Req, k: &str) -> Result<Option<i64>, EsError> {
    match req.param(k) {
        None | Some("") => Ok(None),
        Some(v) => v.parse().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{v}]"))),
    }
}

fn warn(resp: &mut Resp, msg: &str) {
    resp.headers.push((
        "Warning",
        format!("299 Elasticsearch-{}-{} \"{}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH, msg.replace('\\', "\\\\").replace('"', "\\\"")),
    ));
}

const POST_DATA_WARNING: &str = "Posting data directly to anomaly detection jobs is deprecated, in a future major version it will be compulsory to use a datafeed";

/// `ForecastJobAction.Request` validation (before the job is looked at).
fn check_forecast_params(req: &Req, body: &Value) -> Result<(), EsError> {
    if let Some(d) = opt(req, body, "duration") {
        let t = elasticsearch_engine::ml::job::parse_time(&d, "duration")?;
        if t.nanos <= 0 {
            return Err(EsError::illegal_argument(format!("[duration] must be positive: [{}]", t.rep)));
        }
        if t.nanos > 3650i128 * 86_400_000_000_000 {
            return Err(EsError::illegal_argument(format!("[duration] must be 3650d or less: [{}]", t.rep)));
        }
    }
    if let Some(e) = opt(req, body, "expires_in") {
        let t = elasticsearch_engine::ml::job::parse_time(&e, "expires_in")?;
        if t.nanos < 0 {
            return Err(EsError::illegal_argument(format!("[expires_in] must be non-negative: [{}]", t.rep)));
        }
    }
    if let Some(m) = opt(req, body, "max_model_memory") {
        let b = elasticsearch_engine::ml::job::parse_bytes(&m, "max_model_memory")?;
        if b >= 500 << 20 {
            return Err(EsError::illegal_argument(format!("[max_model_memory] must be less than 500mb. Requested [{}]", m)));
        }
    }
    Ok(())
}

/// `from` / `size` of a results API: URL params or `body.page`.
fn page_of(req: &Req, body: &Value) -> Result<Option<results::Page>, EsError> {
    let body_int = |k: &str| -> Result<Option<i64>, EsError> {
        let v = body["page"].get(k).or_else(|| body.get(k)).filter(|v| !v.is_null());
        match v {
            None => Ok(None),
            Some(Value::Number(n)) => n.as_i64().map(Some).ok_or_else(|| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{n}]"))),
            Some(Value::String(s)) => s.parse::<i64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{s}]"))),
            Some(other) => Err(EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{other}]"))),
        }
    };
    let from = match int_param(req, "from")? {
        Some(f) => Some(f),
        None => body_int("from")?,
    };
    let size = match int_param(req, "size")? {
        Some(s) => Some(s),
        None => body_int("size")?,
    };
    if from.is_none() && size.is_none() {
        return Ok(None);
    }
    Ok(Some(results::Page { from: from.unwrap_or(0), size: size.unwrap_or(100) }))
}

/// `_xpack/usage` `ml` block: job / datafeed counts by state (the stats
/// come from the job stats API; when that fails, e.g. a closed results
/// index, only the totals are reported, as ES does).
pub fn usage(node: &Node, template: &Value) -> Value {
    let mut ml = template.clone();
    ml["available"] = json!(true);
    ml["enabled"] = json!(true);
    let st = elasticsearch_engine::ml::store::service(node).read();
    let stats = anomaly::job_stats(node, None, true).ok();
    let agg = |vals: &[f64]| -> Value {
        if vals.is_empty() {
            return json!({"total": 0.0, "min": 0.0, "avg": 0.0, "max": 0.0});
        }
        let total: f64 = vals.iter().sum();
        json!({"total": total, "min": vals.iter().cloned().fold(f64::INFINITY, f64::min), "avg": total / vals.len() as f64, "max": vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)})
    };
    let job_block = |ids: &[&String], stats: Option<&Value>| -> Value {
        let detectors: Vec<f64> = ids.iter().map(|id| st.jobs[*id]["analysis_config"]["detectors"].as_array().map(|a| a.len() as f64).unwrap_or(0.0)).collect();
        let mut created_by: Map<String, Value> = Map::new();
        for id in ids {
            let c = st.jobs[*id]["custom_settings"]["created_by"].as_str().unwrap_or("unknown").replace('.', "_");
            let e = created_by.entry(c).or_insert(json!(0));
            *e = json!(e.as_i64().unwrap_or(0) + 1);
        }
        let mut model_size: Vec<f64> = Vec::new();
        let mut forecasts_total = 0i64;
        let mut forecasted = 0i64;
        if let Some(sv) = stats {
            for id in ids {
                if let Some(j) = sv["jobs"].as_array().into_iter().flatten().find(|j| j["job_id"].as_str() == Some(id.as_str())) {
                    model_size.push(j["model_size_stats"]["model_bytes"].as_f64().unwrap_or(0.0));
                    let t = j["forecasts_stats"]["total"].as_i64().unwrap_or(0);
                    forecasts_total += t;
                    if t > 0 {
                        forecasted += 1;
                    }
                }
            }
        }
        json!({"count": ids.len(), "detectors": agg(&detectors), "created_by": created_by, "model_size": agg(&model_size), "forecasts": {"total": forecasts_total, "forecasted_jobs": forecasted}})
    };
    let all: Vec<&String> = st.jobs.keys().collect();
    let mut jobs = Map::new();
    jobs.insert("_all".into(), job_block(&all, stats.as_ref()));
    if stats.is_some() {
        let mut by_state: std::collections::BTreeMap<String, Vec<&String>> = Default::default();
        for id in &all {
            by_state.entry(st.job_state(id).to_string()).or_default().push(id);
        }
        for (state, ids) in by_state {
            jobs.insert(state, job_block(&ids, stats.as_ref()));
        }
    }
    ml["jobs"] = Value::Object(jobs);
    let mut dfs = Map::new();
    dfs.insert("_all".into(), json!({"count": st.datafeeds.len()}));
    let mut df_state: std::collections::BTreeMap<String, usize> = Default::default();
    for id in st.datafeeds.keys() {
        *df_state.entry(st.datafeed_state(id).to_string()).or_default() += 1;
    }
    for (s, n) in df_state {
        dfs.insert(s, json!({"count": n}));
    }
    ml["datafeeds"] = Value::Object(dfs);
    ml["memory"]["anomaly_detectors_memory_bytes"] = json!(anomaly::anomaly_detector_memory(node));
    ml["memory"]["total_used_memory_bytes"] = json!(anomaly::anomaly_detector_memory(node));
    ml
}

fn opt_value(req: &Req, body: &Value, k: &str) -> Option<Value> {
    req.param(k).map(|v| json!(v)).or_else(|| body.get(k).filter(|v| !v.is_null()).cloned())
}

fn opt_f64(req: &Req, body: &Value, k: &str) -> Result<Option<f64>, EsError> {
    match opt_value(req, body, k) {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) => s.parse::<f64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse double parameter [{k}] with value [{s}]"))),
        Some(other) => Err(EsError::illegal_argument(format!("Failed to parse double parameter [{k}] with value [{other}]"))),
    }
}

fn opt_i64(req: &Req, body: &Value, k: &str) -> Result<Option<i64>, EsError> {
    match opt_value(req, body, k) {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_i64()),
        Some(Value::String(s)) => s.parse::<i64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{s}]"))),
        Some(other) => Err(EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{other}]"))),
    }
}

/// An error response that still carries the `_data` deprecation warning.
fn error_with_warning(e: EsError, msg: &str) -> Resp {
    let mut r = Resp::json(e.status, e.to_json_with_trace(false));
    warn(&mut r, msg);
    r
}

/// The scored-results request (records / influencers) from params + body.
fn scored_req(req: &Req, body: &Value, score_key: &str) -> Result<results::ScoredReq, EsError> {
    Ok(results::ScoredReq {
        exclude_interim: opt_bool(req, body, "exclude_interim", false)?,
        start: opt_value(req, body, "start"),
        end: opt_value(req, body, "end"),
        score: opt_f64(req, body, score_key)?,
        sort: opt(req, body, "sort"),
        desc: match opt(req, body, "desc").as_deref() {
            None => None,
            Some("true") | Some("") => Some(true),
            Some("false") => Some(false),
            Some(v) => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
        },
        page: page_of(req, body)?.unwrap_or_default(),
    })
}

const IGNORE_THROTTLED_WARNING: &str = "[ignore_throttled] parameter is deprecated because frozen indices have been deprecated. Consider cold or frozen tiers in place of frozen indices.";

/// Register the node descriptor ML stats report (once).
fn init_node(node: &Node) {
    let s = crate::sysinfo::sys();
    anomaly::init_node_info(
        json!({"ephemeral_id": crate::observability::EPHEMERAL_ID, "transport_address": crate::observability::TRANSPORT_ADDRESS, "attributes": crate::observability::node_attributes()}),
        s.total_mem,
    );
    let _ = node;
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let m = req.method.clone();
    init_node(&node);
    // Creating and starting ML resources needs a platinum (or trial)
    // license; reads, deletes and upgrade mode do not.
    if m == Method::PUT || (m == Method::POST && matches!(s(n - 1).as_str(), "_open" | "_start")) {
        elasticsearch_engine::license::check(&node, "ml", elasticsearch_engine::license::Mode::Platinum)?;
    }
    match s(1).as_str() {
        "data_frame" => return if s(2) == "_evaluate" { evaluate::handle(node, req).await } else { dfa::handle(node, req).await },
        "anomaly_detectors" => return anomaly_detectors(node, req).await,
        "datafeeds" => return datafeeds(node, req).await,
        "calendars" => return calendars(node, req).await,
        "filters" => return filters(node, req).await,
        "trained_models" => return trained_models(node, req).await,
        _ => {}
    }
    match (s(1).as_str(), n) {
        ("info", 2) => match m {
            Method::GET => {
                let mut info = models::ml_info();
                info["upgrade_mode"] = json!(elasticsearch_engine::ml::upgrade_mode(&node));
                if let Some(max_mb) = anomaly::max_model_memory_mb(&node) {
                    let as_str = elasticsearch_engine::ml::job::bytes_to_string(max_mb << 20);
                    if max_mb < elasticsearch_engine::ml::job::DEFAULT_MML_MB {
                        info["defaults"]["anomaly_detectors"]["model_memory_limit"] = json!(as_str);
                    }
                    info["limits"]["max_model_memory_limit"] = json!(as_str);
                    let eff = info["limits"]["effective_max_model_memory_limit"].as_str().and_then(|e| elasticsearch_engine::ml::job::parse_bytes(e, "x").ok()).unwrap_or(u64::MAX);
                    if (max_mb << 20) < eff {
                        info["limits"]["effective_max_model_memory_limit"] = json!(as_str);
                    }
                }
                Ok(Resp::ok(info))
            }
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("memory", 3) if s(2) == "_stats" => memory_stats(&node, &req, None),
        ("memory", 4) if s(3) == "_stats" => memory_stats(&node, &req, Some(s(2))),
        ("set_upgrade_mode", 2) => match m {
            Method::POST => {
                let enabled = req.bool_param("enabled")?.unwrap_or(false);
                blocking(move || Ok(Resp::ok(anomaly::set_upgrade_mode(&node, enabled)?))).await
            }
            _ => method_not_allowed(&req, &["POST"]),
        },
        ("_delete_expired_data", 2 | 3) => match m {
            Method::DELETE => {
                let job = segs.get(2).cloned();
                let body = req.json()?;
                if let Some(rps) = opt(&req, &body, "requests_per_second") {
                    let v: f64 = rps.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse float parameter [requests_per_second] with value [{rps}]")))?;
                    if v <= 0.0 && v != -1.0 {
                        return Err(EsError::action_validation(&["[requests_per_second] must be positive or -1"]));
                    }
                }
                blocking(move || Ok(Resp::ok(anomaly::delete_expired_data(&node, job.as_deref())?))).await
            }
            _ => method_not_allowed(&req, &["DELETE"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

/// `GET _ml/memory[/{node_id}]/_stats`.
fn memory_stats(node: &Node, req: &Req, filter: Option<String>) -> R {
    if req.method != Method::GET {
        return method_not_allowed(req, &["GET"]);
    }
    let s = crate::sysinfo::sys();
    let (_, heap_max) = crate::sysinfo::heap(&s);
    let attrs = crate::observability::node_attributes();
    let mut a = attrs.as_object().cloned().unwrap_or_default();
    a.remove("testattr");
    a.remove("transform.config_version");
    a.remove("xpack.installed");
    let matches = match filter.as_deref() {
        None | Some("_all") | Some("*") | Some("_local") | Some("_master") => true,
        Some(f) => f.split(',').any(|p| {
            p == NODE_ID || p == node.name || p == "ml:true" || p == "master:true" || p == "data:true" || p == "ingest:true" || elasticsearch_engine::mapping::wildcard_match(p, &node.name) || p.starts_with("_")
        }),
    };
    let max_ml = anomaly::ml_max_memory(node);
    let ad = anomaly::anomaly_detector_memory(node);
    let dfa_mem = elasticsearch_engine::ml::dfa::memory_in_bytes(node);
    let mut nodes = Map::new();
    if matches {
        nodes.insert(
            NODE_ID.into(),
            json!({"name": node.name, "ephemeral_id": crate::observability::EPHEMERAL_ID, "transport_address": crate::observability::TRANSPORT_ADDRESS, "attributes": a,
                "roles": crate::observability::ROLES,
                "mem": {"total_in_bytes": s.total_mem, "adjusted_total_in_bytes": s.total_mem,
                    "ml": {"max_in_bytes": max_ml, "native_code_overhead_in_bytes": if ad + dfa_mem > 0 { 30u64 << 20 } else { 0 }, "anomaly_detectors_in_bytes": ad, "data_frame_analytics_in_bytes": dfa_mem, "native_inference_in_bytes": 0}},
                "jvm": {"heap_max_in_bytes": heap_max, "java_inference_max_in_bytes": heap_max * 2 / 5, "java_inference_in_bytes": 0}}),
        );
    }
    let total = if matches { 1 } else { 0 };
    Ok(Resp::ok(json!({"_nodes": {"total": total, "successful": total, "failed": 0}, "cluster_name": node.cluster_name, "nodes": nodes})))
}

// ---------------------------------------------------------------------------
// Anomaly detectors
// ---------------------------------------------------------------------------

async fn anomaly_detectors(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let m = req.method.clone();
    let allow = || -> Result<bool, EsError> { Ok(req.bool_param("allow_no_match")?.or(req.bool_param("allow_no_jobs")?).unwrap_or(true)) };
    match n {
        2 => match m {
            Method::GET => {
                let (a, ex) = (allow()?, req.flag("exclude_generated")?);
                blocking(move || Ok(Resp::ok(anomaly::get_jobs(&node, None, a, ex)?))).await
            }
            _ => method_not_allowed(&req, &["GET"]),
        },
        3 => match s(2).as_str() {
            "_stats" => match m {
                Method::GET => {
                    let a = allow()?;
                    blocking(move || Ok(Resp::ok(anomaly::job_stats(&node, None, a)?))).await
                }
                _ => method_not_allowed(&req, &["GET"]),
            },
            "_validate" => match m {
                Method::POST => {
                    let (body, raw) = (req.json()?, raw(&req));
                    validate::job(&body, &raw)?;
                    Ok(Resp::ack())
                }
                _ => method_not_allowed(&req, &["POST"]),
            },
            "_estimate_model_memory" => match m {
                Method::POST => {
                    let (body, raw) = (req.json()?, raw(&req));
                    Ok(Resp::ok(memory::estimate_model_memory(&body, &raw)?))
                }
                _ => method_not_allowed(&req, &["POST"]),
            },
            _ => {
                let id = s(2);
                match m {
                    Method::GET => {
                        let (a, ex) = (allow()?, req.flag("exclude_generated")?);
                        blocking(move || Ok(Resp::ok(anomaly::get_jobs(&node, Some(&id), a, ex)?))).await
                    }
                    Method::PUT => {
                        let (body, raw) = (req.json()?, raw(&req));
                        let mut opts = datafeed::indices_options_from_params(&req.params);
                        let throttled = req.param("ignore_throttled").is_some();
                        if body.get("datafeed_config").is_none() {
                            opts.clear();
                        }
                        blocking(move || {
                            let (job, warnings) = anomaly::put_job(&node, &id, &body, &raw, &opts)?;
                            let mut r = Resp::ok(job);
                            for w in warnings {
                                warn(&mut r, &w);
                            }
                            if throttled {
                                warn(&mut r, IGNORE_THROTTLED_WARNING);
                            }
                            Ok(r)
                        })
                        .await
                    }
                    Method::DELETE => {
                        let force = req.flag("force")?;
                        let wait = req.bool_param("wait_for_completion")?.unwrap_or(true);
                        blocking(move || {
                            let r = anomaly::delete_job(&node, &id, force)?;
                            if !wait {
                                let t = node.tasks.register("cluster:admin/xpack/ml/job/delete", Some(format!("delete-job-{id}")), false, false, Map::new());
                                let task_id = t.task_id();
                                node.tasks.finish(&t, Ok(r));
                                return Ok(Resp::ok(json!({"task": task_id})));
                            }
                            Ok(Resp::ok(r))
                        })
                        .await
                    }
                    _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
                }
            }
        },
        4 if s(2) == "_validate" && s(3) == "detector" => match m {
            Method::POST => {
                let (body, raw) = (req.json()?, raw(&req));
                validate::detector(&body, &raw)?;
                Ok(Resp::ack())
            }
            _ => method_not_allowed(&req, &["POST"]),
        },
        4 => {
            let id = s(2);
            // `_data` takes ndjson / delimited data, not a JSON body.
            let body = if s(3) == "_data" { Value::Null } else { req.json()? };
            match (s(3).as_str(), &m) {
                ("_stats", &Method::GET) => {
                    let a = allow()?;
                    blocking(move || Ok(Resp::ok(anomaly::job_stats(&node, Some(&id), a)?))).await
                }
                ("_open", &Method::POST) => blocking(move || Ok(Resp::ok(anomaly::open_job(&node, &id)?))).await,
                ("_close", &Method::POST) => {
                    let a = match opt(&req, &body, "allow_no_match").or_else(|| opt(&req, &body, "allow_no_jobs")) {
                        Some(v) => v == "true" || v.is_empty(),
                        None => true,
                    };
                    let force = opt_bool(&req, &body, "force", false)?;
                    blocking(move || Ok(Resp::ok(anomaly::close_job(&node, &id, a, force)?))).await
                }
                ("_flush", &Method::POST) => {
                    let p = |k: &str| opt(&req, &body, k);
                    let t = |k: &str| -> Result<Option<i64>, EsError> { p(k).map(|v| anomaly::parse_flush_time(&v, k)).transpose() };
                    let fp = anomaly::FlushParams { calc_interim: opt_bool(&req, &body, "calc_interim", false)?, start: t("start")?, end: t("end")?, advance_time: t("advance_time")?, skip_time: t("skip_time")? };
                    blocking(move || Ok(Resp::ok(anomaly::flush_job(&node, &id, &fp)?))).await
                }
                ("_reset", &Method::POST) => {
                    let wait = req.bool_param("wait_for_completion")?.unwrap_or(true);
                    blocking(move || {
                        let r = anomaly::reset_job(&node, &id)?;
                        if !wait {
                            let t = node.tasks.register("cluster:admin/xpack/ml/job/reset", Some(format!("reset-job-{id}")), false, false, Map::new());
                            let task_id = t.task_id();
                            node.tasks.finish(&t, Ok(r));
                            return Ok(Resp::ok(json!({"task": task_id})));
                        }
                        Ok(Resp::ok(r))
                    })
                    .await
                }
                ("_update", &Method::POST) => {
                    let raw = raw(&req);
                    blocking(move || Ok(Resp::ok(anomaly::update_job(&node, &id, &body, &raw)?))).await
                }
                ("model_snapshots", &Method::GET) | ("model_snapshots", &Method::POST) => {
                    let sr = results::SnapshotsReq {
                        snapshot_id: opt(&req, &body, "snapshot_id"),
                        start: opt_value(&req, &body, "start"),
                        end: opt_value(&req, &body, "end"),
                        sort: opt(&req, &body, "sort"),
                        desc: match opt(&req, &body, "desc").as_deref() {
                            None => None,
                            Some("false") => Some(false),
                            Some(_) => Some(true),
                        },
                        page: page_of(&req, &body)?.unwrap_or_default(),
                    };
                    blocking(move || Ok(Resp::ok(results::get_model_snapshots(&node, &id, &sr)?))).await
                }
                ("_forecast", &Method::DELETE) => {
                    let allow = opt_bool(&req, &body, "allow_no_forecasts", true)?;
                    let fids = opt(&req, &body, "forecast_id").unwrap_or_default();
                    blocking(move || Ok(Resp::ok(results::delete_forecasts(&node, &id, &fids, allow)?))).await
                }
                ("_forecast", &Method::POST) => {
                    check_forecast_params(&req, &body)?;
                    let duration = opt(&req, &body, "duration").map(|d| elasticsearch_engine::ml::job::parse_time(&d, "duration")).transpose()?.map(|t| t.millis()).unwrap_or(86_400_000);
                    let expires = opt(&req, &body, "expires_in").map(|d| elasticsearch_engine::ml::job::parse_time(&d, "expires_in")).transpose()?.map(|t| t.millis()).unwrap_or(14 * 86_400_000);
                    blocking(move || Ok(Resp::ok(anomaly::forecast(&node, &id, duration, expires)?))).await
                }
                ("_data", &Method::POST) => {
                    let data = req.body.clone();
                    let r = blocking(move || Ok(Resp::ok(anomaly::post_data(&node, &id, &data)?))).await;
                    Ok(match r {
                        Ok(mut resp) => {
                            warn(&mut resp, POST_DATA_WARNING);
                            resp
                        }
                        Err(e) => error_with_warning(e, POST_DATA_WARNING),
                    })
                }
                ("_stats" | "model_snapshots", _) => method_not_allowed(&req, &["GET"]),
                ("_open" | "_close" | "_flush" | "_reset" | "_update" | "_data", _) => method_not_allowed(&req, &["POST"]),
                _ => crate::not_found_handler(&req),
            }
        }
        5 | 6 if s(3) == "results" => {
            let id = s(2);
            if !matches!(m, Method::GET | Method::POST) {
                return method_not_allowed(&req, &["GET", "POST"]);
            }
            let body = body_obj(&req)?;
            let extra = segs.get(5).cloned();
            match s(4).as_str() {
                "buckets" => {
                    let br = results::BucketsReq {
                        timestamp: extra.map(|t| json!(t)).or_else(|| opt_value(&req, &body, "timestamp")),
                        expand: opt_bool(&req, &body, "expand", false)?,
                        exclude_interim: opt_bool(&req, &body, "exclude_interim", false)?,
                        start: opt_value(&req, &body, "start"),
                        end: opt_value(&req, &body, "end"),
                        anomaly_score: opt_f64(&req, &body, "anomaly_score")?,
                        sort: opt(&req, &body, "sort"),
                        desc: opt_bool(&req, &body, "desc", false)?,
                        page: page_of(&req, &body)?,
                    };
                    blocking(move || Ok(Resp::ok(results::get_buckets(&node, &id, &br)?))).await
                }
                "records" if extra.is_none() => {
                    let sr = scored_req(&req, &body, "record_score")?;
                    blocking(move || Ok(Resp::ok(results::get_records(&node, &id, &sr)?))).await
                }
                "influencers" if extra.is_none() => {
                    let sr = scored_req(&req, &body, "influencer_score")?;
                    blocking(move || Ok(Resp::ok(results::get_influencers(&node, &id, &sr)?))).await
                }
                "categories" => {
                    let cid = match extra {
                        Some(c) => Some(c.parse::<i64>().map_err(|_| EsError::new(400, "number_format_exception", format!("For input string: \"{c}\"")))?),
                        None => opt_i64(&req, &body, "category_id")?,
                    };
                    let pfv = opt(&req, &body, "partition_field_value");
                    let page = page_of(&req, &body)?;
                    blocking(move || Ok(Resp::ok(results::get_categories(&node, &id, cid, pfv.as_deref(), page)?))).await
                }
                "overall_buckets" if extra.is_none() => {
                    let or = results::OverallReq {
                        top_n: opt_i64(&req, &body, "top_n")?,
                        bucket_span: opt(&req, &body, "bucket_span"),
                        overall_score: opt_f64(&req, &body, "overall_score")?,
                        exclude_interim: opt_bool(&req, &body, "exclude_interim", false)?,
                        start: opt_value(&req, &body, "start"),
                        end: opt_value(&req, &body, "end"),
                        allow_no_match: opt_bool(&req, &body, "allow_no_match", opt_bool(&req, &body, "allow_no_jobs", true)?)?,
                    };
                    blocking(move || Ok(Resp::ok(results::get_overall_buckets(&node, &id, &or)?))).await
                }
                _ => crate::not_found_handler(&req),
            }
        }
        5 if s(3) == "model_snapshots" => {
            let (id, snap) = (s(2), s(4));
            match m {
                Method::GET | Method::POST => {
                    let body = body_obj(&req)?;
                    let sr = results::SnapshotsReq {
                        snapshot_id: Some(snap),
                        start: opt_value(&req, &body, "start"),
                        end: opt_value(&req, &body, "end"),
                        sort: opt(&req, &body, "sort"),
                        desc: match opt(&req, &body, "desc").as_deref() {
                            None => None,
                            Some("false") => Some(false),
                            Some(_) => Some(true),
                        },
                        page: page_of(&req, &body)?.unwrap_or_default(),
                    };
                    blocking(move || Ok(Resp::ok(results::get_model_snapshots(&node, &id, &sr)?))).await
                }
                Method::DELETE => blocking(move || Ok(Resp::ok(results::delete_model_snapshot(&node, &id, &snap)?))).await,
                _ => method_not_allowed(&req, &["GET", "POST", "DELETE"]),
            }
        }
        6 if s(3) == "model_snapshots" => {
            let (id, snap) = (s(2), s(4));
            let body = body_obj(&req)?;
            match (s(5).as_str(), &m) {
                ("_update", &Method::POST) => blocking(move || Ok(Resp::ok(results::update_model_snapshot(&node, &id, &snap, &body)?))).await,
                ("_revert", &Method::POST) => {
                    let del = opt_bool(&req, &body, "delete_intervening_results", false)?;
                    blocking(move || Ok(Resp::ok(results::revert_model_snapshot(&node, &id, &snap, del)?))).await
                }
                ("_upgrade", &Method::POST) => {
                    let wait = req.bool_param("wait_for_completion")?.unwrap_or(false);
                    blocking(move || Ok(Resp::ok(results::upgrade_job_snapshot(&node, &id, &snap, wait)?))).await
                }
                ("_update" | "_revert" | "_upgrade", _) => method_not_allowed(&req, &["POST"]),
                _ => crate::not_found_handler(&req),
            }
        }
        7 if s(3) == "model_snapshots" && s(5) == "_upgrade" && s(6) == "_stats" => {
            let (id, snap) = (s(2), s(4));
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            let a = allow()?;
            blocking(move || Ok(Resp::ok(results::snapshot_upgrade_stats(&node, &id, &snap, a)?))).await
        }
        5 if s(3) == "_forecast" && m == Method::DELETE => {
            let (id, fid) = (s(2), s(4));
            let allow = req.bool_param("allow_no_forecasts")?.unwrap_or(true);
            blocking(move || Ok(Resp::ok(results::delete_forecasts(&node, &id, &fid, allow)?))).await
        }
        _ => crate::not_found_handler(&req),
    }
}

// ---------------------------------------------------------------------------
// Datafeeds
// ---------------------------------------------------------------------------

async fn datafeeds(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let m = req.method.clone();
    let allow = || -> Result<bool, EsError> { Ok(req.bool_param("allow_no_match")?.or(req.bool_param("allow_no_datafeeds")?).unwrap_or(true)) };
    match n {
        2 => match m {
            Method::GET => {
                let (a, ex) = (allow()?, req.flag("exclude_generated")?);
                blocking(move || Ok(Resp::ok(anomaly::get_datafeeds(&node, None, a, ex)?))).await
            }
            _ => method_not_allowed(&req, &["GET"]),
        },
        3 if s(2) == "_stats" => match m {
            Method::GET => {
                let a = allow()?;
                blocking(move || Ok(Resp::ok(anomaly::datafeed_stats(&node, None, a)?))).await
            }
            _ => method_not_allowed(&req, &["GET"]),
        },
        3 if s(2) == "_preview" => match m {
            Method::GET | Method::POST => {
                let body = req.json()?;
                let (start, end) = (opt(&req, &body, "start"), opt(&req, &body, "end"));
                blocking(move || Ok(Resp::ok(elasticsearch_engine::ml::preview::preview(&node, None, &body, start.as_deref(), end.as_deref())?))).await
            }
            _ => method_not_allowed(&req, &["GET", "POST"]),
        },
        3 => {
            let id = s(2);
            match m {
                Method::GET => {
                    let (a, ex) = (allow()?, req.flag("exclude_generated")?);
                    blocking(move || Ok(Resp::ok(anomaly::get_datafeeds(&node, Some(&id), a, ex)?))).await
                }
                Method::PUT => {
                    let (body, raw) = (req.json()?, raw(&req));
                    let opts = datafeed::indices_options_from_params(&req.params);
                    let throttled = req.param("ignore_throttled").is_some();
                    blocking(move || {
                        let mut r = Resp::ok(anomaly::put_datafeed(&node, &id, &body, &raw, &opts)?);
                        if throttled {
                            warn(&mut r, IGNORE_THROTTLED_WARNING);
                        }
                        Ok(r)
                    })
                    .await
                }
                Method::DELETE => {
                    let force = req.flag("force")?;
                    blocking(move || Ok(Resp::ok(anomaly::delete_datafeed(&node, &id, force)?))).await
                }
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        4 => {
            let id = s(2);
            // `_data` takes ndjson / delimited data, not a JSON body.
            let body = if s(3) == "_data" { Value::Null } else { req.json()? };
            match (s(3).as_str(), &m) {
                ("_stats", &Method::GET) => {
                    let a = allow()?;
                    blocking(move || Ok(Resp::ok(anomaly::datafeed_stats(&node, Some(&id), a)?))).await
                }
                ("_start", &Method::POST) => {
                    let start = opt(&req, &body, "start");
                    let end = opt(&req, &body, "end");
                    blocking(move || Ok(Resp::ok(anomaly::start_datafeed(&node, &id, start.as_deref(), end.as_deref())?))).await
                }
                ("_stop", &Method::POST) => {
                    let a = match opt(&req, &body, "allow_no_match").or_else(|| opt(&req, &body, "allow_no_datafeeds")) {
                        Some(v) => v == "true" || v.is_empty(),
                        None => true,
                    };
                    let force = opt_bool(&req, &body, "force", false)?;
                    blocking(move || Ok(Resp::ok(anomaly::stop_datafeed(&node, &id, a, force)?))).await
                }
                ("_update", &Method::POST) => {
                    let raw = raw(&req);
                    let opts = datafeed::indices_options_from_params(&req.params);
                    let throttled = req.param("ignore_throttled").is_some();
                    blocking(move || {
                        let mut r = Resp::ok(anomaly::update_datafeed(&node, &id, &body, &raw, &opts)?);
                        if throttled {
                            warn(&mut r, IGNORE_THROTTLED_WARNING);
                        }
                        Ok(r)
                    })
                    .await
                }
                ("_preview", &Method::GET) | ("_preview", &Method::POST) => {
                    let (start, end) = (opt(&req, &body, "start"), opt(&req, &body, "end"));
                    blocking(move || Ok(Resp::ok(elasticsearch_engine::ml::preview::preview(&node, Some(&id), &body, start.as_deref(), end.as_deref())?))).await
                }
                ("_stats", _) => method_not_allowed(&req, &["GET"]),
                ("_start" | "_stop" | "_update", _) => method_not_allowed(&req, &["POST"]),
                _ => crate::not_found_handler(&req),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

// ---------------------------------------------------------------------------
// Calendars
// ---------------------------------------------------------------------------

async fn calendars(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let m = req.method.clone();
    match n {
        // `POST _ml/calendars[/{id}]` is the body form of the GET.
        2 | 3 if matches!(m, Method::GET | Method::POST) => {
            let id = segs.get(2).cloned();
            let body = req.json()?;
            let page = body.get("page").cloned().unwrap_or(Value::Null);
            let from = int_param(&req, "from")?.or_else(|| page.get("from").and_then(Value::as_i64));
            let size = int_param(&req, "size")?.or_else(|| page.get("size").and_then(Value::as_i64));
            blocking(move || Ok(Resp::ok(calendar::get_calendars(&node, id.as_deref(), from, size)?))).await
        }
        3 => {
            let id = s(2);
            match m {
                Method::PUT => {
                    let body = req.json()?;
                    blocking(move || Ok(Resp::ok(calendar::put_calendar(&node, &id, &body)?))).await
                }
                Method::DELETE => blocking(move || Ok(Resp::ok(calendar::delete_calendar(&node, &id)?))).await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE", "POST"]),
            }
        }
        4 if s(3) == "events" => {
            let id = s(2);
            match m {
                Method::GET => {
                    let body = req.json()?;
                    let job = opt(&req, &body, "job_id");
                    let start = opt(&req, &body, "start");
                    let end = opt(&req, &body, "end");
                    let page = body.get("page").cloned().unwrap_or(Value::Null);
                    let from = int_param(&req, "from")?.or_else(|| page.get("from").and_then(Value::as_i64));
                    let size = int_param(&req, "size")?.or_else(|| page.get("size").and_then(Value::as_i64));
                    blocking(move || Ok(Resp::ok(calendar::get_events(&node, &id, job.as_deref(), start.as_deref(), end.as_deref(), from, size)?))).await
                }
                Method::POST => {
                    let body = req.json()?;
                    blocking(move || Ok(Resp::ok(calendar::post_events(&node, &id, &body)?))).await
                }
                _ => method_not_allowed(&req, &["GET", "POST"]),
            }
        }
        5 if s(3) == "events" => {
            let (id, ev) = (s(2), s(4));
            match m {
                Method::DELETE => blocking(move || Ok(Resp::ok(calendar::delete_event(&node, &id, &ev)?))).await,
                _ => method_not_allowed(&req, &["DELETE"]),
            }
        }
        5 if s(3) == "jobs" => {
            let (id, jobs) = (s(2), s(4));
            match m {
                Method::PUT => blocking(move || Ok(Resp::ok(calendar::put_calendar_job(&node, &id, &jobs)?))).await,
                Method::DELETE => blocking(move || Ok(Resp::ok(calendar::delete_calendar_job(&node, &id, &jobs)?))).await,
                _ => method_not_allowed(&req, &["PUT", "DELETE"]),
            }
        }
        _ => crate::not_found_handler(&req),
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

async fn filters(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let m = req.method.clone();
    match (n, &m) {
        (2, &Method::GET) | (3, &Method::GET) => {
            let id = segs.get(2).cloned();
            let (from, size) = (int_param(&req, "from")?, int_param(&req, "size")?);
            blocking(move || Ok(Resp::ok(calendar::get_filters(&node, id.as_deref(), from, size)?))).await
        }
        (3, &Method::PUT) => {
            let id = s(2);
            let body = req.json()?;
            if body.is_null() {
                return Err(EsError::new(400, "parse_exception", "request body is required"));
            }
            blocking(move || Ok(Resp::ok(calendar::put_filter(&node, &id, &body)?))).await
        }
        (3, &Method::DELETE) => {
            let id = s(2);
            blocking(move || Ok(Resp::ok(calendar::delete_filter(&node, &id)?))).await
        }
        (4, &Method::POST) if s(3) == "_update" => {
            let id = s(2);
            let body = body_obj(&req)?;
            blocking(move || Ok(Resp::ok(calendar::update_filter(&node, &id, &body)?))).await
        }
        (2, _) => method_not_allowed(&req, &["GET"]),
        (3, _) => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
        _ => crate::not_found_handler(&req),
    }
}

// ---------------------------------------------------------------------------
// Trained models
// ---------------------------------------------------------------------------

fn get_opts(req: &Req) -> Result<(models::GetOpts, bool), EsError> {
    let list = |k: &str| -> Vec<String> { req.param(k).map(|s| s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect()).unwrap_or_default() };
    let mut include = list("include");
    let deprecated = req.param("include_model_definition").is_some();
    if deprecated {
        include = if req.bool_param("include_model_definition")?.unwrap_or(false) { vec!["definition".to_string()] } else { Vec::new() };
    }
    Ok((
        models::GetOpts {
            allow_no_match: req.bool_param("allow_no_match")?.unwrap_or(true),
            from: int_param(req, "from")?,
            size: int_param(req, "size")?,
            tags: list("tags"),
            include,
            decompress_definition: req.bool_param("decompress_definition")?.unwrap_or(true),
            exclude_generated: req.bool_param("exclude_generated")?.unwrap_or(false),
        },
        deprecated,
    ))
}

async fn get_models(state: std::path::PathBuf, id: Option<String>, req: &Req) -> R {
    let (opts, deprecated) = get_opts(req)?;
    blocking(move || {
        let mut r = Resp::ok(models::get_with(&state, id.as_deref(), &opts)?);
        if deprecated {
            warn(&mut r, "[include_model_definition] parameter is deprecated! Use [include=definition] instead.");
        }
        Ok(r)
    })
    .await
}

async fn trained_models(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let state = node.data.join("_state");
    let m = req.method.clone();
    let node_name = node.name.clone();
    match n {
        2 => match m {
            Method::GET => get_models(state, None, &req).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        3 if s(2) == "_stats" => match m {
            Method::GET => {
                let (allow, from, size) = (req.bool_param("allow_no_match")?.unwrap_or(true), int_param(&req, "from")?, int_param(&req, "size")?);
                blocking(move || Ok(Resp::ok(models::stats_named(&state, None, allow, from, size, &node_name)?))).await
            }
            _ => method_not_allowed(&req, &["GET"]),
        },
        3 => {
            let id = s(2);
            match m {
                Method::GET => get_models(state, Some(id), &req).await,
                Method::PUT => {
                    let body = req.json()?;
                    if body.is_null() {
                        return Err(EsError::new(400, "parse_exception", "request body is required"));
                    }
                    // `TrainedModelConfig` requires `input` (built-in packaged
                    // models are configured by the server).
                    let builtin = id.starts_with(".multilingual-e5-small") || id.starts_with(".elser_model_2");
                    // pytorch / NLP models default `input` to `input`.
                    const NLP: &[&str] = &["pass_through", "text_embedding", "text_expansion", "ner", "fill_mask", "text_classification", "zero_shot_classification", "question_answering", "text_similarity"];
                    let nlp = body.get("model_type").and_then(Value::as_str) == Some("pytorch")
                        || body.get("inference_config").and_then(Value::as_object).map(|o| o.keys().any(|k| NLP.contains(&k.as_str()))).unwrap_or(false);
                    if !builtin && !nlp && body.get("input").map(Value::is_null).unwrap_or(true) {
                        return Err(EsError::illegal_argument("[input] must not be null."));
                    }
                    let opts = models::PutOpts {
                        defer_definition_decompression: req.bool_param("defer_definition_decompression")?.unwrap_or(false),
                        wait_for_completion: req.bool_param("wait_for_completion")?.unwrap_or(false),
                    };
                    blocking(move || Ok(Resp::ok(models::put_with(&state, &id, &body, &opts)?))).await
                }
                Method::DELETE => {
                    let force = req.bool_param("force")?.unwrap_or(false);
                    blocking(move || {
                        let pipelines = node.ingest().get(None);
                        let referenced = models::referenced_model_ids(pipelines.iter().map(|(_, c)| c));
                        Ok(Resp::ok(models::delete_checked(&state, &id, force, &referenced)?))
                    })
                    .await
                }
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        4 if s(3) == "_stats" => {
            let id = s(2);
            let (allow, from, size) = (req.bool_param("allow_no_match")?.unwrap_or(true), int_param(&req, "from")?, int_param(&req, "size")?);
            blocking(move || Ok(Resp::ok(models::stats_named(&state, Some(&id), allow, from, size, &node_name)?))).await
        }
        4 if s(3) == "_infer" => {
            let id = s(2);
            let body = body_obj(&req)?;
            blocking(move || Ok(Resp::ok(models::infer(&state, &id, &body)?))).await
        }
        4 if s(3) == "vocabulary" => {
            let id = s(2);
            let body = body_obj(&req)?;
            blocking(move || Ok(Resp::ok(models::put_vocabulary(&state, &id, &body)?))).await
        }
        5 if s(3) == "model_aliases" => {
            let (id, alias) = (s(2), s(4));
            match m {
                Method::PUT => {
                    let reassign = req.bool_param("reassign")?.unwrap_or(false);
                    blocking(move || Ok(Resp::ok(models::put_alias(&state, &id, &alias, reassign)?))).await
                }
                Method::DELETE => blocking(move || Ok(Resp::ok(models::delete_alias(&state, &id, &alias)?))).await,
                _ => method_not_allowed(&req, &["PUT", "DELETE"]),
            }
        }
        5 if s(3) == "definition" => {
            let id = s(2);
            // Integer.parseInt on the path segment.
            let part: i64 = s(4).parse().map_err(|_| EsError::new(400, "number_format_exception", format!("For input string: \"{}\"", s(4))))?;
            let body = body_obj(&req)?;
            if body.get("definition").map(Value::is_null).unwrap_or(true) {
                return Err(EsError::illegal_argument("[definition] must not be null."));
            }
            if part < 0 {
                return Err(EsError::action_validation(&["[part] must be greater or equal to 0"]));
            }
            blocking(move || Ok(Resp::ok(models::put_definition_part(&state, &id, part as u64, &body)?))).await
        }
        5 if s(3) == "deployment" => {
            let id = s(2);
            let params = req.params.clone();
            let body = body_obj(&req)?;
            match s(4).as_str() {
                "_start" => {
                    if let Some(bid) = body.get("model_id").and_then(Value::as_str) {
                        if bid != id {
                            return Err(EsError::new(400, "status_exception", format!("Inconsistent model_id; '{bid}' specified in the body differs from '{id}' specified as a URL argument")));
                        }
                    }
                    let opts = models::StartOpts::parse(&params, &body)?;
                    blocking(move || Ok(Resp::ok(models::start_deployment_with(&state, &id, &opts)?))).await
                }
                "_stop" => {
                    let allow = opt_bool(&req, &body, "allow_no_match", true)?;
                    let force = opt_bool(&req, &body, "force", false)?;
                    blocking(move || Ok(Resp::ok(models::stop_deployment_with(&state, &id, allow, force)?))).await
                }
                "_update" => {
                    let body = match req.param("number_of_allocations") {
                        Some(v) => {
                            let n: i64 = v.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [number_of_allocations] with value [{v}]")))?;
                            json!({"number_of_allocations": n})
                        }
                        None if req.body.is_empty() => return Err(EsError::new(400, "parse_exception", "request body is required")),
                        None => body,
                    };
                    blocking(move || Ok(Resp::ok(models::update_deployment(&state, &id, &body)?))).await
                }
                "_infer" => {
                    blocking(move || {
                        let mut r = Resp::ok(models::infer(&state, &id, &body)?);
                        warn(&mut r, "[POST /_ml/trained_models/{model_id}/deployment/_infer] is deprecated! Use [POST /_ml/trained_models/{model_id}/_infer] instead.");
                        Ok(r)
                    })
                    .await
                }
                _ => crate::not_found_handler(&req),
            }
        }
        6 if s(3) == "deployment" && s(4) == "cache" && s(5) == "_clear" => {
            let id = s(2);
            blocking(move || Ok(Resp::ok(models::clear_cache(&state, &id)?))).await
        }
        _ => crate::not_found_handler(&req),
    }
}
