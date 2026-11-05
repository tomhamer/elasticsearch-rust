//! Anomaly detection results APIs (`results/buckets`, `records`,
//! `influencers`, `categories`, `overall_buckets`, `model_snapshots` and the
//! snapshot update / delete / revert / upgrade actions).
//!
//! Results are documents in the job's results index read through the
//! `.ml-anomalies-<job>` alias ([`super::resultsidx`]); each API is a search
//! with ES's filters, sort and paging (`JobResultsProvider`), and every hit
//! is re-rendered through the result class's `toXContent` so documents
//! indexed directly (as the YAML suite does) get ES's defaults and field
//! order.

use super::anomaly::expand_jobs;
use super::resultsidx::{self as ridx, hits, read_alias, search, total, ts_millis};
use super::store::service;
use super::xc::bad_request;
use crate::error::{EsError, Result};
use crate::node::Node;
use serde_json::{json, Map, Value};

/// Paging of a results API (`PageParams`).
#[derive(Clone, Copy, Debug)]
pub struct Page {
    pub from: i64,
    pub size: i64,
}

impl Default for Page {
    fn default() -> Self {
        Page { from: 0, size: 100 }
    }
}

fn check_page(p: &Page) -> Result<()> {
    if p.from < 0 {
        return Err(bad_request(format!("Parameter [from] cannot be < 0")));
    }
    if p.size < 0 {
        return Err(bad_request(format!("Parameter [size] cannot be < 0")));
    }
    if p.from + p.size > 10_000 {
        return Err(bad_request("The sum of parameters [from] and [size] cannot be higher than 10000."));
    }
    Ok(())
}

/// Parse a results `start` / `end` param (epoch millis or a date string;
/// ES's `QueryParams` message).
pub fn parse_time_param(v: &Value, name: &str) -> Result<i64> {
    let s = match v {
        Value::Number(n) => return n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).ok_or_else(|| time_error(name, &n.to_string())),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let t = s.trim();
    if t == "now" {
        return Ok(crate::date::now_millis());
    }
    if let Ok(n) = t.parse::<i64>() {
        return Ok(n);
    }
    crate::date::DateFormatter::new("strict_date_optional_time").and_then(|f| f.parse(t)).map_err(|_| time_error(name, &s))
}

fn time_error(name: &str, s: &str) -> EsError {
    EsError::illegal_argument(format!("Query param [{name}] with value [{s}] cannot be parsed as a date or converted to a number (epoch)."))
}

fn f64_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn job_exists(node: &Node, job_id: &str) -> Result<()> {
    let st = service(node).read();
    if !st.jobs.contains_key(job_id) {
        return Err(EsError::resource_not_found(format!("No known job with id '{job_id}'")));
    }
    Ok(())
}

/// `{"count": total, "<key>": [...]}` of a search over the job's alias.
fn page_search(node: &Node, job_id: &str, body: Value, key: &str, render: impl Fn(&Value) -> Value) -> Result<Value> {
    let resp = search(node, &read_alias(job_id), &body)?;
    let items: Vec<Value> = hits(&resp).iter().map(|(_, s)| render(s)).collect();
    Ok(json!({"count": total(&resp), key: items}))
}

fn sort_clause(field: &str, desc: bool) -> Value {
    json!({field: {"order": if desc { "desc" } else { "asc" }, "unmapped_type": "double"}})
}

fn range_filter(filters: &mut Vec<Value>, field: &str, start: Option<i64>, end: Option<i64>) {
    if start.is_none() && end.is_none() {
        return;
    }
    let mut r = Map::new();
    if let Some(s) = start {
        r.insert("gte".into(), json!(s));
    }
    if let Some(e) = end {
        r.insert("lt".into(), json!(e));
    }
    r.insert("format".into(), json!("epoch_millis"));
    filters.push(json!({"range": {field: r}}));
}

fn interim_filter(filters: &mut Vec<Value>, exclude_interim: bool) {
    if exclude_interim {
        filters.push(json!({"bool": {"must_not": {"term": {"is_interim": true}}}}));
    }
}

fn score_filter(filters: &mut Vec<Value>, field: &str, min: Option<f64>) {
    if let Some(m) = min {
        filters.push(json!({"range": {field: {"gte": m}}}));
    }
}

// ---------------------------------------------------------------------------
// Rendering (ES `toXContent` of each result class)
// ---------------------------------------------------------------------------

fn get_or<'a>(src: &'a Value, k: &str, default: Value) -> Value {
    src.get(k).filter(|v| !v.is_null()).cloned().unwrap_or(default)
}

fn ts_value(src: &Value, k: &str) -> Value {
    src.get(k).and_then(ts_millis).map(|m| json!(m)).unwrap_or(Value::Null)
}

/// ES `ReservedFieldNames.isValidFieldName`: a by / over / partition /
/// influencer field name that is not itself a result field.
fn is_valid_field_name(name: &str) -> bool {
    const RESERVED: &[&str] = &[
        "job_id", "result_type", "timestamp", "anomaly_score", "bucket_span", "initial_anomaly_score", "event_count", "is_interim", "bucket_influencers", "processing_time_ms", "scheduled_events",
        "records", "probability", "multi_bucket_impact", "record_score", "initial_record_score", "detector_index", "by_field_name", "by_field_value", "correlated_by_field_value",
        "partition_field_name", "partition_field_value", "function", "function_description", "typical", "actual", "field_name", "over_field_name", "over_field_value", "causes", "influencers",
        "influencer_field_name", "influencer_field_value", "influencer_score", "initial_influencer_score", "raw_anomaly_score", "category_id", "terms", "regex", "max_matching_length",
        "examples", "num_matches", "preferred_to_categories", "grok_pattern", "mlcategory", "snapshot_id", "snapshot_doc_count", "model_size_stats", "quantiles", "retain", "description",
        "min_version", "latest_record_time_stamp", "latest_result_time_stamp", "forecast_id", "model_feature", "model_lower", "model_upper", "model_median", "forecast_lower",
        "forecast_upper", "forecast_prediction", "all_field_values", "geo_results", "anomaly_score_explanation", "model_bytes", "log_time", "@timestamp", "_id", "_index", "_source",
    ];
    !name.is_empty() && !RESERVED.contains(&name)
}

/// `Bucket.toXContent`.
pub fn render_bucket(src: &Value, records: Option<Vec<Value>>) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), get_or(src, "job_id", Value::Null));
    o.insert("timestamp".into(), ts_value(src, "timestamp"));
    o.insert("anomaly_score".into(), json!(f64_of(&src["anomaly_score"]).unwrap_or(0.0)));
    o.insert("bucket_span".into(), json!(src["bucket_span"].as_i64().or_else(|| src["bucket_span"].as_f64().map(|f| f as i64)).unwrap_or(0)));
    o.insert("initial_anomaly_score".into(), json!(f64_of(&src["initial_anomaly_score"]).unwrap_or(0.0)));
    if let Some(r) = records.filter(|r| !r.is_empty()) {
        o.insert("records".into(), Value::Array(r));
    }
    o.insert("event_count".into(), json!(src["event_count"].as_i64().unwrap_or(0)));
    o.insert("is_interim".into(), json!(src["is_interim"].as_bool().unwrap_or(false)));
    o.insert("bucket_influencers".into(), get_or(src, "bucket_influencers", json!([])));
    o.insert("processing_time_ms".into(), json!(src["processing_time_ms"].as_i64().unwrap_or(0)));
    if let Some(se) = src.get("scheduled_events").filter(|v| v.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
        o.insert("scheduled_events".into(), se.clone());
    }
    o.insert("result_type".into(), json!("bucket"));
    Value::Object(o)
}

/// `AnomalyRecord.toXContent`.
pub fn render_record(src: &Value) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), get_or(src, "job_id", Value::Null));
    o.insert("result_type".into(), json!("record"));
    o.insert("probability".into(), json!(f64_of(&src["probability"]).unwrap_or(0.0)));
    if let Some(m) = src.get("multi_bucket_impact").and_then(f64_of) {
        o.insert("multi_bucket_impact".into(), json!(m));
    }
    o.insert("record_score".into(), json!(f64_of(&src["record_score"]).unwrap_or(0.0)));
    o.insert("initial_record_score".into(), json!(f64_of(&src["initial_record_score"]).unwrap_or(0.0)));
    o.insert("bucket_span".into(), json!(src["bucket_span"].as_i64().unwrap_or(0)));
    o.insert("detector_index".into(), json!(src["detector_index"].as_i64().unwrap_or(0)));
    o.insert("is_interim".into(), json!(src["is_interim"].as_bool().unwrap_or(false)));
    o.insert("timestamp".into(), ts_value(src, "timestamp"));
    for k in ["by_field_name", "by_field_value", "correlated_by_field_value", "partition_field_name", "partition_field_value", "function", "function_description", "typical", "actual", "field_name", "over_field_name", "over_field_value"] {
        if let Some(v) = src.get(k).filter(|v| !v.is_null()) {
            o.insert(k.into(), v.clone());
        }
    }
    for k in ["causes", "influencers"] {
        if let Some(v) = src.get(k).filter(|v| v.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
            o.insert(k.into(), v.clone());
        }
    }
    for k in ["geo_results", "anomaly_score_explanation"] {
        if let Some(v) = src.get(k).filter(|v| !v.is_null()) {
            o.insert(k.into(), v.clone());
        }
    }
    // `inputFieldMap`: by / over / partition and influencer fields by name.
    let mut input: Vec<(String, Vec<Value>)> = Vec::new();
    let mut add = |name: Option<&str>, value: Option<&Value>| {
        if let (Some(n), Some(v)) = (name, value) {
            if is_valid_field_name(n) && !v.is_null() {
                match input.iter_mut().find(|(k, _)| k == n) {
                    Some((_, vs)) => {
                        if !vs.contains(v) {
                            vs.push(v.clone());
                        }
                    }
                    None => input.push((n.to_string(), vec![v.clone()])),
                }
            }
        }
    };
    add(src["by_field_name"].as_str(), src.get("by_field_value"));
    add(src["over_field_name"].as_str(), src.get("over_field_value"));
    add(src["partition_field_name"].as_str(), src.get("partition_field_value"));
    for inf in src["influencers"].as_array().into_iter().flatten() {
        for v in inf["influencer_field_values"].as_array().into_iter().flatten() {
            add(inf["influencer_field_name"].as_str(), Some(v));
        }
    }
    for (k, vs) in input {
        if !o.contains_key(&k) {
            o.insert(k, Value::Array(vs));
        }
    }
    Value::Object(o)
}

/// `Influencer.toXContent`.
pub fn render_influencer(src: &Value) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), get_or(src, "job_id", Value::Null));
    o.insert("result_type".into(), json!("influencer"));
    let name = src["influencer_field_name"].as_str().unwrap_or("").to_string();
    o.insert("influencer_field_name".into(), json!(name));
    o.insert("influencer_field_value".into(), get_or(src, "influencer_field_value", json!("")));
    if is_valid_field_name(&name) {
        o.insert(name.clone(), get_or(src, "influencer_field_value", json!("")));
    }
    o.insert("influencer_score".into(), json!(f64_of(&src["influencer_score"]).unwrap_or(0.0)));
    o.insert("initial_influencer_score".into(), json!(f64_of(&src["initial_influencer_score"]).unwrap_or(0.0)));
    o.insert("probability".into(), json!(f64_of(&src["probability"]).unwrap_or(0.0)));
    o.insert("bucket_span".into(), json!(src["bucket_span"].as_i64().unwrap_or(0)));
    o.insert("is_interim".into(), json!(src["is_interim"].as_bool().unwrap_or(false)));
    o.insert("timestamp".into(), ts_value(src, "timestamp"));
    Value::Object(o)
}

/// `CategoryDefinition.toXContent` (the grok pattern is derived when the
/// document has none, as `GrokPatternCreator` does at read time).
pub fn render_category(src: &Value) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), get_or(src, "job_id", Value::Null));
    let cid = src["category_id"].as_i64().unwrap_or(0);
    o.insert("category_id".into(), json!(cid));
    let pname = src["partition_field_name"].as_str().map(str::to_string);
    if let Some(p) = &pname {
        o.insert("partition_field_name".into(), json!(p));
        o.insert("partition_field_value".into(), get_or(src, "partition_field_value", json!("")));
    }
    let terms = src["terms"].as_str().unwrap_or("").to_string();
    let regex = src["regex"].as_str().unwrap_or("").to_string();
    let examples: Vec<String> = src["examples"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect();
    o.insert("terms".into(), json!(terms));
    o.insert("regex".into(), json!(regex));
    o.insert("max_matching_length".into(), json!(src["max_matching_length"].as_i64().unwrap_or(0)));
    o.insert("examples".into(), json!(examples));
    let grok = match src["grok_pattern"].as_str() {
        Some(g) => g.to_string(),
        None => super::process::grok_pattern(&terms, &examples),
    };
    o.insert("grok_pattern".into(), json!(grok));
    if let Some(n) = src.get("num_matches").and_then(Value::as_i64) {
        o.insert("num_matches".into(), json!(n));
    }
    if let Some(p) = src.get("preferred_to_categories").filter(|v| v.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
        o.insert("preferred_to_categories".into(), p.clone());
    }
    if let Some(p) = &pname {
        if is_valid_field_name(p) {
            o.insert(p.clone(), get_or(src, "partition_field_value", json!("")));
        }
    }
    o.insert("result_type".into(), json!("category_definition"));
    o.insert("mlcategory".into(), json!(cid.to_string()));
    Value::Object(o)
}

/// `ModelSnapshot.toXContent`.
pub fn render_snapshot(src: &Value) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), get_or(src, "job_id", Value::Null));
    o.insert("min_version".into(), get_or(src, "min_version", json!("6.3.0")));
    if let Some(t) = src.get("timestamp").and_then(ts_millis) {
        o.insert("timestamp".into(), json!(t));
    }
    if let Some(d) = src.get("description").filter(|v| !v.is_null()) {
        o.insert("description".into(), d.clone());
    }
    if let Some(s) = src.get("snapshot_id").filter(|v| !v.is_null()) {
        o.insert("snapshot_id".into(), s.clone());
    }
    o.insert("snapshot_doc_count".into(), json!(src["snapshot_doc_count"].as_i64().unwrap_or(0)));
    if let Some(m) = src.get("model_size_stats").filter(|v| v.is_object()) {
        o.insert("model_size_stats".into(), m.clone());
    }
    for k in ["latest_record_time_stamp", "latest_result_time_stamp"] {
        if let Some(t) = src.get(k).and_then(ts_millis) {
            o.insert(k.into(), json!(t));
        }
    }
    if let Some(q) = src.get("quantiles").filter(|v| v.is_object()) {
        o.insert("quantiles".into(), q.clone());
    }
    o.insert("retain".into(), json!(src["retain"].as_bool().unwrap_or(false)));
    Value::Object(o)
}

// ---------------------------------------------------------------------------
// Buckets
// ---------------------------------------------------------------------------

/// Parameters of `GET results/buckets[/{timestamp}]` (URL params merged
/// with the body; ES reads either).
#[derive(Default)]
pub struct BucketsReq {
    pub timestamp: Option<Value>,
    pub expand: bool,
    pub exclude_interim: bool,
    pub start: Option<Value>,
    pub end: Option<Value>,
    pub anomaly_score: Option<f64>,
    pub sort: Option<String>,
    pub desc: bool,
    pub page: Option<Page>,
}

fn records_of_bucket(node: &Node, job_id: &str, ts: i64) -> Result<Vec<Value>> {
    let body = json!({"size": 10000, "track_total_hits": true,
        "query": {"bool": {"filter": [{"term": {"result_type": "record"}}, {"term": {"timestamp": ts}}]}},
        "sort": [sort_clause("record_score", true)]});
    let resp = search(node, &read_alias(job_id), &body)?;
    Ok(hits(&resp).iter().map(|(_, s)| render_record(s)).collect())
}

pub fn get_buckets(node: &Node, job_id: &str, req: &BucketsReq) -> Result<Value> {
    job_exists(node, job_id)?;
    if let Some(ts) = &req.timestamp {
        if req.start.is_some() || req.end.is_some() || req.page.is_some() || req.anomaly_score.is_some() {
            return Err(bad_request("Param [timestamp] is incompatible with [start, end, from, size, anomaly_score]."));
        }
        let t = parse_time_param(ts, "timestamp")?;
        let mut filters = vec![json!({"term": {"result_type": "bucket"}}), json!({"term": {"timestamp": t}})];
        interim_filter(&mut filters, req.exclude_interim);
        let resp = search(node, &read_alias(job_id), &json!({"size": 1, "track_total_hits": true, "query": {"bool": {"filter": filters}}}))?;
        let h = hits(&resp);
        if h.is_empty() {
            return Err(EsError::resource_not_found("Could not find requested buckets"));
        }
        let recs = if req.expand { Some(records_of_bucket(node, job_id, t)?) } else { None };
        return Ok(json!({"count": 1, "buckets": [render_bucket(&h[0].1, recs)]}));
    }
    let page = req.page.unwrap_or_default();
    check_page(&page)?;
    let start = req.start.as_ref().map(|v| parse_time_param(v, "start")).transpose()?;
    let end = req.end.as_ref().map(|v| parse_time_param(v, "end")).transpose()?;
    let mut filters = vec![json!({"term": {"result_type": "bucket"}})];
    range_filter(&mut filters, "timestamp", start, end);
    interim_filter(&mut filters, req.exclude_interim);
    score_filter(&mut filters, "anomaly_score", req.anomaly_score);
    let sort_field = req.sort.clone().unwrap_or_else(|| "timestamp".into());
    let mut sort = vec![sort_clause(&sort_field, req.desc)];
    if sort_field != "timestamp" {
        sort.push(sort_clause("timestamp", req.desc));
    }
    let body = json!({"from": page.from, "size": page.size, "track_total_hits": true, "query": {"bool": {"filter": filters}}, "sort": sort});
    let resp = search(node, &read_alias(job_id), &body)?;
    let mut buckets = Vec::new();
    for (_, s) in hits(&resp) {
        let recs = if req.expand { s.get("timestamp").and_then(ts_millis).map(|t| records_of_bucket(node, job_id, t)).transpose()? } else { None };
        buckets.push(render_bucket(&s, recs));
    }
    Ok(json!({"count": total(&resp), "buckets": buckets}))
}

// ---------------------------------------------------------------------------
// Records / influencers
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ScoredReq {
    pub exclude_interim: bool,
    pub start: Option<Value>,
    pub end: Option<Value>,
    /// `record_score` / `influencer_score` lower bound.
    pub score: Option<f64>,
    pub sort: Option<String>,
    pub desc: Option<bool>,
    pub page: Page,
}

pub fn get_records(node: &Node, job_id: &str, req: &ScoredReq) -> Result<Value> {
    job_exists(node, job_id)?;
    check_page(&req.page)?;
    let start = req.start.as_ref().map(|v| parse_time_param(v, "start")).transpose()?;
    let end = req.end.as_ref().map(|v| parse_time_param(v, "end")).transpose()?;
    let mut filters = vec![json!({"term": {"result_type": "record"}})];
    range_filter(&mut filters, "timestamp", start, end);
    interim_filter(&mut filters, req.exclude_interim);
    score_filter(&mut filters, "record_score", req.score);
    let sort_field = req.sort.clone().unwrap_or_else(|| "record_score".into());
    let body = json!({"from": req.page.from, "size": req.page.size, "track_total_hits": true, "query": {"bool": {"filter": filters}}, "sort": [sort_clause(&sort_field, req.desc.unwrap_or(true))]});
    page_search(node, job_id, body, "records", render_record)
}

pub fn get_influencers(node: &Node, job_id: &str, req: &ScoredReq) -> Result<Value> {
    job_exists(node, job_id)?;
    check_page(&req.page)?;
    let start = req.start.as_ref().map(|v| parse_time_param(v, "start")).transpose()?;
    let end = req.end.as_ref().map(|v| parse_time_param(v, "end")).transpose()?;
    let mut filters = vec![json!({"term": {"result_type": "influencer"}})];
    range_filter(&mut filters, "timestamp", start, end);
    interim_filter(&mut filters, req.exclude_interim);
    score_filter(&mut filters, "influencer_score", req.score);
    let sort_field = req.sort.clone().unwrap_or_else(|| "influencer_score".into());
    let body = json!({"from": req.page.from, "size": req.page.size, "track_total_hits": true, "query": {"bool": {"filter": filters}}, "sort": [sort_clause(&sort_field, req.desc.unwrap_or(true))]});
    page_search(node, job_id, body, "influencers", render_influencer)
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

pub fn get_categories(node: &Node, job_id: &str, category_id: Option<i64>, partition_field_value: Option<&str>, page: Option<Page>) -> Result<Value> {
    job_exists(node, job_id)?;
    if category_id.is_some() && page.is_some() {
        return Err(bad_request("Both [category_id] and [page] parameters are specified. Only one may be specified"));
    }
    let mut filters = Vec::new();
    let (from, size) = match category_id {
        Some(c) => {
            filters.push(json!({"term": {"category_id": c}}));
            (0, 1)
        }
        None => {
            let p = page.unwrap_or_default();
            check_page(&p)?;
            filters.push(json!({"exists": {"field": "category_id"}}));
            (p.from, p.size)
        }
    };
    if let Some(p) = partition_field_value {
        filters.push(json!({"term": {"partition_field_value": p}}));
    }
    let body = json!({"from": from, "size": size, "track_total_hits": true, "query": {"bool": {"filter": filters}}, "sort": [{"category_id": {"order": "asc", "unmapped_type": "long"}}]});
    page_search(node, job_id, body, "categories", render_category)
}

// ---------------------------------------------------------------------------
// Overall buckets
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct OverallReq {
    pub top_n: Option<i64>,
    pub bucket_span: Option<String>,
    pub overall_score: Option<f64>,
    pub exclude_interim: bool,
    pub start: Option<Value>,
    pub end: Option<Value>,
    pub allow_no_match: bool,
}

fn align_floor(t: i64, span: i64) -> i64 {
    t.div_euclid(span) * span
}

fn align_ceil(t: i64, span: i64) -> i64 {
    let f = align_floor(t, span);
    if f == t { f } else { f + span }
}

pub fn get_overall_buckets(node: &Node, expr: &str, req: &OverallReq) -> Result<Value> {
    let top_n = req.top_n.unwrap_or(1);
    if top_n <= 0 {
        return Err(EsError::illegal_argument(format!("[topN] parameter must be positive, found [{top_n}]")));
    }
    let start = req.start.as_ref().map(|v| parse_time_param(v, "start")).transpose()?;
    let end = req.end.as_ref().map(|v| parse_time_param(v, "end")).transpose()?;
    let st = service(node).read();
    let ids = expand_jobs(&st, Some(expr), req.allow_no_match)?;
    if ids.is_empty() {
        return Ok(json!({"count": 0, "overall_buckets": []}));
    }
    let mut max_span_ms = 0i64;
    let mut max_span_rep = String::from("5m");
    let mut indices: Vec<String> = Vec::new();
    for id in &ids {
        let job = &st.jobs[id];
        let rep = job["analysis_config"]["bucket_span"].as_str().unwrap_or("5m").to_string();
        let bs = super::job::parse_time(&rep, "bucket_span").map(|t| t.millis()).unwrap_or(300_000);
        if bs > max_span_ms {
            max_span_ms = bs;
            max_span_rep = rep;
        }
        let idx = ridx::results_index(job);
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    drop(st);
    let overall_span_ms = match &req.bucket_span {
        Some(bs) => {
            let t = super::job::parse_time(bs, "bucket_span")?;
            if t.millis() < max_span_ms {
                return Err(EsError::illegal_argument(format!("Param [bucket_span] must be greater or equal to the max bucket_span [{max_span_rep}]")));
            }
            t.millis()
        }
        None => max_span_ms,
    };
    let mut filters = vec![json!({"term": {"result_type": "bucket"}}), json!({"terms": {"job_id": ids}})];
    range_filter(&mut filters, "timestamp", start.map(|s| align_ceil(s, max_span_ms)), end.map(|e| align_floor(e, max_span_ms)));
    interim_filter(&mut filters, req.exclude_interim);
    let docs = ridx::search_all(node, &indices.join(","), json!({"bool": {"filter": filters}}), json!([{"timestamp": {"order": "asc"}}]))?;
    // Stage 1 (`OverallBucketsProvider`): per max-bucket-span bucket, each
    // job's max anomaly score; overall score = mean of the top_n of them.
    let mut buckets: std::collections::BTreeMap<i64, std::collections::BTreeMap<String, (f64, bool)>> = Default::default();
    for (_, s) in docs {
        let Some(t) = s.get("timestamp").and_then(ts_millis) else { continue };
        let job = s["job_id"].as_str().unwrap_or("").to_string();
        let score = f64_of(&s["anomaly_score"]).unwrap_or(0.0);
        let interim = s["is_interim"].as_bool().unwrap_or(false);
        let e = buckets.entry(align_floor(t, max_span_ms)).or_default().entry(job).or_insert((0.0, false));
        e.0 = e.0.max(score);
        e.1 |= interim;
    }
    // (overall score, jobs -> max score, interim) per stage-1 bucket.
    let stage1: Vec<(i64, f64, std::collections::BTreeMap<String, (f64, bool)>)> = buckets
        .into_iter()
        .map(|(ts, jobs)| {
            let mut scores: Vec<f64> = jobs.values().map(|(s, _)| *s).collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let n = (top_n as usize).min(scores.len()).max(1);
            (ts, scores.iter().take(n).sum::<f64>() / n as f64, jobs)
        })
        .collect();
    // Stage 2 (`OverallBucketsAggregator`): a larger bucket_span merges the
    // stage-1 buckets it spans: max overall score, per-job max scores.
    let mut merged: std::collections::BTreeMap<i64, (f64, std::collections::BTreeMap<String, (f64, bool)>)> = Default::default();
    for (ts, overall, jobs) in stage1 {
        let e = merged.entry(align_floor(ts, overall_span_ms)).or_insert((0.0, Default::default()));
        e.0 = e.0.max(overall);
        for (j, (s, i)) in jobs {
            let je = e.1.entry(j).or_insert((0.0, false));
            je.0 = je.0.max(s);
            je.1 |= i;
        }
    }
    let mut out = Vec::new();
    for (ts, (overall, jobs)) in merged {
        if let Some(min) = req.overall_score {
            if overall < min {
                continue;
            }
        }
        let is_interim = jobs.values().any(|(_, i)| *i);
        let job_list: Vec<Value> = jobs.iter().map(|(j, (s, _))| json!({"job_id": j, "max_anomaly_score": s})).collect();
        out.push(json!({"timestamp": ts, "bucket_span": overall_span_ms / 1000, "overall_score": overall, "jobs": job_list, "is_interim": is_interim, "result_type": "overall_bucket"}));
    }
    Ok(json!({"count": out.len(), "overall_buckets": out}))
}

// ---------------------------------------------------------------------------
// Model snapshots
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct SnapshotsReq {
    pub snapshot_id: Option<String>,
    pub start: Option<Value>,
    pub end: Option<Value>,
    pub sort: Option<String>,
    pub desc: Option<bool>,
    pub page: Page,
}

pub fn get_model_snapshots(node: &Node, job_id: &str, req: &SnapshotsReq) -> Result<Value> {
    job_exists(node, job_id)?;
    check_page(&req.page)?;
    let start = req.start.as_ref().map(|v| parse_time_param(v, "start")).transpose()?;
    let end = req.end.as_ref().map(|v| parse_time_param(v, "end")).transpose()?;
    let mut filters = vec![json!({"exists": {"field": "snapshot_doc_count"}})];
    if let Some(s) = req.snapshot_id.as_deref().filter(|s| !s.is_empty() && *s != "_all") {
        filters.push(json!({"term": {"snapshot_id": s}}));
    }
    range_filter(&mut filters, "timestamp", start, end);
    let sort_field = req.sort.clone().unwrap_or_else(|| "timestamp".into());
    let body = json!({"from": req.page.from, "size": req.page.size, "track_total_hits": true, "query": {"bool": {"filter": filters}}, "sort": [sort_clause(&sort_field, req.desc.unwrap_or(true))]});
    page_search(node, job_id, body, "model_snapshots", render_snapshot)
}

/// The snapshot document id (`ModelSnapshot.documentId`).
pub fn snapshot_doc_id(job_id: &str, snapshot_id: &str) -> String {
    format!("{job_id}_model_snapshot_{snapshot_id}")
}

fn no_snapshot(job_id: &str, snapshot_id: &str) -> EsError {
    EsError::resource_not_found(format!("No model snapshot with id [{snapshot_id}] exists for job [{job_id}]"))
}

/// The snapshot descriptor by document id (as the update / revert /
/// upgrade actions look it up).
pub fn get_snapshot_doc(node: &Node, job_id: &str, snapshot_id: &str) -> Result<Value> {
    job_exists(node, job_id)?;
    let resp = search(node, &read_alias(job_id), &json!({"size": 1, "query": {"bool": {"filter": [{"ids": {"values": [snapshot_doc_id(job_id, snapshot_id)]}}, {"exists": {"field": "snapshot_doc_count"}}]}}}))?;
    hits(&resp).into_iter().next().map(|(_, s)| s).ok_or_else(|| no_snapshot(job_id, snapshot_id))
}

/// `POST model_snapshots/{snapshot}/_update`.
pub fn update_model_snapshot(node: &Node, job_id: &str, snapshot_id: &str, body: &Value) -> Result<Value> {
    let mut doc = get_snapshot_doc(node, job_id, snapshot_id)?;
    let o = body.as_object().cloned().unwrap_or_default();
    for (k, v) in &o {
        match k.as_str() {
            "description" => {
                doc["description"] = json!(v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
            }
            "retain" => {
                let b = match v {
                    Value::Bool(b) => *b,
                    Value::String(s) if s == "true" || s == "false" => s == "true",
                    _ => return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
                };
                doc["retain"] = json!(b);
            }
            "job_id" | "snapshot_id" => {}
            other => return Err(super::xc::xparse(format!("[1:2] [update_model_snapshot_request] unknown field [{other}]"))),
        }
    }
    let st = service(node).read();
    let job = st.jobs.get(job_id).cloned().unwrap_or(Value::Null);
    drop(st);
    ridx::write_results(node, job_id, &job, vec![(snapshot_doc_id(job_id, snapshot_id), doc.clone())])?;
    Ok(json!({"acknowledged": true, "model": render_snapshot(&doc)}))
}

/// `DELETE model_snapshots/{snapshot}`.
pub fn delete_model_snapshot(node: &Node, job_id: &str, snapshot_id: &str) -> Result<Value> {
    job_exists(node, job_id)?;
    let resp = search(node, &read_alias(job_id), &json!({"size": 1, "query": {"bool": {"filter": [{"term": {"snapshot_id": snapshot_id}}, {"exists": {"field": "snapshot_doc_count"}}]}}}))?;
    let found = hits(&resp);
    if found.is_empty() {
        return Err(no_snapshot(job_id, snapshot_id));
    }
    let st = service(node).read();
    let job = st.jobs.get(job_id).cloned().unwrap_or(Value::Null);
    drop(st);
    if job["model_snapshot_id"].as_str() == Some(snapshot_id) {
        return Err(EsError::illegal_argument(format!("Model snapshot '{snapshot_id}' is the active snapshot for job '{job_id}', so cannot be deleted")));
    }
    let ids: Vec<String> = found.iter().map(|(id, _)| id.clone()).collect();
    ridx::delete_results(node, job_id, &job, &ids)?;
    ridx::delete_by_query(node, ridx::STATE_PATTERN, json!({"prefix": {"_id": format!("{job_id}_model_state_{snapshot_id}#")}}));
    Ok(json!({"acknowledged": true}))
}

/// `POST model_snapshots/{snapshot}/_revert`.
pub fn revert_model_snapshot(node: &Node, job_id: &str, snapshot_id: &str, delete_intervening: bool) -> Result<Value> {
    let svc = service(node);
    {
        let st = svc.read();
        if !st.jobs.contains_key(job_id) {
            return Err(EsError::resource_not_found(format!("No known job with id '{job_id}'")));
        }
        let state = st.job_state(job_id);
        if state != "closed" {
            return Err(EsError::new(409, "status_exception", format!("Cannot revert to snapshot [{snapshot_id}] because job [{job_id}] is {state}")));
        }
        if let Some(df) = st.datafeed_for_job(job_id) {
            if st.datafeed_state(&df) != "stopped" {
                return Err(EsError::new(409, "status_exception", format!("Cannot revert to snapshot [{snapshot_id}] because datafeed [{df}] is {}", st.datafeed_state(&df))));
            }
        }
    }
    let doc = if snapshot_id == "empty" { json!({"job_id": job_id, "snapshot_id": "empty", "snapshot_doc_count": 0}) } else { get_snapshot_doc(node, job_id, snapshot_id)? };
    let job = svc.mutate(|st| {
        let j = st.jobs.get_mut(job_id).ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{job_id}'")))?;
        if let Some(o) = j.as_object_mut() {
            if snapshot_id == "empty" {
                o.remove("model_snapshot_id");
            } else {
                o.insert("model_snapshot_id".into(), json!(snapshot_id));
            }
            if let Some(mv) = doc.get("min_version").filter(|v| v.is_string()) {
                o.insert("model_snapshot_min_version".into(), mv.clone());
            }
        }
        Ok(j.clone())
    })?;
    super::process::drop_process(node, job_id);
    if delete_intervening {
        let cutoff = doc.get("latest_result_time_stamp").and_then(ts_millis);
        let mut filters = vec![json!({"terms": {"result_type": ["bucket", "record", "influencer", "bucket_influencer", "model_plot"]}})];
        if let Some(c) = cutoff {
            filters.push(json!({"range": {"timestamp": {"gt": c}}}));
        }
        ridx::delete_by_query(node, &read_alias(job_id), json!({"bool": {"filter": filters}}));
        // Data counts restart from the snapshot's latest record.
        let counts_id = format!("{job_id}_data_counts");
        if let Some(mut dc) = ridx::get(node, &read_alias(job_id), &counts_id) {
            match doc.get("latest_record_time_stamp").and_then(ts_millis) {
                Some(t) => {
                    dc["latest_record_timestamp"] = json!(t);
                    dc["latest_bucket_timestamp"] = Value::Null;
                }
                None => {
                    dc = json!({"job_id": job_id, "processed_record_count": 0, "processed_field_count": 0, "input_bytes": 0, "input_field_count": 0, "invalid_date_count": 0, "missing_field_count": 0,
                        "out_of_order_timestamp_count": 0, "empty_bucket_count": 0, "sparse_bucket_count": 0, "bucket_count": 0, "input_record_count": 0, "log_time": crate::date::now_millis()});
                }
            }
            let _ = ridx::write_results(node, job_id, &job, vec![(counts_id, dc)]);
        }
    }
    Ok(json!({"model": render_snapshot(&doc)}))
}

/// `POST model_snapshots/{snapshot}/_upgrade`: the snapshot must exist and
/// the job be closed. Model state written by ES's native process cannot
/// be read here, so the upgrade task is recorded as failed (the stats API
/// reports it as ES does for a corrupt snapshot).
pub fn upgrade_job_snapshot(node: &Node, job_id: &str, snapshot_id: &str, wait: bool) -> Result<Value> {
    let svc = service(node);
    {
        let st = svc.read();
        if !st.jobs.contains_key(job_id) {
            return Err(EsError::resource_not_found(format!("No known job with id '{job_id}'")));
        }
        if st.job_state(job_id) != "closed" {
            return Err(EsError::new(409, "status_exception", format!("Cannot upgrade snapshot [{snapshot_id}] for job [{job_id}] as the job is not closed")));
        }
        let job = &st.jobs[job_id];
        if job["model_snapshot_id"].as_str() == Some(snapshot_id) {
            return Err(EsError::new(409, "status_exception", format!("Cannot upgrade snapshot [{snapshot_id}] for job [{job_id}] as it is the current snapshot")));
        }
    }
    let _doc = get_snapshot_doc(node, job_id, snapshot_id)?;
    let native = !super::process::snapshot_readable(node, job_id, snapshot_id);
    let state = if native { "failed" } else { "stopped" };
    svc.mutate(|st| {
        st.snapshot_upgrades.insert(format!("{job_id}#{snapshot_id}"), json!({"job_id": job_id, "snapshot_id": snapshot_id, "state": state}));
        Ok(())
    })?;
    if wait {
        if native {
            return Err(EsError::new(500, "elasticsearch_exception", format!("[{job_id}] Unable to upgrade snapshot [{snapshot_id}]: the model state is not readable by this node")));
        }
        return Ok(json!({"completed": true, "node": crate::node::NODE_ID}));
    }
    Ok(json!({"completed": false, "node": crate::node::NODE_ID}))
}

/// `GET model_snapshots/{snapshot}/_upgrade/_stats`.
pub fn snapshot_upgrade_stats(node: &Node, job_expr: &str, snapshot_id: &str, allow_no_match: bool) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_jobs(&st, Some(job_expr), allow_no_match)?;
    let any_snapshot = snapshot_id.is_empty() || snapshot_id == "_all" || snapshot_id.contains('*');
    let mut rows: Vec<Value> = st
        .snapshot_upgrades
        .values()
        .filter(|u| ids.iter().any(|i| u["job_id"] == *i) && (any_snapshot && (snapshot_id.is_empty() || snapshot_id == "_all" || crate::mapping::wildcard_match(snapshot_id, u["snapshot_id"].as_str().unwrap_or(""))) || u["snapshot_id"] == snapshot_id))
        .map(|u| {
            let mut o = Map::new();
            o.insert("job_id".into(), u["job_id"].clone());
            o.insert("snapshot_id".into(), u["snapshot_id"].clone());
            o.insert("state".into(), u["state"].clone());
            if u["state"] != "failed" && u["state"] != "stopped" {
                o.insert("node".into(), json!({"id": crate::node::NODE_ID, "name": node.name, "ephemeral_id": "", "transport_address": "", "attributes": {}}));
            }
            o.insert("assignment_explanation".into(), json!(""));
            Value::Object(o)
        })
        .collect();
    if rows.is_empty() && !any_snapshot && !allow_no_match {
        return Err(EsError::resource_not_found(format!("no snapshot upgrade is running for snapshot_id [{snapshot_id}]")));
    }
    if rows.is_empty() && !any_snapshot {
        return Err(EsError::resource_not_found(format!("no snapshot upgrade is running for snapshot_id [{snapshot_id}]")));
    }
    rows.sort_by(|a, b| a["job_id"].as_str().cmp(&b["job_id"].as_str()).then(a["snapshot_id"].as_str().cmp(&b["snapshot_id"].as_str())));
    Ok(json!({"count": rows.len(), "model_snapshot_upgrades": rows}))
}

// ---------------------------------------------------------------------------
// Stats documents (read by the job stats API for closed jobs)
// ---------------------------------------------------------------------------

/// The latest data counts document of a job (`<job>_data_counts`, or the
/// 5.4 `<job>-data-counts` id).
pub fn stored_data_counts(node: &Node, job_id: &str) -> Option<Value> {
    let body = json!({"size": 1, "query": {"ids": {"values": [format!("{job_id}_data_counts"), format!("{job_id}-data-counts")]}},
        "sort": [{"log_time": {"order": "desc", "unmapped_type": "date"}}, {"latest_record_timestamp": {"order": "desc", "unmapped_type": "date"}}]});
    let resp = search(node, &read_alias(job_id), &body).ok()?;
    hits(&resp).into_iter().next().map(|(_, s)| s)
}

/// The latest model size stats document of a job.
pub fn stored_model_size_stats(node: &Node, job_id: &str) -> Option<Value> {
    let body = json!({"size": 1, "query": {"term": {"result_type": "model_size_stats"}}, "sort": [{"log_time": {"order": "desc", "unmapped_type": "date"}}]});
    let resp = search(node, &read_alias(job_id), &body).ok()?;
    hits(&resp).into_iter().next().map(|(_, s)| s)
}

/// The timing stats document of a job.
pub fn stored_timing_stats(node: &Node, job_id: &str) -> Option<Value> {
    ridx::get(node, &read_alias(job_id), &format!("{job_id}_timing_stats"))
}

/// `forecasts_stats` of a job from its forecast request stats documents.
pub fn forecast_stats(node: &Node, job_id: &str) -> Value {
    let docs = ridx::search_all(node, &read_alias(job_id), json!({"term": {"result_type": "model_forecast_request_stats"}}), json!(["_doc"])).unwrap_or_default();
    if docs.is_empty() {
        return json!({"total": 0, "forecasted_jobs": 0});
    }
    let stat = |k: &str| {
        let vals: Vec<f64> = docs.iter().map(|(_, d)| f64_of(&d[k]).unwrap_or(0.0)).collect();
        let total: f64 = vals.iter().sum();
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        json!({"total": total, "min": min, "avg": total / vals.len() as f64, "max": max})
    };
    let mut status: Map<String, Value> = Map::new();
    for (_, d) in &docs {
        let s = d["forecast_status"].as_str().unwrap_or("unknown").to_string();
        let e = status.entry(s).or_insert(json!(0));
        *e = json!(e.as_i64().unwrap_or(0) + 1);
    }
    json!({"total": docs.len(), "forecasted_jobs": 1, "memory_bytes": stat("forecast_memory_bytes"), "records": stat("processed_record_count"), "processing_time_ms": stat("processing_time_ms"), "status": status})
}

/// `DELETE _forecast[/{forecast_ids}]`.
pub fn delete_forecasts(node: &Node, job_id: &str, forecast_ids: &str, allow_no_forecasts: bool) -> Result<Value> {
    // ES does not check that the job exists: a missing job has no forecasts.
    let all = forecast_ids.is_empty() || forecast_ids == "_all" || forecast_ids == "*";
    let ids: Vec<&str> = forecast_ids.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let mut filters = vec![json!({"term": {"result_type": "model_forecast_request_stats"}})];
    if !all {
        filters.push(json!({"terms": {"forecast_id": ids}}));
    }
    let found = ridx::search_all(node, &read_alias(job_id), json!({"bool": {"filter": filters}}), json!(["_doc"]))?;
    let found_ids: Vec<String> = found.iter().filter_map(|(_, d)| d["forecast_id"].as_str().map(str::to_string)).collect();
    if all {
        if found_ids.is_empty() && !allow_no_forecasts {
            return Err(EsError::resource_not_found(format!("No forecast(s) [{}] exists for job [{job_id}]", if forecast_ids.is_empty() { "_all" } else { forecast_ids })));
        }
    } else {
        // `allow_no_forecasts` only covers `_all` / wildcards: an explicit
        // id that does not exist is always an error.
        let missing: Vec<&str> = ids.iter().copied().filter(|i| !found_ids.iter().any(|f| f == i)).collect();
        if !missing.is_empty() {
            return Err(EsError::resource_not_found(format!("No forecast(s) [{forecast_ids}] exists for job [{job_id}]")));
        }
    }
    for (_, d) in &found {
        if d["forecast_status"].as_str() == Some("started") {
            return Err(EsError::new(409, "status_exception", format!("Forecast(s) [{}] for job [{job_id}] needs to be either FAILED or FINISHED to be deleted", d["forecast_id"].as_str().unwrap_or(""))));
        }
    }
    if !found_ids.is_empty() {
        ridx::delete_by_query(node, &read_alias(job_id), json!({"bool": {"filter": [{"terms": {"forecast_id": found_ids}}]}}));
    }
    Ok(json!({"acknowledged": true}))
}
