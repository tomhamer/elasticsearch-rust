//! `_ml/datafeeds[/{id}]/_preview`: the data a datafeed would send to its
//! job — source documents (scroll datafeeds) or flattened aggregation
//! buckets (`AggregationToJsonProcessor`).

use super::anomaly::parse_datafeed_time;
use super::store::service;
use super::xc::{bad_request, Pos};
use super::{datafeed, job};
use crate::error::{EsError, Result};
use crate::node::Node;
use serde_json::{json, Map, Value};

/// Preview a stored datafeed (`id`) or an inline `datafeed_config` /
/// `job_config` body.
pub fn preview(node: &Node, id: Option<&str>, body: &Value, start: Option<&str>, end: Option<&str>) -> Result<Value> {
    let st = service(node).read();
    let dfb = body.get("datafeed_config").filter(|v| !v.is_null()).cloned();
    let jb = body.get("job_config").filter(|v| !v.is_null()).cloned();
    let (df, job) = match id {
        Some(i) => {
            if dfb.is_some() || jb.is_some() {
                return Err(bad_request("Params [datafeed_id] and [datafeed_config|job_config] are mutually exclusive"));
            }
            let df = st.datafeeds.get(i).cloned().ok_or_else(|| EsError::resource_not_found(format!("No datafeed with id [{i}] exists")))?;
            let jid = df["job_id"].as_str().unwrap_or("").to_string();
            let job = st.jobs.get(&jid).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{jid}'")))?;
            (df, job)
        }
        None => {
            let nested = jb.as_ref().and_then(|j| j.get("datafeed_config")).filter(|v| !v.is_null()).cloned();
            let d = match (dfb, nested) {
                (Some(_), Some(_)) => return Err(bad_request("[datafeed_config] must not be present when a [job_config.datafeed_config] is present")),
                (Some(d), None) | (None, Some(d)) => d,
                (None, None) => return Err(bad_request("[datafeed_config] must be present when a [job_config.datafeed_config] is not present")),
            };
            let job = match &jb {
                Some(j) => {
                    let mut j = j.clone();
                    if let Some(o) = j.as_object_mut() {
                        o.remove("datafeed_config");
                        o.remove("job_id");
                    }
                    let raw = j.to_string();
                    let nj = job::parse_put("preview", &j, &raw)?;
                    job::finish("preview", &nj, job::DEFAULT_MML_MB, 0)
                }
                None => {
                    let jid = d.get("job_id").and_then(Value::as_str).ok_or_else(|| bad_request("[job_id] must be set in the datafeed_config when no job_config is supplied"))?;
                    st.jobs.get(jid).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{jid}'")))?
                }
            };
            let mut o = d.as_object().cloned().unwrap_or_default();
            o.insert("job_id".into(), json!(job["job_id"].as_str().unwrap_or("preview")));
            o.remove("datafeed_id");
            (datafeed::build("preview-datafeed", &o, Pos::empty(), &Map::new())?, job)
        }
    };
    drop(st);
    let start = start.map(|s| parse_datafeed_time(s, "start")).transpose()?;
    let end = end.map(|s| parse_datafeed_time(s, "end")).transpose()?;
    run(node, &df, &job, start, end)
}

fn analysis_fields(job: &Value) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut add = |f: &str| {
        if f != "mlcategory" && !fields.iter().any(|x| x == f) {
            fields.push(f.to_string());
        }
    };
    for d in job["analysis_config"]["detectors"].as_array().into_iter().flatten() {
        for k in ["field_name", "by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(f) = d[k].as_str() {
                add(f);
            }
        }
    }
    for k in ["summary_count_field_name", "categorization_field_name"] {
        if let Some(f) = job["analysis_config"][k].as_str() {
            add(f);
        }
    }
    for i in job["analysis_config"]["influencers"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        add(i);
    }
    fields
}

fn run(node: &Node, df: &Value, job: &Value, start: Option<i64>, end: Option<i64>) -> Result<Value> {
    fetch(node, df, job, start, end, 1000)
}

/// The documents a datafeed sends for `[start, end)` (time field as epoch
/// millis), at most `size` of them in time order. Used by the preview and
/// the datafeed runner.
pub fn fetch(node: &Node, df: &Value, job: &Value, start: Option<i64>, end: Option<i64>, size: usize) -> Result<Value> {
    let time_field = job["data_description"]["time_field"].as_str().unwrap_or("time").to_string();
    let indices: Vec<String> = df["indices"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
    // A concrete missing index is a 404 (ES fails the preview search).
    for i in &indices {
        if !i.contains('*') && node.resolve(i, crate::node::ResolveOpts { allow_no_indices: true, ..Default::default() }).map(|t| t.is_empty()).unwrap_or(true) {
            return Err(EsError::index_not_found(i));
        }
    }
    let mut range = Map::new();
    if let Some(s) = start {
        range.insert("gte".into(), json!(s));
    }
    if let Some(e) = end {
        range.insert("lt".into(), json!(e));
    }
    range.insert("format".into(), json!("epoch_millis"));
    let query = json!({"bool": {"filter": [df["query"].clone(), {"range": {time_field.clone(): range}}]}});
    let target = indices.join(",");
    if let Some(aggs) = df.get("aggregations").filter(|a| a.is_object()) {
        let mut search = json!({"size": 0, "query": query, "aggs": aggs});
        if let Some(rm) = df.get("runtime_mappings") {
            search["runtime_mappings"] = rm.clone();
        }
        let resp = super::search(node, &target, &search)?;
        let summary = job["analysis_config"]["summary_count_field_name"].as_str().map(str::to_string);
        let include_doc_count = summary.as_deref() == Some("doc_count");
        let mut out = Vec::new();
        for (name, agg) in resp["aggregations"].as_object().into_iter().flatten() {
            let _ = name;
            for b in agg["buckets"].as_array().into_iter().flatten() {
                if b["doc_count"].as_i64().unwrap_or(0) == 0 {
                    continue;
                }
                let mut base = Map::new();
                let t = b.get(&time_field).and_then(|m| m["value"].as_f64()).or_else(|| b["key"].as_f64()).unwrap_or(0.0);
                base.insert(time_field.clone(), json!(t as i64));
                flatten_bucket(b, base, &time_field, include_doc_count, &mut out);
            }
        }
        return Ok(Value::Array(out));
    }
    let fields = analysis_fields(job);
    let mut all = vec![time_field.clone()];
    all.extend(fields.iter().filter(|f| **f != time_field).cloned());
    let mut search = json!({"size": size, "query": query, "sort": [{time_field.clone(): {"order": "asc"}}], "_source": false,
        "fields": all.iter().map(|f| if *f == time_field { json!({"field": f, "format": "epoch_millis"}) } else { json!({"field": f}) }).collect::<Vec<_>>()});
    if let Some(rm) = df.get("runtime_mappings") {
        search["runtime_mappings"] = rm.clone();
    }
    if let Some(sf) = df.get("script_fields") {
        search["script_fields"] = sf.clone();
    }
    let resp = super::search(node, &target, &search)?;
    let mut out = Vec::new();
    for h in resp["hits"]["hits"].as_array().into_iter().flatten() {
        let mut doc = Map::new();
        for f in &all {
            let Some(arr) = h["fields"].get(f).and_then(Value::as_array) else { continue };
            let v = if *f == time_field {
                let first = arr.first().cloned().unwrap_or(Value::Null);
                match &first {
                    Value::String(s) => s.split('.').next().and_then(|x| x.parse::<i64>().ok()).map(|x| json!(x)).unwrap_or(first),
                    other => other.clone(),
                }
            } else if arr.len() == 1 {
                arr[0].clone()
            } else {
                Value::Array(arr.clone())
            };
            doc.insert(f.clone(), v);
        }
        out.push(Value::Object(doc));
    }
    Ok(Value::Array(out))
}

fn flatten_bucket(b: &Value, base: Map<String, Value>, time_field: &str, include_doc_count: bool, out: &mut Vec<Value>) {
    let mut doc = base;
    let mut multi: Vec<(&String, &Value)> = Vec::new();
    for (k, v) in b.as_object().into_iter().flatten() {
        if matches!(k.as_str(), "key" | "key_as_string" | "doc_count" | "from" | "to" | "from_as_string" | "to_as_string") || k == time_field {
            continue;
        }
        if v.get("buckets").is_some() {
            multi.push((k, v));
        } else if let Some(val) = v.get("value") {
            if !val.is_null() {
                doc.insert(k.clone(), val.clone());
            }
        }
    }
    if multi.is_empty() {
        if include_doc_count {
            doc.insert("doc_count".into(), b["doc_count"].clone());
        }
        out.push(Value::Object(doc));
        return;
    }
    for (name, agg) in multi {
        for sb in agg["buckets"].as_array().into_iter().flatten() {
            if sb["doc_count"].as_i64().unwrap_or(0) == 0 {
                continue;
            }
            let mut d = doc.clone();
            d.insert(name.clone(), sb["key"].clone());
            flatten_bucket(sb, d, time_field, include_doc_count, out);
        }
    }
}
