//! `POST _ml/anomaly_detectors/_estimate_model_memory`
//! (`TransportEstimateModelMemoryAction`).

use super::job;
use super::xc::{self, bad_request, unknown, wrap, xparse};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

const KB: u64 = 1024;
const MB: u64 = 1024 * 1024;
const BASIC_REQUIREMENT: u64 = 10 * MB;
const BYTES_PER_INFLUENCER_VALUE: u64 = 10 * KB;
const PARSER: &str = "cluster:admin/xpack/ml/job/estimate_model_memory";

fn cardinalities(v: &Value, p: &xc::Pos, field: &str) -> Result<HashMap<String, u64>> {
    let mut out = HashMap::new();
    let o = v.as_object().ok_or_else(|| xparse(format!("{} [{PARSER}] {field} doesn't support values of type: {}", xc::fmt(p.start), xc::token_name(v))))?;
    for (k, c) in o {
        let n = c.as_i64().unwrap_or(0);
        if n < 0 {
            return Err(wrap(p.get(k).start, PARSER, field, bad_request(format!("[{field}] contained negative cardinality [{n}]"))));
        }
        out.insert(k.clone(), n as u64);
    }
    Ok(out)
}

fn estimate(card: &HashMap<String, u64>, kind: &str, desc: &str, field: &str) -> Result<u64> {
    if let Some(c) = card.get(field) {
        return Ok(*c);
    }
    if field == "mlcategory" {
        return Ok(if kind == "Overall" { 500 } else { 50 });
    }
    Err(EsError::illegal_argument(format!("[{kind}] cardinality estimate required for [{desc}] [{field}] but not supplied")))
}

fn reduced(cardinality: u64, partition_cardinality: u64, bucket_secs: i64) -> f64 {
    let power = ((bucket_secs.max(1) as f64).log10() + 1.0).min(8.0) / 8.0;
    let power = power.min(1.0);
    cardinality as f64 / (cardinality.min(partition_cardinality) as f64).max(1.0).powf(power)
}

fn mul(a: u64, b: u64) -> u64 {
    a.saturating_mul(b)
}

fn detector_bytes(d: &Value, bucket_secs: i64, overall: &HashMap<String, u64>) -> Result<u64> {
    let f = d["function"].as_str().unwrap_or("metric");
    let mut workspace = false;
    let mut answer: u64 = match f {
        "distinct_count" | "low_distinct_count" | "high_distinct_count" => {
            workspace = true;
            32 * KB
        }
        "count" | "low_count" | "high_count" | "non_zero_count" | "low_non_zero_count" | "high_non_zero_count" => 32 * KB,
        "rare" | "freq_rare" => 2 * KB,
        "info_content" | "low_info_content" | "high_info_content" => {
            workspace = true;
            48 * KB
        }
        "metric" => 120 * KB,
        "median" | "low_median" | "high_median" => 64 * KB,
        "time_of_day" | "time_of_week" => 10 * KB,
        "lat_long" => 64 * KB,
        _ => 48 * KB,
    };
    let partition = d["partition_field_name"].as_str();
    let pcard = match partition {
        Some(p) => estimate(overall, "Overall", "partition_field_name", p)?,
        None => 1,
    };
    if let Some(by) = d["by_field_name"].as_str() {
        let c = estimate(overall, "Overall", "by_field_name", by)?;
        let m = (reduced(c, pcard, bucket_secs) * 2.0 / 3.0).ceil() as u64;
        answer = mul(answer, m);
    }
    if let Some(over) = d["over_field_name"].as_str() {
        let c = estimate(overall, "Overall", "over_field_name", over)?;
        let m = reduced(c, pcard, bucket_secs).ceil() as u64;
        answer = answer.saturating_add(mul(m, 768));
    }
    if partition.is_some() {
        answer = mul(answer, pcard);
    }
    if workspace {
        answer = answer.saturating_add(5 * MB);
    }
    Ok(answer)
}

/// The estimate in bytes for a rendered analysis config.
pub fn estimate_bytes(ac: &Value, overall: &HashMap<String, u64>, max_bucket: &HashMap<String, u64>) -> Result<u64> {
    let bucket_secs = ac["bucket_span"].as_str().and_then(|s| job::parse_time(s, "bucket_span").ok()).map(|t| t.seconds()).unwrap_or(300);
    let detectors = ac["detectors"].as_array().cloned().unwrap_or_default();
    let mut total = BASIC_REQUIREMENT;
    for d in &detectors {
        total = total.saturating_add(detector_bytes(d, bucket_secs, overall)?);
    }
    // Influencers that are not also by/over/partition fields.
    let mut analysis_fields: Vec<&str> = Vec::new();
    for d in &detectors {
        for k in ["by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(s) = d[k].as_str() {
                analysis_fields.push(s);
            }
        }
    }
    let mut infl = 0u64;
    for i in ac["influencers"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        if analysis_fields.contains(&i) {
            continue;
        }
        infl = infl.saturating_add(estimate(max_bucket, "Bucket max", "influencers", i)?);
    }
    total = total.saturating_add(mul(infl, BYTES_PER_INFLUENCER_VALUE));
    // Categorization.
    if ac.get("categorization_field_name").and_then(Value::as_str).is_some() {
        let ppc = &ac["per_partition_categorization"];
        let enabled = ppc["enabled"].as_bool().unwrap_or(false);
        let stop_on_warn = ppc["stop_on_warn"].as_bool().unwrap_or(false);
        let mut partitions = 1u64;
        if enabled {
            if let Some(p) = detectors.iter().find_map(|d| d["partition_field_name"].as_str()) {
                partitions = overall.get(p).copied().unwrap_or(1).max(1);
            }
        }
        let per = if enabled && stop_on_warn { 20 * MB } else { 40 * MB };
        total = total.saturating_add(mul(per, partitions));
    }
    Ok(total)
}

/// `POST _ml/anomaly_detectors/_estimate_model_memory`.
pub fn estimate_model_memory(body: &Value, raw: &str) -> Result<Value> {
    let pos = xc::positions(raw);
    let empty = Map::new();
    let o = body.as_object().unwrap_or(&empty);
    let mut ac = None;
    let mut overall = HashMap::new();
    let mut max_bucket = HashMap::new();
    for (k, v) in o {
        match k.as_str() {
            "analysis_config" => {
                ac = Some(job::parse_analysis_config_only(v, pos.get(k)).map_err(wrap_ac)?);
            }
            "overall_cardinality" => overall = cardinalities(v, pos.get(k), k)?,
            "max_bucket_cardinality" => max_bucket = cardinalities(v, pos.get(k), k)?,
            other => return Err(unknown(pos.key(other), PARSER, other, &["analysis_config", "overall_cardinality", "max_bucket_cardinality"])),
        }
    }
    let Some(ac) = ac else {
        return Err(EsError::action_validation(&["[analysis_config] was not specified"]));
    };
    let bytes = estimate_bytes(&ac, &overall, &max_bucket)?;
    let mb = bytes.saturating_add(MB - 1) / MB;
    Ok(json!({"model_memory_estimate": format!("{mb}mb")}))
}

fn wrap_ac(pe: job::PErr) -> EsError {
    wrap(pe.at, PARSER, "analysis_config", pe.e)
}
