//! Data stream lifecycle (`DataStreamLifecycleService`) and the data
//! stream management APIs: `_lifecycle` get/put/delete/explain/stats,
//! `_data_stream/_modify`, `_migrate`, `_promote` and `_stats`.

use super::{cluster_setting, human_time, parse_time, runner};
use crate::datastream::{DataStream, TIMESTAMP_FIELD};
use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub const MANAGED_ILM: &str = "Index Lifecycle Management";
pub const MANAGED_DLM: &str = "Data stream lifecycle";
pub const UNMANAGED: &str = "Unmanaged";

/// Whether a lifecycle JSON (`{"enabled", "data_retention", ...}`) is enabled.
pub fn lifecycle_enabled(lc: &Option<Value>) -> bool {
    match lc {
        Some(v) => v.get("enabled").and_then(Value::as_bool).unwrap_or(true),
        None => false,
    }
}

fn prefer_ilm(svc: &IndexService) -> bool {
    svc.setting("index.lifecycle.prefer_ilm").map(|s| s != "false").unwrap_or(true)
}

/// `DataStream.isIndexManagedByDataStreamLifecycle`.
pub fn managed_by_dlm(node: &Node, svc: &IndexService) -> bool {
    let ds = match node.data_stream_of_index(&svc.name).and_then(|n| node.data_stream(&n)) {
        Some(d) => d,
        None => return false,
    };
    if !lifecycle_enabled(&ds.lifecycle) {
        return false;
    }
    !(runner::policy_name(svc).is_some() && prefer_ilm(svc))
}

/// `managed_by` of a backing index in `GET _data_stream`.
pub fn managed_by(node: &Node, svc: &IndexService) -> &'static str {
    if managed_by_dlm(node, svc) {
        MANAGED_DLM
    } else if runner::policy_name(svc).is_some() {
        MANAGED_ILM
    } else {
        UNMANAGED
    }
}

/// Template-level ILM policy / prefer_ilm for a data stream's next generation.
pub fn template_ilm(node: &Node, ds: &str) -> (Option<String>, bool) {
    let (settings, _, _) = node.template_for(ds);
    let flat = crate::settings::normalize_index_settings(&settings);
    let policy = flat.get("index.lifecycle.name").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    let prefer = flat.get("index.lifecycle.prefer_ilm").map(|v| v.as_str().map(|s| s != "false").unwrap_or(v.as_bool().unwrap_or(true))).unwrap_or(true);
    (policy, prefer)
}

pub fn next_generation_managed_by(node: &Node, ds: &DataStream) -> &'static str {
    let (policy, prefer) = template_ilm(node, &ds.name);
    if lifecycle_enabled(&ds.lifecycle) {
        if policy.is_some() && prefer {
            MANAGED_ILM
        } else {
            MANAGED_DLM
        }
    } else if policy.is_some() {
        MANAGED_ILM
    } else {
        UNMANAGED
    }
}

/// The lifecycle a new data stream gets from its template (component
/// templates first, the index template's own `template.lifecycle` last).
pub fn template_lifecycle(node: &Node, template: &Value) -> Option<Value> {
    let t = node.templates.read();
    let mut out: Option<Map<String, Value>> = None;
    let mut parts = Vec::new();
    for c in template["composed_of"].as_array().cloned().unwrap_or_default() {
        if let Some(ct) = c.as_str().and_then(|n| t.component.get(n)) {
            parts.push(ct["template"]["lifecycle"].clone());
        }
    }
    parts.push(template["template"]["lifecycle"].clone());
    for p in parts {
        if let Some(o) = p.as_object() {
            let m = out.get_or_insert_with(Map::new);
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    out.map(|m| normalize_lifecycle(&Value::Object(m)).unwrap_or(Value::Object(Map::new())))
}

/// Parse a lifecycle body (`PUT _data_stream/{n}/_lifecycle`, template
/// `lifecycle`) into `{"enabled", "data_retention"?, "downsampling"?}`.
pub fn normalize_lifecycle(v: &Value) -> Result<Value> {
    let empty = Map::new();
    let o = match v {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err(EsError::x_content_parse("[1:1] [lifecycle] expected an object")),
    };
    let mut enabled = true;
    let mut retention: Option<String> = None;
    let mut downsampling: Option<Value> = None;
    let mut method: Option<String> = None;
    let mut frozen_after: Option<String> = None;
    for (k, x) in o {
        match k.as_str() {
            "downsampling_method" => {
                if !x.is_null() {
                    let label = x.as_str().ok_or_else(|| EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [downsampling_method]"))?;
                    method = Some(sampling_method(label).map_err(|e| EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [downsampling_method]").caused_by(e))?.to_string());
                }
            }
            "frozen_after" => {
                if !x.is_null() {
                    let s = match x {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let t = parse_time(&s, "frozen_after").map_err(|e| EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [frozen_after]").caused_by(e))?;
                    if t.millis() <= 0 {
                        return Err(EsError::illegal_argument("frozen_after must be a positive time value"));
                    }
                    frozen_after = Some(t.rep);
                }
            }
            "enabled" => {
                enabled = match x {
                    Value::Bool(b) => *b,
                    Value::String(s) if s == "true" || s == "false" => s == "true",
                    _ => return Err(EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [enabled]")),
                }
            }
            "data_retention" => {
                if !x.is_null() {
                    let s = match x {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let t = parse_time(&s, "data_retention").map_err(|e| EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [data_retention]").caused_by(e))?;
                    if t.millis() <= 0 && t.rep != "-1" {
                        return Err(EsError::illegal_argument("Data retention must be greater than 0"));
                    }
                    retention = Some(t.rep);
                }
            }
            "downsampling" => {
                if !x.is_null() {
                    let arr = x.as_array().ok_or_else(|| EsError::x_content_parse("[1:1] [lifecycle] failed to parse field [downsampling]"))?;
                    let mut out = Vec::new();
                    let mut last: Option<(i64, i64)> = None;
                    for r in arr {
                        let after = r["after"].as_str().ok_or_else(|| EsError::illegal_argument("Required [after]"))?;
                        let fi = r["fixed_interval"].as_str().ok_or_else(|| EsError::illegal_argument("Required [fixed_interval]"))?;
                        let a = parse_time(after, "after")?;
                        let f = crate::date::parse_fixed_interval(fi).ok_or_else(|| EsError::illegal_argument(format!("failed to parse value [{fi}] as a fixed interval")))?;
                        if let Some((la, lf)) = last {
                            if a.millis() <= la {
                                return Err(EsError::illegal_argument("A downsampling round must have a later 'after' value than the proceeding, ..."));
                            }
                            if f % lf != 0 || f <= lf {
                                return Err(EsError::illegal_argument(format!(
                                    "Downsampling interval [{fi}] must be a multiple of the source index interval [{}].",
                                    arr[out.len() - 1]["fixed_interval"].as_str().unwrap_or("")
                                )));
                            }
                        }
                        last = Some((a.millis(), f));
                        out.push(json!({"after": a.rep, "fixed_interval": fi}));
                    }
                    if out.len() > 10 {
                        return Err(EsError::illegal_argument("Downsampling configuration supports maximum 10 configured rounds. Found: ".to_string() + &out.len().to_string()));
                    }
                    downsampling = Some(Value::Array(out));
                }
            }
            other => return Err(EsError::x_content_parse(format!("[1:2] [lifecycle] unknown field [{other}]"))),
        }
    }
    let mut m = Map::new();
    m.insert("enabled".into(), json!(enabled));
    if let Some(r) = retention {
        m.insert("data_retention".into(), json!(r));
    }
    if method.is_some() && downsampling.is_none() {
        return Err(EsError::illegal_argument("Downsampling method can only be set when there is at least one downsampling round."));
    }
    if let Some(d) = downsampling {
        m.insert("downsampling".into(), d);
    }
    if let Some(x) = method {
        m.insert("downsampling_method".into(), json!(x));
    }
    if let Some(x) = frozen_after {
        m.insert("frozen_after".into(), json!(x));
    }
    Ok(Value::Object(m))
}

/// `DownsampleConfig.SamplingMethod.fromString` (case-insensitive).
pub fn sampling_method(label: &str) -> Result<&'static str> {
    match label.to_ascii_lowercase().as_str() {
        "aggregate" => Ok("aggregate"),
        "last_value" => Ok("last_value"),
        _ => Err(EsError::illegal_argument(format!("Sampling method [{label}] is not one of the accepted methods [aggregate, last_value]."))),
    }
}

/// Global retention (`data_streams.lifecycle.retention.default|max`).
pub fn global_retention(node: &Node) -> (Option<String>, Option<String>) {
    let g = |k: &str| cluster_setting(node, k).filter(|s| !s.is_empty()).and_then(|s| parse_time(&s, k).ok()).map(|t| t.rep);
    (g("data_streams.lifecycle.retention.default"), g("data_streams.lifecycle.retention.max"))
}

fn millis(rep: &str) -> i64 {
    parse_time(rep, "").map(|t| t.millis()).unwrap_or(0)
}

/// `DataStreamLifecycle.getEffectiveDataRetentionWithSource`.
pub fn effective_retention(node: &Node, ds_name: &str, lc: &Value) -> Option<(String, &'static str)> {
    let own = lc.get("data_retention").and_then(Value::as_str).map(str::to_string);
    if ds_name.starts_with('.') {
        return own.map(|r| (r, "data_stream_configuration"));
    }
    let (default, max) = global_retention(node);
    match own {
        None => match (default, max) {
            (Some(d), _) => Some((d, "default_global_retention")),
            (None, Some(m)) => Some((m, "max_global_retention")),
            _ => None,
        },
        Some(r) => match max {
            Some(m) if millis(&m) < millis(&r) => Some((m, "max_global_retention")),
            _ => Some((r, "data_stream_configuration")),
        },
    }
}

/// `RolloverConfiguration.evaluateMaxAgeCondition`: `max_age=auto`
/// resolved for a retention.
fn auto_max_age(retention: Option<&str>) -> String {
    let day = 86_400_000;
    match retention.map(millis) {
        None => "30d".into(),
        Some(r) if r <= day => "1h".into(),
        Some(r) if r <= 14 * day => "1d".into(),
        Some(r) if r <= 90 * day => "7d".into(),
        Some(_) => "30d".into(),
    }
}

pub fn rollover_defaults(retention: Option<&str>) -> Value {
    json!({"max_age": format!("{} [automatic]", auto_max_age(retention)), "max_primary_shard_docs": 200_000_000, "max_primary_shard_size": "50gb", "min_docs": 1})
}

/// The lifecycle as rendered by the APIs (effective retention, and the
/// rollover configuration with `include_defaults`).
pub fn render_lifecycle(node: &Node, ds_name: &str, lc: &Value, include_defaults: bool) -> Value {
    let mut m = Map::new();
    m.insert("enabled".into(), json!(lc.get("enabled").and_then(Value::as_bool).unwrap_or(true)));
    if let Some(r) = lc.get("data_retention") {
        m.insert("data_retention".into(), r.clone());
    }
    let eff = effective_retention(node, ds_name, lc);
    if let Some((r, src)) = &eff {
        m.insert("effective_retention".into(), json!(r));
        m.insert("retention_determined_by".into(), json!(src));
    }
    for k in ["downsampling", "downsampling_method", "frozen_after"] {
        if let Some(d) = lc.get(k) {
            m.insert(k.into(), d.clone());
        }
    }
    if include_defaults {
        m.insert("rollover".into(), rollover_defaults(eff.as_ref().map(|(r, _)| r.as_str())));
    }
    Value::Object(m)
}

fn global_retention_json(node: &Node) -> Value {
    let (d, m) = global_retention(node);
    let mut o = Map::new();
    if let Some(m) = m {
        o.insert("max_retention".into(), json!(m));
    }
    if let Some(d) = d {
        o.insert("default_retention".into(), json!(d));
    }
    Value::Object(o)
}

/// Resolve a data stream name expression (comma list, wildcards,
/// exclusions); missing concrete names → 404.
pub fn resolve_data_streams(node: &Node, expr: &str, include_hidden: bool) -> Result<Vec<DataStream>> {
    crate::datastream::reject_selectors(expr)?;
    let mut out: Vec<DataStream> = Vec::new();
    let parts: Vec<&str> = if expr.is_empty() { vec!["*"] } else { expr.split(',').map(str::trim).filter(|p| !p.is_empty()).collect() };
    for p in parts {
        if let Some(ex) = p.strip_prefix('-').filter(|_| !out.is_empty()) {
            out.retain(|d| !crate::mapping::wildcard_match(ex, &d.name));
            continue;
        }
        if p == "_all" || p.contains('*') {
            let pat = if p == "_all" { "*" } else { p };
            for d in node.data_streams_matching(pat, include_hidden) {
                if !out.iter().any(|x| x.name == d.name) {
                    out.push(d);
                }
            }
        } else {
            let d = node.data_stream(p).ok_or_else(|| EsError::index_not_found(p))?;
            if !out.iter().any(|x| x.name == d.name) {
                out.push(d);
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// `GET /_data_stream/{name}/_lifecycle`.
pub fn get_lifecycle(node: &Node, expr: &str, include_hidden: bool, include_defaults: bool) -> Result<Value> {
    let dss = resolve_data_streams(node, expr, include_hidden)?;
    let arr: Vec<Value> = dss
        .iter()
        .map(|d| {
            let mut v = json!({"name": d.name});
            if let Some(lc) = &d.lifecycle {
                v["lifecycle"] = render_lifecycle(node, &d.name, lc, include_defaults);
            }
            v
        })
        .collect();
    Ok(json!({"global_retention": global_retention_json(node), "data_streams": arr}))
}

/// `DataStreamLifecycle.DownsamplingRound.validateRounds` over a request's
/// `downsampling` rounds (each `fixed_interval` a multiple of the previous
/// round's, `after` non-decreasing): the request-validation message.
pub fn downsampling_rounds_error(lifecycle: &Value) -> Option<String> {
    let rounds = lifecycle.get("downsampling")?.as_array()?;
    if rounds.is_empty() {
        return Some("Downsampling configuration should have at least one round configured.".into());
    }
    if rounds.len() > 10 {
        return Some(format!("Downsampling configuration supports maximum 10 configured rounds. Found: {}", rounds.len()));
    }
    let mut prev: Option<(i64, &str, i64, &str)> = None;
    for r in rounds {
        let after = r.get("after").and_then(Value::as_str)?;
        let fi = r.get("fixed_interval").and_then(Value::as_str)?;
        let (a, f) = (parse_time(after, "after").ok()?.millis(), crate::date::parse_fixed_interval(fi)?);
        if let Some((pa, pafter, pf, pfi)) = prev {
            if a < pa {
                return Some(format!("A downsampling round must have a later 'after' value than the proceeding, {after} is not after {pafter}."));
            }
            if pf >= f {
                return Some(format!("Downsampling interval [{fi}] must be greater than the source index interval [{pfi}]."));
            }
            if f % pf != 0 {
                return Some(format!("Downsampling interval [{fi}] must be a multiple of the source index interval [{pfi}]."));
            }
        }
        prev = Some((a, after, f, fi));
    }
    None
}

/// `PUT /_data_stream/{name}/_lifecycle` (`None` body = DELETE).
pub fn put_lifecycle(node: &Node, expr: &str, include_hidden: bool, body: Option<&Value>) -> Result<()> {
    if let Some(e) = body.and_then(downsampling_rounds_error) {
        return Err(EsError::action_validation(&[&format!("downsampling rounds are not valid: {e}")]));
    }
    let lc = match body {
        Some(b) => Some(normalize_lifecycle(b)?),
        None => None,
    };
    let dss = resolve_data_streams(node, expr, include_hidden)?;
    {
        let mut streams = node.data_streams.write();
        for d in &dss {
            if let Some(s) = streams.get_mut(&d.name) {
                s.lifecycle = lc.clone();
            }
        }
    }
    node.persist_state()
}

/// `GET /{index}/_lifecycle/explain`.
pub fn explain(node: &Node, expr: &str, opts: ResolveOpts, include_defaults: bool) -> Result<Value> {
    let targets = node.resolve(expr, opts)?;
    let errors = node.ilm.dlm_stats.lock().errors.clone();
    let mut out = BTreeMap::new();
    let now = crate::date::now_millis();
    for t in targets {
        let svc = &t.svc;
        let ds = node.data_stream_of_index(&svc.name).and_then(|n| node.data_stream(&n));
        let managed = managed_by_dlm(node, svc);
        let mut m = Map::new();
        m.insert("index".into(), json!(svc.name));
        m.insert("managed_by_lifecycle".into(), json!(managed));
        if managed {
            let ds = ds.unwrap();
            m.insert("index_creation_date_millis".into(), json!(svc.creation_date));
            m.insert("time_since_index_creation".into(), json!(human_time(now - svc.creation_date)));
            let rolled = runner::rollover_time(node, svc);
            if let Some(r) = rolled {
                m.insert("rollover_date_millis".into(), json!(r));
                m.insert("time_since_rollover".into(), json!(human_time(now - r)));
            }
            if ds.write_index() != svc.name {
                let gen_time = rolled.unwrap_or(svc.creation_date);
                m.insert("generation_time".into(), json!(human_time(now - gen_time)));
            }
            if let Some(lc) = &ds.lifecycle {
                m.insert("lifecycle".into(), render_lifecycle(node, &ds.name, lc, include_defaults));
            }
            if let Some(e) = errors.get(&svc.name) {
                m.insert("error".into(), json!(e));
            }
        }
        out.insert(svc.name.clone(), Value::Object(m));
    }
    Ok(json!({"indices": out}))
}

/// `GET /_lifecycle/stats`.
/// `data_lifecycle` section of `_xpack/usage`.
pub fn usage(node: &Node) -> Value {
    let dss: Vec<DataStream> = node.data_streams.read().values().filter(|d| lifecycle_enabled(&d.lifecycle)).cloned().collect();
    let summary = |ms: &[i64]| -> Map<String, Value> {
        let mut m = Map::new();
        if !ms.is_empty() {
            m.insert("minimum_millis".into(), json!(ms.iter().min()));
            m.insert("maximum_millis".into(), json!(ms.iter().max()));
            m.insert("average_millis".into(), json!(ms.iter().sum::<i64>() / ms.len() as i64));
        }
        m
    };
    let configured: Vec<i64> = dss.iter().filter_map(|d| d.lifecycle.as_ref()?.get("data_retention")?.as_str().map(millis)).collect();
    let effective: Vec<i64> = dss.iter().filter_map(|d| effective_retention(node, &d.name, d.lifecycle.as_ref()?).map(|(r, _)| millis(&r))).collect();
    let mut data_retention = summary(&configured);
    data_retention.insert("configured_data_streams".into(), json!(configured.len()));
    let frozen: Vec<i64> = dss.iter().filter_map(|d| d.lifecycle.as_ref()?.get("frozen_after")?.as_str().map(millis)).collect();
    let mut frozen_after = summary(&frozen);
    frozen_after.insert("configured_data_streams".into(), json!(frozen.len()));
    let mut eff = summary(&effective);
    eff.insert("retained_data_streams".into(), json!(effective.len()));
    let (default, max) = global_retention(node);
    let g = |v: Option<String>| match v {
        Some(r) => json!({"defined": true, "millis": millis(&r)}),
        None => json!({"defined": false}),
    };
    json!({
        "available": true, "enabled": true, "count": dss.len(), "default_rollover_used": true,
        "data_retention": data_retention, "effective_retention": eff, "frozen_after": frozen_after,
        "global_retention": {"max": g(max), "default": g(default)}
    })
}

pub fn stats(node: &Node) -> Value {
    let st = node.ilm.dlm_stats.lock().clone();
    let mut m = Map::new();
    if let Some(d) = st.last_run_duration {
        m.insert("last_run_duration_in_millis".into(), json!(d));
        m.insert("last_run_duration".into(), json!(human_time(d)));
    }
    if let Some(d) = st.time_between_starts {
        m.insert("time_between_starts_in_millis".into(), json!(d));
        m.insert("time_between_starts".into(), json!(human_time(d)));
    }
    let dss: Vec<DataStream> = node.data_streams.read().values().filter(|d| lifecycle_enabled(&d.lifecycle)).cloned().collect();
    m.insert("data_stream_count".into(), json!(dss.len()));
    let arr: Vec<Value> = dss
        .iter()
        .map(|d| {
            let managed: Vec<&String> = d.indices.iter().filter(|i| node.get_index(i).map(|s| managed_by_dlm(node, &s)).unwrap_or(false)).collect();
            let in_err = managed.iter().filter(|i| st.errors.contains_key(i.as_str())).count();
            json!({"name": d.name, "backing_indices_in_total": managed.len(), "backing_indices_in_error": in_err})
        })
        .collect();
    m.insert("data_streams".into(), Value::Array(arr));
    Value::Object(m)
}

/// One data stream lifecycle run: rollover of managed write indices and
/// retention of managed backing indices.
pub fn run(node: &Node) {
    let start = crate::date::now_millis();
    {
        let mut s = node.ilm.dlm_stats.lock();
        if let Some(prev) = s.last_run_start {
            s.time_between_starts = Some(start - prev);
        }
        s.last_run_start = Some(start);
    }
    let mut errors: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let dss: Vec<DataStream> = node.data_streams.read().values().filter(|d| lifecycle_enabled(&d.lifecycle)).cloned().collect();
    for ds in dss {
        let lc = ds.lifecycle.clone().unwrap_or(Value::Null);
        let eff = effective_retention(node, &ds.name, &lc);
        // rollover
        if let Some(w) = node.get_index(ds.write_index()) {
            if managed_by_dlm(node, &w) {
                let conds = json!({
                    "max_age": auto_max_age(eff.as_ref().map(|(r, _)| r.as_str())),
                    "max_primary_shard_size": "50gb",
                    "max_primary_shard_docs": 200_000_000,
                    "min_docs": 1
                });
                if let Err(e) = node.rollover(&ds.name, None, &json!({"conditions": conds}), false) {
                    errors.insert(w.name.clone(), super::error_json(&e).to_string());
                }
            }
        }
        // retention
        let retention = match &eff {
            Some((r, _)) => millis(r),
            None => continue,
        };
        let cur = match node.data_stream(&ds.name) {
            Some(d) => d,
            None => continue,
        };
        let now = crate::date::now_millis();
        for idx in cur.indices.iter().filter(|i| i.as_str() != cur.write_index()) {
            let svc = match node.get_index(idx) {
                Some(s) => s,
                None => continue,
            };
            if !managed_by_dlm(node, &svc) {
                continue;
            }
            let gen_time = runner::rollover_time(node, &svc).unwrap_or(svc.creation_date);
            if now - gen_time >= retention {
                drop(svc);
                if let Err(e) = node.delete_index(idx) {
                    errors.insert(idx.clone(), super::error_json(&e).to_string());
                }
            }
        }
    }
    // Failure stores: their own lifecycle (enabled by default once the
    // failure store is), rolled over and retained like backing indices.
    let all: Vec<DataStream> = node.data_streams.read().values().filter(|d| !d.failure_indices.is_empty()).cloned().collect();
    for ds in all {
        let Some(flc) = node.failure_store_lifecycle(&ds) else { continue };
        let rendered = node.render_failures_lifecycle(&flc, false);
        if rendered["enabled"].as_bool() != Some(true) {
            continue;
        }
        let eff = rendered["effective_retention"].as_str().map(str::to_string);
        if ds.failure_write_index().is_some() && !ds.failure_rollover_on_write {
            let conds = json!({
                "max_age": auto_max_age(eff.as_deref()),
                "max_primary_shard_size": "50gb",
                "max_primary_shard_docs": 200_000_000,
                "min_docs": 1
            });
            if let Err(e) = node.rollover(&format!("{}::failures", ds.name), None, &json!({"conditions": conds}), false) {
                errors.insert(ds.failure_write_index().unwrap_or_default().to_string(), super::error_json(&e).to_string());
            }
        }
        let Some(retention) = eff.as_deref().map(millis) else { continue };
        let Some(cur) = node.data_stream(&ds.name) else { continue };
        let now = crate::date::now_millis();
        for idx in cur.failure_indices.iter().filter(|i| Some(i.as_str()) != cur.failure_write_index()) {
            let Some(svc) = node.get_index(idx) else { continue };
            let gen_time = runner::rollover_time(node, &svc).unwrap_or(svc.creation_date);
            if now - gen_time >= retention {
                drop(svc);
                if let Err(e) = node.delete_index(idx) {
                    errors.insert(idx.clone(), super::error_json(&e).to_string());
                }
            }
        }
    }
    let mut s = node.ilm.dlm_stats.lock();
    s.errors = errors;
    s.last_run_duration = Some(crate::date::now_millis() - start);
}

/// Swap `old` for `new` in a data stream's backing indices (ILM shrink).
pub fn replace_backing_index(node: &Node, ds: &str, old: &str, new: &str) -> Result<()> {
    {
        let mut streams = node.data_streams.write();
        let d = streams.get_mut(ds).ok_or_else(|| EsError::illegal_argument(format!("data stream [{ds}] not found")))?;
        if d.write_index() == old {
            return Err(EsError::illegal_argument(format!(
                "cannot replace backing index [{old}] of data stream [{ds}] with [{new}] because it is the write index"
            )));
        }
        for i in d.indices.iter_mut() {
            if i == old {
                *i = new.to_string();
            }
        }
        d.generation += 1;
    }
    if let Some(svc) = node.get_index(new) {
        svc.put_settings(crate::settings::normalize_index_settings(&json!({"index.hidden": "true"})), false)?;
    }
    node.persist_state()
}

fn has_timestamp_mapping(svc: &IndexService) -> bool {
    svc.lookup().get(TIMESTAMP_FIELD).map(|f| matches!(f.declared.name(), "date" | "date_nanos")).unwrap_or(false)
}

/// `POST /_data_stream/_modify`.
pub fn modify(node: &Node, body: &Value) -> Result<()> {
    let actions = body["actions"].as_array().ok_or_else(|| EsError::action_validation(&["must specify at least one data stream modification action"]))?;
    if actions.is_empty() {
        return Err(EsError::action_validation(&["must specify at least one data stream modification action"]));
    }
    let mut streams = node.data_streams.read().clone();
    let mut hide: Vec<String> = Vec::new();
    let mut unhide: Vec<String> = Vec::new();
    // `delete_backing_index`: removed from the stream, then deleted.
    let mut delete: Vec<String> = Vec::new();
    for a in actions {
        let (op, args) = a.as_object().and_then(|o| o.iter().next()).ok_or_else(|| EsError::x_content_parse("[1:1] [data_stream_action] expected an object"))?;
        let ds_name = args["data_stream"].as_str().ok_or_else(|| EsError::illegal_argument("Required [data_stream]"))?.to_string();
        let index = args["index"].as_str().ok_or_else(|| EsError::illegal_argument("Required [index]"))?.to_string();
        let failure_store = args["failure_store"].as_bool().unwrap_or(false);
        if !streams.contains_key(&ds_name) {
            return Err(EsError::illegal_argument(format!("data stream [{ds_name}] not found")));
        }
        let svc = node.get_index(&index).ok_or_else(|| EsError::illegal_argument(format!("index [{index}] not found")))?;
        match op.as_str() {
            "add_backing_index" => {
                if let Some(parent) = streams.values().find(|d| d.failure_indices.contains(&index)).map(|d| d.name.clone()) {
                    if !(failure_store && parent == ds_name) {
                        return Err(EsError::illegal_argument(format!(
                            "cannot add index [{index}] to data stream [{ds_name}] because it is already a failure store index on data stream [{parent}]"
                        )));
                    }
                    continue;
                }
                if let Some(parent) = streams.values().find(|d| d.indices.contains(&index)).map(|d| d.name.clone()) {
                    if parent != ds_name || failure_store {
                        return Err(EsError::illegal_argument(format!(
                            "cannot add index [{index}] to data stream [{ds_name}] because it is already a backing index on data stream [{parent}]"
                        )));
                    }
                    continue;
                }
                let aliases: Vec<String> = svc.aliases().keys().cloned().collect();
                if !aliases.is_empty() {
                    return Err(EsError::illegal_argument(format!(
                        "cannot add index [{index}] to data stream [{ds_name}] until its {} [{}] {} removed",
                        if aliases.len() > 1 { "aliases" } else { "alias" },
                        aliases.join(","),
                        if aliases.len() > 1 { "are" } else { "is" }
                    )));
                }
                if !has_timestamp_mapping(&svc) {
                    return Err(EsError::illegal_argument("unable to prepare backing index")
                        .caused_by(EsError::illegal_argument(format!("backing index [{index}] must have mappings for a timestamp field"))));
                }
                let d = streams.get_mut(&ds_name).unwrap();
                if failure_store {
                    d.failure_indices.insert(0, index.clone());
                } else {
                    d.indices.insert(0, index.clone());
                }
                d.generation += 1;
                hide.push(index);
            }
            "remove_backing_index" | "delete_backing_index" if failure_store => {
                let d = streams.get_mut(&ds_name).unwrap();
                if !d.failure_indices.contains(&index) {
                    return Err(EsError::illegal_argument(format!("index [{index}] not found")));
                }
                if d.failure_write_index() == Some(index.as_str()) {
                    d.failure_rollover_on_write = true;
                }
                d.failure_indices.retain(|i| *i != index);
                d.generation += 1;
                if op == "delete_backing_index" {
                    delete.push(index);
                } else {
                    unhide.push(index);
                }
            }
            "remove_backing_index" | "delete_backing_index" => {
                let d = streams.get_mut(&ds_name).unwrap();
                if !d.indices.contains(&index) {
                    return Err(EsError::illegal_argument(format!("index [{index}] not found")));
                }
                if d.write_index() == index {
                    return Err(EsError::illegal_argument(format!(
                        "cannot remove backing index [{index}] of data stream [{ds_name}] because it is the write index"
                    )));
                }
                d.indices.retain(|i| *i != index);
                d.generation += 1;
                if op == "delete_backing_index" {
                    delete.push(index);
                } else {
                    unhide.push(index);
                }
            }
            other => return Err(EsError::x_content_parse(format!("[1:2] [data_stream_action] unknown field [{other}]"))),
        }
    }
    *node.data_streams.write() = streams;
    for i in hide {
        if let Some(s) = node.get_index(&i) {
            s.put_settings(crate::settings::normalize_index_settings(&json!({"index.hidden": "true"})), false)?;
        }
    }
    for i in unhide {
        if let Some(s) = node.get_index(&i) {
            s.put_settings(crate::settings::normalize_index_settings(&json!({"index.hidden": "false"})), false)?;
        }
    }
    node.persist_state()?;
    for i in delete {
        node.delete_index(&i)?;
    }
    Ok(())
}

/// `POST /_data_stream/_migrate/{alias}`.
pub fn migrate(node: &Node, alias: &str) -> Result<()> {
    let members: Vec<(std::sync::Arc<IndexService>, Value)> =
        node.all_indices().into_iter().filter_map(|s| s.aliases().get(alias).cloned().map(|d| (s.clone(), d))).collect();
    if members.is_empty() {
        return Err(EsError::illegal_argument(format!("alias [{alias}] does not exist")));
    }
    let write = members
        .iter()
        .find(|(_, d)| d.get("is_write_index").and_then(Value::as_bool) == Some(true))
        .or_else(|| if members.len() == 1 && members[0].1.get("is_write_index").and_then(Value::as_bool) != Some(false) { members.first() } else { None })
        .map(|(s, _)| s.name.clone())
        .ok_or_else(|| EsError::illegal_argument(format!("alias [{alias}] must specify a write index")))?;
    if members.iter().any(|(_, d)| ["filter", "index_routing", "search_routing", "routing"].iter().any(|k| d.get(*k).map(|v| !v.is_null()).unwrap_or(false))) {
        return Err(EsError::illegal_argument(format!("alias [{alias}] may not have custom filtering or routing")));
    }
    let others: Vec<String> = members.iter().filter(|(s, _)| s.aliases().len() > 1).map(|(s, _)| s.name.clone()).collect();
    if !others.is_empty() {
        return Err(EsError::illegal_argument(format!("other aliases referencing indices [{}] must be removed before migrating to a data stream", others.join(","))));
    }
    let (tname, tmpl) = node.data_stream_template(alias).ok_or_else(|| {
        let has_any = node.templates.read().index.values().any(|v| crate::node::pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, alias)));
        if has_any {
            EsError::illegal_argument(format!("matching index template for data stream [{alias}] has no data stream template"))
        } else {
            EsError::illegal_argument(format!("no matching index template found for data stream [{alias}]"))
        }
    })?;
    for (s, _) in &members {
        if !has_timestamp_mapping(s) {
            return Err(EsError::illegal_argument(format!("backing index [{}] must have mappings for a timestamp field", s.name)));
        }
    }
    let mut indices: Vec<String> = members.iter().map(|(s, _)| s.name.clone()).filter(|n| *n != write).collect();
    indices.sort();
    indices.push(write.clone());
    for (s, _) in &members {
        let mut a = s.aliases();
        a.remove(alias);
        s.set_aliases(a)?;
        s.put_settings(crate::settings::normalize_index_settings(&json!({"index.hidden": "true"})), false)?;
    }
    let lifecycle = template_lifecycle(node, &tmpl);
    let ds = DataStream {
        name: alias.to_string(),
        generation: indices.len() as u64,
        indices,
        template: tname,
        hidden: tmpl["data_stream"]["hidden"].as_bool().unwrap_or(false),
        allow_custom_routing: tmpl["data_stream"]["allow_custom_routing"].as_bool().unwrap_or(false),
        lifecycle,
        meta: tmpl.get("_meta").cloned(),
        replicated: false,
        failure_indices: Vec::new(),
        rollover_on_write: false,
        failure_rollover_on_write: false,
        options: node.template_data_stream_options(&tmpl).filter(|o| o.as_object().map(|m| !m.is_empty()).unwrap_or(false)),
        settings: Map::new(),
        mappings: Map::new(),
        index_mode: None,
    };
    node.data_streams.write().insert(alias.to_string(), ds);
    node.persist_state()
}

/// `POST /_data_stream/{name}/_promote`.
pub fn promote(node: &Node, name: &str) -> Result<()> {
    {
        let mut streams = node.data_streams.write();
        let d = streams.get_mut(name).ok_or_else(|| EsError::resource_not_found(format!("data stream [{name}] does not exist")))?;
        d.replicated = false;
    }
    node.persist_state()
}

fn max_timestamp(node: &Node, indices: &[String]) -> i64 {
    let targets: Vec<crate::search::Target> =
        indices.iter().filter_map(|i| node.get_index(i)).filter(|s| s.is_open()).map(crate::search::Target::new).collect();
    if targets.is_empty() {
        return 0;
    }
    let body = json!({"size": 0, "track_total_hits": false, "aggs": {"m": {"max": {"field": TIMESTAMP_FIELD}}}});
    let req = match crate::search::parse_request(&body, &std::collections::HashMap::new()) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
    match crate::search::execute(&targets, &req, &opts) {
        Ok(out) => serde_json::from_str::<Value>(&out.json).ok().and_then(|v| v["aggregations"]["m"]["value"].as_f64()).map(|f| f as i64).unwrap_or(0),
        Err(_) => 0,
    }
}

/// `GET /_data_stream[/{name}]/_stats`.
pub fn ds_stats(node: &Node, expr: &str, include_hidden: bool) -> Result<Value> {
    let dss = resolve_data_streams(node, expr, include_hidden)?;
    let mut total_shards = 0usize;
    let mut ok_shards = 0usize;
    let mut backing = 0usize;
    let mut total_bytes = 0u64;
    let mut arr = Vec::new();
    for d in &dss {
        let mut bytes = 0u64;
        let mut n = 0usize;
        for i in &d.indices {
            if let Some(s) = node.get_index(i) {
                n += 1;
                bytes += s.store_size();
                let p = s.number_of_shards() as usize;
                let copies = 1 + s.setting_usize("index.number_of_replicas", 1);
                total_shards += p * copies;
                // Stats come from every started copy: primaries plus the
                // replicas the emulated data nodes can hold.
                ok_shards += p * (copies - node.unassigned_replicas(&s));
            }
        }
        backing += n;
        total_bytes += bytes;
        arr.push(json!({"data_stream": d.name, "backing_indices": n, "store_size_bytes": bytes, "maximum_timestamp": max_timestamp(node, &d.indices)}));
    }
    // ES aggregates per data stream in a `HashMap` and lists its entries.
    let by_name: serde_json::Map<String, Value> = arr.into_iter().map(|e| (e["data_stream"].as_str().unwrap_or_default().to_string(), e)).collect();
    let arr: Vec<Value> = crate::jsonpos::java_hashmap_order(&by_name).into_iter().map(|(_, v)| v.clone()).collect();
    Ok(json!({
        "_shards": {"total": total_shards, "successful": ok_shards, "failed": 0},
        "data_stream_count": dss.len(),
        "backing_indices": backing,
        "total_store_size_bytes": total_bytes,
        "data_streams": arr
    }))
}
