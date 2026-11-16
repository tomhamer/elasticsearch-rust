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
    // `DataStreamGlobalRetentionSettings`: `-1` (as unset) is no retention,
    // never a zero one.
    let g = |k: &str| cluster_setting(node, k).filter(|s| !s.is_empty()).and_then(|s| parse_time(&s, k).ok()).filter(|t| t.nanos >= 0).map(|t| t.rep);
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

/// `DataStream.getGenerationLifecycleDate` of a non-write backing (or
/// failure) index: `index.lifecycle.origination_date` when set, else the
/// time it was rolled over, else its creation date. Retention counts from
/// it.
pub fn generation_time(node: &Node, svc: &IndexService) -> i64 {
    let origination = svc.setting("index.lifecycle.origination_date").and_then(|s| s.trim().parse::<i64>().ok()).filter(|v| *v >= 0);
    origination.or_else(|| runner::rollover_time(node, svc)).unwrap_or(svc.creation_date)
}

/// `GET /{index}/_lifecycle/explain`.
pub fn explain(node: &Node, expr: &str, opts: ResolveOpts, include_defaults: bool) -> Result<Value> {
    let targets = node.resolve(expr, opts)?;
    let errors = {
        let st = node.ilm.dlm_stats.lock();
        let mut e = st.errors.clone();
        e.extend(st.frozen_errors.iter().map(|(k, v)| (k.clone(), v.clone())));
        e
    };
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
                let gen_time = generation_time(node, svc);
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
            let in_err = managed.iter().filter(|i| st.errors.contains_key(i.as_str()) || st.frozen_errors.contains_key(i.as_str())).count();
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
            if managed_by_dlm(node, &w) && !lifecycle_skipped(&w) {
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
        if let (Some((r, _)), Some(cur)) = (&eff, node.data_stream(&ds.name)) {
            let retention = millis(r);
            let now = crate::date::now_millis();
            for idx in cur.indices.iter().filter(|i| i.as_str() != cur.write_index()) {
                let svc = match node.get_index(idx) {
                    Some(s) => s,
                    None => continue,
                };
                if !managed_by_dlm(node, &svc) || lifecycle_skipped(&svc) {
                    continue;
                }
                let gen_time = generation_time(node, &svc);
                if now - gen_time >= retention {
                    // A frozen index the conversion mounted goes with its
                    // backing snapshot (`deleteIndexOnce`).
                    let backing = super::frozen::backing_snapshot(&svc);
                    drop(svc);
                    match node.delete_index(idx) {
                        Ok(()) => super::frozen::delete_backing_snapshot(node, backing),
                        Err(e) => {
                            errors.insert(idx.clone(), super::error_json(&e).to_string());
                        }
                    }
                }
            }
        }
        // downsampling (after retention, as ES)
        if let Some(cur) = node.data_stream(&ds.name) {
            for idx in cur.indices.iter().filter(|i| i.as_str() != cur.write_index()) {
                if let Err(e) = downsample_backing_index(node, &cur, idx) {
                    errors.insert(idx.clone(), super::error_json(&e).to_string());
                }
            }
        }
        // frozen_after candidates, marked for the conversion (`ilm::frozen`):
        // not those this run leaves out (lifecycle skip, time series bounds,
        // a downsampling still due).
        if let Some(cur) = node.data_stream(&ds.name) {
            let now = crate::date::now_millis();
            let excluded = |svc: &IndexService| {
                lifecycle_skipped(svc) || within_time_bounds(svc, now) || downsampling_pending(node, &cur, svc, now)
            };
            for (idx, e) in super::frozen::mark_candidates(node, &cur, &excluded) {
                errors.insert(idx, super::error_json(&e).to_string());
            }
        }
    }
    // Failure stores: their own lifecycle (enabled by default once the
    // failure store is), rolled over and retained like backing indices.
    let all: Vec<DataStream> = node.data_streams.read().values().filter(|d| !d.failure_indices.is_empty()).cloned().collect();
    for ds in all {
        let Some(flc) = node.failure_store_lifecycle(&ds) else { continue };
        let rendered = node.render_failures_lifecycle(&ds.name, &flc, false);
        if rendered["enabled"].as_bool() != Some(true) {
            continue;
        }
        let eff = rendered["effective_retention"].as_str().map(str::to_string);
        let failure_write_skipped = ds.failure_write_index().and_then(|w| node.get_index(w)).is_some_and(|w| lifecycle_skipped(&w));
        if ds.failure_write_index().is_some() && !ds.failure_rollover_on_write && !failure_write_skipped {
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
            if lifecycle_skipped(&svc) {
                continue;
            }
            let gen_time = generation_time(node, &svc);
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

/// `DataStream.getDownsamplingRoundsFor`: the rounds of the stream's
/// lifecycle whose `after` has passed since the index's generation time,
/// in order.
fn due_downsampling_rounds(node: &Node, ds: &DataStream, svc: &IndexService, now: i64) -> Vec<Value> {
    let Some(lc) = ds.lifecycle.as_ref().filter(|_| lifecycle_enabled(&ds.lifecycle)) else { return Vec::new() };
    let Some(rounds) = lc.get("downsampling").and_then(Value::as_array) else { return Vec::new() };
    let generation = generation_time(node, svc);
    rounds.iter().filter(|r| r["after"].as_str().is_some_and(|a| now >= generation.saturating_add(millis(a)))).cloned().collect()
}

/// `DownsampleConfig.generateDownsampleIndexName` with the data stream
/// lifecycle's `downsample-` prefix: named after the original source
/// when `svc` is itself a downsample.
pub fn downsample_index_name(svc: &IndexService, fixed_interval: &str) -> String {
    let source = svc.setting("index.downsample.source.name").filter(|s| !s.is_empty()).unwrap_or_else(|| svc.name.clone());
    format!("downsample-{fixed_interval}-{source}")
}

/// A time series index whose `index.time_series.end_time` is still ahead
/// (`timeSeriesIndicesStillWithinTimeBounds`: not downsampled yet).
fn within_time_bounds(svc: &IndexService, now: i64) -> bool {
    let Some(end) = svc.setting(crate::tsdb::END_TIME) else { return false };
    let f = match crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis") {
        Ok(f) => f,
        Err(_) => return true,
    };
    f.parse(&end).map(|e| e > now).unwrap_or(true)
}

/// `index.lifecycle.skip`: a held index is left out of every DLM phase
/// (retention, downsampling, freezing; `DataStreamLifecycleService.run`'s
/// exclusion seed), backing and failure indices alike.
pub fn lifecycle_skipped(svc: &IndexService) -> bool {
    svc.setting("index.lifecycle.skip").as_deref() == Some("true")
}

/// A downsampling round is due for `svc` and not applied yet (the index is
/// left to the downsampling, not frozen, in this run).
fn downsampling_pending(node: &Node, ds: &DataStream, svc: &IndexService, now: i64) -> bool {
    if !svc.is_time_series() {
        return false;
    }
    let rounds = due_downsampling_rounds(node, ds, svc, now);
    let Some(last) = rounds.last() else { return false };
    let Some(want) = last["fixed_interval"].as_str().and_then(crate::date::parse_fixed_interval) else { return false };
    svc.setting("index.downsample.interval").and_then(|i| crate::date::parse_fixed_interval(&i)).is_none_or(|cur| cur < want)
}

/// Data stream lifecycle downsampling of one non-write backing index
/// (`DataStreamLifecycleService.maybeExecuteDownsampling`), one transition
/// per run: block writes to an original index; then downsample it with the
/// last round that is due into `downsample-{interval}-{source}`; once that
/// index is complete, put it in the source's place (carrying the source's
/// generation time as its `index.lifecycle.origination_date`, so retention
/// still counts from it) and delete the source.
pub fn downsample_backing_index(node: &Node, ds: &DataStream, index: &str) -> Result<()> {
    let Some(svc) = node.get_index(index) else { return Ok(()) };
    if !managed_by_dlm(node, &svc) || !svc.is_time_series() || lifecycle_skipped(&svc) {
        return Ok(());
    }
    let now = crate::date::now_millis();
    if within_time_bounds(&svc, now) {
        return Ok(());
    }
    let rounds = due_downsampling_rounds(node, ds, &svc, now);
    let Some(last) = rounds.last() else { return Ok(()) };
    let is_downsample = svc.setting("index.downsample.source.name").is_some_and(|s| !s.is_empty());
    if !is_downsample && svc.setting("index.blocks.write").as_deref() != Some("true") {
        let mut block = crate::settings::FlatSettings::new();
        block.insert("index.blocks.write".into(), json!("true"));
        return svc.put_settings(block, false);
    }
    let fixed_interval = last["fixed_interval"].as_str().unwrap_or_default();
    let target = downsample_index_name(&svc, fixed_interval);
    if target == svc.name {
        return Ok(());
    }
    // A round already applied (this index is that round's downsample, or a
    // coarser one): nothing to do.
    if let Some(cur) = svc.setting("index.downsample.interval").and_then(|i| crate::date::parse_fixed_interval(&i)) {
        if crate::date::parse_fixed_interval(fixed_interval).is_none_or(|f| f <= cur) {
            return Ok(());
        }
    }
    let mut body = json!({"fixed_interval": fixed_interval});
    if let Some(m) = ds.lifecycle.as_ref().and_then(|l| l.get("downsampling_method")).and_then(Value::as_str) {
        body["sampling_method"] = json!(m);
    }
    match node.ilm.cluster_steps() {
        // The cluster's `_downsample` reads every source shard copy; the
        // swap waits until it completed and the target is visible here.
        Some(steps) => match steps.downsample(&svc.name, &target, &body) {
            super::StepProgress::Done => {}
            super::StepProgress::Pending(_) => return Ok(()),
            super::StepProgress::Failed(reason) => return Err(EsError::new(500, "illegal_state_exception", reason)),
        },
        None => {
            let complete = node.get_index(&target).is_some_and(|t| {
                t.setting("index.downsample.source.uuid").as_deref() == Some(svc.uuid.as_str()) && t.setting("index.downsample.status").as_deref() == Some("success")
            });
            if !complete {
                // `_downsample` rebuilds a target an interrupted run left
                // unfinished, and refuses one another run is filling.
                crate::migrate::downsample(node, &svc.name, &target, &body)?;
            }
        }
    }
    let Some(tsvc) = node.get_index(&target) else { return Ok(()) };
    if tsvc.setting("index.downsample.status").as_deref() != Some("success") {
        return Ok(());
    }
    // The target must be this source incarnation's downsample at this
    // round's interval (a completed run reported earlier may have had its
    // target replaced since, by a restore of another downsample under the
    // name): never swapped in, and the source never deleted, otherwise.
    let interval_ok = tsvc.setting("index.downsample.interval").as_deref() == Some(fixed_interval);
    if tsvc.setting("index.downsample.source.uuid").as_deref() != Some(svc.uuid.as_str()) || !interval_ok {
        return Err(EsError::new(
            500,
            "illegal_state_exception",
            format!(
                "downsample target [{target}] ([{}]) is not the [{fixed_interval}] downsample of [{}] ([{}]): the source is kept",
                tsvc.uuid, svc.name, svc.uuid
            ),
        ));
    }
    // `ReplaceSourceWithDownsampleIndexTask`.
    if tsvc.setting("index.lifecycle.origination_date").is_none() {
        let mut origin = crate::settings::FlatSettings::new();
        origin.insert("index.lifecycle.origination_date".into(), json!(generation_time(node, &svc).to_string()));
        tsvc.put_settings(origin, false)?;
    }
    if !ds.indices.iter().any(|i| *i == target) {
        replace_backing_index(node, &ds.name, &svc.name, &target)?;
    }
    drop(svc);
    node.delete_index(index)
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
    // Only an index alias converts: a data stream alias (its members are
    // backing indices) would make a second stream owning the same indices,
    // whose lifecycle could then delete the other stream's data.
    let owned: Vec<String> = members.iter().filter(|(s, _)| node.data_stream_of_index(&s.name).is_some()).map(|(s, _)| s.name.clone()).collect();
    if !owned.is_empty() || node.data_stream(alias).is_some() {
        return Err(EsError::illegal_argument(format!(
            "alias [{alias}] is a data stream alias: indices [{}] already belong to a data stream; only an index alias can be migrated to a data stream",
            owned.join(",")
        )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrate::{DownsampleOps, LocalDownsampleOps};

    const DAY: i64 = 86_400_000;

    fn set(svc: &IndexService, key: &str, value: Value) {
        let mut flat = crate::settings::FlatSettings::new();
        flat.insert(key.into(), value);
        svc.put_settings(flat, false).unwrap();
    }

    /// Retention counts from `index.lifecycle.origination_date` before the
    /// rollover time and the creation date (`getGenerationLifecycleDate`):
    /// an index rolled over 30 days ago but originated 2 days ago is kept
    /// under 7-day retention until 7 days after its origination.
    #[test]
    fn retention_counts_from_the_origination_date() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        node.templates.write().index.insert(
            "ret".into(),
            json!({"index_patterns": ["ret-*"], "data_stream": {}, "template": {"settings": {"index.number_of_replicas": "0"}, "lifecycle": {"data_retention": "7d"}}}),
        );
        let ds = node.create_data_stream("ret-a").unwrap();
        let first = ds.write_index().to_string();
        node.rollover("ret-a", None, &json!({}), false).unwrap();
        let svc = node.get_index(&first).unwrap();
        let now = crate::date::now_millis();
        // Created now, rolled over 30 days ago, originated 2 days ago.
        let mut lc = runner::load(&node, &svc);
        assert!(!lc.rollover_info.is_empty(), "the rollover is recorded");
        for v in lc.rollover_info.values_mut() {
            v["time"] = json!(now - 30 * DAY);
        }
        runner::save(&node, &svc, &lc).unwrap();
        assert_eq!(runner::rollover_time(&node, &svc), Some(now - 30 * DAY));
        // Without an origination date the rollover time decides.
        assert_eq!(generation_time(&node, &svc), now - 30 * DAY);
        set(&svc, "index.lifecycle.origination_date", json!((now - 2 * DAY).to_string()));
        assert_eq!(generation_time(&node, &svc), now - 2 * DAY);
        run(&node);
        assert!(node.get_index(&first).is_some(), "deleted five days before its retention boundary");
        let v = explain(&node, &first, ResolveOpts::default(), false).unwrap();
        assert_eq!(v["indices"][first.as_str()]["generation_time"], "2d", "{v}");
        // Just inside the boundary: kept; just past it: deleted.
        set(&svc, "index.lifecycle.origination_date", json!((now - 7 * DAY + 60_000).to_string()));
        run(&node);
        assert!(node.get_index(&first).is_some(), "deleted before its retention boundary");
        set(&svc, "index.lifecycle.origination_date", json!((now - 7 * DAY - 60_000).to_string()));
        drop(svc);
        run(&node);
        assert!(node.get_index(&first).is_none(), "kept past its retention boundary");
    }

    fn aged_backing_index(node: &Node, pattern: &str, stream: &str, lifecycle: Value, age: i64) -> String {
        node.templates.write().index.insert(
            pattern.into(),
            json!({"index_patterns": [format!("{pattern}-*")], "data_stream": {}, "template": {"settings": {"index.number_of_replicas": "0"}, "lifecycle": lifecycle}}),
        );
        let first = node.create_data_stream(stream).unwrap().write_index().to_string();
        node.rollover(stream, None, &json!({}), false).unwrap();
        set(&node.get_index(&first).unwrap(), "index.lifecycle.origination_date", json!((crate::date::now_millis() - age).to_string()));
        first
    }

    /// Review finding 551: a global default or max retention of `-1` is no
    /// retention (`DataStreamGlobalRetentionSettings`), never zero: a
    /// freshly rolled-over generation is kept; a real cap still applies.
    #[test]
    fn global_retention_minus_one_is_unset_not_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        let explicit = aged_backing_index(&node, "gr", "gr-a", json!({"data_retention": "30d"}), 60_000);
        let inherited = aged_backing_index(&node, "gi", "gi-a", json!({}), 60_000);
        node.cluster_settings.write().0.insert("data_streams.lifecycle.retention.max".into(), json!("-1"));
        node.cluster_settings.write().0.insert("data_streams.lifecycle.retention.default".into(), json!("-1"));
        assert_eq!(global_retention(&node), (None, None));
        assert_eq!(effective_retention(&node, "gr-a", &json!({"data_retention": "30d"})), Some(("30d".into(), "data_stream_configuration")));
        assert_eq!(effective_retention(&node, "gi-a", &json!({})), None);
        run(&node);
        assert!(node.get_index(&explicit).is_some() && node.get_index(&inherited).is_some(), "-1 deleted fresh generations");
        // A real cap still deletes past it.
        node.cluster_settings.write().0.insert("data_streams.lifecycle.retention.max".into(), json!("10s"));
        run(&node);
        assert!(node.get_index(&explicit).is_none() && node.get_index(&inherited).is_none());
    }

    /// Review finding 552: an internal (dot-prefixed) stream's failure
    /// store keeps its own retention under a shorter global max; a public
    /// one is capped.
    #[test]
    fn internal_failure_store_retention_ignores_the_global_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.cluster_settings.write().0.insert("data_streams.lifecycle.retention.max".into(), json!("7d"));
        let lc = json!({"data_retention": "30d"});
        let internal = node.render_failures_lifecycle(".diagnostics", &lc, false);
        assert_eq!((internal["effective_retention"].as_str(), internal["retention_determined_by"].as_str()), (Some("30d"), Some("data_stream_configuration")), "{internal}");
        let public = node.render_failures_lifecycle("diagnostics", &lc, false);
        assert_eq!((public["effective_retention"].as_str(), public["retention_determined_by"].as_str()), (Some("7d"), Some("max_global_retention")), "{public}");
        let none = node.render_failures_lifecycle(".diagnostics", &json!({}), false);
        assert!(none.get("effective_retention").is_none(), "{none}");
    }

    /// Review finding 553: `index.lifecycle.skip` holds an index out of
    /// DLM retention (and rollover) across repeated runs; once the hold is
    /// removed retention resumes.
    #[test]
    fn lifecycle_skip_holds_an_index_out_of_dlm() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        let first = aged_backing_index(&node, "sk", "sk-a", json!({"data_retention": "1d"}), 3 * DAY);
        let svc = node.get_index(&first).unwrap();
        set(&svc, "index.lifecycle.skip", json!("true"));
        assert!(managed_by_dlm(&node, &svc) && lifecycle_skipped(&svc));
        drop(svc);
        for _ in 0..3 {
            run(&node);
            assert!(node.get_index(&first).is_some(), "a held index was deleted");
        }
        // A held write index is not rolled over.
        let write = node.data_stream("sk-a").unwrap().write_index().to_string();
        let w = node.get_index(&write).unwrap();
        w.write(vec![crate::index::WriteReq::Index { id: Some("1".into()), source: json!({"@timestamp": "2026-01-01T00:00:00Z"}).to_string(), routing: None, cond: Default::default() }], true).into_iter().for_each(|r| { r.unwrap(); });
        set(&w, "index.lifecycle.skip", json!("true"));
        set(&w, "index.lifecycle.origination_date", json!((crate::date::now_millis() - 40 * DAY).to_string()));
        drop(w);
        let generations = node.data_stream("sk-a").unwrap().indices.len();
        run(&node);
        assert_eq!(node.data_stream("sk-a").unwrap().indices.len(), generations, "a held write index rolled over");
        set(&node.get_index(&first).unwrap(), "index.lifecycle.skip", json!("false"));
        run(&node);
        assert!(node.get_index(&first).is_none(), "retention did not resume after the hold");
    }

    /// Data stream lifecycle downsampling: once a round is due for a
    /// backing index past its time bounds, the index is write-blocked, then
    /// downsampled into `downsample-{interval}-{index}`, which takes its
    /// place in the stream (with the source's generation time as its
    /// origination date) while the source is deleted. The downsampled
    /// documents keep the metrics.
    #[test]
    fn lifecycle_downsampling_replaces_the_backing_index() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        node.templates.write().index.insert(
            "metrics".into(),
            json!({"index_patterns": ["metrics-*"], "data_stream": {}, "template": {
                "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": "0"},
                "mappings": {"properties": {
                    "@timestamp": {"type": "date"},
                    "host": {"type": "keyword", "time_series_dimension": true},
                    "cpu": {"type": "double", "time_series_metric": "gauge"}}},
                "lifecycle": {"downsampling": [{"after": "1d", "fixed_interval": "1h"}]}}}),
        );
        node.cluster_settings.write().0.insert("data_stream.past_tsdb_index_creation_enabled".into(), json!("true"));
        node.create_data_stream("metrics-a").unwrap();
        let now = crate::date::now_millis();
        // A past backing index (ending before the write index's window, so
        // past its time bounds) holding one hour of documents.
        let hour = now - now.rem_euclid(3_600_000) - 5 * 3_600_000;
        let svc = node.create_past_time_series_service("metrics-a", hour).unwrap();
        let first = svc.name.clone();
        assert!(!within_time_bounds(&svc, now), "{:?}", svc.setting(crate::tsdb::END_TIME));
        let docs: Vec<Value> = (0..6)
            .map(|i| json!({"@timestamp": iso(hour + i * 60_000), "host": if i % 2 == 0 { "a" } else { "b" }, "cpu": (i + 1) as f64}))
            .collect();
        LocalDownsampleOps(&node).index_docs(&first, docs).unwrap();
        svc.refresh().unwrap();
        let mut ds = node.data_stream("metrics-a").unwrap();
        assert_ne!(ds.write_index(), first);
        // Not due yet (generation time now, round after 1d): untouched.
        run(&node);
        assert_eq!(svc.setting("index.blocks.write"), None);
        set(&svc, "index.lifecycle.origination_date", json!((now - 2 * DAY).to_string()));
        // The write index is never downsampled.
        let write = ds.write_index().to_string();
        downsample_backing_index(&node, &ds, &write).unwrap();
        assert_eq!(node.get_index(&write).unwrap().setting("index.blocks.write"), None);
        // First run: the write block; second: downsample and swap.
        downsample_backing_index(&node, &ds, &first).unwrap();
        assert_eq!(svc.setting("index.blocks.write").as_deref(), Some("true"));
        let target = format!("downsample-1h-{first}");
        assert!(node.get_index(&target).is_none());
        ds = node.data_stream("metrics-a").unwrap();
        // A run that died before marking its target successful left a
        // `started` target: never swapped in, the source never deleted.
        let crash = crate::migrate::downsample_safety_tests::CrashAfterFirstBatch {
            inner: LocalDownsampleOps(&node),
            batches: std::cell::Cell::new(0),
            before_success: true,
        };
        assert!(crate::migrate::downsample_with(&node, &first, &target, &json!({"fixed_interval": "1h"}), &crash).is_err());
        assert_eq!(node.get_index(&target).unwrap().setting(crate::settings::DOWNSAMPLE_STATUS).as_deref(), Some("started"));
        {
            // Another run still holds it: the lifecycle waits.
            let _held = crate::migrate::InFlight::claim(&node, &target).unwrap();
            assert!(downsample_backing_index(&node, &ds, &first).is_err());
            assert!(node.get_index(&first).is_some(), "the source is kept while the target is partial");
            assert_eq!(node.data_stream("metrics-a").unwrap().indices[0], first);
        }
        drop(svc);
        downsample_backing_index(&node, &ds, &first).unwrap();
        assert!(node.get_index(&first).is_none(), "the source is deleted");
        let tsvc = node.get_index(&target).expect("downsampled index");
        let ds = node.data_stream("metrics-a").unwrap();
        assert_eq!(ds.indices[0], target, "the downsample took the source's place: {:?}", ds.indices);
        assert_eq!(tsvc.setting("index.downsample.interval").as_deref(), Some("1h"));
        assert_eq!(tsvc.setting("index.lifecycle.origination_date"), Some((now - 2 * DAY).to_string()));
        assert!(managed_by_dlm(&node, &tsvc));
        let mut out = Vec::new();
        LocalDownsampleOps(&node).scan(&target, &mut |src| out.push(src.clone())).unwrap();
        out.sort_by_key(|d| d["host"].as_str().unwrap_or_default().to_string());
        assert_eq!(out.len(), 2, "one bucket per host: {out:?}");
        let metric = |d: &Value| ["min", "max", "sum", "value_count"].map(|k| d["cpu"][k].as_f64().unwrap_or(f64::NAN));
        assert_eq!(metric(&out[0]), [1.0, 5.0, 9.0, 3.0], "{out:?}");
        assert_eq!(metric(&out[1]), [2.0, 6.0, 12.0, 3.0], "{out:?}");
        // The round is applied: further runs leave the downsample alone.
        run(&node);
        assert!(node.get_index(&target).is_some());
        assert_eq!(node.data_stream("metrics-a").unwrap().indices[0], target);
    }

    fn iso(ms: i64) -> String {
        super::super::iso_date(ms)
    }

    /// Review findings 360 / 362: data stream, alias and index names stay
    /// unique: a data stream is not created over an index alias (nor an
    /// alias added under a stream's name), and a data stream's alias is
    /// not migrated into a second stream owning its backing indices; an
    /// ordinary index alias still migrates.
    #[test]
    fn names_and_backing_indices_stay_owned_once() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.templates.write().index.insert(
            "logs".into(),
            json!({"index_patterns": ["logs*", "migr*"], "data_stream": {}, "template": {"settings": {"index.number_of_replicas": "0"},
                "mappings": {"properties": {"@timestamp": {"type": "date"}}}}}),
        );
        node.create_index("old", &json!({"settings": {"index.number_of_replicas": 0}, "mappings": {"properties": {"@timestamp": {"type": "date"}}}})).unwrap();
        node.update_aliases(&[json!({"add": {"index": "old", "alias": "logs"}})]).unwrap();
        let e = node.create_data_stream("logs").unwrap_err();
        assert!(e.reason.contains("names need to be unique"), "{}", e.reason);
        assert!(node.data_stream("logs").is_none());
        assert!(node.get_index("old").unwrap().aliases().contains_key("logs"), "the alias keeps its index");
        // An alias named like an existing stream.
        node.create_data_stream("logs-x").unwrap();
        let e = node.update_aliases(&[json!({"add": {"index": "old", "alias": "logs-x"}})]).unwrap_err();
        assert!(e.reason.contains("Invalid alias name"), "{}", e.reason);
        // A data stream alias does not migrate.
        node.update_aliases(&[json!({"add": {"index": "logs-x", "alias": "migr-a"}})]).unwrap();
        let backing = node.data_stream("logs-x").unwrap().indices.clone();
        let e = migrate(&node, "migr-a").unwrap_err();
        assert!(e.reason.contains("data stream alias"), "{}", e.reason);
        assert!(node.data_stream("migr-a").is_none());
        assert_eq!(node.data_stream("logs-x").unwrap().indices, backing);
        // An ordinary index alias does.
        node.create_index("plain", &json!({"settings": {"index.number_of_replicas": 0}, "mappings": {"properties": {"@timestamp": {"type": "date"}}}})).unwrap();
        node.update_aliases(&[json!({"add": {"index": "plain", "alias": "migr-b", "is_write_index": true}})]).unwrap();
        migrate(&node, "migr-b").unwrap();
        assert_eq!(node.data_stream("migr-b").unwrap().indices, vec!["plain".to_string()]);
    }

    /// Review finding 338: the swap verifies the target's provenance at
    /// the final step (however its completion was learned): a successful
    /// downsample under the target name that is not this source
    /// incarnation's downsample at this round's interval is never swapped
    /// in, and the source is kept. The ordinary downsample is the control.
    #[test]
    fn a_replaced_downsample_target_is_never_swapped_in() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        node.templates.write().index.insert(
            "metrics-r-tmpl".into(),
            json!({"index_patterns": ["metrics-r*"], "data_stream": {}, "template": {
                "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": "0"},
                "mappings": {"properties": {
                    "@timestamp": {"type": "date"},
                    "host": {"type": "keyword", "time_series_dimension": true},
                    "cpu": {"type": "double", "time_series_metric": "gauge"}}},
                "lifecycle": {"downsampling": [{"after": "1d", "fixed_interval": "1h"}]}}}),
        );
        node.cluster_settings.write().0.insert("data_stream.past_tsdb_index_creation_enabled".into(), json!("true"));
        node.create_data_stream("metrics-r").unwrap();
        let now = crate::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 5 * 3_600_000;
        let svc = node.create_past_time_series_service("metrics-r", hour).unwrap();
        let first = svc.name.clone();
        let docs: Vec<Value> = (0..4).map(|i| json!({"@timestamp": iso(hour + i * 60_000), "host": "a", "cpu": i as f64})).collect();
        LocalDownsampleOps(&node).index_docs(&first, docs).unwrap();
        svc.refresh().unwrap();
        set(&svc, "index.lifecycle.origination_date", json!((now - 2 * DAY).to_string()));
        let ds = node.data_stream("metrics-r").unwrap();
        downsample_backing_index(&node, &ds, &first).unwrap();
        let target = format!("downsample-1h-{first}");
        // A successful downsample of this source, but at another interval,
        // now holds the target name.
        crate::migrate::downsample(&node, &first, &target, &json!({"fixed_interval": "30m"})).unwrap();
        assert_eq!(node.get_index(&target).unwrap().setting(crate::settings::DOWNSAMPLE_STATUS).as_deref(), Some("success"));
        let e = downsample_backing_index(&node, &ds, &first).unwrap_err();
        assert!(e.reason.contains("is not the [1h] downsample of"), "{}", e.reason);
        assert!(node.get_index(&first).is_some(), "the source is kept");
        assert_eq!(node.data_stream("metrics-r").unwrap().indices[0], first);
        // Control: the round's own downsample swaps in.
        node.delete_index(&target).unwrap();
        drop(svc);
        downsample_backing_index(&node, &ds, &first).unwrap();
        assert!(node.get_index(&first).is_none());
        assert_eq!(node.data_stream("metrics-r").unwrap().indices[0], target);
    }
}
