//! Per-index search usage that `_stats` reports: `stats` groups on searches
//! and the text `fielddata` fields that sorts / aggregations loaded. Keyed by
//! index uuid so independent nodes in one process never share state.

use elasticsearch_engine::search::Target;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};


#[derive(Default, Clone)]
struct Usage {
    /// group -> (query_total, fetch_total)
    groups: BTreeMap<String, (u64, u64)>,
    fielddata: BTreeSet<String>,
}

fn state() -> &'static Mutex<HashMap<String, Usage>> {
    static S: OnceLock<Mutex<HashMap<String, Usage>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn sort_fields(sort: &Value, out: &mut Vec<String>) {
    match sort {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| sort_fields(x, out)),
        Value::Object(o) => out.extend(o.keys().cloned()),
        _ => {}
    }
}

fn agg_fields(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                if k == "field" {
                    if let Some(s) = x.as_str() {
                        out.push(s.to_string());
                    }
                } else {
                    agg_fields(x, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| agg_fields(x, out)),
        _ => {}
    }
}

/// Note what a completed search used.
pub fn record(targets: &[Target], body: &Value) {
    let groups: Vec<String> = body["stats"].as_array().map(|a| a.iter().filter_map(|g| g.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut fields = Vec::new();
    // Keyword fields an aggregation touched build global ordinals (reported as fielddata).
    let mut agg_f = Vec::new();
    if let Some(s) = body.get("sort") {
        sort_fields(s, &mut fields);
    }
    for k in ["aggs", "aggregations"] {
        if let Some(a) = body.get(k) {
            agg_fields(a, &mut fields);
            agg_fields(a, &mut agg_f);
        }
    }
    if groups.is_empty() && fields.is_empty() {
        return;
    }
    let mut st = state().lock().unwrap_or_else(|e| e.into_inner());
    for t in targets {
        let u = st.entry(t.svc.uuid.clone()).or_default();
        for g in &groups {
            let e = u.groups.entry(g.clone()).or_default();
            e.0 += 1;
            e.1 += 1;
        }
        let lookup = t.svc.lookup();
        for f in &fields {
            if lookup.get(f).is_some_and(|rf| rf.fielddata || (agg_f.contains(f) && rf.ty.is_keyword_like())) {
                u.fielddata.insert(f.clone());
            }
        }
    }
}

pub fn groups(uuid: &str) -> BTreeMap<String, (u64, u64)> {
    state().lock().unwrap_or_else(|e| e.into_inner()).get(uuid).map(|u| u.groups.clone()).unwrap_or_default()
}

pub fn fielddata_fields(uuid: &str) -> BTreeSet<String> {
    state().lock().unwrap_or_else(|e| e.into_inner()).get(uuid).map(|u| u.fielddata.clone()).unwrap_or_default()
}

// ---------------------------------------------------------------- _xpack/usage

/// ES 8.19's `_xpack/usage` sections on a default (basic-licensed) node, with
/// zeroed counters; live sections are merged over it.
static XPACK_TEMPLATE: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| serde_json::from_str(include_str!("data/xpack_usage.json")).unwrap_or(Value::Null));

/// Merge `src` into `dst` (objects recursively, `src` wins elsewhere).
fn merge(dst: &mut Value, src: Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                match d.get_mut(&k) {
                    Some(e) => merge(e, v),
                    None => {
                        d.insert(k, v);
                    }
                }
            }
        }
        (d, s) => *d = s,
    }
}

#[derive(Default)]
struct HealthUsage {
    total: u64,
    verbose_true: u64,
    verbose_false: u64,
    statuses: BTreeMap<String, u64>,
    indicators: BTreeMap<String, BTreeMap<String, u64>>,
    diagnoses: BTreeMap<String, BTreeMap<String, u64>>,
}

fn health_usage() -> &'static Mutex<HealthUsage> {
    static S: OnceLock<Mutex<HealthUsage>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Record one `_health_report` call (`_xpack/usage` `health_api`).
pub fn record_health(verbose: bool, status: Option<&str>, indicators: &serde_json::Map<String, Value>) {
    let mut u = health_usage().lock().unwrap_or_else(|e| e.into_inner());
    u.total += 1;
    if verbose {
        u.verbose_true += 1;
    } else {
        u.verbose_false += 1;
    }
    if let Some(st) = status {
        *u.statuses.entry(st.to_string()).or_default() += 1;
    }
    for (name, ind) in indicators {
        let st = ind["status"].as_str().unwrap_or("green").to_string();
        if st == "green" {
            continue;
        }
        *u.indicators.entry(st.clone()).or_default().entry(name.clone()).or_default() += 1;
        if let Some(ds) = ind["diagnosis"].as_array() {
            for d in ds {
                if let Some(id) = d["id"].as_str() {
                    *u.diagnoses.entry(st.clone()).or_default().entry(id.to_string()).or_default() += 1;
                }
            }
        }
    }
}

fn health_usage_json() -> Value {
    let u = health_usage().lock().unwrap_or_else(|e| e.into_inner());
    let mut v = serde_json::json!({"available": true, "enabled": true});
    if !u.statuses.is_empty() {
        let mut m = serde_json::Map::new();
        m.insert("values".into(), serde_json::json!(u.statuses.keys().collect::<Vec<_>>()));
        for (k, n) in &u.statuses {
            m.insert(k.clone(), serde_json::json!(n));
        }
        v["statuses"] = Value::Object(m);
    }
    let grouped = |g: &BTreeMap<String, BTreeMap<String, u64>>| -> Value {
        let mut out = serde_json::Map::new();
        for (st, by) in g {
            let mut m = serde_json::Map::new();
            for (k, n) in by {
                m.insert(k.clone(), serde_json::json!(n));
            }
            m.insert("values".into(), serde_json::json!(by.keys().collect::<Vec<_>>()));
            out.insert(st.clone(), Value::Object(m));
        }
        Value::Object(out)
    };
    if !u.indicators.is_empty() {
        v["indicators"] = grouped(&u.indicators);
    }
    if !u.diagnoses.is_empty() {
        v["diagnoses"] = grouped(&u.diagnoses);
    }
    let mut inv = serde_json::json!({"total": u.total});
    if u.verbose_true > 0 {
        inv["verbose_true"] = serde_json::json!(u.verbose_true);
    }
    if u.verbose_false > 0 {
        inv["verbose_false"] = serde_json::json!(u.verbose_false);
    }
    v["invocations"] = inv;
    v
}

/// `GET /_xpack/usage`.
/// `analytics` usage: aggregations of the analytics plugin used in search
/// requests (ES counts every parsed occurrence), per node.
const ANALYTICS_AGGS: &[&str] = &["boxplot", "cumulative_cardinality", "string_stats", "top_metrics", "t_test", "moving_percentiles", "normalize", "rate", "multi_terms"];

fn analytics_usage() -> &'static Mutex<HashMap<std::path::PathBuf, BTreeMap<&'static str, u64>>> {
    static S: OnceLock<Mutex<HashMap<std::path::PathBuf, BTreeMap<&'static str, u64>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

pub fn record_analytics(node: &elasticsearch_engine::node::Node, body: &Value) {
    fn walk(aggs: &Value, out: &mut BTreeMap<&'static str, u64>) {
        let Some(m) = aggs.as_object() else { return };
        for spec in m.values() {
            let Some(o) = spec.as_object() else { continue };
            for (k, v) in o {
                if let Some(t) = ANALYTICS_AGGS.iter().find(|t| **t == k.as_str()) {
                    *out.entry(t).or_default() += 1;
                }
                if k == "aggs" || k == "aggregations" {
                    walk(v, out);
                }
            }
        }
    }
    let mut seen = BTreeMap::new();
    for k in ["aggs", "aggregations"] {
        if let Some(a) = body.get(k) {
            walk(a, &mut seen);
        }
    }
    if seen.is_empty() {
        return;
    }
    let mut g = analytics_usage().lock().unwrap_or_else(|e| e.into_inner());
    let e = g.entry(node.data.clone()).or_default();
    for (k, n) in seen {
        *e.entry(k).or_default() += n;
    }
}

/// This node's analytics usage counts, as `xpack.usage` names them.
pub fn analytics_counts(node: &elasticsearch_engine::node::Node) -> BTreeMap<String, u64> {
    analytics_usage().lock().unwrap_or_else(|e| e.into_inner()).get(&node.data)
        .map(|m| m.iter().map(|(k, n)| (format!("{k}_usage"), *n)).collect()).unwrap_or_default()
}

/// ES `SecurityFeatureSetUsage`: `ssl` (whether the HTTP and transport
/// layers actually serve TLS) is reported once security is enabled.
fn security_usage(node: &elasticsearch_engine::node::Node) -> Value {
    if !node.security.enabled() {
        return serde_json::json!({"enabled": false});
    }
    let transport = node.node_settings.read().get("xpack.security.transport.ssl.enabled").is_some_and(|v| v == "true" || *v == Value::Bool(true));
    let http = node.security.cfg.read().http_ssl;
    serde_json::json!({"enabled": true, "ssl": {"http": {"enabled": http}, "transport": {"enabled": transport}}})
}

pub fn xpack_usage(node: &elasticsearch_engine::node::Node) -> Value {
    let mut out = XPACK_TEMPLATE.clone();
    let streams = node.data_streams.read();
    let indices: usize = streams.values().map(|d| d.indices.len()).sum();
    let ds = serde_json::json!({"available": true, "enabled": true, "data_streams": streams.len(), "indices_count": indices});
    drop(streams);
    out["data_streams"] = ds;
    out["data_lifecycle"] = elasticsearch_engine::ilm::dlm::usage(node);
    let vectordb: Vec<_> = node.all_indices().into_iter().filter(|svc| svc.setting("index.mode").as_deref() == Some("vectordb_document")).collect();
    out["vectordb_document"] = serde_json::json!({
        "available": true, "enabled": true,
        "indices_count": vectordb.len(),
        "num_docs": vectordb.iter().map(|svc| svc.doc_count().0).sum::<u64>()
    });
    out["ml"] = crate::ml::usage(node, &out["ml"]);
    merge(&mut out, serde_json::json!({
        "watcher": crate::watcher::xpack_usage(node),
        "security": security_usage(node),
        "ilm": {"policy_count": node.ilm.policy_count()},
        "rollup": {"number_of_rollup_jobs": crate::rollup::job_count(node)},
        "enterprise_search": {"search_applications": {"count": elasticsearch_engine::search_ext::ext(node).search_apps.read().len()}, "analytics_collections": {"count": crate::analytics::collections(node, "*").len()}, "query_rulesets": crate::query_lang::query_rules_usage(node)},
    }));
    let mounted: Vec<_> = node.all_indices().into_iter().filter(|s| elasticsearch_engine::snapshot::is_searchable_snapshot(s)).collect();
    let partial = mounted.iter().filter(|s| elasticsearch_engine::snapshot::is_partial_snapshot(s)).count();
    out["searchable_snapshots"]["indices_count"] = serde_json::json!(mounted.len());
    out["searchable_snapshots"]["full_copy_indices_count"] = serde_json::json!(mounted.len() - partial);
    out["searchable_snapshots"]["shared_cache_indices_count"] = serde_json::json!(partial);
    out["health_api"] = health_usage_json();
    // logsdb (ES `LogsDBUsageTransportAction` with `logsdb_telemetry_stats`).
    {
        let (mut n, mut synth, mut docs, mut bytes) = (0u64, 0u64, 0u64, 0u64);
        for svc in node.all_indices() {
            if svc.setting("index.mode").as_deref() != Some("logsdb") {
                continue;
            }
            n += 1;
            if svc.setting("index.mapping.source.mode").map(|m| m.eq_ignore_ascii_case("synthetic")).unwrap_or(true) {
                synth += 1;
            }
            docs += svc.doc_count().0;
            bytes += svc.store_size();
        }
        out["logsdb"]["indices_count"] = serde_json::json!(n);
        out["logsdb"]["indices_with_synthetic_source"] = serde_json::json!(synth);
        out["logsdb"]["num_docs"] = serde_json::json!(docs);
        out["logsdb"]["size_in_bytes"] = serde_json::json!(bytes);
    }
    // `GpuUsageTransportAction` over this node's GPU stats (a cluster
    // replaces it with every data node's, `TransportGpuStatsAction`).
    {
        use elasticsearch_engine::gpu;
        out["gpu_vector_indexing"] = gpu::usage_json(gpu::license_available(node), &[gpu::local_stats(node)]);
    }
    if let Some(m) = analytics_usage().lock().unwrap_or_else(|e| e.into_inner()).get(&node.data) {
        for (k, n) in m {
            out["analytics"]["stats"][format!("{k}_usage")] = serde_json::json!(n);
        }
    }
    if let Some(svc) = elasticsearch_engine::inference::current() {
        let mut fields: Vec<(String, String)> = Vec::new();
        for ix in node.all_indices() {
            if ix.name.starts_with('.') || ix.setting("index.hidden").as_deref() == Some("true") {
                continue;
            }
            let lk = ix.lookup();
            for f in lk.fields.values().filter(|f| elasticsearch_engine::inference::semantic::is_inference_field(f)) {
                let id = f.ext.as_ref().and_then(|e| e.str("inference_id")).unwrap_or(elasticsearch_engine::inference::semantic::DEFAULT_INFERENCE_ID);
                fields.push((ix.name.clone(), id.to_string()));
            }
        }
        out["inference"] = svc.usage(&fields);
    }
    out["esql"] = crate::query_lang::esql_usage(node);
    out["ccr"] = elasticsearch_engine::ccr::usage(node);
    out["logging"] = logging_usage(node);
    out
}

/// `LoggingUsageTransportAction`: query log and ES|QL query log settings.
fn logging_usage(node: &elasticsearch_engine::node::Node) -> Value {
    let get = |k: &str| -> Option<Value> {
        let cs = node.cluster_settings.read();
        cs.1.get(k).or(cs.0.get(k)).cloned().or_else(|| node.node_settings.read().get(k).cloned())
    };
    let flag = |k: &str, d: bool| match get(k) {
        Some(Value::Bool(b)) => b,
        Some(Value::String(s)) => s == "true",
        _ => d,
    };
    let time = |k: &str| -> Option<String> { get(k).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())) };
    let positive = |t: &str| elasticsearch_engine::date::parse_time_value(t).is_some_and(|ms| ms > 0) || t.ends_with("nanos") && t.trim_end_matches("nanos").parse::<i64>().is_ok_and(|n| n > 0);
    let mut querylog = serde_json::json!({
        "enabled": flag("elasticsearch.querylog.enabled", false),
        "user": flag("elasticsearch.querylog.include.user", true),
        "system": flag("elasticsearch.querylog.include.system_indices", false),
    });
    if let Some(t) = time("elasticsearch.querylog.threshold").filter(|t| positive(t)) {
        querylog["threshold"] = Value::String(t);
    }
    let mut thresholds = serde_json::Map::new();
    for level in ["trace", "debug", "info", "warn"] {
        if let Some(t) = time(&format!("esql.querylog.threshold.{level}")).filter(|t| !t.starts_with('-')) {
            thresholds.insert(level.into(), Value::String(t));
        }
    }
    let mut esql = serde_json::json!({"enabled": !thresholds.is_empty(), "user": flag("esql.querylog.include.user", false)});
    if !thresholds.is_empty() {
        esql["thresholds"] = Value::Object(thresholds);
    }
    serde_json::json!({"querylog": querylog, "esql": esql})
}
