//! Data streams and rollover.
//!
//! A data stream is a named, append-only sequence of hidden backing indices
//! `.ds-<name>-<yyyy.MM.dd>-<generation:06>`; the last one is the write
//! index. It needs a composable index template with `data_stream: {}` whose
//! settings/mappings every backing index gets (plus `@timestamp: date`).
//! Writes must use `op_type=create` and carry `@timestamp`. Searches of the
//! stream name (or a wildcard matching it) expand to all backing indices.
//!
//! Rollover (`POST /{alias|data stream}/_rollover`) creates the next index
//! when its conditions are met and swaps the write index.

use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::node::{pattern_list, Node};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub const TIMESTAMP_FIELD: &str = "@timestamp";

/// Internal node setting: data nodes the cluster topology emulates for shard
/// allocation accounting (see [`Node::data_nodes`]). Never set in production;
/// the YAML runner sets it to mirror ES's multi-node REST test clusters.
pub const DATA_NODES_SETTING: &str = "_internal.data_nodes";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataStream {
    pub name: String,
    pub generation: u64,
    /// Backing indices, oldest first; the last is the write index.
    pub indices: Vec<String>,
    pub template: String,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub allow_custom_routing: bool,
    /// Data stream lifecycle (`{"enabled", "data_retention"?, "downsampling"?}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle: Option<Value>,
    /// `_meta` from the template at creation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    #[serde(default)]
    pub replicated: bool,
    /// Failure store indices (`.fs-<name>-<date>-<generation>`), oldest
    /// first; the last is the failure store write index. Generations are
    /// shared with the backing indices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failure_indices: Vec<String>,
    /// Lazy rollover requested for the backing indices (`?lazy=true`).
    #[serde(default)]
    pub rollover_on_write: bool,
    /// Lazy rollover requested for the failure store.
    #[serde(default)]
    pub failure_rollover_on_write: bool,
    /// `data_stream_options` (`{"failure_store": {"enabled"?, "lifecycle"?}}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
    /// Data stream level index settings (`PUT _data_stream/{n}/_settings`),
    /// flat `index.*` keys; applied on top of the template for new
    /// backing indices.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub settings: Map<String, Value>,
    /// Data stream level mapping overrides (`PUT _data_stream/{n}/_mappings`),
    /// merged over the template's mappings for new backing indices.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub mappings: Map<String, Value>,
    /// `DataStream.indexMode`: the index mode the template set explicitly
    /// (`index.mode`) when the stream was created or last rolled over;
    /// `None` when it set none (GET then resolves it from the current
    /// template and the setting providers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_mode: Option<String>,
}

impl DataStream {
    pub fn write_index(&self) -> &str {
        self.indices.last().map(String::as_str).unwrap_or("")
    }

    pub fn failure_write_index(&self) -> Option<&str> {
        self.failure_indices.last().map(String::as_str)
    }

    /// `options.failure_store.enabled` when explicitly configured.
    pub fn failure_store_option(&self) -> Option<bool> {
        self.options.as_ref().and_then(|o| o["failure_store"]["enabled"].as_bool())
    }

    /// Failure store write index needs creating (uninitialized or lazy rollover).
    pub fn failure_needs_rollover(&self) -> bool {
        self.failure_indices.is_empty() || self.failure_rollover_on_write
    }

    fn new_named(name: &str, template: String, tmpl: &Value) -> DataStream {
        DataStream {
            name: name.to_string(),
            generation: 1,
            indices: Vec::new(),
            template,
            hidden: tmpl["data_stream"]["hidden"].as_bool().unwrap_or(false),
            allow_custom_routing: tmpl["data_stream"]["allow_custom_routing"].as_bool().unwrap_or(false),
            lifecycle: None,
            meta: tmpl.get("_meta").cloned(),
            replicated: false,
            failure_indices: Vec::new(),
            rollover_on_write: false,
            failure_rollover_on_write: false,
            options: None,
            settings: Map::new(),
            mappings: Map::new(),
            index_mode: None,
        }
    }
}

pub fn backing_index_name(ds: &str, generation: u64, millis: i64) -> String {
    let day = crate::date::DateFormatter::new("yyyy.MM.dd").map(|f| f.format(millis, crate::date::Tz::UTC)).unwrap_or_default();
    format!(".ds-{ds}-{day}-{generation:06}")
}

pub fn failure_index_name(ds: &str, generation: u64, millis: i64) -> String {
    let day = crate::date::DateFormatter::new("yyyy.MM.dd").map(|f| f.format(millis, crate::date::Tz::UTC)).unwrap_or_default();
    format!(".fs-{ds}-{day}-{generation:06}")
}

/// Index component selector of an expression (`name::data`, `name::failures`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selector {
    Data,
    Failures,
}

/// Split `expr` into (name, selector). `Ok((expr, None))` without `::`.
pub fn split_selector(expr: &str) -> Result<(&str, Option<Selector>)> {
    let Some(pos) = expr.find("::") else { return Ok((expr, None)) };
    let (name, rest) = (&expr[..pos], &expr[pos + 2..]);
    let bad = |why: String| Err(EsError::new(400, "invalid_index_name_exception", format!("Invalid index name [{expr}], {why}")).with("index_uuid", "_na_").with("index", expr.to_string()));
    if rest.contains("::") {
        return bad("Invalid usage of :: separator, only one :: separator is allowed per expression".into());
    }
    match rest {
        "data" => Ok((name, Some(Selector::Data))),
        "failures" => Ok((name, Some(Selector::Failures))),
        other => bad(format!("invalid usage of :: separator, [{other}] is not a recognized selector")),
    }
}

/// `unsupported_selector_exception` for APIs that do not take selectors.
pub fn reject_selectors(expr: &str) -> Result<()> {
    for p in expr.split(',') {
        if p.contains("::") {
            return Err(EsError::new(400, "unsupported_selector_exception", format!("Index component selectors are not supported in this context but found selector in expression [{}]", p.trim())));
        }
    }
    Ok(())
}

/// Whether the logsdb plugin's provider makes `logs-*-*` data streams
/// logsdb: the plugin is installed (the internal node setting
/// [`LOGSDB_PLUGIN_SETTING`] is not `false`) and `cluster.logsdb.enabled`
/// (default `true` unless `logsdb.prior_logs_usage`).
pub fn logsdb_default_applies(node: &Node) -> bool {
    if node.node_settings.read().get(LOGSDB_PLUGIN_SETTING).map(|v| v.as_str() == Some("false") || v.as_bool() == Some(false)).unwrap_or(false) {
        return false;
    }
    cluster_logsdb_enabled(node)
}

/// `cluster.logsdb.enabled` (`LogsDBPlugin.CLUSTER_LOGSDB_ENABLED`).
pub fn cluster_logsdb_enabled(node: &Node) -> bool {
    let get = |k: &str| crate::ilm::cluster_setting(node, k).or_else(|| node.node_settings.read().get(k).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())));
    match get("cluster.logsdb.enabled").as_deref() {
        Some("true") => true,
        Some("false") => false,
        _ => get("logsdb.prior_logs_usage").as_deref() != Some("true"),
    }
}

/// Internal node setting: `false` when the node has no logsdb plugin (ES's
/// OSS INTEG_TEST distribution); the YAML runner sets it per suite.
pub const LOGSDB_PLUGIN_SETTING: &str = "_internal.xpack.logsdb";

/// Settings `PUT _data_stream/{n}/_settings` may change, and whether each
/// also applies to the existing backing indices.
const DS_SETTINGS_BACKING: &[&str] = &["index.lifecycle.name", "index.lifecycle.prefer_ilm", "index.refresh_interval"];
/// Settings that reach the data stream and its current write index only.
const DS_SETTINGS_WRITE: &[&str] = &["index.number_of_replicas"];
const DS_SETTINGS_DS_ONLY: &[&str] = &["index.number_of_shards"];

/// Template settings a failure store index keeps
/// (`DataStreamFailureStoreDefinition.SUPPORTED_USER_SETTINGS`).
fn failure_store_setting(k: &str) -> bool {
    matches!(k, "index.number_of_shards" | "index.number_of_replicas" | "index.auto_expand_replicas" | "index.hidden" | "index.lifecycle.name" | "index.refresh_interval")
        || k.starts_with("index.routing.allocation.require.")
        || k.starts_with("index.routing.allocation.include.")
        || k.starts_with("index.routing.allocation.exclude.")
}

/// Mappings of every failure store index (`DataStreamFailureStoreDefinition`).
fn failure_store_mappings() -> Value {
    json!({
        "dynamic": "false",
        "_data_stream_timestamp": {"enabled": true},
        "properties": {
            "@timestamp": {"type": "date"},
            "document": {"properties": {
                "id": {"type": "keyword"},
                "index": {"type": "keyword"},
                "routing": {"type": "keyword"}
            }},
            "error": {"properties": {
                "message": {"type": "match_only_text"},
                "pipeline": {"type": "keyword"},
                "pipeline_trace": {"type": "keyword"},
                "processor_tag": {"type": "keyword"},
                "processor_type": {"type": "keyword"},
                "stack_trace": {"type": "text"},
                "type": {"type": "keyword"}
            }}
        }
    })
}

/// Where a write that fails should go (`failure_store` response field).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailureStoreTarget {
    /// Not a data stream write: no `failure_store` field.
    None,
    /// A data stream without the failure store: `not_enabled`.
    Disabled,
    /// Redirect to this data stream's failure store.
    Enabled(String),
}

/// Ingest failure details for the failure document.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FailureContext {
    pub pipeline_trace: Vec<String>,
    pub pipeline: Option<String>,
    pub processor_type: Option<String>,
    pub processor_tag: Option<String>,
}

/// Whether `e` may be redirected to a failure store (everything but
/// version conflicts and rejections).
pub fn failure_store_eligible(e: &EsError) -> bool {
    !matches!(e.kind.as_str(), "version_conflict_engine_exception" | "es_rejected_execution_exception" | "circuit_breaking_exception" | "cluster_block_exception") && e.status != 429
}

#[test]
fn failure_store_exclusions_match_bulk_operation() {
    for kind in ["version_conflict_engine_exception", "es_rejected_execution_exception", "circuit_breaking_exception", "cluster_block_exception"] {
        assert!(!failure_store_eligible(&EsError::new(503, kind, "test")));
    }
    assert!(!failure_store_eligible(&EsError::new(429, "other_exception", "test")));
    // ES 9.5.5 BulkOperation does not exclude all temporary 503s.
    assert!(failure_store_eligible(&EsError::new(503, "mapper_exception", "dynamic mapping timeout")));
    assert!(failure_store_eligible(&EsError::new(400, "document_parsing_exception", "bad value")));
}

/// ES's abbreviated exception class name for a `type`
/// (`document_parsing_exception` -> `o.e.DocumentParsingException`).
fn java_class(kind: &str) -> String {
    let camel: String = kind.split('_').map(|w| {
        let mut c = w.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    }).collect();
    if matches!(kind, "illegal_argument_exception" | "illegal_state_exception" | "number_format_exception" | "unsupported_operation_exception") {
        format!("j.l.{camel}")
    } else {
        format!("o.e.{camel}")
    }
}

fn stack_trace(e: &EsError) -> String {
    let mut s = format!("{}: {}\n", java_class(&e.kind), e.reason);
    let mut cur = e.caused_by.as_deref();
    while let Some(c) = cur {
        s.push_str(&format!("Caused by: {}: {}\n", java_class(&c.kind), c.reason));
        cur = c.caused_by.as_deref();
    }
    s
}

/// The failure document ES writes to a failure store.
pub fn failure_document(index: &str, id: Option<&str>, routing: Option<&str>, source: &Value, e: &EsError, cx: Option<&FailureContext>) -> Value {
    let mut doc = Map::new();
    if let Some(i) = id {
        doc.insert("id".into(), json!(i));
    }
    if let Some(r) = routing {
        doc.insert("routing".into(), json!(r));
    }
    doc.insert("index".into(), json!(index));
    doc.insert("source".into(), source.clone());
    let mut err = Map::new();
    // Wrapper exceptions are unwrapped (`ExceptionsHelper.unwrapCause`).
    let mut e = e;
    while matches!(e.kind.as_str(), "ingest_processor_exception" | "remote_transport_exception") {
        match e.caused_by.as_deref() {
            Some(c) => e = c,
            None => break,
        }
    }
    err.insert("type".into(), json!(e.kind));
    err.insert("message".into(), json!(e.reason));
    err.insert("stack_trace".into(), json!(stack_trace(e)));
    if let Some(cx) = cx {
        if !cx.pipeline_trace.is_empty() {
            err.insert("pipeline_trace".into(), json!(cx.pipeline_trace));
        }
        if let Some(p) = &cx.pipeline {
            err.insert("pipeline".into(), json!(p));
        }
        if let Some(t) = &cx.processor_tag {
            err.insert("processor_tag".into(), json!(t));
        }
        if let Some(t) = &cx.processor_type {
            err.insert("processor_type".into(), json!(t));
        }
    }
    let ts = crate::date::DateFormatter::new("strict_date_optional_time").map(|f| f.format(crate::date::now_millis(), crate::date::Tz::UTC)).unwrap_or_default();
    json!({"@timestamp": ts, "document": doc, "error": err})
}

fn validate_ds_name(name: &str) -> Result<()> {
    // MetadataCreateDataStreamService.validateDataStreamName
    for c in ['#', ':'] {
        if name.contains(c) {
            return Err(EsError::illegal_argument(format!("data_stream [{name}] must not contain '{c}'")));
        }
    }
    let bad = |why: &str| Err(EsError::new(400, "invalid_index_name_exception", format!("Invalid index name [{name}], {why}")).with("index_uuid", "_na_").with("index", name.to_string()));
    if name.is_empty() {
        return bad("must not be empty");
    }
    if name != name.to_lowercase() {
        return bad("must be lowercase");
    }
    if name.starts_with(['-', '_', '+']) {
        return bad("must not start with '_', '-', or '+'");
    }
    if name.starts_with(".ds-") {
        return Err(EsError::illegal_argument(format!("data_stream [{name}] must not start with '.ds-'")));
    }
    if name.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ',' | '#' | ':')) {
        return bad("must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]");
    }
    Ok(())
}

/// Nested JSON of flat data stream settings (`{}` when none).
fn settings_json(m: &Map<String, Value>) -> Value {
    let flat: crate::settings::FlatSettings = m.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())).collect();
    crate::settings::unflatten(&flat)
}

/// Compose `data_stream_options` from templates in order: an absent value
/// keeps what came before, an explicit `null` resets it, objects merge.
pub fn compose_data_stream_options(parts: impl Iterator<Item = Option<Value>>) -> Option<Value> {
    let mut acc: Option<Map<String, Value>> = None;
    for p in parts {
        match p {
            None => {}
            Some(Value::Object(o)) => {
                let a = acc.get_or_insert_with(Map::new);
                match o.get("failure_store") {
                    None => {}
                    Some(Value::Object(fs)) => {
                        let afs = a.entry("failure_store").or_insert_with(|| json!({}));
                        if !afs.is_object() {
                            *afs = json!({});
                        }
                        let afs = afs.as_object_mut().expect("object");
                        for (k, v) in fs {
                            match (k.as_str(), v) {
                                (_, Value::Null) => {
                                    afs.remove(k);
                                }
                                ("lifecycle", Value::Object(lc)) => {
                                    let al = afs.entry("lifecycle").or_insert_with(|| json!({}));
                                    if !al.is_object() {
                                        *al = json!({});
                                    }
                                    let al = al.as_object_mut().expect("object");
                                    for (lk, lv) in lc {
                                        if lv.is_null() {
                                            al.remove(lk);
                                        } else {
                                            al.insert(lk.clone(), lv.clone());
                                        }
                                    }
                                }
                                _ => {
                                    afs.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        if afs.is_empty() {
                            a.remove("failure_store");
                        }
                    }
                    Some(_) => {
                        a.remove("failure_store");
                    }
                }
            }
            Some(_) => acc = None,
        }
    }
    acc.map(|mut a| {
        if let Some(lc) = a.get_mut("failure_store").and_then(|f| f.get_mut("lifecycle")) {
            if let Ok(n) = crate::ilm::dlm::normalize_lifecycle(lc) {
                *lc = n;
            }
        }
        Value::Object(a)
    })
}

/// Validate a `data_stream_options` body (template or `_options` API);
/// `what` is the parser name of the enclosing object.
pub fn validate_data_stream_options(v: &Value, what: &str) -> Result<()> {
    let o = match v {
        Value::Null => return Ok(()),
        Value::Object(o) => o,
        _ => return Err(EsError::x_content_parse(format!("[1:2] [{what}] data_stream_options doesn't support values of type: VALUE_STRING"))),
    };
    for (k, fs) in o {
        if k != "failure_store" {
            return Err(EsError::x_content_parse(format!("[1:2] [{what}] unknown field [{k}]")));
        }
        let Some(fs) = fs.as_object() else {
            if fs.is_null() {
                continue;
            }
            return Err(EsError::x_content_parse(format!("[1:2] [{what}] failed to parse field [failure_store]")));
        };
        for (fk, fv) in fs {
            match fk.as_str() {
                "enabled" => {
                    if !(fv.is_boolean() || fv.is_null() || matches!(fv.as_str(), Some("true" | "false"))) {
                        return Err(EsError::x_content_parse("[1:2] [failure_store] failed to parse field [enabled]"));
                    }
                }
                "lifecycle" => {
                    if let Some(lc) = fv.as_object() {
                        for lk in lc.keys() {
                            if !matches!(lk.as_str(), "enabled" | "data_retention" | "frozen_after") {
                                return Err(EsError::x_content_parse(format!("[1:2] [failure_store] failed to parse field [lifecycle]"))
                                    .caused_by(EsError::x_content_parse(format!("[1:2] [lifecycle] unknown field [{lk}]"))));
                            }
                        }
                        let clean: Map<String, Value> = lc.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())).collect();
                        crate::ilm::dlm::normalize_lifecycle(&Value::Object(clean))?;
                        // Request validation (`PutDataStreamOptionsAction`, template APIs).
                        if lc.get("frozen_after").is_some_and(|v| !v.is_null()) {
                            return Err(EsError::action_validation(&["Failure store lifecycle does not support freezing, please remove the frozen_after configuration."]));
                        }
                    } else if !fv.is_null() {
                        return Err(EsError::x_content_parse("[1:2] [failure_store] failed to parse field [lifecycle]"));
                    }
                }
                other => {
                    return Err(EsError::x_content_parse(format!("[1:2] [{what}] failed to parse field [failure_store]"))
                        .caused_by(EsError::x_content_parse(format!("[1:2] [failure_store] unknown field [{other}]"))))
                }
            }
        }
    }
    Ok(())
}

/// Rollover conditions: `(key as rendered, met)`.
struct Conditions {
    items: Vec<(String, bool, bool)>, // (rendered, met, is_min)
}

fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    let units: [(&str, u64); 6] = [("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)];
    for (u, m) in units {
        if let Some(n) = s.strip_suffix(u) {
            return n.trim().parse::<f64>().ok().map(|x| (x * m as f64) as u64);
        }
    }
    None
}

/// An index's stats for rollover conditions (ES `TransportRolloverAction`
/// reads them from `IndicesStats` across the cluster): primaries' docs and
/// store size, and the largest primary shard's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RolloverStats {
    pub docs: u64,
    pub size: u64,
    pub primary_shard_docs: u64,
    pub primary_shard_size: u64,
}

/// Supplies [`RolloverStats`] for an index by name (`None`: use the local
/// copies). A cluster installs one on the node whose engine runs lifecycle
/// and rollovers, where the local copy may hold no data at all.
pub type RolloverStatsHook = Arc<dyn Fn(&str) -> Option<RolloverStats> + Send + Sync>;

impl Node {
    /// Install (or remove) the [`RolloverStatsHook`].
    pub fn set_rollover_stats_hook(&self, hook: Option<RolloverStatsHook>) {
        *self.rollover_stats_hook.write() = hook;
    }

    fn rollover_stats(&self, svc: &IndexService) -> RolloverStats {
        let hook = self.rollover_stats_hook.read().clone();
        if let Some(stats) = hook.and_then(|h| h(&svc.name)) {
            return stats;
        }
        let (docs, _) = svc.doc_count();
        // An empty index stores nothing (ES counts segment bytes only).
        let size = if docs == 0 { 0 } else { svc.store_size() };
        RolloverStats { docs, size, primary_shard_docs: docs, primary_shard_size: size }
    }
}

fn eval_conditions(svc: &IndexService, conds: &Map<String, Value>, stats: RolloverStats) -> Result<Conditions> {
    let RolloverStats { docs, size, primary_shard_docs, primary_shard_size } = stats;
    let age = crate::date::now_millis() - svc.creation_date;
    let mut items = Vec::new();
    for (k, v) in conds {
        let raw = match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let is_min = k.starts_with("min_");
        let met = match k.trim_start_matches("max_").trim_start_matches("min_") {
            "age" => {
                let ms = crate::date::parse_time_value(&raw).ok_or_else(|| {
                    EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{raw}] as a time value: unit is missing or unrecognized"))
                })?;
                if is_min { age >= ms } else { age >= ms }
            }
            field @ ("docs" | "primary_shard_docs") => {
                let n: u64 = raw.parse().map_err(|_| EsError::new(400, "x_content_parse_exception", format!("[1:1] [conditions] failed to parse field [{k}]")))?;
                (if field == "docs" { docs } else { primary_shard_docs }) >= n
            }
            field @ ("size" | "primary_shard_size") => {
                let n = parse_size(&raw).ok_or_else(|| {
                    EsError::new(400, "x_content_parse_exception", format!("[1:1] [conditions] failed to parse field [{k}]"))
                        .caused_by(EsError::new(400, "parse_exception", format!("failed to parse setting [{k}] with value [{raw}] as a size in bytes: unit is missing or unrecognized")))
                })?;
                (if field == "size" { size } else { primary_shard_size }) >= n
            }
            _ => return Err(EsError::new(400, "x_content_parse_exception", format!("[1:1] [conditions] unknown field [{k}]"))),
        };
        items.push((format!("[{k}: {raw}]"), met, is_min));
    }
    Ok(Conditions { items })
}

impl Conditions {
    fn met(&self) -> bool {
        if self.items.is_empty() {
            return true;
        }
        let maxes: Vec<bool> = self.items.iter().filter(|c| !c.2).map(|c| c.1).collect();
        let mins_ok = self.items.iter().filter(|c| c.2).all(|c| c.1);
        (maxes.is_empty() || maxes.iter().any(|m| *m)) && mins_ok
    }
    fn json(&self) -> Value {
        Value::Object(self.items.iter().map(|(k, m, _)| (k.clone(), json!(m))).collect())
    }
}

/// Next index name for an alias rollover: `-\d+` suffix incremented, 6 wide.
fn next_index_name(old: &str) -> Result<String> {
    let bad = || EsError::illegal_argument(format!("index name [{old}] does not match pattern '^.*-\\d+$'"));
    let (prefix, num) = old.rsplit_once('-').ok_or_else(bad)?;
    if num.is_empty() || !num.chars().all(|c| c.is_ascii_digit()) {
        return Err(bad());
    }
    let n: u64 = num.parse().map_err(|_| bad())?;
    Ok(format!("{prefix}-{:06}", n + 1))
}

impl Node {
    /// `require_data_stream`: the target must be (or auto-create) a data stream.
    pub fn require_data_stream(&self, idx: &str, flag: bool) -> Result<()> {
        if !flag || self.data_stream(idx).is_some() {
            return Ok(());
        }
        if self.get_index(idx).is_some() || self.all_indices().iter().any(|s| s.aliases().contains_key(idx)) {
            return Err(EsError::illegal_argument(format!("The index [{idx}] is not a data stream")));
        }
        if self.data_stream_template(idx).is_none() {
            return Err(EsError::new(404, "index_not_found_exception", format!("no such index [{idx}] and the index creation request requires a data stream, but no matching index template with data stream template was found for it"))
                .with("resource.type", "index_expression")
                .with("resource.id", idx)
                .with("index_uuid", "_na_")
                .with("index", idx));
        }
        Ok(())
    }

    /// The composable template (name, body) that would apply to `name`
    /// when it declares `data_stream`.
    pub fn data_stream_template(&self, name: &str) -> Option<(String, Value)> {
        let t = self.templates.read();
        let best = t
            .index
            .iter()
            .filter(|(_, v)| pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, name)))
            .max_by_key(|(_, v)| v["priority"].as_i64().unwrap_or(0))?;
        if best.1.get("data_stream").map(|d| d.is_object()).unwrap_or(false) {
            Some((best.0.clone(), best.1.clone()))
        } else {
            None
        }
    }

    pub fn data_stream(&self, name: &str) -> Option<DataStream> {
        self.data_streams.read().get(name).cloned()
    }

    /// Whether `name` is a data stream (no copy of it).
    pub fn is_data_stream(&self, name: &str) -> bool {
        self.data_streams.read().contains_key(name)
    }

    /// Backing indices of a data stream name.
    pub fn data_stream_indices(&self, name: &str) -> Option<Vec<String>> {
        self.data_streams.read().get(name).map(|d| d.indices.clone())
    }

    /// Backing + failure-store indices of a data stream name.
    pub fn data_stream_all_indices(&self, name: &str) -> Option<Vec<String>> {
        let dss = self.data_streams.read();
        let d = dss.get(name)?;
        let mut v = d.indices.clone();
        v.extend(d.failure_indices.iter().cloned());
        Some(v)
    }

    /// Data nodes that shard allocation accounts for (cluster health, data
    /// stream status): [`DATA_NODES_SETTING`], default 1 — a true single
    /// node, where every replica stays unassigned exactly as on a one-node ES.
    pub fn data_nodes(&self) -> usize {
        let v = self.node_settings.read().get(DATA_NODES_SETTING).cloned();
        v.and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).map(|n| n as usize).unwrap_or(1).max(1)
    }

    /// Replica copies per shard of `svc` that cannot be allocated: a copy
    /// never shares a node with another copy of its shard, so at most
    /// `data_nodes - 1` replicas per shard are assigned.
    pub fn unassigned_replicas(&self, svc: &IndexService) -> usize {
        svc.setting_usize("index.number_of_replicas", 1).saturating_sub(self.data_nodes() - 1)
    }

    /// Data streams whose name matches a wildcard pattern.
    pub fn data_streams_matching(&self, pat: &str, include_hidden: bool) -> Vec<DataStream> {
        self.data_streams
            .read()
            .values()
            .filter(|d| (include_hidden || !d.hidden || pat.starts_with('.')) && crate::mapping::wildcard_match(pat, &d.name))
            .cloned()
            .collect()
    }

    /// The data stream a backing index belongs to.
    /// The concrete index `name` addresses when every bulk write
    /// resolution of it is the identity: no percent escapes or date math,
    /// a registered index, no data stream of that name, and not a data
    /// stream backing index. For such a name `index_routing_for` keeps the
    /// item's routing, `write_index` and `time_series_write_index` give
    /// this index, `failure_store_target` is `None`, and the backing and
    /// data stream write checks pass.
    pub fn plain_write_index(&self, name: &str) -> Option<std::sync::Arc<crate::index::IndexService>> {
        if name.contains('%') || name.starts_with('<') {
            return None;
        }
        let svc = self.get_index(name)?;
        let streams = self.data_streams.read();
        if streams.contains_key(name) || streams.values().any(|d| d.indices.iter().any(|i| i == name)) {
            return None;
        }
        Some(svc)
    }

    pub fn data_stream_of_index(&self, index: &str) -> Option<String> {
        self.data_streams.read().values().find(|d| d.indices.iter().any(|i| i == index)).map(|d| d.name.clone())
    }

    /// The data stream a failure store index belongs to.
    pub fn data_stream_of_failure_index(&self, index: &str) -> Option<String> {
        self.data_streams.read().values().find(|d| d.failure_indices.iter().any(|i| i == index)).map(|d| d.name.clone())
    }

    /// `data_stream_options` composed from a template's component templates
    /// and its own `template.data_stream_options`; an explicit `null`
    /// resets what earlier templates set (`ResettableValue`).
    pub fn template_data_stream_options(&self, template: &Value) -> Option<Value> {
        let t = self.templates.read();
        let mut parts = Vec::new();
        for c in template["composed_of"].as_array().cloned().unwrap_or_default() {
            if let Some(ct) = c.as_str().and_then(|n| t.component.get(n)) {
                parts.push(ct["template"].clone());
            }
        }
        parts.push(template["template"].clone());
        compose_data_stream_options(parts.iter().map(|p| p.as_object().and_then(|o| o.get("data_stream_options").cloned())))
    }

    /// The cluster setting `data_streams.failure_store.enabled` (name patterns).
    fn failure_store_enabled_by_setting(&self, name: &str) -> bool {
        if name.starts_with('.') {
            return false;
        }
        const KEY: &str = "data_streams.failure_store.enabled";
        {
            // Cheap check first (bulk hot path): no such setting at all.
            let cs = self.cluster_settings.read();
            if ![&cs.0, &cs.1].iter().any(|m| m.contains_key(KEY) || m.contains_key("data_streams")) {
                return false;
            }
        }
        crate::ilm::cluster_setting(self, KEY)
            .map(|s| s.trim_matches(|c| c == '[' || c == ']').split(',').map(|p| p.trim().trim_matches('"')).any(|p| !p.is_empty() && crate::mapping::wildcard_match(p, name)))
            .unwrap_or(false)
    }

    /// Effective failure store state of a data stream (options, else the
    /// cluster setting).
    pub fn failure_store_enabled(&self, ds: &DataStream) -> bool {
        ds.failure_store_option().unwrap_or_else(|| self.failure_store_enabled_by_setting(&ds.name))
    }

    /// Failure store routing for a write addressed to `target`.
    pub fn failure_store_target(&self, target: &str) -> FailureStoreTarget {
        // Hot path (every bulk item): no DataStream clone.
        let ds_opt = self.data_streams.read().get(target).map(|d| d.failure_store_option());
        if let Some(opt) = ds_opt {
            return if opt.unwrap_or_else(|| self.failure_store_enabled_by_setting(target)) { FailureStoreTarget::Enabled(target.to_string()) } else { FailureStoreTarget::Disabled };
        }
        if self.get_index(target).is_some() || target.contains("::") {
            return FailureStoreTarget::None;
        }
        // A write that would auto-create a data stream.
        if let Some((_, tmpl)) = self.data_stream_template(target) {
            let opt = self.template_data_stream_options(&tmpl).and_then(|o| o["failure_store"]["enabled"].as_bool());
            return if opt.unwrap_or_else(|| self.failure_store_enabled_by_setting(target)) {
                FailureStoreTarget::Enabled(target.to_string())
            } else {
                FailureStoreTarget::Disabled
            };
        }
        FailureStoreTarget::None
    }

    /// Effective index settings a data stream gives its next backing index:
    /// the template's, then the data stream's own overrides.
    pub fn data_stream_effective_settings(&self, ds: &str) -> crate::settings::FlatSettings {
        let (settings, _, _) = self.template_for(ds);
        let mut flat = crate::settings::normalize_index_settings(&settings);
        flat.retain(|_, v| !v.is_null());
        if let Some(d) = self.data_stream(ds) {
            for (k, v) in &d.settings {
                if v.is_null() {
                    flat.remove(k);
                } else {
                    flat.insert(k.clone(), v.clone());
                }
            }
        }
        flat
    }

    /// The mappings a data stream gives its next backing index before
    /// index-mode additions: the template's (components, then the
    /// template's own), then `overrides` (the data stream's mapping
    /// overrides; `ComposableIndexTemplate.mergeMappings`), then the
    /// data stream timestamp field.
    pub fn data_stream_mappings_with(&self, ds: &str, overrides: &Map<String, Value>) -> Value {
        let (_settings, mut mappings, _aliases) = self.template_for(ds);
        if !overrides.is_empty() {
            if !mappings.is_object() {
                mappings = json!({});
            }
            crate::node::merge_mapping(&mut mappings, &Value::Object(overrides.clone()));
        }
        mappings
    }

    /// `DataStream.getEffectiveMappings`: what the write index would get if
    /// the stream rolled over now, as the mapper renders it.
    pub fn data_stream_effective_mappings(&self, ds: &str, overrides: &Map<String, Value>) -> Result<Value> {
        let mut mappings = self.data_stream_mappings_with(ds, overrides);
        if !mappings.is_object() {
            mappings = json!({});
        }
        if mappings.get("properties").and_then(|p| p.get(TIMESTAMP_FIELD)).is_none() {
            mappings["properties"][TIMESTAMP_FIELD] = json!({"type": "date"});
        }
        mappings["_data_stream_timestamp"] = json!({"enabled": true});
        let registry = match self.data_stream(ds).and_then(|d| self.get_index(d.write_index())) {
            Some(svc) => svc.registry(),
            None => crate::analysis::AnalysisRegistry::default(),
        };
        let parsed = crate::mapping::Mappings::parse(&mappings, &registry)?;
        Ok(parsed.to_json())
    }

    /// `GET /_data_stream/{name}/_mappings`.
    pub fn get_data_stream_mappings(&self, expr: &str, include_hidden: bool) -> Result<Value> {
        reject_selectors(expr)?;
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        let mut arr = Vec::new();
        for d in &dss {
            arr.push(json!({
                "name": d.name,
                "mappings": Value::Object(d.mappings.clone()),
                "effective_mappings": self.data_stream_effective_mappings(&d.name, &d.mappings)?,
            }));
        }
        Ok(json!({ "data_streams": arr }))
    }

    /// `PUT /_data_stream/{name}/_mappings`: the body replaces the data
    /// stream's mapping overrides (validated as the effective mapping of the
    /// next backing index); existing backing indices are untouched.
    pub fn put_data_stream_mappings(&self, expr: &str, include_hidden: bool, body: &Value, dry_run: bool) -> Result<Value> {
        reject_selectors(expr)?;
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        let overrides = match body {
            Value::Object(o) => o.clone(),
            other => return Err(EsError::illegal_argument(format!("Unexpected token: {}", crate::jsonpos::token_name(other)))),
        };
        let mut arr = Vec::new();
        for d in &dss {
            let failed = |e: String| json!({"name": d.name, "applied_to_data_stream": false, "error": e});
            let eff = match self.data_stream_effective_mappings(&d.name, &overrides) {
                Ok(v) => v,
                Err(e) => {
                    arr.push(failed(e.reason.clone()));
                    continue;
                }
            };
            if !dry_run {
                if let Some(s) = self.data_streams.write().get_mut(&d.name) {
                    s.mappings = overrides.clone();
                }
                self.persist_state()?;
            }
            let mut v = Map::new();
            v.insert("name".into(), json!(d.name));
            v.insert("applied_to_data_stream".into(), json!(true));
            if !overrides.is_empty() {
                v.insert("mappings".into(), Value::Object(overrides.clone()));
            }
            if eff.as_object().is_some_and(|o| !o.is_empty()) {
                v.insert("effective_mappings".into(), eff);
            }
            arr.push(Value::Object(v));
        }
        Ok(json!({ "data_streams": arr }))
    }

    /// `ProjectMetadata.retrieveIndexModeFromTemplate`: `index.mode` as the
    /// matching template (with the stream's setting overrides) sets it.
    pub fn template_index_mode(&self, ds: &str) -> Option<String> {
        self.data_stream_effective_settings(ds).get("index.mode").and_then(Value::as_str).filter(|m| !m.is_empty()).map(str::to_string)
    }

    /// The index mode a new backing index of `ds` gets: the template's, or
    /// `logsdb` from the logsdb plugin's setting provider for `logs-*-*`
    /// data streams (`LogsdbIndexModeSettingsProvider`).
    pub fn resolve_data_stream_index_mode(&self, ds: &str) -> Option<String> {
        self.template_index_mode(ds).or_else(|| (logsdb_default_applies(self) && crate::mapping::wildcard_match("logs-*-*", ds)).then(|| "logsdb".to_string()))
    }

    fn create_backing_index(&self, ds: &str, generation: u64) -> Result<Arc<IndexService>> {
        let overrides = self.data_stream(ds).map(|d| d.mappings).unwrap_or_default();
        let mut mappings = self.data_stream_mappings_with(ds, &overrides);
        let mut flat = self.data_stream_effective_settings(ds);
        if !flat.contains_key("index.mode") {
            if let Some(m) = self.resolve_data_stream_index_mode(ds) {
                flat.insert("index.mode".into(), json!(m));
            }
        }
        flat.insert("index.hidden".into(), json!("true"));
        flat.entry("index.routing.allocation.include._tier_preference".to_string()).or_insert(json!("data_hot"));
        self.time_series_backing_settings(ds, &mut flat, None);
        // logsdb: inject `host.name` (keyword, ignore_above 1024) when the
        // sort is on `host.name` (or no sort is set) and the mapping has no
        // `host.name` field. Also validate the sort fields.
        let mode_l = flat.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase);
        if matches!(mode_l.as_deref(), Some("logsdb" | "logsdb_columnar")) {
            let sort_fields: Vec<String> = flat.get("index.sort.field").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
            if !mappings.is_object() {
                mappings = json!({});
            }
            if !mappings["properties"].is_object() {
                mappings["properties"] = json!({});
            }
            // `LogsdbIndexModeSettingsProvider.getMappingHints`: no hints
            // with a custom sort; otherwise sort on `host.name` when it is
            // added (absent, and `host` is no leaf field) or is a keyword /
            // number with doc values.
            if !flat.contains_key("index.sort.field") {
                let truthy = |k: &str| flat.get(k).is_some_and(|v| v.as_bool() == Some(true) || v.as_str() == Some("true"));
                let props = mappings["properties"].as_object().unwrap();
                // `host.name` may also be a multi-field of `host`.
                let host_name = lookup_field_any(&mappings["properties"], "host.name").or_else(|| props.get("host").and_then(|h| h.get("fields")).and_then(|f| f.get("name")).cloned());
                let is_obj = |d: &Value| d.get("properties").is_some() && d.get("type").is_none() || matches!(d.get("type").and_then(Value::as_str), Some("object" | "nested"));
                // A pass-through object's `host.name` leaf is reachable as
                // `host.name` (an alias: neither added nor sorted on).
                fn passthrough_host_name(props: &serde_json::Map<String, Value>) -> bool {
                    props.values().any(|d| {
                        let sub = d.get("properties").and_then(Value::as_object);
                        if d.get("type").and_then(Value::as_str) == Some("passthrough") {
                            if let Some(p) = sub {
                                if p.contains_key("host.name") || p.get("host").and_then(|h| h.get("properties")).and_then(|x| x.get("name")).is_some() {
                                    return true;
                                }
                            }
                        }
                        sub.is_some_and(passthrough_host_name)
                    })
                }
                let via_passthrough = host_name.is_none() && passthrough_host_name(props);
                let (host_name_field, host_name_object) = match &host_name {
                    Some(d) if is_obj(d) => (None, true),
                    Some(d) => (Some(d), false),
                    None => (None, false),
                };
                let host_leaf = props.get("host").is_some_and(|h| !is_obj(h) && !(h.get("type").is_none() && h.as_object().is_some_and(|o| o.is_empty())));
                let no_subobjects = mappings.get("subobjects").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))) == Some(false) || mode_l.as_deref() == Some("logsdb_columnar");
                let add_host_name = truthy("index.logsdb.add_host_name_field") || (!via_passthrough && host_name_field.is_none() && !host_name_object && (!host_leaf || no_subobjects));
                let dv = |d: &Value| d.get("doc_values").and_then(Value::as_bool) != Some(false);
                let numeric = |t: &str| matches!(t, "long" | "integer" | "short" | "byte" | "double" | "float" | "half_float" | "scaled_float" | "unsigned_long");
                let sort_on_host = truthy("index.logsdb.sort_on_host_name")
                    || add_host_name
                    || host_name_field.is_some_and(|d| {
                        let t = d.get("type").and_then(Value::as_str).unwrap_or("");
                        (t == "keyword" || numeric(t)) && dv(d)
                    });
                // Sort on `message.template_id` for a `pattern_text` message
                // when `index.logsdb.default_sort_on_message_template` asks.
                let message_pattern = lookup_field_any(&mappings["properties"], "message").is_some_and(|d| d.get("type").and_then(Value::as_str) == Some("pattern_text"));
                let sort_on_template = truthy("index.logsdb.sort_on_message_template") || message_pattern;
                let set_template_sort = sort_on_template && (truthy("index.logsdb.default_sort_on_message_template") || truthy("index.logsdb.sort_on_message_template"));
                if add_host_name && host_name_field.is_none() {
                    let props_mut = mappings["properties"].as_object_mut().unwrap();
                    if props_mut.contains_key("host") && props_mut["host"].get("properties").is_some() {
                        props_mut["host"]["properties"]["name"] = json!({"type": "keyword", "ignore_above": 1024});
                    } else if props_mut.contains_key("host") {
                        props_mut.insert("host.name".into(), json!({"type": "keyword", "ignore_above": 1024}));
                    } else {
                        props_mut.insert("host".into(), json!({"properties": {"name": {"type": "keyword", "ignore_above": 1024}}}));
                    }
                }
                if add_host_name {
                    flat.insert("index.logsdb.add_host_name_field".into(), json!("true"));
                }
                if sort_on_host {
                    flat.insert("index.logsdb.sort_on_host_name".into(), json!("true"));
                }
                if set_template_sort {
                    flat.insert("index.logsdb.sort_on_message_template".into(), json!("true"));
                }
            }
            // Validate sort fields: reject alias, keyword without doc values,
            // and non-keyword types. Multi-fields (e.g. host.name.keyword)
            // are resolved via the `fields` sub-object.
            for sf in &sort_fields {
                if sf == "@timestamp" {
                    continue;
                }
                // Resolve the sort field: try flat key first, then nested
                // `properties` traversal.
                let field = lookup_field_any(&mappings["properties"], sf);
                match field {
                    Some(f) => {
                        let ftype = f.get("type").and_then(Value::as_str).unwrap_or("");
                        if ftype == "alias" {
                            return Err(crate::error::EsError::illegal_argument(format!("Cannot use alias [{}] as an index sort field", sf)));
                        }
                        if ftype == "keyword" {
                            if f.get("doc_values").and_then(Value::as_bool) == Some(false) {
                                return Err(crate::error::EsError::illegal_argument(format!("docvalues not found for index sort field:[{}]", sf)));
                            }
                        } else if !ftype.is_empty() && ftype != "date" {
                            // Check if this is a multi-field (e.g. text with a keyword sub-field).
                            let parts: Vec<&str> = sf.split('.').collect();
                            let last = parts.last().copied().unwrap_or("");
                            let parent_path = parts[..parts.len().saturating_sub(1)].join(".");
                            let parent = if parent_path.is_empty() { None } else { lookup_field_any(&mappings["properties"], &parent_path) };
                            let is_multi_field = parent.as_ref().and_then(|p| p.get("fields")).and_then(|fl| fl.get(last)).map(|mf| mf.get("type").and_then(Value::as_str) == Some("keyword")).unwrap_or(false);
                            if !is_multi_field {
                                return Err(crate::error::EsError::illegal_argument(format!("Cannot use field [{}] of type [{}] as an index sort field", sf, ftype)));
                            }
                        }
                    }
                    None => {
                        // Field not in mapping: if it's host.name and we just
                        // injected it, it's fine. Otherwise skip (dynamic).
                    }
                }
            }
        }
        if !mappings.is_object() {
            mappings = json!({});
        }
        if mappings.get("properties").and_then(|p| p.get(TIMESTAMP_FIELD)).is_none() {
            mappings["properties"][TIMESTAMP_FIELD] = json!({"type": "date"});
        }
        mappings["_data_stream_timestamp"] = json!({"enabled": true});
        let settings = crate::settings::unflatten(&flat);
        let name = backing_index_name(ds, generation, crate::date::now_millis());
        // Membership and DLM/ILM precedence are published by the caller.
        // The lifecycle executor initializes only after that context exists.
        self.create_index_impl(&name, &json!({"settings": settings, "mappings": mappings}), None, false)
    }

    /// `DataStreamIndexSettingsProvider` for a time series backing index:
    /// its `[start_time, end_time)` window (unless the template sets one;
    /// `bounds` forces a past window), and sequence numbers off / synthetic
    /// ids on by default.
    fn time_series_backing_settings(&self, ds: &str, flat: &mut crate::settings::FlatSettings, bounds: Option<(i64, i64)>) {
        if flat.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase).as_deref() != Some("time_series") {
            return;
        }
        let ms = |k: &str, d: i64| flat.get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| Some(v.to_string()))).and_then(|v| crate::date::parse_time_value(&v)).unwrap_or(d);
        let look_ahead = ms("index.look_ahead_time", 9 * 60_000).min(2 * 3_600_000);
        let look_back = ms("index.look_back_time", 2 * 3_600_000);
        let now = crate::date::now_millis();
        let canonical = |t: i64| t - t.rem_euclid(1000);
        let (start, end) = match bounds {
            Some(b) => b,
            None => {
                // A rollover continues from the write index's end time.
                let prev_end = self.data_stream(ds).and_then(|d| self.get_index(d.write_index())).and_then(|w| w.setting(crate::tsdb::END_TIME)).and_then(|e| {
                    crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok().and_then(|f| f.parse(&e).ok())
                });
                match prev_end {
                    Some(start) => (start, canonical(if start > now { start + look_ahead } else { now + look_ahead })),
                    None => (canonical(now - look_back), canonical(now + look_ahead)),
                }
            }
        };
        if bounds.is_some() || !flat.contains_key(crate::tsdb::START_TIME) {
            flat.insert(crate::tsdb::START_TIME.into(), json!(crate::ilm::iso_date(start)));
        }
        if bounds.is_some() || !flat.contains_key(crate::tsdb::END_TIME) {
            flat.insert(crate::tsdb::END_TIME.into(), json!(crate::ilm::iso_date(end)));
        }
        // `cluster.time_series.support_seq_no_disabled` /
        // `support_synthetic_id` (on): 9.5.5 backing indices.
        flat.entry("index.disable_sequence_numbers".to_string()).or_insert(json!("true"));
        flat.entry("index.mapping.synthetic_id".to_string()).or_insert(json!("true"));
    }

    fn create_failure_index(&self, ds: &str, generation: u64) -> Result<Arc<IndexService>> {
        let (settings, _, _) = self.template_for(ds);
        let mut flat: crate::settings::FlatSettings = crate::settings::normalize_index_settings(&settings).into_iter().filter(|(k, _)| failure_store_setting(k)).collect();
        flat.insert("index.hidden".into(), json!("true"));
        flat.insert("index.failure_store.version".into(), json!("1"));
        flat.entry("index.routing.allocation.include._tier_preference".to_string()).or_insert(json!("data_hot"));
        // `data_streams.failure_store.refresh_interval` (30s on stateless
        // nodes) is a request setting: it wins over the template's.
        if let Some(interval) = crate::stateless::failure_store_refresh_interval(self) {
            flat.insert("index.refresh_interval".into(), json!(interval));
        }
        let name = failure_index_name(ds, generation, crate::date::now_millis());
        self.create_index_impl(&name, &json!({"settings": crate::settings::unflatten(&flat), "mappings": failure_store_mappings()}), None, false)
    }

    /// Create the next failure store write index (generation + 1, or the
    /// current generation when `same_generation`). Returns it.
    fn roll_failure_store(&self, ds: &str, same_generation: bool) -> Result<Arc<IndexService>> {
        let _g = self.ds_create_lock.lock();
        let d = self.data_stream(ds).ok_or_else(|| EsError::index_not_found(ds))?;
        let generation = if same_generation { d.generation } else { d.generation + 1 };
        let created = self.create_failure_index(ds, generation)?;
        {
            let mut streams = self.data_streams.write();
            if let Some(d) = streams.get_mut(ds) {
                d.generation = generation;
                d.failure_indices.push(created.name.clone());
                d.failure_rollover_on_write = false;
            }
        }
        self.persist_state()?;
        Ok(created)
    }

    /// Roll a data stream over to a new backing write index.
    pub fn roll_data_stream(&self, ds: &str) -> Result<Arc<IndexService>> {
        let _g = self.ds_create_lock.lock();
        let d = self.data_stream(ds).ok_or_else(|| EsError::index_not_found(ds))?;
        let old = self.get_index(d.write_index());
        let created = self.create_backing_index(ds, d.generation + 1)?;
        // Data stream aliases apply to every backing index, including new ones.
        if let Some(old) = &old {
            created.set_aliases(old.aliases())?;
        }
        let mode = self.template_index_mode(ds);
        {
            let mut streams = self.data_streams.write();
            if let Some(d) = streams.get_mut(ds) {
                d.index_mode = mode;
                d.generation += 1;
                d.indices.push(created.name.clone());
                d.rollover_on_write = false;
            }
        }
        self.persist_state()?;
        Ok(created)
    }

    /// The write index of a data stream, performing a pending lazy rollover.
    pub fn data_stream_write_index(&self, ds: &DataStream) -> Result<Arc<IndexService>> {
        if ds.rollover_on_write {
            if let Some(d) = self.data_stream(&ds.name).filter(|d| d.rollover_on_write) {
                let old = self.get_index(d.write_index());
                let created = self.roll_data_stream(&d.name)?;
                if let Some(old) = old {
                    crate::ilm::runner::record_rollover(self, &old, &d.name, &json!({}));
                }
                return Ok(created);
            }
            return self.data_stream(&ds.name).and_then(|d| self.get_index(d.write_index())).ok_or_else(|| EsError::index_not_found(&ds.name));
        }
        self.get_index(ds.write_index()).ok_or_else(|| EsError::index_not_found(&ds.name))
    }

    /// The failure store write index of `ds`, created (or lazily rolled
    /// over) on demand.
    pub fn failure_store_write_index(&self, ds: &str) -> Result<Arc<IndexService>> {
        let d = self.data_stream(ds).ok_or_else(|| EsError::index_not_found(ds))?;
        if !d.failure_needs_rollover() {
            if let Some(svc) = d.failure_write_index().and_then(|n| self.get_index(n)) {
                return Ok(svc);
            }
        }
        self.roll_failure_store(ds, false)
    }

    /// Write a failed document to `ds`'s failure store (auto-creating the
    /// data stream, with an initialized failure store, when missing).
    pub fn store_failure(
        &self,
        ds: &str,
        original_index: &str,
        id: Option<&str>,
        routing: Option<&str>,
        source: &str,
        e: &EsError,
        cx: Option<&FailureContext>,
    ) -> Result<(Arc<IndexService>, crate::shard::WriteResult)> {
        let src: Value = serde_json::from_str(source).map_err(|_| e.clone())?;
        if !src.is_object() {
            return Err(e.clone());
        }
        let svc = if self.data_stream(ds).is_none() {
            match self.create_data_stream_ext(ds, true) {
                Ok(_) => {}
                Err(err) if err.kind == "resource_already_exists_exception" => {}
                Err(err) => return Err(err),
            }
            self.failure_store_write_index(ds)?
        } else {
            self.failure_store_write_index(ds)?
        };
        let doc = failure_document(original_index, id, routing, &src, e, cx);
        let cond = crate::shard::WriteCond { create: true, ..Default::default() };
        let r = svc.write(vec![crate::index::WriteReq::Index { id: None, source: doc.to_string(), routing: None, cond }], false).remove(0)?;
        Ok((svc, r))
    }

    /// `PUT /_data_stream/{name}` (and auto-creation on first write).
    pub fn create_data_stream(&self, name: &str) -> Result<DataStream> {
        self.create_data_stream_ext(name, false)
    }

    /// Create a data stream; `init_failure_store` also creates its first
    /// failure store index (same generation), as ES does when the first
    /// write to a new data stream is a redirected ingest failure.
    pub fn create_data_stream_ext(&self, name: &str, init_failure_store: bool) -> Result<DataStream> {
        validate_ds_name(name)?;
        if crate::esql::views::exists(self, name) {
            return Err(EsError::illegal_argument(format!("data_stream [{name}] already exists as an ESQL view")));
        }
        let exists = || EsError::new(400, "resource_already_exists_exception", format!("data_stream [{name}] already exists"));
        // Never hold the data stream lock while creating indices: resolve()
        // takes the index lock then the data stream lock.
        let _g = self.ds_create_lock.lock();
        if self.data_streams.read().contains_key(name) {
            return Err(exists());
        }
        if let Some(svc) = self.get_index(name) {
            return Err(EsError::index_already_exists(name, &svc.uuid));
        }
        let (tname, tmpl) = {
            let t = self.templates.read();
            let best = t
                .index
                .iter()
                .filter(|(_, v)| pattern_list(&v["index_patterns"]).iter().any(|p| crate::mapping::wildcard_match(p, name)))
                .max_by_key(|(_, v)| v["priority"].as_i64().unwrap_or(0))
                .map(|(k, v)| (k.clone(), v.clone()));
            match best {
                Some(b) => b,
                None => return Err(EsError::illegal_argument(format!("no matching index template found for data stream [{name}]"))),
            }
        };
        if !tmpl.get("data_stream").map(|d| d.is_object()).unwrap_or(false) {
            return Err(EsError::illegal_argument(format!("matching index template [{tname}] for data stream [{name}] has no data stream template")));
        }
        let svc = self.create_backing_index(name, 1)?;
        let mut ds = DataStream::new_named(name, tname, &tmpl);
        ds.indices = vec![svc.name.clone()];
        ds.index_mode = self.template_index_mode(name);
        ds.lifecycle = crate::ilm::dlm::template_lifecycle(self, &tmpl);
        ds.options = self.template_data_stream_options(&tmpl).filter(|o| o.as_object().map(|m| !m.is_empty()).unwrap_or(false));
        if init_failure_store {
            let fs = self.create_failure_index(name, 1)?;
            ds.failure_indices.push(fs.name.clone());
        }
        self.data_streams.write().insert(name.to_string(), ds.clone());
        self.persist_state()?;
        Ok(ds)
    }

    /// `DELETE /_data_stream/{name}` (comma list / wildcards).
    pub fn delete_data_streams(&self, expr: &str) -> Result<()> {
        reject_selectors(expr)?;
        let mut names = Vec::new();
        for p in expr.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if p.contains('*') {
                names.extend(self.data_streams_matching(p, true).into_iter().map(|d| d.name));
            } else if self.data_streams.read().contains_key(p) {
                names.push(p.to_string());
            } else {
                return Err(EsError::index_not_found(p));
            }
        }
        for n in names {
            let ds = self.data_streams.write().remove(&n);
            if let Some(ds) = ds {
                for i in ds.indices.iter().chain(ds.failure_indices.iter()) {
                    if self.get_index(i).is_some() {
                        self.delete_index(i)?;
                    }
                }
            }
        }
        self.persist_state()
    }

    /// `GET /_data_stream[/{name}]` (every stream, hidden included).
    pub fn get_data_streams(&self, expr: Option<&str>) -> Result<Value> {
        self.get_data_streams_opts(expr, true, false)
    }

    /// `GET /_data_stream[/{name}]` with `expand_wildcards` (hidden) and
    /// `include_defaults`.
    pub fn get_data_streams_opts(&self, expr: Option<&str>, include_hidden: bool, include_defaults: bool) -> Result<Value> {
        let out = crate::ilm::dlm::resolve_data_streams(self, expr.unwrap_or(""), include_hidden)?;
        let mut arr = Vec::new();
        let index_mode = |s: &IndexService| s.setting("index.mode").filter(|m| !m.is_empty()).unwrap_or_else(|| "standard".into());
        for ds in out {
            // RED when a backing primary failed to allocate, YELLOW when a
            // replica cannot be allocated ([`Node::unassigned_replicas`]).
            let mut green = true;
            let mut red = false;
            let indices: Vec<Value> = ds
                .indices
                .iter()
                .map(|i| match self.get_index(i) {
                    Some(s) => {
                        red |= !s.failed_shards().is_empty();
                        green &= self.unassigned_replicas(&s) == 0;
                        let mut v = Map::new();
                        v.insert("index_name".into(), json!(i));
                        v.insert("index_uuid".into(), json!(s.uuid));
                        if let Some(p) = crate::ilm::runner::policy_name(&s) {
                            v.insert("ilm_policy".into(), json!(p));
                        }
                        v.insert("managed_by".into(), json!(crate::ilm::dlm::managed_by(self, &s)));
                        v.insert("prefer_ilm".into(), json!(s.setting("index.lifecycle.prefer_ilm").map(|p| p != "false").unwrap_or(true)));
                        v.insert("index_mode".into(), json!(index_mode(&s)));
                        Value::Object(v)
                    }
                    None => json!({"index_name": i, "index_uuid": "", "managed_by": crate::ilm::dlm::UNMANAGED, "prefer_ilm": true, "index_mode": "standard"}),
                })
                .collect();
            let fs_lifecycle = self.failure_store_lifecycle(&ds);
            let fs_managed = fs_lifecycle.as_ref().map(|l| l["enabled"].as_bool().unwrap_or(true)).unwrap_or(false);
            let failure_indices: Vec<Value> = ds
                .failure_indices
                .iter()
                .map(|i| {
                    let s = self.get_index(i);
                    if let Some(s) = &s {
                        green &= self.unassigned_replicas(s) == 0;
                    }
                    let mut v = Map::new();
                    v.insert("index_name".into(), json!(i));
                    v.insert("index_uuid".into(), json!(s.as_ref().map(|s| s.uuid.clone()).unwrap_or_default()));
                    if let Some(p) = s.as_ref().and_then(|s| crate::ilm::runner::policy_name(s)) {
                        v.insert("ilm_policy".into(), json!(p));
                    }
                    v.insert("managed_by".into(), json!(if fs_managed { crate::ilm::dlm::MANAGED_DLM } else { crate::ilm::dlm::UNMANAGED }));
                    v.insert("index_mode".into(), json!(s.as_ref().map(|s| index_mode(s)).unwrap_or_else(|| "standard".into())));
                    Value::Object(v)
                })
                .collect();
            let (tpl_policy, tpl_prefer) = crate::ilm::dlm::template_ilm(self, &ds.name);
            let mut v = serde_json::Map::new();
            v.insert("name".into(), json!(ds.name));
            v.insert("timestamp_field".into(), json!({"name": TIMESTAMP_FIELD}));
            v.insert("indices".into(), Value::Array(indices));
            v.insert("generation".into(), json!(ds.generation));
            if let Some(m) = &ds.meta {
                v.insert("_meta".into(), m.clone());
            }
            v.insert("status".into(), json!(if red { "RED" } else if green { "GREEN" } else { "YELLOW" }));
            v.insert("template".into(), json!(ds.template));
            if let Some(lc) = &ds.lifecycle {
                v.insert("lifecycle".into(), crate::ilm::dlm::render_lifecycle(self, &ds.name, lc, include_defaults));
            }
            if let Some(p) = tpl_policy {
                v.insert("ilm_policy".into(), json!(p));
            }
            v.insert("next_generation_managed_by".into(), json!(crate::ilm::dlm::next_generation_managed_by(self, &ds)));
            v.insert("prefer_ilm".into(), json!(tpl_prefer));
            v.insert("hidden".into(), json!(ds.hidden));
            v.insert("system".into(), json!(false));
            v.insert("allow_custom_routing".into(), json!(ds.allow_custom_routing));
            v.insert("replicated".into(), json!(ds.replicated));
            v.insert("rollover_on_write".into(), json!(ds.rollover_on_write));
            // `TransportGetDataStreamsAction`: the stream's own index mode,
            // else what the current template (and providers) would pick.
            let mode = ds.index_mode.clone().or_else(|| self.resolve_data_stream_index_mode(&ds.name)).unwrap_or_else(|| "standard".into());
            v.insert("index_mode".into(), json!(mode));
            if let Some(ts) = self.tsds_temporal_ranges(&ds) {
                v.insert("time_series".into(), json!({"temporal_ranges": ts}));
            }
            v.insert("settings".into(), settings_json(&ds.settings));
            v.insert("mappings".into(), Value::Object(ds.mappings.clone()));
            let mut fs = Map::new();
            fs.insert("enabled".into(), json!(self.failure_store_enabled(&ds)));
            fs.insert("rollover_on_write".into(), json!(ds.failure_needs_rollover()));
            fs.insert("indices".into(), Value::Array(failure_indices));
            if let Some(lc) = &fs_lifecycle {
                fs.insert("lifecycle".into(), self.render_failures_lifecycle(lc, include_defaults));
            }
            v.insert("failure_store".into(), Value::Object(fs));
            arr.push(Value::Object(v));
        }
        Ok(json!({ "data_streams": arr }))
    }

    /// The failure store lifecycle of a data stream: the configured one, or
    /// the default (enabled) one when the failure store is enabled.
    pub fn failure_store_lifecycle(&self, ds: &DataStream) -> Option<Value> {
        let configured = ds.options.as_ref().and_then(|o| o["failure_store"].get("lifecycle")).filter(|l| l.is_object()).cloned();
        match configured {
            Some(lc) => Some(crate::ilm::dlm::normalize_lifecycle(&lc).unwrap_or(json!({"enabled": true}))),
            None if self.failure_store_enabled(ds) => Some(json!({"enabled": true})),
            None => None,
        }
    }

    /// Failures lifecycle as `GET _data_stream` renders it: effective
    /// retention falls back to `data_streams.lifecycle.retention.failures_default` (30d).
    pub fn render_failures_lifecycle(&self, lc: &Value, include_defaults: bool) -> Value {
        let enabled = lc["enabled"].as_bool().unwrap_or(true);
        let mut m = Map::new();
        m.insert("enabled".into(), json!(enabled));
        if !enabled {
            return Value::Object(m);
        }
        let own = lc.get("data_retention").and_then(Value::as_str).map(str::to_string);
        if let Some(r) = &own {
            m.insert("data_retention".into(), json!(r));
        }
        let (_, max) = crate::ilm::dlm::global_retention(self);
        let ms = |r: &str| crate::date::parse_time_value(r).unwrap_or(i64::MAX);
        let failures_default = crate::ilm::cluster_setting(self, "data_streams.lifecycle.retention.failures_default").filter(|s| !s.is_empty()).unwrap_or_else(|| "30d".into());
        let eff = match (own, max) {
            (Some(r), Some(m)) if ms(&m) < ms(&r) => Some((m, "max_global_retention")),
            (Some(r), _) => Some((r, "data_stream_configuration")),
            (None, Some(m)) if ms(&m) < ms(&failures_default) => Some((m, "max_global_retention")),
            (None, _) if failures_default == "-1" => None,
            (None, _) => Some((failures_default, "default_failures_retention")),
        };
        if let Some((r, src)) = &eff {
            m.insert("effective_retention".into(), json!(r));
            m.insert("retention_determined_by".into(), json!(src));
        }
        if include_defaults {
            m.insert("rollover".into(), crate::ilm::dlm::rollover_defaults(eff.as_ref().map(|(r, _)| r.as_str())));
        }
        Value::Object(m)
    }

    /// `GET /_data_stream/{name}/_options`.
    pub fn get_data_stream_options(&self, expr: &str, include_hidden: bool) -> Result<Value> {
        reject_selectors(expr)?;
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        let arr: Vec<Value> = dss
            .iter()
            .map(|d| {
                let mut m = Map::new();
                m.insert("name".into(), json!(d.name));
                if let Some(o) = d.options.as_ref().filter(|o| o.as_object().map(|m| !m.is_empty()).unwrap_or(false)) {
                    m.insert("options".into(), o.clone());
                }
                Value::Object(m)
            })
            .collect();
        Ok(json!({ "data_streams": arr }))
    }

    /// `PUT /_data_stream/{name}/_options` (`None` = DELETE).
    pub fn put_data_stream_options(&self, expr: &str, include_hidden: bool, body: Option<&Value>) -> Result<()> {
        reject_selectors(expr)?;
        let options = match body {
            None => None,
            Some(b) => {
                let o = b.as_object().ok_or_else(|| EsError::x_content_parse("[1:1] [put_data_stream_options_request] expected an object"))?;
                validate_data_stream_options(b, "put_data_stream_options_request")?;
                if o.get("failure_store").map(|f| f.is_null()).unwrap_or(true) {
                    return Err(EsError::action_validation(&["At least one option needs to be provided"]));
                }
                let mut fs = o["failure_store"].as_object().cloned().unwrap_or_default();
                fs.retain(|_, v| !v.is_null());
                if let Some(e) = fs.get("enabled").and_then(Value::as_str).map(|s| s == "true") {
                    fs.insert("enabled".into(), json!(e));
                }
                if let Some(lc) = fs.get("lifecycle").cloned() {
                    fs.insert("lifecycle".into(), crate::ilm::dlm::normalize_lifecycle(&lc)?);
                }
                Some(json!({ "failure_store": fs }))
            }
        };
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        {
            let mut streams = self.data_streams.write();
            for d in &dss {
                if let Some(s) = streams.get_mut(&d.name) {
                    s.options = options.clone();
                }
            }
        }
        self.persist_state()
    }

    /// `GET /_data_stream/{name}/_settings`.
    pub fn get_data_stream_settings(&self, expr: &str, include_hidden: bool) -> Result<Value> {
        reject_selectors(expr)?;
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        let arr: Vec<Value> = dss
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "settings": settings_json(&d.settings),
                    "effective_settings": crate::settings::unflatten(&self.data_stream_effective_settings(&d.name))
                })
            })
            .collect();
        Ok(json!({ "data_streams": arr }))
    }

    /// `PUT /_data_stream/{name}/_settings`: per data stream result with
    /// `applied_to_data_stream`, `error`, and which settings also reached
    /// the existing backing indices.
    pub fn put_data_stream_settings(&self, expr: &str, include_hidden: bool, body: &Value, dry_run: bool) -> Result<Value> {
        reject_selectors(expr)?;
        let dss = crate::ilm::dlm::resolve_data_streams(self, expr, include_hidden)?;
        // Keep request order for the error list (ES iterates the request's settings).
        let mut requested: Vec<(String, Value)> = Vec::new();
        fn walk(prefix: &str, v: &Value, out: &mut Vec<(String, Value)>) {
            match v {
                Value::Object(m) => {
                    for (k, v) in m {
                        let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                        walk(&key, v, out);
                    }
                }
                Value::Null => out.push((prefix.to_string(), Value::Null)),
                Value::String(s) => out.push((prefix.to_string(), json!(s))),
                other => out.push((prefix.to_string(), json!(other.to_string()))),
            }
        }
        let body = body.get("settings").filter(|_| body.as_object().map(|o| o.len() == 1).unwrap_or(false)).unwrap_or(body);
        walk("", body, &mut requested);
        for (k, _) in requested.iter_mut() {
            if !k.starts_with("index.") {
                *k = format!("index.{k}");
            }
        }
        let bad: Vec<&str> = requested.iter().map(|(k, _)| k.as_str()).filter(|k| !DS_SETTINGS_BACKING.contains(k) && !DS_SETTINGS_DS_ONLY.contains(k) && !DS_SETTINGS_WRITE.contains(k)).collect();
        let mut arr = Vec::new();
        for d in &dss {
            if !bad.is_empty() {
                arr.push(json!({
                    "name": d.name, "applied_to_data_stream": false,
                    "error": format!("Cannot set the following settings on a data stream: [{}]", bad.join(",")),
                    "settings": {}, "effective_settings": {},
                    "index_settings_results": {"applied_to_data_stream_only": [], "applied_to_data_stream_and_write_indices": [], "applied_to_data_stream_and_backing_indices": []}
                }));
                continue;
            }
            let mut new_settings = d.settings.clone();
            for (k, v) in &requested {
                if v.is_null() {
                    new_settings.remove(k);
                } else {
                    new_settings.insert(k.clone(), v.clone());
                }
            }
            let ds_only: Vec<&str> = requested.iter().map(|(k, _)| k.as_str()).filter(|k| DS_SETTINGS_DS_ONLY.contains(k)).collect();
            let backing: Vec<&str> = requested.iter().map(|(k, _)| k.as_str()).filter(|k| DS_SETTINGS_BACKING.contains(k)).collect();
            let write: Vec<&str> = requested.iter().map(|(k, _)| k.as_str()).filter(|k| DS_SETTINGS_WRITE.contains(k)).collect();
            if !dry_run {
                if let Some(s) = self.data_streams.write().get_mut(&d.name) {
                    s.settings = new_settings.clone();
                }
                self.persist_state()?;
                // Settings that apply to the existing backing indices take
                // their new effective value (template value once nulled).
                let eff = self.data_stream_effective_settings(&d.name);
                let flat: crate::settings::FlatSettings = backing.iter().map(|k| (k.to_string(), eff.get(*k).cloned().unwrap_or(Value::Null))).collect();
                let flat_write: crate::settings::FlatSettings = backing.iter().chain(write.iter()).map(|k| (k.to_string(), eff.get(*k).cloned().unwrap_or(Value::Null))).collect();
                for i in &d.indices {
                    let f = if i == d.write_index() { &flat_write } else { &flat };
                    if f.is_empty() {
                        continue;
                    }
                    if let Some(svc) = self.get_index(i) {
                        svc.put_settings(f.clone(), false)?;
                    }
                }
            }
            let mut eff = {
                let (settings, _, _) = self.template_for(&d.name);
                let mut f = crate::settings::normalize_index_settings(&settings);
                f.retain(|_, v| !v.is_null());
                f
            };
            for (k, v) in &new_settings {
                eff.insert(k.clone(), v.clone());
            }
            arr.push(json!({
                "name": d.name, "applied_to_data_stream": true,
                "settings": settings_json(&new_settings),
                "effective_settings": crate::settings::unflatten(&eff),
                "index_settings_results": {"applied_to_data_stream_only": ds_only, "applied_to_data_stream_and_write_indices": write, "applied_to_data_stream_and_backing_indices": backing}
            }));
        }
        Ok(json!({ "data_streams": arr }))
    }

    /// Called before an index is deleted: the write index of a data stream
    /// cannot be deleted; any other backing index leaves its stream
    /// (`DataStream.removeBackingIndex`, generation + 1).
    pub fn detach_backing_index(&self, index: &str) -> Result<()> {
        let (_, changed) = self.detach_backing_index_with(index, || Ok(()))?;
        if changed { self.persist_state()?; }
        Ok(())
    }

    /// Publish removal only after the local index registry has stopped exposing
    /// the deleted generation. The callback must only update the registry: do
    /// not fence shards, persist state or wait for readers while holding this lock.
    pub(crate) fn detach_backing_index_with<T>(&self, index: &str, remove: impl FnOnce() -> Result<T>) -> Result<(T, bool)> {
        let mut streams = self.data_streams.write();
        for d in streams.values() {
            if d.failure_indices.iter().any(|i| i == index) {
                if d.failure_write_index() == Some(index) && d.failure_indices.len() > 1 {
                    return Err(EsError::illegal_argument(format!(
                        "index [{index}] is the failure store write index for data stream [{}] and cannot be deleted", d.name
                    )));
                }
            } else if d.indices.iter().any(|i| i == index) && d.write_index() == index {
                return Err(EsError::illegal_argument(format!(
                    "index [{index}] is the write index for data stream [{}] and cannot be deleted", d.name
                )));
            }
        }
        let removed = remove()?;
        let mut changed = false;
        for d in streams.values_mut() {
            if d.failure_indices.iter().any(|i| i == index) {
                d.failure_indices.retain(|i| i != index);
                d.generation += 1;
                changed = true;
            } else if d.indices.iter().any(|i| i == index) {
                d.indices.retain(|i| i != index);
                d.generation += 1;
                changed = true;
            }
        }
        Ok((removed, changed))
    }

    /// Validate a write addressed to `target`: data streams only accept
    /// `op_type=create` with an `@timestamp`.
    /// Direct writes to a backing index are refused (`op_type` as requested).
    /// `cas`: `if_seq_no` or `if_primary_term` is set (an `op_type=index`
    /// compare-and-set overwrite of a backing index is allowed).
    pub fn check_backing_write(&self, target: &str, op_create: bool, cas: bool) -> Result<()> {
        if !op_create && cas {
            return Ok(());
        }
        if let Some(ds) = self.data_stream_of_index(target) {
            let what = if op_create { "op_type=create" } else { "op_type=index and no if_primary_term and if_seq_no set" };
            return Err(EsError::illegal_argument(format!("index request with {what} targeting backing indices is disallowed, target corresponding data stream [{ds}] instead")));
        }
        Ok(())
    }

    /// `[start, end)` (millis and as configured) of a time series data
    /// stream's backing indices; `None` for any other target.
    fn tsds_ranges(&self, target: &str) -> Option<Vec<(i64, i64, String, String)>> {
        let indices = self.data_streams.read().get(target).map(|d| d.indices.clone())?;
        let write = self.get_index(indices.last()?)?;
        if write.setting("index.mode").as_deref() != Some("time_series") {
            return None;
        }
        let f = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok()?;
        let ranges: Vec<_> = indices
            .iter()
            .filter_map(|i| self.get_index(i))
            .filter_map(|s| {
                let (a, b) = (s.setting("index.time_series.start_time")?, s.setting("index.time_series.end_time")?);
                Some((f.parse(&a).ok()?, f.parse(&b).ok()?, a, b))
            })
            .collect();
        Some(ranges)
    }

    /// `time_series.temporal_ranges` of a time series data stream: the
    /// backing indices' `[start_time, end_time]` bounds, sorted and merged
    /// where one index's end equals the next's start
    /// (`TransportGetDataStreamsAction`). `None` for any other index mode.
    fn tsds_temporal_ranges(&self, ds: &DataStream) -> Option<Vec<Value>> {
        let write = self.get_index(ds.write_index())?;
        if write.setting("index.mode").as_deref() != Some("time_series") {
            return None;
        }
        let mut ranges: Vec<(i64, i64)> = Vec::new();
        for i in &ds.indices {
            let Some(s) = self.get_index(i) else { continue };
            if s.setting("index.mode").as_deref() != Some("time_series") {
                continue;
            }
            let Some((a, b)) = crate::tsdb::bounds(&s.meta().settings) else { continue };
            if let (Some(a), Some(b)) = (a, b) {
                ranges.push((a, b));
            }
        }
        ranges.sort();
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (s, e) in ranges {
            match merged.last_mut() {
                Some(last) if last.1 == s => last.1 = e,
                Some(last) if last.1 < s => merged.push((s, e)),
                Some(_) => {} // colliding / overlapping range: keep the first
                None => merged.push((s, e)),
            }
        }
        Some(merged.into_iter().map(|(s, e)| json!({"start": crate::ilm::iso_date(s), "end": crate::ilm::iso_date(e)})).collect())
    }

    pub fn check_data_stream_write(&self, target: &str, create: bool, source: Option<&str>) -> Result<()> {
        self.check_data_stream_write_in(target, create, source, true).map(|_| ())
    }

    /// [`Node::check_data_stream_write`], returning the document's
    /// `@timestamp` (the first of the root field's values) it read, for
    /// [`Node::time_series_write_index_at`]: one scan of the source serves
    /// the checks and the routing.
    pub fn check_data_stream_write_ts(&self, target: &str, create: bool, source: Option<&str>) -> Result<Option<Value>> {
        self.check_data_stream_write_in(target, create, source, true)
    }

    /// [`Node::check_data_stream_write`] without creating a past backing
    /// index: the cluster coordinator's check of a write it could not route
    /// to a backing index (the master creates past indices).
    pub fn check_data_stream_write_only(&self, target: &str, create: bool, source: Option<&str>) -> Result<()> {
        self.check_data_stream_write_in(target, create, source, false).map(|_| ())
    }

    /// [`Node::check_data_stream_write_only`] of a source already scanned.
    pub fn check_data_stream_write_only_probed(&self, target: &str, create: bool, source: Option<(&str, &TimestampScan)>) -> Result<()> {
        self.check_data_stream_write_scanned(target, create, source.map(|(s, p)| (s, p.probe.clone())), false).map(|_| ())
    }

    /// [`Node::time_series_write_index`] of a source already scanned.
    pub fn time_series_write_index_probed(&self, target: &str, scan: &TimestampScan) -> Option<Arc<IndexService>> {
        if !self.data_streams.read().contains_key(target) {
            return None;
        }
        self.time_series_write_index_at(target, scan.doc_ts.as_ref())
    }

    fn check_data_stream_write_in(&self, target: &str, create: bool, source: Option<&str>, allow_past: bool) -> Result<Option<Value>> {
        // Neither a data stream nor a create: decided without the source.
        if !self.data_streams.read().contains_key(target) || !create {
            return self.check_data_stream_write_scanned(target, create, None, allow_past);
        }
        // One parse serves every check below (a bulk item's source was
        // parsed three or four times here, before the shard parses it).
        // A probe that keeps only the root `@timestamp` (the rest is
        // scanned, not built into a tree: a 10 KB metrics doc's
        // `Value` cost more than the probe of every item of a bulk).
        let scanned = source.map(|src| (src, serde_json::from_str::<TimestampProbe>(src).ok()));
        self.check_data_stream_write_scanned(target, create, scanned, allow_past)
    }

    fn check_data_stream_write_scanned(&self, target: &str, create: bool, source: Option<(&str, Option<TimestampProbe>)>, allow_past: bool) -> Result<Option<Value>> {
        if !self.data_streams.read().contains_key(target) {
            return Ok(None);
        }
        let mut first_ts: Option<Value> = None;
        if !create {
            return Err(EsError::illegal_argument("only write ops with an op_type of create are allowed in data streams"));
        }
        if let Some((src, probe)) = source {
            if let Some(kind) = probe.as_ref().and_then(|p| p.not_object) {
                let found = if kind { "START_ARRAY" } else { "VALUE_STRING" };
                return Err(EsError::illegal_argument(format!(
                    "Error extracting data stream timestamp field: Failed to parse object: expecting token of type [START_OBJECT] but found [{found}]"
                )));
            }
            // `source_values` of the root field: arrays flattened, nulls dropped.
            fn leaves<'v>(v: &'v Value, out: &mut Vec<&'v Value>) {
                match v {
                    Value::Array(a) => a.iter().for_each(|x| leaves(x, out)),
                    Value::Null => {}
                    other => out.push(other),
                }
            }
            let ts_values: Option<Vec<&Value>> = probe.as_ref().map(|p| {
                let mut out = Vec::new();
                if let Some(v) = &p.value {
                    leaves(v, &mut out);
                }
                out
            });
            let has_ts = match (probe.as_ref(), ts_values.as_ref()) {
                (Some(p), Some(values)) => p.value.as_ref().is_some_and(|x| !x.is_null()) || values.iter().any(|x| !x.is_null()),
                _ => true,
            };
            first_ts = ts_values.as_ref().and_then(|values| values.first().map(|x| (*x).clone()));
            if ts_values.as_ref().is_some_and(|values| values.len() > 1) {
                let msg = format!("data stream timestamp field [{TIMESTAMP_FIELD}] encountered multiple values");
                return Err(EsError::document_parsing(format!("[1:1] failed to parse: {msg}")).caused_by(EsError::illegal_argument(msg)));
            }
            // A time series data stream routes on `@timestamp` before the
            // shard sees the document (`timestamp_error`).
            if let Some(ranges) = self.tsds_ranges(target) {
                let ts = ts_values.as_ref().and_then(|values| values.first().map(|x| (*x).clone())).filter(|v| !v.is_null());
                let Some(ts) = ts else {
                    let col = src.trim_end().chars().count();
                    let msg = "Failed to parse object: expecting token of type [START_OBJECT] but found [null]";
                    return Err(EsError::new(400, "timestamp_error", format!("Error extracting data stream timestamp field: {msg}"))
                        .caused_by(EsError::new(400, "parsing_exception", msg).with("line", 1).with("col", col)));
                };
                let millis = TS_FORMATS.with(|(doc, _)| match &ts {
                    Value::Number(n) => n.as_i64(),
                    Value::String(s) => doc.as_ref().and_then(|f| f.parse(s).ok()),
                    _ => None,
                });
                if let Some(ms) = millis {
                    let mut ok = ranges.iter().any(|(s, e, _, _)| ms >= *s && ms < *e);
                    // `data_stream.past_tsdb_index_creation_enabled`: a past
                    // timestamp in the lifecycle's eligible write window gets a
                    // new backing index (`TransportPastTimeSeriesIndexCreationAction`).
                    if !ok && !ranges.is_empty() && create && allow_past && self.past_index_eligible(target, ms) && self.create_past_backing_index(target, ms).is_ok() {
                        ok = true;
                    }
                    if !ok && !ranges.is_empty() {
                        let shown: Vec<String> = ranges.iter().map(|(_, _, s, e)| format!("[{s},{e}]")).collect();
                        return Err(EsError::new(400, "timestamp_error", format!(
                            "the document timestamp [{}] is outside of ranges of currently writable indices [{}]",
                            crate::ilm::iso_date(ms),
                            shown.join(", ")
                        )));
                    }
                }
            }
            if !has_ts {
                return Err(EsError::document_parsing(format!("[1:1] failed to parse: data stream timestamp field [{TIMESTAMP_FIELD}] is missing"))
                    .caused_by(EsError::illegal_argument(format!("data stream timestamp field [{TIMESTAMP_FIELD}] is missing"))));
            }
        }
        Ok(first_ts)
    }

    /// Whether a document at `ms` that no backing index of time series data
    /// stream `ds` covers gets a new past backing index
    /// (`data_stream.past_tsdb_index_creation_enabled`, within the
    /// lifecycle's eligible write window).
    fn past_index_eligible(&self, ds: &str, ms: i64) -> bool {
        if !self.past_tsdb_enabled() {
            return false;
        }
        let now = crate::date::now_millis();
        let window = self.eligible_write_window_start(ds, now);
        ms <= now && (window <= 0 || ms >= window)
    }

    /// The `@timestamp` of a document for time series data stream `target`
    /// that no backing index covers but that gets a past backing index
    /// ([`Node::create_past_time_series_index`]); `None` otherwise. The
    /// cluster coordinator asks the master to create it before routing.
    pub fn past_time_series_timestamp(&self, target: &str, source: &str) -> Option<i64> {
        if !self.past_tsdb_enabled() {
            return None;
        }
        let ranges = self.tsds_ranges(target).filter(|r| !r.is_empty())?;
        let v: Value = serde_json::from_str(source).ok()?;
        let ms = match crate::highlight::source_values(&v, TIMESTAMP_FIELD).into_iter().next()? {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => crate::date::DateFormatter::new("strict_date_optional_time_nanos||strict_date_optional_time||epoch_millis").ok().and_then(|f| f.parse(s).ok()),
            _ => None,
        }?;
        (!ranges.iter().any(|(s, e, _, _)| ms >= *s && ms < *e) && self.past_index_eligible(target, ms)).then_some(ms)
    }

    /// Create (or find) the backing index of time series data stream `ds`
    /// covering past timestamp `ms` (`TransportPastTimeSeriesIndexCreationAction`
    /// on the master); its name.
    pub fn create_past_time_series_index(&self,ds:&str,ms:i64)->Result<String>{
        self.create_past_time_series_service(ds,ms).map(|service|service.name.clone())
    }

    pub fn create_past_time_series_service(&self, ds: &str, ms: i64) -> Result<Arc<IndexService>> {
        if let Some(s) = self.data_stream(ds).and_then(|d| {
            let f = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok()?;
            d.indices.iter().filter_map(|i| self.get_index(i)).find(|s| {
                let b = |k: &str| s.setting(k).and_then(|x| f.parse(&x).ok());
                matches!((b(crate::tsdb::START_TIME), b(crate::tsdb::END_TIME)), (Some(a), Some(e)) if ms >= a && ms < e)
            })
        }) {
            return Ok(s);
        }
        if !self.past_index_eligible(ds, ms) {
            return Err(EsError::illegal_argument(format!("no past backing index may be created for [{ds}] at [{}]", crate::ilm::iso_date(ms))));
        }
        self.create_past_backing_index(ds, ms)
    }

    fn past_tsdb_enabled(&self) -> bool {
        crate::ilm::cluster_setting(self, "data_stream.past_tsdb_index_creation_enabled").as_deref() == Some("true")
    }

    /// `TimeSeriesEligibleWriteWindowLocator.getEligibleWriteWindowStart`
    /// for a data stream lifecycle (`-1`: unbounded).
    fn eligible_write_window_start(&self, ds: &str, now: i64) -> i64 {
        let Some(d) = self.data_stream(ds) else { return -1 };
        let Some(lc) = d.lifecycle.as_ref().filter(|l| l.get("enabled").and_then(Value::as_bool) != Some(false)) else { return -1 };
        let tv = |v: Option<&Value>| v.and_then(Value::as_str).and_then(crate::date::parse_time_value);
        let rounds = lc.get("downsampling").and_then(|d| d.as_array().cloned().or_else(|| d.get("rounds").and_then(Value::as_array).cloned()));
        let window = match rounds.as_ref().and_then(|r| r.first()) {
            Some(first) => tv(first.get("after")),
            None => tv(lc.get("frozen_after")).or_else(|| tv(lc.get("data_retention"))),
        };
        window.map(|w| now - w).unwrap_or(-1)
    }

    /// Create the backing index covering past timestamp `ts`
    /// (`PastTimeSeriesIndexCreationExecutor.executeTask`).
    fn create_past_backing_index(&self, ds: &str, ts: i64) -> Result<Arc<IndexService>> {
        let _g = self.ds_create_lock.lock();
        let d = self.data_stream(ds).ok_or_else(|| EsError::index_not_found(ds))?;
        let f = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").map_err(|_| EsError::internal("formatter"))?;
        let mut windows: Vec<(i64, i64)> = d
            .indices
            .iter()
            .filter_map(|i| self.get_index(i))
            .filter_map(|s| Some((f.parse(&s.setting(crate::tsdb::START_TIME)?).ok()?, f.parse(&s.setting(crate::tsdb::END_TIME)?).ok()?)))
            .collect();
        windows.sort();
        // Consolidated covered windows, oldest first.
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (a, b) in windows {
            match merged.last_mut() {
                Some(last) if last.1 == a => last.1 = b,
                _ => merged.push((a, b)),
            }
        }
        let interval = crate::ilm::cluster_setting(self, "data_streams.past_tsdb_index_interval").and_then(|v| crate::date::parse_time_value(&v)).unwrap_or(86_400_000);
        let prev = merged.iter().filter(|w| w.1 <= ts).next_back().copied();
        let next = merged.iter().find(|w| w.1 > ts).copied().ok_or_else(|| EsError::internal("no backing index after the timestamp"))?;
        if next.0 <= ts {
            return self.get_index(d.write_index()).ok_or_else(|| EsError::index_not_found(ds));
        }
        let (start, end) = match prev {
            Some(p) if (next.0 - p.1) as f64 <= interval as f64 * 1.3 => (p.1, next.0),
            _ => {
                let jumps = (next.0 - ts - 1) / interval;
                let end = next.0 - jumps * interval;
                let start = prev.map_or(end - interval, |p| (end - interval).max(p.1));
                (start, end)
            }
        };
        let generation = d.generation + 1;
        let overrides = d.mappings.clone();
        let mut mappings = self.data_stream_mappings_with(ds, &overrides);
        let mut flat = self.data_stream_effective_settings(ds);
        flat.entry("index.mode".to_string()).or_insert(json!("time_series"));
        flat.insert("index.hidden".into(), json!("true"));
        self.time_series_backing_settings(ds, &mut flat, Some((start, end)));
        if !mappings.is_object() {
            mappings = json!({});
        }
        if mappings.get("properties").and_then(|p| p.get(TIMESTAMP_FIELD)).is_none() {
            mappings["properties"][TIMESTAMP_FIELD] = json!({"type": "date"});
        }
        mappings["_data_stream_timestamp"] = json!({"enabled": true});
        let name = backing_index_name(ds, generation, start);
        let created = self.create_index_impl(&name, &json!({"settings": crate::settings::unflatten(&flat), "mappings": mappings}), None, false)?;
        if let Some(w) = self.get_index(d.write_index()) {
            created.set_aliases(w.aliases())?;
        }
        {
            let mut streams = self.data_streams.write();
            if let Some(d) = streams.get_mut(ds) {
                d.generation = generation;
                d.indices.insert(0, created.name.clone());
            }
        }
        self.persist_state()?;
        Ok(created)
    }

    /// The backing index of a time series data stream whose
    /// `[start_time, end_time)` covers the document's `@timestamp`
    /// (`DataStream.selectTimeSeriesWriteIndex`).
    pub fn time_series_write_index(&self, target: &str, source: &str) -> Option<Arc<IndexService>> {
        if !self.data_streams.read().contains_key(target) {
            return None;
        }
        self.time_series_write_index_at(target, document_timestamp(source).as_ref())
    }

    /// [`Node::time_series_write_index`] of a document whose `@timestamp`
    /// ([`Node::check_data_stream_write_ts`]) is `ts`.
    pub fn time_series_write_index_at(&self, target: &str, ts: Option<&Value>) -> Option<Arc<IndexService>> {
        let d = self.data_streams.read().get(target).cloned()?;
        let write = self.get_index(d.write_index())?;
        if write.setting("index.mode").as_deref() != Some("time_series") {
            return None;
        }
        let ts = ts?;
        let ms = TS_FORMATS.with(|(doc, _)| match &ts {
            Value::Number(n) => n.as_i64(),
            Value::String(s) => doc.as_ref().and_then(|f| f.parse(s).ok()),
            _ => None,
        })?;
        d.indices.iter().rev().filter_map(|i| Some((i, self.get_index(i)?))).find(|(i, s)| {
            let (a, e) = (s.setting(crate::tsdb::START_TIME), s.setting(crate::tsdb::END_TIME));
            matches!(index_bounds(i, a, e), (Some(a), Some(e)) if ms >= a && ms < e)
        }).map(|(_, s)| s)
    }

    /// `POST /{target}/_rollover[/{new_index}]`.
    pub fn rollover(&self, target: &str, new_index: Option<&str>, body: &Value, dry_run: bool) -> Result<Value> {
        self.rollover_ext(target, new_index, body, dry_run, false)
    }

    /// Rollover with `lazy` (mark the data stream, or its failure store,
    /// to roll over on the next write) and `::data` / `::failures` selectors.
    pub fn rollover_ext(&self, target: &str, new_index: Option<&str>, body: &Value, dry_run: bool, lazy: bool) -> Result<Value> {
        let (target, selector) = split_selector(target)?;
        let conds = body.get("conditions").and_then(Value::as_object).cloned().unwrap_or_default();
        if let Some(o) = body.as_object() {
            const FIELDS: &[&str] = &["conditions", "settings", "mappings", "aliases"];
            const CONDITIONS: &[&str] = &["max_age", "max_docs", "max_size", "max_primary_shard_size", "max_primary_shard_docs", "min_age", "min_docs", "min_size", "min_primary_shard_size", "min_primary_shard_docs"];
            for (k, v) in o {
                if !FIELDS.contains(&k.as_str()) {
                    return Err(EsError::unknown_field_xc("rollover", k, None, FIELDS));
                }
                if k == "conditions" {
                    if let Some(c) = v.as_object() {
                        for ck in c.keys() {
                            if !CONDITIONS.contains(&ck.as_str()) {
                                return Err(EsError::field_failure("rollover", "conditions", None, EsError::unknown_field_xc("rollover_conditions", ck, Some("conditions"), CONDITIONS)));
                            }
                        }
                    }
                }
            }
        }
        // RolloverRequest.validate
        if conds.keys().any(|k| k.starts_with("min_")) && !conds.keys().any(|k| k.starts_with("max_")) {
            return Err(EsError::action_validation(&["at least one max_* rollover condition must be set when using min_* conditions"]));
        }
        let resp = |old: &str, new: &str, rolled: bool, c: &Conditions| {
            json!({
                "acknowledged": rolled, "shards_acknowledged": rolled, "old_index": old, "new_index": new,
                "rolled_over": rolled, "dry_run": dry_run, "lazy": false, "conditions": c.json()
            })
        };
        let lazy_resp = |old: &str, new: &str| {
            json!({
                "acknowledged": true, "shards_acknowledged": false, "old_index": old, "new_index": new,
                "rolled_over": false, "dry_run": dry_run, "lazy": true, "conditions": {}
            })
        };
        if lazy && self.data_stream(target).is_none() {
            return Err(EsError::action_validation(&["lazy rollover can be applied only on a data stream."]));
        }
        if lazy && !conds.is_empty() {
            return Err(EsError::action_validation(&["lazy rollover can be used only without any conditions."]));
        }
        // Failure store of a data stream.
        if selector == Some(Selector::Failures) {
            let ds = self.data_stream(target).ok_or_else(|| {
                EsError::new(404, "index_not_found_exception", format!("rollover target [{target}] does not exist")).with("resource.type", "index_or_alias").with("resource.id", target).with("index_uuid", "_na_").with("index", target)
            })?;
            let new_name = failure_index_name(&ds.name, ds.generation + 1, crate::date::now_millis());
            let Some(old) = ds.failure_write_index().map(str::to_string) else {
                if !conds.is_empty() {
                    return Err(EsError::new(500, "illegal_state_exception", "Rolling over/initializing an empty failure store is only supported without conditions."));
                }
                let c = Conditions { items: Vec::new() };
                if dry_run {
                    return Ok(resp("_none_", &new_name, false, &c));
                }
                let created = self.roll_failure_store(&ds.name, false)?;
                return Ok(resp("_none_", &created.name, true, &c));
            };
            if lazy {
                if !dry_run {
                    if let Some(d) = self.data_streams.write().get_mut(&ds.name) {
                        d.failure_rollover_on_write = true;
                    }
                    self.persist_state()?;
                }
                return Ok(lazy_resp(&old, &new_name));
            }
            let svc = self.get_index(&old).ok_or_else(|| EsError::index_not_found(&old))?;
            let c = eval_conditions(&svc, &conds, self.rollover_stats(&svc))?;
            if dry_run || !c.met() {
                return Ok(resp(&old, &new_name, false, &c));
            }
            let created = self.roll_failure_store(&ds.name, false)?;
            return Ok(resp(&old, &created.name, true, &c));
        }
        // Data stream.
        if let Some(ds) = self.data_stream(target) {
            let old = ds.write_index().to_string();
            let new_name = backing_index_name(&ds.name, ds.generation + 1, crate::date::now_millis());
            if lazy {
                if !dry_run {
                    if let Some(d) = self.data_streams.write().get_mut(&ds.name) {
                        d.rollover_on_write = true;
                    }
                    self.persist_state()?;
                }
                return Ok(lazy_resp(&old, &new_name));
            }
            let svc = self.get_index(&old).ok_or_else(|| EsError::index_not_found(&old))?;
            let c = eval_conditions(&svc, &conds, self.rollover_stats(&svc))?;
            if dry_run || !c.met() {
                return Ok(resp(&old, &new_name, false, &c));
            }
            let created = self.roll_data_stream(&ds.name)?;
            crate::ilm::runner::record_rollover(self, &svc, &ds.name, &c.json());
            return Ok(resp(&old, &created.name, true, &c));
        }
        if selector.is_some() {
            if self.get_index(target).is_some() {
                return Err(EsError::illegal_argument("rollover target is a [concrete index] but one of [alias,data_stream] was expected"));
            }
        }
        // Alias.
        let mut members: Vec<(Arc<IndexService>, Value)> = Vec::new();
        for svc in self.all_indices() {
            if let Some(def) = svc.aliases().get(target) {
                members.push((svc.clone(), def.clone()));
            }
        }
        if members.is_empty() {
            if self.get_index(target).is_some() {
                return Err(EsError::illegal_argument("rollover target is a [concrete index] but one of [alias,data_stream] was expected"));
            }
            return Err(EsError::resource_not_found(format!("rollover target [{target}] does not exist")));
        }
        let explicit = members.iter().find(|(_, d)| d.get("is_write_index").and_then(Value::as_bool) == Some(true)).map(|(s, _)| s.clone());
        let (write, explicit_flag) = match explicit {
            Some(s) => (s, true),
            None if members.len() == 1 && members[0].1.get("is_write_index").and_then(Value::as_bool) != Some(false) => (members[0].0.clone(), false),
            None => return Err(EsError::illegal_argument(format!("rollover target [{target}] does not point to a write index"))),
        };
        let new_name = match new_index {
            Some(n) => {
                let n = crate::node::resolve_date_math(n)?;
                crate::node::validate_index_name(&n)?;
                n
            }
            None => next_index_name(&write.name)?,
        };
        let c = eval_conditions(&write, &conds, self.rollover_stats(&write))?;
        if let Some(existing) = self.get_index(&new_name) {
            return Err(EsError::index_already_exists(&new_name, &existing.uuid));
        }
        if dry_run || !c.met() {
            return Ok(resp(&write.name, &new_name, false, &c));
        }
        let mut aliases = body.get("aliases").cloned().unwrap_or(json!({}));
        aliases[target] = if explicit_flag { json!({"is_write_index": true}) } else { json!({}) };
        let mut create_body = json!({"aliases": aliases});
        for k in ["settings", "mappings"] {
            if let Some(v) = body.get(k) {
                create_body[k] = v.clone();
            }
        }
        self.create_index(&new_name, &create_body)?;
        let mut old_aliases = write.aliases();
        if explicit_flag {
            if let Some(d) = old_aliases.get_mut(target) {
                d["is_write_index"] = json!(false);
            }
        } else {
            old_aliases.remove(target);
        }
        write.set_aliases(old_aliases)?;
        crate::ilm::runner::record_rollover(self, &write, target, &c.json());
        Ok(resp(&write.name, &new_name, true, &c))
    }
}

/// Look up a dotted field path in a properties object.
/// `props` is the top-level `properties` map (keys are field names).
/// For a path like `host.name`, it looks up `props["host"]["properties"]["name"]`.
fn lookup_field(props: &serde_json::Map<String, Value>, path: &str) -> Option<Value> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        return None;
    }
    // First part: look up the top-level field directly in props.
    let mut cur = props.get(parts[0])?;
    // Remaining parts: navigate through `properties` sub-objects.
    for part in &parts[1..] {
        let sub_props = cur.get("properties")?;
        cur = sub_props.get(part)?;
    }
    Some(cur.clone())
}

/// Look up a dotted field path, checking both the flat key form
/// (`"host.name"`) and the nested form (`host.properties.name`).
fn lookup_field_any(props: &Value, path: &str) -> Option<Value> {
    // Try the flat key first.
    if let Some(f) = props.get(path) {
        return Some(f.clone());
    }
    // Try the nested form.
    lookup_field(props.as_object()?, path)
}


/// The root `@timestamp` of a document source, read without building the
/// rest of it ([`Node::check_data_stream_write`]): `value` is the last root
/// `@timestamp` (as `serde_json::Value` keeps duplicate keys), and
/// `not_object` is `Some(true)` for an array source, `Some(false)` for any
/// other non-object.
#[derive(Default, Clone)]
struct TimestampProbe {
    /// Root `@timestamp` keys seen.
    count: usize,
    value: Option<Value>,
    not_object: Option<bool>,
}

impl<'de> serde::Deserialize<'de> for TimestampProbe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = TimestampProbe;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON document")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<TimestampProbe, A::Error> {
                let mut out = TimestampProbe::default();
                while let Some(key) = map.next_key::<std::borrow::Cow<'de, str>>()? {
                    if key == TIMESTAMP_FIELD {
                        out.count += 1;
                        out.value = Some(map.next_value::<Value>()?);
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(out)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<TimestampProbe, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(TimestampProbe { count: 0, value: None, not_object: Some(true) })
            }
            fn visit_bool<E>(self, _: bool) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
            fn visit_i64<E>(self, _: i64) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
            fn visit_u64<E>(self, _: u64) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
            fn visit_f64<E>(self, _: f64) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
            fn visit_str<E>(self, _: &str) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
            fn visit_unit<E>(self) -> std::result::Result<TimestampProbe, E> { Ok(TimestampProbe { count: 0, value: None, not_object: Some(false) }) }
        }
        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_conditions() {
        assert_eq!(next_index_name("logs-000001").unwrap(), "logs-000002");
        assert_eq!(next_index_name("logs-9").unwrap(), "logs-000010");
        assert_eq!(next_index_name("logs").unwrap_err().reason, "index name [logs] does not match pattern '^.*-\\d+$'");
        assert!(backing_index_name("x", 3, 0).starts_with(".ds-x-1970.01.01-000003"));
        assert_eq!(parse_size("50gb"), Some(50 << 30));
        assert_eq!(parse_size("1kb"), Some(1024));
    }

    #[test]
    fn failure_store_refresh_template_and_node_override() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.templates.write().index.insert("refresh".into(), json!({
            "index_patterns": ["refresh-*"], "data_stream": {}, "priority": 1000,
            "template": {"settings": {"index.refresh_interval": "7s"}}
        }));
        let inherited = node.create_failure_index("refresh-inherited", 1).unwrap();
        assert_eq!(inherited.setting("index.refresh_interval").as_deref(), Some("7s"));
        node.node_settings.write().insert("data_streams.failure_store.refresh_interval".into(), json!("30s"));
        let overridden = node.create_failure_index("refresh-overridden", 1).unwrap();
        assert_eq!(overridden.setting("index.refresh_interval").as_deref(), Some("30s"));
        let stream = node.create_data_stream("refresh-ordinary").unwrap();
        let ordinary = node.get_index(stream.write_index()).unwrap();
        assert_eq!(ordinary.setting("index.refresh_interval").as_deref(), Some("7s"));
        node.node_settings.write().insert("data_streams.failure_store.refresh_interval".into(), json!("bad-time"));
        assert!(node.create_failure_index("refresh-invalid", 1).is_err());
        node.node_settings.write().remove("data_streams.failure_store.refresh_interval");
        let default = node.create_failure_index("unconfigured", 1).unwrap();
        assert_eq!(default.setting("index.refresh_interval"), None);
    }

    /// A plain write index resolves to itself under every bulk resolution
    /// it lets the bulk handler skip; aliases, data streams, backing
    /// indices, date math and deleted or recreated indices are not plain.
    #[test]
    fn plain_write_index_resolutions_are_identities() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index("plain", &json!({})).unwrap();
        let p = node.plain_write_index("plain").expect("a concrete index is plain");
        assert!(Arc::ptr_eq(&p, &svc));
        assert!(Arc::ptr_eq(&node.write_index("plain", true).unwrap(), &svc));
        assert!(Arc::ptr_eq(&node.write_index("plain", false).unwrap(), &svc));
        assert_eq!(node.index_routing_for("plain", Some("r")).unwrap().as_deref(), Some("r"));
        assert_eq!(node.index_routing_for("plain", None).unwrap(), None);
        assert!(matches!(node.failure_store_target("plain"), FailureStoreTarget::None));
        assert!(node.check_backing_write("plain", true, false).is_ok());
        assert!(node.check_backing_write("plain", false, false).is_ok());
        assert!(node.check_data_stream_write("plain", false, Some(r#"{"a":1}"#)).is_ok());
        assert!(node.time_series_write_index("plain", r#"{"@timestamp":"2026-01-01"}"#).is_none());
        assert!(node.is_registered_index("plain", &svc));

        node.update_aliases(&[json!({"add": {"index": "plain", "alias": "al", "index_routing": "x"}})]).unwrap();
        assert!(node.plain_write_index("al").is_none(), "an alias is not plain");
        assert!(node.plain_write_index("<plain>").is_none() && node.plain_write_index("pl%61in").is_none());
        node.templates.write().index.insert("t".into(), json!({"index_patterns": ["logs-*"], "data_stream": {}}));
        let ds = node.create_data_stream("logs-a").unwrap();
        assert!(node.plain_write_index("logs-a").is_none(), "a data stream is not plain");
        assert!(node.plain_write_index(ds.write_index()).is_none(), "a backing index is not plain");

        node.delete_index("plain").unwrap();
        assert!(!node.is_registered_index("plain", &svc), "a deleted index is no longer registered");
        let again = node.create_index("plain", &json!({})).unwrap();
        assert!(!node.is_registered_index("plain", &svc), "nor is the old copy of a recreated one");
        assert!(node.is_registered_index("plain", &again));
    }

    #[test]
    fn backing_deletion_publishes_registry_and_membership_together() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.templates.write().index.insert("t".into(), json!({"index_patterns": ["logs-*"], "data_stream": {}}));
        let ds = node.create_data_stream("logs-a").unwrap();
        let old = ds.write_index().to_owned();
        node.rollover("logs-a", None, &json!({}), false).unwrap();
        let before = node.data_stream("logs-a").unwrap();
        let refused = node.detach_backing_index_with::<()>(&old, || {
            // A reader cannot observe detached membership while the registry
            // removal is pending, including when that removal fails.
            assert!(node.data_streams.try_read().is_none());
            Err(EsError::illegal_argument("registry removal refused"))
        });
        assert!(refused.is_err());
        assert_eq!(node.data_stream("logs-a").unwrap().indices, before.indices);
        assert_eq!(node.data_stream("logs-a").unwrap().generation, before.generation);
        assert!(node.get_index(&old).is_some());
        assert!(node.delete_index(before.write_index()).is_err());
        assert!(node.get_index(before.write_index()).is_some());
        node.delete_index(&old).unwrap();
        assert!(node.get_index(&old).is_none());
        let after = node.data_stream("logs-a").unwrap();
        assert_eq!(after.indices.len(), 1);
        assert_eq!(after.generation, before.generation + 1);
        drop(node);
        let reopened = Node::open(tmp.path()).unwrap();
        assert!(reopened.get_index(&old).is_none());
        assert_eq!(reopened.data_stream("logs-a").unwrap().indices, after.indices);
    }

    #[test]
    fn data_stream_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        assert_eq!(node.create_data_stream("logs-a").unwrap_err().reason, "no matching index template found for data stream [logs-a]");
        node.templates.write().index.insert("t".into(), json!({"index_patterns": ["logs-*"], "data_stream": {}, "priority": 1}));
        let ds = node.create_data_stream("logs-a").unwrap();
        assert!(ds.indices[0].starts_with(".ds-logs-a-") && ds.indices[0].ends_with("-000001"));
        assert_eq!(node.create_data_stream("logs-a").unwrap_err().kind, "resource_already_exists_exception");
        let svc = node.write_index("logs-a", false).unwrap();
        assert_eq!(svc.name, ds.indices[0]);
        assert!(svc.lookup().get("@timestamp").is_some());
        let r = node.rollover("logs-a", None, &json!({}), false).unwrap();
        assert_eq!(r["rolled_over"], true);
        assert!(r["new_index"].as_str().unwrap().ends_with("-000002"));
        assert_eq!(node.resolve("logs-a", Default::default()).unwrap().len(), 2);
        assert_eq!(node.resolve("logs-*", Default::default()).unwrap().len(), 2);
        drop(svc);
        drop(node);
        let node = Node::open(tmp.path()).unwrap();
        assert_eq!(node.data_stream("logs-a").unwrap().generation, 2);
        node.delete_data_streams("logs-a").unwrap();
        assert!(node.all_indices().is_empty());
    }
}

thread_local! {
    /// The formats [`Node::time_series_write_index`] parses a document's
    /// `@timestamp` and the backing indices' bounds with (built once per
    /// thread, not per document).
    static TS_FORMATS: (Option<crate::date::DateFormatter>, Option<crate::date::DateFormatter>) = (
        crate::date::DateFormatter::new("strict_date_optional_time_nanos||strict_date_optional_time||epoch_millis").ok(),
        crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok(),
    );
    /// Parsed `[start_time, end_time)` of backing indices, keyed by index
    /// name and valid while the raw settings are unchanged.
    #[allow(clippy::type_complexity)]
    static TS_BOUNDS: std::cell::RefCell<std::collections::HashMap<String, (Option<String>, Option<String>, Option<i64>, Option<i64>)>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// A backing index's parsed time bounds (cached per thread while its raw
/// settings `start` / `end` stay the same).
fn index_bounds(index: &str, start: Option<String>, end: Option<String>) -> (Option<i64>, Option<i64>) {
    TS_BOUNDS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((s, e, a, b)) = cache.get(index) {
            if *s == start && *e == end {
                return (*a, *b);
            }
        }
        let parse = |x: &Option<String>| TS_FORMATS.with(|(_, f)| x.as_ref().and_then(|x| f.as_ref()?.parse(x).ok()));
        let (a, b) = (parse(&start), parse(&end));
        if cache.len() > 4096 {
            cache.clear();
        }
        cache.insert(index.to_string(), (start, end, a, b));
        (a, b)
    })
}

/// A document's first `@timestamp` value (`source_values` of the parsed
/// source): read from the top-level key without building the document,
/// else (no such key, a duplicate, an object) from the parsed source.
/// One scan of a document's source for its root `@timestamp`, serving the
/// data stream checks ([`Node::check_data_stream_write_only_probed`]) and
/// the backing-index routing ([`Node::time_series_write_index_probed`])
/// of the same item (each scanned the source again).
#[derive(Clone)]
pub struct TimestampScan {
    probe: Option<TimestampProbe>,
    /// [`document_timestamp`] of the source.
    doc_ts: Option<Value>,
}

impl TimestampScan {
    pub fn of(source: &str) -> Self {
        let probe: Option<TimestampProbe> = serde_json::from_str(source).ok();
        // `document_timestamp`'s fast branch (one key, a scalar or an array
        // of scalars) read from the probe; anything else as it reads it.
        let doc_ts = match probe.as_ref() {
            Some(TimestampProbe { count: 1, value: Some(v), not_object: None })
                if !v.is_object() && !v.is_null() && !v.as_array().is_some_and(|a| a.iter().any(|x| x.is_object() || x.is_null())) => first_timestamp_leaf(v.clone()),
            _ => document_timestamp(source),
        };
        TimestampScan { probe, doc_ts }
    }
}

/// An `@timestamp` value's first leaf (arrays flattened).
fn first_timestamp_leaf(v: Value) -> Option<Value> {
    match v {
        Value::Array(a) => {
            fn first_leaf(v: Value) -> Option<Value> {
                match v {
                    Value::Array(a) => a.into_iter().find_map(first_leaf),
                    x => Some(x),
                }
            }
            a.into_iter().find_map(first_leaf)
        }
        x => Some(x),
    }
}

fn document_timestamp(source: &str) -> Option<Value> {
    #[derive(serde::Deserialize)]
    struct Top {
        #[serde(rename = "@timestamp", default)]
        ts: Option<Value>,
    }
    let first = first_timestamp_leaf;
    match serde_json::from_str::<Top>(source) {
        Ok(Top { ts: Some(v) }) if !v.is_object() && !v.is_null() && !v.as_array().is_some_and(|a| a.iter().any(|x| x.is_object() || x.is_null())) => first(v),
        _ => {
            let v: Value = serde_json::from_str(source).ok()?;
            crate::highlight::source_values(&v, TIMESTAMP_FIELD).into_iter().next().cloned()
        }
    }
}

#[cfg(test)]
mod document_timestamp_tests {
    use super::*;

    /// The top-level read gives `source_values`' first `@timestamp`.
    #[test]
    fn document_timestamp_matches_source_values() {
        for src in [
            r#"{"a":1,"@timestamp":"2024-05-01T10:00:00Z","b":{"c":[1,2]}}"#,
            r#"{"@timestamp":1714557600000}"#,
            r#"{"@timestamp":[["2024-05-01T10:00:00Z"],"x"]}"#,
            r#"{"@timestamp":[null,"2024-05-01T10:00:00Z"]}"#,
            r#"{"@timestamp":null}"#,
            r#"{"@timestamp":{"v":1}}"#,
            r#"{"x":{"@timestamp":"nested"}}"#,
            r#"{"@timestamp":"a","@timestamp":"b"}"#,
            r#"{"b":2}"#,
        ] {
            let v: Value = serde_json::from_str(src).unwrap();
            let want = crate::highlight::source_values(&v, TIMESTAMP_FIELD).into_iter().next().cloned();
            assert_eq!(document_timestamp(src), want, "{src}");
            // One scan reads what the two parses read.
            let scan = TimestampScan::of(src);
            assert_eq!(scan.doc_ts, want, "{src}");
            let probe: Option<TimestampProbe> = serde_json::from_str(src).ok();
            assert_eq!(scan.probe.as_ref().map(|p| (p.value.clone(), p.not_object)), probe.map(|p| (p.value, p.not_object)), "{src}");
        }
        for src in [r#"[1]"#, r#""s""#, "7", r#"{"\u0040timestamp":"2024-05-01T10:00:00Z"}"#] {
            assert_eq!(TimestampScan::of(src).doc_ts, document_timestamp(src), "{src}");
        }
    }
}
