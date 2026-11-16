//! x-pack `migrate` and `downsample` actions:
//!
//! * `PUT|POST /_create_from/{source}/{dest}`: a new index with the source
//!   index's settings and mappings (plus overrides), no documents.
//! * `POST /_migration/reindex`, `GET /_migration/reindex/{n}/_status`,
//!   `POST /_migration/reindex/{n}/_cancel`: upgrade a data stream's
//!   backing indices created by an older version. Every index here is
//!   created by this version, so a run finds nothing to upgrade and
//!   completes at once (what ES reports for an up-to-date data stream).
//! * `POST /{index}/_downsample/{target}`: TSDS downsampling.

use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::node::Node;
use crate::settings::FlatSettings;
use crate::shard::WriteCond;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::OnceLock;

fn no_such_index(name: &str) -> EsError {
    EsError::new(404, "index_not_found_exception", format!("no such index [{name}]")).with("index_uuid", "_na_").with("index", name)
}

/// Settings ES never copies to another index (`IndexMetadata` private /
/// per-index settings).
fn private_setting(k: &str) -> bool {
    matches!(k, "index.uuid" | "index.creation_date" | "index.provided_name" | "index.history.uuid" | "index.frozen" | "index.search.throttled")
        || k.starts_with("index.version.")
        || k.starts_with("index.resize.")
        || k.starts_with("index.downsample.")
        || k.starts_with("index.shrink.")
}

fn block_setting(k: &str) -> bool {
    k.starts_with("index.blocks.")
}

/// Merge a mapping override into a mapping (`properties` recursively, other
/// keys replaced).
fn merge_mappings(base: &mut Value, over: &Value) {
    let (Some(b), Some(o)) = (base.as_object_mut(), over.as_object()) else {
        if !over.is_null() {
            *base = over.clone();
        }
        return;
    };
    for (k, v) in o {
        if k == "properties" {
            let bp = b.entry("properties").or_insert_with(|| json!({}));
            if let (Some(bp), Some(op)) = (bp.as_object_mut(), v.as_object()) {
                for (f, def) in op {
                    match bp.get_mut(f) {
                        // An object field keeps its sub-fields unless the override changes its type.
                        Some(existing) if def.get("type").is_none() && def.get("properties").is_some() => merge_mappings(existing, def),
                        _ => {
                            bp.insert(f.clone(), def.clone());
                        }
                    }
                }
                continue;
            }
        }
        b.insert(k.clone(), v.clone());
    }
}

/// `PUT /_create_from/{source}/{dest}`.
pub fn create_from(node: &Node, source: &str, dest: &str, body: &Value) -> Result<Value> {
    let empty = Map::new();
    let o = match body {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err(EsError::x_content_parse("[1:1] [create_index_from_source_request] expected an object")),
    };
    for k in o.keys() {
        if !matches!(k.as_str(), "settings_override" | "mappings_override" | "remove_index_blocks") {
            return Err(EsError::x_content_parse(format!("[1:2] [create_index_from_source_request] unknown field [{k}]")));
        }
    }
    let remove_blocks = match o.get("remove_index_blocks") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) if s == "true" || s == "false" => s == "true",
        Some(_) => return Err(EsError::x_content_parse("[1:2] [create_index_from_source_request] failed to parse field [remove_index_blocks]")),
    };
    let svc = node.get_index(source).ok_or_else(|| no_such_index(source))?;
    let meta = svc.meta();
    let mut flat: FlatSettings = meta.settings.into_iter().filter(|(k, _)| !private_setting(k)).collect();
    if let Some(over) = o.get("settings_override").filter(|v| !v.is_null()) {
        for (k, v) in crate::settings::normalize_index_settings(over) {
            if v.is_null() {
                flat.remove(&k);
            } else {
                flat.insert(k, v);
            }
        }
    }
    if remove_blocks {
        flat.retain(|k, _| !block_setting(k));
    }
    let mut mappings = meta.mappings.to_json();
    if let Some(over) = o.get("mappings_override").filter(|v| !v.is_null()) {
        merge_mappings(&mut mappings, over);
    }
    let created = node.create_index(dest, &json!({"settings": crate::settings::unflatten(&flat), "mappings": mappings}))?;
    Ok(json!({"acknowledged": true, "shards_acknowledged": true, "index": created.name}))
}

// ---------------------------------------------------------------------------
// _migration/reindex
// ---------------------------------------------------------------------------

/// Migration reindex runs by node (data path) and data stream name.
fn reindex_runs() -> &'static parking_lot::Mutex<HashMap<PathBuf, BTreeMap<String, Value>>> {
    static R: OnceLock<parking_lot::Mutex<HashMap<PathBuf, BTreeMap<String, Value>>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// `POST /_migration/reindex`.
pub fn migrate_reindex(node: &Node, body: &Value) -> Result<Value> {
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        _ => return Err(EsError::x_content_parse("[1:1] [migration_reindex] expected an object")),
    };
    for k in o.keys() {
        if !matches!(k.as_str(), "mode" | "source") {
            return Err(EsError::x_content_parse(format!("[1:2] [migration_reindex] unknown field [{k}]")));
        }
    }
    let mode = o.get("mode").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [mode]"))?;
    let index = o.get("source").ok_or_else(|| EsError::illegal_argument("Required [source]"))?;
    if mode != "upgrade" {
        return Err(EsError::x_content_parse("[1:2] [migration_reindex] failed to parse field [source]").caused_by(
            EsError::x_content_parse("Failed to build [migration_reindex] after last required field arrived")
                .caused_by(EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.migrate.action.ReindexDataStreamAction.Mode.{}", mode.to_uppercase()))),
        ));
    }
    let name = index["index"].as_str().ok_or_else(|| EsError::illegal_argument("Required [index]"))?;
    // Wildcard patterns are rejected as illegal_argument; exact names (including
    // comma-separated lists) that don't exist are resource_not_found (ES behavior).
    if name.contains('*') {
        return Err(EsError::illegal_argument(format!("Data stream named [{name}] does not exist")));
    }
    let ds = node.data_stream(name).ok_or_else(|| EsError::resource_not_found(format!("Data stream named [{name}] does not exist")))?;
    let mut runs = reindex_runs().lock();
    let runs = runs.entry(node.data.clone()).or_default();
    if runs.get(name).map(|r| r["complete"] == json!(false)).unwrap_or(false) {
        return Err(EsError::new(400, "resource_already_exists_exception", format!("task with id {{reindex-data-stream-{name}}} is already running")));
    }
    // Every backing index was created by this version: nothing to upgrade.
    runs.insert(
        name.to_string(),
        json!({
            "start_time_millis": crate::date::now_millis(),
            "complete": true,
            "total_indices_in_data_stream": ds.indices.len(),
            "total_indices_requiring_upgrade": 0,
            "successes": 0,
            "in_progress": [],
            "pending": 0,
            "errors": []
        }),
    );
    Ok(json!({"acknowledged": true}))
}

/// `GET /_migration/reindex/{index}/_status`.
pub fn migrate_reindex_status(node: &Node, name: &str) -> Result<Value> {
    let runs = reindex_runs().lock();
    runs.get(&node.data)
        .and_then(|r| r.get(name))
        .cloned()
        .ok_or_else(|| EsError::resource_not_found(format!("No migration reindex status found for [{name}]")))
}

/// `POST /_migration/reindex/{index}/_cancel`.
pub fn migrate_reindex_cancel(node: &Node, name: &str) -> Result<Value> {
    let mut runs = reindex_runs().lock();
    match runs.get_mut(&node.data).and_then(|r| r.remove(name)) {
        Some(_) => Ok(json!({"acknowledged": true})),
        None => Err(EsError::resource_not_found(format!("the task with id reindex-data-stream-{name} doesn't exist"))),
    }
}

// ---------------------------------------------------------------------------
// _downsample
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Dimension,
    Gauge,
    Counter,
    /// An `aggregate_metric_double` source field (`metric`: a gauge of an
    /// earlier downsample; else a label read per metric).
    Aggregate { metric: bool },
    /// A `histogram` / `tdigest` field (`metric`: `time_series_metric`).
    Histogram { metric: bool, tdigest: bool },
    /// An `exponential_histogram` field.
    ExpHist { metric: bool },
    /// A label; `Some(n)`: keyword `ignore_above` (longer values have no doc values).
    Label(Option<usize>),
}

/// Leaf fields of a mapping by dotted path, classified for downsampling.
fn classify(props: &Map<String, Value>, prefix: &str, out: &mut Vec<(String, Kind)>) -> Result<()> {
    for (name, def) in props {
        let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
        if let Some(sub) = def.get("properties").and_then(Value::as_object) {
            classify(sub, &path, out)?;
            continue;
        }
        // Multi-fields (`TimeSeriesFields.trackMultiFields`): no metrics; a
        // dimension multi-field makes its parent a dimension.
        let mut multi_dim = false;
        for (sub, sd) in def.get("fields").and_then(Value::as_object).into_iter().flatten() {
            if sd.get("time_series_metric").and_then(Value::as_str).is_some() {
                return Err(EsError::illegal_argument(format!("Downsampling failed because index mapping contains time series metrics as a multi field [{path}.{sub}]")));
            }
            if sd["time_series_dimension"].as_bool() == Some(true) {
                multi_dim = true;
            }
        }
        let ty = def["type"].as_str().unwrap_or("object");
        if path == crate::datastream::TIMESTAMP_FIELD || path == "_doc_count" || matches!(ty, "alias" | "object" | "nested" | "passthrough") {
            continue;
        }
        let metric_ty = def.get("time_series_metric").and_then(Value::as_str);
        // Labels are read from doc values: none, nothing to propagate.
        if def["doc_values"].as_bool() == Some(false) && def["time_series_dimension"].as_bool() != Some(true) && metric_ty.is_none() {
            continue;
        }
        if metric_ty == Some("position") {
            return Err(EsError::illegal_argument("Unsupported metric type [position] for downsampling"));
        }
        let kind = if def["time_series_dimension"].as_bool() == Some(true) || (multi_dim && metric_ty.is_none()) {
            Kind::Dimension
        } else {
            match ty {
                "aggregate_metric_double" => Kind::Aggregate { metric: metric_ty.is_some() },
                "histogram" | "tdigest" => Kind::Histogram { metric: metric_ty.is_some(), tdigest: ty == "tdigest" },
                "exponential_histogram" => Kind::ExpHist { metric: metric_ty.is_some() },
                _ => match metric_ty {
                    Some("gauge") => Kind::Gauge,
                    Some("counter") => Kind::Counter,
                    _ => Kind::Label(def["ignore_above"].as_u64().map(|n| n as usize)),
                },
            }
        };
        out.push((path, kind));
    }
    Ok(())
}

/// `TransportDownsampleAction.createDownsampleIndexMapping`: with the
/// `aggregate` method gauges become `aggregate_metric_double`; `@timestamp`
/// records the interval.
fn target_mappings(m: &Value, interval: &str, aggregate: bool) -> Value {
    fn walk(props: &mut Map<String, Value>, prefix: &str, aggregate: bool) {
        for (name, def) in props.iter_mut() {
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
            if let Some(sub) = def.get_mut("properties").and_then(Value::as_object_mut) {
                walk(sub, &path, aggregate);
                continue;
            }
            if aggregate && def["time_series_metric"].as_str() == Some("gauge") && def["type"].as_str() != Some("aggregate_metric_double") {
                // `aggregate_metric_double.default_metric.deprecated`: no default metric.
                *def = json!({"type": "aggregate_metric_double", "metrics": ["min", "max", "sum", "value_count"], "time_series_metric": "gauge"});
            }
        }
    }
    let mut m = m.clone();
    // The downsampled documents are new documents: the source's `_source`
    // filter (whose inputs were read from doc values) does not prune them.
    if let Some(o) = m.get_mut("_source").and_then(Value::as_object_mut) {
        o.remove("includes");
        o.remove("excludes");
        if o.is_empty() {
            m.as_object_mut().map(|o| o.remove("_source"));
        }
    }
    if let Some(props) = m.get_mut("properties").and_then(Value::as_object_mut) {
        props.remove("_doc_count");
        walk(props, "", aggregate);
        let ts = props.entry(crate::datastream::TIMESTAMP_FIELD).or_insert_with(|| json!({"type": "date"}));
        ts["meta"] = json!({"fixed_interval": interval, "time_zone": "UTC"});
    }
    m["_data_stream_timestamp"] = json!({"enabled": true});
    m
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// The doc-value view of a numeric field's source values (sorted, as
/// `SortedNumericDoubleValues` reads them).
fn sorted_nums(vals: &[&Value]) -> Vec<f64> {
    let mut v: Vec<f64> = vals.iter().filter_map(|x| num(x)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// A cumulative metric's reset data point (`ResetDataPoints.ResetPoint`).
#[derive(Clone)]
enum ResetValue {
    Counter(f64),
    Hist(Value),
}

/// `NumericMetricFieldDownsampler.AggregateCounter` /
/// `ExponentialHistogramFieldDownsampler.AggregateHistogram` collector state
/// for one field of one time series, fed in descending time order.
#[derive(Default)]
struct Cumulative {
    /// The bucket's downsampled value (its earliest value).
    value: Option<ResetValue>,
    last_ts: i64,
    previous: Option<ResetValue>,
    previous_bucket: Option<ResetValue>,
    stack: Vec<(i64, ResetValue)>,
}

fn reset_count(v: &ResetValue) -> f64 {
    match v {
        ResetValue::Counter(x) => *x,
        ResetValue::Hist(h) => crate::fieldtypes::exphist::parse("", h, false).map(|h| h.value_count() as f64).unwrap_or(0.0),
    }
}

impl Cumulative {
    fn new() -> Self {
        Cumulative { last_ts: -1, ..Default::default() }
    }

    /// `true` when going back in time from `prev` to `cur` is a reset.
    fn is_reset(prev: &ResetValue, cur: &ResetValue) -> bool {
        match (prev, cur) {
            (ResetValue::Counter(p), ResetValue::Counter(c)) => c > p,
            (ResetValue::Hist(p), ResetValue::Hist(c)) => {
                use crate::fieldtypes::exphist;
                match (exphist::parse("", p, false), exphist::parse("", c, false)) {
                    (Ok(p), Ok(c)) => c.value_count() > p.value_count() || !matches!(exphist::difference(&p, &c), Ok((_, true))),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn collect(&mut self, v: ResetValue, ts: i64) {
        let Some(prev) = self.previous.clone() else {
            self.value = Some(v.clone());
            self.previous = Some(v);
            self.last_ts = ts;
            return;
        };
        if self.value.is_none() {
            self.value = Some(v.clone());
        }
        if Self::is_reset(&prev, &v) {
            if self.last_ts > 0 {
                let already = self.stack.last().is_some_and(|(t, _)| *t == self.last_ts);
                if !already {
                    let last_persisted = self.stack.last().map(|(_, x)| x.clone()).or_else(|| self.previous_bucket.clone());
                    let need = match &last_persisted {
                        None => true,
                        Some(lp) => reset_count(&v) < reset_count(lp),
                    };
                    if need {
                        self.stack.push((self.last_ts, prev.clone()));
                    }
                }
            }
            self.stack.push((ts, v.clone()));
        }
        self.value = Some(v.clone());
        self.previous = Some(v);
        self.last_ts = ts;
    }

    /// End of a bucket: its reset points (`updateResetDataPoints`).
    fn bucket_resets(&mut self) -> Vec<(i64, ResetValue)> {
        let mut out = Vec::new();
        if let Some(first) = self.stack.pop() {
            let same = match (&first.1, &self.value) {
                (ResetValue::Counter(a), Some(ResetValue::Counter(b))) => a == b,
                (ResetValue::Hist(a), Some(ResetValue::Hist(b))) => a == b,
                _ => false,
            };
            if !same {
                out.push(first);
            }
            while let Some(x) = self.stack.pop() {
                out.push(x);
            }
        }
        out
    }

    fn reset_bucket(&mut self) {
        self.previous_bucket = self.value.take();
        self.last_ts = -1;
        self.stack.clear();
    }
}

/// The index operations a downsample run performs. Locally they act on the
/// node's engine; a cluster performs them through its coordinator, so the
/// source is read from every shard and the target index is created in the
/// cluster metadata and filled through its primaries.
pub trait DownsampleOps {
    fn create_index(&self, name: &str, body: &Value) -> Result<()>;
    /// Visit every document `_source` of `index`.
    fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()>;
    /// Visit every hit of the time series index `index` (its `_source`,
    /// `sort` — `_tsid` displayed first — and the requested
    /// `docvalue_fields` under `fields`), in index sort order: `_tsid`
    /// ascending, `@timestamp` descending (one time series at a time,
    /// newest first). Every source shard must be read: a scan that cannot
    /// read all of them (failed shards, a timed-out page) fails. An error
    /// from `f` ends the scan with that error.
    fn scan_series(&self, index: &str, docvalue_fields: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()>;
    /// Index `docs` (sources, auto ids) into `index`.
    fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()>;
    fn refresh(&self, index: &str) -> Result<()>;
    /// Sets `index.blocks.write`.
    fn block_writes(&self, index: &str) -> Result<()>;
    fn delete_index(&self, index: &str) -> Result<()>;
    /// `index.downsample.status` → `success` (every document copied and
    /// refreshed).
    fn mark_success(&self, index: &str) -> Result<()>;
}

/// Targets with a downsample running on this node, keyed by the node's
/// data path and the target: another node (or engine) in the same process
/// may downsample into an index of the same name.
pub(crate) struct InFlight((std::path::PathBuf, String));

static IN_FLIGHT: std::sync::LazyLock<parking_lot::Mutex<std::collections::HashSet<(std::path::PathBuf, String)>>> = std::sync::LazyLock::new(Default::default);

impl InFlight {
    pub(crate) fn claim(node: &Node, target: &str) -> Option<InFlight> {
        let key = (node.data.clone(), target.to_string());
        IN_FLIGHT.lock().insert(key.clone()).then(|| InFlight(key))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT.lock().remove(&self.0);
    }
}

/// [`DownsampleOps`] on the node's own engine.
pub struct LocalDownsampleOps<'a>(pub &'a Node);

impl DownsampleOps for LocalDownsampleOps<'_> {
    fn create_index(&self, name: &str, body: &Value) -> Result<()> {
        self.0.create_index(name, body).map(|_| ())
    }
    fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
        scan_sources(self.0, index, f)
    }
    fn scan_series(&self, index: &str, docvalue_fields: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
        scan_hits(self.0, index, &json!([{"_tsid": "asc"}, {"@timestamp": "desc"}]), docvalue_fields, f)
    }
    fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()> {
        let dest = self.0.get_index(index).ok_or_else(|| no_such_index(index))?;
        let reqs = docs.into_iter().map(|d| WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in dest.write(reqs, false) {
            r?;
        }
        Ok(())
    }
    fn refresh(&self, index: &str) -> Result<()> {
        self.0.get_index(index).ok_or_else(|| no_such_index(index))?.refresh()
    }
    fn block_writes(&self, index: &str) -> Result<()> {
        let dest = self.0.get_index(index).ok_or_else(|| no_such_index(index))?;
        let mut done = FlatSettings::new();
        done.insert("index.blocks.write".into(), json!("true"));
        dest.put_settings(done, false)
    }
    fn delete_index(&self, index: &str) -> Result<()> {
        self.0.delete_index(index)
    }
    fn mark_success(&self, index: &str) -> Result<()> {
        self.0.get_index(index).ok_or_else(|| no_such_index(index))?.set_downsample_status("success")
    }
}

/// `POST /{index}/_downsample/{target_index}` on this node's engine.
pub fn downsample(node: &Node, source: &str, target: &str, body: &Value) -> Result<Value> {
    downsample_with(node, source, target, body, &LocalDownsampleOps(node))
}

/// `POST /{index}/_downsample/{target_index}`: validation against the
/// node's metadata, index work through `ops`.
pub fn downsample_with(node: &Node, source: &str, target: &str, body: &Value, ops: &dyn DownsampleOps) -> Result<Value> {
    let empty = Map::new();
    let o = match body {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err(EsError::x_content_parse("[1:1] [downsample/action/config] expected an object")),
    };
    for k in o.keys() {
        if k != "fixed_interval" && k != "sampling_method" {
            return Err(EsError::x_content_parse(format!("[1:2] [downsample/action/config] unknown field [{k}]")));
        }
    }
    let sampling = match o.get("sampling_method") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("aggregate") => Some("aggregate"),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("last_value") => Some("last_value"),
        Some(v) => {
            let s = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
            return Err(EsError::x_content_parse("[1:2] [downsample/action/config] failed to parse field [sampling_method]")
                .caused_by(EsError::illegal_argument(format!("Unknown value [{s}] for field [sampling_method] - accepted values are [aggregate, last_value]"))));
        }
    };
    let method = sampling.unwrap_or("aggregate");
    let aggregate = method == "aggregate";
    let interval_s = match o.get("fixed_interval") {
        Some(Value::String(s)) => s.clone(),
        Some(v) if !v.is_null() => v.to_string(),
        _ => return Err(EsError::illegal_argument("Required [fixed_interval]")),
    };
    let interval = crate::date::parse_fixed_interval(&interval_s).ok_or_else(|| {
        EsError::x_content_parse("[1:2] [downsample/action/config] failed to parse field [fixed_interval]").caused_by(
            EsError::x_content_parse("Failed to build [downsample/action/config] after last required field arrived").caused_by(EsError::illegal_argument(format!(
                "failed to parse setting [createRounding] with value [{interval_s}] as a time value: unit is missing or unrecognized"
            ))),
        )
    })?;
    let svc = node.get_index(source).ok_or_else(|| no_such_index(source))?;
    let settings = svc.meta().settings;
    if !crate::tsdb::is_time_series(&settings) {
        return Err(EsError::new(500, "exception", format!("Rollup requires setting [index.mode=time_series] for index [{source}]")));
    }
    if svc.setting("index.blocks.write").as_deref() != Some("true") {
        return Err(EsError::new(500, "exception", format!("Downsample requires setting [index.blocks.write = true] for index [{source}]")));
    }
    // One run per target on this node: a rerun never deletes a target
    // another run is still filling.
    let _running = InFlight::claim(node, target).ok_or_else(|| {
        EsError::new(409, "resource_already_exists_exception", format!("downsample task for index [{target}] is already running"))
    })?;
    if let Some(existing) = node.get_index(target) {
        if existing.setting("index.downsample.source.uuid").as_deref() != Some(svc.uuid.as_str()) {
            return Err(EsError::index_already_exists(target, &existing.uuid));
        }
        // A finished downsample of this source into this target is acknowledged again.
        if existing.setting(crate::settings::DOWNSAMPLE_STATUS).as_deref() == Some("success") {
            return Ok(json!({"acknowledged": true}));
        }
        // `TransportDownsampleAction`: a `started` target carrying the
        // write block holds every document (the block is set only after
        // the last one was written and refreshed); the run died before
        // the status update. Resumed from the refresh.
        if existing.setting("index.blocks.write").as_deref() == Some("true") {
            drop(existing);
            ops.refresh(target)?;
            ops.mark_success(target)?;
            return Ok(json!({"acknowledged": true}));
        }
        // An unfinished one (a run that died mid-copy) holds an unknown
        // part of the documents: deleted and rebuilt.
        drop(existing);
        ops.delete_index(target)?;
    }
    if let Some(prev) = svc.setting("index.downsample.interval") {
        let mut errors = Vec::new();
        let p = crate::date::parse_fixed_interval(&prev).unwrap_or(1);
        let why = if interval <= p { "be greater than" } else if interval % p != 0 { "be a multiple of" } else { "" };
        if !why.is_empty() {
            errors.push(format!("Source index is a downsampled index. Downsampling interval [{interval_s}] must {why} the source index interval [{prev}]."));
        }
        // `DownsampleConfig.SamplingMethod.fromIndexMetadata`.
        let source_method = svc.setting("index.downsample.sampling_method").unwrap_or_else(|| "aggregate".into());
        if source_method != method {
            errors.push(format!("Source index is a downsampled index. Downsampling method [{method}] is not compatible with the source index downsampling method [{source_method}]."));
        }
        if !errors.is_empty() {
            let refs: Vec<&str> = errors.iter().map(String::as_str).collect();
            return Err(EsError::action_validation(&refs));
        }
    }
    // Target index: the source's settings (minus private ones and blocks)
    // with downsample bookkeeping; mapping with aggregated gauges.
    // TransportDownsampleAction: only these source settings carry over.
    let carried = |k: &str| {
        matches!(k, "index.number_of_shards" | "index.number_of_replicas" | "index.mode" | "index.routing_path" | "index.time_series.start_time" | "index.time_series.end_time" | "index.routing.allocation.include._tier_preference" | "index.hidden")
            || k.starts_with("index.routing_path")
            || k.starts_with("index.time_series.")
            || k.starts_with("index.analysis.")
    };
    let mut flat: FlatSettings = settings.iter().filter(|(k, _)| carried(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
    let (origin_name, origin_uuid) = match (svc.setting("index.downsample.origin.name"), svc.setting("index.downsample.origin.uuid")) {
        (Some(n), Some(u)) => (n, u),
        _ => (source.to_string(), svc.uuid.clone()),
    };
    flat.insert("index.mapping.source.mode".into(), json!("STORED"));
    flat.insert("index.refresh_interval".into(), json!("-1"));
    flat.insert("index.downsample.interval".into(), json!(interval_s));
    flat.insert("index.downsample.sampling_method".into(), json!(method));
    flat.insert("index.downsample.source.name".into(), json!(source));
    flat.insert("index.downsample.source.uuid".into(), json!(svc.uuid));
    flat.insert("index.downsample.origin.name".into(), json!(origin_name));
    flat.insert("index.downsample.origin.uuid".into(), json!(origin_uuid));
    // `index.downsample.status`: `started` until every document is copied
    // and refreshed, then `success` (only a successful target is used). A
    // failed run deletes its target; one interrupted mid-copy is rebuilt.
    flat.insert(crate::settings::DOWNSAMPLE_STATUS.into(), json!("started"));
    let src_map = svc.meta().mappings.to_json();
    let mut fields = Vec::new();
    if let Some(p) = src_map.get("properties").and_then(Value::as_object) {
        classify(p, "", &mut fields)?;
    }
    let temporality_field = crate::settings::get_str(&settings, "index.time_series.temporality_field").map(str::to_string);
    // ES downsampling reads doc values, never `_source`. Here the source is
    // read, so fields a mapping-level `_source` filter keeps out of the
    // stored source (still indexed, with doc values) are read from their
    // doc values instead; one that has none to read is refused before any
    // target exists.
    let flat_keys = src_map.get("subobjects").and_then(Value::as_bool) == Some(false);
    let doc_value_reads = source_pruned_reads(source, &src_map, &fields, flat_keys)?;
    let docvalue_fields: Vec<Value> = doc_value_reads.iter().map(|(p, ts)| if *ts { json!({"field": p, "format": "epoch_millis"}) } else { json!({"field": p}) }).collect();
    ops.create_index(target, &json!({"settings": crate::settings::unflatten(&flat), "mappings": target_mappings(&src_map, &interval_s, aggregate)}))?;
    // The target this run created: a later step finding another index
    // under the name (a concurrent rerun replaced it) neither marks nor
    // deletes that one.
    let created = node.get_index(target);
    let created_uuid = created.as_ref().map(|t| t.uuid.clone());
    let still_ours = || match &created_uuid {
        Some(u) => node.get_index(target).is_some_and(|t| &t.uuid == u),
        None => true,
    };
    // A target whose shards cannot be allocated on this node (allocation
    // filter on a node attribute it lacks) fails like ES's shard task.
    let unallocatable = created.is_some_and(|dest| {
        dest.meta().settings.keys().any(|k| {
            ["index.routing.allocation.include.", "index.routing.allocation.require."].iter().any(|p| k.strip_prefix(p).map(|a| !a.starts_with('_')).unwrap_or(false))
        })
    });
    if unallocatable {
        let _ = ops.delete_index(target);
        return Err(EsError::new(500, "elasticsearch_exception", format!("downsample task [downsample-{target}-0-{interval_s}] failed")));
    }
    let replaced = || EsError::new(500, "illegal_state_exception", format!("downsample target index [{target}] was replaced while the downsample was running"));
    let result = (|| -> Result<()> {
        // `aggregate_metric_double` fields: their declared metrics.
        let mut agg_metrics: HashMap<String, Vec<String>> = HashMap::new();
        fn agg_walk(props: &Map<String, Value>, prefix: &str, out: &mut HashMap<String, Vec<String>>) {
            for (name, def) in props {
                let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
                if let Some(sub) = def.get("properties").and_then(Value::as_object) {
                    agg_walk(sub, &path, out);
                } else if def["type"].as_str() == Some("aggregate_metric_double") {
                    out.insert(path, def["metrics"].as_array().map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()).unwrap_or_default());
                }
            }
        }
        if let Some(p) = src_map.get("properties").and_then(Value::as_object) {
            agg_walk(p, "", &mut agg_metrics);
        }
        // Bucket timestamps in the field's own format.
        let ts_format = src_map["properties"][crate::datastream::TIMESTAMP_FIELD]["format"].as_str().map(|f| f.split("||").next().unwrap_or(f).to_string());
        let ctx = BucketCtx {
            fields: &fields,
            agg_metrics: &agg_metrics,
            aggregate,
            formatter: ts_format.as_deref().and_then(|f| crate::date::DateFormatter::new(f).ok()),
            // Field paths become nested objects, except under `subobjects: false`.
            flat_keys,
        };
        let parse_fmt = src_map["properties"][crate::datastream::TIMESTAMP_FIELD]["format"].as_str().map(str::to_string);
        let src_meta = svc.meta();
        let mut writer = BatchWriter { ops, target, batch: Vec::new(), batch_bytes: 0, written: 0 };
        let mut charge = DownsampleCharge::default();
        let mut series: Option<SeriesState> = None;
        // `DownsampleShardIndexer`: the source in index order, one time
        // series (`_tsid`) at a time, newest first; each bucket of a series
        // is aggregated as its documents stream by and written once the
        // stream leaves it. Held: one bucket's collectors and one output
        // batch, on the `request` breaker.
        let mut scanned = 0u64;
        ops.scan_series(source, &docvalue_fields, &mut |hit| {
            scanned += 1;
            let tsid = hit["sort"][0].as_str();
            let filled;
            let src = if doc_value_reads.is_empty() {
                hit.get("_source").unwrap_or(&Value::Null)
            } else {
                filled = with_doc_values(hit, &doc_value_reads, flat_keys);
                &filled
            };
            // Every time series document has a timestamp: one that cannot
            // be read fails the run, never silently leaves the output.
            let Some(ts) = src_timestamp(src, parse_fmt.as_deref()) else {
                return Err(EsError::new(
                    500,
                    "illegal_state_exception",
                    format!("downsample of [{source}]: document [{}] has no readable [{}]", hit["_id"].as_str().unwrap_or_default(), crate::datastream::TIMESTAMP_FIELD),
                ));
            };
            // The series identity is the indexed `_tsid` (every dimension,
            // `flattened` sub-key dimensions included), as ES groups.
            let key = match tsid {
                Some(t) => t.to_string(),
                None => crate::tsdb::build_ids(&src_meta.mappings, &settings, src).map(|ids| ids.tsid).unwrap_or_else(|_| serde_json::to_string(&dimension_values(&fields, src)).unwrap_or_default()),
            };
            if series.as_ref().is_none_or(|s| s.tsid != key) {
                if let Some(mut done) = series.take() {
                    done.finish_bucket(&ctx, &mut writer, &mut charge)?;
                }
                let dims = dimension_values(&fields, src);
                let delta = temporality_field.as_ref().and_then(|t| dims.iter().find(|(p, _)| p == t)).and_then(|(_, v)| v.as_str()).map(str::to_string);
                series = Some(SeriesState {
                    tsid: key,
                    counter_delta: delta.as_deref() == Some("delta"),
                    // Histograms are delta unless the series says cumulative.
                    hist_cumulative: delta.as_deref() == Some("cumulative"),
                    dims,
                    cumulative: HashMap::new(),
                    bucket: None,
                });
            }
            let s = series.as_mut().expect("series");
            let start = ts - ts.rem_euclid(interval);
            match s.bucket.as_ref().map(|b| b.start) {
                Some(cur) if cur == start => {}
                Some(cur) if cur < start => {
                    return Err(EsError::internal(format!("downsample of [{source}]: time series [{}] is not in descending @timestamp order", s.tsid)));
                }
                _ => {
                    s.finish_bucket(&ctx, &mut writer, &mut charge)?;
                    s.bucket = Some(Bucket::new(start, &ctx, s.counter_delta, s.hist_cumulative));
                }
            }
            s.feed(&ctx, ts, src);
            charge.set(writer.batch_bytes + s.bucket.as_ref().map_or(0, |b| b.bytes))
        })?;
        if let Some(mut done) = series.take() {
            done.finish_bucket(&ctx, &mut writer, &mut charge)?;
        }
        writer.flush()?;
        drop(charge);
        // Source documents read but nothing written is never a success.
        if scanned > 0 && writer.written == 0 {
            return Err(EsError::new(500, "illegal_state_exception", format!("downsample of [{source}] read {scanned} documents but produced none")));
        }
        ops.refresh(target)?;
        // Each finalisation step only on the target this run filled.
        if !still_ours() {
            return Err(replaced());
        }
        ops.block_writes(target)?;
        if !still_ours() {
            return Err(replaced());
        }
        ops.mark_success(target)
    })();
    if let Err(e) = result {
        // A failed run's target is removed, unless another run's took its place.
        if still_ours() {
            let _ = ops.delete_index(target);
        }
        return Err(e);
    }
    Ok(json!({"acknowledged": true}))
}

/// The breaker label of a downsample's memory.
const DOWNSAMPLE_BREAKER_LABEL: &str = "downsample";

/// A downsample's held memory on the `request` breaker, reserved in
/// 64 KiB steps before it grows and given back as it shrinks.
#[derive(Default)]
struct DownsampleCharge {
    res: Option<crate::memory::Reservation>,
}

#[cfg(test)]
thread_local! {
    /// The largest downsample reservation on this thread (tests).
    pub(crate) static DOWNSAMPLE_PEAK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl DownsampleCharge {
    const STEP: u64 = 64 * 1024;

    fn set(&mut self, bytes: u64) -> Result<()> {
        let want = bytes.div_ceil(Self::STEP) * Self::STEP;
        let have = self.res.as_ref().map_or(0, |r| r.bytes());
        if want > have {
            match &mut self.res {
                Some(r) => r.grow(want - have, DOWNSAMPLE_BREAKER_LABEL)?,
                None => self.res = Some(crate::memory::breakers().reserve_request(want, DOWNSAMPLE_BREAKER_LABEL)?),
            }
        } else if want < have {
            if let Some(r) = &mut self.res {
                r.shrink(have - want);
            }
        }
        #[cfg(test)]
        DOWNSAMPLE_PEAK.with(|p| p.set(p.get().max(want)));
        Ok(())
    }
}

/// Bytes a JSON value holds (a breaker estimate: a node per value and
/// member plus string and key bytes).
fn value_bytes(v: &Value) -> u64 {
    const NODE: u64 = 32;
    match v {
        Value::String(s) => NODE + s.len() as u64,
        Value::Array(a) => NODE + a.iter().map(value_bytes).sum::<u64>(),
        Value::Object(o) => NODE + o.iter().map(|(k, v)| NODE + k.len() as u64 + value_bytes(v)).sum::<u64>(),
        _ => NODE,
    }
}

/// The dimension values of a source document (multi-valued ones in doc
/// value order).
fn dimension_values(fields: &[(String, Kind)], src: &Value) -> Vec<(String, Value)> {
    fields
        .iter()
        .filter(|(_, k)| *k == Kind::Dimension)
        .filter_map(|(p, _)| {
            let vals = crate::highlight::source_values(src, p);
            match vals.len() {
                0 => None,
                1 => Some((p.clone(), vals[0].clone())),
                // Multi-valued dimensions keep all their (doc value ordered) values.
                _ => {
                    let mut a: Vec<Value> = vals.into_iter().cloned().collect();
                    if a.iter().all(Value::is_string) {
                        a.sort_by(|x, y| x.as_str().cmp(&y.as_str()));
                        a.dedup();
                    } else if a.iter().all(Value::is_number) {
                        a.sort_by(|x, y| num(x).partial_cmp(&num(y)).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    Some((p.clone(), Value::Array(a)))
                }
            }
        })
        .collect()
}

/// What every bucket of a downsample run shares.
struct BucketCtx<'a> {
    fields: &'a [(String, Kind)],
    agg_metrics: &'a HashMap<String, Vec<String>>,
    aggregate: bool,
    formatter: Option<crate::date::DateFormatter>,
    flat_keys: bool,
}

impl BucketCtx<'_> {
    fn fmt_ts(&self, ms: i64) -> String {
        match &self.formatter {
            Some(f) => f.format(ms, crate::date::Tz::UTC),
            None => crate::ilm::iso_date(ms),
        }
    }
}

/// Output documents, indexed 500 at a time.
struct BatchWriter<'a> {
    ops: &'a dyn DownsampleOps,
    target: &'a str,
    batch: Vec<Value>,
    batch_bytes: u64,
    /// Documents emitted so far.
    written: u64,
}

impl BatchWriter<'_> {
    fn emit(&mut self, doc: Map<String, Value>, charge: &mut DownsampleCharge) -> Result<()> {
        let doc = Value::Object(doc);
        self.written += 1;
        self.batch_bytes += value_bytes(&doc);
        self.batch.push(doc);
        charge.set(self.batch_bytes)?;
        if self.batch.len() >= 500 {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.batch.is_empty() {
            self.ops.index_docs(self.target, std::mem::take(&mut self.batch))?;
        }
        self.batch_bytes = 0;
        Ok(())
    }
}

/// One field's collector over a bucket of one time series, fed newest
/// document first (`AbstractDownsampleFieldProducer`).
enum Collector {
    None,
    /// An aggregated gauge: min, max, sum, value count.
    Gauge { min: f64, max: f64, sum: crate::fieldtypes::exphist::KahanSum, count: u64 },
    /// The newest value (last-value metrics and labels, non-metric
    /// histograms, `aggregate_metric_double` labels).
    Last(Option<Value>),
    /// A delta counter: the sum of each document's value.
    DeltaSum(Option<crate::fieldtypes::exphist::KahanSum>),
    /// A cumulative counter or histogram, in the series' [`Cumulative`]
    /// state (`true` once the bucket fed it).
    Cumulative(bool),
    /// An aggregated `aggregate_metric_double` gauge: per declared metric.
    AggMetric { seen: bool, min: Option<f64>, max: Option<f64>, sum: Option<crate::fieldtypes::exphist::KahanSum>, count: Option<i64> },
    /// A merged histogram / t-digest (`None` until one parses).
    Hist(Option<BTreeMap<u64, (f64, i64)>>),
    /// A merged exponential histogram.
    ExpHist(Option<crate::fieldtypes::exphist::Merger>),
}

/// The documents of one time series falling in one bucket.
struct Bucket {
    start: i64,
    count: u64,
    collectors: Vec<Collector>,
    /// Estimated bytes the collectors hold.
    bytes: u64,
}

impl Bucket {
    fn new(start: i64, ctx: &BucketCtx<'_>, counter_delta: bool, hist_cumulative: bool) -> Bucket {
        let aggregate = ctx.aggregate;
        let collectors = ctx
            .fields
            .iter()
            .map(|(_, kind)| match *kind {
                Kind::Dimension => Collector::None,
                Kind::Gauge if aggregate => Collector::Gauge { min: f64::MAX, max: -f64::MAX, sum: Default::default(), count: 0 },
                Kind::Gauge | Kind::Counter if !aggregate => Collector::Last(None),
                Kind::Counter if counter_delta => Collector::DeltaSum(None),
                Kind::Counter => Collector::Cumulative(false),
                Kind::Aggregate { metric } if metric && aggregate => Collector::AggMetric { seen: false, min: None, max: None, sum: None, count: None },
                Kind::Histogram { metric, .. } if metric && aggregate => Collector::Hist(None),
                Kind::ExpHist { metric } if metric && aggregate && hist_cumulative => Collector::Cumulative(false),
                Kind::ExpHist { metric } if metric && aggregate => Collector::ExpHist(None),
                Kind::Aggregate { .. } | Kind::Histogram { .. } | Kind::ExpHist { .. } | Kind::Label(_) => Collector::Last(None),
                _ => Collector::None,
            })
            .collect();
        Bucket { start, count: 0, collectors, bytes: 0 }
    }
}

/// The time series a downsample run is in.
struct SeriesState {
    tsid: String,
    dims: Vec<(String, Value)>,
    counter_delta: bool,
    hist_cumulative: bool,
    cumulative: HashMap<String, Cumulative>,
    bucket: Option<Bucket>,
}

impl SeriesState {
    /// Fold one source document (older than every one fed before) into
    /// the current bucket.
    fn feed(&mut self, ctx: &BucketCtx<'_>, ts: i64, src: &Value) {
        use crate::fieldtypes::exphist;
        let Some(b) = self.bucket.as_mut() else { return };
        b.count += src.get("_doc_count").and_then(Value::as_u64).unwrap_or(1);
        for ((p, kind), c) in ctx.fields.iter().zip(b.collectors.iter_mut()) {
            if matches!(c, Collector::None) {
                continue;
            }
            let vals = crate::highlight::source_values(src, p);
            if vals.is_empty() {
                continue;
            }
            match c {
                Collector::None => {}
                Collector::Gauge { min, max, sum, count } => {
                    for x in sorted_nums(&vals) {
                        *min = min.min(x);
                        *max = max.max(x);
                        sum.add(x);
                        *count += 1;
                    }
                }
                Collector::DeltaSum(s) => {
                    if let Some(x) = sorted_nums(&vals).first() {
                        s.get_or_insert_with(Default::default).add(*x);
                    }
                }
                Collector::Cumulative(touched) => match kind {
                    Kind::Counter => {
                        if let Some(x) = sorted_nums(&vals).first() {
                            self.cumulative.entry(p.clone()).or_insert_with(Cumulative::new).collect(ResetValue::Counter(*x), ts);
                            *touched = true;
                        }
                    }
                    _ => {
                        for h in vals.iter().filter_map(|h| exphist::parse("", h, false).ok()) {
                            self.cumulative.entry(p.clone()).or_insert_with(Cumulative::new).collect(ResetValue::Hist(exphist::render(&h)), ts);
                            *touched = true;
                        }
                    }
                },
                Collector::AggMetric { seen, min, max, sum, count } => {
                    let metrics = ctx.agg_metrics.get(p).map(Vec::as_slice).unwrap_or_default();
                    let has = |k: &str| metrics.iter().any(|m| m == k);
                    for o in vals.iter().filter(|v| v.is_object()) {
                        *seen = true;
                        if has("min") {
                            if let Some(x) = o.get("min").and_then(num) {
                                *min = Some(min.unwrap_or(f64::MAX).min(x));
                            }
                        }
                        if has("max") {
                            if let Some(x) = o.get("max").and_then(num) {
                                *max = Some(max.unwrap_or(-f64::MAX).max(x));
                            }
                        }
                        if has("sum") {
                            if let Some(x) = o.get("sum").and_then(num) {
                                sum.get_or_insert_with(Default::default).add(x);
                            }
                        }
                        if has("value_count") {
                            if let Some(x) = o.get("value_count").and_then(num) {
                                *count = Some(count.unwrap_or(0) + x.round() as i64);
                            }
                        }
                    }
                }
                Collector::Hist(merged) => {
                    let Kind::Histogram { tdigest, .. } = *kind else { continue };
                    let ty = if tdigest { crate::mapping::EsType::TDigest } else { crate::mapping::EsType::Histogram };
                    for (vs, cs) in vals.iter().filter_map(|h| crate::fieldtypes::index::parse_histogram(ty, h).ok()) {
                        let m = merged.get_or_insert_with(BTreeMap::new);
                        // A merging t-digest over few centroids keeps
                        // every distinct value.
                        for (x, c) in vs.iter().zip(&cs) {
                            if *c == 0 {
                                continue;
                            }
                            let key = if *x >= 0.0 { x.to_bits() ^ (1 << 63) } else { !x.to_bits() };
                            match m.entry(key) {
                                std::collections::btree_map::Entry::Vacant(e) => {
                                    b.bytes += 48;
                                    e.insert((*x, *c));
                                }
                                std::collections::btree_map::Entry::Occupied(mut e) => e.get_mut().1 += c,
                            }
                        }
                    }
                }
                Collector::ExpHist(m) => {
                    for h in vals.iter().filter_map(|h| exphist::parse("", h, false).ok()) {
                        m.get_or_insert_with(exphist::Merger::new).add(&h);
                    }
                }
                Collector::Last(v) => {
                    if v.is_some() {
                        continue;
                    }
                    *v = last_value(*kind, &vals);
                    if let Some(v) = v {
                        b.bytes += value_bytes(v);
                    }
                }
            }
        }
    }

    /// Write the current bucket's downsampled document (and its counter
    /// reset documents), if any.
    fn finish_bucket(&mut self, ctx: &BucketCtx<'_>, writer: &mut BatchWriter<'_>, charge: &mut DownsampleCharge) -> Result<()> {
        use crate::fieldtypes::exphist;
        let Some(b) = self.bucket.take() else { return Ok(()) };
        let put = |doc: &mut Map<String, Value>, p: &str, v: Value| put_path(doc, p, v, ctx.flat_keys);
        let mut doc = Map::new();
        doc.insert(crate::datastream::TIMESTAMP_FIELD.into(), json!(ctx.fmt_ts(b.start)));
        for (p, v) in &self.dims {
            put(&mut doc, p, v.clone());
        }
        let mut resets: BTreeMap<i64, Vec<(String, ResetValue)>> = BTreeMap::new();
        for ((p, kind), c) in ctx.fields.iter().zip(b.collectors) {
            let value = match c {
                Collector::None => None,
                Collector::Gauge { min, max, sum, count } => (count > 0).then(|| json!({"min": min, "max": max, "sum": sum.value(), "value_count": count})),
                Collector::Last(v) => v,
                Collector::DeltaSum(s) => s.map(|s| json!(s.value())),
                Collector::Cumulative(false) => None,
                Collector::Cumulative(true) => {
                    let c = self.cumulative.entry(p.clone()).or_insert_with(Cumulative::new);
                    let v = match (&c.value, kind) {
                        (Some(ResetValue::Counter(x)), Kind::Counter) => Some(json!(x)),
                        (Some(ResetValue::Hist(h)), Kind::ExpHist { .. }) => Some(h.clone()),
                        _ => None,
                    };
                    for (t, r) in c.bucket_resets() {
                        resets.entry(t).or_default().push((p.clone(), r));
                    }
                    c.reset_bucket();
                    v
                }
                Collector::AggMetric { seen, min, max, sum, count } => seen.then(|| {
                    let mut m = Map::new();
                    if let Some(x) = min {
                        m.insert("min".into(), json!(x));
                    }
                    if let Some(x) = max {
                        m.insert("max".into(), json!(x));
                    }
                    if let Some(s) = sum {
                        m.insert("sum".into(), json!(s.value()));
                    }
                    if let Some(n) = count {
                        m.insert("value_count".into(), json!(n));
                    }
                    Value::Object(m)
                }),
                Collector::Hist(merged) => merged.map(|merged| {
                    let label = if matches!(kind, Kind::Histogram { tdigest: true, .. }) { "centroids" } else { "values" };
                    let values: Vec<Value> = merged.values().map(|(x, _)| json!(x)).collect();
                    let counts: Vec<Value> = merged.values().map(|(_, c)| json!(c)).collect();
                    json!({"counts": counts, label: values})
                }),
                Collector::ExpHist(m) => m.map(|m| exphist::render(&m.get())),
            };
            if let Some(v) = value {
                put(&mut doc, p, v);
            }
        }
        // Reset documents are counted out of the downsampled one.
        let main_count = b.count.saturating_sub(resets.len() as u64).max(1);
        doc.insert("_doc_count".into(), json!(main_count));
        writer.emit(doc, charge)?;
        for (t, values) in resets {
            let mut rd = Map::new();
            rd.insert(crate::datastream::TIMESTAMP_FIELD.into(), json!(ctx.fmt_ts(t)));
            for (p, v) in &self.dims {
                put(&mut rd, p, v.clone());
            }
            for (p, v) in values {
                let v = match v {
                    ResetValue::Counter(x) => json!(x),
                    ResetValue::Hist(h) => h,
                };
                put(&mut rd, &p, v);
            }
            writer.emit(rd, charge)?;
        }
        Ok(())
    }
}

/// The value a last-value collector keeps from the newest document with
/// one (`vals`: that document's values of the field, non-empty).
fn last_value(kind: Kind, vals: &[&Value]) -> Option<Value> {
    use crate::fieldtypes::exphist;
    match kind {
        Kind::Gauge | Kind::Counter => sorted_nums(vals).first().map(|x| json!(x)),
        Kind::Aggregate { .. } => {
            // Last value, per metric.
            let latest = vals.iter().find(|v| v.is_object())?;
            let mut m = Map::new();
            for k in ["min", "max", "sum", "value_count"] {
                if let Some(x) = latest.get(k).and_then(num) {
                    m.insert(k.into(), if k == "value_count" { json!(x as i64) } else { json!(x) });
                }
            }
            Some(Value::Object(m))
        }
        Kind::Histogram { tdigest, .. } => {
            let label = if tdigest { "centroids" } else { "values" };
            let ty = if tdigest { crate::mapping::EsType::TDigest } else { crate::mapping::EsType::Histogram };
            let (vs, cs) = vals.iter().find_map(|h| crate::fieldtypes::index::parse_histogram(ty, h).ok())?;
            Some(json!({"counts": cs, label: vs}))
        }
        Kind::ExpHist { .. } => vals.iter().find_map(|h| exphist::parse("", h, false).ok()).map(|h| exphist::render(&h)),
        Kind::Label(max) => {
            let vals: Vec<&Value> = match max {
                // Values over `ignore_above` are not in doc values.
                Some(max) => vals.iter().copied().filter(|v| v.as_str().map(|s| s.chars().count() <= max).unwrap_or(true)).collect(),
                None => vals.to_vec(),
            };
            match vals.len() {
                0 => None,
                1 => Some(vals[0].clone()),
                // Labels come from doc values: numbers sorted, strings sorted and deduplicated.
                _ => {
                    let mut a: Vec<Value> = vals.into_iter().cloned().collect();
                    if a.iter().all(Value::is_number) {
                        a.sort_by(|x, y| num(x).partial_cmp(&num(y)).unwrap_or(std::cmp::Ordering::Equal));
                    } else if a.iter().all(Value::is_string) {
                        a.sort_by(|x, y| x.as_str().cmp(&y.as_str()));
                        a.dedup();
                    }
                    Some(Value::Array(a))
                }
            }
        }
        Kind::Dimension => None,
    }
}

/// Insert `v` at dotted `path`: nested objects, or one flat key when `flat`.
fn put_path(doc: &mut Map<String, Value>, path: &str, v: Value, flat: bool) {
    if flat {
        doc.insert(path.to_string(), v);
        return;
    }
    let mut parts = path.split('.').peekable();
    let mut cur = doc;
    while let Some(k) = parts.next() {
        if parts.peek().is_none() {
            cur.insert(k.to_string(), v);
            return;
        }
        let e = cur.entry(k.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if !e.is_object() {
            *e = Value::Object(Map::new());
        }
        cur = e.as_object_mut().expect("object");
    }
}

/// Field types whose doc values `docvalue_fields` returns as the values a
/// source would hold.
fn doc_value_readable(ty: &str) -> bool {
    matches!(
        ty,
        "date" | "date_nanos" | "keyword" | "constant_keyword" | "ip" | "boolean" | "long" | "integer" | "short" | "byte" | "double" | "float" | "half_float" | "scaled_float" | "unsigned_long"
    )
}

/// The downsample inputs (`@timestamp` and every classified field) that the
/// source mapping's `_source` filter (`includes` / `excludes`) keeps out of
/// the stored source, as `(path, is_timestamp)`: they are read from doc
/// values. A filtered input that has no doc values to read (`doc_values:
/// false`, or a type whose doc values are not its source form) refuses the
/// downsample: its output would silently lack the field.
fn source_pruned_reads(source: &str, src_map: &Value, fields: &[(String, Kind)], flat: bool) -> Result<Vec<(String, bool)>> {
    let filter = &src_map["_source"];
    let list = |k: &str| -> Vec<String> { filter[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
    let (includes, excludes) = (list("includes"), list("excludes"));
    if includes.is_empty() && excludes.is_empty() {
        return Ok(Vec::new());
    }
    let ts = crate::datastream::TIMESTAMP_FIELD;
    let paths: Vec<&str> = std::iter::once(ts).chain(fields.iter().map(|(p, _)| p.as_str())).collect();
    let mut probe = Map::new();
    for p in &paths {
        put_path(&mut probe, p, json!(0), flat);
    }
    let kept = crate::search::filter_source(&Value::Object(probe), &includes, &excludes);
    let mut out = Vec::new();
    for p in paths {
        if !crate::highlight::source_values(&kept, p).is_empty() {
            continue;
        }
        let def = p.split('.').fold(Some(src_map), |d, k| d.and_then(|d| d["properties"].get(k)));
        let ty = def.and_then(|d| d["type"].as_str()).unwrap_or("object");
        // A text label has no doc values in ES either: never carried over.
        let label = fields.iter().any(|(fp, k)| fp == p && matches!(k, Kind::Label(_)));
        if label && matches!(ty, "text" | "match_only_text" | "annotated_text") {
            continue;
        }
        if !doc_value_readable(ty) || def.is_some_and(|d| d["doc_values"].as_bool() == Some(false)) {
            return Err(EsError::illegal_argument(format!(
                "Downsampling of index [{source}] is not supported: its [_source] filtering removes field [{p}] of type [{ty}], which cannot be read from doc values"
            )));
        }
        out.push((p.to_string(), p == ts));
    }
    Ok(out)
}

/// A hit's `_source` with the doc-value-read fields put back (`reads`:
/// from [`source_pruned_reads`]; the timestamp read as epoch millis).
fn with_doc_values(hit: &Value, reads: &[(String, bool)], flat: bool) -> Value {
    let mut src = match hit.get("_source") {
        Some(Value::Object(o)) => o.clone(),
        _ => Map::new(),
    };
    for (p, is_ts) in reads {
        let Some(vals) = hit["fields"][p.as_str()].as_array().filter(|a| !a.is_empty()) else { continue };
        let v = if *is_ts {
            match vals[0].as_i64().or_else(|| vals[0].as_str().and_then(|s| s.parse::<f64>().ok()).map(|f| f as i64)) {
                Some(ms) => json!(ms),
                None => continue,
            }
        } else if vals.len() == 1 {
            vals[0].clone()
        } else {
            Value::Array(vals.clone())
        };
        put_path(&mut src, p, v, flat);
    }
    Value::Object(src)
}

fn src_timestamp(src: &Value, format: Option<&str>) -> Option<i64> {
    let v = crate::highlight::source_values(src, crate::datastream::TIMESTAMP_FIELD).into_iter().next()?.clone();
    match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => crate::date::DateFormatter::new(format.unwrap_or("strict_date_optional_time||epoch_millis")).ok().and_then(|f| f.parse(&s).ok()),
        _ => None,
    }
}

/// Visit every document `_source` of `index` (a point-in-time scan in `_doc` order).
fn scan_sources(node: &Node, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
    scan_hits(node, index, &json!(["_doc"]), &[], &mut |h| {
        if let Some(s) = h.get("_source") {
            f(s);
        }
        Ok(())
    })
}

/// Visit every hit of `index` in `sort` order, a page of 1000 at a time.
fn scan_hits(node: &Node, index: &str, sort: &Value, docvalue_fields: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
    let svc = node.get_index(index).ok_or_else(|| no_such_index(index))?;
    let targets = vec![crate::search::Target::new(svc)];
    let mut body = json!({"size": 1000, "sort": sort, "track_total_hits": false});
    if !docvalue_fields.is_empty() {
        body["docvalue_fields"] = Value::Array(docvalue_fields.to_vec());
    }
    let req = crate::search::parse_request(&body, &HashMap::new())?;
    let mut cursors: Option<Vec<crate::search::Cursor>> = None;
    loop {
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: cursors.as_deref(), max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&targets, &req, &opts)?;
        let v: Value = serde_json::from_str(&out.json)?;
        let hits = v["hits"]["hits"].as_array().cloned().unwrap_or_default();
        if hits.is_empty() {
            break;
        }
        for h in &hits {
            f(h)?;
        }
        cursors = Some(out.cursors);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod downsample_safety_tests {
    use super::*;
    use std::cell::Cell;

    /// A run that dies after its first batch of target documents: the
    /// second batch fails and, the process being gone, nothing cleans up.
    pub(crate) struct CrashAfterFirstBatch<'a> {
        pub inner: LocalDownsampleOps<'a>,
        pub batches: Cell<usize>,
        /// Die only after every batch, just before the status flips.
        pub before_success: bool,
    }

    impl DownsampleOps for CrashAfterFirstBatch<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<()> {
            self.inner.create_index(name, body)
        }
        fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
            self.inner.scan(index, f)
        }
        fn scan_series(&self, index: &str, dv: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
            self.inner.scan_series(index, dv, f)
        }
        fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()> {
            self.batches.set(self.batches.get() + 1);
            if self.batches.get() > 1 && !self.before_success {
                return Err(EsError::internal("node stopped mid-copy"));
            }
            self.inner.index_docs(index, docs)
        }
        fn refresh(&self, index: &str) -> Result<()> {
            self.inner.refresh(index)
        }
        fn block_writes(&self, index: &str) -> Result<()> {
            self.inner.block_writes(index)
        }
        fn delete_index(&self, _index: &str) -> Result<()> {
            Ok(())
        }
        fn mark_success(&self, index: &str) -> Result<()> {
            if self.before_success {
                return Err(EsError::internal("node stopped before the status update"));
            }
            self.inner.mark_success(index)
        }
    }

    fn iso(ms: i64) -> String {
        crate::ilm::iso_date(ms)
    }

    /// A write-blocked time series index of 600 series × 2 documents in one
    /// hour (600 downsampled documents: more than one target batch).
    pub(crate) fn source(node: &Node, name: &str) -> std::sync::Arc<crate::index::IndexService> {
        let now = crate::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 3_600_000;
        let svc = node
            .create_index(name, &json!({
                "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": 0,
                    "index.time_series.start_time": iso(hour - 3_600_000), "index.time_series.end_time": iso(now + 3_600_000)},
                "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
                    "cpu": {"type": "double", "time_series_metric": "gauge"}}}}))
            .unwrap();
        let docs: Vec<Value> = (0..1200).map(|i| json!({"@timestamp": iso(hour + (i / 600) * 60_000), "host": format!("h{}", i % 600), "cpu": i as f64})).collect();
        LocalDownsampleOps(node).index_docs(name, docs).unwrap();
        svc.refresh().unwrap();
        LocalDownsampleOps(node).block_writes(name).unwrap();
        svc
    }

    fn target_docs(node: &Node, target: &str) -> usize {
        node.get_index(target).unwrap().refresh().unwrap();
        let mut n = 0;
        LocalDownsampleOps(node).scan(target, &mut |_| n += 1).unwrap();
        n
    }

    /// The target is `started` until every document is copied and
    /// refreshed; a run that dies mid-copy leaves a `started` partial
    /// target, which a rerun deletes and rebuilds; a successful target is
    /// acknowledged again without being rebuilt; a target of another
    /// source is never touched; users cannot set the status.
    #[test]
    fn interrupted_downsample_is_rebuilt_never_marked_successful() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        source(&node, "src");
        let body = json!({"fixed_interval": "1h"});
        let crash = CrashAfterFirstBatch { inner: LocalDownsampleOps(&node), batches: Cell::new(0), before_success: false };
        assert!(downsample_with(&node, "src", "dst", &body, &crash).is_err());
        let partial = node.get_index("dst").expect("the dead run left its target");
        assert_eq!(partial.setting(crate::settings::DOWNSAMPLE_STATUS).as_deref(), Some("started"));
        assert_eq!(target_docs(&node, "dst"), 500, "partial: one batch of 600");
        let partial_uuid = partial.uuid.clone();
        drop(partial);
        // Users cannot mark it successful.
        let mut flat = FlatSettings::new();
        flat.insert(crate::settings::DOWNSAMPLE_STATUS.into(), json!("success"));
        assert!(node.get_index("dst").unwrap().put_settings(flat, false).is_err());
        // While another run holds the target, a rerun leaves it alone.
        {
            let _held = InFlight::claim(&node, "dst").unwrap();
            let e = downsample(&node, "src", "dst", &body).unwrap_err();
            assert_eq!(e.kind, "resource_already_exists_exception");
            assert_eq!(node.get_index("dst").unwrap().uuid, partial_uuid);
        }
        // Another node in the process with a target of the same name is
        // not this node's run.
        {
            let other = tempfile::tempdir().unwrap();
            let other = Node::open(other.path()).unwrap();
            let _theirs = InFlight::claim(&other, "dst").expect("another node's target of the same name");
            assert!(InFlight::claim(&other, "dst").is_none(), "one run per target on a node");
            let _ours = InFlight::claim(&node, "dst").expect("not blocked by the other node's run");
        }
        // The rerun rebuilds it completely.
        downsample(&node, "src", "dst", &body).unwrap();
        let done = node.get_index("dst").unwrap();
        assert_ne!(done.uuid, partial_uuid, "rebuilt, not resumed");
        assert_eq!(done.setting(crate::settings::DOWNSAMPLE_STATUS).as_deref(), Some("success"));
        assert_eq!(target_docs(&node, "dst"), 600);
        let done_uuid = done.uuid.clone();
        drop(done);
        // Done: acknowledged again, not rebuilt.
        assert_eq!(downsample(&node, "src", "dst", &body).unwrap(), json!({"acknowledged": true}));
        assert_eq!(node.get_index("dst").unwrap().uuid, done_uuid);
        // A target of another source: refused, untouched.
        source(&node, "other");
        let e = downsample(&node, "other", "dst", &body).unwrap_err();
        assert_eq!(e.kind, "resource_already_exists_exception");
        assert_eq!(node.get_index("dst").unwrap().uuid, done_uuid);
    }

    /// A run whose worker dies (`die_after_batches` target batches
    /// written, or after the write block, before the status update; a dead
    /// process cleans nothing up), or whose target another run replaces
    /// before batch `replace_before_batch`.
    #[derive(Default)]
    struct Faulty<'a> {
        inner: Option<LocalDownsampleOps<'a>>,
        die_after_batches: Option<usize>,
        die_before_success: bool,
        replace_before_batch: Option<usize>,
        batches: Cell<usize>,
        body: std::cell::RefCell<Option<Value>>,
        replacement_uuid: std::cell::RefCell<Option<String>>,
    }

    impl<'a> Faulty<'a> {
        fn new(node: &'a Node) -> Self {
            Faulty { inner: Some(LocalDownsampleOps(node)), ..Default::default() }
        }
        fn ops(&self) -> &LocalDownsampleOps<'a> {
            self.inner.as_ref().unwrap()
        }
        fn dead(&self) -> bool {
            self.die_before_success || self.die_after_batches.is_some()
        }
    }

    impl DownsampleOps for Faulty<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<()> {
            *self.body.borrow_mut() = Some(body.clone());
            self.ops().create_index(name, body)
        }
        fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
            self.ops().scan(index, f)
        }
        fn scan_series(&self, index: &str, dv: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
            self.ops().scan_series(index, dv, f)
        }
        fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()> {
            if self.die_after_batches.is_some_and(|n| self.batches.get() >= n) {
                return Err(EsError::internal("worker killed"));
            }
            self.batches.set(self.batches.get() + 1);
            if self.replace_before_batch == Some(self.batches.get()) {
                self.ops().delete_index(index)?;
                self.ops().create_index(index, self.body.borrow().as_ref().unwrap())?;
                *self.replacement_uuid.borrow_mut() = Some(self.ops().0.get_index(index).unwrap().uuid.clone());
            }
            self.ops().index_docs(index, docs)
        }
        fn refresh(&self, index: &str) -> Result<()> {
            self.ops().refresh(index)
        }
        fn block_writes(&self, index: &str) -> Result<()> {
            self.ops().block_writes(index)
        }
        fn delete_index(&self, index: &str) -> Result<()> {
            if self.dead() {
                return Ok(());
            }
            self.ops().delete_index(index)
        }
        fn mark_success(&self, index: &str) -> Result<()> {
            if self.die_before_success {
                return Err(EsError::internal("worker killed before the status update"));
            }
            self.ops().mark_success(index)
        }
    }

    fn status(node: &Node, index: &str) -> Option<String> {
        node.get_index(index).and_then(|t| t.setting(crate::settings::DOWNSAMPLE_STATUS))
    }

    /// Review finding 64: a worker killed right after creating the target,
    /// or midway through its writes, leaves a `started` target that a
    /// retry rebuilds; one killed after the write block (every document
    /// written and refreshed) is resumed. No incomplete target is ever
    /// reported `success`.
    #[test]
    fn killed_downsample_is_rebuilt_or_resumed_never_reported_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        source(&node, "src");
        let body = json!({"fixed_interval": "1h"});
        for (written, docs) in [(0usize, 0usize), (1, 500)] {
            let killed = Faulty { die_after_batches: Some(written), ..Faulty::new(&node) };
            assert!(downsample_with(&node, "src", "kdst", &body, &killed).is_err());
            assert_eq!(status(&node, "kdst").as_deref(), Some("started"), "after {written} batches");
            assert_eq!(target_docs(&node, "kdst"), docs);
            let dead_uuid = node.get_index("kdst").unwrap().uuid.clone();
            assert_eq!(downsample(&node, "src", "kdst", &body).unwrap(), json!({"acknowledged": true}));
            let done = node.get_index("kdst").unwrap();
            assert_ne!(done.uuid, dead_uuid, "rebuilt after {written} batches");
            assert_eq!(status(&node, "kdst").as_deref(), Some("success"));
            assert_eq!(target_docs(&node, "kdst"), 600);
            drop(done);
            node.delete_index("kdst").unwrap();
        }
        // Killed after the write block, before the status update.
        let killed = Faulty { die_before_success: true, ..Faulty::new(&node) };
        assert!(downsample_with(&node, "src", "kdst", &body, &killed).is_err());
        assert_eq!(status(&node, "kdst").as_deref(), Some("started"));
        let uuid = node.get_index("kdst").unwrap().uuid.clone();
        assert_eq!(downsample(&node, "src", "kdst", &body).unwrap(), json!({"acknowledged": true}));
        assert_eq!(node.get_index("kdst").unwrap().uuid, uuid, "resumed, not rebuilt");
        assert_eq!(status(&node, "kdst").as_deref(), Some("success"));
        assert_eq!(target_docs(&node, "kdst"), 600);
    }

    /// A run whose target another run deleted and created again midway
    /// fails without marking or deleting the other run's target.
    #[test]
    fn replaced_target_is_neither_marked_nor_deleted() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        source(&node, "src");
        let ops = Faulty { replace_before_batch: Some(2), ..Faulty::new(&node) };
        let e = downsample_with(&node, "src", "rdst", &json!({"fixed_interval": "1h"}), &ops).unwrap_err();
        assert_eq!(e.kind, "illegal_state_exception", "{e:?}");
        let replacement = ops.replacement_uuid.borrow().clone().expect("replaced");
        assert_eq!(node.get_index("rdst").expect("the other run's target is kept").uuid, replacement);
        assert_eq!(status(&node, "rdst").as_deref(), Some("started"));
        assert_eq!(node.get_index("rdst").unwrap().setting("index.blocks.write"), None);
    }

    /// 200 series × 60 one-minute documents (12,000, each with a 300-char
    /// label): several MB of source.
    fn big_source(node: &Node, name: &str) -> f64 {
        let now = crate::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 3_600_000;
        node.create_index(name, &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": 0,
                "index.time_series.start_time": iso(hour - 3_600_000), "index.time_series.end_time": iso(now + 3_600_000)},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
                "note": {"type": "keyword"}, "cpu": {"type": "double", "time_series_metric": "gauge"}}}}))
            .unwrap();
        let note = "n".repeat(300);
        let mut total = 0.0;
        for chunk in 0..6 {
            let docs: Vec<Value> = (chunk * 2000..(chunk + 1) * 2000)
                .map(|i| {
                    total += i as f64;
                    json!({"@timestamp": iso(hour + (i / 200) * 60_000), "host": format!("h{:03}", i % 200), "note": note, "cpu": i as f64})
                })
                .collect();
            LocalDownsampleOps(node).index_docs(name, docs).unwrap();
        }
        node.get_index(name).unwrap().refresh().unwrap();
        LocalDownsampleOps(node).block_writes(name).unwrap();
        total
    }

    fn small_request_breaker(limit: &str) -> &'static crate::memory::Breakers {
        let b = crate::memory::Breakers::leak_for_tests(1 << 30);
        let limit = limit.to_string();
        b.apply_settings(&|k| match k {
            "indices.breaker.request.limit" => Some(limit.clone()),
            "indices.breaker.total.limit" => Some("-1".into()),
            "indices.breaker.total.use_real_memory" => Some("false".into()),
            _ => None,
        });
        b
    }

    /// Review finding 65: a source several times the request breaker's
    /// limit downsamples within it: the run holds one bucket of one time
    /// series and one output batch, reserved on the breaker, never the
    /// source.
    #[test]
    fn downsample_streams_a_source_larger_than_the_breaker() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let total = big_source(&node, "big");
        let mut source_bytes = 0;
        LocalDownsampleOps(&node).scan("big", &mut |s| source_bytes += value_bytes(s)).unwrap();
        let limit = 2 << 20;
        assert!(source_bytes > 2 * limit, "{source_bytes}");
        let b = small_request_breaker("2mb");
        DOWNSAMPLE_PEAK.with(|p| p.set(0));
        let r = crate::memory::scoped(Some(crate::memory::RequestScope::new("downsample", b)), || {
            downsample(&node, "big", "big-10m", &json!({"fixed_interval": "10m"}))
        });
        assert_eq!(r.unwrap(), json!({"acknowledged": true}));
        let peak = DOWNSAMPLE_PEAK.with(|p| p.get());
        assert!(peak > 0 && peak <= limit / 2, "peak {peak} of {source_bytes} source bytes");
        let mut out = Vec::new();
        node.get_index("big-10m").unwrap().refresh().unwrap();
        LocalDownsampleOps(&node).scan("big-10m", &mut |s| out.push(s.clone())).unwrap();
        assert_eq!(out.len(), 200 * 6);
        assert_eq!(out.iter().map(|d| d["cpu"]["value_count"].as_u64().unwrap()).sum::<u64>(), 12_000);
        assert_eq!(out.iter().map(|d| d["cpu"]["sum"].as_f64().unwrap()).sum::<f64>(), total);
        // One series' bucket: h007, minutes 10..20 → i = 200*m + 7.
        let d = out.iter().find(|d| d["host"] == "h007" && d["@timestamp"].as_str().is_some_and(|t| t.contains(":10:00"))).expect("h007 second bucket");
        assert_eq!(d["cpu"]["min"], json!(2007.0), "{d}");
        assert_eq!(d["cpu"]["max"], json!(3807.0), "{d}");
        assert_eq!(d["note"].as_str().map(str::len), Some(300));
        assert_eq!(status(&node, "big-10m").as_deref(), Some("success"));
    }

    /// The run's own work under `scope`; the index operations (source
    /// search pages, target writes) on the process's breakers.
    struct Metered<'a> {
        inner: LocalDownsampleOps<'a>,
        scope: std::sync::Arc<crate::memory::RequestScope>,
    }

    impl DownsampleOps for Metered<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<()> {
            crate::memory::scoped(None, || self.inner.create_index(name, body))
        }
        fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
            crate::memory::scoped(None, || self.inner.scan(index, f))
        }
        fn scan_series(&self, index: &str, dv: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
            crate::memory::scoped(None, || self.inner.scan_series(index, dv, &mut |h| crate::memory::scoped(Some(self.scope.clone()), || f(h))))
        }
        fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()> {
            crate::memory::scoped(None, || self.inner.index_docs(index, docs))
        }
        fn refresh(&self, index: &str) -> Result<()> {
            crate::memory::scoped(None, || self.inner.refresh(index))
        }
        fn block_writes(&self, index: &str) -> Result<()> {
            crate::memory::scoped(None, || self.inner.block_writes(index))
        }
        fn delete_index(&self, index: &str) -> Result<()> {
            crate::memory::scoped(None, || self.inner.delete_index(index))
        }
        fn mark_success(&self, index: &str) -> Result<()> {
            crate::memory::scoped(None, || self.inner.mark_success(index))
        }
    }

    /// Past the breaker the run fails with ES's `circuit_breaking_exception`
    /// (its own reservation, or a source page's search) and leaves no
    /// target; with room it then succeeds.
    #[test]
    fn downsample_past_the_breaker_trips_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        big_source(&node, "big");
        let body = json!({"fixed_interval": "10m"});
        // The run's own memory (the output batch outgrows 128kb).
        let b = small_request_breaker("128kb");
        let ops = Metered { inner: LocalDownsampleOps(&node), scope: crate::memory::RequestScope::new("downsample", b) };
        let e = downsample_with(&node, "big", "big-trip", &body, &ops).unwrap_err();
        assert_eq!((e.status, e.kind.as_str()), (429, "circuit_breaking_exception"), "{e:?}");
        assert!(e.reason.starts_with("[request] Data too large, data for [downsample] would be ["), "{}", e.reason);
        assert!(node.get_index("big-trip").is_none(), "the failed target is removed");
        drop(ops);
        assert_eq!(b.request.reserved(), 0, "the run's reservation is released");
        // Everything under a breaker too small for a source page.
        let b = small_request_breaker("64kb");
        let e = crate::memory::scoped(Some(crate::memory::RequestScope::new("downsample", b)), || downsample(&node, "big", "big-trip", &body)).unwrap_err();
        assert_eq!(e.status, 429, "{e:?}");
        let root = e.caused_by.as_deref().unwrap_or(&e);
        assert_eq!(root.kind, "circuit_breaking_exception", "{e:?}");
        assert!(node.get_index("big-trip").is_none(), "the failed target is removed");
        assert!(node.get_index("big").is_some());
        assert_eq!(downsample(&node, "big", "big-trip", &body).unwrap(), json!({"acknowledged": true}));
        assert_eq!(target_docs(&node, "big-trip"), 1200);
    }

    /// Review finding 88: series told apart only by a dimension inside a
    /// `flattened` field (`time_series_dimensions`) stay separate, each
    /// with its own metrics and labels (synthetic source, the default).
    #[test]
    fn flattened_dimensions_keep_series_apart() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let now = crate::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 3_600_000;
        node.create_index("fsrc", &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["region"], "index.number_of_replicas": 0,
                "index.time_series.start_time": iso(hour - 3_600_000), "index.time_series.end_time": iso(now + 3_600_000)},
            "mappings": {"properties": {"@timestamp": {"type": "date"},
                "region": {"type": "keyword", "time_series_dimension": true},
                "labels": {"type": "flattened", "time_series_dimensions": ["host"]},
                "cpu": {"type": "double", "time_series_metric": "gauge"}}}}))
            .unwrap();
        let docs: Vec<Value> = (0..6)
            .map(|i| {
                let host = if i % 2 == 0 { "a" } else { "b" };
                json!({"@timestamp": iso(hour + (i / 2) * 60_000), "region": "eu", "labels": {"host": host, "rack": format!("r-{host}")}, "cpu": (i + 1) as f64})
            })
            .collect();
        LocalDownsampleOps(&node).index_docs("fsrc", docs).unwrap();
        node.get_index("fsrc").unwrap().refresh().unwrap();
        LocalDownsampleOps(&node).block_writes("fsrc").unwrap();
        downsample(&node, "fsrc", "fdst", &json!({"fixed_interval": "1h"})).unwrap();
        node.get_index("fdst").unwrap().refresh().unwrap();
        let mut out = Vec::new();
        LocalDownsampleOps(&node).scan("fdst", &mut |s| out.push(s.clone())).unwrap();
        out.sort_by_key(|d| d["labels"]["host"].as_str().unwrap_or_default().to_string());
        assert_eq!(out.len(), 2, "one series per flattened host: {out:?}");
        let metric = |d: &Value| ["min", "max", "sum", "value_count"].map(|k| d["cpu"][k].as_f64().unwrap_or(f64::NAN));
        assert_eq!((out[0]["labels"]["host"].as_str(), out[0]["labels"]["rack"].as_str()), (Some("a"), Some("r-a")), "{out:?}");
        assert_eq!(metric(&out[0]), [1.0, 5.0, 9.0, 3.0], "{out:?}");
        assert_eq!((out[1]["labels"]["host"].as_str(), out[1]["labels"]["rack"].as_str()), (Some("b"), Some("r-b")), "{out:?}");
        assert_eq!(metric(&out[1]), [2.0, 6.0, 12.0, 3.0], "{out:?}");
        assert!(out.iter().all(|d| d["region"] == "eu"), "{out:?}");
    }
    /// A write-blocked time series index with a stored source filtered by
    /// `source_filter` (`_source` mapping), holding 2 hosts × 3 documents
    /// in one hour; `extra`: more mapped properties.
    fn filtered_source(node: &Node, name: &str, source_filter: Value, extra: Value) {
        let now = crate::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 3_600_000;
        let mut props = json!({"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
            "cpu": {"type": "double", "time_series_metric": "gauge"}, "req": {"type": "long", "time_series_metric": "counter"}});
        for (k, v) in extra.as_object().into_iter().flatten() {
            props[k] = v.clone();
        }
        let mut mappings = json!({"properties": props});
        if !source_filter.is_null() {
            mappings["_source"] = source_filter;
        }
        node.create_index(name, &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": 0, "index.mapping.source.mode": "stored",
                "index.time_series.start_time": iso(hour - 3_600_000), "index.time_series.end_time": iso(now + 3_600_000)},
            "mappings": mappings}))
            .unwrap();
        let docs: Vec<Value> = (0..6)
            .map(|i| json!({"@timestamp": iso(hour + i * 60_000), "host": if i % 2 == 0 { "a" } else { "b" }, "cpu": (i + 1) as f64, "req": 10 * (i + 1)}))
            .collect();
        LocalDownsampleOps(node).index_docs(name, docs).unwrap();
        node.get_index(name).unwrap().refresh().unwrap();
        LocalDownsampleOps(node).block_writes(name).unwrap();
    }

    fn hits_of(node: &Node, index: &str, query: Value) -> usize {
        let svc = node.get_index(index).unwrap();
        let req = crate::search::parse_request(&json!({"query": query, "size": 100}), &HashMap::new()).unwrap();
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        v["hits"]["hits"].as_array().map_or(0, Vec::len)
    }

    fn downsampled(node: &Node, target: &str) -> Vec<Value> {
        node.get_index(target).unwrap().refresh().unwrap();
        let mut out = Vec::new();
        LocalDownsampleOps(node).scan(target, &mut |s| out.push(s.clone())).unwrap();
        out.sort_by_key(|d| d["host"].as_str().unwrap_or_default().to_string());
        out
    }

    /// Review finding 111: a stored source whose mapping-level `_source`
    /// filter drops `@timestamp`, a gauge, a counter or a dimension still
    /// downsamples completely (those inputs are read from doc values, as
    /// ES downsampling reads them): the output equals that of the same
    /// documents with an unfiltered source. The records stay findable by
    /// their indexed timestamp.
    #[test]
    fn source_filtered_inputs_are_read_from_doc_values() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        filtered_source(&node, "plain", Value::Null, json!({}));
        downsample(&node, "plain", "plain-1h", &json!({"fixed_interval": "1h"})).unwrap();
        let expected = downsampled(&node, "plain-1h");
        assert_eq!(expected.len(), 2, "{expected:?}");
        assert!(expected.iter().all(|d| d["@timestamp"].is_string() && d["cpu"]["value_count"] == json!(3) && d["req"].is_number()), "{expected:?}");
        for (name, excludes) in [("no-ts", json!(["@timestamp"])), ("no-cpu", json!(["cpu"])), ("no-req", json!(["req"])), ("no-host", json!(["host"])), ("all", json!(["@timestamp", "cpu", "req", "host"]))] {
            filtered_source(&node, name, json!({"excludes": excludes}), json!({}));
            assert_eq!(hits_of(&node, name, json!({"range": {"@timestamp": {"gte": "now-1d"}}})), 6, "{name}: the timestamp is indexed");
            assert_eq!(hits_of(&node, name, json!({"exists": {"field": "cpu"}})), 6, "{name}");
            let target = format!("{name}-1h");
            downsample(&node, name, &target, &json!({"fixed_interval": "1h"})).unwrap();
            let got = downsampled(&node, &target);
            assert_eq!(got, expected, "{name}: complete summaries");
        }
        // An `includes` filter keeping only the dimension.
        filtered_source(&node, "only-host", json!({"includes": ["host"]}), json!({}));
        downsample(&node, "only-host", "only-host-1h", &json!({"fixed_interval": "1h"})).unwrap();
        assert_eq!(downsampled(&node, "only-host-1h"), expected);
    }

    /// Review finding 111: a filtered-out metric whose doc values cannot
    /// stand in for its source (a histogram here) refuses the downsample
    /// before any target exists, rather than dropping it from the output.
    #[test]
    fn source_filtered_inputs_without_doc_value_form_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        filtered_source(&node, "hsrc", json!({"excludes": ["lat"]}), json!({"lat": {"type": "histogram", "time_series_metric": "histogram"}}));
        let e = downsample(&node, "hsrc", "hdst", &json!({"fixed_interval": "1h"})).unwrap_err();
        assert_eq!(e.status, 400, "{e:?}");
        assert!(e.reason.contains("removes field [lat]"), "{}", e.reason);
        assert!(node.get_index("hdst").is_none());
        // A gauge with `doc_values: false` filtered out is refused too.
        filtered_source(&node, "ndv", json!({"excludes": ["mem"]}), json!({"mem": {"type": "double", "time_series_metric": "gauge", "doc_values": false}}));
        assert!(downsample(&node, "ndv", "ndv-1h", &json!({"fixed_interval": "1h"})).is_err());
        assert!(node.get_index("ndv-1h").is_none());
    }

    /// A scan whose hits lack their timestamp (what the source-only read
    /// saw before finding 111's fix).
    struct NoTimestamps<'a>(LocalDownsampleOps<'a>);

    impl DownsampleOps for NoTimestamps<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<()> {
            self.0.create_index(name, body)
        }
        fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()> {
            self.0.scan(index, f)
        }
        fn scan_series(&self, index: &str, dv: &[Value], f: &mut dyn FnMut(&Value) -> Result<()>) -> Result<()> {
            self.0.scan_series(index, dv, &mut |h| {
                let mut h = h.clone();
                if let Some(o) = h["_source"].as_object_mut() {
                    o.remove("@timestamp");
                }
                f(&h)
            })
        }
        fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()> {
            self.0.index_docs(index, docs)
        }
        fn refresh(&self, index: &str) -> Result<()> {
            self.0.refresh(index)
        }
        fn block_writes(&self, index: &str) -> Result<()> {
            self.0.block_writes(index)
        }
        fn delete_index(&self, index: &str) -> Result<()> {
            self.0.delete_index(index)
        }
        fn mark_success(&self, index: &str) -> Result<()> {
            self.0.mark_success(index)
        }
    }

    /// Records skipped for an unreadable timestamp (a zero-bucket output)
    /// never make a successful target.
    #[test]
    fn unreadable_timestamps_fail_instead_of_an_empty_success() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        filtered_source(&node, "src", Value::Null, json!({}));
        let e = downsample_with(&node, "src", "dst", &json!({"fixed_interval": "1h"}), &NoTimestamps(LocalDownsampleOps(&node))).unwrap_err();
        assert!(e.reason.contains("has no readable [@timestamp]"), "{}", e.reason);
        assert!(node.get_index("dst").is_none(), "no target is left, successful or not");
    }
}
