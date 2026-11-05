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

/// One source document of a time series.
struct SrcDoc {
    ts: i64,
    count: u64,
    src: Value,
}

/// The index operations a downsample run performs. Locally they act on the
/// node's engine; a cluster performs them through its coordinator, so the
/// source is read from every shard and the target index is created in the
/// cluster metadata and filled through its primaries.
pub trait DownsampleOps {
    fn create_index(&self, name: &str, body: &Value) -> Result<()>;
    /// Visit every document `_source` of `index`.
    fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<()>;
    /// Index `docs` (sources, auto ids) into `index`.
    fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<()>;
    fn refresh(&self, index: &str) -> Result<()>;
    /// Sets `index.blocks.write`.
    fn block_writes(&self, index: &str) -> Result<()>;
    fn delete_index(&self, index: &str) -> Result<()>;
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
    if let Some(existing) = node.get_index(target) {
        // A finished downsample of this source into this target is acknowledged again.
        if existing.setting("index.downsample.source.uuid").as_deref() == Some(svc.uuid.as_str()) && existing.setting("index.downsample.status").as_deref() == Some("success") {
            return Ok(json!({"acknowledged": true}));
        }
        return Err(EsError::index_already_exists(target, &existing.uuid));
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
    // `index.downsample.status` is not dynamic: the target is created
    // finished and deleted again if the run fails.
    flat.insert("index.downsample.status".into(), json!("success"));
    let src_map = svc.meta().mappings.to_json();
    let mut fields = Vec::new();
    if let Some(p) = src_map.get("properties").and_then(Value::as_object) {
        classify(p, "", &mut fields)?;
    }
    let temporality_field = crate::settings::get_str(&settings, "index.time_series.temporality_field").map(str::to_string);
    ops.create_index(target, &json!({"settings": crate::settings::unflatten(&flat), "mappings": target_mappings(&src_map, &interval_s, aggregate)}))?;
    // A target whose shards cannot be allocated on this node (allocation
    // filter on a node attribute it lacks) fails like ES's shard task.
    let unallocatable = node.get_index(target).is_some_and(|dest| {
        dest.meta().settings.keys().any(|k| {
            ["index.routing.allocation.include.", "index.routing.allocation.require."].iter().any(|p| k.strip_prefix(p).map(|a| !a.starts_with('_')).unwrap_or(false))
        })
    });
    if unallocatable {
        let _ = ops.delete_index(target);
        return Err(EsError::new(500, "elasticsearch_exception", format!("downsample task [downsample-{target}-0-{interval_s}] failed")));
    }
    let result = (|| -> Result<()> {
        // Source documents per time series (dimension values).
        let parse_fmt = src_map["properties"][crate::datastream::TIMESTAMP_FIELD]["format"].as_str().map(str::to_string);
        let mut series: BTreeMap<String, (Vec<(String, Value)>, Vec<SrcDoc>)> = BTreeMap::new();
        ops.scan(source, &mut |src| {
            let Some(ts) = src_timestamp(src, parse_fmt.as_deref()) else { return };
            let dims: Vec<(String, Value)> = fields
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
                .collect();
            let key = serde_json::to_string(&dims).unwrap_or_default();
            let count = src.get("_doc_count").and_then(Value::as_u64).unwrap_or(1);
            series.entry(key).or_insert_with(|| (dims, Vec::new())).1.push(SrcDoc { ts, count, src: src.clone() });
        })?;
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
        let fmt_ts = |ms: i64| match ts_format.as_deref().and_then(|f| crate::date::DateFormatter::new(f).ok()) {
            Some(f) => f.format(ms, crate::date::Tz::UTC),
            None => crate::ilm::iso_date(ms),
        };
        // Field paths become nested objects, except under `subobjects: false`.
        let flat_keys = src_map.get("subobjects").and_then(Value::as_bool) == Some(false);
        let put = |doc: &mut Map<String, Value>, p: &str, v: Value| put_path(doc, p, v, flat_keys);
        let mut batch = Vec::new();
        let emit = |doc: Map<String, Value>, batch: &mut Vec<Value>| -> Result<()> {
            batch.push(Value::Object(doc));
            if batch.len() >= 500 {
                ops.index_docs(target, std::mem::take(batch))?;
            }
            Ok(())
        };
        for (_, (dims, mut docs)) in series {
            // `TimeSeriesIndexSearcher`: newest first within a time series.
            docs.sort_by(|a, b| b.ts.cmp(&a.ts));
            let delta = temporality_field.as_ref().and_then(|t| dims.iter().find(|(p, _)| p == t)).and_then(|(_, v)| v.as_str()).map(str::to_string);
            let counter_delta = delta.as_deref() == Some("delta");
            // Histograms are delta unless the series says cumulative.
            let hist_cumulative = delta.as_deref() == Some("cumulative");
            let mut cumulative: HashMap<String, Cumulative> = HashMap::new();
            let mut i = 0;
            while i < docs.len() {
                let start = docs[i].ts - docs[i].ts.rem_euclid(interval);
                let mut j = i;
                while j < docs.len() && docs[j].ts - docs[j].ts.rem_euclid(interval) == start {
                    j += 1;
                }
                let bucket = &docs[i..j];
                let count: u64 = bucket.iter().map(|d| d.count).sum();
                let mut doc = Map::new();
                doc.insert(crate::datastream::TIMESTAMP_FIELD.into(), json!(fmt_ts(start)));
                for (p, v) in &dims {
                    put(&mut doc, p, v.clone());
                }
                let mut resets: BTreeMap<i64, Vec<(String, ResetValue)>> = BTreeMap::new();
                for (p, kind) in &fields {
                    // Values per document, newest first.
                    let per_doc: Vec<(i64, Vec<&Value>)> = bucket.iter().map(|d| (d.ts, crate::highlight::source_values(&d.src, p))).filter(|(_, v)| !v.is_empty()).collect();
                    let value: Option<Value> = match *kind {
                        Kind::Dimension => None,
                        Kind::Gauge if aggregate => {
                            let (mut mn, mut mx, mut sum, mut cnt) = (f64::MAX, -f64::MAX, crate::fieldtypes::exphist::KahanSum::default(), 0u64);
                            for (_, vals) in &per_doc {
                                for x in sorted_nums(vals) {
                                    mn = mn.min(x);
                                    mx = mx.max(x);
                                    sum.add(x);
                                    cnt += 1;
                                }
                            }
                            (cnt > 0).then(|| json!({"min": mn, "max": mx, "sum": sum.value(), "value_count": cnt}))
                        }
                        Kind::Gauge | Kind::Counter if !aggregate => per_doc.iter().find_map(|(_, v)| sorted_nums(v).first().copied()).map(|x| json!(x)),
                        Kind::Counter => {
                            let firsts: Vec<(i64, f64)> = per_doc.iter().filter_map(|(t, v)| sorted_nums(v).first().map(|x| (*t, *x))).collect();
                            if firsts.is_empty() {
                                None
                            } else if counter_delta {
                                let mut s = crate::fieldtypes::exphist::KahanSum::default();
                                firsts.iter().for_each(|(_, x)| s.add(*x));
                                Some(json!(s.value()))
                            } else {
                                let c = cumulative.entry(p.clone()).or_insert_with(Cumulative::new);
                                for (t, x) in &firsts {
                                    c.collect(ResetValue::Counter(*x), *t);
                                }
                                let v = match &c.value {
                                    Some(ResetValue::Counter(x)) => Some(json!(x)),
                                    _ => None,
                                };
                                for (t, r) in c.bucket_resets() {
                                    resets.entry(t).or_default().push((p.clone(), r));
                                }
                                c.reset_bucket();
                                v
                            }
                        }
                        Kind::Aggregate { metric } => {
                            let metrics = agg_metrics.get(p).cloned().unwrap_or_default();
                            let objs: Vec<&Value> = per_doc.iter().flat_map(|(_, v)| v.iter().copied()).filter(|v| v.is_object()).collect();
                            if objs.is_empty() {
                                None
                            } else if metric && aggregate {
                                let mut m = Map::new();
                                for k in ["min", "max", "sum", "value_count"] {
                                    if !metrics.iter().any(|x| x == k) {
                                        continue;
                                    }
                                    let xs: Vec<f64> = objs.iter().filter_map(|o| o.get(k).and_then(num)).collect();
                                    if xs.is_empty() {
                                        continue;
                                    }
                                    let v = match k {
                                        "min" => json!(xs.iter().copied().fold(f64::MAX, f64::min)),
                                        "max" => json!(xs.iter().copied().fold(-f64::MAX, f64::max)),
                                        "sum" => {
                                            let mut s = crate::fieldtypes::exphist::KahanSum::default();
                                            xs.iter().for_each(|x| s.add(*x));
                                            json!(s.value())
                                        }
                                        _ => json!(xs.iter().map(|x| x.round() as i64).sum::<i64>()),
                                    };
                                    m.insert(k.into(), v);
                                }
                                Some(Value::Object(m))
                            } else {
                                // Last value, per metric.
                                let latest = objs[0];
                                let mut m = Map::new();
                                for k in ["min", "max", "sum", "value_count"] {
                                    if let Some(x) = latest.get(k).and_then(num) {
                                        m.insert(k.into(), if k == "value_count" { json!(x as i64) } else { json!(x) });
                                    }
                                }
                                Some(Value::Object(m))
                            }
                        }
                        Kind::Histogram { metric, tdigest } => {
                            let label = if tdigest { "centroids" } else { "values" };
                            let parsed: Vec<(Vec<f64>, Vec<i64>)> = per_doc
                                .iter()
                                .flat_map(|(_, v)| v.iter())
                                .filter_map(|h| crate::fieldtypes::index::parse_histogram(if tdigest { crate::mapping::EsType::TDigest } else { crate::mapping::EsType::Histogram }, h).ok())
                                .collect();
                            if parsed.is_empty() {
                                None
                            } else if metric && aggregate {
                                // A merging t-digest over few centroids keeps
                                // every distinct value.
                                let mut merged: BTreeMap<u64, (f64, i64)> = BTreeMap::new();
                                for (vs, cs) in &parsed {
                                    for (x, c) in vs.iter().zip(cs) {
                                        if *c == 0 {
                                            continue;
                                        }
                                        let key = if *x >= 0.0 { x.to_bits() ^ (1 << 63) } else { !x.to_bits() };
                                        merged.entry(key).or_insert((*x, 0)).1 += c;
                                    }
                                }
                                let values: Vec<Value> = merged.values().map(|(x, _)| json!(x)).collect();
                                let counts: Vec<Value> = merged.values().map(|(_, c)| json!(c)).collect();
                                Some(json!({"counts": counts, label: values}))
                            } else {
                                let (vs, cs) = &parsed[0];
                                Some(json!({"counts": cs, label: vs}))
                            }
                        }
                        Kind::ExpHist { metric } => {
                            use crate::fieldtypes::exphist;
                            let hs: Vec<(i64, Value)> = per_doc.iter().flat_map(|(t, v)| v.iter().map(move |h| (*t, (*h).clone()))).filter(|(_, h)| exphist::parse("", h, false).is_ok()).collect();
                            if hs.is_empty() {
                                None
                            } else if !(metric && aggregate) {
                                exphist::parse("", &hs[0].1, false).ok().map(|h| exphist::render(&h))
                            } else if hist_cumulative {
                                let c = cumulative.entry(p.clone()).or_insert_with(Cumulative::new);
                                for (t, h) in &hs {
                                    let r = exphist::parse("", h, false).map(|h| exphist::render(&h)).unwrap_or(Value::Null);
                                    c.collect(ResetValue::Hist(r), *t);
                                }
                                let v = match &c.value {
                                    Some(ResetValue::Hist(h)) => Some(h.clone()),
                                    _ => None,
                                };
                                for (t, r) in c.bucket_resets() {
                                    resets.entry(t).or_default().push((p.clone(), r));
                                }
                                c.reset_bucket();
                                v
                            } else {
                                let mut m = exphist::Merger::new();
                                for (_, h) in &hs {
                                    if let Ok(h) = exphist::parse("", h, false) {
                                        m.add(&h);
                                    }
                                }
                                Some(exphist::render(&m.get()))
                            }
                        }
                        Kind::Label(max) => {
                            // The newest document with doc values for the field.
                            per_doc.iter().find_map(|(_, vals)| {
                                let vals: Vec<&Value> = match max {
                                    // Values over `ignore_above` are not in doc values.
                                    Some(max) => vals.iter().copied().filter(|v| v.as_str().map(|s| s.chars().count() <= max).unwrap_or(true)).collect(),
                                    None => vals.clone(),
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
                            })
                        }
                        _ => None,
                    };
                    if let Some(v) = value {
                        put(&mut doc, p, v);
                    }
                }
                // Reset documents are counted out of the downsampled one.
                let main_count = count.saturating_sub(resets.len() as u64).max(1);
                doc.insert("_doc_count".into(), json!(main_count));
                emit(doc, &mut batch)?;
                for (t, values) in resets {
                    let mut rd = Map::new();
                    rd.insert(crate::datastream::TIMESTAMP_FIELD.into(), json!(fmt_ts(t)));
                    for (p, v) in &dims {
                        put(&mut rd, p, v.clone());
                    }
                    for (p, v) in values {
                        let v = match v {
                            ResetValue::Counter(x) => json!(x),
                            ResetValue::Hist(h) => h,
                        };
                        put(&mut rd, &p, v);
                    }
                    emit(rd, &mut batch)?;
                }
                i = j;
            }
        }
        if !batch.is_empty() {
            ops.index_docs(target, batch)?;
        }
        ops.refresh(target)?;
        ops.block_writes(target)
    })();
    if let Err(e) = result {
        let _ = ops.delete_index(target);
        return Err(e);
    }
    Ok(json!({"acknowledged": true}))
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
    let svc = node.get_index(index).ok_or_else(|| no_such_index(index))?;
    let targets = vec![crate::search::Target::new(svc)];
    let req = crate::search::parse_request(&json!({"size": 1000, "sort": ["_doc"], "track_total_hits": false}), &HashMap::new())?;
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
            if let Some(s) = h.get("_source") {
                f(s);
            }
        }
        cursors = Some(out.cursors);
    }
    Ok(())
}
