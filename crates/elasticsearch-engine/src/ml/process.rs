//! The anomaly detection job process (ES's native `autodetect`, in
//! process).
//!
//! An opened job owns a [`Proc`]: it ingests records (`_data`, or the
//! datafeed), buckets them by `bucket_span`, keeps one statistical model per
//! detector series (partition / by / over values) and writes ES-shaped
//! result documents to the job's results index when a bucket completes:
//! `bucket`, `record`, `influencer`, `bucket_influencer`, `model_plot`,
//! `category_definition`, `model_size_stats`, `categorizer_stats`,
//! `timing_stats` and the data counts. `_flush` finalizes / skips / calc
//! interim results, `_close` persists state as a model snapshot (`.ml-state`)
//! and `_forecast` projects the series models forward.
//!
//! The scoring is a Gaussian (count: Poisson-floored) tail probability per
//! series with ES's probability-to-score shape.

use super::resultsidx::{self as ridx, read_alias, values_to_id};
use super::store::service;
use crate::error::{EsError, Result};
use crate::node::Node;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

/// `min_version` of snapshots written here.
pub const SNAPSHOT_MIN_VERSION: &str = "8.3.0";
/// Marker key of the model state documents this node writes.
const STATE_KEY: &str = "rustic_ml_state";
/// Results are not written for more than this many consecutive empty
/// buckets (a datafeed gap of years would otherwise flood the index).
const MAX_EMPTY_BUCKETS: i64 = 10_000;
/// Buckets a series must have seen before it can be anomalous.
const MIN_BUCKETS: f64 = 12.0;
/// Records are written below this probability (ES's models start reporting
/// records around p = 2e-3).
const RECORD_P: f64 = 0.002;

// ---------------------------------------------------------------------------
// Data counts / timing / model size
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DataCounts {
    pub processed_record_count: u64,
    pub processed_field_count: u64,
    pub input_bytes: u64,
    pub input_field_count: u64,
    pub invalid_date_count: u64,
    pub missing_field_count: u64,
    pub out_of_order_timestamp_count: u64,
    pub empty_bucket_count: u64,
    pub sparse_bucket_count: u64,
    pub bucket_count: u64,
    pub earliest_record_timestamp: Option<i64>,
    pub latest_record_timestamp: Option<i64>,
    pub last_data_time: Option<i64>,
    pub latest_empty_bucket_timestamp: Option<i64>,
    pub latest_sparse_bucket_timestamp: Option<i64>,
    pub latest_bucket_timestamp: Option<i64>,
    pub input_record_count: u64,
    pub log_time: Option<i64>,
}

impl DataCounts {
    /// `DataCounts.toXContent` (`log_time` only when `with_log_time`: the
    /// `_data` response leaves it out).
    pub fn to_json(&self, job_id: &str, with_log_time: bool) -> Value {
        let mut o = Map::new();
        o.insert("job_id".into(), json!(job_id));
        o.insert("processed_record_count".into(), json!(self.processed_record_count));
        o.insert("processed_field_count".into(), json!(self.processed_field_count));
        o.insert("input_bytes".into(), json!(self.input_bytes));
        o.insert("input_field_count".into(), json!(self.input_field_count));
        o.insert("invalid_date_count".into(), json!(self.invalid_date_count));
        o.insert("missing_field_count".into(), json!(self.missing_field_count));
        o.insert("out_of_order_timestamp_count".into(), json!(self.out_of_order_timestamp_count));
        o.insert("empty_bucket_count".into(), json!(self.empty_bucket_count));
        o.insert("sparse_bucket_count".into(), json!(self.sparse_bucket_count));
        o.insert("bucket_count".into(), json!(self.bucket_count));
        for (k, v) in [
            ("earliest_record_timestamp", self.earliest_record_timestamp),
            ("latest_record_timestamp", self.latest_record_timestamp),
            ("last_data_time", self.last_data_time),
            ("latest_empty_bucket_timestamp", self.latest_empty_bucket_timestamp),
            ("latest_sparse_bucket_timestamp", self.latest_sparse_bucket_timestamp),
        ] {
            if let Some(t) = v {
                o.insert(k.into(), json!(t));
            }
        }
        o.insert("input_record_count".into(), json!(self.input_record_count));
        if with_log_time {
            if let Some(t) = self.log_time {
                o.insert("log_time".into(), json!(t));
            }
        }
        Value::Object(o)
    }

    /// Read a stored `data_counts` document.
    pub fn from_json(v: &Value) -> DataCounts {
        let u = |k: &str| v[k].as_u64().or_else(|| v[k].as_f64().map(|f| f as u64)).unwrap_or(0);
        let t = |k: &str| v.get(k).and_then(ridx::ts_millis);
        DataCounts {
            processed_record_count: u("processed_record_count"),
            processed_field_count: u("processed_field_count"),
            input_bytes: u("input_bytes"),
            input_field_count: u("input_field_count"),
            invalid_date_count: u("invalid_date_count"),
            missing_field_count: u("missing_field_count"),
            out_of_order_timestamp_count: u("out_of_order_timestamp_count"),
            empty_bucket_count: u("empty_bucket_count"),
            sparse_bucket_count: u("sparse_bucket_count"),
            bucket_count: u("bucket_count"),
            earliest_record_timestamp: t("earliest_record_timestamp"),
            latest_record_timestamp: t("latest_record_timestamp"),
            last_data_time: t("last_data_time"),
            latest_empty_bucket_timestamp: t("latest_empty_bucket_timestamp"),
            latest_sparse_bucket_timestamp: t("latest_sparse_bucket_timestamp"),
            latest_bucket_timestamp: t("latest_bucket_timestamp"),
            input_record_count: u("input_record_count"),
            log_time: t("log_time"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimingStats {
    pub bucket_count: u64,
    pub total_ms: f64,
    pub min_ms: Option<f64>,
    pub max_ms: Option<f64>,
    pub avg_ms: Option<f64>,
    pub exp_avg_ms: Option<f64>,
    pub exp_avg_per_hour_ms: f64,
}

impl TimingStats {
    fn add(&mut self, ms: f64) {
        self.bucket_count += 1;
        self.total_ms += ms;
        self.min_ms = Some(self.min_ms.map(|m| m.min(ms)).unwrap_or(ms));
        self.max_ms = Some(self.max_ms.map(|m| m.max(ms)).unwrap_or(ms));
        self.avg_ms = Some(self.total_ms / self.bucket_count as f64);
        self.exp_avg_ms = Some(match self.exp_avg_ms {
            Some(e) => 0.99 * e + 0.01 * ms,
            None => ms,
        });
    }

    /// `TimingStats.toXContent` as the stats API renders it.
    pub fn to_json(&self, job_id: &str) -> Value {
        let mut o = Map::new();
        o.insert("job_id".into(), json!(job_id));
        o.insert("bucket_count".into(), json!(self.bucket_count));
        o.insert("total_bucket_processing_time_ms".into(), json!(self.total_ms));
        for (k, v) in [("minimum_bucket_processing_time_ms", self.min_ms), ("maximum_bucket_processing_time_ms", self.max_ms), ("average_bucket_processing_time_ms", self.avg_ms), ("exponential_average_bucket_processing_time_ms", self.exp_avg_ms)] {
            if let Some(x) = v {
                o.insert(k.into(), json!(x));
            }
        }
        o.insert("exponential_average_bucket_processing_time_per_hour_ms".into(), json!(self.exp_avg_per_hour_ms));
        Value::Object(o)
    }

    /// The stored `<job>_timing_stats` document.
    fn to_doc(&self, job_id: &str, latest: Option<i64>) -> Value {
        let mut o = Map::new();
        o.insert("result_type".into(), json!("timing_stats"));
        o.insert("job_id".into(), json!(job_id));
        o.insert("bucket_count".into(), json!(self.bucket_count));
        o.insert("total_bucket_processing_time_ms".into(), json!(self.total_ms));
        for (k, v) in [("minimum_bucket_processing_time_ms", self.min_ms), ("maximum_bucket_processing_time_ms", self.max_ms), ("average_bucket_processing_time_ms", self.avg_ms), ("exponential_average_bucket_processing_time_ms", self.exp_avg_ms)] {
            if let Some(x) = v {
                o.insert(k.into(), json!(x));
            }
        }
        o.insert("exponential_average_calculation_context".into(), json!({"incremental_metric_value_ms": 0.0, "latest_timestamp": latest, "previous_exponential_average_ms": self.exp_avg_per_hour_ms}));
        Value::Object(o)
    }

    /// Read a stored timing stats document.
    pub fn from_json(v: &Value) -> TimingStats {
        TimingStats {
            bucket_count: v["bucket_count"].as_u64().unwrap_or(0),
            total_ms: v["total_bucket_processing_time_ms"].as_f64().unwrap_or(0.0),
            min_ms: v["minimum_bucket_processing_time_ms"].as_f64(),
            max_ms: v["maximum_bucket_processing_time_ms"].as_f64(),
            avg_ms: v["average_bucket_processing_time_ms"].as_f64(),
            exp_avg_ms: v["exponential_average_bucket_processing_time_ms"].as_f64(),
            exp_avg_per_hour_ms: v["exponential_average_calculation_context"]["previous_exponential_average_ms"].as_f64().unwrap_or(0.0),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelSize {
    pub model_bytes: u64,
    pub peak_model_bytes: u64,
    pub by_count: u64,
    pub over_count: u64,
    pub partition_count: u64,
    pub timestamp: Option<i64>,
    pub categorized_doc_count: u64,
    pub total_category_count: u64,
    pub frequent_category_count: u64,
    pub rare_category_count: u64,
    pub dead_category_count: u64,
}

impl ModelSize {
    /// `ModelSizeStats.toXContent`.
    pub fn to_json(&self, job_id: &str, mml_bytes: u64, log_time: i64) -> Value {
        let mut o = Map::new();
        o.insert("job_id".into(), json!(job_id));
        o.insert("result_type".into(), json!("model_size_stats"));
        o.insert("model_bytes".into(), json!(self.model_bytes));
        o.insert("peak_model_bytes".into(), json!(self.peak_model_bytes));
        o.insert("model_bytes_exceeded".into(), json!(0));
        o.insert("model_bytes_memory_limit".into(), json!(mml_bytes));
        o.insert("total_by_field_count".into(), json!(self.by_count));
        o.insert("total_over_field_count".into(), json!(self.over_count));
        o.insert("total_partition_field_count".into(), json!(self.partition_count));
        o.insert("bucket_allocation_failures_count".into(), json!(0));
        o.insert("memory_status".into(), json!("ok"));
        o.insert("assignment_memory_basis".into(), json!("model_memory_limit"));
        o.insert("output_memory_allocator_bytes".into(), json!(0));
        o.insert("categorized_doc_count".into(), json!(self.categorized_doc_count));
        o.insert("total_category_count".into(), json!(self.total_category_count));
        o.insert("frequent_category_count".into(), json!(self.frequent_category_count));
        o.insert("rare_category_count".into(), json!(self.rare_category_count));
        o.insert("dead_category_count".into(), json!(self.dead_category_count));
        o.insert("failed_category_count".into(), json!(0));
        o.insert("categorization_status".into(), json!("ok"));
        o.insert("log_time".into(), json!(log_time));
        if let Some(t) = self.timestamp {
            o.insert("timestamp".into(), json!(t));
        }
        Value::Object(o)
    }
}

// ---------------------------------------------------------------------------
// Series models
// ---------------------------------------------------------------------------

/// Running statistics of one detector series (Welford).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Series {
    n: f64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    /// Last value fed (model plot `actual` of empty buckets).
    last: f64,
}

impl Series {
    fn update(&mut self, x: f64) {
        if self.n == 0.0 {
            self.min = x;
            self.max = x;
        } else {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
        self.last = x;
    }

    fn sigma(&self, count_like: bool) -> f64 {
        let var = if self.n > 1.0 { self.m2 / (self.n - 1.0) } else { 0.0 };
        // A young model is uncertain about its variance: widen it.
        let mut s = var.max(0.0).sqrt() * (1.0 + 20.0 / self.n.max(1.0)).sqrt();
        // Floors: counts are at least Poisson-noisy; metrics keep 5% of the
        // mean so a constant series is not infinitely significant.
        let floor = if count_like { self.mean.abs().max(1.0).sqrt() } else { (self.mean.abs() * 0.05).max(1e-9) };
        if s < floor {
            s = floor;
        }
        s
    }

    /// Tail probability of `x` (`side`: -1 low, 1 high, 0 both).
    fn probability(&self, x: f64, side: i32, count_like: bool) -> f64 {
        if self.n < MIN_BUCKETS {
            return 1.0;
        }
        let z = (x - self.mean) / self.sigma(count_like);
        match side {
            -1 => {
                if z >= 0.0 {
                    1.0
                } else {
                    (0.5 * erfc(-z / std::f64::consts::SQRT_2)).max(1e-300)
                }
            }
            1 => {
                if z <= 0.0 {
                    1.0
                } else {
                    (0.5 * erfc(z / std::f64::consts::SQRT_2)).max(1e-300)
                }
            }
            _ => erfc(z.abs() / std::f64::consts::SQRT_2).max(1e-300),
        }
    }

    fn bounds(&self, count_like: bool) -> (f64, f64, f64) {
        if self.n == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let s = self.sigma(count_like);
        let lo = self.mean - 1.96 * s;
        (if count_like { lo.max(0.0) } else { lo }, self.mean, self.mean + 1.96 * s)
    }
}

/// Complementary error function (Numerical Recipes `erfcc`, relative error
/// below 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// Probability to anomaly score (0..100): the saturating shape fitted to
/// ES's record scores (p = 1e-6 -> ~25, 1e-9 -> ~45, 1e-20 -> ~80).
fn score_of(p: f64) -> f64 {
    if p >= RECORD_P {
        return 0.0;
    }
    let x = -p.log10() - (-RECORD_P.log10());
    (100.0 * (1.0 - (-0.093 * x).exp())).clamp(0.0, 100.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Detector {
    index: usize,
    function: String,
    field: Option<String>,
    by: Option<String>,
    over: Option<String>,
    partition: Option<String>,
    use_null: bool,
    exclude_frequent: Option<String>,
    rules: Vec<Value>,
}

impl Detector {
    fn count_like(&self) -> bool {
        matches!(self.function.as_str(), "count" | "low_count" | "high_count" | "non_zero_count" | "low_non_zero_count" | "high_non_zero_count" | "distinct_count" | "low_distinct_count" | "high_distinct_count" | "rare" | "freq_rare")
    }
    fn side(&self) -> i32 {
        if self.function.starts_with("low_") {
            -1
        } else if self.function.starts_with("high_") {
            1
        } else {
            0
        }
    }
    fn base_function(&self) -> &str {
        self.function.trim_start_matches("low_").trim_start_matches("high_")
    }
    /// `function` / `function_description` of a record.
    fn description(&self) -> &'static str {
        match self.base_function() {
            "count" => "count",
            "non_zero_count" => "non_zero_count",
            "distinct_count" => "distinct_count",
            "rare" => "rare",
            "freq_rare" => "freq_rare",
            "info_content" => "info_content",
            "metric" | "mean" | "avg" => "mean",
            "median" => "median",
            "min" => "min",
            "max" => "max",
            "sum" => "sum",
            "non_null_sum" => "non_null_sum",
            "varp" => "varp",
            "time_of_day" => "time_of_day",
            "time_of_week" => "time_of_week",
            "lat_long" => "lat_long",
            _ => "metric",
        }
    }
    /// The model plot / forecast feature name.
    fn feature(&self) -> &'static str {
        match self.base_function() {
            "count" => "'count per bucket by person'",
            "non_zero_count" => "'non-zero count per bucket by person'",
            "distinct_count" => "'distinct count by person'",
            "metric" | "mean" | "avg" => "'arithmetic mean value by person'",
            "median" => "'median value by person'",
            "min" => "'minimum value by person'",
            "max" => "'maximum value by person'",
            "sum" => "'bucket sum by person'",
            "non_null_sum" => "'bucket non-null sum by person'",
            "varp" => "'variance by person'",
            "info_content" => "'information content by person'",
            "time_of_day" => "'time-of-day by person'",
            "time_of_week" => "'time-of-week by person'",
            "rare" => "'rare in population'",
            "freq_rare" => "'frequency rare in population'",
            _ => "'arithmetic mean value by person'",
        }
    }
    fn forecastable(&self) -> bool {
        !matches!(self.base_function(), "rare" | "freq_rare" | "lat_long" | "time_of_day" | "time_of_week" | "info_content" | "distinct_count")
    }
}

/// A category of a categorization job.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Category {
    id: i64,
    /// Tokens common to every matched message, in order.
    terms: Vec<String>,
    /// Tokens of the first matched message (the matching reference).
    tokens: Vec<String>,
    examples: Vec<String>,
    max_len: usize,
    num_matches: u64,
    partition: Option<String>,
    dirty: bool,
}

/// One input record: the record time and its (flattened) fields.
#[derive(Clone, Debug)]
struct Rec {
    time: i64,
    fields: Map<String, Value>,
}

/// A series of a detector in one bucket.
struct SeriesValue {
    partition: Option<String>,
    by: Option<String>,
    over: Option<String>,
    value: f64,
    /// Influencer field -> values of the contributing records.
    influencers: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Proc {
    pub job_id: String,
    bucket_span_ms: i64,
    time_field: String,
    time_format: String,
    summary_count_field: Option<String>,
    cat_field: Option<String>,
    cat_examples_limit: usize,
    per_partition_cat: bool,
    detectors: Vec<Detector>,
    influencers: Vec<String>,
    model_plot: bool,
    model_plot_terms: Option<Vec<String>>,
    pub counts: DataCounts,
    /// Start of the bucket currently accumulating records.
    current_bucket: Option<i64>,
    #[serde(skip)]
    cur: Vec<Rec>,
    /// `detector|partition|by|over` -> model.
    models: BTreeMap<String, Series>,
    /// `detector|partition` -> by value -> buckets seen (rare / freq_rare).
    frequency: BTreeMap<String, BTreeMap<String, u64>>,
    /// `detector|partition` -> buckets seen.
    frequency_total: BTreeMap<String, u64>,
    pub timing: TimingStats,
    pub size: ModelSize,
    categories: Vec<Category>,
    /// Timestamp of the latest result bucket.
    pub latest_result_ts: Option<i64>,
    /// Interim results are written for the current bucket.
    has_interim: bool,
    /// `log_time` of the latest model size stats document.
    size_log_time: Option<i64>,
    #[serde(skip)]
    time_fmt: Option<Arc<crate::date::DateFormatter>>,
}

type Docs = Vec<(String, Value)>;

fn floor_to(t: i64, span: i64) -> i64 {
    t.div_euclid(span) * span
}

fn str_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Flatten a document into dotted leaf fields (arrays keep their first
/// scalar; `_` prefixed metadata is skipped).
fn flatten(v: &Value, prefix: &str, out: &mut Map<String, Value>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(x, &p, out);
            }
        }
        Value::Array(a) => {
            if let Some(first) = a.iter().find(|x| !x.is_null() && !x.is_object() && !x.is_array()) {
                out.insert(prefix.to_string(), first.clone());
            } else if let Some(o) = a.first() {
                flatten(o, prefix, out);
            }
        }
        Value::Null => {}
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

impl Proc {
    /// A fresh process for a job config.
    pub fn new(job: &Value) -> Proc {
        let ac = &job["analysis_config"];
        let span = super::job::parse_time(ac["bucket_span"].as_str().unwrap_or("5m"), "bucket_span").map(|t| t.millis()).unwrap_or(300_000).max(1000);
        let detectors = ac["detectors"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, d)| Detector {
                index: d["detector_index"].as_u64().map(|x| x as usize).unwrap_or(i),
                function: d["function"].as_str().unwrap_or("metric").to_string(),
                field: d["field_name"].as_str().map(str::to_string),
                by: d["by_field_name"].as_str().map(str::to_string),
                over: d["over_field_name"].as_str().map(str::to_string),
                partition: d["partition_field_name"].as_str().map(str::to_string),
                use_null: d["use_null"].as_bool().unwrap_or(false),
                exclude_frequent: d["exclude_frequent"].as_str().map(str::to_string),
                rules: d["custom_rules"].as_array().cloned().unwrap_or_default(),
            })
            .collect();
        let mut p = Proc {
            job_id: job["job_id"].as_str().unwrap_or("").to_string(),
            bucket_span_ms: span,
            time_field: job["data_description"]["time_field"].as_str().unwrap_or("time").to_string(),
            time_format: job["data_description"]["time_format"].as_str().unwrap_or("epoch_ms").to_string(),
            summary_count_field: ac["summary_count_field_name"].as_str().map(str::to_string),
            cat_field: ac["categorization_field_name"].as_str().map(str::to_string),
            cat_examples_limit: job["analysis_limits"]["categorization_examples_limit"].as_u64().unwrap_or(4) as usize,
            per_partition_cat: ac["per_partition_categorization"]["enabled"].as_bool().unwrap_or(false),
            detectors,
            influencers: ac["influencers"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            model_plot: job["model_plot_config"]["enabled"].as_bool().unwrap_or(false),
            model_plot_terms: job["model_plot_config"]["terms"].as_str().map(|t| t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()),
            counts: DataCounts::default(),
            current_bucket: None,
            cur: Vec::new(),
            models: BTreeMap::new(),
            frequency: BTreeMap::new(),
            frequency_total: BTreeMap::new(),
            timing: TimingStats::default(),
            size: ModelSize::default(),
            categories: Vec::new(),
            latest_result_ts: None,
            has_interim: false,
            size_log_time: None,
            time_fmt: None,
        };
        p.init_fmt();
        p
    }

    fn init_fmt(&mut self) {
        if !matches!(self.time_format.as_str(), "epoch" | "epoch_ms") {
            self.time_fmt = crate::date::DateFormatter::new(&self.time_format).ok().map(Arc::new);
        }
    }

    /// Re-read the parts of the config an update may change (rules,
    /// model plot, categorization filters).
    fn refresh_config(&mut self, job: &Value) {
        let ac = &job["analysis_config"];
        for (i, d) in ac["detectors"].as_array().into_iter().flatten().enumerate() {
            if let Some(det) = self.detectors.get_mut(i) {
                det.rules = d["custom_rules"].as_array().cloned().unwrap_or_default();
            }
        }
        self.model_plot = job["model_plot_config"]["enabled"].as_bool().unwrap_or(false);
        self.model_plot_terms = job["model_plot_config"]["terms"].as_str().map(|t| t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect());
    }

    fn analysis_fields(&self) -> BTreeSet<String> {
        let mut f = BTreeSet::new();
        for d in &self.detectors {
            for x in [&d.field, &d.by, &d.over, &d.partition].into_iter().flatten() {
                if x != "mlcategory" {
                    f.insert(x.clone());
                }
            }
        }
        for x in [&self.summary_count_field, &self.cat_field].into_iter().flatten() {
            f.insert(x.clone());
        }
        for i in &self.influencers {
            f.insert(i.clone());
        }
        f
    }

    fn parse_time(&self, v: Option<&Value>, epoch_ms: bool) -> Option<i64> {
        let v = v?;
        if epoch_ms {
            return match v {
                Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
                Value::String(s) => s.trim().parse::<f64>().ok().map(|f| f as i64).or_else(|| crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").ok().and_then(|f| f.parse(s).ok())),
                _ => None,
            };
        }
        match self.time_format.as_str() {
            "epoch" => num_of(v).map(|f| (f * 1000.0) as i64),
            "epoch_ms" => num_of(v).map(|f| f as i64),
            _ => {
                let s = str_of(v)?;
                self.time_fmt.as_ref().and_then(|f| f.parse(&s).ok())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Ingestion
    // -----------------------------------------------------------------------

    /// Ingest documents (`epoch_ms`: the time field is already epoch
    /// millis, as a datafeed delivers it). Returns the result documents to
    /// write.
    fn ingest(&mut self, docs: Vec<Value>, bytes: usize, epoch_ms: bool, out: &mut Docs) {
        let fields = self.analysis_fields();
        self.counts.input_bytes += bytes as u64;
        let now = crate::date::now_millis();
        for doc in docs {
            let mut flat = Map::new();
            flatten(&doc, "", &mut flat);
            self.counts.input_record_count += 1;
            let n_fields = flat.len() as u64;
            self.counts.input_field_count += n_fields.saturating_sub(if flat.contains_key(&self.time_field) { 1 } else { 0 });
            let Some(t) = self.parse_time(flat.get(&self.time_field), epoch_ms) else {
                self.counts.invalid_date_count += 1;
                continue;
            };
            let mut present = 0u64;
            let mut rec = Map::new();
            for f in &fields {
                match flat.get(f) {
                    Some(v) if !v.is_null() && v.as_str().map(|s| !s.is_empty()).unwrap_or(true) => {
                        present += 1;
                        rec.insert(f.clone(), v.clone());
                    }
                    _ => self.counts.missing_field_count += 1,
                }
            }
            self.counts.processed_field_count += present;
            let b = floor_to(t, self.bucket_span_ms);
            let cb = self.current_bucket;
            match cb {
                None => self.current_bucket = Some(b),
                Some(cur) if b < cur => {
                    self.counts.out_of_order_timestamp_count += 1;
                    continue;
                }
                Some(cur) if b > cur => self.advance_to(b, true, out),
                _ => {}
            }
            self.counts.processed_record_count += 1;
            self.counts.earliest_record_timestamp = Some(self.counts.earliest_record_timestamp.map(|e| e.min(t)).unwrap_or(t));
            self.counts.latest_record_timestamp = Some(self.counts.latest_record_timestamp.map(|e| e.max(t)).unwrap_or(t));
            self.counts.last_data_time = Some(now);
            if let Some(cf) = &self.cat_field {
                let partition = if self.per_partition_cat { self.detectors.iter().find_map(|d| d.partition.as_ref()).and_then(|p| rec.get(p)).and_then(str_of) } else { None };
                if let Some(msg) = rec.get(cf).and_then(str_of) {
                    let cid = self.categorize(&msg, partition);
                    rec.insert("mlcategory".into(), json!(cid.to_string()));
                }
            }
            self.cur.push(Rec { time: t, fields: rec });
        }
        self.counts.log_time = Some(now);
    }

    /// Finalize the current bucket and the empty ones up to `target`
    /// (`from_data`: the advance comes from records, which updates the
    /// data counts; a flush does not).
    fn advance_to(&mut self, target: i64, from_data: bool, out: &mut Docs) {
        let Some(mut cur) = self.current_bucket else {
            self.current_bucket = Some(target);
            return;
        };
        if self.has_interim {
            self.has_interim = false;
            // Interim results of the current bucket are superseded.
            out.push(("__delete_interim__".into(), Value::Null));
        }
        let mut empties = 0i64;
        while cur < target {
            let recs = std::mem::take(&mut self.cur);
            if recs.is_empty() {
                empties += 1;
            }
            if !recs.is_empty() || empties <= MAX_EMPTY_BUCKETS {
                self.finalize(cur, &recs, false, from_data, out);
            }
            cur += self.bucket_span_ms;
        }
        self.current_bucket = Some(target);
    }

    fn series_key(d: &Detector, partition: &Option<String>, by: &Option<String>) -> String {
        format!("{}|{}|{}", d.index, partition.as_deref().unwrap_or(""), by.as_deref().unwrap_or(""))
    }

    /// The per-series values of a detector in a bucket.
    fn series_values(&self, d: &Detector, recs: &[Rec]) -> Vec<SeriesValue> {
        let mut groups: BTreeMap<(Option<String>, Option<String>, Option<String>), Vec<&Rec>> = BTreeMap::new();
        for r in recs {
            let get = |f: &Option<String>| -> Option<Option<String>> {
                match f {
                    None => Some(None),
                    Some(name) => match r.fields.get(name).and_then(str_of) {
                        Some(v) => Some(Some(v)),
                        None if d.use_null => Some(Some(String::new())),
                        None => None,
                    },
                }
            };
            let (Some(p), Some(b), Some(o)) = (get(&d.partition), get(&d.by), get(&d.over)) else { continue };
            if let Some(f) = &d.field {
                if !matches!(d.base_function(), "non_null_sum") && !r.fields.contains_key(f) {
                    continue;
                }
            }
            groups.entry((p, b, o)).or_default().push(r);
        }
        let mut out = Vec::new();
        for ((p, b, o), rs) in groups {
            let weight = |r: &Rec| -> f64 { self.summary_count_field.as_ref().and_then(|s| r.fields.get(s)).and_then(num_of).unwrap_or(1.0) };
            let nums: Vec<(f64, f64)> = d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(num_of).map(|v| (v, weight(r)))).collect()).unwrap_or_default();
            let total_w: f64 = rs.iter().map(|r| weight(r)).sum();
            let value = match d.base_function() {
                "count" | "non_zero_count" => total_w,
                "distinct_count" => d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(str_of)).collect::<BTreeSet<_>>().len() as f64).unwrap_or(0.0),
                "rare" | "freq_rare" => rs.len() as f64,
                "info_content" => d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(str_of)).collect::<BTreeSet<_>>().iter().map(|s| s.len() as f64).sum()).unwrap_or(0.0),
                "metric" | "mean" | "avg" => {
                    let w: f64 = nums.iter().map(|(_, w)| w).sum();
                    if w == 0.0 {
                        continue;
                    }
                    nums.iter().map(|(v, w)| v * w).sum::<f64>() / w
                }
                "median" => {
                    if nums.is_empty() {
                        continue;
                    }
                    let mut v: Vec<f64> = nums.iter().map(|(v, _)| *v).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let m = v.len() / 2;
                    if v.len() % 2 == 0 { (v[m - 1] + v[m]) / 2.0 } else { v[m] }
                }
                "min" => {
                    if nums.is_empty() {
                        continue;
                    }
                    nums.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min)
                }
                "max" => {
                    if nums.is_empty() {
                        continue;
                    }
                    nums.iter().map(|(v, _)| *v).fold(f64::NEG_INFINITY, f64::max)
                }
                "sum" | "non_null_sum" => {
                    if nums.is_empty() && d.base_function() == "non_null_sum" {
                        continue;
                    }
                    nums.iter().map(|(v, w)| v * w).sum()
                }
                "varp" => {
                    if nums.is_empty() {
                        continue;
                    }
                    let n = nums.len() as f64;
                    let mean = nums.iter().map(|(v, _)| *v).sum::<f64>() / n;
                    nums.iter().map(|(v, _)| (v - mean).powi(2)).sum::<f64>() / n
                }
                "time_of_day" => rs.iter().map(|r| (r.time.rem_euclid(86_400_000) / 1000) as f64).sum::<f64>() / rs.len() as f64,
                "time_of_week" => rs.iter().map(|r| (r.time.rem_euclid(604_800_000) / 1000) as f64).sum::<f64>() / rs.len() as f64,
                _ => {
                    if nums.is_empty() {
                        continue;
                    }
                    nums.iter().map(|(v, _)| *v).sum::<f64>() / nums.len() as f64
                }
            };
            let mut influencers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            for inf in &self.influencers {
                for r in &rs {
                    if let Some(v) = r.fields.get(inf).and_then(str_of) {
                        influencers.entry(inf.clone()).or_default().insert(v);
                    }
                }
            }
            out.push(SeriesValue { partition: p, by: b, over: o, value, influencers });
        }
        out
    }

    /// Does a detection rule with `action` apply to this result?
    fn rule_applies(&self, d: &Detector, action: &str, actual: f64, typical: f64, ts: i64, sv: &SeriesValue, filters: &BTreeMap<String, Vec<String>>) -> bool {
        for r in &d.rules {
            let actions: Vec<&str> = r["actions"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
            if !actions.contains(&action) {
                continue;
            }
            let mut ok = true;
            for c in r["conditions"].as_array().into_iter().flatten() {
                let lhs = match c["applies_to"].as_str().unwrap_or("actual") {
                    "typical" => typical,
                    "diff_from_typical" => (actual - typical).abs(),
                    "time" => ts as f64 / 1000.0,
                    _ => actual,
                };
                let rhs = c["value"].as_f64().unwrap_or(0.0);
                ok &= match c["operator"].as_str().unwrap_or("gt") {
                    "gt" => lhs > rhs,
                    "gte" => lhs >= rhs,
                    "lt" => lhs < rhs,
                    "lte" => lhs <= rhs,
                    _ => false,
                };
            }
            if let Some(scope) = r["scope"].as_object() {
                for (field, fref) in scope {
                    let value = if Some(field) == d.by.as_ref() {
                        sv.by.clone()
                    } else if Some(field) == d.over.as_ref() {
                        sv.over.clone()
                    } else if Some(field) == d.partition.as_ref() {
                        sv.partition.clone()
                    } else {
                        None
                    };
                    let items = fref["filter_id"].as_str().and_then(|f| filters.get(f)).cloned().unwrap_or_default();
                    let included = value.map(|v| items.iter().any(|i| crate::mapping::wildcard_match(i, &v))).unwrap_or(false);
                    ok &= if fref["filter_type"].as_str() == Some("exclude") { !included } else { included };
                }
            }
            if ok {
                return true;
            }
        }
        false
    }

    /// Compute and emit the results of one bucket.
    fn finalize(&mut self, start: i64, recs: &[Rec], interim: bool, from_data: bool, out: &mut Docs) {
        let t0 = std::time::Instant::now();
        let span_s = self.bucket_span_ms / 1000;
        let job_id = self.job_id.clone();
        let filters: BTreeMap<String, Vec<String>> = FILTERS.lock().get(&job_id).cloned().unwrap_or_default();
        let mut records: Vec<Value> = Vec::new();
        let mut max_score = 0.0f64;
        let mut min_p = 1.0f64;
        // influencer field -> value -> (max score, min p)
        let mut infl: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
        let mut by_values: BTreeSet<String> = BTreeSet::new();
        let mut over_values: BTreeSet<String> = BTreeSet::new();
        let mut partitions: BTreeSet<String> = BTreeSet::new();
        let detectors = self.detectors.clone();
        for d in &detectors {
            let mut values = self.series_values(d, recs);
            // Count functions see 0 for every known series without records.
            if matches!(d.base_function(), "count") && d.over.is_none() {
                let prefix = format!("{}|", d.index);
                let known: Vec<(String, String)> = self.models.keys().filter(|k| k.starts_with(&prefix)).map(|k| {
                    let parts: Vec<&str> = k.split('|').collect();
                    (parts.get(1).unwrap_or(&"").to_string(), parts.get(2).unwrap_or(&"").to_string())
                }).collect();
                for (p, b) in known {
                    let po = if d.partition.is_some() { Some(p) } else { None };
                    let bo = if d.by.is_some() { Some(b) } else { None };
                    if !values.iter().any(|v| v.partition == po && v.by == bo) {
                        values.push(SeriesValue { partition: po, by: bo, over: None, value: 0.0, influencers: BTreeMap::new() });
                    }
                }
                if d.by.is_none() && d.partition.is_none() && values.is_empty() {
                    values.push(SeriesValue { partition: None, by: None, over: None, value: 0.0, influencers: BTreeMap::new() });
                }
            }
            let is_rare = matches!(d.base_function(), "rare" | "freq_rare");
            for sv in &values {
                if let Some(p) = &sv.partition {
                    partitions.insert(p.clone());
                }
                if let Some(b) = &sv.by {
                    by_values.insert(b.clone());
                }
                if let Some(o) = &sv.over {
                    over_values.insert(o.clone());
                }
                let key = Self::series_key(d, &sv.partition, &sv.by);
                let (p, typical) = if is_rare {
                    let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                    let total = *self.frequency_total.get(&fkey).unwrap_or(&0) as f64;
                    let seen = sv.by.as_ref().and_then(|b| self.frequency.get(&fkey).and_then(|m| m.get(b))).copied().unwrap_or(0) as f64;
                    let p = if total < 20.0 { 1.0 } else { ((seen + 0.5) / (total + 1.0)).min(1.0) };
                    (p, seen)
                } else {
                    let m = self.models.get(&key).cloned().unwrap_or_default();
                    (m.probability(sv.value, d.side(), d.count_like()), if m.n > 0.0 { m.mean } else { sv.value })
                };
                let skip_result = self.rule_applies(d, "skip_result", sv.value, typical, start, sv, &filters);
                let skip_update = self.rule_applies(d, "skip_model_update", sv.value, typical, start, sv, &filters);
                let score = if skip_result { 0.0 } else { score_of(p) };
                if score > 0.0 {
                    max_score = max_score.max(score);
                    min_p = min_p.min(p);
                    let mut r = Map::new();
                    r.insert("job_id".into(), json!(job_id));
                    r.insert("result_type".into(), json!("record"));
                    r.insert("probability".into(), json!(p));
                    r.insert("multi_bucket_impact".into(), json!(-5.0));
                    r.insert("record_score".into(), json!(score));
                    r.insert("initial_record_score".into(), json!(score));
                    r.insert("bucket_span".into(), json!(span_s));
                    r.insert("detector_index".into(), json!(d.index));
                    r.insert("is_interim".into(), json!(interim));
                    r.insert("timestamp".into(), json!(start));
                    if let (Some(n), Some(v)) = (&d.by, &sv.by) {
                        r.insert("by_field_name".into(), json!(n));
                        r.insert("by_field_value".into(), json!(v));
                    }
                    if let (Some(n), Some(v)) = (&d.partition, &sv.partition) {
                        r.insert("partition_field_name".into(), json!(n));
                        r.insert("partition_field_value".into(), json!(v));
                    }
                    r.insert("function".into(), json!(d.description()));
                    r.insert("function_description".into(), json!(d.description()));
                    r.insert("typical".into(), json!([typical]));
                    r.insert("actual".into(), json!([sv.value]));
                    if let Some(f) = &d.field {
                        r.insert("field_name".into(), json!(f));
                    }
                    if let (Some(n), Some(v)) = (&d.over, &sv.over) {
                        r.insert("over_field_name".into(), json!(n));
                        r.insert("over_field_value".into(), json!(v));
                    }
                    if !sv.influencers.is_empty() {
                        let list: Vec<Value> = sv.influencers.iter().map(|(k, vs)| json!({"influencer_field_name": k, "influencer_field_values": vs.iter().collect::<Vec<_>>()})).collect();
                        r.insert("influencers".into(), Value::Array(list));
                    }
                    for (k, vs) in &sv.influencers {
                        for v in vs {
                            let e = infl.entry(k.clone()).or_default().entry(v.clone()).or_insert((0.0, 1.0));
                            e.0 = e.0.max(score);
                            e.1 = e.1.min(p);
                        }
                    }
                    records.push(Value::Object(r));
                }
                if self.model_plot && !is_rare {
                    let allowed = self.model_plot_terms.as_ref().map(|t| [&sv.by, &sv.over, &sv.partition].into_iter().flatten().all(|v| t.iter().any(|x| x == v))).unwrap_or(true);
                    if allowed {
                        let m = self.models.get(&key).cloned().unwrap_or_default();
                        let (lo, med, hi) = m.bounds(d.count_like());
                        let mut mp = Map::new();
                        mp.insert("job_id".into(), json!(job_id));
                        mp.insert("result_type".into(), json!("model_plot"));
                        mp.insert("bucket_span".into(), json!(span_s));
                        mp.insert("detector_index".into(), json!(d.index));
                        mp.insert("timestamp".into(), json!(start));
                        if let (Some(n), Some(v)) = (&d.partition, &sv.partition) {
                            mp.insert("partition_field_name".into(), json!(n));
                            mp.insert("partition_field_value".into(), json!(v));
                        }
                        if let (Some(n), Some(v)) = (&d.over, &sv.over) {
                            mp.insert("over_field_name".into(), json!(n));
                            mp.insert("over_field_value".into(), json!(v));
                        }
                        if let (Some(n), Some(v)) = (&d.by, &sv.by) {
                            mp.insert("by_field_name".into(), json!(n));
                            mp.insert("by_field_value".into(), json!(v));
                        }
                        mp.insert("model_feature".into(), json!(d.feature()));
                        mp.insert("model_lower".into(), json!(lo));
                        mp.insert("model_upper".into(), json!(hi));
                        mp.insert("model_median".into(), json!(med));
                        mp.insert("actual".into(), json!(sv.value));
                        let id = format!("{job_id}_model_plot_{start}_{span_s}_{}_{}", d.index, values_to_id(&[sv.by.as_deref(), sv.over.as_deref(), sv.partition.as_deref()]));
                        if !interim {
                            out.push((id, Value::Object(mp)));
                        }
                    }
                }
                if !interim && !skip_update {
                    if is_rare {
                        let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                        if let Some(b) = &sv.by {
                            *self.frequency.entry(fkey).or_default().entry(b.clone()).or_insert(0) += 1;
                        }
                    } else {
                        self.models.entry(key).or_default().update(sv.value);
                    }
                }
            }
            if is_rare && !interim {
                let fkey = format!("{}|", d.index);
                let parts: BTreeSet<String> = values.iter().map(|v| v.partition.clone().unwrap_or_default()).collect();
                for p in parts {
                    *self.frequency_total.entry(format!("{fkey}{p}")).or_insert(0) += 1;
                }
                if values.is_empty() {
                    *self.frequency_total.entry(fkey).or_insert(0) += 1;
                }
            }
        }
        // Bucket influencers.
        let mut bucket_influencers: Vec<Value> = Vec::new();
        if max_score > 0.0 {
            for (field, vals) in &infl {
                let s = vals.values().map(|(s, _)| *s).fold(0.0, f64::max);
                let p = vals.values().map(|(_, p)| *p).fold(1.0, f64::min);
                bucket_influencers.push(json!({"job_id": job_id, "result_type": "bucket_influencer", "influencer_field_name": field, "initial_anomaly_score": s, "anomaly_score": s,
                    "raw_anomaly_score": -p.log10(), "probability": p, "timestamp": start, "bucket_span": span_s, "is_interim": interim}));
            }
            bucket_influencers.push(json!({"job_id": job_id, "result_type": "bucket_influencer", "influencer_field_name": "bucket_time", "initial_anomaly_score": max_score, "anomaly_score": max_score,
                "raw_anomaly_score": -min_p.log10(), "probability": min_p, "timestamp": start, "bucket_span": span_s, "is_interim": interim}));
        }
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        let bucket = json!({"job_id": job_id, "timestamp": start, "anomaly_score": max_score, "bucket_span": span_s, "initial_anomaly_score": max_score, "event_count": recs.len(),
            "is_interim": interim, "bucket_influencers": bucket_influencers, "processing_time_ms": elapsed as i64, "result_type": "bucket"});
        out.push((format!("{job_id}_bucket_{start}_{span_s}"), bucket));
        for bi in &bucket_influencers {
            let field = bi["influencer_field_name"].as_str().unwrap_or("");
            out.push((format!("{job_id}_bucket_influencer_{start}_{span_s}_{field}"), bi.clone()));
        }
        for r in &records {
            let id = format!("{job_id}_record_{start}_{span_s}_{}_{}", r["detector_index"].as_u64().unwrap_or(0), values_to_id(&[r["by_field_value"].as_str(), r["over_field_value"].as_str(), r["partition_field_value"].as_str()]));
            out.push((id, r.clone()));
        }
        for (field, vals) in &infl {
            for (v, (s, p)) in vals {
                let doc = json!({"job_id": job_id, "result_type": "influencer", "influencer_field_name": field, "influencer_field_value": v, field: v, "influencer_score": s,
                    "initial_influencer_score": s, "probability": p, "bucket_span": span_s, "is_interim": interim, "timestamp": start});
                out.push((format!("{job_id}_influencer_{start}_{span_s}_{field}_{}", values_to_id(&[Some(v.as_str())])), doc));
            }
        }
        if interim {
            self.has_interim = true;
            return;
        }
        self.timing.add(elapsed);
        self.latest_result_ts = Some(start);
        if from_data {
            self.counts.bucket_count += 1;
            self.counts.latest_bucket_timestamp = Some(start);
            if recs.is_empty() {
                self.counts.empty_bucket_count += 1;
                self.counts.latest_empty_bucket_timestamp = Some(start);
            }
        }
        // Model size: a rough footprint of the series models.
        self.size.by_count = self.size.by_count.max(by_values.len() as u64 + self.detectors.iter().filter(|d| d.by.is_none()).count() as u64);
        self.size.over_count = self.size.over_count.max(over_values.len() as u64);
        self.size.partition_count = self.size.partition_count.max(partitions.len() as u64 + self.detectors.len() as u64);
        self.size.model_bytes = 20_000 + 1_500 * self.models.len() as u64 + 400 * self.frequency.values().map(|m| m.len() as u64).sum::<u64>() + 600 * self.categories.len() as u64;
        self.size.peak_model_bytes = self.size.peak_model_bytes.max(self.size.model_bytes);
        self.size.timestamp = Some(start);
        self.size.total_category_count = self.categories.len() as u64;
        if self.timing.bucket_count % 10 == 1 {
            self.emit_model_size(out);
        }
        self.emit_categories(out);
    }

    fn emit_model_size(&mut self, out: &mut Docs) {
        let now = crate::date::now_millis();
        let log_time = match self.size_log_time {
            Some(t) if t >= now => t + 1,
            _ => now,
        };
        self.size_log_time = Some(log_time);
        let mml = MML.lock().get(&self.job_id).copied().unwrap_or(1 << 30);
        out.push((format!("{}_model_size_stats_{log_time}", self.job_id), self.size.to_json(&self.job_id, mml, log_time)));
    }

    // -----------------------------------------------------------------------
    // Categorization
    // -----------------------------------------------------------------------

    fn categorize(&mut self, msg: &str, partition: Option<String>) -> i64 {
        let toks = tokenize(msg);
        let mut best: Option<(usize, f64)> = None;
        for (i, c) in self.categories.iter().enumerate() {
            if c.partition != partition {
                continue;
            }
            let common = common_subsequence(&c.tokens, &toks);
            let denom = (c.tokens.len() + toks.len()) as f64;
            let sim = if denom == 0.0 { 1.0 } else { 2.0 * common.len() as f64 / denom };
            if sim >= 0.7 && best.map(|(_, s)| sim > s).unwrap_or(true) {
                best = Some((i, sim));
            }
        }
        self.size.categorized_doc_count += 1;
        match best {
            Some((i, _)) => {
                let c = &mut self.categories[i];
                c.terms = common_subsequence(&c.terms, &toks);
                c.num_matches += 1;
                c.max_len = c.max_len.max(msg.len());
                if c.examples.len() < self.cat_examples_limit && !c.examples.iter().any(|e| e == msg) {
                    c.examples.push(msg.to_string());
                }
                c.dirty = true;
                c.id
            }
            None => {
                let id = self.categories.len() as i64 + 1;
                self.categories.push(Category { id, terms: toks.clone(), tokens: toks, examples: if self.cat_examples_limit > 0 { vec![msg.to_string()] } else { vec![] }, max_len: msg.len(), num_matches: 1, partition, dirty: true });
                id
            }
        }
    }

    fn emit_categories(&mut self, out: &mut Docs) {
        let job_id = self.job_id.clone();
        let mut frequent = 0u64;
        let mut rare = 0u64;
        let total_docs = self.size.categorized_doc_count.max(1) as f64;
        let pf = self.detectors.iter().find_map(|d| d.partition.clone());
        for c in &mut self.categories {
            if c.num_matches as f64 / total_docs >= 0.01 && c.num_matches > 1 {
                frequent += 1;
            } else {
                rare += 1;
            }
            if !c.dirty {
                continue;
            }
            c.dirty = false;
            let terms = c.terms.join(" ");
            let regex = category_regex(&c.terms);
            let mut doc = Map::new();
            doc.insert("job_id".into(), json!(job_id));
            doc.insert("category_id".into(), json!(c.id));
            if let (Some(p), Some(pf)) = (&c.partition, &pf) {
                doc.insert("partition_field_name".into(), json!(pf));
                doc.insert("partition_field_value".into(), json!(p));
            }
            doc.insert("terms".into(), json!(terms));
            doc.insert("regex".into(), json!(regex));
            doc.insert("max_matching_length".into(), json!((c.max_len as f64 * 1.1).floor() as i64));
            doc.insert("examples".into(), json!(c.examples));
            doc.insert("num_matches".into(), json!(c.num_matches));
            doc.insert("result_type".into(), json!("category_definition"));
            doc.insert("mlcategory".into(), json!(c.id.to_string()));
            out.push((format!("{job_id}_category_definition_{}", c.id), Value::Object(doc)));
        }
        if self.cat_field.is_some() {
            self.size.frequent_category_count = frequent;
            self.size.rare_category_count = rare;
            self.size.total_category_count = self.categories.len() as u64;
        }
    }

    fn categorizer_stats_doc(&self) -> Option<(String, Value)> {
        self.cat_field.as_ref()?;
        let now = crate::date::now_millis();
        Some((
            format!("{}_categorizer_stats_{now}", self.job_id),
            json!({"job_id": self.job_id, "result_type": "categorizer_stats", "categorized_doc_count": self.size.categorized_doc_count, "total_category_count": self.size.total_category_count,
                "frequent_category_count": self.size.frequent_category_count, "rare_category_count": self.size.rare_category_count, "dead_category_count": 0, "failed_category_count": 0,
                "categorization_status": "ok", "log_time": now, "timestamp": self.latest_result_ts}),
        ))
    }

    // -----------------------------------------------------------------------
    // Flush / forecast
    // -----------------------------------------------------------------------

    fn flush(&mut self, params: &super::anomaly::FlushParams, out: &mut Docs) -> Value {
        if let Some(adv) = params.advance_time {
            let target = floor_to(adv, self.bucket_span_ms);
            let cb = self.current_bucket;
            match cb {
                Some(cur) if target > cur => self.advance_to(target, false, out),
                None => self.current_bucket = Some(target),
                _ => {}
            }
        }
        if let Some(skip) = params.skip_time {
            let target = floor_to(skip, self.bucket_span_ms) + self.bucket_span_ms;
            if self.current_bucket.map(|c| target > c).unwrap_or(true) {
                self.cur.clear();
                if self.has_interim {
                    self.has_interim = false;
                    out.push(("__delete_interim__".into(), Value::Null));
                }
                self.current_bucket = Some(target);
            }
        }
        if params.calc_interim {
            if let Some(cur) = self.current_bucket {
                let in_range = params.start.map(|s| cur + self.bucket_span_ms > s).unwrap_or(true) && params.end.map(|e| cur < e).unwrap_or(true);
                if in_range && !self.cur.is_empty() {
                    let recs = self.cur.clone();
                    if self.has_interim {
                        out.push(("__delete_interim__".into(), Value::Null));
                    }
                    self.finalize(cur, &recs, true, false, out);
                }
            }
        }
        let mut o = Map::new();
        o.insert("flushed".into(), json!(true));
        if let Some(cur) = self.current_bucket {
            o.insert("last_finalized_bucket_end".into(), json!(cur));
        }
        Value::Object(o)
    }

    fn forecast(&self, duration_ms: i64, expires_in_ms: i64, out: &mut Docs) -> Result<Value> {
        if self.timing.bucket_count == 0 || self.counts.processed_record_count == 0 {
            let e = EsError::new(400, "status_exception", "Cannot run forecast: Forecast cannot be executed as job requires data to have been processed and modeled");
            return Err(e.clone().caused_by(e));
        }
        let fid = forecast_id();
        let span = self.bucket_span_ms;
        let span_s = span / 1000;
        let start = self.current_bucket.unwrap_or(0);
        let end = start + floor_to(duration_ms.max(span), span).max(span);
        let mut n = 0u64;
        let mut t = start;
        while t < end {
            for d in &self.detectors {
                if !d.forecastable() {
                    continue;
                }
                let prefix = format!("{}|", d.index);
                for (key, m) in self.models.iter().filter(|(k, _)| k.starts_with(&prefix)) {
                    let parts: Vec<&str> = key.split('|').collect();
                    let partition = parts.get(1).filter(|s| !s.is_empty()).map(|s| s.to_string());
                    let by = parts.get(2).filter(|s| !s.is_empty()).map(|s| s.to_string());
                    let (lo, med, hi) = m.bounds(d.count_like());
                    let mut doc = Map::new();
                    doc.insert("job_id".into(), json!(self.job_id));
                    doc.insert("forecast_id".into(), json!(fid));
                    doc.insert("result_type".into(), json!("model_forecast"));
                    doc.insert("bucket_span".into(), json!(span_s));
                    doc.insert("detector_index".into(), json!(d.index));
                    doc.insert("timestamp".into(), json!(t));
                    if let (Some(pn), Some(pv)) = (&d.partition, &partition) {
                        doc.insert("partition_field_name".into(), json!(pn));
                        doc.insert("partition_field_value".into(), json!(pv));
                    }
                    if let (Some(bn), Some(bv)) = (&d.by, &by) {
                        doc.insert("by_field_name".into(), json!(bn));
                        doc.insert("by_field_value".into(), json!(bv));
                    }
                    doc.insert("model_feature".into(), json!(d.feature()));
                    doc.insert("forecast_lower".into(), json!(lo));
                    doc.insert("forecast_upper".into(), json!(hi));
                    doc.insert("forecast_prediction".into(), json!(med));
                    out.push((format!("{}_model_forecast_{fid}_{t}_{span_s}_{}_{}", self.job_id, d.index, values_to_id(&[by.as_deref(), None, partition.as_deref()])), Value::Object(doc)));
                    n += 1;
                }
            }
            t += span;
        }
        let now = crate::date::now_millis() / 1000 * 1000;
        let stats = json!({"job_id": self.job_id, "result_type": "model_forecast_request_stats", "forecast_id": fid, "processed_record_count": n, "forecast_messages": [], "timestamp": start,
            "forecast_start_timestamp": start, "forecast_end_timestamp": end, "forecast_create_timestamp": now, "forecast_expiry_timestamp": now + expires_in_ms, "forecast_progress": 1.0,
            "processing_time_ms": 0, "forecast_memory_bytes": 20_000 + 200 * n, "forecast_status": "finished"});
        out.push((format!("{}_model_forecast_request_stats_{fid}", self.job_id), stats));
        Ok(json!({"acknowledged": true, "forecast_id": fid}))
    }
}

/// `UUIDs.base64UUID()`-shaped forecast id (20 url-safe characters).
fn forecast_id() -> String {
    crate::index::new_uuid()[..20].to_string()
}

// ---------------------------------------------------------------------------
// Categorization helpers
// ---------------------------------------------------------------------------

/// Tokens of a message for categorization: whitespace separated, trimmed of
/// punctuation; tokens that start with a digit or carry no letter are
/// variable parts and dropped.
fn tokenize(msg: &str) -> Vec<String> {
    let first_line = msg.lines().find(|l| l.chars().any(|c| c.is_alphabetic())).unwrap_or(msg);
    first_line
        .split_whitespace()
        .filter_map(|raw| {
            let t = raw.trim_matches(|c: char| !(c.is_alphanumeric() || c == '/' || c == '_'));
            let t = t.trim_end_matches(|c: char| c == '/' );
            if t.is_empty() || !t.chars().any(|c| c.is_alphabetic()) {
                return None;
            }
            let first = t.chars().find(|c| c.is_alphanumeric())?;
            if first.is_ascii_digit() {
                return None;
            }
            Some(t.to_string())
        })
        .collect()
}

/// The ordered tokens common to `a` and `b` (longest common subsequence).
fn common_subsequence(a: &[String], b: &[String]) -> Vec<String> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            dp[i + 1][j + 1] = if a[i] == b[j] { dp[i][j] + 1 } else { dp[i][j + 1].max(dp[i + 1][j]) };
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            out.push(a[i - 1].clone());
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out
}

fn regex_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        if "\\^$|?*+()[]{}".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

fn category_regex(terms: &[String]) -> String {
    if terms.is_empty() {
        return ".*".into();
    }
    format!(".*?{}.*", terms.iter().map(|t| regex_escape(t)).collect::<Vec<_>>().join(".+?"))
}

fn is_number_token(t: &str) -> bool {
    let t = t.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
    !t.is_empty() && t.parse::<f64>().is_ok()
}

fn is_ip_token(t: &str) -> bool {
    let t = t.trim_matches(|c: char| !c.is_alphanumeric() && c != '.');
    let parts: Vec<&str> = t.split('.').collect();
    parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok())
}

/// `GrokPatternCreator.findBestGrokMatchFromExamples`: the category terms
/// with grok placeholders where every example has a number / IP between
/// two terms.
pub fn grok_pattern(terms: &str, examples: &[String]) -> String {
    let terms: Vec<&str> = terms.split_whitespace().collect();
    if terms.is_empty() || examples.is_empty() {
        return String::new();
    }
    // Per example: the raw tokens between consecutive terms.
    let mut gaps: Vec<Vec<Vec<String>>> = vec![Vec::new(); terms.len() + 1];
    for ex in examples {
        let toks: Vec<&str> = ex.split_whitespace().collect();
        let mut pos = 0usize;
        let mut ok = true;
        let mut ex_gaps: Vec<Vec<String>> = Vec::new();
        for term in &terms {
            let Some(found) = toks[pos..].iter().position(|t| t.trim_matches(|c: char| !(c.is_alphanumeric() || c == '/' || c == '_')) == *term) else {
                ok = false;
                break;
            };
            ex_gaps.push(toks[pos..pos + found].iter().map(|s| s.to_string()).collect());
            pos += found + 1;
        }
        if !ok {
            continue;
        }
        ex_gaps.push(toks[pos..].iter().map(|s| s.to_string()).collect());
        for (i, g) in ex_gaps.into_iter().enumerate() {
            gaps[i].push(g);
        }
    }
    let placeholder = |g: &Vec<Vec<String>>| -> Option<&'static str> {
        if g.is_empty() || g.iter().any(|x| x.len() != 1) {
            return None;
        }
        if g.iter().all(|x| is_ip_token(&x[0])) {
            return Some("%{IP:ipaddress}");
        }
        if g.iter().all(|x| is_number_token(&x[0])) {
            return Some("%{NUMBER:field}");
        }
        None
    };
    let mut out = String::new();
    match placeholder(&gaps[0]) {
        Some(p) => out.push_str(&format!(".*?{p}.+?")),
        None => out.push_str(".*?"),
    }
    for (i, term) in terms.iter().enumerate() {
        out.push_str(&regex_escape(term));
        if i + 1 < terms.len() {
            match placeholder(&gaps[i + 1]) {
                Some(p) => out.push_str(&format!(".+?{p}.+?")),
                None => out.push_str(".+?"),
            }
        }
    }
    match placeholder(&gaps[terms.len()]) {
        Some(p) => out.push_str(&format!(".+?{p}.*")),
        None => out.push_str(".*"),
    }
    out
}

// ---------------------------------------------------------------------------
// Registry and API
// ---------------------------------------------------------------------------

fn procs() -> &'static Mutex<HashMap<String, Arc<Mutex<Proc>>>> {
    static P: OnceLock<Mutex<HashMap<String, Arc<Mutex<Proc>>>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// Filter items by filter id, per job (for rule scopes).
static FILTERS: Mutex<BTreeMap<String, BTreeMap<String, Vec<String>>>> = Mutex::new(BTreeMap::new());
/// Model memory limit (bytes) per job id.
static MML: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

fn key(node: &Node, job_id: &str) -> String {
    format!("{}#{job_id}", node.data.display())
}

/// Is a process live for the job on this node?
pub fn is_live(node: &Node, job_id: &str) -> bool {
    procs().lock().contains_key(&key(node, job_id))
}

fn parse_state(v: &Value) -> Option<Proc> {
    let mut p: Proc = serde_json::from_value(v.get(STATE_KEY)?.clone()).ok()?;
    p.init_fmt();
    Some(p)
}

/// Can the snapshot's model state be loaded by this node?
pub fn snapshot_readable(node: &Node, job_id: &str, snapshot_id: &str) -> bool {
    ridx::get(node, ridx::STATE_PATTERN, &format!("{job_id}_model_state_{snapshot_id}#1")).and_then(|v| parse_state(&v)).is_some()
}

fn build(node: &Node, job_id: &str) -> Result<Arc<Mutex<Proc>>> {
    let st = service(node).read();
    let job = st.jobs.get(job_id).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{job_id}'")))?;
    let filters: BTreeMap<String, Vec<String>> = st.filters.iter().map(|(k, f)| (k.clone(), f["items"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect())).collect();
    drop(st);
    FILTERS.lock().insert(job_id.to_string(), filters);
    MML.lock().insert(job_id.to_string(), super::job::mml_bytes(&job));
    let restored = job["model_snapshot_id"].as_str().and_then(|sid| ridx::get(node, ridx::STATE_PATTERN, &format!("{job_id}_model_state_{sid}#1"))).and_then(|v| parse_state(&v));
    let mut p = match restored {
        Some(mut p) => {
            p.refresh_config(&job);
            p
        }
        None => Proc::new(&job),
    };
    // Counts and timing continue from what the index holds.
    if let Some(dc) = super::results::stored_data_counts(node, job_id) {
        p.counts = DataCounts::from_json(&dc);
    }
    if p.timing.bucket_count == 0 {
        if let Some(ts) = super::results::stored_timing_stats(node, job_id) {
            p.timing = TimingStats::from_json(&ts);
        }
    }
    Ok(Arc::new(Mutex::new(p)))
}

/// The job's process, created (restoring the model snapshot) when none is
/// live yet.
fn get_or_build(node: &Node, job_id: &str) -> Result<Arc<Mutex<Proc>>> {
    let k = key(node, job_id);
    if let Some(p) = procs().lock().get(&k) {
        return Ok(p.clone());
    }
    let p = build(node, job_id)?;
    procs().lock().insert(k, p.clone());
    Ok(p)
}

/// Launch the process of a job that was just opened.
pub fn open(node: &Node, job_id: &str) -> Result<()> {
    get_or_build(node, job_id).map(|_| ())
}

/// Forget the process without persisting (revert / reset / failed open).
pub fn drop_process(node: &Node, job_id: &str) {
    procs().lock().remove(&key(node, job_id));
}

fn job_config(node: &Node, job_id: &str) -> Value {
    service(node).read().jobs.get(job_id).cloned().unwrap_or(Value::Null)
}

/// Write result documents (handling the interim-delete marker).
fn write_docs(node: &Node, job_id: &str, job: &Value, docs: Docs) -> Result<()> {
    let mut batch: Docs = Vec::new();
    for (id, doc) in docs {
        if id == "__delete_interim__" {
            if !batch.is_empty() {
                ridx::write_results(node, job_id, job, std::mem::take(&mut batch))?;
            }
            ridx::delete_by_query(node, &read_alias(job_id), json!({"term": {"is_interim": true}}));
        } else {
            batch.push((id, doc));
        }
    }
    ridx::write_results(node, job_id, job, batch)
}

fn stats_docs(p: &Proc) -> Docs {
    let mut docs = vec![(format!("{}_data_counts", p.job_id), p.counts.to_json(&p.job_id, true)), (format!("{}_timing_stats", p.job_id), p.timing.to_doc(&p.job_id, p.latest_result_ts))];
    if let Some(cs) = p.categorizer_stats_doc() {
        docs.push(cs);
    }
    docs
}

/// Parse a `_data` body: NDJSON / concatenated JSON objects or a JSON array.
fn parse_body(body: &[u8]) -> Result<Vec<Value>> {
    let mut docs = Vec::new();
    let stream = serde_json::Deserializer::from_slice(body).into_iter::<Value>();
    for v in stream {
        match v {
            Ok(Value::Array(a)) => docs.extend(a.into_iter().filter(|x| x.is_object())),
            Ok(Value::Object(o)) => docs.push(Value::Object(o)),
            Ok(_) => {}
            Err(e) => {
                if docs.is_empty() {
                    return Err(EsError::new(400, "parse_exception", format!("Failed to parse data: {e}")));
                }
                break;
            }
        }
    }
    Ok(docs)
}

/// `POST _data`: the updated data counts (without `log_time`).
pub fn post_data(node: &Node, job_id: &str, body: &[u8]) -> Result<Value> {
    let docs = parse_body(body)?;
    post_docs(node, job_id, docs, body.len(), false)
}

/// Feed prepared documents (datafeed: `epoch_ms` time values).
pub fn post_docs(node: &Node, job_id: &str, docs: Vec<Value>, bytes: usize, epoch_ms: bool) -> Result<Value> {
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let mut g = p.lock();
    let mut out = Vec::new();
    g.ingest(docs, bytes, epoch_ms, &mut out);
    // Category definitions are written as they are created, not per bucket.
    g.emit_categories(&mut out);
    out.extend(stats_docs(&g));
    let counts = g.counts.to_json(job_id, false);
    drop(g);
    write_docs(node, job_id, &job, out)?;
    Ok(counts)
}

/// `POST _flush`.
pub fn flush(node: &Node, job_id: &str, params: &super::anomaly::FlushParams) -> Result<Value> {
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let mut g = p.lock();
    let mut out = Vec::new();
    let resp = g.flush(params, &mut out);
    out.extend(stats_docs(&g));
    drop(g);
    write_docs(node, job_id, &job, out)?;
    Ok(resp)
}

/// `POST _forecast`.
pub fn forecast(node: &Node, job_id: &str, duration_ms: i64, expires_in_ms: i64) -> Result<Value> {
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let g = p.lock();
    let mut out = Vec::new();
    let resp = g.forecast(duration_ms, expires_in_ms, &mut out)?;
    drop(g);
    write_docs(node, job_id, &job, out)?;
    Ok(resp)
}

/// Close the job's process: persist stats and, when data was processed, a
/// model snapshot. Returns the new snapshot id.
pub fn close(node: &Node, job_id: &str) -> Result<Option<String>> {
    let k = key(node, job_id);
    let Some(p) = procs().lock().remove(&k) else { return Ok(None) };
    let job = job_config(node, job_id);
    let mut g = p.lock();
    let mut out = Vec::new();
    if g.has_interim {
        g.has_interim = false;
        out.push(("__delete_interim__".into(), Value::Null));
    }
    g.emit_categories(&mut out);
    g.emit_model_size(&mut out);
    out.extend(stats_docs(&g));
    let mut snapshot = None;
    if g.counts.processed_record_count > 0 || g.timing.bucket_count > 0 {
        let now = crate::date::now_millis();
        let secs = now / 1000;
        let sid = secs.to_string();
        let mml = MML.lock().get(job_id).copied().unwrap_or(1 << 30);
        let when = crate::date::DateFormatter::new("strict_date_optional_time").map(|f| f.format(secs * 1000, crate::date::Tz::UTC)).unwrap_or_default();
        let when = match when.find('.') {
            Some(i) => format!("{}+0000", &when[..i]),
            None => when.trim_end_matches('Z').to_string() + "+0000",
        };
        let mut snap = Map::new();
        snap.insert("job_id".into(), json!(job_id));
        snap.insert("min_version".into(), json!(SNAPSHOT_MIN_VERSION));
        snap.insert("timestamp".into(), json!(secs * 1000));
        snap.insert("description".into(), json!(format!("State persisted due to job close at {when}")));
        snap.insert("snapshot_id".into(), json!(sid));
        snap.insert("snapshot_doc_count".into(), json!(1));
        snap.insert("model_size_stats".into(), g.size.to_json(job_id, mml, g.size_log_time.unwrap_or(now)));
        if let Some(t) = g.counts.latest_record_timestamp {
            snap.insert("latest_record_time_stamp".into(), json!(t));
        }
        if let Some(t) = g.latest_result_ts {
            snap.insert("latest_result_time_stamp".into(), json!(t));
        }
        snap.insert("quantiles".into(), json!({"job_id": job_id, "timestamp": g.latest_result_ts.unwrap_or(0), "quantile_state": ""}));
        snap.insert("retain".into(), json!(false));
        out.push((super::results::snapshot_doc_id(job_id, &sid), Value::Object(snap)));
        let state = serde_json::to_value(&*g).unwrap_or(Value::Null);
        let state_docs = vec![
            (format!("{job_id}_model_state_{sid}#1"), json!({STATE_KEY: state})),
            (format!("{job_id}_quantiles"), json!({"job_id": job_id, "timestamp": g.latest_result_ts.unwrap_or(0), "quantile_state": ""})),
        ];
        drop(g);
        write_docs(node, job_id, &job, out)?;
        ridx::write_state(node, state_docs)?;
        snapshot = Some(sid);
    } else {
        drop(g);
        write_docs(node, job_id, &job, out)?;
    }
    Ok(snapshot)
}

/// Live stats of an open job's process: (data counts, model size stats,
/// timing stats).
pub fn live_stats(node: &Node, job_id: &str) -> Option<(Value, Value, Value)> {
    let p = procs().lock().get(&key(node, job_id)).cloned()?;
    let g = p.lock();
    let mml = MML.lock().get(job_id).copied().unwrap_or(1 << 30);
    let now = crate::date::now_millis();
    Some((g.counts.to_json(job_id, true), g.size.to_json(job_id, mml, g.size_log_time.unwrap_or(now)), g.timing.to_json(job_id)))
}

/// Model bytes of every live process (`anomaly_detectors_in_bytes`).
pub fn live_model_bytes(node: &Node) -> u64 {
    let prefix = format!("{}#", node.data.display());
    procs().lock().iter().filter(|(k, _)| k.starts_with(&prefix)).map(|(_, p)| p.lock().size.model_bytes).sum()
}

/// Remove every result and state document of a job (`_reset`, delete).
pub fn delete_job_data(node: &Node, job_id: &str, job: &Value) {
    drop_process(node, job_id);
    FILTERS.lock().remove(job_id);
    ridx::delete_by_query(node, &ridx::results_index(job), json!({"term": {"job_id": job_id}}));
    ridx::delete_by_query(node, ridx::STATE_PATTERN, json!({"bool": {"should": [{"prefix": {"_id": format!("{job_id}_model_state_")}}, {"ids": {"values": [format!("{job_id}_quantiles")]}}], "minimum_should_match": 1}}));
}

/// Update the filter items a job's rules see (`_ml/filters` changes).
pub fn update_filters(node: &Node) {
    let st = service(node).read();
    let filters: BTreeMap<String, Vec<String>> = st.filters.iter().map(|(k, f)| (k.clone(), f["items"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect())).collect();
    let mut g = FILTERS.lock();
    for (_, f) in g.iter_mut() {
        *f = filters.clone();
    }
}

/// Re-read a job's config into its live process (`_update`).
pub fn config_updated(node: &Node, job_id: &str) {
    if let Some(p) = procs().lock().get(&key(node, job_id)).cloned() {
        let job = job_config(node, job_id);
        p.lock().refresh_config(&job);
        MML.lock().insert(job_id.to_string(), super::job::mml_bytes(&job));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_terms() {
        assert_eq!(tokenize("Node 1 started successfully"), vec!["Node", "started", "successfully"]);
        assert_eq!(tokenize("Disk usage at 91% on /dev/sda1"), vec!["Disk", "usage", "at", "on", "/dev/sda1"]);
        assert_eq!(tokenize("Failed to connect to db1.example.com: timeout"), vec!["Failed", "to", "connect", "to", "db1.example.com", "timeout"]);
        let a = tokenize("User alice logged in from 10.0.0.1");
        let b = tokenize("User bob logged in from 10.0.0.2");
        assert_eq!(common_subsequence(&a, &b), vec!["User", "logged", "in", "from"]);
    }

    #[test]
    fn grok_patterns_match_es() {
        assert_eq!(grok_pattern("Node started successfully", &["Node 1 started successfully".into(), "Node 2 started successfully".into()]), ".*?Node.+?%{NUMBER:field}.+?started.+?successfully.*");
        assert_eq!(grok_pattern("User logged in from", &["User alice logged in from 10.0.0.1".into(), "User bob logged in from 10.0.0.2".into()]), ".*?User.+?logged.+?in.+?from.+?%{IP:ipaddress}.*");
        assert_eq!(grok_pattern("Disk usage at on /dev/sda1", &["Disk usage at 91% on /dev/sda1".into()]), ".*?Disk.+?usage.+?at.+?%{NUMBER:field}.+?on.+?/dev/sda1.*");
        assert_eq!(grok_pattern("", &[]), "");
    }

    #[test]
    fn scoring_shape() {
        assert_eq!(score_of(0.5), 0.0);
        assert!(score_of(1e-6) > 15.0 && score_of(1e-6) < 35.0);
        assert!(score_of(1e-20) > 60.0 && score_of(1e-20) < 90.0);
        assert!(score_of(1e-300) > 99.0);
        let mut s = Series::default();
        for x in [10.0, 11.0, 9.0, 10.0, 10.5, 9.5, 10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3] {
            s.update(x);
        }
        assert!(s.probability(10.2, 0, false) > 0.5);
        assert!(s.probability(100.0, 0, false) < 1e-6);
        assert_eq!(s.probability(100.0, -1, false), 1.0);
    }

    #[test]
    fn buckets_and_counts() {
        let job = json!({"job_id": "j", "analysis_config": {"bucket_span": "1h", "detectors": [{"function": "metric", "field_name": "responsetime", "by_field_name": "airline", "detector_index": 0}]},
            "data_description": {"time_field": "time", "time_format": "epoch"}});
        let mut p = Proc::new(&job);
        let mut out = Vec::new();
        p.ingest(vec![json!({"airline": "AAL", "responsetime": 132.2, "sourcetype": "x", "time": 1403481600}), json!({"airline": "JZA", "responsetime": 990.4, "sourcetype": "x", "time": 1403481700})], 150, false, &mut out);
        assert_eq!(p.counts.processed_record_count, 2);
        assert_eq!(p.counts.processed_field_count, 4);
        assert_eq!(p.counts.input_field_count, 6);
        assert_eq!(p.counts.earliest_record_timestamp, Some(1403481600000));
        assert!(out.is_empty());
        let fp = super::super::anomaly::FlushParams::default();
        let r = p.flush(&fp, &mut out);
        assert_eq!(r["last_finalized_bucket_end"], json!(1403481600000i64));
        // skip a bucket
        let fp = super::super::anomaly::FlushParams { skip_time: Some(1403488700000), ..Default::default() };
        let r = p.flush(&fp, &mut out);
        assert_eq!(r["last_finalized_bucket_end"], json!(1403488800000i64));
        p.ingest(vec![json!({"airline": "AAL", "responsetime": 1.0, "time": 1403488600})], 10, false, &mut out);
        assert_eq!(p.counts.out_of_order_timestamp_count, 1);
        p.ingest(vec![json!({"airline": "AAL", "responsetime": 1.0, "time": 1403488900}), json!({"airline": "AAL", "responsetime": 1.0, "time": 1403492400})], 10, false, &mut out);
        let buckets: Vec<&(String, Value)> = out.iter().filter(|(_, d)| d["result_type"] == "bucket").collect();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].1["timestamp"], json!(1403488800000i64));
        assert_eq!(buckets[0].1["event_count"], json!(1));
        assert_eq!(p.counts.bucket_count, 1);
    }
}
