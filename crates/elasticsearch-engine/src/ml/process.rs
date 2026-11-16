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
    /// Bytes the model would have needed past its `model_memory_limit`
    /// (the largest refused allocation), 0 while within it.
    #[serde(default)]
    pub model_bytes_exceeded: u64,
    /// Allocations (new series, rare values, categories) refused at the limit.
    #[serde(default)]
    pub bucket_allocation_failures: u64,
    /// `ok`, or `hard_limit` once an allocation was refused.
    #[serde(default)]
    pub memory_status: String,
}

impl ModelSize {
    /// `ModelSizeStats.toXContent`.
    pub fn to_json(&self, job_id: &str, mml_bytes: u64, log_time: i64) -> Value {
        let mut o = Map::new();
        o.insert("job_id".into(), json!(job_id));
        o.insert("result_type".into(), json!("model_size_stats"));
        o.insert("model_bytes".into(), json!(self.model_bytes));
        o.insert("peak_model_bytes".into(), json!(self.peak_model_bytes));
        o.insert("model_bytes_exceeded".into(), json!(self.model_bytes_exceeded));
        o.insert("model_bytes_memory_limit".into(), json!(mml_bytes));
        o.insert("total_by_field_count".into(), json!(self.by_count));
        o.insert("total_over_field_count".into(), json!(self.over_count));
        o.insert("total_partition_field_count".into(), json!(self.partition_count));
        o.insert("bucket_allocation_failures_count".into(), json!(self.bucket_allocation_failures));
        o.insert("memory_status".into(), json!(if self.memory_status.is_empty() { "ok" } else { self.memory_status.as_str() }));
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

const DAY_SECS: f64 = 86_400.0;
const WEEK_SECS: f64 = 604_800.0;

/// The circular mean of `values` on a circle of `period` (in [0, period)).
fn circular_mean(values: impl Iterator<Item = f64>, period: f64) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for v in values {
        let a = v / period * std::f64::consts::TAU;
        s += a.sin();
        c += a.cos();
    }
    (s.atan2(c) / std::f64::consts::TAU * period).rem_euclid(period)
}

/// A circular model of a periodic time (time of day / week): the running
/// sums of its samples' unit vectors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Circular {
    n: f64,
    sin: f64,
    cos: f64,
}

impl Circular {
    fn update(&mut self, x: f64, period: f64) {
        let a = x / period * std::f64::consts::TAU;
        self.n += 1.0;
        self.sin += a.sin();
        self.cos += a.cos();
    }

    /// The learned usual time (`None`: nothing learned, or no direction).
    fn mean(&self, period: f64) -> Option<f64> {
        (self.n > 0.0 && (self.sin != 0.0 || self.cos != 0.0)).then(|| (self.sin.atan2(self.cos) / std::f64::consts::TAU * period).rem_euclid(period))
    }

    /// Two-sided tail probability of `x`'s circular distance from the mean,
    /// with the circular standard deviation `sqrt(-2 ln R)` (floored, and
    /// widened while young, as [`Series::sigma`]).
    fn probability(&self, x: f64, period: f64) -> f64 {
        let Some(mean) = self.mean(period) else { return 1.0 };
        if self.n < MIN_BUCKETS {
            return 1.0;
        }
        let r = ((self.sin * self.sin + self.cos * self.cos).sqrt() / self.n).clamp(1e-12, 1.0);
        let sd = (-2.0 * r.ln()).max(0.0).sqrt() * (1.0 + 20.0 / self.n).sqrt();
        // At least a quarter hour of the day (or its share of a week).
        let sd = sd.max(900.0 / DAY_SECS * std::f64::consts::TAU * DAY_SECS / period);
        let d = ((x - mean).rem_euclid(period)).min((mean - x).rem_euclid(period)) / period * std::f64::consts::TAU;
        erfc((d / sd) / std::f64::consts::SQRT_2).max(1e-300)
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
    /// The period (seconds) of a periodic time function.
    fn period(&self) -> Option<f64> {
        match self.base_function() {
            "time_of_day" => Some(DAY_SECS),
            "time_of_week" => Some(WEEK_SECS),
            _ => None,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// `lat_long`: the bucket's mean location (latitude, longitude).
    point: Option<(f64, f64)>,
    /// `metric`: the bucket's (min, max) next to its mean `value`.
    extremes: Option<(f64, f64)>,
    /// Influencer field -> values of the contributing records.
    influencers: BTreeMap<String, BTreeSet<String>>,
    /// (influencer field, value) -> the series value without that value's
    /// records (`None`: no records left).
    without: BTreeMap<(String, String), Option<f64>>,
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
    /// Start of the earliest bucket not finalized yet (the one accumulating
    /// records; with `latency`, later buckets accumulate too).
    current_bucket: Option<i64>,
    /// Records of the buckets not finalized yet: part of the model state, so
    /// a closed job reopens with its current bucket's accepted records.
    #[serde(default)]
    cur: Vec<Rec>,
    /// Estimated bytes of `cur`.
    #[serde(skip)]
    cur_bytes: u64,
    /// `analysis_config.latency`: a bucket is finalized once records
    /// `latency` past its end arrived, so later records within it still
    /// join their bucket.
    #[serde(default)]
    latency_ms: i64,
    /// The latest record time seen (the finalization horizon with latency).
    #[serde(default)]
    max_time: Option<i64>,
    /// [`Self::series_key`] -> model. Keys of version [`SERIES_KEY_VERSION`]
    /// escape `|` and `\\` in the values; older states are migrated at
    /// restore ([`parse_state`]).
    models: BTreeMap<String, Series>,
    #[serde(default)]
    key_version: u32,
    /// `lat_long` series ([`Self::series_key`]) -> (latitude, longitude) models.
    #[serde(default)]
    geo: BTreeMap<String, (Series, Series)>,
    /// `metric` series ([`Self::series_key`]) -> (min, max) models; the
    /// mean is in `models`.
    #[serde(default)]
    extremes: BTreeMap<String, (Series, Series)>,
    /// `time_of_day` / `time_of_week` series -> circular models.
    #[serde(default)]
    circular: BTreeMap<String, Circular>,
    /// When the model was last persisted (background persistence), and the
    /// interval between persists.
    #[serde(skip)]
    persisted_at: Option<i64>,
    /// `detector|partition` -> by value -> buckets seen (rare / freq_rare).
    frequency: BTreeMap<String, BTreeMap<String, u64>>,
    /// `detector|partition` -> buckets seen.
    frequency_total: BTreeMap<String, u64>,
    /// `exclude_frequent` (#555), as ml-cpp keeps it per detector and
    /// partition: `detector|partition|person|value` -> decayed buckets the
    /// person (the by value of an individual detector, the over value of a
    /// population one) had records in, and `detector|partition` -> decayed
    /// buckets (`CAnomalyDetectorModel::m_PersonBucketCounts` /
    /// `m_BucketCount`).
    #[serde(default)]
    frequent_buckets: BTreeMap<String, f64>,
    /// Population `exclude_frequent: by`: `detector|partition|attribute|by`
    /// -> the people (over values) that had it, and
    /// `detector|partition|people` -> every person seen
    /// (`CPopulationModel::attributeFrequency`).
    #[serde(default)]
    frequent_people: BTreeMap<String, BTreeSet<String>>,
    /// Population `rare` (#619): `detector|partition` -> the population
    /// members (over values) seen, and -> by value -> the members that had it.
    #[serde(default)]
    population: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    prevalence: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    pub timing: TimingStats,
    pub size: ModelSize,
    categories: Vec<Category>,
    /// Timestamp of the latest result bucket.
    pub latest_result_ts: Option<i64>,
    /// Interim results are written for the current bucket.
    has_interim: bool,
    /// `log_time` of the latest model size stats document.
    size_log_time: Option<i64>,
    /// `analysis_limits.model_memory_limit` in bytes: model growth past it
    /// is refused ([`Self::admit`]), input past it rejected.
    #[serde(default)]
    mml_bytes: u64,
    /// Tokens a categorized message keeps (the analyzer's `limit` filter).
    #[serde(default)]
    cat_token_limit: usize,
    /// `categorization_filters` (removed from a message before tokenizing).
    #[serde(default)]
    cat_filters: Vec<String>,
    #[serde(skip)]
    cat_regex: Vec<regex::Regex>,
    /// The `categorization_analyzer` when it is not the ML tokenizers'
    /// (`ml_standard` / `ml_classic`): a custom chain (#589), built from
    /// `cat_analyzer_def` and used instead of the built-in tokenization.
    #[serde(default)]
    cat_analyzer_def: Value,
    #[serde(skip)]
    cat_analyzer: Option<Arc<rustic_core::analysis::Analyzer>>,
    /// Result documents not written yet (a failed results write): written
    /// first by the next write, before newer results. Model progress is
    /// committed once its results are written; `_flush` and `_close` fail
    /// while any are pending.
    #[serde(skip)]
    pending: Docs,
    #[serde(skip)]
    pending_bytes: u64,
    #[serde(skip)]
    time_fmt: Option<Arc<crate::date::DateFormatter>>,
    /// The node-scoped key of this job ([`key`]: node data path + job id)
    /// its rule filters, scheduled events and memory limit are kept under:
    /// two nodes (or engines) in one process may run jobs of the same id.
    /// Empty for a process built without a node (tests): the job id.
    #[serde(skip)]
    scope: String,
}

/// The least share of a record's anomaly an influencer value must explain
/// (its records removed, the rest scored) to be published as influencing:
/// ml-cpp `CAnomalyDetectorModelConfig::DEFAULT_INFLUENCE_CUTOFF` (0.4), the
/// `CProbabilityAndInfluenceCalculator` cutoff on the committed influence
/// (`intersectionInfluence(logp, influence * logp)`, i.e. the influence
/// itself for one feature); a field's lone value has influence 1.
const MIN_INFLUENCE: f64 = 0.4;
/// `exclude_frequent`: an entity whose frequency is above this share is
/// frequent (ml-cpp `SModelParams::s_ExcludePersonFrequency` and
/// `s_ExcludeAttributeFrequency`, both 0.1, tested by
/// `CPersonFrequencyGreaterThan` / `CAttributeFrequencyGreaterThan`).
const FREQUENT_FRACTION: f64 = 0.1;
/// The frequency of anything before a bucket was counted (ml-cpp
/// `personFrequency` / `attributeFrequency` with no buckets / people).
const FREQUENCY_UNSEEN: f64 = 0.5;
/// ml-cpp `CAnomalyDetectorModelConfig::DEFAULT_DECAY_RATE`, scaled by
/// `bucketNormalizationFactor` (bucket length over 1800 s, at most 1): the
/// per-bucket decay of the person bucket counts.
const FREQUENCY_DECAY_RATE: f64 = 0.0005;
const FREQUENCY_STANDARD_BUCKET_SECS: f64 = 1800.0;
/// Version of the [`Proc::series_key`] encoding (2: escaped values).
const SERIES_KEY_VERSION: u32 = 2;
/// Tokens a categorized message keeps at most, whatever its analyzer.
const MAX_CATEGORIZATION_TOKENS: usize = 1000;
/// Estimated bytes of one series model, one rare-value entry, one category.
const SERIES_BYTES: u64 = 1_500;
const FREQUENCY_BYTES: u64 = 400;
const CATEGORY_BYTES: u64 = 600;
/// Fixed part of a model's footprint.
const MODEL_BASE_BYTES: u64 = 20_000;
/// Memory opening a job reserves on the node beyond its model memory
/// limit (`anomaly::open_job`): the job's in-flight input and results.
pub const JOB_OVERHEAD_BYTES: u64 = 30 << 20;

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
/// ml-cpp `CInfoContent`: the information in a bucket's distinct strings
/// is the zlib-compressed length of their concatenation (`CCompressUtils`),
/// not their byte length, so a varied string carries more than a
/// repetitive one of the same size.
fn info_content(values: &BTreeSet<String>) -> f64 {
    use std::io::Write;
    if values.is_empty() {
        return 0.0;
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    for v in values {
        let _ = z.write_all(v.as_bytes());
    }
    z.finish().map_or(0.0, |b| b.len() as f64)
}

fn flatten(v: &Value, prefix: &str, out: &mut Map<String, Value>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(x, &p, out);
            }
        }
        Value::Array(a) => {
            // Several string values are one dimension value (#622): ES
            // concatenates an array's values after sorting them
            // alphabetically and removing duplicates, so `[shared,
            // worker-a]` and `[shared, worker-b]` stay distinct.
            let strings: Vec<&str> = a.iter().filter_map(Value::as_str).collect();
            if strings.len() > 1 && strings.len() == a.iter().filter(|x| !x.is_null()).count() {
                let set: BTreeSet<&str> = strings.into_iter().collect();
                out.insert(prefix.to_string(), Value::String(set.into_iter().collect::<Vec<_>>().join(",")));
            } else if let Some(first) = a.iter().find(|x| !x.is_null() && !x.is_object() && !x.is_array()) {
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
            cur_bytes: 0,
            latency_ms: ac["latency"].as_str().and_then(|l| super::job::parse_time(l, "latency").ok()).map(|t| t.millis()).unwrap_or(0).max(0),
            max_time: None,
            models: BTreeMap::new(),
            key_version: SERIES_KEY_VERSION,
            geo: BTreeMap::new(),
            extremes: BTreeMap::new(),
            circular: BTreeMap::new(),
            persisted_at: None,
            frequency: BTreeMap::new(),
            frequency_total: BTreeMap::new(), frequent_buckets: BTreeMap::new(), frequent_people: BTreeMap::new(), population: BTreeMap::new(), prevalence: BTreeMap::new(),
            timing: TimingStats::default(),
            size: ModelSize::default(),
            categories: Vec::new(),
            latest_result_ts: None,
            has_interim: false,
            size_log_time: None,
            mml_bytes: super::job::mml_bytes(job),
            cat_token_limit: categorization_token_limit(&ac["categorization_analyzer"]),
            cat_filters: categorization_filters(&ac["categorization_analyzer"]),
            cat_regex: Vec::new(),
            cat_analyzer_def: ac["categorization_analyzer"].clone(),
            cat_analyzer: None,
            pending: Vec::new(),
            pending_bytes: 0,
            time_fmt: None,
            scope: String::new(),
        };
        p.init_fmt();
        p
    }

    fn init_fmt(&mut self) {
        if !matches!(self.time_format.as_str(), "epoch" | "epoch_ms") {
            self.time_fmt = crate::date::DateFormatter::new(&self.time_format).ok().map(Arc::new);
        }
        // Filters were validated as regexes when the job was created.
        self.cat_regex = self.cat_filters.iter().filter_map(|f| regex::Regex::new(f).ok()).collect();
        if self.cat_token_limit == 0 {
            self.cat_token_limit = 100;
        }
        self.cat_analyzer = custom_categorization_analyzer(&self.cat_analyzer_def);
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
        self.mml_bytes = super::job::mml_bytes(job);
        self.cat_filters = categorization_filters(&ac["categorization_analyzer"]);
        self.cat_token_limit = categorization_token_limit(&ac["categorization_analyzer"]);
        self.cat_analyzer_def = ac["categorization_analyzer"].clone();
        self.init_fmt();
    }

    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------

    /// Estimated bytes of the retained model: series and `lat_long`
    /// models, rare-value tables, categories.
    fn model_bytes(&self) -> u64 {
        let frequency: u64 = self.frequency.values().map(|m| m.len() as u64).sum();
        let categories: u64 = self.categories.iter().map(|c| CATEGORY_BYTES + c.examples.iter().chain(&c.terms).chain(&c.tokens).map(|x| x.len() as u64 + 24).sum::<u64>()).sum();
        MODEL_BASE_BYTES + SERIES_BYTES * (self.models.len() + self.geo.len() + 2 * self.extremes.len() + self.circular.len()) as u64 + FREQUENCY_BYTES * frequency + categories
    }

    /// Everything the job holds: the model, the records of buckets not
    /// finalized yet and the results not written yet.
    #[cfg(test)]
    fn used_bytes(&self) -> u64 {
        self.model_bytes() + self.cur_bytes + self.pending_bytes
    }

    /// In-flight bytes (records of open buckets, unwritten results) a job
    /// may hold: its model memory limit, at most the per-job overhead
    /// opening reserves on the node beyond it ([`JOB_OVERHEAD_BYTES`]).
    fn transient_limit(&self) -> u64 {
        self.mml_bytes.min(JOB_OVERHEAD_BYTES)
    }

    /// May the model grow by `bytes` (a new series, rare value or
    /// category)? Past `model_memory_limit` it may not: the allocation is
    /// refused and the model size stats report `hard_limit` (known series
    /// keep learning, input keeps flowing).
    fn admit(&mut self, bytes: u64) -> bool {
        let need = self.model_bytes() + bytes;
        if need <= self.mml_bytes {
            return true;
        }
        self.size.bucket_allocation_failures += 1;
        self.size.model_bytes_exceeded = self.size.model_bytes_exceeded.max(need - self.mml_bytes);
        self.size.memory_status = "hard_limit".into();
        false
    }

    /// Refuse input that would take the job past its limit before any of it
    /// is applied (nothing is half-ingested).
    fn admit_input(&self, docs: &[Value], bytes: usize) -> Result<()> {
        let estimate = 2 * bytes as u64 + 128 * docs.len() as u64;
        let held = self.cur_bytes + self.pending_bytes;
        if held + estimate > self.transient_limit() {
            return Err(EsError::new(429, "status_exception", format!(
                "Cannot accept data for job [{}]: its buffered records and unwritten results [{held}] plus this input [{estimate}] would exceed its in-flight memory limit [{}]",
                self.job_id, self.transient_limit()
            )));
        }
        Ok(())
    }

    fn set_pending(&mut self, docs: Docs) {
        self.pending_bytes = docs.iter().map(|(id, d)| (id.len() + 64) as u64 + serde_json::to_vec(d).map(|v| v.len() as u64).unwrap_or(0)).sum();
        self.pending = docs;
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
            match self.current_bucket {
                // With latency, records up to `latency` before the first
                // one still join their buckets.
                None => self.current_bucket = Some(floor_to(t - self.latency_ms, self.bucket_span_ms)),
                Some(cur) if b < cur => {
                    self.counts.out_of_order_timestamp_count += 1;
                    continue;
                }
                _ => {}
            }
            self.counts.processed_record_count += 1;
            self.counts.earliest_record_timestamp = Some(self.counts.earliest_record_timestamp.map(|e| e.min(t)).unwrap_or(t));
            self.counts.latest_record_timestamp = Some(self.counts.latest_record_timestamp.map(|e| e.max(t)).unwrap_or(t));
            self.counts.last_data_time = Some(now);
            if let Some(cf) = &self.cat_field {
                let partition = if self.per_partition_cat { self.detectors.iter().find_map(|d| d.partition.as_ref()).and_then(|p| rec.get(p)).and_then(str_of) } else { None };
                if let Some(msg) = rec.get(cf).and_then(str_of) {
                    if let Some(cid) = self.categorize(&msg, partition) {
                        rec.insert("mlcategory".into(), json!(cid.to_string()));
                    }
                }
            }
            let rec = Rec { time: t, fields: rec };
            self.cur_bytes += rec_bytes(&rec);
            self.cur.push(rec);
            // Buckets ending `latency` or more before the latest record are
            // final (without latency: every bucket before this record's).
            let latest = self.max_time.map_or(t, |m| m.max(t));
            self.max_time = Some(latest);
            let horizon = floor_to(latest - self.latency_ms, self.bucket_span_ms);
            if self.current_bucket.is_some_and(|cur| horizon > cur) {
                self.advance_to(horizon, true, out);
            }
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
            // This bucket's records (later buckets' stay, with latency).
            let end = cur + self.bucket_span_ms;
            let recs = if self.cur.iter().all(|r| r.time < end) {
                std::mem::take(&mut self.cur)
            } else {
                let (now, later): (Vec<Rec>, Vec<Rec>) = std::mem::take(&mut self.cur).into_iter().partition(|r| r.time < end);
                self.cur = later;
                now
            };
            if recs.is_empty() {
                empties += 1;
            }
            if !recs.is_empty() || empties <= MAX_EMPTY_BUCKETS {
                self.finalize(cur, &recs, false, from_data, out);
            }
            cur += self.bucket_span_ms;
        }
        self.cur_bytes = self.cur.iter().map(rec_bytes).sum();
        self.current_bucket = Some(target);
    }

    /// A series' model key: detector index, then the partition and by
    /// values with `\\` and `|` escaped, so distinct value tuples never
    /// share a key.
    fn series_key(d: &Detector, partition: &Option<String>, by: &Option<String>) -> String {
        format!("{}|{}|{}", d.index, escape_key(partition.as_deref().unwrap_or("")), escape_key(by.as_deref().unwrap_or("")))
    }

    /// The (detector index, partition, by) of a [`Self::series_key`].
    fn split_key(key: &str) -> Option<(usize, String, String)> {
        let (index, rest) = key.split_once('|')?;
        let mut parts = Vec::new();
        let mut cur = String::new();
        let mut chars = rest.chars();
        while let Some(c) = chars.next() {
            match c {
                '\\' => cur.push(chars.next()?),
                '|' => parts.push(std::mem::take(&mut cur)),
                c => cur.push(c),
            }
        }
        parts.push(cur);
        let [partition, by]: [String; 2] = parts.try_into().ok()?;
        Some((index.parse().ok()?, partition, by))
    }

    /// A detector's value for one series' records of a bucket, and the
    /// mean location of `lat_long` (`None`: the records give no value).
    fn bucket_value(&self, d: &Detector, rs: &[&Rec]) -> Option<(f64, Option<(f64, f64)>)> {
        let weight = |r: &Rec| -> f64 { self.summary_count_field.as_ref().and_then(|s| r.fields.get(s)).and_then(num_of).unwrap_or(1.0) };
        let nums: Vec<(f64, f64)> = d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(num_of).map(|v| (v, weight(r)))).collect()).unwrap_or_default();
        let total_w: f64 = rs.iter().map(|r| weight(r)).sum();
        let mut point = None;
        let value = match d.base_function() {
            "lat_long" => {
                // `"lat,lon"` strings (as ES's datafeeds send geo points).
                let points: Vec<(f64, f64)> = d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(lat_lon_of)).collect()).unwrap_or_default();
                if points.is_empty() {
                    return None;
                }
                let n = points.len() as f64;
                point = Some((points.iter().map(|p| p.0).sum::<f64>() / n, points.iter().map(|p| p.1).sum::<f64>() / n));
                0.0
            }
            "count" | "non_zero_count" => total_w,
            "distinct_count" => d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(str_of)).collect::<BTreeSet<_>>().len() as f64).unwrap_or(0.0),
            "rare" | "freq_rare" => rs.len() as f64,
            "info_content" => d.field.as_ref().map(|f| info_content(&rs.iter().filter_map(|r| r.fields.get(f).and_then(str_of)).collect::<BTreeSet<_>>())).unwrap_or(0.0),
            "metric" | "mean" | "avg" => {
                let w: f64 = nums.iter().map(|(_, w)| w).sum();
                if w == 0.0 {
                    return None;
                }
                nums.iter().map(|(v, w)| v * w).sum::<f64>() / w
            }
            "median" => {
                if nums.is_empty() {
                    return None;
                }
                let mut v: Vec<f64> = nums.iter().map(|(v, _)| *v).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let m = v.len() / 2;
                if v.len() % 2 == 0 { (v[m - 1] + v[m]) / 2.0 } else { v[m] }
            }
            "min" => {
                if nums.is_empty() {
                    return None;
                }
                nums.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min)
            }
            "max" => {
                if nums.is_empty() {
                    return None;
                }
                nums.iter().map(|(v, _)| *v).fold(f64::NEG_INFINITY, f64::max)
            }
            "sum" | "non_null_sum" => {
                if nums.is_empty() && d.base_function() == "non_null_sum" {
                    return None;
                }
                // A summarized row's value is already its sum (the
                // `summary_count_field_name` count weights means, not
                // sums).
                nums.iter().map(|(v, _)| *v).sum()
            }
            "varp" => {
                if nums.is_empty() {
                    return None;
                }
                let n = nums.len() as f64;
                let mean = nums.iter().map(|(v, _)| *v).sum::<f64>() / n;
                nums.iter().map(|(v, _)| (v - mean).powi(2)).sum::<f64>() / n
            }
            // Periodic time: the circular mean of the bucket's times of
            // day / week (23:59 and 00:01 average to midnight, not noon).
            "time_of_day" => circular_mean(rs.iter().map(|r| (r.time.rem_euclid(86_400_000) / 1000) as f64), DAY_SECS),
            "time_of_week" => circular_mean(rs.iter().map(|r| (r.time.rem_euclid(604_800_000) / 1000) as f64), WEEK_SECS),
            _ => {
                if nums.is_empty() {
                    return None;
                }
                nums.iter().map(|(v, _)| *v).sum::<f64>() / nums.len() as f64
            }
        };
        Some((value, point))
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
            let Some((value, point)) = self.bucket_value(d, &rs) else { continue };
            let nums: Vec<f64> = d.field.as_ref().map(|f| rs.iter().filter_map(|r| r.fields.get(f).and_then(num_of)).collect()).unwrap_or_default();
            let mut influencers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            for inf in &self.influencers {
                for r in &rs {
                    if let Some(v) = r.fields.get(inf).and_then(str_of) {
                        influencers.entry(inf.clone()).or_default().insert(v);
                    }
                }
            }
            let extremes = (d.base_function() == "metric").then(|| {
                let lo = nums.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (lo, hi)
            });
            // Influence (#562): the series' value without each influencer
            // value's records (ES removes an influencer's data and assesses
            // what is left). Only models scored on the value use it.
            let mut without = BTreeMap::new();
            if !matches!(d.base_function(), "lat_long" | "rare" | "freq_rare" | "time_of_day" | "time_of_week") {
                for (field, vs) in &influencers {
                    for v in vs {
                        let rest: Vec<&Rec> = rs.iter().copied().filter(|r| r.fields.get(field).and_then(str_of).as_deref() != Some(v.as_str())).collect();
                        let value_without = if rest.is_empty() { None } else { self.bucket_value(d, &rest).map(|(x, _)| x) };
                        without.insert((field.clone(), v.clone()), value_without);
                    }
                }
            }
            out.push(SeriesValue { partition: p, by: b, over: o, value, point, extremes, influencers, without });
        }
        out
    }

    /// Is `sv`'s entity frequent within the detector's `exclude_frequent`
    /// scope (#555)? ml-cpp, per detector and partition: the person (by
    /// value of an individual detector, over value of a population one) is
    /// frequent when its share of the (decayed) buckets is above
    /// [`FREQUENT_FRACTION`]; a population's by value (attribute) when the
    /// share of the people seen that had it is. Before anything was
    /// counted the frequency is [`FREQUENCY_UNSEEN`].
    fn is_frequent(&self, d: &Detector, sv: &SeriesValue) -> bool {
        let scope = d.exclude_frequent.as_deref().unwrap_or("none");
        if scope == "none" {
            return false;
        }
        let base = format!("{}|{}", d.index, escape_key(sv.partition.as_deref().unwrap_or("")));
        let person_frequency = |person: &str| {
            let total = self.frequent_buckets.get(&base).copied().unwrap_or(0.0);
            if total <= 0.0 {
                return FREQUENCY_UNSEEN;
            }
            self.frequent_buckets.get(&format!("{base}|person|{}", escape_key(person))).copied().unwrap_or(0.0) / total
        };
        if d.over.is_some() {
            // Population: over = person (`E_XF_Over`), by = attribute (`E_XF_By`).
            let over = matches!(scope, "over" | "all") && sv.over.as_deref().is_some_and(|o| person_frequency(o) > FREQUENT_FRACTION);
            let by = matches!(scope, "by" | "all") && d.by.is_some() && sv.by.as_deref().is_some_and(|b| {
                let active = self.frequent_people.get(&format!("{base}|people")).map_or(0, BTreeSet::len);
                let frequency = if active == 0 {
                    FREQUENCY_UNSEEN
                } else {
                    self.frequent_people.get(&format!("{base}|attribute|{}", escape_key(b))).map_or(0, BTreeSet::len) as f64 / active as f64
                };
                frequency > FREQUENT_FRACTION
            });
            over || by
        } else {
            // Individual: the by value is the person (`E_XF_By`).
            matches!(scope, "by" | "all") && d.by.is_some() && sv.by.as_deref().is_some_and(|b| person_frequency(b) > FREQUENT_FRACTION)
        }
    }

    /// Count a finalized bucket's entities for [`Self::is_frequent`]
    /// (ml-cpp `CAnomalyDetectorModel::sampleBucketStatistics`: each person
    /// once per bucket, then every count decays).
    fn count_entities(&mut self, d: &Detector, values: &[SeriesValue]) {
        if d.exclude_frequent.as_deref().unwrap_or("none") == "none" {
            return;
        }
        let alpha = (-FREQUENCY_DECAY_RATE * (self.bucket_span_ms as f64 / 1000.0 / FREQUENCY_STANDARD_BUCKET_SECS).min(1.0)).exp();
        let population = d.over.is_some();
        let partitions: BTreeSet<String> = values.iter().map(|sv| sv.partition.clone().unwrap_or_default()).collect();
        for part in partitions {
            let base = format!("{}|{}", d.index, escape_key(&part));
            let mut people: BTreeSet<&str> = BTreeSet::new();
            for sv in values.iter().filter(|sv| sv.partition.as_deref().unwrap_or("") == part) {
                let person = if population { sv.over.as_deref() } else { sv.by.as_deref() };
                if let Some(person) = person {
                    people.insert(person);
                }
                if let (true, Some(o), Some(b)) = (population, &sv.over, &sv.by) {
                    let known = self.frequent_people.get(&format!("{base}|attribute|{}", escape_key(b))).is_some_and(|s| s.contains(o));
                    if known || self.admit(FREQUENCY_BYTES) {
                        self.frequent_people.entry(format!("{base}|people")).or_default().insert(o.clone());
                        self.frequent_people.entry(format!("{base}|attribute|{}", escape_key(b))).or_default().insert(o.clone());
                    }
                }
            }
            for person in people {
                let key = format!("{base}|person|{}", escape_key(person));
                if self.frequent_buckets.contains_key(&key) || self.admit(FREQUENCY_BYTES) {
                    *self.frequent_buckets.entry(key).or_insert(0.0) += 1.0;
                }
            }
            *self.frequent_buckets.entry(base.clone()).or_insert(0.0) += 1.0;
            // This partition's counts: `base` and the keys under `base|`
            // (values are escaped, so no other partition shares the prefix).
            if let Some(total) = self.frequent_buckets.get_mut(&base) {
                *total *= alpha;
            }
            let prefix = format!("{base}|");
            for (_, v) in self.frequent_buckets.range_mut(prefix.clone()..).take_while(|(k, _)| k.starts_with(&prefix)) {
                *v *= alpha;
            }
        }
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
        let scope = self.scope_key();
        let filters: BTreeMap<String, Vec<String>> = FILTERS.lock().get(&scope).cloned().unwrap_or_default();
        // Scheduled events of the job's calendars overlapping the bucket:
        // their results are skipped and / or the models not updated.
        let end = start + self.bucket_span_ms;
        let events: Vec<ScheduledEvent> = EVENTS.lock().get(&scope).map(|e| e.iter().filter(|e| e.start < end && e.end > start).cloned().collect()).unwrap_or_default();
        let event_skip_result = events.iter().any(|e| e.skip_result);
        let event_skip_update = events.iter().any(|e| e.skip_model_update);
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
            // Count and sum functions see 0 for every known series without
            // records (`non_null_sum` / `non_zero_count` skip them).
            if matches!(d.base_function(), "count" | "sum") && d.over.is_none() {
                let prefix = format!("{}|", d.index);
                let known: Vec<(String, String)> = self.models.keys().filter(|k| k.starts_with(&prefix)).filter_map(|k| Self::split_key(k)).map(|(_, p, b)| (p, b)).collect();
                for (p, b) in known {
                    let po = if d.partition.is_some() { Some(p) } else { None };
                    let bo = if d.by.is_some() { Some(b) } else { None };
                    if !values.iter().any(|v| v.partition == po && v.by == bo) {
                        values.push(SeriesValue { partition: po, by: bo, over: None, value: 0.0, point: None, extremes: None, influencers: BTreeMap::new(), without: BTreeMap::new() });
                    }
                }
                if d.base_function() == "count" && d.by.is_none() && d.partition.is_none() && values.is_empty() {
                    values.push(SeriesValue { partition: None, by: None, over: None, value: 0.0, point: None, extremes: None, influencers: BTreeMap::new(), without: BTreeMap::new() });
                }
            }
            let is_rare = matches!(d.base_function(), "rare" | "freq_rare");
            let is_geo = d.base_function() == "lat_long";
            let mut updates: Vec<(String, usize, Option<(f64, f64)>)> = Vec::new();
            let frequent = |sv: &SeriesValue| self.is_frequent(d, sv);
            let excluded: Vec<bool> = values.iter().map(frequent).collect();
            if !interim {
                self.count_entities(d, &values);
            }
            for (sv_i, sv) in values.iter().enumerate() {
                // A frequent entity (`exclude_frequent`) neither scores nor
                // trains the models.
                if excluded[sv_i] {
                    continue;
                }
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
                // `lat_long`: the bucket's mean location vs the series'
                // learned location (latitude / longitude models), a 2-d
                // Gaussian tail.
                let geo_point = if is_geo { sv.point } else { None };
                let geo_typical = geo_point.map(|_| self.geo.get(&key).map(|(la, lo)| (la.clone(), lo.clone())).unwrap_or_default());
                let (p, typical) = if let (Some((lat, lon)), Some((ml, mo))) = (geo_point, &geo_typical) {
                    let p = if ml.n < MIN_BUCKETS {
                        1.0
                    } else {
                        let z2 = ((lat - ml.mean) / ml.sigma(false).max(1e-3)).powi(2) + ((lon - mo.mean) / mo.sigma(false).max(1e-3)).powi(2);
                        (-z2 / 2.0).exp().max(1e-300)
                    };
                    (p, ml.mean)
                } else if let Some(period) = d.period() {
                    // Periodic models: the circular distance to the learned
                    // usual time, scaled by the circular spread.
                    let c = self.circular.get(&key).cloned().unwrap_or_default();
                    (c.probability(sv.value, period), c.mean(period).unwrap_or(sv.value))
                } else if is_rare && d.over.is_some() && d.base_function() == "rare" {
                    // Population rarity: how few members of the population
                    // ever had this by value, however often they had it.
                    let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                    let members = self.population.get(&fkey).map_or(0, BTreeSet::len) as f64;
                    let having = sv.by.as_ref().and_then(|b| self.prevalence.get(&fkey).and_then(|m| m.get(b))).map_or(0, BTreeSet::len) as f64;
                    let p = if members < 20.0 { 1.0 } else { ((having + 0.5) / (members + 1.0)).min(1.0) };
                    (p, having)
                } else if is_rare {
                    let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                    let total = *self.frequency_total.get(&fkey).unwrap_or(&0) as f64;
                    let seen = sv.by.as_ref().and_then(|b| self.frequency.get(&fkey).and_then(|m| m.get(b))).copied().unwrap_or(0) as f64;
                    let p = if total < 20.0 { 1.0 } else { ((seen + 0.5) / (total + 1.0)).min(1.0) };
                    // `freq_rare`: how often this entity interacts with the
                    // rare value counts, each interaction as unlikely as the
                    // value is rare.
                    let p = if d.base_function() == "freq_rare" && p < 1.0 { p.powf(sv.value.max(1.0)).max(1e-300) } else { p };
                    (p, seen)
                } else {
                    let m = self.models.get(&key).cloned().unwrap_or_default();
                    (m.probability(sv.value, d.side(), d.count_like()), if m.n > 0.0 { m.mean } else { sv.value })
                };
                // `metric`: mean, min and max are modeled apart; the most
                // anomalous of them is the series' result.
                let mut actual = sv.value;
                let mut feature = d.description();
                let (p, typical) = match sv.extremes {
                    Some((lo, hi)) => {
                        let (mmin, mmax) = self.extremes.get(&key).cloned().unwrap_or_default();
                        let typical_of = |m: &Series, x: f64| if m.n > 0.0 { m.mean } else { x };
                        let candidates = [
                            (p, typical, sv.value, "mean"),
                            (mmin.probability(lo, d.side(), false), typical_of(&mmin, lo), lo, "min"),
                            (mmax.probability(hi, d.side(), false), typical_of(&mmax, hi), hi, "max"),
                        ];
                        let best = candidates.into_iter().fold(candidates[0], |a, b| if b.0 < a.0 { b } else { a });
                        actual = best.2;
                        feature = best.3;
                        (best.0, best.1)
                    }
                    None => (p, typical),
                };
                let skip_result = event_skip_result || self.rule_applies(d, "skip_result", sv.value, typical, start, sv, &filters);
                let skip_update = event_skip_update || self.rule_applies(d, "skip_model_update", sv.value, typical, start, sv, &filters);
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
                    r.insert("function".into(), json!(if sv.extremes.is_some() { "metric" } else { d.description() }));
                    r.insert("function_description".into(), json!(feature));
                    match (geo_point, &geo_typical) {
                        (Some((lat, lon)), Some((ml, mo))) => {
                            r.insert("typical".into(), json!([ml.mean, mo.mean]));
                            r.insert("actual".into(), json!([lat, lon]));
                            r.insert("geo_results".into(), json!({"actual_point": format!("{lat},{lon}"), "typical_point": format!("{},{}", ml.mean, mo.mean)}));
                        }
                        _ => {
                            r.insert("typical".into(), json!([typical]));
                            r.insert("actual".into(), json!([actual]));
                        }
                    }
                    if let Some(f) = &d.field {
                        r.insert("field_name".into(), json!(f));
                    }
                    if let (Some(n), Some(v)) = (&d.over, &sv.over) {
                        r.insert("over_field_name".into(), json!(n));
                        r.insert("over_field_value".into(), json!(v));
                    }
                    // Each influencer value's share of the anomaly (#562):
                    // the series without its records, scored by the same
                    // model; one that changes nothing is no influencer.
                    let model = self.models.get(&key).cloned().unwrap_or_default();
                    let sums = matches!(d.base_function(), "sum" | "non_null_sum");
                    let influence = |field: &str, v: &str| -> f64 {
                        let Some(rest) = sv.without.get(&(field.to_string(), v.to_string())) else { return 1.0 };
                        let x = match rest {
                            Some(x) => *x,
                            None if d.count_like() || sums => 0.0,
                            None => return 1.0,
                        };
                        if p >= 1.0 || model.n == 0.0 {
                            return 1.0;
                        }
                        let p_without = model.probability(x, d.side(), d.count_like()).clamp(p, 1.0);
                        (1.0 - p_without.ln() / p.ln()).clamp(0.0, 1.0)
                    };
                    let mut influencing: BTreeMap<&String, Vec<(&String, f64)>> = BTreeMap::new();
                    for (k, vs) in &sv.influencers {
                        for v in vs {
                            // ml-cpp `doComputeInfluences`: a field's lone
                            // value has influence 1.
                            let share = if vs.len() == 1 { 1.0 } else { influence(k, v) };
                            if share >= MIN_INFLUENCE {
                                influencing.entry(k).or_default().push((v, share));
                            }
                        }
                    }
                    if !influencing.is_empty() {
                        let list: Vec<Value> = influencing.iter().map(|(k, vs)| json!({"influencer_field_name": k, "influencer_field_values": vs.iter().map(|(v, _)| v).collect::<Vec<_>>()})).collect();
                        r.insert("influencers".into(), Value::Array(list));
                    }
                    for (k, vs) in &influencing {
                        for (v, share) in vs {
                            let e = infl.entry((*k).clone()).or_default().entry((*v).clone()).or_insert((0.0, 1.0));
                            e.0 = e.0.max(score * share);
                            e.1 = e.1.min(p.powf(*share));
                        }
                    }
                    records.push(Value::Object(r));
                }
                if self.model_plot && !is_rare && !is_geo {
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
                    updates.push((key, sv_i, geo_point));
                }
            }
            // Training happens after every series of the bucket was scored
            // (#561): a population's entities share one model, which must
            // not move between two of them in the same bucket.
            // The partitions with a value the model learns from (#590): a
            // rare model's bucket total moves only with its values.
            let trained_parts: BTreeSet<String> = updates.iter().map(|(_, i, _)| values[*i].partition.clone().unwrap_or_default()).collect();
            for (key, sv_i, geo_point) in updates {
                let sv = &values[sv_i];
                {
                    // A new series / rare value is modeled only within the
                    // model memory limit (else refused, reported as
                    // `hard_limit`); known ones keep learning.
                    if is_rare && d.over.is_some() && d.base_function() == "rare" {
                        let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                        if let (Some(b), Some(o)) = (&sv.by, &sv.over) {
                            let known = self.prevalence.get(&fkey).and_then(|m| m.get(b)).is_some_and(|s| s.contains(o));
                            if known || self.admit(FREQUENCY_BYTES) {
                                self.population.entry(fkey.clone()).or_default().insert(o.clone());
                                self.prevalence.entry(fkey).or_default().entry(b.clone()).or_default().insert(o.clone());
                            }
                        }
                    } else if is_rare {
                        let fkey = format!("{}|{}", d.index, sv.partition.as_deref().unwrap_or(""));
                        if let Some(b) = &sv.by {
                            let known = self.frequency.get(&fkey).is_some_and(|m| m.contains_key(b));
                            if known || self.admit(FREQUENCY_BYTES) {
                                *self.frequency.entry(fkey).or_default().entry(b.clone()).or_insert(0) += 1;
                            }
                        }
                    } else if let Some((lat, lon)) = geo_point {
                        if self.geo.contains_key(&key) || self.admit(SERIES_BYTES) {
                            let (la, lo) = self.geo.entry(key).or_default();
                            la.update(lat);
                            lo.update(lon);
                        }
                    } else if let Some(period) = d.period() {
                        if self.circular.contains_key(&key) || self.admit(SERIES_BYTES) {
                            self.circular.entry(key).or_default().update(sv.value, period);
                        }
                    } else if self.models.contains_key(&key) || self.admit(SERIES_BYTES * if sv.extremes.is_some() { 3 } else { 1 }) {
                        if let Some((lo, hi)) = sv.extremes {
                            let (mmin, mmax) = self.extremes.entry(key.clone()).or_default();
                            mmin.update(lo);
                            mmax.update(hi);
                        }
                        self.models.entry(key).or_default().update(sv.value);
                    }
                }
            }
            if is_rare && !interim && !event_skip_update {
                let fkey = format!("{}|", d.index);
                for p in trained_parts {
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
        let mut bucket = json!({"job_id": job_id, "timestamp": start, "anomaly_score": max_score, "bucket_span": span_s, "initial_anomaly_score": max_score, "event_count": recs.len(),
            "is_interim": interim, "bucket_influencers": bucket_influencers, "processing_time_ms": elapsed as i64, "result_type": "bucket"});
        if !events.is_empty() {
            bucket["scheduled_events"] = json!(events.iter().map(|e| e.description.clone()).collect::<Vec<_>>());
        }
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
        self.size.model_bytes = self.model_bytes();
        self.size.peak_model_bytes = self.size.peak_model_bytes.max(self.size.model_bytes);
        self.size.timestamp = Some(start);
        self.size.total_category_count = self.categories.len() as u64;
        if self.timing.bucket_count % 10 == 1 {
            self.emit_model_size(out);
        }
        self.emit_categories(out);
    }

    /// The key of this job's entries in [`FILTERS`], [`EVENTS`], [`MML`].
    fn scope_key(&self) -> String {
        if self.scope.is_empty() { self.job_id.clone() } else { self.scope.clone() }
    }

    fn emit_model_size(&mut self, out: &mut Docs) {
        let now = crate::date::now_millis();
        let log_time = match self.size_log_time {
            Some(t) if t >= now => t + 1,
            _ => now,
        };
        self.size_log_time = Some(log_time);
        let mml = MML.lock().get(&self.scope_key()).copied().unwrap_or(1 << 30);
        out.push((format!("{}_model_size_stats_{log_time}", self.job_id), self.size.to_json(&self.job_id, mml, log_time)));
    }

    // -----------------------------------------------------------------------
    // Categorization
    // -----------------------------------------------------------------------

    /// The message's category id, or `None` when a new category would take
    /// the model past its memory limit (the message is then uncategorized).
    fn categorize(&mut self, msg: &str, partition: Option<String>) -> Option<i64> {
        // The configured analysis: categorization filters removed, then at
        // most `cat_token_limit` tokens (the analyzer's `limit` filter).
        let filtered;
        let mut msg = msg;
        if !self.cat_regex.is_empty() {
            let mut m = msg.to_string();
            for re in &self.cat_regex {
                m = re.replace_all(&m, "").into_owned();
            }
            filtered = m;
            msg = &filtered;
        }
        let toks = match &self.cat_analyzer {
            // A custom analyzer: its tokens (its own char filters, tokenizer
            // and token filters), on the first line with letters when it
            // asks for that ML char filter.
            Some(a) => {
                let first_line = self.cat_analyzer_def["char_filter"].as_array().into_iter().flatten().any(|c| c.as_str() == Some("first_line_with_letters"));
                let text = if first_line { msg.lines().find(|l| l.chars().any(char::is_alphabetic)).unwrap_or(msg) } else { msg };
                a.analyze(text).into_iter().map(|t| t.text).take(MAX_CATEGORIZATION_TOKENS).collect()
            }
            None => tokenize(msg, self.cat_token_limit.min(MAX_CATEGORIZATION_TOKENS)),
        };
        let mut best: Option<(usize, f64)> = None;
        for (i, c) in self.categories.iter().enumerate() {
            if c.partition != partition {
                continue;
            }
            let common = common_subsequence_len(&c.tokens, &toks);
            let denom = (c.tokens.len() + toks.len()) as f64;
            let sim = if denom == 0.0 { 1.0 } else { 2.0 * common as f64 / denom };
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
                Some(c.id)
            }
            None => {
                let examples = if self.cat_examples_limit > 0 { vec![msg.to_string()] } else { vec![] };
                let bytes = CATEGORY_BYTES + examples.iter().chain(&toks).chain(&toks).map(|x| x.len() as u64 + 24).sum::<u64>();
                if !self.admit(bytes) {
                    return None;
                }
                let id = self.categories.len() as i64 + 1;
                self.categories.push(Category { id, terms: toks.clone(), tokens: toks, examples, max_len: msg.len(), num_matches: 1, partition, dirty: true });
                Some(id)
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
            // ml-cpp `CAnomalyJob::skipTime`: the ceiling to a bucket
            // boundary (#588), so an aligned time resumes at itself.
            let target = -floor_to(-skip, self.bucket_span_ms);
            if self.current_bucket.map(|c| target > c).unwrap_or(true) {
                self.cur.retain(|r| r.time >= target);
                self.cur_bytes = self.cur.iter().map(rec_bytes).sum();
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
                // The interim results of the current bucket (records of
                // later buckets wait for theirs, with latency).
                let recs: Vec<Rec> = self.cur.iter().filter(|r| r.time < cur + self.bucket_span_ms).cloned().collect();
                if in_range && !recs.is_empty() {
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

    /// What a forecast needs from the model: the forecastable series
    /// (detector, partition, by, model).
    fn forecast_series(&self) -> Vec<(Detector, Option<String>, Option<String>, Series)> {
        let mut out = Vec::new();
        for d in self.detectors.iter().filter(|d| d.forecastable()) {
            let prefix = format!("{}|", d.index);
            for (key, m) in self.models.range(prefix.clone()..).take_while(|(k, _)| k.starts_with(&prefix)) {
                let Some((_, partition, by)) = Self::split_key(key) else { continue };
                out.push((d.clone(), (!partition.is_empty()).then_some(partition), (!by.is_empty()).then_some(by), m.clone()));
            }
        }
        out
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
fn tokenize(msg: &str, limit: usize) -> Vec<String> {
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
        .take(limit)
        .collect()
}

#[cfg(test)]
thread_local! {
    /// Comparison cells computed and the largest row workspace (tests).
    static LCS_WORK: std::cell::Cell<(u64, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// The last row of the longest-common-subsequence lengths of `a` and `b`
/// (`b.len() + 1` entries): linear space.
fn lcs_row<T: PartialEq>(a: &[T], b: &[T]) -> Vec<u32> {
    let mut prev = vec![0u32; b.len() + 1];
    let mut row = vec![0u32; b.len() + 1];
    for x in a {
        for (j, y) in b.iter().enumerate() {
            row[j + 1] = if x == y { prev[j] + 1 } else { prev[j + 1].max(row[j]) };
        }
        std::mem::swap(&mut prev, &mut row);
    }
    #[cfg(test)]
    LCS_WORK.with(|w| { let (cells, width) = w.get(); w.set((cells + (a.len() * b.len()) as u64, width.max(b.len() + 1))); });
    prev
}

/// Length of the longest common subsequence of `a` and `b`, in linear space.
fn common_subsequence_len(a: &[String], b: &[String]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };
    lcs_row(long, short)[short.len()] as usize
}

/// The ordered tokens common to `a` and `b` (a longest common
/// subsequence), in linear space (Hirschberg).
fn common_subsequence(a: &[String], b: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    hirschberg(a, b, &mut out);
    out
}

fn hirschberg(a: &[String], b: &[String], out: &mut Vec<String>) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    if a.len() == 1 {
        if b.contains(&a[0]) {
            out.push(a[0].clone());
        }
        return;
    }
    let mid = a.len() / 2;
    let left = lcs_row(&a[..mid], b);
    let rev_a: Vec<&String> = a[mid..].iter().rev().collect();
    let rev_b: Vec<&String> = b.iter().rev().collect();
    let right = lcs_row(&rev_a, &rev_b);
    let split = (0..=b.len()).max_by_key(|&j| (left[j] + right[b.len() - j], std::cmp::Reverse(j))).unwrap_or(0);
    hirschberg(&a[..mid], &b[..split], out);
    hirschberg(&a[mid..], &b[split..], out);
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

/// One job's place in the process registry: its process once built. The
/// first caller builds it (restoring the model snapshot) under `init`;
/// concurrent callers wait for that one process instead of building their
/// own, so a job never has two.
#[derive(Default)]
struct Slot {
    proc: OnceLock<Arc<Mutex<Proc>>>,
    init: Mutex<()>,
}

fn procs() -> &'static Mutex<HashMap<String, Arc<Slot>>> {
    static P: OnceLock<Mutex<HashMap<String, Arc<Slot>>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// Filter items by filter id, per job (for rule scopes).
static FILTERS: Mutex<BTreeMap<String, BTreeMap<String, Vec<String>>>> = Mutex::new(BTreeMap::new());
/// Model memory limit (bytes) per job id.
static MML: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// A scheduled event of one of a job's calendars.
#[derive(Clone, Debug)]
struct ScheduledEvent {
    start: i64,
    end: i64,
    skip_result: bool,
    skip_model_update: bool,
    description: String,
}

/// Scheduled events per job id (its calendars', directly or through its
/// groups), refreshed whenever calendars, events or the job change.
static EVENTS: Mutex<BTreeMap<String, Vec<ScheduledEvent>>> = Mutex::new(BTreeMap::new());

fn job_events(st: &super::store::State, job_id: &str) -> Vec<ScheduledEvent> {
    let mut names = vec![job_id.to_string()];
    if let Some(job) = st.jobs.get(job_id) {
        names.extend(job["groups"].as_array().into_iter().flatten().filter_map(|g| g.as_str().map(str::to_string)));
    }
    let calendars: Vec<&String> = st.calendars.iter().filter(|(_, c)| c["job_ids"].as_array().into_iter().flatten().filter_map(Value::as_str).any(|j| names.iter().any(|n| n == j))).map(|(id, _)| id).collect();
    st.events.iter().filter(|e| e["calendar_id"].as_str().is_some_and(|c| calendars.iter().any(|x| x.as_str() == c))).map(|e| ScheduledEvent {
        start: e["start_time"].as_i64().unwrap_or(0),
        end: e["end_time"].as_i64().unwrap_or(0),
        skip_result: e["skip_result"].as_bool().unwrap_or(true),
        skip_model_update: e["skip_model_update"].as_bool().unwrap_or(true),
        description: e["description"].as_str().unwrap_or("").to_string(),
    }).collect()
}

/// Refresh the scheduled events every job sees (calendar / event / job
/// changes); running jobs apply them from their next bucket on.
pub fn update_calendars(node: &Node) {
    let st = service(node).read();
    let mut events = EVENTS.lock();
    for id in st.jobs.keys() {
        events.insert(key(node, id), job_events(&st, id));
    }
}

fn key(node: &Node, job_id: &str) -> String {
    format!("{}#{job_id}", node.data.display())
}

/// The job's live process, if built.
fn live(node: &Node, job_id: &str) -> Option<Arc<Mutex<Proc>>> {
    procs().lock().get(&key(node, job_id)).and_then(|s| s.proc.get().cloned())
}

/// Is `p` still the job's registered process (not closed / replaced)?
fn is_current(node: &Node, job_id: &str, p: &Arc<Mutex<Proc>>) -> bool {
    live(node, job_id).is_some_and(|c| Arc::ptr_eq(&c, p))
}

fn not_open(job_id: &str) -> EsError {
    EsError::new(409, "status_exception", format!("Cannot perform requested action because job [{job_id}] is not open"))
}

/// Upgrade mode halts job processing (ES unassigns the job tasks): no
/// input, flush or forecast reaches a process meanwhile.
fn refuse_in_upgrade_mode(node: &Node, action: &str) -> Result<()> {
    if service(node).upgrade_mode() {
        return Err(EsError::new(400, "status_exception", format!("Cannot perform {action} action while upgrade mode is enabled")));
    }
    Ok(())
}

/// Is a process live for the job on this node?
pub fn is_live(node: &Node, job_id: &str) -> bool {
    live(node, job_id).is_some()
}

/// The model state of a snapshot: its state document's process, upgraded
/// to the current series key encoding.
fn parse_state(v: &Value) -> Option<Proc> {
    let mut p: Proc = serde_json::from_value(v.get(STATE_KEY)?.clone()).ok()?;
    p.init_fmt();
    p.migrate_series_keys();
    p.cur_bytes = p.cur.iter().map(rec_bytes).sum();
    Some(p)
}

impl Proc {
    /// States written before [`SERIES_KEY_VERSION`] keyed series by the
    /// unescaped `detector|partition|by`. A key is unambiguous when the
    /// detector has at most one of partition / by (or the values hold no
    /// `|`); an ambiguous one (both, with a `|` in a value) cannot be told
    /// apart from another series and is dropped (relearned).
    fn migrate_series_keys(&mut self) {
        if self.key_version >= SERIES_KEY_VERSION {
            return;
        }
        let legacy = std::mem::take(&mut self.models);
        for (k, m) in legacy {
            let Some((index, rest)) = k.split_once('|') else { continue };
            let Some(d) = index.parse::<usize>().ok().and_then(|i| self.detectors.iter().find(|d| d.index == i)) else { continue };
            let (partition, by) = match (d.partition.is_some(), d.by.is_some()) {
                (true, false) => (rest.strip_suffix('|').map(str::to_string), Some(String::new())),
                (false, true) => (Some(String::new()), rest.strip_prefix('|').map(str::to_string)),
                (false, false) => (Some(String::new()), (rest == "|").then(String::new)),
                (true, true) => match rest.split('|').collect::<Vec<_>>()[..] {
                    [p, b] => (Some(p.to_string()), Some(b.to_string())),
                    _ => {
                        tracing::warn!(job = %self.job_id, key = %k, "dropping an ambiguous pre-v2 series model (a value holds '|'); it is relearned");
                        continue;
                    }
                },
            };
            let (Some(partition), Some(by)) = (partition, by) else { continue };
            let d = d.clone();
            self.models.insert(Self::series_key(&d, &Some(partition), &Some(by)), m);
        }
        self.key_version = SERIES_KEY_VERSION;
    }
}

/// Can the snapshot's model state be loaded by this node?
pub fn snapshot_readable(node: &Node, job_id: &str, snapshot_id: &str) -> bool {
    ridx::get(node, ridx::STATE_PATTERN, &format!("{job_id}_model_state_{snapshot_id}#1")).and_then(|v| parse_state(&v)).is_some()
}

/// The state document `id` (`Ok(None)`: it does not exist); a failed read
/// is an error, never "no state".
fn read_state(node: &Node, id: &str) -> Result<Option<Value>> {
    #[cfg(test)]
    if let Some(e) = tests::injected(tests::Fault::StateRead) {
        return Err(e);
    }
    let resp = ridx::search(node, ridx::STATE_PATTERN, &json!({"size": 1, "query": {"ids": {"values": [id]}}}))?;
    Ok(resp["hits"]["hits"].as_array().and_then(|h| h.first()).map(|h| h["_source"].clone()))
}

fn build(node: &Node, job_id: &str) -> Result<Arc<Mutex<Proc>>> {
    let st = service(node).read();
    let job = st.jobs.get(job_id).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{job_id}'")))?;
    if st.job_state(job_id) != "opened" {
        return Err(not_open(job_id));
    }
    let filters: BTreeMap<String, Vec<String>> = st.filters.iter().map(|(k, f)| (k.clone(), f["items"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect())).collect();
    drop(st);
    // A configured snapshot must be restored: an unreadable, missing or
    // unparsable state fails the open (it is retried), never a cold start
    // that a later close would publish over the trained model.
    let restored = match job["model_snapshot_id"].as_str() {
        None => None,
        Some(sid) => {
            let state = read_state(node, &format!("{job_id}_model_state_{sid}#1"))?.ok_or_else(|| {
                EsError::new(500, "status_exception", format!("[{job_id}] cannot restore the model state of snapshot [{sid}]: its state document is missing"))
            })?;
            Some(parse_state(&state).ok_or_else(|| EsError::new(500, "status_exception", format!("[{job_id}] cannot restore the model state of snapshot [{sid}]: its state document cannot be parsed")))?)
        }
    };
    let scope = key(node, job_id);
    FILTERS.lock().insert(scope.clone(), filters);
    MML.lock().insert(scope.clone(), super::job::mml_bytes(&job));
    EVENTS.lock().insert(scope.clone(), job_events(&service(node).read(), job_id));
    let mut p = match restored {
        Some(mut p) => {
            p.refresh_config(&job);
            p
        }
        None => Proc::new(&job),
    };
    // Counts and timing continue from what the index holds.
    p.scope = scope;
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
/// live yet. Concurrent first callers share one build.
/// Jobs closed on this node (#595): no request admitted before the close
/// may build a replacement process; only an open lifts the fence.
static CLOSED: Mutex<std::collections::BTreeSet<String>> = Mutex::new(std::collections::BTreeSet::new());

fn get_or_build(node: &Node, job_id: &str) -> Result<Arc<Mutex<Proc>>> {
    let k = key(node, job_id);
    let slot = {
        let mut all = procs().lock();
        if CLOSED.lock().contains(&k) {
            return Err(not_open(job_id));
        }
        all.entry(k.clone()).or_default().clone()
    };
    if let Some(p) = slot.proc.get() {
        return Ok(p.clone());
    }
    let _init = slot.init.lock();
    if let Some(p) = slot.proc.get() {
        return Ok(p.clone());
    }
    #[cfg(test)]
    tests::building(&k);
    let built = match build(node, job_id) {
        Ok(p) => p,
        Err(e) => {
            let mut all = procs().lock();
            if all.get(&k).is_some_and(|s| Arc::ptr_eq(s, &slot) && s.proc.get().is_none()) {
                all.remove(&k);
            }
            return Err(e);
        }
    };
    let _ = slot.proc.set(built.clone());
    // Closed (the slot dropped) while building: this process is not the job's.
    if !procs().lock().get(&k).is_some_and(|s| Arc::ptr_eq(s, &slot)) {
        return Err(not_open(job_id));
    }
    Ok(built)
}

/// Launch the process of a job that was just opened.
pub fn open(node: &Node, job_id: &str) -> Result<()> {
    {
        let _all = procs().lock();
        CLOSED.lock().remove(&key(node, job_id));
    }
    get_or_build(node, job_id).map(|_| ())
}

/// The latest processed record time of a job's live process (its data
/// counts' `latest_record_timestamp`).
pub fn latest_record_time(node: &Node, job_id: &str) -> Option<i64> {
    live(node, job_id).and_then(|p| p.lock().counts.latest_record_timestamp)
}

/// Forget the process without persisting (revert / reset / failed open).
pub fn drop_process(node: &Node, job_id: &str) {
    procs().lock().remove(&key(node, job_id));
}

fn job_config(node: &Node, job_id: &str) -> Value {
    service(node).read().jobs.get(job_id).cloned().unwrap_or(Value::Null)
}

/// Write result documents in order (handling the interim-delete marker).
/// On failure, the documents not known to be written (the failed batch and
/// everything after it) come back, for the next write to retry: ids make
/// rewriting a document idempotent.
fn write_docs(node: &Node, job_id: &str, job: &Value, docs: Docs) -> std::result::Result<(), (EsError, Docs)> {
    let mut batch: Docs = Vec::new();
    let mut rest = docs.into_iter();
    let write = |batch: Docs| -> std::result::Result<(), (EsError, Docs)> {
        #[cfg(test)]
        if let Some(e) = tests::injected(tests::Fault::ResultsWrite) {
            return Err((e, batch));
        }
        ridx::write_results(node, job_id, job, batch.clone()).map_err(|e| (e, batch))
    };
    while let Some((id, doc)) = rest.next() {
        if id == "__delete_interim__" {
            if !batch.is_empty() {
                if let Err((e, mut failed)) = write(std::mem::take(&mut batch)) {
                    failed.push((id, doc));
                    failed.extend(rest);
                    return Err((e, failed));
                }
            }
            ridx::delete_by_query(node, &read_alias(job_id), json!({"term": {"is_interim": true}}));
        } else {
            batch.push((id, doc));
        }
    }
    write(batch)
}

/// Write the job's pending results and then `out` (under the process
/// lock, so results are written in order). On failure everything not
/// written stays pending and the error is returned.
fn commit(node: &Node, job_id: &str, job: &Value, g: &mut Proc, out: Docs) -> Result<()> {
    let mut docs = std::mem::take(&mut g.pending);
    g.pending_bytes = 0;
    docs.extend(out);
    match write_docs(node, job_id, job, docs) {
        Ok(()) => Ok(()),
        Err((e, unwritten)) => {
            g.set_pending(unwritten);
            Err(e)
        }
    }
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
///
/// The documents are accepted once ingested (the data counts include
/// them). Their results are written before this returns; when that write
/// fails they stay pending, are retried first by the next write, and
/// `_flush` / `_close` fail until they are written: the model never moves
/// past results it has not persisted, and retrying the input is not needed
/// (it would count twice).
pub fn post_docs(node: &Node, job_id: &str, docs: Vec<Value>, bytes: usize, epoch_ms: bool) -> Result<Value> {
    refuse_in_upgrade_mode(node, "cluster:admin/xpack/ml/job/data/post")?;
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let mut g = p.lock();
    if !is_current(node, job_id, &p) {
        return Err(not_open(job_id));
    }
    g.admit_input(&docs, bytes)?;
    let mut out = Vec::new();
    g.ingest(docs, bytes, epoch_ms, &mut out);
    // Category definitions are written as they are created, not per bucket.
    g.emit_categories(&mut out);
    out.extend(stats_docs(&g));
    let counts = g.counts.to_json(job_id, false);
    let had_pending = !g.pending.is_empty();
    match commit(node, job_id, &job, &mut g, out) {
        Ok(()) => maybe_persist(node, job_id, &job, &mut g),
        Err(e) => {
            drop(g);
            if !had_pending {
                ridx::audit(node, job_id, "anomaly_detector", "warning", &format!("Job results could not be persisted and are retained for retry: {}", e.reason));
            }
        }
    }
    Ok(counts)
}

/// `POST _flush`: fails while results cannot be written (they stay
/// pending).
pub fn flush(node: &Node, job_id: &str, params: &super::anomaly::FlushParams) -> Result<Value> {
    refuse_in_upgrade_mode(node, "cluster:admin/xpack/ml/job/flush")?;
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let mut g = p.lock();
    if !is_current(node, job_id, &p) {
        return Err(not_open(job_id));
    }
    let mut out = Vec::new();
    let resp = g.flush(params, &mut out);
    out.extend(stats_docs(&g));
    commit(node, job_id, &job, &mut g, out)?;
    maybe_persist(node, job_id, &job, &mut g);
    Ok(resp)
}

/// Bytes of one forecast result document (its admission estimate).
const FORECAST_DOC_BYTES: u64 = 600;
/// Default `max_model_memory` of a forecast (ES: 20mb).
pub const DEFAULT_FORECAST_MEMORY: u64 = 20 << 20;
/// Forecast documents written per batch at most.
const FORECAST_BATCH: usize = 10_000;

#[cfg(test)]
thread_local! {
    /// The largest forecast batch held (tests).
    static FORECAST_PEAK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `POST _forecast`. The series models are copied under the process lock;
/// the horizon is then generated and written in batches without it,
/// within `max_model_memory` (the copied models plus one batch): a
/// forecast whose models do not fit is refused before generating
/// anything. Closing the job stops a running forecast.
pub fn forecast(node: &Node, job_id: &str, duration_ms: i64, expires_in_ms: i64, max_model_memory: Option<u64>) -> Result<Value> {
    refuse_in_upgrade_mode(node, "cluster:admin/xpack/ml/job/forecast")?;
    let p = get_or_build(node, job_id)?;
    let job = job_config(node, job_id);
    let (series, start, span) = {
        let g = p.lock();
        if g.timing.bucket_count == 0 || g.counts.processed_record_count == 0 {
            let e = EsError::new(400, "status_exception", "Cannot run forecast: Forecast cannot be executed as job requires data to have been processed and modeled");
            return Err(e.clone().caused_by(e));
        }
        if g.size.memory_status == "hard_limit" {
            let e = EsError::new(400, "status_exception", "Cannot run forecast: Forecast cannot be executed as model memory status is not OK");
            return Err(e.clone().caused_by(e));
        }
        (g.forecast_series(), g.current_bucket.unwrap_or(0), g.bucket_span_ms)
    };
    let limit = max_model_memory.unwrap_or(DEFAULT_FORECAST_MEMORY);
    let fid = forecast_id();
    let span_s = span / 1000;
    let end = start + floor_to(duration_ms.max(span), span).max(span);
    let now = crate::date::now_millis() / 1000 * 1000;
    let stats = |n: u64, status: &str, messages: Vec<String>, memory: u64| -> (String, Value) {
        (format!("{job_id}_model_forecast_request_stats_{fid}"), json!({"job_id": job_id, "result_type": "model_forecast_request_stats", "forecast_id": fid, "processed_record_count": n,
            "forecast_messages": messages, "timestamp": start, "forecast_start_timestamp": start, "forecast_end_timestamp": end, "forecast_create_timestamp": now,
            "forecast_expiry_timestamp": now + expires_in_ms, "forecast_progress": if status == "finished" { 1.0 } else { 0.0 }, "processing_time_ms": 0, "forecast_memory_bytes": memory, "forecast_status": status}))
    };
    let models = MODEL_BASE_BYTES + SERIES_BYTES * series.len() as u64;
    if models + FORECAST_DOC_BYTES > limit {
        let message = format!("Forecast cannot be executed as forecast memory usage is predicted to exceed {}MB", limit >> 20);
        let _ = write_docs(node, job_id, &job, vec![stats(0, "failed", vec![message.clone()], models)]);
        let e = EsError::new(400, "status_exception", format!("Cannot run forecast: {message}"));
        return Err(e.clone().caused_by(e));
    }
    let batch_max = (((limit - models) / FORECAST_DOC_BYTES) as usize).clamp(1, FORECAST_BATCH);
    let mut batch: Docs = Vec::with_capacity(batch_max);
    let mut n = 0u64;
    let mut t = start;
    let flush_batch = |batch: &mut Docs| -> Result<()> {
        #[cfg(test)]
        FORECAST_PEAK.with(|peak| peak.set(peak.get().max(batch.len())));
        if !is_current(node, job_id, &p) {
            return Err(EsError::new(409, "status_exception", format!("Forecast [{fid}] of job [{job_id}] stopped: the job was closed")));
        }
        write_docs(node, job_id, &job, std::mem::take(batch)).map_err(|(e, _)| e)
    };
    while t < end {
        for (d, partition, by, m) in &series {
            let (lo, med, hi) = m.bounds(d.count_like());
            let mut doc = Map::new();
            doc.insert("job_id".into(), json!(job_id));
            doc.insert("forecast_id".into(), json!(fid));
            doc.insert("result_type".into(), json!("model_forecast"));
            doc.insert("bucket_span".into(), json!(span_s));
            doc.insert("detector_index".into(), json!(d.index));
            doc.insert("timestamp".into(), json!(t));
            if let (Some(pn), Some(pv)) = (&d.partition, partition) {
                doc.insert("partition_field_name".into(), json!(pn));
                doc.insert("partition_field_value".into(), json!(pv));
            }
            if let (Some(bn), Some(bv)) = (&d.by, by) {
                doc.insert("by_field_name".into(), json!(bn));
                doc.insert("by_field_value".into(), json!(bv));
            }
            doc.insert("model_feature".into(), json!(d.feature()));
            doc.insert("forecast_lower".into(), json!(lo));
            doc.insert("forecast_upper".into(), json!(hi));
            doc.insert("forecast_prediction".into(), json!(med));
            batch.push((format!("{job_id}_model_forecast_{fid}_{t}_{span_s}_{}_{}", d.index, values_to_id(&[by.as_deref(), None, partition.as_deref()])), Value::Object(doc)));
            n += 1;
            if batch.len() >= batch_max {
                if let Err(e) = flush_batch(&mut batch) {
                    let _ = write_docs(node, job_id, &job, vec![stats(n, "failed", vec![e.reason.clone()], models)]);
                    return Err(e);
                }
            }
        }
        t += span;
    }
    batch.push(stats(n, "finished", Vec::new(), models + FORECAST_DOC_BYTES * batch_max as u64));
    flush_batch(&mut batch)?;
    Ok(json!({"acknowledged": true, "forecast_id": fid}))
}

/// Close the job's process: persist stats and, when data was processed, a
/// model snapshot. Returns the new snapshot id.
///
/// The process stays the job's (and retryable) until everything is
/// durable: the model state first, then the pending and final results,
/// then the snapshot document that points at the state. Any failure
/// returns the error with the process still live and nothing lost; the
/// unwritten results stay pending for the retry.
pub fn close(node: &Node, job_id: &str) -> Result<Option<String>> {
    let k = key(node, job_id);
    let Some(p) = live(node, job_id) else { return Ok(None) };
    let job = job_config(node, job_id);
    let mut g = p.lock();
    if !is_current(node, job_id, &p) {
        return Ok(None);
    }
    let snapshot = persist(node, job_id, &job, &mut g, true)?;
    // Durable: the job no longer has a process, and none is built for it
    // until it is opened again (a request admitted before the close finds
    // it closed instead of starting a replacement).
    let mut all = procs().lock();
    if all.get(&k).is_some_and(|s| s.proc.get().is_some_and(|c| Arc::ptr_eq(c, &p))) {
        all.remove(&k);
    }
    CLOSED.lock().insert(k);
    Ok(snapshot)
}

/// Persist the process's model state, its pending and final results and
/// then the snapshot document (in that order: nothing points at state that
/// is not durable); the new snapshot id (`None`: nothing processed yet). On
/// failure the process is unchanged, its unwritten results pending.
/// `closing`: the job's close (its interim results go); else a background
/// persist of a running job.
fn persist(node: &Node, job_id: &str, job: &Value, g: &mut Proc, closing: bool) -> Result<Option<String>> {
    let mut out = Vec::new();
    if closing && g.has_interim {
        g.has_interim = false;
        out.push(("__delete_interim__".into(), Value::Null));
    }
    g.emit_categories(&mut out);
    g.emit_model_size(&mut out);
    out.extend(stats_docs(g));
    if g.counts.processed_record_count == 0 && g.timing.bucket_count == 0 {
        commit(node, job_id, job, g, out)?;
        return Ok(None);
    }
    let now = now_millis();
    let secs = now / 1000;
    let sid = secs.to_string();
    let mml = MML.lock().get(&key(node, job_id)).copied().unwrap_or(1 << 30);
    let when = crate::date::DateFormatter::new("strict_date_optional_time").map(|f| f.format(secs * 1000, crate::date::Tz::UTC)).unwrap_or_default();
    let when = match when.find('.') {
        Some(i) => format!("{}+0000", &when[..i]),
        None => when.trim_end_matches('Z').to_string() + "+0000",
    };
    let mut snap = Map::new();
    snap.insert("job_id".into(), json!(job_id));
    snap.insert("min_version".into(), json!(SNAPSHOT_MIN_VERSION));
    snap.insert("timestamp".into(), json!(secs * 1000));
    let description = if closing { format!("State persisted due to job close at {when}") } else { format!("Periodic persist at {when}") };
    snap.insert("description".into(), json!(description));
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
    let snapshot_doc = (super::results::snapshot_doc_id(job_id, &sid), Value::Object(snap));
    let state = serde_json::to_value(&*g).map_err(|e| EsError::new(500, "status_exception", format!("[{job_id}] cannot serialize the model state: {e}")))?;
    let state_docs = vec![
        (format!("{job_id}_model_state_{sid}#1"), json!({STATE_KEY: state})),
        (format!("{job_id}_quantiles"), json!({"job_id": job_id, "timestamp": g.latest_result_ts.unwrap_or(0), "quantile_state": ""})),
    ];
    #[cfg(test)]
    let state_written = match tests::injected(tests::Fault::StateWrite) { Some(e) => Err(e), None => ridx::write_state(node, state_docs) };
    #[cfg(not(test))]
    let state_written = ridx::write_state(node, state_docs);
    if let Err(e) = state_written {
        g.pending.extend(out);
        let pending = std::mem::take(&mut g.pending);
        g.set_pending(pending);
        return Err(e);
    }
    commit(node, job_id, job, g, out)?;
    write_docs(node, job_id, job, vec![snapshot_doc]).map_err(|(e, _)| e)?;
    g.persisted_at = Some(now);
    Ok(Some(sid))
}

/// The time (epoch millis) the job processes see (tests can move it).
fn now_millis() -> i64 {
    #[cfg(test)]
    if let Some(offset) = tests::clock_offset() {
        return crate::date::now_millis() + offset;
    }
    crate::date::now_millis()
}

/// `background_persist_interval` of `job` in millis; ES's default: between
/// three and four hours, fixed per job.
fn persist_interval_ms(job: &Value) -> i64 {
    if let Some(t) = job["background_persist_interval"].as_str().and_then(|s| super::job::parse_time(s, "background_persist_interval").ok()) {
        return t.millis().max(1000);
    }
    let id = job["job_id"].as_str().unwrap_or("");
    let jitter = id.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)) % 3_600_000;
    3 * 3_600_000 + jitter as i64
}

/// Persist the model of a running job once its background persist interval
/// passed since the last persist (or since its process started): a lost
/// process then loses at most that interval of learning. A failure leaves
/// the process as it was and is tried again at the next input or flush.
fn maybe_persist(node: &Node, job_id: &str, job: &Value, g: &mut Proc) {
    let now = now_millis();
    let Some(last) = g.persisted_at else {
        g.persisted_at = Some(now);
        return;
    };
    if now - last < persist_interval_ms(job) {
        return;
    }
    match persist(node, job_id, job, g, false) {
        Ok(Some(sid)) => {
            if let Err(e) = super::anomaly::record_snapshot(node, job_id, &sid) {
                tracing::warn!(job_id, "background model snapshot [{sid}] is persisted but not recorded on the job: {e}");
            }
        }
        Ok(None) => g.persisted_at = Some(now),
        Err(e) => tracing::warn!(job_id, "background model persist failed (retried later): {e}"),
    }
}

/// Live stats of an open job's process: (data counts, model size stats,
/// timing stats).
pub fn live_stats(node: &Node, job_id: &str) -> Option<(Value, Value, Value)> {
    let p = live(node, job_id)?;
    let g = p.lock();
    let mml = MML.lock().get(&key(node, job_id)).copied().unwrap_or(1 << 30);
    let now = crate::date::now_millis();
    Some((g.counts.to_json(job_id, true), g.size.to_json(job_id, mml, g.size_log_time.unwrap_or(now)), g.timing.to_json(job_id)))
}

/// Model bytes of every live process (`anomaly_detectors_in_bytes`).
pub fn live_model_bytes(node: &Node) -> u64 {
    let prefix = format!("{}#", node.data.display());
    let live: Vec<Arc<Mutex<Proc>>> = procs().lock().iter().filter(|(k, _)| k.starts_with(&prefix)).filter_map(|(_, s)| s.proc.get().cloned()).collect();
    live.iter().map(|p| p.lock().size.model_bytes).sum()
}

/// Remove every result and state document of a job (`_reset`, delete).
pub fn delete_job_data(node: &Node, job_id: &str, job: &Value) -> Result<()> {
    drop_process(node, job_id);
    FILTERS.lock().remove(&key(node, job_id));
    // `_id` takes no prefix query (ES): the model states are found through
    // the job's snapshot descriptors, read before the results go.
    let snapshots = ridx::search_all(node, &ridx::results_index(job), json!({"bool": {"filter": [{"term": {"job_id": job_id}}, {"exists": {"field": "snapshot_doc_count"}}]}}), json!(["_doc"])).unwrap_or_default();
    let mut state_ids: Vec<String> = vec![format!("{job_id}_quantiles")];
    for (_, d) in &snapshots {
        if let Some(sid) = d["snapshot_id"].as_str() {
            state_ids.extend((1..=16).map(|n| format!("{job_id}_model_state_{sid}#{n}")));
        }
    }
    ridx::try_delete_by_query(node, &ridx::results_index(job), json!({"term": {"job_id": job_id}}))?;
    ridx::try_delete_by_query(node, ridx::STATE_PATTERN, json!({"ids": {"values": state_ids}}))?;
    Ok(())
}

/// Update the filter items a job's rules see (`_ml/filters` changes).
pub fn update_filters(node: &Node) {
    let st = service(node).read();
    let filters: BTreeMap<String, Vec<String>> = st.filters.iter().map(|(k, f)| (k.clone(), f["items"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_string)).collect())).collect();
    // Only this node's jobs: another node in the process (a test's, or a
    // cluster engine's) keeps its own filters.
    let prefix = format!("{}#", node.data.display());
    let mut g = FILTERS.lock();
    for (_, f) in g.iter_mut().filter(|(k, _)| k.starts_with(&prefix)) {
        *f = filters.clone();
    }
}

/// Re-read a job's config into its live process (`_update`).
pub fn config_updated(node: &Node, job_id: &str) {
    if let Some(p) = live(node, job_id) {
        let job = job_config(node, job_id);
        p.lock().refresh_config(&job);
        MML.lock().insert(key(node, job_id), super::job::mml_bytes(&job));
    }
    update_calendars(node);
}

/// Test builds: forget this node's live processes, as a process restart
/// does (their in-memory state is gone).
#[cfg(test)]
pub(crate) fn forget_node(node: &Node) {
    let prefix = format!("{}#", node.data.display());
    procs().lock().retain(|k, _| !k.starts_with(&prefix));
}

/// `lat,lon` of a `lat_long` field value (`"40.7,-74.0"`), in range.
fn lat_lon_of(v: &Value) -> Option<(f64, f64)> {
    let s = v.as_str()?;
    let (lat, lon) = s.split_once(',')?;
    let (lat, lon): (f64, f64) = (lat.trim().parse().ok()?, lon.trim().parse().ok()?);
    ((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)).then_some((lat, lon))
}

/// Escape `\` and `|` of a series key part.
fn escape_key(v: &str) -> String {
    let mut o = String::with_capacity(v.len());
    for c in v.chars() {
        if c == '\\' || c == '|' {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

/// Estimated bytes of a held record.
fn rec_bytes(r: &Rec) -> u64 {
    64 + r.fields.iter().map(|(k, v)| 48 + k.len() as u64 + match v { Value::String(s) => s.len() as u64, _ => 16 }).sum::<u64>()
}

/// The `max_token_count` of a categorization analyzer's `limit` filter
/// (the default analyzer's: 100); an analyzer without one keeps
/// [`MAX_CATEGORIZATION_TOKENS`].
fn categorization_token_limit(analyzer: &Value) -> usize {
    if analyzer.is_null() {
        return 100;
    }
    analyzer["filter"].as_array().into_iter().flatten()
        .find(|f| f["type"].as_str() == Some("limit"))
        .and_then(|f| f["max_token_count"].as_u64().or_else(|| f["max_token_count"].as_str().and_then(|s| s.parse().ok())))
        .map(|n| n as usize)
        .unwrap_or(MAX_CATEGORIZATION_TOKENS)
        .clamp(1, MAX_CATEGORIZATION_TOKENS)
}

/// A categorization analyzer other than the ML tokenizers' (`ml_standard`,
/// `ml_classic`, or none), built as an ad hoc analysis chain; `None` uses
/// the built-in ML tokenization (with `pattern_replace` char filters and the
/// `limit` filter applied around it). The ML-only `first_line_with_letters`
/// char filter is applied by the caller.
fn custom_categorization_analyzer(def: &Value) -> Option<Arc<rustic_core::analysis::Analyzer>> {
    let tokenizer = def.get("tokenizer")?;
    if matches!(tokenizer.as_str(), Some("ml_standard" | "ml_classic")) {
        return None;
    }
    let list = |k: &str| -> Vec<Value> { def[k].as_array().cloned().unwrap_or_default() };
    let char_filters: Vec<Value> = list("char_filter").into_iter().filter(|c| c.as_str() != Some("first_line_with_letters")).collect();
    crate::analysis::AnalysisRegistry::default().adhoc(tokenizer, &list("filter"), &char_filters).ok().map(Arc::new)
}

/// The `categorization_filters` of a categorization analyzer (its
/// `pattern_replace` char filters).
fn categorization_filters(analyzer: &Value) -> Vec<String> {
    analyzer["char_filter"].as_array().into_iter().flatten()
        .filter(|f| f["type"].as_str() == Some("pattern_replace"))
        .filter_map(|f| f["pattern"].as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum Fault {
        ResultsWrite,
        StateWrite,
        StateRead,
    }

    thread_local! {
        /// Faults to inject on this thread: (fault, times left).
        static FAULTS: RefCell<Vec<(Fault, u32)>> = const { RefCell::new(Vec::new()) };
    }

    fn inject(fault: Fault, times: u32) {
        FAULTS.with(|f| f.borrow_mut().push((fault, times)));
    }

    thread_local! {
        static CLOCK_OFFSET: std::cell::Cell<Option<i64>> = const { std::cell::Cell::new(None) };
    }

    pub(crate) fn clock_offset() -> Option<i64> {
        CLOCK_OFFSET.with(|c| c.get())
    }

    fn advance_clock(ms: i64) {
        CLOCK_OFFSET.with(|c| c.set(Some(c.get().unwrap_or(0) + ms)));
    }

    pub(crate) fn injected(fault: Fault) -> Option<EsError> {
        FAULTS.with(|f| {
            let mut f = f.borrow_mut();
            let e = f.iter_mut().find(|(x, n)| *x == fault && *n > 0)?;
            e.1 -= 1;
            Some(EsError::new(429, "es_rejected_execution_exception", format!("injected {fault:?} failure")))
        })
    }

    /// Process builds per registry key, and keys whose build lingers.
    static BUILDS: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());
    static SLOW: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub(crate) fn building(k: &str) {
        {
            let mut b = BUILDS.lock();
            match b.iter_mut().find(|(x, _)| x == k) {
                Some(e) => e.1 += 1,
                None => b.push((k.to_string(), 1)),
            }
        }
        if SLOW.lock().iter().any(|x| x == k) {
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    }

    fn builds(k: &str) -> usize {
        BUILDS.lock().iter().find(|(x, _)| x == k).map_or(0, |e| e.1)
    }

    /// A bucket-aligned time (one-minute buckets).
    pub(super) const T0: i64 = 1_699_999_980_000;

    /// A node with job `id` (`analysis_config` / limits given), opened.
    pub(super) fn open_job(analysis: Value, limits: Option<Value>) -> (tempfile::TempDir, Arc<Node>) {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        put_and_open(&node, "j", analysis, limits);
        (data, node)
    }

    pub(super) fn put_and_open(node: &Arc<Node>, id: &str, analysis: Value, limits: Option<Value>) {
        let mut job = json!({"analysis_config": analysis, "data_description": {"time_field": "time", "time_format": "epoch_ms"}});
        if let Some(l) = limits {
            job["analysis_limits"] = l;
        }
        super::super::anomaly::put_job(node, id, &job, &job.to_string(), &Default::default()).unwrap();
        super::super::anomaly::open_job(node, id).unwrap();
    }

    pub(super) fn ndjson(docs: &[Value]) -> Vec<u8> {
        docs.iter().map(|d| d.to_string() + "\n").collect::<String>().into_bytes()
    }

    /// `buckets` one-minute buckets from `from`, `per` records each (user
    /// `u<i % users>`, value around 10).
    fn records(from: i64, buckets: i64, per: i64, users: i64) -> Vec<Value> {
        let mut out = Vec::new();
        for b in 0..buckets {
            for i in 0..per {
                out.push(json!({"time": T0 + (from + b) * 60_000 + i, "user": format!("u{}", i % users), "value": 10.0 + ((b * 7 + i) % 5) as f64}));
            }
        }
        out
    }

    fn model_of(node: &Node, id: &str) -> (Value, u64) {
        let p = live(node, id).expect("live process");
        let g = p.lock();
        (serde_json::to_value(&g.models).unwrap(), g.counts.processed_record_count)
    }

    pub(super) fn results(node: &Node, id: &str, result_type: &str) -> Vec<Value> {
        let r = ridx::search(node, &read_alias(id), &json!({"size": 10000, "query": {"term": {"result_type": result_type}}, "sort": [{"timestamp": "asc"}]})).unwrap();
        r["hits"]["hits"].as_array().cloned().unwrap_or_default().into_iter().map(|h| h["_source"].clone()).collect()
    }

    fn state_of(node: &Node, id: &str) -> String {
        service(node).read().job_state(id).to_string()
    }

    /// #206: a scheduled event of the job's calendar (added while the job
    /// runs) skips its bucket's results and model update; the same spike
    /// outside it is anomalous, and the baseline after it is unaffected.
    #[test]
    fn calendar_events_skip_results_and_learning() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
        super::super::calendar::put_calendar(&node, "maint", &json!({"job_ids": ["j"]})).unwrap();
        super::super::calendar::post_events(&node, "maint", &json!({"events": [{"description": "planned downtime", "start_time": T0 + 20 * 60_000, "end_time": T0 + 21 * 60_000}]})).unwrap();
        let mut docs = Vec::new();
        for b in 0..30i64 {
            let n = if b == 20 || b == 25 { 100 } else { 10 };
            docs.extend((0..n).map(|i| json!({"time": T0 + b * 60_000 + i})));
        }
        super::super::anomaly::post_data(&node, "j", &ndjson(&docs)).unwrap();
        super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(T0 + 30 * 60_000), ..Default::default() }).unwrap();
        let records: Vec<i64> = results(&node, "j", "record").iter().map(|r| r["timestamp"].as_i64().unwrap()).collect();
        assert!(records.contains(&(T0 + 25 * 60_000)), "the spike outside the event is anomalous: {records:?}");
        assert!(!records.contains(&(T0 + 20 * 60_000)), "the spike in the event is not reported");
        let event_bucket = results(&node, "j", "bucket").into_iter().find(|b| b["timestamp"] == json!(T0 + 20 * 60_000)).unwrap();
        assert_eq!(event_bucket["scheduled_events"], json!(["planned downtime"]));
        let p = live(&node, "j").unwrap();
        let m = p.lock().models["0||"].clone();
        // Learned without the event bucket: 30 buckets, one skipped, one spike.
        assert_eq!(m.n, 29.0);
        assert!((m.mean - (28.0 * 10.0 + 100.0) / 29.0).abs() < 1e-9, "{}", m.mean);
    }

    /// #212: reset removes every result of the job, however many (more
    /// than one 10,000-document page).
    #[test]
    fn reset_removes_every_result() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
        let job = service(&node).read().jobs["j"].clone();
        let docs: Vec<(String, Value)> = (0..10_001i64).map(|i| (format!("j_record_{i}"), json!({"job_id": "j", "result_type": "record", "timestamp": T0 + i, "record_score": 1.0}))).collect();
        ridx::write_results(&node, "j", &job, docs).unwrap();
        super::super::anomaly::close_job(&node, "j", false, false).unwrap();
        super::super::anomaly::reset_job(&node, "j").unwrap();
        let r = ridx::search(&node, &ridx::results_index(&job), &json!({"size": 0, "track_total_hits": true, "query": {"term": {"job_id": "j"}}})).unwrap();
        assert_eq!(ridx::total(&r), 0, "every result of the job is gone");
    }

    /// #213: a sum detector fed pre-aggregated rows (`sum` with
    /// `summary_count_field_name`) models the same sums as raw input, at
    /// any document count.
    #[test]
    fn preaggregated_sums_match_raw_sums() {
        let job = |summary: bool| {
            let mut ac = json!({"bucket_span": "60s", "detectors": [{"function": "sum", "field_name": "amount"}]});
            if summary {
                ac["summary_count_field_name"] = json!("doc_count");
            }
            Proc::new(&json!({"job_id": "s", "analysis_config": ac, "data_description": {"time_field": "time", "time_format": "epoch_ms"}}))
        };
        let (mut raw, mut summarized) = (job(false), job(true));
        let (mut out_raw, mut out_sum) = (Vec::new(), Vec::new());
        for b in 0..20i64 {
            let n = 1 + b % 4;
            let values: Vec<f64> = (0..n).map(|i| 30.0 / n as f64 + i as f64 - (n - 1) as f64 / 2.0).collect();
            raw.ingest(values.iter().map(|v| json!({"time": T0 + b * 60_000, "amount": v})).collect(), 10, true, &mut out_raw);
            summarized.ingest(vec![json!({"time": T0 + b * 60_000, "amount": values.iter().sum::<f64>(), "doc_count": n})], 10, true, &mut out_sum);
        }
        let (a, b) = (&raw.models["0||"], &summarized.models["0||"]);
        assert!((a.mean - 30.0).abs() < 1e-9 && (b.mean - a.mean).abs() < 1e-9 && (b.m2 - a.m2).abs() < 1e-6, "raw {a:?} vs summarized {b:?}");
    }

    /// #214: `metric` models min and max next to the mean: pairs (99, 101)
    /// then (0, 200) keep the mean but are anomalous for `metric`, and
    /// normal for a `mean` detector.
    #[test]
    fn metric_detects_changed_extremes() {
        for (function, anomalous) in [("metric", true), ("mean", false)] {
            let mut p = Proc::new(&json!({"job_id": "m", "analysis_config": {"bucket_span": "60s", "detectors": [{"function": function, "field_name": "v"}]}, "data_description": {"time_field": "time", "time_format": "epoch_ms"}}));
            let mut out = Vec::new();
            for b in 0..20i64 {
                p.ingest(vec![json!({"time": T0 + b * 60_000, "v": 99.0}), json!({"time": T0 + b * 60_000 + 1, "v": 101.0})], 10, true, &mut out);
            }
            p.ingest(vec![json!({"time": T0 + 20 * 60_000, "v": 0.0}), json!({"time": T0 + 20 * 60_000 + 1, "v": 200.0}), json!({"time": T0 + 21 * 60_000, "v": 100.0})], 10, true, &mut out);
            let records: Vec<&Value> = out.iter().map(|(_, d)| d).filter(|d| d["result_type"] == "record").collect();
            if anomalous {
                assert_eq!(records.len(), 1, "{records:?}");
                assert_eq!(records[0]["function"], json!("metric"));
                assert!(matches!(records[0]["function_description"].as_str(), Some("min" | "max")), "{}", records[0]);
                assert!(matches!(records[0]["actual"][0].as_f64(), Some(a) if a == 0.0 || a == 200.0));
            } else {
                assert!(records.is_empty(), "the mean did not change: {records:?}");
            }
        }
    }

    /// #215: an empty bucket is a zero sum for `low_sum` (an outage is
    /// anomalous) but nothing for `low_non_null_sum`.
    #[test]
    fn empty_buckets_are_zero_sums() {
        for (function, anomalous) in [("low_sum", true), ("low_non_null_sum", false)] {
            let mut p = Proc::new(&json!({"job_id": "z", "analysis_config": {"bucket_span": "60s", "detectors": [{"function": function, "field_name": "amount"}]}, "data_description": {"time_field": "time", "time_format": "epoch_ms"}}));
            let mut out = Vec::new();
            for b in 0..20i64 {
                p.ingest(vec![json!({"time": T0 + b * 60_000, "amount": 100.0 + (b % 3) as f64})], 10, true, &mut out);
            }
            // Bucket 20 has no records.
            p.ingest(vec![json!({"time": T0 + 21 * 60_000, "amount": 100.0}), json!({"time": T0 + 22 * 60_000, "amount": 100.0})], 10, true, &mut out);
            let records: Vec<&Value> = out.iter().map(|(_, d)| d).filter(|d| d["result_type"] == "record").collect();
            if anomalous {
                assert_eq!(records.len(), 1, "{records:?}");
                assert_eq!(records[0]["timestamp"], json!(T0 + 20 * 60_000));
                assert_eq!(records[0]["actual"], json!([0.0]));
            } else {
                assert!(records.is_empty(), "{records:?}");
            }
        }
    }

    /// #342: a running job persists its model every background persist
    /// interval (no close): a lost process reopens from that snapshot with
    /// the model and counts it had.
    #[test]
    fn running_jobs_persist_their_model_in_the_background() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let job = json!({"analysis_config": {"bucket_span": "60s", "detectors": [{"function": "mean", "field_name": "value", "by_field_name": "user"}]},
            "background_persist_interval": "1h", "data_description": {"time_field": "time", "time_format": "epoch_ms"}});
        super::super::anomaly::put_job(&node, "bp", &job, &job.to_string(), &Default::default()).unwrap();
        super::super::anomaly::open_job(&node, "bp").unwrap();
        super::super::anomaly::post_data(&node, "bp", &ndjson(&records(0, 20, 5, 3))).unwrap();
        assert!(service(&node).read().jobs["bp"].get("model_snapshot_id").is_none(), "not yet: within the interval");
        advance_clock(3_700_000);
        super::super::anomaly::post_data(&node, "bp", &ndjson(&records(20, 5, 5, 3))).unwrap();
        let sid = service(&node).read().jobs["bp"]["model_snapshot_id"].as_str().expect("a background snapshot").to_string();
        let before = model_of(&node, "bp");
        // The process is lost (no close): it reopens from the snapshot.
        forget_node(&node);
        super::super::anomaly::post_data(&node, "bp", &ndjson(&[])).unwrap_or_default();
        assert_eq!(model_of(&node, "bp"), before, "the model of snapshot {sid}");
        CLOCK_OFFSET.with(|c| c.set(None));
    }

    /// #351: a closed job reopens with its current bucket's accepted
    /// records: crossing the bucket afterwards gives the uninterrupted
    /// run's bucket and model.
    #[test]
    fn close_keeps_the_current_buckets_records() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let analysis = json!({"bucket_span": "60s", "detectors": [{"function": "count"}]});
        put_and_open(&node, "c1", analysis.clone(), None);
        put_and_open(&node, "c2", analysis, None);
        let first: Vec<Value> = (0..10).map(|i| json!({"time": T0 + i})).collect();
        let next = vec![json!({"time": T0 + 60_000})];
        for id in ["c1", "c2"] {
            super::super::anomaly::post_data(&node, id, &ndjson(&first)).unwrap();
        }
        super::super::anomaly::close_job(&node, "c1", false, false).unwrap();
        super::super::anomaly::open_job(&node, "c1").unwrap();
        for id in ["c1", "c2"] {
            super::super::anomaly::post_data(&node, id, &ndjson(&next)).unwrap();
        }
        let bucket = |id: &str| results(&node, id, "bucket").into_iter().find(|b| b["timestamp"] == json!(T0)).map(|b| b["event_count"].clone());
        assert_eq!(bucket("c1"), Some(json!(10)), "the closed job's bucket kept its records");
        assert_eq!(bucket("c1"), bucket("c2"));
        assert_eq!(model_of(&node, "c1").0, model_of(&node, "c2").0);
    }

    /// #348: `freq_rare` scores how often an entity hits a rare value: many
    /// interactions are more anomalous than one; `rare` is unchanged.
    #[test]
    fn freq_rare_counts_interactions() {
        let score = |function: &str, hits: usize| -> f64 {
            let mut p = Proc::new(&json!({"job_id": "f", "analysis_config": {"bucket_span": "60s", "detectors": [{"function": function, "by_field_name": "uri", "over_field_name": "client"}]},
                "data_description": {"time_field": "time", "time_format": "epoch_ms"}}));
            let mut out = Vec::new();
            for b in 0..40i64 {
                let docs: Vec<Value> = (0..20).map(|i| json!({"time": T0 + b * 60_000 + i, "uri": format!("/common{}", i % 2), "client": format!("c{i}")})).collect();
                p.ingest(docs, 10, true, &mut out);
            }
            out.clear();
            let mut docs: Vec<Value> = (0..20).map(|i| json!({"time": T0 + 40 * 60_000 + i, "uri": format!("/common{}", i % 2), "client": format!("c{i}")})).collect();
            docs.extend((0..hits).map(|i| json!({"time": T0 + 40 * 60_000 + 30 + i as i64, "uri": "/rare", "client": "attacker"})));
            docs.push(json!({"time": T0 + 41 * 60_000}));
            p.ingest(docs, 10, true, &mut out);
            out.iter().map(|(_, d)| d).filter(|d| d["result_type"] == "record" && d["over_field_value"] == "attacker").map(|d| d["record_score"].as_f64().unwrap()).fold(0.0, f64::max)
        };
        let (one, many) = (score("freq_rare", 1), score("freq_rare", 50));
        assert!(many > one, "freq_rare: 50 interactions {many} vs 1 {one}");
        assert_eq!(score("rare", 1), score("rare", 50), "rare is about the value, not the count");
    }

    /// #349: time of day is periodic: training at 23:59 / 00:01 learns
    /// midnight; a noon event is anomalous, a midnight one is not; the same
    /// holds for the week boundary.
    #[test]
    fn time_of_day_is_circular() {
        for (function, period) in [("time_of_day", 86_400_000i64), ("time_of_week", 604_800_000)] {
            let run = |at: i64| -> usize {
                let mut p = Proc::new(&json!({"job_id": "t", "analysis_config": {"bucket_span": "1h", "detectors": [{"function": function}]},
                    "data_description": {"time_field": "time", "time_format": "epoch_ms"}}));
                let mut out = Vec::new();
                // Bucket starts aligned to the period boundary.
                let base = (T0 / period + 1) * period;
                for k in 0..30i64 {
                    let t = if k % 2 == 0 { base + k * period - 60_000 } else { base + k * period + 60_000 };
                    p.ingest(vec![json!({"time": t})], 10, true, &mut out);
                }
                out.clear();
                let t = base + 40 * period + at;
                p.ingest(vec![json!({"time": t}), json!({"time": t + 600_000}), json!({"time": t + 2 * 3_600_000})], 10, true, &mut out);
                out.iter().filter(|(_, d)| d["result_type"] == "record").count()
            };
            assert_eq!(run(period / 2), 1, "{function}: the half-period event is anomalous");
            assert_eq!(run(0), 0, "{function}: the usual time is not");
        }
    }

    /// #350 / #354: retention counts back from the job's latest bucket; an
    /// expired forecast goes with all its points.
    #[test]
    fn retention_and_forecast_expiry() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "user"}]}), None);
        super::super::anomaly::post_data(&node, "j", &ndjson(&records(0, 20, 5, 2))).unwrap();
        super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(T0 + 20 * 60_000), ..Default::default() }).unwrap();
        service(&node).mutate(|st| { st.jobs.get_mut("j").unwrap()["results_retention_days"] = json!(1); Ok(()) }).unwrap();
        // Historical buckets (2023) are far older than now minus a day,
        // but within a day of the latest bucket: kept.
        let buckets = results(&node, "j", "bucket").len();
        super::super::anomaly::forecast(&node, "j", 600_000, 1, None).unwrap();
        assert!(!results(&node, "j", "model_forecast").is_empty());
        std::thread::sleep(std::time::Duration::from_millis(10));
        super::super::anomaly::delete_expired_data(&node, Some("j")).unwrap();
        assert_eq!(results(&node, "j", "bucket").len(), buckets, "kept: within the retention of the latest bucket");
        assert!(results(&node, "j", "model_forecast_request_stats").is_empty());
        assert!(results(&node, "j", "model_forecast").is_empty(), "the expired forecast's points went too");
    }

    /// #144: a close whose state or results write fails fails, keeps the
    /// job open with its live model; once storage recovers the close
    /// succeeds and the reopened job continues the exact model and counts.
    #[test]
    fn close_fails_without_losing_the_model() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "mean", "field_name": "value", "by_field_name": "user"}]}), None);
        super::super::anomaly::post_data(&node, "j", &ndjson(&records(0, 30, 6, 3))).unwrap();
        let before = model_of(&node, "j");
        assert!(before.1 == 180 && before.0.as_object().unwrap().len() == 3, "{before:?}");
        for fault in [Fault::StateWrite, Fault::ResultsWrite] {
            inject(fault, 1);
            let e = super::super::anomaly::close_job(&node, "j", false, false).unwrap_err();
            assert!(e.reason.contains("injected"), "{fault:?}: {e:?}");
            assert_eq!(state_of(&node, "j"), "opened", "{fault:?}: the failed close left the job open");
            assert_eq!(model_of(&node, "j"), before, "{fault:?}: the live model is kept");
        }
        super::super::anomaly::close_job(&node, "j", false, false).unwrap();
        assert_eq!(state_of(&node, "j"), "closed");
        assert!(!is_live(&node, "j"));
        super::super::anomaly::open_job(&node, "j").unwrap();
        assert_eq!(model_of(&node, "j"), before, "the reopened job continues the model");
    }

    /// #145: a results write failing at a bucket boundary keeps the
    /// bucket's results pending: `_flush` fails until they are written,
    /// and then every bucket is written exactly once with the model and
    /// counts of an uninterrupted run.
    #[test]
    fn failed_results_write_keeps_results_until_written() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let analysis = json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "user"}]});
        put_and_open(&node, "a", analysis.clone(), None);
        put_and_open(&node, "b", analysis, None);
        let input: Vec<Vec<Value>> = (0..10).map(|i| records(i * 2, 2, 4, 2)).collect();
        for (i, chunk) in input.iter().enumerate() {
            if i == 4 {
                inject(Fault::ResultsWrite, 1);
            }
            super::super::anomaly::post_data(&node, "a", &ndjson(chunk)).unwrap();
            if i == 4 {
                assert!(!live(&node, "a").unwrap().lock().pending.is_empty(), "the failed bucket's results are pending");
                inject(Fault::ResultsWrite, 1);
                assert!(super::super::anomaly::flush_job(&node, "a", &Default::default()).is_err(), "a flush cannot claim unwritten results");
            }
            super::super::anomaly::post_data(&node, "b", &ndjson(chunk)).unwrap();
        }
        for id in ["a", "b"] {
            super::super::anomaly::flush_job(&node, id, &super::super::anomaly::FlushParams { advance_time: Some(T0 + 40 * 60_000), ..Default::default() }).unwrap();
        }
        assert!(live(&node, "a").unwrap().lock().pending.is_empty());
        let (a, b) = (results(&node, "a", "bucket"), results(&node, "b", "bucket"));
        let ts = |r: &[Value]| r.iter().map(|d| (d["timestamp"].as_i64().unwrap(), d["event_count"].as_i64().unwrap())).collect::<Vec<_>>();
        assert_eq!(ts(&a), ts(&b), "every bucket exactly once, as uninterrupted");
        let mut seen = std::collections::BTreeSet::new();
        assert!(ts(&a).iter().all(|(t, _)| seen.insert(*t)));
        assert_eq!(model_of(&node, "a"), model_of(&node, "b"));
    }

    /// #146: past the model memory limit new series are not modeled, the
    /// stats report `hard_limit` with the refusals, retained growth stays
    /// bounded, and input that would not fit is refused before any of it
    /// is applied. Opening counts the memory of the jobs already open.
    #[test]
    fn model_memory_limit_bounds_growth() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "user"}]}), Some(json!({"model_memory_limit": "1mb"})));
        for b in 0..8 {
            super::super::anomaly::post_data(&node, "j", &ndjson(&records(b, 1, 400, 400 * (b + 1)).into_iter().map(|mut r| { r["user"] = json!(format!("b{b}-{}", r["user"].as_str().unwrap())); r }).collect::<Vec<_>>())).unwrap();
        }
        let p = live(&node, "j").unwrap();
        let (models, used, size) = { let g = p.lock(); (g.models.len(), g.used_bytes(), g.size.clone()) };
        assert!(models > 100 && (models as u64) * SERIES_BYTES <= 1 << 20, "{models} series");
        assert!(used <= 2 << 20, "retained {used} bytes: the model and in-flight input within their limits");
        assert!(p.lock().model_bytes() <= 1 << 20);
        assert_eq!(size.memory_status, "hard_limit");
        assert!(size.bucket_allocation_failures > 0 && size.model_bytes_exceeded > 0, "{size:?}");
        let (_, stats, _) = live_stats(&node, "j").unwrap();
        assert_eq!(stats["memory_status"], json!("hard_limit"));
        // Input past the limit: refused, nothing applied.
        let count = model_of(&node, "j").1;
        let big: Vec<Value> = (0..20_000).map(|i| json!({"time": T0 + 9 * 60_000, "user": format!("x{i}"), "padding": "y".repeat(40)})).collect();
        let e = super::super::anomaly::post_data(&node, "j", &ndjson(&big)).unwrap_err();
        assert_eq!(e.status, 429);
        assert_eq!(model_of(&node, "j").1, count);
        // Node capacity is shared by the open jobs.
        node.cluster_settings.write().1.insert("xpack.ml.max_machine_memory_percent".into(), json!("1"));
        let max = super::super::anomaly::ml_max_memory(&node);
        let mb = (max * 6 / 10 >> 20).saturating_sub(30).max(1);
        let job = json!({"analysis_config": {"bucket_span": "60s", "detectors": [{"function": "count"}]}, "analysis_limits": {"model_memory_limit": format!("{mb}mb")}, "data_description": {"time_field": "time"}});
        for id in ["k1", "k2"] {
            super::super::anomaly::put_job(&node, id, &job, &job.to_string(), &Default::default()).unwrap();
        }
        super::super::anomaly::open_job(&node, "k1").unwrap();
        let e = super::super::anomaly::open_job(&node, "k2").unwrap_err();
        assert_eq!(e.status, 429, "{e:?}");
    }

    /// #147: categorization keeps at most the analyzer's token limit and
    /// compares in linear space; filters apply; the linear-space common
    /// subsequence matches the quadratic one.
    #[test]
    fn categorization_is_bounded() {
        let job = json!({"job_id": "c", "analysis_config": {"bucket_span": "60s", "categorization_field_name": "msg", "detectors": [{"function": "count", "by_field_name": "mlcategory"}],
            "categorization_analyzer": super::super::job::standard_categorization_analyzer(&["secret-\\d+".to_string()])}, "data_description": {"time_field": "time"}});
        let mut p = Proc::new(&job);
        assert_eq!(p.cat_token_limit, 100);
        let long: String = (0..20_000).map(|i| format!("word{} ", i % 97)).collect();
        LCS_WORK.with(|w| w.set((0, 0)));
        let a = p.categorize(&long, None).unwrap();
        let b = p.categorize(&long, None).unwrap();
        assert_eq!(a, b);
        assert_eq!(p.categories[0].tokens.len(), 100);
        let (cells, width) = LCS_WORK.with(|w| w.get());
        assert!(width <= 101 && cells <= 4 * 100 * 100, "work {cells} cells, rows of {width}");
        p.categorize("token secret-123 here", None);
        assert!(p.categories.iter().all(|c| !c.tokens.iter().any(|t| t.contains("secret"))), "categorization filters applied");
        let mut custom = job.clone();
        custom["analysis_config"]["categorization_analyzer"] = json!({"tokenizer": "ml_standard", "filter": [{"type": "limit", "max_token_count": 5}]});
        let mut q = Proc::new(&custom);
        q.categorize(&long, None);
        assert_eq!(q.categories[0].tokens.len(), 5);
        custom["analysis_config"]["categorization_analyzer"] = json!({"tokenizer": "whitespace"});
        let mut r = Proc::new(&custom);
        r.categorize(&long, None);
        assert_eq!(r.categories[0].tokens.len(), MAX_CATEGORIZATION_TOKENS);
        // Linear space = the quadratic DP's length, on random sequences.
        let mut x = 7u64;
        let mut next = move || { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); x >> 33 };
        for _ in 0..200 {
            let (n1, n2) = ((next() % 12) as usize, (next() % 12) as usize);
            let s1: Vec<String> = (0..n1).map(|_| format!("t{}", next() % 4)).collect();
            let s2: Vec<String> = (0..n2).map(|_| format!("t{}", next() % 4)).collect();
            let mut dp = vec![vec![0usize; s2.len() + 1]; s1.len() + 1];
            for i in 0..s1.len() { for j in 0..s2.len() { dp[i + 1][j + 1] = if s1[i] == s2[j] { dp[i][j] + 1 } else { dp[i][j + 1].max(dp[i + 1][j]) }; } }
            let lcs = common_subsequence(&s1, &s2);
            assert_eq!(lcs.len(), dp[s1.len()][s2.len()]);
            assert_eq!(common_subsequence_len(&s1, &s2), lcs.len());
            let mut it = s1.iter();
            assert!(lcs.iter().all(|t| it.any(|x| x == t)), "a subsequence of the first");
            let mut it = s2.iter();
            assert!(lcs.iter().all(|t| it.any(|x| x == t)), "a subsequence of the second");
        }
    }

    /// #148: a forecast whose models do not fit `max_model_memory` is
    /// refused before generating anything; one that fits is written in
    /// batches no larger than the limit allows.
    #[test]
    fn forecast_honours_max_model_memory() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "user"}]}), None);
        super::super::anomaly::post_data(&node, "j", &ndjson(&records(0, 20, 10, 10))).unwrap();
        FORECAST_PEAK.with(|p| p.set(0));
        let e = super::super::anomaly::forecast(&node, "j", 7_200_000, 86_400_000, Some(1024)).unwrap_err();
        assert!(e.reason.starts_with("Cannot run forecast: Forecast cannot be executed as forecast memory usage is predicted to exceed"), "{e:?}");
        assert_eq!(FORECAST_PEAK.with(|p| p.get()), 0, "nothing generated");
        assert!(results(&node, "j", "model_forecast").is_empty());
        let limit = MODEL_BASE_BYTES + SERIES_BYTES * 10 + FORECAST_DOC_BYTES * 100;
        super::super::anomaly::forecast(&node, "j", 7_200_000, 86_400_000, Some(limit)).unwrap();
        assert!(FORECAST_PEAK.with(|p| p.get()) <= 101, "batches of at most the limit");
        assert_eq!(results(&node, "j", "model_forecast").len(), 120 * 10, "every forecast bucket of every series");
    }

    /// #149: partition `a|b` by `c` and partition `a` by `b|c` are two
    /// series (models, attribution, forecasts), also after a state round
    /// trip; old unescaped keys migrate when unambiguous.
    #[test]
    fn series_keys_do_not_collide() {
        let job = json!({"job_id": "k", "analysis_config": {"bucket_span": "60s", "detectors": [{"function": "mean", "field_name": "value", "partition_field_name": "p", "by_field_name": "b"}]},
            "data_description": {"time_field": "time", "time_format": "epoch_ms"}});
        let mut p = Proc::new(&job);
        let mut out = Vec::new();
        for i in 0..30 {
            p.ingest(vec![json!({"time": T0 + i * 60_000, "p": "a|b", "b": "c", "value": 10.0}), json!({"time": T0 + i * 60_000 + 1, "p": "a", "b": "b|c", "value": 1000.0})], 100, true, &mut out);
        }
        assert_eq!(p.models.len(), 2, "{:?}", p.models.keys().collect::<Vec<_>>());
        let mut means: Vec<(Option<String>, Option<String>, f64)> = p.forecast_series().into_iter().map(|(_, pa, b, m)| (pa, b, m.mean)).collect();
        means.sort_by(|x, y| x.0.cmp(&y.0));
        assert_eq!(means, vec![(Some("a".into()), Some("b|c".into()), 1000.0), (Some("a|b".into()), Some("c".into()), 10.0)]);
        let restored = parse_state(&json!({STATE_KEY: serde_json::to_value(&p).unwrap()})).unwrap();
        assert_eq!(serde_json::to_value(&restored.models).unwrap(), serde_json::to_value(&p.models).unwrap());
        // Pre-v2 keys.
        let mut legacy = serde_json::to_value(&p).unwrap();
        legacy["key_version"] = json!(0);
        legacy["models"] = json!({"0|x|y": legacy["models"]["0|a\\|b|c"].clone(), "0|a|b|c": legacy["models"]["0|a\\|b|c"].clone()});
        let migrated = parse_state(&json!({STATE_KEY: legacy})).unwrap();
        assert_eq!(migrated.models.keys().cloned().collect::<Vec<_>>(), vec!["0|x|y".to_string()], "the unambiguous key kept, the ambiguous one dropped");
        assert_eq!(Proc::split_key("0|a\\|b|c"), Some((0, "a|b".into(), "c".into())));
    }

    /// #150: `lat_long` learns a location and scores a distant one.
    #[test]
    fn lat_long_models_locations() {
        let job = json!({"job_id": "g", "analysis_config": {"bucket_span": "60s", "detectors": [{"function": "lat_long", "field_name": "loc"}]}, "data_description": {"time_field": "time", "time_format": "epoch_ms"}});
        let mut p = Proc::new(&job);
        let mut out = Vec::new();
        for i in 0..30 {
            let jitter = (i % 5) as f64 * 0.01;
            p.ingest(vec![json!({"time": T0 + i * 60_000, "loc": format!("{},{}", 40.7 + jitter, -74.0 - jitter)})], 30, true, &mut out);
        }
        assert!(!out.iter().any(|(_, d)| d["result_type"] == "record"), "normal locations are not anomalous");
        p.ingest(vec![json!({"time": T0 + 30 * 60_000, "loc": "51.5,-0.1"}), json!({"time": T0 + 31 * 60_000, "loc": "40.7,-74.0"})], 30, true, &mut out);
        let records: Vec<&Value> = out.iter().map(|(_, d)| d).filter(|d| d["result_type"] == "record").collect();
        assert_eq!(records.len(), 1, "{records:?}");
        let r = records[0];
        assert_eq!(r["function"], json!("lat_long"));
        assert_eq!(r["actual"], json!([51.5, -0.1]));
        assert!((r["typical"][0].as_f64().unwrap() - 40.72).abs() < 0.05 && r["record_score"].as_f64().unwrap() > 50.0, "{r}");
        assert_eq!(r["geo_results"]["actual_point"], json!("51.5,-0.1"));
    }

    /// #151: an unreadable or unparsable state of the job's snapshot fails
    /// the open (no cold process); once it reads again the job resumes the
    /// trained model and counts.
    #[test]
    fn unreadable_snapshot_state_fails_the_open() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "mean", "field_name": "value", "by_field_name": "user"}]}), None);
        super::super::anomaly::post_data(&node, "j", &ndjson(&records(0, 20, 6, 3))).unwrap();
        let before = model_of(&node, "j");
        super::super::anomaly::close_job(&node, "j", false, false).unwrap();
        inject(Fault::StateRead, 1);
        let e = super::super::anomaly::open_job(&node, "j").unwrap_err();
        assert!(e.reason.contains("injected"), "{e:?}");
        assert_eq!(state_of(&node, "j"), "closed");
        assert!(!is_live(&node, "j"));
        let sid = service(&node).read().jobs["j"]["model_snapshot_id"].as_str().unwrap().to_string();
        let id = format!("j_model_state_{sid}#1");
        let good = read_state(&node, &id).unwrap().unwrap();
        ridx::write_state(&node, vec![(id.clone(), json!({STATE_KEY: "not a model"}))]).unwrap();
        let e = super::super::anomaly::open_job(&node, "j").unwrap_err();
        assert!(e.reason.contains("cannot be parsed"), "{e:?}");
        assert!(!is_live(&node, "j"));
        ridx::write_state(&node, vec![(id, good)]).unwrap();
        super::super::anomaly::open_job(&node, "j").unwrap();
        assert_eq!(model_of(&node, "j"), before);
    }

    /// #154: with a 120s latency an observation from the preceding minute
    /// joins its bucket; one older than the window is out of order; with
    /// no latency (control) the preceding-minute one is out of order too.
    #[test]
    fn latency_keeps_late_observations() {
        for latency in [Some("120s"), None] {
            let mut ac = json!({"bucket_span": "60s", "detectors": [{"function": "count"}]});
            if let Some(l) = latency {
                ac["latency"] = json!(l);
            }
            let mut p = Proc::new(&json!({"job_id": "l", "analysis_config": ac, "data_description": {"time_field": "time", "time_format": "epoch_ms"}}));
            let mut out = Vec::new();
            p.ingest(vec![json!({"time": T0 + 60_000 + 5})], 10, true, &mut out);
            p.ingest(vec![json!({"time": T0 + 30_000})], 10, true, &mut out);
            p.ingest(vec![json!({"time": T0 + 10 * 60_000})], 10, true, &mut out);
            p.ingest(vec![json!({"time": T0 + 5 * 60_000})], 10, true, &mut out);
            let bucket0 = out.iter().find(|(_, d)| d["result_type"] == "bucket" && d["timestamp"] == json!(T0)).map(|(_, d)| d["event_count"].as_i64().unwrap());
            match latency {
                Some(_) => {
                    assert_eq!(bucket0, Some(1), "the late observation joined its bucket");
                    assert_eq!(p.counts.out_of_order_timestamp_count, 1, "only the one past the window");
                }
                None => {
                    assert_eq!(bucket0, None, "no bucket before the first record's");
                    assert_eq!(p.counts.out_of_order_timestamp_count, 2);
                }
            }
        }
    }

    /// #155: two first requests after a restart share one restored
    /// process; both acknowledged inputs survive flush, close and reopen.
    #[test]
    fn concurrent_first_requests_share_one_process() {
        let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "user"}]}), None);
        super::super::anomaly::post_data(&node, "j", &ndjson(&records(0, 5, 4, 2))).unwrap();
        super::super::anomaly::close_job(&node, "j", false, false).unwrap();
        super::super::anomaly::open_job(&node, "j").unwrap();
        let base = model_of(&node, "j").1;
        forget_node(&node);
        let k = key(&node, "j");
        SLOW.lock().push(k.clone());
        let builds_before = builds(&k);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|s| {
            for user in ["x", "y"] {
                let (node, barrier) = (&node, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    let docs: Vec<Value> = (0..3).map(|i| json!({"time": T0 + 6 * 60_000 + i, "user": user})).collect();
                    super::super::anomaly::post_data(node, "j", &ndjson(&docs)).unwrap();
                });
            }
        });
        SLOW.lock().retain(|x| *x != k);
        assert_eq!(builds(&k) - builds_before, 1, "one restored process for both first requests");
        assert_eq!(model_of(&node, "j").1, base + 6, "both acknowledged inputs are in the one process");
        super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(T0 + 8 * 60_000), ..Default::default() }).unwrap();
        super::super::anomaly::close_job(&node, "j", false, false).unwrap();
        super::super::anomaly::open_job(&node, "j").unwrap();
        assert_eq!(model_of(&node, "j").1, base + 6);
    }

    #[test]
    fn tokens_and_terms() {
        assert_eq!(tokenize("Node 1 started successfully", 100), vec!["Node", "started", "successfully"]);
        assert_eq!(tokenize("Disk usage at 91% on /dev/sda1", 100), vec!["Disk", "usage", "at", "on", "/dev/sda1"]);
        assert_eq!(tokenize("Failed to connect to db1.example.com: timeout", 100), vec!["Failed", "to", "connect", "to", "db1.example.com", "timeout"]);
        let a = tokenize("User alice logged in from 10.0.0.1", 100);
        let b = tokenize("User bob logged in from 10.0.0.2", 100);
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

#[cfg(test)]
#[path = "process_b7_tests.rs"]
mod b7_tests;
