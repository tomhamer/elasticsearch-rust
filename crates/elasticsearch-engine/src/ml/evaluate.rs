//! `POST _ml/data_frame/_evaluate`: outlier_detection, classification and
//! regression evaluation metrics computed over a search of the index.
//!
//! A port of `org.elasticsearch.xpack.core.ml.dataframe.evaluation`: every
//! metric contributes aggregations to a search over the documents that
//! hold all the fields it needs (`size: 0`, the request `query` as a
//! filter), then reads its result from the aggregation response. Multi-step
//! metrics (confusion matrix, per-class precision / recall) run further
//! searches until every metric has a result. The ROC curve is built from
//! the 1st..99th percentiles of the predicted probability of the positive
//! and negative documents, exactly as ES does.

use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};

const ACTION: &str = "[cluster:monitor/xpack/ml/data_frame/evaluate]";
const OTHER_BUCKET_KEY: &str = "_other_";
const MAX_CLASSES_CARDINALITY: usize = 1000;
const DEFAULT_TOP_CLASSES_FIELD: &str = "ml.top_classes";

// ---------------------------------------------------------------------------
// Errors.

/// ES `ExceptionsHelper.badRequestException`.
fn bad_request(msg: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", msg)
}

/// Wrap an error raised while parsing the `evaluation` object.
fn in_evaluation(e: EsError) -> EsError {
    EsError::x_content_parse(format!("{ACTION} failed to parse field [evaluation]")).caused_by(e)
}

/// Wrap an error raised while parsing metric `metric` of evaluation `eval`.
fn in_metric(eval: &str, metric: &str, e: EsError) -> EsError {
    in_evaluation(
        EsError::x_content_parse(format!("[{eval}] failed to parse field [metrics]"))
            .caused_by(EsError::x_content_parse(format!("[metrics] failed to parse field [{metric}]")).caused_by(e)),
    )
}

/// `ConstructingObjectParser` build failure of `name`: when every
/// constructor argument was given the object is built as the last of them
/// arrives (`[name] failed to parse field [last]` wraps it), otherwise at
/// the end of the object.
fn build_failure(name: &str, obj: &Map<String, Value>, ctor_args: &[&str], cause: EsError) -> EsError {
    let inner = EsError::x_content_parse(format!("Failed to build [{name}] after last required field arrived")).caused_by(cause);
    if ctor_args.iter().all(|a| obj.contains_key(*a)) {
        if let Some(last) = obj.keys().filter(|k| ctor_args.contains(&k.as_str())).last() {
            return EsError::x_content_parse(format!("[{name}] failed to parse field [{last}]")).caused_by(inner);
        }
    }
    inner
}

// ---------------------------------------------------------------------------
// Number formatting.

/// A double as ES renders it in a response (`NaN` / `Infinity` as strings).
fn dbl(v: f64) -> Value {
    if v.is_nan() {
        Value::String("NaN".into())
    } else if v.is_infinite() {
        Value::String(if v > 0.0 { "Infinity".into() } else { "-Infinity".into() })
    } else {
        crate::values::java_double(v)
    }
}

/// `java.text.MessageFormat` rendering of a double (`#,##0.###`, half-even).
fn message_format_number(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "∞".into() } else { "-∞".into() };
    }
    let s = format!("{:.3}", v.abs());
    let (int, frac) = s.split_once('.').unwrap_or((&s, ""));
    let frac = frac.trim_end_matches('0');
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    let neg = v < 0.0 && (grouped != "0" || !frac.is_empty());
    format!("{}{}{}{}", if neg { "-" } else { "" }, grouped, if frac.is_empty() { "" } else { "." }, frac)
}

/// `Terms.Bucket#getKeyAsString`.
fn key_as_string(b: &Value) -> String {
    if let Some(s) = b.get("key_as_string").and_then(Value::as_str) {
        return s.to_string();
    }
    match &b["key"] {
        Value::String(s) => s.clone(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => crate::script::value::java_double_string(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::Bool(x) => x.to_string(),
        other => other.to_string(),
    }
}

fn doc_count(v: &Value) -> u64 {
    v.get("doc_count").and_then(Value::as_u64).unwrap_or(0)
}

/// A single-value metric aggregation's value (`null` reads as NaN).
fn single_value(v: &Value) -> Option<f64> {
    let v = v.get("value")?;
    Some(match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => s.parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    })
}

// ---------------------------------------------------------------------------
// Request model.

#[derive(Clone, Copy, PartialEq, Eq)]
enum EvalKind {
    OutlierDetection,
    Classification,
    Regression,
}

impl EvalKind {
    fn name(self) -> &'static str {
        match self {
            EvalKind::OutlierDetection => "outlier_detection",
            EvalKind::Classification => "classification",
            EvalKind::Regression => "regression",
        }
    }
}

/// `EvaluationFields`.
#[derive(Default)]
struct Fields {
    actual: String,
    predicted: Option<String>,
    top_classes: Option<String>,
    predicted_class: Option<String>,
    predicted_probability: Option<String>,
    probability_nested: bool,
}

impl Fields {
    /// `listPotentiallyRequiredFields`: (parameter name, field).
    fn potentially_required(&self) -> Vec<(&'static str, Option<&str>)> {
        vec![
            ("actual_field", Some(self.actual.as_str())),
            ("predicted_field", self.predicted.as_deref()),
            ("top_classes_field", self.top_classes.as_deref()),
            ("predicted_class_field", self.predicted_class.as_deref()),
            ("predicted_probability_field", self.predicted_probability.as_deref()),
        ]
    }
}

/// `MulticlassConfusionMatrix` state.
struct Mccm {
    size: usize,
    prefix: String,
    top_actual: Option<Vec<String>>,
    cardinality: Option<u64>,
    processed: usize,
    /// Rows of the batches processed so far.
    pending: Vec<MccmRow>,
    /// (rows, other_actual_class_count)
    result: Option<(Vec<MccmRow>, u64)>,
}

#[derive(Clone)]
struct MccmRow {
    actual: String,
    actual_count: u64,
    predicted: Vec<(String, u64)>,
    other_predicted: u64,
}

impl Mccm {
    fn new(size: usize, prefix: String) -> Mccm {
        Mccm { size, prefix, top_actual: None, cardinality: None, processed: 0, pending: Vec::new(), result: None }
    }

    fn rows_parsed(&self) -> Vec<MccmRow> {
        self.result.as_ref().map(|r| r.0.clone()).unwrap_or_default()
    }

    fn aggs(&self, f: &Fields, max_buckets: usize, out: &mut Map<String, Value>) {
        let actual = &f.actual;
        let predicted = f.predicted.as_deref().unwrap_or_default();
        let p = &self.prefix;
        if self.top_actual.is_none() && self.cardinality.is_none() {
            out.insert(format!("{p}_step_1_by_actual_class"), json!({"terms": {"field": actual, "order": [{"_count": "desc"}, {"_key": "asc"}], "size": self.size}}));
            out.insert(format!("{p}_step_1_cardinality_of_actual_class"), json!({"cardinality": {"field": actual}}));
            return;
        }
        if self.result.is_some() {
            return;
        }
        let names = self.top_actual.clone().unwrap_or_default();
        let per_batch = (max_buckets / (names.len() + 2)).max(1);
        let end = (self.processed + per_batch).min(names.len());
        let lenient_match = |field: &str, c: &str| json!({"match": {field: {"query": c, "lenient": true}}});
        let mut actual_filters = Map::new();
        for c in &names[self.processed.min(end)..end] {
            actual_filters.insert(c.clone(), lenient_match(actual, c));
        }
        let mut predicted_filters = Map::new();
        for c in &names {
            predicted_filters.insert(c.clone(), lenient_match(predicted, c));
        }
        out.insert(
            format!("{p}_step_2_by_actual_class"),
            json!({
                "filters": {"filters": actual_filters},
                "aggs": {format!("{p}_step_2_by_predicted_class"): {"filters": {"filters": predicted_filters, "other_bucket": true, "other_bucket_key": OTHER_BUCKET_KEY}}}
            }),
        );
    }

    fn process(&mut self, aggs: &Value) {
        let p = self.prefix.clone();
        if self.top_actual.is_none() {
            if let Some(t) = aggs.get(format!("{p}_step_1_by_actual_class")) {
                let mut names: Vec<String> = t["buckets"].as_array().map(|b| b.iter().map(key_as_string).collect()).unwrap_or_default();
                names.sort();
                self.top_actual = Some(names);
            }
        }
        if self.cardinality.is_none() {
            if let Some(c) = aggs.get(format!("{p}_step_1_cardinality_of_actual_class")) {
                self.cardinality = Some(c["value"].as_u64().unwrap_or(0));
            }
        }
        if self.result.is_none() && self.top_actual.as_ref().is_some_and(Vec::is_empty) {
            // No actual classes: nothing to break down.
            self.result = Some((Vec::new(), self.cardinality.unwrap_or(0).saturating_sub(self.size as u64)));
            return;
        }
        if self.result.is_none() {
            if let Some(fa) = aggs.get(format!("{p}_step_2_by_actual_class")) {
                let buckets = bucket_list(&fa["buckets"]);
                let names = self.top_actual.clone().unwrap_or_default();
                for (actual, b) in &buckets {
                    let sub = &b[format!("{p}_step_2_by_predicted_class")]["buckets"];
                    let mut predicted = Vec::new();
                    let mut other = 0;
                    for (k, sb) in bucket_list(sub) {
                        if k == OTHER_BUCKET_KEY {
                            other = doc_count(sb);
                        } else {
                            predicted.push((k.clone(), doc_count(sb)));
                        }
                    }
                    predicted.sort_by(|a, b| a.0.cmp(&b.0));
                    self.pending.push(MccmRow { actual: actual.clone(), actual_count: doc_count(b), predicted, other_predicted: other });
                }
                self.processed += buckets.len();
                if self.processed >= names.len() {
                    let other = self.cardinality.unwrap_or(0).saturating_sub(self.size as u64);
                    let rows = std::mem::take(&mut self.pending);
                    self.result = Some((rows, other));
                }
            }
        }
    }

    fn result_json(&self) -> Option<Value> {
        let (rows, other) = self.result.as_ref()?;
        Some(json!({"confusion_matrix": rows.iter().map(mccm_row_json).collect::<Vec<_>>(), "other_actual_class_count": other}))
    }
}

fn mccm_row_json(r: &MccmRow) -> Value {
    json!({
        "actual_class": r.actual,
        "actual_class_doc_count": r.actual_count,
        "predicted_classes": r.predicted.iter().map(|(c, n)| json!({"predicted_class": c, "count": n})).collect::<Vec<_>>(),
        "other_predicted_class_doc_count": r.other_predicted,
    })
}

/// The buckets of a (keyed or array) bucket aggregation, in response order.
fn bucket_list(v: &Value) -> Vec<(String, &Value)> {
    match v {
        Value::Object(m) => m.iter().map(|(k, b)| (k.clone(), b)).collect(),
        Value::Array(a) => a.iter().map(|b| (key_as_string(b), b)).collect(),
        _ => Vec::new(),
    }
}

/// One evaluation metric with its computation state.
enum Metric {
    OutlierAucRoc { include_curve: bool, result: Option<Value> },
    OutlierPrecision { at: Vec<f64>, result: Option<Value> },
    OutlierRecall { at: Vec<f64>, result: Option<Value> },
    OutlierConfusionMatrix { at: Vec<f64>, result: Option<Value> },
    Accuracy { matrix: Mccm, result: Option<Value> },
    MulticlassConfusionMatrix { matrix: Mccm },
    ClassPrecision { top: Option<Vec<String>>, result: Option<Value> },
    ClassRecall { top: Option<Vec<String>>, result: Option<Value> },
    ClassAucRoc { class_name: String, include_curve: bool, result: Option<Value> },
    Mse { result: Option<Value> },
    Msle { offset: f64, result: Option<Value> },
    Huber { delta: f64, result: Option<Value> },
    RSquared { result: Option<Value> },
}

impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Metric::OutlierAucRoc { .. } | Metric::ClassAucRoc { .. } => "auc_roc",
            Metric::OutlierPrecision { .. } | Metric::ClassPrecision { .. } => "precision",
            Metric::OutlierRecall { .. } | Metric::ClassRecall { .. } => "recall",
            Metric::OutlierConfusionMatrix { .. } => "confusion_matrix",
            Metric::Accuracy { .. } => "accuracy",
            Metric::MulticlassConfusionMatrix { .. } => "multiclass_confusion_matrix",
            Metric::Mse { .. } => "mse",
            Metric::Msle { .. } => "msle",
            Metric::Huber { .. } => "huber",
            Metric::RSquared { .. } => "r_squared",
        }
    }

    /// `getRequiredFields`: the evaluation parameters the metric reads.
    fn required(&self) -> &'static [&'static str] {
        match self {
            Metric::OutlierAucRoc { .. } | Metric::OutlierPrecision { .. } | Metric::OutlierRecall { .. } | Metric::OutlierConfusionMatrix { .. } => {
                &["actual_field", "predicted_probability_field"]
            }
            Metric::ClassAucRoc { .. } => &["actual_field", "predicted_class_field", "predicted_probability_field"],
            _ => &["actual_field", "predicted_field"],
        }
    }

    fn result(&self) -> Option<Value> {
        match self {
            Metric::OutlierAucRoc { result, .. }
            | Metric::OutlierPrecision { result, .. }
            | Metric::OutlierRecall { result, .. }
            | Metric::OutlierConfusionMatrix { result, .. }
            | Metric::Accuracy { result, .. }
            | Metric::ClassPrecision { result, .. }
            | Metric::ClassRecall { result, .. }
            | Metric::ClassAucRoc { result, .. }
            | Metric::Mse { result }
            | Metric::Msle { result, .. }
            | Metric::Huber { result, .. }
            | Metric::RSquared { result } => result.clone(),
            Metric::MulticlassConfusionMatrix { matrix } => matrix.result_json(),
        }
    }
}

struct Ctx {
    /// `OutlierDetection.actualIsTrueQuery` for the actual field.
    actual_is_true: Value,
    max_buckets: usize,
}

const PERCENTS: std::ops::RangeInclusive<u32> = 1..=99;

fn percentiles_agg(field: &str) -> Value {
    let percents: Vec<f64> = PERCENTS.map(f64::from).collect();
    json!({"percentiles": {"field": field, "percents": percents}})
}

fn filter_count(field: &str, query: Value) -> Value {
    let _ = field;
    json!({"filter": query})
}

fn painless(source: String) -> Value {
    json!({"source": source, "lang": "painless"})
}

fn diff_squared_script(f: &Fields) -> Value {
    painless(format!(
        "def diff = doc['{}'].value - doc['{}'].value;return diff * diff;",
        f.actual,
        f.predicted.as_deref().unwrap_or_default()
    ))
}

impl Metric {
    fn aggs(&self, i: usize, f: &Fields, ctx: &Ctx, out: &mut Map<String, Value>) {
        let p = format!("m{i}_{}", self.name());
        let prob = f.predicted_probability.as_deref().unwrap_or_default();
        let truth = &ctx.actual_is_true;
        let not_truth = json!({"bool": {"must_not": [truth.clone()]}});
        let conds = |at: &[f64], which: &[(&str, bool, bool)], out: &mut Map<String, Value>| {
            for (ti, t) in at.iter().enumerate() {
                let range = json!({"range": {prob: {"gte": t}}});
                for (cname, actual, predicted) in which {
                    let mut must = Vec::new();
                    let mut must_not = Vec::new();
                    if *actual { must.push(truth.clone()) } else { must_not.push(truth.clone()) }
                    if *predicted { must.push(range.clone()) } else { must_not.push(range.clone()) }
                    out.insert(format!("{p}_at_{ti}_{cname}"), filter_count(prob, json!({"bool": {"must": must, "must_not": must_not}})));
                }
            }
        };
        match self {
            Metric::OutlierAucRoc { result: None, .. } => {
                out.insert(format!("{p}_true"), json!({"filter": truth, "aggs": {"percentiles": percentiles_agg(prob)}}));
                out.insert(format!("{p}_non_true"), json!({"filter": not_truth, "aggs": {"percentiles": percentiles_agg(prob)}}));
            }
            Metric::OutlierPrecision { at, result: None } => conds(at, &[("TP", true, true), ("FP", false, true)], out),
            Metric::OutlierRecall { at, result: None } => conds(at, &[("TP", true, true), ("FN", true, false)], out),
            Metric::OutlierConfusionMatrix { at, result: None } => {
                conds(at, &[("TP", true, true), ("FP", false, true), ("TN", false, false), ("FN", true, false)], out)
            }
            Metric::Accuracy { matrix, result: None } => matrix.aggs(f, ctx.max_buckets, out),
            Metric::MulticlassConfusionMatrix { matrix } => matrix.aggs(f, ctx.max_buckets, out),
            Metric::ClassPrecision { top, result: None } | Metric::ClassRecall { top, result: None } => {
                let actual = &f.actual;
                let predicted = f.predicted.as_deref().unwrap_or_default();
                match top {
                    None => {
                        out.insert(
                            format!("{p}_by_actual_class"),
                            json!({"terms": {"field": actual, "order": [{"_count": "desc"}, {"_key": "asc"}], "size": MAX_CLASSES_CARDINALITY}}),
                        );
                    }
                    Some(names) => {
                        // Precision: by predicted class, share of docs whose actual class is equal;
                        // recall: by actual class, share whose predicted class is equal.
                        let (outer, inner) = if matches!(self, Metric::ClassPrecision { .. }) { (predicted, actual.as_str()) } else { (actual.as_str(), predicted) };
                        let mut filters = Map::new();
                        for c in names {
                            filters.insert(c.clone(), json!({"match": {outer: {"query": c, "lenient": true}}}));
                        }
                        let mut equal = Map::new();
                        for c in names {
                            equal.insert(c.clone(), json!({"match": {inner: {"query": c, "lenient": true}}}));
                        }
                        out.insert(format!("{p}_by_class"), json!({"filters": {"filters": filters}, "aggs": {"equal": {"filters": {"filters": equal}}}}));
                    }
                }
            }
            Metric::ClassAucRoc { class_name, result: None, .. } => {
                let actual = &f.actual;
                let top = f.top_classes.as_deref().unwrap_or_default();
                let pc = f.predicted_class.as_deref().unwrap_or_default();
                let nested = json!({
                    "nested": {"path": top},
                    "aggs": {"nested_filter": {"filter": {"term": {pc: class_name}}, "aggs": {"percentiles": percentiles_agg(prob)}}}
                });
                out.insert(format!("{p}_true"), json!({"filter": {"term": {actual: class_name}}, "aggs": {"nested": nested.clone()}}));
                out.insert(format!("{p}_non_true"), json!({"filter": {"bool": {"must_not": [{"term": {actual: class_name}}]}}, "aggs": {"nested": nested}}));
            }
            Metric::Mse { result: None } => {
                out.insert(p, json!({"avg": {"script": diff_squared_script(f)}}));
            }
            Metric::Msle { offset, result: None } => {
                let src = format!(
                    "def offset = {};def diff = Math.log(doc['{}'].value + offset) - Math.log(doc['{}'].value + offset);return diff * diff;",
                    message_format_number(*offset),
                    f.actual,
                    f.predicted.as_deref().unwrap_or_default()
                );
                out.insert(p, json!({"avg": {"script": painless(src)}}));
            }
            Metric::Huber { delta, result: None } => {
                let src = format!(
                    "def a = doc['{}'].value - doc['{}'].value;def delta2 = {};return delta2 * (Math.sqrt(1.0 + Math.pow(a, 2) / delta2) - 1.0);",
                    f.actual,
                    f.predicted.as_deref().unwrap_or_default(),
                    message_format_number(delta * delta)
                );
                out.insert(p, json!({"avg": {"script": painless(src)}}));
            }
            Metric::RSquared { result: None } => {
                out.insert(format!("{p}_residual_sum_of_squares"), json!({"sum": {"script": diff_squared_script(f)}}));
                out.insert(format!("{p}_extended_stats_actual"), json!({"extended_stats": {"field": f.actual}}));
            }
            _ => {}
        }
    }

    fn process(&mut self, i: usize, aggs: &Value, f: &Fields) -> Result<()> {
        let p = format!("m{i}_{}", self.name());
        let get = |suffix: &str| aggs.get(format!("{p}{suffix}"));
        let score_by_threshold = |at: &[f64], value: &dyn Fn(usize) -> Value| -> Value {
            let mut m = Map::new();
            for (ti, t) in at.iter().enumerate() {
                m.insert(crate::script::value::java_double_string(*t), value(ti));
            }
            Value::Object(m)
        };
        let count = |ti: usize, c: &str| get(&format!("_at_{ti}_{c}")).map(doc_count).unwrap_or(0);
        match self {
            Metric::OutlierAucRoc { include_curve, result } if result.is_none() => {
                let (Some(t), Some(r)) = (get("_true"), get("_non_true")) else { return Ok(()) };
                if doc_count(t) == 0 {
                    return Err(bad_request(format!("[auc_roc] requires at least one [{}] to have the value [true]", f.actual)));
                }
                let tp = percentiles_array(&t["percentiles"])?;
                if doc_count(r) == 0 {
                    return Err(bad_request(format!("[auc_roc] requires at least one [{}] to have a different value than [true]", f.actual)));
                }
                let fp = percentiles_array(&r["percentiles"])?;
                *result = Some(auc_roc_result(&tp, &fp, *include_curve));
            }
            Metric::OutlierPrecision { at, result } if result.is_none() => {
                *result = Some(score_by_threshold(at, &|ti| {
                    let (tp, fp) = (count(ti, "TP"), count(ti, "FP"));
                    dbl(if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 })
                }));
            }
            Metric::OutlierRecall { at, result } if result.is_none() => {
                *result = Some(score_by_threshold(at, &|ti| {
                    let (tp, fn_) = (count(ti, "TP"), count(ti, "FN"));
                    dbl(if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 })
                }));
            }
            Metric::OutlierConfusionMatrix { at, result } if result.is_none() => {
                *result = Some(score_by_threshold(at, &|ti| {
                    json!({"tp": count(ti, "TP"), "fp": count(ti, "FP"), "tn": count(ti, "TN"), "fn": count(ti, "FN")})
                }));
            }
            Metric::Accuracy { matrix, result } if result.is_none() => {
                matrix.process(aggs);
                if let Some((_, other)) = &matrix.result {
                    if *other > 0 {
                        return Err(bad_request(format!("Cannot calculate per-class accuracy. Cardinality of field [{}] is too high", f.actual)));
                    }
                    let rows = matrix.rows_parsed();
                    *result = Some(accuracy_result(&rows));
                }
            }
            Metric::MulticlassConfusionMatrix { matrix } => matrix.process(aggs),
            Metric::ClassPrecision { top, result } | Metric::ClassRecall { top, result } if result.is_none() => {
                let is_precision = p.ends_with("precision");
                if top.is_none() {
                    if let Some(t) = get("_by_actual_class") {
                        if t["sum_other_doc_count"].as_u64().unwrap_or(0) > 0 {
                            let what = if is_precision { "precision" } else { "recall" };
                            return Err(bad_request(format!("Cannot calculate average {what}. Cardinality of field [{}] is too high", f.actual)));
                        }
                        let mut names: Vec<String> = t["buckets"].as_array().map(|b| b.iter().map(key_as_string).collect()).unwrap_or_default();
                        // Precision reports classes by name; recall in the
                        // terms order (doc count desc, then key).
                        if is_precision {
                            names.sort();
                        }
                        if names.is_empty() {
                            let key = if is_precision { "avg_precision" } else { "avg_recall" };
                            *result = Some(json!({"classes": [], key: dbl(f64::NAN)}));
                        }
                        *top = Some(names);
                        return Ok(());
                    }
                }
                if let (Some(names), Some(by)) = (top.as_ref(), get("_by_class")) {
                    let buckets = &by["buckets"];
                    let mut classes = Vec::new();
                    let (mut sum, mut n) = (0.0f64, 0u64);
                    for c in names {
                        let b = &buckets[c.as_str()];
                        let total = doc_count(b);
                        let equal = doc_count(&b["equal"]["buckets"][c.as_str()]);
                        let v = equal as f64 / total as f64;
                        if v.is_finite() {
                            sum += v;
                            n += 1;
                        }
                        if v.is_finite() || !is_precision {
                            classes.push(json!({"class_name": c, "value": dbl(v)}));
                        }
                    }
                    let avg = if n == 0 { f64::NAN } else { sum / n as f64 };
                    let key = if is_precision { "avg_precision" } else { "avg_recall" };
                    *result = Some(json!({"classes": classes, key: dbl(avg)}));
                }
            }
            Metric::ClassAucRoc { class_name, include_curve, result } if result.is_none() => {
                let (Some(t), Some(r)) = (get("_true"), get("_non_true")) else { return Ok(()) };
                let t_nf = &t["nested"]["nested_filter"];
                let r_nf = &r["nested"]["nested_filter"];
                if doc_count(t) == 0 {
                    return Err(bad_request(format!("[auc_roc] requires at least one [{}] to have the value [{class_name}]", f.actual)));
                }
                if doc_count(r) == 0 {
                    return Err(bad_request(format!("[auc_roc] requires at least one [{}] to have a different value than [{class_name}]", f.actual)));
                }
                let appeared = doc_count(t_nf) + doc_count(r_nf);
                let total = doc_count(t) + doc_count(r);
                if appeared < total {
                    return Err(bad_request(format!(
                        "[auc_roc] requires that [{class_name}] appears as one of the [{}] for every document (appeared in {appeared} out of {total}). This is probably caused by the num_top_classes value being less than the total number of actual classes in the dataset.",
                        f.predicted_class.as_deref().unwrap_or_default()
                    )));
                }
                let tp = percentiles_array(&t_nf["percentiles"])?;
                let fp = percentiles_array(&r_nf["percentiles"])?;
                *result = Some(auc_roc_result(&tp, &fp, *include_curve));
            }
            Metric::Mse { result } | Metric::Msle { result, .. } | Metric::Huber { result, .. } if result.is_none() => {
                let v = get("").and_then(single_value).unwrap_or(0.0);
                *result = Some(json!({"value": dbl(v)}));
            }
            Metric::RSquared { result } if result.is_none() => {
                let rss = get("_residual_sum_of_squares");
                let stats = get("_extended_stats_actual");
                let v = match (rss, stats) {
                    (Some(rss), Some(st)) => {
                        let count = st["count"].as_u64().unwrap_or(0);
                        let variance = st["variance"].as_f64().unwrap_or(f64::NAN);
                        if count == 0 || variance == 0.0 {
                            0.0
                        } else {
                            1.0 - single_value(rss).unwrap_or(f64::NAN) / (variance * count as f64)
                        }
                    }
                    _ => 0.0,
                };
                *result = Some(json!({"value": dbl(v)}));
            }
            _ => {}
        }
        Ok(())
    }
}

fn accuracy_result(rows: &[MccmRow]) -> Value {
    let n = rows.len();
    let total: u64 = rows.iter().map(|r| r.actual_count).sum();
    let count = |i: usize, j: usize| rows[i].predicted.get(j).map(|x| x.1).unwrap_or(0) as i64;
    let mut classes = Vec::with_capacity(n);
    let mut diagonal = 0i64;
    for i in 0..n {
        let mut correct = total as i64;
        for j in 0..n {
            if i != j {
                correct -= count(i, j);
                correct -= count(j, i);
            }
        }
        correct -= rows[i].other_predicted as i64;
        diagonal += count(i, i);
        classes.push(json!({"class_name": rows[i].actual, "value": dbl(correct as f64 / total as f64)}));
    }
    // Share of the documents whose predicted class equals the actual one.
    let overall = diagonal as f64 / total as f64;
    json!({"classes": classes, "overall_accuracy": dbl(overall)})
}

// ---------------------------------------------------------------------------
// ROC curve (`AbstractAucRoc`).

#[derive(Clone, Copy)]
struct Point {
    tpr: f64,
    fpr: f64,
    threshold: f64,
}

/// `AbstractAucRoc.percentilesArray`.
fn percentiles_array(p: &Value) -> Result<[f64; 99]> {
    let mut out = [0.0f64; 99];
    let entries: Vec<(f64, f64)> = match &p["values"] {
        Value::Object(m) => m.iter().map(|(k, v)| (k.parse().unwrap_or(f64::NAN), v.as_f64().unwrap_or(f64::NAN))).collect(),
        Value::Array(a) => a.iter().map(|e| (e["key"].as_f64().unwrap_or(f64::NAN), e["value"].as_f64().unwrap_or(f64::NAN))).collect(),
        _ => Vec::new(),
    };
    for (pct, v) in entries {
        if v.is_nan() {
            return Err(bad_request("[auc_roc] requires at all the percentiles values to be finite numbers"));
        }
        let idx = pct as i64 - 1;
        if (0..99).contains(&idx) {
            out[idx as usize] = v;
        }
    }
    Ok(out)
}

/// `Math.ulp`.
fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return if a.is_nan() { f64::NAN } else { f64::INFINITY };
    }
    if a == f64::MAX {
        return 2f64.powi(971);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// `Arrays.binarySearch(double[], double)`.
fn java_binary_search(a: &[f64], key: f64) -> i64 {
    let (mut low, mut high) = (0i64, a.len() as i64 - 1);
    while low <= high {
        let mid = ((low + high) as u64 >> 1) as i64;
        let mv = a[mid as usize];
        if mv < key {
            low = mid + 1;
        } else if mv > key {
            high = mid - 1;
        } else {
            let mb = if mv.is_nan() { 0x7ff8000000000000i64 } else { mv.to_bits() as i64 };
            let kb = if key.is_nan() { 0x7ff8000000000000i64 } else { key.to_bits() as i64 };
            if mb == kb {
                return mid;
            } else if mb < kb {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    -(low + 1)
}

fn rate(i: usize) -> f64 {
    1.0 - 0.01 * ((i + 1) as f64)
}

fn interpolate(x: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    y1 + (x - x1) * (y2 - y1) / (x2 - x1)
}

fn interpolate_rate(p: &[f64; 99], threshold: f64) -> f64 {
    let r = java_binary_search(p, threshold);
    if r >= 0 {
        return rate(r as usize);
    }
    let right = -r - 1;
    let left = right - 1;
    if right >= p.len() as i64 {
        0.0
    } else if left < 0 {
        1.0
    } else {
        let (l, r) = (left as usize, right as usize);
        interpolate(threshold, p[l], rate(l), p[r], rate(r))
    }
}

fn scan_points(own: &[f64; 99], is_tp: bool, against: &[f64; 99]) -> Vec<Point> {
    (0..own.len())
        .map(|i| {
            let rate = rate(i);
            let threshold = f64::max(0.0, own[i] - ulp(own[i]));
            let against_rate = interpolate_rate(against, threshold);
            if is_tp {
                Point { tpr: rate, fpr: against_rate, threshold }
            } else {
                Point { tpr: against_rate, fpr: rate, threshold }
            }
        })
        .collect()
}

fn average_point(points: &[Point]) -> Point {
    if points.len() == 1 {
        return points[0];
    }
    let (mut t, mut f, mut th) = (0.0, 0.0, 0.0);
    for p in points {
        t += p.tpr;
        f += p.fpr;
        th += p.threshold;
    }
    let n = points.len() as f64;
    Point { tpr: t / n, fpr: f / n, threshold: th / n }
}

fn build_curve(tp: &[f64; 99], fp: &[f64; 99]) -> Vec<Point> {
    let mut pts = scan_points(tp, true, fp);
    pts.extend(scan_points(fp, false, tp));
    // Threshold descending, then fpr, then tpr ascending (stable).
    pts.sort_by(|a, b| b.threshold.total_cmp(&a.threshold).then(a.fpr.total_cmp(&b.fpr)).then(a.tpr.total_cmp(&b.tpr)));
    let mut collapsed = Vec::new();
    let mut group: Vec<Point> = Vec::new();
    for p in pts {
        if !group.is_empty() && group[0].threshold != p.threshold {
            collapsed.push(average_point(&group));
            group.clear();
        }
        group.push(p);
    }
    if !group.is_empty() {
        collapsed.push(average_point(&group));
    }
    let mut out = Vec::with_capacity(collapsed.len() + 2);
    out.push(Point { tpr: 0.0, fpr: 0.0, threshold: 1.0 });
    out.extend(collapsed);
    out.push(Point { tpr: 1.0, fpr: 1.0, threshold: 0.0 });
    out
}

fn auc_score(curve: &[Point]) -> f64 {
    let mut auc = 0.0;
    for w in curve.windows(2) {
        let (l, r) = (w[0], w[1]);
        auc += (r.fpr - l.fpr) * (r.tpr + l.tpr) / 2.0;
    }
    auc
}

fn auc_roc_result(tp: &[f64; 99], fp: &[f64; 99], include_curve: bool) -> Value {
    let curve = build_curve(tp, fp);
    let mut m = Map::new();
    m.insert("value".into(), dbl(auc_score(&curve)));
    if include_curve && !curve.is_empty() {
        m.insert(
            "curve".into(),
            Value::Array(curve.iter().map(|p| json!({"tpr": dbl(p.tpr), "fpr": dbl(p.fpr), "threshold": dbl(p.threshold)})).collect()),
        );
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// Parsing.

struct Evaluation {
    kind: EvalKind,
    fields: Fields,
    metrics: Vec<Metric>,
}

fn as_obj<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| EsError::x_content_parse(format!("[{what}] Expected START_OBJECT but was {}", token_name(v))))
}

fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn string_field(v: &Value, owner: &str, field: &str) -> Result<Option<String>> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::Bool(b) => Ok(Some(b.to_string())),
        other => Err(EsError::x_content_parse(format!("[{owner}] {field} doesn't support values of type: {}", token_name(other)))),
    }
}

fn number_field(v: &Value, owner: &str, field: &str) -> Result<Option<f64>> {
    match v {
        Value::Null => Ok(None),
        Value::Number(n) => Ok(n.as_f64()),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|_| EsError::x_content_parse(format!("[{owner}] failed to parse field [{field}]")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\"")))),
        other => Err(EsError::x_content_parse(format!("[{owner}] {field} doesn't support values of type: {}", token_name(other)))),
    }
}

fn bool_field(v: &Value, owner: &str, field: &str) -> Result<Option<bool>> {
    match v {
        Value::Null => Ok(None),
        Value::Bool(b) => Ok(Some(*b)),
        Value::String(s) if s == "true" || s == "false" => Ok(Some(s == "true")),
        other => Err(EsError::x_content_parse(format!("[{owner}] failed to parse field [{field}]")).caused_by(EsError::illegal_argument(format!(
            "Failed to parse value [{}] as only [true] or [false] are allowed.",
            match other {
                Value::String(s) => s.clone(),
                o => o.to_string(),
            }
        )))),
    }
}

/// Parse one metric object of evaluation `eval`.
fn parse_metric(kind: EvalKind, name: &str, body: &Value) -> Result<Metric> {
    let eval = kind.name();
    let wrap = |e: EsError| in_metric(eval, name, e);
    let empty = Map::new();
    let obj = match body {
        Value::Object(m) => m,
        Value::Null => &empty,
        other => return Err(wrap(EsError::x_content_parse(format!("[{name}] Expected START_OBJECT but was {}", token_name(other))))),
    };
    let field = |k: &str| obj.get(k).unwrap_or(&Value::Null);
    let thresholds = || -> Result<Vec<f64>> {
        let at = match obj.get("at") {
            None | Some(Value::Null) => return Err(wrap(EsError::illegal_argument("Required [at]"))),
            Some(Value::Array(a)) => a.iter().map(|x| number_field(x, name, "at").map(|v| v.unwrap_or(f64::NAN))).collect::<Result<Vec<f64>>>().map_err(wrap)?,
            Some(v) => vec![number_field(v, name, "at").map_err(wrap)?.unwrap_or(f64::NAN)],
        };
        let fail = |msg: String| wrap(build_failure(name, obj, &["at"], bad_request(msg)));
        if at.is_empty() {
            return Err(fail(format!("[{name}.at] must have at least one value")));
        }
        if at.iter().any(|t| !(*t >= 0.0 && *t <= 1.0)) {
            return Err(fail(format!("[{name}.at] values must be in [0.0, 1.0]")));
        }
        Ok(at)
    };
    let unknown = || {
        EsError::new(400, "x_content_parse_exception", format!("[metrics] failed to parse field [{name}]"))
            .caused_by(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{eval}.{name}]")))
    };
    Ok(match (kind, name) {
        (EvalKind::OutlierDetection, "auc_roc") => {
            Metric::OutlierAucRoc { include_curve: bool_field(field("include_curve"), name, "include_curve").map_err(wrap)?.unwrap_or(false), result: None }
        }
        (EvalKind::OutlierDetection, "precision") => Metric::OutlierPrecision { at: thresholds()?, result: None },
        (EvalKind::OutlierDetection, "recall") => Metric::OutlierRecall { at: thresholds()?, result: None },
        (EvalKind::OutlierDetection, "confusion_matrix") => Metric::OutlierConfusionMatrix { at: thresholds()?, result: None },
        (EvalKind::Classification, "accuracy") => Metric::Accuracy { matrix: Mccm::new(MAX_CLASSES_CARDINALITY, "accuracy".into()), result: None },
        (EvalKind::Classification, "multiclass_confusion_matrix") => {
            let size = match field("size") {
                Value::Null => None,
                v => Some(number_field(v, name, "size").map_err(wrap)?.unwrap_or(f64::NAN)),
            };
            let prefix = string_field(field("agg_name_prefix"), name, "agg_name_prefix").map_err(wrap)?.unwrap_or_default();
            if let Some(s) = size {
                if !(1.0..=MAX_CLASSES_CARDINALITY as f64).contains(&s) {
                    return Err(wrap(build_failure(
                        name,
                        obj,
                        &["size", "agg_name_prefix"],
                        bad_request(format!("[size] must be an integer in [1, {MAX_CLASSES_CARDINALITY}]")),
                    )));
                }
            }
            Metric::MulticlassConfusionMatrix { matrix: Mccm::new(size.map(|s| s as usize).unwrap_or(10), format!("{prefix}mccm")) }
        }
        (EvalKind::Classification, "precision") => Metric::ClassPrecision { top: None, result: None },
        (EvalKind::Classification, "recall") => Metric::ClassRecall { top: None, result: None },
        (EvalKind::Classification, "auc_roc") => {
            let include_curve = bool_field(field("include_curve"), name, "include_curve").map_err(wrap)?.unwrap_or(false);
            let Some(class_name) = string_field(field("class_name"), name, "class_name").map_err(wrap)? else {
                return Err(wrap(build_failure(name, obj, &["include_curve", "class_name"], EsError::illegal_argument("[class_name] must not be null."))));
            };
            Metric::ClassAucRoc { class_name, include_curve, result: None }
        }
        (EvalKind::Regression, "mse") => Metric::Mse { result: None },
        (EvalKind::Regression, "msle") => Metric::Msle { offset: number_field(field("offset"), name, "offset").map_err(wrap)?.unwrap_or(1.0), result: None },
        (EvalKind::Regression, "huber") => Metric::Huber { delta: number_field(field("delta"), name, "delta").map_err(wrap)?.unwrap_or(1.0), result: None },
        (EvalKind::Regression, "r_squared") => Metric::RSquared { result: None },
        _ => return Err(in_evaluation(EsError::x_content_parse(format!("[{eval}] failed to parse field [metrics]")).caused_by(unknown()))),
    })
}

fn default_metrics(kind: EvalKind) -> Vec<Metric> {
    match kind {
        EvalKind::OutlierDetection => vec![
            Metric::OutlierAucRoc { include_curve: false, result: None },
            Metric::OutlierPrecision { at: vec![0.25, 0.5, 0.75], result: None },
            Metric::OutlierRecall { at: vec![0.25, 0.5, 0.75], result: None },
            Metric::OutlierConfusionMatrix { at: vec![0.25, 0.5, 0.75], result: None },
        ],
        EvalKind::Classification => vec![
            Metric::Accuracy { matrix: Mccm::new(MAX_CLASSES_CARDINALITY, "accuracy".into()), result: None },
            Metric::MulticlassConfusionMatrix { matrix: Mccm::new(10, "mccm".into()) },
            Metric::ClassPrecision { top: None, result: None },
            Metric::ClassRecall { top: None, result: None },
        ],
        EvalKind::Regression => vec![Metric::Mse { result: None }, Metric::RSquared { result: None }, Metric::Huber { result: None, delta: 1.0 }],
    }
}

fn parse_evaluation(v: &Value) -> Result<Evaluation> {
    let obj = as_obj(v, "evaluation").map_err(in_evaluation)?;
    let Some((name, body)) = obj.iter().next() else {
        return Err(in_evaluation(EsError::x_content_parse("[evaluation] no evaluation specified")));
    };
    let kind = match name.as_str() {
        "outlier_detection" | "binary_soft_classification" => EvalKind::OutlierDetection,
        "classification" => EvalKind::Classification,
        "regression" => EvalKind::Regression,
        other => return Err(in_evaluation(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{other}]")))),
    };
    let eval = kind.name();
    let body = as_obj(body, eval).map_err(in_evaluation)?;
    let allowed: &[&str] = match kind {
        EvalKind::OutlierDetection => &["actual_field", "predicted_probability_field", "metrics"],
        EvalKind::Classification => &["actual_field", "predicted_field", "top_classes_field", "metrics"],
        EvalKind::Regression => &["actual_field", "predicted_field", "metrics"],
    };
    let mut strings: Map<String, Value> = Map::new();
    let mut metrics: Option<Vec<Metric>> = None;
    for (k, v) in body {
        if !allowed.contains(&k.as_str()) {
            return Err(in_evaluation(EsError::x_content_parse(format!("[{eval}] unknown field [{k}]"))));
        }
        if k == "metrics" {
            match v {
                Value::Null => metrics = None,
                Value::Object(m) => {
                    let mut list = Vec::new();
                    for (mn, mb) in m {
                        list.push(parse_metric(kind, mn, mb)?);
                    }
                    metrics = Some(list);
                }
                other => {
                    return Err(in_evaluation(EsError::x_content_parse(format!("[{eval}] metrics doesn't support values of type: {}", token_name(other)))));
                }
            }
        } else if let Some(s) = string_field(v, eval, k).map_err(in_evaluation)? {
            strings.insert(k.clone(), Value::String(s));
        }
    }
    let s = |k: &str| strings.get(k).and_then(Value::as_str).map(str::to_string);
    let required: &[&str] = match kind {
        EvalKind::OutlierDetection => &["actual_field", "predicted_probability_field"],
        _ => &["actual_field"],
    };
    let missing: Vec<&str> = required.iter().copied().filter(|r| s(r).is_none()).collect();
    if !missing.is_empty() {
        return Err(in_evaluation(EsError::illegal_argument(format!("Required [{}]", missing.join(", ")))));
    }
    let actual = s("actual_field").unwrap_or_default();
    let fields = match kind {
        EvalKind::OutlierDetection => Fields { actual, predicted_probability: s("predicted_probability_field"), ..Default::default() },
        EvalKind::Classification => {
            let top = s("top_classes_field").unwrap_or_else(|| DEFAULT_TOP_CLASSES_FIELD.to_string());
            Fields {
                actual,
                predicted: s("predicted_field"),
                predicted_class: Some(format!("{top}.class_name")),
                predicted_probability: Some(format!("{top}.class_probability")),
                top_classes: Some(top),
                probability_nested: true,
            }
        }
        EvalKind::Regression => Fields { actual, predicted: s("predicted_field"), ..Default::default() },
    };
    let ctor_args: &[&str] = allowed;
    let build_err = |e: EsError| in_evaluation(build_failure(eval, body, ctor_args, e));
    // `Evaluation.initMetrics`.
    let mut metrics = match metrics {
        Some(m) => m,
        None => default_metrics(kind),
    };
    if metrics.is_empty() {
        return Err(build_err(bad_request(format!("[{eval}] must have one or more metrics"))));
    }
    metrics.sort_by(|a, b| a.name().cmp(b.name()));
    for (param, field) in fields.potentially_required() {
        if field.is_none() {
            let users: Vec<&str> = metrics.iter().filter(|m| m.required().contains(&param)).map(|m| m.name()).collect();
            if !users.is_empty() {
                return Err(build_err(bad_request(format!("[{eval}] must define [{param}] as required by the following metrics [{}]", users.join(", ")))));
            }
        }
    }
    Ok(Evaluation { kind, fields, metrics })
}

struct Request {
    indices: Vec<String>,
    query: Option<Value>,
    evaluation: Evaluation,
}

fn parse_request(body: &Value) -> Result<Request> {
    let empty = Map::new();
    let obj = match body {
        Value::Object(m) => m,
        Value::Null => &empty,
        other => return Err(EsError::x_content_parse(format!("{ACTION} Expected START_OBJECT but was {}", token_name(other)))),
    };
    let mut indices: Option<Vec<String>> = None;
    let mut query = None;
    let mut evaluation = None;
    for (k, v) in obj {
        match k.as_str() {
            "index" => {
                let list = match v {
                    Value::Array(a) => a.iter().map(|x| string_field(x, ACTION, "index").map(|s| s.unwrap_or_default())).collect::<Result<Vec<_>>>()?,
                    Value::Null => continue,
                    other => vec![string_field(other, ACTION, "index")?.unwrap_or_default()],
                };
                indices = Some(list);
            }
            "query" => {
                if !v.is_object() {
                    return Err(EsError::x_content_parse(format!("{ACTION} failed to parse field [query]")));
                }
                query = Some(v.clone());
            }
            "evaluation" => evaluation = Some(parse_evaluation(v)?),
            other => return Err(EsError::x_content_parse(format!("{ACTION} unknown field [{other}]"))),
        }
    }
    let mut missing = Vec::new();
    if indices.is_none() {
        missing.push("index");
    }
    if evaluation.is_none() {
        missing.push("evaluation");
    }
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    let indices = indices.unwrap_or_default();
    if indices.is_empty() {
        return Err(EsError::x_content_parse("Failed to build [cluster:monitor/xpack/ml/data_frame/evaluate] after last required field arrived")
            .caused_by(bad_request("At least one index must be specified")));
    }
    Ok(Request { indices, query, evaluation: evaluation.expect("checked") })
}

// ---------------------------------------------------------------------------
// Execution.

/// `OutlierDetection.actualIsTrueQuery` (`<field>:(1 OR true)`), resolved
/// against the field's mapping: boolean fields match `true`, numeric ones
/// `1`, anything else either string.
fn actual_is_true_query(node: &Node, index_expr: &str, field: &str) -> Value {
    let ty = node.resolve(index_expr, ResolveOpts::default()).ok().and_then(|ts| ts.iter().find_map(|t| t.svc.lookup().get(field).map(|f| f.ty)));
    match ty {
        Some(EsType::Boolean) => json!({"term": {field: true}}),
        Some(t) if t.is_numeric() => json!({"term": {field: 1}}),
        _ => json!({"terms": {field: ["1", "true"]}}),
    }
}

/// `Evaluation.buildSearch`'s query: every required field exists, and the
/// request query matches.
fn base_query(ev: &Evaluation, user_query: &Value) -> Value {
    let required: std::collections::HashSet<&str> = ev.metrics.iter().flat_map(|m| m.required().iter().copied()).collect();
    let f = &ev.fields;
    let mut filter = Vec::new();
    if required.contains("actual_field") {
        filter.push(json!({"exists": {"field": f.actual}}));
    }
    if let (Some(p), true) = (&f.predicted, required.contains("predicted_field")) {
        filter.push(json!({"exists": {"field": p}}));
    }
    let nested_exists = |field: &str| {
        json!({"nested": {"path": f.top_classes.as_deref().unwrap_or_default(), "query": {"exists": {"field": field}}, "score_mode": "none", "ignore_unmapped": true}})
    };
    if let (Some(pc), true) = (&f.predicted_class, required.contains("predicted_class_field")) {
        filter.push(nested_exists(pc));
    }
    if let (Some(pp), true) = (&f.predicted_probability, required.contains("predicted_probability_field")) {
        filter.push(if f.probability_nested { nested_exists(pp) } else { json!({"exists": {"field": pp}}) });
    }
    filter.push(user_query.clone());
    json!({"bool": {"filter": filter}})
}

/// The names of the fields the evaluation's metrics need (`getRequiredFields`).
fn required_field_names(ev: &Evaluation) -> Vec<String> {
    let required: std::collections::HashSet<&str> = ev.metrics.iter().flat_map(|m| m.required().iter().copied()).collect();
    ev.fields.potentially_required().into_iter().filter(|(p, _)| required.contains(p)).map(|(_, f)| f.unwrap_or("null").to_string()).collect()
}

/// `POST _ml/data_frame/_evaluate`: parse the request, run the metric
/// searches over `index` and return the response body
/// (`{"<evaluation>": {"<metric>": result, ...}}`, metrics by name).
pub fn evaluate(node: &Node, body: &Value) -> Result<Value> {
    let Request { indices, query, evaluation: mut ev } = parse_request(body)?;
    // Each entry is one index expression: a comma is part of the name.
    if let Some(bad) = indices.iter().find(|i| i.contains(',')) {
        return Err(EsError::index_not_found(bad));
    }
    let index_expr = indices.join(",");
    // Missing concrete indices fail like the search would.
    node.resolve(&index_expr, ResolveOpts::default())?;
    let user_query = query.unwrap_or_else(|| json!({"match_all": {}}));
    let ctx = Ctx {
        actual_is_true: if ev.kind == EvalKind::OutlierDetection { actual_is_true_query(node, &index_expr, &ev.fields.actual) } else { Value::Null },
        max_buckets: node.max_buckets().max(1),
    };
    let query = base_query(&ev, &user_query);
    let mut steps = 0;
    while ev.metrics.iter().any(|m| m.result().is_none()) {
        steps += 1;
        if steps > 100_000 {
            return Err(EsError::internal("evaluation did not converge"));
        }
        let mut aggs = Map::new();
        for (i, m) in ev.metrics.iter().enumerate() {
            m.aggs(i, &ev.fields, &ctx, &mut aggs);
        }
        let mut search = json!({"size": 0, "query": query});
        if !aggs.is_empty() {
            search["aggs"] = Value::Object(aggs);
        }
        let resp = super::search(node, &index_expr, &search)?;
        let total = match &resp["hits"]["total"] {
            Value::Number(n) => n.as_u64().unwrap_or(0),
            v => v["value"].as_u64().unwrap_or(0),
        };
        if total == 0 {
            return Err(bad_request(format!("No documents found containing all the required fields [{}]", required_field_names(&ev).join(", "))));
        }
        let aggs = resp.get("aggregations").cloned().unwrap_or_else(|| json!({}));
        for (i, m) in ev.metrics.iter_mut().enumerate() {
            m.process(i, &aggs, &ev.fields)?;
        }
    }
    let mut out = Map::new();
    for m in &ev.metrics {
        if let Some(r) = m.result() {
            out.insert(m.name().to_string(), r);
        }
    }
    Ok(json!({ ev.kind.name(): out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_format() {
        assert_eq!(message_format_number(4.0), "4");
        assert_eq!(message_format_number(6.0), "6");
        assert_eq!(message_format_number(1.0), "1");
        assert_eq!(message_format_number(0.25), "0.25");
        assert_eq!(message_format_number(1234.5), "1,234.5");
        assert_eq!(message_format_number(-0.5), "-0.5");
    }

    #[test]
    fn binary_search_like_java() {
        let a = [1.0, 2.0, 2.0, 3.0];
        assert_eq!(java_binary_search(&a, 0.5), -1);
        assert_eq!(java_binary_search(&a, 3.5), -5);
        assert_eq!(java_binary_search(&a, 2.5), -4);
        assert_eq!(java_binary_search(&a, 3.0), 3);
    }
}
