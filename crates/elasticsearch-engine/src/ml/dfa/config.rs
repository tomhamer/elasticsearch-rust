//! Parsing and validation of data frame analytics configs, mirroring ES
//! `DataFrameAnalyticsConfig` (+ `DataFrameAnalyticsSource` / `Dest`,
//! `OutlierDetection`, `Regression`, `Classification`, `BoostedTreeParams`
//! and the feature processors) and `DataFrameAnalyticsConfigUpdate`.
//!
//! The normalized config is a JSON object in ES's rendering order with every
//! default filled in, so it can be stored and returned as is.

use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

/// `MlConfigVersion.CURRENT` as ES 8.19 reports it.
pub const VERSION: &str = "12.0.0";
/// `DataFrameAnalyticsConfig.DEFAULT_MODEL_MEMORY_LIMIT`.
pub const DEFAULT_MML: &str = "1gb";
/// `DataFrameAnalyticsConfig.MIN_MODEL_MEMORY_LIMIT` (1kb).
const MIN_MML_BYTES: u64 = 1024;

const CONFIG: &str = "data_frame_analytics_config";
const UPDATE: &str = "data_frame_analytics_config_update";
const SOURCE: &str = "data_frame_analytics_source";
const DEST: &str = "data_frame_analytics_dest";

/// Top-level key order of a rendered config.
pub const ORDER: [&str; 12] =
    ["id", "create_time", "version", "description", "source", "dest", "analysis", "analyzed_fields", "model_memory_limit", "allow_lazy_start", "max_num_threads", "_meta"];

/// Where a config body comes from (decides which parts are required).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `PUT _ml/data_frame/analytics/{id}`.
    Put,
    /// `_explain` / `_preview` bodies: no id, `dest` optional.
    Explain,
}

/// A parsed config plus what the transport layer still has to check.
pub struct Parsed {
    pub config: Map<String, Value>,
    /// `model_memory_limit` was given (vs the default, which is capped
    /// silently by `xpack.ml.max_model_memory_limit`).
    pub mml_explicit: bool,
}

pub(crate) fn status(msg: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", msg)
}

pub(crate) fn conflict(msg: impl Into<String>) -> EsError {
    EsError::new(409, "status_exception", msg)
}

fn xcp(msg: impl Into<String>) -> EsError {
    EsError::x_content_parse(msg)
}

fn unknown(section: &str, k: &str) -> EsError {
    xcp(format!("[1:1] [{section}] unknown field [{k}]"))
}

fn failed_field(section: &str, field: &str, cause: EsError) -> EsError {
    xcp(format!("[1:1] [{section}] failed to parse field [{field}]")).caused_by(cause)
}

fn failed_build(obj: &str, cause: EsError) -> EsError {
    xcp(format!("Failed to build [{obj}] after last required field arrived")).caused_by(cause)
}

fn token(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

fn bad_type(section: &str, name: &str, v: &Value) -> EsError {
    xcp(format!("[1:1] [{section}] {name} doesn't support values of type: {}", token(v)))
}

fn get_str(section: &str, name: &str, v: &Value) -> Result<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        _ => Err(bad_type(section, name, v)),
    }
}

fn number_format(s: &str) -> EsError {
    EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))
}

fn get_f64(section: &str, name: &str, v: &Value) -> Result<f64> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| failed_field(section, name, number_format(s))),
        _ => Err(bad_type(section, name, v)),
    }
}

fn get_i64(section: &str, name: &str, v: &Value) -> Result<i64> {
    match v {
        Value::Number(n) => Ok(n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64)),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| failed_field(section, name, number_format(s))),
        _ => Err(bad_type(section, name, v)),
    }
}

fn get_bool(section: &str, name: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        _ => Err(failed_field(section, name, EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", crate::ml::dfa::plain(v))))),
    }
}

fn get_obj<'a>(section: &str, name: &str, v: &'a Value) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| bad_type(section, name, v))
}

/// Java `Double.toString`-compatible JSON number (`1` → `1.0`).
pub(crate) fn dbl(f: f64) -> Value {
    json!(f)
}

/// `ByteSizeValue` parse of a `model_memory_limit` → (bytes, rendering).
pub fn parse_mml(v: &Value, section: &str) -> Result<(u64, String)> {
    let s = match v {
        Value::Number(n) => format!("{}b", n),
        Value::String(s) => s.clone(),
        _ => return Err(bad_type(section, "model_memory_limit", v)),
    };
    crate::ilm::parse_size(&s, "model_memory_limit").map_err(|e| failed_field(section, "model_memory_limit", e))
}

fn mml_min_check(bytes: u64, rendered: &str) -> Result<()> {
    if bytes < MIN_MML_BYTES {
        let shown = if rendered == "0" || rendered == "-1" { format!("{}b", rendered) } else { rendered.to_string() };
        return Err(status(format!("model_memory_limit must be at least 1kb. Value = {shown}")));
    }
    Ok(())
}

/// `FetchSourceContext` (`analyzed_fields`, `source._source`) →
/// `{"includes": [..], "excludes": [..]}`.
fn fetch_source(section: &str, name: &str, v: &Value) -> Result<Value> {
    let strs = |v: &Value, what: &str| -> Result<Vec<Value>> {
        match v {
            Value::String(s) => Ok(vec![json!(s)]),
            Value::Array(a) => a
                .iter()
                .map(|x| x.as_str().map(|s| json!(s)).ok_or_else(|| failed_field(section, name, EsError::parsing(format!("Unknown key for a {} in [{what}].", token(x))))))
                .collect(),
            Value::Null => Ok(vec![]),
            _ => Err(failed_field(section, name, EsError::parsing(format!("Unknown key for a {} in [{what}].", token(v))))),
        }
    };
    match v {
        Value::Bool(_) | Value::Null => Ok(json!({"includes": [], "excludes": []})),
        Value::String(_) | Value::Array(_) => Ok(json!({"includes": strs(v, name)?, "excludes": []})),
        Value::Object(o) => {
            let mut inc = vec![];
            let mut exc = vec![];
            for (k, x) in o {
                match k.as_str() {
                    "includes" | "include" => inc = strs(x, k)?,
                    "excludes" | "exclude" => exc = strs(x, k)?,
                    _ => return Err(failed_field(section, name, EsError::parsing(format!("Unknown key for a {} in [{k}].", token(x))))),
                }
            }
            Ok(json!({"includes": inc, "excludes": exc}))
        }
        _ => Err(failed_field(section, name, EsError::parsing(format!("Expected one of [VALUE_BOOLEAN, START_OBJECT] but found [{}]", token(v))))),
    }
}

/// Does a `{"includes", "excludes"}` filter exclude `field`
/// (`DataFrameAnalyticsSource.isFieldExcluded`)?
pub fn is_field_excluded(filter: &Value, field: &str) -> bool {
    let list = |k: &str| -> Vec<String> { filter[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
    let inc = list("includes");
    let exc = list("excludes");
    if !inc.is_empty() && !inc.iter().any(|p| crate::mapping::wildcard_match(p, field)) {
        return true;
    }
    exc.iter().any(|p| crate::mapping::wildcard_match(p, field))
}

fn parse_source(v: &Value) -> Result<Value> {
    let wrap = |e: EsError| failed_field(CONFIG, "source", e);
    let o = get_obj(CONFIG, "source", v)?;
    let mut index: Option<Vec<String>> = None;
    let mut query: Option<Value> = None;
    let mut src: Option<Value> = None;
    let mut rt: Option<Value> = None;
    for (k, x) in o {
        match k.as_str() {
            "index" => {
                index = Some(match x {
                    Value::String(s) => vec![s.clone()],
                    Value::Array(a) => a.iter().map(|i| i.as_str().map(str::to_string).unwrap_or_default()).collect(),
                    Value::Null => vec![],
                    _ => return Err(wrap(bad_type(SOURCE, "index", x))),
                })
            }
            "query" => {
                let q = get_obj(SOURCE, "query", x).map_err(wrap)?;
                if let Err(e) = crate::search::parse_request(&json!({"query": q}), &Default::default()) {
                    return Err(wrap(failed_field(SOURCE, "query", status("Data Frame Analytics config query is not parsable").caused_by(e))));
                }
                query = Some(x.clone());
            }
            "_source" => src = Some(fetch_source(SOURCE, "_source", x).map_err(wrap)?),
            "runtime_mappings" => {
                get_obj(SOURCE, "runtime_mappings", x).map_err(wrap)?;
                rt = Some(x.clone());
            }
            _ => return Err(wrap(unknown(SOURCE, k))),
        }
    }
    let index = index.ok_or_else(|| wrap(EsError::illegal_argument("Required [index]")))?;
    if index.is_empty() {
        return Err(wrap(failed_build(SOURCE, EsError::illegal_argument("source.index must specify at least one index"))));
    }
    if index.iter().any(String::is_empty) {
        return Err(wrap(failed_build(SOURCE, EsError::illegal_argument("source.index must contain non-null and non-empty strings"))));
    }
    let mut m = Map::new();
    m.insert("index".into(), json!(index));
    m.insert("query".into(), query.unwrap_or_else(|| json!({"match_all": {}})));
    if let Some(s) = src {
        m.insert("_source".into(), s);
    }
    if let Some(r) = rt {
        m.insert("runtime_mappings".into(), r);
    }
    Ok(Value::Object(m))
}

fn parse_dest(v: &Value) -> Result<Value> {
    let wrap = |e: EsError| failed_field(CONFIG, "dest", e);
    let o = get_obj(CONFIG, "dest", v)?;
    let mut index: Option<String> = None;
    let mut results: Option<String> = None;
    for (k, x) in o {
        match k.as_str() {
            "index" => index = Some(get_str(DEST, "index", x).map_err(wrap)?),
            "results_field" => results = Some(get_str(DEST, "results_field", x).map_err(wrap)?),
            _ => return Err(wrap(unknown(DEST, k))),
        }
    }
    let index = index.ok_or_else(|| wrap(EsError::illegal_argument("Required [index]")))?;
    if index.is_empty() {
        return Err(wrap(failed_build(DEST, status("[index] must be non-empty"))));
    }
    Ok(json!({"index": index, "results_field": results.unwrap_or_else(|| "ml".into())}))
}

fn random_seed() -> i64 {
    let mut b = [0u8; 8];
    if getrandom::getrandom(&mut b).is_err() {
        b = (crate::date::now_millis() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes();
    }
    i64::from_le_bytes(b)
}

fn enum_err(section: &str, field: &str, class: &str, v: &str) -> EsError {
    failed_field(section, field, EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.ml.dataframe.analyses.{class}.{}", v.to_uppercase())))
}

fn parse_outlier(o: &Map<String, Value>) -> Result<Value> {
    const S: &str = "outlier_detection";
    let mut n_neighbors: Option<i64> = None;
    let mut method: Option<String> = None;
    let mut fit: Option<f64> = None;
    let mut cfi = true;
    let mut frac = 0.05;
    let mut std = true;
    for (k, x) in o {
        match k.as_str() {
            "n_neighbors" => n_neighbors = Some(get_i64(S, k, x)?),
            "method" => {
                let m = get_str(S, k, x)?.to_lowercase();
                if !matches!(m.as_str(), "lof" | "ldof" | "distance_kth_nn" | "distance_knn") {
                    return Err(enum_err(S, k, "OutlierDetection.Method", &m));
                }
                method = Some(m);
            }
            "feature_influence_threshold" => fit = Some(get_f64(S, k, x)?),
            "compute_feature_influence" => cfi = get_bool(S, k, x)?,
            "outlier_fraction" => frac = get_f64(S, k, x)?,
            "standardization_enabled" => std = get_bool(S, k, x)?,
            _ => return Err(unknown(S, k)),
        }
    }
    if n_neighbors.is_some_and(|n| n <= 0) {
        return Err(status("[n_neighbors] must be a positive integer"));
    }
    if fit.is_some_and(|f| !(0.0..=1.0).contains(&f)) {
        return Err(status("[feature_influence_threshold] must be in [0, 1]"));
    }
    if !(0.0..=1.0).contains(&frac) {
        return Err(status("[outlier_fraction] must be in [0, 1]"));
    }
    let mut m = Map::new();
    if let Some(n) = n_neighbors {
        m.insert("n_neighbors".into(), json!(n));
    }
    if let Some(x) = method {
        m.insert("method".into(), json!(x));
    }
    if let Some(f) = fit {
        m.insert("feature_influence_threshold".into(), dbl(f));
    }
    m.insert("compute_feature_influence".into(), json!(cfi));
    m.insert("outlier_fraction".into(), dbl(frac));
    m.insert("standardization_enabled".into(), json!(std));
    Ok(Value::Object(m))
}

/// `BoostedTreeParams` fields in ES rendering order: (name, is_integer).
const BTP: [(&str, bool); 12] = [
    ("alpha", false),
    ("lambda", false),
    ("gamma", false),
    ("eta", false),
    ("eta_growth_rate_per_tree", false),
    ("max_trees", true),
    ("feature_bag_fraction", false),
    ("num_top_feature_importance_values", true),
    ("soft_tree_depth_limit", false),
    ("soft_tree_depth_tolerance", false),
    ("downsample_factor", false),
    ("max_optimization_rounds_per_hyperparameter", true),
];

fn check_btp(p: &Map<String, Value>) -> Result<()> {
    let f = |k: &str| p.get(k).and_then(Value::as_f64);
    let fail = |m: &str| Err(status(m.to_string()));
    if f("lambda").is_some_and(|x| x < 0.0) {
        return fail("[lambda] must be a non-negative double");
    }
    if f("gamma").is_some_and(|x| x < 0.0) {
        return fail("[gamma] must be a non-negative double");
    }
    if f("eta").is_some_and(|x| !(0.001..=1.0).contains(&x)) {
        return fail("[eta] must be a double in [0.001, 1]");
    }
    if f("max_trees").is_some_and(|x| !(1.0..=2000.0).contains(&x)) {
        return fail("[max_trees] must be an integer in [1, 2000]");
    }
    if f("feature_bag_fraction").is_some_and(|x| x <= 0.0 || x > 1.0) {
        return fail("[feature_bag_fraction] must be a double in (0, 1]");
    }
    if f("num_top_feature_importance_values").is_some_and(|x| x < 0.0) {
        return fail("[num_top_feature_importance_values] must be a non-negative integer");
    }
    if f("alpha").is_some_and(|x| x < 0.0) {
        return fail("[alpha] must be a non-negative double");
    }
    if f("downsample_factor").is_some_and(|x| x <= 0.0 || x > 1.0) {
        return fail("[downsample_factor] must be a double in (0, 1]");
    }
    if f("eta_growth_rate_per_tree").is_some_and(|x| !(0.5..=2.0).contains(&x)) {
        return fail("[eta_growth_rate_per_tree] must be a double in [0.5, 2]");
    }
    if f("max_optimization_rounds_per_hyperparameter").is_some_and(|x| !(0.0..=20.0).contains(&x)) {
        return fail("[max_optimization_rounds_per_hyperparameter] must be an integer in [0, 20]");
    }
    if f("soft_tree_depth_limit").is_some_and(|x| x < 0.0) {
        return fail("[soft_tree_depth_limit] must be a non-negative double");
    }
    if f("soft_tree_depth_tolerance").is_some_and(|x| x < 0.01) {
        return fail("[soft_tree_depth_tolerance] must be a double greater than or equal to 0.01");
    }
    Ok(())
}

fn str_map_f64(section: &str, name: &str, v: &Value) -> Result<Value> {
    let o = get_obj(section, name, v)?;
    let mut m = Map::new();
    for (k, x) in o {
        m.insert(k.clone(), dbl(get_f64(section, name, x)?));
    }
    Ok(Value::Object(m))
}

/// One `feature_processors` entry, normalized (`custom: true` added).
fn parse_processor(section: &str, v: &Value) -> Result<Value> {
    let wrap = |e: EsError| failed_field(section, "feature_processors", e);
    let o = get_obj(section, "feature_processors", v).map_err(wrap)?;
    let (name, body) = match o.iter().next() {
        Some((n, b)) if o.len() == 1 => (n.as_str(), b),
        _ => return Err(wrap(EsError::parsing("feature_processors must contain exactly one processor per entry"))),
    };
    let b = get_obj(name, name, body).map_err(wrap)?;
    let req = |k: &str| -> Result<&Value> { b.get(k).ok_or_else(|| wrap(EsError::illegal_argument(format!("Required [{k}]")))) };
    let mut m = Map::new();
    let allowed: &[&str] = match name {
        "n_gram_encoding" => &["field", "feature_prefix", "n_grams", "start", "length", "custom"],
        "one_hot_encoding" => &["field", "hot_map", "custom"],
        "frequency_encoding" => &["field", "feature_name", "frequency_map", "custom"],
        "target_mean_encoding" => &["field", "feature_name", "target_map", "default_value", "custom"],
        _ => return Err(wrap(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{name}]")))),
    };
    if let Some(k) = b.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(wrap(unknown(name, k)));
    }
    m.insert("field".into(), json!(get_str(name, "field", req("field")?).map_err(wrap)?));
    match name {
        "n_gram_encoding" => {
            m.insert("feature_prefix".into(), json!(get_str(name, "feature_prefix", req("feature_prefix")?).map_err(wrap)?));
            let grams: Vec<i64> = match req("n_grams")? {
                Value::Array(a) => a.iter().map(|x| get_i64(name, "n_grams", x)).collect::<Result<_>>().map_err(wrap)?,
                x => vec![get_i64(name, "n_grams", x).map_err(wrap)?],
            };
            if grams.is_empty() || grams.iter().any(|g| !(1..=5).contains(g)) {
                return Err(wrap(failed_build(name, status("[n_grams] is invalid [{}]; minimum supported value is [1]; maximum supported value is [5]".replace("{}", &grams.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(", "))))));
            }
            m.insert("n_grams".into(), json!(grams));
            let start = b.get("start").map(|x| get_i64(name, "start", x)).transpose().map_err(wrap)?.unwrap_or(0);
            let length = b.get("length").map(|x| get_i64(name, "length", x)).transpose().map_err(wrap)?.unwrap_or(50);
            if !(1..=100).contains(&length) {
                return Err(wrap(failed_build(name, status(format!("[length] must be a positive integer between 1 and 100; provided [{length}]")))));
            }
            m.insert("start".into(), json!(start));
            m.insert("length".into(), json!(length));
        }
        "one_hot_encoding" => {
            let hm = get_obj(name, "hot_map", req("hot_map")?).map_err(wrap)?;
            let mut out = Map::new();
            for (k, x) in hm {
                out.insert(k.clone(), json!(get_str(name, "hot_map", x).map_err(wrap)?));
            }
            m.insert("hot_map".into(), Value::Object(out));
        }
        "frequency_encoding" => {
            m.insert("feature_name".into(), json!(get_str(name, "feature_name", req("feature_name")?).map_err(wrap)?));
            m.insert("frequency_map".into(), str_map_f64(name, "frequency_map", req("frequency_map")?).map_err(wrap)?);
        }
        _ => {
            m.insert("feature_name".into(), json!(get_str(name, "feature_name", req("feature_name")?).map_err(wrap)?));
            m.insert("target_map".into(), str_map_f64(name, "target_map", req("target_map")?).map_err(wrap)?);
            m.insert("default_value".into(), dbl(get_f64(name, "default_value", req("default_value")?).map_err(wrap)?));
        }
    }
    m.insert("custom".into(), json!(true));
    Ok(json!({ name: m }))
}

fn parse_supervised(kind: &str, o: &Map<String, Value>) -> Result<Value> {
    let regression = kind == "regression";
    let mut dep: Option<String> = None;
    let mut btp = Map::new();
    let mut pfn: Option<String> = None;
    let mut tp = 100.0;
    let mut seed: Option<i64> = None;
    let mut loss: Option<String> = None;
    let mut loss_param: Option<f64> = None;
    let mut cao: Option<String> = None;
    let mut ntc: Option<i64> = None;
    let mut procs: Option<Vec<Value>> = None;
    let mut early = true;
    for (k, x) in o {
        let key = if k == "maximum_number_trees" { "max_trees" } else { k.as_str() };
        if let Some((_, int)) = BTP.iter().find(|(n, _)| *n == key) {
            btp.insert(key.to_string(), if *int { json!(get_i64(kind, k, x)?) } else { dbl(get_f64(kind, k, x)?) });
            continue;
        }
        match key {
            "dependent_variable" => dep = Some(get_str(kind, k, x)?),
            "prediction_field_name" => pfn = Some(get_str(kind, k, x)?),
            "training_percent" => tp = get_f64(kind, k, x)?,
            "randomize_seed" => seed = Some(get_i64(kind, k, x)?),
            "early_stopping_enabled" => early = get_bool(kind, k, x)?,
            "feature_processors" => {
                let a = x.as_array().ok_or_else(|| bad_type(kind, k, x))?;
                procs = Some(a.iter().map(|p| parse_processor(kind, p)).collect::<Result<_>>()?);
            }
            "loss_function" if regression => {
                let l = get_str(kind, k, x)?.to_lowercase();
                if !matches!(l.as_str(), "mse" | "msle" | "huber") {
                    return Err(enum_err(kind, k, "Regression.LossFunction", &l));
                }
                loss = Some(l);
            }
            "loss_function_parameter" if regression => loss_param = Some(get_f64(kind, k, x)?),
            "class_assignment_objective" if !regression => {
                let c = get_str(kind, k, x)?.to_lowercase();
                if !matches!(c.as_str(), "maximize_accuracy" | "maximize_minimum_recall") {
                    return Err(enum_err(kind, k, "Classification.ClassAssignmentObjective", &c));
                }
                cao = Some(c);
            }
            "num_top_classes" if !regression => ntc = Some(get_i64(kind, k, x)?),
            _ => return Err(unknown(kind, k)),
        }
    }
    let dep = dep.ok_or_else(|| EsError::illegal_argument("Required [dependent_variable]"))?;
    let built = (|| -> Result<()> {
        check_btp(&btp)?;
        if tp <= 0.0 || tp > 100.0 {
            return Err(status("[training_percent] must be a positive double in (0, 100]"));
        }
        if loss_param.is_some_and(|p| p <= 0.0) {
            return Err(status("[loss_function_parameter] must be a positive double"));
        }
        if ntc.is_some_and(|n| n != -1 && !(0..=1000).contains(&n)) {
            return Err(status("[num_top_classes] must be an integer in [0, 1000] or a special value -1"));
        }
        Ok(())
    })();
    built.map_err(|e| failed_build(kind, e))?;
    let mut m = Map::new();
    m.insert("dependent_variable".into(), json!(dep));
    for (k, _) in BTP {
        if let Some(v) = btp.remove(k) {
            m.insert(k.into(), v);
        }
    }
    if !regression {
        m.insert("class_assignment_objective".into(), json!(cao.unwrap_or_else(|| "maximize_minimum_recall".into())));
        m.insert("num_top_classes".into(), json!(ntc.unwrap_or(2)));
    }
    m.insert("prediction_field_name".into(), json!(pfn.unwrap_or_else(|| format!("{dep}_prediction"))));
    m.insert("training_percent".into(), dbl(tp));
    m.insert("randomize_seed".into(), json!(seed.unwrap_or_else(random_seed)));
    if regression {
        m.insert("loss_function".into(), json!(loss.unwrap_or_else(|| "mse".into())));
        if let Some(p) = loss_param {
            m.insert("loss_function_parameter".into(), dbl(p));
        }
    }
    if let Some(p) = procs {
        m.insert("feature_processors".into(), Value::Array(p));
    }
    m.insert("early_stopping_enabled".into(), json!(early));
    Ok(Value::Object(m))
}

fn parse_analysis(v: &Value) -> Result<Value> {
    let wrap = |e: EsError| failed_field(CONFIG, "analysis", e);
    let o = get_obj(CONFIG, "analysis", v)?;
    let mut it = o.iter();
    let (name, body) = match it.next() {
        Some(x) => x,
        None => return Err(wrap(EsError::parsing_at("Failed to parse object: expecting token of type [FIELD_NAME] but found [END_OBJECT]", 1, 1))),
    };
    if it.next().is_some() {
        return Err(wrap(EsError::parsing_at("Failed to parse object: expecting token of type [END_OBJECT] but found [FIELD_NAME]", 1, 1)));
    }
    if !matches!(name.as_str(), "outlier_detection" | "regression" | "classification") {
        return Err(wrap(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{name}]"))));
    }
    let b = get_obj(name, name, body).map_err(wrap)?;
    let inner = if name == "outlier_detection" { parse_outlier(b) } else { parse_supervised(name, b) }.map_err(wrap)?;
    Ok(json!({ name.clone(): inner }))
}

/// Parse a config body (`PUT` or the `_explain` / `_preview` config).
pub fn parse(body: &Value, url_id: Option<&str>, mode: Mode) -> Result<Parsed> {
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        _ => return Err(xcp(format!("[1:1] [{CONFIG}] Expected START_OBJECT but was: {}", token(body)))),
    };
    let mut id: Option<String> = None;
    let mut parts: Map<String, Value> = Map::new();
    let mut mml: Option<(u64, String)> = None;
    let mut threads: Option<i64> = None;
    for (k, x) in o {
        match k.as_str() {
            "id" => id = Some(get_str(CONFIG, k, x)?),
            "description" => {
                match x {
                    Value::String(s) => {
                        parts.insert(k.clone(), json!(s));
                    }
                    Value::Null => {}
                    _ => return Err(bad_type(CONFIG, k, x)),
                }
            }
            "source" => {
                parts.insert(k.clone(), parse_source(x)?);
            }
            "dest" => {
                parts.insert(k.clone(), parse_dest(x)?);
            }
            "analysis" => {
                parts.insert(k.clone(), parse_analysis(x)?);
            }
            "analyzed_fields" => {
                parts.insert(k.clone(), fetch_source(CONFIG, k, x)?);
            }
            "model_memory_limit" => mml = Some(parse_mml(x, CONFIG)?),
            "allow_lazy_start" => {
                parts.insert(k.clone(), json!(get_bool(CONFIG, k, x)?));
            }
            "max_num_threads" => threads = Some(get_i64(CONFIG, k, x)?),
            "_meta" => {
                parts.insert(k.clone(), Value::Object(get_obj(CONFIG, k, x)?.clone()));
            }
            _ => return Err(unknown(CONFIG, k)),
        }
    }
    if let (Some(b), Some(u)) = (&id, url_id) {
        if b != u {
            return Err(EsError::illegal_argument(format!("Inconsistent id; '{b}' specified in the body differs from '{u}' specified as a URL argument")));
        }
    }
    let id = url_id.map(str::to_string).or(id).unwrap_or_else(|| "dummy".into());
    if !parts.contains_key("source") {
        return Err(EsError::illegal_argument("[source] must not be null."));
    }
    if !parts.contains_key("dest") {
        if mode == Mode::Put {
            return Err(EsError::illegal_argument("[dest] must not be null."));
        }
        parts.insert("dest".into(), json!({"index": "", "results_field": "ml"}));
    }
    if !parts.contains_key("analysis") {
        return Err(EsError::illegal_argument("[analysis] must not be null."));
    }
    if let Some((b, r)) = &mml {
        mml_min_check(*b, r)?;
    }
    if threads.is_some_and(|t| t < 1) {
        return Err(status("[max_num_threads] must be a positive integer"));
    }
    let mut m = Map::new();
    m.insert("id".into(), json!(id));
    for k in ["description", "source", "dest", "analysis", "analyzed_fields"] {
        if let Some(v) = parts.remove(k) {
            m.insert(k.into(), v);
        }
    }
    m.insert("model_memory_limit".into(), json!(mml.as_ref().map(|(_, r)| r.clone()).unwrap_or_else(|| DEFAULT_MML.into())));
    m.insert("allow_lazy_start".into(), parts.remove("allow_lazy_start").unwrap_or(json!(false)));
    m.insert("max_num_threads".into(), json!(threads.unwrap_or(1)));
    if let Some(meta) = parts.remove("_meta") {
        m.insert("_meta".into(), meta);
    }
    Ok(Parsed { config: m, mml_explicit: mml.is_some() })
}

/// `MetadataCreateIndexService.validateIndexOrAliasName` message, if invalid.
fn invalid_index_name(name: &str) -> Option<String> {
    let why = if name.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ',')) {
        "must not contain the following characters [' ','\"','*',',','/','<','>','?','\\','|']".to_string()
    } else if name.is_empty() {
        "must not be empty".into()
    } else if name.contains('#') {
        "must not contain '#'".into()
    } else if name.contains(':') {
        "must not contain ':'".into()
    } else if name.starts_with(['_', '-', '+']) {
        "must not start with '_', '-', or '+'".into()
    } else if name.len() > 255 {
        format!("index name is too long, ({} > 255)", name.len())
    } else if name == "." || name == ".." {
        "must not be '.' or '..'".into()
    } else {
        return None;
    };
    Some(format!("Invalid index name [{name}], {why}"))
}

/// `PutDataFrameAnalyticsAction.Request.validate()`.
pub fn validate_put(cfg: &Map<String, Value>) -> Result<()> {
    let mut errs: Vec<String> = Vec::new();
    let id = cfg["id"].as_str().unwrap_or("");
    if id.len() > crate::ml::ID_LENGTH_LIMIT {
        errs.push(format!("Invalid id; '{id}' cannot contain more than {} characters.", crate::ml::ID_LENGTH_LIMIT));
    } else if !crate::ml::is_valid_id(id) {
        errs.push(format!("Invalid id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"));
    }
    let dest = cfg["dest"]["index"].as_str().unwrap_or("");
    match invalid_index_name(dest) {
        Some(e) => errs.push(e),
        None if dest.to_lowercase() != dest => errs.push(format!("Destination index [{dest}] must be lowercase")),
        None => {}
    }
    if let (Some(af), Some(src)) = (cfg.get("analyzed_fields"), cfg["source"].get("_source")) {
        for inc in af["includes"].as_array().into_iter().flatten().filter_map(Value::as_str) {
            if is_field_excluded(src, inc) {
                errs.push(format!("field [{inc}] is included in [analyzed_fields] but not in [source._source]"));
                break;
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        Err(EsError::action_validation(&refs))
    }
}

/// Apply `xpack.ml.max_model_memory_limit` (`Builder(config, maxModelMemoryLimit)`):
/// an explicit limit above it fails, the default is capped to it.
pub fn apply_max_mml(cfg: &mut Map<String, Value>, explicit: bool, max: Option<(u64, String)>) -> Result<()> {
    let Some((max_b, max_s)) = max.filter(|(b, _)| *b > 0) else { return Ok(()) };
    let cur = cfg["model_memory_limit"].as_str().unwrap_or(DEFAULT_MML).to_string();
    let (b, _) = crate::ilm::parse_size(&cur, "model_memory_limit")?;
    if b > max_b {
        if explicit {
            return Err(status(format!("model_memory_limit [{cur}] must be less than the value of the xpack.ml.max_model_memory_limit setting [{max_s}]")));
        }
        cfg.insert("model_memory_limit".into(), json!(max_s));
    }
    Ok(())
}

/// A parsed `_update` body.
pub struct Update {
    pub description: Option<String>,
    pub mml: Option<(u64, String)>,
    pub allow_lazy_start: Option<bool>,
    pub max_num_threads: Option<i64>,
    pub meta: Option<Value>,
}

/// Parse a `DataFrameAnalyticsConfigUpdate` body.
pub fn parse_update(body: &Value, url_id: &str) -> Result<Update> {
    let empty = Map::new();
    let o = match body {
        Value::Object(o) => o,
        Value::Null => &empty,
        _ => return Err(xcp(format!("[1:1] [{UPDATE}] Expected START_OBJECT but was: {}", token(body)))),
    };
    let mut u = Update { description: None, mml: None, allow_lazy_start: None, max_num_threads: None, meta: None };
    for (k, x) in o {
        match k.as_str() {
            "id" => {
                let b = get_str(UPDATE, k, x)?;
                if b != url_id {
                    return Err(EsError::illegal_argument(format!("Inconsistent id; '{b}' specified in the body differs from '{url_id}' specified as a URL argument")));
                }
            }
            "description" => u.description = Some(get_str(UPDATE, k, x)?),
            "model_memory_limit" => u.mml = Some(parse_mml(x, UPDATE)?),
            "allow_lazy_start" => u.allow_lazy_start = Some(get_bool(UPDATE, k, x)?),
            "max_num_threads" => u.max_num_threads = Some(get_i64(UPDATE, k, x)?),
            "_meta" => u.meta = Some(Value::Object(get_obj(UPDATE, k, x)?.clone())),
            _ => return Err(unknown(UPDATE, k)),
        }
    }
    if u.max_num_threads.is_some_and(|t| t < 1) {
        return Err(status("[max_num_threads] must be a positive integer"));
    }
    if let Some((b, r)) = &u.mml {
        mml_min_check(*b, r)?;
    }
    Ok(u)
}

/// Re-order a config's top-level keys into ES's rendering order.
pub fn canonical(m: Map<String, Value>) -> Map<String, Value> {
    let mut m = m;
    let mut out = Map::new();
    for k in ORDER {
        if let Some(v) = m.remove(k) {
            out.insert(k.into(), v);
        }
    }
    out.extend(m);
    out
}

/// Analysis type (`outlier_detection` / `regression` / `classification`).
pub fn analysis_kind(cfg: &Value) -> &str {
    cfg["analysis"].as_object().and_then(|o| o.keys().next()).map(String::as_str).unwrap_or("")
}

/// The analysis' dependent variable (supervised analyses).
pub fn dependent_variable(cfg: &Value) -> Option<&str> {
    cfg["analysis"].as_object().and_then(|o| o.values().next()).and_then(|a| a["dependent_variable"].as_str())
}
