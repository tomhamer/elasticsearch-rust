//! Transform config parsing and validation (ES `TransformConfig` and
//! friends): strict parsing of every section, the request-level checks of
//! `PutTransformAction.Request#validate` (id, frequency, page sizes,
//! settings ranges, latest/pivot field rules) and the normalized JSON
//! rendering used by GET.

use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

pub const MAX_ID_LEN: usize = 64;
pub const DEFAULT_PAGE_SIZE: usize = 500;
pub const DEFAULT_FREQUENCY_MS: i64 = 60_000;
pub const DEFAULT_DELAY: &str = "60s";

/// Aggregation types a pivot accepts (ES `TransformAggregations`).
pub const SUPPORTED_AGGS: &[&str] = &[
    "avg", "cardinality", "geo_bounds", "geo_centroid", "geo_line", "max", "min", "sum", "value_count", "scripted_metric", "weighted_avg", "bucket_selector",
    "bucket_script", "percentiles", "filter", "terms", "rare_terms", "missing", "range", "filters", "top_metrics", "stats", "boxplot", "median_absolute_deviation",
    "extended_stats", "date_range", "multi_terms",
];

#[derive(Clone, Debug)]
pub struct GroupSource {
    pub name: String,
    /// `terms` / `histogram` / `date_histogram` / `geotile_grid`.
    pub kind: String,
    pub params: Map<String, Value>,
}

impl GroupSource {
    pub fn field(&self) -> Option<&str> {
        self.params.get("field").and_then(Value::as_str)
    }
}

#[derive(Clone, Debug)]
pub enum Function {
    Pivot { groups: Vec<GroupSource>, aggs: Map<String, Value>, max_page_search_size: Option<usize> },
    Latest { unique_key: Vec<String>, sort: String },
}

#[derive(Clone, Debug, Default)]
pub struct Settings {
    pub max_page_search_size: Option<usize>,
    pub docs_per_second: Option<f64>,
    pub dates_as_epoch_millis: bool,
    pub align_checkpoints: bool,
    pub deduce_mappings: bool,
    pub num_failure_retries: Option<i64>,
    pub unattended: bool,
}

/// A parsed config (the raw normalized JSON is kept alongside).
#[derive(Clone, Debug)]
pub struct Config {
    pub id: String,
    pub source_indices: Vec<String>,
    pub query: Value,
    pub runtime_mappings: Option<Value>,
    pub dest_index: Option<String>,
    pub dest_pipeline: Option<String>,
    pub dest_aliases: Vec<Value>,
    pub frequency_ms: i64,
    pub sync: Option<(String, i64)>,
    pub retention: Option<(String, i64)>,
    pub function: Function,
    pub settings: Settings,
}

impl Config {
    pub fn is_continuous(&self) -> bool {
        self.sync.is_some()
    }
    pub fn page_size(&self) -> usize {
        self.settings.max_page_search_size.or(match &self.function {
            Function::Pivot { max_page_search_size, .. } => *max_page_search_size,
            _ => None,
        })
        .unwrap_or(DEFAULT_PAGE_SIZE)
    }
    pub fn kind(&self) -> &'static str {
        match self.function {
            Function::Pivot { .. } => "pivot",
            Function::Latest { .. } => "latest",
        }
    }
}

fn unknown(section: &str, k: &str) -> EsError {
    EsError::x_content_parse(format!("[1:1] [{section}] unknown field [{k}]"))
}

fn check_keys(m: &Map<String, Value>, section: &str, allowed: &[&str]) -> Result<()> {
    for k in m.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(unknown(section, k));
        }
    }
    Ok(())
}

fn obj<'a>(v: &'a Value, section: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| EsError::x_content_parse(format!("[1:1] [{section}] failed to parse object")))
}

/// ES time value (`60s`, `999ms`, `1h`) in millis.
pub fn time_value(v: &Value, setting: &str) -> Result<i64> {
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => return Err(EsError::x_content_parse(format!("[1:1] failed to parse field [{setting}]"))),
    };
    crate::date::parse_time_value(&s).filter(|_| s.trim().chars().last().map(|c| c.is_ascii_alphabetic()).unwrap_or(false) || s == "-1" || s == "0").ok_or_else(|| {
        EsError::new(400, "x_content_parse_exception", format!("[1:1] [{setting}] failed to parse field")).caused_by(EsError::illegal_argument(format!(
            "failed to parse setting [{setting}] with value [{s}] as a time value: unit is missing or unrecognized"
        )))
    })
}

pub fn validate_id(id: &str) -> Result<()> {
    let mut errs: Vec<String> = Vec::new();
    let ok = !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.')
        && id.as_bytes()[0].is_ascii_alphanumeric()
        && id.as_bytes()[id.len() - 1].is_ascii_alphanumeric();
    if !ok {
        errs.push(format!("Invalid transform id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"));
    }
    if id.len() > MAX_ID_LEN {
        errs.push(format!("The id cannot contain more than {MAX_ID_LEN} characters."));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        Err(EsError::action_validation(&refs))
    }
}

/// Destination index name rules (ES `MetadataCreateIndexService`).
pub fn validate_dest_name(name: &str) -> Option<String> {
    if name != name.to_lowercase() {
        return Some(format!("Destination index [{name}] must be lowercase"));
    }
    for c in ['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#', ':'] {
        if name.contains(c) {
            if c == '#' {
                return Some(format!("Invalid index name [{name}], must not contain '#'"));
            }
            return Some(format!("Invalid index name [{name}], must not contain the following characters [' ','\"','*',',','/','<','>','?','\\\\','|']"));
        }
    }
    if name.starts_with(['-', '_', '+']) {
        return Some(format!("Invalid index name [{name}], must not start with '_', '-', or '+'"));
    }
    None
}

fn list_of_strings(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => vec![],
    }
}

const GROUP_PARAMS: &[(&str, &str, &[&str])] = &[
    ("terms", "data_frame_terms_group", &["field", "script", "missing_bucket"]),
    ("histogram", "data_frame_histogram_group", &["field", "script", "missing_bucket", "interval"]),
    (
        "date_histogram",
        "data_frame_date_histogram_group",
        &["field", "script", "missing_bucket", "calendar_interval", "fixed_interval", "interval", "time_zone", "offset", "format"],
    ),
    ("geotile_grid", "data_frame_geotile_group", &["field", "script", "missing_bucket", "precision", "bounds"]),
];

fn parse_groups(v: &Value) -> Result<Vec<GroupSource>> {
    let m = obj(v, "group_by")?;
    if m.is_empty() {
        return Err(EsError::x_content_parse("[1:1] [data_frame_transform_pivot] failed to parse field [group_by]").caused_by(EsError::illegal_argument("Group by is empty")));
    }
    let mut out = Vec::new();
    for (name, def) in m {
        let d = obj(def, "group_by")?;
        if d.len() != 1 {
            return Err(EsError::x_content_parse(format!("[1:1] Failed to parse group_by [{name}]")));
        }
        let (kind, params) = d.iter().next().unwrap();
        let Some((_, section, allowed)) = GROUP_PARAMS.iter().find(|(k, _, _)| k == kind) else {
            return Err(EsError::x_content_parse(format!("[1:1] Unknown group_by type [{kind}]")));
        };
        let p = obj(params, section)?;
        check_keys(p, section, allowed)?;
        if !p.contains_key("field") && !p.contains_key("script") {
            return Err(EsError::action_validation(&["Required one of fields [field, script], but none were specified. "]));
        }
        out.push(GroupSource { name: name.clone(), kind: kind.clone(), params: p.clone() });
    }
    Ok(out)
}

fn int_setting(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Field-name checks over the pivot output (ES `PivotConfig#aggFieldValidation`).
fn agg_field_validation(groups: &[GroupSource], aggs: &Map<String, Value>) -> Vec<String> {
    let mut names: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();
    fn collect(prefix: &str, aggs: &Map<String, Value>, out: &mut Vec<String>) {
        for (k, v) in aggs {
            let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            out.push(full.clone());
            if let Some(sub) = v.get("aggs").or_else(|| v.get("aggregations")).and_then(Value::as_object) {
                collect(&full, sub, out);
            }
        }
    }
    let mut agg_names = Vec::new();
    collect("", aggs, &mut agg_names);
    // Sub-aggregation names nest under their parent: only top level and
    // group names compete with each other directly.
    names.extend(agg_names.into_iter().filter(|n| aggs.contains_key(n.as_str()) || !n.contains('.')));
    names.sort();
    let mut errs = Vec::new();
    for w in names.windows(2) {
        if w[0] == w[1] {
            errs.push(format!("duplicate field [{}] detected", w[0]));
        } else if w[1].starts_with(&format!("{}.", w[0])) {
            errs.push(format!("field [{}] cannot be both an object and a field", w[0]));
        }
    }
    errs.dedup();
    errs
}

fn check_agg_types(aggs: &Map<String, Value>) -> Result<()> {
    for (_, def) in aggs {
        let Some(o) = def.as_object() else { continue };
        for (k, v) in o {
            if matches!(k.as_str(), "aggs" | "aggregations") {
                if let Some(sub) = v.as_object() {
                    check_agg_types(sub)?;
                }
                continue;
            }
            if k == "meta" {
                continue;
            }
            if !SUPPORTED_AGGS.contains(&k.as_str()) {
                return Err(EsError::illegal_argument(format!("Unsupported aggregation type [{k}]")));
            }
        }
    }
    Ok(())
}

/// Parse a request body. `id` is the path id (None for previews).
/// `for_put` rejects the generated fields (`create_time`, `version`).
pub fn parse(id: Option<&str>, body: &Value, for_put: bool, require_dest: bool) -> Result<Config> {
    let o = obj(body, "data_frame_transform_config")?;
    for k in o.keys() {
        match k.as_str() {
            "source" | "dest" | "frequency" | "sync" | "pivot" | "latest" | "description" | "settings" | "_meta" | "retention_policy" | "id" => {}
            "create_time" | "version" | "authorization" if for_put => {
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:1] [data_frame_transform_config] failed to parse field [{k}]"))
                    .caused_by(EsError::illegal_argument(format!("Found [{k}], not allowed for strict parsing"))));
            }
            "create_time" | "version" | "authorization" => {}
            other => return Err(unknown("data_frame_transform_config", other)),
        }
    }
    if let (Some(pid), Some(bid)) = (id, o.get("id").and_then(Value::as_str)) {
        if pid != bid {
            return Err(EsError::illegal_argument(format!("Inconsistent id; '{bid}' specified in the body differs from '{pid}' specified as a URL argument")));
        }
    }
    let id = id.map(str::to_string).or_else(|| o.get("id").and_then(Value::as_str).map(str::to_string)).unwrap_or_else(|| "transform-preview".into());
    // Required fields (ObjectParser reports every missing one at once).
    let missing: Vec<&str> = ["source", "dest"].into_iter().filter(|k| (*k == "source" || require_dest) && !o.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    // source
    let src = &o["source"];
    let so = obj(src, "data_frame_config_source")?;
    check_keys(so, "data_frame_config_source", &["index", "query", "runtime_mappings", "_source"])?;
    if !so.contains_key("index") {
        return Err(EsError::x_content_parse("[1:1] [data_frame_transform_config] failed to parse field [source]").caused_by(EsError::illegal_argument("Required [index]")));
    }
    let source_indices = so.get("index").map(list_of_strings).unwrap_or_default();
    if source_indices.is_empty() {
        return Err(EsError::action_validation(&["Source index is missing"]));
    }
    let query = so.get("query").cloned().unwrap_or_else(|| json!({"match_all": {}}));
    let runtime_mappings = so.get("runtime_mappings").cloned();
    // dest
    let (mut dest_index, mut dest_pipeline, mut dest_aliases) = (None, None, Vec::new());
    match o.get("dest") {
        Some(d) => {
            let dm = obj(d, "data_frame_config_dest")?;
            check_keys(dm, "data_frame_config_dest", &["index", "pipeline", "aliases"])?;
            dest_index = dm.get("index").and_then(Value::as_str).map(str::to_string);
            dest_pipeline = dm.get("pipeline").and_then(Value::as_str).map(str::to_string);
            dest_aliases = dm.get("aliases").and_then(Value::as_array).cloned().unwrap_or_default();
            // Preview takes a `dest` without `index` (e.g. only a pipeline).
            if require_dest && !dm.contains_key("index") {
                return Err(EsError::x_content_parse("[1:1] [data_frame_transform_config] failed to parse field [dest]").caused_by(EsError::illegal_argument("Required [index]")));
            }
        }
        None => {}
    }
    let mut errs: Vec<String> = Vec::new();
    // frequency
    let frequency_ms = match o.get("frequency") {
        Some(v) if !v.is_null() => {
            let ms = time_value(v, "frequency")?;
            if ms < 1000 {
                errs.push("minimum permitted [frequency] is [1s]".into());
            } else if ms > 3_600_000 {
                errs.push("highest permitted [frequency] is [1h]".into());
            }
            ms
        }
        _ => DEFAULT_FREQUENCY_MS,
    };
    // sync
    let sync = match o.get("sync") {
        Some(Value::Null) | None => None,
        Some(s) => {
            let sm = obj(s, "sync")?;
            let t = sm.get("time").ok_or_else(|| EsError::x_content_parse("[1:1] unknown sync type"))?;
            let tm = obj(t, "data_frame_transform_pivot_sync_config")?;
            check_keys(tm, "time_sync_config", &["field", "delay"])?;
            let f = tm.get("field").and_then(Value::as_str).ok_or_else(|| EsError::x_content_parse("[1:1] Required [field]"))?.to_string();
            let d = match tm.get("delay") {
                Some(v) => time_value(v, "delay")?,
                None => 60_000,
            };
            Some((f, d))
        }
    };
    let retention = match o.get("retention_policy") {
        Some(Value::Null) | None => None,
        Some(r) => {
            let rm = obj(r, "retention_policy")?;
            let t = rm.get("time").ok_or_else(|| EsError::x_content_parse("[1:1] unknown retention policy type"))?;
            let tm = obj(t, "time_retention_policy_config")?;
            check_keys(tm, "time_retention_policy_config", &["field", "max_age"])?;
            let f = tm.get("field").and_then(Value::as_str).ok_or_else(|| EsError::x_content_parse("[1:1] Required [field]"))?.to_string();
            let age = time_value(tm.get("max_age").ok_or_else(|| EsError::x_content_parse("[1:1] Required [max_age]"))?, "max_age")?;
            if age < 60_000 {
                errs.push("retention_policy.time.max_age must be greater than 60s, found [".to_string() + &age.to_string() + "ms]");
            }
            Some((f, age))
        }
    };
    // settings
    let mut settings = Settings { align_checkpoints: true, deduce_mappings: true, ..Default::default() };
    if let Some(s) = o.get("settings") {
        let sm = obj(s, "transform_config_settings")?;
        check_keys(
            sm,
            "transform_config_settings",
            &["max_page_search_size", "docs_per_second", "dates_as_epoch_millis", "align_checkpoints", "interim_results", "deduce_mappings", "num_failure_retries", "unattended", "use_point_in_time"],
        )?;
        if let Some(v) = sm.get("max_page_search_size").filter(|v| !v.is_null()) {
            let n = int_setting(v).unwrap_or(-1);
            if !(10..=65_536).contains(&n) {
                errs.push(format!("settings.max_page_search_size [{n}] is out of range. The minimum value is 10 and the maximum is 65536"));
            }
            settings.max_page_search_size = Some(n.max(0) as usize);
        }
        if let Some(v) = sm.get("num_failure_retries").filter(|v| !v.is_null()) {
            let n = int_setting(v).unwrap_or(0);
            if !(-1..=100).contains(&n) {
                errs.push(format!("settings.num_failure_retries [{n}] is out of range. The minimum value is -1 (infinity) and the maximum is 100"));
            }
            settings.num_failure_retries = Some(n);
        }
        settings.docs_per_second = sm.get("docs_per_second").and_then(Value::as_f64);
        let b = |k: &str, d: bool| sm.get(k).and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(d);
        settings.dates_as_epoch_millis = b("dates_as_epoch_millis", false);
        settings.align_checkpoints = b("align_checkpoints", true);
        settings.deduce_mappings = b("deduce_mappings", true);
        settings.unattended = b("unattended", false);
    }
    // function
    let function = match (o.get("pivot"), o.get("latest")) {
        (Some(p), None) => {
            let pm = obj(p, "data_frame_transform_pivot")?;
            check_keys(pm, "data_frame_transform_pivot", &["group_by", "aggregations", "aggs", "max_page_search_size"])?;
            let groups = parse_groups(pm.get("group_by").ok_or_else(|| EsError::x_content_parse("[1:1] Required [group_by]"))?)?;
            let aggs = pm
                .get("aggregations")
                .or_else(|| pm.get("aggs"))
                .and_then(Value::as_object)
                .cloned()
                .ok_or_else(|| EsError::x_content_parse("[1:1] Required [aggregations]"))?;
            let mps = match pm.get("max_page_search_size").filter(|v| !v.is_null()) {
                Some(v) => {
                    let n = int_setting(v).unwrap_or(-1);
                    if !(10..=65_536).contains(&n) {
                        errs.push(format!("pivot.max_page_search_size [{n}] is out of range. The minimum value is 10 and the maximum is 65536"));
                    }
                    Some(n.max(0) as usize)
                }
                None => None,
            };
            errs.extend(agg_field_validation(&groups, &aggs));
            if errs.is_empty() {
                check_agg_types(&aggs)?;
            }
            Function::Pivot { groups, aggs, max_page_search_size: mps }
        }
        (None, Some(l)) => {
            let lm = obj(l, "latest_config")?;
            check_keys(lm, "latest_config", &["unique_key", "sort"])?;
            let unique_key = lm.get("unique_key").map(list_of_strings).unwrap_or_default();
            let raw_keys: Vec<String> = lm.get("unique_key").and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()).unwrap_or(unique_key.clone());
            let sort = lm.get("sort").and_then(Value::as_str).unwrap_or("").to_string();
            if raw_keys.is_empty() {
                errs.push("latest.unique_key must be non-empty".into());
            } else {
                let mut seen = std::collections::HashSet::new();
                for (i, k) in raw_keys.iter().enumerate() {
                    if k.is_empty() {
                        errs.push(format!("latest.unique_key[{i}] element must be non-empty"));
                    } else if !seen.insert(k.clone()) {
                        errs.push(format!("latest.unique_key elements must be unique, found duplicate element [{k}]"));
                    }
                }
            }
            if sort.is_empty() {
                errs.push("latest.sort must be non-empty".into());
            }
            Function::Latest { unique_key: raw_keys, sort }
        }
        _ => return Err(EsError::x_content_parse("[1:1] Transform configuration must specify exactly 1 function")),
    };
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    Ok(Config { id, source_indices, query, runtime_mappings, dest_index, dest_pipeline, dest_aliases, frequency_ms, sync, retention, function, settings })
}

/// Normalized stored config JSON (ES `TransformConfig#toXContent` order).
pub fn normalize(body: &Value, cfg: &Config, version: &str, create_time: Option<i64>) -> Value {
    let o = body.as_object().cloned().unwrap_or_default();
    let mut m = Map::new();
    m.insert("id".into(), json!(cfg.id));
    let mut src = Map::new();
    src.insert("index".into(), json!(cfg.source_indices));
    src.insert("query".into(), cfg.query.clone());
    if let Some(r) = &cfg.runtime_mappings {
        src.insert("runtime_mappings".into(), r.clone());
    }
    m.insert("source".into(), Value::Object(src));
    let mut dest = Map::new();
    if let Some(d) = &cfg.dest_index {
        dest.insert("index".into(), json!(d));
    }
    if !cfg.dest_aliases.is_empty() {
        dest.insert("aliases".into(), json!(cfg.dest_aliases));
    }
    if let Some(p) = &cfg.dest_pipeline {
        dest.insert("pipeline".into(), json!(p));
    }
    m.insert("dest".into(), Value::Object(dest));
    if let Some(f) = o.get("frequency").filter(|v| !v.is_null()) {
        m.insert("frequency".into(), f.clone());
    }
    if let Some((f, _)) = &cfg.sync {
        let delay = o.get("sync").and_then(|s| s.get("time")).and_then(|t| t.get("delay")).cloned().unwrap_or_else(|| json!(DEFAULT_DELAY));
        m.insert("sync".into(), json!({"time": {"field": f, "delay": delay}}));
    }
    match o.get("pivot") {
        Some(p) => {
            let mut pm = Map::new();
            pm.insert("group_by".into(), p.get("group_by").cloned().unwrap_or(Value::Null));
            pm.insert("aggregations".into(), p.get("aggregations").or_else(|| p.get("aggs")).cloned().unwrap_or(Value::Null));
            if let Some(x) = p.get("max_page_search_size").filter(|v| !v.is_null()) {
                pm.insert("max_page_search_size".into(), x.clone());
            }
            m.insert("pivot".into(), Value::Object(pm));
        }
        None => {
            if let Function::Latest { unique_key, sort } = &cfg.function {
                m.insert("latest".into(), json!({"unique_key": unique_key, "sort": sort}));
            }
        }
    }
    if let Some(d) = o.get("description").filter(|v| !v.is_null()) {
        m.insert("description".into(), d.clone());
    }
    m.insert("settings".into(), o.get("settings").cloned().filter(Value::is_object).unwrap_or_else(|| json!({})));
    if let Some(meta) = o.get("_meta").filter(|v| !v.is_null()) {
        m.insert("_meta".into(), meta.clone());
    }
    if let Some(r) = o.get("retention_policy").filter(|v| !v.is_null()) {
        m.insert("retention_policy".into(), r.clone());
    }
    m.insert("version".into(), json!(version));
    if let Some(ct) = create_time {
        m.insert("create_time".into(), json!(ct));
    }
    m.insert("authorization".into(), json!({"roles": ["superuser"]}));
    Value::Object(m)
}

/// The body of a stored config for re-parsing (generated fields removed).
pub fn strip_generated(stored: &Value) -> Value {
    let mut v = stored.clone();
    if let Some(o) = v.as_object_mut() {
        for k in ["version", "create_time", "authorization"] {
            o.remove(k);
        }
    }
    v
}

/// Warnings ES attaches to continuous transforms over runtime fields.
pub fn runtime_warnings(cfg: &Config) -> Vec<String> {
    let mut out = Vec::new();
    let Some((sync_field, _)) = &cfg.sync else { return out };
    let rt = cfg.runtime_mappings.as_ref().and_then(Value::as_object);
    let is_rt = |f: &str| rt.map(|m| m.get(f).map(|d| d.get("script").is_some()).unwrap_or(false)).unwrap_or(false);
    if let Function::Pivot { groups, .. } = &cfg.function {
        if !groups.is_empty() && groups.iter().all(|g| g.field().map(is_rt).unwrap_or(true)) {
            out.push("all the group-by fields are script-based runtime fields, this transform might run slowly, please check your configuration.".to_string());
        }
    }
    if is_rt(sync_field) {
        out.push("sync time field is a script-based runtime field, this transform might run slowly, please check your configuration.".to_string());
    }
    out
}
