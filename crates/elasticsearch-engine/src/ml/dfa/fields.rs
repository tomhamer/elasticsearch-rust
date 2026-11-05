//! Field selection (ES `ExtractedFieldsDetector`), data checks, `_explain`
//! memory estimation and `_preview` feature extraction.

use super::config::{analysis_kind, dependent_variable, status};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

const NUMERICAL: [&str; 9] = ["byte", "double", "float", "half_float", "integer", "long", "scaled_float", "short", "unsigned_long"];
const DISCRETE: [&str; 5] = ["byte", "integer", "long", "short", "unsigned_long"];
const CATEGORICAL: [&str; 3] = ["ip", "keyword", "text"];
/// `Classification.MAX_DEPENDENT_VARIABLE_CARDINALITY`.
const MAX_CLASSES: u64 = 100;

fn is_numeric(t: &str) -> bool {
    NUMERICAL.contains(&t) || t == "boolean"
}

/// Field types an analysis can use, sorted as ES lists them.
fn supported_types(kind: &str) -> Vec<&'static str> {
    let mut v: Vec<&str> = NUMERICAL.to_vec();
    v.push("boolean");
    if kind != "outlier_detection" {
        v.extend(CATEGORICAL);
    }
    v.sort();
    v
}

/// One mapped field of the source indices (from field caps).
#[derive(Clone, Debug)]
pub struct FieldInfo {
    pub name: String,
    pub types: Vec<String>,
}

/// The fields picked for an analysis.
pub struct Selection {
    /// `field_selection` entries of `_explain`, sorted by name.
    pub entries: Vec<Value>,
    /// Included fields: (name, types, feature_type).
    pub included: Vec<(String, Vec<String>, &'static str)>,
    /// Required fields (the dependent variable).
    pub required: Vec<String>,
}

fn source_indices(cfg: &Value) -> Vec<String> {
    cfg["source"]["index"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// `[a, b]` (Java `Arrays.toString` of the source indices).
pub fn source_list(cfg: &Value) -> String {
    format!("[{}]", source_indices(cfg).join(", "))
}

fn source_expr(cfg: &Value) -> String {
    source_indices(cfg).join(",")
}

/// Search the source indices (with the source's runtime mappings).
pub fn search_source(node: &Node, cfg: &Value, mut body: Value) -> Result<Value> {
    if let Some(rt) = cfg["source"].get("runtime_mappings").filter(|v| v.is_object()) {
        body["runtime_mappings"] = rt.clone();
    }
    crate::ml::search(node, &source_expr(cfg), &body)
}

/// Mapped fields of the source indices, minus metadata, objects and the
/// results field.
pub fn field_infos(node: &Node, cfg: &Value) -> Result<Vec<FieldInfo>> {
    let targets = node.resolve(&source_expr(cfg), ResolveOpts { allow_no_indices: true, ..Default::default() })?;
    let fields = vec!["*".to_string()];
    let filters = vec!["-metadata".to_string()];
    let caps = crate::field_caps::field_caps(
        &targets,
        &crate::field_caps::FieldCapsOpts {
            fields: &fields,
            include_unmapped: false,
            filters: &filters,
            types: &[],
            runtime_mappings: cfg["source"].get("runtime_mappings").filter(|v| v.is_object()),
            include_empty_fields: true,
        },
    )?;
    let results = cfg["dest"]["results_field"].as_str().unwrap_or("ml");
    let mut out = Vec::new();
    for (name, by_type) in caps["fields"].as_object().into_iter().flatten() {
        if name.starts_with('_') {
            continue;
        }
        if !results.is_empty() && (name == results || name.starts_with(&format!("{results}."))) {
            continue;
        }
        let mut types: Vec<String> = by_type.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
        types.retain(|t| t != "unmapped");
        if types.is_empty() || types.iter().any(|t| matches!(t.as_str(), "object" | "nested")) {
            continue;
        }
        types.sort();
        out.push(FieldInfo { name: name.clone(), types });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The aggregatable keyword multi-field preferred over text field `f`.
fn preferred_child<'a>(infos: &'a [FieldInfo], f: &FieldInfo) -> Option<&'a FieldInfo> {
    if !f.types.iter().any(|t| t == "text") {
        return None;
    }
    let prefix = format!("{}.", f.name);
    infos.iter().find(|c| c.name.starts_with(&prefix) && !c.name[prefix.len()..].contains('.') && c.types.iter().all(|t| t == "keyword"))
}

fn list(v: &Value, k: &str) -> Vec<String> {
    v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

/// Check the classification dependent variable's cardinality
/// (`Classification.getFieldCardinalityConstraints`).
fn check_cardinality(node: &Node, cfg: &Value, infos: &[FieldInfo], dep: &str) -> Result<()> {
    let Some(info) = infos.iter().find(|f| f.name == dep) else {
        return Err(status(format!("no mappings could be found for field [{dep}]")));
    };
    let field = preferred_child(infos, info).map(|c| c.name.clone()).unwrap_or_else(|| dep.to_string());
    let body = json!({"size": 0, "query": cfg["source"]["query"].clone(), "aggs": {"c": {"cardinality": {"field": field}}}});
    let r = search_source(node, cfg, body)?;
    let n = r["aggregations"]["c"]["value"].as_u64().unwrap_or(0);
    if n < 2 {
        return Err(status(format!("Field [{dep}] must have at least [2] distinct values but there were [{n}]")));
    }
    if n > MAX_CLASSES {
        return Err(status(format!("Field [{dep}] must have at most [{MAX_CLASSES}] distinct values but there were at least [{n}]")));
    }
    Ok(())
}

/// Select the fields of an analysis (ES `ExtractedFieldsDetector.detect`).
pub fn detect(node: &Node, cfg: &Value) -> Result<Selection> {
    let kind = analysis_kind(cfg).to_string();
    let dep = dependent_variable(cfg).map(str::to_string);
    let infos = field_infos(node, cfg)?;
    if kind == "classification" {
        if let Some(d) = &dep {
            check_cardinality(node, cfg, &infos, d)?;
        }
    }
    let af = cfg.get("analyzed_fields").cloned().unwrap_or(Value::Null);
    let includes = list(&af, "includes");
    let excludes = list(&af, "excludes");
    for inc in &includes {
        if !infos.iter().any(|f| crate::mapping::wildcard_match(inc, &f.name)) {
            return Err(status(format!("No field [{inc}] could be detected")));
        }
    }
    let supported = supported_types(&kind);
    let mut entries = Vec::new();
    let mut included = Vec::new();
    let excluded = |name: &str, types: &[String], reason: String| json!({"name": name, "mapping_types": types, "is_included": false, "is_required": false, "reason": reason});
    for f in &infos {
        let required = dep.as_deref() == Some(f.name.as_str());
        if !includes.is_empty() && !includes.iter().any(|p| crate::mapping::wildcard_match(p, &f.name)) && !required {
            entries.push(excluded(&f.name, &f.types, "field not in includes list".into()));
            continue;
        }
        if excludes.iter().any(|p| crate::mapping::wildcard_match(p, &f.name)) && !required {
            entries.push(excluded(&f.name, &f.types, "field in excludes list".into()));
            continue;
        }
        if required {
            let allowed: Vec<&str> = if kind == "regression" {
                NUMERICAL.to_vec()
            } else {
                let mut v: Vec<&str> = DISCRETE.iter().chain(CATEGORICAL.iter()).copied().collect();
                v.push("boolean");
                v.sort();
                v
            };
            if f.types.iter().any(|t| !allowed.contains(&t.as_str())) {
                return Err(status(format!(
                    "invalid types [{}] for required field [{}]; expected types are [{}]",
                    f.types.join(", "),
                    f.name,
                    allowed.join(", ")
                )));
            }
            let ft = if kind == "regression" { "numerical" } else { "categorical" };
            entries.push(json!({"name": f.name, "mapping_types": f.types, "is_included": true, "is_required": true, "feature_type": ft}));
            included.push((f.name.clone(), f.types.clone(), ft));
            continue;
        }
        if f.types.iter().any(|t| !supported.contains(&t.as_str())) {
            entries.push(excluded(&f.name, &f.types, format!("unsupported type; supported types are [{}]", supported.join(", "))));
            continue;
        }
        if let Some(c) = preferred_child(&infos, f) {
            entries.push(excluded(&f.name, &f.types, format!("[{}] is preferred because it is aggregatable", c.name)));
            continue;
        }
        let ft = if f.types.iter().all(|t| is_numeric(t)) { "numerical" } else { "categorical" };
        entries.push(json!({"name": f.name, "mapping_types": f.types, "is_included": true, "is_required": false, "feature_type": ft}));
        included.push((f.name.clone(), f.types.clone(), ft));
    }
    let required: Vec<String> = dep.into_iter().collect();
    for r in &required {
        if !included.iter().any(|(n, _, _)| n == r) {
            return Err(status(format!("required field [{r}] is missing; analysis requires fields [{}]", required.join(", "))));
        }
    }
    Ok(Selection { entries, included, required })
}

/// `at least one field must be included in the analysis` (start only).
pub fn check_has_features(sel: &Selection) -> Result<()> {
    if sel.included.iter().any(|(n, _, _)| !sel.required.contains(n)) {
        return Ok(());
    }
    let mut msg = "at least one field must be included in the analysis".to_string();
    if !sel.required.is_empty() {
        msg.push_str(&format!(" (excluding fields [{}])", sel.required.join(", ")));
    }
    Err(status(msg))
}

/// Number of documents usable by the analysis (outlier detection needs
/// every included field present; supervised analyses allow missing values).
pub fn usable_docs(node: &Node, cfg: &Value, sel: &Selection) -> Result<u64> {
    let mut filters = vec![cfg["source"]["query"].clone()];
    if analysis_kind(cfg) == "outlier_detection" {
        for (n, _, _) in &sel.included {
            filters.push(json!({"exists": {"field": n}}));
        }
    }
    let body = json!({"size": 0, "track_total_hits": true, "query": {"bool": {"filter": filters}}});
    let r = search_source(node, cfg, body)?;
    Ok(r["hits"]["total"]["value"].as_u64().unwrap_or(0))
}

pub const NO_DATA_SUFFIX: &str = "contained all the fields selected for analysis. If you are relying on automatic field selection then there are currently mapped fields that do not exist in any indexed documents, and you will have to switch to explicit field selection and include only fields that exist in indexed documents.";

/// Rough memory estimate (7b: replace with the analytics process's
/// `memory_usage_estimation`), rendered like ES's `ByteSizeValue`.
fn estimate(rows: u64, cols: usize) -> String {
    let bytes = 1_048_576u64 + rows.saturating_mul(cols as u64).saturating_mul(64);
    let mb = bytes.div_ceil(1_048_576);
    format!("{mb}mb")
}

static ESTIMATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `_explain` for a resolved config.
pub fn explain(node: &Node, cfg: &Value) -> Result<Value> {
    let sel = detect(node, cfg)?;
    let (without, with) = if sel.included.is_empty() {
        ("0".to_string(), "0".to_string())
    } else {
        let rows = usable_docs(node, cfg, &sel)?;
        if rows == 0 {
            let n = ESTIMATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(status(format!(
                "[memory_usage_estimation_{n}] Unable to estimate memory usage as no documents in the source indices {} {NO_DATA_SUFFIX}",
                source_list(cfg)
            )));
        }
        let e = estimate(rows, sel.included.len());
        (e.clone(), e)
    };
    Ok(json!({"field_selection": sel.entries, "memory_estimation": {"expected_memory_without_disk": without, "expected_memory_with_disk": with}}))
}

/// Value at a dotted path of a `_source` (also through dotted keys).
fn source_value<'a>(src: &'a Value, path: &str) -> Option<&'a Value> {
    if let Some(v) = src.get(path) {
        return Some(v);
    }
    let o = src.as_object()?;
    for (k, v) in o {
        if let Some(rest) = path.strip_prefix(k.as_str()).and_then(|r| r.strip_prefix('.')) {
            if let Some(x) = source_value(v, rest) {
                return Some(x);
            }
        }
    }
    None
}

/// A field's value in a hit: its own path, or its parent's for a
/// multi-field (`f.keyword` → `f`).
fn field_value<'a>(src: &'a Value, name: &str) -> Option<&'a Value> {
    let v = source_value(src, name).or_else(|| name.rfind('.').and_then(|i| source_value(src, &name[..i])))?;
    let v = match v {
        Value::Array(a) => a.first()?,
        x => x,
    };
    (!v.is_null() && !v.is_object()).then_some(v)
}

/// Java `Double.toString`-like rendering.
pub fn fmt_double(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e7 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn render_value(v: &Value, types: &[String]) -> String {
    let t = types.first().map(String::as_str).unwrap_or("");
    match v {
        Value::Bool(b) => if *b { "1" } else { "0" }.into(),
        Value::String(s) if t == "boolean" => if s == "true" { "1" } else { "0" }.into(),
        Value::Number(n) if DISCRETE.contains(&t) => n.as_i64().map(|i| i.to_string()).unwrap_or_else(|| format!("{}", n.as_f64().unwrap_or(0.0) as i64)),
        Value::Number(n) if NUMERICAL.contains(&t) => n.as_f64().map(fmt_double).unwrap_or_else(|| n.to_string()),
        Value::String(s) => s.clone(),
        x => super::plain(x),
    }
}

/// Feature values a processor derives from `v` (the input field's value).
fn process(p: &Value, v: Option<&Value>, out: &mut Map<String, Value>) {
    let Some((name, b)) = p.as_object().and_then(|o| o.iter().next()) else { return };
    let s = v.map(super::plain);
    match name.as_str() {
        "n_gram_encoding" => {
            let Some(s) = s else { return };
            let chars: Vec<char> = s.chars().collect();
            let prefix = b["feature_prefix"].as_str().unwrap_or("");
            let start = b["start"].as_i64().unwrap_or(0).max(0) as usize;
            let length = b["length"].as_i64().unwrap_or(50).max(0) as usize;
            for n in b["n_grams"].as_array().into_iter().flatten().filter_map(Value::as_u64) {
                let n = n as usize;
                for pos in 0..(length + 1).saturating_sub(n) {
                    let from = start + pos;
                    if from + n <= chars.len() {
                        out.insert(format!("{prefix}.{n}{pos}"), json!(chars[from..from + n].iter().collect::<String>()));
                    }
                }
            }
        }
        "one_hot_encoding" => {
            for (val, feat) in b["hot_map"].as_object().into_iter().flatten() {
                let hit = s.as_deref() == Some(val.as_str());
                out.insert(feat.as_str().unwrap_or("").to_string(), json!(if hit { "1" } else { "0" }));
            }
        }
        "frequency_encoding" => {
            let f = s.as_deref().and_then(|s| b["frequency_map"][s].as_f64()).unwrap_or(0.0);
            out.insert(b["feature_name"].as_str().unwrap_or("").to_string(), json!(fmt_double(f)));
        }
        "target_mean_encoding" => {
            let f = s.as_deref().and_then(|s| b["target_map"][s].as_f64()).unwrap_or_else(|| b["default_value"].as_f64().unwrap_or(0.0));
            out.insert(b["feature_name"].as_str().unwrap_or("").to_string(), json!(fmt_double(f)));
        }
        _ => {}
    }
}

/// Rows `_preview` returns at most.
const PREVIEW_SIZE: usize = 100;

/// `_preview`: the feature values the analytics process would receive.
pub fn preview(node: &Node, cfg: &Value) -> Result<Value> {
    let sel = detect(node, cfg)?;
    let kind = analysis_kind(cfg);
    let procs: Vec<Value> = cfg["analysis"][kind]["feature_processors"].as_array().cloned().unwrap_or_default();
    let inputs: BTreeSet<String> = procs.iter().filter_map(|p| p.as_object().and_then(|o| o.values().next()).and_then(|b| b["field"].as_str().map(str::to_string))).collect();
    // Processor inputs (and the multi-field standing in for one) are not
    // features themselves.
    let fields: Vec<&(String, Vec<String>, &str)> = sel
        .included
        .iter()
        .filter(|(n, _, _)| !inputs.contains(n) && !n.rfind('.').is_some_and(|i| inputs.contains(&n[..i])))
        .collect();
    let body = json!({"size": PREVIEW_SIZE, "query": cfg["source"]["query"].clone()});
    let r = search_source(node, cfg, body)?;
    let mut rows = Vec::new();
    for hit in r["hits"]["hits"].as_array().into_iter().flatten() {
        let src = &hit["_source"];
        let mut row = Map::new();
        let mut complete = true;
        for (n, types, _) in &fields {
            match field_value(src, n) {
                Some(v) => {
                    row.insert(n.clone(), json!(render_value(v, types)));
                }
                None => complete = false,
            }
        }
        if !complete && kind == "outlier_detection" {
            continue;
        }
        for p in &procs {
            let input = p.as_object().and_then(|o| o.values().next()).and_then(|b| b["field"].as_str()).unwrap_or("");
            process(p, field_value(src, input), &mut row);
        }
        rows.push(Value::Object(row));
    }
    Ok(json!({"feature_values": rows}))
}

/// Merged `properties` of the source indices' mappings (for the dest index).
pub fn source_properties(node: &Node, cfg: &Value) -> Map<String, Value> {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    if let Ok(targets) = node.resolve(&source_expr(cfg), ResolveOpts { allow_no_indices: true, ignore_unavailable: true, ..Default::default() }) {
        for t in targets {
            let m = t.svc.meta().mappings.to_json();
            let p = m.get("properties").or_else(|| m.get("_doc").and_then(|d| d.get("properties")));
            for (k, v) in p.and_then(Value::as_object).into_iter().flatten() {
                props.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    props.into_iter().collect()
}

/// Is the expression token remote (`cluster:index`)?
pub fn is_remote(tok: &str) -> bool {
    tok.contains(':')
}

/// `SourceDestValidator` (source missing, remote source, dest in source,
/// dest alias without a write index), as one `validation_exception`.
pub fn validate_source_dest(node: &Node, cfg: &Value) -> Result<()> {
    let fail = |m: String| EsError::new(400, "validation_exception", format!("Validation Failed: 1: {m};"));
    let toks = source_indices(cfg);
    let mut remote = false;
    let mut resolved: Vec<(String, BTreeSet<String>)> = Vec::new();
    for t in &toks {
        if is_remote(t) {
            remote = true;
            continue;
        }
        match node.resolve(t, ResolveOpts { allow_no_indices: true, ..Default::default() }) {
            Ok(ts) => resolved.push((t.clone(), ts.iter().map(|x| x.svc.name.clone()).collect())),
            Err(_) => return Err(fail(format!("no such index [{t}]"))),
        }
    }
    if remote {
        return Err(fail("remote source and cross-project indices are not supported".into()));
    }
    let dest = cfg["dest"]["index"].as_str().unwrap_or("");
    if dest.is_empty() {
        return Ok(());
    }
    // The concrete index the dest name writes to.
    let concrete = if node.get_index(dest).is_some() {
        Some(dest.to_string())
    } else {
        let mut members: Vec<(String, bool)> = Vec::new();
        for svc in node.all_indices() {
            if let Some(def) = svc.aliases().get(dest) {
                members.push((svc.name.clone(), def.get("is_write_index").and_then(Value::as_bool).unwrap_or(false)));
            }
        }
        match members.len() {
            0 => None,
            1 => Some(members[0].0.clone()),
            _ => match members.iter().find(|(_, w)| *w) {
                Some((n, _)) => Some(n.clone()),
                None => {
                    return Err(fail(format!(
                        "no write index is defined for alias [{dest}]. The write index may be explicitly disabled using is_write_index=false or the alias points to multiple indices without one being designated as a write index"
                    )))
                }
            },
        }
    };
    if let Some(c) = concrete {
        if let Some((tok, _)) = resolved.iter().find(|(_, set)| set.contains(&c)) {
            return Err(fail(format!("Destination index [{c}] is included in source expression [{tok}]")));
        }
    }
    Ok(())
}
