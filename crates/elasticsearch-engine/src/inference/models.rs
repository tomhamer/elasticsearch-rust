//! `_ml/trained_models`: model configs, definitions, deployments and
//! inference.
//!
//! Configs live in `<path.data>/ml-models/models.json`; uploaded definition
//! parts / vocabularies under `<path.data>/ml-models/<id>/`; deployments
//! (`TrainedModelAssignment`s) in `deployments.json`. The prepackaged
//! `lang_ident_model_1` is always listed. Built-in `.multilingual-e5-small`
//! and `.elser_model_2` are "downloaded" by a model import task when put
//! and then run on the CPU fallbacks in [`super::local`]; uploaded PyTorch
//! models run on the TorchScript interpreter in [`super::torch`].

use super::local;
use super::torch;
use crate::error::{EsError, Result};
use base64::Engine as _;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

static LOCK: Mutex<()> = Mutex::new(());

const NATIVE_INDEX: &str = ".ml-inference-native-000002";
const TREE_INDEX: &str = ".ml-inference-000005";
/// How long the model import task of a packaged model runs.
const DOWNLOAD_MILLIS: i64 = 20_000;

fn root(state_dir: &Path) -> PathBuf {
    state_dir.parent().unwrap_or(state_dir).join("ml-models")
}

fn read_map(path: PathBuf) -> Map<String, Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default()
}

fn write_map(state_dir: &Path, file: &str, m: &Map<String, Value>) -> Result<()> {
    let r = root(state_dir);
    std::fs::create_dir_all(&r)?;
    std::fs::write(r.join(file), serde_json::to_vec_pretty(&Value::Object(m.clone()))?)?;
    Ok(())
}

fn load(state_dir: &Path) -> Map<String, Value> {
    read_map(root(state_dir).join("models.json"))
}

fn save(state_dir: &Path, m: &Map<String, Value>) -> Result<()> {
    write_map(state_dir, "models.json", m)
}

fn not_found(id: &str) -> EsError {
    EsError::resource_not_found(format!("No known trained model with model_id [{id}], you may need to create it or load it into the cluster with eland"))
}

fn bad_request(msg: impl Into<String>) -> EsError {
    EsError::new(400, "status_exception", msg)
}

/// Does a trained model `id` exist (stored or built-in downloaded)?
pub fn exists(state_dir: &Path, id: &str) -> bool {
    load(state_dir).contains_key(id)
}

// ---------------------------------------------------------------------------
// Aliases

/// Model aliases: alias -> model id (`.ml-inference` `model_aliases` metadata).
fn aliases(state_dir: &Path) -> Map<String, Value> {
    read_map(root(state_dir).join("aliases.json"))
}

fn save_aliases(state_dir: &Path, a: &Map<String, Value>) -> Result<()> {
    write_map(state_dir, "aliases.json", a)
}

/// Resolve a model alias to its model id.
pub fn resolve_alias(state_dir: &Path, id: &str) -> Option<String> {
    aliases(state_dir).get(id).and_then(Value::as_str).map(str::to_string)
}

/// `metadata.model_aliases` of a config (sorted).
fn with_aliases(mut cfg: Value, aliases: &Map<String, Value>) -> Value {
    let mid = cfg["model_id"].as_str().unwrap_or("").to_string();
    let mut mine: Vec<&str> = aliases.iter().filter(|(_, m)| m.as_str() == Some(mid.as_str())).map(|(a, _)| a.as_str()).collect();
    if mine.is_empty() {
        return cfg;
    }
    mine.sort();
    if !cfg["metadata"].is_object() {
        cfg["metadata"] = json!({});
    }
    cfg["metadata"]["model_aliases"] = json!(mine);
    cfg
}

/// `MlStrings.isValidId` plus ES's alias rule: must not end with a digit.
fn valid_alias(a: &str) -> bool {
    !a.is_empty()
        && a.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        && a.as_bytes()[0].is_ascii_alphanumeric()
        && !a.as_bytes()[a.len() - 1].is_ascii_digit()
        && a.len() <= 64
}

fn inference_type(cfg: &Value) -> Option<String> {
    cfg["inference_config"].as_object().and_then(|o| o.keys().next().cloned())
}

/// `PUT _ml/trained_models/{model_id}/model_aliases/{alias}`.
pub fn put_alias(state_dir: &Path, model_id: &str, alias: &str, reassign: bool) -> Result<Value> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut errors: Vec<String> = Vec::new();
    if alias == model_id {
        errors.push(format!("model_alias [{alias}] cannot equal model_id [{model_id}]"));
    }
    if !valid_alias(alias) {
        errors.push(format!("Invalid model_alias; '{alias}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start with alphanumeric and cannot end with numbers"));
    }
    if !errors.is_empty() {
        let refs: Vec<&str> = errors.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let models = load(state_dir);
    let model = if model_id == "lang_ident_model_1" { lang_ident() } else { models.get(model_id).cloned().ok_or_else(|| not_found(model_id))? };
    if models.contains_key(alias) || alias == "lang_ident_model_1" {
        return Err(EsError::new(400, "status_exception", format!("requested model_alias [{alias}] is the same as an existing model_id. Model Ids and model_aliases must be unique")));
    }
    let mut a = aliases(state_dir);
    if let Some(old) = a.get(alias).and_then(Value::as_str).map(str::to_string) {
        if old == model_id {
            return Ok(json!({"acknowledged": true}));
        }
        if !reassign {
            return Err(EsError::new(400, "status_exception", format!("cannot assign model_alias [{alias}] to model_id [{model_id}] as model_alias already refers to [{old}]. Set parameter reassign to [true] if model_alias should be reassigned.")));
        }
        let old_cfg = if old == "lang_ident_model_1" { Some(lang_ident()) } else { models.get(&old).cloned() };
        if let Some(oc) = old_cfg {
            let (ot, nt) = (inference_type(&oc), inference_type(&model));
            if ot.is_some() && nt.is_some() && ot != nt {
                return Err(EsError::new(400, "status_exception", format!(
                    "cannot reassign model_alias [{alias}] to model [{model_id}] with inference config type [{}] from model [{old}] with type [{}]",
                    nt.unwrap_or_default(),
                    ot.unwrap_or_default()
                )));
            }
            let deps = deployments(state_dir);
            let deployed = |m: &str| deps.values().any(|d| d["model_id"] == m);
            if model["model_type"] == "pytorch" && !deployed(model_id) && deployed(&old) {
                return Err(EsError::new(400, "status_exception", format!("cannot reassign model_alias [{alias}] to model [{model_id}] from model [{old}] as it is not yet deployed")));
            }
        }
    }
    a.insert(alias.to_string(), json!(model_id));
    save_aliases(state_dir, &a)?;
    Ok(json!({"acknowledged": true}))
}

/// `DELETE _ml/trained_models/{model_id}/model_aliases/{alias}`.
pub fn delete_alias(state_dir: &Path, model_id: &str, alias: &str) -> Result<Value> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut a = aliases(state_dir);
    let Some(cur) = a.get(alias).and_then(Value::as_str).map(str::to_string) else {
        return Err(EsError::new(404, "status_exception", format!("model_alias [{alias}] could not be found")));
    };
    if cur != model_id {
        return Err(EsError::new(409, "status_exception", format!("model_alias [{alias}] does not refer to [{model_id}]")));
    }
    a.remove(alias);
    save_aliases(state_dir, &a)?;
    Ok(json!({"acknowledged": true}))
}

/// Aliases pointing at `model_id` (for the delete-model checks).
pub fn aliases_of(state_dir: &Path, model_id: &str) -> Vec<String> {
    aliases(state_dir).iter().filter(|(_, m)| m.as_str() == Some(model_id)).map(|(a, _)| a.clone()).collect()
}

// ---------------------------------------------------------------------------
// Configs

fn lang_ident() -> Value {
    json!({"model_id": "lang_ident_model_1", "model_type": "lang_ident", "created_by": "_xpack", "version": "7.6.0", "create_time": 1575548914594i64,
        "model_size_bytes": 1053992, "estimated_operations": 39629, "license_level": "basic",
        "description": "Model used for identifying language from arbitrary input text.", "tags": ["lang_ident", "prepackaged"],
        "input": {"field_names": ["text"]},
        "inference_config": {"classification": {"num_top_classes": 0, "top_classes_results_field": "top_classes", "results_field": "predicted_value", "num_top_feature_importance_values": 0, "prediction_field_type": "string"}}})
}

fn builtin_config(id: &str) -> Option<Value> {
    let base = id.trim_end_matches("_linux-x86_64");
    let (desc, cfg, size, ops): (&str, Value, i64, i64) = match base {
        ".multilingual-e5-small" => ("E5 (EmbEddings from bidirEctional Encoder rEpresentations) multilingual small", json!({"text_embedding": {"vocabulary": {"index": NATIVE_INDEX}, "tokenization": {"xlm_roberta": {"do_lower_case": false, "with_special_tokens": true, "max_sequence_length": 512, "truncate": "first", "span": -1}}, "embedding_size": 384, "results_field": "predicted_value"}}), 471_000_000, 0),
        ".elser_model_2" => ("Elastic Learned Sparse EncodeR v2", json!({"text_expansion": {"vocabulary": {"index": NATIVE_INDEX}, "tokenization": {"bert": {"do_lower_case": true, "with_special_tokens": true, "max_sequence_length": 512, "truncate": "first", "span": -1}}, "results_field": "tokens"}}), 438_000_000, 0),
        _ => return None,
    };
    Some(json!({"model_id": id, "model_type": "pytorch", "created_by": "api_user", "version": "12.0.0", "create_time": crate::date::now_millis(),
        "model_size_bytes": size, "estimated_operations": ops, "license_level": "platinum", "description": desc,
        "tags": [], "metadata": {}, "input": {"field_names": ["text_field"]}, "inference_config": cfg,
        "location": {"index": {"name": NATIVE_INDEX}}}))
}

/// Internal (not rendered) keys of a stored config.
const DOWNLOAD_KEY: &str = "__download_until";
const DEFINITION_KEY: &str = "__definition";

/// The stored config of `id` (built-in `lang_ident_model_1` included).
fn config_of(models: &Map<String, Value>, id: &str) -> Option<Value> {
    if id == "lang_ident_model_1" { Some(lang_ident()) } else { models.get(id).cloned() }
}

/// Is the model import (download) task of `cfg` still running?
fn downloading(cfg: &Value) -> bool {
    cfg.get(DOWNLOAD_KEY).and_then(Value::as_i64).is_some_and(|t| t > crate::date::now_millis())
}

/// `GET _ml/trained_models` options.
#[derive(Default, Clone)]
pub struct GetOpts {
    pub allow_no_match: bool,
    pub from: Option<i64>,
    pub size: Option<i64>,
    pub tags: Vec<String>,
    pub include: Vec<String>,
    pub decompress_definition: bool,
    pub exclude_generated: bool,
}

/// Expand an id expression (aliases, wildcards, `_all`) to the sorted
/// matching configs.
fn expand(state_dir: &Path, id: Option<&str>, allow_no_match: bool, tags: &[String]) -> Result<Vec<Value>> {
    let stored = load(state_dir);
    let al = aliases(state_dir);
    let mut all: Vec<Value> = vec![lang_ident()];
    all.extend(stored.values().cloned());
    all.sort_by(|a, b| a["model_id"].as_str().cmp(&b["model_id"].as_str()));
    let resolved: Option<String> = id.map(|p| p.split(',').map(|pt| al.get(pt).and_then(Value::as_str).unwrap_or(pt).to_string()).collect::<Vec<_>>().join(","));
    let id = resolved.as_deref();
    let mut sel: Vec<Value> = match id {
        None | Some("_all") | Some("*") | Some("") => all,
        Some(p) => {
            let pats: Vec<&str> = p.split(',').collect();
            for pt in &pats {
                if !pt.contains('*') && pt != &"_all" && !all.iter().any(|m| m["model_id"] == *pt) {
                    return Err(not_found(pt));
                }
            }
            all.into_iter().filter(|m| pats.iter().any(|pt| *pt == "_all" || glob(pt, m["model_id"].as_str().unwrap_or("")))).collect()
        }
    };
    if !tags.is_empty() {
        sel.retain(|m| tags.iter().all(|t| m["tags"].as_array().is_some_and(|a| a.iter().any(|x| x == t.as_str()))));
    }
    if sel.is_empty() && !allow_no_match {
        return Err(not_found(id.unwrap_or("")));
    }
    Ok(sel)
}

/// `GET _ml/trained_models[/{id}]`.
pub fn get(state_dir: &Path, id: Option<&str>, allow_no_match: bool) -> Result<Value> {
    get_with(state_dir, id, &GetOpts { allow_no_match, decompress_definition: true, size: Some(10_000), ..Default::default() })
}

/// `GET _ml/trained_models[/{id}]` with paging (`count` is the total).
pub fn get_page(state_dir: &Path, id: Option<&str>, allow_no_match: bool, from: Option<i64>, size: Option<i64>) -> Result<Value> {
    get_with(state_dir, id, &GetOpts { allow_no_match, from, size, decompress_definition: true, ..Default::default() })
}

/// `GET _ml/trained_models[/{id}]` with every option.
pub fn get_with(state_dir: &Path, id: Option<&str>, o: &GetOpts) -> Result<Value> {
    page_check(o.from, o.size)?;
    let all = expand(state_dir, id, o.allow_no_match, &o.tags)?;
    let total = all.len();
    let from = o.from.unwrap_or(0).max(0) as usize;
    let size = o.size.unwrap_or(100).max(0) as usize;
    let paged: Vec<Value> = all.into_iter().skip(from).take(size).collect();
    let inc = |k: &str| o.include.iter().any(|i| i == k);
    if inc("definition") && paged.len() > 1 {
        return Err(bad_request("Getting model definition is not supported when getting more than one model"));
    }
    if inc("definition_status") && paged.len() > 1 {
        return Err(bad_request("Getting the model download status is not supported when getting more than one model"));
    }
    if inc("definition_status") && paged.first().is_some_and(|c| c["model_type"] != "pytorch") {
        return Err(bad_request("Definition status is only relevant to PyTorch model types"));
    }
    let al = aliases(state_dir);
    let mut out = Vec::new();
    for cfg in paged {
        let mid = cfg["model_id"].as_str().unwrap_or("").to_string();
        let mut c = with_aliases(cfg.clone(), &al);
        let def = c.as_object_mut().and_then(|m| m.shift_remove(DEFINITION_KEY));
        if let Some(m) = c.as_object_mut() {
            m.shift_remove(DOWNLOAD_KEY);
        }
        if inc("definition") {
            if let Some(d) = def {
                if o.decompress_definition {
                    c["definition"] = d;
                } else {
                    c["compressed_definition"] = json!(compress(&d)?);
                }
            } else if mid == "lang_ident_model_1" && !o.decompress_definition {
                c["compressed_definition"] = json!(compress(&json!({}))?);
            }
        }
        if o.exclude_generated {
            if let Some(m) = c.as_object_mut() {
                for k in ["created_by", "version", "create_time", "model_size_bytes", "estimated_heap_memory_usage_bytes", "estimated_operations", "license_level"] {
                    m.shift_remove(k);
                }
            }
        }
        if inc("definition_status") {
            let full = !downloading(&cfg) && definition_length(state_dir, &mid).is_ok_and(|n| n > 0);
            c["fully_defined"] = json!(full);
        }
        out.push(c);
    }
    Ok(json!({"count": total, "trained_model_configs": out}))
}

fn page_check(from: Option<i64>, size: Option<i64>) -> Result<()> {
    if from.unwrap_or(0) < 0 {
        return Err(EsError::illegal_argument("Parameter [from] cannot be < 0"));
    }
    if size.unwrap_or(0) < 0 {
        return Err(EsError::illegal_argument("Parameter [size] cannot be < 0"));
    }
    Ok(())
}

fn glob(pat: &str, s: &str) -> bool {
    match pat.split_once('*') {
        None => pat == s,
        Some((a, b)) => s.starts_with(a) && s[a.len()..].ends_with(b) && s.len() >= a.len() + b.len(),
    }
}

/// base64(gzip(json)) (`InferenceToXContentCompressor`).
fn compress(v: &Value) -> Result<String> {
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&serde_json::to_vec(v)?)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(gz.finish()?))
}

fn decompress(s: &str) -> Option<Value> {
    let raw = base64::engine::general_purpose::STANDARD.decode(s.trim()).ok()?;
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(&raw[..]).read_to_end(&mut out).ok()?;
    serde_json::from_slice(&out).ok()
}

// ---------------------------------------------------------------------------
// PUT

const NLP_TYPES: &[&str] = &["pass_through", "text_embedding", "text_expansion", "ner", "fill_mask", "text_classification", "zero_shot_classification", "question_answering", "text_similarity"];

/// `XContentParseException` chain of a failure inside `inference_config`.
fn config_parse_error(kind: &str, field: Option<&str>, cause: EsError) -> EsError {
    let inner = match field {
        Some(f) => EsError::x_content_parse(format!("[1:2] [{kind}] failed to parse field [{f}]")).caused_by(cause),
        None => cause,
    };
    EsError::x_content_parse("[1:2] [trained_model_config] failed to parse field [inference_config]").caused_by(inner)
}

fn mask_token(tokenizer: &str) -> &'static str {
    match tokenizer {
        "roberta" | "mpnet" | "xlm_roberta" => "<mask>",
        _ => "[MASK]",
    }
}

/// An NLP inference config with ES's defaults filled in (the parsed
/// `NlpConfig` as rendered back).
fn normalize_nlp(kind: &str, cfg: &Value) -> Result<Value> {
    let mut o = cfg.as_object().cloned().unwrap_or_default();
    if o.contains_key("vocabulary") {
        return Err(config_parse_error(kind, Some("vocabulary"), bad_request("illegal setting [vocabulary] on inference model creation")));
    }
    let (tok_name, tok_body) = match o.get("tokenization").and_then(Value::as_object).and_then(|t| t.iter().next()) {
        Some((n, b)) => (n.clone(), b.as_object().cloned().unwrap_or_default()),
        None => ("bert".to_string(), Map::new()),
    };
    let mut tok = Map::new();
    tok.insert("do_lower_case".into(), tok_body.get("do_lower_case").cloned().unwrap_or(json!(false)));
    tok.insert("with_special_tokens".into(), tok_body.get("with_special_tokens").cloned().unwrap_or(json!(true)));
    tok.insert("max_sequence_length".into(), tok_body.get("max_sequence_length").cloned().unwrap_or(json!(512)));
    tok.insert("truncate".into(), tok_body.get("truncate").cloned().unwrap_or(json!("first")));
    tok.insert("span".into(), tok_body.get("span").cloned().unwrap_or(json!(-1)));
    if tok_name == "roberta" {
        tok.insert("add_prefix_space".into(), tok_body.get("add_prefix_space").cloned().unwrap_or(json!(false)));
    }
    let mut out = Map::new();
    out.insert("vocabulary".into(), json!({"index": NATIVE_INDEX}));
    out.insert("tokenization".into(), json!({ tok_name.clone(): tok }));
    match kind {
        "fill_mask" => {
            let want = mask_token(&tok_name);
            if let Some(m) = o.get("mask_token").and_then(Value::as_str) {
                if m != want {
                    return Err(config_parse_error(kind, None, EsError::illegal_argument(format!("Mask token requested was [{m}] but must be [{want}] for this model"))));
                }
            }
            out.insert("num_top_classes".into(), o.get("num_top_classes").cloned().unwrap_or(json!(5)));
        }
        "text_classification" => {
            out.insert("classification_labels".into(), o.get("classification_labels").cloned().unwrap_or(json!([])));
            out.insert("num_top_classes".into(), o.get("num_top_classes").cloned().unwrap_or(json!(-1)));
        }
        "zero_shot_classification" => {
            out.insert("classification_labels".into(), o.get("classification_labels").cloned().unwrap_or(json!([])));
            out.insert("multi_label".into(), o.get("multi_label").cloned().unwrap_or(json!(false)));
            out.insert("hypothesis_template".into(), o.get("hypothesis_template").cloned().unwrap_or(json!("This example is {}.")));
            if let Some(l) = o.get("labels") {
                out.insert("labels".into(), l.clone());
            }
        }
        "question_answering" => {
            out.insert("num_top_classes".into(), o.get("num_top_classes").cloned().unwrap_or(json!(0)));
            out.insert("max_answer_length".into(), o.get("max_answer_length").cloned().unwrap_or(json!(15)));
            if let Some(q) = o.get("question") {
                out.insert("question".into(), q.clone());
            }
        }
        "ner" => {
            if let Some(l) = o.get("classification_labels").filter(|l| l.as_array().is_some_and(|a| !a.is_empty())) {
                out.insert("classification_labels".into(), l.clone());
            }
        }
        "text_similarity" => {
            if let Some(t) = o.get("text") {
                out.insert("text".into(), t.clone());
            }
            out.insert("span_score_combination_function".into(), o.get("span_score_combination_function").cloned().unwrap_or(json!("max")));
        }
        "text_embedding" => {
            if let Some(e) = o.get("embedding_size") {
                out.insert("embedding_size".into(), e.clone());
            }
        }
        _ => {}
    }
    if let Some(r) = o.shift_remove("results_field") {
        out.insert("results_field".into(), r);
    }
    if kind == "fill_mask" {
        out.insert("mask_token".into(), json!(mask_token(&tok_name)));
    }
    Ok(Value::Object(out))
}

/// `Tree` / `Ensemble` checks (`TrainedModel.validate`) and the target type.
/// Returns (target type, estimated operations, ram bytes).
fn validate_model(m: &Value) -> Result<(String, i64, i64)> {
    if let Some(t) = m.get("tree") {
        let names = t["feature_names"].as_array().map(Vec::len).unwrap_or(0);
        let nodes = t["tree_structure"].as_array().cloned().unwrap_or_default();
        let max_feature = nodes.iter().filter(|n| n.get("leaf_value").is_none()).filter_map(|n| n.get("split_feature").and_then(Value::as_i64)).max().unwrap_or(-1);
        if max_feature >= names as i64 {
            return Err(bad_request(format!("feature index [{max_feature}] is out of bounds for the [feature_names] array")));
        }
        if nodes.len() > 1 && names == 0 {
            return Err(bad_request(format!("[feature_names] is empty and the tree has > 1 nodes; num nodes [{}]. The model Must have features if tree is not a stump", nodes.len())));
        }
        let target = t["target_type"].as_str().unwrap_or(if t.get("classification_labels").is_some() { "classification" } else { "regression" }).to_string();
        if t.get("classification_labels").is_some() && target != "classification" {
            return Err(bad_request("[target_type] should be [classification] if [classification_labels] are provided"));
        }
        let mut missing = Vec::new();
        for n in &nodes {
            for k in ["left_child", "right_child"] {
                if let Some(c) = n.get(k).and_then(Value::as_i64) {
                    if c >= nodes.len() as i64 {
                        missing.push(c);
                    }
                }
            }
        }
        if !missing.is_empty() {
            let s: Vec<String> = missing.iter().map(|x| x.to_string()).collect();
            return Err(bad_request(format!("[tree] contains missing nodes [{}]", s.join(", "))));
        }
        let ops = ((nodes.len().max(1) as f64).ln().ceil() as i64) + names as i64;
        let ram = 64 + nodes.len() as i64 * 88 + t["feature_names"].as_array().map(|a| a.iter().map(|f| 40 + 2 * f.as_str().map(str::len).unwrap_or(0) as i64).sum()).unwrap_or(0);
        return Ok((target, ops, ram));
    }
    if let Some(e) = m.get("ensemble") {
        let models = e["trained_models"].as_array().cloned().unwrap_or_default();
        if models.is_empty() {
            return Err(bad_request("[trained_models] must not be empty"));
        }
        let target = e["target_type"].as_str().unwrap_or(if e.get("classification_labels").is_some() { "classification" } else { "regression" }).to_string();
        if (e.get("classification_labels").is_some() || e.get("classification_weights").is_some()) && target != "classification" {
            return Err(bad_request("[target_type] should be [classification] if [classification_labels] or [classification_weights] are provided"));
        }
        let (mut ops_sum, mut ram) = (0i64, 96i64);
        for sub in &models {
            let (_, o, r) = validate_model(sub)?;
            ops_sum += o;
            ram += r;
        }
        let avg = (ops_sum as f64 / models.len() as f64).ceil() as i64;
        return Ok((target, avg + 2 * (models.len() as i64 - 1), ram));
    }
    let name = m.as_object().and_then(|o| o.keys().next().cloned()).unwrap_or_default();
    Err(EsError::x_content_parse(format!("[1:2] [trained_model] unknown field [{name}]")))
}

fn valid_ml_id(id: &str) -> bool {
    !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.')
        && id.as_bytes()[0].is_ascii_alphanumeric()
        && id.as_bytes()[id.len() - 1].is_ascii_alphanumeric()
}

/// `PUT _ml/trained_models/{id}` options.
#[derive(Default, Clone)]
pub struct PutOpts {
    pub defer_definition_decompression: bool,
    pub wait_for_completion: bool,
}

/// `PUT _ml/trained_models/{id}`.
pub fn put(state_dir: &Path, id: &str, body: &Value) -> Result<Value> {
    put_with(state_dir, id, body, &PutOpts::default())
}

/// `PUT _ml/trained_models/{id}` with the request options.
pub fn put_with(state_dir: &Path, id: &str, body: &Value, opts: &PutOpts) -> Result<Value> {
    let o = body.as_object().cloned().unwrap_or_default();
    let packaged = id.starts_with('.');
    // Parsing: the named inference config.
    let mut inference_config = None;
    if let Some((kind, c)) = o.get("inference_config").and_then(Value::as_object).and_then(|m| m.iter().next()) {
        inference_config = Some(if NLP_TYPES.contains(&kind.as_str()) { json!({ kind.clone(): normalize_nlp(kind, c)? }) } else { json!({ kind.clone(): c.clone() }) });
    }
    let has_definition = o.get("definition").is_some_and(|d| !d.is_null());
    let compressed = o.get("compressed_definition").and_then(Value::as_str);
    let model_size_given = o.get("model_size_bytes").and_then(Value::as_i64).unwrap_or(0);
    let is_nlp = inference_config.as_ref().and_then(|c| c.as_object()).and_then(|m| m.keys().next()).is_some_and(|k| NLP_TYPES.contains(&k.as_str()));
    let model_type = o.get("model_type").and_then(Value::as_str).map(str::to_string);
    // `TrainedModelConfig.Builder.validate(true)`.
    let mut errs: Vec<String> = Vec::new();
    if !packaged {
        if !has_definition && compressed.is_none() && model_type.is_none() && !is_nlp {
            errs.push("[model_type] must be set if [definition] is not defined.".into());
        }
        if inference_config.is_none() {
            errs.push("[inference_config] must not be null.".into());
        }
        if !valid_ml_id(id) {
            errs.push(format!("Invalid model_id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric"));
        }
    }
    let tags: Vec<String> = o.get("tags").and_then(Value::as_array).map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()).unwrap_or_default();
    if tags.iter().any(|t| t == id) {
        errs.push("none of the tags must equal the model_id".into());
    }
    if let Some(f) = o.get("input").and_then(|i| i.get("field_names")) {
        let names: Vec<String> = match f {
            Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            Value::String(s) => vec![s.clone()],
            _ => vec![],
        };
        if names.is_empty() {
            errs.push("[input.field_names] must not be empty".into());
        } else if names.iter().filter(|n| n.contains('.')).any(|n| n.split('.').any(str::is_empty)) {
            errs.push("[input.field_names] must only contain valid dot delimited field names".into());
        }
    }
    for k in ["version", "created_by", "create_time", "license_level", "location", "model_package"] {
        if o.get(k).is_some_and(|v| !v.is_null()) {
            errs.push(format!("illegal to set [{k}] at inference model creation"));
        }
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    // `PutTrainedModelAction.Request.validate`.
    if opts.defer_definition_decompression && model_size_given == 0 && compressed.is_some() {
        return Err(EsError::action_validation(&["when [defer_definition_decompression] is true and a compressed definition is provided, model_size_bytes must be set"]));
    }
    // Transport: parse and validate the definition.
    let mut definition = if has_definition { o.get("definition").cloned() } else { None };
    if definition.is_none() && !opts.defer_definition_decompression {
        if let Some(c) = compressed {
            definition = Some(decompress(c).ok_or_else(|| bad_request(format!("Failed to parse definition for [{id}]")))?);
        }
    }
    let mut size_ops = None;
    let mut model_type = model_type;
    if let Some(def) = &definition {
        let tm = def.get("trained_model").ok_or_else(|| bad_request(format!("Failed to parse definition for [{id}]")))?;
        let (target, ops, ram) = validate_model(tm).map_err(|e| if e.kind == "status_exception" { bad_request(format!("Definition for [{id}] has validation failures.")).caused_by(e) } else { e })?;
        let def_type = "tree_ensemble";
        match model_type.as_deref() {
            None => model_type = Some(def_type.into()),
            Some(t) if t != def_type => return Err(bad_request(format!("model_type [{t}] does not match the model definition type [{def_type}]"))),
            _ => {}
        }
        if let Some(kind) = inference_config.as_ref().and_then(|c| c.as_object()).and_then(|m| m.keys().next().cloned()) {
            let ok = match kind.as_str() {
                "classification" => target == "classification",
                "regression" => target == "regression",
                _ => true,
            };
            if !ok {
                return Err(bad_request(format!("Model [{id}] inference config type [{kind}] does not support definition target type [{target}]")));
            }
        }
        size_ops = Some((ram + serde_json::to_vec(def).map(|v| v.len() as i64).unwrap_or(0), ops));
    }
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = load(state_dir);
    if aliases(state_dir).contains_key(id) {
        return Err(bad_request(format!("requested model_id [{id}] is the same as an existing model_alias. Model model_aliases and ids must be unique")));
    }
    if deployments(state_dir).contains_key(id) {
        return Err(bad_request(format!("Cannot create model [{id}] the model id is the same as the deployment id of a current model deployment")));
    }
    if let Some(cur) = m.get(id) {
        if downloading(cur) {
            return Err(EsError::new(409, "status_exception", format!("Model download task is currently running. Wait for trained model [{id}] download task to complete then try again")));
        }
    }
    if id == "lang_ident_model_1" || m.contains_key(id) {
        return Err(EsError::new(409, "resource_already_exists_exception", format!("trained machine learning model [{id}] already exists")));
    }
    let mut cfg = if let Some(c) = builtin_config(id) {
        c
    } else {
        let model_type = model_type.unwrap_or_else(|| if is_nlp { "pytorch".into() } else { "tree_ensemble".into() });
        let input = match o.get("input") {
            Some(i) if !i.is_null() => {
                let mut i = i.clone();
                if let Some(s) = i.get("field_names").and_then(Value::as_str).map(str::to_string) {
                    i["field_names"] = json!([s]);
                }
                i
            }
            _ => json!({"field_names": ["input"]}),
        };
        let (size, ops) = size_ops.unwrap_or((model_size_given, o.get("estimated_operations").and_then(Value::as_i64).unwrap_or(0)));
        let mut c = Map::new();
        c.insert("model_id".into(), json!(id));
        c.insert("model_type".into(), json!(model_type));
        if let Some(p) = o.get("platform_architecture") {
            c.insert("platform_architecture".into(), p.clone());
        }
        c.insert("created_by".into(), json!("api_user"));
        c.insert("version".into(), json!("12.0.0"));
        c.insert("create_time".into(), json!(crate::date::now_millis()));
        c.insert("model_size_bytes".into(), json!(size));
        c.insert("estimated_operations".into(), json!(ops));
        c.insert("license_level".into(), json!("platinum"));
        if let Some(d) = o.get("description") {
            c.insert("description".into(), d.clone());
        }
        c.insert("tags".into(), json!(tags));
        if let Some(md) = o.get("metadata").filter(|v| !v.is_null()) {
            c.insert("metadata".into(), md.clone());
        }
        c.insert("input".into(), input);
        if let Some(dfm) = o.get("default_field_map").filter(|v| v.as_object().is_some_and(|m| !m.is_empty())) {
            c.insert("default_field_map".into(), dfm.clone());
        }
        if let Some(ic) = &inference_config {
            c.insert("inference_config".into(), ic.clone());
        }
        if definition.is_none() {
            let loc = if model_type == "pytorch" { NATIVE_INDEX } else { TREE_INDEX };
            c.insert("location".into(), json!({"index": {"name": loc}}));
        }
        if let Some(ps) = o.get("prefix_strings").filter(|v| !v.is_null()) {
            c.insert("prefix_strings".into(), ps.clone());
        }
        for k in ["per_deployment_memory_bytes", "per_allocation_memory_bytes"] {
            if let Some(v) = o.get(k) {
                c.insert(k.into(), v.clone());
            }
        }
        Value::Object(c)
    };
    let response = cfg.clone();
    if let Some(d) = definition {
        cfg[DEFINITION_KEY] = d;
    } else if let Some(c) = compressed {
        if let Some(d) = decompress(c) {
            cfg[DEFINITION_KEY] = d;
        }
    }
    if packaged {
        // The model import task fetches the definition.
        let until = crate::date::now_millis() + DOWNLOAD_MILLIS;
        cfg[DOWNLOAD_KEY] = json!(until);
    }
    m.insert(id.to_string(), cfg);
    save(state_dir, &m)?;
    drop(_g);
    if packaged && opts.wait_for_completion {
        wait_for_download(state_dir, id, None);
    }
    Ok(response)
}

/// Wait (up to `timeout_ms`) for the import task of `id`; true when done.
fn wait_for_download(state_dir: &Path, id: &str, timeout_ms: Option<i64>) -> bool {
    let deadline = timeout_ms.map(|t| crate::date::now_millis() + t);
    loop {
        let Some(cfg) = load(state_dir).get(id).cloned() else { return true };
        if !downloading(&cfg) {
            return true;
        }
        let until = cfg[DOWNLOAD_KEY].as_i64().unwrap_or(0);
        let now = crate::date::now_millis();
        if deadline.is_some_and(|d| now >= d) {
            return false;
        }
        let next = deadline.map_or(until, |d| d.min(until));
        std::thread::sleep(std::time::Duration::from_millis(((next - now).clamp(1, 200)) as u64));
    }
}

/// `DELETE _ml/trained_models/{id}`.
pub fn delete(state_dir: &Path, id: &str) -> Result<Value> {
    delete_with(state_dir, id, false)
}

/// `DELETE _ml/trained_models/{id}[?force]`.
pub fn delete_with(state_dir: &Path, id: &str, force: bool) -> Result<Value> {
    delete_checked(state_dir, id, force, &std::collections::HashSet::new())
}

/// Model ids (or aliases) the `inference` processors of the given ingest
/// pipeline configs refer to (`InferenceProcessorInfoExtractor.getModelIdsFromInferenceProcessors`).
pub fn referenced_model_ids<'a>(pipelines: impl IntoIterator<Item = &'a Value>) -> std::collections::HashSet<String> {
    fn walk(v: &Value, out: &mut std::collections::HashSet<String>) {
        match v {
            Value::Object(o) => {
                if let Some(id) = o.get("inference").and_then(|p| p.get("model_id")).and_then(Value::as_str) {
                    out.insert(id.to_string());
                }
                o.values().for_each(|x| walk(x, out));
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = std::collections::HashSet::new();
    for p in pipelines {
        walk(p, &mut out);
    }
    out
}

/// `DELETE _ml/trained_models/{id}` (`TransportDeleteTrainedModelAction`):
/// without `force`, a model (or one of its aliases) that an ingest
/// pipeline's `inference` processor references cannot be deleted.
pub fn delete_checked(state_dir: &Path, id: &str, force: bool, referenced: &std::collections::HashSet<String>) -> Result<Value> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if id == "lang_ident_model_1" {
        return Err(EsError::new(409, "status_exception", format!("Unable to delete model [{id}] as it is a prepackaged model")));
    }
    let mut m = load(state_dir);
    if !m.contains_key(id) {
        return Err(not_found(id));
    }
    if !force {
        if referenced.contains(id) {
            return Err(EsError::new(409, "status_exception", format!("Cannot delete model [{id}] as it is still referenced by ingest processors; use force to delete the model")));
        }
        let a = aliases(state_dir);
        if let Some(alias) = a.iter().filter(|(_, v)| v.as_str() == Some(id)).map(|(k, _)| k).find(|k| referenced.contains(k.as_str())) {
            return Err(EsError::new(409, "status_exception", format!("Cannot delete model [{id}] as it has a model_alias [{alias}] that is still referenced by ingest processors; use force to delete the model")));
        }
    }
    let mut d = deployments(state_dir);
    if d.values().any(|x| x["model_id"] == id) {
        if !force {
            return Err(EsError::new(409, "status_exception", format!("Cannot delete model [{id}] as it is currently deployed; use force to delete the model")));
        }
        let gone: Vec<String> = d.iter().filter(|(_, x)| x["model_id"] == id).map(|(k, _)| k.clone()).collect();
        for k in gone {
            d.shift_remove(&k);
            runtime_drop(state_dir, &k);
        }
        save_deployments(state_dir, &d)?;
    }
    m.shift_remove(id);
    save(state_dir, &m)?;
    let mut a = aliases(state_dir);
    let before = a.len();
    a.retain(|_, v| v.as_str() != Some(id));
    if a.len() != before {
        save_aliases(state_dir, &a)?;
    }
    let _ = std::fs::remove_dir_all(root(state_dir).join(id));
    program_cache().lock().unwrap_or_else(|e| e.into_inner()).retain(|k, _| k.1 != id);
    Ok(json!({"acknowledged": true}))
}

// ---------------------------------------------------------------------------
// Vocabulary and definition parts

/// Reset user model artifacts through the normal forced deletion path, which
/// also drops deployments, aliases and cached programs. Built-ins remain.
pub fn reset(state_dir: &Path) -> Result<()> {
    let ids: Vec<_> = load(state_dir).keys().filter(|id| id.as_str() != "lang_ident_model_1").cloned().collect();
    for id in ids { delete_with(state_dir, &id, true)?; }
    let _guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
    for deployment in deployments(state_dir).keys() { runtime_drop(state_dir, deployment); }
    save_deployments(state_dir, &Map::new())?;
    save_aliases(state_dir, &Map::new())?;
    Ok(())
}

/// `PUT _ml/trained_models/{id}/vocabulary`.
pub fn put_vocabulary(state_dir: &Path, id: &str, body: &Value) -> Result<Value> {
    if !exists(state_dir, id) {
        return Err(not_found(id));
    }
    let dir = root(state_dir).join(id);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("vocabulary.json"), serde_json::to_vec(body)?)?;
    Ok(json!({"acknowledged": true}))
}

/// `PUT _ml/trained_models/{id}/definition/{part}`.
pub fn put_definition_part(state_dir: &Path, id: &str, part: u64, body: &Value) -> Result<Value> {
    if !exists(state_dir, id) {
        return Err(not_found(id));
    }
    let dir = root(state_dir).join(id);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("definition-{part}.json")), serde_json::to_vec(body)?)?;
    program_cache().lock().unwrap_or_else(|e| e.into_inner()).retain(|k, _| k.1 != id);
    Ok(json!({"acknowledged": true}))
}

/// The definition parts of `id` (part number, body), by part number.
fn parts(state_dir: &Path, id: &str) -> Vec<(u64, Value)> {
    let dir = root(state_dir).join(id);
    let mut out: Vec<(u64, Value)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            let p: u64 = n.strip_prefix("definition-")?.strip_suffix(".json")?.parse().ok()?;
            let v: Value = serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok()?;
            Some((p, v))
        })
        .collect();
    out.sort_by_key(|p| p.0);
    out
}

fn decoded_len(b64: &str) -> u64 {
    base64::engine::general_purpose::STANDARD.decode(b64.trim()).map(|v| v.len() as u64).unwrap_or(0)
}

/// `checkFullModelDefinitionIsPresent`: the total definition length, or the
/// error why the definition is incomplete.
fn definition_length(state_dir: &Path, id: &str) -> Result<u64> {
    let ps = parts(state_dir, id);
    if ps.is_empty() {
        return Err(EsError::resource_not_found(format!("Could not find trained model definition [{id}]")));
    }
    let first_total = ps[0].1["total_definition_length"].as_u64().unwrap_or(0);
    let mut sum = 0;
    for (n, p) in &ps {
        let total = p["total_definition_length"].as_u64().unwrap_or(0);
        if total != first_total {
            return Err(bad_request(format!(
                "[{id}] [total_definition_length] must be the same in all model definition parts. The value [{total}] in model definition part [{n}] does not match the value [{first_total}] in part [{}]. Unable to deploy model, please delete and recreate the model definition",
                ps[0].0
            )));
        }
        sum += decoded_len(p["definition"].as_str().unwrap_or(""));
    }
    if sum != first_total {
        return Err(bad_request(format!("Model definition truncated. Unable to deserialize trained model definition [{id}]")));
    }
    Ok(sum)
}

/// The raw definition (TorchScript archive) bytes of `id`.
fn definition_bytes(state_dir: &Path, id: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for (_, p) in parts(state_dir, id) {
        if let Ok(b) = base64::engine::general_purpose::STANDARD.decode(p["definition"].as_str().unwrap_or("").trim()) {
            out.extend(b);
        }
    }
    out
}

type ProgramKey = (PathBuf, String);

struct SendProgram(torch::Program);
// SAFETY: a parsed `Program` is immutable data (no `Rc` inside); tensors
// with shared buffers only exist while one thread runs it.
unsafe impl Send for SendProgram {}
unsafe impl Sync for SendProgram {}

fn program_cache() -> &'static Mutex<HashMap<ProgramKey, Arc<SendProgram>>> {
    static C: std::sync::OnceLock<Mutex<HashMap<ProgramKey, Arc<SendProgram>>>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn program(state_dir: &Path, id: &str) -> Result<Arc<SendProgram>> {
    let key = (state_dir.to_path_buf(), id.to_string());
    if let Some(p) = program_cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let p = Arc::new(SendProgram(torch::Program::from_archive(&definition_bytes(state_dir, id))?));
    program_cache().lock().unwrap_or_else(|e| e.into_inner()).insert(key, p.clone());
    Ok(p)
}

fn vocabulary(state_dir: &Path, id: &str) -> Vec<String> {
    std::fs::read(root(state_dir).join(id).join("vocabulary.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["vocabulary"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect()))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Deployments

fn deployments(state_dir: &Path) -> Map<String, Value> {
    read_map(root(state_dir).join("deployments.json"))
}

fn save_deployments(state_dir: &Path, d: &Map<String, Value>) -> Result<()> {
    write_map(state_dir, "deployments.json", d)
}

/// `ByteSizeValue.parseBytesSizeValue`: (bytes, `getStringRep`).
fn parse_byte_size(s: &str, setting: &str) -> Result<(u64, String)> {
    let t = s.trim().to_lowercase();
    if matches!(t.as_str(), "0" | "0b") {
        return Ok((0, "0".into()));
    }
    if t == "-1" || t == "-1b" {
        return Ok((0, "-1".into()));
    }
    let units: [(&str, u64); 11] = [("pb", 1 << 50), ("p", 1 << 50), ("tb", 1 << 40), ("t", 1 << 40), ("gb", 1 << 30), ("g", 1 << 30), ("mb", 1 << 20), ("m", 1 << 20), ("kb", 1 << 10), ("k", 1 << 10), ("b", 1)];
    for (suf, mult) in units {
        if let Some(num) = t.strip_suffix(suf) {
            let num = num.trim();
            let canon = match suf {
                "p" => "pb",
                "t" => "tb",
                "g" => "gb",
                "m" => "mb",
                "k" => "kb",
                x => x,
            };
            if let Ok(n) = num.parse::<u64>() {
                return Ok((n * mult, if n == 0 { "0".into() } else { format!("{n}{canon}") }));
            }
            if let Ok(f) = num.parse::<f64>() {
                if f >= 0.0 {
                    let b = (f * mult as f64) as u64;
                    return Ok((b, if b == 0 { "0".into() } else { format!("{b}b") }));
                }
            }
            return Err(EsError::new(400, "parse_exception", format!("failed to parse [{s}]")));
        }
    }
    Err(EsError::new(400, "parse_exception", format!("failed to parse setting [{setting}] with value [{s}] as a size in bytes: unit is missing or unrecognized")))
}

/// `ByteSizeValue.toString()` of a string rep (e.g. `1630b` -> `1.5kb`).
fn byte_size_human(rep: &str) -> String {
    let bytes = parse_byte_size(rep, "cache_size").map(|x| x.0).unwrap_or(0);
    let units: [(&str, u64); 5] = [("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)];
    for (suf, mult) in units {
        if bytes >= mult {
            // `Strings.format1Decimals`: the first decimal digit, truncated.
            let v = bytes as f64 / mult as f64;
            let tenths = ((v * 10.0).floor() as u64) % 10;
            return if tenths == 0 { format!("{}{suf}", v.floor() as u64) } else { format!("{}.{tenths}{suf}", v.floor() as u64) };
        }
    }
    format!("{bytes}b")
}

/// `POST _ml/trained_models/{id}/deployment/_start` parameters.
#[derive(Default, Clone)]
pub struct StartOpts {
    pub deployment_id: Option<String>,
    pub number_of_allocations: Option<i64>,
    pub threads_per_allocation: Option<i64>,
    pub queue_capacity: Option<i64>,
    pub cache_size: Option<String>,
    pub priority: Option<String>,
    pub wait_for: Option<String>,
    pub timeout_ms: Option<i64>,
    pub adaptive_allocations: Option<Value>,
}

impl StartOpts {
    /// From URL params (`StartTrainedModelDeploymentAction.Request` fields)
    /// and the optional body.
    pub fn parse(params: &HashMap<String, String>, body: &Value) -> Result<StartOpts> {
        let get = |k: &str, alt: Option<&str>| -> Option<String> {
            params.get(k).cloned().or_else(|| alt.and_then(|a| params.get(a).cloned())).or_else(|| {
                body.get(k).or_else(|| alt.and_then(|a| body.get(a))).map(|v| match v {
                    Value::String(s) => s.clone(),
                    o => o.to_string(),
                })
            })
        };
        let int = |k: &str, alt: Option<&str>| -> Result<Option<i64>> {
            match get(k, alt) {
                None => Ok(None),
                Some(s) => s.parse::<i64>().map(Some).map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{k}] with value [{s}]"))),
            }
        };
        let timeout_ms = match get("timeout", None) {
            None => None,
            Some(t) => Some(crate::security::parse_time_ms(&t, "timeout")?),
        };
        Ok(StartOpts {
            deployment_id: get("deployment_id", None),
            number_of_allocations: int("number_of_allocations", Some("model_threads"))?,
            threads_per_allocation: int("threads_per_allocation", Some("inference_threads"))?,
            queue_capacity: int("queue_capacity", None)?,
            cache_size: get("cache_size", None),
            priority: get("priority", None),
            wait_for: get("wait_for", None),
            timeout_ms,
            adaptive_allocations: body.get("adaptive_allocations").filter(|v| !v.is_null()).cloned(),
        })
    }
}

/// `AdaptiveAllocationsSettings.validate`.
fn adaptive_errors(a: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let min = a.get("min_number_of_allocations").and_then(Value::as_i64).filter(|m| *m != -1);
    let max = a.get("max_number_of_allocations").and_then(Value::as_i64).filter(|m| *m != -1);
    if min.is_some_and(|m| m < 0) {
        errs.push("[min_number_of_allocations] must be a non-negative integer or null".into());
    }
    if max.is_some_and(|m| m < 1) {
        errs.push("[max_number_of_allocations] must be a positive integer or null".into());
    }
    if let (Some(lo), Some(hi)) = (min, max) {
        if lo > hi {
            errs.push("[min_number_of_allocations] must not be larger than [max_number_of_allocations]".into());
        }
    }
    errs
}

fn render_adaptive(a: &Value) -> Value {
    let mut m = Map::new();
    m.insert("enabled".into(), json!(a.get("enabled").and_then(Value::as_bool).unwrap_or(false)));
    for k in ["min_number_of_allocations", "max_number_of_allocations"] {
        if let Some(v) = a.get(k).filter(|v| !v.is_null()) {
            m.insert(k.into(), v.clone());
        }
    }
    Value::Object(m)
}

/// `POST _ml/trained_models/{id}/deployment/_start`.
pub fn start_deployment(state_dir: &Path, id: &str, params: &HashMap<String, String>) -> Result<Value> {
    start_deployment_with(state_dir, id, &StartOpts::parse(params, &Value::Null)?)
}

/// `POST _ml/trained_models/{id}/deployment/_start` with parsed options.
pub fn start_deployment_with(state_dir: &Path, id: &str, o: &StartOpts) -> Result<Value> {
    let deployment_id = o.deployment_id.clone().unwrap_or_else(|| id.to_string());
    let threads = o.threads_per_allocation.unwrap_or(1);
    let queue = o.queue_capacity.unwrap_or(10_000);
    let priority = match o.priority.as_deref().map(str::to_lowercase).as_deref() {
        None | Some("normal") => "normal",
        Some("low") => "low",
        Some(other) => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.xpack.core.ml.inference.assignment.Priority.{}", other.to_uppercase()))),
    };
    let wait_for = o.wait_for.clone().unwrap_or_else(|| "started".into());
    let cache = match &o.cache_size {
        Some(c) => Some(parse_byte_size(c, "cache_size")?),
        None => None,
    };
    // `Request.validate`.
    let mut errs: Vec<String> = Vec::new();
    if !matches!(wait_for.as_str(), "started" | "starting" | "fully_allocated") {
        errs.push(format!("invalid [wait_for] state [{wait_for}]; must be one of [started,starting,fully_allocated]"));
    }
    let adaptive_enabled = o.adaptive_allocations.as_ref().and_then(|a| a.get("enabled")).and_then(Value::as_bool) == Some(true);
    if let Some(n) = o.number_of_allocations {
        if n < 1 {
            errs.push("[number_of_allocations] must be a positive integer".into());
        }
        if adaptive_enabled {
            errs.push("[number_of_allocations] cannot be set if adaptive allocations is enabled".into());
        }
    }
    if threads < 1 {
        errs.push("[threads_per_allocation] must be a positive integer".into());
    }
    if let Some(a) = &o.adaptive_allocations {
        errs.extend(adaptive_errors(a));
    }
    if threads > 32 || (threads as u64).count_ones() != 1 {
        errs.push("[threads_per_allocation] must be a power of 2 less than or equal to 32".into());
    }
    if queue < 1 {
        errs.push("[queue_capacity] must be a positive integer".into());
    }
    if queue > 100_000 {
        errs.push("[queue_capacity] must be less than 100000".into());
    }
    if o.timeout_ms.is_some_and(|t| t < 1) {
        errs.push("[timeout] must be positive".into());
    }
    if priority == "low" {
        if o.number_of_allocations.is_some_and(|n| n > 1) {
            errs.push("[number_of_allocations] must be 1 when [priority] is low".into());
        }
        if threads > 1 {
            errs.push("[threads_per_allocation] must be 1 when [priority] is low".into());
        }
        if o.adaptive_allocations.as_ref().and_then(|a| a.get("max_number_of_allocations")).and_then(Value::as_i64).is_some_and(|m| m > 1) {
            errs.push("[max_number_of_allocations] must be 1 when [priority] is low".into());
        }
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let allocs = o.number_of_allocations.unwrap_or_else(|| {
        o.adaptive_allocations.as_ref().and_then(|a| a.get("min_number_of_allocations")).and_then(Value::as_i64).filter(|m| *m > 0).unwrap_or(1)
    });
    {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if deployments(state_dir).contains_key(&deployment_id) {
            return Err(bad_request(format!("Could not start model deployment because an existing deployment with the same id [{deployment_id}] exist")));
        }
    }
    if super::current().is_some_and(|s| s.get_endpoint(id).is_some()) {
        return Err(bad_request(format!("Model IDs must be unique. Requested model ID [{id}] matches existing model IDs but must not.")));
    }
    let cfgs = expand(state_dir, Some(id), false, &[])?;
    if cfgs.len() > 1 {
        let ids: Vec<&str> = cfgs.iter().filter_map(|c| c["model_id"].as_str()).collect();
        return Err(bad_request(format!("cannot deploy more than one model at the same time; [{id}] matches models [{}]", ids.join(", "))));
    }
    let cfg = cfgs.into_iter().next().ok_or_else(|| not_found(id))?;
    let model_id = cfg["model_id"].as_str().unwrap_or(id).to_string();
    if cfg["model_type"] != "pytorch" {
        return Err(bad_request(format!("model [{model_id}] of type [{}] cannot be deployed. Only PyTorch models can be deployed", cfg["model_type"].as_str().unwrap_or(""))));
    }
    if cfg.get("location").is_none() {
        return Err(EsError::new(500, "status_exception", format!("model [{model_id}] does not have location")));
    }
    if model_id != deployment_id && expand(state_dir, Some(&deployment_id), false, &[]).is_ok() {
        return Err(bad_request(format!(
            "Deployment id [{deployment_id}] is the same as an another model which is not the model being deployed. Deployment id can be the same as the model being deployed but cannot match a different model"
        )));
    }
    // Step 1: wait for a running model import task.
    if !wait_for_download(state_dir, &model_id, Some(o.timeout_ms.unwrap_or(30_000))) {
        return Err(EsError::new(408, "status_exception", format!("Model download task is currently running. Wait for trained model [{model_id}] download task to complete then try again")));
    }
    let packaged = builtin_config(&model_id).is_some();
    let model_bytes = if packaged { cfg["model_size_bytes"].as_u64().unwrap_or(0) } else { definition_length(state_dir, &model_id)? };
    let (cache_bytes, cache_rep) = cache.unwrap_or_else(|| (model_bytes, if model_bytes == 0 { "0".into() } else { format!("{model_bytes}b") }));
    let per_dep = cfg["per_deployment_memory_bytes"].as_i64().unwrap_or(0);
    let per_alloc = cfg["per_allocation_memory_bytes"].as_i64().unwrap_or(0);
    let now = crate::date::now_millis();
    let dep = json!({
        "deployment_id": deployment_id, "model_id": model_id, "model_bytes": model_bytes, "threads_per_allocation": threads,
        "number_of_allocations": allocs, "queue_capacity": queue, "cache_size": cache_rep, "cache_bytes": cache_bytes, "priority": priority,
        "per_deployment_memory_bytes": per_dep, "per_allocation_memory_bytes": per_alloc,
        "adaptive_allocations": o.adaptive_allocations.as_ref().map(render_adaptive), "start_time": now,
    });
    {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut d = deployments(state_dir);
        if d.contains_key(&deployment_id) {
            return Err(EsError::new(409, "status_exception", format!("Cannot start deployment [{deployment_id}] because it has already been started")));
        }
        d.insert(deployment_id.clone(), dep.clone());
        save_deployments(state_dir, &d)?;
    }
    runtime_drop(state_dir, &deployment_id);
    Ok(json!({"assignment": assignment(&dep)}))
}

/// `TrainedModelAssignment` of a deployment.
fn assignment(d: &Value) -> Value {
    let allocs = d["number_of_allocations"].clone();
    let mut tp = Map::new();
    for k in ["model_id", "deployment_id", "model_bytes", "threads_per_allocation", "number_of_allocations", "queue_capacity", "cache_size", "priority", "per_deployment_memory_bytes", "per_allocation_memory_bytes"] {
        tp.insert(k.into(), d[k].clone());
    }
    json!({
        "task_parameters": tp,
        "routing_table": { crate::node::NODE_ID: {"current_allocations": allocs, "target_allocations": allocs, "routing_state": "started", "reason": ""} },
        "assignment_state": "started",
        "start_time": crate::ilm::iso_date(d["start_time"].as_i64().unwrap_or(0)),
        "max_assigned_allocations": allocs,
        "adaptive_allocations": d["adaptive_allocations"],
    })
}

/// `POST _ml/trained_models/{deployment_id}/deployment/_stop`.
pub fn stop_deployment(state_dir: &Path, id: &str) -> Result<Value> {
    stop_deployment_with(state_dir, id, true, false)
}

/// `_stop` with `allow_no_match` / `force`.
pub fn stop_deployment_with(state_dir: &Path, id: &str, allow_no_match: bool, _force: bool) -> Result<Value> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut d = deployments(state_dir);
    if d.shift_remove(id).is_none() {
        if allow_no_match {
            return Ok(json!({"stopped": true}));
        }
        return Err(EsError::resource_not_found(format!("No known model deployment with id [{id}]")));
    }
    save_deployments(state_dir, &d)?;
    runtime_drop(state_dir, id);
    Ok(json!({"stopped": true}))
}

/// `POST _ml/trained_models/{deployment_id}/deployment/_update`:
/// `number_of_allocations` (URL param or body) / `adaptive_allocations`.
pub fn update_deployment(state_dir: &Path, id: &str, body: &Value) -> Result<Value> {
    let n = body.get("number_of_allocations").filter(|v| !v.is_null()).map(|v| v.as_i64().ok_or_else(|| EsError::illegal_argument("[number_of_allocations] must be an integer"))).transpose()?;
    let adaptive = body.get("adaptive_allocations").filter(|v| !v.is_null()).cloned();
    // `Request.validate`.
    let mut errs: Vec<String> = Vec::new();
    if let Some(n) = n {
        if n < 1 {
            errs.push("[number_of_allocations] must be a positive integer".into());
        }
        if adaptive.as_ref().and_then(|a| a.get("enabled")).and_then(Value::as_bool) == Some(true) {
            errs.push("[number_of_allocations] cannot be set if adaptive allocations is enabled".into());
        }
    }
    if let Some(a) = &adaptive {
        errs.extend(adaptive_errors(a));
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut d = deployments(state_dir);
    let Some(dep) = d.get_mut(id) else { return Err(EsError::resource_not_found(format!("No known model deployment with id [{id}]"))) };
    let validation = |m: &str| EsError::new(400, "validation_exception", format!("Validation Failed: 1: {m};"));
    // Merged adaptive settings (`AdaptiveAllocationsSettings.merge`).
    let merged = match (&dep["adaptive_allocations"], &adaptive) {
        (_, None) => dep["adaptive_allocations"].clone(),
        (Value::Object(cur), Some(Value::Object(up))) => {
            let mut m = cur.clone();
            for (k, v) in up {
                if v.as_i64() == Some(-1) {
                    m.shift_remove(k);
                } else {
                    m.insert(k.clone(), v.clone());
                }
            }
            render_adaptive(&Value::Object(m))
        }
        (_, Some(up)) => render_adaptive(up),
    };
    if merged.is_object() {
        if merged["enabled"] == true && n.is_some() {
            return Err(validation("[number_of_allocations] cannot be set if adaptive allocations is enabled"));
        }
        let e = adaptive_errors(&merged);
        if !e.is_empty() {
            let refs: Vec<&str> = e.iter().map(String::as_str).collect();
            return Err(EsError::action_validation(&refs));
        }
    }
    if dep["priority"] == "low" {
        if n.is_some_and(|n| n > 1) {
            return Err(validation("[number_of_allocations] must be 1 when [priority] is low"));
        }
        if merged.get("max_number_of_allocations").and_then(Value::as_i64).is_some_and(|m| m > 1) {
            return Err(validation("[max_number_of_allocations] must be 1 when [priority] is low"));
        }
    }
    if let Some(n) = n {
        dep["number_of_allocations"] = json!(n);
    }
    dep["adaptive_allocations"] = merged;
    let out = dep.clone();
    save_deployments(state_dir, &d)?;
    Ok(json!({"assignment": assignment(&out)}))
}

/// `POST _ml/trained_models/{deployment_id}/deployment/cache/_clear`.
pub fn clear_cache(state_dir: &Path, id: &str) -> Result<Value> {
    if !deployments(state_dir).contains_key(id) {
        return Err(EsError::resource_not_found(format!("No known model deployment with id [{id}]")));
    }
    let mut rt = runtimes().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = rt.get_mut(&(state_dir.to_path_buf(), id.to_string())) {
        r.cache.clear();
        r.cache_used = 0;
    }
    Ok(json!({"cleared": true}))
}

// ---------------------------------------------------------------------------
// Deployment runtime: inference counters and the result cache

#[derive(Default)]
struct Runtime {
    inference_count: u64,
    cache_hits: u64,
    errors: u64,
    total_ms: f64,
    last_access: Option<i64>,
    cache: VecDeque<(String, Value, u64)>,
    cache_used: u64,
}

fn runtimes() -> &'static Mutex<HashMap<ProgramKey, Runtime>> {
    static R: std::sync::OnceLock<Mutex<HashMap<ProgramKey, Runtime>>> = std::sync::OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn runtime_drop(state_dir: &Path, deployment_id: &str) {
    runtimes().lock().unwrap_or_else(|e| e.into_inner()).remove(&(state_dir.to_path_buf(), deployment_id.to_string()));
}

/// The deployment serving `id` (a deployment id, a model id or an alias).
fn deployment_for(state_dir: &Path, id: &str) -> Option<Value> {
    let d = deployments(state_dir);
    if let Some(x) = d.get(id) {
        return Some(x.clone());
    }
    let mid = resolve_alias(state_dir, id).unwrap_or_else(|| id.to_string());
    if let Some(x) = d.get(&mid) {
        return Some(x.clone());
    }
    d.values().find(|x| x["model_id"] == mid.as_str()).cloned()
}

// ---------------------------------------------------------------------------
// Stats

/// `GET _ml/trained_models/_stats`.
pub fn stats(state_dir: &Path, id: Option<&str>, allow_no_match: bool, from: Option<i64>, size: Option<i64>) -> Result<Value> {
    stats_named(state_dir, id, allow_no_match, from, size, "node-0")
}

/// `GET _ml/trained_models/_stats` (`node_name`: the local node's name).
pub fn stats_named(state_dir: &Path, id: Option<&str>, allow_no_match: bool, from: Option<i64>, size: Option<i64>, node_name: &str) -> Result<Value> {
    page_check(from, size)?;
    let all = expand(state_dir, id, allow_no_match, &[])?;
    let total = all.len();
    let cfgs: Vec<Value> = all.into_iter().skip(from.unwrap_or(0).max(0) as usize).take(size.unwrap_or(100).max(0) as usize).collect();
    let deps = deployments(state_dir);
    let rts = runtimes().lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: Vec<Value> = Vec::new();
    for c in &cfgs {
        let mid = c["model_id"].as_str().unwrap_or("");
        let pytorch = c["model_type"] == "pytorch";
        let mut mine: Vec<&Value> = deps.values().filter(|d| d["model_id"] == mid).collect();
        mine.sort_by(|a, b| a["deployment_id"].as_str().cmp(&b["deployment_id"].as_str()));
        let size_stats = if pytorch {
            let len = if builtin_config(mid).is_some() { c["model_size_bytes"].as_u64().unwrap_or(0) } else { definition_length(state_dir, mid).unwrap_or(0) };
            let allocs: i64 = mine.iter().map(|d| d["number_of_allocations"].as_i64().unwrap_or(1)).sum::<i64>().max(1);
            let required = if len == 0 {
                0
            } else if mid.starts_with(".elser_model_1") || mid.starts_with(".elser_model_2") {
                2004u64 << 20
            } else {
                let base = (240u64 << 20) + 2 * len;
                let (pd, pa) = (c["per_deployment_memory_bytes"].as_u64().unwrap_or(0), c["per_allocation_memory_bytes"].as_u64().unwrap_or(0));
                if pd == 0 && pa == 0 { base } else { base.max(pd + pa * allocs as u64 + len) }
            };
            json!({"model_size_bytes": len, "required_native_memory_bytes": required})
        } else {
            json!({"model_size_bytes": c["model_size_bytes"], "required_native_memory_bytes": 0})
        };
        if mine.is_empty() {
            let mut o = json!({"model_id": mid, "model_size_stats": size_stats, "pipeline_count": 0});
            if !pytorch {
                o["inference_stats"] = json!({"failure_count": 0, "inference_count": 0, "cache_miss_count": 0, "missing_all_fields_count": 0, "timestamp": crate::date::now_millis()});
            }
            rows.push(o);
            continue;
        }
        for d in mine {
            let did = d["deployment_id"].as_str().unwrap_or("");
            let rt = rts.get(&(state_dir.to_path_buf(), did.to_string()));
            let count = rt.map_or(0, |r| r.inference_count);
            let hits = rt.map_or(0, |r| r.cache_hits);
            let errors = rt.map_or(0, |r| r.errors);
            let start = d["start_time"].as_i64().unwrap_or(0);
            let mut node = Map::new();
            node.insert("node".into(), json!({ crate::node::NODE_ID: {"name": node_name, "ephemeral_id": crate::node::NODE_ID, "transport_address": "127.0.0.1:9300", "external_id": node_name, "attributes": {}, "roles": ["data", "ingest", "master", "ml", "remote_cluster_client", "transform"], "version": crate::node::ES_VERSION} }));
            node.insert("routing_state".into(), json!({"routing_state": "started"}));
            node.insert("inference_count".into(), json!(count));
            if count > 0 {
                let avg = rt.map_or(0.0, |r| r.total_ms) / count as f64;
                node.insert("average_inference_time_ms".into(), crate::values::java_double(avg));
                let misses = count.saturating_sub(hits);
                node.insert("average_inference_time_ms_excluding_cache_hits".into(), crate::values::java_double(if misses > 0 { avg * count as f64 / misses as f64 } else { 0.0 }));
            }
            node.insert("inference_cache_hit_count".into(), json!(hits));
            if let Some(la) = rt.and_then(|r| r.last_access) {
                node.insert("last_access".into(), json!(la));
            }
            node.insert("number_of_pending_requests".into(), json!(0));
            if errors > 0 {
                node.insert("error_count".into(), json!(errors));
            }
            node.insert("start_time".into(), json!(start));
            node.insert("threads_per_allocation".into(), d["threads_per_allocation"].clone());
            node.insert("number_of_allocations".into(), d["number_of_allocations"].clone());
            node.insert("peak_throughput_per_minute".into(), json!(count));
            node.insert("throughput_last_minute".into(), json!(count));
            node.insert("inference_cache_hit_count_last_minute".into(), json!(hits));
            let mut ds = Map::new();
            ds.insert("deployment_id".into(), json!(did));
            ds.insert("model_id".into(), json!(mid));
            ds.insert("threads_per_allocation".into(), d["threads_per_allocation"].clone());
            ds.insert("number_of_allocations".into(), d["number_of_allocations"].clone());
            if d["adaptive_allocations"].is_object() {
                ds.insert("adaptive_allocations".into(), d["adaptive_allocations"].clone());
            }
            ds.insert("queue_capacity".into(), d["queue_capacity"].clone());
            ds.insert("state".into(), json!("started"));
            let allocs = d["number_of_allocations"].clone();
            ds.insert("allocation_status".into(), json!({"allocation_count": allocs, "target_allocation_count": allocs, "state": "fully_allocated"}));
            if let Some(cs) = d["cache_size"].as_str() {
                ds.insert("cache_size".into(), json!(byte_size_human(cs)));
            }
            ds.insert("priority".into(), d["priority"].clone());
            ds.insert("start_time".into(), json!(start));
            if errors > 0 {
                ds.insert("error_count".into(), json!(errors));
            }
            if count > 0 {
                ds.insert("inference_count".into(), json!(count));
            }
            ds.insert("peak_throughput_per_minute".into(), json!(count));
            ds.insert("nodes".into(), json!([Value::Object(node)]));
            rows.push(json!({
                "model_id": mid, "model_size_stats": size_stats, "pipeline_count": 0,
                "inference_stats": {"failure_count": errors, "inference_count": count, "cache_miss_count": count.saturating_sub(hits), "missing_all_fields_count": 0, "timestamp": crate::date::now_millis()},
                "deployment_stats": Value::Object(ds),
            }));
        }
    }
    Ok(json!({"count": total, "trained_model_stats": rows}))
}

// ---------------------------------------------------------------------------
// Inference

/// The input text of an inference doc (`NlpInferenceInput.extractInput`).
fn input_text(doc: &Value, field: &str) -> Result<String> {
    let mut cur = Some(doc);
    if doc.get(field).is_none() {
        for part in field.split('.') {
            cur = cur.and_then(|v| v.get(part));
        }
    } else {
        cur = doc.get(field);
    }
    match cur {
        None | Some(Value::Null) => Err(bad_request(format!("Input field [{field}] does not exist in the source document"))),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(bad_request(format!("Input value [{other}] for field [{field}] must be a string"))),
    }
}

/// Run an NLP model over one text (no counters).
fn run_nlp(state_dir: &Path, cfg: &Value, text: &str, kind: &str) -> Result<Value> {
    let mid = cfg["model_id"].as_str().unwrap_or("");
    let ic = cfg["inference_config"].as_object().and_then(|o| o.get(kind)).cloned().unwrap_or(json!({}));
    let results_field = ic.get("results_field").and_then(Value::as_str).unwrap_or("predicted_value").to_string();
    if builtin_config(mid).is_some() {
        return Ok(match kind {
            "text_expansion" => json!({results_field: local::embed_sparse(text), "is_truncated": false}),
            _ => json!({results_field: local::embed_dense(text, None)}),
        });
    }
    let vocab = vocabulary(state_dir, mid);
    let tok = ic.get("tokenization").and_then(Value::as_object).and_then(|t| t.values().next()).cloned().unwrap_or(json!({}));
    let lower = tok.get("do_lower_case").and_then(Value::as_bool).unwrap_or(false);
    let special = tok.get("with_special_tokens").and_then(Value::as_bool).unwrap_or(true);
    let max_len = tok.get("max_sequence_length").and_then(Value::as_u64).unwrap_or(512) as usize;
    let (ids, truncated) = torch::wordpiece(text, &vocab, lower, special, max_len);
    let prog = program(state_dir, mid)?;
    let out = prog.0.run(&ids)?.as_3d();
    let first = out.first().and_then(|r| r.first()).cloned().unwrap_or_default();
    let num = |x: f64| crate::values::java_double(x);
    let mut res = match kind {
        "text_embedding" => json!({ results_field.clone(): first.iter().map(|x| num(*x)).collect::<Vec<_>>() }),
        "text_expansion" => {
            let mut toks: Vec<(String, f64)> = first.iter().enumerate().filter(|(_, w)| **w > 0.0).map(|(i, w)| (vocab.get(i).cloned().unwrap_or_else(|| i.to_string()), *w)).collect();
            toks.sort_by(|a, b| b.1.total_cmp(&a.1));
            let m: Map<String, Value> = toks.into_iter().map(|(t, w)| (t, num(w as f32 as f64))).collect();
            json!({ results_field.clone(): m })
        }
        "pass_through" => {
            let rows: Vec<Value> = out.first().map(|r| r.iter().map(|row| Value::Array(row.iter().map(|x| num(*x)).collect())).collect()).unwrap_or_default();
            json!({ results_field.clone(): rows })
        }
        "text_classification" | "zero_shot_classification" => json!({ results_field.clone(): "", "prediction_probability": 0.0, "top_classes": [] }),
        "ner" => json!({ results_field.clone(): text, "entities": [] }),
        "fill_mask" => json!({ results_field.clone(): "", "prediction_probability": 0.0, "top_classes": [] }),
        "question_answering" => json!({ results_field.clone(): "", "start_offset": 0, "end_offset": 0, "prediction_probability": 0.0 }),
        "text_similarity" => json!({ results_field.clone(): 0.0 }),
        _ => json!({}),
    };
    if truncated {
        res["is_truncated"] = json!(true);
    }
    Ok(res)
}

/// Infer one text through deployment `dep` (counters and cache updated).
fn infer_deployed(state_dir: &Path, dep: &Value, cfg: &Value, text: &str, kind: &str) -> Result<Value> {
    let did = dep["deployment_id"].as_str().unwrap_or("").to_string();
    let key = (state_dir.to_path_buf(), did);
    let cache_key = format!("{kind}\u{0}{text}");
    {
        let mut rts = runtimes().lock().unwrap_or_else(|e| e.into_inner());
        let rt = rts.entry(key.clone()).or_default();
        rt.last_access = Some(crate::date::now_millis());
        if let Some((_, v, _)) = rt.cache.iter().find(|(k, _, _)| *k == cache_key) {
            rt.inference_count += 1;
            rt.cache_hits += 1;
            return Ok(v.clone());
        }
    }
    let t0 = std::time::Instant::now();
    let res = run_nlp(state_dir, cfg, text, kind);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut rts = runtimes().lock().unwrap_or_else(|e| e.into_inner());
    let rt = rts.entry(key).or_default();
    match &res {
        Ok(v) => {
            rt.inference_count += 1;
            rt.total_ms += ms;
            let cap = dep["cache_bytes"].as_u64().unwrap_or(0);
            let sz = (cache_key.len() + serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0)) as u64;
            if cap > 0 && sz <= cap {
                while rt.cache_used + sz > cap {
                    match rt.cache.pop_front() {
                        Some((_, _, s)) => rt.cache_used -= s,
                        None => break,
                    }
                }
                rt.cache.push_back((cache_key, v.clone(), sz));
                rt.cache_used += sz;
            }
        }
        Err(_) => rt.errors += 1,
    }
    res
}

/// `POST _ml/trained_models/{id}/_infer`: `docs: [{<input field>: ..}]`.
pub fn infer(state_dir: &Path, id: &str, body: &Value) -> Result<Value> {
    let docs = body.get("docs").and_then(Value::as_array).ok_or_else(|| EsError::action_validation(&["[docs] must not be empty"]))?;
    if docs.is_empty() {
        return Err(EsError::action_validation(&["[docs] must not be empty"]));
    }
    let dep = deployment_for(state_dir, id);
    let model_id = match &dep {
        Some(d) => d["model_id"].as_str().unwrap_or(id).to_string(),
        None => resolve_alias(state_dir, id).unwrap_or_else(|| id.to_string()),
    };
    let cfgs = get(state_dir, Some(&model_id), false)?;
    let cfg = cfgs["trained_model_configs"][0].clone();
    let ic = cfg["inference_config"].as_object().cloned().unwrap_or_default();
    let body_ic = body.get("inference_config").and_then(Value::as_object);
    let kind = body_ic.and_then(|o| o.keys().next()).or_else(|| ic.keys().next()).map(String::as_str).unwrap_or("text_embedding").to_string();
    let field = cfg["input"]["field_names"][0].as_str().unwrap_or("text_field").to_string();
    if cfg["model_type"] == "pytorch" {
        let Some(dep) = dep else {
            return Err(EsError::new(409, "status_exception", format!("Model [{id}] must be deployed to use. Please deploy with the start trained model deployment API.")));
        };
        let mut out = Vec::new();
        for d in docs {
            let text = input_text(d, &field)?;
            out.push(infer_deployed(state_dir, &dep, &cfg, &text, &kind)?);
        }
        return Ok(json!({"inference_results": out}));
    }
    let results_field = ic.get(&kind).and_then(|c| c.get("results_field")).and_then(Value::as_str).unwrap_or("predicted_value").to_string();
    let mut out = Vec::new();
    for d in docs {
        let text = d.get(&field).or_else(|| d.get("text_field")).and_then(Value::as_str).unwrap_or("");
        out.push(match kind.as_str() {
            "text_embedding" => json!({results_field.clone(): local::embed_dense(text, None)}),
            "text_expansion" => json!({results_field.clone(): local::embed_sparse(text), "is_truncated": false}),
            _ => json!({}),
        });
    }
    Ok(json!({"inference_results": out}))
}

/// Query-time inference through a deployed trained model (the
/// `CoordinatedInferenceAction` fallback when `id` is no inference
/// endpoint): the result object of `text`, e.g. `{predicted_value: [..]}`.
pub fn infer_text(state_dir: &Path, id: &str, text: &str) -> Result<Value> {
    let dep = deployment_for(state_dir, id);
    let model_id = match &dep {
        Some(d) => d["model_id"].as_str().unwrap_or(id).to_string(),
        None => resolve_alias(state_dir, id).unwrap_or_else(|| id.to_string()),
    };
    let models = load(state_dir);
    let cfg = config_of(&models, &model_id).ok_or_else(|| not_found(id))?;
    let Some(dep) = dep else {
        return Err(EsError::new(409, "status_exception", format!("Model [{id}] must be deployed to use. Please deploy with the start trained model deployment API.")));
    };
    let kind = inference_type(&cfg).unwrap_or_else(|| "text_embedding".into());
    infer_deployed(state_dir, &dep, &cfg, text, &kind)
}

/// [`infer_text`] and the model's [`results_field`]. On a cluster the
/// trained models are owned by the elected master, so a model this node does
/// not know is inferred through the coordinator (as ES's coordinated
/// inference action routes to the node running the deployment).
pub fn infer_text_with_field(state_dir: &Path, id: &str, text: &str) -> Result<(Value, String)> {
    match infer_text(state_dir, id, text) {
        Ok(r) => Ok((r, results_field(state_dir, id))),
        Err(e) if e.status == 404 => match crate::coord::current() {
            Some(c) => infer_text_on(&c, id, text).ok_or(e),
            None => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// `_infer` of one text through the coordinator; `None` when the model is
/// unknown there too.
fn infer_text_on(c: &crate::coord::Coordinator, id: &str, text: &str) -> Option<(Value, String)> {
    let seg = crate::coord::path_segment(id);
    let cfg = c("GET", &format!("/_ml/trained_models/{seg}"), None).ok()?;
    let cfg = cfg["trained_model_configs"].get(0)?.clone();
    let input = cfg["input"]["field_names"].get(0).and_then(Value::as_str).unwrap_or("text_field").to_string();
    let field = cfg["inference_config"].as_object().and_then(|o| o.values().next()).and_then(|v| v.get("results_field")).and_then(Value::as_str).unwrap_or("predicted_value").to_string();
    let mut doc = Map::new();
    doc.insert(input, json!(text));
    let resp = c("POST", &format!("/_ml/trained_models/{seg}/_infer"), Some(json!({"docs": [doc]}))).ok()?;
    Some((resp["inference_results"].get(0)?.clone(), field))
}

/// The `results_field` of a model's inference config (`predicted_value`).
pub fn results_field(state_dir: &Path, id: &str) -> String {
    let mid = deployment_for(state_dir, id).and_then(|d| d["model_id"].as_str().map(str::to_string)).unwrap_or_else(|| resolve_alias(state_dir, id).unwrap_or_else(|| id.to_string()));
    config_of(&load(state_dir), &mid)
        .and_then(|c| c["inference_config"].as_object().and_then(|o| o.values().next()).and_then(|v| v.get("results_field")).and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| "predicted_value".into())
}

/// `MlPlatformArchitecturesUtil.resolveModelPlatformVariant` (9.5): this node
/// is the cluster's only ML node, so its own platform decides.
fn model_platform_variant() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x86_64"
    } else {
        "platform_agnostic"
    }
}

/// `GET _ml/info`.
pub fn ml_info() -> Value {
    json!({"defaults": {"anomaly_detectors": {"model_memory_limit": "1gb", "categorization_examples_limit": 4, "model_snapshot_retention_days": 10, "daily_model_snapshot_retention_after_days": 1,
        "categorization_analyzer": {"char_filter": ["first_line_with_letters"], "tokenizer": "ml_standard", "filter": [{"type": "stop", "stopwords": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","Mon","Tue","Wed","Thu","Fri","Sat","Sun","January","February","March","April","May","June","July","August","September","October","November","December","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","GMT","UTC"]}, {"type": "limit", "max_token_count": "100"}]}},
        "datafeeds": {"scroll_size": 1000}, "model_platform_variant": model_platform_variant()}, "upgrade_mode": false, "native_code": {"version": crate::node::ES_VERSION, "build_hash": "d628c5c3cc794c"},
        "limits": {"total_ml_memory": "18977mb", "total_ml_processors": 16, "effective_max_model_memory_limit": "18937mb", "max_single_ml_node_processors": 16}})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_reset_removes_models_deployments_aliases_and_artifacts() {
        let data = tempfile::tempdir().unwrap();
        let state = data.path().join("_state");
        std::fs::create_dir_all(&state).unwrap();
        let model_dir = root(&state).join("saved-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("definition-part-0"), b"saved definition").unwrap();
        save(&state, &json!({"saved-model":{"model_id":"saved-model", "model_type":"tree_ensemble"}}).as_object().unwrap().clone()).unwrap();
        save_aliases(&state, &json!({"saved-alias":"saved-model", "builtin-alias":"lang_ident_model_1"}).as_object().unwrap().clone()).unwrap();
        save_deployments(&state, &json!({"saved-deployment":{"model_id":"saved-model"}}).as_object().unwrap().clone()).unwrap();
        reset(&state).unwrap();
        assert!(!exists(&state, "saved-model"));
        assert!(get(&state, Some("lang_ident_model_1"), false).is_ok());
        assert!(aliases(&state).is_empty());
        assert!(deployments(&state).is_empty());
        assert!(!model_dir.exists());
        assert!(load(&state).is_empty());
    }

    #[test]
    fn byte_sizes() {
        assert_eq!(parse_byte_size("10kb", "x").unwrap(), (10240, "10kb".into()));
        assert_eq!(parse_byte_size("0", "x").unwrap(), (0, "0".into()));
        assert_eq!(byte_size_human("1630b"), "1.5kb");
        assert_eq!(byte_size_human("10kb"), "10kb");
    }
}
