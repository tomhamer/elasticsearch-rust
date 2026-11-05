//! Mock inference services of ES's `inference-service-test` plugin
//! (`x-pack/plugin/inference/qa/test-service-plugin`), which the x-pack
//! `inference` and `rank-rrf` YAML suites register endpoints against:
//!
//! * `test_service` (`sparse_embedding`): five `feature_N` tokens weighted
//!   `|hashCode(input)| + 1 + N`.
//! * `text_embedding_test_service` (`text_embedding`): the UTF-8 bytes of
//!   `Integer.toString(hashCode(input))` repeated to `dimensions`
//!   (unit-normalised for `dot_product`).
//! * `test_reranking_service` (`rerank`): inputs parsed as float scores
//!   when they all parse, else `min_score + result_diff * (n - i)` in
//!   reverse input order (or `1 / len` with `use_text_length`).
//! * `completion_test_service` (`completion`): echoes the input upper-cased.
//!
//! Like the plugin, they exist only on test clusters: they are enabled by
//! [`enable`] or the `ES_INFERENCE_TEST_SERVICES` environment variable
//! (set by `es-yaml-runner`), never on a production node.

use super::providers::Task;
use super::Endpoint;
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicBool, Ordering};

pub const SPARSE: &str = "test_service";
/// `TestAlternateSparseInferenceService`: the sparse test service under another name.
pub const SPARSE_ALT: &str = "alternate_sparse_embedding_test_service";
pub const DENSE: &str = "text_embedding_test_service";
pub const RERANK: &str = "test_reranking_service";
pub const COMPLETION: &str = "completion_test_service";

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Register the test services (the `inference-service-test` plugin).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed) || std::env::var("ES_INFERENCE_TEST_SERVICES").is_ok_and(|v| v == "1" || v == "true")
}

/// Whether `service` names a mock service (regardless of enablement).
pub fn is_mock(service: &str) -> bool {
    matches!(service, SPARSE | SPARSE_ALT | DENSE | RERANK | COMPLETION)
}

/// Whether `service` names a registered mock service.
pub fn is_registered(service: &str) -> bool {
    is_mock(service) && enabled()
}

fn supported(service: &str) -> Task {
    match service {
        SPARSE | SPARSE_ALT => Task::SparseEmbedding,
        DENSE => Task::TextEmbedding,
        RERANK => Task::Rerank,
        _ => Task::Completion,
    }
}

fn validation(errors: &[&str]) -> EsError {
    let mut e = EsError::action_validation(errors);
    e.kind = "validation_exception".into();
    e
}

/// Java `String.hashCode` (UTF-16 code units, wrapping).
pub fn java_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// Java `Float.parseFloat` (decimal forms, `NaN`, `Infinity`, `f`/`d` suffix).
fn java_parse_float(s: &str) -> Option<f32> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    let neg = t.starts_with('-');
    match body {
        "NaN" => return Some(f32::NAN),
        "Infinity" => return Some(if neg { f32::NEG_INFINITY } else { f32::INFINITY }),
        _ => {}
    }
    let t = t.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(t);
    if t.is_empty() || t.chars().any(|c| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))) {
        return None;
    }
    t.parse::<f32>().ok()
}

fn take_str(m: &mut Map<String, Value>, k: &str) -> Option<String> {
    m.shift_remove(k).and_then(|v| v.as_str().map(str::to_string))
}

/// `PUT _inference` for a mock service: the plugin's `parseRequestConfig`
/// (unknown settings are ignored, like the plugin does).
pub fn parse(task: Task, id: &str, service: &str, body: &Map<String, Value>) -> Result<Endpoint> {
    // `text_embedding_test_service` also serves the 9.x `embedding` task.
    if task != supported(service) && !(service == DENSE && task == Task::Embedding) {
        return Err(EsError::new(400, "status_exception", format!("The [{service}] service does not support task type [{}]", task.name())));
    }
    let mut ss = body.get("service_settings").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut ts_in = body.get("task_settings").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut errors: Vec<&str> = Vec::new();
    let model = take_str(&mut ss, "model_id").or_else(|| take_str(&mut ss, "model"));
    if model.is_none() {
        errors.push("missing model");
    }
    let mut out = Map::new();
    match service {
        RERANK => {
            out.insert("model_id".into(), json!(model));
        }
        DENSE => {
            out.insert("model".into(), json!(model));
            let dims = ss.shift_remove("dimensions").filter(Value::is_i64);
            if dims.is_none() {
                errors.push("missing dimensions");
            }
            out.insert("dimensions".into(), dims.unwrap_or(Value::Null));
            if let Some(s) = take_str(&mut ss, "similarity") {
                out.insert("similarity".into(), json!(s.to_ascii_lowercase()));
            }
            if let Some(s) = take_str(&mut ss, "element_type") {
                out.insert("element_type".into(), json!(s.to_ascii_lowercase()));
            }
        }
        _ => {
            out.insert("model".into(), json!(model));
            if let Some(h) = take_str(&mut ss, "hidden_field") {
                out.insert("hidden_field".into(), json!(h));
            }
            if service == SPARSE || service == SPARSE_ALT {
                let ret = ss.shift_remove("should_return_hidden_field").and_then(|v| v.as_bool()).unwrap_or(false);
                out.insert("should_return_hidden_field".into(), json!(ret));
            }
        }
    }
    if !errors.is_empty() {
        return Err(validation(&errors));
    }
    let mut secrets = Map::new();
    match take_str(&mut ss, "api_key") {
        Some(k) => {
            secrets.insert("api_key".into(), json!(k));
        }
        None => return Err(validation(&["missing api_key"])),
    }
    let task_settings = if service == RERANK {
        rerank_settings(&Map::new(), &mut ts_in)
    } else {
        let mut t = Map::new();
        if let Some(v) = ts_in.shift_remove("temperature").filter(Value::is_i64) {
            t.insert("temperature".into(), v);
        }
        t
    };
    Ok(Endpoint {
        inference_id: id.to_string(),
        task_type: task,
        service: service.to_string(),
        service_settings: out,
        task_settings,
        chunking_settings: None,
        secrets,
        preconfigured: false,
    })
}

fn as_f32(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|f| f as f32),
        Value::String(s) => java_parse_float(s),
        _ => None,
    }
}

/// The reranking `TestTaskSettings`: `base` updated by the keys in `new`.
/// A missing `min_score` is random in `[-1, 1)` in the plugin (seeded per
/// test run); here it is a fixed value in that range.
fn rerank_settings(base: &Map<String, Value>, new: &mut Map<String, Value>) -> Map<String, Value> {
    let mut t = Map::new();
    let ul = new.get("use_text_length").map(|v| v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"))).or_else(|| base.get("use_text_length").and_then(Value::as_bool)).unwrap_or(false);
    let ms = new.get("min_score").and_then(as_f32).or_else(|| base.get("min_score").and_then(as_f32)).unwrap_or(0.0);
    let rd = new.get("result_diff").and_then(as_f32).or_else(|| base.get("result_diff").and_then(as_f32)).unwrap_or(0.2);
    t.insert("use_text_length".into(), json!(ul));
    t.insert("min_score".into(), json!(ms as f64));
    t.insert("result_diff".into(), json!(rd as f64));
    let int = |v: &Value| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    if let Some(n) = new.get("top_n").and_then(int).or_else(|| base.get("top_n").and_then(int)) {
        t.insert("top_n".into(), json!(n));
    }
    let flag = |v: &Value| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"));
    if let Some(h) = new.get("hide_top_n").and_then(flag).or_else(|| base.get("hide_top_n").and_then(flag)) {
        t.insert("hide_top_n".into(), json!(h));
    }
    t
}

/// The `service_settings` the GET API shows (`getFilteredXContentObject`).
pub fn filtered_settings(ep: &Endpoint) -> Map<String, Value> {
    let ss = &ep.service_settings;
    match ep.service.as_str() {
        SPARSE | SPARSE_ALT => {
            let mut m = Map::new();
            m.insert("model".into(), ss.get("model").cloned().unwrap_or(Value::Null));
            if ss.get("should_return_hidden_field").and_then(Value::as_bool) == Some(true) {
                if let Some(h) = ss.get("hidden_field") {
                    m.insert("hidden_field".into(), h.clone());
                }
            }
            m
        }
        _ => ss.clone(),
    }
}

/// `similarity` of a dense mock endpoint (`cosine` when unset).
pub fn similarity(ep: &Endpoint) -> String {
    ep.service_settings.get("similarity").and_then(Value::as_str).unwrap_or("cosine").to_string()
}

fn element_type(ep: &Endpoint) -> String {
    ep.service_settings.get("element_type").and_then(Value::as_str).unwrap_or("float").to_string()
}

/// `TestSparseInferenceServiceExtension.generateEmbedding`.
pub fn sparse_embedding(input: &str) -> Map<String, Value> {
    let h = java_hash(input);
    let abs = if h == i32::MIN { i32::MAX } else { h.abs() };
    let mut m = Map::new();
    for j in 0..5 {
        let w = abs as f32 + 1.0f32 + j as f32;
        m.insert(format!("feature_{j}"), json!(w as f64));
    }
    m
}

/// `TestDenseInferenceServiceExtension.generateEmbedding`.
pub fn dense_embedding(input: &str, dims: usize, element_type: &str, similarity: &str) -> Vec<f32> {
    let len = if element_type == "bit" { dims / 8 } else { dims };
    let bytes: Vec<f32> = java_hash(input).to_string().bytes().map(|b| b as f32).collect();
    let mut out: Vec<f32> = Vec::with_capacity(len);
    while out.len() + bytes.len() <= len {
        out.extend_from_slice(&bytes);
    }
    let rest = len - out.len();
    out.extend_from_slice(&bytes[..rest]);
    if similarity == "dot_product" {
        let mag = out.iter().fold(0f32, |a, b| a + b * b).sqrt();
        out = out.into_iter().map(|v| v / mag).collect();
    }
    out
}

/// One ranked document of the reranking mock.
fn ranked(input: &[String], ts: &Map<String, Value>) -> Vec<(usize, f32)> {
    let n = input.len();
    let parsed: Option<Vec<f32>> = input.iter().map(|s| java_parse_float(s)).collect();
    let mut rows: Vec<(usize, f32)> = match parsed {
        Some(scores) => scores.into_iter().enumerate().collect(),
        None => {
            let ul = ts.get("use_text_length").and_then(Value::as_bool).unwrap_or(false);
            let ms = ts.get("min_score").and_then(as_f32).unwrap_or(0.0);
            let rd = ts.get("result_diff").and_then(as_f32).unwrap_or(0.2);
            (0..n)
                .map(|i| {
                    let idx = n - 1 - i;
                    let s = if ul { 1f32 / input[idx].chars().count() as f32 } else { ms + rd * (n - i) as f32 };
                    (idx, s)
                })
                .collect()
        }
    };
    // Descending score; stable for ties (`Comparator.reverseOrder()` / `-Float.compare`).
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    rows
}

/// Run an inference call against a mock endpoint (the `_inference` result).
pub fn infer(ep: &Endpoint, input: &[String], task_settings: &Map<String, Value>) -> Result<Value> {
    match ep.service.as_str() {
        SPARSE | SPARSE_ALT => Ok(json!({"sparse_embedding": input.iter().map(|t| json!({"is_truncated": false, "embedding": sparse_embedding(t)})).collect::<Vec<_>>()})),
        DENSE => {
            let dims = ep.service_settings.get("dimensions").and_then(Value::as_u64).unwrap_or(0) as usize;
            let (et, sim) = (element_type(ep), similarity(ep));
            let rows: Vec<Value> = input.iter().map(|t| json!({"embedding": dense_embedding(t, dims, &et, &sim).into_iter().map(|f| f as f64).collect::<Vec<_>>()})).collect();
            // `GenericDenseEmbeddingFloatResults` (`embeddings`) for the `embedding` task.
            Ok(if ep.task_type == Task::Embedding { json!({"embeddings": rows}) } else { json!({"text_embedding": rows}) })
        }
        RERANK => {
            let mut over = task_settings.clone();
            let ts = rerank_settings(&ep.task_settings, &mut over);
            let mut rows = ranked(input, &ts);
            // `TestTaskSettings.topN` truncates the ranked results.
            let top_n = over.get("top_n").or_else(|| ep.task_settings.get("top_n")).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
            if let Some(n) = top_n {
                rows.truncate(n as usize);
            }
            Ok(json!({"rerank": rows.into_iter().map(|(i, s)| json!({"index": i, "relevance_score": s as f64, "text": input[i]})).collect::<Vec<_>>()}))
        }
        _ => Ok(json!({"completion": input.iter().map(|t| json!({"result": t.to_uppercase()})).collect::<Vec<_>>()})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_hash_matches() {
        assert_eq!(java_hash(""), 0);
        assert_eq!(java_hash("a"), 97);
        assert_eq!(java_hash("hello"), 99162322);
        assert_eq!(java_hash("inference test"), "inference test".encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)));
    }

    #[test]
    fn dense_repeats_hash_digits() {
        // hashCode("a") = 97 -> "97" -> [57, 55]
        assert_eq!(dense_embedding("a", 5, "float", "cosine"), vec![57.0, 55.0, 57.0, 55.0, 57.0]);
        let u = dense_embedding("a", 2, "float", "dot_product");
        assert!((u[0] * u[0] + u[1] * u[1] - 1.0).abs() < 1e-6);
        assert_eq!(dense_embedding("a", 16, "bit", "cosine").len(), 2);
    }

    #[test]
    fn rerank_parses_scores_or_falls_back() {
        let inp: Vec<String> = ["0", "1", "-1"].iter().map(|s| s.to_string()).collect();
        assert_eq!(ranked(&inp, &Map::new()), vec![(1, 1.0), (0, 0.0), (2, -1.0)]);
        let inp: Vec<String> = ["a", "bb", "ccc"].iter().map(|s| s.to_string()).collect();
        let ts = json!({"min_score": 0.0, "result_diff": 0.2}).as_object().unwrap().clone();
        let r = ranked(&inp, &ts);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2, 1, 0]);
        let ts = json!({"use_text_length": true}).as_object().unwrap().clone();
        assert_eq!(ranked(&inp, &ts).iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn parse_validation() {
        let b = json!({"service_settings": {"api_key": "k"}}).as_object().unwrap().clone();
        let e = parse(Task::SparseEmbedding, "x", SPARSE, &b).unwrap_err();
        assert_eq!(e.reason, "Validation Failed: 1: missing model;");
        let b = json!({"service_settings": {"model": "m"}}).as_object().unwrap().clone();
        assert_eq!(parse(Task::TextEmbedding, "x", DENSE, &b).unwrap_err().reason, "Validation Failed: 1: missing dimensions;");
        let b = json!({"service_settings": {"model": "m", "api_key": "k", "dimensions": 4, "similarity": "COSINE"}}).as_object().unwrap().clone();
        let ep = parse(Task::TextEmbedding, "x", DENSE, &b).unwrap();
        assert_eq!(Value::Object(ep.service_settings), json!({"model": "m", "dimensions": 4, "similarity": "cosine"}));
    }
}
