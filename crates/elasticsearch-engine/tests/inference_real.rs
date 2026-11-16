//! Runs the real multilingual-e5-small model when its files are present
//! (`ES_ML_E5_DIR`, default `~/ml-e5`) and the `candle-models` feature is on.
#![cfg(feature = "candle-models")]
use elasticsearch_engine::inference::real;
use std::path::PathBuf;

#[test]
fn e5_semantic_similarity_is_sensible() {
    let dir = std::env::var("ES_ML_E5_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("ml-e5"));
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no model files in {}", dir.display());
        return;
    }
    let texts: Vec<String> = ["Whales and dolphins are marine mammals that live in the ocean.", "Simmer the tomato sauce with garlic and basil.", "The central bank raised interest rates."].iter().map(|s| s.to_string()).collect();
    let docs = real::try_embed(&dir, &texts, "ingest").expect("model loads");
    let q = real::try_embed(&dir, &["which animals live in the sea".to_string()], "search").unwrap().remove(0);
    assert_eq!(q.len(), 384);
    let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let s: Vec<f32> = docs.iter().map(|d| cos(&q, d)).collect();
    eprintln!("scores {s:?}");
    assert!(s[0] > s[1] && s[0] > s[2], "ocean doc must rank first: {s:?}");
    assert!((cos(&q, &q) - 1.0).abs() < 1e-3);
}

/// With the model files installed in the data path, the default E5 endpoint
/// runs the real model without any opt-in, and reports `candle`.
#[test]
fn e5_endpoint_runs_real_model_from_data_path() {
    use elasticsearch_engine::inference::{backend, InferenceService};
    use serde_json::json;
    let src = std::env::var("ES_ML_E5_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("ml-e5"));
    if !src.join("model.safetensors").exists() {
        eprintln!("skipping: no model files in {}", src.display());
        return;
    }
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("ml-models")).unwrap();
    std::os::unix::fs::symlink(&src, d.path().join("ml-models").join(".multilingual-e5-small")).unwrap();
    let s = InferenceService::open(d.path());
    let out = s.infer(None, ".multilingual-e5-small-elasticsearch", &json!({"input": ["hello world"], "input_type": "ingest"})).unwrap();
    let v: Vec<f32> = out["text_embedding"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    assert_eq!(v.len(), 384);
    let direct = real::try_embed(&src, &["hello world".to_string()], "ingest").unwrap().remove(0);
    assert!(v.iter().zip(&direct).all(|(a, b)| (a - b).abs() < 1e-5), "the endpoint must return the real model's embedding");
    assert!(s.fallback_warning(".multilingual-e5-small-elasticsearch").is_none());
    assert_eq!(backend::pin_of(s.state_dir(), ".multilingual-e5-small-elasticsearch").as_deref(), Some("candle"));
    // Enabling the stand-ins does not replace a model that can run.
    s.set_builtin_fallback(true);
    assert_eq!(s.get(None, Some(".multilingual-e5-small-elasticsearch")).unwrap()["endpoints"][0]["implementation"], "candle");
}
