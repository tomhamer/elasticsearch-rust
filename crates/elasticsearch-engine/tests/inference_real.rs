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
