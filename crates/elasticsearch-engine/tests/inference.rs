//! `_inference` / local model / provider-adapter tests (public API only).
use elasticsearch_engine::inference::{local, models, providers::{self, Call, Task}, InferenceService};
use serde_json::{json, Map, Value};
use std::io::{Read, Write};
use std::time::Duration;

fn svc() -> (tempfile::TempDir, InferenceService) {
    let d = tempfile::tempdir().unwrap();
    let s = InferenceService::open(d.path());
    (d, s)
}

#[test]
fn defaults_and_error_wording() {
    let (_d, s) = svc();
    let all = s.get(None, Some("_all")).unwrap();
    assert_eq!(all["endpoints"].as_array().unwrap().len(), 3);
    assert_eq!(all["endpoints"][2]["task_settings"], json!({"return_documents": true}));
    assert_eq!(s.get(Some("badtype"), None).unwrap_err().reason, "Unknown task_type [badtype]");
    assert_eq!(s.get(None, Some("nope")).unwrap_err().reason, "Inference endpoint not found [nope]");
    assert_eq!(s.delete(Some("text_embedding"), ".elser-2-elasticsearch", false, false).unwrap_err().reason, "Requested task type [text_embedding] does not match the inference endpoint's task type [sparse_embedding]");
}

#[test]
fn put_validation_messages() {
    let (_d, s) = svc();
    let e = s.put("text_embedding", ".bad", &json!({})).unwrap_err();
    assert!(e.reason.starts_with("Validation Failed: 1: Invalid inference_id; '.bad' can contain"), "{}", e.reason);
    assert_eq!(s.put("text_embedding", "e1", &json!({"service": "nosuch"})).unwrap_err().reason, "Unknown service [nosuch]");
    assert_eq!(s.put("text_embedding", "e1", &json!({"service_settings": {}})).unwrap_err().reason, "Inference endpoint configuration is missing the [service] setting");
    assert_eq!(s.put("sparse_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k"}})).unwrap_err().reason, "The [openai] service does not support task type [sparse_embedding]");
    // ES 8.19: required service settings are checked before leftovers.
    assert_eq!(s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k", "bogus": 1}})).unwrap_err().reason, "Validation Failed: 1: [service_settings] does not contain the required setting [model_id];");
    assert_eq!(s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k", "model_id": "m", "bogus": 1}})).unwrap_err().reason, "Configuration contains settings [{bogus=1}] unknown to the [openai] service");
    let e = s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"model_id": "m"}})).unwrap_err();
    assert_eq!(e.reason, "Validation Failed: 1: [secret_settings] does not contain the required setting [api_key];");
}

#[test]
fn local_endpoint_lifecycle_and_infer() {
    let (d, s) = svc();
    let r = s.put("text_embedding", "my-e5", &json!({"service": "elasticsearch", "service_settings": {"model_id": ".multilingual-e5-small", "num_allocations": 1, "num_threads": 1}})).unwrap();
    assert_eq!(r["inference_id"], "my-e5");
    let out = s.infer(Some("text_embedding"), "my-e5", &json!({"input": ["hello world", "bye"]})).unwrap();
    assert_eq!(out["text_embedding"].as_array().unwrap().len(), 2);
    assert_eq!(out["text_embedding"][0]["embedding"].as_array().unwrap().len(), 384);
    let s2 = InferenceService::open(d.path());
    assert!(s2.get_endpoint("my-e5").is_some());
    assert_eq!(s.infer(Some("completion"), "my-e5", &json!({"input": "x"})).unwrap_err().reason, "Incompatible task_type, the requested type [completion] does not match the model type [text_embedding]");
    assert_eq!(s.infer(None, "my-e5", &json!({})).unwrap_err().reason, "Validation Failed: 1: Field [input] cannot be null;");
    assert_eq!(s.delete(None, "my-e5", false, false).unwrap(), json!({"acknowledged": true, "pipelines": [], "indexes": []}));
}

#[test]
fn rerank_default_endpoint() {
    let (_d, s) = svc();
    let out = s.infer(None, ".rerank-v1-elasticsearch", &json!({"query": "fox", "input": ["database revenue", "the quick fox"]})).unwrap();
    assert_eq!(out["rerank"][0]["index"], 1);
    assert_eq!(out["rerank"][0]["text"], "the quick fox");
}

#[test]
fn hosted_provider_against_mock_server_and_validation_probe() {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for _ in 0..3 {
            let (mut sk, _) = l.accept().unwrap();
            let mut raw = Vec::new();
            let mut buf = [0u8; 8192];
            loop {
                let n = sk.read(&mut buf).unwrap();
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(h) = text.find("\r\n\r\n") {
                    let cl: usize = text[..h].to_ascii_lowercase().split("content-length:").nth(1).and_then(|r| r.trim().split_whitespace().next().and_then(|v| v.parse().ok())).unwrap_or(0);
                    if raw.len() >= h + 4 + cl {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let req = String::from_utf8_lossy(&raw).to_string();
            let count = if req.contains("\"a\"") && req.contains("\"b\"") { 2 } else { 1 };
            let data: Vec<Value> = (0..count).map(|i| json!({"index": i, "embedding": [0.1 * (i as f64 + 1.0), 0.2, 0.3]})).collect();
            let body = json!({"data": data}).to_string();
            let _ = sk.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    let (_d, s) = svc();
    let put = s.put("text_embedding", "oe", &json!({"service": "openai", "service_settings": {"api_key": "sekret", "model_id": "m", "url": format!("http://127.0.0.1:{port}/v1/embeddings")}})).unwrap();
    assert_eq!(put["service_settings"]["dimensions"], 3);
    assert!(put["service_settings"].get("api_key").is_none(), "secrets must not be echoed");
    let out = s.infer(None, "oe", &json!({"input": ["a", "b"]})).unwrap();
    assert_eq!(out["text_embedding"].as_array().unwrap().len(), 2);
    assert!(!s.get(None, Some("oe")).unwrap().to_string().contains("sekret"));
}

#[test]
fn unreachable_provider_creation_error() {
    let (_d, s) = svc();
    let e = s.put("text_embedding", "oe", &json!({"service": "openai", "service_settings": {"api_key": "k", "model_id": "m", "url": "http://127.0.0.1:1/v1"}})).unwrap_err();
    assert_eq!(e.reason, "Could not complete inference endpoint creation as validation call to service threw an exception.");
    assert_eq!(e.caused_by.as_ref().unwrap().reason, "Failed to send OpenAI embeddings request. Cause: Connection refused (connect failed)");
}

#[test]
fn adapters_map_requests() {
    let ss: Map<String, Value> = json!({"model_id": "m"}).as_object().unwrap().clone();
    let sec: Map<String, Value> = json!({"api_key": "k"}).as_object().unwrap().clone();
    let input = vec!["hi".to_string()];
    let c = Call { service: "anthropic", task: Task::Completion, service_settings: &ss, secrets: &sec, task_settings: Map::new(), input: &input, query: None };
    let r = providers::build_request(&c).unwrap();
    assert_eq!(r.body["max_tokens"], 1024);
    let out = providers::parse_response(&c, &json!({"content": [{"type": "text", "text": "yo"}]})).unwrap();
    assert_eq!(out, json!({"completion": [{"result": "yo"}]}));
    let c = Call { service: "cohere", task: Task::Rerank, service_settings: &ss, secrets: &sec, task_settings: json!({"return_documents": true}).as_object().unwrap().clone(), input: &["a".to_string(), "b".to_string()], query: Some("q") };
    let out = providers::parse_response(&c, &json!({"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}}]})).unwrap();
    assert_eq!(out["rerank"][0], json!({"index": 1, "relevance_score": 0.9, "text": "b"}));
    let _ = Duration::from_secs(1);
}

#[test]
fn local_models_and_chunking() {
    let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let a = local::embed_dense("the quick brown fox jumps over the lazy dog", None);
    let near = local::embed_dense("a quick brown fox jumping", None);
    let far = local::embed_dense("quarterly revenue forecast for cloud databases", None);
    assert_eq!(a.len(), 384);
    assert!(cos(&a, &near) > cos(&a, &far));
    let text = "a b c d e f g h";
    let c = local::Chunking { strategy: "word".into(), max_chunk_size: 4, overlap: 2, sentence_overlap: 0 };
    let ch: Vec<&str> = local::chunk(text, &c).into_iter().map(|(s, e)| &text[s..e]).collect();
    // ES's WordBoundaryChunker starts an overlapping chunk at the preceding
    // word boundary, so it carries the whitespace before its first word
    // (WordBoundaryChunkerTests.assertChunkContents).
    assert_eq!(ch, vec!["a b c d", " c d e f", " e f g h"]);
    let text = "One two. Three four. Five six.";
    let c = local::Chunking { strategy: "sentence".into(), max_chunk_size: 4, overlap: 0, sentence_overlap: 0 };
    let ch: Vec<&str> = local::chunk(text, &c).into_iter().map(|(s, e)| &text[s..e]).collect();
    assert_eq!(ch, vec!["One two. Three four.", "Five six."]);
}

#[test]
fn trained_models_crud_and_infer() {
    let d = tempfile::tempdir().unwrap();
    let st = d.path().join("_state");
    std::fs::create_dir_all(&st).unwrap();
    assert_eq!(models::get(&st, None, true).unwrap()["count"], 1);
    models::put(&st, ".multilingual-e5-small", &json!({})).unwrap();
    assert_eq!(models::get(&st, None, true).unwrap()["count"], 2);
    // The model import task is still running: a second put conflicts and
    // deployments wait for it (here: time out).
    assert_eq!(models::put(&st, ".multilingual-e5-small", &json!({})).unwrap_err().status, 409);
    let params: std::collections::HashMap<String, String> = [("timeout".to_string(), "10ms".to_string())].into_iter().collect();
    assert_eq!(models::start_deployment(&st, ".multilingual-e5-small", &params).unwrap_err().status, 408);
    // Pytorch models must be deployed to infer.
    assert_eq!(models::infer(&st, ".multilingual-e5-small", &json!({"docs": [{"text_field": "hello"}]})).unwrap_err().status, 409);
    assert_eq!(models::delete(&st, ".multilingual-e5-small").unwrap(), json!({"acknowledged": true}));
    assert!(models::get(&st, Some("nope"), false).is_err());
}
