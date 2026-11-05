//! Kibana compatibility: request tracing and the few endpoints Kibana calls
//! that no other module owns.
//!
//! Tracing: set `ES_TRACE_HTTP=/path/to/file.ndjson` to append one JSON line
//! per request (`method`, `uri`, `status`, `took_ms`, request body, response
//! body; bodies truncated to `ES_TRACE_HTTP_MAX` bytes, default 4096). Set
//! `ES_TRACE_HTTP_ERRORS=1` to log only responses with status >= 400.

use crate::{RespBody, R};
use axum::body::Bytes;
use serde_json::json;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

struct Tracer {
    out: Mutex<std::fs::File>,
    max: usize,
    errors_only: bool,
}

fn tracer() -> Option<&'static Tracer> {
    static T: OnceLock<Option<Tracer>> = OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var("ES_TRACE_HTTP").ok().filter(|p| !p.is_empty())?;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?;
        let max = std::env::var("ES_TRACE_HTTP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let errors_only = matches!(std::env::var("ES_TRACE_HTTP_ERRORS").as_deref(), Ok("1") | Ok("true"));
        Some(Tracer { out: Mutex::new(file), max, errors_only })
    })
    .as_ref()
}

/// Whether request tracing is on (checked once per request before cloning).
pub fn trace_enabled() -> bool {
    tracer().is_some()
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...[{} bytes]", &s[..end], s.len())
}

/// Append one trace line for a finished request.
pub fn trace(method: &str, uri: &str, req_body: &Bytes, r: &R, started: std::time::Instant) {
    let t = match tracer() {
        Some(t) => t,
        None => return,
    };
    let (status, body) = match r {
        Ok(resp) => {
            let b = match &resp.body {
                RespBody::Json(v) | RespBody::RetainedJson {value:v,..} => v.to_string(),
                RespBody::Raw(s) | RespBody::Text(s) => s.clone(),
                RespBody::Bytes(b) => format!("<{} bytes>", b.len()),
                RespBody::Owned {bytes,json} => if *json {String::from_utf8_lossy(bytes).into_owned()} else {format!("<{} bytes>",bytes.len())},
                RespBody::Empty => String::new(),
            };
            (resp.status, b)
        }
        Err(e) => (e.status, e.to_json_with_trace(false).to_string()),
    };
    if t.errors_only && status < 400 {
        return;
    }
    let line = json!({
        "ts": chrono_now(),
        "method": method,
        "uri": uri,
        "status": status,
        "took_ms": started.elapsed().as_millis() as u64,
        "req": clip(&String::from_utf8_lossy(req_body), t.max),
        "resp": clip(&body, t.max),
    });
    if let Ok(mut f) = t.out.lock() {
        let _ = writeln!(f, "{line}");
    }
}

fn chrono_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The endpoints ES 8.19 ships preconfigured (`.elser-2-elasticsearch`,
/// `.multilingual-e5-small-elasticsearch`, `.rerank-v1-elasticsearch`).
fn preconfigured_endpoints() -> Vec<serde_json::Value> {
    let alloc = json!({"enabled": true, "min_number_of_allocations": 0, "max_number_of_allocations": 32});
    let chunking = json!({"strategy": "sentence", "max_chunk_size": 250, "sentence_overlap": 1});
    vec![
        json!({"inference_id": ".elser-2-elasticsearch", "task_type": "sparse_embedding", "service": "elasticsearch",
               "service_settings": {"num_threads": 1, "model_id": ".elser_model_2_linux-x86_64", "adaptive_allocations": alloc}, "chunking_settings": chunking}),
        json!({"inference_id": ".multilingual-e5-small-elasticsearch", "task_type": "text_embedding", "service": "elasticsearch",
               "service_settings": {"num_threads": 1, "model_id": ".multilingual-e5-small_linux-x86_64", "adaptive_allocations": alloc}, "chunking_settings": chunking}),
        json!({"inference_id": ".rerank-v1-elasticsearch", "task_type": "rerank", "service": "elasticsearch",
               "service_settings": {"num_threads": 1, "model_id": ".rerank-v1", "adaptive_allocations": alloc}, "task_settings": {"return_documents": true}}),
    ]
}

/// `/_inference[/{task_type}][/{id}]`: this node hosts only the preconfigured
/// endpoints; a named endpoint that is not one of them is a 404
/// `resource_not_found_exception`, as ES answers for an unknown id.
pub fn inference(req: &crate::Req) -> R {
    use elasticsearch_engine::error::EsError;
    if req.method != axum::http::Method::GET {
        return crate::not_found_handler(req);
    }
    const TASK_TYPES: &[&str] = &["sparse_embedding", "text_embedding", "rerank", "completion", "chat_completion"];
    let (task_type, id) = match (req.segs.get(1).map(String::as_str), req.segs.get(2).map(String::as_str)) {
        (None, _) => (None, None),
        (Some(a), None) if TASK_TYPES.contains(&a) => (Some(a), None),
        (Some(a), None) => (None, Some(a)),
        (Some(a), Some(b)) => (Some(a), Some(b)),
    };
    if let Some(t) = task_type {
        if !TASK_TYPES.contains(&t) {
            return Err(EsError::new(400, "illegal_argument_exception", format!("No enum constant org.elasticsearch.inference.TaskType.{}", t.to_uppercase())));
        }
    }
    let all = preconfigured_endpoints();
    let pick = |f: &dyn Fn(&serde_json::Value) -> bool| all.iter().filter(|e| f(e)).cloned().collect::<Vec<_>>();
    let by_type = |e: &serde_json::Value| task_type.is_none_or(|t| e["task_type"] == t);
    match id {
        None | Some("_all") => Ok(crate::Resp::ok(json!({"endpoints": pick(&by_type)}))),
        Some(id) => {
            let found = pick(&|e| by_type(e) && e["inference_id"] == id);
            if found.is_empty() {
                Err(EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))
            } else {
                Ok(crate::Resp::ok(json!({"endpoints": found})))
            }
        }
    }
}
