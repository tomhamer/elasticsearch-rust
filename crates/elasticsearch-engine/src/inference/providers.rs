//! Provider adapters for the `_inference` API: request building and
//! response parsing for the hosted services, behind a [`Transport`] so the
//! unit tests can run against a local mock server.
//!
//! The built-in [`PlainHttp`] speaks HTTP/1.1 over `std::net`; another
//! transport can be installed with [`set_transport`].

use serde_json::{json, Map, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Result of an HTTP call: status and parsed JSON body (`Null` when empty).
pub type HttpResult = Result<(u16, Value), String>;

/// A blocking HTTP POST of a JSON body.
pub trait Transport: Send + Sync {
    fn post(&self, url: &str, headers: &[(String, String)], body: &Value, timeout: Duration) -> HttpResult;
}

static TRANSPORT: OnceLock<Arc<dyn Transport>> = OnceLock::new();

/// Install the transport used for provider calls (first call wins).
pub fn set_transport(t: Arc<dyn Transport>) {
    let _ = TRANSPORT.set(t);
}

pub fn transport() -> Arc<dyn Transport> {
    TRANSPORT.get_or_init(|| Arc::new(PlainHttp)).clone()
}

/// HTTP/1.1 over a plain TCP socket (no TLS).
pub struct PlainHttp;

fn split_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        if url.starts_with("https://") {
            "https is not supported by the built-in transport".to_string()
        } else {
            format!("unsupported url [{url}]")
        }
    })?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| format!("bad port in [{url}]"))?),
        None => (hostport.to_string(), 80),
    };
    Ok((host, port, path))
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else { break };
        let size = usize::from_str_radix(std::str::from_utf8(&b[..nl]).unwrap_or("0").trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
        b = &b[nl + 2..];
        if size == 0 || b.len() < size {
            break;
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size..];
        if b.len() >= 2 {
            b = &b[2..];
        }
    }
    out
}

impl Transport for PlainHttp {
    fn post(&self, url: &str, headers: &[(String, String)], body: &Value, timeout: Duration) -> HttpResult {
        let (host, port, path) = split_url(url)?;
        let addr = format!("{host}:{port}");
        let sock = std::net::ToSocketAddrs::to_socket_addrs(&addr).map_err(|e| e.to_string())?.next().ok_or("no address")?;
        let mut s = TcpStream::connect_timeout(&sock, timeout).map_err(|e| match e.kind() {
            std::io::ErrorKind::ConnectionRefused => "Connection refused (connect failed)".to_string(),
            _ => e.to_string(),
        })?;
        s.set_read_timeout(Some(timeout)).ok();
        s.set_write_timeout(Some(timeout)).ok();
        let payload = serde_json::to_vec(body).map_err(|e| e.to_string())?;
        let mut req = format!("POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: application/json\r\nConnection: close\r\nContent-Length: {}\r\n", payload.len());
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        s.write_all(&payload).map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("malformed http response")?;
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("malformed status line")?;
        let mut body = raw[split + 4..].to_vec();
        if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            body = dechunk(&body);
        }
        let v = if body.is_empty() { Value::Null } else { serde_json::from_slice(&body).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).to_string())) };
        Ok((status, v))
    }
}

// ---------------------------------------------------------------- adapters

/// Task types a hosted service handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    TextEmbedding,
    /// ES 9.4+ `embedding`: multimodal-capable dense embeddings (text here).
    Embedding,
    SparseEmbedding,
    Rerank,
    Completion,
    ChatCompletion,
}

impl Task {
    pub fn parse(s: &str) -> Option<Task> {
        Some(match s {
            "text_embedding" => Task::TextEmbedding,
            "embedding" => Task::Embedding,
            "sparse_embedding" => Task::SparseEmbedding,
            "rerank" => Task::Rerank,
            "completion" => Task::Completion,
            "chat_completion" => Task::ChatCompletion,
            _ => return None,
        })
    }
    /// A dense embedding task (`text_embedding` or `embedding`).
    pub fn is_dense(self) -> bool {
        matches!(self, Task::TextEmbedding | Task::Embedding)
    }
    pub fn name(self) -> &'static str {
        match self {
            Task::TextEmbedding => "text_embedding",
            Task::Embedding => "embedding",
            Task::SparseEmbedding => "sparse_embedding",
            Task::Rerank => "rerank",
            Task::Completion => "completion",
            Task::ChatCompletion => "chat_completion",
        }
    }
}

/// A fully built provider request.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

/// Everything an adapter needs from an endpoint.
pub struct Call<'a> {
    pub service: &'a str,
    pub task: Task,
    pub service_settings: &'a Map<String, Value>,
    pub secrets: &'a Map<String, Value>,
    pub task_settings: Map<String, Value>,
    pub input: &'a [String],
    pub query: Option<&'a str>,
}

fn s<'a>(m: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(Value::as_str)
}

fn join(base: &str, suffix: &str) -> String {
    if base.ends_with(suffix) {
        base.to_string()
    } else {
        format!("{}{}", base.trim_end_matches('/'), suffix)
    }
}

/// Build the HTTP request for a call.
pub fn build_request(c: &Call) -> Result<ProviderRequest, String> {
    let ss = c.service_settings;
    let key = s(c.secrets, "api_key").unwrap_or("").to_string();
    let model = s(ss, "model_id").or_else(|| s(ss, "model")).unwrap_or("").to_string();
    let bearer = vec![("Authorization".to_string(), format!("Bearer {key}"))];
    let chat_messages = |inputs: &[String]| -> Value { json!(inputs.iter().map(|t| json!({"role": "user", "content": t})).collect::<Vec<_>>()) };
    match c.service {
        "openai" | "mistral" | "deepseek" => {
            let default_host = match c.service {
                "mistral" => "https://api.mistral.ai/v1",
                "deepseek" => "https://api.deepseek.com/v1",
                _ => "https://api.openai.com/v1",
            };
            let mut headers = bearer;
            if let Some(o) = s(ss, "organization_id") {
                headers.push(("OpenAI-Organization".into(), o.into()));
            }
            match c.task {
                Task::TextEmbedding => {
                    let url = match s(ss, "url") {
                        Some(u) => u.to_string(),
                        None => join(default_host, "/embeddings"),
                    };
                    let mut body = json!({"input": c.input, "model": model});
                    if let Some(d) = ss.get("dimensions") {
                        if c.service == "openai" && ss.get("dimensions_set_by_user").and_then(Value::as_bool).unwrap_or(true) {
                            body["dimensions"] = d.clone();
                        }
                    }
                    if let Some(u) = c.task_settings.get("user") {
                        body["user"] = u.clone();
                    }
                    Ok(ProviderRequest { url, headers, body })
                }
                Task::Completion | Task::ChatCompletion => {
                    let url = match s(ss, "url") {
                        Some(u) => u.to_string(),
                        None => join(default_host, "/chat/completions"),
                    };
                    let mut body = json!({"messages": chat_messages(c.input), "model": model, "n": 1});
                    if let Some(u) = c.task_settings.get("user") {
                        body["user"] = u.clone();
                    }
                    Ok(ProviderRequest { url, headers, body })
                }
                t => Err(format!("The [{}] service does not support task type [{}]", c.service, t.name())),
            }
        }
        "azureopenai" => {
            let resource = s(ss, "resource_name").unwrap_or("");
            let dep = s(ss, "deployment_id").unwrap_or("");
            let ver = s(ss, "api_version").unwrap_or("");
            let tail = if c.task == Task::TextEmbedding { "embeddings" } else { "chat/completions" };
            let url = format!("https://{resource}.openai.azure.com/openai/deployments/{dep}/{tail}?api-version={ver}");
            let headers = if let Some(tok) = s(c.secrets, "entra_id") { vec![("Authorization".to_string(), format!("Bearer {tok}"))] } else { vec![("api-key".to_string(), key)] };
            let body = match c.task {
                Task::TextEmbedding => {
                    let mut b = json!({"input": c.input});
                    if let Some(u) = c.task_settings.get("user") {
                        b["user"] = u.clone();
                    }
                    b
                }
                Task::Completion => json!({"messages": chat_messages(c.input), "n": 1}),
                t => return Err(format!("The [azureopenai] service does not support task type [{}]", t.name())),
            };
            Ok(ProviderRequest { url, headers, body })
        }
        "cohere" => {
            let base = s(ss, "url").unwrap_or("https://api.cohere.ai");
            match c.task {
                Task::TextEmbedding => {
                    let mut body = json!({"texts": c.input, "embedding_types": [s(ss, "embedding_type").map(|t| if t == "byte" { "int8" } else { t }).unwrap_or("float")]});
                    if !model.is_empty() {
                        body["model"] = json!(model);
                    }
                    if let Some(i) = c.task_settings.get("input_type") {
                        body["input_type"] = i.clone();
                    }
                    if let Some(t) = c.task_settings.get("truncate") {
                        body["truncate"] = t.clone();
                    }
                    Ok(ProviderRequest { url: join(base, "/v2/embed"), headers: bearer, body })
                }
                Task::Rerank => {
                    let mut body = json!({"query": c.query.unwrap_or(""), "documents": c.input});
                    if !model.is_empty() {
                        body["model"] = json!(model);
                    }
                    if let Some(v) = c.task_settings.get("top_n") {
                        body["top_n"] = v.clone();
                    }
                    if let Some(v) = c.task_settings.get("return_documents") {
                        body["return_documents"] = v.clone();
                    }
                    Ok(ProviderRequest { url: join(base, "/v2/rerank"), headers: bearer, body })
                }
                Task::Completion => Ok(ProviderRequest { url: join(base, "/v2/chat"), headers: bearer, body: json!({"messages": chat_messages(c.input), "model": model}) }),
                t => Err(format!("The [cohere] service does not support task type [{}]", t.name())),
            }
        }
        "hugging_face" | "huggingface" => {
            let url = s(ss, "url").ok_or("[service_settings] does not contain the required setting [url]")?.to_string();
            match c.task {
                Task::Rerank => Ok(ProviderRequest { url, headers: bearer, body: json!({"query": c.query.unwrap_or(""), "texts": c.input, "return_text": c.task_settings.get("return_documents").cloned().unwrap_or(json!(false))}) }),
                Task::Completion | Task::ChatCompletion => Ok(ProviderRequest { url, headers: bearer, body: json!({"messages": chat_messages(c.input), "model": model, "n": 1}) }),
                _ => Ok(ProviderRequest { url, headers: bearer, body: json!({"inputs": c.input}) }),
            }
        }
        "anthropic" => {
            if c.task != Task::Completion {
                return Err(format!("The [anthropic] service does not support task type [{}]", c.task.name()));
            }
            let max = c.task_settings.get("max_tokens").cloned().unwrap_or(json!(1024));
            let mut body = json!({"model": model, "messages": chat_messages(c.input), "max_tokens": max});
            for k in ["temperature", "top_p", "top_k"] {
                if let Some(v) = c.task_settings.get(k) {
                    body[k] = v.clone();
                }
            }
            let headers = vec![("x-api-key".to_string(), key), ("anthropic-version".to_string(), "2023-06-01".to_string())];
            Ok(ProviderRequest { url: s(ss, "url").map(str::to_string).unwrap_or_else(|| "https://api.anthropic.com/v1/messages".into()), headers, body })
        }
        "voyageai" | "jinaai" => {
            let base = if c.service == "voyageai" { "https://api.voyageai.com/v1" } else { "https://api.jina.ai/v1" };
            match c.task {
                Task::TextEmbedding => {
                    let mut body = json!({"input": c.input, "model": model});
                    if c.service == "jinaai" {
                        if let Some(t) = c.task_settings.get("task").or_else(|| c.task_settings.get("input_type")) {
                            body["task"] = t.clone();
                        }
                    } else if let Some(t) = c.task_settings.get("input_type") {
                        body["input_type"] = t.clone();
                    }
                    Ok(ProviderRequest { url: s(ss, "url").map(str::to_string).unwrap_or_else(|| format!("{base}/embeddings")), headers: bearer, body })
                }
                Task::Rerank => {
                    let mut body = json!({"query": c.query.unwrap_or(""), "documents": c.input, "model": model});
                    if let Some(v) = c.task_settings.get("top_k").or_else(|| c.task_settings.get("top_n")) {
                        body[if c.service == "voyageai" { "top_k" } else { "top_n" }] = v.clone();
                    }
                    if let Some(v) = c.task_settings.get("return_documents") {
                        body["return_documents"] = v.clone();
                    }
                    Ok(ProviderRequest { url: s(ss, "url").map(str::to_string).unwrap_or_else(|| format!("{base}/rerank")), headers: bearer, body })
                }
                t => Err(format!("The [{}] service does not support task type [{}]", c.service, t.name())),
            }
        }
        other => Err(format!("provider adapter for service [{other}] is not available in this build")),
    }
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

/// Parse a provider response into the ES `_inference` result object
/// (`{"text_embedding": [...]}` etc.).
pub fn parse_response(c: &Call, resp: &Value) -> Result<Value, String> {
    let bad = |what: &str| format!("Failed to parse {} response from [{}]: {what}", c.task.name(), c.service);
    match c.task {
        Task::TextEmbedding | Task::Embedding => {
            let rows: Vec<Vec<f64>> = if let Some(d) = resp.get("data").and_then(Value::as_array) {
                let mut rows: Vec<(i64, Vec<f64>)> = Vec::new();
                for (i, e) in d.iter().enumerate() {
                    let idx = e.get("index").and_then(Value::as_i64).unwrap_or(i as i64);
                    rows.push((idx, e.get("embedding").and_then(floats).ok_or_else(|| bad("missing embedding"))?));
                }
                rows.sort_by_key(|r| r.0);
                rows.into_iter().map(|r| r.1).collect()
            } else if let Some(e) = resp.get("embeddings") {
                // cohere: {"embeddings": {"float": [[..]]}} or {"embeddings": [[..]]}
                let arr = e.get("float").or_else(|| e.get("int8")).or(Some(e)).and_then(Value::as_array).ok_or_else(|| bad("missing embeddings"))?;
                arr.iter().map(|r| floats(r).ok_or_else(|| bad("bad embedding"))).collect::<Result<_, _>>()?
            } else if let Some(arr) = resp.as_array() {
                // huggingface: [[..], [..]]
                arr.iter().map(|r| floats(r).ok_or_else(|| bad("bad embedding"))).collect::<Result<_, _>>()?
            } else {
                return Err(bad("unrecognised response"));
            };
            let key = if c.task == Task::Embedding { "embeddings" } else { "text_embedding" };
            Ok(json!({ key: rows.into_iter().map(|r| json!({"embedding": r})).collect::<Vec<_>>() }))
        }
        Task::SparseEmbedding => {
            let arr = resp.as_array().ok_or_else(|| bad("expected array"))?;
            let out: Vec<Value> = arr.iter().map(|m| json!({"is_truncated": false, "embedding": m})).collect();
            Ok(json!({"sparse_embedding": out}))
        }
        Task::Rerank => {
            let arr = resp.get("results").or_else(|| resp.get("data")).or(Some(resp)).and_then(Value::as_array).ok_or_else(|| bad("missing results"))?;
            let ret_docs = c.task_settings.get("return_documents").and_then(Value::as_bool).unwrap_or(false);
            let mut out: Vec<Value> = Vec::new();
            for r in arr {
                let idx = r.get("index").and_then(Value::as_i64).ok_or_else(|| bad("missing index"))?;
                let score = r.get("relevance_score").or_else(|| r.get("score")).and_then(Value::as_f64).ok_or_else(|| bad("missing score"))?;
                let mut o = json!({"index": idx, "relevance_score": score});
                // Pass through the document text only when the provider returned it.
                if ret_docs {
                    if let Some(t) = r.pointer("/document/text").and_then(Value::as_str).or_else(|| r.get("document").and_then(Value::as_str)) {
                        o["text"] = json!(t);
                    }
                }
                out.push(o);
            }
            out.sort_by(|a, b| b["relevance_score"].as_f64().partial_cmp(&a["relevance_score"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
            Ok(json!({"rerank": out}))
        }
        Task::Completion | Task::ChatCompletion => {
            let text = if let Some(t) = resp.pointer("/choices/0/message/content").and_then(Value::as_str) {
                t.to_string()
            } else if let Some(arr) = resp.get("content").and_then(Value::as_array) {
                arr.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("")
            } else if let Some(t) = resp.pointer("/message/content/0/text").and_then(Value::as_str) {
                t.to_string()
            } else {
                return Err(bad("missing completion text"));
            };
            Ok(json!({"completion": [{"result": text}]}))
        }
    }
}

/// Run a call against a provider and return the ES result object.
pub fn execute(c: &Call, timeout: Duration) -> Result<Value, (u16, String)> {
    let req = build_request(c).map_err(|e| (400, e))?;
    let (status, body) = transport().post(&req.url, &req.headers, &req.body, timeout).map_err(|e| (502, e))?;
    if status >= 400 {
        let detail = body.pointer("/error/message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| body.to_string());
        return Err((status, format!("Error message: [{detail}]")));
    }
    parse_response(c, &body).map_err(|e| (502, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call<'a>(service: &'a str, task: Task, ss: &'a Map<String, Value>, sec: &'a Map<String, Value>, input: &'a [String]) -> Call<'a> {
        Call { service, task, service_settings: ss, secrets: sec, task_settings: Map::new(), input, query: Some("q") }
    }

    #[test]
    fn openai_embedding_request_and_response() {
        let ss = json!({"model_id": "m", "url": "http://x/v1/embeddings"}).as_object().unwrap().clone();
        let sec = json!({"api_key": "k"}).as_object().unwrap().clone();
        let input = vec!["a".to_string(), "b".to_string()];
        let c = call("openai", Task::TextEmbedding, &ss, &sec, &input);
        let r = build_request(&c).unwrap();
        assert_eq!(r.body, json!({"input": ["a", "b"], "model": "m"}));
        assert_eq!(r.headers[0], ("Authorization".to_string(), "Bearer k".to_string()));
        let out = parse_response(&c, &json!({"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})).unwrap();
        assert_eq!(out, json!({"text_embedding": [{"embedding": [1.0]}, {"embedding": [2.0]}]}));
    }

    #[test]
    fn cohere_rerank_sorted_with_text() {
        let ss = json!({"model_id": "m"}).as_object().unwrap().clone();
        let sec = json!({"api_key": "k"}).as_object().unwrap().clone();
        let input = vec!["a".to_string(), "b".to_string()];
        let mut c = call("cohere", Task::Rerank, &ss, &sec, &input);
        c.task_settings = json!({"return_documents": true}).as_object().unwrap().clone();
        let out = parse_response(&c, &json!({"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]})).unwrap();
        // The provider response carries no document text, so `text` is omitted.
        assert_eq!(out["rerank"][0], json!({"index": 1, "relevance_score": 0.9}));
        // When the provider returns the document, the text is passed through.
        let out = parse_response(&c, &json!({"results": [{"index": 0, "relevance_score": 0.1, "document": {"text": "a"}}, {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}}]})).unwrap();
        assert_eq!(out["rerank"][0], json!({"index": 1, "relevance_score": 0.9, "text": "b"}));
    }

    #[test]
    fn anthropic_completion() {
        let ss = json!({"model_id": "claude"}).as_object().unwrap().clone();
        let sec = json!({"api_key": "k"}).as_object().unwrap().clone();
        let input = vec!["hi".to_string()];
        let c = call("anthropic", Task::Completion, &ss, &sec, &input);
        let r = build_request(&c).unwrap();
        assert_eq!(r.body["max_tokens"], 1024);
        let out = parse_response(&c, &json!({"content": [{"type": "text", "text": "yo"}]})).unwrap();
        assert_eq!(out, json!({"completion": [{"result": "yo"}]}));
    }

    #[test]
    fn plain_http_roundtrip_against_local_server() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut sk, _) = l.accept().unwrap();
            // Read the full request (headers + body) before replying.
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = sk.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&raw[..pos]).to_ascii_lowercase();
                    if let Some(i) = head.find("content-length: ") {
                        let len: usize = head[i + 16..].split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
                        if raw.len() >= pos + 4 + len {
                            break;
                        }
                    }
                }
            }
            let body = r#"{"data":[{"index":0,"embedding":[0.5,0.25]}]}"#;
            let _ = sk.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        });
        let ss = json!({"model_id": "m", "url": format!("http://127.0.0.1:{port}/v1/embeddings")}).as_object().unwrap().clone();
        let sec = json!({"api_key": "k"}).as_object().unwrap().clone();
        let input = vec!["a".to_string()];
        let c = call("openai", Task::TextEmbedding, &ss, &sec, &input);
        let out = execute(&c, Duration::from_secs(5)).unwrap();
        assert_eq!(out, json!({"text_embedding": [{"embedding": [0.5, 0.25]}]}));
    }
}
