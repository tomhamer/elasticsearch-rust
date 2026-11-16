//! Provider adapters for the `_inference` API: request building and
//! response parsing for the hosted services, behind a [`Transport`] so the
//! unit tests can run against a local mock server.
//!
//! The built-in [`HttpClient`] speaks HTTP/1.1 over `std::net`, and HTTPS
//! over rustls (full certificate and host name verification against the
//! Mozilla roots plus the operating system's trust store, which honours
//! `SSL_CERT_FILE` / `SSL_CERT_DIR`), with connect, read/write and overall
//! timeouts. Another transport can be installed with [`set_transport`].

use elasticsearch_ssl::rustls;
use serde_json::{json, Map, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

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
    TRANSPORT.get_or_init(|| Arc::new(HttpClient::default())).clone()
}

/// Longest a provider connection may take to establish (ES
/// `xpack.inference.http.connect_timeout`, 5s), capped by the request timeout.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Largest provider response body read (ES `xpack.inference.http.max_response_size`, 50mb).
pub const MAX_RESPONSE_BYTES: usize = 50 << 20;

/// HTTP/1.1 client for `http://` and `https://` provider URLs.
#[derive(Clone)]
pub struct HttpClient {
    tls: Option<Arc<rustls::ClientConfig>>,
}

impl Default for HttpClient {
    /// Verifies servers against the default trust ([`default_tls`]).
    fn default() -> Self {
        HttpClient { tls: None }
    }
}

impl HttpClient {
    /// A client that verifies servers against `config`'s roots instead.
    pub fn with_tls(config: Arc<rustls::ClientConfig>) -> HttpClient {
        HttpClient { tls: Some(config) }
    }
}

/// The former plain-TCP transport's name.
pub type PlainHttp = HttpClient;

/// The default client TLS config: Mozilla's roots (`webpki-roots`) plus the
/// platform's trusted certificates (`rustls-native-certs`, which also reads
/// `SSL_CERT_FILE` / `SSL_CERT_DIR`); TLS 1.3 and 1.2, full verification.
pub fn default_tls() -> Arc<rustls::ClientConfig> {
    static C: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    C.get_or_init(|| {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let native = rustls_native_certs::load_native_certs();
        for e in &native.errors {
            tracing::debug!("inference http client: skipping native certificates: {e}");
        }
        let (added, _ignored) = roots.add_parsable_certificates(native.certs);
        tracing::debug!("inference http client: {added} native root certificates");
        Arc::new(
            rustls::ClientConfig::builder_with_provider(elasticsearch_ssl::provider())
                .with_safe_default_protocol_versions()
                .expect("default TLS versions")
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    })
    .clone()
}

struct Target {
    tls: bool,
    host: String,
    port: u16,
    path: String,
}

fn split_url(url: &str) -> Result<Target, String> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err(format!("unsupported url [{url}]"));
    };
    let (hostport, path) = match rest.find(['/', '?']) {
        Some(i) if rest.as_bytes()[i] == b'?' => (&rest[..i], format!("/{}", &rest[i..])),
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    let default_port = if tls { 443 } else { 80 };
    let (host, port) = if let Some(h) = hostport.strip_prefix('[') {
        // IPv6 literal: `[::1]:8443`.
        let (h, tail) = h.split_once(']').ok_or_else(|| format!("bad host in [{url}]"))?;
        let port = match tail.strip_prefix(':') {
            Some(p) => p.parse::<u16>().map_err(|_| format!("bad port in [{url}]"))?,
            None => default_port,
        };
        (h.to_string(), port)
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| format!("bad port in [{url}]"))?),
            None => (hostport.to_string(), default_port),
        }
    };
    if host.is_empty() {
        return Err(format!("no host in [{url}]"));
    }
    Ok(Target { tls, host, port, path })
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

fn io_text(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::ConnectionRefused => "Connection refused (connect failed)".to_string(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => "Read timed out".to_string(),
        _ => e.to_string(),
    }
}

/// Read a whole HTTP/1.1 response: up to `Content-Length`, the end of a
/// chunked body, or the connection's close; at most [`MAX_RESPONSE_BYTES`]
/// and until `deadline`.
fn read_response(s: &mut dyn Read, deadline: Instant) -> Result<Vec<u8>, String> {
    read_response_max(s, deadline, MAX_RESPONSE_BYTES)
}

fn read_response_max(s: &mut dyn Read, deadline: Instant, max: usize) -> Result<Vec<u8>, String> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        if Instant::now() > deadline {
            return Err("Read timed out".into());
        }
        let n = match s.read(&mut buf) {
            Ok(n) => n,
            // A server that closes without TLS close_notify: what was read is the response.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && !raw.is_empty() => 0,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_text(&e)),
        };
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if raw.len() > max {
            return Err(format!("response exceeds the maximum size of [{max}] bytes"));
        }
        if let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
            let body = &raw[split + 4..];
            let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok());
            if len.is_some_and(|len| body.len() >= len) || (head.contains("transfer-encoding: chunked") && body.ends_with(b"0\r\n\r\n")) {
                break;
            }
        }
    }
    Ok(raw)
}

impl Transport for HttpClient {
    fn post(&self, url: &str, headers: &[(String, String)], body: &Value, timeout: Duration) -> HttpResult {
        let t = split_url(url)?;
        let deadline = Instant::now() + timeout;
        let addr = format!("{}:{}", t.host, t.port);
        // Every resolved address in turn (`localhost` may be `::1` first).
        let mut last = "no address".to_string();
        let mut tcp = None;
        for sock in std::net::ToSocketAddrs::to_socket_addrs(&(t.host.as_str(), t.port)).map_err(|e| e.to_string())? {
            match TcpStream::connect_timeout(&sock, timeout.min(CONNECT_TIMEOUT)) {
                Ok(s) => {
                    tcp = Some(s);
                    break;
                }
                Err(e) => last = io_text(&e),
            }
        }
        let tcp = tcp.ok_or(last)?;
        tcp.set_read_timeout(Some(timeout)).ok();
        tcp.set_write_timeout(Some(timeout)).ok();
        let _ = tcp.set_nodelay(true);
        let mut stream: Box<dyn ReadWrite> = if t.tls {
            let config = self.tls.clone().unwrap_or_else(default_tls);
            let name = rustls::pki_types::ServerName::try_from(t.host.clone()).map_err(|e| format!("invalid TLS server name [{}]: {e}", t.host))?;
            let conn = rustls::ClientConnection::new(config, name).map_err(|e| e.to_string())?;
            Box::new(rustls::StreamOwned::new(conn, tcp))
        } else {
            Box::new(tcp)
        };
        let payload = serde_json::to_vec(body).map_err(|e| e.to_string())?;
        let default_port = if t.tls { 443 } else { 80 };
        let host = if t.port == default_port { t.host.clone() } else { addr.clone() };
        let mut req = format!("POST {} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json\r\nConnection: close\r\nContent-Length: {}\r\n", t.path, payload.len());
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        let mut msg = req.into_bytes();
        msg.extend_from_slice(&payload);
        // The TLS handshake runs on the first write.
        stream.write_all(&msg).and_then(|_| stream.flush()).map_err(|e| io_text(&e))?;
        let raw = read_response(&mut stream, deadline)?;
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

impl HttpClient {
    /// A blocking `GET` of `url` (http or https, following no redirects
    /// beyond `max_redirects`): status and body bytes, at most `max` bytes.
    pub fn get_bytes(&self, url: &str, headers: &[(String, String)], timeout: Duration, max: usize, max_redirects: usize) -> Result<(u16, Vec<u8>), String> {
        let t = split_url(url)?;
        let deadline = Instant::now() + timeout;
        let mut last = "no address".to_string();
        let mut tcp = None;
        for sock in std::net::ToSocketAddrs::to_socket_addrs(&(t.host.as_str(), t.port)).map_err(|e| e.to_string())? {
            match TcpStream::connect_timeout(&sock, timeout.min(CONNECT_TIMEOUT)) {
                Ok(s) => {
                    tcp = Some(s);
                    break;
                }
                Err(e) => last = io_text(&e),
            }
        }
        let tcp = tcp.ok_or(last)?;
        tcp.set_read_timeout(Some(timeout)).ok();
        tcp.set_write_timeout(Some(timeout)).ok();
        let mut stream: Box<dyn ReadWrite> = if t.tls {
            let config = self.tls.clone().unwrap_or_else(default_tls);
            let name = rustls::pki_types::ServerName::try_from(t.host.clone()).map_err(|e| format!("invalid TLS server name [{}]: {e}", t.host))?;
            let conn = rustls::ClientConnection::new(config, name).map_err(|e| e.to_string())?;
            Box::new(rustls::StreamOwned::new(conn, tcp))
        } else {
            Box::new(tcp)
        };
        let default_port = if t.tls { 443 } else { 80 };
        let host = if t.port == default_port { t.host.clone() } else { format!("{}:{}", t.host, t.port) };
        let mut req = format!("GET {} HTTP/1.1\r\nHost: {host}\r\nAccept: */*\r\nUser-Agent: Elasticsearch\r\nConnection: close\r\n", t.path);
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        stream.write_all(req.as_bytes()).and_then(|_| stream.flush()).map_err(|e| io_text(&e))?;
        let raw = read_response_max(&mut stream, deadline, max)?;
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("malformed http response")?;
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("malformed status line")?;
        if (300..400).contains(&status) && max_redirects > 0 {
            if let Some(loc) = head.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.trim().eq_ignore_ascii_case("location")).map(|(_, v)| v.trim().to_string())) {
                // Credentials are not sent on to the redirect target (a
                // pre-signed storage URL).
                return self.get_bytes(&loc, &[], deadline.saturating_duration_since(Instant::now()), max, max_redirects - 1);
            }
        }
        let mut body = raw[split + 4..].to_vec();
        if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            body = dechunk(&body);
        }
        Ok((status, body))
    }
}

trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

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
    /// The request's root `input_type` (`ingest`, `search`,
    /// `classification`, `clustering`, `internal_ingest`,
    /// `internal_search`); it takes precedence over
    /// `task_settings.input_type` (finding 344).
    pub input_type: Option<&'a str>,
}

/// The effective ES input type of a call: the request's, else the
/// endpoint's (merged) `task_settings.input_type`.
fn input_type<'a>(c: &'a Call) -> Option<&'a str> {
    c.input_type.or_else(|| s(&c.task_settings, "input_type"))
}

/// Cohere's spelling of an ES input type (`CohereUtils`).
fn cohere_input_type(t: &str) -> &str {
    match t {
        "ingest" | "internal_ingest" => "search_document",
        "search" | "internal_search" => "search_query",
        other => other,
    }
}

/// Voyage's spelling (`document` / `query`); other types are not sent.
fn voyage_input_type(t: &str) -> Option<&'static str> {
    match t {
        "ingest" | "internal_ingest" => Some("document"),
        "search" | "internal_search" => Some("query"),
        _ => None,
    }
}

/// Jina's embedding `task` for an ES input type.
fn jina_task(t: &str) -> &str {
    match t {
        "ingest" | "internal_ingest" => "retrieval.passage",
        "search" | "internal_search" => "retrieval.query",
        "clustering" => "separation",
        other => other,
    }
}

/// The user-configured output width of an embedding endpoint (`dimensions`
/// the user set, not one learned from the creation probe), which the
/// provider request must ask for (finding 356).
fn user_dimensions(ss: &Map<String, Value>) -> Option<&Value> {
    ss.get("dimensions").filter(|_| ss.get("dimensions_set_by_user").and_then(Value::as_bool).unwrap_or(false))
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
                        // Only a width the user asked for is requested
                        // (`text-embedding-ada-002` rejects the option); an
                        // endpoint persisted without the flag learned its
                        // width (ES reads a missing persisted flag as false).
                        if c.service == "openai" && ss.get("dimensions_set_by_user").and_then(Value::as_bool).unwrap_or(false) {
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
                    if let Some(d) = user_dimensions(ss) {
                        b["dimensions"] = d.clone();
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
                    if let Some(i) = input_type(c) {
                        body["input_type"] = json!(cohere_input_type(i));
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
                        let task = c.input_type.map(jina_task).or_else(|| s(&c.task_settings, "task")).or_else(|| s(&c.task_settings, "input_type").map(jina_task));
                        if let Some(t) = task {
                            body["task"] = json!(t);
                        }
                        if let Some(d) = user_dimensions(ss) {
                            body["dimensions"] = d.clone();
                        }
                    } else {
                        if let Some(t) = input_type(c).and_then(voyage_input_type) {
                            body["input_type"] = json!(t);
                        }
                        if let Some(d) = user_dimensions(ss) {
                            body["output_dimension"] = d.clone();
                        }
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

/// ES `EmbeddingRequestChunker`'s failure when a provider answers with a
/// different number of embeddings than it was sent inputs.
pub fn count_mismatch(got: usize, sent: usize) -> String {
    format!("Error the number of embedding responses [{got}] does not equal the number of requests [{sent}]")
}

/// Parse a provider response into the ES `_inference` result object
/// (`{"text_embedding": [...]}` etc.).
pub fn parse_response(c: &Call, resp: &Value) -> Result<Value, String> {
    let bad = |what: &str| format!("Failed to parse {} response from [{}]: {what}", c.task.name(), c.service);
    match c.task {
        Task::TextEmbedding | Task::Embedding => {
            let n = c.input.len();
            let rows: Vec<Vec<f64>> = if let Some(d) = resp.get("data").and_then(Value::as_array) {
                // Exactly one row per input: indices unique, in range and
                // covering every input (any order); a malformed response
                // never drops a chunk or attaches a vector to another.
                if d.len() != n {
                    return Err(count_mismatch(d.len(), n));
                }
                let mut slots: Vec<Option<Vec<f64>>> = vec![None; n];
                for (i, e) in d.iter().enumerate() {
                    let idx = match e.get("index") {
                        None | Some(Value::Null) => i as i64,
                        Some(v) => v.as_i64().ok_or_else(|| bad("invalid embedding index"))?,
                    };
                    let slot = usize::try_from(idx).ok().filter(|i| *i < n).ok_or_else(|| bad(&format!("embedding index [{idx}] is out of range for [{n}] inputs")))?;
                    if slots[slot].is_some() {
                        return Err(bad(&format!("embedding index [{idx}] is repeated")));
                    }
                    slots[slot] = Some(e.get("embedding").and_then(floats).ok_or_else(|| bad("missing embedding"))?);
                }
                slots.into_iter().map(|r| r.expect("every index filled: n distinct in-range indices")).collect()
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
            if rows.len() != n {
                return Err(count_mismatch(rows.len(), n));
            }
            // One embedding shape for the whole response.
            if let Some(first) = rows.first() {
                if first.is_empty() || rows.iter().any(|r| r.len() != first.len()) {
                    return Err(bad("embeddings of different or zero dimensions"));
                }
            }
            let key = if c.task == Task::Embedding { "embeddings" } else { "text_embedding" };
            Ok(json!({ key: rows.into_iter().map(|r| json!({"embedding": r})).collect::<Vec<_>>() }))
        }
        Task::SparseEmbedding => {
            let arr = resp.as_array().ok_or_else(|| bad("expected array"))?;
            if arr.len() != c.input.len() {
                return Err(count_mismatch(arr.len(), c.input.len()));
            }
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
        Call { service, task, service_settings: ss, secrets: sec, task_settings: Map::new(), input, query: Some("q"), input_type: None }
    }

    /// Findings 344 and 356: the request's root input type wins over the
    /// endpoint's and is translated to each provider's spelling; a
    /// user-configured width is requested from Voyage / Jina.
    #[test]
    fn input_type_precedence_translation_and_dimensions() {
        let sec = json!({"api_key": "k"}).as_object().unwrap().clone();
        let input = vec!["a".to_string()];
        let ss = json!({"model_id": "embed-v4"}).as_object().unwrap().clone();
        let body = |svc: &str, ss: &Map<String, Value>, saved: Option<&str>, req: Option<&'static str>| {
            let mut c = call(svc, Task::TextEmbedding, ss, &sec, &input);
            if let Some(t) = saved {
                c.task_settings.insert("input_type".into(), json!(t));
            }
            c.input_type = req;
            build_request(&c).unwrap().body
        };
        assert_eq!(body("cohere", &ss, Some("classification"), Some("search"))["input_type"], json!("search_query"));
        assert_eq!(body("cohere", &ss, Some("classification"), Some("ingest"))["input_type"], json!("search_document"));
        assert_eq!(body("cohere", &ss, Some("classification"), None)["input_type"], json!("classification"));
        assert_eq!(body("cohere", &ss, Some("ingest"), None)["input_type"], json!("search_document"));
        assert_eq!(body("cohere", &ss, None, Some("internal_search"))["input_type"], json!("search_query"));
        assert!(body("cohere", &ss, None, None).get("input_type").is_none());
        assert_eq!(body("voyageai", &ss, Some("search"), Some("ingest"))["input_type"], json!("document"));
        assert_eq!(body("voyageai", &ss, Some("ingest"), None)["input_type"], json!("document"));
        assert_eq!(body("jinaai", &ss, Some("ingest"), Some("search"))["task"], json!("retrieval.query"));
        assert_eq!(body("jinaai", &ss, Some("clustering"), None)["task"], json!("separation"));
        // Dimensions: sent only when the user configured them.
        let user = json!({"model_id": "voyage-3-large", "dimensions": 512, "dimensions_set_by_user": true}).as_object().unwrap().clone();
        let probed = json!({"model_id": "voyage-3-large", "dimensions": 1024, "dimensions_set_by_user": false}).as_object().unwrap().clone();
        assert_eq!(body("voyageai", &user, None, None)["output_dimension"], json!(512));
        assert!(body("voyageai", &probed, None, None).get("output_dimension").is_none());
        assert_eq!(body("jinaai", &user, None, None)["dimensions"], json!(512));
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
