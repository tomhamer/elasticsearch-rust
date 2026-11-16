//! Minimal blocking HTTP/1.1 client for the watcher `http` input, the
//! `webhook` action and the notification integrations.

use base64::Engine;
use serde_json::{json, Map, Value};
use elasticsearch_ssl::rustls;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct HttpRequest {
    pub method: String,
    pub scheme: String,
    pub host: String,
    pub port: u16,
    pub path: String,
    pub params: Vec<(String, String)>,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
    pub auth: Option<(String, String)>,
    pub connect_timeout_ms: u64,
    pub read_timeout_ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    pub fn body_str(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Marker key on a request template whose `body` is already rendered.
pub const BODY_RENDERED: &str = "\u{0}body_rendered";

impl HttpRequest {
    /// Parse a request definition (`url` or `host`/`port`/`scheme`/`path`,
    /// `params`, `headers`, `body`, `auth.basic`, timeouts). Strings are
    /// expected to be rendered already.
    pub fn from_json(v: &Value, render: &dyn Fn(&str) -> String) -> Result<HttpRequest, String> {
        let o = v.as_object().ok_or("request must be an object")?;
        let mut r = HttpRequest { method: "GET".into(), scheme: "http".into(), port: 80, connect_timeout_ms: 10_000, read_timeout_ms: 10_000, ..Default::default() };
        if let Some(m) = o.get("method").and_then(Value::as_str) {
            r.method = m.to_ascii_uppercase();
        }
        if let Some(u) = o.get("url").and_then(Value::as_str) {
            let u = render(u);
            let (scheme, rest) = u.split_once("://").unwrap_or(("http", &u));
            r.scheme = scheme.to_string();
            let (hostport, path) = match rest.find('/') {
                Some(i) => (&rest[..i], &rest[i..]),
                None => (rest, "/"),
            };
            let (path, query) = path.split_once('?').map(|(a, b)| (a, Some(b))).unwrap_or((path, None));
            r.path = path.to_string();
            if let Some(q) = query {
                for kv in q.split('&').filter(|s| !s.is_empty()) {
                    let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                    r.params.push((k.to_string(), v.to_string()));
                }
            }
            match hostport.rsplit_once(':') {
                Some((h, p)) if p.parse::<u16>().is_ok() => {
                    r.host = h.to_string();
                    r.port = p.parse().unwrap();
                }
                _ => {
                    r.host = hostport.to_string();
                    r.port = if scheme == "https" { 443 } else { 80 };
                }
            }
        } else {
            r.host = o.get("host").and_then(Value::as_str).map(render).ok_or("request requires [host] or [url]")?;
            if let Some(s) = o.get("scheme").and_then(Value::as_str) {
                r.scheme = s.to_string();
            }
            r.port = match o.get("port") {
                Some(Value::Number(n)) => n.as_u64().unwrap_or(80) as u16,
                Some(Value::String(s)) => render(s).parse().unwrap_or(80),
                _ => {
                    if r.scheme == "https" {
                        443
                    } else {
                        80
                    }
                }
            };
            r.path = o.get("path").and_then(Value::as_str).map(render).unwrap_or_else(|| "/".into());
            if !r.path.starts_with('/') {
                r.path.insert(0, '/');
            }
        }
        if let Some(Value::Object(p)) = o.get("params") {
            for (k, v) in p {
                let s = match v {
                    Value::String(s) => render(s),
                    other => other.to_string(),
                };
                r.params.push((k.clone(), s));
            }
        }
        if let Some(Value::Object(h)) = o.get("headers") {
            for (k, v) in h {
                let s = match v {
                    Value::String(s) => render(s),
                    other => other.to_string(),
                };
                r.headers.push((k.clone(), s));
            }
        }
        let body_rendered = o.get(BODY_RENDERED) == Some(&Value::Bool(true));
        if let Some(b) = o.get("body") {
            r.body = Some(match b {
                Value::String(s) if body_rendered => s.clone(),
                Value::String(s) => render(s),
                other => render(&other.to_string()),
            });
        }
        if let Some(b) = o.get("auth").and_then(|a| a.get("basic")) {
            let u = b.get("username").and_then(Value::as_str).unwrap_or_default().to_string();
            let p = b.get("password").and_then(Value::as_str).unwrap_or_default().to_string();
            r.auth = Some((u, p));
        }
        let t = |k: &str| o.get(k).and_then(|v| match v {
            Value::String(s) => crate::ilm::parse_time_millis(s),
            Value::Number(n) => n.as_i64(),
            _ => None,
        });
        if let Some(ms) = t("connection_timeout") {
            r.connect_timeout_ms = ms.max(1) as u64;
        }
        if let Some(ms) = t("read_timeout") {
            r.read_timeout_ms = ms.max(1) as u64;
        }
        Ok(r)
    }

    /// The request as ES records it in action results (`request` object).
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("scheme".into(), json!(self.scheme));
        m.insert("host".into(), json!(self.host));
        m.insert("port".into(), json!(self.port));
        m.insert("method".into(), json!(self.method.to_ascii_lowercase()));
        m.insert("path".into(), json!(self.path));
        if !self.params.is_empty() {
            m.insert("params".into(), Value::Object(self.params.iter().map(|(k, v)| (k.clone(), json!(v))).collect()));
        }
        self.put_headers_auth(&mut m);
        if let Some(b) = &self.body {
            m.insert("body".into(), json!(b));
        }
        Value::Object(m)
    }

    /// Simulated-request rendering (`host`, `port`, `scheme`, `method`, `path`, `body`).
    pub fn to_json_sim(&self) -> Value {
        let mut m = Map::new();
        m.insert("host".into(), json!(self.host));
        m.insert("port".into(), json!(self.port));
        m.insert("scheme".into(), json!(self.scheme));
        m.insert("method".into(), json!(self.method.to_ascii_lowercase()));
        m.insert("path".into(), json!(self.path));
        if !self.params.is_empty() {
            m.insert("params".into(), Value::Object(self.params.iter().map(|(k, v)| (k.clone(), json!(v))).collect()));
        }
        self.put_headers_auth(&mut m);
        if let Some(b) = &self.body {
            m.insert("body".into(), json!(b));
        }
        Value::Object(m)
    }

    /// `headers` (Authorization redacted) and `auth` (password redacted),
    /// as `HttpRequest.toXContent` with `hide_secrets`.
    fn put_headers_auth(&self, m: &mut Map<String, Value>) {
        if !self.headers.is_empty() {
            let auth_key = ["Authorization", "authorization"].into_iter().find(|n| self.headers.iter().any(|(k, _)| k == n));
            m.insert(
                "headers".into(),
                Value::Object(self.headers.iter().map(|(k, v)| (k.clone(), if Some(k.as_str()) == auth_key { json!(super::model::REDACTED) } else { json!(v) })).collect()),
            );
        }
        if let Some((u, _)) = &self.auth {
            m.insert("auth".into(), json!({"basic": {"username": u, "password": super::model::REDACTED}}));
        }
    }

    pub fn url(&self) -> String {
        format!("{}://{}:{}{}", self.scheme, self.host, self.port, self.path)
    }
}

fn pct(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

/// ES `xpack.http.max_response_size` default (10mb).
pub const DEFAULT_MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024;
/// The longest response head (status line and headers) accepted.
const MAX_HEAD: usize = 64 * 1024;
/// The watcher HTTP client's TLS settings (`xpack.http.ssl.*`).
pub const HTTP_SSL: &str = "xpack.http.ssl.";

/// How one request is sent: TLS for `https` and the response bound.
#[derive(Clone)]
pub struct Transport {
    /// The client config for `https` (`xpack.http.ssl.*`); `None` refuses https.
    pub tls: Option<Arc<rustls::ClientConfig>>,
    /// `xpack.http.max_response_size`.
    pub max_response_size: usize,
}

/// A blocking connection: plain TCP or TLS over it.
trait Conn: Read + Write {}
impl<T: Read + Write> Conn for T {}

/// Send `req` with the node's watcher HTTP settings: `https` connects with
/// `xpack.http.ssl.*` (full verification against the default trust unless
/// configured), responses are bounded by `xpack.http.max_response_size`.
pub fn send(node: &crate::node::Node, req: &HttpRequest) -> Result<HttpResponse, String> {
    let settings = node.node_settings.read().clone();
    let tls = if req.scheme.eq_ignore_ascii_case("https") { Some(crate::remote::client_tls(&settings, HTTP_SSL, false)?) } else { None };
    let max_response_size = super::setting(node, "xpack.http.max_response_size")
        .map(|v| crate::settings::parse_byte_size(&v).filter(|n| *n >= 0).map(|n| n as usize).ok_or_else(|| format!("failed to parse setting [xpack.http.max_response_size] with value [{v}]")))
        .transpose()?
        .unwrap_or(DEFAULT_MAX_RESPONSE_SIZE);
    send_with(req, &Transport { tls, max_response_size })
}

/// Send `req` over `transport`. The whole exchange (connect, request,
/// response) must finish within the connection timeout plus twice the read
/// timeout; every read also stops after the read timeout. The response is
/// framed strictly (Content-Length, chunked, or until the server closes):
/// a truncated body, a malformed chunk or a body over the size bound is an
/// error, never a partial success.
pub fn send_with(req: &HttpRequest, transport: &Transport) -> Result<HttpResponse, String> {
    let https = req.scheme.eq_ignore_ascii_case("https");
    if !https && !req.scheme.eq_ignore_ascii_case("http") {
        return Err(format!("unsupported scheme [{}] in [{}]", req.scheme, req.url()));
    }
    let read_timeout = Duration::from_millis(req.read_timeout_ms.max(1));
    let deadline = Instant::now() + Duration::from_millis(req.connect_timeout_ms.max(1)) + read_timeout * 2;
    let addr = (req.host.as_str(), req.port).to_socket_addrs().map_err(|e| format!("{}: {e}", req.host))?.next().ok_or_else(|| format!("{}: unknown host", req.host))?;
    let tcp = TcpStream::connect_timeout(&addr, Duration::from_millis(req.connect_timeout_ms.max(1))).map_err(|e| format!("{}: {e}", req.host))?;
    tcp.set_read_timeout(Some(read_timeout)).ok();
    tcp.set_write_timeout(Some(read_timeout)).ok();
    let control = tcp.try_clone().map_err(|e| e.to_string())?;
    let mut s: Box<dyn Conn> = if https {
        let config = transport.tls.clone().ok_or_else(|| format!("no TLS configuration for [{}]", req.url()))?;
        let name = rustls::pki_types::ServerName::try_from(req.host.clone()).map_err(|e| format!("invalid TLS server name [{}]: {e}", req.host))?;
        let conn = rustls::ClientConnection::new(config, name).map_err(|e| format!("TLS setup for [{}]: {e}", req.url()))?;
        let mut tls = rustls::StreamOwned::new(conn, tcp);
        // Handshake now: certificate and hostname failures surface as such.
        while tls.conn.is_handshaking() {
            tls.conn.complete_io(&mut tls.sock).map_err(|e| format!("TLS handshake with [{}:{}] failed: {e}", req.host, req.port))?;
        }
        Box::new(tls)
    } else {
        Box::new(tcp)
    };
    let mut target = req.path.clone();
    if !req.params.is_empty() {
        target.push('?');
        target.push_str(&req.params.iter().map(|(k, v)| format!("{}={}", pct(k), pct(v))).collect::<Vec<_>>().join("&"));
    }
    let mut head = format!("{} {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nAccept: */*\r\n", req.method, target, req.host, req.port);
    let has = |n: &str| req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(n));
    if let Some((u, p)) = &req.auth {
        if !has("authorization") {
            head.push_str(&format!("Authorization: Basic {}\r\n", base64::engine::general_purpose::STANDARD.encode(format!("{u}:{p}"))));
        }
    }
    for (k, v) in &req.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = &req.body {
        if !has("content-length") {
            head.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    if let Some(b) = &req.body {
        s.write_all(b.as_bytes()).map_err(|e| e.to_string())?;
    }
    s.flush().map_err(|e| e.to_string())?;
    let mut r = BoundedReader { inner: s, control, deadline, read_timeout, buf: Vec::new(), pos: 0, eof: false };
    read_response(&mut r, req.method.eq_ignore_ascii_case("HEAD"), transport.max_response_size)
}

/// Reads with the per-read timeout capped by the exchange deadline.
struct BoundedReader {
    inner: Box<dyn Conn>,
    control: TcpStream,
    deadline: Instant,
    read_timeout: Duration,
    buf: Vec<u8>,
    pos: usize,
    eof: bool,
}

impl BoundedReader {
    /// Read more bytes into the buffer; `Ok(false)` at end of stream.
    fn fill(&mut self) -> Result<bool, String> {
        if self.eof {
            return Ok(false);
        }
        if self.pos > 0 && self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        }
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err("timed out waiting for the response".into());
        }
        self.control.set_read_timeout(Some(left.min(self.read_timeout))).ok();
        let mut chunk = [0u8; 16 * 1024];
        loop {
            match self.inner.read(&mut chunk) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(false);
                }
                Ok(n) => {
                    self.buf.extend_from_slice(&chunk[..n]);
                    return Ok(true);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // A peer closing without TLS close_notify: the end of the
                // stream (framed bodies still detect truncation).
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    self.eof = true;
                    return Ok(false);
                }
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => return Err("timed out reading the response".into()),
                Err(e) => return Err(format!("failed to read the response: {e}")),
            }
        }
    }

    fn available(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    /// A CRLF-terminated line (without the CRLF), at most `max` bytes.
    fn line(&mut self, max: usize, what: &str) -> Result<String, String> {
        loop {
            if let Some(i) = self.available().windows(2).position(|w| w == b"\r\n") {
                let line = String::from_utf8_lossy(&self.available()[..i]).into_owned();
                self.pos += i + 2;
                return Ok(line);
            }
            if self.available().len() > max {
                return Err(format!("{what} is too long"));
            }
            if !self.fill()? {
                return Err(format!("connection closed before the end of the {what}"));
            }
        }
    }

    /// Exactly `n` bytes appended to `out`.
    fn exact(&mut self, n: usize, out: &mut Vec<u8>, what: &str) -> Result<(), String> {
        let mut left = n;
        while left > 0 {
            if self.available().is_empty() && !self.fill()? {
                return Err(format!("response body truncated: {what}"));
            }
            let take = left.min(self.available().len());
            out.extend_from_slice(&self.available()[..take]);
            self.pos += take;
            left -= take;
        }
        Ok(())
    }
}

fn too_large(max: usize) -> String {
    format!("Maximum limit of [{max}] bytes reached")
}

fn read_response(r: &mut BoundedReader, head_request: bool, max: usize) -> Result<HttpResponse, String> {
    // Status line, skipping interim (1xx) responses.
    let (status, headers) = loop {
        let status_line = r.line(MAX_HEAD, "response status line")?;
        let mut parts = status_line.splitn(3, ' ');
        let version = parts.next().unwrap_or("");
        if !version.starts_with("HTTP/1.") {
            return Err(format!("malformed HTTP response status line [{status_line}]"));
        }
        let status: u16 = parts.next().and_then(|s| s.parse().ok()).filter(|s| (100..1000).contains(s)).ok_or_else(|| format!("malformed HTTP response status line [{status_line}]"))?;
        let mut headers = Vec::new();
        let mut head_bytes = status_line.len();
        loop {
            let l = r.line(MAX_HEAD, "response headers")?;
            if l.is_empty() {
                break;
            }
            head_bytes += l.len() + 2;
            if head_bytes > MAX_HEAD {
                return Err("response headers are too long".into());
            }
            let (k, v) = l.split_once(':').ok_or_else(|| format!("malformed HTTP response header [{l}]"))?;
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
        if (100..200).contains(&status) && status != 101 {
            continue;
        }
        break (status, headers);
    };
    let header = |n: &str| headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case(n)).map(|(_, v)| v.as_str()).collect::<Vec<_>>();
    let mut body = Vec::new();
    if head_request || status == 204 || status == 304 || (100..200).contains(&status) {
        return Ok(HttpResponse { status, headers, body });
    }
    let te = header("transfer-encoding");
    let chunked = te.iter().any(|v| v.split(',').any(|t| t.trim().eq_ignore_ascii_case("chunked")));
    let lengths = header("content-length");
    if chunked {
        // RFC 9112: chunk-size [ext] CRLF data CRLF ... 0 CRLF trailers CRLF.
        loop {
            let size_line = r.line(4096, "chunk size line")?;
            let size_text = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_text, 16).ok().filter(|_| !size_text.is_empty()).ok_or_else(|| format!("malformed chunk size [{size_line}]"))?;
            if size == 0 {
                // Trailers until the empty line.
                let mut trailer_bytes = 0;
                loop {
                    let t = r.line(MAX_HEAD, "chunked trailer")?;
                    if t.is_empty() {
                        break;
                    }
                    trailer_bytes += t.len();
                    if trailer_bytes > MAX_HEAD {
                        return Err("chunked trailers are too long".into());
                    }
                }
                break;
            }
            if body.len().saturating_add(size) > max {
                return Err(too_large(max));
            }
            r.exact(size, &mut body, "incomplete chunk")?;
            let end = r.line(2, "chunk")?;
            if !end.is_empty() {
                return Err("malformed chunk: data longer than its size".into());
            }
        }
    } else if !lengths.is_empty() {
        let n: u64 = lengths[0].parse().map_err(|_| format!("malformed Content-Length [{}]", lengths[0]))?;
        if lengths.iter().any(|l| *l != lengths[0]) {
            return Err("conflicting Content-Length headers".into());
        }
        if n > max as u64 {
            return Err(too_large(max));
        }
        r.exact(n as usize, &mut body, &format!("received fewer than the [{n}] bytes of its Content-Length"))?;
    } else {
        // Delimited by the server closing the connection.
        loop {
            let avail = r.available().len();
            if body.len() + avail > max {
                return Err(too_large(max));
            }
            body.extend_from_slice(r.available());
            r.pos += avail;
            if !r.fill()? {
                break;
            }
        }
    }
    Ok(HttpResponse { status, headers, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn plain() -> Transport {
        Transport { tls: None, max_response_size: DEFAULT_MAX_RESPONSE_SIZE }
    }

    /// Read one request head (and its Content-Length body) from `c`.
    fn read_request(c: &mut dyn Read) -> String {
        let mut raw = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = match c.read(&mut buf) {
                Ok(n) => n,
                Err(_) => break,
            };
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n]);
            if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&raw[..p]).to_ascii_lowercase();
                let len: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                if raw.len() >= p + 4 + len {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }

    /// A one-shot plain HTTP server: reads the request, then runs `respond`.
    fn serve(respond: impl FnOnce(&mut TcpStream) + Send + 'static) -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            read_request(&mut c);
            respond(&mut c);
        });
        port
    }

    fn get(port: u16, read_timeout_ms: u64) -> HttpRequest {
        HttpRequest::from_json(&json!({"host": "127.0.0.1", "port": port, "path": "/", "read_timeout": format!("{read_timeout_ms}ms"), "connection_timeout": "1s"}), &|s| s.to_string()).unwrap()
    }

    #[test]
    fn strict_framing_and_bounds() {
        // A Content-Length body cut short is an error, not a partial success.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nonly ten b").unwrap();
        });
        let e = send_with(&get(port, 2000), &plain()).unwrap_err();
        assert!(e.contains("truncated"), "{e}");
        // An incomplete chunk.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab").unwrap();
        });
        let e = send_with(&get(port, 2000), &plain()).unwrap_err();
        assert!(e.contains("incomplete chunk"), "{e}");
        // Chunks without the terminating zero chunk.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nab\r\n").unwrap();
        });
        assert!(send_with(&get(port, 2000), &plain()).is_err());
        // A malformed chunk size.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nab\r\n0\r\n\r\n").unwrap();
        });
        let e = send_with(&get(port, 2000), &plain()).unwrap_err();
        assert!(e.contains("malformed chunk size"), "{e}");
        // A complete chunked body with trailers decodes.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=y\r\nde\r\n0\r\nT: 1\r\n\r\n").unwrap();
        });
        assert_eq!(send_with(&get(port, 2000), &plain()).unwrap().body, b"abcde");
        // Over the size bound: declared, and streamed until close.
        let small = Transport { tls: None, max_response_size: 1024 };
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n").unwrap();
            let _ = c.write_all(&[b'x'; 4096]);
        });
        let e = send_with(&get(port, 2000), &small).unwrap_err();
        assert!(e.contains("Maximum limit of [1024] bytes reached"), "{e}");
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\n\r\n").unwrap();
            for _ in 0..64 {
                if c.write_all(&[b'x'; 1024]).is_err() {
                    break;
                }
            }
        });
        let e = send_with(&get(port, 2000), &small).unwrap_err();
        assert!(e.contains("Maximum limit"), "{e}");
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
            for _ in 0..8 {
                let _ = c.write_all(b"400\r\n");
                let _ = c.write_all(&[b'x'; 1024]);
                let _ = c.write_all(b"\r\n");
            }
            let _ = c.write_all(b"0\r\n\r\n");
        });
        let e = send_with(&get(port, 2000), &small).unwrap_err();
        assert!(e.contains("Maximum limit"), "{e}");
    }

    #[test]
    fn timeouts_bound_the_whole_exchange() {
        // A valid status line, then silence: the read timeout fires.
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\n").unwrap();
            std::thread::sleep(Duration::from_secs(3));
        });
        let started = Instant::now();
        let e = send_with(&get(port, 300), &plain()).unwrap_err();
        assert!(e.contains("timed out"), "{e}");
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
        // A body that keeps streaming slowly (each read within the read
        // timeout) stops at the total deadline (connect + 2 x read timeout).
        let port = serve(|c| {
            c.write_all(b"HTTP/1.1 200 OK\r\n\r\n").unwrap();
            for _ in 0..200 {
                if c.write_all(b"x").is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let started = Instant::now();
        let e = send_with(&get(port, 300), &plain()).unwrap_err();
        assert!(e.contains("timed out"), "{e}");
        assert!(started.elapsed() < Duration::from_millis(4000), "{:?}", started.elapsed());
    }

    fn data(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../elasticsearch-ssl/tests/data").join(name)
    }

    /// A one-shot HTTPS server with the `elasticsearch-ssl` fixture
    /// certificate (`node.crt`: `localhost`, `127.0.0.1`, signed by `ca.crt`).
    fn serve_tls(body: &'static str) -> u16 {
        let settings = [
            ("xpack.security.http.ssl.enabled", "true".to_string()),
            ("xpack.security.http.ssl.certificate", data("node.crt").display().to_string()),
            ("xpack.security.http.ssl.key", data("node.key").display().to_string()),
        ];
        let get = |k: &str| settings.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
        let config = elasticsearch_ssl::http_server(&get, &|_| Vec::new(), &data("")).unwrap().unwrap();
        // Every loopback address (the hostname check connects to 127.0.0.2).
        let l = TcpListener::bind("0.0.0.0:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (c, _) = l.accept().unwrap();
            c.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let conn = rustls::ServerConnection::new(config).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, c);
            let req = read_request(&mut tls);
            if req.is_empty() {
                return;
            }
            let _ = tls.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes());
            tls.conn.send_close_notify();
            let _ = tls.flush();
        });
        port
    }

    #[test]
    fn https_verifies_the_server_with_the_configured_trust() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let req = |port: u16, host: &str| HttpRequest::from_json(&json!({"url": format!("https://{host}:{port}/x"), "connection_timeout": "2s", "read_timeout": "2s"}), &|s| s.to_string()).unwrap();
        // Default trust: the fixture CA is not trusted.
        let port = serve_tls("{}");
        let e = send(&node, &req(port, "localhost")).unwrap_err();
        assert!(e.contains("TLS handshake") && e.to_ascii_lowercase().contains("certificate"), "{e}");
        // `xpack.http.ssl.certificate_authorities` trusts it.
        node.node_settings.write().insert("xpack.http.ssl.certificate_authorities".into(), json!([data("ca.crt").display().to_string()]));
        let port = serve_tls("{\"ok\":true}");
        let resp = send(&node, &req(port, "localhost")).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body_str(), "{\"ok\":true}");
        // Hostname verification: the certificate does not name this host.
        let port = serve_tls("{}");
        let e = send(&node, &req(port, "127.0.0.2")).unwrap_err();
        assert!(e.contains("TLS handshake"), "{e}");
    }

    #[test]
    fn roundtrip() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            // Read the whole request (headers + Content-Length body): a single
            // read() can return the headers alone.
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = c.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&raw[..p]).to_ascii_lowercase();
                    let len: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                    if raw.len() >= p + 4 + len {
                        break;
                    }
                }
            }
            let req = String::from_utf8_lossy(&raw).into_owned();
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n{\"a\":1}\n").unwrap();
            req
        });
        let r = HttpRequest::from_json(&json!({"host": "127.0.0.1", "port": port, "path": "/x", "method": "post", "body": "hi {{n}}", "params": {"q": "a b"}}), &|s| s.replace("{{n}}", "7")).unwrap();
        let resp = send_with(&r, &plain()).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body_str().trim(), "{\"a\":1}");
        let req = h.join().unwrap();
        assert!(req.starts_with("POST /x?q=a%20b HTTP/1.1"));
        assert!(req.ends_with("hi 7"));
    }
}
