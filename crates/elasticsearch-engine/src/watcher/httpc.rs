//! Minimal blocking HTTP/1.1 client for the watcher `http` input, the
//! `webhook` action and the notification integrations.

use base64::Engine;
use serde_json::{json, Map, Value};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

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

pub fn send(req: &HttpRequest) -> Result<HttpResponse, String> {
    if req.scheme == "https" {
        return Err(format!("https is not supported by this client [{}]", req.url()));
    }
    let addr = (req.host.as_str(), req.port).to_socket_addrs().map_err(|e| format!("{}: {e}", req.host))?.next().ok_or_else(|| format!("{}: unknown host", req.host))?;
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(req.connect_timeout_ms)).map_err(|e| format!("{}: {e}", req.host))?;
    s.set_read_timeout(Some(Duration::from_millis(req.read_timeout_ms))).ok();
    s.set_write_timeout(Some(Duration::from_millis(req.read_timeout_ms))).ok();
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
    let mut raw = Vec::new();
    let read = s.read_to_end(&mut raw);
    if raw.is_empty() {
        if let Err(e) = read {
            return Err(e.to_string());
        }
    }
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> Result<HttpResponse, String> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("malformed HTTP response")?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.lines();
    let status_line = lines.next().ok_or("empty response")?;
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("malformed status line")?;
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))).collect();
    let mut body = raw[split + 4..].to_vec();
    let chunked = headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked"));
    if chunked {
        let mut out = Vec::new();
        let mut rest = &body[..];
        loop {
            let Some(p) = rest.windows(2).position(|w| w == b"\r\n") else { break };
            let size = usize::from_str_radix(String::from_utf8_lossy(&rest[..p]).split(';').next().unwrap_or("0").trim(), 16).unwrap_or(0);
            rest = &rest[p + 2..];
            if size == 0 || rest.len() < size {
                break;
            }
            out.extend_from_slice(&rest[..size]);
            rest = rest.get(size + 2..).unwrap_or(&[]);
        }
        body = out;
    }
    Ok(HttpResponse { status, headers, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

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
        let resp = send(&r).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body_str().trim(), "{\"a\":1}");
        let req = h.join().unwrap();
        assert!(req.starts_with("POST /x?q=a%20b HTTP/1.1"));
        assert!(req.ends_with("hi 7"));
    }
}
