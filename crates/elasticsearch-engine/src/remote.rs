//! Minimal blocking HTTP/1.1 client for `_reindex` from a remote cluster
//! (`source.remote`): `http://` or `https://`, one request per connection,
//! optional basic auth and extra headers, connect / socket timeouts.
//!
//! `https://` hosts are always spoken to over TLS (rustls), never in
//! plaintext, configured by ES's `reindex.ssl.*` node settings
//! (`certificate_authorities`, `truststore.*`, `verification_mode`, default
//! `full`: chain and host name; `certificate`, `key`, `key_passphrase`,
//! `keystore.*` for a client certificate; `supported_protocols`). Without
//! `certificate_authorities` / `truststore.path` the server must chain to
//! the default roots (Mozilla's, as ES uses the JDK's). See
//! [`RemoteSpec::with_node_tls`] and `elasticsearch_ssl::reindex_client`.
//! `reindex.remote.whitelist` is checked by the caller, before any
//! connection.

use crate::error::{EsError, Result};
use base64::Engine as _;
use elasticsearch_ssl::rustls;
use serde_json::{Map, Value};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct RemoteSpec {
    pub scheme: String,
    pub host: String,
    pub port: u16,
    /// Path prefix without trailing slash (`""` or `"/foo"`).
    pub path_prefix: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub headers: Vec<(String, String)>,
    pub socket_timeout: Duration,
    pub connect_timeout: Duration,
    /// The `reindex.ssl.*` client config for an `https://` host
    /// ([`RemoteSpec::with_node_tls`]); `None` uses the defaults (default
    /// roots, full verification). Unused for `http://`.
    pub tls: Option<Arc<rustls::ClientConfig>>,
}

/// `reindex.ssl.*` from the node settings: the client config remote
/// reindexing over `https://` connects with. Invalid settings are ES's
/// `SslConfigException`s (the cluster node already refuses to start with
/// them).
pub fn reindex_tls(settings: &Map<String, Value>) -> Result<Arc<rustls::ClientConfig>> {
    let context = reindex_context(settings).map_err(|e| EsError::new(500, "ssl_config_exception", e))?;
    // ES `ReindexSslConfig` reloads the context when its files change (the
    // resource watcher, `resource.reload.enabled`). The cluster node also
    // polls this context on `resource.reload.interval.high`; checking here
    // too covers the engine-only server. A failed reload keeps the
    // previous context.
    if settings.get("resource.reload.enabled").and_then(scalar).as_deref() != Some("false") {
        elasticsearch_ssl::Reload::check(&*context);
    }
    Ok(context.load())
}

fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::Null | Value::Array(_) | Value::Object(_) => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// The node's `reindex.ssl.*` context (built once per distinct settings and
/// config directory, then reloaded when its files change). Errors are the
/// SSL configuration errors' messages.
pub fn reindex_context(settings: &Map<String, Value>) -> std::result::Result<Arc<elasticsearch_ssl::Reloadable<rustls::ClientConfig>>, String> {
    use std::collections::HashMap;
    use std::sync::LazyLock;
    static CONTEXTS: LazyLock<parking_lot::Mutex<HashMap<String, Arc<elasticsearch_ssl::Reloadable<rustls::ClientConfig>>>>> = LazyLock::new(Default::default);
    let get = |k: &str| settings.get(k).and_then(scalar);
    let get_list = |k: &str| -> Vec<String> {
        match settings.get(k) {
            Some(Value::Array(a)) => a.iter().filter_map(scalar).collect(),
            Some(Value::String(s)) => {
                let t = s.trim();
                let t = t.strip_prefix('[').and_then(|t| t.strip_suffix(']')).unwrap_or(t);
                t.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
            }
            _ => Vec::new(),
        }
    };
    let conf = settings
        .get("path.conf")
        .and_then(Value::as_str)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("ES_PATH_CONF").unwrap_or_else(|_| "config".into())));
    let mut key = conf.display().to_string();
    let mut relevant: Vec<(&String, &Value)> = settings.iter().filter(|(k, _)| k.starts_with(elasticsearch_ssl::REINDEX_SSL)).collect();
    relevant.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in relevant {
        key.push('\n');
        key.push_str(k);
        key.push('=');
        key.push_str(&v.to_string());
    }
    let mut contexts = CONTEXTS.lock();
    if let Some(c) = contexts.get(&key) {
        return Ok(c.clone());
    }
    let context = Arc::new(elasticsearch_ssl::reindex_client_reloadable(&get, &get_list, &conf)?);
    contexts.insert(key, context.clone());
    Ok(context)
}

/// A blocking connection: plain TCP or TLS over it.
trait Conn: Read + Write {}
impl<T: Read + Write> Conn for T {}

fn handshake_error(e: std::io::Error) -> EsError {
    match e.kind() {
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => EsError::new(500, "socket_timeout_exception", format!("TLS handshake timed out: {e}")),
        _ => EsError::new(500, "s_s_l_handshake_exception", e.to_string()),
    }
}

impl RemoteSpec {
    /// `scheme://host:port`.
    pub fn base(&self) -> String {
        format!("{}://{}:{}", self.scheme, self.host, self.port)
    }

    /// Parse `source.remote`.
    pub fn parse(o: &serde_json::Map<String, Value>) -> Result<RemoteSpec> {
        let mut host = None;
        let (mut username, mut password) = (None, None);
        let mut headers = Vec::new();
        let (mut socket, mut connect) = (Duration::from_secs(30), Duration::from_secs(30));
        for (k, v) in o {
            let t = |v: &Value| v.as_str().map(str::to_string);
            match k.as_str() {
                "host" => host = t(v),
                "username" => username = t(v),
                "password" => password = t(v),
                "headers" => {
                    if let Some(m) = v.as_object() {
                        for (hk, hv) in m {
                            headers.push((hk.clone(), hv.as_str().unwrap_or("").to_string()));
                        }
                    }
                }
                "socket_timeout" | "connect_timeout" => {
                    let s = v.as_str().unwrap_or("");
                    let ms = crate::date::parse_time_value(s).ok_or_else(|| {
                        EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{s}] as a time value: unit is missing or unrecognized"))
                    })?;
                    let d = Duration::from_millis(ms.max(0) as u64);
                    if k == "socket_timeout" {
                        socket = d
                    } else {
                        connect = d
                    }
                }
                _ => {}
            }
        }
        let unsupported: Vec<&str> = o.keys().map(String::as_str).filter(|k| !matches!(*k, "host" | "username" | "password" | "headers" | "socket_timeout" | "connect_timeout")).collect();
        if !unsupported.is_empty() {
            return Err(EsError::illegal_argument(format!("Unsupported fields in [remote]: [{}]", unsupported.join(","))));
        }
        let host = host.ok_or_else(|| EsError::illegal_argument("[host] must be specified to reindex from a remote cluster"))?;
        if username.is_some() != password.is_some() {
            return Err(EsError::illegal_argument(if username.is_some() { "[username] was specified but [password] was not" } else { "[password] was specified but [username] was not" }));
        }
        let bad = |cause: &str| {
            EsError::illegal_argument(format!("[host] must be of the form [scheme]://[host]:[port](/[pathPrefix])? but was [{host}]"))
                .caused_by(EsError::new(400, "u_r_i_syntax_exception", cause))
        };
        let Some((scheme, rest)) = host.split_once("://") else {
            return Err(bad(&format!("The port was not defined in the [host]: {host}")));
        };
        let (authority, prefix) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].trim_end_matches('/').to_string()),
            None => (rest, String::new()),
        };
        let Some((h, p)) = authority.rsplit_once(':') else {
            return Err(bad(&format!("The port was not defined in the [host]: {host}")));
        };
        let port: u16 = p.parse().map_err(|_| bad(&format!("The port was not defined in the [host]: {host}")))?;
        Ok(RemoteSpec { scheme: scheme.to_string(), host: h.to_string(), port, path_prefix: prefix, username, password, headers, socket_timeout: socket, connect_timeout: connect, tls: None })
    }

    /// Whether this host is spoken to over TLS.
    pub fn is_https(&self) -> bool {
        self.scheme.eq_ignore_ascii_case("https")
    }

    /// This spec with the node's `reindex.ssl.*` client config attached
    /// (for `https://` hosts; `http://` ones are returned unchanged).
    pub fn with_node_tls(&self, node_settings: &Map<String, Value>) -> Result<RemoteSpec> {
        let mut spec = self.clone();
        if spec.is_https() {
            spec.tls = Some(reindex_tls(node_settings)?);
        }
        Ok(spec)
    }

    /// Connect, completing the TLS handshake for `https://` (a failed
    /// handshake is the request's error; there is no plaintext fallback).
    fn connect(&self, s: TcpStream) -> Result<Box<dyn Conn>> {
        if !self.is_https() {
            return Ok(Box::new(s));
        }
        let config = match &self.tls {
            Some(c) => c.clone(),
            None => reindex_tls(&Map::new())?,
        };
        let host = self.host.trim_start_matches('[').trim_end_matches(']').to_string();
        let name = rustls::pki_types::ServerName::try_from(host).map_err(|e| EsError::new(500, "s_s_l_handshake_exception", format!("invalid TLS server name [{}]: {e}", self.host)))?;
        let mut conn = rustls::ClientConnection::new(config, name).map_err(|e| EsError::new(500, "s_s_l_handshake_exception", e.to_string()))?;
        let mut s = s;
        while conn.is_handshaking() {
            let (read, written) = conn.complete_io(&mut s).map_err(handshake_error)?;
            if read == 0 && written == 0 && conn.is_handshaking() {
                return Err(EsError::new(500, "s_s_l_handshake_exception", "Remote host terminated the handshake"));
            }
        }
        Ok(Box::new(rustls::StreamOwned::new(conn, s)))
    }

    /// `reindex.remote.whitelist` check (`host:port` patterns, comma separated, `*` wildcards).
    pub fn check_whitelist(&self, whitelist: &[String]) -> Result<()> {
        let hp = format!("{}:{}", self.host, self.port);
        if whitelist.iter().any(|w| crate::mapping::wildcard_match(w, &hp)) {
            return Ok(());
        }
        Err(EsError::illegal_argument(format!("[{hp}] not whitelisted in reindex.remote.whitelist")))
    }

    /// One request; returns `(status, body)`.
    pub fn request(&self, method: &str, path_and_query: &str, body: Option<&str>) -> Result<(u16, String)> {
        let addrs: Vec<std::net::SocketAddr> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| EsError::new(500, "unknown_host_exception", format!("{}: {e}", self.host)))?
            .collect();
        if addrs.is_empty() {
            return Err(EsError::new(500, "unknown_host_exception", self.host.clone()));
        }
        // Every resolved address in turn, as ES's HTTP client does (a name
        // such as `localhost` can resolve to `::1` first while the remote
        // listens on IPv4 only); the last failure is reported.
        let mut last = None;
        let mut connected = None;
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, self.connect_timeout) {
                Ok(s) => {
                    connected = Some(s);
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        let s = match connected {
            Some(s) => s,
            None => {
                let e = last.expect("at least one address was tried");
                let reason = match e.kind() {
                    std::io::ErrorKind::ConnectionRefused => "Connection refused (connect failed)".to_string(),
                    std::io::ErrorKind::TimedOut => "Connect timed out".to_string(),
                    _ => e.to_string(),
                };
                return Err(EsError::new(500, "connect_exception", reason));
            }
        };
        s.set_read_timeout(Some(self.socket_timeout)).ok();
        s.set_write_timeout(Some(self.socket_timeout)).ok();
        let mut s = self.connect(s)?;
        let mut head = format!("{method} {}{path_and_query} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nAccept: application/json\r\n", self.path_prefix, self.host, self.port);
        if let (Some(u), Some(p)) = (&self.username, &self.password) {
            head.push_str(&format!("Authorization: Basic {}\r\n", base64::engine::general_purpose::STANDARD.encode(format!("{u}:{p}"))));
        }
        for (k, v) in &self.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        match body {
            Some(b) => head.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len())),
            None => head.push_str("\r\n"),
        }
        s.write_all(head.as_bytes()).and_then(|_| s.flush()).map_err(|e| EsError::new(500, "socket_exception", e.to_string()))?;
        let mut raw = Vec::new();
        match s.read_to_end(&mut raw) {
            Ok(_) => {}
            // A TLS peer that closes without `close_notify` after a whole
            // response (`Connection: close`) ended the exchange normally.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && raw.windows(4).any(|w| w == b"\r\n\r\n") => {}
            Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => {
                return Err(EsError::new(500, "socket_timeout_exception", "1,000 milliseconds timeout on connection http-outgoing-0 [ACTIVE]"));
            }
            Err(e) => return Err(EsError::new(500, "socket_exception", e.to_string())),
        }
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| EsError::new(500, "i_o_exception", "malformed HTTP response"))?;
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(500);
        let chunked = head.lines().any(|l| l.to_ascii_lowercase().starts_with("transfer-encoding:") && l.to_ascii_lowercase().contains("chunked"));
        let payload = &raw[split + 4..];
        let body = if chunked { dechunk(payload) } else { payload.to_vec() };
        Ok((status, String::from_utf8_lossy(&body).into_owned()))
    }

    /// Request that must succeed; non-2xx becomes ES's `status_exception` wrapper.
    pub fn call(&self, method: &str, path_and_query: &str, body: Option<&str>) -> Result<String> {
        let (status, text) = self.request(method, path_and_query, body)?;
        if (200..300).contains(&status) {
            return Ok(text);
        }
        let phrase = match status {
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            409 => "Conflict",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "",
        };
        let uri = format!("{}{path_and_query}", if path_and_query == "/" { self.path_prefix.clone() } else { self.path_prefix.clone() });
        let uri = if path_and_query == "/" { format!("{}/", self.path_prefix) } else { uri };
        let cause = EsError::new(status, "response_exception", format!("method [{method}], host [{}], URI [{uri}], status line [HTTP/1.1 {status} {phrase}]\n{text}", self.base()));
        Err(EsError::new(status, "status_exception", format!("body={text}")).caused_by(cause))
    }
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(p) = b.windows(2).position(|w| w == b"\r\n") else { break };
        let n = usize::from_str_radix(String::from_utf8_lossy(&b[..p]).split(';').next().unwrap_or("0").trim(), 16).unwrap_or(0);
        b = &b[p + 2..];
        if n == 0 || b.len() < n {
            break;
        }
        out.extend_from_slice(&b[..n]);
        b = &b[n..];
        if b.len() >= 2 {
            b = &b[2..];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_host() {
        let r = RemoteSpec::parse(json!({"host": "http://localhost:9257/foo/"}).as_object().unwrap()).unwrap();
        assert_eq!((r.host.as_str(), r.port, r.path_prefix.as_str()), ("localhost", 9257, "/foo"));
        let e = RemoteSpec::parse(json!({"host": "localhost:9257"}).as_object().unwrap()).unwrap_err();
        assert!(e.reason.starts_with("[host] must be of the form"));
        assert!(r.check_whitelist(&["localhost:*".into()]).is_ok());
        assert!(r.check_whitelist(&["other:*".into()]).is_err());
    }

    #[test]
    fn https_never_falls_back_to_plaintext() {
        // A plaintext HTTP server: an `https://` request must fail its
        // handshake rather than read this answer.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
        });
        let spec = RemoteSpec::parse(json!({"host": format!("https://127.0.0.1:{port}")}).as_object().unwrap()).unwrap();
        assert!(spec.is_https());
        let spec = spec.with_node_tls(&Map::new()).unwrap();
        assert!(spec.tls.is_some());
        let e = spec.request("GET", "/", None).unwrap_err();
        assert_eq!(e.kind, "s_s_l_handshake_exception", "{e:?}");
        server.join().unwrap();
        // `http://` hosts carry no TLS config.
        let plain = RemoteSpec::parse(json!({"host": "http://127.0.0.1:9200"}).as_object().unwrap()).unwrap().with_node_tls(&Map::new()).unwrap();
        assert!(plain.tls.is_none());
    }

    #[test]
    fn invalid_reindex_ssl_settings_are_reported() {
        let settings = json!({"reindex.ssl.verification_mode": "sometimes"}).as_object().unwrap().clone();
        let spec = RemoteSpec::parse(json!({"host": "https://localhost:9200"}).as_object().unwrap()).unwrap();
        let e = spec.with_node_tls(&settings).unwrap_err();
        assert_eq!(e.kind, "ssl_config_exception");
        assert!(e.reason.starts_with("could not resolve ssl client verification mode, unknown value [sometimes]"), "{}", e.reason);
        let settings = json!({"reindex.ssl.certificate_authorities": ["/nonexistent/ca.crt"]}).as_object().unwrap().clone();
        let e = spec.with_node_tls(&settings).unwrap_err();
        assert!(e.reason.ends_with("/nonexistent/ca.crt] because the file does not exist"), "{}", e.reason);
    }

    #[test]
    fn dechunk_basic() {
        assert_eq!(dechunk(b"5\r\nhello\r\n0\r\n\r\n"), b"hello");
    }
}
