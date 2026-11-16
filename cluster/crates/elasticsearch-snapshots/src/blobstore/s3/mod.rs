//! Minimal S3 client (SigV4 over reqwest) and the `s3` blob store.
//!
//! Supports AWS S3 and S3-compatible stores (MinIO): virtual-hosted or
//! path-style addressing, custom endpoints, session tokens, SSE-S3, storage
//! classes, canned ACLs, conditional create (`If-None-Match: *`), multipart
//! uploads for blobs larger than `buffer_size`, and batch deletes.
//!
//! Every request is bounded (ES `s3.client.*` settings): connecting is
//! limited by [`CONNECT_TIMEOUT`]; every wait for progress (the connection
//! taking more of the request body, the response headers, the next chunk of
//! the response body) by `read_timeout` (default 50s), like a socket read
//! timeout; and the whole call, retries included, by `api_call_timeout`
//! when set. Transport errors, timeouts, truncated bodies,
//! 5xx and 429 responses are retried up to `max_retries` times with
//! exponential backoff and full jitter (the AWS SDK's standard strategy).

pub mod credentials;
pub mod sigv4;
mod xml;

use super::{BlobStore, ByteStream, DeleteResult, WriteOptions, validate_key};
use crate::error::{Result, SnapshotError};
use async_trait::async_trait;
use base64::Engine;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use sigv4::{Credentials, EMPTY_SHA256, SignableRequest, sha256_hex, uri_encode_path};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Minimum S3 multipart part size.
pub const MIN_PART_SIZE: u64 = 5 << 20;
/// ES default `buffer_size`: 100mb (or 5% of heap, whichever is smaller).
pub const DEFAULT_BUFFER_SIZE: u64 = 100 << 20;
/// ES `s3.client.*.read_timeout` default.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(50);
/// TCP/TLS connection establishment limit.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// ES `s3.client.*.connection_max_idle_time` default.
pub const DEFAULT_MAX_IDLE_TIME: Duration = Duration::from_secs(60);
/// Backoff cap (AWS SDK standard retry strategy).
pub const MAX_BACKOFF: Duration = Duration::from_secs(20);

#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    /// Key prefix (no leading/trailing slash), may be empty.
    pub base_path: String,
    /// `host[:port]` or full URL; default `s3.<region>.amazonaws.com`.
    pub endpoint: Option<String>,
    pub region: String,
    /// `https` (default) or `http`, used when `endpoint` has no scheme.
    pub protocol: String,
    pub path_style_access: bool,
    /// `None` → the AWS default provider chain (environment, web identity,
    /// container, instance profile), see [`credentials`].
    pub credentials: Option<Credentials>,
    /// Where the provider chain looks; `None`: the standard AWS environment
    /// variables and endpoints ([`credentials::ChainConfig::from_env`]).
    pub credential_chain: Option<credentials::ChainConfig>,
    pub server_side_encryption: bool,
    pub storage_class: Option<String>,
    pub canned_acl: Option<String>,
    pub buffer_size: u64,
    pub max_retries: u32,
    /// Limit on any wait for progress: the connection taking more of the
    /// request body, the response headers, the next response body chunk
    /// (ES `read_timeout`).
    pub read_timeout: Duration,
    /// Limit on one whole call including its retries (ES
    /// `api_call_timeout`); `None`: bounded by the per-attempt limits only.
    pub api_call_timeout: Option<Duration>,
    /// Base of the exponential backoff between retries.
    pub retry_base_delay: Duration,
    /// Idle pooled connections are closed after this long.
    pub max_idle_time: Duration,
}

impl S3Config {
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            base_path: String::new(),
            endpoint: None,
            region: std::env::var("AWS_REGION").ok().filter(|r| !r.is_empty()).unwrap_or_else(|| "us-east-1".into()),
            protocol: "https".into(),
            path_style_access: false,
            credentials: None,
            credential_chain: None,
            server_side_encryption: false,
            storage_class: None,
            canned_acl: None,
            buffer_size: DEFAULT_BUFFER_SIZE,
            max_retries: 3,
            read_timeout: DEFAULT_READ_TIMEOUT,
            api_call_timeout: None,
            retry_base_delay: Duration::from_millis(100),
            max_idle_time: DEFAULT_MAX_IDLE_TIME,
        }
    }
}

#[derive(Clone)]
pub struct S3BlobStore {
    cfg: S3Config,
    client: reqwest::Client,
    /// scheme://host[:port]
    origin: String,
    host: String,
    prefix: String,
    chain: Arc<credentials::CredentialChain>,
}

/// When a request last made progress (its body consumed by the connection).
struct Progress {
    origin: Instant,
    last_ms: std::sync::atomic::AtomicU64,
}

impl Progress {
    fn new() -> Self {
        Self { origin: Instant::now(), last_ms: std::sync::atomic::AtomicU64::new(0) }
    }

    fn touch(&self) {
        self.last_ms.store(self.origin.elapsed().as_millis() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn idle(&self) -> Duration {
        let last = Duration::from_millis(self.last_ms.load(std::sync::atomic::Ordering::Relaxed));
        self.origin.elapsed().saturating_sub(last)
    }
}

/// A request body handed to the connection in small pieces, recording
/// progress: the connection pulls the next piece only once it has room, so
/// a peer that stops reading stops the progress.
struct ProgressBody {
    data: Bytes,
    off: usize,
    progress: Arc<Progress>,
}

impl http_body::Body for ProgressBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<std::result::Result<http_body::Frame<Bytes>, Self::Error>>> {
        const PIECE: usize = 64 << 10;
        let this = &mut *self;
        this.progress.touch();
        if this.off >= this.data.len() {
            return std::task::Poll::Ready(None);
        }
        let end = (this.off + PIECE).min(this.data.len());
        let piece = this.data.slice(this.off..end);
        this.off = end;
        std::task::Poll::Ready(Some(Ok(http_body::Frame::data(piece))))
    }

    fn is_end_stream(&self) -> bool {
        self.off >= self.data.len()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::with_exact((self.data.len() - self.off) as u64)
    }
}

/// Why an attempt produced no response.
enum Failure {
    /// Transport error, timeout or truncated body: worth another attempt.
    Retryable(String),
    /// E.g. the caller's admission refused the body.
    Fatal(SnapshotError),
}

/// `e` with its sources (reqwest's top-level message alone hides the cause).
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(s) = cur {
        out.push_str(": ");
        out.push_str(&s.to_string());
        cur = s.source();
    }
    out
}

struct Resp {
    status: u16,
    headers: reqwest::header::HeaderMap,
    body: Bytes,
}

impl S3BlobStore {
    pub fn new(cfg: S3Config) -> Result<Self> {
        let endpoint = cfg.endpoint.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| format!("s3.{}.amazonaws.com", cfg.region));
        let with_scheme = if endpoint.contains("://") { endpoint } else { format!("{}://{}", cfg.protocol, endpoint) };
        let url = reqwest::Url::parse(&with_scheme)
            .map_err(|e| SnapshotError::illegal_argument(format!("invalid s3 endpoint [{with_scheme}]: {e}")))?;
        let mut host = url.host_str().unwrap_or_default().to_string();
        if !cfg.path_style_access {
            host = format!("{}.{}", cfg.bucket, host);
        }
        if let Some(p) = url.port() {
            host = format!("{host}:{p}");
        }
        let origin = format!("{}://{}", url.scheme(), host);
        let base = cfg.base_path.trim_matches('/');
        let prefix = if base.is_empty() { String::new() } else { format!("{base}/") };
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT.min(cfg.read_timeout.max(Duration::from_millis(1))))
            // No reqwest `read_timeout`: it is armed on connection reads
            // even while a large body is still being written. The limits
            // are applied per attempt instead (`attempt`).
            .pool_idle_timeout(cfg.max_idle_time)
            .build()
            .map_err(|e| SnapshotError::io("building s3 client", e))?;
        let chain_cfg = cfg.credential_chain.clone().unwrap_or_else(|| credentials::ChainConfig::from_env(&cfg.region));
        let chain = Arc::new(credentials::CredentialChain::new(chain_cfg)?);
        Ok(Self { cfg, client, origin, host, prefix, chain })
    }

    pub fn config(&self) -> &S3Config {
        &self.cfg
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    fn object_path(&self, full_key: &str) -> String {
        if self.cfg.path_style_access {
            format!("/{}/{}", sigv4::uri_encode(&self.cfg.bucket), uri_encode_path(full_key))
        } else {
            format!("/{}", uri_encode_path(full_key))
        }
    }

    fn bucket_path(&self) -> String {
        if self.cfg.path_style_access { format!("/{}", sigv4::uri_encode(&self.cfg.bucket)) } else { "/".into() }
    }

    /// Keystore credentials, else the provider chain's current ones
    /// (refreshed before they expire).
    async fn credentials(&self) -> Result<Credentials> {
        match &self.cfg.credentials {
            Some(c) => Ok(c.clone()),
            None => self.chain.credentials().await,
        }
    }

    async fn request(
        &self,
        method: &str,
        path: &str,
        query: &[(String, String)],
        headers: &[(String, String)],
        body: Bytes,
    ) -> Result<Resp> {
        self.request_with_admission(method, path, query, headers, body, None).await
    }

    async fn request_with_admission(&self, method: &str, path: &str, query: &[(String,String)], headers: &[(String,String)], body: Bytes, admission: Option<super::ReadAdmission>) -> Result<Resp> {
        let payload_hash = if body.is_empty() { EMPTY_SHA256.to_string() } else { sha256_hex(&body) };
        let qs = sigv4::canonical_query(query);
        let url = if qs.is_empty() { format!("{}{}", self.origin, path) } else { format!("{}{}?{}", self.origin, path, qs) };
        let m = reqwest::Method::from_bytes(method.as_bytes()).map_err(|e| SnapshotError::illegal_argument(e.to_string()))?;
        let deadline = self.cfg.api_call_timeout.map(|t| Instant::now() + t);
        let ctx = || format!("s3 {method} [{url}]");
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            // Per attempt: temporary credentials may rotate during a long
            // operation (e.g. between the parts of a multipart upload).
            let creds = self.credentials().await?;
            let date = sigv4::amz_date(chrono::Utc::now());
            let req = SignableRequest { method, host: &self.host, path, query, headers, payload_sha256: &payload_hash };
            let signed = sigv4::sign(&req, &creds, &self.cfg.region, "s3", &date);
            let mut rb = self.client.request(m.clone(), &url);
            // `signed` already contains the caller's headers; sending both
            // would duplicate them (reqwest appends), and S3 would then sign
            // the comma-joined value and reject the signature.
            for (k, v) in signed.iter() {
                rb = rb.header(k.as_str(), v.as_str());
            }
            // hyper omits Content-Length for an empty body, which S3 rejects
            // (411 MissingContentLength) on PUT/POST, e.g. zero-byte files.
            if body.is_empty() && matches!(method, "PUT" | "POST") {
                rb = rb.header("content-length", "0");
            }
            let attempt_fut = self.attempt(rb, &body, admission.as_ref());
            let outcome = match deadline {
                Some(d) => match tokio::time::timeout_at(d.into(), attempt_fut).await {
                    Ok(o) => o,
                    Err(_) => {
                        return Err(SnapshotError::io(ctx(), format!("api call timed out after [{}ms] ({attempt} attempts)", self.cfg.api_call_timeout.unwrap_or_default().as_millis())));
                    }
                },
                None => attempt_fut.await,
            };
            let (err, throttled) = match outcome {
                Ok(resp) if matches!(resp.status, 400 | 403) && self.cfg.credentials.is_none() && attempt <= self.cfg.max_retries && {
                    let code = xml::tag(&String::from_utf8_lossy(&resp.body), "Code").unwrap_or_default();
                    matches!(code.as_str(), "ExpiredToken" | "InvalidToken" | "TokenRefreshRequired" | "InvalidAccessKeyId")
                } => {
                    // Rotated or expired temporary credentials: reload them.
                    self.chain.invalidate().await;
                    (format!("credentials rejected (status code {})", resp.status), false)
                }
                Ok(resp) => {
                    let throttled = resp.status == 429 || (resp.status == 503 && String::from_utf8_lossy(&resp.body).contains("SlowDown"));
                    if (resp.status >= 500 || resp.status == 429) && attempt <= self.cfg.max_retries {
                        (format!("status code {}", resp.status), throttled)
                    } else {
                        return Ok(resp);
                    }
                }
                Err(Failure::Fatal(e)) => return Err(e),
                Err(Failure::Retryable(e)) if attempt <= self.cfg.max_retries => (e, false),
                Err(Failure::Retryable(e)) => return Err(SnapshotError::io(ctx(), format!("{e} (after {attempt} attempts)"))),
            };
            let delay = self.backoff(attempt, throttled);
            if let Some(d) = deadline {
                if Instant::now() + delay >= d {
                    return Err(SnapshotError::io(ctx(), format!("api call timed out after [{}ms] ({attempt} attempts, last: {err})", self.cfg.api_call_timeout.unwrap_or_default().as_millis())));
                }
            }
            tracing::debug!("s3 {method} {url} attempt {attempt} failed, retrying in {delay:?}: {err}");
            tokio::time::sleep(delay).await;
        }
    }

    /// Exponential backoff with full jitter: uniform in `[0, min(cap, base * 2^(attempt-1))]`.
    fn backoff(&self, attempt: u32, throttled: bool) -> Duration {
        use rand::Rng;
        // Throttling responses (429, 503 SlowDown) back off longer, as the
        // AWS SDK v2 always does.
        let base = if throttled { self.cfg.retry_base_delay * 5 } else { self.cfg.retry_base_delay };
        let ceiling = base.saturating_mul(1u32 << attempt.saturating_sub(1).min(16)).min(MAX_BACKOFF);
        let ms = ceiling.as_millis() as u64;
        Duration::from_millis(if ms == 0 { 0 } else { rand::thread_rng().gen_range(0..=ms) })
    }

    /// One attempt. Bounded like a socket read timeout: the request body
    /// must keep being consumed and, once sent, the response headers and
    /// then each body chunk must arrive, each within `read_timeout`.
    async fn attempt(&self, rb: reqwest::RequestBuilder, body: &Bytes, admission: Option<&super::ReadAdmission>) -> std::result::Result<Resp, Failure> {
        let read_timeout = self.cfg.read_timeout;
        let progress = Arc::new(Progress::new());
        let rb = if body.is_empty() {
            rb.body(Bytes::new())
        } else {
            rb.header("content-length", body.len().to_string())
                .body(reqwest::Body::wrap(ProgressBody { data: body.clone(), off: 0, progress: progress.clone() }))
        };
        let watchdog = async {
            loop {
                let idle = progress.idle();
                if idle >= read_timeout {
                    return;
                }
                tokio::time::sleep(read_timeout - idle).await;
            }
        };
        let sent = tokio::select! {
            r = rb.send() => r,
            _ = watchdog => {
                return Err(Failure::Retryable(format!("Read timed out: no progress sending the request or awaiting the response for [{}ms]", read_timeout.as_millis())));
            }
        };
        let mut r = sent.map_err(|e| Failure::Retryable(error_chain(&e)))?;
        let status = r.status().as_u16();
        let headers = r.headers().clone();
        let mut admitted = admission.map(|a| super::admitted::Buffer::new(a.clone()));
        let mut plain = BytesMut::with_capacity(if admitted.is_some() { 0 } else { r.content_length().unwrap_or(0).min(64 << 20) as usize });
        loop {
            let chunk = match tokio::time::timeout(read_timeout, r.chunk()).await {
                Err(_) => return Err(Failure::Retryable(format!("Read timed out: response body stalled for [{}ms]", read_timeout.as_millis()))),
                Ok(Err(e)) => return Err(Failure::Retryable(format!("reading the response body: {}", error_chain(&e)))),
                Ok(Ok(None)) => break,
                Ok(Ok(Some(c))) => c,
            };
            match admitted.as_mut() {
                Some(b) => b.extend(&chunk).map_err(Failure::Fatal)?,
                None => plain.extend_from_slice(&chunk),
            }
        }
        let body = match admitted {
            Some(b) => b.finish(),
            None => plain.freeze(),
        };
        Ok(Resp { status, headers, body })
    }

    fn s3_error(op: &str, key: &str, r: &Resp) -> SnapshotError {
        let body = String::from_utf8_lossy(&r.body);
        let code = xml::tag(&body, "Code").unwrap_or_default();
        let msg = xml::tag(&body, "Message").unwrap_or_default();
        SnapshotError::new(
            "amazon_s3_exception",
            500,
            format!("{op} [{key}] failed: status code {} ({code}) {msg}", r.status),
        )
    }

    fn put_headers(&self, fail_if_exists: bool) -> Vec<(String, String)> {
        let mut h = Vec::new();
        if fail_if_exists {
            h.push(("if-none-match".into(), "*".into()));
        }
        if self.cfg.server_side_encryption {
            h.push(("x-amz-server-side-encryption".into(), "AES256".into()));
        }
        if let Some(sc) = &self.cfg.storage_class {
            h.push(("x-amz-storage-class".into(), sc.to_ascii_uppercase()));
        }
        if let Some(acl) = &self.cfg.canned_acl {
            h.push(("x-amz-acl".into(), acl.clone()));
        }
        h
    }

    async fn get(&self, key: &str, range: Option<(u64, u64)>) -> Result<Bytes> {
        validate_key(key)?;
        let fk = self.full_key(key);
        let mut headers = Vec::new();
        if let Some((off, len)) = range {
            if len == 0 {
                return Ok(Bytes::new());
            }
            headers.push(("range".to_string(), format!("bytes={}-{}", off, off + len - 1)));
        }
        let r = self.request("GET", &self.object_path(&fk), &[], &headers, Bytes::new()).await?;
        match r.status {
            200 | 206 => Ok(r.body),
            404 => Err(SnapshotError::no_such_blob(key)),
            416 => Ok(Bytes::new()),
            _ => Err(Self::s3_error("GetObject", key, &r)),
        }
    }

    async fn put_object(&self, key: &str, data: Bytes, fail_if_exists: bool) -> Result<()> {
        let fk = self.full_key(key);
        let headers = self.put_headers(fail_if_exists);
        let r = self.request("PUT", &self.object_path(&fk), &[], &headers, data).await?;
        match r.status {
            200..=299 => Ok(()),
            412 if fail_if_exists => Err(SnapshotError::blob_exists(key)),
            409 if fail_if_exists => Err(SnapshotError::blob_exists(key)),
            _ => Err(Self::s3_error("PutObject", key, &r)),
        }
    }

    async fn multipart_upload(&self, key: &str, mut stream: ByteStream, fail_if_exists: bool) -> Result<()> {
        let fk = self.full_key(key);
        let path = self.object_path(&fk);
        let q = vec![("uploads".to_string(), String::new())];
        let mut init_headers = self.put_headers(false);
        init_headers.retain(|(k, _)| k != "if-none-match");
        let r = self.request("POST", &path, &q, &init_headers, Bytes::new()).await?;
        if !(200..300).contains(&r.status) {
            return Err(Self::s3_error("CreateMultipartUpload", key, &r));
        }
        let upload_id = xml::tag(&String::from_utf8_lossy(&r.body), "UploadId")
            .ok_or_else(|| SnapshotError::io("CreateMultipartUpload", "missing UploadId"))?;
        let part_size = self.cfg.buffer_size.max(MIN_PART_SIZE) as usize;
        // A caller that stops waiting (cancellation drops this future) must
        // not leave the upload's parts behind: abort it in the background.
        let mut guard = AbortUploadOnDrop { store: Some(self.clone()), path: path.clone(), upload_id: upload_id.clone() };

        let result: Result<()> = async {
            let mut parts: Vec<(u32, String)> = Vec::new();
            let mut buf = BytesMut::with_capacity(part_size);
            let mut part_no = 1u32;
            let mut done = false;
            while !done {
                match stream.next().await {
                    Some(chunk) => buf.extend_from_slice(&chunk?),
                    None => done = true,
                }
                while buf.len() >= part_size || (done && !buf.is_empty()) {
                    let n = buf.len().min(part_size);
                    let part = buf.split_to(n).freeze();
                    let q = vec![
                        ("partNumber".to_string(), part_no.to_string()),
                        ("uploadId".to_string(), upload_id.clone()),
                    ];
                    let r = self.request("PUT", &path, &q, &[], part).await?;
                    if !(200..300).contains(&r.status) {
                        return Err(Self::s3_error("UploadPart", key, &r));
                    }
                    let etag = r
                        .headers
                        .get("etag")
                        .and_then(|v| v.to_str().ok())
                        .ok_or_else(|| SnapshotError::io("UploadPart", "missing ETag"))?
                        .to_string();
                    parts.push((part_no, etag));
                    part_no += 1;
                }
            }
            let mut body = String::from("<CompleteMultipartUpload>");
            for (n, etag) in &parts {
                body.push_str(&format!("<Part><PartNumber>{n}</PartNumber><ETag>{}</ETag></Part>", xml::escape(etag)));
            }
            body.push_str("</CompleteMultipartUpload>");
            let q = vec![("uploadId".to_string(), upload_id.clone())];
            let mut h = Vec::new();
            if fail_if_exists {
                h.push(("if-none-match".to_string(), "*".to_string()));
            }
            let r = self.request("POST", &path, &q, &h, Bytes::from(body)).await?;
            let text = String::from_utf8_lossy(&r.body);
            if r.status == 412 && fail_if_exists {
                return Err(SnapshotError::blob_exists(key));
            }
            if !(200..300).contains(&r.status) || text.contains("<Error>") {
                return Err(Self::s3_error("CompleteMultipartUpload", key, &r));
            }
            Ok(())
        }
        .await;
        guard.store = None;
        if result.is_err() {
            self.abort_upload(&path, &upload_id).await;
        }
        result
    }

    async fn abort_upload(&self, path: &str, upload_id: &str) {
        let q = vec![("uploadId".to_string(), upload_id.to_string())];
        if let Err(e) = self.request("DELETE", path, &q, &[], Bytes::new()).await {
            tracing::warn!("failed to abort multipart upload [{upload_id}] of [{path}]: {e}");
        }
    }

    /// ListObjectsV2 over `prefix`; returns (key → size, common prefixes).
    async fn list_objects(&self, prefix: &str, delimiter: Option<&str>) -> Result<(BTreeMap<String, u64>, Vec<String>)> {
        let mut objects = BTreeMap::new();
        let mut prefixes = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut q = vec![("list-type".to_string(), "2".to_string()), ("prefix".to_string(), prefix.to_string())];
            if let Some(d) = delimiter {
                q.push(("delimiter".into(), d.into()));
            }
            if let Some(t) = &token {
                q.push(("continuation-token".into(), t.clone()));
            }
            let r = self.request("GET", &self.bucket_path(), &q, &[], Bytes::new()).await?;
            if !(200..300).contains(&r.status) {
                return Err(Self::s3_error("ListObjectsV2", prefix, &r));
            }
            let text = String::from_utf8_lossy(&r.body).to_string();
            for c in xml::tags(&text, "Contents") {
                let key = xml::tag(c, "Key").unwrap_or_default();
                let size = xml::tag(c, "Size").and_then(|s| s.parse().ok()).unwrap_or(0);
                objects.insert(key, size);
            }
            for c in xml::tags(&text, "CommonPrefixes") {
                if let Some(p) = xml::tag(c, "Prefix") {
                    prefixes.push(p);
                }
            }
            let truncated = xml::tag(&text, "IsTruncated").map(|s| s == "true").unwrap_or(false);
            token = xml::tag(&text, "NextContinuationToken");
            if !truncated || token.is_none() {
                break;
            }
        }
        Ok((objects, prefixes))
    }

    async fn delete_full_keys(&self, keys: &[String]) -> Result<()> {
        for batch in keys.chunks(1000) {
            let mut body = String::from("<Delete><Quiet>true</Quiet>");
            for k in batch {
                body.push_str(&format!("<Object><Key>{}</Key></Object>", xml::escape(k)));
            }
            body.push_str("</Delete>");
            let md5 = {
                use md5::Digest;
                base64::engine::general_purpose::STANDARD.encode(md5::Md5::digest(body.as_bytes()))
            };
            let q = vec![("delete".to_string(), String::new())];
            let h = vec![("content-md5".to_string(), md5), ("content-type".to_string(), "application/xml".to_string())];
            let r = self.request("POST", &self.bucket_path(), &q, &h, Bytes::from(body)).await?;
            let text = String::from_utf8_lossy(&r.body).to_string();
            if !(200..300).contains(&r.status) {
                return Err(Self::s3_error("DeleteObjects", &batch[0], &r));
            }
            let errors: Vec<String> = xml::tags(&text, "Error")
                .into_iter()
                .filter(|e| xml::tag(e, "Code").as_deref() != Some("NoSuchKey"))
                .map(|e| format!("{}: {}", xml::tag(e, "Key").unwrap_or_default(), xml::tag(e, "Code").unwrap_or_default()))
                .collect();
            if !errors.is_empty() {
                return Err(SnapshotError::io("DeleteObjects", format!("failed to delete some blobs {errors:?}")));
            }
        }
        Ok(())
    }
}

/// Aborts a multipart upload whose upload future was dropped before it
/// completed or aborted the upload itself.
struct AbortUploadOnDrop {
    store: Option<S3BlobStore>,
    path: String,
    upload_id: String,
}

impl Drop for AbortUploadOnDrop {
    fn drop(&mut self) {
        let Some(store) = self.store.take() else { return };
        let (path, upload_id) = (std::mem::take(&mut self.path), std::mem::take(&mut self.upload_id));
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move { store.abort_upload(&path, &upload_id).await });
        }
    }
}

#[async_trait]
impl BlobStore for S3BlobStore {
    fn kind(&self) -> &'static str {
        "s3"
    }

    fn describe(&self) -> String {
        format!("s3://{}/{}", self.cfg.bucket, self.prefix)
    }

    async fn read(&self, key: &str) -> Result<Bytes> {
        self.get(key, None).await
    }

    async fn read_admitted(&self, key: &str, admission: super::ReadAdmission) -> Result<Bytes> {
        validate_key(key)?;
        let fk = self.full_key(key);
        let response = self.request_with_admission("GET", &self.object_path(&fk), &[], &[], Bytes::new(), Some(admission)).await?;
        match response.status {
            200 | 206 => Ok(response.body),
            404 => Err(SnapshotError::no_such_blob(key)),
            _ => Err(Self::s3_error("GetObject",key,&response)),
        }
    }

    async fn read_range(&self, key: &str, offset: u64, len: u64) -> Result<Bytes> {
        self.get(key, Some((offset, len))).await
    }

    async fn write(&self, key: &str, data: Bytes, opts: WriteOptions) -> Result<()> {
        validate_key(key)?;
        // S3 PUTs are atomic; multipart for large single-buffer writes.
        if data.len() as u64 > self.cfg.buffer_size.max(MIN_PART_SIZE) {
            let s = futures::stream::once(async move { Ok(data) }).boxed();
            return self.multipart_upload(key, s, opts.fail_if_exists).await;
        }
        self.put_object(key, data, opts.fail_if_exists).await
    }

    async fn write_stream(&self, key: &str, len: u64, mut stream: ByteStream, opts: WriteOptions) -> Result<()> {
        validate_key(key)?;
        if len > self.cfg.buffer_size.max(MIN_PART_SIZE) {
            return self.multipart_upload(key, stream, opts.fail_if_exists).await;
        }
        let mut buf = BytesMut::with_capacity(len as usize);
        while let Some(c) = stream.next().await {
            buf.extend_from_slice(&c?);
        }
        self.put_object(key, buf.freeze(), opts.fail_if_exists).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        validate_key(key)?;
        let fk = self.full_key(key);
        let r = self.request("HEAD", &self.object_path(&fk), &[], &[], Bytes::new()).await?;
        match r.status {
            200..=299 => Ok(true),
            404 => Ok(false),
            _ => Err(Self::s3_error("HeadObject", key, &r)),
        }
    }

    async fn list(&self, dir: &str, name_prefix: &str) -> Result<BTreeMap<String, u64>> {
        let p = format!("{}{}", self.prefix, dir);
        let (objs, _) = self.list_objects(&format!("{p}{name_prefix}"), Some("/")).await?;
        Ok(objs.into_iter().filter_map(|(k, s)| k.strip_prefix(&p).map(|n| (n.to_string(), s))).collect())
    }

    async fn list_children(&self, dir: &str) -> Result<Vec<String>> {
        let p = format!("{}{}", self.prefix, dir);
        let (_, prefixes) = self.list_objects(&p, Some("/")).await?;
        let mut out: Vec<String> = prefixes
            .into_iter()
            .filter_map(|cp| cp.strip_prefix(&p).map(|c| c.trim_end_matches('/').to_string()))
            .filter(|c| !c.is_empty())
            .collect();
        out.sort();
        Ok(out)
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<()> {
        let full: Vec<String> = keys.iter().map(|k| self.full_key(k)).collect();
        self.delete_full_keys(&full).await
    }

    async fn delete_dir(&self, dir: &str) -> Result<DeleteResult> {
        let p = format!("{}{}", self.prefix, dir);
        let (objs, _) = self.list_objects(&p, None).await?;
        let mut res = DeleteResult::default();
        let keys: Vec<String> = objs
            .into_iter()
            .map(|(k, s)| {
                res.blobs_deleted += 1;
                res.bytes_deleted += s;
                k
            })
            .collect();
        self.delete_full_keys(&keys).await?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addressing() {
        let mut c = S3Config::new("bkt");
        c.region = "eu-west-1".into();
        c.base_path = "/a/b/".into();
        let s = S3BlobStore::new(c.clone()).unwrap();
        assert_eq!(s.origin, "https://bkt.s3.eu-west-1.amazonaws.com");
        assert_eq!(s.object_path(&s.full_key("x y")), "/a/b/x%20y");
        c.endpoint = Some("localhost:9000".into());
        c.protocol = "http".into();
        c.path_style_access = true;
        let s = S3BlobStore::new(c).unwrap();
        assert_eq!(s.origin, "http://localhost:9000");
        assert_eq!(s.host, "localhost:9000");
        assert_eq!(s.object_path(&s.full_key("k")), "/bkt/a/b/k");
        assert_eq!(s.bucket_path(), "/bkt");
    }
}
