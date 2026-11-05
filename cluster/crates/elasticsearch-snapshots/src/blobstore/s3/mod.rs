//! Minimal S3 client (SigV4 over reqwest) and the `s3` blob store.
//!
//! Supports AWS S3 and S3-compatible stores (MinIO): virtual-hosted or
//! path-style addressing, custom endpoints, session tokens, SSE-S3, storage
//! classes, canned ACLs, conditional create (`If-None-Match: *`), multipart
//! uploads for blobs larger than `buffer_size`, and batch deletes.

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
use std::time::Duration;

/// Minimum S3 multipart part size.
pub const MIN_PART_SIZE: u64 = 5 << 20;
/// ES default `buffer_size`: 100mb (or 5% of heap, whichever is smaller).
pub const DEFAULT_BUFFER_SIZE: u64 = 100 << 20;

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
    /// `None` → environment variables at request time.
    pub credentials: Option<Credentials>,
    pub server_side_encryption: bool,
    pub storage_class: Option<String>,
    pub canned_acl: Option<String>,
    pub buffer_size: u64,
    pub max_retries: u32,
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
            server_side_encryption: false,
            storage_class: None,
            canned_acl: None,
            buffer_size: DEFAULT_BUFFER_SIZE,
            max_retries: 3,
        }
    }
}

pub struct S3BlobStore {
    cfg: S3Config,
    client: reqwest::Client,
    /// scheme://host[:port]
    origin: String,
    host: String,
    prefix: String,
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
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| SnapshotError::io("building s3 client", e))?;
        Ok(Self { cfg, client, origin, host, prefix })
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

    fn credentials(&self) -> Result<Credentials> {
        self.cfg.credentials.clone().or_else(Credentials::from_env).ok_or_else(|| {
            SnapshotError::new(
                "sdk_client_exception",
                500,
                "Unable to load AWS credentials: set access_key/secret_key or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY",
            )
        })
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
        let creds = self.credentials()?;
        let payload_hash = if body.is_empty() { EMPTY_SHA256.to_string() } else { sha256_hex(&body) };
        let qs = sigv4::canonical_query(query);
        let url = if qs.is_empty() { format!("{}{}", self.origin, path) } else { format!("{}{}?{}", self.origin, path, qs) };
        let m = reqwest::Method::from_bytes(method.as_bytes()).map_err(|e| SnapshotError::illegal_argument(e.to_string()))?;
        let mut attempt = 0;
        loop {
            attempt += 1;
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
            let rb = rb.body(body.clone());
            let result = rb.send().await;
            match result {
                Ok(r) => {
                    let status = r.status().as_u16();
                    let rheaders = r.headers().clone();
                    let collected = match &admission {
                        Some(admit) => super::admitted::response(r, admit.clone()).await,
                        None => r.bytes().await.map_err(|e| SnapshotError::io(format!("s3 {method} [{url}]"),e)),
                    };
                    let rbody = match collected {
                        Ok(b) => b,
                        Err(e) if e.kind == "circuit_breaking_exception" => return Err(e),
                        Err(e) if attempt <= self.cfg.max_retries => {
                            tracing::debug!("s3 {method} {url} body read failed, retrying: {e}");
                            tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                            continue;
                        }
                        Err(e) => return Err(SnapshotError::io(format!("s3 {method} [{url}]"), e)),
                    };
                    if (status >= 500 || status == 429) && attempt <= self.cfg.max_retries {
                        tokio::time::sleep(Duration::from_millis(200 * attempt as u64)).await;
                        continue;
                    }
                    return Ok(Resp { status, headers: rheaders, body: rbody });
                }
                Err(e) if attempt <= self.cfg.max_retries => {
                    tracing::debug!("s3 {method} {url} failed, retrying: {e}");
                    tokio::time::sleep(Duration::from_millis(200 * attempt as u64)).await;
                }
                Err(e) => return Err(SnapshotError::io(format!("s3 {method} [{url}]"), e)),
            }
        }
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
        if result.is_err() {
            let q = vec![("uploadId".to_string(), upload_id.clone())];
            let _ = self.request("DELETE", &path, &q, &[], Bytes::new()).await;
        }
        result
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
