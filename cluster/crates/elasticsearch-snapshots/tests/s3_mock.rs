//! The S3 client against an in-process mock S3 (path-style, MinIO-like).
//! The mock re-derives every request's SigV4 signature from the raw bytes it
//! received, paginates listings, implements conditional PUT, multipart
//! uploads and batch deletes (checking Content-MD5).

mod common;

use base64::Engine;
use bytes::Bytes;
use common::*;
use elasticsearch_snapshots::blobstore::s3::sigv4::{self, Credentials, SignableRequest};
use elasticsearch_snapshots::blobstore::s3::{S3BlobStore, S3Config};
use elasticsearch_snapshots::blobstore::{BlobStore, WriteOptions};
use elasticsearch_snapshots::repository::RepositorySettings;
use elasticsearch_snapshots::{Repository, RepositoryContext, RepositoryMetadata, SnapshotsService};
use futures::StreamExt;
use md5::Digest;
use parking_lot::Mutex;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const BUCKET: &str = "test-bucket";
const AK: &str = "MOCKACCESSKEY";
const SK: &str = "mock/secret/key";
const TOKEN: &str = "session-token-123";

/// A network fault the mock injects instead of (or into) a response.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fault {
    /// Read the request, then never answer (connection stays open).
    Stall,
    /// Send the headers and half the body, then hang.
    StallMidBody,
    /// Close the connection without answering.
    Drop,
    /// Send the headers and half the body, then close.
    DropMidBody,
    /// Answer 500 InternalError.
    Error500,
    /// Answer 503 SlowDown.
    SlowDown,
    /// Stop reading the request body and never answer.
    StallReadingBody,
}

struct FaultRule {
    method: &'static str,
    /// Matched against the decoded request path.
    path_contains: String,
    fault: Fault,
    /// How many requests it applies to (`usize::MAX`: until cleared).
    remaining: usize,
}

#[derive(Default)]
struct MockS3 {
    /// Temporary credentials (access key → credentials, expiry).
    temp_keys: Mutex<HashMap<String, (Credentials, Option<Instant>)>>,
    /// Access key that signed each UploadPart.
    part_keys: Mutex<Vec<String>>,
    expired_rejections: AtomicUsize,
    faults: Mutex<Vec<FaultRule>>,
    /// Requests that hit a fault.
    faulted: AtomicUsize,
    /// Requests currently hanging in a stall.
    stalled: AtomicUsize,
    objects: Mutex<BTreeMap<String, Vec<u8>>>,
    uploads: Mutex<HashMap<String, BTreeMap<u32, Vec<u8>>>>,
    completed_multipart: AtomicUsize,
    requests: AtomicUsize,
    bad_signatures: AtomicUsize,
}

struct Req {
    method: String,
    raw_path: String,
    raw_query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn query(&self) -> Vec<(String, String)> {
        if self.raw_query.is_empty() {
            return vec![];
        }
        self.raw_query
            .split('&')
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
                let d = |s: &str| percent_encoding::percent_decode_str(s).decode_utf8_lossy().to_string();
                (d(k), d(v))
            })
            .collect()
    }

    fn q(&self, k: &str) -> Option<String> {
        self.query().into_iter().find(|(a, _)| a == k).map(|(_, v)| v)
    }
}

fn resp(status: u16, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status} X\r\ncontent-length: {}\r\n", body.len());
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    let mut b = out.into_bytes();
    b.extend_from_slice(body);
    b
}

fn xml_err(status: u16, code: &str) -> Vec<u8> {
    resp(status, &[], format!("<Error><Code>{code}</Code><Message>{code}</Message></Error>").as_bytes())
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

impl MockS3 {
    fn inject(&self, method: &'static str, path_contains: &str, fault: Fault, times: usize) {
        self.faults.lock().push(FaultRule { method, path_contains: path_contains.into(), fault, remaining: times });
    }

    fn clear_faults(&self) {
        self.faults.lock().clear();
    }

    fn take_fault(&self, method: &str, raw_path: &str) -> Option<Fault> {
        let path = percent_encoding::percent_decode_str(raw_path).decode_utf8_lossy().to_string();
        let mut rules = self.faults.lock();
        let rule = rules.iter_mut().find(|r| r.remaining > 0 && r.method == method && path.contains(&r.path_contains))?;
        rule.remaining = rule.remaining.saturating_sub(1);
        self.faulted.fetch_add(1, Ordering::SeqCst);
        Some(rule.fault)
    }

    /// Credentials of `ak`: the fixed test keys or temporary ones issued by
    /// a mock credential endpoint, with their server-side expiry.
    fn lookup(&self, ak: &str) -> Option<(Credentials, Option<Instant>)> {
        if ak == AK {
            return Some((Credentials { access_key: AK.into(), secret_key: SK.into(), session_token: Some(TOKEN.into()) }, None));
        }
        self.temp_keys.lock().get(ak).cloned()
    }

    fn verify_signature(&self, r: &Req) -> bool {
        let Some(auth) = r.header("authorization") else { return false };
        let signed: Vec<&str> = auth.split("SignedHeaders=").nth(1).and_then(|s| s.split(',').next()).unwrap_or("").split(';').collect();
        let sig = auth.rsplit("Signature=").next().unwrap_or("");
        let Some(ak) = auth.strip_prefix("AWS4-HMAC-SHA256 Credential=").and_then(|c| c.split('/').next()) else { return false };
        let Some((creds, _)) = self.lookup(ak) else { return false };
        if !auth.contains("/us-east-1/s3/aws4_request") {
            return false;
        }
        let payload = r.header("x-amz-content-sha256").unwrap_or("");
        if payload != sigv4::sha256_hex(&r.body) {
            return false;
        }
        if r.header("x-amz-security-token") != creds.session_token.as_deref() {
            return false;
        }
        let auto = ["host", "x-amz-date", "x-amz-content-sha256", "x-amz-security-token"];
        let extra: Vec<(String, String)> = signed
            .iter()
            .filter(|h| !auto.contains(h))
            .map(|h| (h.to_string(), r.header(h).unwrap_or("").to_string()))
            .collect();
        let query = r.query();
        let req = SignableRequest {
            method: &r.method,
            host: r.header("host").unwrap_or(""),
            path: &r.raw_path,
            query: &query,
            headers: &extra,
            payload_sha256: payload,
        };
        let h = sigv4::sign(&req, &creds, "us-east-1", "s3", r.header("x-amz-date").unwrap_or(""));
        let expected = h.iter().find(|(k, _)| k == "authorization").unwrap().1.rsplit("Signature=").next().unwrap().to_string();
        expected == sig
    }

    fn handle(&self, r: Req) -> Vec<u8> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        // Real S3 joins repeated headers with ',' before verifying the
        // signature, so a duplicated signed header fails there; and it
        // requires Content-Length on PUT/POST. Reject both like S3 does.
        let mut seen = std::collections::HashSet::new();
        if r.headers.iter().any(|(k, _)| !seen.insert(k.clone())) {
            self.bad_signatures.fetch_add(1, Ordering::Relaxed);
            return xml_err(403, "SignatureDoesNotMatch");
        }
        if matches!(r.method.as_str(), "PUT" | "POST") && r.header("content-length").is_none() {
            return xml_err(411, "MissingContentLength");
        }
        if !self.verify_signature(&r) {
            self.bad_signatures.fetch_add(1, Ordering::Relaxed);
            return xml_err(403, "SignatureDoesNotMatch");
        }
        let ak = r.header("authorization").and_then(|a| a.strip_prefix("AWS4-HMAC-SHA256 Credential=")).and_then(|c| c.split('/').next()).unwrap_or("").to_string();
        if let Some((_, Some(exp))) = self.lookup(&ak) {
            if Instant::now() >= exp {
                self.expired_rejections.fetch_add(1, Ordering::SeqCst);
                return xml_err(400, "ExpiredToken");
            }
        }
        if r.q("partNumber").is_some() {
            self.part_keys.lock().push(ak);
        }
        let decoded = percent_encoding::percent_decode_str(&r.raw_path).decode_utf8_lossy().to_string();
        let Some(rest) = decoded.strip_prefix(&format!("/{BUCKET}")) else { return xml_err(404, "NoSuchBucket") };
        let key = rest.trim_start_matches('/').to_string();
        let m = r.method.as_str();
        if key.is_empty() {
            if m == "GET" && r.q("list-type").as_deref() == Some("2") {
                return self.list(&r);
            }
            if m == "POST" && r.q("delete").is_some() {
                let md5 = base64::engine::general_purpose::STANDARD.encode(md5::Md5::digest(&r.body));
                if r.header("content-md5") != Some(md5.as_str()) {
                    return xml_err(400, "InvalidDigest");
                }
                let body = String::from_utf8_lossy(&r.body).to_string();
                let mut objs = self.objects.lock();
                for part in body.split("<Key>").skip(1) {
                    let k = part.split("</Key>").next().unwrap().replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&");
                    objs.remove(&k);
                }
                return resp(200, &[], b"<DeleteResult></DeleteResult>");
            }
            return xml_err(400, "BadRequest");
        }
        match m {
            "GET" => {
                let objs = self.objects.lock();
                let Some(data) = objs.get(&key) else { return xml_err(404, "NoSuchKey") };
                if let Some(range) = r.header("range") {
                    let (a, b) = range.trim_start_matches("bytes=").split_once('-').unwrap();
                    let a: usize = a.parse().unwrap();
                    let b: usize = b.parse().unwrap();
                    if a >= data.len() {
                        return xml_err(416, "InvalidRange");
                    }
                    let b = b.min(data.len() - 1);
                    return resp(206, &[], &data[a..=b]);
                }
                resp(200, &[], data)
            }
            "HEAD" => {
                if self.objects.lock().contains_key(&key) { resp(200, &[], b"") } else { resp(404, &[], b"") }
            }
            "PUT" => {
                if let (Some(n), Some(id)) = (r.q("partNumber"), r.q("uploadId")) {
                    let mut ups = self.uploads.lock();
                    let Some(u) = ups.get_mut(&id) else { return xml_err(404, "NoSuchUpload") };
                    u.insert(n.parse().unwrap(), r.body.clone());
                    return resp(200, &[("etag", format!("\"etag-{n}\""))], b"");
                }
                let mut objs = self.objects.lock();
                if r.header("if-none-match") == Some("*") && objs.contains_key(&key) {
                    return xml_err(412, "PreconditionFailed");
                }
                objs.insert(key, r.body.clone());
                resp(200, &[("etag", "\"x\"".into())], b"")
            }
            "POST" if r.q("uploads").is_some() => {
                let id = format!("upload-{}", self.requests.load(Ordering::Relaxed));
                self.uploads.lock().insert(id.clone(), BTreeMap::new());
                resp(200, &[], format!("<InitiateMultipartUploadResult><UploadId>{id}</UploadId></InitiateMultipartUploadResult>").as_bytes())
            }
            "POST" if r.q("uploadId").is_some() => {
                let id = r.q("uploadId").unwrap();
                let Some(parts) = self.uploads.lock().remove(&id) else { return xml_err(404, "NoSuchUpload") };
                let body = String::from_utf8_lossy(&r.body).to_string();
                let mut data = Vec::new();
                for p in body.split("<PartNumber>").skip(1) {
                    let n: u32 = p.split("</PartNumber>").next().unwrap().parse().unwrap();
                    data.extend_from_slice(&parts[&n]);
                }
                let mut objs = self.objects.lock();
                if r.header("if-none-match") == Some("*") && objs.contains_key(&key) {
                    return xml_err(412, "PreconditionFailed");
                }
                objs.insert(key, data);
                self.completed_multipart.fetch_add(1, Ordering::Relaxed);
                resp(200, &[], b"<CompleteMultipartUploadResult></CompleteMultipartUploadResult>")
            }
            "DELETE" if r.q("uploadId").is_some() => {
                self.uploads.lock().remove(&r.q("uploadId").unwrap());
                resp(204, &[], b"")
            }
            _ => xml_err(400, "BadRequest"),
        }
    }

    fn list(&self, r: &Req) -> Vec<u8> {
        const PAGE: usize = 3; // tiny pages to exercise continuation tokens
        let prefix = r.q("prefix").unwrap_or_default();
        let delim = r.q("delimiter");
        let after = r.q("continuation-token");
        let objs = self.objects.lock();
        // Entries: either keys or common prefixes, in key order.
        let mut entries: Vec<(String, Option<u64>)> = Vec::new();
        for (k, v) in objs.iter().filter(|(k, _)| k.starts_with(&prefix)) {
            let rest = &k[prefix.len()..];
            match (&delim, rest.find('/')) {
                (Some(d), Some(i)) if d == "/" => {
                    let cp = format!("{prefix}{}", &rest[..=i]);
                    if entries.last().map(|e| e.0 != cp).unwrap_or(true) {
                        entries.push((cp, None));
                    }
                }
                _ => entries.push((k.clone(), Some(v.len() as u64))),
            }
        }
        let start = after.map(|a| entries.iter().position(|e| e.0 > a).unwrap_or(entries.len())).unwrap_or(0);
        let page: Vec<_> = entries[start..].iter().take(PAGE).cloned().collect();
        let truncated = start + page.len() < entries.len();
        let mut x = String::from("<ListBucketResult>");
        x.push_str(&format!("<IsTruncated>{truncated}</IsTruncated>"));
        for (k, size) in &page {
            match size {
                Some(s) => x.push_str(&format!("<Contents><Key>{}</Key><Size>{s}</Size></Contents>", esc(k))),
                None => x.push_str(&format!("<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>", esc(k))),
            }
        }
        if truncated {
            x.push_str(&format!("<NextContinuationToken>{}</NextContinuationToken>", esc(&page.last().unwrap().0)));
        }
        x.push_str("</ListBucketResult>");
        resp(200, &[], x.as_bytes())
    }
}

async fn serve(mock: Arc<MockS3>) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let mock = mock.clone();
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
                loop {
                    // headers
                    let head_end = loop {
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break i;
                        }
                        let mut tmp = [0u8; 65536];
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let mut lines = head.split("\r\n");
                    let mut rl = lines.next().unwrap().split(' ');
                    let method = rl.next().unwrap().to_string();
                    let target = rl.next().unwrap().to_string();
                    let headers: Vec<(String, String)> = lines
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers.iter().find(|(k, _)| k == "content-length").map(|(_, v)| v.parse().unwrap()).unwrap_or(0);
                    let total = head_end + 4 + len;
                    let raw_path = target.split_once('?').map(|(p, _)| p).unwrap_or(&target).to_string();
                    let fault = mock.take_fault(&method, &raw_path);
                    if fault == Some(Fault::StallReadingBody) {
                        // Stop reading: the client's body upload backs up.
                        mock.stalled.fetch_add(1, Ordering::SeqCst);
                        let _hold = sock;
                        futures::future::pending::<()>().await;
                        return;
                    }
                    while buf.len() < total {
                        let mut tmp = [0u8; 65536];
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                    }
                    let body = buf[head_end + 4..total].to_vec();
                    buf.drain(..total);
                    let (raw_path, raw_query) = match target.split_once('?') {
                        Some((p, q)) => (p.to_string(), q.to_string()),
                        None => (target.clone(), String::new()),
                    };
                    let is_head = method == "HEAD";
                    match fault {
                        Some(Fault::Stall) => {
                            mock.stalled.fetch_add(1, Ordering::SeqCst);
                            let _hold = sock;
                            futures::future::pending::<()>().await;
                            return;
                        }
                        Some(Fault::Drop) => return,
                        Some(Fault::Error500) => {
                            if sock.write_all(&xml_err(500, "InternalError")).await.is_err() {
                                return;
                            }
                            continue;
                        }
                        Some(Fault::SlowDown) => {
                            if sock.write_all(&xml_err(503, "SlowDown")).await.is_err() {
                                return;
                            }
                            continue;
                        }
                        _ => {}
                    }
                    let mut out = mock.handle(Req { method, raw_path, raw_query, headers, body });
                    if let Some(f @ (Fault::StallMidBody | Fault::DropMidBody)) = fault {
                        let head_end = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                        let cut = head_end + (out.len() - head_end) / 2;
                        let _ = sock.write_all(&out[..cut]).await;
                        let _ = sock.flush().await;
                        if f == Fault::StallMidBody {
                            mock.stalled.fetch_add(1, Ordering::SeqCst);
                            let _hold = sock;
                            futures::future::pending::<()>().await;
                        }
                        return;
                    }
                    if is_head {
                        // HEAD responses carry no body.
                        let i = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                        out.truncate(i + 4);
                    }
                    if sock.write_all(&out).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    port
}

fn config(port: u16, base_path: &str) -> S3Config {
    let mut c = S3Config::new(BUCKET);
    c.endpoint = Some(format!("127.0.0.1:{port}"));
    c.protocol = "http".into();
    c.path_style_access = true;
    c.region = "us-east-1".into();
    c.base_path = base_path.into();
    c.credentials = Some(Credentials { access_key: AK.into(), secret_key: SK.into(), session_token: Some(TOKEN.into()) });
    c.buffer_size = 5 << 20;
    c
}

#[tokio::test]
async fn s3_store_against_mock() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let s = S3BlobStore::new(config(port, "base/path")).unwrap();

    s.write("dir/a b+c.txt", Bytes::from_static(b"hello"), WriteOptions::CREATE_NEW).await.unwrap();
    let e = s.write("dir/a b+c.txt", Bytes::from_static(b"x"), WriteOptions::CREATE_NEW).await.unwrap_err();
    assert!(e.is_already_exists(), "{e}");
    assert_eq!(&s.read("dir/a b+c.txt").await.unwrap()[..], b"hello");
    assert_eq!(&s.read_range("dir/a b+c.txt", 1, 3).await.unwrap()[..], b"ell");
    assert!(s.exists("dir/a b+c.txt").await.unwrap());
    assert!(!s.exists("dir/nope").await.unwrap());
    assert!(s.read("dir/nope").await.unwrap_err().is_not_found());
    assert!(mock.objects.lock().contains_key("base/path/dir/a b+c.txt"));

    for i in 0..7 {
        s.write(&format!("dir/f{i}"), Bytes::from(vec![0u8; i]), WriteOptions::OVERWRITE).await.unwrap();
        s.write(&format!("dir/sub{i}/x"), Bytes::from_static(b"1"), WriteOptions::OVERWRITE).await.unwrap();
    }
    let l = s.list("dir/", "f").await.unwrap();
    assert_eq!(l.len(), 7);
    assert_eq!(l["f3"], 3);
    assert_eq!(s.list("dir/", "").await.unwrap().len(), 8);
    assert_eq!(s.list_children("dir/").await.unwrap().len(), 7);

    // Multipart: 11mb streamed with a 5mb buffer → 3 parts.
    let big: Vec<u8> = (0..(11 << 20)).map(|i| (i % 251) as u8).collect();
    let chunks: Vec<Result<Bytes, _>> = big.chunks(1 << 20).map(|c| Ok(Bytes::copy_from_slice(c))).collect();
    s.write_stream("dir/big", big.len() as u64, futures::stream::iter(chunks).boxed(), WriteOptions::CREATE_NEW)
        .await
        .unwrap();
    assert_eq!(mock.completed_multipart.load(Ordering::Relaxed), 1);
    assert_eq!(s.read("dir/big").await.unwrap().len(), big.len());
    assert!(s.read("dir/big").await.unwrap() == big);
    let chunks: Vec<Result<Bytes, _>> = vec![Ok(Bytes::from(big.clone()))];
    let e = s
        .write_stream("dir/big", big.len() as u64, futures::stream::iter(chunks).boxed(), WriteOptions::CREATE_NEW)
        .await
        .unwrap_err();
    assert!(e.is_already_exists());
    assert!(mock.uploads.lock().is_empty(), "failed multipart must be aborted");

    s.delete_batch(&["dir/f0".into(), "dir/f1".into(), "dir/missing".into()]).await.unwrap();
    assert_eq!(s.list("dir/", "f").await.unwrap().len(), 5);
    let r = s.delete_dir("dir/").await.unwrap();
    assert_eq!(r.blobs_deleted, 1 + 5 + 7 + 1);
    assert!(mock.objects.lock().is_empty());
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn s3_full_cycle_against_mock() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let svc = SnapshotsService::new(RepositoryContext::default());
    let store = Arc::new(S3BlobStore::new(config(port, "cluster-a")).unwrap());
    let settings = RepositorySettings { chunk_size: Some(64 * 1024), ..Default::default() };
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("mock", "s3", json!({"bucket": BUCKET, "base_path": "cluster-a"})),
        store,
        settings,
    ));
    let work = tempfile::tempdir().unwrap();
    full_cycle(&svc, "mock", work.path(), 64 * 1024).await;
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);
    assert!(mock.requests.load(Ordering::Relaxed) > 50);
}

#[tokio::test]
async fn s3_repository_from_settings_and_client() {
    // Named client settings drive endpoint / path-style / protocol.
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let mut clients = BTreeMap::new();
    clients.insert(
        "minio".to_string(),
        elasticsearch_snapshots::S3ClientSettings {
            endpoint: Some(format!("http://127.0.0.1:{port}")),
            region: Some("us-east-1".into()),
            path_style_access: true,
            access_key: Some(AK.into()),
            secret_key: Some(SK.into()),
            session_token: Some(TOKEN.into()),
            ..Default::default()
        },
    );
    let svc = SnapshotsService::new(RepositoryContext { s3_clients: clients, ..Default::default() });
    svc.put_repository(
        RepositoryMetadata::new(
            "m",
            "s3",
            json!({"bucket": BUCKET, "client": "minio", "base_path": "bp", "storage_class": "standard_ia", "server_side_encryption": true}),
        ),
        true,
    )
    .await
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_file(dir.path(), "f", b"data");
    svc.create_snapshot(request(
        "m",
        "s",
        vec![elasticsearch_snapshots::IndexSnapshotInput {
            index_name: "i".into(),
            index_metadata_json: index_meta("i", 1),
            shards: vec![shard(dir.path(), 0, &["f"])],
        }],
    ))
    .await
    .unwrap();
    assert!(mock.objects.lock().keys().all(|k| k.starts_with("bp/")));
    assert!(mock.objects.lock().contains_key("bp/index.latest"));
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);
    let e = svc
        .put_repository(RepositoryMetadata::new("x", "s3", json!({"bucket": BUCKET, "client": "nope"})), false)
        .await
        .unwrap_err();
    assert_eq!(e.reason, "Unknown s3 client name [nope]. Existing client configs: minio");
}

// ----- network faults: timeouts, retries, deadlines, cancellation ----------

use std::time::{Duration, Instant};

/// Short limits so faults resolve in milliseconds.
fn fast_config(port: u16, base_path: &str, read_timeout: Duration) -> S3Config {
    let mut c = config(port, base_path);
    c.read_timeout = read_timeout;
    c.retry_base_delay = Duration::from_millis(10);
    c.max_retries = 3;
    c
}

async fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let until = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn s3_transient_faults_are_retried() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let s = S3BlobStore::new(fast_config(port, "flaky", Duration::from_millis(300))).unwrap();
    let data: Vec<u8> = (0..200_000u32).map(|i| (i % 253) as u8).collect();
    s.write("k", Bytes::from(data.clone()), WriteOptions::OVERWRITE).await.unwrap();
    for fault in [Fault::Stall, Fault::StallMidBody, Fault::Drop, Fault::DropMidBody, Fault::Error500, Fault::SlowDown] {
        mock.inject("GET", "flaky/k", fault, 1);
        let started = Instant::now();
        let got = s.read("k").await.unwrap_or_else(|e| panic!("{fault:?}: {e}"));
        assert!(got == data, "{fault:?}: wrong bytes after retry");
        assert!(started.elapsed() < Duration::from_secs(5), "{fault:?} took {:?}", started.elapsed());
        let got = s.read_range("k", 1000, 5000).await.unwrap();
        assert_eq!(&got[..], &data[1000..6000]);
    }
    // Uploads too: a dropped / stalled PUT is retried with the same body.
    for fault in [Fault::Stall, Fault::Drop, Fault::Error500] {
        mock.inject("PUT", "flaky/w", fault, 1);
        s.write("w", Bytes::from_static(b"payload"), WriteOptions::OVERWRITE).await.unwrap_or_else(|e| panic!("{fault:?}: {e}"));
        assert_eq!(&s.read("w").await.unwrap()[..], b"payload");
    }
    assert_eq!(mock.faulted.load(Ordering::SeqCst), 9);
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn s3_stalled_requests_fail_within_the_deadline() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let read_timeout = Duration::from_millis(250);
    let s = S3BlobStore::new(fast_config(port, "stall", read_timeout)).unwrap();
    s.write("k", Bytes::from(vec![7u8; 10_000]), WriteOptions::OVERWRITE).await.unwrap();

    // Every attempt stalls (before or in the middle of the body): the read
    // fails after max_retries + 1 attempts, each bounded by read_timeout.
    for fault in [Fault::Stall, Fault::StallMidBody] {
        mock.clear_faults();
        let before = mock.faulted.load(Ordering::SeqCst);
        mock.inject("GET", "stall/k", fault, usize::MAX);
        let started = Instant::now();
        let e = s.read("k").await.unwrap_err();
        let took = started.elapsed();
        assert!(e.reason.contains("Read timed out"), "{fault:?}: {e}");
        assert!(e.reason.contains("after 4 attempts"), "{fault:?}: {e}");
        assert_eq!(mock.faulted.load(Ordering::SeqCst) - before, 4, "{fault:?}");
        assert!(took >= read_timeout * 4 && took < Duration::from_secs(5), "{fault:?} took {took:?}");
    }

    // A connection that keeps being dropped also gives up.
    mock.clear_faults();
    mock.inject("GET", "stall/k", Fault::Drop, usize::MAX);
    let e = s.read("k").await.unwrap_err();
    assert_eq!(e.kind, "i_o_exception", "{e}");
    assert!(e.reason.contains("after 4 attempts"), "{e}");

    // api_call_timeout bounds the whole call, retries included, even when
    // a single attempt could wait much longer.
    mock.clear_faults();
    let mut c = fast_config(port, "stall", Duration::from_secs(30));
    c.api_call_timeout = Some(Duration::from_millis(400));
    let s2 = S3BlobStore::new(c).unwrap();
    mock.inject("GET", "stall/k", Fault::Stall, usize::MAX);
    let started = Instant::now();
    let e = s2.read("k").await.unwrap_err();
    assert!(e.reason.contains("api call timed out"), "{e}");
    assert!(started.elapsed() < Duration::from_secs(3), "{:?}", started.elapsed());
    mock.clear_faults();
    assert_eq!(s2.read("k").await.unwrap().len(), 10_000);
}

#[tokio::test]
async fn s3_stalled_multipart_upload_fails_and_is_aborted() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let s = S3BlobStore::new(fast_config(port, "mp", Duration::from_millis(250))).unwrap();
    let big: Vec<u8> = (0..(11u32 << 20)).map(|i| (i % 251) as u8).collect();
    let stream = || {
        let chunks: Vec<Result<Bytes, _>> = big.chunks(1 << 20).map(|c| Ok(Bytes::copy_from_slice(c))).collect();
        futures::stream::iter(chunks).boxed()
    };
    // Every part upload is read by the peer, which then never answers.
    mock.inject("PUT", "mp/big", Fault::Stall, usize::MAX);
    let started = Instant::now();
    let e = s.write_stream("big", big.len() as u64, stream(), WriteOptions::CREATE_NEW).await.unwrap_err();
    assert!(e.reason.contains("Read timed out"), "{e}");
    assert!(e.reason.contains("after 4 attempts"), "{e}");
    assert!(started.elapsed() < Duration::from_secs(20), "{:?}", started.elapsed());
    wait_for("the upload to be aborted", || mock.uploads.lock().is_empty()).await;
    assert!(!mock.objects.lock().contains_key("mp/big"));

    // A peer that stops reading the part body: no upload progress.
    mock.clear_faults();
    mock.inject("PUT", "mp/big", Fault::StallReadingBody, usize::MAX);
    let started = Instant::now();
    let e = s.write_stream("big", big.len() as u64, stream(), WriteOptions::CREATE_NEW).await.unwrap_err();
    assert!(e.reason.contains("Read timed out"), "{e}");
    assert!(started.elapsed() < Duration::from_secs(20), "{:?}", started.elapsed());
    wait_for("the upload to be aborted", || mock.uploads.lock().is_empty()).await;

    // One stalled part attempt is retried and the upload completes. (A
    // longer read timeout: the mock hashes each 5mb part before answering.)
    mock.clear_faults();
    let s = S3BlobStore::new(fast_config(port, "mp", Duration::from_secs(5))).unwrap();
    mock.inject("PUT", "mp/big", Fault::StallMidBody, 1);
    s.write_stream("big", big.len() as u64, stream(), WriteOptions::CREATE_NEW).await.unwrap();
    assert!(s.read("big").await.unwrap() == big);
    assert!(mock.uploads.lock().is_empty());
}

/// An S3 repository on the mock, registered with `svc`.
fn s3_service(port: u16, read_timeout: Duration) -> Arc<SnapshotsService> {
    s3_service_with(port, read_timeout, 3)
}

fn s3_service_with(port: u16, read_timeout: Duration, max_retries: u32) -> Arc<SnapshotsService> {
    let svc = SnapshotsService::new(RepositoryContext::default());
    let mut cfg = fast_config(port, "drill", read_timeout);
    cfg.max_retries = max_retries;
    let store = Arc::new(S3BlobStore::new(cfg).unwrap());
    svc.register_repository(Repository::with_store(
        RepositoryMetadata::new("mock", "s3", json!({"bucket": BUCKET, "base_path": "drill"})),
        store,
        RepositorySettings { chunk_size: Some(64 * 1024), ..Default::default() },
    ));
    svc
}

fn drill_index(dir: &std::path::Path) -> Vec<elasticsearch_snapshots::IndexSnapshotInput> {
    vec![elasticsearch_snapshots::IndexSnapshotInput {
        index_name: "i".into(),
        index_metadata_json: index_meta("i", 1),
        shards: vec![shard(dir, 0, &["seg", "segments_1"])],
    }]
}

#[tokio::test]
async fn s3_stalled_snapshot_is_aborted_by_deletion() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    // A long read timeout: only the abort can end the stalled upload quickly.
    let svc = s3_service(port, Duration::from_secs(120));
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    write_file(&data, "seg", &random_bytes(200_000));
    write_file(&data, "segments_1", b"commit");
    mock.inject("PUT", "drill/indices/", Fault::Stall, usize::MAX);
    let (_, handle) = svc.start_snapshot(request("mock", "stuck", drill_index(&data))).await.unwrap();
    wait_for("an upload to stall", || mock.stalled.load(Ordering::SeqCst) > 0).await;

    let started = Instant::now();
    tokio::time::timeout(Duration::from_secs(10), svc.delete_snapshots("mock", &["stuck".into()]))
        .await
        .expect("deleting a snapshot with a stalled upload must not hang")
        .unwrap();
    let e = tokio::time::timeout(Duration::from_secs(10), handle).await.unwrap().unwrap().unwrap_err();
    assert_eq!(e.reason, "[mock:stuck] Snapshot was aborted by deletion");
    assert!(started.elapsed() < Duration::from_secs(10));
    // Its slots are released: no running entry, no in-progress status, and
    // the shard (lock) and name are free for the next snapshot.
    assert!(svc.running_snapshots(None).is_empty());
    assert_eq!(svc.snapshot_status(None, &[], false, false).await.unwrap(), json!({"snapshots": []}));

    // The repository is healthy afterwards.
    mock.clear_faults();
    // The same name and shard again (would block on a leaked shard lock or
    // in-progress entry).
    let info = tokio::time::timeout(Duration::from_secs(20), svc.create_snapshot(request("mock", "stuck", drill_index(&data)))).await.unwrap().unwrap();
    assert_eq!(info.state, Some(elasticsearch_snapshots::SnapshotState::Success));
    let out = work.path().join("out");
    svc.restore_shard("mock", "stuck", "i", 0, &out, &elasticsearch_snapshots::CancelToken::new()).await.unwrap();
    assert_same_dir_files(&data, &["seg", "segments_1"], &out);
}

#[tokio::test]
async fn s3_snapshot_with_persistently_stalled_storage_fails() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let svc = s3_service(port, Duration::from_millis(200));
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    write_file(&data, "seg", &random_bytes(100_000));
    write_file(&data, "segments_1", b"commit");
    // Data blob uploads always stall: the shard fails within its retry
    // budget and the snapshot is not reported successful.
    mock.inject("PUT", "drill/indices/", Fault::Stall, usize::MAX);
    let started = Instant::now();
    let r = svc.create_snapshot(request("mock", "s", drill_index(&data))).await;
    assert!(started.elapsed() < Duration::from_secs(30), "{:?}", started.elapsed());
    match r {
        Err(e) => assert!(e.reason.contains("timed out"), "{e}"),
        Ok(info) => {
            assert_ne!(info.state, Some(elasticsearch_snapshots::SnapshotState::Success));
            assert!(info.failures[0].reason.contains("timed out"), "{:?}", info.failures);
        }
    }
}

#[tokio::test]
async fn s3_restore_drill_network_flaps_then_succeeds() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    // The flap below fails up to 5 consecutive attempts of one request.
    let svc = s3_service_with(port, Duration::from_millis(300), 6);
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    write_file(&data, "seg", &random_bytes(64 * 1024 * 3 + 100)); // 4 parts
    write_file(&data, "segments_1", b"commit");
    svc.create_snapshot(request("mock", "s", drill_index(&data))).await.unwrap();

    // Flapping network while restoring: dropped connections, truncated and
    // stalled bodies, 5xx. Each request recovers within its retries.
    let faults = [Fault::DropMidBody, Fault::Drop, Fault::StallMidBody, Fault::Error500, Fault::Stall];
    for f in faults {
        mock.inject("GET", "drill/indices/", f, 1);
    }
    let out = work.path().join("out");
    let res = svc.restore_shard("mock", "s", "i", 0, &out, &elasticsearch_snapshots::CancelToken::new()).await.unwrap();
    assert_eq!(res.files, 2);
    assert_same_dir_files(&data, &["seg", "segments_1"], &out);
    assert_eq!(mock.faulted.load(Ordering::SeqCst), faults.len());
}

#[tokio::test]
async fn s3_restore_drill_outage_fails_clearly_and_cancels() {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let svc = s3_service(port, Duration::from_millis(200));
    let work = tempfile::tempdir().unwrap();
    let data = work.path().join("data");
    write_file(&data, "seg", &random_bytes(100_000));
    write_file(&data, "segments_1", b"commit");
    svc.create_snapshot(request("mock", "s", drill_index(&data))).await.unwrap();

    // Storage unreachable for data blobs: the restore fails (no hang), and
    // nothing partial is left in the target directory.
    mock.inject("GET", "drill/indices/", Fault::Stall, usize::MAX);
    let out = work.path().join("out");
    let started = Instant::now();
    let e = svc.restore_shard("mock", "s", "i", 0, &out, &elasticsearch_snapshots::CancelToken::new()).await.unwrap_err();
    assert!(e.reason.contains("timed out"), "{e}");
    assert!(started.elapsed() < Duration::from_secs(20), "{:?}", started.elapsed());
    let left: Vec<_> = std::fs::read_dir(&out).map(|r| r.flatten().map(|e| e.file_name()).collect()).unwrap_or_default();
    assert!(left.is_empty(), "{left:?}");

    // A restore stalled on a long read timeout is ended by cancellation.
    let svc_slow = s3_service(port, Duration::from_secs(120));
    let cancel = elasticsearch_snapshots::CancelToken::new();
    let c2 = cancel.clone();
    let m2 = mock.clone();
    tokio::spawn(async move {
        wait_for("a restore read to stall", || m2.stalled.load(Ordering::SeqCst) > 4).await;
        c2.cancel("restore aborted: the index was deleted");
    });
    let out2 = work.path().join("out2");
    let started = Instant::now();
    let e = svc_slow.restore_shard("mock", "s", "i", 0, &out2, &cancel).await.unwrap_err();
    assert_eq!(e.kind, "task_cancelled_exception", "{e}");
    assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
    let left: Vec<_> = std::fs::read_dir(&out2).map(|r| r.flatten().map(|e| e.file_name()).collect()).unwrap_or_default();
    assert!(left.is_empty(), "{left:?}");

    // Storage back: the same restore succeeds.
    mock.clear_faults();
    svc.restore_shard("mock", "s", "i", 0, &out, &elasticsearch_snapshots::CancelToken::new()).await.unwrap();
    assert_same_dir_files(&data, &["seg", "segments_1"], &out);
}

// ----- credential provider chain: IMDSv2, container, web identity ---------

use elasticsearch_snapshots::blobstore::s3::credentials::ChainConfig;

/// Mock credential endpoints (IMDSv2, ECS container, STS) that issue
/// short-lived keys the S3 mock accepts until they expire.
struct CredServer {
    s3: Arc<MockS3>,
    ttl: Duration,
    issued: AtomicUsize,
    imds_tokens: AtomicUsize,
    last_key: Mutex<String>,
}

impl CredServer {
    fn issue(&self, source: &str) -> Credentials {
        let n = self.issued.fetch_add(1, Ordering::SeqCst) + 1;
        let c = Credentials {
            access_key: format!("TEMP{}KEY{n}", source.to_ascii_uppercase()),
            secret_key: format!("temp/secret/{n}"),
            session_token: Some(format!("temp-token-{n}")),
        };
        self.s3.temp_keys.lock().insert(c.access_key.clone(), (c.clone(), Some(Instant::now() + self.ttl)));
        *self.last_key.lock() = c.access_key.clone();
        c
    }

    /// Expire `ak` at the S3 side now (rotated out early).
    fn revoke(&self, ak: &str) {
        if let Some(e) = self.s3.temp_keys.lock().get_mut(ak) {
            e.1 = Some(Instant::now());
        }
    }

    fn expiration(&self) -> String {
        (chrono::Utc::now() + chrono::Duration::from_std(self.ttl).unwrap()).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    }

    fn handle(&self, method: &str, path: &str, headers: &[(String, String)], body: &[u8]) -> Vec<u8> {
        let h = |k: &str| headers.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        let json = |c: &Credentials| {
            resp(200, &[], json!({"Code": "Success", "AccessKeyId": c.access_key, "SecretAccessKey": c.secret_key, "Token": c.session_token, "Expiration": self.expiration()}).to_string().as_bytes())
        };
        match (method, path) {
            ("PUT", "/latest/api/token") => {
                if h("x-aws-ec2-metadata-token-ttl-seconds").is_none() {
                    return resp(400, &[], b"");
                }
                let n = self.imds_tokens.fetch_add(1, Ordering::SeqCst);
                resp(200, &[], format!("imds-session-{n}").as_bytes())
            }
            ("GET", p) if p.starts_with("/latest/meta-data/iam/security-credentials/") => {
                // IMDSv2 only: reads without a session token are refused.
                if !h("x-aws-ec2-metadata-token").is_some_and(|t| t.starts_with("imds-session-")) {
                    return resp(401, &[], b"");
                }
                let role = &p["/latest/meta-data/iam/security-credentials/".len()..];
                if role.is_empty() { resp(200, &[], b"test-role\n") } else if role == "test-role" { json(&self.issue("imds")) } else { resp(404, &[], b"") }
            }
            ("GET", "/ecs/creds") => {
                if h("authorization") != Some("ecs-auth-token") {
                    return resp(403, &[], b"");
                }
                json(&self.issue("ecs"))
            }
            ("POST", "/") => {
                let form = String::from_utf8_lossy(body).to_string();
                if !form.contains("Action=AssumeRoleWithWebIdentity") || !form.contains("WebIdentityToken=jwt.web.identity") || !form.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123%3Arole%2Fes") {
                    return resp(400, &[], b"<ErrorResponse><Error><Code>InvalidParameter</Code><Message>bad request</Message></Error></ErrorResponse>");
                }
                let c = self.issue("sts");
                let x = format!(
                    "<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>{}</AccessKeyId><SecretAccessKey>{}</SecretAccessKey><SessionToken>{}</SessionToken><Expiration>{}</Expiration></Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>",
                    c.access_key, esc(&c.secret_key), c.session_token.unwrap(), self.expiration()
                );
                resp(200, &[], x.as_bytes())
            }
            _ => resp(404, &[], b""),
        }
    }
}

/// One request per connection.
async fn serve_creds(srv: Arc<CredServer>) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let srv = srv.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let head_end = loop {
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i;
                    }
                    let mut tmp = [0u8; 8192];
                    match sock.read(&mut tmp).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    }
                };
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                let mut lines = head.split("\r\n");
                let mut rl = lines.next().unwrap().split(' ');
                let (method, path) = (rl.next().unwrap().to_string(), rl.next().unwrap().to_string());
                let headers: Vec<(String, String)> =
                    lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string())).collect();
                let len: usize = headers.iter().find(|(k, _)| k == "content-length").map(|(_, v)| v.parse().unwrap()).unwrap_or(0);
                while buf.len() < head_end + 4 + len {
                    let mut tmp = [0u8; 8192];
                    match sock.read(&mut tmp).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&tmp[..n]),
                    }
                }
                let mut out = srv.handle(&method, &path, &headers, &buf[head_end + 4..head_end + 4 + len]);
                // Close after answering.
                let i = out.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
                out.splice(i + 2..i + 2, b"connection: close\r\n".iter().copied());
                let _ = sock.write_all(&out).await;
            });
        }
    });
    port
}

async fn cred_setup(ttl: Duration) -> (Arc<MockS3>, u16, Arc<CredServer>, u16) {
    let mock = Arc::new(MockS3::default());
    let port = serve(mock.clone()).await;
    let srv = Arc::new(CredServer { s3: mock.clone(), ttl, issued: AtomicUsize::new(0), imds_tokens: AtomicUsize::new(0), last_key: Mutex::new(String::new()) });
    let cport = serve_creds(srv.clone()).await;
    (mock, port, srv, cport)
}

fn chain_config(port: u16, base: &str, chain: ChainConfig) -> S3Config {
    let mut c = fast_config(port, base, Duration::from_secs(10));
    c.credentials = None;
    c.credential_chain = Some(chain);
    c
}

#[tokio::test]
async fn s3_instance_profile_credentials_rotate_during_multipart_upload() {
    let (mock, port, srv, cport) = cred_setup(Duration::from_secs(3)).await;
    let mut chain = ChainConfig::empty("us-east-1");
    chain.imds_endpoint = Some(format!("http://127.0.0.1:{cport}"));
    chain.refresh_before = Duration::from_secs(1);
    let s = S3BlobStore::new(chain_config(port, "rot", chain)).unwrap();

    // 30mb in 5mb parts, produced slowly: the upload outlives several
    // generations of 3s credentials.
    let big: Vec<u8> = (0..(30u32 << 20)).map(|i| (i % 249) as u8).collect();
    let chunks: Vec<Result<Bytes, elasticsearch_snapshots::SnapshotError>> = big.chunks(1 << 20).map(|c| Ok(Bytes::copy_from_slice(c))).collect();
    let slow = futures::stream::iter(chunks)
        .then(|c| async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            c
        })
        .boxed();
    let started = Instant::now();
    s.write_stream("big", big.len() as u64, slow, WriteOptions::CREATE_NEW).await.unwrap();
    assert!(started.elapsed() > Duration::from_secs(6), "{:?}", started.elapsed());
    let parts = mock.part_keys.lock().clone();
    let distinct: std::collections::BTreeSet<&String> = parts.iter().collect();
    assert_eq!(parts.len(), 6);
    assert!(distinct.len() >= 2, "credentials never rotated: {parts:?}");
    assert!(srv.issued.load(Ordering::SeqCst) >= 2);
    assert!(srv.imds_tokens.load(Ordering::SeqCst) >= 2, "IMDSv2 session token per refresh");
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);
    assert_eq!(mock.expired_rejections.load(Ordering::SeqCst), 0, "refreshed before expiry");
    assert!(s.read("big").await.unwrap() == big);
    assert!(mock.uploads.lock().is_empty());
}

#[tokio::test]
async fn s3_rejected_expired_token_reloads_credentials() {
    let (mock, port, srv, cport) = cred_setup(Duration::from_secs(3600)).await;
    let mut chain = ChainConfig::empty("us-east-1");
    chain.imds_endpoint = Some(format!("http://127.0.0.1:{cport}"));
    let s = S3BlobStore::new(chain_config(port, "exp", chain)).unwrap();
    s.write("a", Bytes::from_static(b"1"), WriteOptions::OVERWRITE).await.unwrap();
    assert_eq!(srv.issued.load(Ordering::SeqCst), 1);
    // Cached credentials are reused while valid.
    s.write("b", Bytes::from_static(b"2"), WriteOptions::OVERWRITE).await.unwrap();
    assert_eq!(srv.issued.load(Ordering::SeqCst), 1);
    // Rotated out early on the service side: S3 answers ExpiredToken, the
    // client reloads and the same call succeeds.
    let key = srv.last_key.lock().clone();
    srv.revoke(&key);
    s.write("c", Bytes::from_static(b"3"), WriteOptions::OVERWRITE).await.unwrap();
    assert_eq!(mock.expired_rejections.load(Ordering::SeqCst), 1);
    assert_eq!(srv.issued.load(Ordering::SeqCst), 2);
    assert_eq!(&s.read("c").await.unwrap()[..], b"3");
}

#[tokio::test]
async fn s3_container_and_web_identity_credentials() {
    let (mock, port, srv, cport) = cred_setup(Duration::from_secs(3600)).await;
    let tmp = tempfile::tempdir().unwrap();

    // ECS task role (full URI + authorization token from a file).
    let token_file = tmp.path().join("ecs-token");
    std::fs::write(&token_file, "ecs-auth-token\n").unwrap();
    let mut chain = ChainConfig::empty("us-east-1");
    chain.container_uri = Some(format!("http://127.0.0.1:{cport}/ecs/creds"));
    chain.container_token_file = Some(token_file.to_string_lossy().to_string());
    let s = S3BlobStore::new(chain_config(port, "ecs", chain)).unwrap();
    s.write("x", Bytes::from_static(b"ecs"), WriteOptions::OVERWRITE).await.unwrap();
    assert!(srv.last_key.lock().starts_with("TEMPECS"));

    // Web identity (IRSA): token file + role ARN exchanged at STS.
    let jwt = tmp.path().join("token");
    std::fs::write(&jwt, "jwt.web.identity").unwrap();
    let mut chain = ChainConfig::empty("us-east-1");
    chain.web_identity_token_file = Some(jwt.to_string_lossy().to_string());
    chain.role_arn = Some("arn:aws:iam::123:role/es".into());
    chain.sts_endpoint = Some(format!("http://127.0.0.1:{cport}"));
    // Lower priority providers are not consulted when web identity works.
    chain.imds_endpoint = Some(format!("http://127.0.0.1:{cport}"));
    let s = S3BlobStore::new(chain_config(port, "irsa", chain)).unwrap();
    s.write("y", Bytes::from_static(b"sts"), WriteOptions::OVERWRITE).await.unwrap();
    assert!(srv.last_key.lock().starts_with("TEMPSTS"));
    assert_eq!(srv.imds_tokens.load(Ordering::SeqCst), 0);
    assert_eq!(mock.bad_signatures.load(Ordering::Relaxed), 0);

    // A wrong container token falls through to the next provider (none here).
    let mut chain = ChainConfig::empty("us-east-1");
    chain.container_uri = Some(format!("http://127.0.0.1:{cport}/ecs/creds"));
    chain.container_token = Some("wrong".into());
    let s = S3BlobStore::new(chain_config(port, "bad", chain)).unwrap();
    let e = s.write("z", Bytes::from_static(b"!"), WriteOptions::OVERWRITE).await.unwrap_err();
    assert_eq!(e.kind, "sdk_client_exception", "{e}");
    assert!(e.reason.contains("Unable to load AWS credentials") && e.reason.contains("container"), "{e}");
}
