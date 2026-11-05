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

#[derive(Default)]
struct MockS3 {
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
    fn verify_signature(&self, r: &Req) -> bool {
        let Some(auth) = r.header("authorization") else { return false };
        let signed: Vec<&str> = auth.split("SignedHeaders=").nth(1).and_then(|s| s.split(',').next()).unwrap_or("").split(';').collect();
        let sig = auth.rsplit("Signature=").next().unwrap_or("");
        if !auth.starts_with(&format!("AWS4-HMAC-SHA256 Credential={AK}/")) || !auth.contains("/us-east-1/s3/aws4_request") {
            return false;
        }
        let payload = r.header("x-amz-content-sha256").unwrap_or("");
        if payload != sigv4::sha256_hex(&r.body) {
            return false;
        }
        if r.header("x-amz-security-token") != Some(TOKEN) {
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
        let creds = Credentials { access_key: AK.into(), secret_key: SK.into(), session_token: Some(TOKEN.into()) };
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
                    let mut out = mock.handle(Req { method, raw_path, raw_query, headers, body });
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
