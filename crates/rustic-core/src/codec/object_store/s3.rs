//! S3-compatible [`ObjectStore`] (feature `s3`). AWS Signature V4 is
//! implemented here (SHA-256 / HMAC-SHA256 included) so the feature adds
//! no crate dependencies; HTTP goes through [`HttpTransport`].
//! [`PlainHttp`] speaks HTTP/1.1 over a TCP socket (`http://` endpoints:
//! MinIO, in-VPC gateways, tests). For `https://` plug a transport backed
//! by your HTTP client.

use super::ObjectStore;
use crate::error::{Result, RusticError};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

// ---------------------------------------------------------------- crypto

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (FIPS 180-4).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for block in msg.chunks_exact(64) {
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// HMAC-SHA256 (RFC 2104).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&sha256(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Vec::with_capacity(64 + data.len());
    inner.extend(k.iter().map(|b| b ^ 0x36));
    inner.extend_from_slice(data);
    let ih = sha256(&inner);
    let mut outer = Vec::with_capacity(96);
    outer.extend(k.iter().map(|b| b ^ 0x5c));
    outer.extend_from_slice(&ih);
    sha256(&outer)
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// RFC 3986 percent-encoding as SigV4 wants it (`/` kept when `keep_slash`).
pub fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') || (keep_slash && c == '/') {
            out.push(c);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// SigV4 signature of a request. `headers` are (lowercase name, value), all
/// of them signed; returns (signed header list, hex signature).
#[allow(clippy::too_many_arguments)]
pub fn sign_v4(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    headers: &[(String, String)],
    payload_sha256_hex: &str,
    secret_key: &str,
    region: &str,
    service: &str,
    amz_date: &str,
) -> (String, String) {
    let mut hs: Vec<(String, String)> = headers.iter().map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string())).collect();
    hs.sort();
    let canonical_headers: String = hs.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed: Vec<&str> = hs.iter().map(|(k, _)| k.as_str()).collect();
    let signed = signed.join(";");
    let creq = format!("{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed}\n{payload_sha256_hex}");
    let date = &amz_date[..8];
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let sts = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", hex(&sha256(creq.as_bytes())));
    let k_date = hmac_sha256(format!("AWS4{secret_key}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    (signed, hex(&hmac_sha256(&k_signing, sts.as_bytes())))
}

// ---------------------------------------------------------------- transport

#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub method: String,
    /// Absolute URL (`scheme://host[:port]/path?query`).
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Sends one HTTP request (blocking).
pub trait HttpTransport: Send + Sync + fmt::Debug {
    fn send(&self, req: &HttpRequest) -> Result<HttpResponse>;
}

/// HTTP/1.1 over plain TCP (`http://` URLs only), one connection per request.
#[derive(Debug, Default)]
pub struct PlainHttp {
    pub timeout: Option<std::time::Duration>,
}

fn io_err(m: impl Into<String>) -> RusticError {
    RusticError::Io(std::io::Error::other(m.into()))
}

impl HttpTransport for PlainHttp {
    fn send(&self, req: &HttpRequest) -> Result<HttpResponse> {
        use std::io::{Read, Write};
        let rest = req.url.strip_prefix("http://").ok_or_else(|| io_err(format!("PlainHttp supports http:// only: {}", req.url)))?;
        let (hostport, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let addr = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
        let mut s = std::net::TcpStream::connect(&addr)?;
        s.set_read_timeout(self.timeout)?;
        s.set_write_timeout(self.timeout)?;
        let mut head = format!("{} {path} HTTP/1.1\r\nConnection: close\r\nContent-Length: {}\r\n", req.method, req.body.len());
        for (k, v) in &req.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str("\r\n");
        s.write_all(head.as_bytes())?;
        s.write_all(&req.body)?;
        let mut raw = Vec::new();
        s.read_to_end(&mut raw)?;
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| io_err("malformed HTTP response"))?;
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let mut lines = head.split("\r\n");
        let status: u16 = lines.next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse().ok()).ok_or_else(|| io_err("bad status line"))?;
        let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))).collect();
        let mut body = raw[split + 4..].to_vec();
        if headers.iter().any(|(k, v)| k == "transfer-encoding" && v.eq_ignore_ascii_case("chunked")) {
            body = dechunk(&body)?;
        } else if let Some(n) = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse::<usize>().ok()) {
            body.truncate(n);
        }
        Ok(HttpResponse { status, headers, body })
    }
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = b.windows(2).position(|w| w == b"\r\n").ok_or_else(|| io_err("bad chunk"))?;
        let size_s = String::from_utf8_lossy(&b[..eol]);
        let size = usize::from_str_radix(size_s.split(';').next().unwrap_or("").trim(), 16).map_err(|_| io_err("bad chunk size"))?;
        b = &b[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size + 2 {
            return Err(io_err("truncated chunk"));
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size + 2..];
    }
}

// ---------------------------------------------------------------- store

#[derive(Clone)]
pub struct S3Config {
    /// `http(s)://host[:port]` (no trailing slash).
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    /// `endpoint/bucket/key` (MinIO, most gateways) instead of `bucket.host/key`.
    pub path_style: bool,
}

impl fmt::Debug for S3Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S3Config({} bucket={} region={})", self.endpoint, self.bucket, self.region)
    }
}

#[derive(Debug)]
pub struct S3ObjectStore {
    cfg: S3Config,
    http: Arc<dyn HttpTransport>,
}

fn amz_now() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

impl S3ObjectStore {
    pub fn new(cfg: S3Config, http: Arc<dyn HttpTransport>) -> Self {
        Self { cfg, http }
    }

    fn base(&self) -> (String, String) {
        // (url prefix up to the key, host header)
        let (scheme, host) = self.cfg.endpoint.split_once("://").unwrap_or(("https", &self.cfg.endpoint));
        let host = host.trim_end_matches('/');
        if self.cfg.path_style {
            (format!("{scheme}://{host}/{}", self.cfg.bucket), host.to_string())
        } else {
            let h = format!("{}.{host}", self.cfg.bucket);
            (format!("{scheme}://{h}"), h)
        }
    }

    /// Build, sign and send one request. `key` is the object key (may be
    /// empty for bucket-level calls); `query` pairs are unencoded.
    fn call(&self, method: &str, key: &str, query: &[(&str, &str)], extra: &[(&str, String)], body: Vec<u8>) -> Result<HttpResponse> {
        let (prefix, host) = self.base();
        let path_prefix = prefix.split_once("://").map(|(_, r)| r.find('/').map(|i| r[i..].to_string()).unwrap_or_default()).unwrap_or_default();
        let key_path = format!("/{}", uri_encode(key, true));
        let canonical_uri = format!("{path_prefix}{key_path}");
        let mut q: Vec<(String, String)> = query.iter().map(|(k, v)| (uri_encode(k, false), uri_encode(v, false))).collect();
        q.sort();
        let canonical_query = q.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
        let payload = hex(&sha256(&body));
        let date = amz_now();
        let mut headers: Vec<(String, String)> = vec![("host".into(), host), ("x-amz-content-sha256".into(), payload.clone()), ("x-amz-date".into(), date.clone())];
        if let Some(t) = &self.cfg.session_token {
            headers.push(("x-amz-security-token".into(), t.clone()));
        }
        for (k, v) in extra {
            headers.push((k.to_ascii_lowercase(), v.clone()));
        }
        let (signed, sig) = sign_v4(method, &canonical_uri, &canonical_query, &headers, &payload, &self.cfg.secret_access_key, &self.cfg.region, "s3", &date);
        let scope = format!("{}/{}/s3/aws4_request", &date[..8], self.cfg.region);
        headers.push(("authorization".into(), format!("AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={sig}", self.cfg.access_key_id)));
        let base_host = prefix.split_once("://").map(|(s, r)| format!("{s}://{}", r.split('/').next().unwrap_or(r))).unwrap_or(prefix.clone());
        let url = if canonical_query.is_empty() { format!("{base_host}{canonical_uri}") } else { format!("{base_host}{canonical_uri}?{canonical_query}") };
        self.http.send(&HttpRequest { method: method.into(), url, headers, body })
    }

    fn fail(key: &str, r: &HttpResponse) -> RusticError {
        if r.status == 404 {
            return RusticError::IndexNotFound(format!("object {key}"));
        }
        io_err(format!("s3 {key}: HTTP {} {}", r.status, String::from_utf8_lossy(&r.body).chars().take(300).collect::<String>()))
    }
}

/// Text of every `<tag>...</tag>` in `xml` (S3 list responses; values are
/// entity-decoded for the five XML entities).
fn xml_values(xml: &str, tag: &str) -> Vec<String> {
    let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        rest = &rest[i + open.len()..];
        let Some(j) = rest.find(&close) else { break };
        out.push(rest[..j].replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&"));
        rest = &rest[j + close.len()..];
    }
    out
}

impl ObjectStore for S3ObjectStore {
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let r = self.call("GET", key, &[], &[], Vec::new())?;
        if r.status == 200 { Ok(r.body) } else { Err(Self::fail(key, &r)) }
    }

    fn get_range(&self, key: &str, range: Range<u64>) -> Result<Vec<u8>> {
        if range.end <= range.start {
            return Ok(Vec::new());
        }
        let r = self.call("GET", key, &[], &[("range", format!("bytes={}-{}", range.start, range.end - 1))], Vec::new())?;
        match r.status {
            200 | 206 => Ok(r.body),
            416 => Ok(Vec::new()),
            _ => Err(Self::fail(key, &r)),
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let r = self.call("PUT", key, &[], &[], data.to_vec())?;
        if r.status == 200 { Ok(()) } else { Err(Self::fail(key, &r)) }
    }

    fn delete(&self, key: &str) -> Result<()> {
        let r = self.call("DELETE", key, &[], &[], Vec::new())?;
        if matches!(r.status, 200 | 204 | 404) { Ok(()) } else { Err(Self::fail(key, &r)) }
    }

    fn list_admitted(&self, prefix:&str)->Result<crate::codec::directory::AdmittedNames> {
        let mut out=crate::codec::directory::AdmittedNames::new()?;
        let mut tokens=crate::codec::directory::AdmittedNames::new()?;
        loop {
            let query=[("list-type","2"),("prefix",prefix),("continuation-token",tokens.last().map_or("",|t|t.as_str()))];
            let response=self.call("GET","",if !tokens.is_empty(){&query[..]}else{&query[..2]},&[],Vec::new())?;
            if response.status!=200{return Err(Self::fail(prefix,&response));}
            // Transport owns its response. Reserve XML decoding scratch before
            // lossy conversion/unescaping and preserve copied token separately.
            let _scratch=crate::admission::shared().reserve((response.body.len() as u64).saturating_mul(16).saturating_add(4096),"S3 listing XML decode")?;
            let xml=String::from_utf8_lossy(&response.body);
            for key in xml_values(&xml,"Key"){out.push(&key)?;}
            let truncated=xml_values(&xml,"IsTruncated").first().is_some_and(|v|v=="true");
            let next=xml_values(&xml,"NextContinuationToken").into_iter().next();
            if !truncated { break; }
            let value=next.filter(|value|!value.is_empty()).ok_or_else(||RusticError::Corrupted("truncated S3 listing without continuation token".into()))?;
            if tokens.iter().any(|previous|previous == &value) {
                return Err(RusticError::Corrupted("S3 listing repeated continuation token".into()));
            }
            tokens.push(&value)?;
        }out.sort();Ok(out)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut q: Vec<(&str, &str)> = vec![("list-type", "2"), ("prefix", prefix)];
            if let Some(t) = &token {
                q.push(("continuation-token", t));
            }
            let r = self.call("GET", "", &q, &[], Vec::new())?;
            if r.status != 200 {
                return Err(Self::fail(prefix, &r));
            }
            let xml = String::from_utf8_lossy(&r.body).to_string();
            out.extend(xml_values(&xml, "Key"));
            let truncated = xml_values(&xml, "IsTruncated").first().map(|v| v == "true").unwrap_or(false);
            token = xml_values(&xml, "NextContinuationToken").into_iter().next();
            if !truncated || token.is_none() {
                break;
            }
        }
        out.sort();
        Ok(out)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let r = self.call("HEAD", key, &[], &[], Vec::new())?;
        match r.status {
            200 => Ok(true),
            404 => Ok(false),
            _ => Err(Self::fail(key, &r)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::collections::BTreeMap;

    #[test]
    fn sha256_and_hmac_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // RFC 4231 test case 2.
        assert_eq!(hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }

    #[test]
    fn sigv4_get_vanilla() {
        // get-vanilla inputs of the AWS SigV4 test suite; expected value
        // cross-checked with an independent Python hashlib/hmac computation.
        let headers = vec![("host".to_string(), "example.amazon.com".to_string()), ("x-amz-date".to_string(), "20150830T123600Z".to_string())];
        let (signed, sig) = sign_v4("GET", "/", "", &headers, &hex(&sha256(b"")), "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "us-east-1", "service", "20150830T123600Z");
        assert_eq!(signed, "host;x-amz-date");
        assert_eq!(sig, "7ab4567ae243ee168f6bf18206b2b40b61ce08277323168138fa113ed23c538e");
    }

    /// An in-memory fake S3 endpoint behind the transport trait.
    #[derive(Debug, Default)]
    struct FakeS3 {
        objects: Mutex<BTreeMap<String, Vec<u8>>>,
    }

    impl HttpTransport for FakeS3 {
        fn send(&self, req: &HttpRequest) -> Result<HttpResponse> {
            assert!(req.headers.iter().any(|(k, v)| k == "authorization" && v.starts_with("AWS4-HMAC-SHA256 Credential=AK/")));
            let rest = req.url.strip_prefix("http://minio:9000/bkt").unwrap();
            let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
            let key = path.trim_start_matches('/').replace("%2F", "/");
            let mut objs = self.objects.lock();
            let ok = |body: Vec<u8>| Ok(HttpResponse { status: 200, headers: vec![], body });
            let missing = || Ok(HttpResponse { status: 404, headers: vec![], body: vec![] });
            match req.method.as_str() {
                "PUT" => {
                    objs.insert(key, req.body.clone());
                    ok(vec![])
                }
                "DELETE" => {
                    objs.remove(&key);
                    Ok(HttpResponse { status: 204, headers: vec![], body: vec![] })
                }
                "HEAD" => if objs.contains_key(&key) { ok(vec![]) } else { missing() },
                "GET" if key.is_empty() => {
                    let prefix = query.split('&').find_map(|kv| kv.strip_prefix("prefix=")).unwrap_or("").replace("%2F", "/");
                    let keys: String = objs.keys().filter(|k| k.starts_with(&prefix)).map(|k| format!("<Contents><Key>{k}</Key></Contents>")).collect();
                    ok(format!("<ListBucketResult><IsTruncated>false</IsTruncated>{keys}</ListBucketResult>").into_bytes())
                }
                "GET" => match objs.get(&key) {
                    Some(v) => match req.headers.iter().find(|(k, _)| k == "range") {
                        Some((_, r)) => {
                            let (a, b) = r.trim_start_matches("bytes=").split_once('-').unwrap();
                            let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
                            Ok(HttpResponse { status: 206, headers: vec![], body: v[a..=b.min(v.len() - 1)].to_vec() })
                        }
                        None => ok(v.clone()),
                    },
                    None => missing(),
                },
                m => panic!("unexpected {m}"),
            }
        }
    }

    #[test]
    fn admitted_listing_rejects_incomplete_and_repeating_pagination() {
        #[derive(Debug)]
        struct BrokenPage { cycle: usize, calls: std::sync::atomic::AtomicUsize }
        impl HttpTransport for BrokenPage {
            fn send(&self, _: &HttpRequest) -> Result<HttpResponse> {
                let call=self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let continuation=if self.cycle == 0 { String::new() } else { format!("<NextContinuationToken>token-{}</NextContinuationToken>", call % self.cycle) };
                Ok(HttpResponse {status:200,headers:vec![],body:format!("<ListBucketResult><IsTruncated>true</IsTruncated><Contents><Key>idx/a</Key></Contents>{continuation}</ListBucketResult>").into_bytes()})
            }
        }
        for cycle in [0,1,2,3] {
            let http=Arc::new(BrokenPage{cycle,calls:std::sync::atomic::AtomicUsize::new(0)});
            let cfg=S3Config{endpoint:"http://minio:9000".into(),region:"us-east-1".into(),bucket:"bkt".into(),access_key_id:"AK".into(),secret_access_key:"SK".into(),session_token:None,path_style:true};
            let store=S3ObjectStore::new(cfg,http.clone());
            assert!(matches!(store.list_admitted("idx/"),Err(RusticError::Corrupted(_))));
            assert_eq!(http.calls.load(std::sync::atomic::Ordering::Relaxed),cycle+1);
        }
    }

    #[test]
    fn s3_store_against_fake_endpoint() {
        let cfg = S3Config {
            endpoint: "http://minio:9000".into(),
            region: "us-east-1".into(),
            bucket: "bkt".into(),
            access_key_id: "AK".into(),
            secret_access_key: "SK".into(),
            session_token: None,
            path_style: true,
        };
        let s = S3ObjectStore::new(cfg, Arc::new(FakeS3::default()));
        s.put("idx/seg_00000001.rstc", b"0123456789").unwrap();
        s.put("idx/meta.json", b"{}").unwrap();
        assert_eq!(s.get("idx/seg_00000001.rstc").unwrap(), b"0123456789");
        assert_eq!(s.get_range("idx/seg_00000001.rstc", 3..6).unwrap(), b"345");
        assert!(s.exists("idx/meta.json").unwrap());
        assert_eq!(s.list("idx/").unwrap(), vec!["idx/meta.json".to_string(), "idx/seg_00000001.rstc".to_string()]);
        s.delete("idx/meta.json").unwrap();
        assert!(!s.exists("idx/meta.json").unwrap());
        assert!(matches!(s.get("idx/meta.json"), Err(RusticError::IndexNotFound(_))));
    }
}
