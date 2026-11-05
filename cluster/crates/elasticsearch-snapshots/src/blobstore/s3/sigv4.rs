//! AWS Signature Version 4 for S3 (header-based auth).

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("access_key", &self.access_key)
            .field("secret_key", &"<redacted>")
            .field("session_token", &self.session_token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Credentials {
    /// `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`.
    pub fn from_env() -> Option<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok().filter(|s| !s.is_empty())?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok().filter(|s| !s.is_empty())?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok().filter(|s| !s.is_empty());
        Some(Self { access_key, secret_key, session_token })
    }
}

fn is_unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~')
}

/// RFC 3986 encoding as required by SigV4 (everything but unreserved chars).
pub fn uri_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if is_unreserved(b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Encode an object key / path, keeping `/` separators.
pub fn uri_encode_path(s: &str) -> String {
    s.split('/').map(uri_encode).collect::<Vec<_>>().join("/")
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

/// A request to sign. `path` must already be URI-encoded (see [`uri_encode_path`]);
/// `query` holds raw (unencoded) pairs.
#[derive(Debug, Clone)]
pub struct SignableRequest<'a> {
    pub method: &'a str,
    /// Host header value (with `:port` if non-default).
    pub host: &'a str,
    pub path: &'a str,
    pub query: &'a [(String, String)],
    /// Extra headers to sign (any case); `host`, `x-amz-date`,
    /// `x-amz-content-sha256` and `x-amz-security-token` are added automatically.
    pub headers: &'a [(String, String)],
    pub payload_sha256: &'a str,
}

pub fn canonical_query(query: &[(String, String)]) -> String {
    let mut q: Vec<(String, String)> = query.iter().map(|(k, v)| (uri_encode(k), uri_encode(v))).collect();
    q.sort();
    q.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// Returns `(canonical_request, signed_headers, all_headers_to_send)`.
pub fn canonical_request(req: &SignableRequest<'_>, amz_date: &str, creds: &Credentials) -> (String, String, Vec<(String, String)>) {
    let mut headers: Vec<(String, String)> = req
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.split_whitespace().collect::<Vec<_>>().join(" ")))
        .collect();
    headers.push(("host".into(), req.host.to_string()));
    headers.push(("x-amz-date".into(), amz_date.to_string()));
    headers.push(("x-amz-content-sha256".into(), req.payload_sha256.to_string()));
    if let Some(t) = &creds.session_token {
        headers.push(("x-amz-security-token".into(), t.clone()));
    }
    headers.sort();
    headers.dedup_by(|a, b| a.0 == b.0);
    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers = headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");
    let path = if req.path.is_empty() { "/" } else { req.path };
    let creq = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        req.method,
        path,
        canonical_query(req.query),
        canonical_headers,
        signed_headers,
        req.payload_sha256
    );
    (creq, signed_headers, headers)
}

/// Sign a request. `amz_date` is `YYYYMMDDTHHMMSSZ`. Returns the headers to
/// send (everything that was signed except `host`, plus `authorization`).
pub fn sign(req: &SignableRequest<'_>, creds: &Credentials, region: &str, service: &str, amz_date: &str) -> Vec<(String, String)> {
    let (creq, signed_headers, headers) = canonical_request(req, amz_date, creds);
    let date = &amz_date[..8];
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let sts = format!("AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}", sha256_hex(creq.as_bytes()));
    let k_date = hmac(format!("AWS4{}", creds.secret_key).as_bytes(), date.as_bytes());
    let k_region = hmac(&k_date, region.as_bytes());
    let k_service = hmac(&k_region, service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
    let signature = hex::encode(hmac(&k_signing, sts.as_bytes()));
    let mut out: Vec<(String, String)> = headers.into_iter().filter(|(k, _)| k != "host").collect();
    out.push((
        "authorization".into(),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            creds.access_key
        ),
    ));
    out
}

pub fn amz_date(now: chrono::DateTime<chrono::Utc>) -> String {
    now.format("%Y%m%dT%H%M%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_creds() -> Credentials {
        Credentials {
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
        }
    }

    fn signature(h: &[(String, String)]) -> String {
        let auth = &h.iter().find(|(k, _)| k == "authorization").unwrap().1;
        auth.rsplit("Signature=").next().unwrap().to_string()
    }

    // Test vectors from the AWS S3 SigV4 documentation
    // ("Signature Calculations for the Authorization Header").
    #[test]
    fn aws_vector_get_object() {
        let headers = vec![("Range".to_string(), "bytes=0-9".to_string())];
        let req = SignableRequest {
            method: "GET",
            host: "examplebucket.s3.amazonaws.com",
            path: "/test.txt",
            query: &[],
            headers: &headers,
            payload_sha256: EMPTY_SHA256,
        };
        let (creq, signed, _) = canonical_request(&req, "20130524T000000Z", &example_creds());
        assert_eq!(
            creq,
            "GET\n/test.txt\n\nhost:examplebucket.s3.amazonaws.com\nrange:bytes=0-9\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130524T000000Z\n\nhost;range;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(signed, "host;range;x-amz-content-sha256;x-amz-date");
        let h = sign(&req, &example_creds(), "us-east-1", "s3", "20130524T000000Z");
        assert_eq!(signature(&h), "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
    }

    #[test]
    fn aws_vector_put_object() {
        let payload = b"Welcome to Amazon S3.";
        let hash = sha256_hex(payload);
        assert_eq!(hash, "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072");
        let headers = vec![
            ("Date".to_string(), "Fri, 24 May 2013 00:00:00 GMT".to_string()),
            ("x-amz-storage-class".to_string(), "REDUCED_REDUNDANCY".to_string()),
        ];
        let path = format!("/{}", uri_encode_path("test$file.text"));
        assert_eq!(path, "/test%24file.text");
        let req = SignableRequest {
            method: "PUT",
            host: "examplebucket.s3.amazonaws.com",
            path: &path,
            query: &[],
            headers: &headers,
            payload_sha256: &hash,
        };
        let h = sign(&req, &example_creds(), "us-east-1", "s3", "20130524T000000Z");
        assert_eq!(signature(&h), "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd");
    }

    #[test]
    fn aws_vector_get_lifecycle_and_list() {
        let q = vec![("lifecycle".to_string(), String::new())];
        let req = SignableRequest {
            method: "GET",
            host: "examplebucket.s3.amazonaws.com",
            path: "/",
            query: &q,
            headers: &[],
            payload_sha256: EMPTY_SHA256,
        };
        let h = sign(&req, &example_creds(), "us-east-1", "s3", "20130524T000000Z");
        assert_eq!(signature(&h), "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543");

        let q = vec![("prefix".to_string(), "J".to_string()), ("max-keys".to_string(), "2".to_string())];
        let req = SignableRequest { query: &q, ..req };
        assert_eq!(canonical_query(&q), "max-keys=2&prefix=J");
        let h = sign(&req, &example_creds(), "us-east-1", "s3", "20130524T000000Z");
        assert_eq!(signature(&h), "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
    }

    /// MinIO-style path-style addressing: bucket in the path, host with port.
    /// The signing math is the GET-object vector's; only the canonical
    /// request's URI and host change, which we check structurally, and the
    /// session token must be signed.
    #[test]
    fn path_style_minio_request() {
        let mut creds = example_creds();
        creds.session_token = Some("TOKEN".into());
        let key = "base/indices/abc/0/__x y";
        let path = format!("/{}/{}", "examplebucket", uri_encode_path(key));
        let q = vec![("list-type".to_string(), "2".to_string()), ("prefix".to_string(), "base/a b/".to_string())];
        let req = SignableRequest {
            method: "GET",
            host: "localhost:9000",
            path: &path,
            query: &q,
            headers: &[],
            payload_sha256: EMPTY_SHA256,
        };
        let (creq, signed, _) = canonical_request(&req, "20130524T000000Z", &creds);
        assert_eq!(
            creq,
            "GET\n/examplebucket/base/indices/abc/0/__x%20y\nlist-type=2&prefix=base%2Fa%20b%2F\nhost:localhost:9000\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130524T000000Z\nx-amz-security-token:TOKEN\n\nhost;x-amz-content-sha256;x-amz-date;x-amz-security-token\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(signed, "host;x-amz-content-sha256;x-amz-date;x-amz-security-token");
        let h = sign(&req, &creds, "us-east-1", "s3", "20130524T000000Z");
        assert!(h.iter().any(|(k, v)| k == "x-amz-security-token" && v == "TOKEN"));
        assert!(!h.iter().any(|(k, _)| k == "host"));
        let auth = &h.iter().find(|(k, _)| k == "authorization").unwrap().1;
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, Signature="));
        // Deterministic.
        assert_eq!(signature(&h), signature(&sign(&req, &creds, "us-east-1", "s3", "20130524T000000Z")));
    }
}
