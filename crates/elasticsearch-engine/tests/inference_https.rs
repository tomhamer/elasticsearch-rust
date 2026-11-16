//! Hosted inference providers over HTTPS (against a local TLS mock): the
//! default client verifies certificates and host names against the default
//! trust (here a test CA supplied through `SSL_CERT_FILE`, as an operator
//! would add a corporate CA), and calls time out.
//!
//! One test in this binary: it sets `SSL_CERT_FILE` before the default trust
//! is first loaded.
use elasticsearch_engine::inference::providers::{HttpClient, Transport};
use elasticsearch_engine::inference::InferenceService;
use elasticsearch_ssl::rustls;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A CA and a certificate it signed for `localhost` only: `(ca, cert, key)` PEM.
fn pki(cn: &str) -> (String, String, String) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(DnType::CommonName, cn);
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.distinguished_name.push(DnType::CommonName, "provider-mock");
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let cert = p.signed_by(&key, &ca, &ca_key).unwrap();
    (ca.pem(), cert.pem(), key.serialize_pem())
}

/// Serve OpenAI-style embeddings over TLS on 127.0.0.1; returns the port.
fn tls_mock(cert: &str, key: &str) -> u16 {
    let certs = elasticsearch_ssl::parse_certificates(cert.as_bytes()).unwrap();
    let key = elasticsearch_ssl::parse_private_key(key.as_bytes(), None, "mock").unwrap();
    let cfg = Arc::new(
        rustls::ServerConfig::builder_with_provider(elasticsearch_ssl::provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .unwrap(),
    );
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for sk in l.incoming() {
            let Ok(sk) = sk else { continue };
            let cfg = cfg.clone();
            std::thread::spawn(move || {
                sk.set_read_timeout(Some(Duration::from_secs(10))).ok();
                let conn = rustls::ServerConnection::new(cfg).unwrap();
                let mut s = rustls::StreamOwned::new(conn, sk);
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                loop {
                    let n = match s.read(&mut buf) {
                        Ok(n) => n,
                        Err(_) => return, // e.g. the client rejected our certificate
                    };
                    if n == 0 {
                        return;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(h) = text.find("\r\n\r\n") {
                        let cl: usize = text[..h].to_ascii_lowercase().split("content-length:").nth(1).and_then(|r| r.split_whitespace().next().and_then(|v| v.parse().ok())).unwrap_or(0);
                        if raw.len() >= h + 4 + cl {
                            break;
                        }
                    }
                }
                let req = String::from_utf8_lossy(&raw).to_string();
                assert!(req.starts_with("POST /v1/embeddings HTTP/1.1\r\n"), "{req}");
                assert!(req.contains("Authorization: Bearer sekret"), "{req}");
                let count = if req.contains("\"a\"") && req.contains("\"b\"") { 2 } else { 1 };
                let data: Vec<Value> = (0..count).map(|i| json!({"index": i, "embedding": [0.1 * (i as f64 + 1.0), 0.2, 0.3]})).collect();
                let body = json!({"data": data}).to_string();
                let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes());
                // Keep-alive style: the response ends at Content-Length, not at close.
                let _ = s.flush();
                std::thread::sleep(Duration::from_millis(200));
                s.conn.send_close_notify();
                let _ = s.flush();
            });
        }
    });
    port
}

#[test]
fn hosted_provider_over_https() {
    let dir = tempfile::tempdir().unwrap();
    let (ca, cert, key) = pki("inference-https-test-ca");
    let ca_file = dir.path().join("ca.pem");
    std::fs::write(&ca_file, &ca).unwrap();
    // The default trust includes the platform store, which honours SSL_CERT_FILE.
    unsafe { std::env::set_var("SSL_CERT_FILE", &ca_file) };
    let port = tls_mock(&cert, &key);
    let url = format!("https://localhost:{port}/v1/embeddings");

    // The default transport speaks HTTPS: endpoint creation (validation
    // call, learns the dimensions) and inference.
    let s = InferenceService::open(&dir.path().join("data"));
    let put = s.put("text_embedding", "oe", &json!({"service": "openai", "service_settings": {"api_key": "sekret", "model_id": "m", "url": url}})).unwrap();
    assert_eq!(put["service_settings"]["dimensions"], 3);
    let out = s.infer(None, "oe", &json!({"input": ["a", "b"]})).unwrap();
    assert_eq!(out["text_embedding"].as_array().unwrap().len(), 2);

    let body = json!({"input": ["a"], "model": "m"});
    let auth = vec![("Authorization".to_string(), "Bearer sekret".to_string())];
    // Host name verification: the certificate names `localhost` only.
    let e = HttpClient::default().post(&format!("https://127.0.0.1:{port}/v1/embeddings"), &auth, &body, Duration::from_secs(5)).unwrap_err();
    assert!(e.to_ascii_lowercase().contains("certificate"), "{e}");
    // Chain verification: a client trusting another CA refuses the server.
    let (other_ca, _, _) = pki("some-other-ca");
    let mut roots = rustls::RootCertStore::empty();
    roots.add_parsable_certificates(elasticsearch_ssl::parse_certificates(other_ca.as_bytes()).unwrap());
    let strict = rustls::ClientConfig::builder_with_provider(elasticsearch_ssl::provider()).with_safe_default_protocol_versions().unwrap().with_root_certificates(roots).with_no_client_auth();
    let e = HttpClient::with_tls(Arc::new(strict)).post(&url, &auth, &body, Duration::from_secs(5)).unwrap_err();
    assert!(e.to_ascii_lowercase().contains("certificate") || e.contains("UnknownIssuer"), "{e}");

    // Timeouts: a server that accepts and never answers the handshake.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let silent_port = silent.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let held: Vec<_> = silent.incoming().take(1).collect();
        std::thread::sleep(Duration::from_secs(10));
        drop(held);
    });
    let t0 = Instant::now();
    let e = HttpClient::default().post(&format!("https://localhost:{silent_port}/v1/embeddings"), &auth, &body, Duration::from_millis(300)).unwrap_err();
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    assert!(e.contains("timed out") || e.contains("Resource temporarily unavailable"), "{e}");
}
