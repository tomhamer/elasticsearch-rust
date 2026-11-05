//! HTTPS on the REST port (`xpack.security.http.ssl.*`): the node serves
//! TLS with the configured certificate, a plaintext client only loses its
//! own connection, and invalid TLS settings fail the start.
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A CA and a node certificate it signed (SANs `localhost`, `127.0.0.1`),
/// as PEM: `(ca, cert, key)`.
fn pki() -> (String, String, String) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair, SanType};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(DnType::CommonName, "http-tls-test-ca");
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let node_key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
    p.distinguished_name.push(DnType::CommonName, "http-tls-node");
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let cert = p.signed_by(&node_key, &ca, &ca_key).unwrap();
    (ca.pem(), cert.pem(), node_key.serialize_pem())
}

fn base_settings(dir: &std::path::Path, name: &str) -> Settings {
    let mut settings = Settings::new();
    for (key, value) in [
        ("cluster.name", json!(name)), ("node.name", json!("n0")),
        ("path.data", json!(dir.join("data").to_string_lossy())),
        ("path.logs", json!(dir.join("logs").to_string_lossy())),
        ("path.conf", json!(dir.join("config").to_string_lossy())),
        ("network.host", json!("127.0.0.1")), ("http.port", json!("0")),
        ("transport.port", json!("0")), ("discovery.type", json!("single-node")),
        ("cluster.routing.allocation.disk.watermark.low", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.high", json!("1b")),
        ("cluster.routing.allocation.disk.watermark.flood_stage", json!("1b")),
    ] { settings.insert(key.into(), value); }
    settings
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rest_port_serves_https_and_survives_plaintext_clients() {
    let dir = evidence::retained_dir("http_tls");
    let conf = dir.path().join("config");
    std::fs::create_dir_all(&conf).unwrap();
    let (ca, cert, key) = pki();
    std::fs::write(conf.join("http-ca.crt"), &ca).unwrap();
    std::fs::write(conf.join("http.crt"), &cert).unwrap();
    std::fs::write(conf.join("http.key"), &key).unwrap();
    let mut settings = base_settings(dir.path(), "http-tls");
    // Relative paths resolve against the config directory, as in ES.
    settings.insert("xpack.security.http.ssl.enabled".into(), json!("true"));
    settings.insert("xpack.security.http.ssl.certificate".into(), json!("http.crt"));
    settings.insert("xpack.security.http.ssl.key".into(), json!("http.key"));
    // HTTPS is served with security on (ES's default); authenticate as elastic.
    settings.insert("xpack.security.enabled".into(), json!("true"));
    settings.insert("bootstrap.password".into(), json!("test-password"));
    let config = NodeConfig::from_settings(settings).unwrap();
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(config).await.unwrap();
    assert!(node.http_tls.is_some());
    assert!(node.engine.security.cfg.read().http_ssl, "http_ssl reflects the served TLS");
    let port = ports::http_port(&node);
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(node, listener).await.unwrap(); });

    let mut auth = reqwest::header::HeaderMap::new();
    auth.insert(reqwest::header::AUTHORIZATION, reqwest::header::HeaderValue::from_static("Basic ZWxhc3RpYzp0ZXN0LXBhc3N3b3Jk"));
    let client = reqwest::Client::builder()
        .default_headers(auth)
        .use_rustls_tls()
        .add_root_certificate(reqwest::Certificate::from_pem(ca.as_bytes()).unwrap())
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let https = format!("https://localhost:{port}");
    let mut ready = false;
    let mut last = String::new();
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        match client.get(format!("{https}/_cluster/health")).send().await {
            Ok(r) if r.status().is_success() => { ready = true; break; }
            Ok(r) => last = format!("{} {}", r.status(), r.text().await.unwrap_or_default()),
            Err(e) => last = format!("{e:?}"),
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "HTTPS node did not become ready: {last}");
    let root: Value = client.get(format!("{https}/")).send().await.unwrap().json().await.unwrap();
    assert_eq!(root["tagline"], "You Know, for Search", "{root}");
    // The certificate also names the IP.
    let r = client.get(format!("https://127.0.0.1:{port}/_cluster/health")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    // A body round-trips over TLS.
    let r = client.put(format!("{https}/tls-index/_doc/1?refresh=true")).json(&json!({"f": "v"})).send().await.unwrap();
    assert!(r.status().is_success(), "{:?}", r.text().await);
    let doc: Value = client.get(format!("{https}/tls-index/_doc/1")).send().await.unwrap().json().await.unwrap();
    assert_eq!(doc["_source"]["f"], "v", "{doc}");

    // Plaintext HTTP to the TLS port gets no HTTP response (ES closes the
    // channel), over a raw socket and through an HTTP client.
    let mut raw = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    raw.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").await.unwrap();
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(15), raw.read_to_end(&mut buf)).await.expect("the node closes a plaintext connection");
    assert!(!buf.starts_with(b"HTTP/"), "plaintext request was answered: {}", String::from_utf8_lossy(&buf));
    let plain = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().unwrap();
    assert!(plain.get(format!("http://127.0.0.1:{port}/")).send().await.is_err());
    // A client that doesn't trust the node's CA fails its handshake.
    let untrusting = reqwest::Client::builder().use_rustls_tls().timeout(Duration::from_secs(10)).build().unwrap();
    assert!(untrusting.get(format!("{https}/")).send().await.is_err());

    // The node keeps serving TLS clients.
    assert!(!server.is_finished());
    let r = client.get(format!("{https}/_cluster/health")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_http_ssl_settings_fail_the_start() {
    let dir = evidence::retained_dir("http_tls_invalid");
    std::fs::create_dir_all(dir.path().join("config")).unwrap();
    let start = |extra: &[(&str, &str)]| {
        let mut settings = base_settings(dir.path(), "http-tls-invalid");
        for (k, v) in extra {
            settings.insert((*k).into(), json!(v));
        }
        NodeConfig::from_settings(settings).unwrap()
    };
    let err = |r: Result<_, Box<dyn std::error::Error + Send + Sync>>| match r {
        Ok(_) => panic!("the node started"),
        Err(e) => e.to_string(),
    };
    let e = err(elasticsearch_node::node::ClusterNode::start(start(&[("xpack.security.http.ssl.enabled", "true")])).await);
    assert!(e.contains("server ssl configuration requires a key and certificate"), "{e}");
    let e = err(elasticsearch_node::node::ClusterNode::start(start(&[
        ("xpack.security.http.ssl.enabled", "true"), ("xpack.security.http.ssl.certificate", "missing.crt"), ("xpack.security.http.ssl.key", "missing.key"),
    ])).await);
    assert!(e.starts_with("failed to load SSL configuration [xpack.security.http.ssl] - cannot read configured PEM certificate ["), "{e}");
    assert!(e.ends_with("missing.crt] because the file does not exist"), "{e}");
    let e = err(elasticsearch_node::node::ClusterNode::start(start(&[
        ("xpack.security.http.ssl.enabled", "true"), ("xpack.security.http.ssl.keystore.path", "http.p12"), ("xpack.security.http.ssl.key", "http.key"),
    ])).await);
    assert_eq!(e, "cannot specify both [xpack.security.http.ssl.keystore.path] and [xpack.security.http.ssl.key]");
    let e = err(elasticsearch_node::node::ClusterNode::start(start(&[("reindex.ssl.verification_mode", "sometimes")])).await);
    assert!(e.starts_with("could not resolve ssl client verification mode, unknown value [sometimes]"), "{e}");
}

/// A fresh client (its own connection pool: every request below is a new
/// connection) trusting only `ca`.
fn client_trusting(ca: &str) -> reqwest::Client {
    let mut auth = reqwest::header::HeaderMap::new();
    auth.insert(reqwest::header::AUTHORIZATION, reqwest::header::HeaderValue::from_static("Basic ZWxhc3RpYzp0ZXN0LXBhc3N3b3Jk"));
    reqwest::Client::builder()
        .default_headers(auth)
        .use_rustls_tls()
        .tls_built_in_root_certs(false)
        .add_root_certificate(reqwest::Certificate::from_pem(ca.as_bytes()).unwrap())
        .pool_max_idle_per_host(0)
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

async fn healthy(ca: &str, url: &str) -> Result<(), String> {
    match client_trusting(ca).get(url).send().await {
        Ok(r) if r.status().is_success() => Ok(()),
        Ok(r) => Err(format!("{} {}", r.status(), r.text().await.unwrap_or_default())),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Write `contents` to `path` through a rename, as certificate rotation
/// tooling does (a reader never sees a half-written file).
fn replace(path: &std::path::Path, contents: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, contents).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

/// ES `SSLConfigurationReloader`: rotating the HTTP certificate and key on
/// disk applies to new connections without a restart; a broken file keeps
/// the current certificate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rotated_http_certificate_is_served_without_restart() {
    let dir = evidence::retained_dir("http_tls_reload");
    let conf = dir.path().join("config");
    std::fs::create_dir_all(&conf).unwrap();
    let (ca1, cert1, key1) = pki();
    std::fs::write(conf.join("http.crt"), &cert1).unwrap();
    std::fs::write(conf.join("http.key"), &key1).unwrap();
    let mut settings = base_settings(dir.path(), "http-tls-reload");
    settings.insert("xpack.security.http.ssl.enabled".into(), json!("true"));
    settings.insert("xpack.security.http.ssl.certificate".into(), json!("http.crt"));
    settings.insert("xpack.security.http.ssl.key".into(), json!("http.key"));
    settings.insert("xpack.security.enabled".into(), json!("true"));
    settings.insert("bootstrap.password".into(), json!("test-password"));
    // ES's resource watcher interval for SSL files (default 5s), shortened.
    settings.insert("resource.reload.interval.high".into(), json!("200ms"));
    let config = NodeConfig::from_settings(settings).unwrap();
    let (node, listener) = elasticsearch_node::node::ClusterNode::start(config).await.unwrap();
    let port = ports::http_port(&node);
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let server = tokio::spawn(async move { elasticsearch_node::front::serve(node, listener).await.unwrap(); });
    let url = format!("https://localhost:{port}/_cluster/health");

    let mut last = String::new();
    let mut ready = false;
    for _ in 0..100 {
        assert!(!server.is_finished(), "node exited at startup");
        match healthy(&ca1, &url).await {
            Ok(()) => { ready = true; break; }
            Err(e) => last = e,
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "HTTPS node did not become ready: {last}");
    // A connection opened before the rotation (kept alive by its client).
    let established = {
        let mut auth = reqwest::header::HeaderMap::new();
        auth.insert(reqwest::header::AUTHORIZATION, reqwest::header::HeaderValue::from_static("Basic ZWxhc3RpYzp0ZXN0LXBhc3N3b3Jk"));
        reqwest::Client::builder()
            .default_headers(auth)
            .use_rustls_tls()
            .tls_built_in_root_certs(false)
            .add_root_certificate(reqwest::Certificate::from_pem(ca1.as_bytes()).unwrap())
            .pool_idle_timeout(Duration::from_secs(600))
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    };
    assert!(established.get(&url).send().await.unwrap().status().is_success());

    // Rotate to a certificate from another CA.
    let (ca2, cert2, key2) = pki();
    assert!(healthy(&ca2, &url).await.is_err(), "the new CA is not in use yet");
    replace(&conf.join("http.key"), &key2);
    replace(&conf.join("http.crt"), &cert2);
    let mut rotated = false;
    for _ in 0..100 {
        if healthy(&ca2, &url).await.is_ok() {
            rotated = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(rotated, "new connections still get the old certificate");
    // New connections no longer present the old chain.
    let e = healthy(&ca1, &url).await.unwrap_err();
    assert!(e.to_lowercase().contains("certificate") || e.contains("UnknownIssuer") || e.contains("connect"), "{e}");
    // The established connection keeps working on its handshake.
    assert!(established.get(&url).send().await.unwrap().status().is_success());

    // A broken certificate file: the reload fails, the current certificate stays.
    replace(&conf.join("http.crt"), "-----BEGIN CERTIFICATE-----\nnot base64\n-----END CERTIFICATE-----\n");
    tokio::time::sleep(Duration::from_millis(1500)).await;
    healthy(&ca2, &url).await.expect("a failed reload keeps the previous certificate");
    assert!(healthy(&ca1, &url).await.is_err());
    // A key that doesn't match: still the current certificate.
    replace(&conf.join("http.crt"), &cert1);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    healthy(&ca2, &url).await.expect("a mismatched key and certificate keep the previous certificate");
    // Restoring a consistent pair applies again.
    replace(&conf.join("http.key"), &key1);
    let mut back = false;
    for _ in 0..100 {
        if healthy(&ca1, &url).await.is_ok() {
            back = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(back, "the restored certificate was not picked up");
    assert!(healthy(&ca2, &url).await.is_err());
    assert!(!server.is_finished());
    server.abort();
}
