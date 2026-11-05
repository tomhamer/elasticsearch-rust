//! `_reindex` from an `https://` remote (`reindex.ssl.*`) and HTTPS on the
//! engine-only REST server (`serve_listener_with_tls`). The certificates are
//! `elasticsearch-ssl`'s fixtures: `node.crt` (SANs `localhost`,
//! `127.0.0.1`) signed by `ca.crt`.

use elasticsearch_engine::node::Node;
use reqwest::Client;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

fn data(name: &str) -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../elasticsearch-ssl/tests/data").join(name)
}

/// `xpack.security.http.ssl.*` with the fixture certificate.
fn server_tls() -> Arc<elasticsearch_ssl::rustls::ServerConfig> {
    let settings = [
        ("xpack.security.http.ssl.enabled", "true".to_string()),
        ("xpack.security.http.ssl.certificate", data("node.crt").display().to_string()),
        ("xpack.security.http.ssl.key", data("node.key").display().to_string()),
    ];
    let get = |k: &str| settings.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
    elasticsearch_ssl::http_server(&get, &|_| Vec::new(), &data("")).unwrap().unwrap()
}

/// Serve `node` (over TLS with `tls`); returns its port.
async fn serve(node: Arc<Node>, tls: Option<Arc<elasticsearch_ssl::rustls::ServerConfig>>) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener_with_tls(node, listener, tls).await;
    });
    port
}

async fn call(c: &Client, method: reqwest::Method, url: String, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, url);
    if let Some(b) = body {
        r = r.header("content-type", "application/json").body(b.to_string());
    }
    let resp = r.send().await.unwrap();
    let st = resp.status().as_u16();
    let t = resp.text().await.unwrap();
    (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reindex_from_an_https_remote() {
    // `reindex.remote.whitelist` (read from the environment by this server).
    // SAFETY: set before any reindex runs; this binary has a single test.
    unsafe { std::env::set_var("ES_REINDEX_REMOTE_WHITELIST", "localhost:*,127.0.0.1:*") };
    let tmp = tempfile::tempdir().unwrap();
    for d in ["source", "trusting", "untrusting", "lenient"] {
        std::fs::create_dir_all(tmp.path().join(d)).unwrap();
    }

    // The remote: HTTPS only.
    let source = Node::open(&tmp.path().join("source")).unwrap();
    let source_port = serve(source, Some(server_tls())).await;
    let ca = std::fs::read(data("ca.crt")).unwrap();
    let https = Client::builder().use_rustls_tls().add_root_certificate(reqwest::Certificate::from_pem(&ca).unwrap()).build().unwrap();
    let remote = format!("https://localhost:{source_port}");
    let (st, v) = call(&https, reqwest::Method::PUT, format!("{remote}/src"), Some(json!({"settings": {"number_of_replicas": 0}}))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..7 {
        let (st, v) = call(&https, reqwest::Method::PUT, format!("{remote}/src/_doc/{i}?refresh=true"), Some(json!({"n": i}))).await;
        assert!(st == 200 || st == 201, "{v}");
    }
    // Plain HTTP to the HTTPS port gets no answer.
    assert!(Client::new().get(format!("http://127.0.0.1:{source_port}/")).send().await.is_err());

    // A destination trusting the remote's CA (`reindex.ssl.certificate_authorities`).
    let trusting = Node::open(&tmp.path().join("trusting")).unwrap();
    trusting.node_settings.write().insert("reindex.ssl.certificate_authorities".into(), json!([data("ca.crt").display().to_string()]));
    let trusting_port = serve(trusting, None).await;
    let plain = Client::new();
    let local = format!("http://127.0.0.1:{trusting_port}");
    let body = |host: &str, dest: &str| json!({"source": {"remote": {"host": host}, "index": "src", "size": 3}, "dest": {"index": dest}});
    let (st, v) = call(&plain, reqwest::Method::POST, format!("{local}/_reindex?refresh=true"), Some(body(&remote, "dst"))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["created"], 7, "{v}");
    let (_, c) = call(&plain, reqwest::Method::GET, format!("{local}/dst/_count"), None).await;
    assert_eq!(c["count"], 7, "{c}");
    // By IP too (the certificate names 127.0.0.1).
    let (st, v) = call(&plain, reqwest::Method::POST, format!("{local}/_reindex?refresh=true"), Some(body(&format!("https://127.0.0.1:{source_port}"), "dst-ip"))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["created"], 7, "{v}");
    // `https://` to a plaintext server fails its handshake: never plaintext.
    let (st, v) = call(&plain, reqwest::Method::POST, format!("{local}/_reindex"), Some(body(&format!("https://127.0.0.1:{trusting_port}"), "never"))).await;
    assert_ne!(st, 200, "{v}");
    assert!(v.to_string().contains("s_s_l_handshake_exception"), "{v}");

    // A destination with the default trust (`verification_mode: full`
    // against the default roots) rejects the remote's self-made CA.
    let untrusting = Node::open(&tmp.path().join("untrusting")).unwrap();
    let untrusting_port = serve(untrusting, None).await;
    let (st, v) = call(&plain, reqwest::Method::POST, format!("http://127.0.0.1:{untrusting_port}/_reindex"), Some(body(&remote, "untrusted"))).await;
    assert_ne!(st, 200, "{v}");
    assert!(v.to_string().contains("s_s_l_handshake_exception"), "{v}");
    let (_, c) = call(&plain, reqwest::Method::GET, format!("http://127.0.0.1:{untrusting_port}/untrusted/_count"), None).await;
    assert_ne!(c["count"], 7, "nothing was copied: {c}");

    // `verification_mode: certificate` checks the chain but not the name.
    let lenient = Node::open(&tmp.path().join("lenient")).unwrap();
    {
        let mut s = lenient.node_settings.write();
        s.insert("reindex.ssl.certificate_authorities".into(), json!(data("ca.crt").display().to_string()));
        s.insert("reindex.ssl.verification_mode".into(), json!("certificate"));
    }
    let lenient_port = serve(lenient, None).await;
    // `localhost.localdomain` resolves to loopback on common hosts; fall back
    // to checking the name-mismatch path only where it resolves.
    use std::net::ToSocketAddrs;
    if ("localhost.localdomain", source_port).to_socket_addrs().map(|mut a| a.next().is_some()).unwrap_or(false) {
        unsafe { std::env::set_var("ES_REINDEX_REMOTE_WHITELIST", "localhost:*,127.0.0.1:*,localhost.localdomain:*") };
        let mismatched = format!("https://localhost.localdomain:{source_port}");
        let (st, v) = call(&plain, reqwest::Method::POST, format!("http://127.0.0.1:{lenient_port}/_reindex?refresh=true"), Some(body(&mismatched, "lenient"))).await;
        assert_eq!(st, 200, "{v}");
        let (st, v) = call(&plain, reqwest::Method::POST, format!("{local}/_reindex"), Some(body(&mismatched, "strict"))).await;
        assert_ne!(st, 200, "full verification rejects the host name: {v}");
    }
}
