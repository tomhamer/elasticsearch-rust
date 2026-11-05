use super::*;
use rustls::pki_types::ServerName;
use std::collections::HashMap;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Settings `pairs` (keys relative to `prefix`); list values comma-separated.
fn settings(prefix: &str, pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (format!("{prefix}{k}"), v.to_string())).collect()
}

fn parse(prefix: &str, pairs: &[(&str, &str)]) -> Result<SslConfig, String> {
    let m = settings(prefix, pairs);
    SslConfig::parse(
        prefix,
        &|k| m.get(k).cloned(),
        &|k| m.get(k).map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default(),
        &data(),
        Defaults { verification_mode: VerificationMode::Full, client_authentication: ClientAuth::None },
    )
}

fn http(pairs: &[(&str, &str)]) -> Result<Option<Arc<ServerConfig>>, String> {
    let m = settings(HTTP_SSL, pairs);
    http_server(&|k| m.get(k).cloned(), &|k| m.get(k).map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default(), &data())
}

fn reindex(pairs: &[(&str, &str)]) -> Result<Arc<ClientConfig>, String> {
    let m = settings(REINDEX_SSL, pairs);
    reindex_client(&|k| m.get(k).cloned(), &|k| m.get(k).map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default(), &data())
}

/// A deterministic RSA PKCS#1 v1.5 signature identifies the key.
fn fingerprint(key: &PrivateKeyDer<'static>) -> Vec<u8> {
    let k = rustls::crypto::ring::sign::any_supported_type(key).expect("supported key");
    let signer = k.choose_scheme(&[rustls::SignatureScheme::RSA_PKCS1_SHA256]).expect("rsa signer");
    signer.sign(b"ssl key fingerprint").unwrap()
}

fn node_key() -> PrivateKeyDer<'static> {
    parse_private_key(&std::fs::read(data().join("node.key")).unwrap(), None, "node.key").unwrap()
}

fn cert(name: &str) -> CertificateDer<'static> {
    parse_certificates(&std::fs::read(data().join(name)).unwrap()).unwrap().remove(0)
}

/// Run a TLS handshake in memory; `Err` is the first side's failure.
fn handshake(client: Arc<ClientConfig>, server: Arc<ServerConfig>, name: &str) -> Result<(), rustls::Error> {
    let mut c = rustls::ClientConnection::new(client, ServerName::try_from(name.to_string()).unwrap())?;
    let mut s = rustls::ServerConnection::new(server)?;
    for _ in 0..16 {
        let mut buf = Vec::new();
        c.write_tls(&mut buf).unwrap();
        if !buf.is_empty() {
            s.read_tls(&mut &buf[..]).unwrap();
            s.process_new_packets()?;
        }
        let mut buf = Vec::new();
        s.write_tls(&mut buf).unwrap();
        if !buf.is_empty() {
            c.read_tls(&mut &buf[..]).unwrap();
            c.process_new_packets()?;
        }
        if !c.is_handshaking() && !s.is_handshaking() {
            return Ok(());
        }
    }
    panic!("handshake did not finish");
}

#[test]
fn http_ssl_is_off_unless_enabled() {
    assert!(http(&[]).unwrap().is_none());
    assert!(http(&[("enabled", "false"), ("certificate", "missing.crt")]).unwrap().is_none());
    let e = http(&[("enabled", "yes")]).unwrap_err();
    assert_eq!(e, "Failed to parse value [yes] for setting [xpack.security.http.ssl.enabled] as only [true] or [false] are allowed.");
}

#[test]
fn http_ssl_enabled_without_key_material_fails_like_es() {
    let e = http(&[("enabled", "true")]).unwrap_err();
    assert_eq!(
        e,
        "invalid SSL configuration for xpack.security.http.ssl - server ssl configuration requires a key and certificate, but these have not been configured; \
         you must set either [xpack.security.http.ssl.keystore.path], or both [xpack.security.http.ssl.key] and [xpack.security.http.ssl.certificate]"
    );
}

#[test]
fn conflicting_and_incomplete_settings_are_rejected() {
    let p = HTTP_SSL;
    assert_eq!(parse(p, &[("certificate", "node.crt")]).unwrap_err(), "cannot specify [xpack.security.http.ssl.certificate] without also setting [xpack.security.http.ssl.key]");
    assert_eq!(parse(p, &[("key", "node.key")]).unwrap_err(), "cannot specify [xpack.security.http.ssl.key] without also setting [xpack.security.http.ssl.certificate]");
    assert_eq!(
        parse(p, &[("keystore.path", "node-aes.p12"), ("certificate", "node.crt"), ("key", "node.key")]).unwrap_err(),
        "cannot specify both [xpack.security.http.ssl.keystore.path] and [xpack.security.http.ssl.certificate]"
    );
    assert_eq!(
        parse(p, &[("certificate_authorities", "ca.crt"), ("truststore.path", "truststore.p12")]).unwrap_err(),
        "cannot specify both [xpack.security.http.ssl.certificate_authorities] and [xpack.security.http.ssl.truststore.path]"
    );
    assert_eq!(
        parse(p, &[("keystore.path", "node-aes.p12"), ("keystore.password", "a"), ("keystore.secure_password", "b")]).unwrap_err(),
        "cannot specify both [xpack.security.http.ssl.keystore.secure_password] and [xpack.security.http.ssl.keystore.password]"
    );
    assert_eq!(
        parse(p, &[("certificate", "node.crt"), ("key", "node.key"), ("key_passphrase", "a"), ("secure_key_passphrase", "b")]).unwrap_err(),
        "cannot specify both [xpack.security.http.ssl.secure_key_passphrase] and [xpack.security.http.ssl.key_passphrase]"
    );
    assert!(parse(p, &[("verification_mode", "strict")]).unwrap_err().starts_with("could not resolve ssl client verification mode, unknown value [strict]"));
    assert!(parse(p, &[("client_authentication", "maybe")]).unwrap_err().starts_with("could not resolve ssl client authentication, unknown value [maybe]"));
    assert!(parse(p, &[("supported_protocols", "TLSv9")]).unwrap_err().contains("unsupported protocol [TLSv9]"));
    assert!(parse(p, &[("supported_protocols", "TLSv1.1")]).unwrap_err().contains("does not include a protocol this node supports"));
    assert!(parse(p, &[("keystore.path", "node.jks"), ("keystore.type", "bcfks")]).unwrap_err().starts_with("unsupported keystore type [bcfks]"));
}

#[test]
fn settings_parse_with_es_defaults() {
    let c = parse(REINDEX_SSL, &[]).unwrap();
    assert_eq!(c.key, None);
    assert_eq!(c.trust, TrustMaterial::Default);
    assert_eq!(c.verification_mode, VerificationMode::Full);
    assert_eq!(c.client_authentication, ClientAuth::None);
    assert_eq!(c.supported_protocols, vec![Protocol::Tls13, Protocol::Tls12]);
    let c = parse(HTTP_SSL, &[("certificate", "node.crt"), ("key", "/abs/node.key"), ("secure_key_passphrase", "pw"), ("certificate_authorities", "ca.crt,other.crt"), ("verification_mode", "CERTIFICATE"), ("client_authentication", "optional"), ("supported_protocols", "TLSv1.2,TLSv1.1")]).unwrap();
    assert_eq!(c.key, Some(KeyMaterial::Pem { certificate: data().join("node.crt"), key: PathBuf::from("/abs/node.key"), passphrase: Some("pw".into()) }));
    assert_eq!(c.trust, TrustMaterial::Pem(vec![data().join("ca.crt"), data().join("other.crt")]));
    assert_eq!(c.verification_mode, VerificationMode::Certificate);
    assert_eq!(c.client_authentication, ClientAuth::Optional);
    assert_eq!(c.supported_protocols, vec![Protocol::Tls12]);
    // Keystore type from the extension; the key password defaults to the store password.
    let c = parse(HTTP_SSL, &[("keystore.path", "node-aes.p12"), ("keystore.secure_password", "testnode")]).unwrap();
    assert_eq!(c.key, Some(KeyMaterial::Keystore { path: data().join("node-aes.p12"), kind: StoreType::Pkcs12, password: "testnode".into(), key_password: "testnode".into() }));
    let c = parse(HTTP_SSL, &[("keystore.path", "node.jks"), ("keystore.password", "a"), ("keystore.key_password", "b"), ("truststore.path", "trust.pfx")]).unwrap();
    assert_eq!(c.key, Some(KeyMaterial::Keystore { path: data().join("node.jks"), kind: StoreType::Jks, password: "a".into(), key_password: "b".into() }));
    assert_eq!(c.trust, TrustMaterial::Truststore { path: data().join("trust.pfx"), kind: StoreType::Pkcs12, password: String::new() });
}

#[test]
fn pkcs12_keystores_load_the_key_and_its_chain() {
    let want_key = fingerprint(&node_key());
    for store in ["node-aes.p12", "node-3des.p12"] {
        let c = parse(HTTP_SSL, &[("keystore.path", store), ("keystore.password", "testnode")]).unwrap();
        let (chain, key) = c.load_key().unwrap().unwrap();
        assert_eq!(fingerprint(&key), want_key, "{store}");
        assert_eq!(chain, vec![cert("node.crt"), cert("ca.crt")], "{store}: leaf first, then its issuer");
        let e = parse(HTTP_SSL, &[("keystore.path", store), ("keystore.password", "wrong")]).unwrap().load_key().unwrap_err();
        assert!(e.starts_with("failed to load SSL configuration [xpack.security.http.ssl] - cannot read configured [PKCS12] keystore ["), "{e}");
        assert!(e.ends_with("this is usually caused by an incorrect password"), "{e}");
    }
    let e = parse(HTTP_SSL, &[("keystore.path", "missing.p12")]).unwrap().load_key().unwrap_err();
    assert!(e.ends_with("missing.p12] because the file does not exist"), "{e}");
    let e = parse(HTTP_SSL, &[("keystore.path", "node.jks")]).unwrap().load_key().unwrap_err();
    assert!(e.contains("JKS stores are not supported"), "{e}");
}

#[test]
fn pkcs12_truststore_and_pem_cas_are_trust_anchors() {
    let roots = parse(REINDEX_SSL, &[("truststore.path", "truststore.p12"), ("truststore.password", "changeit")]).unwrap().load_trust(None).unwrap();
    assert_eq!(roots.len(), 1);
    let roots = parse(REINDEX_SSL, &[("certificate_authorities", "ca.crt")]).unwrap().load_trust(None).unwrap();
    assert_eq!(roots.len(), 1);
    let roots = parse(REINDEX_SSL, &[]).unwrap().load_trust(None).unwrap();
    assert!(roots.len() > 50, "default trust is the Mozilla root set");
    let e = parse(REINDEX_SSL, &[("truststore.path", "truststore.p12"), ("truststore.password", "nope")]).unwrap().load_trust(None).unwrap_err();
    assert!(e.contains("incorrect password"), "{e}");
}

#[test]
fn pem_and_keystore_servers_handshake_with_verifying_clients() {
    let pem = http(&[("enabled", "true"), ("certificate", "node.crt"), ("key", "node.key")]).unwrap().unwrap();
    assert_eq!(pem.alpn_protocols, vec![b"http/1.1".to_vec()]);
    let p12 = http(&[("enabled", "true"), ("keystore.path", "node-3des.p12"), ("keystore.secure_password", "testnode")]).unwrap().unwrap();
    let trusting = reindex(&[("certificate_authorities", "ca.crt")]).unwrap();
    for server in [pem, p12] {
        handshake(trusting.clone(), server.clone(), "localhost").unwrap();
        handshake(trusting.clone(), server.clone(), "127.0.0.1").unwrap();
        // `full` checks the name, `certificate` does not, `none` checks nothing.
        assert!(handshake(trusting.clone(), server.clone(), "elsewhere.example").is_err());
        handshake(reindex(&[("certificate_authorities", "ca.crt"), ("verification_mode", "certificate")]).unwrap(), server.clone(), "elsewhere.example").unwrap();
        // An untrusted chain fails `full` (the default roots don't know this CA).
        let e = handshake(reindex(&[]).unwrap(), server.clone(), "localhost").unwrap_err();
        assert!(matches!(e, rustls::Error::InvalidCertificate(_)), "{e:?}");
        handshake(reindex(&[("verification_mode", "none")]).unwrap(), server.clone(), "localhost").unwrap();
        let truststore = reindex(&[("truststore.path", "truststore.p12"), ("truststore.secure_password", "changeit")]).unwrap();
        handshake(truststore, server, "localhost").unwrap();
    }
}

#[test]
fn required_client_authentication_needs_a_trusted_client_certificate() {
    let server = http(&[("enabled", "true"), ("certificate", "node.crt"), ("key", "node.key"), ("certificate_authorities", "ca.crt"), ("client_authentication", "required")]).unwrap().unwrap();
    let anonymous = reindex(&[("certificate_authorities", "ca.crt")]).unwrap();
    assert!(handshake(anonymous, server.clone(), "localhost").is_err());
    let with_cert = reindex(&[("certificate_authorities", "ca.crt"), ("keystore.path", "node-aes.p12"), ("keystore.password", "testnode")]).unwrap();
    handshake(with_cert, server, "localhost").unwrap();
}

#[test]
fn encrypted_pem_keys_need_their_passphrase() {
    let ok = http(&[("enabled", "true"), ("certificate", "testnode.crt"), ("key", "testnode.pem"), ("key_passphrase", "testnode")]);
    assert!(ok.unwrap().is_some());
    let e = http(&[("enabled", "true"), ("certificate", "testnode.crt"), ("key", "testnode.pem")]).unwrap_err();
    assert!(e.contains("without a password"), "{e}");
}

/// The certificate a server config presents (a handshake with a
/// non-verifying client).
fn presented(server: Arc<ServerConfig>) -> CertificateDer<'static> {
    let client = reindex(&[("verification_mode", "none")]).unwrap();
    let mut c = rustls::ClientConnection::new(client, ServerName::try_from("localhost").unwrap()).unwrap();
    let mut s = rustls::ServerConnection::new(server).unwrap();
    for _ in 0..16 {
        let mut buf = Vec::new();
        c.write_tls(&mut buf).unwrap();
        if !buf.is_empty() {
            s.read_tls(&mut &buf[..]).unwrap();
            s.process_new_packets().unwrap();
        }
        let mut buf = Vec::new();
        s.write_tls(&mut buf).unwrap();
        if !buf.is_empty() {
            c.read_tls(&mut &buf[..]).unwrap();
            c.process_new_packets().unwrap();
        }
        if !c.is_handshaking() {
            break;
        }
    }
    c.peer_certificates().expect("server certificate")[0].clone().into_owned()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-ssl-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn reloadable_contexts_follow_their_files_and_keep_the_old_one_on_failure() {
    let dir = scratch("reload");
    let (crt, key) = (dir.join("http.crt"), dir.join("http.key"));
    std::fs::copy(data().join("node.crt"), &crt).unwrap();
    std::fs::copy(data().join("node.key"), &key).unwrap();
    let m = settings(HTTP_SSL, &[("enabled", "true"), ("certificate", "http.crt"), ("key", "http.key")]);
    let ctx = http_server_reloadable(&|k| m.get(k).cloned(), &|_| Vec::new(), &dir).unwrap().unwrap();
    assert_eq!(ctx.files(), &[crt.clone(), key.clone()]);
    assert_eq!(Reload::context(&*ctx), "xpack.security.http.ssl");
    let first = ctx.load();
    assert_eq!(first.alpn_protocols, vec![b"http/1.1".to_vec()]);
    assert_eq!(presented(first.clone()), cert("node.crt"));
    assert_eq!(ctx.check(), ReloadOutcome::Unchanged);
    assert!(Arc::ptr_eq(&first, &ctx.load()));

    // Rotate to another key pair: the next check swaps it in.
    std::fs::copy(data().join("testnode.crt"), &crt).unwrap();
    std::fs::copy(data().join("testnode-unprotected.pem"), &key).unwrap();
    assert_eq!(ctx.check(), ReloadOutcome::Reloaded);
    let second = ctx.load();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(presented(second.clone()), cert("testnode.crt"));
    assert_eq!(second.alpn_protocols, vec![b"http/1.1".to_vec()]);
    // The old config still works for whoever holds it (established connections).
    assert_eq!(presented(first), cert("node.crt"));
    assert_eq!(ctx.check(), ReloadOutcome::Unchanged);

    // A broken certificate fails the rebuild and keeps the current context,
    // without retrying until the next change.
    std::fs::write(&crt, b"not a certificate").unwrap();
    match ctx.check() {
        ReloadOutcome::Failed(e) => assert!(e.starts_with("failed to load SSL configuration [xpack.security.http.ssl] - cannot load PEM certificate ["), "{e}"),
        other => panic!("{other:?}"),
    }
    assert!(Arc::ptr_eq(&second, &ctx.load()));
    assert_eq!(ctx.check(), ReloadOutcome::Unchanged);
    // A deleted file is not a reload (ES reloads on creation and change).
    std::fs::remove_file(&crt).unwrap();
    assert_eq!(ctx.check(), ReloadOutcome::Unchanged);
    // Fixing the files applies on the next change.
    std::fs::copy(data().join("node.key"), &key).unwrap();
    std::fs::copy(data().join("node.crt"), &crt).unwrap();
    assert_eq!(ctx.check(), ReloadOutcome::Reloaded);
    assert_eq!(presented(ctx.load()), cert("node.crt"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reloadable_client_contexts_follow_their_trust_files() {
    let dir = scratch("reload-client");
    let ca = dir.join("ca.crt");
    std::fs::copy(data().join("testnode.crt"), &ca).unwrap();
    let m = settings(REINDEX_SSL, &[("certificate_authorities", "ca.crt")]);
    let ctx = reindex_client_reloadable(&|k| m.get(k).cloned(), &|k| m.get(k).map(|v| vec![v.clone()]).unwrap_or_default(), &dir).unwrap();
    assert_eq!(ctx.files(), &[ca.clone()]);
    let server = http(&[("enabled", "true"), ("certificate", "node.crt"), ("key", "node.key")]).unwrap().unwrap();
    assert!(handshake(ctx.load(), server.clone(), "localhost").is_err(), "the node CA is not trusted yet");
    std::fs::copy(data().join("ca.crt"), &ca).unwrap();
    assert_eq!(ctx.check(), ReloadOutcome::Reloaded);
    handshake(ctx.load(), server, "localhost").unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fixed_contexts_never_reload() {
    let server = http(&[("enabled", "true"), ("certificate", "node.crt"), ("key", "node.key")]).unwrap().unwrap();
    let ctx = Reloadable::fixed("xpack.security.http.ssl", server.clone());
    assert_eq!(ctx.check(), ReloadOutcome::Unchanged);
    assert!(Arc::ptr_eq(&server, &ctx.load()));
}

#[test]
fn reload_interval_follows_the_resource_watcher_settings() {
    let ms = |s: &str| -> Option<i64> {
        let s = s.trim();
        if let Some(n) = s.strip_suffix("ms") {
            return n.parse().ok();
        }
        s.strip_suffix('s').and_then(|n| n.parse::<i64>().ok()).map(|n| n * 1000)
    };
    let with = |pairs: &[(&str, &str)]| {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        reload_interval(&|k| m.get(k).cloned(), &ms)
    };
    assert_eq!(with(&[]).unwrap(), Some(std::time::Duration::from_secs(5)));
    assert_eq!(with(&[("resource.reload.interval.high", "250ms")]).unwrap(), Some(std::time::Duration::from_millis(250)));
    assert_eq!(with(&[("resource.reload.enabled", "false"), ("resource.reload.interval.high", "1s")]).unwrap(), None);
    assert!(with(&[("resource.reload.enabled", "nope")]).unwrap_err().contains("only [true] or [false]"));
    assert!(with(&[("resource.reload.interval.high", "-5s")]).unwrap_err().contains("negative durations are not supported"));
    assert!(with(&[("resource.reload.interval.high", "5x")]).unwrap_err().contains("unit is missing or unrecognized"));
}
