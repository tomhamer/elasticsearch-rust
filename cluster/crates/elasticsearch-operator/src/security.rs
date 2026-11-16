//! Security of the clusters the operator manages, provisioned the way ECK
//! does it: every cluster runs with `xpack.security.enabled`, HTTPS on the
//! REST port and TLS on the transport port, and the `elastic` superuser has
//! a generated password. The operator itself talks to Elasticsearch over
//! HTTPS, trusting only the cluster's HTTP CA, as `elastic`.
//!
//! Secrets per cluster `<c>` (ECK's names, so tooling written for ECK finds
//! them):
//!
//! | Secret | keys | |
//! |---|---|---|
//! | `<c>-es-elastic-user` | `elastic` | the `elastic` password (the node's bootstrap password, `ELASTIC_PASSWORD`) |
//! | `<c>-es-http-ca-internal` | `tls.crt`, `tls.key` | self-signed HTTP CA |
//! | `<c>-es-http-certs-internal` | `tls.crt`, `tls.key`, `ca.crt` | HTTP certificate mounted into the pods |
//! | `<c>-es-http-certs-public` | `ca.crt`, `tls.crt` | what clients need to trust the cluster |
//! | `<c>-es-transport-ca-internal` | `tls.crt`, `tls.key` | self-signed transport CA |
//! | `<c>-es-transport-certs` | `tls.crt`, `tls.key`, `ca.crt` | transport certificate shared by the nodes |
//!
//! Deviation from ECK: one transport certificate for all nodes (ECK issues
//! one per pod). Transport TLS uses `verification_mode: certificate`
//! (chain only, as ECK and Elastic's Helm chart), so names are not checked.
//!
//! `spec.http.tls.certificate.secretName` (ECK's field) replaces the
//! generated HTTP certificate with the user's (`tls.crt`, `tls.key`,
//! optional `ca.crt`); its certificate must name `<c>-es-http.<ns>.svc`.
//! `spec.http.tls.selfSignedCertificate.subjectAltNames` adds names to the
//! generated one. Plain-text HTTP (`selfSignedCertificate.disabled`) is not
//! supported.
//!
//! Lifetimes: CAs 5 years, certificates 1 year; certificates are re-issued
//! 30 days before they expire (or when their names change), CAs 30 days
//! before theirs, keeping the previous CA in the trust bundle until it
//! expires, so nodes holding old and new certificates trust each other
//! during the rollover. The nodes reload changed certificate files without
//! a restart.

use crate::crd::{Elasticsearch, ElasticsearchSpec};
use crate::names;
use chrono::{DateTime, Duration, Utc};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::net::IpAddr;

pub const ELASTIC_USER: &str = "elastic";
pub const ELASTIC_USER_KEY: &str = "elastic";
pub const TLS_CRT: &str = "tls.crt";
pub const TLS_KEY: &str = "tls.key";
pub const CA_CRT: &str = "ca.crt";
/// Previous CA certificate kept in a CA Secret during a CA rollover.
pub const PREVIOUS_CA_CRT: &str = "previous.crt";

pub const HTTP_CERTS_DIR: &str = "/usr/share/elasticsearch/certs/http";
pub const TRANSPORT_CERTS_DIR: &str = "/usr/share/elasticsearch/certs/transport";
pub const HTTP_CERTS_VOLUME: &str = "elastic-internal-http-certificates";
pub const TRANSPORT_CERTS_VOLUME: &str = "elastic-internal-transport-certificates";

/// Expiry (RFC 3339) of the certificate in a Secret.
pub const ANNOTATION_NOT_AFTER: &str = "elasticsearch.k8s.elastic.co/not-after";
/// Hash of what a certificate was issued for (names + CA + trust bundle).
pub const ANNOTATION_ISSUED_FOR: &str = "elasticsearch.k8s.elastic.co/issued-for";

pub const CA_DAYS: i64 = 5 * 365;
pub const CERT_DAYS: i64 = 365;
pub const RENEW_BEFORE_DAYS: i64 = 30;
pub const PASSWORD_LEN: usize = 24;

pub fn elastic_user_secret(cluster: &str) -> String {
    format!("{cluster}-es-elastic-user")
}
pub fn http_ca_secret(cluster: &str) -> String {
    format!("{cluster}-es-http-ca-internal")
}
pub fn http_certs_internal_secret(cluster: &str) -> String {
    format!("{cluster}-es-http-certs-internal")
}
pub fn http_certs_public_secret(cluster: &str) -> String {
    format!("{cluster}-es-http-certs-public")
}
pub fn transport_ca_secret(cluster: &str) -> String {
    format!("{cluster}-es-transport-ca-internal")
}
pub fn transport_certs_secret(cluster: &str) -> String {
    format!("{cluster}-es-transport-certs")
}

/// `<c>-es-http.<ns>.svc`: the name the operator, the readiness probe and
/// in-cluster clients use, so every HTTP certificate must carry it.
pub fn http_service_host(cluster: &str, namespace: &str) -> String {
    format!("{}.{namespace}.svc", names::http_service(cluster))
}

pub fn http_url(cluster: &str, namespace: &str) -> String {
    format!("https://{}:{}", http_service_host(cluster, namespace), names::HTTP_PORT)
}

/// Decoded content of a Secret (data + annotations) as the planners see it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SecretData {
    pub data: BTreeMap<String, Vec<u8>>,
    pub annotations: BTreeMap<String, String>,
}

impl SecretData {
    pub fn get(&self, k: &str) -> Option<&[u8]> {
        self.data.get(k).map(Vec::as_slice).filter(|v| !v.is_empty())
    }
    fn get_str(&self, k: &str) -> Option<&str> {
        self.get(k).and_then(|v| std::str::from_utf8(v).ok())
    }
    fn not_after(&self) -> Option<DateTime<Utc>> {
        self.annotations.get(ANNOTATION_NOT_AFTER).and_then(|s| DateTime::parse_from_rfc3339(s).ok()).map(|d| d.with_timezone(&Utc))
    }
}

/// How the operator connects to a cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct EsConnection {
    pub url: String,
    /// PEM CA bundle to trust (only these CAs). `None`: the platform roots
    /// (a user-provided HTTP certificate from a public CA).
    pub ca_pem: Option<Vec<u8>>,
    pub username: String,
    pub password: String,
}

// ---------------------------------------------------------------- password

pub fn generate_password() -> String {
    use rand::Rng;
    rand::thread_rng().sample_iter(&rand::distributions::Alphanumeric).take(PASSWORD_LEN).map(char::from).collect()
}

/// The `elastic` user Secret: `None` when the existing one is usable.
pub fn plan_elastic_user(existing: Option<&SecretData>) -> Option<SecretData> {
    if existing.and_then(|s| s.get(ELASTIC_USER_KEY)).is_some() {
        return None;
    }
    Some(SecretData { data: BTreeMap::from([(ELASTIC_USER_KEY.to_string(), generate_password().into_bytes())]), annotations: BTreeMap::new() })
}

// ---------------------------------------------------------------- certificates

fn ymd(t: DateTime<Utc>) -> (i32, u8, u8) {
    use chrono::Datelike;
    (t.year(), t.month() as u8, t.day() as u8)
}

fn params(cn: &str, now: DateTime<Utc>, days: i64) -> (rcgen::CertificateParams, DateTime<Utc>) {
    let mut p = rcgen::CertificateParams::default();
    let mut dn = rcgen::DistinguishedName::new();
    dn.push(rcgen::DnType::CommonName, cn.to_string());
    p.distinguished_name = dn;
    // Back-dated a day against clock skew between operator and nodes.
    let (y, m, d) = ymd(now - Duration::days(1));
    p.not_before = rcgen::date_time_ymd(y, m, d);
    let end = now + Duration::days(days);
    let (y, m, d) = ymd(end);
    p.not_after = rcgen::date_time_ymd(y, m, d);
    let end_day = DateTime::parse_from_rfc3339(&format!("{y:04}-{m:02}-{d:02}T00:00:00Z")).map(|t| t.with_timezone(&Utc)).unwrap_or(end);
    (p, end_day)
}

fn ca_params(cn: &str, now: DateTime<Utc>) -> (rcgen::CertificateParams, DateTime<Utc>) {
    use rcgen::{BasicConstraints, IsCa, KeyUsagePurpose as Ku};
    let (mut p, end) = params(cn, now, CA_DAYS);
    p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    p.key_usages = vec![Ku::KeyCertSign, Ku::CrlSign, Ku::DigitalSignature];
    (p, end)
}

fn renew_due(not_after: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
    not_after.is_none_or(|t| t - now < Duration::days(RENEW_BEFORE_DAYS))
}

/// The CA Secret (`tls.crt`, `tls.key`, optionally `previous.crt`): `None`
/// when the existing CA is valid for more than RENEW_BEFORE_DAYS.
pub fn plan_ca(existing: Option<&SecretData>, cn: &str, now: DateTime<Utc>) -> Result<Option<SecretData>, String> {
    let usable = existing.filter(|s| s.get(TLS_CRT).is_some() && s.get_str(TLS_KEY).is_some_and(|k| rcgen::KeyPair::from_pem(k).is_ok()));
    if usable.is_some_and(|s| !renew_due(s.not_after(), now)) {
        return Ok(None);
    }
    let key = rcgen::KeyPair::generate().map_err(|e| e.to_string())?;
    let (p, end) = ca_params(cn, now);
    let cert = p.self_signed(&key).map_err(|e| e.to_string())?;
    let mut data = BTreeMap::from([(TLS_CRT.to_string(), cert.pem().into_bytes()), (TLS_KEY.to_string(), key.serialize_pem().into_bytes())]);
    // Rollover: keep trusting the old CA while it is still valid.
    if let Some(old) = usable.filter(|s| s.not_after().is_some_and(|t| t > now)) {
        data.insert(PREVIOUS_CA_CRT.to_string(), old.get(TLS_CRT).unwrap_or_default().to_vec());
    }
    Ok(Some(SecretData { data, annotations: BTreeMap::from([(ANNOTATION_NOT_AFTER.to_string(), end.to_rfc3339())]) }))
}

/// PEM bundle of the CAs to trust: the current CA, then the previous one.
pub fn trust_bundle(ca: &SecretData) -> Vec<u8> {
    let mut out = ca.get(TLS_CRT).unwrap_or_default().to_vec();
    if let Some(prev) = ca.get(PREVIOUS_CA_CRT) {
        if !out.ends_with(b"\n") {
            out.push(b'\n');
        }
        out.extend_from_slice(prev);
    }
    out
}

/// What a certificate is issued for.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafSpec {
    pub cn: String,
    pub dns: Vec<String>,
    pub ips: Vec<IpAddr>,
}

fn issued_for(spec: &LeafSpec, ca: &SecretData) -> String {
    let mut h = Sha256::new();
    h.update(spec.cn.as_bytes());
    for d in &spec.dns {
        h.update(b"\0dns:");
        h.update(d.as_bytes());
    }
    for ip in &spec.ips {
        h.update(b"\0ip:");
        h.update(ip.to_string().as_bytes());
    }
    h.update(b"\0ca:");
    h.update(trust_bundle(ca));
    h.finalize().iter().take(12).map(|b| format!("{b:02x}")).collect()
}

/// A certificate Secret (`tls.crt`, `tls.key`, `ca.crt`) signed by `ca`:
/// `None` when the existing one was issued for the same names and CA and is
/// not due for renewal.
pub fn plan_leaf(existing: Option<&SecretData>, ca: &SecretData, ca_cn: &str, spec: &LeafSpec, now: DateTime<Utc>) -> Result<Option<SecretData>, String> {
    use rcgen::{ExtendedKeyUsagePurpose as Eku, KeyUsagePurpose as Ku, SanType};
    let fingerprint = issued_for(spec, ca);
    if let Some(s) = existing {
        let complete = s.get(TLS_CRT).is_some() && s.get(TLS_KEY).is_some() && s.get(CA_CRT).is_some();
        if complete && s.annotations.get(ANNOTATION_ISSUED_FOR) == Some(&fingerprint) && !renew_due(s.not_after(), now) {
            return Ok(None);
        }
    }
    let ca_key = rcgen::KeyPair::from_pem(ca.get_str(TLS_KEY).ok_or("CA secret has no tls.key")?).map_err(|e| format!("CA key: {e}"))?;
    // The issuer is rebuilt from its subject and key: `signed_by` takes the
    // issuer name and key identifier from these, which match the stored CA.
    let issuer = ca_params(ca_cn, now).0.self_signed(&ca_key).map_err(|e| e.to_string())?;
    let key = rcgen::KeyPair::generate().map_err(|e| e.to_string())?;
    let (mut p, end) = params(&spec.cn, now, CERT_DAYS);
    let mut sans = Vec::new();
    for d in &spec.dns {
        sans.push(SanType::DnsName(d.clone().try_into().map_err(|e| format!("DNS name [{d}]: {e}"))?));
    }
    sans.extend(spec.ips.iter().map(|ip| SanType::IpAddress(*ip)));
    p.subject_alt_names = sans;
    p.key_usages = vec![Ku::DigitalSignature, Ku::KeyEncipherment];
    p.extended_key_usages = vec![Eku::ServerAuth, Eku::ClientAuth];
    let cert = p.signed_by(&key, &issuer, &ca_key).map_err(|e| e.to_string())?;
    Ok(Some(SecretData {
        data: BTreeMap::from([
            (TLS_CRT.to_string(), cert.pem().into_bytes()),
            (TLS_KEY.to_string(), key.serialize_pem().into_bytes()),
            (CA_CRT.to_string(), trust_bundle(ca)),
        ]),
        annotations: BTreeMap::from([(ANNOTATION_NOT_AFTER.to_string(), end.to_rfc3339()), (ANNOTATION_ISSUED_FOR.to_string(), fingerprint)]),
    }))
}

pub fn http_ca_cn(cluster: &str) -> String {
    format!("{cluster}-http")
}
pub fn transport_ca_cn(cluster: &str) -> String {
    format!("{cluster}-transport")
}

/// Names of the generated HTTP certificate: the HTTP Service (short and
/// qualified), every pod through its StatefulSet's headless Service,
/// localhost, and `spec.http.tls.selfSignedCertificate.subjectAltNames`.
pub fn http_leaf_spec(cluster: &str, namespace: &str, spec: &ElasticsearchSpec) -> LeafSpec {
    let svc = names::http_service(cluster);
    let mut dns = vec![
        svc.clone(),
        format!("{svc}.{namespace}"),
        format!("{svc}.{namespace}.svc"),
        format!("{svc}.{namespace}.svc.cluster.local"),
        "localhost".to_string(),
    ];
    for ns in &spec.node_sets {
        let sts = names::statefulset(cluster, &ns.name);
        dns.push(format!("*.{sts}.{namespace}.svc"));
        dns.push(format!("*.{sts}.{namespace}.svc.cluster.local"));
    }
    let mut ips: Vec<IpAddr> = vec![IpAddr::from([127, 0, 0, 1]), IpAddr::from(std::net::Ipv6Addr::LOCALHOST)];
    for san in spec.http.as_ref().and_then(|h| h.tls.as_ref()).and_then(|t| t.self_signed_certificate.as_ref()).map(|s| s.subject_alt_names.as_slice()).unwrap_or_default() {
        if let Some(d) = san.dns.as_ref().filter(|d| !d.is_empty()) {
            dns.push(d.clone());
        }
        if let Some(ip) = san.ip.as_ref().and_then(|ip| ip.parse::<IpAddr>().ok()) {
            ips.push(ip);
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    dns.retain(|d| seen.insert(d.clone()));
    let mut seen = std::collections::BTreeSet::new();
    ips.retain(|ip| seen.insert(*ip));
    LeafSpec { cn: http_service_host(cluster, namespace), dns, ips }
}

pub fn transport_leaf_spec(cluster: &str, namespace: &str) -> LeafSpec {
    let svc = names::transport_service(cluster);
    LeafSpec { cn: format!("{cluster}-transport"), dns: vec![format!("{svc}.{namespace}.svc"), format!("*.{svc}.{namespace}.svc")], ips: vec![] }
}

/// The user-provided HTTP certificate Secret, if `spec.http.tls.certificate.secretName` is set.
pub fn custom_http_secret(es: &Elasticsearch) -> Option<String> {
    es.spec.http.as_ref().and_then(|h| h.tls.as_ref()).and_then(|t| t.certificate.as_ref()).map(|c| c.secret_name.clone()).filter(|s| !s.is_empty())
}

/// The internal HTTP certificate Secret built from the user's
/// (`tls.crt`, `tls.key`, optional `ca.crt`): `None` when unchanged.
pub fn plan_custom_http(user: &SecretData, existing: Option<&SecretData>, secret_name: &str) -> Result<Option<SecretData>, String> {
    let crt = user.get(TLS_CRT).ok_or_else(|| format!("secret [{secret_name}] has no tls.crt"))?;
    let key = user.get(TLS_KEY).ok_or_else(|| format!("secret [{secret_name}] has no tls.key"))?;
    let mut data = BTreeMap::from([(TLS_CRT.to_string(), crt.to_vec()), (TLS_KEY.to_string(), key.to_vec())]);
    if let Some(ca) = user.get(CA_CRT) {
        data.insert(CA_CRT.to_string(), ca.to_vec());
    }
    if existing.is_some_and(|e| e.data == data) {
        return Ok(None);
    }
    Ok(Some(SecretData { data, annotations: BTreeMap::new() }))
}

/// The public HTTP Secret (`ca.crt`, `tls.crt`) from the internal one: `None` when unchanged.
pub fn plan_http_public(internal: &SecretData, existing: Option<&SecretData>) -> Option<SecretData> {
    let mut data = BTreeMap::new();
    for k in [CA_CRT, TLS_CRT] {
        if let Some(v) = internal.get(k) {
            data.insert(k.to_string(), v.to_vec());
        }
    }
    if existing.is_some_and(|e| e.data == data) {
        return None;
    }
    Some(SecretData { data, annotations: BTreeMap::new() })
}

/// Settings the operator sets for security (the files are mounted from the
/// Secrets above). Users cannot override them.
pub fn security_settings() -> Vec<(&'static str, serde_json::Value)> {
    use serde_json::{json, Value};
    let file = |dir: &str, f: &str| Value::String(format!("{dir}/{f}"));
    vec![
        ("xpack.security.enabled", json!("true")),
        ("xpack.security.http.ssl.enabled", json!("true")),
        ("xpack.security.http.ssl.key", file(HTTP_CERTS_DIR, TLS_KEY)),
        ("xpack.security.http.ssl.certificate", file(HTTP_CERTS_DIR, TLS_CRT)),
        ("xpack.security.transport.ssl.enabled", json!("true")),
        ("xpack.security.transport.ssl.verification_mode", json!("certificate")),
        ("xpack.security.transport.ssl.key", file(TRANSPORT_CERTS_DIR, TLS_KEY)),
        ("xpack.security.transport.ssl.certificate", file(TRANSPORT_CERTS_DIR, TLS_CRT)),
        ("xpack.security.transport.ssl.certificate_authorities", json!([format!("{TRANSPORT_CERTS_DIR}/{CA_CRT}")])),
    ]
}

/// Security settings users may not set in `nodeSets[].config` (the ones
/// above plus the keystore/truststore alternatives to them).
pub const MANAGED_SECURITY_SETTINGS: &[&str] = &[
    "xpack.security.enabled",
    "xpack.security.http.ssl.enabled",
    "xpack.security.http.ssl.key",
    "xpack.security.http.ssl.certificate",
    "xpack.security.http.ssl.keystore.path",
    "xpack.security.transport.ssl.enabled",
    "xpack.security.transport.ssl.verification_mode",
    "xpack.security.transport.ssl.key",
    "xpack.security.transport.ssl.certificate",
    "xpack.security.transport.ssl.certificate_authorities",
    "xpack.security.transport.ssl.keystore.path",
    "xpack.security.transport.ssl.truststore.path",
];

/// Spec errors about security (ECK fields this operator rejects).
pub fn validate(spec: &ElasticsearchSpec) -> Vec<String> {
    let mut errs = Vec::new();
    if let Some(tls) = spec.http.as_ref().and_then(|h| h.tls.as_ref()) {
        if tls.self_signed_certificate.as_ref().is_some_and(|s| s.disabled) {
            errs.push("spec.http.tls.selfSignedCertificate.disabled: plain-text HTTP is not supported; the operator always serves HTTPS (use spec.http.tls.certificate.secretName for your own certificate)".into());
        }
        if tls.certificate.as_ref().is_some_and(|c| c.secret_name.is_empty()) {
            errs.push("spec.http.tls.certificate.secretName must not be empty".into());
        }
        for san in tls.self_signed_certificate.as_ref().map(|s| s.subject_alt_names.as_slice()).unwrap_or_default() {
            if let Some(ip) = &san.ip {
                if ip.parse::<IpAddr>().is_err() {
                    errs.push(format!("spec.http.tls.selfSignedCertificate.subjectAltNames: invalid IP [{ip}]"));
                }
            }
        }
    }
    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::{HttpConfig, NodeSet, SecretRef, SelfSignedCertificate, SubjectAlternativeName, TlsOptions};

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-11-08T12:00:00Z").unwrap().with_timezone(&Utc)
    }

    fn spec() -> ElasticsearchSpec {
        ElasticsearchSpec { version: "9.5.5".into(), node_sets: vec![NodeSet { name: "default".into(), count: 3, ..Default::default() }], ..Default::default() }
    }

    fn parse_cert(pem: &[u8]) -> Vec<u8> {
        let text = std::str::from_utf8(pem).unwrap();
        let b64: String = text.lines().skip_while(|l| !l.starts_with("-----BEGIN CERTIFICATE")).skip(1).take_while(|l| !l.starts_with("-----END")).collect();
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(b64).unwrap()
    }

    #[test]
    fn password_generated_once() {
        let s = plan_elastic_user(None).unwrap();
        let pw = std::str::from_utf8(s.get(ELASTIC_USER_KEY).unwrap()).unwrap().to_string();
        assert_eq!(pw.len(), PASSWORD_LEN);
        assert!(pw.chars().all(|c| c.is_ascii_alphanumeric()));
        assert!(plan_elastic_user(Some(&s)).is_none(), "an existing password is kept");
        assert_ne!(plan_elastic_user(None).unwrap(), s, "random");
        let empty = SecretData { data: BTreeMap::from([(ELASTIC_USER_KEY.to_string(), vec![])]), ..Default::default() };
        assert!(plan_elastic_user(Some(&empty)).is_some(), "an empty password is replaced");
    }

    #[test]
    fn ca_and_leaf_issue_verify_and_are_kept() {
        let ca = plan_ca(None, "c-http", now()).unwrap().unwrap();
        assert!(ca.get(TLS_KEY).is_some() && ca.get(PREVIOUS_CA_CRT).is_none());
        assert!(plan_ca(Some(&ca), "c-http", now()).unwrap().is_none(), "valid CA kept");
        let leaf_spec = http_leaf_spec("c", "ns", &spec());
        assert!(leaf_spec.dns.contains(&"c-es-http.ns.svc".to_string()));
        assert!(leaf_spec.dns.contains(&"*.c-es-default.ns.svc".to_string()));
        assert!(leaf_spec.ips.contains(&"127.0.0.1".parse().unwrap()));
        let leaf = plan_leaf(None, &ca, "c-http", &leaf_spec, now()).unwrap().unwrap();
        assert_eq!(leaf.get(CA_CRT).unwrap(), ca.get(TLS_CRT).unwrap());
        // The leaf chains to the stored CA (rustls/webpki path validation for the service name).
        let ca_der = parse_cert(ca.get(TLS_CRT).unwrap());
        let leaf_der = parse_cert(leaf.get(TLS_CRT).unwrap());
        let ca_cert = rustls_pki_types::CertificateDer::from(ca_der.as_slice());
        let leaf_cert = rustls_pki_types::CertificateDer::from(leaf_der.as_slice());
        let anchor = webpki::anchor_from_trusted_cert(&ca_cert).unwrap();
        let ee = webpki::EndEntityCert::try_from(&leaf_cert).unwrap();
        let time = rustls_pki_types::UnixTime::since_unix_epoch(std::time::Duration::from_secs(now().timestamp() as u64));
        ee.verify_for_usage(webpki::ALL_VERIFICATION_ALGS, &[anchor], &[], time, webpki::KeyUsage::server_auth(), None, None).expect("chains to the CA");
        ee.verify_is_valid_for_subject_name(&rustls_pki_types::ServerName::try_from("c-es-http.ns.svc").unwrap()).expect("names the service");
        ee.verify_is_valid_for_subject_name(&rustls_pki_types::ServerName::try_from("c-es-default-1.c-es-default.ns.svc").unwrap()).expect("names the pods");
        // Same names, same CA, far from expiry: kept.
        assert!(plan_leaf(Some(&leaf), &ca, "c-http", &leaf_spec, now()).unwrap().is_none());
        // New names: re-issued.
        let mut more = spec();
        more.http = Some(HttpConfig { tls: Some(TlsOptions { self_signed_certificate: Some(SelfSignedCertificate { subject_alt_names: vec![SubjectAlternativeName { dns: Some("search.example.com".into()), ip: Some("10.0.0.7".into()) }], disabled: false }), certificate: None }) });
        let spec2 = http_leaf_spec("c", "ns", &more);
        assert!(spec2.dns.contains(&"search.example.com".to_string()) && spec2.ips.contains(&"10.0.0.7".parse().unwrap()));
        assert!(plan_leaf(Some(&leaf), &ca, "c-http", &spec2, now()).unwrap().is_some());
        // Close to expiry: re-issued.
        let later = now() + Duration::days(CERT_DAYS - RENEW_BEFORE_DAYS + 1);
        assert!(plan_leaf(Some(&leaf), &ca, "c-http", &leaf_spec, later).unwrap().is_some());
    }

    #[test]
    fn ca_rollover_keeps_the_previous_ca_trusted() {
        let ca = plan_ca(None, "c-transport", now()).unwrap().unwrap();
        let leaf_spec = transport_leaf_spec("c", "ns");
        let leaf = plan_leaf(None, &ca, "c-transport", &leaf_spec, now()).unwrap().unwrap();
        let late = now() + Duration::days(CA_DAYS - RENEW_BEFORE_DAYS + 1);
        let ca2 = plan_ca(Some(&ca), "c-transport", late).unwrap().expect("CA renewed");
        assert_ne!(ca2.get(TLS_CRT), ca.get(TLS_CRT));
        assert_eq!(ca2.get(PREVIOUS_CA_CRT), ca.get(TLS_CRT));
        let bundle = trust_bundle(&ca2);
        assert_eq!(String::from_utf8_lossy(&bundle).matches("BEGIN CERTIFICATE").count(), 2);
        // The certificates are re-issued from the new CA with both CAs trusted.
        let leaf2 = plan_leaf(Some(&leaf), &ca2, "c-transport", &leaf_spec, late).unwrap().expect("re-issued");
        assert_eq!(leaf2.get(CA_CRT).unwrap(), bundle.as_slice());
    }

    #[test]
    fn custom_and_public_http_secrets() {
        let user = SecretData { data: BTreeMap::from([(TLS_CRT.into(), b"C".to_vec()), (TLS_KEY.into(), b"K".to_vec())]), ..Default::default() };
        let internal = plan_custom_http(&user, None, "mine").unwrap().unwrap();
        assert!(internal.get(CA_CRT).is_none());
        assert!(plan_custom_http(&user, Some(&internal), "mine").unwrap().is_none());
        assert!(plan_custom_http(&SecretData::default(), None, "mine").unwrap_err().contains("no tls.crt"));
        let public = plan_http_public(&internal, None).unwrap();
        assert_eq!(public.data.keys().collect::<Vec<_>>(), vec![TLS_CRT]);
        assert!(plan_http_public(&internal, Some(&public)).is_none());
    }

    #[test]
    fn plain_http_is_rejected() {
        let mut s = spec();
        assert!(validate(&s).is_empty());
        s.http = Some(HttpConfig { tls: Some(TlsOptions { self_signed_certificate: Some(SelfSignedCertificate { subject_alt_names: vec![], disabled: true }), certificate: None }) });
        assert!(validate(&s)[0].contains("plain-text HTTP is not supported"));
        s.http = Some(HttpConfig { tls: Some(TlsOptions { self_signed_certificate: None, certificate: Some(SecretRef { secret_name: String::new() }) }) });
        assert!(validate(&s)[0].contains("must not be empty"));
    }
}
