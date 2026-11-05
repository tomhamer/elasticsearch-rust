//! Node and Kibana enrollment, as ES's security plugin implements it:
//! `EnrollmentToken` (the token format), `BaseEnrollmentTokenGenerator`
//! (address filtering, the HTTP CA fingerprint), `InternalEnrollmentTokenGenerator`
//! (tokens the first node prints), `TransportNodeEnrollmentAction` /
//! `TransportKibanaEnrollmentAction` (the material `GET /_security/enroll/*`
//! hands out) and `CommandLineHttpClient`'s CA-fingerprint pinning (the
//! enrolling node's client).
//!
//! A token is the URL-safe base64 of `{"ver","adr","fgr","key"}`: the token
//! version, the HTTP addresses of the node that created it, the hex SHA-256
//! of its HTTP CA certificate and an API key (`id:secret`) whose only
//! privilege is `cluster:admin/xpack/security/enroll/{node,kibana}`, valid
//! for 30 minutes.
//!
//! Key material: ES reads it from the PKCS#12 keystores its
//! auto-configuration writes (`StoreKeyConfig.getKeys()`); a keystore
//! configured here is read the same way. The PEM layout of rustic's
//! auto-configuration (see `security_autoconfig`) has no keystore: the HTTP
//! CA's key is `http_ca.key` beside `xpack.security.http.ssl.certificate`, and
//! the transport key is `xpack.security.transport.ssl.key` with its CA from
//! `certificate_authorities`. Keys travel as base64 PKCS#8 DER and
//! certificates as base64 DER, as in ES.

use crate::config::{get_list, Settings};
use crate::security_autoconfig::hex;
use base64::Engine;
use elasticsearch_engine::error::EsError;
use elasticsearch_ssl::rustls;
use rustls::client::danger::ServerCertVerifier;
use serde_json::{json, Map, Value};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// ES `EnrollmentToken.CURRENT_TOKEN_VERSION`.
pub const CURRENT_TOKEN_VERSION: &str = "8.14.0";
/// ES `NodeEnrollmentAction.NAME`.
pub const NODE_ENROLLMENT_ACTION: &str = "cluster:admin/xpack/security/enroll/node";
/// ES `KibanaEnrollmentAction.NAME`.
pub const KIBANA_ENROLLMENT_ACTION: &str = "cluster:admin/xpack/security/enroll/kibana";
/// ES `ExternalEnrollmentTokenGenerator.ENROLL_API_KEY_EXPIRATION`.
pub const ENROLL_API_KEY_EXPIRATION: &str = "30m";
/// ES `BaseEnrollmentTokenGenerator.ENROLL_API_KEY_EXPIRATION_MINUTES`.
pub const ENROLL_API_KEY_EXPIRATION_MINUTES: i64 = 30;
/// The auto-configured HTTP CA key, beside the HTTP certificate (PEM layout).
pub const HTTP_CA_KEY_FILE: &str = crate::security_autoconfig::HTTP_CA_KEY;

// ------------------------------------------------------------------ token

/// ES `EnrollmentToken`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentToken {
    /// `id:secret` of the enrollment API key.
    pub api_key: String,
    /// Hex SHA-256 of the HTTP CA certificate (DER).
    pub fingerprint: String,
    pub version: String,
    /// `host:port` HTTP addresses of the node that created the token.
    pub bound_address: Vec<String>,
}

impl EnrollmentToken {
    pub fn new(api_key: String, fingerprint: String, bound_address: Vec<String>) -> Self {
        EnrollmentToken { api_key, fingerprint, version: CURRENT_TOKEN_VERSION.to_string(), bound_address }
    }

    /// ES `getRaw`: compact JSON, fields `ver`, `adr`, `fgr`, `key` in this order.
    pub fn raw(&self) -> String {
        let mut m = Map::new();
        m.insert("ver".into(), json!(self.version));
        m.insert("adr".into(), json!(self.bound_address));
        m.insert("fgr".into(), json!(self.fingerprint));
        m.insert("key".into(), json!(self.api_key));
        Value::Object(m).to_string()
    }

    /// ES `getEncoded`: `Base64.getUrlEncoder()` (URL-safe alphabet, padded).
    pub fn encoded(&self) -> String {
        base64::engine::general_purpose::URL_SAFE.encode(self.raw())
    }

    /// ES `decodeFromString` (a strict object parser: every field required,
    /// no unknown field). ES decodes with the basic alphabet only, which
    /// rejects tokens whose URL-safe encoding contains `-` or `_`; both
    /// alphabets are accepted here.
    pub fn decode(encoded: &str) -> Result<EnrollmentToken, String> {
        if encoded.is_empty() {
            return Err("Cannot decode enrollment token from an empty string".into());
        }
        let bytes = decode_base64(encoded.trim())?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| format!("Failed to parse enrollment token: {e}"))?;
        let Value::Object(o) = value else {
            return Err("[enrollment_token] Expected START_OBJECT".into());
        };
        for k in o.keys() {
            if !matches!(k.as_str(), "key" | "fgr" | "ver" | "adr") {
                return Err(format!("[enrollment_token] unknown field [{k}]"));
            }
        }
        let text = |k: &str| -> Result<Option<String>, String> {
            match o.get(k) {
                None => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(other) => Err(format!("[enrollment_token] failed to parse field [{k}]: expected a string, got [{other}]")),
            }
        };
        let (key, fgr, ver) = (text("key")?, text("fgr")?, text("ver")?);
        let adr = match o.get("adr") {
            None => None,
            Some(Value::Array(a)) => Some(
                a.iter()
                    .map(|v| v.as_str().map(str::to_string).ok_or_else(|| "[enrollment_token] failed to parse field [adr]: expected strings".to_string()))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(Value::String(s)) => Some(vec![s.clone()]),
            Some(other) => return Err(format!("[enrollment_token] failed to parse field [adr]: expected an array, got [{other}]")),
        };
        let mut missing = Vec::new();
        for (name, present) in [("key", key.is_some()), ("fgr", fgr.is_some()), ("ver", ver.is_some()), ("adr", adr.is_some())] {
            if !present {
                missing.push(name);
            }
        }
        if !missing.is_empty() {
            return Err(format!("Required [{}]", missing.join(", ")));
        }
        Ok(EnrollmentToken { api_key: key.unwrap_or_default(), fingerprint: fgr.unwrap_or_default(), version: ver.unwrap_or_default(), bound_address: adr.unwrap_or_default() })
    }
}

fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    use base64::engine::{general_purpose::GeneralPurposeConfig, DecodePaddingMode, GeneralPurpose};
    let config = || GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent);
    let standard = GeneralPurpose::new(&base64::alphabet::STANDARD, config());
    let url_safe = GeneralPurpose::new(&base64::alphabet::URL_SAFE, config());
    standard.decode(s).or_else(|e| url_safe.decode(s).map_err(|_| format!("Illegal base64 enrollment token: {e}")))
}

// -------------------------------------------------------------- addresses

/// The IP of a `host:port` address (ES `getInetAddressFromString`: a host
/// name is resolved).
pub fn address_ip(address: &str) -> Option<IpAddr> {
    if let Ok(sa) = address.parse::<SocketAddr>() {
        return Some(sa.ip());
    }
    let host = match address.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !h.is_empty() => h,
        _ => address,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(ip);
    }
    (host, 0).to_socket_addrs().ok()?.next().map(|sa| sa.ip())
}

/// ES `splitAddresses`: (loopback, non-loopback) addresses, wildcard
/// addresses dropped, IPv4 before IPv6 (otherwise in order), deduplicated.
pub fn split_addresses(addresses: &[String]) -> (Vec<String>, Vec<String>) {
    let mut local = Vec::new();
    let mut non_local = Vec::new();
    for a in addresses {
        let Some(ip) = address_ip(a) else { continue };
        if ip.is_loopback() {
            local.push((ip.is_ipv6(), a.clone()));
        } else if !ip.is_unspecified() {
            non_local.push((ip.is_ipv6(), a.clone()));
        }
    }
    let finish = |mut v: Vec<(bool, String)>| -> Vec<String> {
        // A stable sort, as `List.sort`.
        v.sort_by_key(|(v6, _)| *v6);
        let mut out: Vec<String> = Vec::new();
        for (_, a) in v {
            if !out.contains(&a) {
                out.push(a);
            }
        }
        out
    };
    (finish(local), finish(non_local))
}

/// ES `getFilteredAddresses`: the non-loopback addresses, or the loopback
/// ones when there are none.
pub fn filtered_addresses(addresses: &[String]) -> Vec<String> {
    let (local, non_local) = split_addresses(addresses);
    if non_local.is_empty() { local } else { non_local }
}

/// ES `getIpFromPublishAddress`: `host/ip:port` -> `ip:port`.
pub fn ip_from_publish_address(publish: &str) -> String {
    match publish.split('/').nth(1) {
        Some(ip) if publish.contains('/') => ip.to_string(),
        _ => publish.to_string(),
    }
}

/// ES `AutoConfigureNode.anyRemoteHostNodeAddress`: some node of the cluster
/// publishes a non-loopback transport address that is not one of this
/// host's addresses.
pub fn any_remote_host_node_address(transport_addresses: &[String], host_addresses: &[IpAddr]) -> bool {
    transport_addresses.iter().filter_map(|a| address_ip(a)).any(|ip| !ip.is_loopback() && !host_addresses.contains(&ip))
}

// ------------------------------------------------------------- X.509 DER

/// One DER TLV: (tag, content, rest).
fn tlv(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = b.split_first()?;
    let (&l0, rest) = rest.split_first()?;
    let (len, rest) = if l0 < 0x80 {
        (l0 as usize, rest)
    } else {
        let n = (l0 & 0x7f) as usize;
        if n == 0 || n > 4 || rest.len() < n {
            return None;
        }
        let len = rest[..n].iter().fold(0usize, |acc, x| (acc << 8) | *x as usize);
        (len, &rest[n..])
    };
    if rest.len() < len {
        return None;
    }
    Some((tag, &rest[..len], &rest[len..]))
}

fn oid_arcs(b: &[u8]) -> Option<Vec<u64>> {
    let mut subs = Vec::new();
    let mut cur: u64 = 0;
    for &x in b {
        cur = (cur << 7) | u64::from(x & 0x7f);
        if x & 0x80 == 0 {
            subs.push(cur);
            cur = 0;
        }
    }
    let first = *subs.first()?;
    let (a, b) = if first < 40 { (0, first) } else if first < 80 { (1, first - 40) } else { (2, first - 80) };
    let mut out = vec![a, b];
    out.extend_from_slice(&subs[1..]);
    Some(out)
}

/// What enrollment needs to know of a certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertInfo {
    pub der: Vec<u8>,
    /// Subject attributes in order: (OID arcs, string tag, value bytes).
    pub subject: Vec<(Vec<u64>, u8, Vec<u8>)>,
    /// The subject `Name`, DER.
    pub subject_der: Vec<u8>,
    /// The subject public key (the SPKI bit string's bytes).
    pub public_key: Vec<u8>,
    /// `basicConstraints` with `cA` set (ES `getBasicConstraints() != -1`).
    pub is_ca: bool,
    pub subject_key_id: Option<Vec<u8>>,
}

/// Parse an X.509 certificate (DER).
pub fn parse_certificate(der: &[u8]) -> Option<CertInfo> {
    let (0x30, cert, _) = tlv(der)? else { return None };
    let (0x30, tbs, _) = tlv(cert)? else { return None };
    let mut r = tbs;
    if let Some((0xA0, _, rest)) = tlv(r) {
        r = rest; // [0] version
    }
    let (_, _, r) = tlv(r)?; // serialNumber
    let (_, _, r) = tlv(r)?; // signature
    let (_, _, r) = tlv(r)?; // issuer
    let (_, _, r) = tlv(r)?; // validity
    let subject_start = r;
    let (0x30, subject, r) = tlv(r)? else { return None };
    let subject_der = subject_start[..subject_start.len() - r.len()].to_vec();
    let (0x30, spki, mut r) = tlv(r)? else { return None };
    let (_, _, spki_rest) = tlv(spki)?;
    let (0x03, bits, _) = tlv(spki_rest)? else { return None };
    let public_key = bits.get(1..)?.to_vec();
    let mut attrs = Vec::new();
    let mut s = subject;
    while let Some((_, set, rest)) = tlv(s) {
        s = rest;
        let mut a = set;
        while let Some((_, atv, rest)) = tlv(a) {
            a = rest;
            let (_, oid, after) = tlv(atv)?;
            let (tag, value, _) = tlv(after)?;
            attrs.push((oid_arcs(oid)?, tag, value.to_vec()));
        }
    }
    let mut is_ca = false;
    let mut subject_key_id = None;
    while let Some((tag, content, rest)) = tlv(r) {
        r = rest;
        if tag != 0xA3 {
            continue;
        }
        let (_, exts, _) = tlv(content)?;
        let mut e = exts;
        while let Some((_, ext, rest)) = tlv(e) {
            e = rest;
            let (_, oid, after_oid) = tlv(ext)?;
            let (mut t, mut value, after) = tlv(after_oid)?;
            if t == 0x01 {
                // critical
                let (t2, v2, _) = tlv(after)?;
                t = t2;
                value = v2;
            }
            if t != 0x04 {
                continue;
            }
            match oid_arcs(oid)?.as_slice() {
                [2, 5, 29, 19] => {
                    if let Some((0x30, bc, _)) = tlv(value) {
                        if let Some((0x01, b, _)) = tlv(bc) {
                            is_ca = b.first().is_some_and(|x| *x != 0);
                        }
                    }
                }
                [2, 5, 29, 14] => {
                    if let Some((0x04, id, _)) = tlv(value) {
                        subject_key_id = Some(id.to_vec());
                    }
                }
                _ => {}
            }
        }
    }
    Some(CertInfo { der: der.to_vec(), subject: attrs, subject_der, public_key, is_ca, subject_key_id })
}

/// PEM text of one DER object.
pub fn pem_encode(label: &str, der: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(der);
    let mut out = format!("-----BEGIN {label}-----\n");
    for chunk in b64.as_bytes().chunks(64) {
        out.push_str(std::str::from_utf8(chunk).unwrap_or_default());
        out.push('\n');
    }
    out.push_str(&format!("-----END {label}-----\n"));
    out
}

/// An rcgen issuer for an existing CA certificate and its key: the same
/// subject (attribute encodings kept, so issued certificates chain to it by
/// name) and key identifier. Refused when the subject cannot be reproduced
/// byte for byte.
pub fn issuer_certificate(ca: &CertInfo, ca_key: &rcgen::KeyPair) -> Result<rcgen::Certificate, String> {
    if ca.public_key != ca_key.public_key_raw() {
        return Err("the CA private key does not match the CA certificate".into());
    }
    let mut dn = rcgen::DistinguishedName::new();
    for (oid, tag, value) in &ca.subject {
        let unsupported = || format!("unsupported CA subject attribute encoding [{tag}]");
        let text = String::from_utf8(value.clone()).map_err(|_| unsupported())?;
        let v = match tag {
            0x0C => rcgen::DnValue::Utf8String(text),
            0x13 => rcgen::DnValue::PrintableString(rcgen::PrintableString::try_from(text).map_err(|_| unsupported())?),
            0x16 => rcgen::DnValue::Ia5String(rcgen::Ia5String::try_from(text).map_err(|_| unsupported())?),
            0x14 => rcgen::DnValue::TeletexString(rcgen::TeletexString::try_from(text).map_err(|_| unsupported())?),
            _ => return Err(unsupported()),
        };
        dn.push(rcgen::DnType::from_oid(oid), v);
    }
    let mut p = rcgen::CertificateParams::default();
    p.distinguished_name = dn;
    p.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    if let Some(id) = &ca.subject_key_id {
        p.key_identifier_method = rcgen::KeyIdMethod::PreSpecified(id.clone());
    }
    let issuer = p.self_signed(ca_key).map_err(|e| e.to_string())?;
    let reproduced = parse_certificate(issuer.der().as_ref()).ok_or("cannot re-read the CA subject")?;
    if reproduced.subject_der != ca.subject_der {
        return Err("cannot reproduce the CA certificate's subject".into());
    }
    Ok(issuer)
}

// ------------------------------------------------------- the node's keys

/// A private key entry and its certificate (ES `getKeys()` element).
pub struct KeyEntry {
    pub key: rcgen::KeyPair,
    /// PKCS#8 DER (ES `PrivateKey.getEncoded()`).
    pub pkcs8: Vec<u8>,
    pub cert: CertInfo,
}

fn key_entry(key: &rustls::pki_types::PrivateKeyDer<'_>, cert: &[u8]) -> Option<KeyEntry> {
    let rustls::pki_types::PrivateKeyDer::Pkcs8(p) = key else { return None };
    let pkcs8 = p.secret_pkcs8_der().to_vec();
    let key = rcgen::KeyPair::try_from(pkcs8.as_slice()).ok()?;
    Some(KeyEntry { key, pkcs8, cert: parse_certificate(cert)? })
}

/// A setting as text (booleans and numbers too).
fn setting_text(settings: &Settings, key: &str) -> Option<String> {
    match settings.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        Value::Array(_) | Value::Object(_) => None,
        other => Some(other.to_string()),
    }
}

pub fn conf_dir(settings: &Settings) -> PathBuf {
    PathBuf::from(setting_text(settings, "path.conf").unwrap_or_else(|| "config".into()))
}

/// The key material of an SSL context: its key entries and every
/// configured certificate (`None`: no key material).
struct Material {
    entries: Vec<KeyEntry>,
    certificates: Vec<CertInfo>,
}

fn material(settings: &Settings, prefix: &str, ca_key_file: Option<&str>) -> Result<Option<Material>, String> {
    let get = |k: &str| setting_text(settings, k);
    let list = |k: &str| get_list(settings, k);
    let defaults = elasticsearch_ssl::Defaults { verification_mode: elasticsearch_ssl::VerificationMode::Full, client_authentication: elasticsearch_ssl::ClientAuth::None };
    let cfg = elasticsearch_ssl::SslConfig::parse(prefix, &get, &list, &conf_dir(settings), defaults)?;
    if let Some((keys, certs)) = cfg.keystore_contents()? {
        return Ok(Some(Material {
            entries: keys.iter().filter_map(|(k, c)| key_entry(k, c.as_ref())).collect(),
            certificates: certs.iter().filter_map(|c| parse_certificate(c.as_ref())).collect(),
        }));
    }
    let Some(elasticsearch_ssl::KeyMaterial::Pem { certificate, .. }) = cfg.key.clone() else { return Ok(None) };
    let Some((chain, key)) = cfg.load_key()? else { return Ok(None) };
    let mut certificates: Vec<CertInfo> = chain.iter().filter_map(|c| parse_certificate(c.as_ref())).collect();
    if let elasticsearch_ssl::TrustMaterial::Pem(files) = &cfg.trust {
        for f in files {
            let pem = std::fs::read(f).map_err(|e| format!("cannot read [{}]: {e}", f.display()))?;
            for c in elasticsearch_ssl::parse_certificates(&pem)? {
                certificates.extend(parse_certificate(c.as_ref()));
            }
        }
    }
    let mut entries: Vec<KeyEntry> = chain.first().and_then(|leaf| key_entry(&key, leaf.as_ref())).into_iter().collect();
    if let Some(name) = ca_key_file {
        let path = certificate.parent().unwrap_or(Path::new(".")).join(name);
        if let Ok(pem) = std::fs::read(&path) {
            let key = elasticsearch_ssl::parse_private_key(&pem, None, &path.display().to_string())?;
            if let rustls::pki_types::PrivateKeyDer::Pkcs8(p) = &key {
                let pkcs8 = p.secret_pkcs8_der().to_vec();
                if let Ok(kp) = rcgen::KeyPair::try_from(pkcs8.as_slice()) {
                    if let Some(cert) = certificates.iter().find(|c| c.public_key == kp.public_key_raw()) {
                        entries.push(KeyEntry { cert: cert.clone(), key: kp, pkcs8 });
                    }
                }
            }
        }
    }
    Ok(Some(Material { entries, certificates }))
}

/// Who asks for the HTTP CA: the messages differ.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CaUse {
    /// `TransportNodeEnrollmentAction`.
    NodeEnrollment,
    /// `BaseEnrollmentTokenGenerator.getHttpsCaFingerprint`.
    Token,
}

/// The HTTP CA key entry: the only HTTP key entry whose certificate is a CA.
pub fn http_ca(settings: &Settings, purpose: CaUse) -> Result<KeyEntry, String> {
    let prefix = match purpose {
        CaUse::NodeEnrollment => "Unable to enroll node.",
        CaUse::Token => "Unable to create an enrollment token.",
    };
    let Some(m) = material(settings, elasticsearch_ssl::HTTP_SSL, Some(HTTP_CA_KEY_FILE))? else {
        return Err(format!("{prefix} Elasticsearch node HTTP layer SSL configuration is not configured with a keystore"));
    };
    let mut cas: Vec<KeyEntry> = m.entries.into_iter().filter(|e| e.cert.is_ca).collect();
    match cas.len() {
        0 => Err(format!("{prefix} Elasticsearch node HTTP layer SSL configuration Keystore doesn't contain any PrivateKey entries where the associated certificate is a CA certificate")),
        1 => Ok(cas.remove(0)),
        _ if purpose == CaUse::NodeEnrollment => Err(format!("{prefix} Elasticsearch node HTTP layer SSL configuration Keystore contain multiple PrivateKey entries where the associated certificate is a CA certificate")),
        _ => Err(format!("{prefix} Elasticsearch node HTTP layer SSL configuration Keystore contains multiple PrivateKey entries where the associated certificate is a CA certificate")),
    }
}

/// ES `getHttpsCaFingerprint`.
pub fn https_ca_fingerprint(settings: &Settings) -> Result<String, String> {
    http_ca(settings, CaUse::Token).map(|e| hex(&elasticsearch_engine::security::hash::sha256(&e.cert.der)))
}

fn b64(der: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(der)
}

fn illegal_state(msg: impl Into<String>) -> EsError {
    EsError::new(500, "illegal_state_exception", msg.into())
}

/// `GET /_security/enroll/node` (ES `TransportNodeEnrollmentAction`): the
/// HTTP CA key and certificate, the transport CA certificate, the transport
/// key and certificate, and the transport publish address of every node.
pub fn node_enrollment_response(settings: &Settings, nodes_addresses: Vec<String>) -> Result<Value, EsError> {
    let config = |e: String| EsError::new(500, "exception", e);
    let transport = material(settings, "xpack.security.transport.ssl.", None).map_err(config)?
        .ok_or_else(|| illegal_state("Unable to enroll node. Elasticsearch node transport layer SSL configuration is not configured with a keystore"))?;
    let http = material(settings, elasticsearch_ssl::HTTP_SSL, Some(HTTP_CA_KEY_FILE)).map_err(config)?
        .ok_or_else(|| illegal_state("Unable to enroll node. Elasticsearch node HTTP layer SSL configuration is not configured with a keystore"))?;
    let transport_key = match transport.entries.len() {
        0 => return Err(illegal_state("Unable to enroll node. Elasticsearch node transport layer SSL configuration doesn't contain any keys")),
        1 => &transport.entries[0],
        _ => return Err(illegal_state("Unable to enroll node. Elasticsearch node transport layer SSL configuration contains multiple keys")),
    };
    let mut transport_cas: Vec<&CertInfo> = Vec::new();
    for c in transport.certificates.iter().filter(|c| c.is_ca) {
        if !transport_cas.iter().any(|x| x.der == c.der) {
            transport_cas.push(c);
        }
    }
    if transport_cas.len() != 1 {
        return Err(EsError::new(500, "exception", "Unable to enroll Elasticsearch node. Elasticsearch node transport layer SSL configuration Keystore [xpack.security.transport.ssl.keystore] doesn't contain a single CA certificate"));
    }
    let http_cas: Vec<&KeyEntry> = http.entries.iter().filter(|e| e.cert.is_ca).collect();
    let http_ca = match http_cas.len() {
        0 => return Err(illegal_state("Unable to enroll node. Elasticsearch node HTTP layer SSL configuration Keystore doesn't contain any PrivateKey entries where the associated certificate is a CA certificate")),
        1 => http_cas[0],
        _ => return Err(illegal_state("Unable to enroll node. Elasticsearch node HTTP layer SSL configuration Keystore contain multiple PrivateKey entries where the associated certificate is a CA certificate")),
    };
    let mut m = Map::new();
    m.insert("http_ca_key".into(), json!(b64(&http_ca.pkcs8)));
    m.insert("http_ca_cert".into(), json!(b64(&http_ca.cert.der)));
    m.insert("transport_ca_cert".into(), json!(b64(&transport_cas[0].der)));
    m.insert("transport_key".into(), json!(b64(&transport_key.pkcs8)));
    m.insert("transport_cert".into(), json!(b64(&transport_key.cert.der)));
    m.insert("nodes_addresses".into(), json!(nodes_addresses));
    Ok(Value::Object(m))
}

/// `GET /_security/enroll/kibana` (ES `TransportKibanaEnrollmentAction`):
/// the base64 DER of the HTTP CA certificate.
pub fn kibana_http_ca(settings: &Settings) -> Result<String, EsError> {
    let http = material(settings, elasticsearch_ssl::HTTP_SSL, Some(HTTP_CA_KEY_FILE))
        .map_err(|e| EsError::new(500, "exception", "Unable to enroll kibana instance. Cannot retrieve CA certificate for the HTTP layer of the Elasticsearch node.").caused_by(EsError::new(500, "exception", e)))?
        .ok_or_else(|| EsError::new(500, "exception", "Unable to enroll kibana instance. Elasticsearch node HTTP layer SSL configuration is not configured with a keystore"))?;
    let cas: Vec<&KeyEntry> = http.entries.iter().filter(|e| e.cert.is_ca).collect();
    if cas.len() != 1 {
        return Err(EsError::new(500, "exception", "Unable to enroll kibana instance. Elasticsearch node HTTP layer SSL configuration Keystore [xpack.security.http.ssl.keystore] doesn't contain a single PrivateKey entry where the associated certificate is a CA certificate"));
    }
    Ok(b64(&cas[0].cert.der))
}

/// Whether `xpack.security.enrollment.enabled` is true.
pub fn enrollment_enabled(settings: &Settings) -> bool {
    setting_text(settings, "xpack.security.enrollment.enabled").as_deref() == Some("true")
}

/// The request extension the cluster front attaches to `/_security/enroll/*`.
pub fn request_extension(node: &Arc<crate::node::ClusterNode>) -> elasticsearch_http::Enrollment {
    let enabled = enrollment_enabled(&node.config.settings) && node.engine.security.enabled();
    let for_node = node.clone();
    let for_kibana = node.clone();
    elasticsearch_http::Enrollment {
        enabled,
        node: Arc::new(move || {
            let state = for_node.state();
            let addresses: Vec<String> = state.nodes.values().map(|n| n.transport_address.clone()).collect();
            node_enrollment_response(&for_node.config.settings, addresses)
        }),
        http_ca: Arc::new(move || kibana_http_ca(&for_kibana.config.settings)),
    }
}

// ---------------------------------------------------- in-node token creation

/// ES `InternalEnrollmentTokenGenerator.assembleToken`: an API key owned by
/// the internal security user, granting `action` for 30 minutes, created
/// through the master's metadata service.
pub async fn create_internal_token(node: &Arc<crate::node::ClusterNode>, action: &str, http_addresses: &[String]) -> Result<EnrollmentToken, String> {
    let settings = &node.config.settings;
    if !enrollment_enabled(settings) {
        return Err(format!("Cannot create enrollment token [{action}] because enrollment is disabled with setting [xpack.security.enrollment.enabled] set to [false]"));
    }
    let fingerprint = https_ca_fingerprint(settings).map_err(|e| format!("Failed to create enrollment token when computing HTTPS CA fingerprint, possibly the certs are not auto-generated: {e}"))?;
    let token_addresses = filtered_addresses(http_addresses);
    let descriptor = elasticsearch_engine::security::roles::parse_role("create_enrollment_token", &json!({"cluster": [action]}), true).map_err(|e| e.reason.clone())?;
    let created = node.mutate_engine_metadata(move |engine| {
        let sec = &engine.security;
        let system = elasticsearch_engine::security::Security::system();
        let limited = sec.user_descriptors(&system);
        let mut owner = system;
        owner.username = "_xpack_security".into();
        let mut rds = Map::new();
        rds.insert("create_enrollment_token".into(), descriptor);
        let expiration = elasticsearch_engine::security::now_ms() + ENROLL_API_KEY_EXPIRATION_MINUTES * 60 * 1000;
        let name = format!("enrollment_token_API_key_{}", elasticsearch_engine::security::hash::random_b64url(15));
        sec.create_api_key(&owner, limited, name, Some(expiration), rds, json!({}))
    }).await;
    match created {
        Ok((key, _)) => Ok(EnrollmentToken::new(format!("{}:{}", key.id, key.hash), fingerprint, token_addresses)),
        Err(e) => Err(format!("Failed to create enrollment token when generating API key: {}", e.reason)),
    }
}

// ------------------------------------------------------ pinned HTTPS client

/// ES `CommandLineHttpClient`'s fingerprint trust: the server's chain must
/// carry the CA whose SHA-256 is pinned right after its leaf. The leaf must
/// also chain to that CA (host names are not checked, as in ES; ES checks
/// only the fingerprint).
#[derive(Debug)]
struct PinnedCa {
    fingerprint: String,
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl ServerCertVerifier for PinnedCa {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        intermediates: &[rustls::pki_types::CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let untrusted = || rustls::Error::InvalidCertificate(rustls::CertificateError::UnknownIssuer);
        let ca = intermediates.first().ok_or_else(untrusted)?;
        if !hex(&elasticsearch_engine::security::hash::sha256(ca.as_ref())).eq_ignore_ascii_case(&self.fingerprint) {
            return Err(untrusted());
        }
        let mut roots = rustls::RootCertStore::empty();
        roots.add(ca.clone().into_owned())?;
        let verifier = elasticsearch_ssl::PeerVerifier::new(Arc::new(roots), elasticsearch_ssl::VerificationMode::Certificate, &self.provider).map_err(rustls::Error::General)?;
        verifier.verify_server_cert(end_entity, &[], server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &rustls::pki_types::CertificateDer<'_>, dss: &rustls::DigitallySignedStruct) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &rustls::pki_types::CertificateDer<'_>, dss: &rustls::DigitallySignedStruct) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

/// An HTTPS client trusting the CA with this fingerprint only.
pub fn pinned_client(fingerprint: &str) -> Result<reqwest::Client, String> {
    let provider = elasticsearch_ssl::provider();
    let config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinnedCa { fingerprint: fingerprint.to_string(), provider }))
        .with_no_client_auth();
    reqwest::Client::builder().use_preconfigured_tls(config).timeout(std::time::Duration::from_secs(35)).build().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_round_trips_in_es_format() {
        let t = EnrollmentToken::new("id:secret".into(), "ab12".into(), vec!["192.168.1.2:9200".into(), "[::1]:9200".into()]);
        assert_eq!(t.raw(), r#"{"ver":"8.14.0","adr":["192.168.1.2:9200","[::1]:9200"],"fgr":"ab12","key":"id:secret"}"#);
        let enc = t.encoded();
        assert_eq!(base64::engine::general_purpose::URL_SAFE.decode(&enc).unwrap(), t.raw().as_bytes());
        assert_eq!(EnrollmentToken::decode(&enc).unwrap(), t);
        // A token ES encoded (basic alphabet, as ES's tests build them), and
        // one without padding.
        let es = base64::engine::general_purpose::STANDARD.encode(r#"{"ver":"8.0.0","adr":["[192.168.0.1]:9201","172.16.254.1:9202","[2001:db8:0:1234:0:567:8:1]:9203"],"fgr":"ce480d53728605674fcfd8ffb51000d8a33bf32de7c7f1e26b4d428f8a91362d","key":"DR6CzXkBDf8amV_48yYX:x3YqU_rqQwm-ESrkExcnOg"}"#);
        let d = EnrollmentToken::decode(&es).unwrap();
        assert_eq!(d.version, "8.0.0");
        assert_eq!(d.bound_address, vec!["[192.168.0.1]:9201", "172.16.254.1:9202", "[2001:db8:0:1234:0:567:8:1]:9203"]);
        assert_eq!(d.api_key, "DR6CzXkBDf8amV_48yYX:x3YqU_rqQwm-ESrkExcnOg");
        assert_eq!(EnrollmentToken::decode(es.trim_end_matches('=')).unwrap(), d);
        assert_eq!(EnrollmentToken::decode("").unwrap_err(), "Cannot decode enrollment token from an empty string");
        assert!(EnrollmentToken::decode("not base64 !").is_err());
        let partial = base64::engine::general_purpose::STANDARD.encode(r#"{"ver":"8.14.0","adr":[]}"#);
        assert_eq!(EnrollmentToken::decode(&partial).unwrap_err(), "Required [key, fgr]");
        let unknown = base64::engine::general_purpose::STANDARD.encode(r#"{"ver":"1","adr":[],"fgr":"f","key":"k","x":1}"#);
        assert!(EnrollmentToken::decode(&unknown).unwrap_err().contains("unknown field [x]"));
    }

    #[test]
    fn addresses_follow_es() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let (local, non_local) = split_addresses(&a(&["[::1]:9200", "127.0.0.1:9200", "[2001:db8::1]:9200", "0.0.0.0:9200", "10.0.0.5:9200", "10.0.0.5:9200"]));
        assert_eq!(local, a(&["127.0.0.1:9200", "[::1]:9200"]));
        assert_eq!(non_local, a(&["10.0.0.5:9200", "[2001:db8::1]:9200"]));
        assert_eq!(filtered_addresses(&a(&["127.0.0.1:9200", "10.0.0.5:9200"])), a(&["10.0.0.5:9200"]));
        assert_eq!(filtered_addresses(&a(&["127.0.0.1:9200", "[::1]:9200"])), a(&["127.0.0.1:9200", "[::1]:9200"]));
        assert_eq!(ip_from_publish_address("host.example/10.1.2.3:9200"), "10.1.2.3:9200");
        assert_eq!(ip_from_publish_address("10.1.2.3:9200"), "10.1.2.3:9200");
        let here = [IpAddr::from([10, 0, 0, 5]), IpAddr::from([127, 0, 0, 1])];
        assert!(!any_remote_host_node_address(&a(&["127.0.0.1:9300", "10.0.0.5:9300"]), &here));
        assert!(any_remote_host_node_address(&a(&["10.0.0.6:9300"]), &here));
    }

    #[test]
    fn ca_certificates_parse_and_reissue() {
        let now = chrono::Utc::now();
        let g = crate::security_autoconfig::generate_certs("n1", vec![], now).unwrap();
        let pem = |name: &str| g.files.iter().find(|f| f.name == name).unwrap().pem.clone();
        let ca_der = elasticsearch_ssl::parse_certificates(pem(crate::security_autoconfig::HTTP_CA_CERT).as_bytes()).unwrap().remove(0);
        let ca = parse_certificate(ca_der.as_ref()).unwrap();
        assert!(ca.is_ca);
        assert_eq!(hex(&elasticsearch_engine::security::hash::sha256(&ca.der)), g.http_ca_fingerprint);
        assert_eq!(ca.subject.len(), 1);
        assert_eq!(ca.subject[0].0, vec![2, 5, 4, 3]);
        assert_eq!(ca.subject[0].2, crate::security_autoconfig::AUTO_CONFIG_HTTP_ALT_DN.as_bytes());
        let leaf_der = elasticsearch_ssl::parse_certificates(pem(crate::security_autoconfig::HTTP_CERT).as_bytes()).unwrap().remove(0);
        assert!(!parse_certificate(leaf_der.as_ref()).unwrap().is_ca);
        // The CA key re-issues certificates under the same subject.
        let ca_key = rcgen::KeyPair::from_pem(&pem(crate::security_autoconfig::HTTP_CA_KEY)).unwrap();
        let issuer = issuer_certificate(&ca, &ca_key).unwrap();
        assert_eq!(parse_certificate(issuer.der().as_ref()).unwrap().subject_der, ca.subject_der);
        let other = rcgen::KeyPair::generate().unwrap();
        assert!(issuer_certificate(&ca, &other).is_err(), "a key of another CA is refused");
        // PEM round trip.
        let again = pem_encode("CERTIFICATE", &ca.der);
        assert_eq!(elasticsearch_ssl::parse_certificates(again.as_bytes()).unwrap()[0].to_vec(), ca.der);
    }

    #[test]
    fn enrollment_material_from_the_auto_configured_layout() {
        let conf = tempfile::tempdir().unwrap();
        std::fs::write(conf.path().join("elasticsearch.yml"), "cluster.name: c\n").unwrap();
        let mut settings: Settings = [("path.data", conf.path().join("data")), ("path.conf", conf.path().to_path_buf())]
            .into_iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string_lossy().into_owned()))).collect();
        settings.insert("node.name".into(), json!("n1"));
        crate::security_autoconfig::run_at(&settings, conf.path(), chrono::Utc::now(), &[IpAddr::from([127, 0, 0, 1])]).unwrap();
        let mut s = crate::config::parse_yaml(&std::fs::read_to_string(conf.path().join("elasticsearch.yml")).unwrap()).unwrap();
        s.insert("path.conf".into(), json!(conf.path().to_string_lossy()));
        assert!(enrollment_enabled(&s));
        let fingerprint = https_ca_fingerprint(&s).unwrap();
        assert_eq!(Some(fingerprint.clone()), crate::security_autoconfig::https_ca_fingerprint(&s));
        let r = node_enrollment_response(&s, vec!["127.0.0.1:9300".into()]).unwrap();
        let keys: Vec<&str> = r.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["http_ca_key", "http_ca_cert", "transport_ca_cert", "transport_key", "transport_cert", "nodes_addresses"]);
        let der = |k: &str| base64::engine::general_purpose::STANDARD.decode(r[k].as_str().unwrap()).unwrap();
        assert_eq!(hex(&elasticsearch_engine::security::hash::sha256(&der("http_ca_cert"))), fingerprint);
        assert!(parse_certificate(&der("transport_ca_cert")).unwrap().is_ca);
        assert!(!parse_certificate(&der("transport_cert")).unwrap().is_ca);
        assert!(rcgen::KeyPair::try_from(der("transport_key").as_slice()).is_ok());
        assert!(rcgen::KeyPair::try_from(der("http_ca_key").as_slice()).is_ok());
        assert_eq!(r["nodes_addresses"], json!(["127.0.0.1:9300"]));
        assert_eq!(kibana_http_ca(&s).unwrap(), r["http_ca_cert"].as_str().unwrap());
        // Without the CA key the HTTP layer has no CA entry.
        std::fs::remove_file(conf.path().join("certs").join(HTTP_CA_KEY_FILE)).unwrap();
        let e = node_enrollment_response(&s, vec![]).unwrap_err();
        assert_eq!(e.reason, "Unable to enroll node. Elasticsearch node HTTP layer SSL configuration Keystore doesn't contain any PrivateKey entries where the associated certificate is a CA certificate");
        assert!(https_ca_fingerprint(&s).unwrap_err().starts_with("Unable to create an enrollment token."));
        // No HTTP TLS at all.
        let e = node_enrollment_response(&Settings::new(), vec![]).unwrap_err();
        assert!(e.reason.contains("transport layer SSL configuration is not configured with a keystore"), "{}", e.reason);
    }
}
