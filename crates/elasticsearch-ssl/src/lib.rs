//! ES SSL configuration, shared by every TLS context of the node: the
//! `<prefix>.*` settings ES's `SslConfigurationLoader` reads
//! (`xpack.security.http.ssl.`, `reindex.ssl.`, ...), key material from PEM
//! files (plain or encrypted keys) or a PKCS#12 keystore, trust from PEM CAs,
//! a PKCS#12 truststore or the default roots, ES's verification modes and
//! client authentication, turned into rustls server and client configs
//! (`ring` provider, no global state).
//!
//! Transport TLS (`elasticsearch-transport`) builds its own configs from the
//! PEM and verifier helpers here ([`parse_certificates`],
//! [`parse_private_key`], [`PeerVerifier`]).

mod pem_key;
mod pkcs12;
mod reload;
pub mod restrictions;
mod verify;

pub use reload::{reload_interval, watch, Reload, ReloadOutcome, Reloadable};
pub use rustls;
pub use verify::{PeerVerifier, RestrictedClientVerifier};
pub use restrictions::{TrustRestrictions, TrustRestrictionsConfig, X509Field};

use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `xpack.security.http.ssl.` (the REST layer).
pub const HTTP_SSL: &str = "xpack.security.http.ssl.";
/// `reindex.ssl.` (`_reindex` from a remote cluster).
pub const REINDEX_SSL: &str = "reindex.ssl.";

/// The `ring` crypto provider every config here is built with.
pub fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// `<prefix>.verification_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationMode {
    /// Verify the chain and that the certificate names the peer.
    Full,
    /// Verify the chain only.
    Certificate,
    /// No verification. Testing only.
    None,
}

impl VerificationMode {
    /// ES `SslVerificationMode.parse` (case-insensitive).
    pub fn parse(value: &str) -> Result<VerificationMode, String> {
        match value.to_ascii_lowercase().as_str() {
            "full" => Ok(VerificationMode::Full),
            "certificate" => Ok(VerificationMode::Certificate),
            "none" => Ok(VerificationMode::None),
            _ => Err(format!("could not resolve ssl client verification mode, unknown value [{value}], recognised values are [NONE, CERTIFICATE, FULL]")),
        }
    }
}

/// `<prefix>.client_authentication`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientAuth {
    None,
    Optional,
    Required,
}

impl ClientAuth {
    /// ES `SslClientAuthenticationMode.parse` (case-insensitive).
    pub fn parse(value: &str) -> Result<ClientAuth, String> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Ok(ClientAuth::None),
            "optional" => Ok(ClientAuth::Optional),
            "required" => Ok(ClientAuth::Required),
            _ => Err(format!("could not resolve ssl client authentication, unknown value [{value}], recognised values are [NONE, OPTIONAL, REQUIRED]")),
        }
    }
}

/// `keystore.type` / `truststore.type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreType {
    Pkcs12,
    /// Recognised (ES's default for paths without a PKCS#12 extension) but
    /// not readable here: loading fails with a conversion hint.
    Jks,
}

impl StoreType {
    fn name(self) -> &'static str {
        match self {
            StoreType::Pkcs12 => "PKCS12",
            StoreType::Jks => "jks",
        }
    }
}

/// ES `KeyStoreUtil.inferKeyStoreType`: `.p12`, `.pfx` and `.pkcs12` files
/// are PKCS#12, anything else JKS; an explicit `type` wins.
fn store_type(explicit: Option<String>, path: &str, setting: &str) -> Result<StoreType, String> {
    match explicit.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None => {
            let lower = path.to_ascii_lowercase();
            Ok(if [".p12", ".pfx", ".pkcs12"].iter().any(|ext| lower.ends_with(ext)) { StoreType::Pkcs12 } else { StoreType::Jks })
        }
        Some("pkcs12") => Ok(StoreType::Pkcs12),
        Some("jks") => Ok(StoreType::Jks),
        Some(_) => Err(format!("unsupported keystore type [{}] for [{setting}], supported types are [PKCS12]", explicit.unwrap_or_default())),
    }
}

/// Where the context's own certificate and private key come from.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyMaterial {
    /// `certificate` + `key` (+ `secure_key_passphrase` / `key_passphrase`).
    Pem { certificate: PathBuf, key: PathBuf, passphrase: Option<String> },
    /// `keystore.path` (+ `keystore.type`, `keystore.secure_password` /
    /// `keystore.password`, `keystore.secure_key_password` /
    /// `keystore.key_password`, which defaults to the store password).
    Keystore { path: PathBuf, kind: StoreType, password: String, key_password: String },
}

/// What peers' certificates are verified against.
#[derive(Clone, Debug, PartialEq)]
pub enum TrustMaterial {
    /// Neither `certificate_authorities` nor `truststore.path`: ES's default
    /// trust (the JDK's CAs, here the Mozilla roots of `webpki-roots`) plus
    /// the key material's own certificates.
    Default,
    /// `certificate_authorities`: PEM CA files.
    Pem(Vec<PathBuf>),
    /// `truststore.path` (+ `truststore.type`, `truststore.secure_password`
    /// / `truststore.password`): every certificate in the store is trusted.
    Truststore { path: PathBuf, kind: StoreType, password: String },
}

/// `<prefix>.supported_protocols` entries this node can speak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Tls13,
    Tls12,
}

/// Defaults that differ between contexts.
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub verification_mode: VerificationMode,
    pub client_authentication: ClientAuth,
}

/// One parsed SSL context (`<prefix>.*`).
#[derive(Clone, Debug)]
pub struct SslConfig {
    /// The settings prefix, with its trailing dot (`xpack.security.http.ssl.`).
    pub prefix: String,
    pub key: Option<KeyMaterial>,
    pub trust: TrustMaterial,
    pub verification_mode: VerificationMode,
    pub client_authentication: ClientAuth,
    /// In preference order; never empty.
    pub supported_protocols: Vec<Protocol>,
    /// `trust_restrictions.path` / `.x509_fields`.
    pub trust_restrictions: Option<TrustRestrictionsConfig>,
}

impl SslConfig {
    /// The context name ES uses in messages (`xpack.security.http.ssl`).
    pub fn context(&self) -> &str {
        self.prefix.trim_end_matches('.')
    }

    /// Read `<prefix>.*` (`prefix` ends with a dot) from node settings:
    /// `get` reads a scalar (secure settings injected from the keystore
    /// under their own names), `get_list` a list. Relative paths resolve
    /// against `conf_dir`. Errors are ES's configuration errors.
    pub fn parse(prefix: &str, get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path, defaults: Defaults) -> Result<SslConfig, String> {
        let key = |k: &str| format!("{prefix}{k}");
        let s = |k: &str| get(&key(k)).filter(|v| !v.is_empty());
        let path = |p: String| -> PathBuf {
            let p = PathBuf::from(p);
            if p.is_absolute() { p } else { conf_dir.join(p) }
        };
        // ES `resolvePasswordSetting`: the secure (keystore) setting or the
        // legacy one, never both. An empty password is a password.
        let secret = |secure: &str, legacy: &str| -> Result<Option<String>, String> {
            match (get(&key(secure)), get(&key(legacy))) {
                (Some(_), Some(_)) => Err(format!("cannot specify both [{}] and [{}]", key(secure), key(legacy))),
                (a, b) => Ok(a.or(b)),
            }
        };
        let verification_mode = match s("verification_mode") {
            None => defaults.verification_mode,
            Some(v) => VerificationMode::parse(&v)?,
        };
        let client_authentication = match s("client_authentication") {
            None => defaults.client_authentication,
            Some(v) => ClientAuth::parse(&v)?,
        };
        let key_material = match (s("keystore.path"), s("certificate"), s("key")) {
            (Some(_), Some(_), _) => return Err(format!("cannot specify both [{}] and [{}]", key("keystore.path"), key("certificate"))),
            (Some(_), None, Some(_)) => return Err(format!("cannot specify both [{}] and [{}]", key("keystore.path"), key("key"))),
            (Some(ks), None, None) => {
                let kind = store_type(s("keystore.type"), &ks, &key("keystore.type"))?;
                let password = secret("keystore.secure_password", "keystore.password")?.unwrap_or_default();
                // The key password defaults to the store password.
                let key_password = secret("keystore.secure_key_password", "keystore.key_password")?.unwrap_or_else(|| password.clone());
                Some(KeyMaterial::Keystore { path: path(ks), kind, password, key_password })
            }
            (None, Some(c), Some(k)) => Some(KeyMaterial::Pem { certificate: path(c), key: path(k), passphrase: secret("secure_key_passphrase", "key_passphrase")? }),
            (None, Some(_), None) => return Err(format!("cannot specify [{}] without also setting [{}]", key("certificate"), key("key"))),
            (None, None, Some(_)) => return Err(format!("cannot specify [{}] without also setting [{}]", key("key"), key("certificate"))),
            (None, None, None) => None,
        };
        let cas: Vec<String> = get_list(&key("certificate_authorities")).into_iter().filter(|c| !c.is_empty()).collect();
        let trust = match (cas.is_empty(), s("truststore.path")) {
            (false, Some(_)) => return Err(format!("cannot specify both [{}] and [{}]", key("certificate_authorities"), key("truststore.path"))),
            (false, None) => TrustMaterial::Pem(cas.into_iter().map(&path).collect()),
            (true, Some(ts)) => TrustMaterial::Truststore {
                kind: store_type(s("truststore.type"), &ts, &key("truststore.type"))?,
                password: secret("truststore.secure_password", "truststore.password")?.unwrap_or_default(),
                path: path(ts),
            },
            (true, None) => TrustMaterial::Default,
        };
        let supported_protocols = parse_protocols(&get_list(&key("supported_protocols")), &key("supported_protocols"))?;
        let trust_restrictions = TrustRestrictionsConfig::parse(prefix, get, get_list, conf_dir)?;
        Ok(SslConfig { prefix: prefix.to_string(), key: key_material, trust, verification_mode, client_authentication, supported_protocols, trust_restrictions })
    }

    /// Every file this context reads (ES `SslConfiguration.getDependentFiles`):
    /// the certificate and key or the keystore, the CA files or the
    /// truststore. The default trust has none.
    pub fn dependent_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        match &self.key {
            None => {}
            Some(KeyMaterial::Pem { certificate, key, .. }) => files.extend([certificate.clone(), key.clone()]),
            Some(KeyMaterial::Keystore { path, .. }) => files.push(path.clone()),
        }
        match &self.trust {
            TrustMaterial::Default => {}
            TrustMaterial::Pem(cas) => files.extend(cas.iter().cloned()),
            TrustMaterial::Truststore { path, .. } => files.push(path.clone()),
        }
        // `RestrictedTrustConfig.getDependentFiles`: the restrictions file is
        // watched with the rest, so editing it rebuilds the context.
        if let Some(r) = &self.trust_restrictions {
            files.push(r.path.clone());
        }
        let mut seen = std::collections::HashSet::new();
        files.retain(|f| seen.insert(f.clone()));
        files
    }

    /// [`SslConfig::server_config`], rebuilt whenever a
    /// [dependent file](SslConfig::dependent_files) changes.
    pub fn reloadable_server(&self, alpn: &[&[u8]]) -> Result<Reloadable<ServerConfig>, String> {
        let cfg = self.clone();
        let alpn: Vec<Vec<u8>> = alpn.iter().map(|p| p.to_vec()).collect();
        Reloadable::new(self.context(), self.dependent_files(), move || {
            let alpn: Vec<&[u8]> = alpn.iter().map(Vec::as_slice).collect();
            cfg.server_config(&alpn)
        })
    }

    /// [`SslConfig::client_config`], rebuilt whenever a
    /// [dependent file](SslConfig::dependent_files) changes.
    pub fn reloadable_client(&self) -> Result<Reloadable<ClientConfig>, String> {
        let cfg = self.clone();
        Reloadable::new(self.context(), self.dependent_files(), move || cfg.client_config())
    }

    fn fail(&self, detail: impl std::fmt::Display) -> String {
        format!("failed to load SSL configuration [{}] - {detail}", self.context())
    }

    /// The certificate chain (leaf first) and private key, if configured.
    pub fn load_key(&self) -> Result<Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>, String> {
        match &self.key {
            None => Ok(None),
            Some(KeyMaterial::Pem { certificate, key, passphrase }) => {
                let pem = read(certificate, "PEM certificate").map_err(|e| self.fail(e))?;
                let chain = parse_certificates(&pem).map_err(|e| self.fail(format!("cannot load PEM certificate [{}]: {e}", certificate.display())))?;
                let key_pem = read(key, "PEM private key").map_err(|e| self.fail(e))?;
                let key = parse_private_key(&key_pem, passphrase.as_deref(), &key.display().to_string()).map_err(|e| self.fail(e))?;
                Ok(Some((chain, key)))
            }
            Some(KeyMaterial::Keystore { path, kind, password, key_password }) => {
                let store = self.load_store(path, *kind, password, key_password, "keystore")?;
                match store.key_and_chain() {
                    Some(kc) => Ok(Some(kc)),
                    None => Err(self.fail(format!("the configured [{}] keystore [{}] does not contain a private key entry", kind.name(), path.display()))),
                }
            }
        }
    }

    /// The configured keystore's contents (`None` for PEM key material or no
    /// key material): ES `StoreKeyConfig.getKeys()` (every private key entry
    /// with its certificate) and `getConfiguredCertificates()` (every
    /// certificate in the store).
    #[allow(clippy::type_complexity)]
    pub fn keystore_contents(&self) -> Result<Option<(Vec<(PrivateKeyDer<'static>, CertificateDer<'static>)>, Vec<CertificateDer<'static>>)>, String> {
        match &self.key {
            Some(KeyMaterial::Keystore { path, kind, password, key_password }) => {
                let store = self.load_store(path, *kind, password, key_password, "keystore")?;
                let certs = store.certs.iter().map(|(_, c)| c.clone()).collect();
                Ok(Some((store.key_entries(), certs)))
            }
            _ => Ok(None),
        }
    }

    fn load_store(&self, path: &Path, kind: StoreType, password: &str, key_password: &str, what: &str) -> Result<pkcs12::Store, String> {
        if kind == StoreType::Jks {
            return Err(self.fail(format!(
                "cannot read configured [jks] {what} [{}]: JKS stores are not supported, convert it to PKCS#12 (keytool -importkeystore -srckeystore {} -destkeystore {}.p12 -deststoretype PKCS12) or use PEM files",
                path.display(), path.display(), path.display()
            )));
        }
        let der = read(path, &format!("[PKCS12] {what}")).map_err(|e| self.fail(e))?;
        pkcs12::parse(&der, password, key_password).map_err(|e| match e {
            pkcs12::StoreError::Password => self.fail(format!("cannot read configured [PKCS12] {what} [{}] - this is usually caused by an incorrect password", path.display())),
            pkcs12::StoreError::Invalid(m) => self.fail(format!("cannot read configured [PKCS12] {what} [{}]: {m}", path.display())),
        })
    }

    /// The trust anchors; `own` (the key material's chain) joins the
    /// default trust, as in ES.
    pub fn load_trust(&self, own: Option<&[CertificateDer<'static>]>) -> Result<RootCertStore, String> {
        let mut roots = RootCertStore::empty();
        let add = |roots: &mut RootCertStore, c: CertificateDer<'static>, origin: &str| -> Result<(), String> {
            roots.add(c).map_err(|e| self.fail(format!("cannot use certificate from [{origin}] as a trust anchor: {e}")))
        };
        match &self.trust {
            TrustMaterial::Default => {
                roots.roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
                for c in own.unwrap_or_default() {
                    // Best effort, as ES: a key chain that isn't a valid anchor
                    // just isn't trusted.
                    let _ = roots.add(c.clone());
                }
            }
            TrustMaterial::Pem(files) => {
                for f in files {
                    let pem = read(f, "PEM certificate_authorities").map_err(|e| self.fail(e))?;
                    let certs = parse_certificates(&pem).map_err(|e| self.fail(format!("cannot load PEM certificate_authorities [{}]: {e}", f.display())))?;
                    for c in certs {
                        add(&mut roots, c, &f.display().to_string())?;
                    }
                }
            }
            TrustMaterial::Truststore { path, kind, password } => {
                let store = self.load_store(path, *kind, password, password, "truststore")?;
                if store.certs.is_empty() {
                    return Err(self.fail(format!("the configured [PKCS12] truststore [{}] does not contain any trusted certificate entries", path.display())));
                }
                for (_, c) in store.certs {
                    add(&mut roots, c, &path.display().to_string())?;
                }
            }
        }
        Ok(roots)
    }

    fn versions(&self) -> Vec<&'static rustls::SupportedProtocolVersion> {
        self.supported_protocols
            .iter()
            .map(|p| match p {
                Protocol::Tls13 => &rustls::version::TLS13,
                Protocol::Tls12 => &rustls::version::TLS12,
            })
            .collect()
    }

    /// A server config (the node accepts connections in this context),
    /// offering `alpn` protocols. ES requires a key: without one this is
    /// `SSLService`'s server configuration error.
    pub fn server_config(&self, alpn: &[&[u8]]) -> Result<Arc<ServerConfig>, String> {
        let Some((chain, key)) = self.load_key()? else {
            let p = &self.prefix;
            return Err(format!(
                "invalid SSL configuration for {} - server ssl configuration requires a key and certificate, but these have not been configured; you must set either [{p}keystore.path], or both [{p}key] and [{p}certificate]",
                self.context()
            ));
        };
        let provider = provider();
        let client_verifier: Arc<dyn rustls::server::danger::ClientCertVerifier> = match (self.client_authentication, self.verification_mode) {
            (ClientAuth::None, _) => WebPkiClientVerifier::no_client_auth(),
            (mode, VerificationMode::None) => Arc::new(verify::AnyClientCert { mandatory: mode == ClientAuth::Required, algs: provider.signature_verification_algorithms }),
            (mode, _) => {
                let roots = Arc::new(self.load_trust(Some(&chain))?);
                let b = WebPkiClientVerifier::builder_with_provider(roots, provider.clone());
                let b = if mode == ClientAuth::Optional { b.allow_unauthenticated() } else { b };
                b.build().map_err(|e| self.fail(format!("client certificate verifier: {e}")))?
            }
        };
        let client_verifier = match &self.trust_restrictions {
            Some(r) if self.client_authentication != ClientAuth::None => {
                RestrictedClientVerifier::wrap(client_verifier, TrustRestrictions::load(r.clone()).map_err(|e| self.fail(e))?)
            }
            _ => client_verifier,
        };
        let mut config = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&self.versions())
            .map_err(|e| self.fail(format!("protocol versions: {e}")))?
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(chain, key)
            .map_err(|e| self.fail(format!("server certificate: {e}")))?;
        config.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
        Ok(Arc::new(config))
    }

    /// A client config (the node connects out in this context): peers are
    /// verified per `verification_mode`, the key material (if any) is the
    /// client certificate.
    pub fn client_config(&self) -> Result<Arc<ClientConfig>, String> {
        let provider = provider();
        let key = self.load_key()?;
        let roots = Arc::new(self.load_trust(key.as_ref().map(|(c, _)| c.as_slice()))?);
        let restrictions = match &self.trust_restrictions {
            Some(r) => Some(TrustRestrictions::load(r.clone()).map_err(|e| self.fail(e))?),
            None => None,
        };
        let verifier = PeerVerifier::with_restrictions(roots, self.verification_mode, &provider, restrictions).map_err(|e| self.fail(e))?;
        let builder = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&self.versions())
            .map_err(|e| self.fail(format!("protocol versions: {e}")))?
            .dangerous()
            .with_custom_certificate_verifier(verifier);
        let config = match key {
            Some((chain, key)) => builder.with_client_auth_cert(chain, key).map_err(|e| self.fail(format!("client certificate: {e}")))?,
            None => builder.with_no_client_auth(),
        };
        Ok(Arc::new(config))
    }
}

/// ES's default `supported_protocols` is `TLSv1.3, TLSv1.2` (plus older
/// versions only when explicitly listed, which rustls cannot speak).
fn parse_protocols(list: &[String], setting: &str) -> Result<Vec<Protocol>, String> {
    if list.iter().all(|p| p.is_empty()) {
        return Ok(vec![Protocol::Tls13, Protocol::Tls12]);
    }
    let mut out = Vec::new();
    for p in list.iter().filter(|p| !p.is_empty()) {
        let v = match p.as_str() {
            "TLSv1.3" => Some(Protocol::Tls13),
            "TLSv1.2" => Some(Protocol::Tls12),
            "TLSv1.1" | "TLSv1" | "SSLv3" | "SSLv2Hello" => None,
            other => return Err(format!("[{setting}] contains unsupported protocol [{other}], supported protocols are [TLSv1.3, TLSv1.2, TLSv1.1, TLSv1]")),
        };
        if let Some(v) = v {
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    if out.is_empty() {
        return Err(format!("[{setting}] [{}] does not include a protocol this node supports; supported protocols are [TLSv1.3, TLSv1.2]", list.join(", ")));
    }
    Ok(out)
}

fn read(path: &Path, what: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => format!("cannot read configured {what} [{}] because the file does not exist", path.display()),
        std::io::ErrorKind::PermissionDenied => format!("not permitted to read the {what} file [{}]", path.display()),
        _ => format!("cannot read configured {what} [{}]: {e}", path.display()),
    })
}

/// Every `CERTIFICATE` block of a PEM file, in order.
pub fn parse_certificates(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>, String> {
    let certs = CertificateDer::pem_slice_iter(pem).collect::<Result<Vec<_>, _>>().map_err(|e| format!("invalid certificate PEM: {e}"))?;
    if certs.is_empty() {
        return Err("no certificates found in PEM".into());
    }
    Ok(certs)
}

/// The private key in a PEM file (PKCS#8, PKCS#1 or SEC1; plain, PKCS#8
/// encrypted or OpenSSL `Proc-Type: 4,ENCRYPTED`), decrypted with
/// `passphrase`. `origin` names the file in errors.
pub fn parse_private_key(pem: &[u8], passphrase: Option<&str>, origin: &str) -> Result<PrivateKeyDer<'static>, String> {
    pem_key::parse(pem, passphrase, origin).map_err(|e| e.0)
}

/// ES's strict boolean parsing of `setting`.
fn parse_bool(value: Option<String>, setting: &str) -> Result<bool, String> {
    match value.as_deref() {
        None | Some("") | Some("false") => Ok(false),
        Some("true") => Ok(true),
        Some(other) => Err(format!("Failed to parse value [{other}] for setting [{setting}] as only [true] or [false] are allowed.")),
    }
}

/// `xpack.security.http.ssl.*`: `Ok(None)` unless `enabled` is true, else
/// the HTTPS server config (ALPN `http/1.1`; the REST server speaks
/// HTTP/1.1 only). Client authentication defaults to `none`, verification
/// to `full`, as ES's HTTP context.
pub fn http_server(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Option<Arc<ServerConfig>>, String> {
    match http_ssl(get, get_list, conf_dir)? {
        None => Ok(None),
        Some(cfg) => cfg.server_config(&[b"http/1.1"]).map(Some),
    }
}

/// [`http_server`], reloaded when its files change ([`Reloadable`]).
pub fn http_server_reloadable(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Option<Arc<Reloadable<ServerConfig>>>, String> {
    match http_ssl(get, get_list, conf_dir)? {
        None => Ok(None),
        Some(cfg) => cfg.reloadable_server(&[b"http/1.1"]).map(|r| Some(Arc::new(r))),
    }
}

fn http_ssl(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Option<SslConfig>, String> {
    let enabled = format!("{HTTP_SSL}enabled");
    if !parse_bool(get(&enabled), &enabled)? {
        return Ok(None);
    }
    SslConfig::parse(HTTP_SSL, get, get_list, conf_dir, Defaults { verification_mode: VerificationMode::Full, client_authentication: ClientAuth::None }).map(Some)
}

/// `reindex.ssl.*`: the client config `_reindex` from a remote `https://`
/// host connects with (full verification against the default roots unless
/// configured otherwise).
pub fn reindex_client(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Arc<ClientConfig>, String> {
    reindex_ssl(get, get_list, conf_dir)?.client_config()
}

/// [`reindex_client`], reloaded when its files change ([`Reloadable`]).
pub fn reindex_client_reloadable(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<Reloadable<ClientConfig>, String> {
    reindex_ssl(get, get_list, conf_dir)?.reloadable_client()
}

fn reindex_ssl(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &Path) -> Result<SslConfig, String> {
    SslConfig::parse(REINDEX_SSL, get, get_list, conf_dir, Defaults { verification_mode: VerificationMode::Full, client_authentication: ClientAuth::None })
}

#[cfg(test)]
mod tests;
