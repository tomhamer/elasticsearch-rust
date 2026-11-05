//! Transport configuration: the `transport.*` and
//! `xpack.security.transport.ssl.*` settings this layer honours.

use elasticsearch_cluster::state::DiscoveryNode;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Tunables, named after their ES settings.
#[derive(Clone, Debug)]
pub struct TransportSettings {
    /// `transport.compress`: lz4-compress payloads of at least
    /// `compress_threshold` bytes of indexing data requests
    /// (ES's default `INDEXING_DATA`: replication, resync and recovery
    /// operations, see [`crate::indexing_data`]), or of every request and
    /// response with `compress_all` (ES `true`). Search traffic is not
    /// compressed by default: its lz4 would run on the runtime's workers
    /// beside every request.
    pub compress: bool,
    pub compress_all: bool,
    pub compress_threshold: usize,
    /// Largest frame (header + possibly-compressed payload) we send or
    /// accept, and the largest decompressed payload we accept. Oversized
    /// outgoing messages fail with `FrameTooLarge`; an oversized incoming frame
    /// is a protocol violation and closes the connection.
    pub max_frame_size: usize,
    /// `transport.connect_timeout` (ES default 30s): TCP connect + TLS +
    /// handshake.
    pub connect_timeout: Duration,
    /// `transport.ping_schedule` (ES default -1 = disabled): when set, every
    /// outbound connection is pinged at this interval and closed (firing a
    /// disconnect event) if a ping is not answered within the interval.
    pub ping_schedule: Option<Duration>,
}

impl Default for TransportSettings {
    fn default() -> Self {
        TransportSettings {
            compress: true,
            compress_all: false,
            compress_threshold: 8 * 1024,
            max_frame_size: 1 << 30,
            connect_timeout: Duration::from_secs(30),
            ping_schedule: None,
        }
    }
}

/// Everything needed to start a [`crate::TransportService`].
#[derive(Clone, Debug)]
pub struct TransportConfig {
    /// `cluster.name`; peers with a different name are rejected at handshake.
    pub cluster_name: String,
    /// This node's identity, sent in every handshake. Its `transport_address`
    /// is overwritten with the publish address once the listener is bound.
    pub local_node: DiscoveryNode,
    /// `transport.host` / `transport.port`; port 0 picks a free port.
    pub bind_addr: SocketAddr,
    /// `transport.publish_host`/`publish_port`; defaults to the bound address.
    pub publish_addr: Option<SocketAddr>,
    pub settings: TransportSettings,
    /// `xpack.security.transport.ssl.*`; `None` = plaintext.
    pub tls: Option<TlsConfig>,
}

impl TransportConfig {
    pub fn new(cluster_name: impl Into<String>, local_node: DiscoveryNode, bind_addr: SocketAddr) -> Self {
        TransportConfig {
            cluster_name: cluster_name.into(),
            local_node,
            bind_addr,
            publish_addr: None,
            settings: TransportSettings::default(),
            tls: None,
        }
    }
}

/// PEM material: a file path (as configured in `elasticsearch.yml`) or the
/// PEM text itself.
#[derive(Clone, Debug)]
pub enum PemSource {
    Path(PathBuf),
    Inline(String),
}

impl PemSource {
    #[cfg_attr(not(feature = "tls"), allow(dead_code))]
    pub(crate) fn load(&self) -> std::io::Result<Vec<u8>> {
        match self {
            PemSource::Path(p) => std::fs::read(p),
            PemSource::Inline(s) => Ok(s.as_bytes().to_vec()),
        }
    }
}

/// `xpack.security.transport.ssl.verification_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerificationMode {
    /// Verify the chain and that the certificate matches the peer's address.
    Full,
    /// Verify the chain only (the usual choice for node certificates).
    #[default]
    Certificate,
    /// No verification. Testing only.
    None,
}

/// `xpack.security.transport.ssl.client_authentication`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClientAuthentication {
    /// Mutual TLS: every inbound connection must present a CA-signed cert.
    #[default]
    Required,
    Optional,
    None,
}

/// Transport TLS. The same key pair serves as server certificate (inbound)
/// and client certificate (outbound), as in ES.
#[derive(Clone, Debug)]
pub struct TlsConfig {
    /// `.certificate`: PEM chain, leaf first.
    pub certificate: PemSource,
    /// `.key`: PEM private key (PKCS#8, PKCS#1 or SEC1; plain or encrypted).
    pub key: PemSource,
    /// `.secure_key_passphrase` (keystore) or the legacy `.key_passphrase`:
    /// decrypts an encrypted `key`.
    pub key_passphrase: Option<String>,
    /// `.certificate_authorities`: trusted CAs for verifying peers. When
    /// empty, the node's own certificate chain is trusted (ES's default
    /// trust: the JDK's CAs plus the key configuration's certificates).
    pub certificate_authorities: Vec<PemSource>,
    pub verification_mode: VerificationMode,
    pub client_authentication: ClientAuthentication,
    /// `.trust_restrictions.path` / `.x509_fields`: peers' certificates must
    /// also carry a trusted name (ES `RestrictedTrustConfig`).
    #[cfg(feature = "tls")]
    pub trust_restrictions: Option<elasticsearch_ssl::TrustRestrictionsConfig>,
}

/// Prefix of the transport TLS settings.
pub const TRANSPORT_SSL: &str = "xpack.security.transport.ssl.";

impl TlsConfig {
    /// `xpack.security.transport.ssl.*` from node settings (`get` reads a
    /// scalar, `get_list` a list setting), relative paths resolved against
    /// the config directory. `Ok(None)` when transport TLS is disabled.
    pub fn from_settings(get: &dyn Fn(&str) -> Option<String>, get_list: &dyn Fn(&str) -> Vec<String>, conf_dir: &std::path::Path) -> Result<Option<TlsConfig>, String> {
        let key = |k: &str| format!("{TRANSPORT_SSL}{k}");
        let s = |k: &str| get(&key(k)).filter(|v| !v.is_empty());
        if s("enabled").as_deref() != Some("true") {
            return Ok(None);
        }
        let path = |p: String| -> PemSource {
            let p = std::path::PathBuf::from(p);
            PemSource::Path(if p.is_absolute() { p } else { conf_dir.join(p) })
        };
        for unsupported in ["keystore.path", "truststore.path"] {
            if s(unsupported).is_some() {
                return Err(format!("[{}] is not supported: configure PEM files with [{}] and [{}]", key(unsupported), key("certificate"), key("key")));
            }
        }
        let certificate = s("certificate").ok_or_else(|| format!("[{}] is required when [{}] is true", key("certificate"), key("enabled")))?;
        let private_key = s("key").ok_or_else(|| format!("[{}] is required when [{}] is true", key("key"), key("enabled")))?;
        let verification_mode = match s("verification_mode").as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("full") => VerificationMode::Full,
            Some("certificate") => VerificationMode::Certificate,
            Some("none") => VerificationMode::None,
            Some(o) => return Err(format!("could not resolve ssl client authentication, unknown value [{o}] for [{}]", key("verification_mode"))),
        };
        let client_authentication = match s("client_authentication").as_deref().map(str::to_ascii_lowercase).as_deref() {
            None | Some("required") => ClientAuthentication::Required,
            Some("optional") => ClientAuthentication::Optional,
            Some("none") => ClientAuthentication::None,
            Some(o) => return Err(format!("could not resolve ssl client authentication, unknown value [{o}] for [{}]", key("client_authentication"))),
        };
        let key_passphrase = match (s("secure_key_passphrase"), s("key_passphrase")) {
            (Some(_), Some(_)) => return Err(format!("cannot specify both [{}] and [{}]", key("secure_key_passphrase"), key("key_passphrase"))),
            (a, b) => a.or(b),
        };
        Ok(Some(TlsConfig {
            certificate: path(certificate),
            key: path(private_key),
            key_passphrase,
            certificate_authorities: get_list(&key("certificate_authorities")).into_iter().map(path).collect(),
            verification_mode,
            client_authentication,
            #[cfg(feature = "tls")]
            trust_restrictions: elasticsearch_ssl::TrustRestrictionsConfig::parse(TRANSPORT_SSL, get, get_list, conf_dir)?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(pairs: &[(&str, &str)]) -> Result<Option<TlsConfig>, String> {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (format!("{TRANSPORT_SSL}{k}"), v.to_string())).collect();
        TlsConfig::from_settings(&|k| m.get(k).cloned(), &|k| m.get(k).map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default(), std::path::Path::new("/conf"))
    }

    #[test]
    fn transport_ssl_settings_follow_es() {
        assert!(cfg(&[]).unwrap().is_none());
        let c = cfg(&[("enabled", "true"), ("certificate", "testnode.crt"), ("key", "testnode.pem"), ("verification_mode", "certificate"), ("secure_key_passphrase", "testnode")]).unwrap().unwrap();
        assert!(matches!(&c.certificate, PemSource::Path(p) if p == std::path::Path::new("/conf/testnode.crt")));
        assert_eq!(c.key_passphrase.as_deref(), Some("testnode"));
        assert_eq!(c.verification_mode, VerificationMode::Certificate);
        assert_eq!(c.client_authentication, ClientAuthentication::Required);
        assert!(c.certificate_authorities.is_empty());
        // ES defaults: full verification, required client authentication.
        let d = cfg(&[("enabled", "true"), ("certificate", "/abs/a.crt"), ("key", "a.pem"), ("key_passphrase", "x"), ("certificate_authorities", "ca.crt")]).unwrap().unwrap();
        assert_eq!(d.verification_mode, VerificationMode::Full);
        assert_eq!(d.certificate_authorities.len(), 1);
        assert!(cfg(&[("enabled", "true"), ("key", "a.pem")]).unwrap_err().contains("certificate] is required"));
        assert!(cfg(&[("enabled", "true"), ("certificate", "a"), ("key", "b"), ("key_passphrase", "x"), ("secure_key_passphrase", "y")]).unwrap_err().contains("cannot specify both"));
    }
}
