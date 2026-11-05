//! rustls glue for transport TLS (`ring` provider, no global state).
//!
//! Verification modes follow ES: `full` checks chain + that the certificate
//! names the peer's IP, `certificate` checks the chain only (hostname
//! mismatches are ignored), `none` accepts anything but still checks the
//! handshake signatures. PEM parsing, encrypted keys and the verifier are
//! shared with the node's other TLS contexts (`elasticsearch-ssl`).
//!
//! The server and client configs are rebuilt together when a PEM file they
//! read changes ([`elasticsearch_ssl::Reloadable`], polled by the node's
//! SSL watcher): new connections, inbound and outbound, use the current
//! pair; established connections keep the material they handshook with.

use crate::config::{ClientAuthentication, PemSource, TlsConfig, VerificationMode};
use crate::error::TransportError;
use elasticsearch_ssl::{PeerVerifier, Reloadable};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// One built server + client pair.
pub(crate) struct TlsPair {
    acceptor: TlsAcceptor,
    connector: TlsConnector,
}

pub(crate) struct TlsState {
    current: Arc<Reloadable<TlsPair>>,
}

fn cfg_err(what: &str, e: impl std::fmt::Display) -> TransportError {
    TransportError::Config(format!("xpack.security.transport.ssl: {what}: {e}"))
}

fn load_certs(src: &PemSource) -> Result<Vec<CertificateDer<'static>>, TransportError> {
    let pem = src.load().map_err(|e| cfg_err("cannot read certificate", e))?;
    elasticsearch_ssl::parse_certificates(&pem).map_err(|e| cfg_err("certificate", e))
}

impl TlsState {
    pub(crate) fn new(cfg: &TlsConfig) -> Result<TlsState, TransportError> {
        let files = std::iter::once(&cfg.certificate)
            .chain(std::iter::once(&cfg.key))
            .chain(cfg.certificate_authorities.iter())
            .filter_map(|src| match src {
                PemSource::Path(p) => Some(p.clone()),
                PemSource::Inline(_) => None,
            })
            // The trust restrictions file is watched too (ES
            // `RestrictedTrustConfig.getDependentFiles`): editing it rebuilds
            // the pair, and every rebuilt pair carries the restrictions.
            .chain(cfg.trust_restrictions.iter().map(|r| r.path.clone()))
            .collect();
        let own = cfg.clone();
        let current = Reloadable::new(crate::config::TRANSPORT_SSL.trim_end_matches('.'), files, move || {
            TlsPair::build(&own).map(Arc::new).map_err(|e| match e {
                TransportError::Config(m) => m,
                other => other.to_string(),
            })
        })
            .map_err(TransportError::Config)?;
        Ok(TlsState { current: Arc::new(current) })
    }

    /// The reloadable pair, for the node's SSL watcher.
    pub(crate) fn reloadable(&self) -> Arc<dyn elasticsearch_ssl::Reload> {
        self.current.clone()
    }

    pub(crate) async fn connect(&self, addr: SocketAddr, tcp: TcpStream) -> std::io::Result<tokio_rustls::client::TlsStream<TcpStream>> {
        let pair = self.current.load();
        pair.connector.connect(ServerName::IpAddress(addr.ip().into()), tcp).await
    }

    pub(crate) async fn accept(&self, tcp: TcpStream) -> std::io::Result<tokio_rustls::server::TlsStream<TcpStream>> {
        let pair = self.current.load();
        pair.acceptor.accept(tcp).await
    }
}

impl TlsPair {
    fn build(cfg: &TlsConfig) -> Result<TlsPair, TransportError> {
        let provider = elasticsearch_ssl::provider();
        let chain = load_certs(&cfg.certificate)?;
        let key_pem = cfg.key.load().map_err(|e| cfg_err("cannot read key", e))?;
        let origin = match &cfg.key {
            PemSource::Path(p) => p.display().to_string(),
            PemSource::Inline(_) => "<inline>".to_string(),
        };
        let key = elasticsearch_ssl::parse_private_key(&key_pem, cfg.key_passphrase.as_deref(), &origin).map_err(|e| cfg_err("key", e))?;

        // Without `certificate_authorities` ES trusts the key configuration's
        // own certificates (`buildDefaultTrustConfig`; nodes sharing one
        // certificate trust each other).
        let mut roots = RootCertStore::empty();
        if cfg.certificate_authorities.is_empty() {
            for c in &chain {
                roots.add(c.clone()).map_err(|e| cfg_err("invalid certificate (as trust anchor)", e))?;
            }
        }
        for ca in &cfg.certificate_authorities {
            for c in load_certs(ca)? {
                roots.add(c).map_err(|e| cfg_err("invalid CA certificate", e))?;
            }
        }
        let roots = Arc::new(roots);

        // `trust_restrictions.*`: after the chain, the peer's certificate
        // must carry a trusted name, inbound and outbound (ES
        // `RestrictedTrustManager`).
        let restrictions = match &cfg.trust_restrictions {
            Some(r) => Some(elasticsearch_ssl::TrustRestrictions::load(r.clone()).map_err(|e| cfg_err("trust restrictions", e))?),
            None => None,
        };

        // Server side (inbound connections).
        let client_verifier = match cfg.client_authentication {
            ClientAuthentication::None => WebPkiClientVerifier::no_client_auth(),
            mode => {
                let b = WebPkiClientVerifier::builder_with_provider(roots.clone(), provider.clone());
                let b = if mode == ClientAuthentication::Optional { b.allow_unauthenticated() } else { b };
                let verifier = b.build().map_err(|e| cfg_err("client verifier", e))?;
                match &restrictions {
                    Some(r) => elasticsearch_ssl::RestrictedClientVerifier::wrap(verifier, r.clone()),
                    None => verifier,
                }
            }
        };
        let server = ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .map_err(|e| cfg_err("protocol versions", e))?
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(chain.clone(), key.clone_key())
            .map_err(|e| cfg_err("server certificate", e))?;

        // Client side (outbound connections).
        let mode = match cfg.verification_mode {
            VerificationMode::Full => elasticsearch_ssl::VerificationMode::Full,
            VerificationMode::Certificate => elasticsearch_ssl::VerificationMode::Certificate,
            VerificationMode::None => elasticsearch_ssl::VerificationMode::None,
        };
        let verifier = PeerVerifier::with_restrictions(roots.clone(), mode, &provider, restrictions).map_err(|e| cfg_err("server verifier", e))?;
        let client = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| cfg_err("protocol versions", e))?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_client_auth_cert(chain, key)
            .map_err(|e| cfg_err("client certificate", e))?;

        Ok(TlsPair { acceptor: TlsAcceptor::from(Arc::new(server)), connector: TlsConnector::from(Arc::new(client)) })
    }
}
