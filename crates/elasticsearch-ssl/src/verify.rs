//! Certificate verifiers implementing ES's verification modes.
//!
//! `full` checks the chain and that the certificate names the peer (host
//! name or IP), `certificate` checks the chain only (name mismatches are
//! ignored), `none` accepts any certificate but still checks the handshake
//! signatures.

use crate::restrictions::TrustRestrictions;
use crate::VerificationMode;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{CertificateError, DigitallySignedStruct, DistinguishedName, RootCertStore, SignatureScheme};
use std::sync::Arc;

/// Server-certificate verifier (the client side of a connection).
#[derive(Debug)]
pub struct PeerVerifier {
    inner: Option<Arc<rustls::client::WebPkiServerVerifier>>,
    mode: VerificationMode,
    algs: WebPkiSupportedAlgorithms,
    /// `trust_restrictions.*`: checked after the chain (ES's
    /// `RestrictedTrustManager` wraps the delegate trust manager).
    restrictions: Option<Arc<TrustRestrictions>>,
}

impl PeerVerifier {
    /// Verify against `roots` with `mode` (`roots` is unused for `none`).
    pub fn new(roots: Arc<RootCertStore>, mode: VerificationMode, provider: &Arc<CryptoProvider>) -> Result<Arc<PeerVerifier>, String> {
        Self::with_restrictions(roots, mode, provider, None)
    }

    /// As [`Self::new`], also enforcing trust restrictions.
    pub fn with_restrictions(roots: Arc<RootCertStore>, mode: VerificationMode, provider: &Arc<CryptoProvider>, restrictions: Option<Arc<TrustRestrictions>>) -> Result<Arc<PeerVerifier>, String> {
        let inner = match mode {
            VerificationMode::None => None,
            _ => Some(
                rustls::client::WebPkiServerVerifier::builder_with_provider(roots, provider.clone())
                    .build()
                    .map_err(|e| format!("server certificate verifier: {e}"))?,
            ),
        };
        Ok(Arc::new(PeerVerifier { inner, mode, algs: provider.signature_verification_algorithms, restrictions }))
    }
}

fn is_name_mismatch(e: &CertificateError) -> bool {
    // `NotValidForNameContext` only exists in newer rustls 0.23.x; match it by
    // name so either version compiles.
    matches!(e, CertificateError::NotValidForName) || format!("{e:?}").starts_with("NotValidForName")
}

impl ServerCertVerifier for PeerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let verified = match &self.inner {
            None => Ok(ServerCertVerified::assertion()),
            Some(inner) => match inner.verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now) {
                Err(rustls::Error::InvalidCertificate(e)) if self.mode == VerificationMode::Certificate && is_name_mismatch(&e) => Ok(ServerCertVerified::assertion()),
                r => r,
            },
        }?;
        check_restrictions(self.restrictions.as_deref(), end_entity)?;
        Ok(verified)
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// Client-certificate verifier for `verification_mode: none` with client
/// authentication: asks for (or requires) a certificate and accepts any.
#[derive(Debug)]
pub(crate) struct AnyClientCert {
    pub(crate) mandatory: bool,
    pub(crate) algs: WebPkiSupportedAlgorithms,
}

impl ClientCertVerifier for AnyClientCert {
    fn client_auth_mandatory(&self) -> bool {
        self.mandatory
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(&self, _end_entity: &CertificateDer<'_>, _intermediates: &[CertificateDer<'_>], _now: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

fn check_restrictions(restrictions: Option<&TrustRestrictions>, end_entity: &CertificateDer<'_>) -> Result<(), rustls::Error> {
    match restrictions {
        None => Ok(()),
        Some(r) => r.verify(end_entity.as_ref()).map_err(|reason| {
            tracing::info!("Rejecting certificate: {reason}");
            // ES's `CertificateException` message travels with the error.
            let e: Arc<dyn std::error::Error + Send + Sync> = Arc::new(std::io::Error::other(reason));
            rustls::Error::InvalidCertificate(CertificateError::Other(rustls::OtherError(e)))
        }),
    }
}

/// A client-certificate verifier that also enforces trust restrictions on
/// the presented certificate (`RestrictedTrustManager.checkClientTrusted`).
#[derive(Debug)]
pub struct RestrictedClientVerifier {
    inner: Arc<dyn ClientCertVerifier>,
    restrictions: Arc<TrustRestrictions>,
}

impl RestrictedClientVerifier {
    pub fn wrap(inner: Arc<dyn ClientCertVerifier>, restrictions: Arc<TrustRestrictions>) -> Arc<dyn ClientCertVerifier> {
        Arc::new(RestrictedClientVerifier { inner, restrictions })
    }
}

impl ClientCertVerifier for RestrictedClientVerifier {
    fn offer_client_auth(&self) -> bool {
        self.inner.offer_client_auth()
    }

    fn client_auth_mandatory(&self) -> bool {
        self.inner.client_auth_mandatory()
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        self.inner.root_hint_subjects()
    }

    fn verify_client_cert(&self, end_entity: &CertificateDer<'_>, intermediates: &[CertificateDer<'_>], now: UnixTime) -> Result<ClientCertVerified, rustls::Error> {
        let verified = self.inner.verify_client_cert(end_entity, intermediates, now)?;
        check_restrictions(Some(&self.restrictions), end_entity)?;
        Ok(verified)
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}
