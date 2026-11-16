//! Node-to-node transport for the `elasticsearch` cluster layer: the
//! equivalent of ES's `TransportService`, with its own compact binary
//! protocol designed for this implementation. Clusters contain only
//! elasticsearch-rust nodes; migrate documents from Java ES over HTTP with
//! `_reindex` and `source.remote` (see the repository's `MIGRATION.md`).
//!
//! - [`TransportService`] binds one TCP listener per node and keeps a pool of
//!   persistent, multiplexed outbound connections (one per remote address).
//! - Messages are length-prefixed frames ([`frame`](crate::frame) module docs)
//!   carrying opaque payloads; [`TransportService::send_request`] and
//!   [`TransportService::register_handler`] add bincode-typed actions on top.
//! - Every connection starts with a handshake exchanging [`NodeIdentity`];
//!   peers from another `cluster.name` are rejected.
//! - Failures are [`TransportError`]s; handler failures travel as
//!   [`RemoteError`] (`{type, reason, status, caused_by}`) so the REST layer
//!   can render them as ES would.
//! - Optional TLS (feature `tls`, on by default) mirrors
//!   `xpack.security.transport.ssl.*`, including mutual authentication.

mod config;
mod error;
mod frame;
pub use frame::{indexing_data, FrameAdmission};
mod service;
#[cfg(feature = "tls")]
mod tls;

pub use config::{ClientAuthentication, PemSource, TlsConfig, TlsDefaults, TransportConfig, TransportSettings, VerificationMode, TRANSPORT_SSL};
pub use error::{decode_admission_error, RemoteError, TransportError, BREAKING_FIELDS};
pub use service::{inbound_remote_peer, AdmittedDiscoveryNode, ConnectionEvent, InboundGate, NodeIdentity, RemoteCompress, RemoteConnectOptions, RemotePeer, TransportService, TransportStats};

mod control;
