//! HTTPS on the REST listener (`xpack.security.http.ssl.*`).
//!
//! [`TlsListener`] is an [`axum::serve::Listener`]: a background task
//! accepts TCP connections and runs each TLS handshake in its own task, so
//! a slow or failing handshake (a plaintext client, an untrusted client
//! certificate, a port scanner) only drops that connection and never holds
//! up the accept loop. Completed streams are served by `axum::serve` exactly
//! as plaintext ones are (same router, same per-connection
//! [`crate::HttpConnection`] bookkeeping).
//!
//! The server config is a [`Reloadable`]: each accepted connection
//! handshakes with the config current at accept time, so a certificate
//! rotated on disk (ES `SSLConfigurationReloader`) applies to new
//! connections while established ones keep theirs.

use elasticsearch_ssl::rustls::ServerConfig;
use elasticsearch_ssl::Reloadable;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::server::TlsStream;
use tokio_rustls::TlsAcceptor;

/// Netty's `SslHandler` default handshake timeout, which ES keeps.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// A TCP listener whose connections are TLS-terminated before serving.
pub struct TlsListener {
    rx: mpsc::Receiver<(TlsStream<TcpStream>, SocketAddr)>,
    local: SocketAddr,
}

impl TlsListener {
    /// Start accepting on `listener` with the server config `config`
    /// currently holds (must be called inside a tokio runtime). Accepting
    /// stops once the listener is dropped.
    pub fn new(listener: TcpListener, config: Arc<Reloadable<ServerConfig>>) -> std::io::Result<TlsListener> {
        let local = listener.local_addr()?;
        let (tx, rx) = mpsc::channel(256);
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted,
                    _ = tx.closed() => return,
                };
                let (tcp, remote) = match accepted {
                    Ok(c) => c,
                    Err(e) => {
                        // As axum's own accept loop: per-connection errors are
                        // retried at once, resource errors (EMFILE) after a pause.
                        if !matches!(e.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::ConnectionReset) {
                            tracing::error!(error = %e, "http accept error");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        continue;
                    }
                };
                // ES `http.tcp.no_delay` (default true), as for plaintext.
                let _ = tcp.set_nodelay(true);
                // The config current now (an atomic load, no lock).
                let acceptor = TlsAcceptor::from(config.load());
                let tx = tx.clone();
                tokio::spawn(async move {
                    match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(tcp)).await {
                        Ok(Ok(tls)) => {
                            let _ = tx.send((tls, remote)).await;
                        }
                        // ES logs these at debug/warn and closes the channel
                        // ("received plaintext http traffic on an https channel").
                        Ok(Err(e)) => tracing::debug!(%remote, error = %e, "http TLS handshake failed, closing connection"),
                        Err(_) => tracing::debug!(%remote, "http TLS handshake timed out, closing connection"),
                    }
                });
            }
        });
        Ok(TlsListener { rx, local })
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.rx.recv().await {
            Some(conn) => conn,
            // The accept task only ends with the listener; never resolve.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Ok(self.local)
    }
}
