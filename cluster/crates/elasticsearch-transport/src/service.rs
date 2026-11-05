//! `TransportService`: the node-to-node RPC endpoint (ES's `TransportService`
//! + `TcpTransport` + `ConnectionManager` in one).
//!
//! Design:
//! - **One listener, many connections.** Each node binds one TCP listener.
//!   Outbound connections are pooled: at most one live connection per remote
//!   address, opened lazily on first use (or explicitly via `connect`) and
//!   re-opened lazily after it closes. All requests to a peer are multiplexed
//!   over that connection and matched to responses by request id, so
//!   responses may arrive in any order.
//! - **Two tasks per connection.** A reader task decodes frames, completes
//!   pending requests and spawns one task per inbound request (a slow handler
//!   never blocks the connection). A writer task owns the socket's write half
//!   and drains a bounded queue of encoded frames, batching flushes.
//! - **Handshake first.** The opener sends its `NodeIdentity`; the acceptor
//!   answers with its own. Each side checks `cluster.name`; the acceptor
//!   refuses requests on a connection until a valid handshake arrived. The
//!   opener learns who is listening at an address (what discovery needs).
//! - **Events.** Establishing / losing an outbound connection is broadcast as
//!   a [`ConnectionEvent`]; that is the "node disconnected" signal higher
//!   layers (fault detection, replication) react to. Inbound connections do
//!   not produce events: peers that care about us connect to us.
//! - **Local short-circuit.** A request addressed to our own address runs the
//!   handler directly (spawned, with the same timeout and error semantics)
//!   without touching TCP, as ES does.
//! - **Closing** a connection fails all its in-flight requests with
//!   `NodeDisconnected`; `close()` on the service stops accepting, closes
//!   every connection and makes later sends fail with `Closed`.

use crate::config::{TransportConfig, TransportSettings};
use crate::error::{ACTION_NOT_FOUND_TYPE, RemoteError, TransportError};
use crate::frame::{self, FLAG_COMPRESSED, FLAG_ERROR, FLAG_HANDSHAKE, FLAG_PING, FLAG_REQUEST, Frame, HANDSHAKE_ACTION, PING_ACTION};
use bytes::Bytes;
use elasticsearch_cluster::state::DiscoveryNode;
use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::{Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufWriter, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tracing::{debug, warn};

/// What each side sends in the handshake.
/// A received DiscoveryNode whose owned strings and containers remain charged
/// through every clone. Serialize has the original DiscoveryNode wire layout.
#[derive(Clone,Debug)]
pub struct AdmittedDiscoveryNode(Arc<DiscoveryNodeOwner>);
#[derive(Debug)]
pub(crate) struct DiscoveryNodeOwner {node:DiscoveryNode,_owner:Arc<crate::control::Owner>}
impl AdmittedDiscoveryNode {
    pub(crate) fn new(node:DiscoveryNode,owner:Arc<crate::control::Owner>)->Self {Self(Arc::new(DiscoveryNodeOwner{node,_owner:owner}))}
}
impl std::ops::Deref for AdmittedDiscoveryNode {type Target=DiscoveryNode;fn deref(&self)->&DiscoveryNode{&self.0.node}}
impl Serialize for AdmittedDiscoveryNode {fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{self.0.node.serialize(s)}}

#[derive(Clone, Debug)]
pub struct NodeIdentity(Arc<IdentityOwner>);
#[derive(Debug)]
pub(crate) struct IdentityOwner { data: NodeIdentityData, _owner: Option<Arc<crate::control::Owner>> }
/// Immutable handshake fields. Clones of `NodeIdentity` share these fields and
/// their allocation owner instead of deep-copying uncharged discovery data.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeIdentityData {
    pub cluster_name: String,
    pub node: DiscoveryNode,
}
impl std::ops::Deref for NodeIdentity {
    type Target = NodeIdentityData;
    fn deref(&self) -> &Self::Target { &self.0.data }
}
impl PartialEq for NodeIdentity { fn eq(&self, other: &Self) -> bool { self.0.data == other.0.data } }
impl Eq for NodeIdentity {}
impl Serialize for NodeIdentity {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok,S::Error> { self.0.data.serialize(s) }
}
impl NodeIdentity {
    pub(crate) fn from_data(data: NodeIdentityData, owner: Option<Arc<crate::control::Owner>>) -> Self {
        Self(Arc::new(IdentityOwner { data, _owner: owner }))
    }
}

/// Outbound connection lifecycle notifications.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionEvent {
    /// A connection to `addr` completed its handshake.
    Connected { addr: SocketAddr, node: NodeIdentity },
    /// An established connection to `addr` closed (peer went away, I/O error,
    /// failed ping, `disconnect`, or `close`).
    Disconnected { addr: SocketAddr, node: NodeIdentity },
}

/// The `transport` section of `_nodes/stats`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TransportStats {
    /// Currently open inbound connections.
    pub server_open: u64,
    /// Outbound connections ever opened (ES's counter is cumulative too).
    pub total_outbound_connections: u64,
    pub rx_count: u64,
    pub rx_size_in_bytes: u64,
    pub tx_count: u64,
    pub tx_size_in_bytes: u64,
    /// Currently open outbound connections (not part of ES's output).
    #[serde(skip_serializing)]
    pub outbound_open: u64,
}

/// A raw handler: payload in, payload or failure out.
type RawHandler = Arc<dyn Fn(Bytes) -> BoxFuture<'static, Result<Bytes, RemoteError>> + Send + Sync>;

/// Anything we can run the protocol over (plain TCP or a TLS stream).
trait Io: AsyncRead + AsyncWrite + Send + Unpin + 'static {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin + 'static> Io for T {}
type BoxIo = Box<dyn Io>;

/// Pool slot for one remote address. The async mutex serializes connection
/// attempts so concurrent first requests share one connection.
type Slot = Arc<tokio::sync::Mutex<Option<Arc<Connection>>>>;

#[derive(Default)]
struct Counters {
    rx_count: AtomicU64,
    rx_bytes: AtomicU64,
    tx_count: AtomicU64,
    tx_bytes: AtomicU64,
    outbound_total: AtomicU64,
    outbound_open: AtomicU64,
}

/// Admission of inbound request payloads (ES charges them to the
/// `inflight_requests` breaker): given the action and payload bytes, a
/// guard held until the handler's response is queued, or the failure the
/// request is answered with.
pub type InboundGate = Arc<dyn Fn(&str, usize) -> Result<Box<dyn std::any::Any + Send>, RemoteError> + Send + Sync>;

struct Inner {
    local: NodeIdentity,
    bound_addr: SocketAddr,
    publish_addr: SocketAddr,
    settings: TransportSettings,
    frame_admission: Option<crate::frame::FrameAdmission>,
    handlers: RwLock<HashMap<String, RawHandler>>,
    outbound: Mutex<HashMap<SocketAddr, Slot>>,
    inbound: Mutex<HashMap<u64, Arc<Connection>>>,
    next_request_id: AtomicU64,
    next_conn_id: AtomicU64,
    counters: Counters,
    events: broadcast::Sender<ConnectionEvent>,
    shutdown: watch::Sender<bool>,
    gate: RwLock<Option<InboundGate>>,
    #[cfg(feature = "tls")]
    tls: Option<crate::tls::TlsState>,
}

impl Inner {
    fn is_closed(&self) -> bool {
        *self.shutdown.borrow()
    }

    /// `indexing`: the payload is indexing data whatever its action (a
    /// forwarded write, the response to a compressed request).
    fn compress_threshold(&self, action: &str, indexing: bool) -> Option<usize> {
        (self.settings.compress && (self.settings.compress_all || indexing || frame::indexing_data(action))).then_some(self.settings.compress_threshold)
    }

    fn encode(&self, flags: u8, id: u64, action: &str, payload: &[u8]) -> Result<Bytes, TransportError> {
        self.encode_as(flags, id, action, payload, false)
    }

    fn encode_as(&self, flags: u8, id: u64, action: &str, payload: &[u8], indexing: bool) -> Result<Bytes, TransportError> {
        frame::encode_admitted(flags, id, action, payload, self.compress_threshold(action, indexing), self.settings.max_frame_size, self.frame_admission.as_ref())
    }
}

// ---------------------------------------------------------- connection ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Inbound,
    Outbound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnState {
    Handshaking,
    Established,
    Closed,
}

struct Pending {
    action: String,
    tx: oneshot::Sender<Result<Bytes, TransportError>>,
}

struct Connection {
    id: u64,
    /// Remote address: the dialed address (outbound) or the peer's ephemeral
    /// socket address (inbound).
    addr: SocketAddr,
    direction: Direction,
    remote: OnceLock<NodeIdentity>,
    state: Mutex<ConnState>,
    writer: mpsc::Sender<Bytes>,
    pending: Mutex<HashMap<u64, Pending>>,
    /// Flipped to `true` exactly once; the reader/writer/ping tasks watch it.
    closed: watch::Sender<bool>,
}

/// Removes a pending entry when the requesting future finishes or is dropped
/// (timeouts, cancelled callers).
struct PendingGuard<'a> {
    conn: &'a Connection,
    id: u64,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        self.conn.pending.lock().remove(&self.id);
    }
}

impl Connection {
    /// Wraps an open stream and starts its reader and writer tasks.
    fn start(inner: &Arc<Inner>, io: BoxIo, addr: SocketAddr, direction: Direction) -> Arc<Connection> {
        let (rd, wr) = tokio::io::split(io);
        let (tx, rx) = mpsc::channel(1024);
        let conn = Arc::new(Connection {
            id: inner.next_conn_id.fetch_add(1, Relaxed),
            addr,
            direction,
            remote: OnceLock::new(),
            state: Mutex::new(ConnState::Handshaking),
            writer: tx,
            pending: Mutex::new(HashMap::new()),
            closed: watch::channel(false).0,
        });
        if direction == Direction::Inbound {
            inner.inbound.lock().insert(conn.id, conn.clone());
        }
        tokio::spawn(writer_loop(inner.clone(), conn.clone(), wr, rx));
        tokio::spawn(reader_loop(inner.clone(), conn.clone(), rd));
        // `close()` may have drained the connection maps just before we
        // registered; re-check after registration.
        if inner.is_closed() {
            conn.close(inner);
        }
        conn
    }

    fn is_closed(&self) -> bool {
        *self.closed.borrow()
    }

    fn is_established(&self) -> bool {
        *self.state.lock() == ConnState::Established
    }

    fn disconnected(&self, action: &str) -> TransportError {
        TransportError::NodeDisconnected { addr: self.addr, action: action.to_string() }
    }

    /// Handshake done. Returns false if the connection closed meanwhile.
    fn mark_established(&self, inner: &Inner) -> bool {
        let mut st = self.state.lock();
        if *st != ConnState::Handshaking || inner.is_closed() {
            return false;
        }
        *st = ConnState::Established;
        if self.direction == Direction::Outbound {
            inner.counters.outbound_total.fetch_add(1, Relaxed);
            inner.counters.outbound_open.fetch_add(1, Relaxed);
            if let Some(node) = self.remote.get() {
                let _ = inner.events.send(ConnectionEvent::Connected { addr: self.addr, node: node.clone() });
            }
        }
        true
    }

    /// Idempotent. Fails in-flight requests, unregisters the connection and
    /// stops its tasks (which drops the socket).
    fn close(&self, inner: &Inner) {
        let prev = std::mem::replace(&mut *self.state.lock(), ConnState::Closed);
        if prev == ConnState::Closed {
            return;
        }
        // Flag first, then drain: a request registering concurrently either
        // gets drained here or sees the flag after inserting.
        self.closed.send_replace(true);
        let pending: Vec<Pending> = self.pending.lock().drain().map(|(_, p)| p).collect();
        for p in pending {
            let _ = p.tx.send(Err(self.disconnected(&p.action)));
        }
        match self.direction {
            Direction::Inbound => {
                inner.inbound.lock().remove(&self.id);
            }
            Direction::Outbound => {
                if prev == ConnState::Established {
                    inner.counters.outbound_open.fetch_sub(1, Relaxed);
                    if let Some(node) = self.remote.get() {
                        let _ = inner.events.send(ConnectionEvent::Disconnected { addr: self.addr, node: node.clone() });
                    }
                }
                // Clear the pool slot if it still points at us (if it is
                // locked, a connector is busy and will notice we're closed).
                let slot = inner.outbound.lock().get(&self.addr).cloned();
                if let Some(slot) = slot {
                    if let Ok(mut g) = slot.try_lock() {
                        if g.as_ref().is_some_and(|c| c.id == self.id) {
                            *g = None;
                        }
                    }
                }
            }
        }
        debug!(addr = %self.addr, direction = ?self.direction, "transport connection closed");
    }

    /// Sends a request and waits for its response (or the timeout).
    async fn request(&self, inner: &Inner, flags: u8, action: &str, payload: &[u8], timeout: Option<Duration>) -> Result<Bytes, TransportError> {
        self.request_as(inner, flags, action, payload, timeout, false).await
    }

    async fn request_as(&self, inner: &Inner, flags: u8, action: &str, payload: &[u8], timeout: Option<Duration>, indexing: bool) -> Result<Bytes, TransportError> {
        let id = inner.next_request_id.fetch_add(1, Relaxed);
        let frame = inner.encode_as(FLAG_REQUEST | flags, id, action, payload, indexing)?;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id, Pending { action: action.to_string(), tx });
        let _guard = PendingGuard { conn: self, id };
        if self.is_closed() {
            return Err(self.disconnected(action));
        }
        let exchange = async {
            self.writer.send(frame).await.map_err(|_| self.disconnected(action))?;
            rx.await.unwrap_or_else(|_| Err(self.disconnected(action)))
        };
        match timeout {
            None => exchange.await,
            Some(t) => tokio::time::timeout(t, exchange).await.unwrap_or_else(|_| {
                Err(TransportError::ReceiveTimeout { addr: self.addr, action: action.to_string(), request_id: id, timeout_ms: t.as_millis() as u64 })
            }),
        }
    }

    /// Queues a response frame; a failure to encode the success payload
    /// (e.g. too large) is reported to the caller as a remote failure.
    async fn respond(&self, inner: &Inner, id: u64, flags: u8, result: Result<Bytes, RemoteError>) {
        self.respond_as(inner, id, flags, result, false).await
    }

    /// [`Self::respond`]; `indexing`: the request came compressed, so the
    /// response is (ES `InboundHandler`: a response follows its request).
    async fn respond_as(&self, inner: &Inner, id: u64, flags: u8, result: Result<Bytes, RemoteError>, indexing: bool) {
        let frame = match result {
            Ok(payload) => inner.encode_as(flags, id, "", &payload, indexing).or_else(|e| encode_error(inner, id, &e.to_remote_error())),
            Err(e) => encode_error(inner, id, &e),
        };
        match frame {
            Ok(f) => {
                let _ = self.writer.send(f).await;
            }
            Err(e) => warn!(addr = %self.addr, "failed to encode transport response: {e}"),
        }
    }

    /// Handles one decoded frame on the reader task.
    async fn on_frame(self: &Arc<Self>, inner: &Arc<Inner>, frame: Frame) {
        if !frame.is_request() {
            let pending = self.pending.lock().remove(&frame.request_id);
            let Some(p) = pending else {
                debug!(addr = %self.addr, id = frame.request_id, "response for unknown or timed-out request");
                return;
            };
            let result = if frame.has(FLAG_ERROR) { Err(decode_error(&frame.payload, &p.action, inner.frame_admission.as_ref())) } else { Ok(frame.payload) };
            let _ = p.tx.send(result);
            return;
        }
        if frame.has(FLAG_PING) {
            self.respond(inner, frame.request_id, FLAG_PING, Ok(Bytes::new())).await;
            return;
        }
        if frame.has(FLAG_HANDSHAKE) {
            self.on_handshake(inner, frame).await;
            return;
        }
        if !self.is_established() {
            let err = RemoteError::new("illegal_state_exception", "transport handshake required before sending requests", 500);
            self.respond(inner, frame.request_id, 0, Err(err)).await;
            self.close(inner);
            return;
        }
        let handler = inner.handlers.read().get(&frame.action).cloned();
        let gate = inner.gate.read().clone();
        let admitted = match gate.map(|g| g(&frame.action, frame.payload.len())).transpose() {
            Ok(guard) => guard,
            Err(e) => {
                self.respond(inner, frame.request_id, 0, Err(e)).await;
                return;
            }
        };
        let conn = self.clone();
        let inner = inner.clone();
        let compressed = frame.has(FLAG_COMPRESSED);
        tokio::spawn(async move {
            let result = match handler {
                Some(h) => run_handler(h, frame.payload).await,
                None => Err(action_not_found(&frame.action)),
            };
            conn.respond_as(&inner, frame.request_id, 0, result, compressed).await;
            drop(admitted);
        });
    }

    /// Acceptor side of the handshake.
    async fn on_handshake(self: &Arc<Self>, inner: &Arc<Inner>, frame: Frame) {
        let remote: NodeIdentity = match crate::control::decode_identity(&frame.payload, inner.frame_admission.as_ref()) {
            Ok(r) => r,
            Err(e) => {
                let err = match e {
                    TransportError::Remote(error)=>error,
                    other=>RemoteError::new("transport_serialization_exception",format!("invalid handshake: {other}"),500),
                };
                self.respond(inner, frame.request_id, FLAG_HANDSHAKE, Err(err)).await;
                self.close(inner);
                return;
            }
        };
        // Always answer with our identity: the opener produces the precise
        // cluster-name error from it.
        let me = match frame::serialize_admitted(&inner.local, inner.frame_admission.as_ref()) {
            Ok(me) => me,
            Err(error) => { self.respond(inner, frame.request_id, FLAG_HANDSHAKE, Err(error.to_remote_error())).await; self.close(inner); return; }
        };
        self.respond(inner, frame.request_id, FLAG_HANDSHAKE, Ok(me)).await;
        if remote.cluster_name != inner.local.cluster_name {
            warn!(
                addr = %self.addr,
                "rejecting connection from node [{}] of cluster [{}]: local cluster name is [{}]",
                remote.node.name, remote.cluster_name, inner.local.cluster_name
            );
            // Give the response a moment to flush, then drop the connection.
            let (conn, inner) = (self.clone(), inner.clone());
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(1)).await;
                conn.close(&inner);
            });
            return;
        }
        let _ = self.remote.set(remote);
        self.mark_established(inner);
    }
}

fn encode_error(inner: &Inner, id: u64, err: &RemoteError) -> Result<Bytes, TransportError> {
    let payload = frame::serialize_admitted(err, inner.frame_admission.as_ref())?;
    inner.encode(FLAG_ERROR, id, "", &payload)
}

fn decode_error(payload: &[u8], action: &str, admission: Option<&crate::frame::FrameAdmission>) -> TransportError {
    match crate::control::decode_error(payload, admission) {
        Ok(e) if e.error_type == ACTION_NOT_FOUND_TYPE => TransportError::ActionNotFound { action: action.to_string() },
        Ok(e) => TransportError::Remote(e),
        Err(e) => e,
    }
}

fn action_not_found(action: &str) -> RemoteError {
    RemoteError::new(ACTION_NOT_FOUND_TYPE, format!("No handler for action [{action}]"), 500)
}

/// Runs a handler, turning a panic into a failure response.
async fn run_handler(h: RawHandler, payload: Bytes) -> Result<Bytes, RemoteError> {
    AssertUnwindSafe(async move { h(payload).await }).catch_unwind().await.unwrap_or_else(|panic| {
        let msg = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "handler panicked".into());
        Err(RemoteError::internal(msg))
    })
}

/// Resolves once the watched flag is `true` (or its sender is gone).
async fn wait_closed(rx: &mut watch::Receiver<bool>) {
    let _ = rx.wait_for(|c| *c).await;
}

async fn reader_loop(inner: Arc<Inner>, conn: Arc<Connection>, mut rd: ReadHalf<BoxIo>) {
    let mut closed = conn.closed.subscribe();
    loop {
        let res = tokio::select! {
            r = frame::read_frame_admitted(&mut rd, inner.settings.max_frame_size, inner.frame_admission.as_ref()) => r,
            _ = wait_closed(&mut closed) => break,
        };
        match res {
            Ok(Some((frame, size))) => {
                inner.counters.rx_count.fetch_add(1, Relaxed);
                inner.counters.rx_bytes.fetch_add(size as u64, Relaxed);
                conn.on_frame(&inner, frame).await;
            }
            Ok(None) => break,
            Err(e) => {
                debug!(addr = %conn.addr, "transport read failed: {e}");
                break;
            }
        }
    }
    conn.close(&inner);
}

async fn writer_loop(inner: Arc<Inner>, conn: Arc<Connection>, wr: WriteHalf<BoxIo>, mut rx: mpsc::Receiver<Bytes>) {
    let mut closed = conn.closed.subscribe();
    let mut w = BufWriter::with_capacity(64 * 1024, wr);
    loop {
        let first = tokio::select! {
            biased;
            f = rx.recv() => match f { Some(f) => f, None => break },
            _ = wait_closed(&mut closed) => break,
        };
        // Write everything already queued, then flush once.
        let batch = async {
            let mut next = Some(first);
            while let Some(f) = next {
                w.write_all(&f).await?;
                inner.counters.tx_count.fetch_add(1, Relaxed);
                inner.counters.tx_bytes.fetch_add(f.len() as u64, Relaxed);
                next = rx.try_recv().ok();
            }
            w.flush().await
        };
        let res = tokio::select! {
            r = batch => r,
            _ = wait_closed(&mut closed) => break,
        };
        if let Err(e) = res {
            debug!(addr = %conn.addr, "transport write failed: {e}");
            break;
        }
    }
    let _ = tokio::time::timeout(Duration::from_millis(500), w.shutdown()).await;
    conn.close(&inner);
}

/// Pings an outbound connection every `interval`; closes it on failure.
async fn ping_loop(inner: Arc<Inner>, conn: Arc<Connection>, interval: Duration) {
    let mut closed = conn.closed.subscribe();
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = wait_closed(&mut closed) => return,
        }
        if let Err(e) = conn.request(&inner, FLAG_PING, PING_ACTION, &[], Some(interval)).await {
            warn!(addr = %conn.addr, "transport ping failed, closing connection: {e}");
            conn.close(&inner);
            return;
        }
    }
}

async fn accept_loop(weak: Weak<Inner>, listener: TcpListener, mut shutdown: watch::Receiver<bool>) {
    loop {
        let (tcp, peer) = tokio::select! {
            r = listener.accept() => match r {
                Ok(x) => x,
                Err(e) => {
                    // e.g. EMFILE: back off instead of spinning.
                    warn!("transport accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            _ = wait_closed(&mut shutdown) => return,
        };
        let Some(inner) = weak.upgrade() else { return };
        let _ = tcp.set_nodelay(true);
        tokio::spawn(async move {
            let deadline = inner.settings.connect_timeout;
            let Some(io) = wrap_inbound(&inner, tcp, peer).await else { return };
            let conn = Connection::start(&inner, io, peer, Direction::Inbound);
            // A peer that never handshakes is dropped after connect_timeout.
            tokio::time::sleep(deadline).await;
            if *conn.state.lock() == ConnState::Handshaking {
                debug!(addr = %peer, "no transport handshake within [{deadline:?}], closing");
                conn.close(&inner);
            }
        });
    }
}

#[cfg(feature = "tls")]
async fn wrap_inbound(inner: &Inner, tcp: TcpStream, peer: SocketAddr) -> Option<BoxIo> {
    match &inner.tls {
        None => Some(Box::new(tcp)),
        Some(tls) => match tokio::time::timeout(inner.settings.connect_timeout, tls.accept(tcp)).await {
            Ok(Ok(s)) => Some(Box::new(s)),
            Ok(Err(e)) => {
                warn!(addr = %peer, "transport TLS handshake failed: {e}");
                None
            }
            Err(_) => None,
        },
    }
}

#[cfg(not(feature = "tls"))]
async fn wrap_inbound(_inner: &Inner, tcp: TcpStream, _peer: SocketAddr) -> Option<BoxIo> {
    Some(Box::new(tcp))
}

#[cfg(feature = "tls")]
async fn wrap_outbound(inner: &Inner, tcp: TcpStream, addr: SocketAddr) -> std::io::Result<BoxIo> {
    match &inner.tls {
        None => Ok(Box::new(tcp)),
        Some(tls) => Ok(Box::new(tls.connect(addr, tcp).await?)),
    }
}

#[cfg(not(feature = "tls"))]
async fn wrap_outbound(_inner: &Inner, tcp: TcpStream, _addr: SocketAddr) -> std::io::Result<BoxIo> {
    Ok(Box::new(tcp))
}

// ------------------------------------------------------------- service ----

/// The transport endpoint of one node. Cheap to clone; all clones share the
/// listener, connection pool and handler registry.
#[derive(Clone)]
pub struct TransportService {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for TransportService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportService")
            .field("node", &self.inner.local.node.name)
            .field("cluster", &self.inner.local.cluster_name)
            .field("address", &self.inner.publish_addr)
            .finish()
    }
}

impl TransportService {
    /// Make an independently admitted immutable copy before any owned node
    /// strings or containers are cloned (including local control responses).
    pub fn admit_discovery_node(&self,node:&DiscoveryNode)->Result<AdmittedDiscoveryNode,TransportError>{
        crate::control::copy_node(node,self.inner.frame_admission.as_ref())
    }
    /// Decode a same-wire DiscoveryNode response with retained structural
    /// admission. Intended for raw control actions such as capability probes.
    pub fn decode_discovery_node(&self,payload:&[u8])->Result<AdmittedDiscoveryNode,TransportError>{
        crate::control::decode_node(payload,self.inner.frame_admission.as_ref())
    }
    /// Construct an owned control error without copying borrowed text before admission.
    pub fn admitted_error(&self,kind:&str,reason:&str,status:u16)->Result<RemoteError,TransportError>{
        #[derive(serde::Serialize)]
        struct Error<'a>{error_type:&'a str,reason:&'a str,status:u16,metadata:std::collections::BTreeMap<String,String>,caused_by:Option<Box<RemoteError>>}
        let data=Error{error_type:kind,reason,status,metadata:Default::default(),caused_by:None};
        let bytes=frame::serialize_admitted(&data,self.inner.frame_admission.as_ref())?;
        crate::control::decode_error(&bytes,self.inner.frame_admission.as_ref())
    }

    /// Binds the listener and starts accepting connections.
    pub async fn bind(config: TransportConfig) -> Result<TransportService, TransportError> {
        Self::bind_with_admission(config, None).await
    }

    /// Node embeddings supply shared admission before any connection is accepted.
    /// `None` explicitly opts standalone transport users out of memory admission.
    pub async fn bind_with_admission(config: TransportConfig, frame_admission: Option<crate::frame::FrameAdmission>) -> Result<TransportService, TransportError> {
        #[cfg(feature = "tls")]
        let tls = config.tls.as_ref().map(crate::tls::TlsState::new).transpose()?;
        #[cfg(not(feature = "tls"))]
        if config.tls.is_some() {
            return Err(TransportError::Config("TLS configured but the `tls` feature is disabled".into()));
        }

        let bind_err = |e: std::io::Error| TransportError::Config(format!("failed to bind transport to [{}]: {e}", config.bind_addr));
        let socket = if config.bind_addr.is_ipv4() { TcpSocket::new_v4() } else { TcpSocket::new_v6() }.map_err(bind_err)?;
        // Lets a restarted node rebind its port while old sockets linger in
        // TIME_WAIT.
        socket.set_reuseaddr(true).map_err(bind_err)?;
        socket.bind(config.bind_addr).map_err(bind_err)?;
        let listener = socket.listen(1024).map_err(bind_err)?;
        let bound_addr = listener.local_addr().map_err(bind_err)?;
        let publish_addr = config.publish_addr.unwrap_or(bound_addr);

        let mut node = config.local_node;
        node.transport_address = publish_addr.to_string();
        if node.ip.is_empty() {
            node.ip = publish_addr.ip().to_string();
        }
        if node.host.is_empty() {
            node.host = publish_addr.ip().to_string();
        }

        let inner = Arc::new(Inner {
            local: NodeIdentity::from_data(NodeIdentityData { cluster_name: config.cluster_name, node }, None),
            bound_addr,
            publish_addr,
            settings: config.settings,
            frame_admission,
            handlers: RwLock::new(HashMap::new()),
            outbound: Mutex::new(HashMap::new()),
            inbound: Mutex::new(HashMap::new()),
            next_request_id: AtomicU64::new(1),
            next_conn_id: AtomicU64::new(1),
            counters: Counters::default(),
            events: broadcast::channel(1024).0,
            shutdown: watch::channel(false).0,
            gate: RwLock::new(None),
            #[cfg(feature = "tls")]
            tls,
        });
        tokio::spawn(accept_loop(Arc::downgrade(&inner), listener, inner.shutdown.subscribe()));
        debug!(addr = %bound_addr, "transport bound");
        Ok(TransportService { inner })
    }

    /// The transport's SSL context (`xpack.security.transport.ssl`), for
    /// the node's SSL file watcher; `None` without transport TLS.
    #[cfg(feature = "tls")]
    pub fn ssl_reloader(&self) -> Option<std::sync::Arc<dyn elasticsearch_ssl::Reload>> {
        self.inner.tls.as_ref().map(crate::tls::TlsState::reloadable)
    }

    // ---- identity ----

    /// Install the admission of inbound request payloads.
    pub fn set_inbound_gate(&self, gate: InboundGate) {
        *self.inner.gate.write() = Some(gate);
    }

    /// This node as advertised in handshakes (with its publish address).
    pub fn local_node(&self) -> &DiscoveryNode {
        &self.inner.local.node
    }

    pub fn identity(&self) -> &NodeIdentity {
        &self.inner.local
    }

    pub fn cluster_name(&self) -> &str {
        &self.inner.local.cluster_name
    }

    /// The address the listener is actually bound to (real port if 0 was
    /// requested).
    pub fn bound_address(&self) -> SocketAddr {
        self.inner.bound_addr
    }

    /// The address advertised to peers (`transport.publish_*`).
    pub fn publish_address(&self) -> SocketAddr {
        self.inner.publish_addr
    }

    pub fn settings(&self) -> &TransportSettings {
        &self.inner.settings
    }

    /// Whether `addr` is this node (requests to it short-circuit).
    pub fn is_local_address(&self, addr: SocketAddr) -> bool {
        let b = self.inner.bound_addr;
        addr == b || addr == self.inner.publish_addr || (b.ip().is_unspecified() && addr.port() == b.port() && addr.ip().is_loopback())
    }

    // ---- handlers ----

    /// Registers (or replaces) the handler for `action`. Each request runs in
    /// its own task; a panic becomes a 500 `exception` failure.
    pub fn register_raw_handler<F, Fut>(&self, action: &str, f: F)
    where
        F: Fn(Bytes) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Bytes, RemoteError>> + Send + 'static,
    {
        let h: RawHandler = Arc::new(move |b| f(b).boxed());
        self.inner.handlers.write().insert(action.to_string(), h);
    }

    /// Typed variant of [`Self::register_raw_handler`]: requests and responses
    /// are bincode-encoded serde types.
    pub fn register_handler<Req, Resp, F, Fut>(&self, action: &str, f: F)
    where
        Req: DeserializeOwned + Send + 'static,
        Resp: Serialize + Send + 'static,
        F: Fn(Req) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Resp, RemoteError>> + Send + 'static,
    {
        let f = Arc::new(f);
        let action_name = action.to_string();
        let admission = self.inner.frame_admission.clone();
        self.register_raw_handler(action, move |payload: Bytes| {
            let f = f.clone();
            let action = action_name.clone();
            let admission = admission.clone();
            async move {
                let req: Req = bincode::deserialize(&payload).map_err(|e| crate::error::typed_decode_error(e, &action))?;
                let resp = f(req).await?;
                frame::serialize_admitted(&resp, admission.as_ref()).map_err(|e| e.to_remote_error())
            }
        });
    }

    pub fn unregister_handler(&self, action: &str) -> bool {
        self.inner.handlers.write().remove(action).is_some()
    }

    pub fn has_handler(&self, action: &str) -> bool {
        self.inner.handlers.read().contains_key(action)
    }

    // ---- connections ----

    /// Connects to `addr` (or reuses the pooled connection) and returns the
    /// remote node's identity from the handshake. For discovery: dial a seed
    /// address and learn who is there. Connecting to ourselves returns our own
    /// identity without opening a socket.
    pub async fn connect(&self, addr: SocketAddr) -> Result<NodeIdentity, TransportError> {
        if self.is_local_address(addr) {
            return if self.inner.is_closed() { Err(TransportError::Closed) } else { Ok(self.inner.local.clone()) };
        }
        let conn = self.connection(addr).await?;
        Ok(conn.remote.get().cloned().expect("established connections have an identity"))
    }

    /// Identity of the node behind a currently open connection to `addr`.
    pub fn remote_identity(&self, addr: SocketAddr) -> Option<NodeIdentity> {
        self.live_connection(addr).and_then(|c| c.remote.get().cloned())
    }

    pub fn is_connected(&self, addr: SocketAddr) -> bool {
        self.live_connection(addr).is_some()
    }

    /// Every currently established outbound connection.
    pub fn connected_nodes(&self) -> Vec<(SocketAddr, NodeIdentity)> {
        let slots: Vec<(SocketAddr, Slot)> = self.inner.outbound.lock().iter().map(|(a, s)| (*a, s.clone())).collect();
        let mut out: Vec<(SocketAddr, NodeIdentity)> = slots
            .into_iter()
            .filter_map(|(a, s)| {
                let g = s.try_lock().ok()?;
                let c = g.as_ref().filter(|c| c.is_established())?;
                Some((a, c.remote.get()?.clone()))
            })
            .collect();
        out.sort_by_key(|(a, _)| *a);
        out
    }

    /// Closes the pooled connection to `addr` (fires `Disconnected`, fails its
    /// in-flight requests). The next request reconnects.
    pub async fn disconnect(&self, addr: SocketAddr) {
        let slot = self.inner.outbound.lock().get(&addr).cloned();
        if let Some(slot) = slot {
            let conn = slot.lock().await.take();
            if let Some(c) = conn {
                c.close(&self.inner);
            }
        }
    }

    /// Subscribes to outbound connect/disconnect events.
    pub fn subscribe(&self) -> broadcast::Receiver<ConnectionEvent> {
        self.inner.events.subscribe()
    }

    fn live_connection(&self, addr: SocketAddr) -> Option<Arc<Connection>> {
        let slot = self.inner.outbound.lock().get(&addr).cloned()?;
        let g = slot.try_lock().ok()?;
        g.as_ref().filter(|c| c.is_established()).cloned()
    }

    /// The pooled connection to `addr`, opening one if needed.
    async fn connection(&self, addr: SocketAddr) -> Result<Arc<Connection>, TransportError> {
        if self.inner.is_closed() {
            return Err(TransportError::Closed);
        }
        let slot = self.inner.outbound.lock().entry(addr).or_default().clone();
        let mut g = slot.lock().await;
        if let Some(c) = g.as_ref() {
            if c.is_established() {
                return Ok(c.clone());
            }
        }
        let timeout = self.inner.settings.connect_timeout;
        let conn = tokio::time::timeout(timeout, self.open_connection(addr)).await.unwrap_or_else(|_| {
            Err(TransportError::ConnectTransport { addr, reason: format!("connect_timeout[{}ms]", timeout.as_millis()) })
        })?;
        *g = Some(conn.clone());
        Ok(conn)
    }

    /// TCP connect + optional TLS + handshake.
    async fn open_connection(&self, addr: SocketAddr) -> Result<Arc<Connection>, TransportError> {
        let inner = &self.inner;
        let connect_err = |reason: String| TransportError::ConnectTransport { addr, reason };
        let tcp = TcpStream::connect(addr).await.map_err(|e| connect_err(e.to_string()))?;
        let _ = tcp.set_nodelay(true);
        let io = wrap_outbound(inner, tcp, addr).await.map_err(|e| connect_err(format!("TLS handshake failed: {e}")))?;
        let conn = Connection::start(inner, io, addr, Direction::Outbound);

        // Make sure the half-open connection dies with this future (timeout,
        // cancellation) unless the handshake succeeds.
        struct CloseOnDrop<'a>(Option<(Arc<Connection>, &'a Inner)>);
        impl Drop for CloseOnDrop<'_> {
            fn drop(&mut self) {
                if let Some((c, i)) = self.0.take() {
                    c.close(i);
                }
            }
        }
        let mut guard = CloseOnDrop(Some((conn.clone(), inner)));

        let hello = frame::serialize_admitted(&inner.local, inner.frame_admission.as_ref())?;
        let resp = conn.request(inner, FLAG_HANDSHAKE, HANDSHAKE_ACTION, &hello, None).await.map_err(|e| match e {
            TransportError::Remote(r) if r.status==429 => TransportError::Remote(r),
            TransportError::Remote(r) => connect_err(format!("handshake failed: {r}")),
            other => connect_err(format!("handshake failed: {other}")),
        })?;
        let remote = crate::control::decode_identity(&resp, inner.frame_admission.as_ref())?;
        if remote.cluster_name != inner.local.cluster_name {
            return Err(TransportError::HandshakeFailed {
                addr,
                reason: format!(
                    "remote cluster name [{}] does not match local cluster name [{}]",
                    remote.cluster_name, inner.local.cluster_name
                ),
            });
        }
        let _ = conn.remote.set(remote);
        if !conn.mark_established(inner) {
            return Err(conn.disconnected(HANDSHAKE_ACTION));
        }
        guard.0 = None;
        if let Some(interval) = inner.settings.ping_schedule {
            tokio::spawn(ping_loop(inner.clone(), conn.clone(), interval));
        }
        Ok(conn)
    }

    // ---- requests ----

    /// Sends `payload` to `action` on the node at `target` and waits for the
    /// response. `timeout` bounds the wait for the response (connecting is
    /// bounded separately by `connect_timeout`); `None` waits indefinitely.
    pub async fn send_request_bytes(
        &self,
        target: SocketAddr,
        action: &str,
        payload: Bytes,
        timeout: impl Into<Option<Duration>>,
    ) -> Result<Bytes, TransportError> {
        self.send_request_bytes_as(target, action, payload, timeout.into(), false).await
    }

    /// [`Self::send_request`] of a payload that is indexing data whatever
    /// its action (a forwarded bulk / document write): compressed as such
    /// (`transport.compress: indexing_data`), and so is its response.
    pub async fn send_indexing_request<Req, Resp>(&self, target: SocketAddr, action: &str, req: &Req, timeout: impl Into<Option<Duration>>) -> Result<Resp, TransportError>
    where
        Req: Serialize + ?Sized,
        Resp: DeserializeOwned,
    {
        let payload = frame::serialize_admitted(req, self.inner.frame_admission.as_ref())?;
        let resp = self.send_request_bytes_as(target, action, payload, timeout.into(), true).await?;
        bincode::deserialize(&resp).map_err(|e| crate::error::response_decode_error(e, action))
    }

    async fn send_request_bytes_as(&self, target: SocketAddr, action: &str, payload: Bytes, timeout: Option<Duration>, indexing: bool) -> Result<Bytes, TransportError> {
        if self.inner.is_closed() {
            return Err(TransportError::Closed);
        }
        if self.is_local_address(target) {
            return self.send_local(target, action, payload, timeout).await;
        }
        let conn = self.connection(target).await?;
        conn.request_as(&self.inner, 0, action, &payload, timeout, indexing).await
    }

    /// Typed request: bincode-encodes `req`, decodes the response as `Resp`.
    pub async fn send_request<Req, Resp>(
        &self,
        target: SocketAddr,
        action: &str,
        req: &Req,
        timeout: impl Into<Option<Duration>>,
    ) -> Result<Resp, TransportError>
    where
        Req: Serialize + ?Sized,
        Resp: DeserializeOwned,
    {
        let payload = frame::serialize_admitted(req, self.inner.frame_admission.as_ref())?;
        let resp = self.send_request_bytes(target, action, payload, timeout).await?;
        bincode::deserialize(&resp).map_err(|e| crate::error::response_decode_error(e, action))
    }

    /// Local short-circuit: run the handler in its own task, no TCP.
    async fn send_local(&self, target: SocketAddr, action: &str, payload: Bytes, timeout: Option<Duration>) -> Result<Bytes, TransportError> {
        let request_id = self.inner.next_request_id.fetch_add(1, Relaxed);
        let handler = self.inner.handlers.read().get(action).cloned();
        let Some(h) = handler else { return Err(TransportError::ActionNotFound { action: action.to_string() }) };
        let task = tokio::spawn(run_handler(h, payload));
        let joined = match timeout {
            None => task.await,
            Some(t) => match tokio::time::timeout(t, task).await {
                Ok(j) => j,
                Err(_) => {
                    return Err(TransportError::ReceiveTimeout { addr: target, action: action.to_string(), request_id, timeout_ms: t.as_millis() as u64 });
                }
            },
        };
        match joined {
            Ok(r) => r.map_err(TransportError::Remote),
            // The runtime aborts in-flight handlers while this node shuts
            // down. That is the local form of a connection closing under the
            // request, so callers retry it as a node failure, not a 500.
            Err(e) if e.is_cancelled() => Err(TransportError::NodeDisconnected { addr: target, action: action.to_string() }),
            Err(e) => Err(TransportError::Remote(RemoteError::internal(format!("local handler for [{action}] failed: {e}")))),
        }
    }

    // ---- stats / lifecycle ----

    pub fn stats(&self) -> TransportStats {
        let c = &self.inner.counters;
        TransportStats {
            server_open: self.inner.inbound.lock().len() as u64,
            total_outbound_connections: c.outbound_total.load(Relaxed),
            rx_count: c.rx_count.load(Relaxed),
            rx_size_in_bytes: c.rx_bytes.load(Relaxed),
            tx_count: c.tx_count.load(Relaxed),
            tx_size_in_bytes: c.tx_bytes.load(Relaxed),
            outbound_open: c.outbound_open.load(Relaxed),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Stops accepting, closes every connection (in-flight requests fail with
    /// `NodeDisconnected`) and makes further sends fail with `Closed`.
    /// Idempotent.
    pub async fn close(&self) {
        // (async for API stability: a future version may drain gracefully)
        if self.inner.shutdown.send_replace(true) {
            return;
        }
        let inbound: Vec<Arc<Connection>> = self.inner.inbound.lock().values().cloned().collect();
        for c in inbound {
            c.close(&self.inner);
        }
        let slots: Vec<Slot> = self.inner.outbound.lock().drain().map(|(_, s)| s).collect();
        for s in slots {
            // A slot locked by an in-progress connect is skipped: that connect
            // fails because `mark_established` re-checks the shutdown flag.
            if let Ok(mut g) = s.try_lock() {
                if let Some(c) = g.take() {
                    c.close(&self.inner);
                }
            }
        }
        debug!(addr = %self.inner.bound_addr, "transport closed");
    }
}
