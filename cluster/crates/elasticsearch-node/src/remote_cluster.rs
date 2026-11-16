//! Remote cluster connections (ES `RemoteClusterService`,
//! `RemoteClusterConnection`, `SniffConnectionStrategy`,
//! `ProxyConnectionStrategy`).
//!
//! Every node with the `remote_cluster_client` role keeps connections to the
//! remote clusters its `cluster.remote.<alias>.*` settings (cluster settings,
//! transient over persistent, then node settings) enable:
//!
//! * **sniff** (default): the seeds are tried in order. A transient
//!   remote-cluster connection to a seed handshakes (any cluster name; the
//!   first one learned is required from then on), asks the seed for its
//!   cluster's nodes ([`ACTION_CLUSTER_INFO`]) and opens one managed
//!   connection to each gateway node (not a dedicated master; with
//!   `cluster.remote.node.attr`, only nodes with that attribute `true`) up to
//!   `node_connections` (`cluster.remote.connections_per_cluster`, 3).
//! * **proxy**: `proxy_socket_connections` (18) sockets to `proxy_address`;
//!   a socket whose peer reports another cluster name than the first is
//!   refused.
//!
//! The connections are the transport's remote-cluster connections
//! ([`TransportService::connect_remote`]): their handshake does not require
//! this cluster's name, and the peer serves only the actions allowed to
//! remote clusters on them. Transport TLS applies as to any connection.
//!
//! A reconcile loop follows the settings (an alias that is no longer enabled
//! is closed, a changed mode / addresses / transport settings rebuild its
//! connections, `skip_unavailable` changes in place) and reconnects lost
//! connections, with exponential backoff (100ms doubling to 30s) between
//! failed attempts. A request that needs a remote it is not connected to
//! (`skip_unavailable: false`) tries to connect at once ([`ensure_connected`]).
//!
//! The live state is reported to the engine ([`RemoteConnections`]):
//! `_remote/info`, the planners and the connection errors read it.
//!
//! Requests from a remote cluster to this one arrive over the same
//! listener; [`register`] serves the actions allowed to them:
//! [`ACTION_CLUSTER_INFO`] (the sniff) and [`ACTION_REMOTE_REST`], which runs
//! an allow-listed read API through this node's front as the calling user
//! (the user's name and role names travel with the request; this cluster's
//! own role definitions authorize it, ES's certificate-based remote cluster
//! security). CCR registers its leader actions in `crate::ccr`.
//!
//! [`ensure_connected`]: RemoteClusterService::ensure_connected

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::Role;
use elasticsearch_engine::ccr::remote::{self as eremote, RemoteConfig, RemoteMode};
use elasticsearch_engine::error::EsError;
use elasticsearch_transport::{ConnectionEvent, NodeIdentity, RemoteCompress, RemoteConnectOptions, RemoteError, TransportError};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

/// A remote cluster's nodes and identity (the sniff; ES sends a
/// `cluster:monitor/state` request for the nodes).
pub const ACTION_CLUSTER_INFO: &str = "cluster:internal/remote/cluster_info";
/// Run an allow-listed REST read API as a remote cluster's user.
pub const ACTION_REMOTE_REST: &str = "internal:remote_cluster/rest";
/// Internal header: the request runs for a remote cluster (CCS) and answers
/// unmerged per-shard results (`front::ccs`).
pub const CCS_PARTS_HEADER: &str = "x-rustic-internal-ccs-parts";
/// Internal header: the request came from a remote cluster (its name).
pub const CCS_ORIGIN_HEADER: &str = "x-rustic-internal-ccs-origin";
/// Internal header of a CCS parts request: ES's DFS phase across clusters,
/// `collect` (answer the shards' term statistics) or `use:<hex JSON stats>`
/// (score with these global statistics).
pub const CCS_DFS_HEADER: &str = "x-rustic-internal-ccs-dfs";
/// Internal header: a retriever's final (rank plan apply) search.
pub const RANK_APPLY_HEADER: &str = "x-rustic-internal-rank-plan";
/// Internal header of a CCS parts request: the other clusters of the search
/// (each holds at least one shard), for ES's can_match pre-filter threshold
/// (`pre_filter_shard_size` counts every cluster's shards).
pub const CCS_OTHER_SHARDS_HEADER: &str = "x-rustic-internal-ccs-other-shards";

/// The transient connection to a seed (sniff) uses this socket index; managed
/// connections use 0 (sniff) or 0..sockets (proxy).
const SEED_SOCKET: u16 = u16::MAX;
const SNIFF_TIMEOUT: Duration = Duration::from_secs(30);
const BACKOFF_MIN: Duration = Duration::from_millis(100);
const BACKOFF_MAX: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------- wire types

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClusterInfoReq {
    /// The asking cluster's name (logged).
    pub from_cluster: String,
    /// On a remote cluster server port: the cross-cluster API key
    /// (`cluster.remote.<alias>.credentials`), required there.
    pub credential: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteNodeInfo {
    pub id: String,
    pub name: String,
    pub transport_address: String,
    pub roles: Vec<String>,
    pub attributes: BTreeMap<String, String>,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub cluster_name: String,
    pub cluster_uuid: String,
    pub version: String,
    pub nodes: Vec<RemoteNodeInfo>,
}

/// A REST read API a remote cluster runs here ([`ACTION_REMOTE_REST`]).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RemoteRestReq {
    pub method: String,
    /// Path and query string (`/idx/_search?size=0`).
    pub uri: String,
    pub body: Vec<u8>,
    /// The calling user (`StoredAuthn::encode`), authorized by this
    /// cluster's role definitions; `None` when the caller runs without
    /// security.
    pub identity: Option<String>,
    /// Answer unmerged per-shard results (CCS reduce on the caller).
    pub parts: bool,
    /// A DFS round of a parts request ([`CCS_DFS_HEADER`]'s value).
    pub dfs: Option<String>,
    /// A retriever search's final phase (the body carries the rank plan).
    pub rank_apply: bool,
    /// Shards (at least) the search reads on other clusters.
    pub other_shards: u32,
}

/// The answer to a [`RemoteRestReq`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemoteRestResp {
    pub status: u16,
    pub body: Vec<u8>,
}

impl RemoteRestResp {
    pub fn json(&self) -> Option<Value> {
        serde_json::from_slice(&self.body).ok()
    }
}

// -------------------------------------------------------------- the service

/// One managed connection of an alias.
#[derive(Clone, Debug)]
pub struct RemoteConn {
    pub addr: SocketAddr,
    pub socket: u16,
    pub node: NodeIdentity,
}

#[derive(Debug, Default)]
struct RemoteState {
    conns: Vec<RemoteConn>,
    /// The remote cluster's name, learned from the first connection.
    cluster_name: Option<String>,
    cluster_uuid: Option<String>,
    version: Option<String>,
    last_error: Option<EsError>,
    failures: u32,
    next_attempt: Option<Instant>,
    closed: bool,
    /// Gateway nodes (sniff) or sockets (proxy) the last attempt found.
    candidates: Option<usize>,
    /// Consecutive top-up attempts that left connections missing.
    topups: u32,
    /// Connected to the remote cluster server port with credentials.
    rcs: bool,
}

pub struct RemoteCluster {
    config: RwLock<RemoteConfig>,
    st: Mutex<RemoteState>,
    connecting: tokio::sync::Mutex<()>,
    rr: AtomicUsize,
}

impl RemoteCluster {
    fn new(config: RemoteConfig) -> Arc<Self> {
        Arc::new(RemoteCluster { config: RwLock::new(config), st: Mutex::new(RemoteState::default()), connecting: tokio::sync::Mutex::new(()), rr: AtomicUsize::new(0) })
    }

    pub fn config(&self) -> RemoteConfig {
        self.config.read().clone()
    }

    fn max_conns(&self) -> usize {
        let c = self.config.read();
        match c.mode {
            RemoteMode::Sniff => c.node_connections as usize,
            RemoteMode::Proxy => c.proxy_socket_connections as usize,
        }
    }

    pub fn connected(&self) -> usize {
        self.st.lock().conns.len()
    }

    /// The connections worth having: the configured maximum, or fewer when
    /// the remote cluster has fewer gateway nodes.
    fn wanted(&self) -> usize {
        let max = self.max_conns();
        self.st.lock().candidates.map_or(max, |c| c.clamp(1, max))
    }

    pub fn cluster_name(&self) -> Option<String> {
        self.st.lock().cluster_name.clone()
    }

    pub fn cluster_uuid(&self) -> Option<String> {
        self.st.lock().cluster_uuid.clone()
    }

    pub fn version(&self) -> Option<String> {
        self.st.lock().version.clone()
    }

    fn options(&self, rcs: bool) -> RemoteConnectOptions {
        let c = self.config.read();
        RemoteConnectOptions {
            remote_cluster_server: rcs,
            ping_schedule: c.ping_schedule_millis.map(Duration::from_millis),
            compress: Some(match c.compress.as_str() {
                "true" => RemoteCompress::True,
                "false" => RemoteCompress::False,
                _ => RemoteCompress::IndexingData,
            }),
            connect_timeout: None,
        }
    }

    /// `cluster.remote.<alias>.credentials` (API key based remote cluster
    /// security) were configured when the connections opened.
    fn rcs(&self) -> bool {
        self.st.lock().rcs
    }

    /// The next connection to use (round robin).
    fn pick(&self) -> Option<RemoteConn> {
        let st = self.st.lock();
        if st.conns.is_empty() {
            return None;
        }
        let i = self.rr.fetch_add(1, Ordering::Relaxed) % st.conns.len();
        Some(st.conns[i].clone())
    }
}

/// The node's remote cluster connections.
pub struct RemoteClusterService {
    node: Weak<ClusterNode>,
    clusters: Mutex<BTreeMap<String, Arc<RemoteCluster>>>,
    wake: tokio::sync::Notify,
    /// The node has the `remote_cluster_client` role.
    client: bool,
}

static SERVICES: OnceLock<Mutex<Vec<(usize, Weak<RemoteClusterService>)>>> = OnceLock::new();

/// The remote cluster service of `node` (started by [`register`]).
pub fn service(node: &ClusterNode) -> Option<Arc<RemoteClusterService>> {
    let key = node as *const ClusterNode as usize;
    SERVICES.get()?.lock().iter().find(|(k, _)| *k == key).and_then(|(_, w)| w.upgrade())
}

/// ES `ConnectTransportException` for a node at `addr` (`[name][addr] msg`).
fn connect_error(name: &str, addr: &str, msg: &str) -> EsError {
    EsError::new(502, "connect_transport_exception", format!("[{name}][{addr}] {msg}"))
}

/// A transport failure as the ES exception ES reports for it.
pub fn transport_error(e: &TransportError, node_name: &str) -> EsError {
    match e {
        TransportError::ConnectTransport { addr, .. } => connect_error(node_name, &addr.to_string(), "connect_exception"),
        TransportError::NodeDisconnected { addr, action } => EsError::new(502, "node_disconnected_exception", format!("[{node_name}][{addr}][{action}] disconnected")),
        TransportError::Remote(r) => remote_error(r),
        other => remote_error(&other.to_remote_error()),
    }
}

/// A [`RemoteError`] (with its causes and metadata) as an [`EsError`].
pub fn remote_error(r: &RemoteError) -> EsError {
    let mut e = EsError::new(r.status, &r.error_type, r.reason.clone());
    for (k, v) in &r.metadata {
        e = e.with(k, Value::String(v.clone()));
    }
    if let Some(c) = &r.caused_by {
        e = e.caused_by(remote_error(c));
    }
    e
}

/// An error's own object (`type`, `reason`, ...), without the REST
/// envelope (`{"error": ..., "status": ...}`).
pub fn error_body(e: &EsError) -> Value {
    let v = e.to_json();
    match v.get("error") {
        Some(inner) if inner.is_object() => inner.clone(),
        _ => v,
    }
}

/// An [`EsError`] for the wire.
pub fn to_remote(e: &EsError) -> RemoteError {
    let v = error_body(e);
    fn conv(v: &Value, status: u16) -> RemoteError {
        let mut r = RemoteError::new(v["type"].as_str().unwrap_or("exception"), v["reason"].as_str().unwrap_or_default().to_string(), status);
        if let Some(o) = v.as_object() {
            for (k, x) in o {
                if matches!(k.as_str(), "type" | "reason" | "caused_by" | "root_cause" | "stack_trace" | "suppressed") {
                    continue;
                }
                let text = match x {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                r = r.with_metadata(k.clone(), text);
            }
        }
        if let Some(c) = v.get("caused_by").filter(|c| c.is_object()) {
            r = r.with_cause(conv(c, status));
        }
        r
    }
    conv(&v, e.status)
}

fn retryable(e: &EsError) -> bool {
    matches!(e.kind.as_str(), "connect_transport_exception" | "node_disconnected_exception" | "illegal_state_exception" | "receive_timeout_transport_exception" | "transport_exception")
}

async fn resolve(address: &str) -> Result<SocketAddr, EsError> {
    let (host, port) = eremote::parse_remote_address(address)?;
    let mut it = tokio::net::lookup_host((host.as_str(), port)).await.map_err(|_| EsError::illegal_argument(format!("unknown host [{host}]")))?;
    it.next().ok_or_else(|| EsError::illegal_argument(format!("unknown host [{host}]")))
}

/// ES `RemoteClusterSettings` node predicate: not a dedicated master (data
/// or ingest roles, or not master-eligible) and, with
/// `cluster.remote.node.attr`, the attribute set to `true`.
fn gateway(n: &RemoteNodeInfo, attr: Option<&str>) -> bool {
    let roles: BTreeSet<&str> = n.roles.iter().map(String::as_str).collect();
    let master = roles.contains("master");
    let data = roles.iter().any(|r| r.starts_with("data") || *r == "index" || *r == "search");
    let ingest = roles.contains("ingest");
    let attr_ok = attr.is_none_or(|a| n.attributes.get(a).map(String::as_str) == Some("true"));
    (!master || data || ingest) && attr_ok
}

impl RemoteClusterService {
    fn node(&self) -> Option<Arc<ClusterNode>> {
        self.node.upgrade()
    }

    /// The configured aliases (enabled ones).
    pub fn aliases(&self) -> Vec<String> {
        self.clusters.lock().keys().cloned().collect()
    }

    pub fn cluster(&self, alias: &str) -> Option<Arc<RemoteCluster>> {
        self.clusters.lock().get(alias).cloned()
    }

    fn no_such_remote(alias: &str) -> EsError {
        eremote::no_such_remote(alias)
    }

    /// Bring the alias table in line with the settings.
    fn reconcile(self: &Arc<Self>) {
        let Some(node) = self.node() else { return };
        let desired = eremote::remote_configs(&node.engine);
        let mut closing: Vec<Arc<RemoteCluster>> = Vec::new();
        {
            let mut clusters = self.clusters.lock();
            let gone: Vec<String> = clusters.keys().filter(|a| !desired.contains_key(*a)).cloned().collect();
            for a in gone {
                if let Some(c) = clusters.remove(&a) {
                    tracing::info!(alias = %a, "remote cluster removed: closing its connections");
                    closing.push(c);
                }
            }
            for (alias, cfg) in desired {
                match clusters.get(&alias) {
                    None => {
                        clusters.insert(alias, RemoteCluster::new(cfg));
                    }
                    Some(c) => {
                        let current = c.config();
                        let mut cmp = current.clone();
                        cmp.skip_unavailable = cfg.skip_unavailable;
                        if cmp != cfg {
                            tracing::info!(alias = %alias, "remote cluster settings changed: reconnecting");
                            closing.push(c.clone());
                            clusters.insert(alias, RemoteCluster::new(cfg));
                        } else if current.skip_unavailable != cfg.skip_unavailable {
                            c.config.write().skip_unavailable = cfg.skip_unavailable;
                        }
                    }
                }
            }
        }
        // Aliases pointing at the same address share its pooled
        // connections: those another alias still uses stay open.
        let in_use: std::collections::HashSet<(SocketAddr, u16)> =
            self.clusters.lock().values().flat_map(|c| c.st.lock().conns.iter().map(|r| (r.addr, r.socket)).collect::<Vec<_>>()).collect();
        for c in closing {
            let node = node.clone();
            let in_use = in_use.clone();
            tokio::spawn(async move { close_cluster(&node, &c, &in_use).await });
        }
    }

    /// Drop connections that closed.
    fn prune(&self) {
        let Some(node) = self.node() else { return };
        for c in self.clusters.lock().values() {
            let mut st = c.st.lock();
            st.conns.retain(|r| node.transport.remote_connection_identity(r.addr, r.socket).is_some());
        }
    }

    /// Start connect attempts that are due.
    fn connect_due(self: &Arc<Self>) {
        if !self.client {
            return;
        }
        let now = Instant::now();
        let due: Vec<(String, Arc<RemoteCluster>)> = self
            .clusters
            .lock()
            .iter()
            .filter(|(_, c)| {
                let st = c.st.lock();
                !st.closed && st.conns.len() < st.candidates.map_or(c.max_conns(), |n| n.clamp(1, c.max_conns())) && st.next_attempt.is_none_or(|t| now >= t)
            })
            .map(|(a, c)| (a.clone(), c.clone()))
            .collect();
        for (alias, c) in due {
            if c.connecting.try_lock().is_err() {
                continue;
            }
            let me = self.clone();
            tokio::spawn(async move {
                let _ = me.connect(&alias, &c).await;
            });
        }
    }

    /// Connect `c` (coalesced: one attempt at a time). Ok when at least one
    /// connection is up afterwards.
    async fn connect(&self, alias: &str, c: &Arc<RemoteCluster>) -> Result<(), EsError> {
        let _g = c.connecting.lock().await;
        if c.connected() >= c.wanted() {
            return Ok(());
        }
        let Some(node) = self.node() else { return Err(EsError::new(500, "node_closed_exception", "node closed")) };
        let r = match c.config().mode {
            RemoteMode::Sniff => self.connect_sniff(&node, alias, c).await,
            RemoteMode::Proxy => self.connect_proxy(&node, alias, c).await,
        };
        let wanted = c.wanted();
        let mut st = c.st.lock();
        match &r {
            Ok(()) if st.conns.len() < wanted => {
                // Some gateway nodes did not answer: top up later, backing off.
                st.failures = 0;
                st.last_error = None;
                st.topups = st.topups.saturating_add(1);
                let backoff = Duration::from_secs(1).saturating_mul(1u32 << st.topups.saturating_sub(1).min(5)).min(BACKOFF_MAX);
                st.next_attempt = Some(Instant::now() + backoff);
            }
            Ok(()) => {
                st.failures = 0;
                st.topups = 0;
                st.next_attempt = None;
                st.last_error = None;
            }
            Err(e) if st.conns.is_empty() => {
                st.failures = st.failures.saturating_add(1);
                let backoff = BACKOFF_MIN.saturating_mul(1u32 << st.failures.saturating_sub(1).min(16)).min(BACKOFF_MAX);
                st.next_attempt = Some(Instant::now() + backoff);
                st.last_error = Some(e.clone());
                tracing::debug!(%alias, "remote cluster connect failed (retry in {backoff:?}): {e}");
            }
            Err(_) => {}
        }
        if st.conns.is_empty() { r } else { Ok(()) }
    }

    fn check_name(c: &RemoteCluster, identity: &NodeIdentity, addr: &str) -> Result<(), EsError> {
        let st = c.st.lock();
        if let Some(expected) = &st.cluster_name {
            if expected != &identity.cluster_name {
                return Err(EsError::new(500, "illegal_state_exception", format!(
                    "handshake with [{}][{addr}] failed: remote cluster name [{}] does not match expected remote cluster name [{expected}]",
                    identity.node.name, identity.cluster_name
                )));
            }
        }
        Ok(())
    }

    async fn connect_sniff(&self, node: &Arc<ClusterNode>, alias: &str, c: &Arc<RemoteCluster>) -> Result<(), EsError> {
        let cfg = c.config();
        let rcs = self.has_credentials(alias);
        c.st.lock().rcs = rcs;
        if cfg.seeds.is_empty() {
            return Err(EsError::new(500, "no_seed_node_left_exception", format!("no seed node left for cluster: [{alias}]")));
        }
        let mut last = None;
        for seed in &cfg.seeds {
            match self.try_seed(node, alias, c, seed).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let again = retryable(&e);
                    last = Some(e);
                    if !again {
                        break;
                    }
                }
            }
        }
        Err(last.unwrap_or_else(|| EsError::new(500, "no_seed_node_left_exception", format!("no seed node left for cluster: [{alias}]"))))
    }

    async fn try_seed(&self, node: &Arc<ClusterNode>, alias: &str, c: &Arc<RemoteCluster>, seed: &str) -> Result<(), EsError> {
        let cfg = c.config();
        let dial = cfg.sniff_proxy.clone().unwrap_or_else(|| seed.to_string());
        let addr = resolve(&dial).await?;
        let rcs = c.rcs();
        let options = c.options(rcs);
        // API key based remote cluster security: the seeds are remote
        // cluster server addresses, and the sniff answers each gateway
        // node's remote cluster server address (ES `RemoteClusterNodesAction`).
        let _ = rcs;
        let identity = node.transport.connect_remote(addr, SEED_SOCKET, options.clone()).await.map_err(|e| transport_error(&e, ""))?;
        let result = async {
            Self::check_name(c, &identity, &addr.to_string())?;
            let info: ClusterInfo = node
                .transport
                .send_remote_request(addr, SEED_SOCKET, ACTION_CLUSTER_INFO, &ClusterInfoReq { from_cluster: node.transport.cluster_name().to_string(), credential: node.config.settings.get(&format!("cluster.remote.{alias}.credentials")).and_then(Value::as_str).map(str::to_string) }, SNIFF_TIMEOUT)
                .await
                .map_err(|e| transport_error(&e, &identity.node.name))?;
            let attr = eremote::gateway_node_attr(&node.engine);
            let max = c.max_conns();
            c.st.lock().candidates = Some(info.nodes.iter().filter(|n| gateway(n, attr.as_deref())).count());
            let mut opened = 0usize;
            for n in info.nodes.iter().filter(|n| gateway(n, attr.as_deref())) {
                if c.connected() >= max {
                    break;
                }
                let target = cfg.sniff_proxy.clone().unwrap_or_else(|| n.transport_address.clone());
                let Ok(naddr) = resolve(&target).await else { continue };
                if c.st.lock().conns.iter().any(|r| r.addr == naddr && r.node.node.id == n.id) {
                    continue;
                }
                match node.transport.connect_remote(naddr, 0, options.clone()).await {
                    Ok(id) => {
                        if Self::check_name(c, &id, &naddr.to_string()).is_err() || id.cluster_name != info.cluster_name {
                            node.transport.disconnect_remote(naddr, 0).await;
                            continue;
                        }
                        let mut st = c.st.lock();
                        if st.cluster_name.is_none() {
                            st.cluster_name = Some(info.cluster_name.clone());
                        }
                        st.cluster_uuid = Some(info.cluster_uuid.clone());
                        st.version = Some(info.version.clone());
                        st.conns.push(RemoteConn { addr: naddr, socket: 0, node: id });
                        opened += 1;
                    }
                    Err(e) => tracing::debug!(%alias, node = %n.name, "failed to open managed connection to remote node: {e}"),
                }
            }
            if c.connected() == 0 {
                return Err(EsError::new(500, "illegal_state_exception", format!("Unable to open any connections to remote cluster [{alias}]")));
            }
            tracing::info!(%alias, cluster = %info.cluster_name, opened, "connected to remote cluster");
            Ok(())
        }
        .await;
        node.transport.disconnect_remote(addr, SEED_SOCKET).await;
        result
    }

    async fn connect_proxy(&self, node: &Arc<ClusterNode>, alias: &str, c: &Arc<RemoteCluster>) -> Result<(), EsError> {
        let cfg = c.config();
        let rcs = self.has_credentials(alias);
        c.st.lock().rcs = rcs;
        let configured = cfg.proxy_address.clone().unwrap_or_default();
        let server_name = cfg.server_name.clone().unwrap_or_else(|| "null".into());
        let no_sockets = || EsError::new(500, "no_seed_node_left_exception", format!(
            "Unable to open any proxy connections to cluster [{alias}] at address [{configured}], configuredAddress=[{configured}], configuredServerName=[{server_name}]"
        ));
        let addr = match resolve(&configured).await {
            Ok(a) => a,
            Err(_) => return Err(no_sockets()),
        };
        let options = c.options(rcs);
        c.st.lock().candidates = Some(cfg.proxy_socket_connections as usize);
        let wanted: Vec<u16> = {
            let st = c.st.lock();
            (0..cfg.proxy_socket_connections as u16).filter(|s| !st.conns.iter().any(|r| r.socket == *s)).collect()
        };
        let attempts = wanted.iter().map(|s| {
            let (transport, options, s) = (node.transport.clone(), options.clone(), *s);
            async move { (s, transport.connect_remote(addr, s, options).await) }
        });
        let results = futures::future::join_all(attempts).await;
        for (s, r) in results {
            match r {
                Ok(id) => {
                    let mismatch = {
                        let mut st = c.st.lock();
                        let mismatch = st.cluster_name.as_ref().is_some_and(|n| n != &id.cluster_name);
                        if !mismatch {
                            if st.cluster_name.is_none() {
                                st.cluster_name = Some(id.cluster_name.clone());
                            }
                            st.conns.push(RemoteConn { addr, socket: s, node: id.clone() });
                        }
                        mismatch
                    };
                    if mismatch {
                        tracing::warn!(%alias, "handshake failed. unexpected remote cluster name {}", id.cluster_name);
                        node.transport.disconnect_remote(addr, s).await;
                    }
                }
                Err(e) => tracing::debug!(%alias, "failed to open remote proxy connection: {e}"),
            }
        }
        if c.connected() == 0 {
            return Err(no_sockets());
        }
        // The remote cluster's uuid and version for stats and resolve/cluster.
        if c.cluster_uuid().is_none() {
            if let Some(conn) = c.pick() {
                if let Ok(info) = node.transport.send_remote_request::<_, ClusterInfo>(conn.addr, conn.socket, ACTION_CLUSTER_INFO, &ClusterInfoReq { from_cluster: node.transport.cluster_name().to_string(), credential: node.config.settings.get(&format!("cluster.remote.{alias}.credentials")).and_then(Value::as_str).map(str::to_string) }, SNIFF_TIMEOUT).await {
                    let mut st = c.st.lock();
                    st.cluster_uuid = Some(info.cluster_uuid);
                    st.version = Some(info.version);
                }
            }
        }
        Ok(())
    }

    /// `cluster.remote.<alias>.credentials` is in the keystore.
    pub fn has_credentials(&self, alias: &str) -> bool {
        self.node().is_some_and(|n| n.config.settings.contains_key(&format!("cluster.remote.{alias}.credentials")))
    }

    /// Connect `alias` now unless it is connected (ES `ensureConnected`).
    pub async fn ensure_connected(&self, alias: &str) -> Result<(), EsError> {
        if !self.client {
            return Err(EsError::illegal_argument(format!(
                "node [{}] does not have the [remote_cluster_client] role",
                self.node().map(|n| n.local.name.clone()).unwrap_or_default()
            )));
        }
        let c = self.cluster(alias).ok_or_else(|| Self::no_such_remote(alias))?;
        if c.connected() > 0 {
            return Ok(());
        }
        self.connect(alias, &c).await
    }

    /// The error a request to a disconnected `alias` gets right now.
    pub fn unavailable(&self, alias: &str) -> EsError {
        match self.cluster(alias) {
            None => Self::no_such_remote(alias),
            Some(_) => EsError::new(502, "connect_transport_exception", format!("Unable to connect to [{alias}]")),
        }
    }

    /// Send `req` to `alias` (any connected node; another one if that node
    /// went away). `ensure`: connect first if not connected (requests that
    /// must not skip the cluster); otherwise a disconnected alias fails at
    /// once with `Unable to connect to [alias]` (and reconnects in the
    /// background), as ES's skippable remotes do.
    pub async fn send<Req, Resp>(&self, alias: &str, action: &str, req: &Req, timeout: Option<Duration>, ensure: bool) -> Result<Resp, EsError>
    where
        Req: Serialize + ?Sized + Sync,
        Resp: serde::de::DeserializeOwned,
    {
        let node = self.node().ok_or_else(|| EsError::new(500, "node_closed_exception", "node closed"))?;
        let c = self.cluster(alias).ok_or_else(|| Self::no_such_remote(alias))?;
        if ensure {
            self.ensure_connected(alias).await?;
        } else if c.connected() == 0 && c.st.lock().failures == 0 {
            // A remote that was just configured and never failed: ES connects
            // it while applying the settings (up to 10s), so its first
            // request waits for that first connection instead of skipping it.
            let _ = tokio::time::timeout(Duration::from_secs(10), self.ensure_connected(alias)).await;
        }
        let mut last = None;
        for _ in 0..c.max_conns().clamp(1, 3) {
            let Some(conn) = c.pick() else { break };
            match node.transport.send_remote_request::<Req, Resp>(conn.addr, conn.socket, action, req, timeout).await {
                Ok(r) => return Ok(r),
                Err(e) if e.is_connection_failure() => {
                    c.st.lock().conns.retain(|r| !(r.addr == conn.addr && r.socket == conn.socket));
                    self.wake.notify_one();
                    last = Some(transport_error(&e, &conn.node.node.name));
                }
                Err(e) => return Err(transport_error(&e, &conn.node.node.name)),
            }
        }
        if c.connected() == 0 {
            self.wake.notify_one();
        }
        Err(last.unwrap_or_else(|| self.unavailable(alias)))
    }

    /// Run `method uri` (an allow-listed read API) on `alias` as `identity`.
    pub async fn rest(&self, alias: &str, mut req: RemoteRestReq, ensure: bool) -> Result<RemoteRestResp, EsError> {
        if let Some(node) = self.node() {
            req.identity = outgoing_identity(&node, alias, req.identity.take());
        }
        self.send(alias, ACTION_REMOTE_REST, &req, Some(Duration::from_secs(300)), ensure).await
    }
}

/// The identity a request to `alias` carries: with
/// `cluster.remote.<alias>.credentials` (ES RCS 2.0), the user's name and
/// the role descriptors its roles grant on that cluster (`remote_indices`
/// and `remote_cluster` whose `clusters` match the alias), with the
/// cross-cluster API key that the remote cluster authenticates and limits
/// them by; otherwise (certificate-based) the user as it is.
pub fn outgoing_identity(node: &ClusterNode, alias: &str, identity: Option<String>) -> Option<String> {
    let key = format!("cluster.remote.{alias}.credentials");
    let Some(credential) = node.config.settings.get(&key).and_then(Value::as_str).map(str::to_string) else { return identity };
    let stored = identity.as_deref().and_then(elasticsearch_engine::security::owner::StoredAuthn::decode);
    if stored.as_ref().is_some_and(|s| s.cross_cluster.is_some()) {
        return identity;
    }
    let username = stored.as_ref().map(|s| s.principal().to_string()).unwrap_or_else(|| "_anonymous".into());
    let descriptors = stored.map(|s| remote_descriptors(node, &s.to_authn(), alias)).unwrap_or_default();
    let mut out = elasticsearch_engine::security::owner::StoredAuthn {
        username, roles: Vec::new(), full_name: None, email: None, metadata: Value::Null, realm_name: "_es_cross_cluster".into(),
        realm_type: "_es_cross_cluster".into(), auth_type: "api_key".into(), api_key: None, service_token: None, anonymous: false, run_as: None,
        cross_cluster: None,
    };
    out.cross_cluster = Some((credential, descriptors));
    Some(out.encode())
}

/// The role descriptors `authn`'s roles grant on remote cluster `alias`.
fn remote_descriptors(node: &ClusterNode, authn: &elasticsearch_engine::security::Authn, alias: &str) -> Vec<Value> {
    let sec = &node.engine.security;
    let matches = |v: &Value| v["clusters"].as_array().is_some_and(|c| c.iter().filter_map(Value::as_str).any(|p| elasticsearch_engine::mapping::wildcard_match(p, alias)));
    let mut out = Vec::new();
    for name in sec.assigned_roles(authn.effective()) {
        if name == "superuser" {
            out.push(json!({"cluster": ["all"], "indices": [{"names": ["*"], "privileges": ["all"], "allow_restricted_indices": false}]}));
            continue;
        }
        let Some(d) = sec.role_descriptor(&name) else { continue };
        let indices: Vec<Value> = d["remote_indices"].as_array().into_iter().flatten().filter(|g| matches(g)).map(|g| {
            let mut g = g.clone();
            if let Some(o) = g.as_object_mut() {
                o.remove("clusters");
            }
            g
        }).collect();
        let cluster: Vec<Value> = d["remote_cluster"].as_array().into_iter().flatten().filter(|g| matches(g)).flat_map(|g| g["privileges"].as_array().cloned().unwrap_or_default()).collect();
        if !indices.is_empty() || !cluster.is_empty() {
            out.push(json!({"cluster": cluster, "indices": indices}));
        }
    }
    out
}

/// Who a remote cluster's request runs as: the user it names
/// (certificate-based remote cluster security) or, with a cross-cluster API
/// key (API key based remote cluster security; required on the remote
/// cluster server port), that user limited by the key, authenticated here.
pub fn authenticate_remote(node: &ClusterNode, peer: &Option<elasticsearch_transport::RemotePeer>, identity: &Option<String>) -> Result<Option<elasticsearch_engine::security::owner::StoredAuthn>, EsError> {
    let rcs_port = peer.as_ref().is_some_and(|p| p.remote_cluster_server);
    let stored = match identity {
        Some(id) => Some(elasticsearch_engine::security::owner::StoredAuthn::decode(id).ok_or_else(|| EsError::new(403, "security_exception", "invalid remote cluster user identity"))?),
        None => None,
    };
    match stored {
        Some(s) if s.cross_cluster.is_some() => {
            let (credential, descriptors) = s.cross_cluster.clone().unwrap_or_default();
            let user = s.username.clone();
            let authn = node.engine.security.authenticate_cross_cluster(&credential, &user, descriptors)
                .map_err(|why| EsError::new(401, "security_exception", format!("unable to authenticate the cross-cluster request of user [{user}]: {why}")))?;
            Ok(Some(elasticsearch_engine::security::owner::StoredAuthn::of(&authn)))
        }
        _ if rcs_port => Err(EsError::new(401, "security_exception", "requests to the remote cluster server port must carry a cross-cluster API key (cluster.remote.<alias>.credentials)")),
        Some(s) if s.api_key.is_some() || s.service_token.is_some() => Err(EsError::new(403, "security_exception",
            "cross-cluster requests authenticated by an API key or a service account token are not supported with certificate-based remote cluster security")),
        other => Ok(other),
    }
}

async fn close_cluster(node: &ClusterNode, c: &RemoteCluster, in_use: &std::collections::HashSet<(SocketAddr, u16)>) {
    let conns = {
        let mut st = c.st.lock();
        st.closed = true;
        std::mem::take(&mut st.conns)
    };
    for r in conns.into_iter().filter(|r| !in_use.contains(&(r.addr, r.socket))) {
        node.transport.disconnect_remote(r.addr, r.socket).await;
    }
}

struct Connections(Weak<RemoteClusterService>);

impl eremote::RemoteConnections for Connections {
    fn connected(&self, alias: &str) -> usize {
        self.0.upgrade().and_then(|s| s.cluster(alias)).map(|c| c.connected()).unwrap_or(0)
    }

    fn connection_error(&self, alias: &str) -> Option<EsError> {
        let s = self.0.upgrade()?;
        let c = s.cluster(alias)?;
        if c.connected() > 0 {
            return None;
        }
        Some(c.st.lock().last_error.clone().unwrap_or_else(|| s.unavailable(alias)))
    }

    fn has_credentials(&self, alias: &str) -> bool {
        self.0.upgrade().is_some_and(|s| s.has_credentials(alias))
    }
}

/// The node's remote cluster info for the sniff of another cluster.
/// A node's remote cluster server publish address (local action).
const ACTION_RCS_ADDRESS: &str = "cluster:internal/remote_cluster/server_address";

/// This node's remote cluster server publish address (`None`: not
/// enabled): `remote_cluster.publish_host`, else the bound host unless it
/// is a wildcard, else the transport publish host; the bound port.
fn rcs_publish_address(node: &ClusterNode) -> Option<String> {
    let bound = node.transport.remote_cluster_server_address()?;
    let text = |k: &str| node.config.settings.get(k).and_then(Value::as_str).map(str::to_string);
    let host = text("remote_cluster.publish_host").or_else(|| (!bound.ip().is_unspecified()).then(|| bound.ip().to_string())).unwrap_or_else(|| {
        node.local.transport_address.parse::<SocketAddr>().map(|a| a.ip().to_string()).unwrap_or_else(|_| "127.0.0.1".into())
    });
    let port = text("remote_cluster.publish_port").and_then(|p| p.parse::<u16>().ok()).unwrap_or(bound.port());
    Some(if host.contains(':') && !host.starts_with('[') { format!("[{host}]:{port}") } else { format!("{host}:{port}") })
}

/// The sniff answer of the remote cluster server port: each node's remote
/// cluster server address in place of its transport address; nodes without
/// one are left out (ES `TransportRemoteClusterNodesAction`, nodes info
/// `remote_cluster_server`).
async fn remote_cluster_server_nodes(node: &Arc<ClusterNode>, nodes: Vec<RemoteNodeInfo>) -> Vec<RemoteNodeInfo> {
    let st = node.state();
    let asks = nodes.into_iter().map(|mut n| {
        let addr = st.nodes.get(&n.id).and_then(|d| d.transport_address.parse::<SocketAddr>().ok());
        let transport = node.transport.clone();
        let own = (n.id == node.local.id).then(|| rcs_publish_address(node));
        async move {
            let address = match (own, addr) {
                (Some(own), _) => own,
                (None, Some(addr)) => transport.send_request::<_, Option<String>>(addr, ACTION_RCS_ADDRESS, &(), Duration::from_secs(5)).await.ok().flatten(),
                (None, None) => None,
            };
            address.map(|a| {
                n.transport_address = a;
                n
            })
        }
    });
    futures::future::join_all(asks).await.into_iter().flatten().collect()
}

fn cluster_info(node: &ClusterNode) -> ClusterInfo {
    let st = node.state();
    let nodes = st
        .nodes
        .values()
        .map(|n| RemoteNodeInfo {
            id: n.id.clone(),
            name: n.name.clone(),
            transport_address: n.transport_address.clone(),
            roles: n.roles.iter().map(|r| r.name().to_string()).collect(),
            attributes: n.attributes.clone(),
            version: n.version.clone(),
        })
        .collect();
    ClusterInfo { cluster_name: node.transport.cluster_name().to_string(), cluster_uuid: st.metadata.cluster_uuid.clone(), version: elasticsearch_engine::node::ES_VERSION.to_string(), nodes }
}

/// REST read APIs a remote cluster may run here.
fn allowed_rest(method: &str, segs: &[String]) -> bool {
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    let read = matches!(method, "GET" | "POST" | "HEAD");
    match s.as_slice() {
        [_, "_search"] | ["_search"] => read,
        [_, "_count"] | ["_count"] => read,
        [_, "_mapping"] | [_, "_settings"] => method == "GET",
        [_, "_field_caps"] | ["_field_caps"] => read,
        [_, "_terms_enum"] => read,
        ["_resolve", "index", _] | ["_resolve", "cluster", _] | ["_resolve", "cluster"] => read,
        ["_search", "scroll"] | ["_search", "scroll", _] => read || method == "DELETE",
        [_, "_pit"] => method == "POST",
        ["_pit"] => method == "DELETE",
        [_, "_eql", "search"] => read,
        ["_query"] => method == "POST",
        ["_sql"] => read,
        ["_cluster", "stats"] => method == "GET",
        _ => false,
    }
}

async fn serve_rest(node: Arc<ClusterNode>, req: RemoteRestReq, peer: Option<elasticsearch_transport::RemotePeer>) -> Result<RemoteRestResp, RemoteError> {
    let origin = peer.as_ref().map(|p| p.cluster_name.clone()).unwrap_or_default();
    let uri = req.uri.clone();
    let path = uri.split('?').next().unwrap_or("");
    let segs: Vec<String> = path.split('/').filter(|s| !s.is_empty()).map(|s| crate::front::decode_path(s)).collect();
    if !allowed_rest(&req.method, &segs) {
        return Err(RemoteError::new("security_exception", format!("request [{} {path}] is not allowed over a remote cluster connection", req.method), 403));
    }
    let mut headers: Vec<(String, String)> = vec![("content-type".into(), "application/json".into())];
    if node.engine.security.enabled() {
        if let Some(stored) = authenticate_remote(&node, &peer, &req.identity).map_err(|e| to_remote(&e))? {
            headers.push((crate::front::IDENTITY_HEADER.into(), stored.encode()));
        }
    }
    if req.parts {
        headers.push((CCS_PARTS_HEADER.into(), "1".into()));
        if let Some(dfs) = &req.dfs {
            headers.push((CCS_DFS_HEADER.into(), dfs.clone()));
        }
        if req.other_shards > 0 {
            headers.push((CCS_OTHER_SHARDS_HEADER.into(), req.other_shards.to_string()));
        }
    }
    if req.rank_apply {
        headers.push((RANK_APPLY_HEADER.into(), "apply".into()));
    }
    headers.push((CCS_ORIGIN_HEADER.into(), origin));
    let rest = RestReq::try_new(&req.method, &uri, &headers, &req.body, 0, Default::default(), None).map_err(|e| to_remote(&e))?;
    // Through the same validation and security pipeline as a client
    // request of this cluster (wildcards expand to what the user may read).
    let resp: RestResp = if node.engine.security.enabled() {
        Box::pin(crate::front::authorized_frontend(&node, &rest)).await
    } else {
        crate::front::dispatch(&node, rest).await
    };
    Ok(RemoteRestResp { status: resp.status, body: resp.body.to_vec() })
}

/// Start `node`'s remote cluster service and serve the actions remote
/// clusters call.
pub fn register(node: &Arc<ClusterNode>) {
    let client = node.local.roles.contains(&Role::RemoteClusterClient);
    let svc = Arc::new(RemoteClusterService { node: Arc::downgrade(node), clusters: Mutex::new(BTreeMap::new()), wake: tokio::sync::Notify::new(), client });
    let key = Arc::as_ptr(node) as usize;
    {
        let mut all = SERVICES.get_or_init(Default::default).lock();
        all.retain(|(_, w)| w.strong_count() > 0);
        all.push((key, Arc::downgrade(&svc)));
    }
    // ES `remote_cluster_server.enabled`: the API key based remote cluster
    // server port (`remote_cluster.host` / `remote_cluster.port`, 9443).
    {
        let settings = &node.config.settings;
        let text = |k: &str| settings.get(k).map(|v| match v { Value::String(s) => s.clone(), other => other.to_string() });
        // Validated at start ([`crate::node::remote_cluster_tls`]).
        let (server_tls, client_tls) = crate::node::remote_cluster_tls(settings).unwrap_or_else(|e| {
            tracing::error!("remote cluster TLS: {e}");
            (None, None)
        });
        if let Err(e) = node.transport.set_remote_cluster_client_tls(client_tls) {
            tracing::error!("remote cluster client TLS: {e}");
        }
        if text("remote_cluster_server.enabled").as_deref() == Some("true") {
            let host = text("remote_cluster.host").or_else(|| text("network.host")).unwrap_or_else(|| "127.0.0.1".into());
            let host = if host == "0" || host == "_site_" || host == "_global_" { "0.0.0.0".to_string() } else { host.trim_matches(|c| c == '[' || c == ']').to_string() };
            let port: u16 = text("remote_cluster.port").and_then(|p| p.parse().ok()).unwrap_or(9443);
            let tls: Result<_, String> = Ok(server_tls);
            let transport = node.transport.clone();
            tokio::spawn(async move {
                let addr: Result<SocketAddr, _> = format!("{host}:{port}").parse().or_else(|_| format!("[{host}]:{port}").parse());
                match (addr, tls) {
                    (Ok(addr), Ok(tls)) => match transport.bind_remote_cluster_server(addr, tls).await {
                        Ok(bound) => tracing::info!(%bound, "remote cluster server started"),
                        Err(e) => tracing::error!("remote cluster server: {e}"),
                    },
                    (Err(e), _) => tracing::error!("remote cluster server address [{host}:{port}]: {e}"),
                    (_, Err(e)) => tracing::error!("remote cluster server TLS: {e}"),
                }
            });
        }
    }
    // The engines report the live state (`_remote/info`, planners).
    let conns: Arc<dyn eremote::RemoteConnections> = Arc::new(Connections(Arc::downgrade(&svc)));
    eremote::set_remote_connections(&node.engine, Some(conns));

    let n = node.clone();
    node.transport.register_handler(ACTION_CLUSTER_INFO, move |req: ClusterInfoReq| {
        let n = n.clone();
        let peer = elasticsearch_transport::inbound_remote_peer();
        async move {
            tracing::debug!(from = %req.from_cluster, "remote cluster sniff");
            // The remote cluster server port serves clusters holding a
            // cross-cluster API key only.
            if peer.as_ref().is_some_and(|p| p.remote_cluster_server) && n.engine.security.enabled() {
                let Some(credential) = &req.credential else {
                    return Err(RemoteError::new("security_exception", "the remote cluster server port requires a cross-cluster API key (cluster.remote.<alias>.credentials)", 401));
                };
                n.engine.security.authenticate_cross_cluster(credential, "_cross_cluster", Vec::new())
                    .map_err(|why| RemoteError::new("security_exception", format!("unable to authenticate the remote cluster: {why}"), 401))?;
            }
            let mut info = cluster_info(&n);
            if peer.as_ref().is_some_and(|p| p.remote_cluster_server) {
                info.nodes = remote_cluster_server_nodes(&n, info.nodes).await;
            }
            Ok::<_, RemoteError>(info)
        }
    });
    node.transport.allow_remote_action(ACTION_CLUSTER_INFO);
    let n = node.clone();
    node.transport.register_handler(ACTION_RCS_ADDRESS, move |_: ()| {
        let n = n.clone();
        async move { Ok::<_, RemoteError>(rcs_publish_address(&n)) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_REMOTE_REST, move |req: RemoteRestReq| {
        let n = n.clone();
        let peer = elasticsearch_transport::inbound_remote_peer();
        async move { serve_rest(n, req, peer).await }
    });
    node.transport.allow_remote_action(ACTION_REMOTE_REST);
    crate::front::ccs::register_esql(node);

    // The reconcile / reconnect loop owns the service; it ends with the node
    // (closing every remote connection).
    let mut states = node.subscribe();
    let mut events = node.transport.subscribe_remote();
    tokio::spawn(async move {
        loop {
            if svc.node().is_none() || svc.node().is_some_and(|n| n.transport.is_closed()) {
                break;
            }
            svc.reconcile();
            svc.prune();
            svc.connect_due();
            let state_closed = tokio::select! {
                r = states.changed() => r.is_err(),
                e = events.recv() => {
                    if let Ok(ConnectionEvent::Disconnected { addr, .. }) = &e {
                        tracing::debug!(%addr, "remote cluster connection closed");
                    }
                    false
                }
                _ = svc.wake.notified() => false,
                _ = tokio::time::sleep(Duration::from_millis(500)) => false,
            };
            if state_closed {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        let clusters: Vec<Arc<RemoteCluster>> = std::mem::take(&mut *svc.clusters.lock()).into_values().collect();
        if let Some(node) = svc.node() {
            for c in clusters {
                close_cluster(&node, &c, &Default::default()).await;
            }
        }
    });
}

/// `_remote/info` with the live connection state (the engine renders it).
pub fn remote_info(node: &ClusterNode) -> Value {
    eremote::remote_info(&node.engine)
}

/// `ccs.clusters.<alias>` of `_cluster/stats?include_remotes` for one
/// connected remote: its stats summary (ES `RemoteClusterStatsResponse`).
pub async fn remote_cluster_stats(node: &Arc<ClusterNode>, alias: &str, identity: Option<String>) -> Option<Value> {
    let svc = service(node)?;
    let c = svc.cluster(alias)?;
    if c.connected() == 0 {
        return None;
    }
    let req = RemoteRestReq { method: "GET".into(), uri: "/_cluster/stats".into(), body: Vec::new(), identity, parts: false, ..Default::default() };
    let r = svc.rest(alias, req, false).await.ok()?;
    if r.status >= 300 {
        return None;
    }
    let s = r.json()?;
    let cfg = c.config();
    Some(json!({
        "cluster_uuid": s["cluster_uuid"],
        "mode": if cfg.mode == RemoteMode::Proxy { "proxy" } else { "sniff" },
        "skip_unavailable": cfg.skip_unavailable,
        "transport.compress": cfg.compress,
        "status": s["status"],
        "version": s["nodes"]["versions"],
        "nodes_count": s["nodes"]["count"]["total"],
        "shards_count": s["indices"]["shards"]["total"],
        "indices_count": s["indices"]["count"],
        "indices_total_size_in_bytes": s["indices"]["store"]["size_in_bytes"],
        "indices_total_size": s["indices"]["store"]["size"],
        "max_heap_in_bytes": s["nodes"]["jvm"]["mem"]["heap_max_in_bytes"],
        "max_heap": s["nodes"]["jvm"]["mem"]["heap_max"],
        "mem_total_in_bytes": s["nodes"]["os"]["mem"]["total_in_bytes"],
        "mem_total": s["nodes"]["os"]["mem"]["total"],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(roles: &[&str], attrs: &[(&str, &str)]) -> RemoteNodeInfo {
        RemoteNodeInfo {
            id: "id".into(),
            name: "n".into(),
            transport_address: "127.0.0.1:9300".into(),
            roles: roles.iter().map(|r| r.to_string()).collect(),
            attributes: attrs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            version: "9.5.5".into(),
        }
    }

    #[test]
    fn gateway_nodes_exclude_dedicated_masters_and_follow_node_attr() {
        assert!(!gateway(&n(&["master"], &[]), None));
        assert!(gateway(&n(&["master", "data"], &[]), None));
        assert!(gateway(&n(&["master", "ingest"], &[]), None));
        assert!(gateway(&n(&["data_hot"], &[]), None));
        assert!(gateway(&n(&["ml"], &[]), None));
        assert!(!gateway(&n(&["data"], &[]), Some("gateway")));
        assert!(gateway(&n(&["data"], &[("gateway", "true")]), Some("gateway")));
        assert!(!gateway(&n(&["data"], &[("gateway", "false")]), Some("gateway")));
    }

    #[test]
    fn remote_rest_allow_list_is_read_only() {
        let segs = |p: &str| p.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect::<Vec<_>>();
        assert!(allowed_rest("POST", &segs("/logs/_search")));
        assert!(allowed_rest("GET", &segs("/_resolve/index/logs*")));
        assert!(allowed_rest("DELETE", &segs("/_search/scroll")));
        assert!(!allowed_rest("PUT", &segs("/logs/_doc/1")));
        assert!(!allowed_rest("POST", &segs("/logs/_bulk")));
        assert!(!allowed_rest("DELETE", &segs("/logs")));
        assert!(!allowed_rest("POST", &segs("/_security/user/x")));
        assert!(!allowed_rest("PUT", &segs("/_cluster/settings")));
    }

    #[test]
    fn errors_round_trip_over_the_wire() {
        let e = EsError::new(404, "index_not_found_exception", "no such index [x]").with("index", "x").caused_by(EsError::new(500, "exception", "inner"));
        let r = to_remote(&e);
        let back = remote_error(&r);
        assert_eq!(back.status, 404);
        assert_eq!(back.kind, "index_not_found_exception");
        assert_eq!(error_body(&back)["index"], "x");
        assert_eq!(error_body(&back)["caused_by"]["reason"], "inner");
    }
}
