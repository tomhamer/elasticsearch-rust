//! The cluster-aware `elasticsearch` server node: configuration, the
//! coordination driver, the cluster-state applier, replication + peer
//! recovery, request routing and the cluster REST APIs. See
//! `cluster/ARCHITECTURE.md`.

pub mod bootstrap_checks;
pub mod cli;
pub mod config;
pub mod coord;
pub mod create_enrollment_token;
pub mod enrollment;
pub mod front;
pub mod gateway;
pub mod keystore;
pub(crate) mod internal_capabilities;
pub(crate) mod capability_probe;
pub(crate) mod capacity;
pub mod lifecycle_steps;
pub mod mapping_update;
pub(crate) mod owned_json;
pub mod master;
pub mod master_service;
pub mod meta_extra;
pub mod mirror;
pub mod natives;
pub mod node;
pub mod readiness;
pub mod replication;
pub mod reset_password;
pub mod resize;
pub mod scratch;
pub mod security_autoconfig;
pub mod settings_validation;
pub mod stats_shards;
pub mod reindex_multi;
pub mod rest;
pub mod shard_engines;
pub mod shards;
pub mod store_records;
mod control_memory;
pub mod snapshot_api;
pub mod stateless;
pub mod stats;

/// mimalloc for every binary built on the node (thread-local free lists:
/// the bulk path allocates and frees from many threads at once).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

unsafe extern "C" {
    /// mimalloc's `mi_collect` (libmimalloc, linked by the allocator above).
    fn mi_collect(force: bool);
}

/// How the parent circuit breaker gets freed memory back to the OS before
/// measuring again: mimalloc's `mi_collect`, forced from the background
/// trim, the calling thread's pages only from idle workers (see
/// `elasticsearch_engine::memory::idle_collect`). mimalloc's own options
/// (`MIMALLOC_PURGE_DELAY` and the rest) are left as configured: the
/// allocator behaves as usual unless the parent is over its limit.
pub(crate) fn install_breaker_release_hook() {
    fn collect(force: bool) {
        // SAFETY: `mi_collect` only returns free pages to the OS.
        unsafe { mi_collect(force) }
    }
    elasticsearch_engine::memory::set_release_hook(collect);
}

/// Start a node from settings and serve HTTP until shutdown.
pub async fn run(config: config::NodeConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let readiness_port = readiness::port(&config.settings).map_err(|e| e.0)?;
    let (node, http) = node::ClusterNode::start(config).await?;
    if let Some(port) = readiness_port {
        readiness::spawn(port, node.local.id.clone(), node.subscribe());
    }
    let _ = RUNNING.set(std::sync::Arc::downgrade(&node));
    rest::register(&node);
    snapshot_api::register(&node);
    security_autoconfig::spawn_initial_node_password(&node);
    tracing::info!(
        node = %node.local.name,
        id = %node.local.id,
        http = %node.local.http_address,
        transport = %node.local.transport_address,
        "started"
    );
    if portsfile_enabled(&node.config.settings)? {
        let http_bound = http.local_addr()?;
        let transport_port = node.local.transport_address.parse::<std::net::SocketAddr>()
            .map(|a| a.port())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("transport address [{}]: {e}", node.local.transport_address)))?;
        let transport_bound = std::net::SocketAddr::new(node.config.transport_bind, transport_port);
        write_ports_file(&node.config.logs_path, "transport", transport_bound)?;
        write_ports_file(&node.config.logs_path, "http", http_bound)?;
    }
    front::serve(node, http).await?;
    Ok(())
}

/// The node [`run`] started (see [`shutdown`]).
static RUNNING: std::sync::OnceLock<std::sync::Weak<node::ClusterNode>> = std::sync::OnceLock::new();

/// Graceful-shutdown work before the process exits (SIGTERM / SIGINT):
/// make the local translogs durable, bounded by `limit`. The process then
/// exits at once (`cli`): every acknowledged write is already durable,
/// and an exit never waits on a blocking task or cancels the runtime's
/// tasks under their waiters (which aborted the process and kept its
/// listeners open while a long blocking task ran).
pub async fn shutdown(limit: std::time::Duration) {
    // ES stops the readiness service first (`doStop`): the node is no
    // longer ready once it has started to shut down.
    readiness::stop();
    let Some(node) = RUNNING.get().and_then(std::sync::Weak::upgrade) else { return };
    let sync = tokio::task::spawn_blocking(move || node.sync_translogs_for_shutdown());
    if tokio::time::timeout(limit, sync).await.is_err() {
        tracing::warn!("translog sync at shutdown did not finish within {limit:?}; exiting");
    }
}

/// `node.portsfile` (ES, default false): after binding, write the bound
/// `http` and `transport` addresses to `http.ports` / `transport.ports` in
/// `path.logs`, so a launcher that asked for port `0` (or a range) can find
/// the ports the node actually took.
fn portsfile_enabled(settings: &config::Settings) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    match settings.get("node.portsfile") {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(serde_json::Value::Bool(b)) => Ok(*b),
        Some(serde_json::Value::String(s)) if s == "true" => Ok(true),
        Some(serde_json::Value::String(s)) if s == "false" => Ok(false),
        Some(other) => Err(format!("Failed to parse value [{other}] for setting [node.portsfile] as only [true] or [false] are allowed.").into()),
    }
}

/// One bound address per line, written to a temporary file and renamed into
/// place (as ES does), so a reader never sees a partial file.
fn write_ports_file(dir: &std::path::Path, kind: &str, addr: std::net::SocketAddr) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{kind}.ports.tmp"));
    std::fs::write(&tmp, format!("{addr}\n"))?;
    std::fs::rename(&tmp, dir.join(format!("{kind}.ports")))
}

pub mod shared_sequence;
mod telemetry_owned;
