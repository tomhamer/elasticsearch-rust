//! ES's `ReadinessService` (`readiness.port`).
//!
//! With `readiness.port` set (default `-1`: off; `0`: any free port), the
//! node opens a plain TCP listener on that port, on all interfaces, while it
//! is ready, and closes it while it is not. A connection is accepted and
//! closed at once: the open port is the signal (ECK's readiness probe is
//! `nc -z 127.0.0.1 8080`; it writes `readiness.port: "8080"` for ES >= 8.2).
//!
//! Ready means, on every applied cluster state (`checkReadyState`):
//! * the local node is not in the node shutdown metadata (any type; a node
//!   being shut down stops answering at once), and otherwise
//! * a master is elected (`masterElected`) and file-based settings are
//!   applied (`fileSettingsApplied`). Rustic has no file-based settings
//!   service, so there is nothing to wait for: the second condition holds.
//!
//! The port is closed when the node stops (`doStop`: [`stop`], from the
//! graceful shutdown). Once bound, a re-open always binds the same address
//! (`socketAddress`), so `readiness.port: 0` keeps its port.

use crate::config::{ConfigError, Settings};
use elasticsearch_cluster::state::ClusterState;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::watch;

const SETTING: &str = "readiness.port";

static STOPPING: OnceLock<watch::Sender<bool>> = OnceLock::new();

fn stopping() -> &'static watch::Sender<bool> {
    STOPPING.get_or_init(|| watch::channel(false).0)
}

/// `readiness.port` (`Setting.intSetting("readiness.port", -1)`): `None`
/// when disabled (`-1`).
pub fn port(settings: &Settings) -> Result<Option<u16>, ConfigError> {
    let raw = match settings.get(SETTING) {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::String(s)) => s.trim().to_string(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    };
    let value: i32 = raw.parse().map_err(|_| ConfigError(format!("Failed to parse value [{raw}] for setting [{SETTING}]")))?;
    match value {
        -1 => Ok(None),
        // ES's int setting has no bounds; a port outside 0..=65535 fails
        // the bind (`port out of range`), at start rather than when ready.
        0..=65535 => Ok(Some(value as u16)),
        _ => Err(ConfigError(format!("Failed to bind readiness service to port [{value}]: port out of range"))),
    }
}

/// `PluginShutdownService.shutdownNodes`: every node with a shutdown
/// record, whatever its type.
fn shutting_down(state: &ClusterState, local_id: &str) -> bool {
    state
        .metadata
        .customs
        .get(crate::rest::node_shutdown::CUSTOM)
        .and_then(|c| c.get("nodes").and_then(|n| n.as_object()))
        .is_some_and(|nodes| nodes.contains_key(local_id))
}

/// Whether the node is ready in `state` (`checkReadyState`).
pub fn ready(state: &ClusterState, local_id: &str) -> bool {
    !shutting_down(state, local_id) && state.master_node_id.is_some()
}

/// Close the readiness port for good (node stop).
pub fn stop() {
    let _ = stopping().send(true);
}

async fn accept_loop(listener: tokio::net::TcpListener) {
    loop {
        match listener.accept().await {
            // Accept and close: the connection succeeding is the answer.
            Ok((stream, _)) => drop(stream),
            Err(e) => {
                tracing::debug!("encountered exception while responding to readiness check request: {e}");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}

/// Run the readiness service for `local_id` on `port`, following the
/// applied cluster states in `states`.
pub fn spawn(port: u16, local_id: String, mut states: watch::Receiver<Arc<ClusterState>>) {
    let mut stop = stopping().subscribe();
    tokio::spawn(async move {
        let mut bound: Option<SocketAddr> = None;
        let mut listening: Option<tokio::task::JoinHandle<()>> = None;
        let (mut was_master_elected, mut was_shutting_down) = (false, false);
        loop {
            let state = states.borrow_and_update().clone();
            let master_elected = state.master_node_id.is_some();
            let down = shutting_down(&state, &local_id);
            if master_elected != was_master_elected && !down {
                tracing::info!("readiness change: masterElected={master_elected}");
            }
            if down && !was_shutting_down && listening.is_some() {
                tracing::info!("marking node as not ready because it's shutting down");
            }
            (was_master_elected, was_shutting_down) = (master_elected, down);
            let want = !*stop.borrow_and_update() && ready(&state, &local_id);
            if want && listening.is_none() {
                let addr = bound.unwrap_or(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port));
                match tokio::net::TcpListener::bind(addr).await {
                    Ok(listener) => {
                        let local = listener.local_addr().unwrap_or(addr);
                        bound = Some(local);
                        tracing::info!("readiness service up and running on {local}");
                        listening = Some(tokio::spawn(accept_loop(listener)));
                    }
                    Err(e) => tracing::error!("Failed to bind to {addr} for the readiness service: {e}"),
                }
            } else if !want {
                if let Some(task) = listening.take() {
                    tracing::info!("stopping readiness service on channel {}", bound.map(|a| a.to_string()).unwrap_or_else(|| "None".into()));
                    task.abort();
                    // The listener is dropped (the port closed) when the
                    // aborted task is.
                    let _ = task.await;
                    tracing::info!("readiness service stopped");
                }
            }
            if *stop.borrow() && listening.is_none() {
                break;
            }
            // A failed bind (port still held) is retried.
            let retry = want && listening.is_none();
            tokio::select! {
                changed = states.changed() => if changed.is_err() { break },
                changed = stop.changed() => if changed.is_err() { break },
                _ = tokio::time::sleep(Duration::from_secs(1)), if retry => {}
            }
        }
        if let Some(task) = listening.take() {
            task.abort();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn port_setting() {
        let s = |v: Value| -> Settings { [(SETTING.to_string(), v)].into_iter().collect() };
        assert_eq!(port(&Settings::new()).unwrap(), None);
        assert_eq!(port(&s(json!("-1"))).unwrap(), None);
        assert_eq!(port(&s(json!("8080"))).unwrap(), Some(8080));
        assert_eq!(port(&s(json!(0))).unwrap(), Some(0));
        assert_eq!(port(&s(json!("x"))).unwrap_err().0, "Failed to parse value [x] for setting [readiness.port]");
        assert!(port(&s(json!("70000"))).is_err());
    }

    #[test]
    fn ready_needs_a_master_and_no_shutdown_record() {
        let mut state = ClusterState::empty("c");
        assert!(!ready(&state, "n1"));
        state.master_node_id = Some("m".into());
        assert!(ready(&state, "n1"));
        state.metadata.customs.insert(crate::rest::node_shutdown::CUSTOM.into(), json!({"nodes": {"n2": {"type": "restart"}}}));
        assert!(ready(&state, "n1"));
        state.metadata.customs.insert(crate::rest::node_shutdown::CUSTOM.into(), json!({"nodes": {"n1": {"type": "restart"}}}));
        assert!(!ready(&state, "n1"));
    }

    #[tokio::test]
    async fn port_opens_when_ready_and_closes_when_not() {
        // Find a free port, then serve readiness on it.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let (tx, rx) = watch::channel(Arc::new(ClusterState::empty("c")));
        spawn(port, "n1".into(), rx);
        let connect = || async move { tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() };
        let wait_for = |want: bool| async move {
            for _ in 0..100 {
                if connect().await == want {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            false
        };
        assert!(wait_for(false).await, "closed without a master");
        let mut with_master = ClusterState::empty("c");
        with_master.master_node_id = Some("m".into());
        tx.send(Arc::new(with_master.clone())).unwrap();
        assert!(wait_for(true).await, "open once a master is elected");
        let mut shutting = with_master.clone();
        shutting.metadata.customs.insert(crate::rest::node_shutdown::CUSTOM.into(), json!({"nodes": {"n1": {"type": "restart"}}}));
        tx.send(Arc::new(shutting)).unwrap();
        assert!(wait_for(false).await, "closed while shutting down");
        tx.send(Arc::new(with_master)).unwrap();
        assert!(wait_for(true).await, "re-opened on the same port");
    }
}
