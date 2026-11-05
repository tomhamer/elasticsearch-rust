//! Listener ports for test nodes. Every node binds port 0 (the OS picks a
//! free port) and the test reads back what it took: an in-process node from
//! its published addresses, a node process from the `node.portsfile` files.
//! Fixed ports collided with sockets of earlier tests, with parallel tests
//! and with ephemeral client ports (`Address already in use`).
//!
//! Only a binary that cannot report its ports (an old release under a
//! rolling-upgrade test) gets ports chosen here ([`probe_free_ports`]), and
//! its launcher retries when one was taken in between ([`is_addr_in_use`]).
#![allow(dead_code)]

use elasticsearch_node::node::ClusterNode;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// The port of a published address (`127.0.0.1:1234`, `[::1]:1234`).
pub fn port_of(addr: &str) -> u16 {
    addr.parse::<SocketAddr>().map(|a| a.port()).ok()
        .or_else(|| addr.rsplit(':').next().and_then(|p| p.trim().parse().ok()))
        .unwrap_or_else(|| panic!("no port in address [{addr}]"))
}

/// The HTTP port an in-process node bound.
pub fn http_port(node: &ClusterNode) -> u16 {
    port_of(&node.local.http_address)
}

/// The transport port an in-process node bound.
pub fn transport_port(node: &ClusterNode) -> u16 {
    port_of(&node.local.transport_address)
}

/// `127.0.0.1:{port}` seed hosts for `discovery.seed_hosts`.
pub fn seeds(ports: &[u16]) -> Vec<String> {
    ports.iter().filter(|p| **p != 0).map(|p| format!("127.0.0.1:{p}")).collect()
}

/// A tag unique within this test run (cluster names): ports no longer
/// tell clusters apart.
pub fn unique(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    format!("{prefix}-{}-{}-{nanos}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Bind a probe listener on an OS-assigned loopback port with
/// `SO_REUSEADDR` (so a port of a closed socket in TIME_WAIT is no obstacle
/// to the node binding it next). Rust's `TcpListener::bind` sets
/// `SO_REUSEADDR` on every Unix; Windows has no TIME_WAIT bind conflict
/// for a listener (and its `SO_REUSEADDR` would allow a double bind).
fn probe() -> TcpListener {
    TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind a probe listener on 127.0.0.1:0")
}

/// `n` distinct free loopback ports, for a node binary that cannot report
/// the ports it bound. Every probe is held until all are chosen (so the
/// ports are distinct) and released on return: spawn the node right after,
/// and retry with fresh ports when it fails with [`is_addr_in_use`].
pub fn probe_free_ports(n: usize) -> Vec<u16> {
    let held: Vec<TcpListener> = (0..n).map(|_| probe()).collect();
    held.iter().map(|l| l.local_addr().expect("probe address").port()).collect()
}

/// Whether a node's output says it could not bind its port.
pub fn is_addr_in_use(text: &str) -> bool {
    text.contains("Address already in use") || text.contains("AddrInUse") || text.contains("os error 98") || text.contains("os error 48")
        || text.contains("os error 10048")
}

/// The port a node process recorded in `{logs}/{kind}.ports` (`node.portsfile`),
/// if it has written it yet.
pub fn read_ports_file(logs: &Path, kind: &str) -> Option<u16> {
    let text = std::fs::read_to_string(logs.join(format!("{kind}.ports"))).ok()?;
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(port_of(line))
}

/// Remove `{logs}/{kind}.ports` left by an earlier run of the node.
pub fn clear_ports_files(logs: &Path) {
    for kind in ["http", "transport"] {
        let _ = std::fs::remove_file(logs.join(format!("{kind}.ports")));
    }
}

/// Wait (up to `within`) until the cluster of the nodes listening on `http`
/// has formed and recovered: every node sees `count` nodes and has applied a
/// recovered cluster state (no "state not recovered" block; with no red index
/// that is health yellow or green). The node that answers health first is
/// not enough: a node reports a node count as soon as it applied the state
/// with both members, while the elected master may still be applying it and
/// refuse metadata requests with `503 state not recovered / initialized`.
/// `auth`: basic credentials of a secured cluster. `Err`: what the last node
/// not yet ready answered.
pub async fn wait_formed(c: &reqwest::Client, http: &[u16], auth: Option<&(String, String)>, count: usize, within: std::time::Duration) -> Result<(), String> {
    let deadline = std::time::Instant::now() + within;
    let get = |port: u16, path: String| {
        let mut r = c.get(format!("http://127.0.0.1:{port}{path}")).timeout(std::time::Duration::from_secs(10));
        if let Some((user, password)) = auth {
            r = r.basic_auth(user, Some(password));
        }
        async move {
            let resp = r.send().await.map_err(|e| e.to_string())?;
            let status = resp.status().as_u16();
            let body = resp.json::<serde_json::Value>().await.map_err(|e| e.to_string())?;
            Ok::<_, String>((status, body))
        }
    };
    let mut ready = 0;
    let mut last = String::new();
    while ready < http.len() {
        let port = http[ready];
        let ok = match get(port, format!("/_cluster/health?wait_for_nodes={count}&timeout=1s")).await {
            Ok((_, h)) if h["number_of_nodes"] == count && h["timed_out"] == false => {
                match get(port, "/_cluster/state/blocks?local=true".to_string()).await {
                    Ok((200, s)) if !s["blocks"]["global"].as_object().is_some_and(|g| g.values().any(|b| b["description"] == "state not recovered / initialized")) => true,
                    Ok((st, s)) => {
                        last = format!("port {port}: blocks {st} {s}");
                        false
                    }
                    Err(e) => {
                        last = format!("port {port}: blocks {e}");
                        false
                    }
                }
            }
            Ok((st, h)) => {
                last = format!("port {port}: health {st} {h}");
                false
            }
            Err(e) => {
                last = format!("port {port}: {e}");
                false
            }
        };
        if ok {
            ready += 1;
            continue;
        }
        if std::time::Instant::now() >= deadline {
            return Err(last);
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    Ok(())
}
