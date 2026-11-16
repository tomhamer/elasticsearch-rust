//! GeoIP databases on a cluster (ES `GeoIpDownloaderTaskExecutor` /
//! `DatabaseNodeService`): the elected master downloads databases into the
//! `.geoip_databases` system index (chunks and a checksum state), and every
//! node installs the stored databases it does not have yet, pulling the
//! chunks through its coordinator. See
//! `elasticsearch_engine::ingest::geoip_download`.

use crate::node::ClusterNode;
use elasticsearch_engine::ingest::{geoip, geoip_download};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The databases this node installs: the engine's own `ingest-geoip`
/// directory, searched by every engine of the process.
fn install_dir(node: &ClusterNode) -> std::path::PathBuf {
    node.engine.data.join("ingest-geoip")
}

pub fn install(node: &Arc<ClusterNode>) {
    geoip_download::set_clustered();
    geoip::add_search_dir(install_dir(node));
}

/// One pass (every few seconds): the master's downloader when due, then
/// this node's installs.
pub fn tick(node: &Arc<ClusterNode>) {
    static RUNNING: std::sync::OnceLock<parking_lot::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    let running = RUNNING.get_or_init(Default::default);
    if !running.lock().insert(node.local.id.clone()) {
        return;
    }
    struct Done<'a>(&'a parking_lot::Mutex<std::collections::HashSet<String>>, String);
    impl Drop for Done<'_> {
        fn drop(&mut self) {
            self.0.lock().remove(&self.1);
        }
    }
    let _done = Done(running, node.local.id.clone());
    let rest = match crate::front::watcher_rest_factory(node)(None) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("geoip: no coordinator call: {}", e.reason);
            return;
        }
    };
    let store = geoip_download::RestStore(rest);
    if node.is_master() {
        let mut flat: BTreeMap<String, String> = node.config.settings.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
        crate::keystore::overlay_secure(&node.config.settings, &mut flat);
        let secret = move |k: &str| flat.get(k).cloned();
        geoip_download::master_tick(&store, &node.engine, &secret, geoip_download::take_force());
    }
    // Before the cluster forms (or while `.geoip_databases` is not
    // searchable) this fails: the next pass retries.
    if let Err(e) = geoip_download::sync_node(&store, &install_dir(node)) {
        tracing::debug!("installing geoip databases failed: {e}");
    }
}
