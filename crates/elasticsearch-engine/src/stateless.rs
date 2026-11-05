//! Stateless mode (ES `x-pack/plugin/stateless`, `stateless.enabled`):
//! the node-level defaults the stateless plugin adds
//! (`StatelessPlugin.additionalSettings`). Shard placement by the `index`
//! / `search` node roles lives in the cluster allocator.

use serde_json::{Map, Value};

use crate::node::Node;

/// `DiscoveryNode.STATELESS_ENABLED_SETTING_NAME`.
pub const ENABLED_SETTING: &str = "stateless.enabled";
/// `DataStreamFailureStoreDefinition.FAILURE_STORE_REFRESH_INTERVAL_SETTING_NAME`.
pub const FAILURE_STORE_REFRESH_INTERVAL: &str = "data_streams.failure_store.refresh_interval";
/// `StatelessPlugin.FAILURE_STORE_REFRESH_INTERVAL_SETTING` default.
pub const STATELESS_FAILURE_STORE_REFRESH_INTERVAL: &str = "30s";

fn truthy(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

/// `DiscoveryNode.isStateless(settings)`.
pub fn enabled_in(settings: &Map<String, Value>) -> bool {
    truthy(settings.get(ENABLED_SETTING))
}

/// This node runs in stateless mode.
pub fn enabled(node: &Node) -> bool {
    enabled_in(&node.node_settings.read())
}

/// The refresh interval failure store indices are created with, if any:
/// the node setting, else the stateless default (30s), else none (ES
/// `buildFailureStoreIndexSettings`).
pub fn failure_store_refresh_interval_in(settings: &Map<String, Value>) -> Option<String> {
    match settings.get(FAILURE_STORE_REFRESH_INTERVAL) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => enabled_in(settings).then(|| STATELESS_FAILURE_STORE_REFRESH_INTERVAL.to_string()),
        Some(other) => Some(other.to_string()),
    }
}

/// [`failure_store_refresh_interval_in`] of this node.
pub fn failure_store_refresh_interval(node: &Node) -> Option<String> {
    failure_store_refresh_interval_in(&node.node_settings.read())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn settings(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn stateless_flag() {
        assert!(!enabled_in(&Map::new()));
        assert!(enabled_in(&settings(json!({"stateless.enabled": "true"}))));
        assert!(enabled_in(&settings(json!({"stateless.enabled": true}))));
        assert!(!enabled_in(&settings(json!({"stateless.enabled": "false"}))));
    }

    #[test]
    fn failure_store_refresh_interval_defaults() {
        assert_eq!(failure_store_refresh_interval_in(&Map::new()), None);
        assert_eq!(failure_store_refresh_interval_in(&settings(json!({"stateless.enabled": "true"}))).as_deref(), Some("30s"));
        assert_eq!(failure_store_refresh_interval_in(&settings(json!({"data_streams.failure_store.refresh_interval": "5s"}))).as_deref(), Some("5s"));
        assert_eq!(failure_store_refresh_interval_in(&settings(json!({"stateless.enabled": "true", "data_streams.failure_store.refresh_interval": "1m"}))).as_deref(), Some("1m"));
    }
}
