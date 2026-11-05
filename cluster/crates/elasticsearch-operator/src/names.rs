//! Object names, labels and annotations. They follow ECK's conventions so
//! that tooling written against ECK (label selectors, `kubectl` one-liners)
//! keeps working.

use std::collections::BTreeMap;

pub const LABEL_CLUSTER_NAME: &str = "elasticsearch.k8s.elastic.co/cluster-name";
pub const LABEL_STATEFULSET_NAME: &str = "elasticsearch.k8s.elastic.co/statefulset-name";
pub const LABEL_NODE_MASTER: &str = "elasticsearch.k8s.elastic.co/node-master";
pub const LABEL_NODE_DATA: &str = "elasticsearch.k8s.elastic.co/node-data";
pub const LABEL_TYPE: &str = "common.k8s.elastic.co/type";
pub const LABEL_MANAGED_BY: &str = "app.kubernetes.io/managed-by";

/// Set on the `Elasticsearch` resource once the cluster has formed (value:
/// the cluster UUID). While absent, `cluster.initial_master_nodes` is
/// rendered into the configuration.
pub const ANNOTATION_CLUSTER_UUID: &str = "elasticsearch.k8s.elastic.co/cluster-uuid";
/// Hash of the rendered configuration on the pod template: a change rolls the pods.
pub const ANNOTATION_CONFIG_HASH: &str = "elasticsearch.k8s.elastic.co/config-hash";

/// Reason recorded in node shutdown records the operator creates; used to
/// recognise (and clean up) only our own records.
pub const SHUTDOWN_REASON: &str = "elasticsearch-operator rolling restart";

pub const CONTAINER_NAME: &str = "elasticsearch";
pub const DATA_VOLUME: &str = "elasticsearch-data";
pub const LOGS_VOLUME: &str = "elasticsearch-logs";
pub const CONFIG_VOLUME: &str = "elasticsearch-config";
pub const HOME: &str = "/usr/share/elasticsearch";
pub const CONFIG_DIR: &str = "/usr/share/elasticsearch/config";
pub const DATA_DIR: &str = "/usr/share/elasticsearch/data";
pub const LOGS_DIR: &str = "/usr/share/elasticsearch/logs";
pub const HTTP_PORT: i32 = 9200;
pub const TRANSPORT_PORT: i32 = 9300;
pub const FIELD_MANAGER: &str = "elasticsearch-operator";

pub fn statefulset(cluster: &str, node_set: &str) -> String {
    format!("{cluster}-es-{node_set}")
}

/// Headless service governing a StatefulSet (same name, like ECK).
pub fn headless_service(cluster: &str, node_set: &str) -> String {
    statefulset(cluster, node_set)
}

pub fn pod(cluster: &str, node_set: &str, ordinal: i32) -> String {
    format!("{}-{ordinal}", statefulset(cluster, node_set))
}

pub fn transport_service(cluster: &str) -> String {
    format!("{cluster}-es-transport")
}

pub fn http_service(cluster: &str) -> String {
    format!("{cluster}-es-http")
}

pub fn config_map(cluster: &str, node_set: &str) -> String {
    format!("{}-es-config", statefulset(cluster, node_set))
}

pub fn pdb(cluster: &str) -> String {
    format!("{cluster}-es-default")
}

/// `<pod>` -> ordinal, for pods named `<sts>-<n>`.
pub fn ordinal_of(pod: &str) -> Option<i32> {
    pod.rsplit_once('-').and_then(|(_, n)| n.parse().ok())
}

pub fn cluster_labels(cluster: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (LABEL_CLUSTER_NAME.to_string(), cluster.to_string()),
        (LABEL_TYPE.to_string(), "elasticsearch".to_string()),
    ])
}

pub fn cluster_selector(cluster: &str) -> BTreeMap<String, String> {
    BTreeMap::from([(LABEL_CLUSTER_NAME.to_string(), cluster.to_string())])
}

pub fn node_set_selector(cluster: &str, node_set: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (LABEL_CLUSTER_NAME.to_string(), cluster.to_string()),
        (LABEL_STATEFULSET_NAME.to_string(), statefulset(cluster, node_set)),
    ])
}

pub fn label_selector_string(labels: &BTreeMap<String, String>) -> String {
    labels.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",")
}

/// DNS-1123 label check for names that become part of object names.
pub fn valid_dns_label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 63
        && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !s.starts_with('-')
        && !s.ends_with('-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eck_naming() {
        assert_eq!(statefulset("quickstart", "default"), "quickstart-es-default");
        assert_eq!(pod("quickstart", "default", 2), "quickstart-es-default-2");
        assert_eq!(http_service("quickstart"), "quickstart-es-http");
        assert_eq!(transport_service("quickstart"), "quickstart-es-transport");
        assert_eq!(config_map("q", "hot"), "q-es-hot-es-config");
        assert_eq!(ordinal_of("q-es-hot-12"), Some(12));
        assert_eq!(ordinal_of("nope"), None);
    }

    #[test]
    fn dns_labels() {
        assert!(valid_dns_label("data-hot-1"));
        assert!(!valid_dns_label("Data"));
        assert!(!valid_dns_label("-a"));
        assert!(!valid_dns_label(&"a".repeat(64)));
        assert_eq!(label_selector_string(&node_set_selector("c", "m")), "elasticsearch.k8s.elastic.co/cluster-name=c,elasticsearch.k8s.elastic.co/statefulset-name=c-es-m");
    }
}
