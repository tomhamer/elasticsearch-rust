//! Rendering the per-nodeSet `elasticsearch.yml` and validating the spec.
//!
//! The node reads this file from `$ES_PATH_CONF/elasticsearch.yml` and
//! substitutes `${POD_NAME}` / `${POD_IP}` from the environment (ES
//! semantics: an unresolvable placeholder fails startup).

use crate::crd::{ElasticsearchSpec, NodeSet};
use crate::names;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub type FlatSettings = BTreeMap<String, Value>;

/// Roles accepted in `node.roles` (ES 8/9 role names).
pub const KNOWN_ROLES: &[&str] = &[
    "master",
    "voting_only",
    "data",
    "data_content",
    "data_hot",
    "data_warm",
    "data_cold",
    "data_frozen",
    "ingest",
    "ml",
    "remote_cluster_client",
    "transform",
];

/// Settings the operator owns; user values for them are ignored (and reported).
pub const MANAGED_SETTINGS: &[&str] = &[
    "cluster.name",
    "node.name",
    "network.host",
    "network.publish_host",
    "http.port",
    "transport.port",
    "path.data",
    "path.logs",
    "discovery.seed_hosts",
    "discovery.seed_providers",
    "discovery.type",
    "cluster.initial_master_nodes",
];

pub const INITIAL_MASTER_NODES: &str = "cluster.initial_master_nodes";

/// Flatten nested config into dotted keys (ES `Settings` semantics).
pub fn flatten(v: &Value) -> Result<FlatSettings, String> {
    fn go(prefix: &str, v: &Value, out: &mut FlatSettings) -> Result<(), String> {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    go(&key, v, out)?;
                }
                Ok(())
            }
            other => {
                if prefix.is_empty() {
                    return Err("config must be an object".into());
                }
                if out.insert(prefix.to_string(), other.clone()).is_some() {
                    return Err(format!("duplicate settings key [{prefix}]"));
                }
                Ok(())
            }
        }
    }
    let mut out = FlatSettings::new();
    match v {
        Value::Null => Ok(out),
        Value::Object(_) => {
            go("", v, &mut out)?;
            Ok(out)
        }
        _ => Err("config must be an object".into()),
    }
}

pub fn user_settings(ns: &NodeSet) -> Result<FlatSettings, String> {
    ns.config.as_ref().map(flatten).unwrap_or_else(|| Ok(FlatSettings::new()))
}

/// `node.roles` of a nodeSet; `None` = not set = ES default (all roles).
pub fn roles(ns: &NodeSet) -> Result<Option<Vec<String>>, String> {
    let s = user_settings(ns)?;
    match s.get("node.roles") {
        None => Ok(None),
        Some(Value::Array(a)) => a.iter().map(|r| r.as_str().map(str::to_string).ok_or_else(|| "node.roles must be a list of strings".to_string())).collect::<Result<Vec<_>, _>>().map(Some),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(Some(vec![])),
        Some(Value::String(s)) => Ok(Some(s.split(',').map(|p| p.trim().to_string()).collect())),
        Some(_) => Err("node.roles must be a list of strings".into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct NodeSetRoles {
    pub master: bool,
    pub data: bool,
}

pub fn node_set_roles(ns: &NodeSet) -> NodeSetRoles {
    match roles(ns) {
        Ok(None) | Err(_) => NodeSetRoles { master: true, data: true },
        Ok(Some(r)) => NodeSetRoles {
            master: r.iter().any(|x| x == "master"),
            data: r.iter().any(|x| x == "data" || x.starts_with("data_")),
        },
    }
}

/// Validate the spec before touching any object. Returns human-readable errors.
pub fn validate(cluster: &str, spec: &ElasticsearchSpec) -> Vec<String> {
    let mut errs = Vec::new();
    if spec.version.trim().is_empty() {
        errs.push("spec.version must be set".into());
    }
    if spec.node_sets.is_empty() {
        errs.push("spec.nodeSets must contain at least one node set".into());
    }
    let mut seen = BTreeSet::new();
    let mut masters = 0;
    for ns in &spec.node_sets {
        if !names::valid_dns_label(&ns.name) {
            errs.push(format!("nodeSet name [{}] must be a DNS-1123 label", ns.name));
        }
        if !seen.insert(ns.name.clone()) {
            errs.push(format!("duplicate nodeSet name [{}]", ns.name));
        }
        // StatefulSet names feed the `controller-revision-hash` label (63
        // chars incl. an 11-char suffix), hence ECK's 52-char limit.
        let sts = names::statefulset(cluster, &ns.name);
        if sts.len() > 52 {
            errs.push(format!("StatefulSet name [{sts}] exceeds 52 characters; shorten the cluster or nodeSet name"));
        }
        if ns.count < 0 {
            errs.push(format!("nodeSet [{}] count must be >= 0", ns.name));
        }
        match user_settings(ns) {
            Err(e) => errs.push(format!("nodeSet [{}]: {e}", ns.name)),
            Ok(_) => match roles(ns) {
                Err(e) => errs.push(format!("nodeSet [{}]: {e}", ns.name)),
                Ok(Some(rs)) => {
                    for r in &rs {
                        if !KNOWN_ROLES.contains(&r.as_str()) {
                            errs.push(format!("nodeSet [{}]: unknown role [{r}]", ns.name));
                        }
                    }
                    if rs.iter().any(|r| r == "voting_only") && !rs.iter().any(|r| r == "master") {
                        errs.push(format!("nodeSet [{}]: voting_only requires the master role", ns.name));
                    }
                }
                Ok(None) => {}
            },
        }
        if node_set_roles(ns).master {
            masters += ns.count.max(0);
        }
        if let Some(pt) = &ns.pod_template {
            if let Err(e) = serde_json::from_value::<k8s_openapi::api::core::v1::PodTemplateSpec>(pt.clone()) {
                errs.push(format!("nodeSet [{}]: invalid podTemplate: {e}", ns.name));
            }
        }
        for v in ns.volume_claim_templates.iter().flatten() {
            if let Err(e) = serde_json::from_value::<k8s_openapi::api::core::v1::PersistentVolumeClaim>(v.clone()) {
                errs.push(format!("nodeSet [{}]: invalid volumeClaimTemplate: {e}", ns.name));
            }
        }
    }
    if !spec.node_sets.is_empty() && masters == 0 {
        errs.push("at least one master-eligible node is required".into());
    }
    errs
}

/// Master-eligible pod names of the desired topology, for
/// `cluster.initial_master_nodes`.
pub fn initial_master_nodes(cluster: &str, spec: &ElasticsearchSpec) -> Vec<String> {
    spec.node_sets
        .iter()
        .filter(|ns| node_set_roles(ns).master)
        .flat_map(|ns| (0..ns.count.max(0)).map(move |i| names::pod(cluster, &ns.name, i)))
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderedConfig {
    pub yaml: String,
    /// Hash of every setting except `cluster.initial_master_nodes` (which
    /// is dropped after bootstrap and must not cause a rolling restart).
    pub hash: String,
    /// User settings ignored because the operator manages them.
    pub ignored: Vec<String>,
}

/// Render `elasticsearch.yml` for one nodeSet. `initial_masters` is `Some`
/// only while the cluster has never formed.
pub fn render(cluster: &str, namespace: &str, ns: &NodeSet, initial_masters: Option<&[String]>) -> Result<RenderedConfig, String> {
    let user = user_settings(ns)?;
    let mut out = FlatSettings::new();
    let mut ignored = Vec::new();
    for (k, v) in user {
        if MANAGED_SETTINGS.contains(&k.as_str()) {
            ignored.push(k);
        } else {
            out.insert(k, v);
        }
    }
    let s = |x: &str| Value::String(x.to_string());
    out.insert("cluster.name".into(), s(cluster));
    out.insert("node.name".into(), s("${POD_NAME}"));
    out.insert("network.host".into(), s("0.0.0.0"));
    out.insert("network.publish_host".into(), s("${POD_IP}"));
    out.insert("http.port".into(), s(&names::HTTP_PORT.to_string()));
    out.insert("transport.port".into(), s(&names::TRANSPORT_PORT.to_string()));
    out.insert("path.data".into(), s(names::DATA_DIR));
    out.insert("path.logs".into(), s(names::LOGS_DIR));
    // The cluster-wide transport service is headless with
    // publishNotReadyAddresses: its name resolves to every pod's IP.
    out.insert("discovery.seed_hosts".into(), Value::Array(vec![s(&format!("{}.{namespace}.svc", names::transport_service(cluster)))]));

    let hash = {
        let yaml = serde_yaml::to_string(&out).map_err(|e| e.to_string())?;
        let digest = Sha256::digest(yaml.as_bytes());
        digest.iter().take(12).map(|b| format!("{b:02x}")).collect::<String>()
    };
    if let Some(m) = initial_masters {
        out.insert(INITIAL_MASTER_NODES.into(), Value::Array(m.iter().map(|x| s(x)).collect()));
    }
    // The node binary defaults `xpack.security.enabled` to true (ES 8+), but
    // the operator's probes and client speak plain HTTP without credentials
    // and it provisions no transport certificates: keep security off unless
    // the nodeSet config sets it. (Outside the hash: an operator upgrade
    // alone does not roll existing clusters.)
    if !out.contains_key("xpack.security.enabled") {
        out.insert("xpack.security.enabled".into(), s("false"));
    }
    let header = "# Managed by elasticsearch-operator; edit spec.nodeSets[].config instead.\n";
    let yaml = format!("{header}{}", serde_yaml::to_string(&out).map_err(|e| e.to_string())?);
    Ok(RenderedConfig { yaml, hash, ignored })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ns(name: &str, count: i32, config: Value) -> NodeSet {
        NodeSet { name: name.into(), count, config: Some(config), ..Default::default() }
    }

    #[test]
    fn flatten_nested_and_dotted() {
        let f = flatten(&json!({"node": {"roles": ["master"], "attr": {"zone": "a"}}, "cluster.routing.allocation.awareness.attributes": "zone"})).unwrap();
        assert_eq!(f["node.roles"], json!(["master"]));
        assert_eq!(f["node.attr.zone"], json!("a"));
        assert!(f.contains_key("cluster.routing.allocation.awareness.attributes"));
        assert!(flatten(&json!({"a.b": 1, "a": {"b": 2}})).is_err());
        assert!(flatten(&json!([1])).is_err());
    }

    #[test]
    fn roles_detection() {
        assert_eq!(node_set_roles(&NodeSet::default()), NodeSetRoles { master: true, data: true });
        assert_eq!(node_set_roles(&ns("m", 3, json!({"node.roles": ["master"]}))), NodeSetRoles { master: true, data: false });
        assert_eq!(node_set_roles(&ns("h", 3, json!({"node": {"roles": ["data_hot", "ingest"]}}))), NodeSetRoles { master: false, data: true });
        assert_eq!(node_set_roles(&ns("c", 2, json!({"node.roles": []}))), NodeSetRoles { master: false, data: false });
    }

    #[test]
    fn initial_masters_and_render() {
        let spec = ElasticsearchSpec {
            version: "9.1.0".into(),
            node_sets: vec![ns("master", 3, json!({"node.roles": ["master"]})), ns("data", 2, json!({"node.roles": ["data"], "cluster.name": "evil"}))],
            ..Default::default()
        };
        let im = initial_master_nodes("prod", &spec);
        assert_eq!(im, vec!["prod-es-master-0", "prod-es-master-1", "prod-es-master-2"]);
        let boot = render("prod", "ns1", &spec.node_sets[1], Some(&im)).unwrap();
        let after = render("prod", "ns1", &spec.node_sets[1], None).unwrap();
        assert!(boot.yaml.contains("cluster.initial_master_nodes"));
        assert!(!after.yaml.contains("initial_master_nodes"));
        // dropping initial_master_nodes must not roll the pods
        assert_eq!(boot.hash, after.hash);
        assert_eq!(after.ignored, vec!["cluster.name".to_string()]);
        let parsed: BTreeMap<String, Value> = serde_yaml::from_str(&after.yaml).unwrap();
        assert_eq!(parsed["cluster.name"], json!("prod"));
        assert_eq!(parsed["node.name"], json!("${POD_NAME}"));
        assert_eq!(parsed["discovery.seed_hosts"], json!(["prod-es-transport.ns1.svc"]));
        assert_eq!(parsed["node.roles"], json!(["data"]));
        // a user config change changes the hash
        let changed = render("prod", "ns1", &ns("data", 2, json!({"node.roles": ["data", "ingest"]})), None).unwrap();
        assert_ne!(changed.hash, after.hash);
    }

    #[test]
    fn validation() {
        let ok = ElasticsearchSpec { version: "9.1.0".into(), node_sets: vec![ns("default", 3, json!({}))], ..Default::default() };
        assert!(validate("c", &ok).is_empty());
        let no_master = ElasticsearchSpec { version: "9".into(), node_sets: vec![ns("d", 3, json!({"node.roles": ["data"]}))], ..Default::default() };
        assert_eq!(validate("c", &no_master), vec!["at least one master-eligible node is required".to_string()]);
        let bad = ElasticsearchSpec {
            version: "9".into(),
            node_sets: vec![
                ns("Bad_Name", 1, json!({})),
                ns("x", -1, json!({"node.roles": ["wizard"]})),
                ns("x", 1, json!({})),
                NodeSet { name: "pt".into(), count: 1, pod_template: Some(json!({"spec": {"containers": "nope"}})), ..Default::default() },
            ],
            ..Default::default()
        };
        let errs = validate("c", &bad);
        assert!(errs.iter().any(|e| e.contains("DNS-1123")), "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("duplicate nodeSet")));
        assert!(errs.iter().any(|e| e.contains("count must be")));
        assert!(errs.iter().any(|e| e.contains("unknown role [wizard]")));
        assert!(errs.iter().any(|e| e.contains("invalid podTemplate")));
        let long = ElasticsearchSpec { version: "9".into(), node_sets: vec![ns(&"n".repeat(50), 1, json!({}))], ..Default::default() };
        assert!(validate("cluster", &long).iter().any(|e| e.contains("52 characters")));
    }
}
