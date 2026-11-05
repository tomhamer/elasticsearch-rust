//! The `Elasticsearch` custom resource: an ECK-compatible subset of
//! `elasticsearch.k8s.elastic.co/v1`.
//!
//! Field names and semantics follow ECK so that a manifest written for ECK
//! (restricted to the fields below) works unchanged:
//!
//! ```yaml
//! apiVersion: elasticsearch.k8s.elastic.co/v1
//! kind: Elasticsearch
//! metadata: { name: quickstart }
//! spec:
//!   version: 9.1.0
//!   nodeSets:
//!   - name: default
//!     count: 3
//!     config: { node.store.allow_mmap: true }
//! ```
//!
//! `config`, `podTemplate` and `volumeClaimTemplates` are free-form
//! (`x-kubernetes-preserve-unknown-fields`) exactly like ECK's, and are
//! decoded into typed Kubernetes objects at reconcile time so that a bad
//! value surfaces as an `Invalid` phase instead of a rejected watch event.

use kube::CustomResource;
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// API group, shared with ECK.
pub const GROUP: &str = "elasticsearch.k8s.elastic.co";
pub const VERSION: &str = "v1";

#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[kube(
    group = "elasticsearch.k8s.elastic.co",
    version = "v1",
    kind = "Elasticsearch",
    plural = "elasticsearches",
    singular = "elasticsearch",
    shortname = "es",
    namespaced,
    status = "ElasticsearchStatus",
    printcolumn = r#"{"name":"health","type":"string","jsonPath":".status.health"}"#,
    printcolumn = r#"{"name":"nodes","type":"integer","description":"Available nodes","jsonPath":".status.availableNodes"}"#,
    printcolumn = r#"{"name":"version","type":"string","description":"Elasticsearch version","jsonPath":".status.version"}"#,
    printcolumn = r#"{"name":"phase","type":"string","jsonPath":".status.phase"}"#,
    printcolumn = r#"{"name":"age","type":"date","jsonPath":".metadata.creationTimestamp"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct ElasticsearchSpec {
    /// Elasticsearch version (used for the default image tag and reported in status).
    pub version: String,
    /// Container image; defaults to `<operator default image repository>:<version>`
    /// (repository from the operator's `--default-image-repository`, default `elasticsearch-rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Groups of identically configured nodes; each becomes one StatefulSet.
    pub node_sets: Vec<NodeSet>,
    /// Constraints applied to rolling changes and topology changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_strategy: Option<UpdateStrategy>,
    /// PodDisruptionBudget template (`{metadata, spec}`). Omitted: a default
    /// budget allowing one disruption while the cluster is green. Set with an
    /// empty `spec` (`podDisruptionBudget: {spec: {}}`) to disable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "free_form_object")]
    pub pod_disruption_budget: Option<Value>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeSet {
    /// Name of the node set; part of the StatefulSet and pod names.
    pub name: String,
    /// Number of Elasticsearch nodes in this set.
    #[serde(default = "default_count")]
    #[schemars(range(min = 0))]
    pub count: i32,
    /// Free-form `elasticsearch.yml` settings (nested or dotted keys).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "free_form_object")]
    pub config: Option<Value>,
    /// Free-form PodTemplateSpec merged over the operator's defaults; the
    /// main container is named `elasticsearch`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "free_form_object")]
    pub pod_template: Option<Value>,
    /// PersistentVolumeClaim templates. The claim named `elasticsearch-data`
    /// is mounted at the data path; default 1Gi ReadWriteOnce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "free_form_array")]
    pub volume_claim_templates: Option<Vec<Value>>,
}

fn default_count() -> i32 {
    1
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateStrategy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_budget: Option<ChangeBudget>,
}

/// ECK semantics: `maxSurge` = how many pods may exist above the total
/// desired count during changes (absent or negative: unbounded);
/// `maxUnavailable` = how many pods may be unavailable (absent: 1; negative:
/// unbounded).
#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChangeBudget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_surge: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_unavailable: Option<i32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ElasticsearchStatus {
    /// `green`, `yellow`, `red` or `unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health: Option<String>,
    /// `Ready`, `ApplyingChanges`, `MigratingData`, `Stalled` or `Invalid`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default)]
    pub available_nodes: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Condition>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    /// `ReconciliationComplete`, `RunningDesiredVersion`, `ElasticsearchIsReachable`, `ResourcesAwareManagement`.
    #[serde(rename = "type")]
    pub type_: String,
    /// `True`, `False` or `Unknown`.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_transition_time: Option<String>,
}

fn free_form_object(_: &mut SchemaGenerator) -> Schema {
    json_schema!({ "type": "object", "x-kubernetes-preserve-unknown-fields": true })
}

fn free_form_array(_: &mut SchemaGenerator) -> Schema {
    json_schema!({
        "type": "array",
        "items": { "type": "object", "x-kubernetes-preserve-unknown-fields": true }
    })
}

impl ElasticsearchSpec {
    pub fn max_unavailable(&self) -> Option<i32> {
        let v = self.update_strategy.as_ref().and_then(|u| u.change_budget.as_ref()).and_then(|c| c.max_unavailable);
        match v {
            None => Some(1),
            Some(n) if n < 0 => None,
            Some(n) => Some(n),
        }
    }

    pub fn max_surge(&self) -> Option<i32> {
        match self.update_strategy.as_ref().and_then(|u| u.change_budget.as_ref()).and_then(|c| c.max_surge) {
            Some(n) if n >= 0 => Some(n),
            _ => None,
        }
    }
}

/// The CRD as YAML (what `elasticsearch-operator crd` prints).
pub fn crd_yaml() -> String {
    use kube::CustomResourceExt;
    serde_yaml::to_string(&Elasticsearch::crd()).expect("CRD serializes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kube::CustomResourceExt;

    #[test]
    fn crd_identity_matches_eck() {
        let crd = Elasticsearch::crd();
        assert_eq!(crd.metadata.name.as_deref(), Some("elasticsearches.elasticsearch.k8s.elastic.co"));
        assert_eq!(crd.spec.group, GROUP);
        assert_eq!(crd.spec.names.kind, "Elasticsearch");
        assert_eq!(crd.spec.names.short_names.as_deref(), Some(&["es".to_string()][..]));
        assert_eq!(crd.spec.scope, "Namespaced");
        let v = &crd.spec.versions[0];
        assert_eq!(v.name, VERSION);
        assert!(v.subresources.as_ref().unwrap().status.is_some());
        let yaml = crd_yaml();
        assert!(yaml.contains("x-kubernetes-preserve-unknown-fields: true"));
        assert!(yaml.contains("nodeSets"));
        assert!(yaml.contains("volumeClaimTemplates"));
    }

    #[test]
    fn eck_manifest_parses() {
        let doc = r#"
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata: { name: quickstart, namespace: default }
spec:
  version: 9.1.0
  updateStrategy: { changeBudget: { maxSurge: 3, maxUnavailable: 1 } }
  nodeSets:
  - name: masters
    count: 3
    config:
      node.roles: ["master"]
      node:
        attr: { zone: a }
    podTemplate:
      spec:
        containers:
        - name: elasticsearch
          resources: { limits: { memory: 4Gi } }
    volumeClaimTemplates:
    - metadata: { name: elasticsearch-data }
      spec: { accessModes: [ReadWriteOnce], resources: { requests: { storage: 5Gi } } }
"#;
        let es: Elasticsearch = serde_yaml::from_str(doc).unwrap();
        assert_eq!(es.spec.node_sets[0].count, 3);
        assert_eq!(es.spec.max_surge(), Some(3));
        assert_eq!(es.spec.max_unavailable(), Some(1));
        assert!(es.spec.pod_disruption_budget.is_none());
    }

    #[test]
    fn change_budget_defaults() {
        let mut spec = ElasticsearchSpec::default();
        assert_eq!(spec.max_unavailable(), Some(1));
        assert_eq!(spec.max_surge(), None);
        spec.update_strategy = Some(UpdateStrategy { change_budget: Some(ChangeBudget { max_surge: Some(-1), max_unavailable: Some(-1) }) });
        assert_eq!(spec.max_unavailable(), None);
        assert_eq!(spec.max_surge(), None);
    }
}
