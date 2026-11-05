//! Pure builders for the Kubernetes objects the operator owns.

use crate::crd::{Elasticsearch, NodeSet};
use crate::es_client::Health;
use crate::names::{self, *};
use crate::settings::{self, RenderedConfig};
use k8s_openapi::api::apps::v1::{StatefulSet, StatefulSetPersistentVolumeClaimRetentionPolicy, StatefulSetSpec, StatefulSetUpdateStrategy};
use k8s_openapi::api::core::v1::{
    Affinity, ConfigMap, ConfigMapVolumeSource, Container, ContainerPort, EmptyDirVolumeSource, EnvVar, EnvVarSource, ExecAction, ObjectFieldSelector, PersistentVolumeClaim,
    PersistentVolumeClaimSpec, PodAffinityTerm, PodAntiAffinity, PodSecurityContext, PodSpec, PodTemplateSpec, Probe, ResourceRequirements, Service, ServicePort, ServiceSpec,
    Volume, VolumeMount, VolumeResourceRequirements, WeightedPodAffinityTerm,
};
use k8s_openapi::api::policy::v1::{PodDisruptionBudget, PodDisruptionBudgetSpec};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta, OwnerReference};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::{Resource, ResourceExt};
use std::collections::BTreeMap;

pub const DEFAULT_IMAGE_REPOSITORY: &str = "elasticsearch-rs";
pub const DEFAULT_MEMORY: &str = "2Gi";
pub const DEFAULT_STORAGE: &str = "1Gi";
pub const DEFAULT_TERMINATION_GRACE_SECONDS: i64 = 180;

/// Readiness: the node's HTTP layer answers `GET /_cluster/health?local=true`
/// with 200 within 3 seconds (same contract as ECK's readiness script).
pub const READINESS_SCRIPT: &str = r#"code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:9200/_cluster/health?local=true") && [ "$code" = 200 ]"#;

pub fn image(es: &Elasticsearch, default_repo: &str) -> String {
    es.spec.image.clone().unwrap_or_else(|| format!("{default_repo}:{}", es.spec.version))
}

pub fn owner_ref(es: &Elasticsearch) -> OwnerReference {
    es.controller_owner_ref(&()).unwrap_or_else(|| OwnerReference {
        api_version: "elasticsearch.k8s.elastic.co/v1".into(),
        kind: "Elasticsearch".into(),
        name: es.name_any(),
        uid: es.meta().uid.clone().unwrap_or_default(),
        controller: Some(true),
        block_owner_deletion: Some(true),
    })
}

fn meta(es: &Elasticsearch, name: String, extra: BTreeMap<String, String>) -> ObjectMeta {
    let mut labels = cluster_labels(&es.name_any());
    labels.extend(extra);
    labels.insert(LABEL_MANAGED_BY.into(), FIELD_MANAGER.into());
    ObjectMeta { name: Some(name), namespace: es.namespace(), labels: Some(labels), owner_references: Some(vec![owner_ref(es)]), ..Default::default() }
}

fn port(name: &str, p: i32) -> ServicePort {
    ServicePort { name: Some(name.into()), port: p, target_port: Some(IntOrString::Int(p)), protocol: Some("TCP".into()), ..Default::default() }
}

/// Cluster-wide headless transport service: every pod, ready or not. Its
/// DNS name is the discovery seed.
pub fn transport_service(es: &Elasticsearch) -> Service {
    let c = es.name_any();
    Service {
        metadata: meta(es, names::transport_service(&c), BTreeMap::new()),
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()),
            publish_not_ready_addresses: Some(true),
            selector: Some(cluster_selector(&c)),
            ports: Some(vec![port("transport", TRANSPORT_PORT)]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Client-facing HTTP service (ready pods only).
pub fn http_service(es: &Elasticsearch) -> Service {
    let c = es.name_any();
    Service {
        metadata: meta(es, names::http_service(&c), BTreeMap::new()),
        spec: Some(ServiceSpec { type_: Some("ClusterIP".into()), selector: Some(cluster_selector(&c)), ports: Some(vec![port("http", HTTP_PORT)]), ..Default::default() }),
        ..Default::default()
    }
}

/// Per-nodeSet headless service governing the StatefulSet (stable pod DNS).
pub fn headless_service(es: &Elasticsearch, ns: &NodeSet) -> Service {
    let c = es.name_any();
    let sel = node_set_selector(&c, &ns.name);
    Service {
        metadata: meta(es, names::headless_service(&c, &ns.name), sel.clone()),
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()),
            publish_not_ready_addresses: Some(true),
            selector: Some(sel),
            ports: Some(vec![port("http", HTTP_PORT), port("transport", TRANSPORT_PORT)]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub fn config_map(es: &Elasticsearch, ns: &NodeSet, cfg: &RenderedConfig) -> ConfigMap {
    let c = es.name_any();
    ConfigMap {
        metadata: meta(es, names::config_map(&c, &ns.name), node_set_selector(&c, &ns.name)),
        data: Some(BTreeMap::from([("elasticsearch.yml".to_string(), cfg.yaml.clone())])),
        ..Default::default()
    }
}

fn env_field(name: &str, path: &str) -> EnvVar {
    EnvVar {
        name: name.into(),
        value_from: Some(EnvVarSource { field_ref: Some(ObjectFieldSelector { field_path: path.into(), ..Default::default() }), ..Default::default() }),
        ..Default::default()
    }
}

fn default_pvc() -> PersistentVolumeClaim {
    PersistentVolumeClaim {
        metadata: ObjectMeta { name: Some(DATA_VOLUME.into()), ..Default::default() },
        spec: Some(PersistentVolumeClaimSpec {
            access_modes: Some(vec!["ReadWriteOnce".into()]),
            resources: Some(VolumeResourceRequirements { requests: Some(BTreeMap::from([("storage".into(), Quantity(DEFAULT_STORAGE.into()))])), ..Default::default() }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build the StatefulSet for a nodeSet. `replicas` comes from the scale
/// planner (not necessarily `ns.count`).
pub fn statefulset(es: &Elasticsearch, ns: &NodeSet, cfg: &RenderedConfig, replicas: i32, default_repo: &str) -> Result<StatefulSet, String> {
    let c = es.name_any();
    let roles = settings::node_set_roles(ns);
    let selector = node_set_selector(&c, &ns.name);

    let mut tpl: PodTemplateSpec = match &ns.pod_template {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| format!("invalid podTemplate: {e}"))?,
        None => PodTemplateSpec::default(),
    };
    // --- metadata
    let tmeta = tpl.metadata.get_or_insert_with(Default::default);
    let labels = tmeta.labels.get_or_insert_with(Default::default);
    labels.extend(cluster_labels(&c));
    labels.extend(selector.clone());
    labels.insert(LABEL_NODE_MASTER.into(), roles.master.to_string());
    labels.insert(LABEL_NODE_DATA.into(), roles.data.to_string());
    tmeta.annotations.get_or_insert_with(Default::default).insert(ANNOTATION_CONFIG_HASH.into(), cfg.hash.clone());

    // --- pod spec defaults
    let spec = tpl.spec.get_or_insert_with(PodSpec::default);
    spec.termination_grace_period_seconds.get_or_insert(DEFAULT_TERMINATION_GRACE_SECONDS);
    if spec.security_context.is_none() {
        spec.security_context = Some(PodSecurityContext { fs_group: Some(1000), run_as_user: Some(1000), run_as_non_root: Some(true), ..Default::default() });
    }
    if spec.affinity.is_none() {
        spec.affinity = Some(Affinity {
            pod_anti_affinity: Some(PodAntiAffinity {
                preferred_during_scheduling_ignored_during_execution: Some(vec![WeightedPodAffinityTerm {
                    weight: 100,
                    pod_affinity_term: PodAffinityTerm {
                        topology_key: "kubernetes.io/hostname".into(),
                        label_selector: Some(LabelSelector { match_labels: Some(cluster_selector(&c)), ..Default::default() }),
                        ..Default::default()
                    },
                }]),
                ..Default::default()
            }),
            ..Default::default()
        });
    }

    // --- main container
    if !spec.containers.iter().any(|ct| ct.name == CONTAINER_NAME) {
        spec.containers.insert(0, Container { name: CONTAINER_NAME.into(), ..Default::default() });
    }
    let ct = spec.containers.iter_mut().find(|ct| ct.name == CONTAINER_NAME).expect("inserted above");
    if ct.image.as_deref().unwrap_or("").is_empty() {
        ct.image = Some(image(es, default_repo));
    }
    if ct.ports.as_ref().is_none_or(|p| p.is_empty()) {
        ct.ports = Some(vec![
            ContainerPort { name: Some("http".into()), container_port: HTTP_PORT, protocol: Some("TCP".into()), ..Default::default() },
            ContainerPort { name: Some("transport".into()), container_port: TRANSPORT_PORT, protocol: Some("TCP".into()), ..Default::default() },
        ]);
    }
    let env = ct.env.get_or_insert_with(Vec::new);
    let ours = [
        env_field("POD_NAME", "metadata.name"),
        env_field("POD_IP", "status.podIP"),
        env_field("NODE_NAME", "spec.nodeName"),
        env_field("NAMESPACE", "metadata.namespace"),
        EnvVar { name: "ES_PATH_CONF".into(), value: Some(CONFIG_DIR.into()), ..Default::default() },
    ];
    for (i, e) in ours.into_iter().enumerate() {
        if !env.iter().any(|x| x.name == e.name) {
            env.insert(i.min(env.len()), e);
        }
    }
    let mounts = ct.volume_mounts.get_or_insert_with(Vec::new);
    for (vol, path) in [(DATA_VOLUME, DATA_DIR), (LOGS_VOLUME, LOGS_DIR), (CONFIG_VOLUME, CONFIG_DIR)] {
        if !mounts.iter().any(|m| m.mount_path == path) {
            mounts.push(VolumeMount { name: vol.into(), mount_path: path.into(), ..Default::default() });
        }
    }
    if ct.resources.is_none() {
        let mem = BTreeMap::from([("memory".to_string(), Quantity(DEFAULT_MEMORY.into()))]);
        ct.resources = Some(ResourceRequirements { requests: Some(mem.clone()), limits: Some(mem), ..Default::default() });
    }
    if ct.readiness_probe.is_none() {
        ct.readiness_probe = Some(Probe {
            exec: Some(ExecAction { command: Some(vec!["bash".into(), "-c".into(), READINESS_SCRIPT.into()]) }),
            initial_delay_seconds: Some(10),
            period_seconds: Some(5),
            timeout_seconds: Some(5),
            failure_threshold: Some(3),
            success_threshold: Some(1),
            ..Default::default()
        });
    }

    // --- volumes
    let vols = spec.volumes.get_or_insert_with(Vec::new);
    if !vols.iter().any(|v| v.name == CONFIG_VOLUME) {
        vols.push(Volume {
            name: CONFIG_VOLUME.into(),
            config_map: Some(ConfigMapVolumeSource { name: names::config_map(&c, &ns.name), ..Default::default() }),
            ..Default::default()
        });
    }
    if !vols.iter().any(|v| v.name == LOGS_VOLUME) {
        vols.push(Volume { name: LOGS_VOLUME.into(), empty_dir: Some(EmptyDirVolumeSource::default()), ..Default::default() });
    }

    let mut claims: Vec<PersistentVolumeClaim> = match &ns.volume_claim_templates {
        Some(v) if !v.is_empty() => v.iter().map(|x| serde_json::from_value(x.clone()).map_err(|e| format!("invalid volumeClaimTemplate: {e}"))).collect::<Result<_, _>>()?,
        _ => vec![],
    };
    // An emptyDir named elasticsearch-data in the pod template opts out of persistence (like ECK).
    let data_in_pod = spec.volumes.as_ref().is_some_and(|v| v.iter().any(|v| v.name == DATA_VOLUME));
    if !data_in_pod && !claims.iter().any(|p| p.metadata.name.as_deref() == Some(DATA_VOLUME)) {
        claims.push(default_pvc());
    }

    Ok(StatefulSet {
        metadata: meta(es, names::statefulset(&c, &ns.name), selector.clone()),
        spec: Some(StatefulSetSpec {
            replicas: Some(replicas),
            service_name: Some(names::headless_service(&c, &ns.name)),
            selector: LabelSelector { match_labels: Some(selector), ..Default::default() },
            pod_management_policy: Some("Parallel".into()),
            update_strategy: Some(StatefulSetUpdateStrategy { type_: Some("OnDelete".into()), ..Default::default() }),
            persistent_volume_claim_retention_policy: Some(StatefulSetPersistentVolumeClaimRetentionPolicy {
                when_deleted: Some("Delete".into()),
                when_scaled: Some("Delete".into()),
            }),
            template: tpl,
            volume_claim_templates: Some(claims),
            ..Default::default()
        }),
        ..Default::default()
    })
}

/// The PodDisruptionBudget, or `None` when disabled (`podDisruptionBudget: {}`).
/// Default: one voluntary disruption while green, none otherwise.
pub fn pdb(es: &Elasticsearch, health: Option<Health>) -> Result<Option<PodDisruptionBudget>, String> {
    let c = es.name_any();
    let selector = LabelSelector { match_labels: Some(cluster_selector(&c)), ..Default::default() };
    let mut out = PodDisruptionBudget { metadata: meta(es, names::pdb(&c), BTreeMap::new()), ..Default::default() };
    match &es.spec.pod_disruption_budget {
        None => {
            let allowed = if health == Some(Health::Green) { 1 } else { 0 };
            out.spec = Some(PodDisruptionBudgetSpec { max_unavailable: Some(IntOrString::Int(allowed)), selector: Some(selector), ..Default::default() });
        }
        Some(v) => {
            let spec_v = v.get("spec");
            let empty = v.as_object().is_none_or(|o| o.is_empty()) || spec_v.is_some_and(|s| s.as_object().is_some_and(|o| o.is_empty()));
            if empty {
                return Ok(None);
            }
            let mut spec: PodDisruptionBudgetSpec = serde_json::from_value(spec_v.cloned().unwrap_or_default()).map_err(|e| format!("invalid podDisruptionBudget.spec: {e}"))?;
            if spec.selector.is_none() {
                spec.selector = Some(selector);
            }
            if let Some(m) = v.get("metadata") {
                let m: ObjectMeta = serde_json::from_value(m.clone()).map_err(|e| format!("invalid podDisruptionBudget.metadata: {e}"))?;
                out.metadata.labels.get_or_insert_with(Default::default).extend(m.labels.unwrap_or_default());
                out.metadata.annotations = m.annotations;
            }
            out.spec = Some(spec);
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::ElasticsearchSpec;
    use serde_json::json;

    pub(crate) fn es(spec: ElasticsearchSpec) -> Elasticsearch {
        let mut e = Elasticsearch::new("prod", spec);
        e.metadata.namespace = Some("search".into());
        e.metadata.uid = Some("uid-1".into());
        e
    }

    fn cfg() -> RenderedConfig {
        RenderedConfig { yaml: "cluster.name: prod\n".into(), hash: "abc".into(), ignored: vec![] }
    }

    #[test]
    fn statefulset_defaults() {
        let ns = NodeSet { name: "data".into(), count: 3, config: Some(json!({"node.roles": ["data"]})), ..Default::default() };
        let e = es(ElasticsearchSpec { version: "9.1.0".into(), node_sets: vec![ns.clone()], ..Default::default() });
        let sts = statefulset(&e, &ns, &cfg(), 2, DEFAULT_IMAGE_REPOSITORY).unwrap();
        assert_eq!(sts.metadata.name.as_deref(), Some("prod-es-data"));
        assert_eq!(sts.metadata.owner_references.as_ref().unwrap()[0].uid, "uid-1");
        let spec = sts.spec.unwrap();
        assert_eq!(spec.replicas, Some(2));
        assert_eq!(spec.service_name.as_deref(), Some("prod-es-data"));
        assert_eq!(spec.update_strategy.unwrap().type_.as_deref(), Some("OnDelete"));
        assert_eq!(spec.pod_management_policy.as_deref(), Some("Parallel"));
        let tpl = spec.template;
        let labels = tpl.metadata.as_ref().unwrap().labels.as_ref().unwrap();
        assert_eq!(labels[LABEL_NODE_MASTER], "false");
        assert_eq!(labels[LABEL_NODE_DATA], "true");
        assert_eq!(tpl.metadata.as_ref().unwrap().annotations.as_ref().unwrap()[ANNOTATION_CONFIG_HASH], "abc");
        let pod = tpl.spec.unwrap();
        assert_eq!(pod.security_context.unwrap().fs_group, Some(1000));
        assert_eq!(pod.termination_grace_period_seconds, Some(180));
        let ct = &pod.containers[0];
        assert_eq!(ct.image.as_deref(), Some("elasticsearch-rs:9.1.0"));
        let env: Vec<_> = ct.env.as_ref().unwrap().iter().map(|e| e.name.as_str()).collect();
        assert!(env.contains(&"POD_NAME") && env.contains(&"POD_IP"));
        assert!(ct.volume_mounts.as_ref().unwrap().iter().any(|m| m.mount_path == DATA_DIR));
        assert_eq!(spec.volume_claim_templates.unwrap()[0].metadata.name.as_deref(), Some(DATA_VOLUME));
    }

    #[test]
    fn user_pod_template_is_merged_not_replaced() {
        let ns = NodeSet {
            name: "m".into(),
            count: 1,
            pod_template: Some(json!({
                "metadata": {"labels": {"team": "search"}},
                "spec": {
                    "nodeSelector": {"pool": "es"},
                    "containers": [{"name": "elasticsearch", "image": "custom:1", "env": [{"name": "POD_NAME", "value": "override"}],
                                    "resources": {"limits": {"memory": "8Gi"}}},
                                   {"name": "sidecar", "image": "busybox"}]
                }
            })),
            volume_claim_templates: Some(vec![json!({"metadata": {"name": "elasticsearch-data"}, "spec": {"storageClassName": "fast", "resources": {"requests": {"storage": "50Gi"}}}})]),
            ..Default::default()
        };
        let e = es(ElasticsearchSpec { version: "9.1.0".into(), node_sets: vec![ns.clone()], ..Default::default() });
        let sts = statefulset(&e, &ns, &cfg(), 1, "repo").unwrap();
        let spec = sts.spec.unwrap();
        let pod = spec.template.spec.unwrap();
        assert_eq!(pod.containers.len(), 2);
        assert_eq!(pod.containers[0].image.as_deref(), Some("custom:1"));
        assert_eq!(pod.node_selector.unwrap()["pool"], "es");
        let env = pod.containers[0].env.as_ref().unwrap();
        assert_eq!(env.iter().filter(|e| e.name == "POD_NAME").count(), 1);
        assert_eq!(pod.containers[0].resources.as_ref().unwrap().limits.as_ref().unwrap()["memory"].0, "8Gi");
        assert_eq!(spec.template.metadata.unwrap().labels.unwrap()["team"], "search");
        let claims = spec.volume_claim_templates.unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].spec.as_ref().unwrap().storage_class_name.as_deref(), Some("fast"));
    }

    #[test]
    fn services() {
        let e = es(ElasticsearchSpec { version: "9".into(), ..Default::default() });
        let t = transport_service(&e);
        assert_eq!(t.metadata.name.as_deref(), Some("prod-es-transport"));
        let ts = t.spec.unwrap();
        assert_eq!(ts.cluster_ip.as_deref(), Some("None"));
        assert_eq!(ts.publish_not_ready_addresses, Some(true));
        let h = http_service(&e).spec.unwrap();
        assert_eq!(h.ports.unwrap()[0].port, 9200);
        assert_eq!(h.publish_not_ready_addresses, None);
    }

    #[test]
    fn pdb_default_tracks_health_and_can_be_disabled() {
        let mut e = es(ElasticsearchSpec { version: "9".into(), ..Default::default() });
        let g = pdb(&e, Some(Health::Green)).unwrap().unwrap();
        assert_eq!(g.spec.unwrap().max_unavailable, Some(IntOrString::Int(1)));
        let y = pdb(&e, Some(Health::Yellow)).unwrap().unwrap();
        assert_eq!(y.spec.unwrap().max_unavailable, Some(IntOrString::Int(0)));
        e.spec.pod_disruption_budget = Some(json!({}));
        assert!(pdb(&e, Some(Health::Green)).unwrap().is_none());
        e.spec.pod_disruption_budget = Some(json!({"spec": {"minAvailable": 2}}));
        let u = pdb(&e, None).unwrap().unwrap().spec.unwrap();
        assert_eq!(u.min_available, Some(IntOrString::Int(2)));
        assert!(u.selector.is_some());
    }
}
