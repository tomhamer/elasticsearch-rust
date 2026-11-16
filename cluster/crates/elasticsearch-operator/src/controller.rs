//! kube-rs controller wiring: observe Kubernetes + Elasticsearch, run the
//! pure planners, apply their output.
//!
//! One reconcile pass:
//! 1. validate the spec (`Invalid` phase on error, nothing touched);
//! 2. provision the `elastic` user and the TLS certificates (Secrets, see
//!    [`crate::security`]); observe Elasticsearch through the HTTP service
//!    over HTTPS as `elastic`; record the cluster UUID annotation the first
//!    time the cluster answers (bootstrap done);
//! 3. apply services; plan topology changes ([`crate::scale`]) and run the
//!    Elasticsearch side of the plan; apply ConfigMaps + StatefulSets with
//!    the planned replica counts; delete removed nodeSets once drained;
//! 4. apply the PodDisruptionBudget;
//! 5. when the topology is stable, plan and run the rolling restart
//!    ([`crate::rolling`]);
//! 6. write status; requeue (10s while changes are in flight, 60s otherwise).

use crate::crd::{Elasticsearch, ElasticsearchStatus};
use crate::es_client::{self, EsClient, EsError, HttpEsClient};
use crate::executor::{self, K8sAction};
use crate::names::{self, *};
use crate::resources;
use crate::rolling::{self, PodView};
use crate::scale::{self, NodeSetScale};
use crate::security::{self, EsConnection, SecretData};
use crate::settings;
use crate::status::{self, StatusInputs};
use futures::StreamExt;
use k8s_openapi::api::apps::v1::StatefulSet;
use k8s_openapi::api::core::v1::{ConfigMap, Pod, Secret, Service};
use k8s_openapi::api::policy::v1::PodDisruptionBudget;
use kube::api::{Api, DeleteParams, ListParams, Patch, PatchParams};
use kube::runtime::controller::{Action, Controller};
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher;
use kube::{Client, Resource, ResourceExt};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("kubernetes: {0}")]
    Kube(#[from] kube::Error),
    #[error("elasticsearch: {0}")]
    Es(#[from] EsError),
    #[error("{0}")]
    Build(String),
}

pub type EsClientFactory = Arc<dyn Fn(&Elasticsearch, &EsConnection) -> Result<Arc<dyn EsClient>, EsError> + Send + Sync>;

pub struct Context {
    pub client: Client,
    pub default_image_repository: String,
    pub es_client: EsClientFactory,
}

/// Default factory: HTTPS to `https://<name>-es-http.<ns>.svc:9200`,
/// trusting the cluster's HTTP CA, as `elastic`.
pub fn https_es_client_factory(timeout: Duration) -> EsClientFactory {
    Arc::new(move |_es: &Elasticsearch, conn: &EsConnection| Ok(Arc::new(HttpEsClient::connect(conn, timeout)?) as Arc<dyn EsClient>))
}

fn secret_data(s: &Secret) -> SecretData {
    SecretData {
        data: s.data.as_ref().map(|d| d.iter().map(|(k, v)| (k.clone(), v.0.clone())).collect()).unwrap_or_default(),
        annotations: s.metadata.annotations.clone().unwrap_or_default(),
    }
}

async fn get_secret(api: &Api<Secret>, name: &str) -> Result<Option<SecretData>, kube::Error> {
    Ok(api.get_opt(name).await?.as_ref().map(secret_data))
}

/// Write `planned` if `Some`, return the Secret's content either way.
async fn put_secret(api: &Api<Secret>, es: &Elasticsearch, name: &str, existing: Option<SecretData>, planned: Option<SecretData>) -> Result<SecretData, Error> {
    match planned {
        Some(d) => {
            info!(cluster = %es.name_any(), secret = %name, "writing secret");
            apply(api, &resources::secret(es, name, &d)).await?;
            Ok(d)
        }
        None => existing.ok_or_else(|| Error::Build(format!("secret {name} missing"))),
    }
}

/// Provision the `elastic` user and the TLS certificates (see
/// [`crate::security`]) and return how to reach the cluster.
async fn reconcile_security(es: &Elasticsearch, ns: &str, secret_api: &Api<Secret>) -> Result<EsConnection, Error> {
    let name = es.name_any();
    let now = chrono::Utc::now();
    let build = Error::Build;

    let user_name = security::elastic_user_secret(&name);
    let existing = get_secret(secret_api, &user_name).await?;
    let planned = security::plan_elastic_user(existing.as_ref());
    let user = put_secret(secret_api, es, &user_name, existing, planned).await?;
    let password = String::from_utf8(user.get(security::ELASTIC_USER_KEY).unwrap_or_default().to_vec()).map_err(|_| build("elastic password is not UTF-8".into()))?;

    // Transport: CA + shared node certificate.
    let ca_name = security::transport_ca_secret(&name);
    let existing = get_secret(secret_api, &ca_name).await?;
    let planned = security::plan_ca(existing.as_ref(), &security::transport_ca_cn(&name), now).map_err(build)?;
    let ca = put_secret(secret_api, es, &ca_name, existing, planned).await?;
    let leaf_name = security::transport_certs_secret(&name);
    let existing = get_secret(secret_api, &leaf_name).await?;
    let planned = security::plan_leaf(existing.as_ref(), &ca, &security::transport_ca_cn(&name), &security::transport_leaf_spec(&name, ns), now).map_err(build)?;
    put_secret(secret_api, es, &leaf_name, existing, planned).await?;

    // HTTP: the user's certificate, or CA + generated certificate.
    let internal_name = security::http_certs_internal_secret(&name);
    let existing_internal = get_secret(secret_api, &internal_name).await?;
    let planned = match security::custom_http_secret(es) {
        Some(user_secret) => {
            let user = get_secret(secret_api, &user_secret).await?.ok_or_else(|| build(format!("spec.http.tls.certificate.secretName: secret {user_secret} not found")))?;
            security::plan_custom_http(&user, existing_internal.as_ref(), &user_secret).map_err(build)?
        }
        None => {
            let ca_name = security::http_ca_secret(&name);
            let existing = get_secret(secret_api, &ca_name).await?;
            let planned = security::plan_ca(existing.as_ref(), &security::http_ca_cn(&name), now).map_err(build)?;
            let ca = put_secret(secret_api, es, &ca_name, existing, planned).await?;
            security::plan_leaf(existing_internal.as_ref(), &ca, &security::http_ca_cn(&name), &security::http_leaf_spec(&name, ns, &es.spec), now).map_err(build)?
        }
    };
    let internal = put_secret(secret_api, es, &internal_name, existing_internal, planned).await?;
    let public_name = security::http_certs_public_secret(&name);
    let existing = get_secret(secret_api, &public_name).await?;
    let planned = security::plan_http_public(&internal, existing.as_ref());
    put_secret(secret_api, es, &public_name, existing, planned).await?;

    Ok(EsConnection {
        url: security::http_url(&name, ns),
        ca_pem: internal.get(security::CA_CRT).map(<[u8]>::to_vec),
        username: security::ELASTIC_USER.into(),
        password,
    })
}

fn apply_params() -> PatchParams {
    PatchParams::apply(FIELD_MANAGER).force()
}

async fn apply<K>(api: &Api<K>, obj: &K) -> Result<K, kube::Error>
where
    K: Resource + Clone + serde::de::DeserializeOwned + Serialize + std::fmt::Debug,
{
    let name = obj.meta().name.clone().unwrap_or_default();
    api.patch(&name, &apply_params(), &Patch::Apply(obj)).await
}

async fn delete_ignore_missing<K>(api: &Api<K>, name: &str) -> Result<(), kube::Error>
where
    K: Resource + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
{
    match api.delete(name, &DeleteParams::background()).await {
        Ok(_) => Ok(()),
        Err(kube::Error::Api(e)) if e.code == 404 => Ok(()),
        Err(e) => Err(e),
    }
}

fn now() -> String {
    k8s_openapi::chrono::Utc::now().to_rfc3339_opts(k8s_openapi::chrono::SecondsFormat::Secs, true)
}

async fn write_status(api: &Api<Elasticsearch>, name: &str, st: &ElasticsearchStatus) -> Result<(), kube::Error> {
    api.patch_status(name, &PatchParams::default(), &Patch::Merge(json!({ "status": st }))).await?;
    Ok(())
}

pub async fn reconcile(es: Arc<Elasticsearch>, ctx: Arc<Context>) -> Result<Action, Error> {
    let name = es.name_any();
    let ns = es.namespace().unwrap_or_else(|| "default".into());
    let client = ctx.client.clone();
    let es_api: Api<Elasticsearch> = Api::namespaced(client.clone(), &ns);
    let sts_api: Api<StatefulSet> = Api::namespaced(client.clone(), &ns);
    let svc_api: Api<Service> = Api::namespaced(client.clone(), &ns);
    let cm_api: Api<ConfigMap> = Api::namespaced(client.clone(), &ns);
    let pdb_api: Api<PodDisruptionBudget> = Api::namespaced(client.clone(), &ns);
    let pod_api: Api<Pod> = Api::namespaced(client.clone(), &ns);
    let secret_api: Api<Secret> = Api::namespaced(client.clone(), &ns);
    let prev_status = es.status.clone();

    if es.meta().deletion_timestamp.is_some() {
        return Ok(Action::await_change()); // children are garbage-collected via owner references
    }

    // 1. validation
    let invalid = settings::validate(&name, &es.spec);
    if !invalid.is_empty() {
        warn!(cluster = %name, errors = ?invalid, "invalid Elasticsearch spec");
        let st = status::compute(prev_status.as_ref(), &StatusInputs { generation: es.metadata.generation, version: es.spec.version.clone(), invalid, ..Default::default() }, &now());
        write_status(&es_api, &name, &st).await?;
        return Ok(Action::requeue(Duration::from_secs(300)));
    }

    // 2. security (elastic user, certificates), then observe Elasticsearch
    //    over HTTPS as `elastic`
    let conn = reconcile_security(&es, &ns, &secret_api).await?;
    let esc = (ctx.es_client)(&es, &conn)?;
    let obs = match es_client::observe(esc.as_ref()).await {
        Ok(o) => o,
        Err(e) => {
            warn!(cluster = %name, error = %e, "observing elasticsearch failed");
            Default::default()
        }
    };
    let mut bootstrapped = es.annotations().contains_key(ANNOTATION_CLUSTER_UUID);
    if !bootstrapped && obs.reachable() {
        let uuid = esc.cluster_uuid().await.ok().flatten();
        if let Some(uuid) = uuid {
            info!(cluster = %name, %uuid, "cluster formed; dropping cluster.initial_master_nodes");
            let patch = json!({ "metadata": { "annotations": { ANNOTATION_CLUSTER_UUID: uuid } } });
            es_api.patch(&name, &PatchParams::default(), &Patch::Merge(&patch)).await?;
            bootstrapped = true;
        }
    }

    // 3. services, topology, configs, statefulsets
    apply(&svc_api, &resources::transport_service(&es)).await?;
    apply(&svc_api, &resources::http_service(&es)).await?;

    let existing: BTreeMap<String, StatefulSet> = sts_api
        .list(&ListParams::default().labels(&label_selector_string(&cluster_selector(&name))))
        .await?
        .items
        .into_iter()
        .map(|s| (s.name_any(), s))
        .collect();
    let replicas_of = |s: &StatefulSet| s.spec.as_ref().and_then(|x| x.replicas).unwrap_or(0);
    let mut sets: Vec<NodeSetScale> = es
        .spec
        .node_sets
        .iter()
        .map(|n| {
            let sts = names::statefulset(&name, &n.name);
            let r = settings::node_set_roles(n);
            // The roles the existing pods may still hold count until they
            // have moved (production review #286).
            let labels = existing.get(&sts).and_then(|s| s.spec.as_ref()).and_then(|x| x.template.metadata.as_ref()).and_then(|m| m.labels.clone());
            let pod_prefix = format!("{sts}-");
            let observed: Vec<&str> = obs.nodes.iter().filter(|nd| nd.name.strip_prefix(&pod_prefix).is_some_and(|o| o.parse::<i32>().is_ok())).map(|nd| nd.roles.as_str()).collect();
            let (master, data) = scale::conservative_roles(r.master, r.data, labels.as_ref(), &observed);
            NodeSetScale { current: existing.get(&sts).map(replicas_of).unwrap_or(0), sts, desired: n.count, master, data, in_spec: true }
        })
        .collect();
    for (sts_name, s) in &existing {
        if !sets.iter().any(|x| &x.sts == sts_name) {
            let labels = s.spec.as_ref().and_then(|x| x.template.metadata.as_ref()).and_then(|m| m.labels.clone()).unwrap_or_default();
            let flag = |k: &str| labels.get(k).is_none_or(|v| v == "true");
            sets.push(NodeSetScale { sts: sts_name.clone(), current: replicas_of(s), desired: 0, master: flag(LABEL_NODE_MASTER), data: flag(LABEL_NODE_DATA), in_spec: false });
        }
    }
    let plan = scale::plan(&sets, &obs, bootstrapped, es.spec.max_surge());
    let k8s_actions = match executor::execute_scale(esc.as_ref(), &plan).await {
        Ok(a) => a,
        Err(e) => {
            warn!(cluster = %name, error = %e, "elasticsearch side of topology change failed; retrying");
            Vec::new()
        }
    };
    let planned = |sts: &str| k8s_actions.iter().find_map(|a| match a {
        K8sAction::SetReplicas { sts: s, replicas } if s == sts => Some(*replicas),
        _ => None,
    });

    let initial_masters = if bootstrapped { None } else { Some(settings::initial_master_nodes(&name, &es.spec)) };
    let mut ignored_settings = Vec::new();
    for n in &es.spec.node_sets {
        let rendered = settings::render(&name, &ns, n, initial_masters.as_deref()).map_err(Error::Build)?;
        ignored_settings.extend(rendered.ignored.iter().map(|k| format!("{}: {k}", n.name)));
        apply(&cm_api, &resources::config_map(&es, n, &rendered)).await?;
        apply(&svc_api, &resources::headless_service(&es, n)).await?;
        let sts_name = names::statefulset(&name, &n.name);
        let current = sets.iter().find(|s| s.sts == sts_name).map(|s| s.current).unwrap_or(0);
        let replicas = planned(&sts_name).unwrap_or(current);
        let sts = resources::statefulset(&es, n, &rendered, replicas, &ctx.default_image_repository).map_err(Error::Build)?;
        apply(&sts_api, &sts).await?;
    }
    if !ignored_settings.is_empty() {
        warn!(cluster = %name, settings = ?ignored_settings, "ignoring operator-managed settings in nodeSet config");
    }
    for a in &k8s_actions {
        match a {
            K8sAction::SetReplicas { sts, replicas } if !es.spec.node_sets.iter().any(|n| &names::statefulset(&name, &n.name) == sts) => {
                sts_api.patch(sts, &PatchParams::default(), &Patch::Merge(json!({ "spec": { "replicas": replicas } }))).await?;
            }
            K8sAction::DeleteStatefulSet { sts } => {
                info!(cluster = %name, %sts, "deleting drained StatefulSet of removed nodeSet");
                delete_ignore_missing(&sts_api, sts).await?;
                delete_ignore_missing(&svc_api, sts).await?;
                delete_ignore_missing(&cm_api, &format!("{sts}-es-config")).await?;
            }
            _ => {}
        }
    }

    // 4. PDB
    match resources::pdb(&es, obs.health.as_ref().map(|h| h.status)).map_err(Error::Build)? {
        Some(p) => {
            apply(&pdb_api, &p).await?;
        }
        None => delete_ignore_missing(&pdb_api, &names::pdb(&name)).await?,
    }

    // 5. rolling restart
    let stses: BTreeMap<String, StatefulSet> = sts_api
        .list(&ListParams::default().labels(&label_selector_string(&cluster_selector(&name))))
        .await?
        .items
        .into_iter()
        .map(|s| (s.name_any(), s))
        .collect();
    let pods = pod_api.list(&ListParams::default().labels(&label_selector_string(&cluster_selector(&name)))).await?.items;
    let mut views = Vec::new();
    let mut revisions_settled = true;
    for p in &pods {
        let labels = p.labels();
        let Some(sts) = labels.get(LABEL_STATEFULSET_NAME).and_then(|s| stses.get(s)) else { continue };
        let st = sts.status.as_ref();
        if st.and_then(|s| s.observed_generation) < sts.metadata.generation {
            revisions_settled = false;
        }
        let update_rev = st.and_then(|s| s.update_revision.clone());
        let ready = p.status.as_ref().and_then(|s| s.conditions.as_ref()).is_some_and(|c| c.iter().any(|c| c.type_ == "Ready" && c.status == "True"));
        views.push(PodView {
            name: p.name_any(),
            sts: sts.name_any(),
            ordinal: names::ordinal_of(&p.name_any()).unwrap_or(0),
            master: labels.get(LABEL_NODE_MASTER).is_some_and(|v| v == "true"),
            data: labels.get(LABEL_NODE_DATA).is_some_and(|v| v == "true"),
            outdated: update_rev.is_some_and(|r| labels.get("controller-revision-hash") != Some(&r)),
            ready,
            terminating: p.metadata.deletion_timestamp.is_some(),
        });
    }
    let expected: i32 = stses.values().map(replicas_of).sum();
    let topology_stable = plan.actions.iter().all(|a| matches!(a, scale::ScaleAction::SetAllocationExcludes(_) | scale::ScaleAction::ClearVotingExclusions)) && plan.waiting.is_empty();
    let mut outdated = views.iter().filter(|v| v.outdated).count();
    let mut waiting = plan.waiting.clone();
    if topology_stable && revisions_settled {
        let rplan = rolling::plan(&views, expected.max(0) as usize, &obs, es.spec.max_unavailable());
        outdated = rplan.remaining;
        waiting.extend(rplan.waiting.iter().cloned());
        match executor::execute_rolling(esc.as_ref(), &rplan).await {
            Ok(acts) => {
                for a in acts {
                    if let K8sAction::DeletePod { pod } = a {
                        info!(cluster = %name, %pod, "restarting pod to apply changes");
                        delete_ignore_missing(&pod_api, &pod).await?;
                    }
                }
            }
            Err(e) => warn!(cluster = %name, error = %e, "rolling restart step failed; retrying"),
        }
    }

    // 6. status
    let desired_pods: i32 = es.spec.node_sets.iter().map(|n| n.count).sum();
    let inputs = StatusInputs {
        generation: es.metadata.generation,
        version: es.spec.version.clone(),
        invalid: vec![],
        ready_pods: views.iter().filter(|v| v.ready && !v.terminating).count() as i32,
        desired_pods,
        scale_pending: !plan.actions.iter().all(|a| matches!(a, scale::ScaleAction::SetAllocationExcludes(_) | scale::ScaleAction::ClearVotingExclusions)) || !plan.waiting.is_empty() || expected != desired_pods,
        migrating_data: plan.migrating_data,
        outdated_pods: outdated,
        waiting,
        es: obs,
    };
    let st = status::compute(prev_status.as_ref(), &inputs, &now());
    if prev_status.as_ref() != Some(&st) {
        write_status(&es_api, &name, &st).await?;
    }
    Ok(if st.phase.as_deref() == Some(status::PHASE_READY) { Action::requeue(Duration::from_secs(60)) } else { Action::requeue(Duration::from_secs(10)) })
}

pub fn error_policy(es: Arc<Elasticsearch>, err: &Error, _ctx: Arc<Context>) -> Action {
    warn!(cluster = %es.name_any(), error = %err, "reconcile failed");
    Action::requeue(Duration::from_secs(15))
}

/// Run the controller until SIGTERM/SIGINT. `namespace = None` watches all namespaces.
pub async fn run(ctx: Context, namespace: Option<String>) {
    let client = ctx.client.clone();
    fn api<K>(c: &Client, ns: &Option<String>) -> Api<K>
    where
        K: Resource<Scope = k8s_openapi::NamespaceResourceScope, DynamicType = ()> + Clone + serde::de::DeserializeOwned + std::fmt::Debug,
    {
        match ns {
            Some(n) => Api::namespaced(c.clone(), n),
            None => Api::all(c.clone()),
        }
    }
    let managed = watcher::Config::default().labels(&format!("{LABEL_MANAGED_BY}={FIELD_MANAGER}"));
    Controller::new(api::<Elasticsearch>(&client, &namespace), watcher::Config::default())
        .owns(api::<StatefulSet>(&client, &namespace), managed.clone())
        .owns(api::<Service>(&client, &namespace), managed.clone())
        .owns(api::<PodDisruptionBudget>(&client, &namespace), managed)
        .watches(api::<Pod>(&client, &namespace), watcher::Config::default().labels(LABEL_CLUSTER_NAME), |pod: Pod| {
            let ns = pod.namespace().unwrap_or_default();
            pod.labels().get(LABEL_CLUSTER_NAME).map(|c| ObjectRef::<Elasticsearch>::new(c).within(&ns))
        })
        .shutdown_on_signal()
        .run(reconcile, error_policy, Arc::new(ctx))
        .for_each(|res| async move {
            match res {
                Ok((obj, _)) => tracing::debug!(cluster = %obj.name, "reconciled"),
                Err(e) => tracing::debug!(error = %e, "reconcile error"),
            }
        })
        .await;
}
