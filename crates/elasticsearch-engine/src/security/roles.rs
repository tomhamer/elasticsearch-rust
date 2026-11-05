//! Role descriptors (parse / validate / render), the reserved (built-in)
//! roles, and compiled role permissions used for authorization.

use super::privilege::{self, glob, Grant};
use super::Authn;
use crate::error::EsError;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::OnceLock;

fn parse_err(reason: String) -> EsError {
    EsError::new(400, "parse_exception", reason)
}

fn str_list(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_string)).collect(),
        Value::Null => Some(vec![]),
        _ => None,
    }
}

/// Parse a role descriptor body into ES's display form.
/// `api_key` = descriptor inside an API key's `role_descriptors`.
pub fn parse_role(name: &str, body: &Value, api_key: bool) -> Result<Value, EsError> {
    let obj = match body {
        Value::Object(o) => o,
        Value::Null => return Ok(normalize_empty()),
        _ => return Err(parse_err(format!("failed to parse role [{name}]. expected an object but found [{}] instead", json_token(body)))),
    };
    let mut cluster: Vec<String> = Vec::new();
    let mut indices: Vec<Value> = Vec::new();
    let mut remote_indices: Vec<Value> = Vec::new();
    let mut apps: Vec<Value> = Vec::new();
    let mut run_as: Vec<String> = Vec::new();
    let mut metadata = Value::Object(Map::new());
    let mut global: Option<Value> = None;
    let mut description: Option<String> = None;
    let mut remote_cluster: Option<Value> = None;
    let mut restriction: Option<Value> = None;
    for (k, v) in obj {
        match k.as_str() {
            "cluster" => {
                cluster = str_list(v).ok_or_else(|| parse_err(format!("failed to parse role [{name}]. expected field [cluster] value to be an array of strings")))?;
            }
            "indices" | "index" => {
                let arr = v.as_array().ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{name}]. expected field [{k}] value to be an array, but found [{}] instead", json_token(v))))?;
                for e in arr {
                    indices.push(parse_index_group(name, e, false)?);
                }
            }
            "remote_indices" => {
                if let Some(arr) = v.as_array() {
                    for e in arr {
                        remote_indices.push(parse_index_group(name, e, true)?);
                    }
                }
            }
            "applications" | "application" => {
                let arr = v.as_array().ok_or_else(|| parse_err(format!("failed to parse application privileges for role [{name}]. expected field [{k}] value to be an array")))?;
                for e in arr {
                    let o = e.as_object().ok_or_else(|| parse_err(format!("failed to parse application privileges for role [{name}]. expected an object")))?;
                    let app = o.get("application").and_then(Value::as_str).ok_or_else(|| parse_err(format!("failed to parse application privileges for role [{name}]. missing required [application] field")))?;
                    let privs = str_list(o.get("privileges").unwrap_or(&Value::Null)).unwrap_or_default();
                    let res = str_list(o.get("resources").unwrap_or(&Value::Null)).unwrap_or_default();
                    if privs.is_empty() {
                        return Err(parse_err(format!("failed to parse application privileges for role [{name}]. missing required [privileges] field")));
                    }
                    if res.is_empty() {
                        return Err(parse_err(format!("failed to parse application privileges for role [{name}]. missing required [resources] field")));
                    }
                    apps.push(json!({"application": app, "privileges": privs, "resources": res}));
                }
            }
            "run_as" => run_as = str_list(v).unwrap_or_default(),
            "metadata" => {
                if !v.is_object() && !v.is_null() {
                    return Err(parse_err(format!("expected field [metadata] to be of type object, but found [{}] instead", json_token(v))));
                }
                if v.is_object() {
                    metadata = v.clone();
                }
            }
            "global" => global = Some(v.clone()),
            "description" if !api_key => description = v.as_str().map(str::to_string),
            "remote_cluster" => remote_cluster = Some(v.clone()),
            "restriction" if api_key => restriction = Some(v.clone()),
            "transient_metadata" => {}
            other => return Err(parse_err(format!("failed to parse role [{name}]. unexpected field [{other}]"))),
        }
    }
    // Validate privilege names.
    let mut errors: Vec<String> = Vec::new();
    for c in &cluster {
        if let Err(e) = privilege::resolve_cluster(c) {
            errors.push(e);
        }
    }
    for g in &indices {
        for p in g["privileges"].as_array().into_iter().flatten().filter_map(Value::as_str) {
            if let Err(e) = privilege::resolve_index(p) {
                errors.push(e);
            }
        }
    }
    if !errors.is_empty() {
        let refs: Vec<&str> = errors.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let mut out = Map::new();
    out.insert("cluster".into(), json!(cluster));
    if let Some(g) = global {
        out.insert("global".into(), g);
    }
    out.insert("indices".into(), Value::Array(indices));
    out.insert("applications".into(), Value::Array(apps));
    out.insert("run_as".into(), json!(run_as));
    out.insert("metadata".into(), metadata);
    out.insert("transient_metadata".into(), json!({"enabled": true}));
    if !remote_indices.is_empty() {
        out.insert("remote_indices".into(), Value::Array(remote_indices));
    }
    if let Some(rc) = remote_cluster {
        out.insert("remote_cluster".into(), rc);
    }
    if let Some(r) = restriction {
        out.insert("restriction".into(), r);
    }
    if let Some(d) = description {
        out.insert("description".into(), Value::String(d));
    }
    Ok(Value::Object(out))
}

fn normalize_empty() -> Value {
    json!({"cluster": [], "indices": [], "applications": [], "run_as": [], "metadata": {}, "transient_metadata": {"enabled": true}})
}

fn json_token(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
    }
}

fn parse_index_group(role: &str, e: &Value, remote: bool) -> Result<Value, EsError> {
    let o = e.as_object().ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{role}]. expected field [indices] value to be an array of objects, but found an array element of type [{}]", json_token(e))))?;
    let mut names: Option<Vec<String>> = None;
    let mut privs: Option<Vec<String>> = None;
    let mut fls: Option<Value> = None;
    let mut query: Option<Value> = None;
    let mut allow_restricted = false;
    let mut clusters: Option<Vec<String>> = None;
    for (k, v) in o {
        match k.as_str() {
            "names" => {
                let n = str_list(v).ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{role}]. expected field [names] value to be a string or an array of strings, but found [{}] instead", json_token(v))))?;
                if n.is_empty() {
                    return Err(parse_err(format!("failed to parse indices privileges for role [{role}]. [names] cannot be an empty array")));
                }
                names = Some(n);
            }
            "privileges" => privs = Some(str_list(v).ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{role}]. expected field [privileges] value to be an array of strings")))?),
            "field_security" => {
                if !v.is_null() {
                    let bad = |m: &str| parse_err(format!("failed to parse indices privileges for role [{role}]. {m}"));
                    let obj = v.as_object().ok_or_else(|| bad(&format!("expected field [field_security] to be an object, but found [{}] instead", json_token(v))))?;
                    for (fk, fv) in obj {
                        if !matches!(fk.as_str(), "grant" | "except") {
                            return Err(bad(&format!("unexpected field [{fk}] in [field_security]")));
                        }
                        if !fv.is_null() && str_list(fv).is_none() {
                            return Err(bad(&format!("expected field [{fk}] value to be an array of strings")));
                        }
                    }
                    fls = Some(v.clone());
                }
            }
            "query" => {
                if !v.is_null() {
                    query = Some(match v {
                        Value::String(s) => Value::String(s.clone()),
                        other => Value::String(other.to_string()),
                    });
                }
            }
            "allow_restricted_indices" => allow_restricted = v.as_bool().unwrap_or(false),
            "clusters" if remote => clusters = str_list(v),
            other => return Err(parse_err(format!("failed to parse indices privileges for role [{role}]. unexpected field [{other}]"))),
        }
    }
    let names = names.ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{role}]. missing required [names] field")))?;
    let privs = privs.ok_or_else(|| parse_err(format!("failed to parse indices privileges for role [{role}]. missing required [privileges] field")))?;
    let mut m = Map::new();
    m.insert("names".into(), json!(names));
    m.insert("privileges".into(), json!(privs));
    if let Some(f) = fls {
        m.insert("field_security".into(), f);
    }
    if let Some(q) = query {
        m.insert("query".into(), q);
    }
    m.insert("allow_restricted_indices".into(), Value::Bool(allow_restricted));
    if let Some(c) = clusters {
        m.insert("clusters".into(), json!(c));
    }
    Ok(Value::Object(m))
}

// ---------------------------------------------------------------------------
// Reserved roles

fn reserved_md() -> Value {
    json!({"_reserved": true})
}

fn deprecated_md(reason: &str) -> Value {
    json!({"_reserved": true, "_deprecated": true, "_deprecated_reason": reason})
}

fn idx(names: &[&str], privs: &[&str], restricted: bool) -> Value {
    json!({"names": names, "privileges": privs, "allow_restricted_indices": restricted})
}

fn app(app: &str, privs: &[&str], res: &[&str]) -> Value {
    json!({"application": app, "privileges": privs, "resources": res})
}

fn role(cluster: &[&str], indices: Vec<Value>, apps: Vec<Value>, run_as: &[&str], md: Value, description: &str) -> Value {
    let mut v = json!({
        "cluster": cluster, "indices": indices, "applications": apps, "run_as": run_as,
        "metadata": md, "transient_metadata": {"enabled": true}
    });
    if !description.is_empty() {
        v["description"] = Value::String(description.into());
    }
    v
}

/// ES 9.5.5 `KibanaOwnedReservedRoleDescriptors.kibanaSystem` cluster
/// privileges, in ES order (action-name constants resolved).
const KIBANA_SYSTEM_CLUSTER: &[&str] = &[
    "monitor",
    "manage_index_templates",
    "manage_inference",
    "cluster:admin/xpack/monitoring/bulk", // MonitoringBulkAction.NAME
    "manage_saml",
    "manage_token",
    "manage_oidc",
    "manage_enrich",
    "manage_pipeline",
    "manage_ilm",
    "monitor_connector",
    "manage_transform",
    "cluster:admin/xpack/security/api_key/invalidate", // InvalidateApiKeyAction.NAME
    "grant_api_key",
    "clone_api_key",
    "manage_own_api_key",
    "cluster:admin/xpack/security/privilege/builtin/get", // GetBuiltinPrivilegesAction.NAME
    "delegate_pki",
    "cluster:admin/xpack/security/profile/get",            // GetProfilesAction.NAME
    "cluster:admin/xpack/security/profile/activate",       // ActivateProfileAction.NAME
    "cluster:admin/xpack/security/profile/suggest",        // SuggestProfilesAction.NAME
    "cluster:admin/xpack/security/profile/has_privileges", // ProfileHasPrivilegesAction.NAME
    "write_fleet_secrets",
    "manage_ml",
    "cluster:admin/analyze",
    "cluster:admin/script/put",
    "cluster:admin/script/get",
    "cluster:admin/project_routing/delete",
    "monitor_text_structure",
    "cancel_task",
    "cluster:admin/streams/logs/toggle",
];

// ES action names used as index privileges by `kibanaSystem`.
const ROLLOVER: &str = "indices:admin/rollover"; // RolloverAction.NAME
const ALIASES: &str = "indices:admin/aliases"; // TransportIndicesAliasesAction.NAME
const PUT_MAPPING: &str = "indices:admin/mapping/put"; // TransportPutMappingAction.TYPE
const AUTO_PUT_MAPPING: &str = "indices:admin/mapping/auto_put"; // TransportAutoPutMappingAction.TYPE
const UPDATE_SETTINGS: &str = "indices:admin/settings/update"; // TransportUpdateSettingsAction.TYPE
const DELETE_INDEX: &str = "indices:admin/delete"; // TransportDeleteIndexAction.TYPE

/// ES 9.5.5 `kibanaSystem` index privileges, in ES order.
fn kibana_system_indices() -> Vec<Value> {
    let rw_ro = ["read", "read_cross_cluster"];
    let mut v = vec![
        idx(&[".kibana*", ".reindexed-v8-kibana*", ".reporting-*", ".chat-*", ".workflows-*"], &["all"], true),
        idx(&[".monitoring-*"], &rw_ro, false),
        idx(&[".management-beats"], &["create_index", "read", "write"], false),
        idx(&[".ml-anomalies*", ".ml-stats-*"], &["read"], false),
        idx(&[".ml-annotations*", ".ml-notifications*"], &["read", "write"], false),
        idx(&[".reindexed-v8-ml-annotations*", ".reindexed-v8-ml-notifications*"], &["read", "write"], false),
        idx(&[".apm-agent-configuration"], &["all"], true),
        idx(&[".apm-custom-link"], &["all"], true),
        idx(&[".apm-source-map"], &["all"], true),
        idx(&["apm-*"], &rw_ro, false),
        idx(&["logs-apm.*"], &rw_ro, false),
        idx(&["metrics-apm.*"], &rw_ro, false),
        idx(&["traces-apm.*"], &rw_ro, false),
        idx(&["traces-apm-*"], &rw_ro, false),
        idx(&["metrics-logstash.*"], &["read"], false),
        idx(&["*"], &["view_index_metadata", "monitor"], false),
        idx(&[".logs-endpoint.diagnostic.collection-*"], &["read", "create_index"], false),
        idx(&[".fleet-secrets*"], &["write", "delete", "create_index"], true),
        idx(&[".integration_knowledge*"], &["read", "write", "create_index"], true),
    ];
    for n in [
        ".fleet-actions*", ".fleet-agents*", ".fleet-artifacts*", ".fleet-enrollment-api-keys*", ".fleet-policies*", ".fleet-policies-leader*",
        ".fleet-servers*", ".fleet-fileds*", ".fleet-file-data-*", ".fleet-files-*", ".fleet-filedelivery-data-*", ".fleet-filedelivery-meta-*",
    ] {
        v.push(idx(&[n], &["all"], true));
    }
    v.extend([
        idx(&["logs-elastic_agent.status_change-*"], &["all"], false),
        idx(&["logs-elastic_agent*"], &["read"], false),
        idx(&["metrics-fleet_server*"], &["all"], false),
        idx(&["logs-fleet_server*"], &["read", "delete_index"], false),
        idx(&["fleet-synced-integrations", "fleet-synced-integrations-ccr*"], &["create_index", "manage", "read", "write"], false),
        // ALERTS_LEGACY_INDEX, ALERTS_LEGACY_INDEX_REINDEXED_V8
        idx(&[".siem-signals*", ".reindexed-v8-siem-signals*"], &["all"], false),
        // LISTS_INDEX, LISTS_ITEMS_INDEX, LISTS_INDEX_REINDEXED_V8, LISTS_ITEMS_INDEX_REINDEXED_V8
        idx(&[".lists-*", ".items-*", ".reindexed-v8-lists-*", ".reindexed-v8-items-*"], &["all"], false),
        // ALERTS_BACKING_INDEX, ALERTS_BACKING_INDEX_REINDEXED
        idx(&[".internal.alerts*", ".reindexed-v8-internal.alerts*"], &["all"], false),
        // ALERTING_V2_ALERT_VIEWS, ALERTING_V2_RULE_VIEWS
        idx(&["$.alert*", "$.rule*"], &["indices:admin/esql/view/put"], false),
        // ALERTS_INDEX_ALIAS
        idx(&[".alerts*"], &["all"], false),
        // CASES_ANALYTICS_INDEXES, CASES_ANALYTICS_ALIASES
        idx(&[".internal.cases*"], &["all"], false),
        idx(&[".cases*"], &["all"], false),
        // ALERTING_V2_ALERT_INDEX_ALIAS, ALERTING_V2_RULE_INDEX_ALIAS
        idx(&[".alert*"], &["all"], false),
        idx(&[".rule*"], &["all"], false),
        // ADHOC_ALERTS_BACKING_INDEX, ADHOC_ALERTS_INDEX_ALIAS
        idx(
            &[".internal.adhoc.alerts*", ".adhoc.alerts*"],
            &["create_index", "read", "write", "view_index_metadata", "maintenance", ROLLOVER, ALIASES, PUT_MAPPING, AUTO_PUT_MAPPING, UPDATE_SETTINGS],
            false,
        ),
        // PREVIEW_ALERTS_INDEX_ALIAS
        idx(&[".preview.alerts*"], &["all"], false),
        // PREVIEW_ALERTS_BACKING_INDEX, PREVIEW_ALERTS_BACKING_INDEX_REINDEXED
        idx(&[".internal.preview.alerts*", ".reindexed-v8-internal.preview.alerts*"], &["all"], false),
        idx(&["metrics-endpoint.policy-*"], &["read"], false),
        idx(&["metrics-endpoint.metrics-*"], &["read"], false),
        idx(&["logs-endpoint.events.*"], &["read"], false),
        idx(&[".logs-endpoint.action.responses-*"], &rw_ro, false),
        idx(&[".fleet-actions-results*"], &rw_ro, true),
        idx(&["metrics-endpoint.metadata_current_*"], &rw_ro, false),
        idx(&[".metrics-endpoint.metadata_united_default*"], &rw_ro, false),
        idx(&["metrics-endpoint.policy-*"], &rw_ro, false),
        idx(&["logs-endpoint.events.*"], &rw_ro, false),
        idx(
            &[
                "logs-*",
                "synthetics-*",
                "traces-*",
                "/metrics-.*&~(metrics-endpoint\\.metadata_current_default.*)/",
                ".logs-endpoint.action.responses-*",
                ".logs-endpoint.diagnostic.collection-*",
                ".logs-endpoint.actions-*",
                ".logs-endpoint.heartbeat-*",
                ".logs-osquery_manager.actions-*",
                ".logs-osquery_manager.action.responses-*",
                "logs-osquery_manager.action.responses-*",
                "logs-osquery_manager.result-*",
                "profiling-*",
            ],
            &[UPDATE_SETTINGS, PUT_MAPPING, ROLLOVER, "indices:admin/data_stream/lifecycle/put", "indices:admin/forcemerge*"],
            false,
        ),
        idx(&["logs", "logs.*", "$.logs", "$.logs.*"], &["read", "manage"], false),
        idx(&["logs-*.*"], &["read"], false),
        idx(&["traces-*.otel-*", "metrics-*.otel-*"], &["read"], false),
        idx(&[".endpoint-fleetfiles-*"], &["auto_configure", "read", "write", "delete", "create_index", "manage"], false),
        idx(&[".logs-endpoint.action.responses-*"], &["auto_configure", "read", "write", "create_index"], false),
        idx(&[".logs-endpoint.actions-*"], &["auto_configure", "read", "write", "create_index"], false),
        idx(&[".logs-osquery_manager.action.responses-*"], &["auto_configure", "create_index", "read", "index", "delete"], false),
        idx(&["logs-osquery_manager.action.responses-*"], &["read", "view_index_metadata"], false),
        idx(&[".logs-osquery_manager.actions-*"], &["auto_configure", "create_index", "read", "index", "write", "delete"], false),
        idx(&["logs-osquery_manager.result-*"], &["read", "view_index_metadata"], false),
        idx(&["logs-sentinel_one.*", "logs-crowdstrike.*", "logs-microsoft_defender_endpoint.*", "logs-m365_defender.*"], &["read"], false),
        idx(
            &[
                ".logs-endpoint.diagnostic.collection-*",
                "logs-apm-*",
                "logs-apm.*-*",
                "logs-cloud_security_posture.findings-*",
                "logs-cloud_security_posture.vulnerabilities-*",
                "metrics-apm-*",
                "metrics-apm.*-*",
                "traces-apm-*",
                "traces-apm.*-*",
                "synthetics-http-*",
                "synthetics-icmp-*",
                "synthetics-tcp-*",
                "synthetics-browser-*",
                "synthetics-browser.network-*",
                "synthetics-browser.screenshot-*",
            ],
            &[DELETE_INDEX],
            false,
        ),
        idx(&["metrics-endpoint.metadata*"], &["read", "view_index_metadata"], false),
        idx(
            &["metrics-endpoint.metadata_current_default*", ".metrics-endpoint.metadata_current_default*", ".metrics-endpoint.metadata_united_default*"],
            &["create_index", "delete_index", "read", "index", ALIASES, UPDATE_SETTINGS],
            false,
        ),
        idx(&["logs-ti_*_latest.*"], &["create_index", "delete_index", "read", "index", "delete", "manage", ALIASES, UPDATE_SETTINGS], false),
        idx(&["logs-ti_*.*-*"], &[DELETE_INDEX, "read", "view_index_metadata"], false),
        idx(&["kibana_sample_data_*"], &["create_index", "delete_index", "read", "index", "view_index_metadata", ALIASES, UPDATE_SETTINGS], false),
        idx(&["logs-cloud_security_posture.findings-*", "logs-cloud_security_posture.vulnerabilities-*"], &["read", "view_index_metadata"], false),
        idx(
            &["logs-cloud_security_posture.findings_latest-default*", "logs-cloud_security_posture.vulnerabilities_latest-default*"],
            &["create_index", "read", "index", "delete", DELETE_INDEX, "manage", ALIASES, UPDATE_SETTINGS],
            false,
        ),
        idx(&["logs-cloud_security_posture.scores-default*"], &["create_index", "read", "index", "delete", DELETE_INDEX, ALIASES, UPDATE_SETTINGS, AUTO_PUT_MAPPING], false),
        idx(
            &[
                "logs-wiz.vulnerability-*",
                "logs-wiz.cloud_configuration_finding-*",
                "logs-wiz.cloud_configuration_finding_full_posture-*",
                "logs-google_scc.finding-*",
                "logs-aws.securityhub_findings-*",
                "logs-aws.securityhub_findings_full_posture-*",
                "logs-aws_securityhub.finding-*",
                "logs-aws.inspector-*",
                "logs-aws.config-*",
                "logs-amazon_security_lake.findings-*",
                "logs-qualys_vmdr.asset_host_detection-*",
                "logs-tenable_sc.vulnerability-*",
                "logs-tenable_io.vulnerability-*",
                "logs-rapid7_insightvm.vulnerability-*",
                "logs-rapid7_insightvm.asset_vulnerability-*",
                "logs-carbon_black_cloud.asset_vulnerability_summary-*",
            ],
            &["read", "view_index_metadata"],
            false,
        ),
        idx(
            &[
                "logs-m365_defender.vulnerability-*",
                "logs-microsoft_defender_endpoint.vulnerability-*",
                "logs-microsoft_defender_cloud.assessment-*",
                "logs-prisma_cloud.misconfiguration-*",
                "logs-prisma_cloud.vulnerability-*",
                "logs-sentinel_one.application_risk-*",
            ],
            &["read", "view_index_metadata", DELETE_INDEX],
            false,
        ),
        idx(
            &[
                "logs-extrahop.investigation-*",
                "logs-qualys_gav.asset-*",
                "logs-sentinel_one.application-*",
                "logs-sentinel_one.threat_event-*",
                "logs-sentinel_one.unified_alert-*",
                "logs-island_browser.user-*",
                "logs-island_browser.device-*",
                "logs-cyera.classification-*",
                "logs-cyera.issue-*",
                "logs-cyera.datastore-*",
                "logs-ironscales.incident-*",
                "logs-axonius.adapter-*",
                "logs-axonius.alert_finding-*",
                "logs-axonius.application-*",
                "logs-axonius.compute-*",
                "logs-axonius.exposure-*",
                "logs-axonius.gateway-*",
                "logs-axonius.identity-*",
                "logs-axonius.incident-*",
                "logs-axonius.network-*",
                "logs-axonius.storage-*",
                "logs-axonius.ticket-*",
                "logs-axonius.user-*",
                "logs-jupiter_one.risks_and_alerts-*",
                "logs-ping_directory.user-*",
                "logs-xm_cyber.device-*",
                "logs-xm_cyber.product-*",
                "logs-xm_cyber.vulnerability_instance-*",
            ],
            &["manage", "create_index", "read", "index", "write", "delete", DELETE_INDEX, ALIASES, AUTO_PUT_MAPPING],
            false,
        ),
        idx(&["security_solution-*.vulnerability_latest", "security_solution-*.misconfiguration_latest"], &["manage", "read", ALIASES, UPDATE_SETTINGS], false),
        idx(
            &["security_solution-*.vulnerability_latest-*", "security_solution-*.misconfiguration_latest-*"],
            &["create_index", "index", "manage", "read", "delete", ALIASES, UPDATE_SETTINGS],
            false,
        ),
        idx(&["risk-score.risk-*"], &["all"], false),
        idx(&[".entity_analytics.*"], &["all"], false),
        idx(&[".asset-criticality.asset-criticality-*"], &["create_index", "manage", "read", "write"], false),
        idx(&[".entities.*"], &["auto_configure", "create_index", "read", "write"], false),
        idx(&[".entities.*history*"], &["create_index", "manage", "read", "write"], false),
        idx(&[".entities.*updates*"], &["create_index", "manage", "read", "write"], false),
        idx(&[".entities.*reset*"], &["create_index", "manage", "read", "write"], false),
        idx(&["logs-cloud_defend.*", "metrics-cloud_defend.*"], &["read", "view_index_metadata"], false),
        idx(&[".slo-observability.*"], &["all"], false),
        idx(&[".evaluation-*"], &["all"], false),
        idx(&[".logs-endpoint.heartbeat-*"], &["read", "create_index"], false),
        idx(
            &[".edr-workflow-insights-*"],
            &["create_index", "auto_configure", "manage", "read", "write", "delete", UPDATE_SETTINGS, PUT_MAPPING, ROLLOVER],
            false,
        ),
        idx(&[".elastic-connectors*"], &["read"], false),
        idx(
            &[".significant_events-*"],
            &["auto_configure", "create_index", "read", "write", ROLLOVER, PUT_MAPPING, AUTO_PUT_MAPPING, UPDATE_SETTINGS, "indices:admin/data_stream/lifecycle/put"],
            false,
        ),
        idx(&["traces-agent_builder.otel-*", "logs-agent_builder.otel-*"], &["auto_configure", "create_doc"], false),
    ]);
    v
}

/// `kibanaSystem` remote indices (`getRemoteIndicesReadPrivileges(..)`: read +
/// read_cross_cluster on every remote cluster), in ES order.
fn kibana_system_remote_indices() -> Vec<Value> {
    let remote = |n: &str, restricted: bool| json!({"names": [n], "privileges": ["read", "read_cross_cluster"], "allow_restricted_indices": restricted, "clusters": ["*"]});
    vec![
        remote(".monitoring-*", false),
        remote("apm-*", false),
        remote("logs-apm.*", false),
        remote("metrics-apm.*", false),
        remote("traces-apm.*", false),
        remote("traces-apm-*", false),
        remote(".logs-endpoint.action.responses-*", false),
        remote(".fleet-actions-results*", true),
        remote("metrics-endpoint.metadata_current_*", false),
        remote(".metrics-endpoint.metadata_united_default*", false),
        remote("metrics-endpoint.policy-*", false),
        remote("logs-endpoint.events.*", false),
    ]
}

/// The `kibana_system` descriptor (ES 9.5.5 `kibanaSystem`), keys in ES
/// `RoleDescriptor.toXContent` order. Its configurable cluster privileges
/// render under `global` grouped by category (`ConfigurableClusterPrivileges
/// .toXContent` writes every category, empty ones included).
fn kibana_system_role() -> Value {
    let mut m = Map::new();
    m.insert("cluster".into(), json!(KIBANA_SYSTEM_CLUSTER));
    m.insert(
        "global".into(),
        json!({
            "application": {"manage": {"applications": ["kibana-*"]}},
            "profile": {"write": {"applications": ["kibana*"]}},
            "role": {},
            "data_source": []
        }),
    );
    m.insert("indices".into(), Value::Array(kibana_system_indices()));
    m.insert("applications".into(), json!([]));
    m.insert("run_as".into(), json!([]));
    m.insert("metadata".into(), reserved_md());
    m.insert("transient_metadata".into(), json!({"enabled": true}));
    m.insert("remote_indices".into(), Value::Array(kibana_system_remote_indices()));
    m.insert("remote_cluster".into(), json!([{"privileges": ["monitor_stats"], "clusters": ["*"]}]));
    m.insert("description".into(), Value::String(
        "Grants access necessary for the Kibana system user to read from and write to the Kibana indices, manage index templates and tokens, and check the availability of the Elasticsearch cluster. It also permits activating, searching, and retrieving user profiles, as well as updating user profile data for the kibana-* namespace. Additionally, this role grants read access to the .monitoring-* indices and read and write access to the .reporting-* indices. Note: This role should not be assigned to users as the granted permissions may change between releases.".into(),
    ));
    Value::Object(m)
}

fn viewer_editor(editor: bool) -> Value {
    let base = if editor { vec!["read", "view_index_metadata", "write", "maintenance"] } else { vec!["read", "view_index_metadata"] };
    let mut indices = vec![idx(&["/~(([.]|ilm-history-).*)/"], &["read", "view_index_metadata"], false)];
    if editor {
        indices.push(idx(&["observability-annotations"], &["read", "view_index_metadata", "write"], false));
        indices.push(idx(&[".slo-observability.*"], &["read", "view_index_metadata", "write", "manage"], false));
    } else {
        indices.push(idx(&[".slo-observability.*"], &["read", "view_index_metadata"], false));
    }
    indices.push(idx(&[".siem-signals-*", ".lists-*", ".items-*"], &base, false));
    indices.push(idx(&[".alerts*", ".preview.alerts*", ".internal.alerts*", ".internal.preview.alerts*"], &base, false));
    indices.push(idx(&["profiling-*", ".profiling-*"], &["read", "view_index_metadata"], false));
    role(
        &[],
        indices,
        vec![app("kibana-.kibana", if editor { &["all"] } else { &["read"] }, &["*"])],
        &[],
        reserved_md(),
        if editor {
            "Grants full access to all features in Kibana (including Solutions) and read-only access to data indices."
        } else {
            "Grants read-only access to all features in Kibana (including Solutions) and to data indices."
        },
    )
}

/// The built-in roles, by name.
pub fn reserved_roles() -> &'static BTreeMap<String, Value> {
    static R: OnceLock<BTreeMap<String, Value>> = OnceLock::new();
    R.get_or_init(|| {
        let mut m = BTreeMap::new();
        let mut add = |n: &str, v: Value| {
            m.insert(n.to_string(), v);
        };
        add("superuser", role(
            &["all"],
            vec![idx(&["*"], &["all"], false), idx(&["*"], &["monitor", "read", "view_index_metadata", "read_cross_cluster"], true)],
            vec![app("*", &["*"], &["*"])],
            &["*"],
            reserved_md(),
            "Grants full access to cluster management and data indices. This role also grants direct read-only access to restricted indices like .security. A user with this role can impersonate any other user in the system, manage security and create roles with unlimited privileges. Take extra care when assigning it to a user.",
        ));
        add("transport_client", role(&["transport_client"], vec![], vec![], &[], reserved_md(), "Grants the privileges required to access the cluster through the Java Transport Client. The Java Transport Client fetches information about the nodes in the cluster using the Node Liveness API and the Cluster State API (when sniffing is enabled). Assign your users this role if they use the Transport Client."));
        let kibana_admin = |md: Value| role(&[], vec![], vec![app("kibana-.kibana", &["all"], &["*"])], &[], md, "Grants access to all features in Kibana.");
        add("kibana_admin", kibana_admin(reserved_md()));
        add("kibana_user", kibana_admin(deprecated_md("Please use the [kibana_admin] role instead")));
        add("monitoring_user", role(
            &["cluster:monitor/main", "cluster:monitor/xpack/info", "cluster:monitor/remote/info"],
            vec![idx(&[".monitoring-*"], &["read", "read_cross_cluster"], false), idx(&["/metrics-(beats|elasticsearch|enterprisesearch|kibana|logstash).*/"], &["read", "read_cross_cluster"], false), idx(&["metricbeat-*"], &["read", "read_cross_cluster"], false)],
            vec![app("kibana-*", &["reserved_monitoring"], &["*"])],
            &[], reserved_md(), "",
        ));
        add("remote_monitoring_agent", role(
            &["manage_index_templates", "manage_ingest_pipelines", "monitor", "cluster:admin/ilm/get", "cluster:admin/ilm/put", "cluster:monitor/xpack/watcher/watch/get", "cluster:admin/xpack/watcher/watch/put", "cluster:admin/xpack/watcher/watch/delete"],
            vec![idx(&[".monitoring-*"], &["all"], false), idx(&["metricbeat-*"], &["index", "create_index", "view_index_metadata", "indices:admin/aliases", "indices:admin/rollover"], false)],
            vec![], &[], reserved_md(), "",
        ));
        add("remote_monitoring_collector", role(&["monitor"], vec![idx(&["*"], &["monitor"], true), idx(&[".kibana*"], &["read"], true)], vec![], &[], reserved_md(), ""));
        add("ingest_admin", role(&["manage_index_templates", "manage_pipeline"], vec![], vec![], &[], reserved_md(), ""));
        add("reporting_user", role(&[], vec![], vec![], &[], deprecated_md("Please use Kibana feature privileges instead"), ""));
        add("kibana_system", kibana_system_role());
        add("logstash_system", role(&["monitor", "cluster:admin/xpack/monitoring/bulk"], vec![], vec![], &[], reserved_md(), ""));
        add("beats_admin", role(&[], vec![idx(&[".management-beats"], &["all"], false)], vec![], &[], reserved_md(), ""));
        add("beats_system", role(&["monitor", "cluster:admin/xpack/monitoring/bulk"], vec![idx(&[".monitoring-beats-*"], &["create_index", "create"], false)], vec![], &[], reserved_md(), ""));
        add("apm_system", role(&["monitor", "cluster:admin/xpack/monitoring/bulk"], vec![idx(&[".monitoring-beats-*"], &["create_index", "create_doc"], false)], vec![], &[], reserved_md(), ""));
        add("apm_user", role(
            &[],
            ["apm-*", "logs-apm.*", "logs-apm-*", "metrics-apm.*", "metrics-apm-*", "traces-apm.*", "traces-apm-*", ".ml-anomalies*", "observability-annotations"].iter().map(|n| idx(&[n], &["read", "view_index_metadata"], false)).collect(),
            vec![app("kibana-*", &["reserved_ml_apm_user"], &["*"])],
            &[], deprecated_md("This role will be removed in a future major release. Please use editor and viewer roles instead"), "",
        ));
        add("inference_admin", role(&["manage_inference"], vec![], vec![], &[], reserved_md(), ""));
        add("inference_user", role(&["monitor_inference"], vec![], vec![], &[], reserved_md(), ""));
        add("machine_learning_user", role(&["monitor_ml"], vec![idx(&[".ml-anomalies*", ".ml-notifications*"], &["view_index_metadata", "read"], false), idx(&[".ml-annotations*"], &["view_index_metadata", "read", "write"], false)], vec![app("kibana-*", &["reserved_ml_user"], &["*"])], &[], reserved_md(), ""));
        add("machine_learning_admin", role(&["manage_ml"], vec![idx(&[".ml-anomalies*", ".ml-notifications*", ".ml-state*", ".ml-meta*", ".ml-stats-*"], &["view_index_metadata", "read"], true), idx(&[".ml-annotations*"], &["view_index_metadata", "read", "write"], false)], vec![app("kibana-*", &["reserved_ml_admin"], &["*"])], &[], reserved_md(), ""));
        add("data_frame_transforms_admin", role(&["manage_data_frame_transforms"], vec![idx(&[".data-frame-notifications*", ".transform-notifications*"], &["view_index_metadata", "read"], false)], vec![app("kibana-*", &["reserved_ml_user"], &["*"])], &[], deprecated_md("Please use the [transform_admin] role instead"), ""));
        add("data_frame_transforms_user", role(&["monitor_data_frame_transforms"], vec![idx(&[".data-frame-notifications*", ".transform-notifications*"], &["view_index_metadata", "read"], false)], vec![app("kibana-*", &["reserved_ml_user"], &["*"])], &[], deprecated_md("Please use the [transform_user] role instead"), ""));
        add("transform_admin", role(&["manage_transform"], vec![idx(&[".data-frame-notifications*", ".transform-notifications*"], &["view_index_metadata", "read"], false)], vec![], &[], reserved_md(), ""));
        add("transform_user", role(&["monitor_transform"], vec![idx(&[".data-frame-notifications*", ".transform-notifications*"], &["view_index_metadata", "read"], false)], vec![], &[], reserved_md(), ""));
        add("watcher_admin", role(&["manage_watcher"], vec![idx(&[".watches", ".triggered_watches", ".watcher-history-*"], &["read"], true)], vec![], &[], reserved_md(), ""));
        add("watcher_user", role(&["monitor_watcher"], vec![idx(&[".watches"], &["read"], true), idx(&[".watcher-history-*"], &["read"], false)], vec![], &[], reserved_md(), ""));
        add("logstash_admin", role(&["manage_logstash_pipelines"], vec![idx(&[".logstash*"], &["create", "delete", "index", "manage", "read"], true)], vec![], &[], reserved_md(), ""));
        add("rollup_user", role(&["monitor_rollup"], vec![], vec![], &[], reserved_md(), ""));
        add("rollup_admin", role(&["manage_rollup"], vec![], vec![], &[], reserved_md(), ""));
        add("snapshot_user", role(&["create_snapshot", "cluster:admin/repository/get"], vec![idx(&["*"], &["view_index_metadata"], true)], vec![], &[], reserved_md(), ""));
        add("enrich_user", role(&["manage_enrich", "manage_ingest_pipelines", "monitor"], vec![idx(&[".enrich-*"], &["manage", "write"], false)], vec![], &[], reserved_md(), ""));
        add("viewer", viewer_editor(false));
        add("editor", viewer_editor(true));
        m
    })
}

// ---------------------------------------------------------------------------
// Compiled permissions

#[derive(Clone, Debug)]
enum NamePat {
    Glob(String),
    /// Lucene-style `/regex/`: `pos` must match (if any), `neg` must not.
    Regex { pos: Option<regex::Regex>, neg: Option<regex::Regex> },
}

fn anchored(src: &str) -> Option<regex::Regex> {
    regex::Regex::new(&format!("^(?:{src})$")).ok()
}

impl NamePat {
    fn new(p: &str) -> NamePat {
        if p.len() >= 2 && p.starts_with('/') && p.ends_with('/') {
            let inner = &p[1..p.len() - 1];
            // Lucene `~(x)` is a complement; `a&~(b)` = a and not b.
            let (pos, neg) = match inner.split_once("&~(") {
                Some((a, b)) => (Some(a), Some(b.strip_suffix(')').unwrap_or(b))),
                None => match inner.strip_prefix("~(").and_then(|r| r.strip_suffix(')')) {
                    Some(r) => (None, Some(r)),
                    None => (Some(inner), None),
                },
            };
            let pos_re = pos.map(anchored);
            let neg_re = neg.map(anchored);
            if pos_re.as_ref().map(|r| r.is_none()).unwrap_or(false) || neg_re.as_ref().map(|r| r.is_none()).unwrap_or(false) {
                return NamePat::Glob(p.to_string());
            }
            NamePat::Regex { pos: pos_re.flatten(), neg: neg_re.flatten() }
        } else {
            NamePat::Glob(p.to_string())
        }
    }

    fn matches(&self, name: &str) -> bool {
        match self {
            NamePat::Glob(g) => glob(g, name),
            NamePat::Regex { pos, neg } => pos.as_ref().map(|r| r.is_match(name)).unwrap_or(true) && !neg.as_ref().map(|r| r.is_match(name)).unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug)]
struct IndexGroup {
    pats: Vec<NamePat>,
    privs: Vec<Grant>,
    restricted: bool,
    /// Document-level security: the role `query` (JSON text, possibly a `{"template": ..}`).
    query: Option<String>,
    /// Field-level security (`None` = every field).
    fields: Option<FieldPermissions>,
}

/// Metadata fields every field-level-security grant includes.
pub const ALWAYS_GRANTED_FIELDS: &[&str] = &["_id", "_index", "_routing", "_type", "_version", "_seq_no", "_primary_term", "_ignored", "_field_names", "_nested_path"];

/// One role group's `field_security`: `grant` patterns minus `except` patterns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldPermissions {
    pub grant: Vec<String>,
    pub except: Vec<String>,
}

impl FieldPermissions {
    /// Parse `{"grant": [..], "except": [..]}`; a missing `grant` grants everything.
    fn from_json(v: &Value) -> Option<FieldPermissions> {
        if v.is_null() {
            return None;
        }
        let grant = match v.get("grant") {
            None | Some(Value::Null) => vec!["*".to_string()],
            Some(g) => str_list(g).unwrap_or_default(),
        };
        let except = v.get("except").and_then(str_list).unwrap_or_default();
        Some(FieldPermissions { grant, except })
    }

    /// Does this group grant every field (no effective restriction)?
    pub fn grants_all(&self) -> bool {
        self.except.is_empty() && self.grant.iter().any(|g| g == "*")
    }

    pub fn allows(&self, field: &str) -> bool {
        ALWAYS_GRANTED_FIELDS.contains(&field) || (self.grant.iter().any(|p| glob(p, field)) && !self.except.iter().any(|p| glob(p, field)))
    }
}

/// Effective field access on one index: a field is visible when, in every
/// layer, some group allows it (layers: a role and an API key's `limited_by`
/// role). No layers = unrestricted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldAccess {
    layers: Vec<Vec<FieldPermissions>>,
}

impl FieldAccess {
    pub fn from_layer(groups: Vec<FieldPermissions>) -> FieldAccess {
        FieldAccess { layers: vec![groups] }
    }

    /// Metadata fields are always visible; any other field must pass every layer.
    pub fn allows(&self, field: &str) -> bool {
        ALWAYS_GRANTED_FIELDS.contains(&field) || self.layers.iter().all(|l| l.iter().any(|g| g.allows(field)))
    }

    /// Both accesses must allow a field.
    pub fn and(mut self, other: FieldAccess) -> FieldAccess {
        self.layers.extend(other.layers);
        self
    }

    /// The `_source` with only the permitted leaf paths. Objects survive when
    /// some child does (or when empty and granted); arrays are transparent.
    pub fn filter_source(&self, src: &Value) -> Value {
        fn walk(a: &FieldAccess, v: &Value, prefix: &str) -> Option<Value> {
            match v {
                Value::Object(m) => {
                    let mut out = Map::new();
                    for (k, child) in m {
                        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                        if let Some(c) = walk(a, child, &path) {
                            out.insert(k.clone(), c);
                        }
                    }
                    if !out.is_empty() || (m.is_empty() && !prefix.is_empty() && a.allows(prefix)) {
                        Some(Value::Object(out))
                    } else {
                        None
                    }
                }
                Value::Array(items) => {
                    let kept: Vec<Value> = items.iter().filter_map(|i| walk(a, i, prefix)).collect();
                    if !kept.is_empty() || (items.is_empty() && a.allows(prefix)) {
                        Some(Value::Array(kept))
                    } else {
                        None
                    }
                }
                leaf => a.allows(prefix).then(|| leaf.clone()),
            }
        }
        walk(self, src, "").unwrap_or_else(|| Value::Object(Map::new()))
    }
}

/// What a role/API key may read of one index.
#[derive(Clone, Debug, Default)]
pub struct IndexAccess {
    /// Document-level security filter (query DSL) every read must AND in.
    pub dls: Option<Value>,
    /// Field-level security (`None` = every field).
    pub fields: Option<FieldAccess>,
}

impl IndexAccess {
    pub fn is_restricted(&self) -> bool {
        self.dls.is_some() || self.fields.is_some()
    }

    /// Both accesses must hold: filters AND together, field layers stack.
    pub fn and(self, other: IndexAccess) -> IndexAccess {
        let dls = match (self.dls, other.dls) {
            (Some(a), Some(b)) => Some(json!({"bool": {"filter": [a, b]}})),
            (a, b) => a.or(b),
        };
        let fields = match (self.fields, other.fields) {
            (Some(a), Some(b)) => Some(a.and(b)),
            (a, b) => a.or(b),
        };
        IndexAccess { dls, fields }
    }
}

#[derive(Clone, Debug)]
pub struct AppGrant {
    pub app: String,
    pub privileges: Vec<String>,
    pub resources: Vec<String>,
}

/// A set of role descriptors compiled for checks.
#[derive(Clone, Debug, Default)]
pub struct Role {
    cluster: Vec<Grant>,
    indices: Vec<IndexGroup>,
    pub apps: Vec<AppGrant>,
    pub run_as: Vec<String>,
    /// `global.application.manage.applications` (conditional app-privilege management).
    pub manage_apps: Vec<String>,
    /// `global.profile.write.applications` (conditional profile-data updates).
    pub write_profile_apps: Vec<String>,
}

/// Restricted (security-owned system) index names.
pub fn is_restricted(name: &str) -> bool {
    name.starts_with(".security") || name.starts_with(".async-search") || name.starts_with(".fleet-secrets")
}

impl Role {
    pub fn compile(descs: &[&Value]) -> Role {
        let mut r = Role::default();
        for d in descs {
            for c in d["cluster"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                if let Ok(g) = privilege::resolve_cluster(c) {
                    r.cluster.push(g);
                }
            }
            for g in d["indices"].as_array().into_iter().flatten() {
                let pats = str_list(&g["names"]).unwrap_or_default().iter().map(|p| NamePat::new(p)).collect();
                let privs = str_list(&g["privileges"]).unwrap_or_default().iter().filter_map(|p| privilege::resolve_index(p).ok()).collect();
                let query = match &g["query"] {
                    Value::Null => None,
                    Value::String(q) => Some(q.clone()),
                    other => Some(other.to_string()),
                };
                let fields = FieldPermissions::from_json(&g["field_security"]);
                r.indices.push(IndexGroup { pats, privs, restricted: g["allow_restricted_indices"].as_bool().unwrap_or(false), query, fields });
            }
            for a in d["applications"].as_array().into_iter().flatten() {
                r.apps.push(AppGrant {
                    app: a["application"].as_str().unwrap_or("").to_string(),
                    privileges: str_list(&a["privileges"]).unwrap_or_default(),
                    resources: str_list(&a["resources"]).unwrap_or_default(),
                });
            }
            r.run_as.extend(str_list(&d["run_as"]).unwrap_or_default());
            r.manage_apps.extend(str_list(&d["global"]["application"]["manage"]["applications"]).unwrap_or_default());
            r.write_profile_apps.extend(str_list(&d["global"]["profile"]["write"]["applications"]).unwrap_or_default());
        }
        r
    }

    pub fn check_cluster(&self, action: &str) -> bool {
        self.cluster.iter().any(|g| g.matches(action))
    }

    /// Some cluster privilege other than the conditional `manage_own_api_key`
    /// grants `action` for every request (ES: an unconditional privilege in
    /// the role's `ClusterPermission`). An explicit action privilege such as
    /// `cluster:admin/xpack/security/api_key/invalidate` grants the action on
    /// any API key, not only the caller's own.
    pub fn check_cluster_unconditional(&self, action: &str) -> bool {
        self.cluster.iter().any(|g| !matches!(g, Grant::Named(d) if d.name == "manage_own_api_key") && g.matches(action))
    }

    /// `ManageApplicationPrivileges` (`global.application.manage`): grants
    /// `cluster:admin/xpack/security/privilege/*` when every application the
    /// request names matches; a request naming none needs `*`.
    pub fn manages_applications(&self, apps: &[String]) -> bool {
        if apps.is_empty() {
            return self.manage_apps.iter().any(|p| p == "*");
        }
        apps.iter().all(|a| self.manage_apps.iter().any(|p| glob(p, a)))
    }

    /// `WriteProfileDataPrivileges` (`global.profile.write`): grants
    /// `cluster:admin/xpack/security/profile/put/data` when every application
    /// (top-level key of the request's `labels` and `data`) matches.
    pub fn writes_profile_data(&self, apps: &[String]) -> bool {
        !self.write_profile_apps.is_empty() && apps.iter().all(|a| self.write_profile_apps.iter().any(|p| glob(p, a)))
    }

    /// Does the role grant the cluster privilege named `name` (or an action pattern) entirely?
    pub fn has_cluster_privilege(&self, name: &str) -> bool {
        match privilege::resolve_cluster(name) {
            Ok(Grant::Named(d)) => {
                if d.include.is_empty() {
                    return true;
                }
                self.cluster.iter().any(|g| match g {
                    // manage_own_api_key is conditional: it never implies another privilege.
                    Grant::Named(o) if o.name == "manage_own_api_key" => o.name == d.name,
                    Grant::Named(o) => std::ptr::eq(*o, d) || o.implies(d),
                    Grant::Pattern(p) => d.include.iter().all(|q| glob(p, q.trim_end_matches('*')) && (p.ends_with('*') || !q.ends_with('*'))),
                })
            }
            Ok(Grant::Pattern(p)) => self.check_cluster(p.trim_end_matches('*')),
            Err(_) => false,
        }
    }

    /// Any index group grants `action` (on some name).
    pub fn check_index_action(&self, action: &str) -> bool {
        self.indices.iter().any(|g| g.privs.iter().any(|p| p.matches(action)))
    }

    pub fn check_index(&self, action: &str, name: &str) -> bool {
        let restricted = is_restricted(name);
        self.indices.iter().any(|g| (g.restricted || !restricted) && g.pats.iter().any(|p| p.matches(name)) && g.privs.iter().any(|p| p.matches(action)))
    }

    /// Does the role grant index privilege `privilege` (named or action) on `name`?
    pub fn has_index_privilege(&self, name: &str, priv_name: &str) -> bool {
        let restricted = is_restricted(name);
        let want = match privilege::resolve_index(priv_name) {
            Ok(g) => g,
            Err(_) => return false,
        };
        let groups: Vec<&IndexGroup> = self
            .indices
            .iter()
            .filter(|g| (g.restricted || !restricted) && (g.pats.iter().any(|p| p.matches(name)) || (name.contains('*') && g.pats.iter().any(|p| matches!(p, NamePat::Glob(gp) if covers_name(gp, name))))))
            .collect();
        let probes: Vec<String> = match &want {
            Grant::Named(d) => d.include.iter().map(|s| s.to_string()).collect(),
            Grant::Pattern(p) => vec![p.clone()],
        };
        if probes.is_empty() {
            return true;
        }
        probes.iter().all(|probe| {
            groups.iter().any(|g| {
                g.privs.iter().any(|gp| match gp {
                    Grant::Named(d) => d.include.iter().any(|inc| covers_glob(inc, probe)),
                    Grant::Pattern(p) => covers_glob(p, probe),
                })
            })
        })
    }

    pub fn can_run_as(&self, user: &str) -> bool {
        self.run_as.iter().any(|p| glob(p, user))
    }

    /// Does any index group carry document- or field-level security?
    pub fn has_dls_fls(&self) -> bool {
        self.indices.iter().any(|g| g.query.is_some() || g.fields.as_ref().is_some_and(|f| !f.grants_all()))
    }

    /// DLS/FLS of the groups that grant `action` on any of `names` (the
    /// concrete index first, then the alias names the request used).
    /// Multiple groups union: unrestricted when any group has no query /
    /// grants all fields. Default-deny: no matching group is an error.
    pub fn index_access(&self, action: &str, names: &[&str], user: &Authn) -> Result<IndexAccess, EsError> {
        let groups: Vec<&IndexGroup> = self
            .indices
            .iter()
            .filter(|g| g.privs.iter().any(|p| p.matches(action)) && names.iter().any(|n| (g.restricted || !is_restricted(n)) && g.pats.iter().any(|p| p.matches(n))))
            .collect();
        if groups.is_empty() {
            return Err(EsError::new(403, "security_exception", format!("no index privilege grants action [{action}] on [{}]", names.join(","))));
        }
        let dls = if groups.iter().any(|g| g.query.is_none()) {
            None
        } else {
            let mut qs = Vec::new();
            for g in &groups {
                let q = render_dls_query(g.query.as_deref().unwrap_or(""), user)?;
                if !qs.contains(&q) {
                    qs.push(q);
                }
            }
            Some(if qs.len() == 1 { qs.remove(0) } else { json!({"bool": {"should": qs, "minimum_should_match": 1}}) })
        };
        let fields = if groups.iter().any(|g| g.fields.as_ref().map(FieldPermissions::grants_all).unwrap_or(true)) {
            None
        } else {
            let mut fp: Vec<FieldPermissions> = Vec::new();
            for g in &groups {
                if let Some(f) = &g.fields {
                    if !fp.contains(f) {
                        fp.push(f.clone());
                    }
                }
            }
            Some(FieldAccess::from_layer(fp))
        };
        Ok(IndexAccess { dls, fields })
    }
}

fn dls_err(reason: String) -> EsError {
    EsError::new(403, "security_exception", reason)
}

/// Parse a role `query` (JSON text) and expand `{"template": ..}` with the
/// `_user` parameters. Any failure denies the request.
fn render_dls_query(text: &str, user: &Authn) -> Result<Value, EsError> {
    let q: Value = serde_json::from_str(text).map_err(|e| dls_err(format!("invalid role query: {e}")))?;
    let Some(t) = q.get("template").filter(|_| q.as_object().map(|o| o.len() == 1).unwrap_or(false)) else {
        return Ok(q);
    };
    let src = match t.get("source") {
        Some(s) => crate::template::source_text(s),
        None => return Err(dls_err("role query templates support only inline [source]".into())),
    };
    let u = user.effective();
    let mut params = t.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
    params["_user"] = json!({
        "username": u.username,
        "roles": u.roles,
        "full_name": u.full_name,
        "email": u.email,
        "metadata": if u.metadata.is_null() { json!({}) } else { u.metadata.clone() },
        "enabled": u.enabled,
    });
    crate::template::render_to_json(&src, &params).map_err(|e| dls_err(format!("unable to render role query template: {}", e.reason)))
}

/// Does glob `p` cover glob `q`?
fn covers_glob(p: &str, q: &str) -> bool {
    match p.strip_suffix('*') {
        Some(prefix) if !prefix.contains('*') => q.starts_with(prefix),
        _ => p == q,
    }
}

fn covers_name(p: &str, q: &str) -> bool {
    covers_glob(p, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_regex() {
        let r = reserved_roles();
        let v = Role::compile(&[&r["viewer"]]);
        assert!(v.check_index("indices:data/read/search", "logs"));
        assert!(!v.check_index("indices:data/read/search", ".kibana"));
        assert!(!v.check_index("indices:data/read/search", "ilm-history-1"));
        let su = Role::compile(&[&r["superuser"]]);
        assert!(su.check_index("indices:data/read/get", ".security-7"));
        assert!(!su.check_index("indices:data/write/index", ".security-7"));
        assert!(su.check_index("indices:data/write/index", "foo"));
    }

    // ---- kibana_system (ES 9.5.5 `kibanaSystem`) -------------------------------

    fn kibana_system() -> Role {
        Role::compile(&[&reserved_roles()["kibana_system"]])
    }

    #[test]
    fn kibana_system_descriptor_shape() {
        let d = &reserved_roles()["kibana_system"];
        assert_eq!(d["cluster"].as_array().unwrap().len(), 31);
        assert_eq!(d["indices"].as_array().unwrap().len(), 98);
        assert_eq!(d["remote_indices"].as_array().unwrap().len(), 12);
        assert_eq!(d["applications"], json!([]));
        assert_eq!(d["global"]["application"], json!({"manage": {"applications": ["kibana-*"]}}));
        assert_eq!(d["global"]["profile"], json!({"write": {"applications": ["kibana*"]}}));
        assert_eq!(d["remote_cluster"], json!([{"privileges": ["monitor_stats"], "clusters": ["*"]}]));
        assert_eq!(d["metadata"], json!({"_reserved": true}));
        let keys: Vec<&str> = d.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["cluster", "global", "indices", "applications", "run_as", "metadata", "transient_metadata", "remote_indices", "remote_cluster", "description"]);
        // Every privilege and field validates and round-trips through the role parser.
        assert_eq!(parse_role("kibana_system", d, false).unwrap(), *d);
    }

    #[test]
    fn kibana_system_manages_kibana_application_privileges() {
        let r = kibana_system();
        // Not an unconditional cluster privilege: `GET _security/privilege` with no application is denied.
        assert!(!r.check_cluster("cluster:admin/xpack/security/privilege/get"));
        assert!(r.manages_applications(&["kibana-.kibana".to_string()]));
        assert!(r.manages_applications(&["kibana-.kibana".to_string(), "kibana-other".to_string()]));
        assert!(!r.manages_applications(&["kibana-.kibana".to_string(), "myapp".to_string()]));
        assert!(!r.manages_applications(&[]));
        assert!(r.writes_profile_data(&["kibana".to_string(), "kibana-x".to_string()]));
        assert!(!r.writes_profile_data(&["other".to_string()]));
        // A role without `global` grants neither.
        let plain = role_of(json!([]));
        assert!(!plain.manages_applications(&["kibana-.kibana".to_string()]));
        assert!(!plain.writes_profile_data(&[]));
    }

    #[test]
    fn kibana_system_invalidates_any_api_key() {
        let r = kibana_system();
        // The explicit action privilege is unconditional (any owner's keys) ...
        assert!(r.check_cluster_unconditional("cluster:admin/xpack/security/api_key/invalidate"));
        assert!(r.check_cluster_unconditional("cluster:admin/xpack/security/api_key/grant"));
        // ... while get/query only come from the conditional manage_own_api_key.
        assert!(r.check_cluster("cluster:admin/xpack/security/api_key/get"));
        assert!(!r.check_cluster_unconditional("cluster:admin/xpack/security/api_key/get"));
        let own = Role::compile(&[&json!({"cluster": ["manage_own_api_key"]})]);
        assert!(own.check_cluster("cluster:admin/xpack/security/api_key/invalidate"));
        assert!(!own.check_cluster_unconditional("cluster:admin/xpack/security/api_key/invalidate"));
        let admin = Role::compile(&[&json!({"cluster": ["manage_api_key"]})]);
        assert!(admin.check_cluster_unconditional("cluster:admin/xpack/security/api_key/invalidate"));
    }

    #[test]
    fn kibana_system_index_privileges_of_955() {
        let r = kibana_system();
        for ix in [".cases", ".cases-attachments", ".cases-activity", ".entity_analytics.watchlists.default", ".internal.cases-x"] {
            assert!(r.check_index("indices:admin/create", ix), "{ix}");
        }
        for ds in [".rule-events", ".workflows-events", ".edr-workflow-insights-default", ".alert-actions", ".kibana-event-log-ds"] {
            assert!(r.check_index("indices:admin/data_stream/create", ds), "{ds}");
        }
        assert!(r.check_index("indices:data/read/search", ".workflows-workflows"));
        assert!(r.check_index("indices:data/read/get", "fleet-synced-integrations"));
        assert!(r.check_index("indices:data/write/bulk[s]", ".fleet-files-x"));
        assert!(r.check_index("indices:data/write/bulk[s]", ".fleet-secrets-x"));
        // `/metrics-.*&~(metrics-endpoint\.metadata_current_default.*)/`
        assert!(r.check_index("indices:admin/mapping/put", "metrics-system.cpu-default"));
        assert!(!r.check_index("indices:admin/mapping/put", "metrics-endpoint.metadata_current_default"));
        assert!(r.check_index("indices:admin/forcemerge", "logs-a-b"));
        assert!(!r.check_index("indices:data/write/bulk[s]", "logs-a-b"));
        assert!(r.check_index("indices:admin/esql/view/put", "$.alerts-x"));
        // No longer granted by ES 9.5.5.
        assert!(!r.apps.iter().any(|a| a.privileges.iter().any(|p| p == "reserved_fleet-setup")));
    }

    #[test]
    fn kibana_system_event_log_data_stream() {
        // Kibana writes to the `.kibana-event-log-ds` data stream; the cluster
        // coordinator authorizes each bulk item on the concrete backing index.
        // ES's IndicesPermission grants a backing index through its parent
        // data stream's name, so the caller must check the parent name when
        // the backing index name itself is not granted.
        let r = kibana_system();
        let backing = ".ds-.kibana-event-log-ds-2026.11.04-000001";
        for a in ["indices:data/write/bulk[s]", "indices:data/write/index:op_type/create", "indices:admin/auto_create"] {
            assert!(r.check_index(a, ".kibana-event-log-ds"), "{a}");
            assert!(!r.check_index(a, backing), "{a} granted on the backing name alone");
        }
    }

    #[test]
    fn parse_validation() {
        let e = parse_role("r", &json!({"cluster": ["foo"]}), false).unwrap_err();
        assert_eq!(e.kind, "action_request_validation_exception");
        let ok = parse_role("r", &json!({"indices": [{"names": "*", "privileges": ["all"]}]}), false).unwrap();
        assert_eq!(ok["indices"][0]["names"], json!(["*"]));
    }

    // ---- DLS / FLS composition ------------------------------------------------

    fn user(name: &str) -> Authn {
        Authn {
            username: name.into(),
            roles: vec!["r1".into()],
            full_name: None,
            email: None,
            metadata: json!({"dept": "eng\"x"}),
            enabled: true,
            realm_name: "native".into(),
            realm_type: "native".into(),
            auth_type: "realm",
            principal: super::super::Principal::User,
            run_as: None,
        }
    }

    fn role_of(indices: Value) -> Role {
        Role::compile(&[&json!({"indices": indices})])
    }

    const SEARCH: &str = "indices:data/read/search";

    #[test]
    fn dls_union_and_unrestricted() {
        let r = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "query": "{\"term\":{\"x\":1}}"},
            {"names": ["a*"], "privileges": ["read"], "query": {"term": {"y": 2}}},
        ]));
        let f = r.index_access(SEARCH, &["a"], &user("u")).unwrap().dls.unwrap();
        assert_eq!(f, json!({"bool": {"should": [{"term": {"x": 1}}, {"term": {"y": 2}}], "minimum_should_match": 1}}));
        // single granting group: its query as is
        let one = r.index_access(SEARCH, &["abc"], &user("u")).unwrap().dls.unwrap();
        assert_eq!(one, json!({"term": {"y": 2}}));
        // a granting group without a query makes the index unrestricted
        let r2 = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "query": {"term": {"x": 1}}},
            {"names": ["a"], "privileges": ["read"]},
        ]));
        assert!(r2.index_access(SEARCH, &["a"], &user("u")).unwrap().dls.is_none());
        assert!(!r2.has_dls_fls() || r2.has_dls_fls());
        // a group that does not grant the action does not widen access
        let r3 = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "query": {"term": {"x": 1}}},
            {"names": ["a"], "privileges": ["view_index_metadata"]},
        ]));
        assert!(r3.index_access(SEARCH, &["a"], &user("u")).unwrap().dls.is_some());
        assert!(r3.index_access("indices:admin/mappings/get", &["a"], &user("u")).unwrap().dls.is_none());
    }

    #[test]
    fn no_granting_group_is_denied() {
        let r = role_of(json!([{"names": ["a"], "privileges": ["read"]}]));
        assert!(r.index_access(SEARCH, &["b"], &user("u")).is_err());
        assert!(r.index_access("indices:data/write/index", &["a"], &user("u")).is_err());
    }

    #[test]
    fn fls_grant_except_and_union() {
        let r = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["*"], "except": ["secret*", "obj.private"]}},
        ]));
        let fa = r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        assert!(fa.allows("name") && fa.allows("obj.public"));
        assert!(!fa.allows("secret") && !fa.allows("secret_2") && !fa.allows("obj.private"));
        for m in ALWAYS_GRANTED_FIELDS {
            assert!(fa.allows(m));
        }
        // union of two groups
        let r = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["name"]}},
            {"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["age"]}},
        ]));
        let fa = r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        assert!(fa.allows("name") && fa.allows("age") && !fa.allows("ssn"));
        // an unrestricted group (no field_security) lifts the restriction
        let r = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["name"]}},
            {"names": ["a"], "privileges": ["read"]},
        ]));
        assert!(r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.is_none());
        // empty grant grants only metadata
        let r = role_of(json!([{"names": ["a"], "privileges": ["read"], "field_security": {"grant": []}}]));
        let fa = r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        assert!(!fa.allows("name") && fa.allows("_id"));
    }

    #[test]
    fn fls_source_filtering() {
        let r = role_of(json!([
            {"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["*"], "except": ["object.secret"]}},
        ]));
        let fa = r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        let src = json!({"name": "n", "object": {"secret": "s", "public": "p"}, "list": [{"secret": 1}, {"x": 2}]});
        assert_eq!(fa.filter_source(&src), json!({"name": "n", "object": {"public": "p"}, "list": [{"secret": 1}, {"x": 2}]}));
        let fa2 = role_of(json!([{"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["*"], "except": ["list.secret"]}}])).index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        assert_eq!(fa2.filter_source(&src)["list"], json!([{"x": 2}]));
        let r = role_of(json!([{"names": ["a"], "privileges": ["read"], "field_security": {"grant": ["name"]}}]));
        let fa = r.index_access(SEARCH, &["a"], &user("u")).unwrap().fields.unwrap();
        assert_eq!(fa.filter_source(&src), json!({"name": "n"}));
        // a fully hidden object disappears (no empty shell leaks its existence)
        assert_eq!(fa.filter_source(&json!({"object": {"secret": "s"}})), json!({}));
    }

    #[test]
    fn dls_template_substitutes_user() {
        let r = role_of(json!([{
            "names": ["a"], "privileges": ["read"],
            "query": "{\"template\":{\"source\":\"{\\\"bool\\\":{\\\"filter\\\":[{\\\"term\\\":{\\\"owner\\\":\\\"{{_user.username}}\\\"}},{\\\"term\\\":{\\\"dept\\\":\\\"{{_user.metadata.dept}}\\\"}}]}}\"}}"
        }]));
        let f = r.index_access(SEARCH, &["a"], &user("alice")).unwrap().dls.unwrap();
        assert_eq!(f, json!({"bool": {"filter": [{"term": {"owner": "alice"}}, {"term": {"dept": "eng\"x"}}]}}));
    }

    #[test]
    fn broken_dls_query_denies() {
        let r = role_of(json!([{"names": ["a"], "privileges": ["read"], "query": "{not json"}]));
        assert!(r.index_access(SEARCH, &["a"], &user("u")).is_err());
        // a template with a stored-script id cannot be rendered: deny rather than skip the filter
        let r = role_of(json!([{"names": ["a"], "privileges": ["read"], "query": "{\"template\":{\"id\":\"x\"}}"}]));
        assert!(r.index_access(SEARCH, &["a"], &user("u")).is_err());
    }

    #[test]
    fn api_key_intersects_with_limited_by() {
        use std::sync::Arc;
        let key_role = Arc::new(role_of(json!([{"names": ["a"], "privileges": ["read"], "query": {"term": {"k": 1}}, "field_security": {"grant": ["name", "age"]}}])));
        let owner = Arc::new(role_of(json!([{"names": ["a"], "privileges": ["read"], "query": {"term": {"o": 2}}, "field_security": {"grant": ["name", "ssn"]}}])));
        let perm = super::super::Permission { role: key_role.clone(), limit: Some(owner), effective_roles: vec![], missing_roles: vec![], is_superuser: false };
        assert!(perm.has_dls_fls());
        let f = perm.doc_filter("a", &user("u")).unwrap().unwrap();
        assert_eq!(f, json!({"bool": {"filter": [{"term": {"k": 1}}, {"term": {"o": 2}}]}}));
        let fa = perm.field_permissions("a", &user("u")).unwrap().unwrap();
        assert!(fa.allows("name"));
        assert!(!fa.allows("age") && !fa.allows("ssn"));
        // owner unrestricted: the key's own restrictions stand
        let owner = Arc::new(role_of(json!([{"names": ["a"], "privileges": ["all"]}])));
        let perm = super::super::Permission { role: key_role, limit: Some(owner), effective_roles: vec![], missing_roles: vec![], is_superuser: false };
        assert_eq!(perm.doc_filter("a", &user("u")).unwrap().unwrap(), json!({"term": {"k": 1}}));
        assert!(perm.field_permissions("a", &user("u")).unwrap().unwrap().allows("age"));
        // key without restrictions but owner restricted: the owner's stand
        let key = Arc::new(role_of(json!([{"names": ["a"], "privileges": ["read"]}])));
        let owner = Arc::new(role_of(json!([{"names": ["a"], "privileges": ["read"], "query": {"term": {"o": 2}}}])));
        let perm = super::super::Permission { role: key, limit: Some(owner), effective_roles: vec![], missing_roles: vec![], is_superuser: false };
        assert_eq!(perm.doc_filter("a", &user("u")).unwrap().unwrap(), json!({"term": {"o": 2}}));
    }

    #[test]
    fn alias_names_select_groups() {
        // A group granted on the alias name applies when the request used that alias.
        let r = role_of(json!([
            {"names": ["al"], "privileges": ["read"], "query": {"term": {"x": 1}}},
        ]));
        assert!(r.index_access(SEARCH, &["idx", "al"], &user("u")).unwrap().dls.is_some());
        assert!(r.index_access(SEARCH, &["idx"], &user("u")).is_err());
    }
}
