//! Built-in index template registries (ES `IndexTemplateRegistry`
//! subclasses): the x-pack `stack` plugin's `StackTemplateRegistry` and
//! `LegacyStackTemplateRegistry`, and the YAML registries of the `apm-data`
//! (`APMIndexTemplateRegistry`) and `otel-data` (`OTelIndexTemplateRegistry`)
//! plugins.
//!
//! The resource files under `resources/` are ES 8.19's own, copied verbatim
//! (`x-pack/plugin/core/template-resources`, `x-pack/plugin/{apm,otel}-data/
//! src/main/resources`); they are rendered the way `TemplateUtils` does
//! (`${var}` substitution) and installed when the node starts serving
//! ([`ensure`]), replacing any older (lower `version`) or missing copy, as
//! the registries do on cluster-state changes.
//!
//! Lifecycle policies are installed by the ILM service
//! (`ilm/builtin_policies.json`).

mod files;

use crate::node::Node;
use serde_json::Value;
use std::sync::OnceLock;

/// `StackTemplateRegistry.REGISTRY_VERSION`.
const STACK_VERSION: i64 = 22;
/// `LegacyStackTemplateRegistry.REGISTRY_VERSION`.
const LEGACY_STACK_VERSION: i64 = 5;

/// `stack.templates.enabled` (default true). Gates the whole stack registry
/// here (ES still installs its component templates when it is off).
pub const STACK_TEMPLATES_ENABLED: &str = "stack.templates.enabled";
/// `QueryLoggingTemplateRegistry` (stack plugin).
pub const QUERYLOG_REGISTRY_ENABLED: &str = "xpack.stack.querylog.registry.enabled";
const QUERYLOG_VERSION: i64 = 9;
/// `PrometheusPlugin.PROMETHEUS_REGISTRY_ENABLED`.
pub const PROMETHEUS_REGISTRY_ENABLED: &str = "xpack.prometheus.registry.enabled";

macro_rules! stack {
    ($p:literal) => {
        ($p, include_str!(concat!("resources/stack/", $p, ".json")))
    };
}

/// (name, resource) of the stack registry's component templates.
const STACK_COMPONENTS: &[(&str, &str)] = &[
    stack!("data-streams@mappings"),
    stack!("logs@mappings"),
    stack!("ecs@mappings"),
    stack!("logs@settings"),
    stack!("metrics@mappings"),
    stack!("metrics@settings"),
    stack!("metrics@tsdb-settings"),
    stack!("traces@settings"),
    stack!("traces@mappings"),
    stack!("synthetics@mappings"),
    stack!("synthetics@settings"),
    stack!("agentless@mappings"),
    stack!("agentless@settings"),
    stack!("kibana-reporting@settings"),
];

/// (template name, resource) of the stack registry's composable templates.
const STACK_INDEX_TEMPLATES: &[(&str, (&str, &str))] = &[
    ("logs", stack!("logs@template")),
    ("metrics", stack!("metrics@template")),
    ("synthetics", stack!("synthetics@template")),
    ("agentless", stack!("agentless@template")),
    (".kibana-reporting", stack!("kibana-reporting@template")),
];

/// (pipeline name, resource) of the stack registry's pipelines.
const STACK_PIPELINES: &[(&str, (&str, &str))] = &[("logs@json-pipeline", stack!("logs@json-pipeline")), ("logs@default-pipeline", stack!("logs@default-pipeline"))];

/// `LegacyStackTemplateRegistry`: deprecated names for the same resources.
const LEGACY_COMPONENTS: &[(&str, (&str, &str))] = &[
    ("data-streams-mappings", stack!("data-streams@mappings")),
    ("logs-mappings", stack!("logs@mappings")),
    ("ecs@dynamic_templates", stack!("ecs@mappings")),
    ("logs-settings", stack!("logs@settings")),
    ("metrics-mappings", stack!("metrics@mappings")),
    ("metrics-settings", stack!("metrics@settings")),
    ("metrics-tsdb-settings", stack!("metrics@tsdb-settings")),
    ("synthetics-mappings", stack!("synthetics@mappings")),
    ("synthetics-settings", stack!("synthetics@settings")),
];
const LEGACY_PIPELINES: &[(&str, (&str, &str))] = &[("logs@json-message", stack!("logs@json-pipeline")), ("logs-default-pipeline", stack!("logs@default-pipeline"))];

/// One registry's rendered resources.
struct Registry {
    /// Node settings that must all be true (with their defaults).
    enabled_by: &'static [(&'static str, bool)],
    components: Vec<(String, Value)>,
    index_templates: Vec<(String, Value)>,
    pipelines: Vec<(String, Value)>,
}

/// `TemplateUtils.replaceVariables`.
fn substitute(raw: &str, vars: &[(&str, String)]) -> String {
    let mut s = raw.to_string();
    for (k, v) in vars {
        s = s.replace(&format!("${{{k}}}"), v);
    }
    s
}

fn parse_json(name: &str, raw: &str, vars: &[(&str, String)]) -> Value {
    serde_json::from_str(&substitute(raw, vars)).unwrap_or_else(|e| panic!("built-in resource [{name}] does not parse: {e}"))
}

fn parse_yaml(name: &str, raw: &str, vars: &[(&str, String)]) -> Value {
    serde_yaml::from_str::<Value>(&substitute(raw, vars)).unwrap_or_else(|e| panic!("built-in resource [{name}] does not parse: {e}"))
}

fn stack_registry() -> Registry {
    let vars = [("xpack.stack.template.version", STACK_VERSION.to_string()), ("xpack.stack.template.deprecated", "false".into())];
    Registry {
        enabled_by: &[(STACK_TEMPLATES_ENABLED, true)],
        components: STACK_COMPONENTS.iter().map(|(n, r)| (n.to_string(), parse_json(n, r, &vars))).collect(),
        index_templates: STACK_INDEX_TEMPLATES.iter().map(|(n, (f, r))| (n.to_string(), parse_json(f, r, &vars))).collect(),
        pipelines: STACK_PIPELINES.iter().map(|(n, (f, r))| (n.to_string(), parse_json(f, r, &vars))).collect(),
    }
}

fn legacy_stack_registry() -> Registry {
    let vars = [("xpack.stack.template.version", LEGACY_STACK_VERSION.to_string()), ("xpack.stack.template.deprecated", "true".into())];
    Registry {
        enabled_by: &[(STACK_TEMPLATES_ENABLED, true)],
        components: LEGACY_COMPONENTS.iter().map(|(n, (f, r))| (n.to_string(), parse_json(f, r, &vars))).collect(),
        index_templates: Vec::new(),
        pipelines: LEGACY_PIPELINES.iter().map(|(n, (f, r))| (n.to_string(), parse_json(f, r, &vars))).collect(),
    }
}

fn querylog_registry() -> Registry {
    let vars = [
        ("xpack.stack.querylog.template.version", QUERYLOG_VERSION.to_string()),
        ("xpack.stack.querylog.template.deprecated", "false".into()),
        ("xpack.stack.querylog.template.pattern", "logs-elasticsearch.querylog-*".into()),
    ];
    let res = |n: &str, raw: &str| (n.to_string(), parse_json(n, raw, &vars));
    Registry {
        enabled_by: &[(QUERYLOG_REGISTRY_ENABLED, true)],
        components: vec![
            res("logs-elasticsearch.querylog@mappings", include_str!("resources/stack/activitylog/logs-elasticsearch.querylog@mappings.json")),
            res("logs-elasticsearch.querylog@settings", include_str!("resources/stack/activitylog/logs-elasticsearch.querylog@settings.json")),
        ],
        index_templates: vec![res("logs-elasticsearch.querylog@template", include_str!("resources/stack/activitylog/logs-elasticsearch.querylog@template.json"))],
        pipelines: Vec::new(),
    }
}

fn yaml_file(path: &str) -> &'static str {
    files::YAML_FILES.iter().find(|(p, _)| *p == path).map(|(_, r)| *r).unwrap_or_else(|| panic!("missing built-in resource [{path}]"))
}

/// `YamlTemplateRegistry`: the resources listed in `<plugin>/resources.yaml`.
fn yaml_registry(plugin: &str, resources: &str, version_var: &'static str, enabled_by: &'static [(&'static str, bool)]) -> Registry {
    let res: Value = serde_yaml::from_str(resources).expect("resources.yaml");
    let version = res["version"].as_i64().expect("resources.yaml version");
    let vars = [(version_var, version.to_string())];
    let names = |k: &str| -> Vec<String> { res[k].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default() };
    let load = |dir: &str, n: &str| parse_yaml(n, yaml_file(&format!("{plugin}/{dir}/{n}.yaml")), &vars);
    Registry {
        enabled_by,
        components: names("component-templates").into_iter().map(|n| (n.clone(), load("component-templates", &n))).collect(),
        index_templates: names("index-templates").into_iter().map(|n| (n.clone(), load("index-templates", &n))).collect(),
        pipelines: res["ingest-pipelines"]
            .as_array()
            .map(|a| a.iter().filter_map(|e| e.as_object().and_then(|o| o.keys().next().cloned())).map(|n| (n.clone(), load("ingest-pipelines", &n))).collect())
            .unwrap_or_default(),
    }
}

fn registries() -> &'static [Registry] {
    static R: OnceLock<Vec<Registry>> = OnceLock::new();
    R.get_or_init(|| {
        vec![
            legacy_stack_registry(),
            stack_registry(),
            yaml_registry(
                "apm-data",
                include_str!("resources/apm-data/resources.yaml"),
                "xpack.apmdata.template.version",
                &[("xpack.apm_data.enabled", true), ("xpack.apm_data.registry.enabled", true)],
            ),
            yaml_registry(
                "otel-data",
                include_str!("resources/otel-data/resources.yaml"),
                "xpack.oteldata.template.version",
                &[("xpack.otel_data.enabled", true), ("xpack.otel_data.registry.enabled", true)],
            ),
            querylog_registry(),
            yaml_registry("prometheus", include_str!("resources/prometheus/resources.yaml"), "xpack.prometheus.template.version", &[(PROMETHEUS_REGISTRY_ENABLED, true)]),
        ]
    })
}

fn bool_setting(node: &Node, key: &str, default: bool) -> bool {
    let v = crate::ilm::cluster_setting(node, key).or_else(|| {
        node.node_settings.read().get(key).map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    });
    match v.as_deref() {
        Some("true") => true,
        Some("false") => false,
        _ => default,
    }
}

fn version_of(v: &Value) -> Option<i64> {
    v.get("version").and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}

/// Whether `current` must be (re)installed with `new` (missing or older).
fn needs_install(current: Option<&Value>, new: &Value) -> bool {
    match current {
        None => true,
        Some(c) => match (version_of(c), version_of(new)) {
            (Some(a), Some(b)) => a < b,
            (None, Some(_)) => true,
            _ => false,
        },
    }
}

/// Install every enabled registry's missing or outdated resources.
pub fn ensure(node: &Node) -> bool {
    let mut changed = false;
    for r in registries() {
        if !r.enabled_by.iter().all(|(k, d)| bool_setting(node, k, *d)) {
            continue;
        }
        {
            let mut t = node.templates.write();
            // Installed through the template service: tracking dates set.
            let tracked = |v: &Value| {
                let mut v = v.clone();
                let now = crate::date::now_millis();
                if let Some(o) = v.as_object_mut() {
                    o.insert("created_date_millis".into(), now.into());
                    o.insert("modified_date_millis".into(), now.into());
                }
                v
            };
            for (n, v) in &r.components {
                if needs_install(t.component.get(n), v) {
                    t.component.insert(n.clone(), tracked(v));
                    changed = true;
                }
            }
            for (n, v) in &r.index_templates {
                if needs_install(t.index.get(n), v) {
                    t.index.insert(n.clone(), tracked(v));
                    changed = true;
                }
            }
        }
        let ingest = node.ingest();
        let current = ingest.get(None);
        let missing: Vec<(String, Value)> =
            r.pipelines.iter().filter(|(n, v)| needs_install(current.iter().find(|(k, _)| k == n).map(|(_, c)| c), v)).cloned().collect();
        if !missing.is_empty() {
            ingest.put_builtin(missing);
            changed = true;
        }
    }
    // Built-in lifecycle policies: missing or older ones (re)installed.
    changed |= node.ilm.ensure_builtins();
    // `ProfilingIndexTemplateRegistry` (behind `xpack.profiling.templates.enabled`).
    changed |= crate::profiling::ensure_registry(node);
    // `WatcherIndexTemplateRegistry`: the history policy and template.
    changed |= crate::watcher::ensure_registry(node);
    if changed {
        if let Err(e) = node.persist_state() {
            tracing::warn!("persisting built-in templates failed: {e}");
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ProfilingIndexTemplateRegistry`: nothing until
    /// `xpack.profiling.templates.enabled`; then the reconcile installs its
    /// templates and `profiling-60-days` (no profiling call needed), and an
    /// older policy (`_meta.version`) is upgraded.
    #[test]
    fn profiling_registry_follows_its_setting_and_version() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        ensure(&node);
        assert!(node.ilm.policy("profiling-60-days").is_none());
        assert!(!node.templates.read().index.contains_key("profiling-events"));
        node.cluster_settings.write().0.insert(crate::profiling::TEMPLATES_ENABLED.into(), serde_json::json!("true"));
        assert!(ensure(&node));
        let p = node.ilm.policy("profiling-60-days").expect("installed by the registry");
        assert_eq!(p.policy["_meta"]["version"], crate::profiling::INDEX_TEMPLATE_VERSION, "{:?}", p.policy);
        assert_eq!(p.policy["phases"]["delete"]["min_age"], "60d");
        for t in ["profiling-events", "profiling-metrics", "profiling-hosts", "profiling-stacktraces"] {
            assert!(node.templates.read().index.contains_key(t), "{t}");
        }
        // An older install's policy (lower `_meta.version`) is upgraded.
        let mut old = p.policy.clone();
        old["_meta"]["version"] = serde_json::json!(crate::profiling::INDEX_TEMPLATE_VERSION - 1);
        old["phases"]["delete"]["min_age"] = serde_json::json!("30d");
        node.ilm.replace_policy("profiling-60-days", old).unwrap();
        assert!(ensure(&node));
        assert_eq!(node.ilm.policy("profiling-60-days").unwrap().policy["phases"]["delete"]["min_age"], "60d");
        assert!(!ensure(&node));
    }

    #[test]
    fn registries_reconcile_missing_cluster_metadata_without_downgrading() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        assert!(ensure(&node));
        assert!(!ensure(&node));
        {
            let mut templates = node.templates.write();
            templates.index.remove("traces-apm@template");
            templates.component.remove("apm@mappings");
            templates.component.get_mut("logs@mappings").unwrap()["version"] = 999999.into();
        }
        assert!(ensure(&node));
        assert!(node.templates.read().index.contains_key("traces-apm@template"));
        assert!(node.templates.read().component.contains_key("apm@mappings"));
        assert_eq!(node.templates.read().component["logs@mappings"]["version"], 999999);
        assert!(!ensure(&node));
    }

    /// A user template named like a built-in one (`metrics`), without a
    /// `version`, is replaced by the built-in on the next reconcile, as ES's
    /// `IndexTemplateRegistry` does ("upgrading composable index template"
    /// when the current one has no version). Its `metrics-*` pattern goes
    /// with it, so `metrics-a` matches no template (the built-in is
    /// `metrics-*-*`) — ES's answer too. A user template with its own name
    /// and a higher priority wins; one at the built-in's name with a higher
    /// `version` is kept.
    #[test]
    fn user_template_named_like_a_builtin_follows_es_registry_rules() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        ensure(&node);
        let builtin = node.templates.read().index["metrics"].clone();
        assert_eq!(builtin["index_patterns"], serde_json::json!(["metrics-*-*"]));
        let user = serde_json::json!({"index_patterns": ["metrics-*"], "data_stream": {}, "priority": 500, "template": {"settings": {"index.number_of_replicas": "0"}}});
        node.templates.write().index.insert("metrics".into(), user.clone());
        assert!(ensure(&node), "the version-less user template is upgraded");
        assert_eq!(node.templates.read().index["metrics"]["index_patterns"], builtin["index_patterns"]);
        let e = node.create_data_stream("metrics-a").unwrap_err();
        assert_eq!(e.reason, "no matching index template found for data stream [metrics-a]");
        // Its own name and a higher priority: used for `metrics-a`, and for
        // `metrics-x-y` over the built-in.
        node.templates.write().index.insert("my-metrics".into(), user.clone());
        assert!(!ensure(&node));
        node.create_data_stream("metrics-a").unwrap();
        node.create_data_stream("metrics-x-y").unwrap();
        assert_eq!(node.template_for("metrics-x-y").0, node.template_for("metrics-a").0);
        // At the built-in's name with a higher version: kept.
        let mut versioned = user;
        versioned["index_patterns"] = serde_json::json!(["vmetrics-*"]);
        versioned["version"] = serde_json::json!(i64::MAX);
        node.templates.write().index.insert("metrics".into(), versioned);
        assert!(!ensure(&node));
        assert_eq!(node.templates.read().index["metrics"]["index_patterns"], serde_json::json!(["vmetrics-*"]));
    }
}
