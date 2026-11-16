//! Universal Profiling (ES x-pack `profiling` plugin).
//!
//! * Resource management: with `xpack.profiling.templates.enabled` the
//!   plugin installs its component/composable templates and ILM policy
//!   (`ProfilingIndexTemplateRegistry`), the K/V indices behind the
//!   `profiling-*` aliases (`ProfilingIndexManager`) and the events / hosts
//!   / metrics data streams (`ProfilingDataStreamManager`). ES does that on
//!   cluster-state changes; here [`ensure_resources`] runs whenever a
//!   profiling API is called.
//! * `GET /_profiling/status` ([`status`]).
//! * `POST /_profiling/stacktraces|flamegraph|topn/functions`
//!   ([`stacktraces`], [`graph`]).

pub mod costs;
pub mod graph;
pub mod java;
pub mod stacktraces;

use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// `ProfilingIndexTemplateRegistry.INDEX_TEMPLATE_VERSION` (ES 9.5.5; 15:
/// LogsDB mode for `profiling-events-*`).
pub const INDEX_TEMPLATE_VERSION: i64 = 15;
/// `PROFILING_EVENTS_VERSION` (6 with the LogsDB events template: a new
/// backing index, by rollover).
const PROFILING_EVENTS_VERSION: i64 = 6;
/// `xpack.profiling.check_outdated_indices` (default true).
pub const CHECK_OUTDATED_INDICES: &str = "xpack.profiling.check_outdated_indices";
const VERSION_VARIABLE: &str = "xpack.profiling.template.version";
pub const TEMPLATES_ENABLED: &str = "xpack.profiling.templates.enabled";
pub const PROFILING_ENABLED: &str = "xpack.profiling.enabled";

macro_rules! res {
    ($p:literal) => {
        include_str!(concat!("resources/", $p))
    };
}

/// (name, raw resource, index-version variable).
const COMPONENT_TEMPLATES: &[(&str, &str, Option<(&str, i64)>)] = &[
    ("profiling-events", res!("component-template/profiling-events.json"), Some(("events", PROFILING_EVENTS_VERSION))),
    ("profiling-executables", res!("component-template/profiling-executables.json"), Some(("executables", 1))),
    ("profiling-ilm", res!("component-template/profiling-ilm.json"), None),
    ("profiling-hot-tier", res!("component-template/profiling-hot-tier.json"), None),
    ("profiling-metrics", res!("component-template/profiling-metrics.json"), Some(("metrics", 2))),
    ("profiling-hosts", res!("component-template/profiling-hosts.json"), Some(("hosts", 2))),
    ("profiling-stackframes", res!("component-template/profiling-stackframes.json"), Some(("stackframes", 1))),
    ("profiling-stacktraces", res!("component-template/profiling-stacktraces.json"), Some(("stacktraces", 1))),
    ("profiling-symbols", res!("component-template/profiling-symbols.json"), Some(("symbols", 1))),
];

const INDEX_TEMPLATES: &[(&str, &str, Option<(&str, i64)>)] = &[
    ("profiling-events", res!("index-template/profiling-events.json"), None),
    ("profiling-metrics", res!("index-template/profiling-metrics.json"), None),
    ("profiling-hosts", res!("index-template/profiling-hosts.json"), None),
    ("profiling-executables", res!("index-template/profiling-executables.json"), None),
    ("profiling-stackframes", res!("index-template/profiling-stackframes.json"), None),
    ("profiling-stacktraces", res!("index-template/profiling-stacktraces.json"), None),
    ("profiling-returnpads-private", res!("index-template/profiling-returnpads-private.json"), Some(("returnpads.private", 1))),
    ("profiling-sq-executables", res!("index-template/profiling-sq-executables.json"), Some(("sq.executables", 1))),
    ("profiling-sq-leafframes", res!("index-template/profiling-sq-leafframes.json"), Some(("sq.leafframes", 1))),
    ("profiling-symbols-global", res!("index-template/profiling-symbols-global.json"), None),
    ("profiling-symbols-private", res!("index-template/profiling-symbols-private.json"), None),
];

const ILM_POLICY: (&str, &str) = ("profiling-60-days", res!("ilm-policy/profiling-60-days.json"));

/// A K/V or regular index managed by `ProfilingIndexManager`:
/// (alias / name prefix, version, K/V generation).
const MANAGED_INDICES: &[(&str, i64, Option<&str>)] = &[
    ("profiling-returnpads-private", 1, None),
    ("profiling-sq-executables", 1, None),
    ("profiling-sq-leafframes", 1, None),
    ("profiling-symbols-private", 1, None),
    ("profiling-executables", 1, Some("000001")),
    ("profiling-stackframes", 1, Some("000001")),
    ("profiling-stacktraces", 1, Some("000001")),
    ("profiling-symbols-global", 1, Some("000001")),
];

/// `ProfilingIndexManager.OnVersionBump.DELETE_OLD`: the regular indices
/// whose older versions are deleted on a version bump (the rest keep them;
/// K/V indices age out through ILM).
const DELETE_OLD_ON_BUMP: &[&str] = &["profiling-sq-executables", "profiling-sq-leafframes"];

/// One `Migration` (ES `Migration.Builder`): a mapping change or dynamic
/// settings, reaching index template version `target`.
#[derive(Clone, Debug, PartialEq)]
pub enum Migration {
    PutMapping { target: i64, body: Value },
    DynamicSettings { target: i64, settings: Value },
}

impl Migration {
    fn target(&self) -> i64 {
        match self {
            Migration::PutMapping { target, .. } | Migration::DynamicSettings { target, .. } => *target,
        }
    }
}

/// The migrations of a managed index or data stream (ES 9.5.5 defines none:
/// every `ProfilingIndex` / `ProfilingDataStream` is built without a
/// `Migration.Builder`). `Migration.Builder.build` appends the `_meta`
/// version update to a non-empty list.
fn migrations_of(_name: &str) -> Vec<Migration> {
    #[cfg(test)]
    if let Some(m) = TEST_MIGRATIONS.with(|t| t.borrow().get(_name).cloned()) {
        return m;
    }
    Vec::new()
}

#[cfg(test)]
thread_local! {
    static TEST_MIGRATIONS: std::cell::RefCell<std::collections::HashMap<String, Vec<Migration>>> = std::cell::RefCell::new(Default::default());
}

/// `Migration.Builder.build(indexVersion)`.
fn built_migrations(name: &str, index_version: i64) -> Vec<Migration> {
    let mut m = migrations_of(name);
    if let Some(last) = m.last().map(Migration::target) {
        m.push(Migration::PutMapping { target: last, body: json!({"_meta": {"index-template-version": last, "index-version": index_version}}) });
    }
    m
}

const EVENTS_PREFIX: &str = "profiling-events";
pub const ALL_EVENTS: &str = "profiling-events-all";
const SAMPLING_FACTOR: i32 = 5;
const MIN_EXPONENT: i32 = 1;
const MAX_EXPONENT: i32 = 11;

/// `EventsIndex.indexName`.
pub fn events_index_name(exp: i32) -> String {
    if exp == 0 {
        ALL_EVENTS.to_string()
    } else {
        format!("{EVENTS_PREFIX}-{SAMPLING_FACTOR}pow{exp:02}")
    }
}

/// `EventsIndex.getSampleRate`.
pub fn events_sample_rate(exp: i32) -> f64 {
    (1.0f64 / SAMPLING_FACTOR as f64).powi(exp)
}

/// `EventsIndex.getSampledIndex` (exponent 0 = the full index).
pub fn resampled_exponent(target: i64, current: i64, initial_exp: i32) -> i32 {
    if current == 0 {
        return 0;
    }
    let x = ((target * SAMPLING_FACTOR as i64) as f64 / current as f64).ln() / (SAMPLING_FACTOR as f64).ln();
    let exp = initial_exp - java_round(x) as i32 + 1;
    if exp < MIN_EXPONENT {
        return 0;
    }
    exp.min(MAX_EXPONENT)
}

/// `Math.round(double)`.
pub fn java_round(x: f64) -> i64 {
    if x.is_nan() {
        return 0;
    }
    (x + 0.5).floor() as i64
}

/// Data streams managed by `ProfilingDataStreamManager`: (name, version).
fn managed_data_streams() -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = (0..=MAX_EXPONENT).filter(|e| *e == 0 || *e >= MIN_EXPONENT).map(|e| (events_index_name(e), PROFILING_EVENTS_VERSION)).collect();
    out.push(("profiling-metrics".into(), 2));
    out.push(("profiling-hosts".into(), 2));
    out
}

/// Strip `/* */` comments and substitute `${...}` variables
/// (ES `TemplateUtils.loadTemplate`).
fn render(raw: &str, index_var: Option<(&str, i64)>) -> Value {
    let mut s = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(i) = rest.find("/*") {
        s.push_str(&rest[..i]);
        rest = match rest[i + 2..].find("*/") {
            Some(j) => &rest[i + 2 + j + 2..],
            None => "",
        };
    }
    s.push_str(rest);
    let mut s = s.replace(&format!("${{{VERSION_VARIABLE}}}"), &INDEX_TEMPLATE_VERSION.to_string());
    if let Some((name, v)) = index_var {
        s = s.replace(&format!("${{xpack.profiling.index.{name}.version}}"), &v.to_string());
    }
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("profiling resource does not parse: {e}"))
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

pub fn templates_enabled(node: &Node) -> bool {
    bool_setting(node, TEMPLATES_ENABLED, false)
}

pub fn plugin_enabled(node: &Node) -> bool {
    bool_setting(node, PROFILING_ENABLED, true)
}

#[derive(PartialEq, Debug, Clone)]
enum IndexStatus {
    NeedsCreation,
    NeedsVersionBump,
    /// Template version behind with pending migrations (applied to the
    /// write index).
    NeedsMappingsUpdate(String, Vec<Migration>),
    Closed,
    UpToDate,
    Unhealthy,
    /// Created before 8.9.1 (`checkOutdatedIndices`): the managers abort.
    TooOld,
}

/// `IndexStateResolver.getIndexState`: `svc` is the index (or the data
/// stream's write index) named `name` in the manager, `version` its
/// managed version.
fn index_status(node: &Node, svc: Option<std::sync::Arc<crate::index::IndexService>>, name: &str, version: i64) -> IndexStatus {
    let Some(svc) = svc else { return IndexStatus::NeedsCreation };
    if !svc.is_open() {
        return IndexStatus::Closed;
    }
    // A red index waits (pending mapping upgrades included).
    if !svc.failed_shards().is_empty() {
        return IndexStatus::Unhealthy;
    }
    if bool_setting(node, CHECK_OUTDATED_INDICES, true)
        && svc.setting("index.version.created").is_some_and(|v| crate::version::IndexVersion::from_setting(Some(&json!(v))) < crate::version::IndexVersion(8_09_01_99))
    {
        return IndexStatus::TooOld;
    }
    let mapping = svc.meta().mappings.to_json();
    let (index_version, template_version) = match mapping.get("_meta") {
        None => (1, 1),
        Some(m) => {
            let f = |k: &str| m.get(k).and_then(Value::as_i64).filter(|v| *v >= i32::MIN as i64 && *v <= i32::MAX as i64).unwrap_or(-1);
            (f("index-version"), f("index-template-version"))
        }
    };
    if index_version == -1 || template_version == -1 {
        return IndexStatus::Unhealthy;
    }
    if version > index_version {
        return IndexStatus::NeedsVersionBump;
    }
    if INDEX_TEMPLATE_VERSION > template_version {
        // No pending migrations: up to date whatever the template version.
        let pending: Vec<Migration> = built_migrations(name, version).into_iter().filter(|m| m.target() > template_version).collect();
        if !pending.is_empty() {
            return IndexStatus::NeedsMappingsUpdate(svc.name.clone(), pending);
        }
    }
    IndexStatus::UpToDate
}

fn creation_date(svc: &crate::index::IndexService) -> i64 {
    svc.setting("index.creation_date").and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// `ProfilingIndex.indexMetadata`: exact name, else the newest generation,
/// else the newest index of any version.
fn managed_index_state(node: &Node, prefix: &str, version: i64, generation: Option<&str>) -> IndexStatus {
    let index_prefix = format!(".{prefix}-v{version:03}");
    let name = match generation {
        Some(g) => format!("{index_prefix}-{g}"),
        None => index_prefix.clone(),
    };
    let all = node.all_indices();
    let newest = |f: &dyn Fn(&str) -> bool| all.iter().filter(|s| f(&s.name)).max_by_key(|s| creation_date(s)).cloned();
    let mut svc = node.get_index(&name).filter(|s| s.name == name);
    if svc.is_none() && generation.is_some() {
        svc = newest(&|n| n.starts_with(&index_prefix));
    }
    if svc.is_none() {
        let p = format!(".{prefix}-v");
        svc = newest(&|n| n.starts_with(&p) && n.len() >= p.len() + 3 && n[p.len()..p.len() + 3].bytes().all(|b| b.is_ascii_digit()));
    }
    index_status(node, svc, prefix, version)
}

fn data_stream_state(node: &Node, name: &str, version: i64) -> IndexStatus {
    let Some(ds) = node.data_stream(name) else { return IndexStatus::NeedsCreation };
    let svc = ds.indices.last().and_then(|w| node.get_index(w));
    if svc.is_none() {
        return IndexStatus::NeedsCreation;
    }
    index_status(node, svc, name, version)
}

fn template_version(v: &Value) -> i64 {
    v.get("version").and_then(Value::as_i64).unwrap_or(i64::MIN)
}

/// `ProfilingIndexTemplateRegistry.isAllResourcesCreated`.
fn templates_created(node: &Node) -> bool {
    {
        let t = node.templates.read();
        for (name, _, _) in COMPONENT_TEMPLATES {
            if t.component.get(*name).map(template_version).unwrap_or(i64::MIN) < INDEX_TEMPLATE_VERSION {
                return false;
            }
        }
        for (name, _, _) in INDEX_TEMPLATES {
            if t.index.get(*name).map(template_version).unwrap_or(i64::MIN) < INDEX_TEMPLATE_VERSION {
                return false;
            }
        }
    }
    policy_version(node) >= INDEX_TEMPLATE_VERSION
}

fn policy_version(node: &Node) -> i64 {
    match node.ilm.policy(ILM_POLICY.0) {
        None => i64::MIN,
        Some(p) => match p.policy.get("_meta").and_then(|m| m.get("version")) {
            Some(Value::Number(n)) => n.as_i64().unwrap_or(i64::MIN),
            Some(Value::String(s)) => s.parse().unwrap_or(i64::MIN),
            _ => i64::MIN,
        },
    }
}

/// `StatusResolver.isResourcesCreated`.
pub fn resources_created(node: &Node) -> bool {
    templates_created(node)
        && MANAGED_INDICES.iter().all(|(p, v, g)| managed_index_state(node, p, *v, *g) == IndexStatus::UpToDate)
        && managed_data_streams().iter().all(|(n, v)| data_stream_state(node, n, *v) == IndexStatus::UpToDate)
}

static ENSURE: Mutex<()> = Mutex::new(());

/// Install every missing profiling resource (no-op unless
/// `xpack.profiling.templates.enabled`).
pub fn ensure_resources(node: &Node) -> Result<()> {
    if !templates_enabled(node) || !plugin_enabled(node) {
        return Ok(());
    }
    let _g = ENSURE.lock().unwrap_or_else(|e| e.into_inner());
    install_registry(node)?;
    if !templates_created(node) {
        return Ok(());
    }
    manage_indices(node)
}

/// `ProfilingIndexTemplateRegistry` alone (component templates, the
/// `profiling-60-days` policy, composable templates), missing or older
/// ones installed, as ES does on every cluster-state change once
/// `xpack.profiling.templates.enabled` is set; called by
/// [`crate::registry::ensure`] (they used to appear only with the first
/// profiling API call). Returns whether anything changed.
pub fn ensure_registry(node: &Node) -> bool {
    if !templates_enabled(node) || !plugin_enabled(node) {
        return false;
    }
    let _g = ENSURE.lock().unwrap_or_else(|e| e.into_inner());
    match install_registry(node) {
        Ok(changed) => changed,
        Err(e) => {
            tracing::warn!("installing the profiling templates failed: {e}");
            false
        }
    }
}

fn install_registry(node: &Node) -> Result<bool> {
    // Registry: component templates, ILM policy, composable templates.
    let mut changed = false;
    {
        let mut t = node.templates.write();
        for (name, raw, var) in COMPONENT_TEMPLATES {
            if t.component.get(*name).map(template_version).unwrap_or(i64::MIN) < INDEX_TEMPLATE_VERSION {
                t.component.insert(name.to_string(), render(raw, *var));
                changed = true;
            }
        }
    }
    // ES `isUpgradeRequired`: the policy's `_meta.version`.
    if policy_version(node) < INDEX_TEMPLATE_VERSION {
        let p = crate::ilm::policy::parse_put_body(&json!({"policy": render(ILM_POLICY.1, None)}))?;
        node.ilm.put_policy(node, ILM_POLICY.0, p)?;
        changed = true;
    }
    {
        let mut t = node.templates.write();
        for (name, raw, var) in INDEX_TEMPLATES {
            if t.index.get(*name).map(template_version).unwrap_or(i64::MIN) < INDEX_TEMPLATE_VERSION {
                t.index.insert(name.to_string(), render(raw, *var));
                changed = true;
            }
        }
    }
    if changed {
        node.persist_state()?;
    }
    Ok(changed)
}

/// `ProfilingIndexManager` / `ProfilingDataStreamManager`.
fn manage_indices(node: &Node) -> Result<()> {
    // `AbstractProfilingPersistenceManager.onClusterChanged`: an index
    // considered too old aborts the whole pass (it must be deleted first).
    let indices: Vec<_> = MANAGED_INDICES.iter().map(|(p, v, g)| (*p, *v, *g, managed_index_state(node, p, *v, *g))).collect();
    if indices.iter().any(|(_, _, _, s)| *s == IndexStatus::TooOld) {
        tracing::info!("aborting profiling index creation: an index is considered too old");
        return Ok(());
    }
    // Index manager.
    for (prefix, version, generation, status) in indices {
        match status {
            IndexStatus::NeedsCreation | IndexStatus::NeedsVersionBump => {
                let base = format!(".{prefix}-v{version:03}");
                let name = match generation {
                    Some(g) => format!("{base}-{g}"),
                    None => base.clone(),
                };
                // `OnVersionBump.DELETE_OLD`: older versions go first.
                if status == IndexStatus::NeedsVersionBump && DELETE_OLD_ON_BUMP.contains(&prefix) {
                    let older: Vec<String> = node.all_indices().iter().map(|s| s.name.clone())
                        .filter(|n| n != &name && is_version_match(n, prefix)).collect();
                    for n in older {
                        node.delete_index(&n)?;
                    }
                }
                // An alias left behind by an older version: move the write
                // alias to the new index.
                if node.get_index(prefix).is_some() || alias_exists(node, prefix) {
                    for svc in node.all_indices() {
                        let mut a = svc.aliases();
                        if let Some(def) = a.get_mut(prefix) {
                            if let Some(o) = def.as_object_mut() {
                                o.insert("is_write_index".into(), json!(false));
                            }
                            svc.set_aliases(a)?;
                        }
                    }
                }
                let body = json!({"aliases": {prefix: {"is_write_index": true}}});
                match node.create_index(&name, &body) {
                    Ok(_) => {}
                    Err(e) if e.kind.contains("already_exists") => {}
                    Err(e) => return Err(e),
                }
            }
            IndexStatus::NeedsMappingsUpdate(write, pending) => apply_migrations(node, &write, &pending)?,
            _ => {}
        }
    }
    // Data stream manager.
    let streams: Vec<_> = managed_data_streams().into_iter().map(|(n, v)| { let s = data_stream_state(node, &n, v); (n, s) }).collect();
    if streams.iter().any(|(_, s)| *s == IndexStatus::TooOld) {
        tracing::info!("aborting profiling data stream creation: an index is considered too old");
        return Ok(());
    }
    for (name, status) in streams {
        match status {
            IndexStatus::NeedsCreation => match node.create_data_stream(&name) {
                Ok(_) => {}
                Err(e) if e.kind.contains("already_exists") => {}
                Err(e) => return Err(e),
            },
            // A new backing index from the current template (v15: LogsDB
            // events).
            IndexStatus::NeedsVersionBump => {
                node.rollover(&name, None, &json!({}), false)?;
            }
            IndexStatus::NeedsMappingsUpdate(write, pending) => apply_migrations(node, &write, &pending)?,
            _ => {}
        }
    }
    Ok(())
}

/// `ProfilingIndex.isMatchWithoutVersion`: `.<prefix>-v<3 latin digits>…`.
fn is_version_match(name: &str, prefix: &str) -> bool {
    let p = format!(".{prefix}-v");
    name.starts_with(&p) && name.len() >= p.len() + 3 && name.as_bytes()[p.len()..p.len() + 3].iter().all(u8::is_ascii_digit)
}

/// `AbstractProfilingPersistenceManager.applyMigrations`: each pending
/// migration on the write index, in order (put mapping / update settings).
fn apply_migrations(node: &Node, write_index: &str, pending: &[Migration]) -> Result<()> {
    let Some(svc) = node.get_index(write_index) else { return Ok(()) };
    for m in pending {
        match m {
            Migration::PutMapping { body, .. } => {
                svc.put_mapping(body)?;
            }
            Migration::DynamicSettings { settings, .. } => {
                let flat: crate::settings::FlatSettings = settings.as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.clone())).collect();
                svc.put_settings(flat, false)?;
            }
        }
    }
    Ok(())
}

fn alias_exists(node: &Node, alias: &str) -> bool {
    node.all_indices().iter().any(|s| s.aliases().contains_key(alias))
}

/// `profiling-events-all` holds at least one searchable document.
fn has_data(node: &Node) -> bool {
    let opts = ResolveOpts { ignore_unavailable: true, allow_no_indices: true, ..Default::default() };
    let Ok(targets) = node.resolve(ALL_EVENTS, opts) else { return false };
    if targets.is_empty() {
        return false;
    }
    let body = json!({"size": 0, "track_total_hits": 1});
    match crate::transform::pivot::search(node, &targets, &body) {
        Ok(v) => v["hits"]["total"]["value"].as_i64().unwrap_or(0) > 0,
        Err(_) => false,
    }
}

/// `GET /_profiling/status`: `(status code, body)`.
pub fn status(node: &Node, wait_for_resources: bool, timeout: Duration) -> Result<(u16, Value)> {
    status_with(node, wait_for_resources, timeout, None)
}

/// [`status`] with `has_data` answered by `data` (on a cluster: a search
/// through the coordinator, which reaches every shard of the events indices).
pub fn status_with(node: &Node, wait_for_resources: bool, timeout: Duration, data: Option<&dyn Fn() -> bool>) -> Result<(u16, Value)> {
    ensure_resources(node)?;
    let mut timed_out = false;
    if wait_for_resources && !resources_created(node) {
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() >= deadline {
                timed_out = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50).min(deadline.saturating_duration_since(Instant::now())));
            ensure_resources(node)?;
            if resources_created(node) {
                break;
            }
        }
    }
    let created = resources_created(node);
    let body = json!({
        "profiling": {"enabled": plugin_enabled(node)},
        "resource_management": {"enabled": templates_enabled(node)},
        "resources": {"created": created, "pre_8_9_1_data": false, "has_data": created && data.map_or_else(|| has_data(node), |f| f())},
    });
    Ok((if timed_out { 408 } else { 200 }, body))
}

/// Universal Profiling needs an enterprise (or trial) license.
pub fn check_license(node: &Node) -> Result<()> {
    crate::license::check(node, "universal_profiling", crate::license::Mode::Enterprise)
}

pub(crate) fn not_found(e: &EsError) -> bool {
    e.kind == "index_not_found_exception"
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    fn node() -> (tempfile::TempDir, std::sync::Arc<Node>) {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        node.cluster_settings.write().0.insert(TEMPLATES_ENABLED.into(), json!("true"));
        (dir, node)
    }

    fn write_index(node: &Node, ds: &str) -> std::sync::Arc<crate::index::IndexService> {
        let d = node.data_stream(ds).unwrap();
        node.get_index(d.write_index()).unwrap()
    }

    /// ES 9.5.5 registry v15: LogsDB events; an events data stream at events
    /// version 5 (an older install) is rolled over to a LogsDB backing
    /// index; the other streams stay.
    #[test]
    fn v15_events_are_logsdb_and_older_streams_roll_over() {
        let (_d, node) = node();
        ensure_resources(&node).unwrap();
        assert!(resources_created(&node));
        let events = write_index(&node, ALL_EVENTS);
        assert_eq!(events.setting("index.mode").as_deref(), Some("logsdb"), "v15 events template");
        let meta = events.meta().mappings.to_json()["_meta"].clone();
        assert_eq!((meta["index-version"].as_i64(), meta["index-template-version"].as_i64()), (Some(6), Some(15)), "{meta}");
        // An older install's write index (events version 5, template 14).
        events.put_mapping(&json!({"_meta": {"index-version": 5, "index-template-version": 14}})).unwrap();
        let hosts_before = write_index(&node, "profiling-hosts").name.clone();
        assert!(!resources_created(&node));
        ensure_resources(&node).unwrap();
        let rolled = write_index(&node, ALL_EVENTS);
        assert_ne!(rolled.name, events.name, "rolled over");
        assert_eq!(rolled.setting("index.mode").as_deref(), Some("logsdb"));
        assert_eq!(write_index(&node, "profiling-hosts").name, hosts_before, "no migrations, no version bump: untouched");
        assert!(resources_created(&node));
    }

    /// Pending migrations (`Migration.Builder`) apply to the write index and
    /// end with the `_meta` version update; without any (ES 9.5.5) an older
    /// template version is up to date.
    #[test]
    fn migrations_apply_to_the_write_index() {
        let (_d, node) = node();
        ensure_resources(&node).unwrap();
        let hosts = write_index(&node, "profiling-hosts");
        hosts.put_mapping(&json!({"_meta": {"index-version": 2, "index-template-version": 14}})).unwrap();
        assert!(resources_created(&node), "no migrations: up to date at an older template version");
        TEST_MIGRATIONS.with(|t| t.borrow_mut().insert("profiling-hosts".into(), vec![
            Migration::PutMapping { target: 15, body: json!({"properties": {"migrated.field": {"type": "keyword"}}}) },
        ]));
        assert!(!resources_created(&node));
        ensure_resources(&node).unwrap();
        let m = write_index(&node, "profiling-hosts").meta().mappings.to_json();
        assert_eq!(m["properties"]["migrated"]["properties"]["field"]["type"], "keyword", "{m}");
        assert_eq!(m["_meta"]["index-template-version"], 15);
        assert!(resources_created(&node));
        TEST_MIGRATIONS.with(|t| t.borrow_mut().clear());
    }

    /// `OnVersionBump.DELETE_OLD` (sq-executables, sq-leafframes): an older
    /// version is deleted when the new one is created; KEEP_OLD indices keep
    /// theirs.
    #[test]
    fn version_bumps_delete_old_only_where_es_does() {
        let (_d, node) = node();
        install_registry(&node).unwrap();
        for prefix in ["profiling-sq-executables", "profiling-symbols-private"] {
            let old = format!(".{prefix}-v000");
            node.create_index(&old, &json!({"aliases": {prefix: {"is_write_index": true}}, "mappings": {"_meta": {"index-version": 0, "index-template-version": 14}}})).unwrap();
        }
        ensure_resources(&node).unwrap();
        assert!(node.get_index(".profiling-sq-executables-v000").is_none(), "DELETE_OLD");
        assert!(node.get_index(".profiling-sq-executables-v001").is_some());
        assert!(node.get_index(".profiling-symbols-private-v000").is_some(), "KEEP_OLD");
        assert!(node.get_index(".profiling-symbols-private-v001").is_some());
    }
}
