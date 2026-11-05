//! Node-level metadata for the search extensions: synonyms sets, search
//! applications, query rulesets, plus transient
//! ES|QL async results and SQL cursors.
//!
//! The state is keyed by the node's data directory (so tests running
//! several nodes in one process stay isolated) and persisted to
//! `<data>/_ext_state.json` (ES keeps these in cluster state / system
//! indices). Synonyms sets are additionally mirrored into a process-wide
//! table, keyed by data directory, that the analysis chain reads
//! (`synonyms_set` on a `synonym` / `synonym_graph` filter).

use crate::error::{EsError, Result};
use crate::node::Node;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub const MAX_SYNONYM_RULES: usize = 100_000;

pub enum SqlRows {
    Materialized(std::vec::IntoIter<Value>),
    /// Executor rows, converted to JSON a page at a time (a large sorted or
    /// grouped result is not converted whole when one page is read).
    Table(std::vec::IntoIter<Vec<crate::esql::Val>>),
    Scan(crate::esql::exec::SqlScan),
}

impl SqlRows {
    pub fn len(&self) -> usize {
        match self { Self::Materialized(rows) => rows.len(), Self::Table(rows) => rows.len(), Self::Scan(scan) => scan.len() }
    }
    pub fn page(&mut self, node: &Node, fetch: usize) -> Result<Vec<Value>> {
        match self {
            Self::Materialized(rows) => Ok(rows.by_ref().take(fetch).collect()),
            Self::Table(rows) => Ok(rows.by_ref().take(fetch).map(|row| Value::Array(row.into_iter().map(|v| v.to_json()).collect())).collect()),
            Self::Scan(scan) => scan.page(node, fetch),
        }
    }
}

pub struct SqlCursor {
    pub columns: Vec<Value>,
    pub rows: SqlRows,
    pub fetch: usize,
    pub expires: Instant,
}

#[cfg(test)]
mod sql_cursor_tests {
    use super::*;
    use std::time::Duration;

    fn cursor(expires: Instant) -> SqlCursor {
        SqlCursor { columns: vec![json!({"name": "value"})],
            rows: SqlRows::Materialized(vec![json!([1]), json!([2])].into_iter()), fetch: 1, expires }
    }

    #[test]
    fn expiry_removes_only_elapsed_contexts_and_returns_owned_rows() {
        let state = ExtState::default();
        let now = Instant::now();
        {
            let mut cursors = state.sql_cursors.lock();
            cursors.insert("expired".into(), cursor(now - Duration::from_secs(1)));
            cursors.insert("boundary".into(), cursor(now));
            cursors.insert("live".into(), cursor(now + Duration::from_secs(1)));
        }
        let expired = state.take_expired_sql_cursors(now);
        assert_eq!(expired.len(), 2);
        assert!(expired.iter().all(|cursor| cursor.rows.len() == 2));
        // Returned buffers remain owned outside the map lock, so their
        // potentially expensive deallocation can run on a blocking worker.
        let cursors = state.sql_cursors.try_lock().unwrap();
        assert_eq!(cursors.len(), 1);
        assert!(cursors.contains_key("live"));
        drop(cursors);
        drop(expired);
        assert_eq!(state.take_expired_sql_cursors(now + Duration::from_secs(1)).len(), 1);
        assert!(state.sql_cursors.lock().is_empty());
    }

    #[test]
    fn maintenance_defers_busy_cursor_map_without_blocking() {
        let state = ExtState::default();
        let now = Instant::now();
        let mut cursors = state.sql_cursors.lock();
        cursors.insert("expired".into(), cursor(now));
        assert!(state.take_expired_sql_cursors(now).is_empty());
        drop(cursors);
        assert_eq!(state.take_expired_sql_cursors(now).len(), 1);
    }
}

#[derive(Default)]
pub struct ExtState {
    path: PathBuf,
    /// set id → ordered rules `(rule id, synonyms)`.
    pub synonyms: RwLock<BTreeMap<String, Vec<(String, String)>>>,
    /// name → application definition (with `updated_at_millis`).
    pub search_apps: RwLock<BTreeMap<String, Value>>,
    /// ruleset id → rules array.
    pub rulesets: RwLock<BTreeMap<String, Vec<Value>>>,
    /// Rollup jobs (ES: persistent tasks): id → `{"config": .., "state": "stopped"|"started"}`.
    pub rollup_jobs: RwLock<BTreeMap<String, Value>>,
    /// Autoscaling policies: name → `{"roles": [..], "deciders": {..}}`.
    pub autoscaling_policies: RwLock<BTreeMap<String, Value>>,
    /// ES|QL async results: id → (response, is_running).
    pub esql_async: Mutex<HashMap<String, Value>>,
    /// SQL cursors with a finite retention deadline.
    /// Advancing the iterator moves only the requested page, rather than
    /// reallocating and moving every remaining row on each continuation.
    pub sql_cursors: Mutex<HashMap<String, SqlCursor>>,
    /// SQL async results.
    pub sql_async: Mutex<HashMap<String, Value>>,
    /// EQL async/stored results.
    pub eql_async: Mutex<HashMap<String, Value>>,
    pub sql_stats: Mutex<BTreeMap<String, u64>>,
    /// ES|QL usage counters (`queries.rest.total`, `features.where`, `functions.max`, ...).
    pub esql_stats: Mutex<BTreeMap<String, u64>>,
    /// Running ES|QL queries (`GET /_query/queries`): task id (`node:n`) →
    /// (encoded id, start millis, start instant, query).
    pub esql_running: Mutex<BTreeMap<String, (String, i64, std::time::Instant, String)>>,
}

static STATES: OnceLock<Mutex<HashMap<PathBuf, Arc<ExtState>>>> = OnceLock::new();

/// Remove expired contexts without dropping their potentially large row buffers
/// under a mutex. Cluster callers must drop the returned values off async workers.
/// A busy metadata/cursor lock is retried on the next maintenance tick.
pub fn take_expired_sql_cursors(node: &Node, now: Instant) -> Vec<SqlCursor> {
    let Some(states) = STATES.get() else { return Vec::new(); };
    let state = {
        let Some(states) = states.try_lock() else { return Vec::new(); };
        states.get(&node.data).cloned()
    };
    state.map(|state| state.take_expired_sql_cursors(now)).unwrap_or_default()
}
/// data dir → set id → rules (one node's sets never leak into another's).
static SYNONYMS: OnceLock<RwLock<HashMap<PathBuf, HashMap<String, Vec<String>>>>> = OnceLock::new();

fn global_synonyms() -> &'static RwLock<HashMap<PathBuf, HashMap<String, Vec<String>>>> {
    SYNONYMS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Rules of synonyms set `set` of the node at `data` (`None`: no such set);
/// read by the analysis chain when a shard recovers / analyzers reload.
pub fn synonym_set_rules(data: &Path, set: &str) -> Option<Vec<String>> {
    global_synonyms().read().get(data)?.get(set).cloned()
}

/// The extension state of `node` (loaded from disk on first use).
pub fn ext(node: &Node) -> Arc<ExtState> {
    let map = STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock();
    if let Some(s) = g.get(&node.data) {
        return s.clone();
    }
    let st = Arc::new(load(node.data.join("_ext_state.json")));
    g.insert(node.data.clone(), st.clone());
    drop(g);
    // Indices opened before the synonyms were known: rebuild their analyzers.
    let sets: Vec<String> = st.synonyms.read().keys().cloned().collect();
    for s in sets {
        st.publish_synonyms(&s);
        let _ = reload_dependents(node, &s);
    }
    st
}

fn load(path: PathBuf) -> ExtState {
    let st = ExtState { path: path.clone(), ..Default::default() };
    let Ok(raw) = std::fs::read_to_string(&path) else { return st };
    let Ok(v) = serde_json::from_str::<Value>(&raw) else { return st };
    if let Some(m) = v.get("synonyms").and_then(Value::as_object) {
        let mut g = st.synonyms.write();
        for (k, rules) in m {
            let rs = rules
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|r| (r["id"].as_str().unwrap_or("").to_string(), r["synonyms"].as_str().unwrap_or("").to_string()))
                        .collect()
                })
                .unwrap_or_default();
            g.insert(k.clone(), rs);
        }
    }
    if let Some(m) = v.get("search_apps").and_then(Value::as_object) {
        st.search_apps.write().extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    if let Some(m) = v.get("rulesets").and_then(Value::as_object) {
        st.rulesets.write().extend(m.iter().map(|(k, v)| (k.clone(), v.as_array().cloned().unwrap_or_default())));
    }
    if let Some(m) = v.get("rollup_jobs").and_then(Value::as_object) {
        st.rollup_jobs.write().extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    if let Some(m) = v.get("autoscaling_policies").and_then(Value::as_object) {
        st.autoscaling_policies.write().extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    st
}

impl ExtState {
    pub fn take_expired_sql_cursors(&self, now: Instant) -> Vec<SqlCursor> {
        let Some(mut cursors) = self.sql_cursors.try_lock() else { return Vec::new(); };
        let expired: Vec<String> = cursors.iter().filter(|(_, cursor)| cursor.expires <= now).map(|(id, _)| id.clone()).collect();
        expired.into_iter().filter_map(|id| cursors.remove(&id)).collect()
    }

    pub fn persist(&self) -> Result<()> {
        let syn: Map<String, Value> = self
            .synonyms
            .read()
            .iter()
            .map(|(k, rs)| (k.clone(), Value::Array(rs.iter().map(|(id, s)| json!({"id": id, "synonyms": s})).collect())))
            .collect();
        let v = json!({
            "synonyms": syn,
            "search_apps": self.search_apps.read().clone(),
            "rulesets": self.rulesets.read().clone(),
            "rollup_jobs": self.rollup_jobs.read().clone(),
            "autoscaling_policies": self.autoscaling_policies.read().clone(),
        });
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, v.to_string()).map_err(|e| EsError::internal(format!("failed to persist state: {e}")))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| EsError::internal(format!("failed to persist state: {e}")))?;
        Ok(())
    }

    pub fn publish_synonyms(&self, set: &str) {
        let rules: Option<Vec<String>> = self.synonyms.read().get(set).map(|rs| rs.iter().map(|(_, s)| s.clone()).collect());
        // `<data>/_ext_state.json` → `<data>`.
        let data = self.path.parent().map(Path::to_path_buf).unwrap_or_default();
        let mut g = global_synonyms().write();
        let sets = g.entry(data).or_default();
        match rules {
            Some(r) => {
                sets.insert(set.to_string(), r);
            }
            None => {
                sets.remove(set);
            }
        }
    }
}

// ---------------------------------------------------------------- scripts

/// A stored Mustache template (`PUT _scripts/{id}` with `lang: mustache`),
/// as template source text.
pub fn stored_template(node: &Node, id: &str) -> Result<String> {
    let st = crate::script::stored::for_data_dir(&node.data);
    match st.get(id) {
        Some(s) if s.lang == "mustache" => Ok(s.source),
        Some(s) => Err(EsError::illegal_argument(format!("unable to execute stored script [{id}] with lang [{}] as a template", s.lang))),
        None => Err(EsError::resource_not_found(format!("unable to find script [{id}] in cluster state"))),
    }
}

// --------------------------------------------------------------- synonyms

/// Whether a `synonyms_set` setting (one name or a list) names `set`.
fn setting_names(v: &Value, set: &str) -> bool {
    match v {
        Value::String(s) => s == set,
        Value::Array(a) => a.iter().any(|x| x.as_str() == Some(set)),
        _ => false,
    }
}

/// `synonyms.max_synonym_rules` (dynamic cluster setting, default 100000).
pub fn max_synonym_rules(node: &Node) -> usize {
    const KEY: &str = "synonyms.max_synonym_rules";
    let cs = node.cluster_settings.read();
    cs.1.get(KEY).or(cs.0.get(KEY)).and_then(|v| v.as_u64().map(|n| n as usize).or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(MAX_SYNONYM_RULES)
}

fn check_rule_count(node: &Node, count: usize) -> Result<()> {
    let max = max_synonym_rules(node);
    if count > max {
        return Err(EsError::illegal_argument(format!("The number of synonym rules in a synonym set cannot exceed {max}")));
    }
    Ok(())
}

/// Indices whose analysis settings reference synonyms set `set`.
pub fn dependents(node: &Node, set: &str) -> Vec<Arc<crate::index::IndexService>> {
    node.all_indices()
        .into_iter()
        .filter(|svc| {
            svc.meta().settings.iter().any(|(k, v)| k.starts_with("index.analysis.filter.") && k.ends_with(".synonyms_set") && setting_names(v, set))
        })
        .collect()
}

/// Rebuild the analyzers of the open indices using `set`; the ES reload
/// details (`TransportReloadAnalyzersAction`, one shard per index here).
/// A shard whose analyzers no longer build (a `lenient: false` filter and
/// an invalid rule) keeps its current analyzers and is reported under
/// `_shards.failures`; an index whose primary is unassigned counts in
/// `_shards.total` only.
pub fn reload_dependents(node: &Node, set: &str) -> Result<Value> {
    let deps: Vec<_> = dependents(node, set).into_iter().filter(|s| s.is_open()).collect();
    let mut details = Vec::new();
    let mut failures = Vec::new();
    let mut successful = 0;
    let mut total = 0;
    for svc in &deps {
        if svc.local_shards().is_empty() {
            // No copy here (a cluster mirror): its analyzers still serve
            // `_analyze` and query parsing, but there is no shard to report.
            // Once they build, drop the failures recorded while the set was
            // missing (the cluster layer opens the real copies).
            if svc.reload_search_analyzers().is_ok() {
                svc.clear_failed_without_copies();
            }
            continue;
        }
        total += 1;
        if let Err(e) = svc.reload_search_analyzers() {
            let mut reason = e.to_json()["error"].clone();
            if let Some(o) = reason.as_object_mut() {
                o.remove("root_cause");
            }
            failures.push(json!({"shard": 0, "index": svc.name, "status": rest_status_name(e.status), "reason": reason}));
            continue;
        }
        successful += 1;
        let settings = svc.meta().settings;
        let filters: Vec<String> = settings
            .iter()
            .filter(|(k, v)| k.ends_with(".synonyms_set") && setting_names(v, set))
            .filter_map(|(k, _)| k.strip_prefix("index.analysis.filter.").and_then(|r| r.strip_suffix(".synonyms_set")).map(str::to_string))
            .collect();
        let analyzers: Vec<String> = settings
            .iter()
            .filter_map(|(k, v)| {
                let name = k.strip_prefix("index.analysis.analyzer.")?.strip_suffix(".filter")?;
                let uses = match v {
                    Value::Array(a) => a.iter().any(|f| f.as_str().map(|x| filters.iter().any(|y| y == x)).unwrap_or(false)),
                    Value::String(x) => filters.iter().any(|y| y == x),
                    _ => false,
                };
                uses.then(|| name.to_string())
            })
            .collect();
        details.push(json!({"index": svc.name, "reloaded_analyzers": analyzers, "reloaded_node_ids": [crate::node::NODE_ID]}));
    }
    let mut shards = json!({"total": total, "successful": successful, "failed": failures.len()});
    if !failures.is_empty() {
        shards["failures"] = Value::Array(failures);
    }
    Ok(json!({"_shards": shards, "reload_details": details}))
}

/// ES `RestStatus` constant name of an HTTP status (shard failure `status`).
fn rest_status_name(status: u16) -> &'static str {
    match status {
        400 => "BAD_REQUEST",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        409 => "CONFLICT",
        429 => "TOO_MANY_REQUESTS",
        503 => "SERVICE_UNAVAILABLE",
        _ => "INTERNAL_SERVER_ERROR",
    }
}

fn parse_rules(body: &Value) -> Result<Vec<(String, String)>> {
    // `PutSynonymsAction.Request` (ObjectParser `synonyms_set`).
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "synonyms_set" {
                return Err(EsError::unknown_field_xc("synonyms_set", k, None, &["synonyms_set"]));
            }
            if let Value::Array(a) = v {
                for r in a {
                    if let Some(ro) = r.as_object() {
                        for rk in ro.keys() {
                            if !matches!(rk.as_str(), "id" | "synonyms") {
                                return Err(EsError::field_failure("synonyms_set", "synonyms_set", None, EsError::unknown_field_xc("synonym_rule", rk, Some("synonyms_set"), &["id", "synonyms"])));
                            }
                        }
                    }
                }
            } else if !v.is_object() {
                return Err(EsError::unsupported_type_xc("synonyms_set", "synonyms_set", v, None));
            }
        }
    }
    let arr = match body.get("synonyms_set") {
        Some(Value::Array(a)) => a.clone(),
        Some(o @ Value::Object(_)) => vec![o.clone()],
        _ => return Err(EsError::illegal_argument("Required [synonyms_set]")),
    };
    let mut out = Vec::new();
    for r in arr {
        let syn = r.get("synonyms").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["synonyms rule must have synonyms"]))?;
        validate_rule(syn)?;
        let id = r.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(crate::index::auto_id);
        out.push((id, syn.to_string()));
    }
    Ok(out)
}

fn validate_rule(s: &str) -> Result<()> {
    if s.trim().is_empty() {
        return Err(EsError::action_validation(&["[synonyms] field can't be empty"]));
    }
    let parts: Vec<&str> = s.split("=>").collect();
    if parts.len() > 2 {
        return Err(EsError::illegal_argument("More than one explicit mapping specified in the same synonyms rule"));
    }
    for side in &parts {
        if side.split(',').any(|t| t.trim().is_empty()) {
            return Err(EsError::illegal_argument(format!("Incorrect syntax for [synonyms]: [{s}]")));
        }
    }
    Ok(())
}

/// `PUT _synonyms/{set}[?refresh][?append]`: returns (created, reload
/// details; `None` with `refresh=false`).
pub fn put_synonyms(node: &Node, set: &str, body: &Value, refresh: bool, append: bool) -> Result<(bool, Option<Value>)> {
    let rules = parse_rules(body)?;
    let st = ext(node);
    let created = {
        let mut g = st.synonyms.write();
        match (append, g.get_mut(set)) {
            (true, Some(existing)) => {
                // `addToSynonymsSet`: rules overwriting an id still count as new.
                check_rule_count(node, existing.len() + rules.len())?;
                for (id, syn) in rules {
                    match existing.iter_mut().find(|(x, _)| *x == id) {
                        Some(r) => r.1 = syn,
                        None => existing.push((id, syn)),
                    }
                }
                false
            }
            (_, existing) => {
                check_rule_count(node, rules.len())?;
                let created = existing.is_none();
                g.insert(set.to_string(), rules);
                created
            }
        }
    };
    st.publish_synonyms(set);
    st.persist()?;
    Ok((created, if refresh { Some(reload_dependents(node, set)?) } else { None }))
}

pub fn put_synonym_rule(node: &Node, set: &str, rule: &str, body: &Value, refresh: bool) -> Result<(bool, Option<Value>)> {
    // `PutSynonymRuleAction.Request` (ObjectParser `synonyms`).
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "synonyms" {
                return Err(EsError::unknown_field_xc("synonyms", k, None, &["synonyms"]));
            }
            if !v.is_string() {
                return Err(EsError::unsupported_type_xc("synonyms", "synonyms", v, None));
            }
        }
    }
    let syn = body.get("synonyms").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [synonyms]"))?;
    validate_rule(syn)?;
    let st = ext(node);
    let created = {
        let mut g = st.synonyms.write();
        let rs = g.get_mut(set).ok_or_else(|| EsError::resource_not_found(format!("synonyms set [{set}] not found")))?;
        // The rule being replaced does not count against the limit.
        check_rule_count(node, rs.iter().filter(|(id, _)| id != rule).count() + 1)?;
        match rs.iter_mut().find(|(id, _)| id == rule) {
            Some(r) => {
                r.1 = syn.to_string();
                false
            }
            None => {
                rs.push((rule.to_string(), syn.to_string()));
                true
            }
        }
    };
    st.publish_synonyms(set);
    st.persist()?;
    Ok((created, if refresh { Some(reload_dependents(node, set)?) } else { None }))
}

pub fn delete_synonym_rule(node: &Node, set: &str, rule: &str, refresh: bool) -> Result<Option<Value>> {
    let st = ext(node);
    {
        let mut g = st.synonyms.write();
        let rs = g.get_mut(set).ok_or_else(|| EsError::resource_not_found(format!("synonyms set [{set}] not found")))?;
        let before = rs.len();
        rs.retain(|(id, _)| id != rule);
        if rs.len() == before {
            return Err(EsError::resource_not_found(format!("synonym rule [{rule}] not found on synonyms set [{set}]")));
        }
    }
    st.publish_synonyms(set);
    st.persist()?;
    if refresh { reload_dependents(node, set).map(Some) } else { Ok(None) }
}

pub fn delete_synonyms(node: &Node, set: &str) -> Result<()> {
    let st = ext(node);
    if !st.synonyms.read().contains_key(set) {
        return Err(EsError::resource_not_found(format!("synonyms set [{set}] not found")));
    }
    let deps = dependents(node, set);
    if !deps.is_empty() {
        let names: Vec<String> = deps.iter().map(|d| d.name.clone()).collect();
        return Err(EsError::illegal_argument(format!(
            "synonyms set [{set}] cannot be deleted as it is used in the following indices: {}",
            names.join(", ")
        )));
    }
    st.synonyms.write().remove(set);
    st.publish_synonyms(set);
    st.persist()
}

// ------------------------------------------------------------ query rules

pub const CRITERIA_TYPES: &[&str] = &["global", "exact", "exact_fuzzy", "fuzzy", "prefix", "suffix", "contains", "lt", "lte", "gt", "gte", "always"];

/// Java `Double.parseDouble` acceptance (leading/trailing whitespace,
/// a `d`/`f` suffix, `NaN`, `Infinity`).
fn java_parses_double(s: &str) -> bool {
    let t = s.trim();
    let t = t.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(t);
    let unsigned = t.strip_prefix(['+', '-']).unwrap_or(t);
    if unsigned == "NaN" || unsigned == "Infinity" {
        return true;
    }
    !unsigned.is_empty() && unsigned.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')) && t.parse::<f64>().is_ok()
}

/// `QueryRuleCriteriaType.isValidForInput`.
fn criteria_value_valid(ct: &str, v: &Value) -> bool {
    match ct {
        "exact" => v.is_string() || v.is_number(),
        "fuzzy" | "prefix" | "suffix" | "contains" => v.is_string(),
        "lt" | "lte" | "gt" | "gte" => match v {
            Value::Number(_) => true,
            Value::String(s) => java_parses_double(s),
            Value::Bool(_) | Value::Null | Value::Array(_) | Value::Object(_) => false,
        },
        _ => true,
    }
}

pub fn validate_rule_def(rule: &Value) -> Result<()> {
    let ty = rule.get("type").and_then(Value::as_str).ok_or_else(|| EsError::action_validation(&["type is required"]))?;
    if !matches!(ty, "pinned" | "exclude") {
        return Err(EsError::illegal_argument(format!("Unknown QueryRuleType: {ty}")));
    }
    let crit = rule.get("criteria").ok_or_else(|| EsError::action_validation(&["criteria cannot be empty"]))?;
    let crits: Vec<Value> = match crit {
        Value::Array(a) => a.clone(),
        o => vec![o.clone()],
    };
    if crits.is_empty() {
        return Err(EsError::action_validation(&["criteria cannot be empty"]));
    }
    for c in &crits {
        let ct = c.get("type").and_then(Value::as_str).unwrap_or("");
        if !CRITERIA_TYPES.contains(&ct) {
            return Err(EsError::illegal_argument(format!("Unknown QueryRuleCriteriaType: {ct}")));
        }
        if ct != "always" && ct != "global" && (c.get("metadata").is_none() || c.get("values").is_none()) {
            return Err(EsError::illegal_argument(format!("{ct} query rule criteria requires metadata and values")));
        }
        // `QueryRuleCriteriaType.validateInput` on every criteria value.
        let values: Vec<&Value> = match c.get("values") {
            Some(Value::Array(a)) => a.iter().collect(),
            Some(v) => vec![v],
            None => vec![],
        };
        for v in values {
            if !criteria_value_valid(ct, v) {
                let shown = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                return Err(EsError::illegal_argument(format!("Input [{shown}] is not valid for CriteriaType [{ct}]")));
            }
        }
    }
    let actions = rule.get("actions").ok_or_else(|| EsError::action_validation(&["actions cannot be empty"]))?;
    if actions.get("ids").is_none() && actions.get("docs").is_none() {
        return Err(EsError::illegal_argument("Pinned query rule actions must contain either ids or docs"));
    }
    Ok(())
}

fn crit_match(c: &Value, criteria: &Map<String, Value>) -> bool {
    let ct = c.get("type").and_then(Value::as_str).unwrap_or("");
    if ct == "always" || ct == "global" {
        return true;
    }
    let md = c.get("metadata").and_then(Value::as_str).unwrap_or("");
    let Some(input) = criteria.get(md) else { return false };
    let inputs: Vec<&Value> = match input {
        Value::Array(a) => a.iter().collect(),
        v => vec![v],
    };
    let values: Vec<&Value> = match c.get("values") {
        Some(Value::Array(a)) => a.iter().collect(),
        Some(v) => vec![v],
        None => vec![],
    };
    let s = |v: &Value| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let num = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|x| x.parse().ok()));
    inputs.iter().any(|inp| {
        values.iter().any(|val| match ct {
            "exact" => s(inp) == s(val),
            "fuzzy" | "exact_fuzzy" => crate::esql::levenshtein(&s(inp), &s(val)) <= 2 || s(inp) == s(val),
            "prefix" => s(inp).starts_with(&s(val)),
            "suffix" => s(inp).ends_with(&s(val)),
            "contains" => s(inp).contains(&s(val)),
            "lt" | "lte" | "gt" | "gte" => match (num(inp), num(val)) {
                (Some(a), Some(b)) => match ct {
                    "lt" => a < b,
                    "lte" => a <= b,
                    "gt" => a > b,
                    _ => a >= b,
                },
                _ => false,
            },
            _ => false,
        })
    })
}

/// Rules of `ruleset_ids` matching `criteria`, in priority order: `(ruleset, rule)`.
pub fn matching_rules(node: &Node, ruleset_ids: &[String], criteria: &Map<String, Value>) -> Result<Vec<(String, Value)>> {
    let st = ext(node);
    let g = st.rulesets.read();
    let mut out = Vec::new();
    for rs in ruleset_ids {
        let rules = g.get(rs).ok_or_else(|| EsError::resource_not_found(format!("query ruleset [{rs}] not found")))?;
        let mut sorted: Vec<&Value> = rules.iter().collect();
        sorted.sort_by_key(|r| r.get("priority").and_then(Value::as_i64).unwrap_or(i64::MAX));
        for r in sorted {
            let crits: Vec<Value> = match r.get("criteria") {
                Some(Value::Array(a)) => a.clone(),
                Some(o) => vec![o.clone()],
                None => vec![],
            };
            if crits.iter().all(|c| crit_match(c, criteria)) {
                out.push((rs.clone(), r.clone()));
            }
        }
    }
    Ok(out)
}

fn action_ids(r: &Value) -> Vec<Value> {
    let a = &r["actions"];
    if let Some(ids) = a.get("ids").and_then(Value::as_array) {
        return ids.iter().map(|i| json!({"ids": {"values": [i]}})).collect();
    }
    a.get("docs")
        .and_then(Value::as_array)
        .map(|d| {
            d.iter()
                .map(|doc| {
                    let id = doc.get("_id").cloned().unwrap_or(Value::Null);
                    match doc.get("_index").and_then(Value::as_str) {
                        Some(ix) => json!({"bool": {"filter": [{"ids": {"values": [id]}}, {"term": {"_index": ix}}]}}),
                        None => json!({"ids": {"values": [id]}}),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}


/// The XContent token name of a JSON value (ES parser error wording).
fn token_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "VALUE_NULL",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
    }
}

/// Rewrite a `rule` query into pinned/excluded bool queries.
fn rewrite_rule_query(node: &Node, rq: &Value) -> Result<Value> {
    let organic = rq.get("organic").cloned().unwrap_or(json!({"match_all": {}}));
    let organic = rewrite_query(node, &organic)?;
    if let Some(m) = rq.as_object() {
        for k in m.keys() {
            if !matches!(k.as_str(), "organic" | "ruleset_ids" | "ruleset_id" | "match_criteria" | "boost" | "_name") {
                return Err(EsError::x_content_parse(format!("[1:2] [rule] unknown field [{k}]")));
            }
        }
    }
    // `ruleset_ids` (a list) or the legacy single `ruleset_id`, never both.
    let ids: Vec<String> = match (rq.get("ruleset_ids"), rq.get("ruleset_id")) {
        (Some(_), Some(_)) | (None, None) => return Err(EsError::parsing("ruleset information not provided correctly")),
        (Some(Value::Array(a)), None) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        (Some(Value::String(s)), None) => vec![s.clone()],
        (Some(other), None) => return Err(EsError::x_content_parse(format!("[1:2] [rule] ruleset_ids doesn't support values of type: {}", token_name(other)))),
        (None, Some(Value::String(s))) => vec![s.clone()],
        (None, Some(other)) => return Err(EsError::x_content_parse(format!("[1:2] [rule] ruleset_id doesn't support values of type: {}", token_name(other)))),
    };
    if ids.is_empty() {
        return Err(EsError::parsing("rulesetIds must not be null or empty"));
    }
    if ids.len() > 10 {
        return Err(EsError::parsing("rulesetIds must not contain more than 10 rulesets"));
    }
    let criteria = match rq.get("match_criteria") {
        Some(Value::Object(m)) => m.clone(),
        _ => return Err(EsError::parsing("Required [match_criteria]")),
    };
    let matched = matching_rules(node, &ids, &criteria)?;
    let mut pinned = Vec::new();
    let mut excluded = Vec::new();
    for (_, r) in &matched {
        if r["type"] == "exclude" {
            excluded.extend(action_ids(r));
        } else {
            pinned.extend(action_ids(r));
        }
    }
    if pinned.is_empty() && excluded.is_empty() {
        return Ok(organic);
    }
    let n = pinned.len();
    let mut should: Vec<Value> = pinned
        .into_iter()
        .enumerate()
        .map(|(i, q)| json!({"constant_score": {"filter": q, "boost": 1.0e7_f64 * (n - i) as f64}}))
        .collect();
    should.push(organic);
    let mut b = json!({"bool": {"should": should, "minimum_should_match": 1}});
    if !excluded.is_empty() {
        b["bool"]["must_not"] = Value::Array(excluded);
    }
    Ok(b)
}

/// Walk a query, rewriting `rule` queries.
pub fn rewrite_query(node: &Node, q: &Value) -> Result<Value> {
    match q {
        Value::Object(m) => {
            if m.len() == 1 {
                if let Some(rq) = m.get("rule").or_else(|| m.get("rule_query")) {
                    return rewrite_rule_query(node, rq);
                }
            }
            let mut out = Map::new();
            for (k, v) in m {
                out.insert(k.clone(), rewrite_query(node, v)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(a) => Ok(Value::Array(a.iter().map(|x| rewrite_query(node, x)).collect::<Result<_>>()?)),
        other => Ok(other.clone()),
    }
}

fn mentions(v: &Value, keys: &[&str]) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| keys.contains(&k.as_str()) || mentions(x, keys)),
        Value::Array(a) => a.iter().any(|x| mentions(x, keys)),
        _ => false,
    }
}

/// Search body preprocessing: `rule` queries (query rules).
/// Returns `None` when the body needs no change (the common case).
pub fn preprocess_search_body(node: &Node, body: &Value) -> Result<Option<Value>> {
    if !mentions(body, &["rule", "rule_query"]) {
        return Ok(None);
    }
    let mut out = body.clone();
    if let Some(q) = body.get("query") {
        out["query"] = rewrite_query(node, q)?;
    }
    if let Some(r) = body.get("retriever") {
        out["retriever"] = crate::retriever::resolve_rules(node, r)?;
    }
    if &out == body {
        return Ok(None);
    }
    Ok(Some(out))
}
