//! Snapshot lifecycle management (`_slm`): policy model and validation,
//! schedules (cron + interval), snapshot name generation, retention, and the
//! REST shapes of `GET _slm/policy`, `_slm/stats`, `_slm/status`.
//!
//! Scheduling itself (timers, executing snapshots, persisting policy
//! metadata in the cluster state) belongs to the node crate; this module
//! provides the pure logic.

pub mod cron;
pub mod datemath;
pub mod retention;

pub use cron::Cron;
pub use retention::RetentionConfig;

use crate::error::{Result, SnapshotError};
use crate::service::CreateSnapshotParams;
use crate::util::{format_duration_millis, format_millis, parse_time_value, random_base64_uuid};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// ES default for `slm.minimum_interval`.
pub const DEFAULT_MINIMUM_INTERVAL_MILLIS: i64 = 15 * 60 * 1000;

#[derive(Debug, Clone, PartialEq)]
pub enum Schedule {
    Cron(Cron),
    /// Interval schedule (`30m`, 8.15+): fires every N ms from the policy's modification time.
    Interval(i64),
}

impl Schedule {
    /// Parse a cron expression or a time value.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let s = s.trim();
        if s.split_whitespace().count() >= 6 {
            return Cron::parse(s).map(Schedule::Cron).map_err(|e| e.0);
        }
        match parse_time_value("schedule", s) {
            Ok(ms) if ms > 0 => Ok(Schedule::Interval(ms)),
            Ok(_) => Err(format!("invalid schedule [{s}]: interval must be greater than 0")),
            Err(_) => Err(format!("invalid schedule [{s}]: must be a valid cron expression or time unit")),
        }
    }

    /// Next execution strictly after `now` (`modified_date` anchors interval schedules).
    pub fn next_after(&self, modified_date: i64, now: i64) -> Option<i64> {
        match self {
            Schedule::Cron(c) => c.next_after(now),
            Schedule::Interval(iv) => {
                if now < modified_date {
                    return Some(modified_date + iv);
                }
                let k = (now - modified_date) / iv + 1;
                Some(modified_date + k * iv)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotLifecyclePolicy {
    pub id: String,
    pub name: String,
    pub schedule: String,
    pub repository: String,
    pub config: Option<Map<String, Value>>,
    pub retention: Option<RetentionConfig>,
}

const INVALID_FILENAME_CHARS: &str = "[ , \", *, \\, <, |, ,, >, /, ?]";

fn valid_file_name(s: &str) -> bool {
    !s.chars().any(|c| matches!(c, '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' | ','))
}

impl SnapshotLifecyclePolicy {
    /// Parse a `PUT _slm/policy/{id}` body.
    pub fn from_json(id: &str, body: &Value) -> Result<Self> {
        let obj = body.as_object().ok_or_else(|| SnapshotError::illegal_argument("request body is required"))?;
        let mut p = Self {
            id: id.to_string(),
            name: String::new(),
            schedule: String::new(),
            repository: String::new(),
            config: None,
            retention: None,
        };
        for (k, v) in obj {
            let s = || v.as_str().map(|s| s.to_string()).ok_or_else(|| SnapshotError::illegal_argument(format!("[{k}] must be a string")));
            match k.as_str() {
                "name" => p.name = s()?,
                "schedule" => p.schedule = s()?,
                "repository" => p.repository = s()?,
                "config" => {
                    p.config = Some(v.as_object().cloned().ok_or_else(|| SnapshotError::illegal_argument("[config] must be an object"))?)
                }
                "retention" => p.retention = Some(RetentionConfig::from_json(v)?),
                other => {
                    return Err(SnapshotError::new(
                        "x_content_parse_exception",
                        400,
                        format!("[snapshot_lifecycle] unknown field [{other}]"),
                    ));
                }
            }
        }
        for (field, v) in [("name", &p.name), ("schedule", &p.schedule), ("repository", &p.repository)] {
            if v.is_empty() && !obj.contains_key(field) {
                return Err(SnapshotError::new(
                    "x_content_parse_exception",
                    400,
                    format!("[snapshot_lifecycle] failed to parse object: Required [{field}]"),
                ));
            }
        }
        Ok(p)
    }

    pub fn parsed_schedule(&self) -> Result<Schedule> {
        Schedule::parse(&self.schedule).map_err(SnapshotError::illegal_argument)
    }

    /// ES `SnapshotLifecyclePolicy.validate` + `slm.minimum_interval`.
    /// Returns an `action_request_validation_exception` listing every problem.
    pub fn validate(&self, minimum_interval_millis: Option<i64>) -> Result<()> {
        let mut errs = Vec::new();
        let id = &self.id;
        if !valid_file_name(id) {
            errs.push(format!("invalid policy id [{id}]: must not contain the following characters {INVALID_FILENAME_CHARS}"));
        }
        if id.starts_with('_') {
            errs.push(format!("invalid policy id [{id}]: must not start with '_'"));
        }
        if id.len() > 255 {
            errs.push(format!("invalid policy id [{id}]: name is too long, ({} > 255)", id.len()));
        }
        if id.contains('#') {
            errs.push(format!("invalid policy id [{id}]: must not contain '#'"));
        }
        if id.to_lowercase() != *id {
            errs.push(format!("invalid policy id [{id}]: must be lowercase"));
        }
        // Snapshot name: validate the resolved (date-math) name.
        match datemath::resolve(&self.name, crate::util::now_millis()) {
            Ok(resolved) => {
                let n = &self.name;
                if resolved.is_empty() {
                    errs.push(format!("invalid snapshot name [{n}]: cannot be empty"));
                }
                if resolved.contains('#') {
                    errs.push(format!("invalid snapshot name [{n}]: must not contain '#'"));
                }
                if resolved.starts_with('_') {
                    errs.push(format!("invalid snapshot name [{n}]: must not start with '_'"));
                }
                if resolved.to_lowercase() != resolved {
                    errs.push(format!("invalid snapshot name [{n}]: must be lowercase"));
                }
                if !valid_file_name(&resolved) {
                    errs.push(format!(
                        "invalid snapshot name [{n}]: must not contain contain the following characters {INVALID_FILENAME_CHARS}"
                    ));
                }
            }
            Err(e) => errs.push(format!("invalid snapshot name [{}]: {}", self.name, e.reason)),
        }
        if self.schedule.trim().is_empty() {
            errs.push(format!("invalid schedule [{}]: must not be empty", self.schedule));
        } else {
            match Schedule::parse(&self.schedule) {
                Err(e) if e.starts_with("invalid schedule") => errs.push(e),
                Err(e) => errs.push(format!("invalid schedule: {e}")),
                Ok(s) => {
                    if let Some(min) = minimum_interval_millis.filter(|m| *m > 0) {
                        let now = crate::util::now_millis();
                        let too_frequent = match &s {
                            Schedule::Interval(iv) => *iv < min,
                            Schedule::Cron(c) => match c.next_after(now).and_then(|a| c.next_after(a).map(|b| (a, b))) {
                                Some((a, b)) => b - a < min,
                                None => false,
                            },
                        };
                        if too_frequent {
                            errs.push(format!(
                                "invalid schedule [{}]: schedule would be too frequent, executing more than every [{}]",
                                self.schedule,
                                format_duration_millis(min)
                            ));
                        }
                    }
                }
            }
        }
        if self.repository.trim().is_empty() {
            errs.push(format!("invalid repository name [{}]: cannot be empty", self.repository));
        }
        if let Some(cfg) = &self.config {
            if let Some(md) = cfg.get("metadata") {
                match md {
                    Value::Object(m) => {
                        if m.contains_key("policy") {
                            errs.push("invalid configuration.metadata: [policy] is a reserved field and may not be used".into());
                        }
                        let size = serde_json::to_vec(m).map(|b| b.len()).unwrap_or(0);
                        if size > 1004 {
                            errs.push(format!(
                                "invalid configuration.metadata: must be smaller than [1004] bytes, but is [{size}] bytes"
                            ));
                        }
                    }
                    _ => errs.push("invalid configuration.metadata: must be an object if present".into()),
                }
            }
        }
        if errs.is_empty() { Ok(()) } else { Err(SnapshotError::validation(&errs)) }
    }

    /// Next execution time (millis) after `now`.
    pub fn next_execution_millis(&self, modified_date: i64, now: i64) -> Option<i64> {
        Schedule::parse(&self.schedule).ok()?.next_after(modified_date, now)
    }

    /// `<date-math name>-<lowercase random suffix>` (ES `generateSnapshotName`).
    pub fn generate_snapshot_name(&self, now: i64) -> Result<String> {
        let base = datemath::resolve(&self.name, now)?;
        Ok(format!("{base}-{}", random_base64_uuid().to_lowercase()))
    }

    /// Create-snapshot parameters from `config`, with `metadata.policy = id`.
    pub fn snapshot_params(&self) -> Result<CreateSnapshotParams> {
        let cfg = self.config.clone().map(Value::Object);
        let mut p = CreateSnapshotParams::from_json(cfg.as_ref())?;
        let mut md = p.metadata.take().unwrap_or_default();
        md.insert("policy".into(), json!(self.id));
        p.metadata = Some(md);
        Ok(p)
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("name".into(), json!(self.name));
        m.insert("schedule".into(), json!(self.schedule));
        m.insert("repository".into(), json!(self.repository));
        if let Some(c) = &self.config {
            m.insert("config".into(), Value::Object(c.clone()));
        }
        if let Some(r) = &self.retention {
            m.insert("retention".into(), r.to_json());
        }
        Value::Object(m)
    }
}

/// `last_success` / `last_failure` (ES `SnapshotInvocationRecord`).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotInvocationRecord {
    pub snapshot_name: String,
    pub start_time: Option<i64>,
    pub time: i64,
    pub details: Option<String>,
}

impl SnapshotInvocationRecord {
    pub fn to_json(&self, human: bool) -> Value {
        let mut m = Map::new();
        m.insert("snapshot_name".into(), json!(self.snapshot_name));
        if let Some(st) = self.start_time {
            if human {
                m.insert("start_time_string".into(), json!(format_millis(st)));
            }
            m.insert("start_time".into(), json!(st));
        }
        if human {
            m.insert("time_string".into(), json!(format_millis(self.time)));
        }
        m.insert("time".into(), json!(self.time));
        if let Some(d) = &self.details {
            m.insert("details".into(), json!(d));
        }
        Value::Object(m)
    }
}

/// Per-policy counters (ES `SnapshotPolicyStats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PolicyStats {
    pub snapshots_taken: u64,
    pub snapshots_failed: u64,
    pub snapshots_deleted: u64,
    pub snapshot_deletion_failures: u64,
}

impl PolicyStats {
    pub fn to_json(&self, policy: &str) -> Value {
        json!({
            "policy": policy,
            "snapshots_taken": self.snapshots_taken,
            "snapshots_failed": self.snapshots_failed,
            "snapshots_deleted": self.snapshots_deleted,
            "snapshot_deletion_failures": self.snapshot_deletion_failures,
        })
    }
}

/// Global SLM stats (`GET _slm/stats`, ES `SnapshotLifecycleStats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SlmStats {
    pub retention_runs: u64,
    pub retention_failed: u64,
    pub retention_timed_out: u64,
    pub retention_deletion_time_millis: i64,
    pub policies: BTreeMap<String, PolicyStats>,
}

impl SlmStats {
    pub fn policy(&mut self, id: &str) -> &mut PolicyStats {
        self.policies.entry(id.to_string()).or_default()
    }

    pub fn snapshot_taken(&mut self, id: &str) {
        self.policy(id).snapshots_taken += 1;
    }

    pub fn snapshot_failed(&mut self, id: &str) {
        self.policy(id).snapshots_failed += 1;
    }

    pub fn snapshot_deleted(&mut self, id: &str) {
        self.policy(id).snapshots_deleted += 1;
    }

    pub fn snapshot_deletion_failure(&mut self, id: &str) {
        self.policy(id).snapshot_deletion_failures += 1;
    }

    pub fn retention_run(&mut self, took_millis: i64, failed: bool, timed_out: bool) {
        self.retention_runs += 1;
        self.retention_deletion_time_millis += took_millis;
        if failed {
            self.retention_failed += 1;
        }
        if timed_out {
            self.retention_timed_out += 1;
        }
    }

    pub fn to_json(&self) -> Value {
        let sum = |f: fn(&PolicyStats) -> u64| self.policies.values().map(f).sum::<u64>();
        json!({
            "retention_runs": self.retention_runs,
            "retention_failed": self.retention_failed,
            "retention_timed_out": self.retention_timed_out,
            "retention_deletion_time": format_duration_millis(self.retention_deletion_time_millis),
            "retention_deletion_time_millis": self.retention_deletion_time_millis,
            "total_snapshots_taken": sum(|p| p.snapshots_taken),
            "total_snapshots_failed": sum(|p| p.snapshots_failed),
            "total_snapshots_deleted": sum(|p| p.snapshots_deleted),
            "total_snapshot_deletion_failures": sum(|p| p.snapshot_deletion_failures),
            "policy_stats": self.policies.iter().map(|(id, p)| p.to_json(id)).collect::<Vec<_>>(),
        })
    }
}

/// SLM operation mode (`GET _slm/status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperationMode {
    #[default]
    Running,
    Stopping,
    Stopped,
}

impl OperationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "RUNNING",
            Self::Stopping => "STOPPING",
            Self::Stopped => "STOPPED",
        }
    }

    pub fn status_json(self) -> Value {
        json!({"operation_mode": self.as_str()})
    }
}

/// A running SLM snapshot (`in_progress` in `GET _slm/policy`).
#[derive(Debug, Clone, PartialEq)]
pub struct SlmInProgress {
    pub name: String,
    pub uuid: String,
    pub state: String,
    pub start_time: i64,
    pub failure: Option<String>,
}

/// Policy as stored in cluster state (ES `SnapshotLifecyclePolicyMetadata`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyMetadata {
    pub policy: SnapshotLifecyclePolicy,
    pub version: u64,
    pub modified_date: i64,
    pub last_success: Option<SnapshotInvocationRecord>,
    pub last_failure: Option<SnapshotInvocationRecord>,
    pub invocations_since_last_success: u64,
}

impl PolicyMetadata {
    pub fn new(policy: SnapshotLifecyclePolicy, now: i64, previous: Option<&PolicyMetadata>) -> Self {
        Self {
            policy,
            version: previous.map(|p| p.version + 1).unwrap_or(1),
            modified_date: now,
            last_success: previous.and_then(|p| p.last_success.clone()),
            last_failure: previous.and_then(|p| p.last_failure.clone()),
            invocations_since_last_success: previous.map(|p| p.invocations_since_last_success).unwrap_or(0),
        }
    }

    pub fn next_execution_millis(&self, now: i64) -> Option<i64> {
        self.policy.next_execution_millis(self.modified_date, now)
    }

    /// One entry of `GET _slm/policy` (`{id: {...}}` is built by the caller or [`policies_json`]).
    pub fn to_json(&self, stats: &PolicyStats, in_progress: Option<&SlmInProgress>, now: i64, human: bool) -> Value {
        let mut m = Map::new();
        m.insert("version".into(), json!(self.version));
        if human {
            m.insert("modified_date".into(), json!(format_millis(self.modified_date)));
        }
        m.insert("modified_date_millis".into(), json!(self.modified_date));
        m.insert("policy".into(), self.policy.to_json());
        if let Some(s) = &self.last_success {
            m.insert("last_success".into(), s.to_json(human));
        }
        if let Some(f) = &self.last_failure {
            m.insert("last_failure".into(), f.to_json(human));
        }
        let next = self.next_execution_millis(now).unwrap_or(-1);
        if human {
            m.insert("next_execution".into(), json!(format_millis(next)));
        }
        m.insert("next_execution_millis".into(), json!(next));
        if let Some(ip) = in_progress {
            let mut p = Map::new();
            p.insert("name".into(), json!(ip.name));
            p.insert("uuid".into(), json!(ip.uuid));
            p.insert("state".into(), json!(ip.state));
            if human {
                p.insert("start_time".into(), json!(format_millis(ip.start_time)));
            }
            p.insert("start_time_millis".into(), json!(ip.start_time));
            if let Some(f) = &ip.failure {
                p.insert("failure".into(), json!(f));
            }
            m.insert("in_progress".into(), Value::Object(p));
        }
        m.insert("stats".into(), stats.to_json(&self.policy.id));
        Value::Object(m)
    }

    /// Record a successful run.
    pub fn record_success(&mut self, snapshot_name: &str, start: i64, now: i64) {
        self.last_success = Some(SnapshotInvocationRecord {
            snapshot_name: snapshot_name.into(),
            start_time: Some(start),
            time: now,
            details: None,
        });
        self.invocations_since_last_success = 0;
    }

    /// Record a failed run with the ES error JSON (as a string) in `details`.
    pub fn record_failure(&mut self, snapshot_name: &str, start: i64, now: i64, details: &str) {
        self.last_failure = Some(SnapshotInvocationRecord {
            snapshot_name: snapshot_name.into(),
            start_time: Some(start),
            time: now,
            details: Some(details.into()),
        });
        self.invocations_since_last_success += 1;
    }
}

/// `GET _slm/policy[/{ids}]` body. Missing ids → `resource_not_found_exception`.
pub fn policies_json(
    policies: &BTreeMap<String, PolicyMetadata>,
    ids: &[String],
    stats: &SlmStats,
    in_progress: &BTreeMap<String, SlmInProgress>,
    now: i64,
    human: bool,
) -> Result<Value> {
    let available: Vec<String> = policies.keys().cloned().collect();
    let (selected, missing) = crate::util::resolve_names(ids, &available);
    if !missing.is_empty() {
        return Err(SnapshotError::resource_not_found(format!(
            "snapshot lifecycle policy or policies {missing:?} not found, no policies are configured"
        )
        .replace('"', "")));
    }
    if selected.is_empty() && !ids.is_empty() && !ids.iter().all(|i| i == "_all" || i == "*") {
        return Err(SnapshotError::resource_not_found(format!("snapshot lifecycle policy or policies {ids:?} not found").replace('"', "")));
    }
    let mut out = Map::new();
    let empty = PolicyStats::default();
    for id in selected {
        let pm = &policies[&id];
        out.insert(id.clone(), pm.to_json(stats.policies.get(&id).unwrap_or(&empty), in_progress.get(&id), now, human));
    }
    Ok(Value::Object(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(s: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis()
    }

    fn policy() -> SnapshotLifecyclePolicy {
        SnapshotLifecyclePolicy::from_json(
            "daily-snapshots",
            &json!({
                "schedule": "0 30 1 * * ?",
                "name": "<daily-snap-{now/d}>",
                "repository": "my_repository",
                "config": {"indices": ["data-*", "important"], "ignore_unavailable": false, "include_global_state": false},
                "retention": {"expire_after": "30d", "min_count": 5, "max_count": 50}
            }),
        )
        .unwrap()
    }

    #[test]
    fn parse_validate_render() {
        let p = policy();
        p.validate(Some(DEFAULT_MINIMUM_INTERVAL_MILLIS)).unwrap();
        assert_eq!(p.next_execution_millis(0, ms("2019-04-23T01:30:00Z")), Some(ms("2019-04-24T01:30:00Z")));
        let name = p.generate_snapshot_name(ms("2019-04-23T01:30:00Z")).unwrap();
        assert!(name.starts_with("daily-snap-2019.04.23-"), "{name}");
        assert_eq!(name.len(), "daily-snap-2019.04.23-".len() + 22);
        let params = p.snapshot_params().unwrap();
        assert_eq!(params.indices, vec!["data-*", "important"]);
        assert_eq!(params.metadata.unwrap()["policy"], "daily-snapshots");
        assert_eq!(
            p.to_json()["retention"],
            json!({"expire_after":"30d","min_count":5,"max_count":50})
        );

        let pm = PolicyMetadata::new(p, ms("2019-04-23T01:00:00Z"), None);
        let mut stats = SlmStats::default();
        stats.snapshot_taken("daily-snapshots");
        let mut policies = BTreeMap::new();
        policies.insert("daily-snapshots".to_string(), pm);
        let v = policies_json(&policies, &[], &stats, &BTreeMap::new(), ms("2019-04-23T01:00:00Z"), false).unwrap();
        let e = &v["daily-snapshots"];
        assert_eq!(e["version"], 1);
        assert_eq!(e["next_execution_millis"], ms("2019-04-23T01:30:00Z"));
        assert_eq!(e["stats"]["snapshots_taken"], 1);
        assert!(e.get("next_execution").is_none());
        let keys: Vec<&String> = e.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["version", "modified_date_millis", "policy", "next_execution_millis", "stats"]);
        let v = policies_json(&policies, &[], &stats, &BTreeMap::new(), 0, true).unwrap();
        assert_eq!(v["daily-snapshots"]["modified_date"], "2019-04-23T01:00:00.000Z");
        assert!(policies_json(&policies, &["nope".into()], &stats, &BTreeMap::new(), 0, false).is_err());
    }

    #[test]
    fn validation_errors() {
        let mut p = policy();
        p.id = "_Bad".into();
        p.schedule = "nope".into();
        p.repository = String::new();
        p.name = "Snap".into();
        let e = p.validate(None).unwrap_err();
        assert_eq!(e.kind, "action_request_validation_exception");
        assert!(e.reason.contains("invalid policy id [_Bad]: must not start with '_'"), "{}", e.reason);
        assert!(e.reason.contains("invalid snapshot name [Snap]: must be lowercase"));
        assert!(e.reason.contains("invalid schedule [nope]: must be a valid cron expression or time unit"));
        assert!(e.reason.contains("invalid repository name []: cannot be empty"));

        let mut p = policy();
        p.schedule = "0 * * * * ?".into();
        let e = p.validate(Some(DEFAULT_MINIMUM_INTERVAL_MILLIS)).unwrap_err();
        assert!(e.reason.contains("schedule would be too frequent, executing more than every [15m]"), "{}", e.reason);
        p.schedule = "0 0 0 * * *".into();
        let e = p.validate(None).unwrap_err();
        assert!(e.reason.contains("invalid schedule: invalid cron expression [0 0 0 * * *]"), "{}", e.reason);
    }

    #[test]
    fn interval_schedule() {
        let s = Schedule::parse("30m").unwrap();
        let base = ms("2024-01-01T00:00:00Z");
        assert_eq!(s.next_after(base, base), Some(base + 1_800_000));
        assert_eq!(s.next_after(base, base + 1_800_000), Some(base + 3_600_000));
        assert_eq!(s.next_after(base, base + 2_000_000), Some(base + 3_600_000));
        assert!(Schedule::parse("0s").is_err());
        let mut p = policy();
        p.schedule = "5m".into();
        assert!(p.validate(Some(DEFAULT_MINIMUM_INTERVAL_MILLIS)).is_err());
        p.schedule = "1h".into();
        p.validate(Some(DEFAULT_MINIMUM_INTERVAL_MILLIS)).unwrap();
    }

    #[test]
    fn stats_and_status_shapes() {
        let mut s = SlmStats::default();
        s.snapshot_taken("a");
        s.snapshot_failed("a");
        s.snapshot_deleted("b");
        s.retention_run(1500, false, false);
        let v = s.to_json();
        assert_eq!(v["retention_runs"], 1);
        assert_eq!(v["retention_deletion_time"], "1.5s");
        assert_eq!(v["total_snapshots_taken"], 1);
        assert_eq!(v["total_snapshots_deleted"], 1);
        assert_eq!(v["policy_stats"][0], json!({"policy":"a","snapshots_taken":1,"snapshots_failed":1,"snapshots_deleted":0,"snapshot_deletion_failures":0}));
        assert_eq!(OperationMode::Running.status_json(), json!({"operation_mode":"RUNNING"}));
    }
}
