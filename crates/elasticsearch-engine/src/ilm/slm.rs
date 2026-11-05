//! Snapshot lifecycle management: policies, stats and status.
//!
//! Snapshot repositories are not part of this node, so executing a policy
//! (`_execute` or its schedule) records the failure ES records for a
//! missing repository (`repository_missing_exception`) in the policy's
//! `last_failure` and stats, exactly like an ES cluster whose repository
//! was removed.

use super::{iso_date, parse_time, RUNNING, STOPPED};
use crate::error::{EsError, Result};
use crate::node::Node;
use chrono::{Datelike, NaiveDate, Timelike};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PolicyStats {
    pub snapshots_taken: u64,
    pub snapshots_failed: u64,
    pub snapshots_deleted: u64,
    pub snapshot_deletion_failures: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SlmStats {
    pub retention_runs: u64,
    pub retention_failed: u64,
    pub retention_timed_out: u64,
    pub retention_deletion_time_millis: u64,
    pub total_snapshots_taken: u64,
    pub total_snapshots_failed: u64,
    pub total_snapshots_deleted: u64,
    pub total_snapshot_deletion_failures: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlmPolicyMeta {
    pub policy: Value,
    pub version: u64,
    pub modified_date: i64,
    #[serde(default)]
    pub last_success: Option<Value>,
    #[serde(default)]
    pub last_failure: Option<Value>,
    #[serde(default)]
    pub stats: PolicyStats,
    #[serde(default)]
    pub next_execution: Option<i64>,
}

// ---------------------------------------------------------------------------
// Quartz-style cron (`sec min hour dom month dow [year]`)
// ---------------------------------------------------------------------------

pub struct Cron {
    sec: Vec<u32>,
    min: Vec<u32>,
    hour: Vec<u32>,
    dom: Option<Vec<u32>>,
    dom_last: bool,
    month: Vec<u32>,
    /// Quartz day of week: 1 = SUN .. 7 = SAT.
    dow: Option<Vec<u32>>,
    year: Option<Vec<u32>>,
}

fn parse_field(s: &str, lo: u32, hi: u32, names: &[&str], what: &str) -> std::result::Result<Vec<u32>, String> {
    let val = |t: &str| -> std::result::Result<u32, String> {
        let up = t.to_ascii_uppercase();
        if let Some(i) = names.iter().position(|n| *n == up) {
            return Ok(lo + i as u32);
        }
        let v: u32 = t.parse().map_err(|_| format!("invalid {what} value [{t}]"))?;
        if v < lo || v > hi {
            return Err(format!("invalid {what} value [{t}], must be between [{lo}] and [{hi}]"));
        }
        Ok(v)
    };
    let mut out = Vec::new();
    for part in s.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, st)) => (r, st.parse::<u32>().map_err(|_| format!("invalid {what} increment [{st}]"))?),
            None => (part, 1),
        };
        if step == 0 {
            return Err(format!("invalid {what} increment [0]"));
        }
        let (a, b) = if range == "*" || range == "?" {
            (lo, hi)
        } else if let Some((x, y)) = range.split_once('-') {
            (val(x)?, val(y)?)
        } else {
            let v = val(range)?;
            (v, if part.contains('/') { hi } else { v })
        };
        let mut v = a;
        while v <= b {
            out.push(v);
            v += step;
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

impl Cron {
    pub fn parse(expr: &str) -> std::result::Result<Cron, String> {
        let f: Vec<&str> = expr.split_whitespace().collect();
        if f.len() < 6 || f.len() > 7 {
            return Err(format!("invalid cron expression [{expr}]"));
        }
        const MONTHS: [&str; 12] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
        const DAYS: [&str; 7] = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
        let (dom, dom_last) = match f[3] {
            "?" => (None, false),
            "L" => (None, true),
            s => (Some(parse_field(s, 1, 31, &[], "day of month")?), false),
        };
        let dow = match f[5] {
            "?" => None,
            s => Some(parse_field(s, 1, 7, &DAYS, "day of week")?),
        };
        if dom.is_some() && dow.is_some() && f[3] != "*" && f[5] != "*" {
            return Err("support for specifying both a day-of-week AND a day-of-month parameter is not implemented.".into());
        }
        Ok(Cron {
            sec: parse_field(f[0], 0, 59, &[], "seconds")?,
            min: parse_field(f[1], 0, 59, &[], "minutes")?,
            hour: parse_field(f[2], 0, 23, &[], "hours")?,
            dom: if f[3] == "*" && dow.is_some() && f[5] != "*" { None } else { dom },
            dom_last,
            month: parse_field(f[4], 1, 12, &MONTHS, "month")?,
            dow: if f[5] == "*" { None } else { dow },
            year: match f.get(6) {
                Some(y) => Some(parse_field(y, 1970, 2199, &[], "year")?),
                None => None,
            },
        })
    }

    fn day_matches(&self, d: NaiveDate) -> bool {
        if !self.month.contains(&d.month()) {
            return false;
        }
        if let Some(y) = &self.year {
            if !y.contains(&(d.year() as u32)) {
                return false;
            }
        }
        if self.dom_last {
            let next = d.succ_opt();
            if next.map(|n| n.month() == d.month()).unwrap_or(false) {
                return false;
            }
        }
        if let Some(dom) = &self.dom {
            if !dom.contains(&d.day()) {
                return false;
            }
        }
        if let Some(dow) = &self.dow {
            let q = d.weekday().num_days_from_sunday() + 1;
            if !dow.contains(&q) {
                return false;
            }
        }
        true
    }

    /// First fire time strictly after `after` (epoch millis, UTC).
    pub fn next_after(&self, after: i64) -> Option<i64> {
        let start = crate::date::from_millis(after.div_euclid(1000) * 1000 + 1000);
        let mut day = start.date();
        let first_tod = (start.hour(), start.minute(), start.second());
        for i in 0..(366 * 30) {
            if self.day_matches(day) {
                for h in &self.hour {
                    for m in &self.min {
                        for s in &self.sec {
                            if i == 0 && (*h, *m, *s) < first_tod {
                                continue;
                            }
                            let dt = day.and_hms_opt(*h, *m, *s)?;
                            return Some(crate::date::to_millis(dt));
                        }
                    }
                }
            }
            day = day.succ_opt()?;
        }
        None
    }
}

/// A policy schedule: a cron expression or a fixed interval (`1d`, `30m`).
pub enum Schedule {
    Cron(Cron),
    Interval(i64),
}

impl Schedule {
    pub fn parse(s: &str) -> Option<Schedule> {
        if let Ok(c) = Cron::parse(s) {
            return Some(Schedule::Cron(c));
        }
        let t = parse_time(s, "schedule").ok()?;
        (t.millis() > 0).then(|| Schedule::Interval(t.millis()))
    }

    /// Next fire time after `after`; intervals count from `start`
    /// (`SchedulerEngine.IntervalSchedule`).
    pub fn next_after(&self, after: i64, start: i64) -> Option<i64> {
        match self {
            Schedule::Cron(c) => c.next_after(after),
            Schedule::Interval(iv) => {
                if after < start {
                    Some(start)
                } else {
                    Some(start + ((after - start) / iv + 1) * iv)
                }
            }
        }
    }
}

/// Validate a policy body; returns the normalized policy.
/// (`SnapshotLifecyclePolicy.parse` + `validate`, then the service checks.)
fn parse_policy(id: &str, body: &Value, node: &Node, raw: &str) -> Result<Value> {
    use super::policy::{locate, set_cur, wrap, xparse, At, PathGuard};
    let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
    let s = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
    // constructor arguments: name, schedule, repository
    let missing: Vec<&str> = ["name", "schedule", "repository"].into_iter().filter(|k| !matches!(o.get(*k), Some(v) if !v.is_null())).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    // retention is built by its own ConstructingObjectParser (at the object's end)
    let mut retention_err = None;
    if let Some(r) = o.get("retention").filter(|r| !r.is_null()) {
        super::policy::with_source(raw, || {
            let _g = PathGuard::push("retention");
            let fail = |leaf: EsError| {
                set_cur(At::End, None);
                let _ = locate(At::End, None);
                wrap("snapshot_lifecycle", "retention", xparse("Failed to build [snapshot_retention] after last required field arrived").caused_by(leaf))
            };
            let ro = match r.as_object() {
                Some(m) => m,
                None => return,
            };
            let mut expire_err = None;
            if let Some(e) = ro.get("expire_after").and_then(Value::as_str) {
                if let Err(e) = super::policy::tv(e, "expire_after") {
                    expire_err = Some(e);
                }
            }
            let min = ro.get("min_count").and_then(Value::as_i64);
            let max = ro.get("max_count").and_then(Value::as_i64);
            if let Some(e) = expire_err {
                retention_err = Some(fail(e));
            } else if let Some(m) = min.filter(|m| *m < 1) {
                retention_err = Some(fail(EsError::illegal_argument(format!("minimum snapshot count must be at least 1, but was: {m}"))));
            } else if let Some(m) = max.filter(|m| *m < 1) {
                retention_err = Some(fail(EsError::illegal_argument(format!("maximum snapshot count must be at least 1, but was: {m}"))));
            } else if let (Some(a), Some(b)) = (min, max) {
                if a > b {
                    retention_err = Some(fail(EsError::illegal_argument(format!("minimum snapshot count {a} cannot be larger than maximum snapshot count {b}"))));
                }
            }
        });
    }
    if let Some(e) = retention_err {
        return Err(e);
    }
    let mut errs: Vec<String> = Vec::new();
    if id.is_empty() {
        errs.push(format!("invalid policy id [{id}]: cannot be empty"));
    } else if id.contains(',') {
        errs.push(format!("invalid policy id [{id}]: must not contain ','"));
    } else if id.contains(' ') {
        errs.push(format!("invalid policy id [{id}]: must not contain spaces"));
    } else if id.starts_with('_') {
        errs.push(format!("invalid policy id [{id}]: must not start with '_'"));
    }
    let name = s("name").unwrap_or_default();
    if name.is_empty() {
        errs.push("invalid snapshot name []: cannot be empty".into());
    } else if let Ok(resolved) = crate::node::resolve_date_math(&name) {
        if resolved != resolved.to_lowercase() {
            errs.push(format!("invalid snapshot name [{name}]: must be lowercase"));
        }
        if resolved.contains(['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#']) {
            errs.push(format!("invalid snapshot name [{name}]: must not contain contain the following characters [' ','\"','*',',','/','<','>','?','\\','|']"));
        }
    } else {
        errs.push(format!("invalid snapshot name [{name}]: must be a valid date math expression"));
    }
    let schedule = s("schedule").unwrap_or_default();
    let cron = match Schedule::parse(&schedule) {
        Some(c) => Some(c),
        None => {
            errs.push(format!("invalid schedule [{schedule}]: must be a valid cron expression or time unit"));
            None
        }
    };
    let repo = s("repository").unwrap_or_default();
    if repo.is_empty() {
        errs.push("invalid repository name []: cannot be empty".into());
    }
    if let Some(r) = o.get("retention").filter(|r| !r.is_null()) {
        let ro = r.as_object().ok_or_else(|| EsError::x_content_parse("[1:1] [snapshot_retention] expected an object"))?;
        if let Some(e) = ro.get("expire_after").and_then(Value::as_str) {
            parse_time(e, "expire_after")?;
        }
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    // SnapshotLifecycleService.validateRepositoryExists
    if !node.snapshots.repository_names().iter().any(|r| *r == repo) {
        return Err(EsError::illegal_argument(format!("no such repository [{repo}]")));
    }
    // SnapshotLifecycleService.validateMinimumInterval
    if let Some(c) = cron {
        let min_s = super::cluster_setting(node, "slm.minimum_interval").unwrap_or_else(|| "15m".into());
        let min_ms = parse_time(&min_s, "slm.minimum_interval").map(|t| t.millis()).unwrap_or(900_000);
        let now = crate::date::now_millis();
        if let Some(a) = c.next_after(now, now) {
            if let Some(b) = c.next_after(a, now) {
                if b - a < min_ms {
                    return Err(EsError::illegal_argument(format!("invalid schedule [{schedule}]: schedule would be too frequent, executing more than every [{min_s}]")));
                }
            }
        }
    }
    let mut out = Map::new();
    out.insert("name".into(), json!(name));
    out.insert("schedule".into(), json!(schedule));
    out.insert("repository".into(), json!(repo));
    if let Some(c) = o.get("config").filter(|c| !c.is_null()) {
        out.insert("config".into(), c.clone());
    }
    if let Some(r) = o.get("retention").filter(|r| !r.is_null()) {
        out.insert("retention".into(), r.clone());
    }
    Ok(Value::Object(out))
}

fn next_execution(policy: &Value, start: i64) -> Option<i64> {
    Schedule::parse(policy["schedule"].as_str()?)?.next_after(crate::date::now_millis(), start)
}

/// `PUT /_slm/policy/{id}`.
pub fn put_policy(node: &Node, id: &str, body: &Value, raw: &str) -> Result<()> {
    let p = parse_policy(id, body, node, raw)?;
    let now = crate::date::now_millis();
    {
        let mut st = node.ilm.st.write();
        match st.slm.get_mut(id) {
            Some(existing) if existing.policy == p => return Ok(()),
            Some(existing) => {
                existing.policy = p.clone();
                existing.version += 1;
                existing.modified_date = now;
                existing.next_execution = next_execution(&p, now);
            }
            None => {
                let next = next_execution(&p, now);
                st.slm.insert(
                    id.to_string(),
                    SlmPolicyMeta { policy: p, version: 1, modified_date: now, last_success: None, last_failure: None, stats: PolicyStats::default(), next_execution: next },
                );
            }
        }
    }
    node.ilm.persist()
}

fn policy_item(id: &str, m: &SlmPolicyMeta, human: bool) -> Value {
    let mut o = Map::new();
    o.insert("version".into(), json!(m.version));
    if human {
        o.insert("modified_date".into(), json!(iso_date(m.modified_date)));
    }
    o.insert("modified_date_millis".into(), json!(m.modified_date));
    o.insert("policy".into(), m.policy.clone());
    for (k, rec) in [("last_success", &m.last_success), ("last_failure", &m.last_failure)] {
        if let Some(r) = rec {
            let mut r = r.clone();
            if human {
                for f in ["start_time", "time"] {
                    if let Some(ms) = r[f].as_i64() {
                        r[format!("{f}_string")] = json!(iso_date(ms));
                    }
                }
            }
            o.insert(k.into(), r);
        }
    }
    if let Some(n) = m.next_execution.or_else(|| next_execution(&m.policy, m.modified_date)) {
        if human {
            o.insert("next_execution".into(), json!(iso_date(n)));
        }
        o.insert("next_execution_millis".into(), json!(n));
    }
    o.insert(
        "stats".into(),
        json!({"policy": id, "snapshots_taken": m.stats.snapshots_taken, "snapshots_failed": m.stats.snapshots_failed,
               "snapshots_deleted": m.stats.snapshots_deleted, "snapshot_deletion_failures": m.stats.snapshot_deletion_failures}),
    );
    Value::Object(o)
}

/// `GET /_slm/policy[/{ids}]`.
pub fn get_policies(node: &Node, ids: Option<&str>, human: bool) -> Result<Value> {
    let st = node.ilm.st.read();
    let mut out = Map::new();
    match ids.filter(|s| !s.is_empty()) {
        None => {
            for (k, m) in &st.slm {
                out.insert(k.clone(), policy_item(k, m, human));
            }
        }
        Some(list) => {
            // ids are literal (no wildcards / `_all`), like ES
            for id in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if let Some(m) = st.slm.get(id) {
                    out.insert(id.to_string(), policy_item(id, m, human));
                }
            }
            if out.is_empty() {
                return Err(EsError::resource_not_found(format!("snapshot lifecycle policy or policies {} not found{}", format_ids(list), if st.slm.is_empty() { ", no policies are configured" } else { "" })));
            }
        }
    }
    Ok(Value::Object(out))
}

fn format_ids(list: &str) -> String {
    format!("[{}]", list.split(',').map(str::trim).collect::<Vec<_>>().join(", "))
}

/// `DELETE /_slm/policy/{id}`.
pub fn delete_policy(node: &Node, id: &str) -> Result<()> {
    let removed = node.ilm.st.write().slm.remove(id);
    if removed.is_none() {
        return Err(EsError::resource_not_found(format!("snapshot lifecycle policy not found: {id}")));
    }
    node.ilm.persist()
}

fn snapshot_name(pattern: &str) -> String {
    let base = crate::node::resolve_date_math(pattern).unwrap_or_else(|_| pattern.to_string());
    format!("{base}-{}", crate::index::new_uuid().to_lowercase())
}

/// `PUT /_slm/policy/{id}/_execute` (and scheduled runs): takes a real
/// snapshot in the policy's repository. The REST call returns the snapshot
/// name; a failed snapshot (missing repository, ...) is recorded in the
/// policy's `last_failure` and stats like ES does.
pub fn execute(node: &Node, id: &str) -> Result<Value> {
    let policy = {
        let st = node.ilm.st.read();
        st.slm.get(id).map(|p| p.policy.clone())
    }
    .ok_or_else(|| EsError::illegal_argument(format!("no such snapshot lifecycle policy [{id}]")))?;
    let name = snapshot_name(policy["name"].as_str().unwrap_or("snap"));
    let repo = policy["repository"].as_str().unwrap_or("").to_string();
    let mut body = policy.get("config").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
    let mut metadata = body.get("metadata").and_then(Value::as_object).cloned().unwrap_or_default();
    metadata.insert("policy".into(), json!(id));
    body["metadata"] = Value::Object(metadata);
    let start = crate::date::now_millis();
    let res = node.snapshots.create_snapshot(node, &repo, &name, &body);
    let now = crate::date::now_millis();
    {
        let mut st = node.ilm.st.write();
        let mut ok = res.is_ok();
        let mut fail_details = None;
        match &res {
            Ok(meta) if meta["state"].as_str() != Some("SUCCESS") && meta["state"].as_str() != Some("PARTIAL") => {
                ok = false;
                fail_details = Some(json!({"type": "snapshot_exception", "reason": format!("snapshot finished in state [{}]", meta["state"].as_str().unwrap_or("FAILED"))}));
            }
            Err(e) => {
                fail_details = Some(json!({"type": e.kind, "reason": e.reason, "stack_trace": format!("{}: {}", e.kind, e.reason)}));
            }
            _ => {}
        }
        if let Some(p) = st.slm.get_mut(id) {
            if ok {
                p.last_success = Some(json!({"snapshot_name": name, "start_time": start, "time": now}));
                p.stats.snapshots_taken += 1;
            } else {
                p.last_failure = Some(json!({"snapshot_name": name, "start_time": start, "time": now, "details": fail_details.unwrap_or(Value::Null).to_string()}));
                p.stats.snapshots_failed += 1;
            }
            p.next_execution = next_execution(&p.policy, p.modified_date);
        }
        if ok {
            st.slm_stats.total_snapshots_taken += 1;
        } else {
            st.slm_stats.total_snapshots_failed += 1;
        }
    }
    node.ilm.persist()?;
    Ok(json!({"snapshot_name": name}))
}

/// `GET /_slm/stats`.
pub fn stats(node: &Node) -> Value {
    let st = node.ilm.st.read();
    let s = &st.slm_stats;
    let policy_stats: Vec<Value> = st
        .slm
        .iter()
        .map(|(k, m)| json!({"policy": k, "snapshots_taken": m.stats.snapshots_taken, "snapshots_failed": m.stats.snapshots_failed, "snapshots_deleted": m.stats.snapshots_deleted, "snapshot_deletion_failures": m.stats.snapshot_deletion_failures}))
        .collect();
    json!({
        "retention_runs": s.retention_runs,
        "retention_failed": s.retention_failed,
        "retention_timed_out": s.retention_timed_out,
        "retention_deletion_time": super::human_time(s.retention_deletion_time_millis as i64),
        "retention_deletion_time_millis": s.retention_deletion_time_millis,
        "total_snapshots_taken": s.total_snapshots_taken,
        "total_snapshots_failed": s.total_snapshots_failed,
        "total_snapshots_deleted": s.total_snapshots_deleted,
        "total_snapshot_deletion_failures": s.total_snapshot_deletion_failures,
        "policy_stats": policy_stats
    })
}

/// Snapshots of `policy_id` in `repo` that retention would delete now
/// (`SnapshotRetentionConfiguration.getSnapshotDeletionPredicate`).
fn expired_snapshots(node: &Node, policy_id: &str, policy: &Value, now: i64) -> Result<Vec<String>> {
    let ret = &policy["retention"];
    if !ret.is_object() {
        return Ok(Vec::new());
    }
    let repo = policy["repository"].as_str().unwrap_or("");
    let mut snaps: Vec<Value> = node.snapshots.get_snapshots(repo, "*", true)?.into_iter().filter(|m| m["metadata"]["policy"].as_str() == Some(policy_id)).collect();
    snaps.sort_by_key(|m| m["start_time_in_millis"].as_i64().unwrap_or(0));
    let expire = ret["expire_after"].as_str().and_then(|e| parse_time(e, "expire_after").ok()).map(|t| t.millis());
    let min_count = ret["min_count"].as_i64().unwrap_or(1).max(0) as usize;
    let max_count = ret["max_count"].as_i64().map(|m| m.max(0) as usize);
    let successful = snaps.iter().filter(|m| m["state"].as_str() == Some("SUCCESS")).count();
    let mut out = Vec::new();
    let mut seen_successful = 0usize;
    for m in &snaps {
        let ok = m["state"].as_str() == Some("SUCCESS");
        let started = m["start_time_in_millis"].as_i64().unwrap_or(0);
        let expired = expire.map(|e| now - started > e).unwrap_or(false);
        let mut del = false;
        if ok {
            // successful snapshots remaining if this and all older ones go
            let remaining = successful - seen_successful - 1;
            if expired && successful - seen_successful > min_count {
                del = true;
            }
            if let Some(max) = max_count {
                if successful - seen_successful > max && remaining + 1 > min_count {
                    del = true;
                }
            }
            seen_successful += 1;
        } else if expired && m["state"].as_str() != Some("IN_PROGRESS") {
            del = true;
        }
        if del {
            if let Some(n) = m["snapshot"].as_str() {
                out.push(n.to_string());
            }
        }
    }
    Ok(out)
}

/// One retention pass over every policy (`SnapshotRetentionTask`).
pub fn run_retention(node: &Node) {
    let start = std::time::Instant::now();
    let now = crate::date::now_millis();
    let policies: Vec<(String, Value)> = node.ilm.st.read().slm.iter().map(|(k, m)| (k.clone(), m.policy.clone())).collect();
    let (mut deleted, mut failed_del, mut failed_run) = (Vec::new(), Vec::new(), 0u64);
    for (id, pol) in &policies {
        let repo = pol["repository"].as_str().unwrap_or("").to_string();
        match expired_snapshots(node, id, pol, now) {
            Ok(names) => {
                for n in names {
                    match node.snapshots.delete_snapshots(&repo, &n) {
                        Ok(()) => deleted.push(id.clone()),
                        Err(_) => failed_del.push(id.clone()),
                    }
                }
            }
            Err(_) => failed_run += 1,
        }
    }
    {
        let mut st = node.ilm.st.write();
        st.slm_stats.retention_runs += 1;
        st.slm_stats.retention_failed += failed_run;
        st.slm_stats.retention_deletion_time_millis += start.elapsed().as_millis() as u64;
        st.slm_stats.total_snapshots_deleted += deleted.len() as u64;
        st.slm_stats.total_snapshot_deletion_failures += failed_del.len() as u64;
        for id in &deleted {
            if let Some(p) = st.slm.get_mut(id) {
                p.stats.snapshots_deleted += 1;
            }
        }
        for id in &failed_del {
            if let Some(p) = st.slm.get_mut(id) {
                p.stats.snapshot_deletion_failures += 1;
            }
        }
    }
    let _ = node.ilm.persist();
}

/// `POST /_slm/_execute_retention`.
pub fn execute_retention(node: &Node) -> Result<()> {
    run_retention(node);
    Ok(())
}

/// `_slm/start` → RUNNING; `_slm/stop` → STOPPING (the next tick settles it
/// to STOPPED, there being no snapshot to wait for).
pub fn set_mode(node: &Node, running: bool) -> Result<()> {
    {
        let mut st = node.ilm.st.write();
        st.slm_mode = if running {
            RUNNING.into()
        } else if st.slm_mode == STOPPED {
            STOPPED.into()
        } else {
            super::STOPPING.into()
        };
    }
    node.ilm.persist()
}

/// Scheduled executions (background thread).
pub fn tick(node: &Node) {
    if node.ilm.slm_mode() == super::STOPPING {
        node.ilm.st.write().slm_mode = STOPPED.into();
        let _ = node.ilm.persist();
        return;
    }
    if node.ilm.slm_mode() != RUNNING {
        return;
    }
    let now = crate::date::now_millis();
    let due: Vec<String> = node.ilm.st.read().slm.iter().filter(|(_, m)| m.next_execution.map(|n| n <= now).unwrap_or(false)).map(|(k, _)| k.clone()).collect();
    for id in due {
        let _ = execute(node, &id);
    }
    // `slm.retention_schedule` (default daily 01:30)
    static NEXT_RETENTION: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    let sched = super::cluster_setting(node, "slm.retention_schedule").unwrap_or_else(|| "0 30 1 * * ?".into());
    if let Ok(c) = Cron::parse(&sched) {
        use std::sync::atomic::Ordering::SeqCst;
        let next = NEXT_RETENTION.load(SeqCst);
        if next == 0 {
            NEXT_RETENTION.store(c.next_after(now).unwrap_or(i64::MAX), SeqCst);
        } else if next <= now {
            NEXT_RETENTION.store(c.next_after(now).unwrap_or(i64::MAX), SeqCst);
            run_retention(node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cron_next() {
        let c = Cron::parse("0 1 2 3 4 ?").unwrap();
        let t = c.next_after(0).unwrap();
        assert_eq!(iso_date(t), "1970-04-03T02:01:00.000Z");
        let every_min = Cron::parse("0 * * * * ?").unwrap();
        let a = every_min.next_after(0).unwrap();
        assert_eq!(every_min.next_after(a).unwrap() - a, 60_000);
        let c = Cron::parse("0 30 1 * * ?").unwrap();
        assert_eq!(iso_date(c.next_after(0).unwrap()), "1970-01-01T01:30:00.000Z");
        assert!(Cron::parse("bad").is_err());
    }

    #[test]
    fn interval_schedule() {
        let s = Schedule::parse("20m").unwrap();
        // counts from the policy start
        assert_eq!(s.next_after(5_000, 5_000), Some(5_000 + 1_200_000));
        assert_eq!(s.next_after(5_000 + 1_200_000, 5_000), Some(5_000 + 2_400_000));
        assert_eq!(s.next_after(0, 5_000), Some(5_000));
        assert!(matches!(Schedule::parse("0 0 1 * * ?"), Some(Schedule::Cron(_))));
        assert!(Schedule::parse("1.5h").is_none());
        assert!(Schedule::parse("bad").is_none());
    }

    #[test]
    fn cron_next_variants() {
        // 2021-03-04T05:06:07Z
        let t0 = 1_614_834_367_000i64;
        let c = Cron::parse("0 0/15 * * * ?").unwrap();
        assert_eq!(iso_date(c.next_after(t0).unwrap()), "2021-03-04T05:15:00.000Z");
        // strictly after: exact fire time moves to the next one
        let fire = c.next_after(t0).unwrap();
        assert_eq!(iso_date(c.next_after(fire).unwrap()), "2021-03-04T05:30:00.000Z");
        // weekday names (2021-03-04 is a Thursday)
        let c = Cron::parse("0 0 9 ? * MON-FRI").unwrap();
        assert_eq!(iso_date(c.next_after(t0).unwrap()), "2021-03-04T09:00:00.000Z");
        let c = Cron::parse("0 0 9 ? * SAT").unwrap();
        assert_eq!(iso_date(c.next_after(t0).unwrap()), "2021-03-06T09:00:00.000Z");
        // last day of month, with year field
        let c = Cron::parse("0 0 0 L * ? 2021").unwrap();
        assert_eq!(iso_date(c.next_after(t0).unwrap()), "2021-03-31T00:00:00.000Z");
        assert_eq!(c.next_after(1_640_995_200_000), None);
        // month names and ranges
        let c = Cron::parse("30 10 8 1 JAN,JUL ?").unwrap();
        assert_eq!(iso_date(c.next_after(t0).unwrap()), "2021-07-01T08:10:30.000Z");
        // errors
        assert!(Cron::parse("0 0 0 * *").is_err());
        assert!(Cron::parse("0 61 * * * ?").is_err());
        assert!(Cron::parse("0 0 0 1 * 2").is_err());
    }
}
