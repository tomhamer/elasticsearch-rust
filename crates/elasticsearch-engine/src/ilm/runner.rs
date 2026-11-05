//! ILM execution: per-index lifecycle state (`LifecycleExecutionState`,
//! persisted in `<index dir>/lifecycle.json` together with the index's
//! rollover info), the step runner (`IndexLifecycleRunner`), explain,
//! move-to-step, retry, remove-policy and the ILM history store.

use super::policy::{self, Step, StepKey, StepKind};
use super::{error_json, human_time, iso_date, PolicyMeta, RUNNING, STOPPED, STOPPING};
use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::node::{Node, ResolveOpts};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

const LIFECYCLE_FILE: &str = "lifecycle.json";
/// `WaitForRolloverReadyStep.MAX_PRIMARY_SHARD_DOCS`.
const MAX_PRIMARY_SHARD_DOCS: i64 = 200_000_000;
pub const HISTORY_DATA_STREAM: &str = "ilm-history-7";

/// `LifecycleExecutionState` (ES index metadata custom `ilm`).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_auto_retryable_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_step_retry_count: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_info: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_step_info: Option<Value>,
    /// `{"policy", "phase_definition"?, "version", "modified_date_in_millis"}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_definition: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_date: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_time: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_time: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_time: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_repository: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shrink_index_name: Option<String>,
}

impl ExecState {
    pub fn key(&self) -> Option<StepKey> {
        match (&self.phase, &self.action, &self.step) {
            (Some(p), Some(a), Some(s)) => Some(StepKey::new(p, a, s)),
            _ => None,
        }
    }
    pub fn is_empty(&self) -> bool {
        *self == ExecState::default()
    }
}

/// Everything lifecycle-related stored next to one index.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexLifecycle {
    #[serde(default)]
    pub ilm: ExecState,
    /// Rollover target (alias / data stream) → `{"met_conditions": {..}, "time": ms}`.
    #[serde(default)]
    pub rollover_info: BTreeMap<String, Value>,
}

pub fn load(node: &Node, svc: &IndexService) -> IndexLifecycle {
    if let Some(lc) = node.ilm.exec.lock().get(&svc.uuid) {
        return lc.clone();
    }
    let lc: IndexLifecycle = std::fs::read(svc.path().join(LIFECYCLE_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    node.ilm.exec.lock().insert(svc.uuid.clone(), lc.clone());
    lc
}

pub fn save(node: &Node, svc: &IndexService, lc: &IndexLifecycle) -> Result<()> {
    let old = node.ilm.exec.lock().insert(svc.uuid.clone(), lc.clone());
    if old.is_none_or(|o| o.ilm != lc.ilm || o.rollover_info != lc.rollover_info) {
        *node.ilm.exec_gen.lock().entry(svc.uuid.clone()).or_insert(0) += 1;
    }
    if !svc.path().exists() {
        return Ok(());
    }
    let tmp = svc.path().join(format!("{LIFECYCLE_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(lc)?)?;
    std::fs::rename(&tmp, svc.path().join(LIFECYCLE_FILE))?;
    Ok(())
}

/// Record a successful rollover of `old` (ES `IndexMetadata.rolloverInfos`).
pub fn record_rollover(node: &Node, old: &IndexService, target: &str, met_conditions: &Value) {
    let mut lc = load(node, old);
    lc.rollover_info.insert(target.to_string(), json!({"met_conditions": met_conditions, "time": crate::date::now_millis()}));
    let _ = save(node, old, &lc);
}

/// Rollover time of an index for any target (data stream lifecycle
/// generation time).
pub fn rollover_time(node: &Node, svc: &IndexService) -> Option<i64> {
    load(node, svc).rollover_info.values().filter_map(|v| v["time"].as_i64()).max()
}

pub fn policy_name(svc: &IndexService) -> Option<String> {
    svc.setting("index.lifecycle.name").filter(|s| !s.is_empty())
}

/// `Metadata.isIndexManagedByILM`: a policy is set and the data stream
/// lifecycle does not take precedence.
pub fn managed_by_ilm(node: &Node, svc: &IndexService) -> bool {
    policy_name(svc).is_some() && !super::dlm::managed_by_dlm(node, svc)
}

fn now() -> i64 {
    crate::date::now_millis()
}

fn origination_or_lifecycle_date(svc: &IndexService, st: &ExecState) -> Option<i64> {
    svc.setting("index.lifecycle.origination_date").and_then(|s| s.parse::<i64>().ok()).filter(|v| *v >= 0).or(st.lifecycle_date)
}

fn phase_def_value(pm: &PolicyMeta, policy_name: &str, phase: &str) -> Value {
    let mut v = json!({"policy": policy_name});
    if let Some(p) = pm.policy["phases"].get(phase) {
        v["phase_definition"] = p.clone();
    }
    v["version"] = json!(pm.version);
    v["modified_date_in_millis"] = json!(pm.modified_date);
    v
}

/// `IndexLifecycleTransition.updateExecutionStateToStep`.
fn move_to(st: &mut ExecState, pm: &PolicyMeta, policy_name: &str, next: &StepKey, refresh_phase: bool, keep_step_info: bool) {
    let t = now();
    let phase_changed = st.phase.as_deref() != Some(next.phase.as_str());
    let action_changed = st.action.as_deref() != Some(next.action.as_str()) || phase_changed;
    if keep_step_info {
        st.previous_step_info = st.step_info.take();
    } else {
        st.step_info = None;
    }
    st.failed_step = None;
    st.is_auto_retryable_error = None;
    st.failed_step_retry_count = None;
    if phase_changed || refresh_phase {
        st.phase_definition = Some(phase_def_value(pm, policy_name, &next.phase));
        st.phase_time = Some(t);
    }
    if action_changed {
        st.action_time = Some(t);
    }
    st.phase = Some(next.phase.clone());
    st.action = Some(next.action.clone());
    st.step = Some(next.name.clone());
    st.step_time = Some(t);
}

/// The phase definition the index executes (cached in its state), or the
/// policy's current one.
fn current_phase_def(st: &ExecState, pm: &PolicyMeta, phase: &str) -> Value {
    if let Some(pd) = &st.phase_definition {
        if st.phase.as_deref() == Some(phase) {
            if let Some(d) = pd.get("phase_definition") {
                return d.clone();
            }
        }
    }
    pm.policy["phases"].get(phase).cloned().unwrap_or(Value::Null)
}

fn steps_of(st: &ExecState, pm: &PolicyMeta, phase: &str) -> Vec<Step> {
    if phase == "new" {
        return vec![
            Step { key: StepKey::new("new", "init", "init"), kind: StepKind::Init, config: Value::Null },
            Step { key: StepKey::new("new", "complete", "complete"), kind: StepKind::PhaseComplete, config: Value::Null },
        ];
    }
    let def = current_phase_def(st, pm, phase);
    if def.is_null() {
        return Vec::new();
    }
    policy::phase_steps(phase, &def)
}

fn next_phase(pm: &PolicyMeta, phase: &str) -> Option<String> {
    let phases = policy::ordered_phases(&pm.policy);
    if phase == "new" {
        return phases.first().cloned();
    }
    let i = phases.iter().position(|p| p == phase)?;
    phases.get(i + 1).cloned()
}

fn first_step_of_phase(pm: &PolicyMeta, phase: &str) -> Option<StepKey> {
    let def = pm.policy["phases"].get(phase)?;
    policy::phase_steps(phase, def).first().map(|s| s.key.clone())
}

enum Outcome {
    Next,
    /// Condition not met yet (optional `step_info`).
    Wait(Option<Value>),
    /// Branch: skip the rest of the action.
    SkipAction,
    /// The index was deleted (or replaced) by the step.
    Gone,
}

fn retryable(step: &str) -> bool {
    matches!(
        step,
        "check-rollover-ready" | "attempt-rollover" | "readonly" | "forcemerge" | "segment-count" | "shrink" | "delete" | "set_priority" | "update-settings" | "allocate"
            | "migrate" | "set-indexing-complete" | "update-rollover-lifecycle-date" | "wait-for-snapshot" | "set-single-node-allocation" | "aliases"
            | "close-index" | "open-index" | "generate-index-name" | "copy-execution-state"
    )
}

fn update_settings(svc: &IndexService, kv: Vec<(&str, Value)>) -> Result<()> {
    let flat: crate::settings::FlatSettings = kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    svc.put_settings(flat, false)
}

fn rollover_target(node: &Node, svc: &IndexService) -> Result<(String, bool)> {
    if let Some(ds) = node.data_stream_of_index(&svc.name) {
        return Ok((ds, true));
    }
    match svc.setting("index.lifecycle.rollover_alias").filter(|s| !s.is_empty()) {
        Some(a) => Ok((a, false)),
        None => Err(EsError::illegal_argument(format!("setting [index.lifecycle.rollover_alias] for index [{}] is empty or not defined", svc.name))),
    }
}

fn is_ds_write_index(node: &Node, svc: &IndexService) -> Option<String> {
    let ds = node.data_stream_of_index(&svc.name)?;
    let d = node.data_stream(&ds)?;
    if d.write_index() == svc.name {
        Some(ds)
    } else {
        None
    }
}

fn indexing_complete(svc: &IndexService) -> bool {
    svc.setting("index.lifecycle.indexing_complete").as_deref() == Some("true")
}

/// Rollover conditions with ES's defaults (`applyDefaultConditions`).
pub fn rollover_conditions_with_defaults(node: &Node, cfg: &Value) -> Value {
    let mut c = cfg.as_object().cloned().unwrap_or_default();
    let only_if_docs = super::cluster_setting(node, "indices.lifecycle.rollover.only_if_has_documents").map(|s| s != "false").unwrap_or(true);
    if only_if_docs && !c.contains_key("min_docs") && !c.contains_key("min_primary_shard_docs") {
        c.insert("min_docs".into(), json!(1));
    }
    let cur = c.get("max_primary_shard_docs").and_then(Value::as_i64).unwrap_or(i64::MAX);
    c.insert("max_primary_shard_docs".into(), json!(cur.min(MAX_PRIMARY_SHARD_DOCS)));
    Value::Object(c)
}

fn conditions_met(resp: &Value) -> bool {
    let conds = match resp["conditions"].as_object() {
        Some(c) => c,
        None => return false,
    };
    if conds.is_empty() {
        return true;
    }
    let mut any_max = false;
    let mut max_met = false;
    for (k, v) in conds {
        let met = v.as_bool().unwrap_or(false);
        if k.starts_with("[min_") {
            if !met {
                return false;
            }
        } else {
            any_max = true;
            max_met |= met;
        }
    }
    !any_max || max_met
}

fn logical_shards(svc: &IndexService) -> i64 {
    svc.setting("index.number_of_shards").and_then(|s| s.parse().ok()).unwrap_or(1)
}

fn replicas_wait_info(svc: &IndexService) -> Option<Value> {
    let auto = svc.setting("index.auto_expand_replicas").unwrap_or_default();
    let replicas = svc.setting_usize("index.number_of_replicas", 1);
    if replicas > 0 && !auto.starts_with("0-") {
        Some(json!({"message": "Waiting for all shard copies to be active", "shards_left_to_allocate": -1, "all_shards_active": false, "number_of_replicas": replicas}))
    } else {
        None
    }
}

/// `new Date(millis).toString()` (Java, UTC): `Sat Oct 03 12:00:00 UTC 2026`.
fn java_date(millis: i64) -> String {
    crate::date::DateFormatter::new("EEE MMM dd HH:mm:ss 'UTC' yyyy").map(|f| f.format(millis, crate::date::Tz::UTC)).unwrap_or_else(|_| iso_date(millis))
}

fn progress(p: super::StepProgress) -> Result<Outcome> {
    match p {
        super::StepProgress::Done => Ok(Outcome::Next),
        super::StepProgress::Pending(info) => Ok(Outcome::Wait(info)),
        super::StepProgress::Failed(reason) => Err(EsError::new(500, "illegal_state_exception", reason)),
    }
}

/// Before a shrink's source is deleted: the target holds its documents.
fn check_shrunk_docs(node: &Node, source: &str, target: &str) -> Result<Option<Outcome>> {
    let Some(steps) = node.ilm.cluster_steps() else { return Ok(None) };
    match progress(steps.holds_all_docs(source, target))? {
        Outcome::Next => Ok(None),
        other => Ok(Some(other)),
    }
}

/// Execute one step. `periodic` = wait steps are evaluated.
fn exec_step(node: &Node, svc: &Arc<IndexService>, lc: &mut IndexLifecycle, pname: &str, step: &Step, periodic: bool) -> Result<Option<Outcome>> {
    let cfg = &step.config;
    let name = svc.name.clone();
    let key = &step.key;
    Ok(Some(match key.name.as_str() {
        "init" => {
            if lc.ilm.lifecycle_date.is_none() {
                lc.ilm.lifecycle_date = Some(svc.creation_date);
            }
            if svc.setting("index.lifecycle.parse_origination_date").as_deref() == Some("true") && svc.setting("index.lifecycle.origination_date").is_none() {
                let date = parse_origination_date(&name)?;
                update_settings(svc, vec![("index.lifecycle.origination_date", json!(date.to_string()))])?;
            }
            Outcome::Next
        }
        "set_priority" => {
            let v = match &cfg["priority"] {
                Value::Null => Value::Null,
                p => json!(p.to_string()),
            };
            update_settings(svc, vec![("index.priority", v)])?;
            Outcome::Next
        }
        "branch-check-unfollow-prerequisites" | "branch-freeze-check-prerequisites" | "branch-check-skip-action" | "branch-forcemerge-check-prerequisites" => {
            if key.name == "branch-check-unfollow-prerequisites" {
                Outcome::SkipAction
            } else {
                Outcome::Next
            }
        }
        "check-rollover-ready" => {
            if !periodic {
                return Ok(None);
            }
            let (target, is_ds) = rollover_target(node, svc)?;
            if is_ds {
                if is_ds_write_index(node, svc).is_none() {
                    return Ok(Some(Outcome::Next));
                }
            } else {
                if lc.rollover_info.contains_key(&target) {
                    return Ok(Some(Outcome::Next));
                }
                let aliases = svc.aliases();
                let points = aliases.get(&target);
                let is_write = points.and_then(|d| d.get("is_write_index")).and_then(Value::as_bool);
                if indexing_complete(svc) {
                    if points.is_some() && is_write == Some(true) {
                        return Err(EsError::new(500, "illegal_state_exception", format!(
                            "index [{name}] has [index.lifecycle.indexing_complete] set to [true], but is still the write index for alias [{target}]"
                        )));
                    }
                    return Ok(Some(Outcome::Next));
                }
                if points.is_none() {
                    return Err(EsError::illegal_argument(format!("index.lifecycle.rollover_alias [{target}] does not point to index [{name}]")));
                }
                if is_write == Some(false) {
                    return Err(EsError::illegal_argument(format!("index [{name}] is not the write index for alias [{target}]")));
                }
            }
            let conds = rollover_conditions_with_defaults(node, cfg);
            let resp = node.rollover(&target, None, &json!({"conditions": conds}), true)?;
            if conditions_met(&resp) {
                Outcome::Next
            } else {
                Outcome::Wait(None)
            }
        }
        "attempt-rollover" => {
            let (target, is_ds) = rollover_target(node, svc)?;
            if is_ds {
                if is_ds_write_index(node, svc).is_none() {
                    return Ok(Some(Outcome::Next));
                }
            } else if indexing_complete(svc) || lc.rollover_info.contains_key(&target) {
                return Ok(Some(Outcome::Next));
            }
            node.rollover(&target, None, &json!({}), false)?;
            *lc = IndexLifecycle { ilm: lc.ilm.clone(), rollover_info: load(node, svc).rollover_info };
            Outcome::Next
        }
        // On a cluster these wait for the copies (the shrink needs one copy
        // of every source shard on one node; the target must be allocated).
        // After a rollover, the new write index's primaries are active
        // (ES `WaitForActiveShardsStep`, active shards 1).
        "wait-for-active-shards" if node.ilm.cluster_steps().is_some() => {
            let steps = node.ilm.cluster_steps().expect("checked");
            match rollover_target(node, svc).ok().and_then(|(target, _)| node.write_index(&target, false).ok()) {
                Some(write) if write.name != name => progress(steps.primaries_active(&write.name))?,
                _ => Outcome::Next,
            }
        }
        "check-shrink-allocation" if node.ilm.cluster_steps().is_some() => {
            let steps = node.ilm.cluster_steps().expect("checked");
            match svc.setting("index.routing.allocation.require._id") {
                Some(id) => progress(steps.allocated_on(&name, &id))?,
                None => Outcome::Wait(Some(json!({"message": "waiting for the shrink node to be chosen"}))),
            }
        }
        "shrunk-shards-allocated" if node.ilm.cluster_steps().is_some() => {
            let target = lc.ilm.shrink_index_name.clone().unwrap_or_default();
            progress(node.ilm.cluster_steps().expect("checked").primaries_active(&target))?
        }
        "wait-for-active-shards" | "check-ts-end-time-passed" | "wait-for-shard-history-leases" | "cleanup-shrink-index" | "check-shrink-allocation"
        | "shrunk-shards-allocated" | "wait-for-index-color" | "is-shrunken-index" | "freeze" => Outcome::Next,
        "update-rollover-lifecycle-date" => {
            let (target, _) = rollover_target(node, svc).unwrap_or_default();
            let time = lc.rollover_info.get(&target).and_then(|r| r["time"].as_i64());
            match time {
                Some(t) => lc.ilm.lifecycle_date = Some(t),
                None if indexing_complete(svc) || node.data_stream_of_index(&name).is_some() => {
                    if node.data_stream_of_index(&name).is_none() {
                        lc.ilm.lifecycle_date = Some(now());
                    }
                }
                None => {
                    return Err(EsError::new(500, "illegal_state_exception", format!(
                        "no rollover info found for [{name}] with rollover target [{target}], the index has not yet rolled over with that target"
                    )))
                }
            }
            Outcome::Next
        }
        "set-indexing-complete" => {
            update_settings(svc, vec![("index.lifecycle.indexing_complete", json!("true"))])?;
            Outcome::Next
        }
        "check-not-write-index" => match is_ds_write_index(node, svc) {
            Some(ds) => Outcome::Wait(Some(json!({"message": format!(
                "index [{name}] is the write index for data stream [{ds}], pausing ILM execution of lifecycle [{pname}] until this index is no longer the write index for the data stream via manual or automated rollover"
            )}))),
            None => Outcome::Next,
        },
        "readonly" => {
            update_settings(svc, vec![("index.blocks.write", json!("true"))])?;
            Outcome::Next
        }
        "close-index" => {
            svc.set_open(false)?;
            Outcome::Next
        }
        "update-settings" => {
            if key.action == "shrink" {
                update_settings(svc, vec![("index.blocks.write", Value::Null)])?;
            } else if let Some(c) = cfg.get("index_codec").and_then(Value::as_str) {
                update_settings(svc, vec![("index.codec", json!(c))])?;
            }
            Outcome::Next
        }
        "open-index" => {
            if !svc.is_open() {
                svc.set_open(true)?;
            }
            Outcome::Next
        }
        "forcemerge" if key.action == "forcemerge" => {
            let n = cfg["max_num_segments"].as_u64().unwrap_or(1) as usize;
            match node.ilm.cluster_steps() {
                // Every copy, wherever it is (the local copy may hold no data).
                Some(steps) => progress(steps.forcemerge(&name, n))?,
                None => {
                    svc.force_merge(n, false)?;
                    Outcome::Next
                }
            }
        }
        // The cluster's force-merge completed on every copy.
        "segment-count" if node.ilm.cluster_steps().is_some() => Outcome::Next,
        "segment-count" => {
            let n = cfg["max_num_segments"].as_u64().unwrap_or(1) as usize;
            let segs = svc.local_shards().iter().map(|s| s.shard.core.read().num_segments()).max().unwrap_or(0);
            if segs <= n.max(1) {
                Outcome::Next
            } else {
                Outcome::Wait(Some(json!({"message": format!("Waiting for at least [{}] shards to force merge", 1), "shards_left_to_merge": 1})))
            }
        }
        "branch-check-prerequisites" => {
            if key.action == "searchable_snapshot" {
                return Ok(Some(Outcome::SkipAction));
            }
            // shrink: skip when the index already has the target shard count
            let src = logical_shards(svc);
            let target = shrink_target_shards(svc, cfg);
            if src == 1 || target >= src {
                Outcome::SkipAction
            } else {
                Outcome::Next
            }
        }
        "check-target-shards-count" => {
            let src = logical_shards(svc);
            if let Some(t) = cfg["number_of_shards"].as_i64() {
                if t <= 0 || src % t != 0 {
                    return Err(EsError::illegal_argument(format!(
                        "lifecycle action of policy [{pname}] for index [{name}] cannot make progress because the target shards count [{t}] must be a factor of the source index's shards count [{src}]"
                    )));
                }
            }
            Outcome::Next
        }
        "generate-index-name" => {
            if lc.ilm.shrink_index_name.is_none() {
                lc.ilm.shrink_index_name = Some(format!("shrink-{}-{name}", random_alpha(4)));
            }
            Outcome::Next
        }
        "set-single-node-allocation" => {
            let id = match node.ilm.cluster_steps() {
                Some(steps) => match steps.single_node(&name) {
                    Some(id) => id,
                    None => return Ok(Some(Outcome::Wait(Some(json!({"message": "waiting for a node that can hold a copy of every shard"}))))),
                },
                None => crate::node::NODE_ID.to_string(),
            };
            update_settings(svc, vec![("index.routing.allocation.require._id", json!(id))])?;
            Outcome::Next
        }
        "shrink" if key.action == "shrink" => {
            let target = lc.ilm.shrink_index_name.clone().unwrap_or_else(|| format!("shrink-{}-{name}", random_alpha(4)));
            // On a cluster the step waits for the whole resize (the target
            // exists before its documents are copied).
            let cluster = node.ilm.cluster_steps().is_some();
            if cluster || node.get_index(&target).is_none() {
                let shards = shrink_target_shards(svc, cfg);
                let mut settings = json!({
                    "index.number_of_shards": shards.to_string(),
                    "index.number_of_replicas": svc.setting("index.number_of_replicas").unwrap_or_else(|| "1".into()),
                    "index.routing.allocation.require._id": null,
                    "index.lifecycle.name": pname,
                });
                if let Some(a) = svc.setting("index.lifecycle.rollover_alias") {
                    settings["index.lifecycle.rollover_alias"] = json!(a);
                }
                match node.ilm.cluster_steps() {
                    // The cluster's resize: the target is created through the
                    // master and the documents copied from the source copies.
                    Some(steps) => return Ok(Some(progress(steps.shrink(&name, &target, &json!({"settings": settings})))?)),
                    None => {
                        super::resize::resize(node, &name, &target, super::resize::ResizeType::Shrink, &json!({"settings": settings}))?;
                    }
                }
            }
            Outcome::Next
        }
        "copy-execution-state" => {
            let target = match (&lc.ilm.shrink_index_name, key.action.as_str()) {
                (Some(t), "shrink") => t.clone(),
                _ => return Ok(Some(Outcome::Next)),
            };
            if let Some(tsvc) = node.get_index(&target) {
                let mut tlc = load(node, &tsvc);
                let mut st = lc.ilm.clone();
                st.step = Some("is-shrunken-index".into());
                st.step_time = Some(now());
                tlc.ilm = st;
                save(node, &tsvc, &tlc)?;
            }
            Outcome::Next
        }
        "branch-on-datastream-check" => {
            if key.action != "shrink" {
                return Ok(Some(Outcome::Next));
            }
            let target = lc.ilm.shrink_index_name.clone().unwrap_or_default();
            if let Some(ds) = node.data_stream_of_index(&name) {
                if let Some(wait) = check_shrunk_docs(node, &name, &target)? {
                    return Ok(Some(wait));
                }
                super::dlm::replace_backing_index(node, &ds, &name, &target)?;
                node.delete_index(&name)?;
                return Ok(Some(Outcome::Gone));
            }
            Outcome::Next
        }
        "aliases" => {
            // ShrinkSetAliasStep: move the source's aliases to the shrunken
            // index, add the source name as an alias, delete the source.
            let target = lc.ilm.shrink_index_name.clone().unwrap_or_default();
            let tsvc = node.get_index(&target).ok_or_else(|| EsError::index_not_found(&target))?;
            // Never delete a source whose target lacks its documents.
            if let Some(wait) = check_shrunk_docs(node, &name, &target)? {
                return Ok(Some(wait));
            }
            let mut aliases = tsvc.aliases();
            for (a, mut def) in svc.aliases() {
                if let Some(o) = def.as_object_mut() {
                    o.remove("is_write_index");
                }
                aliases.insert(a, def);
            }
            aliases.insert(name.clone(), json!({}));
            node.delete_index(&name)?;
            tsvc.set_aliases(aliases)?;
            Outcome::Gone
        }
        "allocate" => {
            let mut kv: Vec<(String, Value)> = Vec::new();
            if let Some(r) = cfg.get("number_of_replicas").filter(|v| !v.is_null()) {
                kv.push(("index.number_of_replicas".into(), json!(r.to_string())));
            }
            if let Some(t) = cfg.get("total_shards_per_node").filter(|v| !v.is_null()) {
                kv.push(("index.routing.allocation.total_shards_per_node".into(), json!(t.to_string())));
            }
            for kind in ["include", "exclude", "require"] {
                if let Some(m) = cfg[kind].as_object() {
                    for (attr, v) in m {
                        kv.push((format!("index.routing.allocation.{kind}.{attr}"), v.clone()));
                    }
                }
            }
            let flat: crate::settings::FlatSettings = kv.into_iter().collect();
            svc.put_settings(flat, false)?;
            Outcome::Next
        }
        "check-allocation" | "check-migration" => match replicas_wait_info(svc) {
            Some(info) => Outcome::Wait(Some(info)),
            None => Outcome::Next,
        },
        "migrate" => {
            let pref = match key.phase.as_str() {
                "warm" => "data_warm,data_hot",
                "cold" => "data_cold,data_warm,data_hot",
                "frozen" => "data_frozen",
                _ => "data_hot",
            };
            update_settings(svc, vec![("index.routing.allocation.include._tier_preference", json!(pref))])?;
            Outcome::Next
        }
        "branch-on-timeseries-check" => {
            if svc.setting("index.mode").as_deref() == Some("time_series") {
                return Err(EsError::illegal_argument(format!("downsampling of time series index [{name}] is not supported by this node")));
            }
            Outcome::SkipAction
        }
        "wait-for-snapshot" => {
            if !periodic {
                return Ok(None);
            }
            let p = cfg["policy"].as_str().unwrap_or_default().to_string();
            let slm = node.ilm.st.read().slm.get(&p).cloned();
            match slm {
                None => return Err(EsError::new(500, "illegal_state_exception", format!("configured policy '{p}' not found"))),
                Some(sp) => {
                    let since = lc.ilm.action_time.unwrap_or_else(now);
                    let done = sp.last_success.as_ref().and_then(|s| s["start_time"].as_i64()).map(|t| t > since).unwrap_or(false);
                    if done {
                        Outcome::Next
                    } else {
                        Outcome::Wait(Some(json!({"message": format!("waiting for policy '{p}' to be executed since {}", java_date(since))})))
                    }
                }
            }
        }
        "delete" if key.action == "delete" => {
            if let Some(ds) = is_ds_write_index(node, svc) {
                return Err(EsError::illegal_argument(format!(
                    "index [{name}] is the write index for data stream [{ds}]. stopping execution of lifecycle [{pname}] as a data stream's write index cannot be deleted. manually rolling over the index will resume the execution of the policy as the index will not be the data stream's write index anymore"
                )));
            }
            node.delete_index(&name)?;
            Outcome::Gone
        }
        _ => Outcome::Next,
    }))
}

fn shrink_target_shards(svc: &IndexService, cfg: &Value) -> i64 {
    if let Some(n) = cfg["number_of_shards"].as_i64() {
        return n;
    }
    let src = logical_shards(svc);
    let max = cfg["max_primary_shard_size"].as_str().and_then(|s| super::parse_size(s, "").ok()).map(|x| x.0).unwrap_or(0);
    if max == 0 {
        return 1;
    }
    let size = svc.local_shards().iter().map(|s| s.shard.store_size()).max().unwrap_or(0);
    let mut min = (size / max) as i64;
    if (min as u64) * max < size {
        min += 1;
    }
    if min > src {
        return src;
    }
    super::resize::acceptable_shrink_shards(src, min.max(1))
}

fn random_alpha(n: usize) -> String {
    let u = uuid::Uuid::new_v4();
    u.as_bytes().iter().take(n).map(|b| (b'a' + b % 26) as char).collect()
}

/// `IndexLifecycleOriginationDateParser`.
fn parse_origination_date(name: &str) -> Result<i64> {
    let bad = || EsError::illegal_argument(format!("index name [{name}] does not match pattern '^.*-\\d{{4}}.\\d{{2}}.\\d{{2}}(-[\\d]+)?$'"));
    let parts: Vec<&str> = name.rsplit('-').collect();
    let date_part = if parts.len() >= 2 && parts[0].chars().all(|c| c.is_ascii_digit()) && !parts[0].is_empty() && parts[1].len() == 10 { parts[1] } else { parts[0] };
    if date_part.len() != 10 {
        return Err(bad());
    }
    let f = crate::date::DateFormatter::new("yyyy.MM.dd").map_err(|_| bad())?;
    f.parse(date_part).map_err(|_| bad())
}

/// Run the lifecycle of one index. `periodic` also evaluates async wait
/// steps (`indices.lifecycle.poll_interval`).
pub fn run_index(node: &Node, svc: &Arc<IndexService>, periodic: bool, history: &mut Vec<Value>) {
    let pname = match policy_name(svc) {
        Some(p) => p,
        None => return,
    };
    if !svc.is_open() || !managed_by_ilm(node, svc) {
        return;
    }
    let mut lc = load(node, svc);
    let pm = match node.ilm.policy(&pname) {
        Some(p) => p,
        None => {
            let info = json!({"type": "illegal_argument_exception", "reason": format!("policy [{pname}] does not exist")});
            if lc.ilm.step_info.as_ref() != Some(&info) {
                lc.ilm.step_info = Some(info);
                let _ = save(node, svc, &lc);
            }
            return;
        }
    };
    if svc.setting("index.lifecycle.skip").as_deref() == Some("true") {
        return;
    }
    let before = lc.ilm.clone();
    if lc.ilm.key().is_none() {
        // InitializePolicyContextStep runs from "no state".
        move_to(&mut lc.ilm, &pm, &pname, &StepKey::new("new", "init", "init"), true, false);
    }
    // Refresh a cached phase definition that is compatible with the policy.
    if let (Some(phase), Some(pd)) = (lc.ilm.phase.clone(), lc.ilm.phase_definition.clone()) {
        if pd["version"].as_u64() != Some(pm.version) {
            let cur = lc.ilm.key();
            let compatible = phase == "new"
                || pm.policy["phases"].get(&phase).map(|d| policy::phase_steps(&phase, d).iter().any(|s| Some(&s.key) == cur.as_ref())).unwrap_or(false);
            if compatible {
                lc.ilm.phase_definition = Some(phase_def_value(&pm, &pname, &phase));
            }
        }
    }
    let mut gone = false;
    for _ in 0..256 {
        let key = match lc.ilm.key() {
            Some(k) => k,
            None => break,
        };
        if key.name == "ERROR" {
            let auto = lc.ilm.is_auto_retryable_error == Some(true);
            if periodic && auto {
                let count = lc.ilm.failed_step_retry_count.unwrap_or(0) + 1;
                retry_to_failed_step(&mut lc.ilm, &pm, &pname);
                lc.ilm.failed_step_retry_count = Some(count);
                lc.ilm.is_auto_retryable_error = Some(true);
                continue;
            }
            break;
        }
        let steps = steps_of(&lc.ilm, &pm, &key.phase);
        let idx = match steps.iter().position(|s| s.key == key) {
            Some(i) => i,
            None => {
                if key.phase == "completed" {
                    break;
                }
                // The step no longer exists (policy changed): stay put.
                break;
            }
        };
        let step = steps[idx].clone();
        match step.kind {
            StepKind::PhaseComplete => {
                let next = match next_phase(&pm, &key.phase) {
                    Some(n) => n,
                    None => break,
                };
                let min_age = pm.policy["phases"].get(&next).map(policy::phase_min_age_millis).unwrap_or(0);
                let base = origination_or_lifecycle_date(svc, &lc.ilm);
                let ready = match base {
                    Some(b) => now() >= b + min_age,
                    None => false,
                };
                if !ready {
                    break;
                }
                match first_step_of_phase(&pm, &next) {
                    Some(first) => {
                        move_to(&mut lc.ilm, &pm, &pname, &first, false, true);
                        history.push(history_item(svc, &pname, &lc.ilm, None));
                    }
                    None => break,
                }
            }
            _ => {
                let res = exec_step(node, svc, &mut lc, &pname, &step, periodic);
                match res {
                    Ok(None) => break,
                    Ok(Some(Outcome::Next)) => {
                        let next = steps.get(idx + 1).map(|s| s.key.clone());
                        match next {
                            Some(n) => {
                                move_to(&mut lc.ilm, &pm, &pname, &n, false, true);
                                history.push(history_item(svc, &pname, &lc.ilm, None));
                            }
                            None => break,
                        }
                    }
                    Ok(Some(Outcome::SkipAction)) => {
                        let next = steps[idx..].iter().find(|s| s.key.action != key.action).map(|s| s.key.clone());
                        match next {
                            Some(n) => {
                                move_to(&mut lc.ilm, &pm, &pname, &n, false, true);
                                history.push(history_item(svc, &pname, &lc.ilm, None));
                            }
                            None => break,
                        }
                    }
                    Ok(Some(Outcome::Wait(info))) => {
                        if info.is_some() && lc.ilm.step_info != info {
                            lc.ilm.step_info = info;
                        }
                        break;
                    }
                    Ok(Some(Outcome::Gone)) => {
                        history.push(history_item(svc, &pname, &lc.ilm, None));
                        gone = true;
                        break;
                    }
                    Err(e) => {
                        move_to_error(&mut lc.ilm, &pm, &pname, &key, &e);
                        history.push(history_item(svc, &pname, &lc.ilm, Some(&e)));
                        break;
                    }
                }
            }
        }
    }
    if gone {
        node.ilm.exec.lock().remove(&svc.uuid);
        *node.ilm.exec_gen.lock().entry(svc.uuid.clone()).or_insert(0) += 1;
        return;
    }
    if lc.ilm != before || !svc.path().join(LIFECYCLE_FILE).exists() {
        let _ = save(node, svc, &lc);
    }
}

fn move_to_error(st: &mut ExecState, pm: &PolicyMeta, pname: &str, key: &StepKey, e: &EsError) {
    let count = st.failed_step_retry_count;
    move_to(st, pm, pname, &StepKey::new(&key.phase, &key.action, "ERROR"), false, false);
    st.failed_step = Some(key.name.clone());
    st.step_info = Some(error_json(e));
    st.is_auto_retryable_error = Some(retryable(&key.name));
    st.failed_step_retry_count = count;
}

fn retry_to_failed_step(st: &mut ExecState, pm: &PolicyMeta, pname: &str) {
    let failed = st.failed_step.clone().unwrap_or_default();
    let key = StepKey::new(st.phase.as_deref().unwrap_or(""), st.action.as_deref().unwrap_or(""), &failed);
    move_to(st, pm, pname, &key, false, true);
}

/// Initialize a newly created, not-yet-published index in its creation
/// transaction. Only the initialization step runs here: policy actions remain
/// the lifecycle executor's responsibility, and explain remains read-only on
/// cluster hosts. The caller owns the unpublished service, so no run_lock is
/// taken (a lifecycle action can itself create an index while holding it).
pub(crate) fn initialize_new_index(node: &Node, svc: &Arc<IndexService>) -> Result<()> {
    if node.ilm.mode() != RUNNING || svc.setting("index.lifecycle.skip").as_deref() == Some("true") {
        return Ok(());
    }
    let Some(pname) = policy_name(svc) else { return Ok(()) };
    let Some(pm) = node.ilm.policy(&pname) else { return Ok(()) };
    if !managed_by_ilm(node, svc) { return Ok(()); }
    let mut lc = load(node, svc);
    if lc.ilm.key().is_some() { return Ok(()); }
    let key = StepKey::new("new", "init", "init");
    move_to(&mut lc.ilm, &pm, &pname, &key, true, false);
    let step = Step { key: key.clone(), kind: StepKind::Init, config: Value::Null };
    match exec_step(node, svc, &mut lc, &pname, &step, false) {
        Ok(Some(Outcome::Next)) => move_to(&mut lc.ilm, &pm, &pname, &StepKey::new("new", "complete", "complete"), false, true),
        Err(error) => move_to_error(&mut lc.ilm, &pm, &pname, &key, &error),
        _ => {},
    }
    save(node, svc, &lc)
}

/// One executor pass over every managed index.
pub fn run_all(node: &Node, periodic: bool) {
    let _g = node.ilm.run_lock.lock();
    let mode = node.ilm.mode();
    if mode == STOPPING {
        // Safe to stop once no index is inside an unsafe action (shrink).
        let unsafe_any = node.all_indices().iter().any(|s| {
            let lc = load(node, s);
            lc.ilm.action.as_deref() == Some("shrink") && lc.ilm.step.as_deref() != Some("ERROR")
        });
        if !unsafe_any {
            node.ilm.st.write().mode = STOPPED.into();
            let _ = node.ilm.persist();
        }
        return;
    }
    if mode != RUNNING {
        return;
    }
    let mut history = Vec::new();
    for svc in node.all_indices() {
        if policy_name(&svc).is_some() {
            run_index(node, &svc, periodic, &mut history);
        }
    }
    write_history(node, history);
}

/// Run the "state changed" pass for the given indices (API-triggered).
pub fn run_indices(node: &Node, names: &[String]) {
    let _g = node.ilm.run_lock.lock();
    if node.ilm.mode() != RUNNING {
        return;
    }
    let mut history = Vec::new();
    for n in names {
        if let Some(svc) = node.get_index(n) {
            run_index(node, &svc, false, &mut history);
        }
    }
    write_history(node, history);
}

// ---------------------------------------------------------------------------
// History (`ilm-history-7` data stream)
// ---------------------------------------------------------------------------

fn history_item(svc: &IndexService, policy: &str, st: &ExecState, err: Option<&EsError>) -> Value {
    let mut state = Map::new();
    let mut put = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            state.insert(k.to_string(), json!(v));
        }
    };
    put("phase", st.phase.clone());
    put("action", st.action.clone());
    put("step", st.step.clone());
    put("failed_step", st.failed_step.clone());
    put("is_auto_retryable_error", st.is_auto_retryable_error.map(|b| b.to_string()));
    put("failed_step_retry_count", st.failed_step_retry_count.map(|b| b.to_string()));
    put("step_info", st.step_info.as_ref().map(|v| v.to_string()));
    put("phase_definition", st.phase_definition.as_ref().map(|v| v.to_string()));
    put("creation_date", st.lifecycle_date.map(|v| v.to_string()));
    put("phase_time", st.phase_time.map(|v| v.to_string()));
    put("action_time", st.action_time.map(|v| v.to_string()));
    put("step_time", st.step_time.map(|v| v.to_string()));
    let t = now();
    let mut v = json!({
        "index": svc.name,
        "policy": policy,
        "@timestamp": t,
        "index_age": st.lifecycle_date.map(|d| t - d).unwrap_or(0),
        "success": err.is_none(),
        "state": Value::Object(state),
    });
    if let Some(e) = err {
        v["error_details"] = json!(json!({"type": e.kind, "reason": e.reason}).to_string());
    }
    v
}

pub fn history_template() -> Value {
    json!({
        "index_patterns": [format!("{HISTORY_DATA_STREAM}*")],
        "priority": 2147483647,
        "data_stream": {"hidden": true},
        "template": {
            "settings": {"index.number_of_shards": 1, "index.auto_expand_replicas": "0-1", "index.lifecycle.name": "ilm-history-ilm-policy", "index.hidden": true, "index.format": 1},
            "mappings": {"dynamic": false, "properties": {
                "@timestamp": {"type": "date", "format": "epoch_millis"},
                "policy": {"type": "keyword"},
                "index": {"type": "keyword"},
                "index_age": {"type": "long"},
                "success": {"type": "boolean"},
                "state": {"type": "object", "dynamic": true, "properties": {
                    "phase": {"type": "keyword"}, "action": {"type": "keyword"}, "step": {"type": "keyword"}, "failed_step": {"type": "keyword"},
                    "is_auto-retryable_error": {"type": "keyword"}, "creation_date": {"type": "date", "format": "epoch_millis"},
                    "phase_time": {"type": "date", "format": "epoch_millis"}, "action_time": {"type": "date", "format": "epoch_millis"},
                    "step_time": {"type": "date", "format": "epoch_millis"}, "phase_definition": {"type": "text"}, "step_info": {"type": "text"}
                }},
                "error_details": {"type": "text"}
            }}
        },
        "version": 7,
        "_meta": {"description": "index template for ILM history indices", "managed": true}
    })
}

fn write_history(node: &Node, items: Vec<Value>) {
    if items.is_empty() {
        return;
    }
    let enabled = super::cluster_setting(node, "indices.lifecycle.history_index_enabled").map(|s| s != "false").unwrap_or(true);
    if !enabled {
        return;
    }
    // A cluster indexes history through its own write path (a local
    // write here would create the history data stream on a copy).
    if let Some(steps) = node.ilm.cluster_steps() {
        steps.write_history(items);
        return;
    }
    let installed = node.templates.read().index.contains_key(HISTORY_DATA_STREAM);
    if !installed {
        node.templates.write().index.insert(HISTORY_DATA_STREAM.into(), history_template());
        let _ = node.persist_state();
    }
    let svc = match node.write_index(HISTORY_DATA_STREAM, true) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("failed to write ILM history: {e}");
            return;
        }
    };
    let reqs: Vec<crate::index::WriteReq> = items
        .into_iter()
        .map(|v| crate::index::WriteReq::Index {
            id: None,
            source: v.to_string(),
            routing: None,
            cond: crate::shard::WriteCond { create: true, ..Default::default() },
        })
        .collect();
    let _ = svc.write(reqs, false);
}

// ---------------------------------------------------------------------------
// APIs
// ---------------------------------------------------------------------------

/// `GET /{index}/_ilm/explain`.
pub fn explain(node: &Node, expr: &str, opts: ResolveOpts, only_managed: bool, only_errors: bool) -> Result<Value> {
    let targets = node.resolve(expr, opts)?;
    let names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
    // A host that drives lifecycle runs steps only in its own passes:
    // explain is a read (on a cluster, a step run here would change an
    // engine outside the cluster state).
    if !node.ilm.driven_externally() {
        run_indices(node, &names);
    }
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for n in names {
        let svc = match node.get_index(&n) {
            Some(s) => s,
            None => continue,
        };
        if let Some(v) = explain_index(node, &svc, only_managed, only_errors) {
            out.insert(n, v);
        }
    }
    Ok(json!({"indices": out}))
}

fn explain_index(node: &Node, svc: &IndexService, only_managed: bool, only_errors: bool) -> Option<Value> {
    if !managed_by_ilm(node, svc) {
        if only_managed || only_errors {
            return None;
        }
        return Some(json!({"index": svc.name, "managed": false}));
    }
    let pname = policy_name(svc).unwrap_or_default();
    let lc = load(node, svc);
    let st = &lc.ilm;
    let policy_exists = node.ilm.policy(&pname).is_some();
    if only_errors && st.step.as_deref() != Some("ERROR") && policy_exists {
        return None;
    }
    let t = now();
    let mut m = Map::new();
    m.insert("index".into(), json!(svc.name));
    m.insert("managed".into(), json!(true));
    m.insert("policy".into(), json!(pname));
    m.insert("index_creation_date_millis".into(), json!(svc.creation_date));
    m.insert("time_since_index_creation".into(), json!(human_time(t - svc.creation_date)));
    if let Some(ld) = origination_or_lifecycle_date(svc, st) {
        m.insert("lifecycle_date_millis".into(), json!(ld));
        m.insert("age".into(), json!(human_time(t - ld)));
    }
    if let Some(p) = &st.phase {
        m.insert("phase".into(), json!(p));
    }
    if let Some(v) = st.phase_time {
        m.insert("phase_time_millis".into(), json!(v));
    }
    if let Some(a) = &st.action {
        m.insert("action".into(), json!(a));
    }
    if let Some(v) = st.action_time {
        m.insert("action_time_millis".into(), json!(v));
    }
    let step = if !policy_exists { Some("ERROR".to_string()) } else { st.step.clone() };
    if let Some(s) = step {
        m.insert("step".into(), json!(s));
    }
    if let Some(v) = st.step_time {
        m.insert("step_time_millis".into(), json!(v));
    }
    if let Some(f) = st.failed_step.as_ref().filter(|f| !f.is_empty()) {
        m.insert("failed_step".into(), json!(f));
    }
    if let Some(b) = st.is_auto_retryable_error {
        m.insert("is_auto_retryable_error".into(), json!(b));
    }
    if let Some(c) = st.failed_step_retry_count {
        m.insert("failed_step_retry_count".into(), json!(c));
    }
    if let Some(r) = &st.snapshot_repository {
        m.insert("repository_name".into(), json!(r));
    }
    if let Some(s) = &st.snapshot_name {
        m.insert("snapshot_name".into(), json!(s));
    }
    if let Some(s) = &st.shrink_index_name {
        m.insert("shrink_index_name".into(), json!(s));
    }
    if let Some(i) = st.step_info.as_ref().filter(|i| !i.is_null()) {
        m.insert("step_info".into(), i.clone());
    }
    if let Some(i) = st.previous_step_info.as_ref().filter(|i| !i.is_null()) {
        m.insert("previous_step_info".into(), i.clone());
    }
    if let Some(pd) = &st.phase_definition {
        let mut pd = pd.clone();
        if let Some(r) = pd.get("phase_definition").and_then(|d| d["actions"].get("rollover")).cloned() {
            pd["phase_definition"]["actions"]["rollover"] = rollover_conditions_with_defaults(node, &r);
        }
        m.insert("phase_execution".into(), pd);
    }
    m.insert("skip".into(), json!(svc.setting("index.lifecycle.skip").as_deref() == Some("true")));
    Some(Value::Object(m))
}

/// Add the human-readable siblings of `*_millis` explain fields (`?human`).
pub fn humanize_explain(v: &mut Value) {
    if let Some(ix) = v["indices"].as_object_mut() {
        for e in ix.values_mut() {
            if let Some(o) = e.as_object_mut() {
                let mut add = Vec::new();
                for (k, x) in o.iter() {
                    if let (Some(base), Some(ms)) = (k.strip_suffix("_millis"), x.as_i64()) {
                        add.push((base.to_string(), iso_date(ms)));
                    }
                }
                let mut out = Map::new();
                for (k, x) in o.iter() {
                    if let Some(base) = k.strip_suffix("_millis") {
                        if let Some((_, h)) = add.iter().find(|(b, _)| b == base) {
                            out.insert(base.to_string(), json!(h));
                        }
                    }
                    out.insert(k.clone(), x.clone());
                }
                *o = out;
            }
        }
    }
}

/// `POST /_ilm/move/{index}`.
pub fn move_to_step(node: &Node, index: &str, body: &Value) -> Result<()> {
    let parse_key = |v: &Value, field: &str, require_all: bool| -> Result<(Option<String>, Option<String>, Option<String>)> {
        let o = v.as_object().ok_or_else(|| EsError::x_content_parse(format!("[1:1] [move_to_step_request] failed to parse field [{field}]")))?;
        for k in o.keys() {
            if !matches!(k.as_str(), "phase" | "action" | "name") {
                return Err(EsError::x_content_parse(format!("[1:2] [{field}] unknown field [{k}]")));
            }
        }
        let g = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
        let (p, a, n) = (g("phase"), g("action"), g("name"));
        if require_all && (p.is_none() || a.is_none() || n.is_none()) {
            return Err(EsError::illegal_argument(format!("Required [{}]", if p.is_none() { "phase" } else if a.is_none() { "action" } else { "name" })));
        }
        if p.is_none() {
            return Err(EsError::illegal_argument("Required [phase]"));
        }
        if a.is_none() && n.is_some() {
            return Err(EsError::illegal_argument(
                "phase; phase and action; or phase, action, and step must be provided, but a step name was specified without a corresponding action",
            ));
        }
        Ok((p, a, n))
    };
    let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "current_step" | "next_step") {
            return Err(EsError::x_content_parse(format!("[1:2] [move_to_step_request] unknown field [{k}]")));
        }
    }
    let missing: Vec<&str> = ["current_step", "next_step"].into_iter().filter(|k| !o.contains_key(*k)).collect();
    if !missing.is_empty() {
        return Err(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))));
    }
    let cur = parse_key(&o["current_step"], "current_step", true)?;
    let next = parse_key(&o["next_step"], "next_step", false)?;
    let cur_key = StepKey::new(cur.0.as_deref().unwrap(), cur.1.as_deref().unwrap(), cur.2.as_deref().unwrap());
    let svc = node.get_index(index).ok_or_else(|| EsError::illegal_argument(format!("index [{index}] does not exist")))?;
    let pname = policy_name(&svc).ok_or_else(|| EsError::illegal_argument(format!("index [{index}] is not associated with an Index Lifecycle Policy")))?;
    run_indices(node, &[index.to_string()]);
    let _g = node.ilm.run_lock.lock();
    let pm = node.ilm.policy(&pname);
    let mut lc = load(node, &svc);
    // Resolve the (possibly partial) target key against the policy.
    let all = pm.as_ref().map(|p| policy::all_step_keys(&p.policy)).unwrap_or_default();
    let (np, na, nn) = next;
    let np = np.unwrap();
    let concrete = all.iter().find(|k| k.phase == np && na.as_ref().map(|a| *a == k.action).unwrap_or(true) && nn.as_ref().map(|n| *n == k.name).unwrap_or(true)).cloned();
    let concrete = match concrete {
        Some(k) => k,
        None => {
            let mut partial = format!("{{\"phase\":\"{np}\"");
            if let Some(a) = &na {
                partial.push_str(&format!(",\"action\":\"{a}\""));
            }
            if let Some(n) = &nn {
                partial.push_str(&format!(",\"name\":\"{n}\""));
            }
            partial.push('}');
            return Err(EsError::illegal_argument(format!(
                "cannot move index [{index}] with policy [{pname}]: unable to determine concrete step key from target next step key: {partial}"
            )));
        }
    };
    let real = lc.ilm.key();
    if real.as_ref() != Some(&cur_key) {
        let cur_txt = real.map(|k| k.render()).unwrap_or_else(|| "null".into());
        return Err(EsError::illegal_argument(format!("index [{index}] is not on current step [{}], currently: [{cur_txt}]", cur_key.render())));
    }
    let pm = pm.ok_or_else(|| EsError::illegal_argument(format!("policy [{pname}] does not exist")))?;
    move_to(&mut lc.ilm, &pm, &pname, &concrete, true, true);
    save(node, &svc, &lc)?;
    drop(_g);
    run_indices(node, &[index.to_string()]);
    Ok(())
}

/// `POST /{index}/_ilm/retry`.
pub fn retry(node: &Node, expr: &str) -> Result<()> {
    let mut names = Vec::new();
    for raw in expr.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if raw.contains('*') {
            for t in node.resolve(raw, ResolveOpts::default())? {
                names.push(t.svc.name.clone());
            }
        } else {
            names.push(raw.to_string());
        }
    }
    {
        let _g = node.ilm.run_lock.lock();
        for n in &names {
            let svc = node.get_index(n).ok_or_else(|| EsError::illegal_argument(format!("index [{n}] does not exist")))?;
            let lc = load(node, &svc);
            if lc.ilm.step.as_deref() != Some("ERROR") {
                return Err(EsError::illegal_argument(format!(
                    "cannot retry an action for an index [{n}] that has not encountered an error when running a Lifecycle Policy"
                )));
            }
        }
        for n in &names {
            let svc = match node.get_index(n) {
                Some(s) => s,
                None => continue,
            };
            let pname = policy_name(&svc).unwrap_or_default();
            let pm = node.ilm.policy(&pname).ok_or_else(|| EsError::illegal_argument(format!("policy [{pname}] does not exist")))?;
            let mut lc = load(node, &svc);
            let count = lc.ilm.failed_step_retry_count;
            let auto = lc.ilm.is_auto_retryable_error;
            retry_to_failed_step(&mut lc.ilm, &pm, &pname);
            lc.ilm.failed_step_retry_count = count;
            lc.ilm.is_auto_retryable_error = auto;
            save(node, &svc, &lc)?;
        }
    }
    run_indices(node, &names);
    Ok(())
}

/// `POST /{index}/_ilm/remove`.
pub fn remove_policy(node: &Node, expr: &str, opts: ResolveOpts) -> Result<Value> {
    let targets = node.resolve(expr, opts)?;
    let _g = node.ilm.run_lock.lock();
    let failed: Vec<String> = Vec::new();
    for t in targets {
        let svc = &t.svc;
        let keys = ["index.lifecycle.name", "index.lifecycle.indexing_complete", "index.lifecycle.skip", "index.lifecycle.rollover_alias"];
        if keys.iter().any(|k| svc.setting(k).is_some()) {
            let flat: crate::settings::FlatSettings = keys.iter().map(|k| (k.to_string(), Value::Null)).collect();
            svc.put_settings(flat, false)?;
        }
        let mut lc = load(node, svc);
        if !lc.ilm.is_empty() {
            lc.ilm = ExecState::default();
            save(node, svc, &lc)?;
        }
    }
    Ok(json!({"has_failures": !failed.is_empty(), "failed_indexes": failed}))
}

/// Indices that did not progress in a long time (health indicator).
pub fn stagnating_indices(node: &Node) -> usize {
    let day = 24 * 3600 * 1000;
    node.all_indices()
        .iter()
        .filter(|s| policy_name(s).is_some())
        .filter(|s| {
            let lc = load(node, s);
            lc.ilm.failed_step_retry_count.unwrap_or(0) >= 100 && lc.ilm.step_time.map(|t| now() - t > day).unwrap_or(false)
        })
        .count()
}

/// Indices (and data streams / templates) using a policy (`in_use_by`).
pub fn policy_usage(node: &Node, name: &str) -> Value {
    let indices: Vec<String> = node.all_indices().iter().filter(|s| policy_name(s).as_deref() == Some(name)).map(|s| s.name.clone()).collect();
    let mut templates: Vec<String> = Vec::new();
    {
        let t = node.templates.read();
        for (tn, body) in &t.index {
            let mut settings = crate::settings::flatten(&body["template"]["settings"]);
            for c in body["composed_of"].as_array().cloned().unwrap_or_default() {
                if let Some(ct) = c.as_str().and_then(|n| t.component.get(n)) {
                    for (k, v) in crate::settings::flatten(&ct["template"]["settings"]) {
                        settings.entry(k).or_insert(v);
                    }
                }
            }
            let lname = settings.get("index.lifecycle.name").or_else(|| settings.get("lifecycle.name")).and_then(Value::as_str).map(str::to_string);
            if lname.as_deref() == Some(name) {
                templates.push(tn.clone());
            }
        }
    }
    let ds: Vec<String> = node
        .data_streams
        .read()
        .values()
        .filter(|d| templates.contains(&d.template))
        .map(|d| d.name.clone())
        .collect();
    json!({"indices": indices, "data_streams": ds, "composable_templates": templates})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditions_and_dates() {
        assert!(conditions_met(&json!({"conditions": {"[max_docs: 1]": true, "[min_docs: 1]": true}})));
        assert!(!conditions_met(&json!({"conditions": {"[max_docs: 1]": true, "[min_docs: 1]": false}})));
        assert!(!conditions_met(&json!({"conditions": {"[max_docs: 1]": false}})));
        assert!(parse_origination_date("logs-2024.01.02-000001").is_ok());
        assert!(parse_origination_date("logs-2024.01.02").is_ok());
        assert!(parse_origination_date("logs").is_err());
    }

    #[test]
    fn managed_creation_publishes_initial_state_without_explain_side_effects() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        let policy = policy::parse_put_body(&json!({"policy":{"phases":{"hot":{"actions":{"set_priority":{"priority":123}}}}}})).unwrap();
        node.ilm.put_policy(&node, "initial", policy).unwrap();
        let svc = node.create_index("managed", &json!({"settings":{"index.lifecycle.name":"initial"}})).unwrap();
        let before = load(&node, &svc).ilm;
        assert_eq!(before.key(), Some(StepKey::new("new", "complete", "complete")));
        assert_eq!(before.lifecycle_date, Some(svc.creation_date));
        assert_ne!(svc.setting("index.priority").as_deref(), Some("123"));
        let response = explain(&node, "managed", ResolveOpts::default(), false, false).unwrap();
        assert_eq!(response["indices"]["managed"]["step"], "complete");
        assert_eq!(load(&node, &svc).ilm, before);
        assert_ne!(svc.setting("index.priority").as_deref(), Some("123"));
    }

    #[test]
    fn data_stream_creation_defers_ilm_until_lifecycle_precedence_is_known() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        let policy = policy::parse_put_body(&json!({"policy":{"phases":{"hot":{"actions":{}}}}})).unwrap();
        node.ilm.put_policy(&node, "unused", policy).unwrap();
        node.templates.write().index.insert("dlm".into(), json!({"index_patterns":["dlm-*"],"data_stream":{},"template":{
            "settings":{"index.lifecycle.name":"unused","index.lifecycle.prefer_ilm":false,"index.lifecycle.parse_origination_date":true},
            "lifecycle":{"data_retention":"7d"}
        }}));
        let ds = node.create_data_stream("dlm-test").unwrap();
        let svc = node.get_index(ds.write_index()).unwrap();
        assert!(!managed_by_ilm(&node, &svc));
        assert!(load(&node, &svc).ilm.is_empty());
        assert!(svc.setting("index.lifecycle.origination_date").is_none());
        run_indices(&node, &[svc.name.clone()]);
        assert!(load(&node, &svc).ilm.is_empty());
    }

    #[test]
    fn move_into_warm_runs_actions() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let p = policy::parse_put_body(&json!({"policy": {"phases": {"warm": {"min_age": "1000s", "actions": {"forcemerge": {"max_num_segments": 10000}}}, "hot": {"min_age": "1000s", "actions": {}}}}})).unwrap();
        node.ilm.put_policy(&node, "mv", p).unwrap();
        node.create_index("my_index", &json!({"settings": {"index.lifecycle.name": "mv"}})).unwrap();
        let v = explain(&node, "my_index", ResolveOpts::default(), false, false).unwrap();
        assert_eq!(v["indices"]["my_index"]["step"], "complete", "{v}");
        move_to_step(&node, "my_index", &json!({"current_step": {"phase": "new", "action": "complete", "name": "complete"}, "next_step": {"phase": "warm", "action": "forcemerge"}})).unwrap();
        let v = explain(&node, "my_index", ResolveOpts::default(), false, false).unwrap();
        let svc = node.get_index("my_index").unwrap();
        let st = load(&node, &svc);
        let pm = node.ilm.policy("mv").unwrap();
        let keys: Vec<String> = steps_of(&st.ilm, &pm, "warm").iter().map(|s| s.key.render()).collect();
        // warm runs migrate before forcemerge: moving to forcemerge skips it
        assert_eq!(v["indices"]["my_index"]["step"], "complete", "{v}");
        assert_eq!(keys[0], "{\"phase\":\"warm\",\"action\":\"migrate\",\"name\":\"branch-check-skip-action\"}");
        assert_eq!(svc.setting("index.blocks.write").as_deref(), Some("true"));
    }
}
