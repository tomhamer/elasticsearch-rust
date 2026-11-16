//! Cluster-owned transform storage and task ownership (ES
//! `TransformConfigManager` + `TransformPersistentTasksExecutor`).
//!
//! On a cluster a transform's config, state/stats and checkpoints are
//! documents in the `.transform-internal-007` system index (replicated
//! like any index, so every master candidate can read them), and a started
//! transform is a persistent task in the cluster state custom
//! `persistent_tasks` (ES `PersistentTasksCustomMetadata`), assigned to the
//! elected master with a fresh allocation id. The task is the authority for
//! "started": a newly elected master finds the tasks assigned to another
//! node, re-assigns them to itself (new allocation id) and resumes from the
//! stored state; an old master stops executing as soon as it is not the
//! elected master, or its allocation id is no longer the task's. State
//! writes are additionally fenced with `if_seq_no`/`if_primary_term`, so a
//! stale executor can never overwrite the new owner's checkpoint.
//!
//! The functions here are pure (JSON in, JSON out) so the ownership rules are
//! unit-testable without a cluster.

use crate::error::Result;
use serde_json::{json, Map, Value};

/// The cluster a transform service runs in, installed by the cluster node.
pub trait TransformCluster: Send + Sync {
    /// REST through the local coordinator as the internal `_system` user
    /// (internal index documents; work resumed without a caller).
    fn system_rest(&self) -> super::RestCall;
    /// REST as a transform's stored subject: its source and destination
    /// work runs with the privileges (and DLS/FLS) of the subject that
    /// created or last updated it (an API key limited by its owner, as ES
    /// runs a transform with its stored authentication). `None`: as
    /// [`Self::system_rest`] (security disabled).
    fn rest_as(&self, subject: Option<&super::RunAs>) -> super::RestCall;
    /// This node's id.
    fn local_node(&self) -> String;
    /// The master term when this node is the elected master of its applied
    /// cluster state, `None` otherwise.
    fn master_term(&self) -> Option<u64>;
    /// The applied `persistent_tasks` custom (`Null` when absent).
    fn persistent_tasks(&self) -> Value;
    /// Publish a change of the `persistent_tasks` custom computed from the
    /// committed one (`None`: no change) and wait until this node applied
    /// it; returns the custom as published.
    fn update_persistent_tasks(&self, f: TasksChange) -> Result<Value>;
}

/// A change of the `persistent_tasks` custom: from the committed custom to
/// the new one (`None`: no change). Run again on a newer custom when the
/// publication raced another change.
pub type TasksChange = std::sync::Arc<dyn Fn(&Value) -> Option<Value> + Send + Sync>;

/// The cluster state custom holding persistent tasks.
pub const TASKS_CUSTOM: &str = "persistent_tasks";
/// ES `TransformField.TASK_NAME`.
pub const TASK_NAME: &str = "data_frame/transforms";
/// ES `TransformInternalIndexConstants.LATEST_INDEX_NAME`.
pub const INTERNAL_INDEX: &str = ".transform-internal-007";
pub const CONFIG_DOC_TYPE: &str = "data_frame_transform_config";
pub const STATE_DOC_TYPE: &str = "data_frame_transform_state_and_stats";
pub const CHECKPOINT_DOC_TYPE: &str = "data_frame_transform_checkpoint";

pub fn config_doc_id(id: &str) -> String {
    format!("{CONFIG_DOC_TYPE}-{id}")
}

pub fn state_doc_id(id: &str) -> String {
    format!("{STATE_DOC_TYPE}-{id}")
}

pub fn checkpoint_doc_id(id: &str, checkpoint: u64) -> String {
    format!("{CHECKPOINT_DOC_TYPE}-{id}-{checkpoint}")
}

/// Settings and mappings of the internal index (hidden, one shard, a
/// replica wherever a second node exists; documents are kept, only the
/// lookup fields are indexed).
pub fn internal_index_body() -> Value {
    json!({
        "settings": {"index": {"number_of_shards": 1, "auto_expand_replicas": "0-1", "hidden": true}},
        "mappings": {
            "dynamic": false,
            "_meta": {"version": crate::node::ES_VERSION},
            "properties": {
                "id": {"type": "keyword"},
                "transform_id": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "checkpoint": {"type": "long"},
                "timestamp_millis": {"type": "date", "format": "epoch_millis"},
                "time_upper_bound_millis": {"type": "date", "format": "epoch_millis"}
            }
        }
    })
}

/// A transform's persistent task as the cluster state has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRef {
    pub allocation_id: u64,
    pub executor: Option<String>,
}

fn tasks(custom: &Value) -> Vec<Value> {
    custom.get("tasks").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn is_transform_task(t: &Value) -> bool {
    t.get("task").and_then(|x| x.get(TASK_NAME)).is_some()
}

fn task_ref(t: &Value) -> TaskRef {
    TaskRef {
        allocation_id: t["allocation_id"].as_u64().unwrap_or(0),
        executor: t["assignment"]["executor_node"].as_str().map(str::to_string),
    }
}

/// The transform task `id`, if started.
pub fn task_of(custom: &Value, id: &str) -> Option<TaskRef> {
    tasks(custom).iter().find(|t| t["id"] == id && is_transform_task(t)).map(task_ref)
}

/// Every transform task: `(transform id, task)`.
pub fn transform_tasks(custom: &Value) -> Vec<(String, TaskRef)> {
    tasks(custom).iter().filter(|t| is_transform_task(t)).filter_map(|t| Some((t["id"].as_str()?.to_string(), task_ref(t)))).collect()
}

/// ES `TransformTaskParams`.
pub fn task_params(id: &str, frequency: Option<&str>, from: Option<i64>) -> Value {
    let mut p = Map::new();
    p.insert("transform_id".into(), json!(id));
    p.insert("version".into(), json!(crate::node::ES_VERSION));
    if let Some(f) = frequency {
        p.insert("frequency".into(), json!(f));
    }
    if let Some(f) = from {
        p.insert("from".into(), json!(f));
    }
    p.insert("requires_remote".into(), json!(false));
    Value::Object(p)
}

fn with_tasks(custom: &Value, tasks: Vec<Value>, last: u64) -> Value {
    let mut out = custom.as_object().cloned().unwrap_or_default();
    out.insert("last_allocation_id".into(), json!(last));
    out.insert("tasks".into(), Value::Array(tasks));
    Value::Object(out)
}

fn assignment(executor: &str, term: u64) -> Value {
    json!({"executor_node": executor, "explanation": format!("assigned by the elected master in term [{term}]")})
}

/// Create transform task `id` assigned to `executor`; `None` when it
/// already exists. Returns the new custom and the task's allocation id.
pub fn create_task(custom: &Value, id: &str, params: Value, executor: &str, term: u64) -> Option<(Value, u64)> {
    if task_of(custom, id).is_some() {
        return None;
    }
    let alloc = custom["last_allocation_id"].as_u64().unwrap_or(0) + 1;
    let mut ts = tasks(custom);
    ts.push(json!({
        "id": id,
        "task": {TASK_NAME: {"params": params}},
        "allocation_id": alloc,
        "assignment": assignment(executor, term),
        "allocation_id_on_last_status_update": 0,
    }));
    Some((with_tasks(custom, ts, alloc), alloc))
}

/// Remove transform task `id` (`None`: there is none).
pub fn remove_task(custom: &Value, id: &str) -> Option<Value> {
    let ts = tasks(custom);
    let kept: Vec<Value> = ts.iter().filter(|t| !(t["id"] == id && is_transform_task(t))).cloned().collect();
    if kept.len() == ts.len() {
        return None;
    }
    Some(with_tasks(custom, kept, custom["last_allocation_id"].as_u64().unwrap_or(0)))
}

/// Remove every transform task (`None`: there is none).
pub fn remove_all(custom: &Value) -> Option<Value> {
    let ts = tasks(custom);
    let kept: Vec<Value> = ts.iter().filter(|t| !is_transform_task(t)).cloned().collect();
    if kept.len() == ts.len() {
        return None;
    }
    Some(with_tasks(custom, kept, custom["last_allocation_id"].as_u64().unwrap_or(0)))
}

/// Re-assign the transform tasks in `ids` that are not assigned to
/// `executor` to it, each with a new allocation id (ES re-assigns a
/// persistent task whose executor left with a fresh allocation id, which is
/// what fences the old executor). `None` when nothing changes.
pub fn reassign(custom: &Value, ids: &[String], executor: &str, term: u64) -> Option<Value> {
    let mut last = custom["last_allocation_id"].as_u64().unwrap_or(0);
    let mut changed = false;
    let ts: Vec<Value> = tasks(custom)
        .into_iter()
        .map(|mut t| {
            let mine = t["assignment"]["executor_node"].as_str() == Some(executor);
            if is_transform_task(&t) && !mine && t["id"].as_str().is_some_and(|id| ids.iter().any(|x| x == id)) {
                last += 1;
                t["allocation_id"] = json!(last);
                t["assignment"] = assignment(executor, term);
                changed = true;
            }
            t
        })
        .collect();
    changed.then(|| with_tasks(custom, ts, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_reassign_remove_keep_es_shape_and_fence_by_allocation_id() {
        let empty = Value::Null;
        let (c1, a1) = create_task(&empty, "t1", task_params("t1", Some("1s"), None), "nodeA", 3).unwrap();
        assert_eq!(a1, 1);
        assert!(create_task(&c1, "t1", task_params("t1", None, None), "nodeA", 3).is_none(), "a started transform has one task");
        let (c2, a2) = create_task(&c1, "t2", task_params("t2", None, None), "nodeA", 3).unwrap();
        assert_eq!(a2, 2);
        assert_eq!(c2["last_allocation_id"], 2);
        assert_eq!(c2["tasks"][0]["task"][TASK_NAME]["params"]["transform_id"], "t1");
        assert_eq!(task_of(&c2, "t1"), Some(TaskRef { allocation_id: 1, executor: Some("nodeA".into()) }));
        // A new master takes both: new allocation ids fence the old executor.
        let c3 = reassign(&c2, &["t1".into(), "t2".into()], "nodeB", 4).unwrap();
        assert_eq!(task_of(&c3, "t1"), Some(TaskRef { allocation_id: 3, executor: Some("nodeB".into()) }));
        assert_eq!(task_of(&c3, "t2"), Some(TaskRef { allocation_id: 4, executor: Some("nodeB".into()) }));
        assert!(c3["tasks"][0]["assignment"]["explanation"].as_str().unwrap().contains("term [4]"));
        assert!(reassign(&c3, &["t1".into()], "nodeB", 4).is_none(), "already the executor: unchanged");
        let c4 = remove_task(&c3, "t1").unwrap();
        assert!(task_of(&c4, "t1").is_none());
        assert_eq!(c4["last_allocation_id"], 4, "allocation ids are never reused");
        assert!(remove_task(&c4, "t1").is_none());
        assert_eq!(transform_tasks(&c4).len(), 1);
        assert!(remove_all(&c4).is_some_and(|c| transform_tasks(&c).is_empty()));
    }

    #[test]
    fn other_persistent_tasks_are_left_alone() {
        let custom = json!({"last_allocation_id": 7, "tasks": [{"id": "job", "task": {"xpack/ml/job": {"params": {}}}, "allocation_id": 7, "assignment": {"executor_node": "x"}}]});
        let (c, a) = create_task(&custom, "job", task_params("job", None, None), "n", 1).unwrap();
        assert_eq!(a, 8, "a transform may share an id with another task type");
        assert_eq!(c["tasks"].as_array().unwrap().len(), 2);
        let c = remove_all(&c).unwrap();
        assert_eq!(c["tasks"].as_array().unwrap().len(), 1);
        assert_eq!(c["tasks"][0]["id"], "job");
        assert!(remove_all(&c).is_none());
        assert!(reassign(&custom, &["job".into()], "n", 1).is_none(), "only transform tasks are re-assigned");
    }
}
