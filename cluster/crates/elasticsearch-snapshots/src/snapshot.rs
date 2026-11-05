//! Snapshot model: `SnapshotInfo`, states, shard failures, live progress, and
//! their ES REST renderings (`GET _snapshot/{repo}/{snap}`, `_status`).

use crate::util::{format_duration_millis, format_millis};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// Version this implementation reports for snapshots it writes: ES 9.5.5's
/// `IndexVersion.CURRENT` id and its release range (`IndexVersions.VERSION_LOOKUP`).
pub const ES_VERSION: &str = "9.5.2-9.5.5";
pub const ES_VERSION_ID: u32 = 9_111_000;
/// Lucene version recorded as `written_by` on snapshotted files.
pub const LUCENE_VERSION: &str = "10.5.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SnapshotState {
    InProgress,
    Success,
    Failed,
    Partial,
    Incompatible,
}

impl SnapshotState {
    /// Byte value used in `index-N` (ES `SnapshotState.value()`).
    pub fn value(self) -> u8 {
        match self {
            Self::InProgress => 0,
            Self::Success => 1,
            Self::Failed => 2,
            Self::Partial => 3,
            Self::Incompatible => 4,
        }
    }

    pub fn from_value(v: u8) -> Self {
        match v {
            0 => Self::InProgress,
            1 => Self::Success,
            2 => Self::Failed,
            3 => Self::Partial,
            _ => Self::Incompatible,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "IN_PROGRESS",
            Self::Success => "SUCCESS",
            Self::Failed => "FAILED",
            Self::Partial => "PARTIAL",
            Self::Incompatible => "INCOMPATIBLE",
        }
    }

    pub fn completed(self) -> bool {
        matches!(self, Self::Success | Self::Partial | Self::Failed)
    }

    pub fn restorable(self) -> bool {
        matches!(self, Self::Success | Self::Partial)
    }
}

impl std::fmt::Display for SnapshotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardFailure {
    pub index: String,
    pub index_uuid: String,
    pub shard_id: u32,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// REST status name, e.g. `INTERNAL_SERVER_ERROR`.
    pub status: String,
}

impl ShardFailure {
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("index".into(), json!(self.index));
        m.insert("index_uuid".into(), json!(self.index_uuid));
        m.insert("shard_id".into(), json!(self.shard_id));
        m.insert("reason".into(), json!(self.reason));
        if let Some(n) = &self.node_id {
            m.insert("node_id".into(), json!(n));
        }
        m.insert("status".into(), json!(self.status));
        Value::Object(m)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureState {
    pub feature_name: String,
    #[serde(default)]
    pub indices: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct IndexDetails {
    pub shard_count: u32,
    pub size_in_bytes: u64,
    pub max_segments_per_shard: u32,
}

/// ES `SnapshotInfo`. Also the on-disk content of root `snap-{uuid}.dat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub repository: String,
    pub snapshot: String,
    pub uuid: String,
    #[serde(default)]
    pub version_id: Option<u32>,
    #[serde(default)]
    pub version: Option<String>,
    pub indices: Vec<String>,
    #[serde(default)]
    pub data_streams: Vec<String>,
    #[serde(default)]
    pub feature_states: Vec<FeatureState>,
    #[serde(default)]
    pub include_global_state: Option<bool>,
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
    pub state: Option<SnapshotState>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub start_time: i64,
    #[serde(default)]
    pub end_time: i64,
    #[serde(default)]
    pub total_shards: u32,
    #[serde(default)]
    pub successful_shards: u32,
    #[serde(default)]
    pub failures: Vec<ShardFailure>,
    #[serde(default)]
    pub index_details: BTreeMap<String, IndexDetails>,
}

/// Rendering flags (REST params of `GET _snapshot`).
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    pub verbose: bool,
    pub include_repository: bool,
    pub index_names: bool,
    pub index_details: bool,
    pub human: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self { verbose: true, include_repository: true, index_names: true, index_details: false, human: false }
    }
}

impl SnapshotInfo {
    pub fn failed_shards(&self) -> u32 {
        self.total_shards.saturating_sub(self.successful_shards)
    }

    pub fn duration_millis(&self) -> i64 {
        (self.end_time - self.start_time).max(0)
    }

    pub fn snapshot_ref(&self) -> String {
        format!("{}:{}/{}", self.repository, self.snapshot, self.uuid)
    }

    /// SLM policy id from `metadata.policy`.
    pub fn slm_policy(&self) -> Option<&str> {
        self.metadata.as_ref()?.get("policy")?.as_str()
    }

    /// ES `SnapshotInfo.toXContentExternal`.
    pub fn render(&self, opts: RenderOptions) -> Value {
        let verbose = opts.verbose;
        let mut m = Map::new();
        m.insert("snapshot".into(), json!(self.snapshot));
        m.insert("uuid".into(), json!(self.uuid));
        if opts.include_repository {
            m.insert("repository".into(), json!(self.repository));
        }
        if let (Some(id), Some(v)) = (self.version_id, &self.version) {
            m.insert("version_id".into(), json!(id));
            m.insert("version".into(), json!(v));
        }
        if opts.index_names {
            m.insert("indices".into(), json!(self.indices));
        }
        if verbose || !self.data_streams.is_empty() {
            m.insert("data_streams".into(), json!(self.data_streams));
        }
        if let Some(g) = self.include_global_state {
            m.insert("include_global_state".into(), json!(g));
        }
        if let Some(md) = &self.metadata {
            m.insert("metadata".into(), Value::Object(md.clone()));
        }
        if let Some(s) = self.state {
            m.insert("state".into(), json!(s.as_str()));
        }
        if let Some(r) = &self.reason {
            m.insert("reason".into(), json!(r));
        }
        if verbose || self.start_time != 0 {
            m.insert("start_time".into(), json!(format_millis(self.start_time)));
            m.insert("start_time_in_millis".into(), json!(self.start_time));
        }
        if verbose || self.end_time != 0 {
            m.insert("end_time".into(), json!(format_millis(self.end_time)));
            m.insert("end_time_in_millis".into(), json!(self.end_time));
            let d = if self.end_time == 0 { 0 } else { self.duration_millis() };
            if opts.human {
                m.insert("duration".into(), json!(format_duration_millis(d)));
            }
            m.insert("duration_in_millis".into(), json!(d));
        }
        if verbose || !self.failures.is_empty() {
            m.insert("failures".into(), Value::Array(self.failures.iter().map(|f| f.to_json()).collect()));
        }
        if verbose || self.total_shards != 0 {
            m.insert(
                "shards".into(),
                json!({"total": self.total_shards, "failed": self.failed_shards(), "successful": self.successful_shards}),
            );
        }
        if verbose || !self.feature_states.is_empty() {
            m.insert(
                "feature_states".into(),
                Value::Array(
                    self.feature_states
                        .iter()
                        .map(|f| json!({"feature_name": f.feature_name, "indices": f.indices}))
                        .collect(),
                ),
            );
        }
        if opts.index_details && verbose {
            let mut d = Map::new();
            for (k, v) in &self.index_details {
                let mut e = Map::new();
                e.insert("shard_count".into(), json!(v.shard_count));
                if opts.human {
                    e.insert("size".into(), json!(crate::util::human_bytes(v.size_in_bytes)));
                }
                e.insert("size_in_bytes".into(), json!(v.size_in_bytes));
                e.insert("max_segments_per_shard".into(), json!(v.max_segments_per_shard));
                d.insert(k.clone(), Value::Object(e));
            }
            m.insert("index_details".into(), Value::Object(d));
        }
        Value::Object(m)
    }

    /// `PUT _snapshot/{repo}/{snap}?wait_for_completion=true` response body.
    pub fn create_response(&self) -> Value {
        json!({ "snapshot": self.render(RenderOptions { include_repository: false, ..Default::default() }) })
    }
}

// ---------------------------------------------------------------------------
// Live progress / `_status`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ShardStage {
    Init,
    Started,
    Finalize,
    Done,
    Failure,
}

impl ShardStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Init => "INIT",
            Self::Started => "STARTED",
            Self::Finalize => "FINALIZE",
            Self::Done => "DONE",
            Self::Failure => "FAILURE",
        }
    }
}

/// ES `SnapshotStats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotStats {
    pub incremental_file_count: u64,
    pub incremental_size: u64,
    pub processed_file_count: u64,
    pub processed_size: u64,
    pub total_file_count: u64,
    pub total_size: u64,
    pub start_time: i64,
    pub time: i64,
}

impl SnapshotStats {
    /// ES `SnapshotStats.add`: sums counters and widens the time window.
    pub fn add(&mut self, o: &SnapshotStats) {
        let was_empty = self.start_time == 0;
        self.incremental_file_count += o.incremental_file_count;
        self.incremental_size += o.incremental_size;
        self.processed_file_count += o.processed_file_count;
        self.processed_size += o.processed_size;
        self.total_file_count += o.total_file_count;
        self.total_size += o.total_size;
        if o.start_time == 0 {
            return;
        }
        if was_empty {
            self.start_time = o.start_time;
            self.time = o.time;
        } else {
            let end = (self.start_time + self.time).max(o.start_time + o.time);
            self.start_time = self.start_time.min(o.start_time);
            self.time = end - self.start_time;
        }
    }

    pub fn to_json(&self, human: bool) -> Value {
        let mut m = Map::new();
        let block = |count: u64, size: u64| {
            let mut b = Map::new();
            b.insert("file_count".into(), json!(count));
            if human {
                b.insert("size".into(), json!(crate::util::human_bytes(size)));
            }
            b.insert("size_in_bytes".into(), json!(size));
            Value::Object(b)
        };
        m.insert("incremental".into(), block(self.incremental_file_count, self.incremental_size));
        if self.processed_file_count != self.incremental_file_count {
            m.insert("processed".into(), block(self.processed_file_count, self.processed_size));
        }
        m.insert("total".into(), block(self.total_file_count, self.total_size));
        m.insert("start_time_in_millis".into(), json!(self.start_time));
        if human {
            m.insert("time".into(), json!(format_duration_millis(self.time)));
        }
        m.insert("time_in_millis".into(), json!(self.time));
        Value::Object(m)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardStatus {
    pub stage: ShardStage,
    pub stats: SnapshotStats,
    pub node: Option<String>,
    pub reason: Option<String>,
}

impl ShardStatus {
    pub fn new(stage: ShardStage) -> Self {
        Self { stage, stats: SnapshotStats::default(), node: None, reason: None }
    }

    pub fn to_json(&self, human: bool) -> Value {
        let mut m = Map::new();
        m.insert("stage".into(), json!(self.stage.as_str()));
        m.insert("stats".into(), self.stats.to_json(human));
        if let Some(n) = &self.node {
            m.insert("node".into(), json!(n));
        }
        if let Some(r) = &self.reason {
            m.insert("reason".into(), json!(r));
        }
        Value::Object(m)
    }
}

/// `SnapshotShardsStats`.
pub fn shards_stats_json<'a>(shards: impl Iterator<Item = &'a ShardStatus>) -> Value {
    let (mut init, mut started, mut fin, mut done, mut failed, mut total) = (0, 0, 0, 0, 0, 0);
    for s in shards {
        total += 1;
        match s.stage {
            ShardStage::Init => init += 1,
            ShardStage::Started => started += 1,
            ShardStage::Finalize => fin += 1,
            ShardStage::Done => done += 1,
            ShardStage::Failure => failed += 1,
        }
    }
    json!({"initializing": init, "started": started, "finalizing": fin, "done": done, "failed": failed, "total": total})
}

/// One entry of the `_snapshot/{repo}/{snap}/_status` response.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotStatus {
    pub snapshot: String,
    pub repository: String,
    pub uuid: String,
    /// `STARTED`, `SUCCESS`, `FAILED`, `ABORTED`, `INIT` (SnapshotsInProgress.State names).
    pub state: String,
    pub include_global_state: bool,
    /// index → shard → status.
    pub indices: BTreeMap<String, BTreeMap<u32, ShardStatus>>,
    /// Overall start/time if known (otherwise derived from shards).
    pub start_time: i64,
    pub time: i64,
}

impl SnapshotStatus {
    pub fn to_json(&self, human: bool) -> Value {
        let mut overall = SnapshotStats::default();
        let mut indices = Map::new();
        for (name, shards) in &self.indices {
            let mut istats = SnapshotStats::default();
            let mut sh = Map::new();
            for (id, s) in shards {
                istats.add(&s.stats);
                sh.insert(id.to_string(), s.to_json(human));
            }
            overall.add(&istats);
            indices.insert(
                name.clone(),
                json!({
                    "shards_stats": shards_stats_json(shards.values()),
                    "stats": istats.to_json(human),
                    "shards": Value::Object(sh),
                }),
            );
        }
        if self.start_time != 0 {
            overall.start_time = self.start_time;
            overall.time = self.time;
        }
        let mut m = Map::new();
        m.insert("snapshot".into(), json!(self.snapshot));
        m.insert("repository".into(), json!(self.repository));
        m.insert("uuid".into(), json!(self.uuid));
        m.insert("state".into(), json!(self.state));
        m.insert("include_global_state".into(), json!(self.include_global_state));
        m.insert("shards_stats".into(), shards_stats_json(self.indices.values().flat_map(|s| s.values())));
        m.insert("stats".into(), overall.to_json(human));
        m.insert("indices".into(), Value::Object(indices));
        Value::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> SnapshotInfo {
        SnapshotInfo {
            repository: "repo".into(),
            snapshot: "snap".into(),
            uuid: "u1".into(),
            version_id: Some(ES_VERSION_ID),
            version: Some(ES_VERSION.into()),
            indices: vec!["idx".into()],
            data_streams: vec![],
            feature_states: vec![],
            include_global_state: Some(true),
            metadata: None,
            state: Some(SnapshotState::Success),
            reason: None,
            start_time: 1594072518129,
            end_time: 1594072519129,
            total_shards: 2,
            successful_shards: 2,
            failures: vec![],
            index_details: BTreeMap::new(),
        }
    }

    #[test]
    fn verbose_shape() {
        let v = info().render(RenderOptions::default());
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "snapshot",
                "uuid",
                "repository",
                "version_id",
                "version",
                "indices",
                "data_streams",
                "include_global_state",
                "state",
                "start_time",
                "start_time_in_millis",
                "end_time",
                "end_time_in_millis",
                "duration_in_millis",
                "failures",
                "shards",
                "feature_states"
            ]
        );
        assert_eq!(v["start_time"], "2020-07-06T21:55:18.129Z");
        assert_eq!(v["duration_in_millis"], 1000);
        assert_eq!(v["shards"], json!({"total":2,"failed":0,"successful":2}));
    }

    #[test]
    fn non_verbose_shape() {
        let mut i = info();
        i.version = None;
        i.version_id = None;
        i.include_global_state = None;
        i.start_time = 0;
        i.end_time = 0;
        i.total_shards = 0;
        i.successful_shards = 0;
        let v = i.render(RenderOptions { verbose: false, ..Default::default() });
        assert_eq!(v, json!({"snapshot":"snap","uuid":"u1","repository":"repo","indices":["idx"],"state":"SUCCESS"}));
    }

    #[test]
    fn stats_add() {
        let mut a = SnapshotStats { start_time: 100, time: 50, total_file_count: 1, ..Default::default() };
        a.add(&SnapshotStats { start_time: 120, time: 100, total_file_count: 2, ..Default::default() });
        assert_eq!((a.start_time, a.time, a.total_file_count), (100, 120, 3));
    }
}
