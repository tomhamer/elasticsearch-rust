//! Cross-cluster replication (x-pack `ccr`).
//!
//! * **Follower indices** (`PUT /{index}/_ccr/follow`): the follower is
//!   created from a copy of every leader shard's files (ES restores it from
//!   the `_ccr_<alias>` repository), with the leader's settings, mappings
//!   and aliases, the `index.xpack.ccr.following_index` setting and CCR
//!   metadata (leader alias / name / uuid / shard history uuids). A
//!   retention lease on every leader shard keeps the history the follower
//!   still needs.
//! * **Shard follow tasks** (one per follower shard while following, ES
//!   `ShardFollowTask`): read the leader shard's operations by seq_no from
//!   its history ([`Shard::changes_range`](crate::shard::Shard)) up to the
//!   leader's global checkpoint and apply them to the follower shard with
//!   the leader's `_id`, `_seq_no` and `_version` (ES
//!   `TransportBulkShardOperationsAction`), keep the follower's mapping,
//!   replicated settings and aliases in line with the leader's, and renew
//!   the leader retention lease. Their state is persisted
//!   (`<data>/_state/ccr.json`) and they resume on restart.
//! * **Follower write block**: a following index rejects every write that
//!   does not come from its follow task (ES `FollowingEngine`).
//! * **Auto-follow patterns**: new leader indices matching a pattern (and
//!   no exclusion pattern) are followed automatically.
//!
//! The leader is reached through [`remote::LeaderCluster`].

pub mod remote;
mod task;

use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::node::Node;
use crate::settings::FlatSettings;
use parking_lot::Mutex;
use remote::LeaderCluster;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use task::follow_stats_json;

/// `CcrSettings.CCR_FOLLOWING_INDEX_SETTING`.
pub const FOLLOWING_INDEX_SETTING: &str = "index.xpack.ccr.following_index";

/// `FollowingEngine.preFlight`: a following index accepts only operations
/// that carry the leader's sequence number.
pub fn following_write_error() -> EsError {
    EsError::new(403, "status_exception", "a following engine does not accept operations without an assigned sequence number")
}

/// `AutoFollowCoordinator.AUTO_FOLLOW_PATTERN_REPLACEMENT`.
pub const AUTO_FOLLOW_PATTERN_REPLACEMENT: &str = "{{leader_index}}";

// ---------------------------------------------------------------------------
// Follow parameters
// ---------------------------------------------------------------------------

const DEFAULT_MAX_READ_REQUEST_OPERATION_COUNT: i64 = 5120;
const DEFAULT_MAX_WRITE_REQUEST_OPERATION_COUNT: i64 = 5120;
const DEFAULT_MAX_OUTSTANDING_READ_REQUESTS: i64 = 12;
const DEFAULT_MAX_OUTSTANDING_WRITE_REQUESTS: i64 = 9;
const DEFAULT_MAX_WRITE_BUFFER_COUNT: i64 = i32::MAX as i64;
const RETRY_DELAY_MAX_MILLIS: i64 = 5 * 60 * 1000;

/// A `ByteSizeValue`: bytes and its string representation.
pub type Size = (u64, String);
/// A `TimeValue`: millis and its string representation.
pub type Time = (i64, String);

/// `FollowParameters`: every field optional (unset = the default).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FollowParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_read_request_operation_count: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_write_request_operation_count: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_outstanding_read_requests: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_outstanding_write_requests: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_read_request_size: Option<Size>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_write_request_size: Option<Size>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_write_buffer_count: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_write_buffer_size: Option<Size>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retry_delay: Option<Time>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_poll_timeout: Option<Time>,
}

pub(crate) const PARAM_FIELDS: [&str; 10] = [
    "max_read_request_operation_count",
    "max_write_request_operation_count",
    "max_outstanding_read_requests",
    "max_outstanding_write_requests",
    "max_read_request_size",
    "max_write_request_size",
    "max_write_buffer_count",
    "max_write_buffer_size",
    "max_retry_delay",
    "read_poll_timeout",
];

fn parse_int(parser: &str, k: &str, v: &Value) -> Result<i64> {
    let n = match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => match s.trim().parse::<i64>() {
            Ok(n) => Some(n),
            Err(_) => return Err(EsError::field_failure(parser, k, None, EsError::number_format(s).at_key(k, None))),
        },
        _ => return Err(EsError::unsupported_type_xc(parser, k, v, None)),
    };
    match n {
        Some(n) if n >= i32::MIN as i64 && n <= i32::MAX as i64 => Ok(n),
        _ => Err(EsError::field_failure(parser, k, None, EsError::new(400, "json_parse_exception", format!("Numeric value ({v}) out of range of int (-2147483648 - 2147483647)")).at_key(k, None))),
    }
}

fn parse_size_field(parser: &str, k: &str, v: &Value) -> Result<Size> {
    let s = match v {
        Value::String(s) => s.clone(),
        _ => return Err(EsError::unsupported_type_xc(parser, k, v, None)),
    };
    crate::ilm::parse_size(&s, k).map_err(|e| EsError::field_failure(parser, k, None, e.at_key(k, None)))
}

fn parse_time_field(parser: &str, k: &str, v: &Value) -> Result<Time> {
    let s = match v {
        Value::String(s) => s.clone(),
        _ => return Err(EsError::unsupported_type_xc(parser, k, v, None)),
    };
    let t = crate::ilm::parse_time(&s, k).map_err(|e| {
        // `TimeValue.parseTimeValue` fails with an IllegalArgumentException.
        let cause = EsError::illegal_argument(e.reason.clone()).at_key(k, None);
        EsError::field_failure(parser, k, None, cause)
    })?;
    Ok((t.millis(), t.rep))
}

impl FollowParams {
    /// Parse one body field; `false` when `k` is not a follow parameter.
    pub fn set(&mut self, parser: &str, k: &str, v: &Value) -> Result<bool> {
        match k {
            "max_read_request_operation_count" => self.max_read_request_operation_count = Some(parse_int(parser, k, v)?),
            "max_write_request_operation_count" => self.max_write_request_operation_count = Some(parse_int(parser, k, v)?),
            "max_outstanding_read_requests" => self.max_outstanding_read_requests = Some(parse_int(parser, k, v)?),
            "max_outstanding_write_requests" => self.max_outstanding_write_requests = Some(parse_int(parser, k, v)?),
            "max_read_request_size" => self.max_read_request_size = Some(parse_size_field(parser, k, v)?),
            "max_write_request_size" => self.max_write_request_size = Some(parse_size_field(parser, k, v)?),
            "max_write_buffer_count" => self.max_write_buffer_count = Some(parse_int(parser, k, v)?),
            "max_write_buffer_size" => self.max_write_buffer_size = Some(parse_size_field(parser, k, v)?),
            "max_retry_delay" => self.max_retry_delay = Some(parse_time_field(parser, k, v)?),
            "read_poll_timeout" => self.read_poll_timeout = Some(parse_time_field(parser, k, v)?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// `FollowParameters.validate` messages, in ES order.
    pub fn validate(&self, errs: &mut Vec<String>) {
        let pos = |v: Option<i64>, name: &str, errs: &mut Vec<String>| {
            if v.is_some_and(|n| n < 1) {
                errs.push(format!("{name} must be larger than 0"));
            }
        };
        let pos_size = |v: &Option<Size>, name: &str, errs: &mut Vec<String>| {
            if v.as_ref().is_some_and(|s| s.0 == 0) {
                errs.push(format!("{name} must be larger than 0"));
            }
        };
        pos(self.max_read_request_operation_count, "max_read_request_operation_count", errs);
        pos_size(&self.max_read_request_size, "max_read_request_size", errs);
        pos(self.max_outstanding_read_requests, "max_outstanding_read_requests", errs);
        pos(self.max_write_request_operation_count, "max_write_request_operation_count", errs);
        pos_size(&self.max_write_request_size, "max_write_request_size", errs);
        pos(self.max_outstanding_write_requests, "max_outstanding_write_requests", errs);
        pos(self.max_write_buffer_count, "max_write_buffer_count", errs);
        pos_size(&self.max_write_buffer_size, "max_write_buffer_size", errs);
        if let Some((ms, rep)) = &self.max_retry_delay {
            if *ms <= 0 {
                errs.push(format!("[max_retry_delay] must be positive but was [{rep}]"));
            }
            if *ms > RETRY_DELAY_MAX_MILLIS {
                errs.push(format!("[max_retry_delay] must be less than [5m] but was [{rep}]"));
            }
        }
    }

    /// `FollowParameters.toXContentFragment` (set fields only).
    pub fn fragment(&self, m: &mut Map<String, Value>) {
        let mut put = |k: &str, v: Option<Value>| {
            if let Some(v) = v {
                m.insert(k.into(), v);
            }
        };
        put("max_read_request_operation_count", self.max_read_request_operation_count.map(Value::from));
        put("max_write_request_operation_count", self.max_write_request_operation_count.map(Value::from));
        put("max_outstanding_read_requests", self.max_outstanding_read_requests.map(Value::from));
        put("max_outstanding_write_requests", self.max_outstanding_write_requests.map(Value::from));
        put("max_read_request_size", self.max_read_request_size.as_ref().map(|s| json!(s.1)));
        put("max_write_request_size", self.max_write_request_size.as_ref().map(|s| json!(s.1)));
        put("max_write_buffer_count", self.max_write_buffer_count.map(Value::from));
        put("max_write_buffer_size", self.max_write_buffer_size.as_ref().map(|s| json!(s.1)));
        put("max_retry_delay", self.max_retry_delay.as_ref().map(|s| json!(s.1)));
        put("read_poll_timeout", self.read_poll_timeout.as_ref().map(|s| json!(s.1)));
    }

    /// The `ShardFollowTask` parameters: every unset field at its default
    /// (`TransportResumeFollowAction.createShardFollowTask`).
    pub fn resolve(&self) -> FollowParams {
        FollowParams {
            max_read_request_operation_count: Some(self.max_read_request_operation_count.unwrap_or(DEFAULT_MAX_READ_REQUEST_OPERATION_COUNT)),
            max_write_request_operation_count: Some(self.max_write_request_operation_count.unwrap_or(DEFAULT_MAX_WRITE_REQUEST_OPERATION_COUNT)),
            max_outstanding_read_requests: Some(self.max_outstanding_read_requests.unwrap_or(DEFAULT_MAX_OUTSTANDING_READ_REQUESTS)),
            max_outstanding_write_requests: Some(self.max_outstanding_write_requests.unwrap_or(DEFAULT_MAX_OUTSTANDING_WRITE_REQUESTS)),
            max_read_request_size: Some(self.max_read_request_size.clone().unwrap_or((32 << 20, "32mb".into()))),
            max_write_request_size: Some(self.max_write_request_size.clone().unwrap_or((i64::MAX as u64, format!("{}b", i64::MAX)))),
            max_write_buffer_count: Some(self.max_write_buffer_count.unwrap_or(DEFAULT_MAX_WRITE_BUFFER_COUNT)),
            max_write_buffer_size: Some(self.max_write_buffer_size.clone().unwrap_or((512 << 20, "512mb".into()))),
            max_retry_delay: Some(self.max_retry_delay.clone().unwrap_or((500, "500ms".into()))),
            read_poll_timeout: Some(self.read_poll_timeout.clone().unwrap_or((60_000, "1m".into()))),
        }
    }

    pub(crate) fn read_ops(&self) -> usize {
        self.max_read_request_operation_count.unwrap_or(DEFAULT_MAX_READ_REQUEST_OPERATION_COUNT).max(1) as usize
    }
    pub(crate) fn read_bytes(&self) -> u64 {
        self.max_read_request_size.as_ref().map(|s| s.0).unwrap_or(32 << 20).max(1)
    }
    pub(crate) fn write_ops(&self) -> usize {
        self.max_write_request_operation_count.unwrap_or(DEFAULT_MAX_WRITE_REQUEST_OPERATION_COUNT).max(1) as usize
    }
    pub(crate) fn write_bytes(&self) -> u64 {
        self.max_write_request_size.as_ref().map(|s| s.0).unwrap_or(i64::MAX as u64).max(1)
    }
    pub(crate) fn retry_delay_ms(&self) -> i64 {
        self.max_retry_delay.as_ref().map(|s| s.0).unwrap_or(500).max(1)
    }
    pub(crate) fn poll_timeout_ms(&self) -> i64 {
        self.read_poll_timeout.as_ref().map(|s| s.0).unwrap_or(60_000).max(1)
    }
}

// ---------------------------------------------------------------------------
// Settings filtering (TransportResumeFollowAction)
// ---------------------------------------------------------------------------

/// Settings that are never replicated and may differ between leader and
/// follower (`NON_REPLICATED_SETTINGS`, plus the per-index identity ones).
fn non_replicated(key: &str) -> bool {
    const EXACT: &[&str] = &[
        FOLLOWING_INDEX_SETTING,
        "index.soft_deletes.enabled",
        "index.version.created",
        "index.uuid",
        "index.history.uuid",
        "index.provided_name",
        "index.creation_date",
        "index.version.upgraded",
        "index.version.upgraded_string",
        "index.number_of_replicas",
        "index.auto_expand_replicas",
        "index.blocks.read_only",
        "index.blocks.read",
        "index.blocks.write",
        "index.blocks.metadata",
        "index.blocks.read_only_allow_delete",
        "index.priority",
        "index.write.wait_for_active_shards",
        "index.hidden",
        "index.routing.rebalance.enable",
        "index.routing.allocation.enable",
        "index.routing.allocation.total_shards_per_node",
        "index.allocation.max_retries",
        "index.unassigned.node_left.delayed_timeout",
        "index.max_result_window",
        "index.warmer.enabled",
        "index.refresh_interval",
        "index.max_rescore_window",
        "index.max_inner_result_window",
        "index.query.default_field",
        "index.query_string.lenient",
        "indices.query.query_string.analyze_wildcard",
        "indices.query.query_string.allowLeadingWildcard",
        "index.query.parse.allow_unmapped_fields",
        "index.search.idle.after",
        "index.soft_deletes.retention.operations",
        "index.soft_deletes.retention_lease.period",
        "index.max_script_fields",
        "index.max_regex_length",
        "index.max_terms_count",
        "index.highlight.max_analyzed_offset",
        "index.highlight.weight_matches_mode.enabled",
        "index.max_docvalue_fields_search",
        "index.analyze.max_token_count",
        "index.analyze.max_char_count",
        "index.max_slices_per_scroll",
        "index.default_pipeline",
        "index.final_pipeline",
        "index.search.throttled",
        "index.translog.retention.age",
        "index.translog.retention.size",
        "index.translog.generation_threshold_size",
        "index.translog.flush_threshold_size",
        "index.translog.flush_threshold_age",
        "index.translog.durability",
        "index.translog.sync_interval",
        "index.flush_after_merge",
        "index.gc_deletes",
        "index.max_refresh_listeners",
        "index.requests.cache.enable",
        "index.load_fixed_bitset_filters_eagerly",
        "index.search.slowlog.include.user",
        "index.indexing.slowlog.reformat",
        "index.indexing.slowlog.source",
        "index.indexing.slowlog.include.user",
        "index.compound_format",
        "index.codec",
        "index.routing.allocation.include._tier_preference",
        "index.bloom_filter_for_id_field.enabled",
        "index.verified_read_only",
    ];
    const PREFIX: &[&str] = &[
        "index.routing.allocation.exclude.",
        "index.routing.allocation.include.",
        "index.routing.allocation.require.",
        "index.search.slowlog.threshold.",
        "index.indexing.slowlog.threshold.",
        "index.merge.policy.",
        "index.merge.scheduler.",
    ];
    EXACT.contains(&key) || PREFIX.iter().any(|p| key.starts_with(p))
}

/// `TransportResumeFollowAction.filter`: the settings that must match.
pub fn filter_settings(s: &FlatSettings) -> FlatSettings {
    s.iter().filter(|(k, _)| !non_replicated(k)).map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// `Settings.toString()`.
fn settings_string(s: &FlatSettings) -> String {
    let m: Map<String, Value> = s.iter().map(|(k, v)| (k.clone(), Value::String(setting_str(v)))).collect();
    Value::Object(m).to_string()
}

fn setting_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(setting_str).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

/// Is `key` a registered index-scope setting (`IndexScopedSettings.get`)?
fn known_index_setting(key: &str) -> bool {
    static KNOWN: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    let known = KNOWN.get_or_init(|| crate::settings::index_defaults(&FlatSettings::new()).into_keys().collect());
    const GROUPS: &[&str] = &[
        "index.analysis.",
        "index.similarity.",
        "index.routing.allocation.",
        "index.sort.",
        "index.mapping.",
        "index.search.slowlog.",
        "index.indexing.slowlog.",
        "index.merge.",
        "index.store.",
        "index.translog.",
        "index.blocks.",
        "index.lifecycle.",
        "index.query.",
        "index.highlight.",
        "index.time_series.",
        "index.routing_path",
        "index.dimensions",
        "index.version.",
        "index.history.",
        "index.xpack.",
        "index.frozen",
        "index.verified_before_close",
        "index.downsample.",
        "index.resize.",
        "index.shrink.",
        "index.plugin.",
        "index.look_ahead_time",
        "index.look_back_time",
    ];
    const CORE: &[&str] = &[
        "index.number_of_shards",
        "index.number_of_routing_shards",
        "index.routing_partition_size",
        "index.mode",
        "index.shard.check_on_startup",
        "index.format",
        "index.load_fixed_bitset_filters_eagerly",
        "index.max_ngram_diff",
        "index.max_shingle_diff",
        "index.max_adjacency_matrix_filters",
        "index.data_path",
        "index.write.wait_for_active_shards",
    ];
    known.contains(key) || CORE.contains(&key) || GROUPS.iter().any(|g| key.starts_with(g)) || non_replicated(key)
}

/// Request `settings`: flattened, values as strings.
fn parse_settings(parser: &str, v: &Value) -> Result<FlatSettings> {
    match v {
        Value::Object(_) => Ok(crate::settings::flatten(v).into_iter().map(|(k, v)| {
            let v = match v {
                Value::String(_) | Value::Array(_) | Value::Null => v,
                other => Value::String(other.to_string()),
            };
            (k, v)
        }).collect()),
        _ => Err(EsError::unsupported_type_xc(parser, "settings", v, None)),
    }
}

// ---------------------------------------------------------------------------
// Persistent state
// ---------------------------------------------------------------------------

/// `AutoFollowMetadata.AutoFollowPattern`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoFollowPattern {
    pub remote_cluster: String,
    pub leader_index_patterns: Vec<String>,
    #[serde(default)]
    pub leader_index_exclusion_patterns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow_index_pattern: Option<String>,
    #[serde(default)]
    pub settings: FlatSettings,
    pub active: bool,
    #[serde(default)]
    pub params: FollowParams,
}

impl AutoFollowPattern {
    /// `AutoFollowPattern.toXContent`.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("active".into(), json!(self.active));
        m.insert("remote_cluster".into(), json!(self.remote_cluster));
        m.insert("leader_index_patterns".into(), json!(self.leader_index_patterns));
        m.insert("leader_index_exclusion_patterns".into(), json!(self.leader_index_exclusion_patterns));
        if let Some(p) = &self.follow_index_pattern {
            m.insert("follow_index_pattern".into(), json!(p));
        }
        if !self.settings.is_empty() {
            m.insert("settings".into(), crate::settings::unflatten(&self.settings));
        }
        self.params.fragment(&mut m);
        Value::Object(m)
    }

    /// `AutoFollowPattern.match` for a leader index (or its data stream).
    fn matches(&self, name: &str, data_stream: Option<&str>) -> bool {
        auto_follow_match(&self.leader_index_patterns, &self.leader_index_exclusion_patterns, name, data_stream)
    }
}

fn simple_match_any(patterns: &[String], name: &str) -> bool {
    patterns.iter().any(|p| crate::mapping::wildcard_match(p, name))
}

/// `AutoFollowPattern.match`.
fn auto_follow_match(patterns: &[String], exclusions: &[String], name: &str, data_stream: Option<&str>) -> bool {
    if !is_system_index(name) && !simple_match_any(exclusions, name) && simple_match_any(patterns, name) {
        return true;
    }
    match data_stream {
        Some(ds) => !ds.starts_with('.') && !simple_match_any(exclusions, ds) && simple_match_any(patterns, ds),
        None => false,
    }
}

/// Indices owned by a system feature (never auto-followed).
fn is_system_index(name: &str) -> bool {
    crate::system_indices::is_system(name)
}

/// The CCR custom metadata of a follower index (`Ccr.CCR_CUSTOM_METADATA_KEY`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FollowerMeta {
    pub follower_index: String,
    pub remote_cluster: String,
    pub leader_index: String,
    pub leader_uuid: String,
    pub leader_history_uuids: Vec<String>,
    #[serde(default)]
    pub created_millis: i64,
}

impl FollowerMeta {
    /// The custom metadata map as ES stores it (`ccr` in the index metadata).
    pub fn custom_json(&self) -> Value {
        json!({
            "leader_index_shard_history_uuids": self.leader_history_uuids.join(","),
            "leader_index_uuid": self.leader_uuid,
            "leader_index_name": self.leader_index,
            "remote_cluster_name": self.remote_cluster,
        })
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Persisted {
    #[serde(default)]
    pub patterns: BTreeMap<String, AutoFollowPattern>,
    /// Pattern → leader index uuids already (auto-)followed.
    #[serde(default)]
    pub followed: BTreeMap<String, Vec<String>>,
    /// Follower index uuid → its CCR metadata.
    #[serde(default)]
    pub followers: BTreeMap<String, FollowerMeta>,
    /// Follower index uuid → the resolved parameters of its shard follow
    /// tasks (present while following, absent while paused).
    #[serde(default)]
    pub tasks: BTreeMap<String, FollowParams>,
}

impl Persisted {
    fn has_work(&self) -> bool {
        !self.tasks.is_empty() || self.patterns.values().any(|p| p.active)
    }
}

type LeaderConnector = fn(&Arc<Node>, &str) -> Result<LeaderCluster>;

/// The per-node CCR service.
pub struct CcrService {
    path: PathBuf,
    // Explicit dependency for internal adapter tests; production never substitutes local indices.
    connector: parking_lot::RwLock<LeaderConnector>,
    pub(crate) st: Mutex<Persisted>,
    pub(crate) rt: task::Runtime,
    worker: AtomicBool,
    /// A cluster's way of changing a follower's metadata (`None`: in place).
    pub(crate) metadata_hook: parking_lot::RwLock<Option<FollowerMetadataHook>>,
}

/// What a follower's metadata sync changes (the leader's mapping, the
/// replicated settings that differ, the aliases).
#[derive(Clone, Debug, Default)]
pub struct FollowerMetadataChange {
    pub mapping: Option<Value>,
    pub settings: FlatSettings,
    /// Non-dynamic settings among them: applied with the index reopened.
    pub reopen: bool,
    pub aliases: Option<BTreeMap<String, Value>>,
}

/// The error kind a [`FollowerMetadataHook`] answers while the change is
/// being committed: the sync treats it as neither applied nor failed.
pub const FOLLOWER_METADATA_PENDING: &str = "ccr_follower_metadata_pending";

/// Applies a [`FollowerMetadataChange`] to the follower index named by its
/// first argument and returns once it is committed (a cluster commits it
/// through its master before any engine holds it).
pub type FollowerMetadataHook = Arc<dyn Fn(&str, FollowerMetadataChange) -> Result<()> + Send + Sync>;

/// Install (or remove) `node`'s [`FollowerMetadataHook`].
pub fn set_follower_metadata_hook(node: &Node, hook: Option<FollowerMetadataHook>) {
    *service(node).metadata_hook.write() = hook;
}

/// Apply `change` to follower `name` on `node`'s engine, in place.
pub fn apply_follower_change(node: &Node, name: &str, change: &FollowerMetadataChange) -> Result<()> {
    let fsvc = node.get_index(name).ok_or_else(|| EsError::index_not_found(name))?;
    task::apply_follower_in_place(&fsvc, change)
}

/// Services by node (data path + node identity: a node reopened on the
/// same path in this process gets a fresh service loaded from disk).
type RegKey = (PathBuf, usize);

fn registry() -> &'static Mutex<HashMap<RegKey, Arc<CcrService>>> {
    static R: OnceLock<Mutex<HashMap<RegKey, Arc<CcrService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn reg_key(node: &Node) -> RegKey {
    (node.data.clone(), node as *const Node as usize)
}

fn state_path(data: &Path) -> PathBuf {
    data.join("_state").join("ccr.json")
}

/// The node's CCR service (loaded on first use).
pub fn service(node: &Node) -> Arc<CcrService> {
    let mut r = registry().lock();
    r.entry(reg_key(node))
        .or_insert_with(|| {
            let path = state_path(&node.data);
            let st: Persisted = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
            Arc::new(CcrService { path, connector: parking_lot::RwLock::new(LeaderCluster::connect), st: Mutex::new(st), rt: task::Runtime::default(), worker: AtomicBool::new(false), metadata_hook: parking_lot::RwLock::new(None) })
        })
        .clone()
}

/// The service when CCR was used on this node (never creates one).
fn existing(node: &Node) -> Option<Arc<CcrService>> {
    if let Some(s) = registry().lock().get(&reg_key(node)) {
        return Some(s.clone());
    }
    if state_path(&node.data).exists() {
        return Some(service(node));
    }
    None
}

/// Resume persisted shard follow tasks and auto-follow patterns (node start).
pub fn start(node: &Arc<Node>) {
    if let Some(s) = existing(node) {
        s.ensure_worker(node);
    }
}

impl CcrService {
    fn connect(&self, node: &Arc<Node>, alias: &str) -> Result<LeaderCluster> {
        let connect = *self.connector.read();
        connect(node, alias)
    }

    pub(crate) fn persist(&self, st: &Persisted) -> Result<()> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        crate::hooks::write_durable(&self.path, &serde_json::to_vec_pretty(st)?)?;
        Ok(())
    }

    /// Start the follow/auto-follow thread unless it runs.
    pub(crate) fn ensure_worker(self: &Arc<Self>, node: &Arc<Node>) {
        if !self.st.lock().has_work() {
            return;
        }
        if self.worker.swap(true, Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(node);
        let key = reg_key(node);
        let me = self.clone();
        let spawned = std::thread::Builder::new().name("es-ccr".into()).spawn(move || task::worker_loop(weak, key, me));
        if spawned.is_err() {
            self.worker.store(false, Ordering::Release);
        }
    }

    /// Called by the worker when it is about to exit for lack of work.
    pub(crate) fn worker_idle_exit(&self) -> bool {
        let st = self.st.lock();
        if st.has_work() {
            return false;
        }
        self.worker.store(false, Ordering::Release);
        true
    }

    pub(crate) fn worker_stopped(&self) {
        self.worker.store(false, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_license(node: &Node) -> Result<()> {
    crate::license::check(node, "ccr", crate::license::Mode::Platinum)
}

/// `CcrRetentionLeases.retentionLeaseId`.
pub fn retention_lease_id(local_cluster: &str, follower: &str, follower_uuid: &str, remote_alias: &str, leader: &str, leader_uuid: &str) -> String {
    format!("{local_cluster}/{follower}/{follower_uuid}-following-{remote_alias}/{leader}/{leader_uuid}")
}

fn find_by_uuid(node: &Node, name: &str, uuid: &str) -> Option<Arc<IndexService>> {
    node.get_index(name).filter(|s| s.uuid == uuid)
}

/// The leader index of a follow request (`CcrLicenseChecker`): concrete,
/// open, soft deletes on, not a searchable snapshot.
fn leader_index(leader: &LeaderCluster, name: &str) -> Result<Arc<IndexService>> {
    let svc = match leader.index(name) {
        Some(s) => s,
        None => {
            return Err(match leader.abstraction_type(name) {
                Some(t) => EsError::illegal_argument(format!("cannot follow [{name}], because it is a {t}")),
                None => EsError::index_not_found(name),
            })
        }
    };
    if !svc.is_open() {
        return Err(EsError::index_closed(&svc.name, &svc.uuid));
    }
    Ok(svc)
}

fn soft_deletes(svc: &IndexService) -> bool {
    svc.setting("index.soft_deletes.enabled").map(|v| v != "false").unwrap_or(true)
}

// ---------------------------------------------------------------------------
// Put follow
// ---------------------------------------------------------------------------

/// A parsed `PUT /{index}/_ccr/follow`.
#[derive(Clone, Debug, Default)]
pub struct PutFollowRequest {
    pub follower_index: String,
    pub remote_cluster: Option<String>,
    pub leader_index: Option<String>,
    pub data_stream_name: Option<String>,
    pub settings: FlatSettings,
    pub params: FollowParams,
    /// `wait_for_active_shards` other than `0` / default.
    pub wait_for_active_shards: bool,
}

const PUT_FOLLOW_PARSER: &str = "indices:admin/xpack/ccr/put_follow";

impl PutFollowRequest {
    pub fn parse(follower: &str, body: &Value) -> Result<PutFollowRequest> {
        let mut r = PutFollowRequest { follower_index: follower.to_string(), ..Default::default() };
        let o = match body {
            Value::Object(o) => o,
            Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
            other => return Err(EsError::x_content_parse(format!("[{PUT_FOLLOW_PARSER}] Expected START_OBJECT but was: {}", crate::jsonpos::token_name(other)))),
        };
        for (k, v) in o {
            match k.as_str() {
                "remote_cluster" | "leader_index" | "data_stream_name" => {
                    let s = match v {
                        Value::String(s) => s.clone(),
                        Value::Number(_) | Value::Bool(_) => v.to_string(),
                        _ => return Err(EsError::unsupported_type_xc(PUT_FOLLOW_PARSER, k, v, None)),
                    };
                    match k.as_str() {
                        "remote_cluster" => r.remote_cluster = Some(s),
                        "leader_index" => r.leader_index = Some(s),
                        _ => r.data_stream_name = Some(s),
                    }
                }
                "settings" => r.settings = parse_settings(PUT_FOLLOW_PARSER, v)?,
                _ => {
                    if !r.params.set(PUT_FOLLOW_PARSER, k, v)? {
                        let mut cands: Vec<&str> = vec!["remote_cluster", "leader_index", "data_stream_name", "settings"];
                        cands.extend(PARAM_FIELDS);
                        return Err(EsError::unknown_field_xc(PUT_FOLLOW_PARSER, k, None, &cands));
                    }
                }
            }
        }
        Ok(r)
    }

    fn validate(&self) -> Result<()> {
        let mut errs = Vec::new();
        self.params.validate(&mut errs);
        if self.remote_cluster.is_none() {
            errs.push("remote_cluster is missing".into());
        }
        if self.leader_index.is_none() {
            errs.push("leader_index is missing".into());
        }
        if self.data_stream_name.as_deref().is_some_and(|s| s.trim().is_empty()) {
            errs.push("data stream name must contain text if present".into());
        }
        if errs.is_empty() {
            return Ok(());
        }
        Err(EsError::action_validation(&errs.iter().map(String::as_str).collect::<Vec<_>>()))
    }
}

/// `TransportPutFollowAction`: create the follower from the leader's shard
/// files, then start following. Returns `(follow_index_created,
/// follow_index_shards_acked, index_following_started)`.
pub fn put_follow(node: &Arc<Node>, req: &PutFollowRequest) -> Result<(bool, bool, bool)> {
    req.validate()?;
    check_license(node)?;
    let alias = req.remote_cluster.clone().unwrap_or_default();
    let leader = service(node).connect(node, &alias)?;
    leader.check_license()?;
    let leader_name = req.leader_index.clone().unwrap_or_default();
    let lsvc = leader_index(&leader, &leader_name)?;
    if !soft_deletes(&lsvc) {
        return Err(EsError::illegal_argument(format!("leader index [{leader_name}] does not have soft deletes enabled")));
    }
    if crate::snapshot::is_searchable_snapshot(&lsvc) {
        return Err(EsError::illegal_argument(format!(
            "leader index [{leader_name}] is a searchable snapshot index and cannot be used as a leader index for cross-cluster replication purpose"
        )));
    }
    let replicated = filter_settings(&req.settings);
    if !replicated.is_empty() {
        let unknown: Vec<&str> = replicated.keys().map(String::as_str).filter(|k| !known_index_setting(k)).collect();
        let msg = if unknown.is_empty() {
            format!("can not put follower index that could override leader settings {}", settings_string(&replicated))
        } else {
            format!("unknown setting{} [{}]", if unknown.len() == 1 { "" } else { "s" }, unknown.join(","))
        };
        return Err(EsError::illegal_argument(msg));
    }
    let follower = req.follower_index.clone();
    crate::node::validate_index_name(&follower)?;
    if let Some(existing) = node.get_index(&follower) {
        if existing.is_open() {
            return Err(EsError::new(
                500,
                "snapshot_restore_exception",
                format!(
                    "[{}_ccr_{alias}:_latest_/_latest_] cannot restore index [{follower}] because an open index with same name already exists in the cluster. Either close or delete the existing index or restore the index under a different name by providing a rename pattern and replacement name",
                    // ES 9 snapshots are project-scoped (`default:`).
                    if crate::node::ES_VERSION.starts_with("8.") { "" } else { "default:" }
                ),
            ));
        }
        node.delete_index(&follower)?;
    }
    // A data stream backing index: its follower joins a replicated local
    // data stream (`TransportPutFollowAction.updateLocalDataStream`).
    let remote_ds = leader.data_stream_of(&lsvc.name);
    let local_ds = remote_ds.as_ref().map(|r| req.data_stream_name.clone().filter(|n| !n.trim().is_empty()).unwrap_or_else(|| r.name.clone()));
    if let Some(ln) = &local_ds {
        if let Some(local) = node.data_streams.read().get(ln) {
            if !local.replicated {
                return Err(EsError::illegal_argument(format!(
                    "cannot follow backing index [{follower}], because local data stream [{}] is no longer marked as replicated",
                    local.name
                )));
            }
        }
    }
    let fsvc = restore_follower(node, &leader, &lsvc, &alias, &follower, &req.settings)?;
    if let (Some(r), Some(ln)) = (&remote_ds, &local_ds) {
        update_local_data_stream(node, &follower, ln, r)?;
    }
    let svc = service(node);
    {
        let mut st = svc.st.lock();
        st.followers.insert(
            fsvc.uuid.clone(),
            FollowerMeta {
                follower_index: follower.clone(),
                remote_cluster: alias.clone(),
                leader_index: lsvc.name.clone(),
                leader_uuid: lsvc.uuid.clone(),
                leader_history_uuids: (0..lsvc.number_of_shards()).map(|_| lsvc.uuid.clone()).collect(),
                created_millis: crate::date::now_millis(),
            },
        );
        svc.persist(&st)?;
    }
    // Aliases (and anything else the leader changed meanwhile) before answering.
    task::initial_sync(&svc, &lsvc, &fsvc);
    let started = resume_follow_inner(node, &svc, &follower, &req.params);
    if req.wait_for_active_shards {
        started?;
        let acked = fsvc.active_shards() >= fsvc.number_of_shards() as usize;
        Ok((true, acked, true))
    } else {
        // ES answers before the restore completes; following still starts.
        let _ = started;
        Ok((true, false, false))
    }
}

/// `TransportPutFollowAction.updateLocalDataStream`.
fn update_local_data_stream(node: &Node, follower: &str, local_name: &str, remote: &crate::datastream::DataStream) -> Result<()> {
    {
        let mut streams = node.data_streams.write();
        match streams.get_mut(local_name) {
            None => {
                let mut ds = remote.clone();
                ds.name = local_name.to_string();
                ds.indices = vec![follower.to_string()];
                ds.rollover_on_write = false;
                ds.failure_rollover_on_write = false;
                ds.failure_indices = Vec::new();
                ds.replicated = true;
                streams.insert(local_name.to_string(), ds);
            }
            Some(local) => {
                if !local.replicated {
                    return Err(EsError::illegal_argument(format!(
                        "cannot follow backing index [{follower}], because local data stream [{}] is no longer marked as replicated",
                        local.name
                    )));
                }
                if !local.indices.iter().any(|i| i == follower) {
                    local.indices.push(follower.to_string());
                    // Non-standard backing indices first, then by the name
                    // from `.ds-` on (the newest generation stays the write index).
                    let base = format!(".ds-{}", local.name);
                    local.indices.sort_by(|a, b| {
                        let key = |n: &str| (n.contains(&base), n.find(".ds-").map(|p| n[p..].to_string()).unwrap_or_else(|| n.to_string()));
                        key(a).cmp(&key(b))
                    });
                }
                local.generation = remote.generation;
                local.meta = remote.meta.clone();
            }
        }
    }
    node.persist_state()
}

/// Copy every leader shard into a new follower index (the CCR repository
/// restore) and add the follower's retention lease on each leader shard.
fn restore_follower(node: &Arc<Node>, leader: &LeaderCluster, lsvc: &Arc<IndexService>, alias: &str, follower: &str, overrides: &FlatSettings) -> Result<Arc<IndexService>> {
    let uuid = crate::index::new_uuid();
    let dir = node.data.join("indices").join(&uuid);
    let lease_id = retention_lease_id(&node.cluster_name, follower, &uuid, alias, &lsvc.name, &lsvc.uuid);
    let mut leased: Vec<Arc<crate::index::IndexShard>> = Vec::new();
    let res = (|| -> Result<Arc<IndexService>> {
        let meta = lsvc.meta();
        let n = lsvc.number_of_shards();
        for s in 0..n {
            let sh = lsvc.shard(s).ok_or_else(|| EsError::shard_not_found(&lsvc.name, &lsvc.uuid, s))?;
            let files = sh.snapshot_files(true)?;
            let sdir = dir.join(format!("shard-{s}"));
            files.write_to_dir(&sdir, matches!(leader, LeaderCluster::Local(_)))?;
            for f in [crate::hooks::files::MANIFEST_FILE, crate::hooks::retention::LEASES_FILE] {
                let _ = std::fs::remove_file(sdir.join(f));
            }
            // Retain the leader's history above what the copy holds before
            // the file set's retention lock goes.
            let retain = files.manifest.local_checkpoint + 1;
            match sh.shard.add_retention_lease(&lease_id, retain, "ccr") {
                Ok(_) => {}
                Err(e) if e.kind == "retention_lease_already_exists_exception" => {
                    let _ = sh.shard.renew_retention_lease(&lease_id, retain, "ccr");
                }
                Err(e) => return Err(e),
            }
            leased.push(sh);
            drop(files);
        }
        let mut settings = meta.settings.clone();
        for k in ["index.uuid", "index.history.uuid", "index.provided_name", "index.verified_before_close"] {
            settings.remove(k);
        }
        settings.insert("index.uuid".into(), json!(uuid));
        settings.insert("index.provided_name".into(), json!(follower));
        settings.insert(FOLLOWING_INDEX_SETTING.into(), json!("true"));
        for (k, v) in overrides {
            if v.is_null() {
                settings.remove(k);
            } else {
                settings.insert(k.clone(), v.clone());
            }
        }
        // Aliases arrive with the first metadata sync of the follow task
        // (`innerUpdateAliases`), as non-write aliases.
        let aliases = Map::new();
        let m = json!({
            "name": follower, "uuid": uuid, "creation_date": lsvc.creation_date,
            "settings": crate::settings::as_flat_json(&settings), "mappings": meta.mappings.to_persisted(), "aliases": aliases, "open": true,
        });
        std::fs::create_dir_all(&dir)?;
        crate::hooks::write_durable(&dir.join("index_meta.json"), &serde_json::to_vec_pretty(&m)?)?;
        let svc = Arc::new(IndexService::open(&dir)?);
        node.register_index(svc.clone())?;
        Ok(svc)
    })();
    if res.is_err() {
        let _ = std::fs::remove_dir_all(&dir);
        for sh in leased {
            let _ = sh.shard.remove_retention_lease(&lease_id);
        }
    }
    res
}

// ---------------------------------------------------------------------------
// Resume / pause / unfollow
// ---------------------------------------------------------------------------

const RESUME_FOLLOW_PARSER: &str = "indices:admin/xpack/ccr/resume_follow";

/// Parse a `POST /{index}/_ccr/resume_follow` body (optional).
pub fn parse_resume_body(body: &Value) -> Result<FollowParams> {
    let mut p = FollowParams::default();
    match body {
        Value::Null => {}
        Value::Object(o) => {
            for (k, v) in o {
                if !p.set(RESUME_FOLLOW_PARSER, k, v)? {
                    return Err(EsError::unknown_field_xc(RESUME_FOLLOW_PARSER, k, None, &PARAM_FIELDS));
                }
            }
        }
        other => return Err(EsError::x_content_parse(format!("[{RESUME_FOLLOW_PARSER}] Expected START_OBJECT but was: {}", crate::jsonpos::token_name(other)))),
    }
    Ok(p)
}

/// `TransportResumeFollowAction`.
pub fn resume_follow(node: &Arc<Node>, follower: &str, params: &FollowParams) -> Result<()> {
    let mut errs = Vec::new();
    params.validate(&mut errs);
    if !errs.is_empty() {
        return Err(EsError::action_validation(&errs.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    check_license(node)?;
    let svc = service(node);
    resume_follow_inner(node, &svc, follower, params)
}

fn resume_follow_inner(node: &Arc<Node>, svc: &Arc<CcrService>, follower: &str, params: &FollowParams) -> Result<()> {
    let fsvc = node.get_index(follower).ok_or_else(|| EsError::index_not_found(follower))?;
    let meta = svc.st.lock().followers.get(&fsvc.uuid).cloned().ok_or_else(|| EsError::illegal_argument(format!("follow index [{follower}] does not have ccr metadata")))?;
    let leader = svc.connect(node, &meta.remote_cluster)?;
    leader.check_license()?;
    let lsvc = leader_index(&leader, &meta.leader_index)?;
    // TransportResumeFollowAction.validate
    if lsvc.uuid != meta.leader_uuid {
        return Err(EsError::illegal_argument(format!(
            "follow index [{follower}] should reference [{}] as leader index but instead reference [{}] as leader index",
            lsvc.uuid, meta.leader_uuid
        )));
    }
    if !soft_deletes(&lsvc) {
        return Err(EsError::illegal_argument(format!("leader index [{}] does not have soft deletes enabled", lsvc.name)));
    }
    if crate::snapshot::is_searchable_snapshot(&lsvc) {
        return Err(EsError::illegal_argument(format!("leader index [{}] is a searchable snapshot index and cannot be used for cross-cluster replication purpose", lsvc.name)));
    }
    if !soft_deletes(&fsvc) {
        return Err(EsError::illegal_argument(format!("follower index [{follower}] does not have soft deletes enabled")));
    }
    if crate::snapshot::is_searchable_snapshot(&fsvc) {
        return Err(EsError::illegal_argument(format!("follower index [{follower}] is a searchable snapshot index and cannot be used for cross-cluster replication purpose")));
    }
    if lsvc.number_of_shards() != fsvc.number_of_shards() {
        return Err(EsError::illegal_argument(format!(
            "leader index primary shards [{}] does not match with the number of shards of the follow index [{}]",
            lsvc.number_of_shards(),
            fsvc.number_of_shards()
        )));
    }
    let routing = |s: &IndexService| s.setting("index.number_of_routing_shards").unwrap_or_else(|| s.number_of_shards().to_string());
    if routing(&lsvc) != routing(&fsvc) {
        return Err(EsError::illegal_argument(format!(
            "leader index number_of_routing_shards [{}] does not match with the number_of_routing_shards of the follow index [{}]",
            routing(&lsvc),
            routing(&fsvc)
        )));
    }
    if !lsvc.is_open() || !fsvc.is_open() {
        return Err(EsError::illegal_argument("leader and follow index must be open"));
    }
    if fsvc.setting(FOLLOWING_INDEX_SETTING).as_deref() != Some("true") {
        return Err(EsError::illegal_argument(format!(
            "the following index [{follower}] is not ready to follow; the setting [{FOLLOWING_INDEX_SETTING}] must be enabled."
        )));
    }
    let strip = |s: &FlatSettings| {
        let mut f = filter_settings(s);
        f.remove("index.lifecycle.indexing_complete");
        f
    };
    let (ls, fs) = (strip(&lsvc.meta().settings), strip(&fsvc.meta().settings));
    if ls != fs {
        return Err(EsError::illegal_argument(format!(
            "the leader index settings [{}] and follower index settings [{}] must be identical",
            settings_string(&ls),
            settings_string(&fs)
        )));
    }
    fsvc.check_put_mapping(&lsvc.meta().mappings.to_json())?;
    {
        let mut st = svc.st.lock();
        if st.tasks.contains_key(&fsvc.uuid) {
            // One failed persistent-task start per shard, the rest suppressed.
            let exists = |n: u32| format!("task with id {{{}-{n}}} already exist", fsvc.uuid);
            let mut e = EsError::new(400, "resource_already_exists_exception", exists(0));
            if fsvc.number_of_shards() > 1 {
                let sup: Vec<Value> = (1..fsvc.number_of_shards()).map(|n| json!({"type": "resource_already_exists_exception", "reason": exists(n)})).collect();
                e = e.with("suppressed", Value::Array(sup));
            }
            return Err(e);
        }
        st.tasks.insert(fsvc.uuid.clone(), params.resolve());
        svc.persist(&st)?;
    }
    svc.rt.drop_index(&fsvc.uuid);
    svc.ensure_worker(node);
    Ok(())
}

/// `TransportPauseFollowAction`.
pub fn pause_follow(node: &Arc<Node>, follower: &str) -> Result<()> {
    let fsvc = node.get_index(follower).ok_or_else(|| EsError::index_not_found(follower))?;
    let svc = service(node);
    let mut st = svc.st.lock();
    if !st.followers.contains_key(&fsvc.uuid) {
        return Err(EsError::illegal_argument(format!("index [{follower}] is not a follower index")));
    }
    if st.tasks.remove(&fsvc.uuid).is_none() {
        return Err(EsError::illegal_argument(format!("no shard follow tasks for [{follower}]")));
    }
    svc.persist(&st)?;
    drop(st);
    svc.rt.drop_index(&fsvc.uuid);
    Ok(())
}

/// `TransportUnfollowAction`: turn a paused, closed follower into a
/// regular index and remove its retention leases on the leader.
pub fn unfollow(node: &Arc<Node>, follower: &str) -> Result<()> {
    let fsvc = node.get_index(follower).ok_or_else(|| EsError::index_not_found(follower))?;
    let svc = service(node);
    let meta = {
        let mut st = svc.st.lock();
        let Some(meta) = st.followers.get(&fsvc.uuid).cloned() else {
            return Err(EsError::illegal_argument(format!("index [{follower}] is not a follower index")));
        };
        if fsvc.is_open() {
            return Err(EsError::illegal_argument(format!("cannot convert the follower index [{follower}] to a non-follower, because it has not been closed")));
        }
        if st.tasks.contains_key(&fsvc.uuid) {
            return Err(EsError::illegal_argument(format!("cannot convert the follower index [{follower}] to a non-follower, because it has not been paused")));
        }
        let mut flat = FlatSettings::new();
        flat.insert(FOLLOWING_INDEX_SETTING.into(), Value::Null);
        fsvc.put_settings(flat, false)?;
        st.followers.remove(&fsvc.uuid);
        svc.persist(&st)?;
        meta
    };
    svc.rt.drop_index(&fsvc.uuid);
    // Remove the follower's retention lease on every leader shard.
    let lease_id = retention_lease_id(&node.cluster_name, follower, &fsvc.uuid, &meta.remote_cluster, &meta.leader_index, &meta.leader_uuid);
    let wrap = |e: EsError| {
        EsError::new(500, "exception", format!("{}: {}", java_exception_class(&e.kind), e.reason)).with("failed_to_remove_retention_leases", lease_id.clone()).caused_by(e)
    };
    let leader = svc.connect(node, &meta.remote_cluster).map_err(wrap)?;
    if let Some(lsvc) = leader.index(&meta.leader_index).filter(|s| s.uuid == meta.leader_uuid) {
        for sh in lsvc.local_shards() {
            sh.shard.remove_retention_lease(&lease_id).map_err(wrap)?;
        }
    }
    Ok(())
}

fn java_exception_class(kind: &str) -> String {
    match kind {
        "no_such_remote_cluster_exception" => "org.elasticsearch.transport.NoSuchRemoteClusterException".into(),
        "connect_transport_exception" => "org.elasticsearch.transport.ConnectTransportException".into(),
        "index_not_found_exception" => "org.elasticsearch.index.IndexNotFoundException".into(),
        other => {
            let camel: String = other.split('_').map(|p| {
                let mut c = p.chars();
                match c.next() {
                    Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
                    None => String::new(),
                }
            }).collect();
            format!("org.elasticsearch.{camel}")
        }
    }
}

// ---------------------------------------------------------------------------
// Forget follower
// ---------------------------------------------------------------------------

const FORGET_FOLLOWER_PARSER: &str = "forget_follower";

/// `POST /{index}/_ccr/forget_follower` (`TransportForgetFollowerAction`):
/// remove a follower's retention lease from every primary of the leader
/// index. Returns the broadcast `_shards` header.
pub fn forget_follower(node: &Arc<Node>, leader_index: &str, body: &Value) -> Result<Value> {
    let o = match body {
        Value::Object(o) => o,
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        other => return Err(EsError::x_content_parse(format!("[{FORGET_FOLLOWER_PARSER}] Expected START_OBJECT but was: {}", crate::jsonpos::token_name(other)))),
    };
    const FIELDS: [&str; 4] = ["follower_cluster", "follower_index", "follower_index_uuid", "leader_remote_cluster"];
    let mut vals: [Option<String>; 4] = Default::default();
    for (k, v) in o {
        let Some(i) = FIELDS.iter().position(|f| f == k) else {
            return Err(EsError::unknown_field_xc(FORGET_FOLLOWER_PARSER, k, None, &FIELDS));
        };
        vals[i] = Some(match v {
            Value::String(s) => s.clone(),
            Value::Number(_) | Value::Bool(_) => v.to_string(),
            _ => return Err(EsError::unsupported_type_xc(FORGET_FOLLOWER_PARSER, k, v, None)),
        });
    }
    // ES builds the request without checking for missing fields and fails
    // with a NullPointerException.
    if vals.iter().any(Option::is_none) {
        return Err(EsError::new(500, "null_pointer_exception", crate::error::NULL_REASON));
    }
    let [fc, fi, fu, lrc] = vals.map(Option::unwrap_or_default);
    let targets = node.resolve(leader_index, crate::node::ResolveOpts::default())?;
    let mut total = 0;
    let mut ok = 0;
    let mut failures = Vec::new();
    for t in &targets {
        let id = retention_lease_id(&fc, &fi, &fu, &lrc, &t.svc.name, &t.svc.uuid);
        for sh in t.svc.local_shards() {
            total += 1;
            let failure = match sh.shard.remove_retention_lease(&id) {
                Ok(true) => {
                    ok += 1;
                    continue;
                }
                Ok(false) => json!({
                    "shard": sh.id, "index": t.svc.name, "status": "NOT_FOUND",
                    "reason": {"type": "retention_lease_not_found_exception", "reason": format!("retention lease with ID [{id}] not found")}
                }),
                Err(e) => json!({"shard": sh.id, "index": t.svc.name, "status": "INTERNAL_SERVER_ERROR", "reason": e.shard_reason()}),
            };
            failures.push(failure);
        }
    }
    let failed = failures.len();
    // `ExceptionsHelper.groupBy`: one entry per index and cause.
    let mut grouped: Vec<Value> = Vec::new();
    for f in failures {
        if !grouped.iter().any(|g| g["index"] == f["index"] && g["reason"] == f["reason"]) {
            grouped.push(f);
        }
    }
    let mut shards = json!({"total": total, "successful": ok, "failed": failed});
    if !grouped.is_empty() {
        shards["failures"] = Value::Array(grouped);
    }
    Ok(json!({"_shards": shards}))
}

// ---------------------------------------------------------------------------
// Follow info / stats
// ---------------------------------------------------------------------------

/// `GET /{index}/_ccr/info` (`TransportFollowInfoAction`).
pub fn follow_info(node: &Node, expr: &str) -> Result<Value> {
    let opts = crate::node::ResolveOpts { ignore_unavailable: false, allow_no_indices: true, expand_open: true, expand_closed: true, expand_hidden: false };
    let targets = node.resolve(expr, opts)?;
    let svc = existing(node);
    let st = svc.as_ref().map(|s| s.st.lock());
    let mut out = Vec::new();
    for t in &targets {
        let Some(st) = &st else { break };
        let Some(meta) = st.followers.get(&t.svc.uuid) else { continue };
        let mut m = Map::new();
        m.insert("follower_index".into(), json!(t.svc.name));
        m.insert("remote_cluster".into(), json!(meta.remote_cluster));
        m.insert("leader_index".into(), json!(meta.leader_index));
        match st.tasks.get(&t.svc.uuid) {
            Some(p) => {
                m.insert("status".into(), json!("active"));
                let mut pm = Map::new();
                p.fragment(&mut pm);
                m.insert("parameters".into(), Value::Object(pm));
            }
            None => {
                m.insert("status".into(), json!("paused"));
            }
        }
        out.push(Value::Object(m));
    }
    Ok(json!({"follower_indices": out}))
}

/// The follower indices with shard follow tasks (`findFollowerIndicesFromShardFollowTasks`).
fn followed_indices(node: &Node, svc: &CcrService, names: Option<&[String]>) -> Vec<(String, String)> {
    let st = svc.st.lock();
    let mut out: Vec<(String, String)> = st
        .tasks
        .keys()
        .filter_map(|uuid| {
            let meta = st.followers.get(uuid)?;
            find_by_uuid(node, &meta.follower_index, uuid)?;
            Some((meta.follower_index.clone(), uuid.clone()))
        })
        .filter(|(name, _)| names.is_none_or(|n| n.iter().any(|x| x == name)))
        .collect();
    out.sort();
    out
}

fn all_or_wildcard(names: &[String]) -> bool {
    names.is_empty() || (names.len() == 1 && (names[0] == "_all" || names[0] == "*"))
}

/// `GET /{index}/_ccr/stats` (`TransportFollowStatsAction`).
pub fn follow_stats(node: &Node, expr: &str) -> Result<Value> {
    check_license(node)?;
    let names: Vec<String> = expr.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let all = all_or_wildcard(&names);
    let svc = existing(node);
    let list = match &svc {
        Some(s) => followed_indices(node, s, if all { None } else { Some(&names) }),
        None => Vec::new(),
    };
    if !all && list.is_empty() {
        return Err(EsError::resource_not_found(format!("No shard follow tasks for follower indices [{}]", names.join(","))));
    }
    Ok(match svc {
        Some(s) => follow_stats_json(node, &s, &list),
        None => json!({"indices": []}),
    })
}

/// `GET /_ccr/stats` (`TransportCcrStatsAction`).
pub fn ccr_stats(node: &Node) -> Result<Value> {
    check_license(node)?;
    let svc = existing(node);
    let (auto, follow) = match &svc {
        Some(s) => {
            let list = followed_indices(node, s, None);
            (s.rt.auto_stats_json(), follow_stats_json(node, s, &list))
        }
        None => (task::Runtime::default().auto_stats_json(), json!({"indices": []})),
    };
    Ok(json!({"auto_follow_stats": auto, "follow_stats": follow}))
}

/// `xpack.usage.ccr` (`CCRUsageTransportAction`).
pub fn usage(node: &Node) -> Value {
    let available = crate::license::current(node).effective_mode(crate::date::now_millis()).allows(crate::license::Mode::Platinum);
    let mut m = Map::new();
    m.insert("available".into(), json!(available));
    m.insert("enabled".into(), json!(true));
    let (mut followers, mut patterns, mut last) = (0usize, 0usize, None::<i64>);
    if let Some(s) = existing(node) {
        let st = s.st.lock();
        for (uuid, meta) in &st.followers {
            if let Some(svc) = find_by_uuid(node, &meta.follower_index, uuid) {
                followers += 1;
                last = Some(last.map_or(svc.creation_date, |l: i64| l.max(svc.creation_date)));
            }
        }
        patterns = st.patterns.len();
    }
    m.insert("follower_indices_count".into(), json!(followers));
    m.insert("auto_follow_patterns_count".into(), json!(patterns));
    if let Some(l) = last {
        m.insert("last_follow_time_in_millis".into(), json!((crate::date::now_millis() - l).max(0)));
    }
    Value::Object(m)
}

/// The CCR custom metadata of `index` (`None` when it is not a follower).
pub fn follower_custom(node: &Node, svc: &IndexService) -> Option<Value> {
    let s = existing(node)?;
    let st = s.st.lock();
    st.followers.get(&svc.uuid).map(FollowerMeta::custom_json)
}

// ---------------------------------------------------------------------------
// Auto-follow patterns
// ---------------------------------------------------------------------------

const PUT_PATTERN_PARSER: &str = "put_auto_follow_pattern_request";

/// `PUT /_ccr/auto_follow/{name}` (`TransportPutAutoFollowPatternAction`).
pub fn put_auto_follow_pattern(node: &Arc<Node>, name: &str, body: &Value) -> Result<()> {
    let o = match body {
        Value::Object(o) => o.clone(),
        Value::Null => return Err(EsError::new(400, "parse_exception", "request body is required")),
        other => return Err(EsError::x_content_parse(format!("[{PUT_PATTERN_PARSER}] Expected START_OBJECT but was: {}", crate::jsonpos::token_name(other)))),
    };
    let mut remote_cluster = None;
    let mut patterns: Option<Vec<String>> = None;
    let mut exclusions: Vec<String> = Vec::new();
    let mut follow_pattern = None;
    let mut settings = FlatSettings::new();
    let mut params = FollowParams::default();
    let str_array = |k: &str, v: &Value| -> Result<Vec<String>> {
        match v {
            Value::Array(a) => a
                .iter()
                .map(|x| match x {
                    Value::String(s) => Ok(s.clone()),
                    Value::Number(_) | Value::Bool(_) => Ok(x.to_string()),
                    _ => Err(EsError::x_content_parse(format!("[{PUT_PATTERN_PARSER}] failed to parse field [{k}]")).at_key(k, None).prefixed("key")),
                })
                .collect(),
            Value::String(s) => Ok(vec![s.clone()]),
            _ => Err(EsError::unsupported_type_xc(PUT_PATTERN_PARSER, k, v, None)),
        }
    };
    for (k, v) in &o {
        match k.as_str() {
            "remote_cluster" | "follow_index_pattern" => {
                let s = match v {
                    Value::String(s) => s.clone(),
                    Value::Number(_) | Value::Bool(_) => v.to_string(),
                    _ => return Err(EsError::unsupported_type_xc(PUT_PATTERN_PARSER, k, v, None)),
                };
                if k == "remote_cluster" {
                    remote_cluster = Some(s);
                } else {
                    follow_pattern = Some(s);
                }
            }
            "leader_index_patterns" => patterns = Some(str_array(k, v)?),
            "leader_index_exclusion_patterns" => exclusions = str_array(k, v)?,
            "settings" => settings = parse_settings(PUT_PATTERN_PARSER, v)?,
            _ => {
                if !params.set(PUT_PATTERN_PARSER, k, v)? {
                    let mut cands: Vec<&str> = vec!["remote_cluster", "leader_index_patterns", "follow_index_pattern", "settings", "leader_index_exclusion_patterns"];
                    cands.extend(PARAM_FIELDS);
                    return Err(EsError::unknown_field_xc(PUT_PATTERN_PARSER, k, None, &cands));
                }
            }
        }
    }
    // PutAutoFollowPatternAction.Request.validate
    let mut errs = Vec::new();
    params.validate(&mut errs);
    if name.contains(',') {
        errs.push("[name] name must not contain a ','".into());
    }
    if name.starts_with('_') {
        errs.push("[name] name must not start with '_'".into());
    }
    if name.len() > 255 {
        errs.push(format!("[name] name is too long ({} > 255)", name.len()));
    }
    if remote_cluster.is_none() {
        errs.push("[remote_cluster] is missing".into());
    }
    if patterns.as_ref().is_none_or(|p| p.is_empty()) {
        errs.push("[leader_index_patterns] is missing".into());
    }
    if !errs.is_empty() {
        return Err(EsError::action_validation(&errs.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    check_license(node)?;
    let replicated = filter_settings(&settings);
    if !replicated.is_empty() {
        return Err(EsError::illegal_argument(format!("can not put auto-follow pattern that could override leader settings {}", settings_string(&replicated))));
    }
    let remote_cluster = remote_cluster.unwrap_or_default();
    let patterns = patterns.unwrap_or_default();
    let leader = service(node).connect(node, &remote_cluster)?;
    leader.check_license()?;
    let svc = service(node);
    {
        let mut st = svc.st.lock();
        let previous = st.patterns.get(name).cloned();
        let mut followed = st.followed.get(name).cloned().unwrap_or_default();
        // Existing leader indices count as followed (only new patterns' on update).
        let mark: Vec<String> = match &previous {
            Some(p) => patterns.iter().filter(|x| !p.leader_index_patterns.contains(x)).cloned().collect(),
            None => patterns.clone(),
        };
        for l in leader.indices() {
            let ds = leader.parent_data_stream(&l.name);
            if auto_follow_match(&mark, &exclusions, &l.name, ds.as_deref()) && !followed.contains(&l.uuid) {
                followed.push(l.uuid.clone());
            }
        }
        st.followed.insert(name.to_string(), followed);
        st.patterns.insert(
            name.to_string(),
            AutoFollowPattern {
                remote_cluster,
                leader_index_patterns: patterns,
                leader_index_exclusion_patterns: exclusions,
                follow_index_pattern: follow_pattern,
                settings,
                active: true,
                params,
            },
        );
        svc.persist(&st)?;
    }
    svc.rt.patterns_changed();
    svc.ensure_worker(node);
    Ok(())
}

fn pattern_missing(name: &str) -> EsError {
    EsError::resource_not_found(format!("auto-follow pattern [{name}] is missing"))
}

/// `GET /_ccr/auto_follow[/{name}]`.
pub fn get_auto_follow_patterns(node: &Node, name: Option<&str>) -> Result<Value> {
    let svc = existing(node);
    let st = svc.as_ref().map(|s| s.st.lock());
    let empty = BTreeMap::new();
    let all = st.as_ref().map(|s| &s.patterns).unwrap_or(&empty);
    let mut out = Vec::new();
    match name {
        None => {
            for (n, p) in all {
                out.push(json!({"name": n, "pattern": p.to_json()}));
            }
        }
        Some(n) => {
            let p = all.get(n).ok_or_else(|| pattern_missing(n))?;
            out.push(json!({"name": n, "pattern": p.to_json()}));
        }
    }
    Ok(json!({"patterns": out}))
}

/// `DELETE /_ccr/auto_follow/{name}`.
pub fn delete_auto_follow_pattern(node: &Node, name: &str) -> Result<()> {
    let svc = service(node);
    {
        let mut st = svc.st.lock();
        if st.patterns.remove(name).is_none() {
            return Err(pattern_missing(name));
        }
        st.followed.remove(name);
        svc.persist(&st)?;
    }
    svc.rt.patterns_changed();
    Ok(())
}

/// `POST /_ccr/auto_follow/{name}/pause|resume` (`TransportActivateAutoFollowPatternAction`).
pub fn activate_auto_follow_pattern(node: &Arc<Node>, name: &str, active: bool) -> Result<()> {
    let svc = service(node);
    {
        let mut st = svc.st.lock();
        let p = st.patterns.get_mut(name).ok_or_else(|| pattern_missing(name))?;
        if p.active == active {
            return Ok(());
        }
        p.active = active;
        svc.persist(&st)?;
    }
    svc.rt.patterns_changed();
    svc.ensure_worker(node);
    Ok(())
}

/// `AutoFollower.getFollowerIndexName`.
fn follower_index_name(pattern: &AutoFollowPattern, leader: &str) -> String {
    match &pattern.follow_index_pattern {
        None => leader.to_string(),
        Some(fp) => {
            if let Some(pos) = leader.find(".ds-") {
                // `<prefix>.ds-<data stream>-<yyyy.MM.dd>-<generation>`: rename
                // the data stream part only.
                let (prefix, rest) = leader.split_at(pos + 4);
                let re = regex::Regex::new(r"^(.+)-(\d{4}.\d{2}.\d{2})(-[\d]+)?$").expect("static regex");
                if let Some(c) = re.captures(rest) {
                    let ds = c.get(1).map_or("", |m| m.as_str());
                    let date = c.get(2).map_or("", |m| m.as_str());
                    let generation = c.get(3).map_or("", |m| m.as_str());
                    return format!("{prefix}{}-{date}{generation}", fp.replace(AUTO_FOLLOW_PATTERN_REPLACEMENT, ds));
                }
            }
            fp.replace(AUTO_FOLLOW_PATTERN_REPLACEMENT, leader)
        }
    }
}

#[cfg(test)]
mod tests;
