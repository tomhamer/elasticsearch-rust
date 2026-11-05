//! `SnapshotsService`: repository registry plus the snapshot operations the
//! node crate exposes over REST (`_snapshot/*`).

use crate::error::{Result, SnapshotError};
use crate::repository::shard::ShardSnapshotTask;
use crate::repository::{FinalizeRequest, Repository, RepositoryContext, RepositoryData, RepositoryMetadata};
use crate::repository::FinalizeIndex;
use crate::snapshot::{
    ES_VERSION, ES_VERSION_ID, FeatureState, IndexDetails, RenderOptions, ShardFailure, ShardStage, ShardStatus, SnapshotInfo,
    SnapshotState, SnapshotStatus,
};
use crate::util::{CancelToken, now_millis, random_base64_uuid, resolve_names, simple_match};
use base64::Engine;
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `SourceOnlySnapshotRepository`'s shard failure for an index whose
/// `_source` is disabled, synthetic or filtered.
const SOURCE_ONLY_INCOMPLETE: &str = "IllegalStateException[Can't snapshot _source only on an index that has incomplete source ie. has _source disabled or filters the source]";

/// Immutable file-name source supplied by the engine integration. Implementors
/// retain their allocation owners; snapshot task clones share that source.
pub trait SnapshotFileSource: std::fmt::Debug + Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&str>;
    /// Independent admission retained by derived upload metadata/buffers.
    fn workspace(&self) -> Result<Box<dyn SnapshotWorkspace>>;
}
pub trait SnapshotWorkspace: Send + Sync {
    fn grow(&mut self, bytes: usize, label: &'static str) -> Result<()>;
}
// Standalone storage-crate callers have no engine budget. Production node
// integration supplies its real reservation through SnapshotNames instead.
struct StandaloneWorkspace;
impl SnapshotWorkspace for StandaloneWorkspace {
    fn grow(&mut self, _: usize, _: &'static str) -> Result<()> { Ok(()) }
}
impl SnapshotFileSource for Vec<String> {
    fn len(&self)->usize { Vec::len(self) }
    fn get(&self,index:usize)->Option<&str> { self.as_slice().get(index).map(String::as_str) }
    fn workspace(&self)->Result<Box<dyn SnapshotWorkspace>> { Ok(Box::new(StandaloneWorkspace)) }
}
#[derive(Debug, Clone)]
pub struct SnapshotFiles(pub Arc<dyn SnapshotFileSource>);
impl SnapshotFiles {
    pub fn workspace(&self)->Result<Box<dyn SnapshotWorkspace>> { self.0.workspace() }
    pub fn len(&self)->usize { self.0.len() }
    pub fn is_empty(&self)->bool { self.len()==0 }
    pub fn iter(&self)->impl Iterator<Item=&str> { (0..self.len()).map(|i|self.0.get(i).expect("immutable file source length")) }
}
impl From<Vec<String>> for SnapshotFiles {fn from(v:Vec<String>)->Self{Self(Arc::new(v))}}
/// One shard of an index to snapshot. Files are read from `dir`; the caller
/// guarantees they are immutable for the duration of the snapshot.
#[derive(Debug, Clone)]
pub struct ShardSnapshotInput {
    pub shard: u32,
    pub dir: PathBuf,
    /// Paths relative to `dir`.
    pub files: SnapshotFiles,
    /// `Some(reason)` if the shard cannot be snapshotted (primary missing).
    pub unavailable: Option<String>,
    /// The shard is snapshotted by another node (the one holding its
    /// primary) through [`CreateSnapshotRequest::remote`]; `dir` and `files`
    /// are unused.
    pub remote: Option<RemoteShard>,
}

/// Where a remotely snapshotted shard lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteShard {
    pub node_id: String,
}

/// One shard snapshot handed to a [`RemoteShardSnapshots`].
#[derive(Debug, Clone)]
pub struct RemoteShardRequest {
    pub snapshot_name: String,
    pub snapshot_uuid: String,
    pub index_name: String,
    pub index_id: String,
    pub shard: u32,
    /// Node holding the shard's primary when the snapshot was planned.
    pub node_id: String,
    /// Shard generation the new shard index must build on (`None`: the
    /// shard has none yet). Set under the coordinating node's shard lock.
    pub base_gen: Option<String>,
}

/// Snapshots shards on the nodes holding them (ES `SnapshotShardsService`):
/// the implementation writes the shard-level blobs (`__*` data blobs,
/// `snap-{uuid}.dat`, `index-{gen}`) on top of `req.base_gen`, keeps
/// `status` up to date while it runs, stops when `cancel` fires, and returns
/// the new shard generation. `repo` is the coordinating node's repository
/// (for implementations that fall back to uploading from there).
#[async_trait::async_trait]
pub trait RemoteShardSnapshots: Send + Sync {
    async fn snapshot_shard(
        &self,
        repo: Arc<Repository>,
        req: RemoteShardRequest,
        status: Arc<Mutex<ShardStatus>>,
        cancel: CancelToken,
    ) -> Result<crate::repository::ShardSnapshotResult>;
}

/// The [`RemoteShardSnapshots`] of a create request.
#[derive(Clone)]
pub struct RemoteUploader(pub Arc<dyn RemoteShardSnapshots>);

impl std::fmt::Debug for RemoteUploader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RemoteUploader")
    }
}

#[derive(Debug, Clone)]
pub struct IndexSnapshotInput {
    pub index_name: String,
    pub index_metadata_json: Value,
    pub shards: Vec<ShardSnapshotInput>,
}

/// A fully resolved create-snapshot request.
#[derive(Debug, Clone, Default)]
pub struct CreateSnapshotRequest {
    pub repository: String,
    pub snapshot: String,
    /// Fixed uuid (tests / replays); random if `None`.
    pub uuid: Option<String>,
    pub indices: Vec<IndexSnapshotInput>,
    pub global_metadata_json: Option<Value>,
    pub include_global_state: bool,
    pub partial: bool,
    pub metadata: Option<Map<String, Value>>,
    pub feature_states: Vec<FeatureState>,
    pub data_streams: Vec<String>,
    /// Node doing the work (reported in failures and `_status`).
    pub node_id: Option<String>,
    /// Snapshots the shards marked [`ShardSnapshotInput::remote`].
    pub remote: Option<RemoteUploader>,
}

/// Parsed body of `PUT _snapshot/{repo}/{snap}`; the node resolves `indices`
/// against its cluster state and builds a [`CreateSnapshotRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct CreateSnapshotParams {
    pub indices: Vec<String>,
    pub ignore_unavailable: bool,
    pub allow_no_indices: bool,
    pub expand_wildcards: Vec<String>,
    pub include_global_state: bool,
    pub partial: bool,
    pub metadata: Option<Map<String, Value>>,
    pub feature_states: Vec<String>,
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
        _ => vec![],
    }
}

fn as_bool(field: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        _ => Err(SnapshotError::illegal_argument(format!("malformed {field}, expected boolean but got [{v}]"))),
    }
}

impl CreateSnapshotParams {
    pub fn from_json(body: Option<&Value>) -> Result<Self> {
        let mut p = Self {
            indices: vec![],
            ignore_unavailable: false,
            allow_no_indices: true,
            expand_wildcards: vec!["open".into(), "hidden".into()],
            include_global_state: true,
            partial: false,
            metadata: None,
            feature_states: vec![],
        };
        let Some(obj) = body.and_then(|b| b.as_object()) else { return Ok(p) };
        for (k, v) in obj {
            match k.as_str() {
                "indices" => p.indices = str_list(v),
                "ignore_unavailable" => p.ignore_unavailable = as_bool(k, v)?,
                "allow_no_indices" => p.allow_no_indices = as_bool(k, v)?,
                "expand_wildcards" => p.expand_wildcards = str_list(v),
                "include_global_state" => p.include_global_state = as_bool(k, v)?,
                "partial" => p.partial = as_bool(k, v)?,
                "feature_states" => p.feature_states = str_list(v),
                "metadata" => match v {
                    Value::Object(m) => {
                        let size = serde_json::to_vec(m).map(|b| b.len()).unwrap_or(0);
                        if size > 1024 {
                            return Err(SnapshotError::validation(&[format!(
                                "metadata must be smaller than 1024 bytes, but was [{size}]"
                            )]));
                        }
                        p.metadata = Some(m.clone())
                    }
                    Value::Null => {}
                    _ => return Err(SnapshotError::illegal_argument(format!("malformed metadata, should be an object"))),
                },
                // Accepted and ignored (handled by REST layer / not applicable).
                "uuid" | "wait_for_completion" | "master_timeout" => {}
                // CreateSnapshotRequest.source ignores unrecognized body entries.
                // They must not be confused with restore index settings.
                _ => {}
            }
        }
        Ok(p)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    StartTime,
    Name,
    Duration,
    IndexCount,
    ShardCount,
    FailedShardCount,
    Repository,
}

impl SortBy {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "start_time" => Self::StartTime,
            "name" => Self::Name,
            "duration" => Self::Duration,
            "index_count" => Self::IndexCount,
            "shard_count" => Self::ShardCount,
            "failed_shard_count" => Self::FailedShardCount,
            "repository" => Self::Repository,
            _ => return Err(SnapshotError::illegal_argument(format!("unknown sort order [{s}]"))),
        })
    }

    fn value(self, info: &SnapshotInfo) -> SortValue {
        match self {
            Self::StartTime => SortValue::Num(info.start_time),
            Self::Name => SortValue::Str(info.snapshot.clone()),
            Self::Duration => SortValue::Num(info.duration_millis()),
            Self::IndexCount => SortValue::Num(info.indices.len() as i64),
            Self::ShardCount => SortValue::Num(info.total_shards as i64),
            Self::FailedShardCount => SortValue::Num(info.failed_shards() as i64),
            Self::Repository => SortValue::Str(info.repository.clone()),
        }
    }

    fn parse_value(self, s: &str) -> Result<SortValue> {
        match self {
            Self::Name | Self::Repository => Ok(SortValue::Str(s.to_string())),
            _ => s
                .parse::<i64>()
                .map(SortValue::Num)
                .map_err(|_| SnapshotError::illegal_argument(format!("invalid sort value [{s}]"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SortValue {
    Num(i64),
    Str(String),
}

impl SortValue {
    fn render(&self) -> String {
        match self {
            Self::Num(n) => n.to_string(),
            Self::Str(s) => s.clone(),
        }
    }
}

/// `GET _snapshot/{repo}/{snapshot}` request.
#[derive(Debug, Clone)]
pub struct GetSnapshotsRequest {
    pub repositories: Vec<String>,
    pub snapshots: Vec<String>,
    pub ignore_unavailable: bool,
    pub verbose: bool,
    pub sort: SortBy,
    pub desc: bool,
    /// `None` = all (`size=-1`).
    pub size: Option<usize>,
    pub offset: usize,
    pub after: Option<String>,
    pub from_sort_value: Option<String>,
    pub slm_policy_filter: Vec<String>,
    /// Optional exact snapshot states, applied before pagination.
    pub states: Option<Vec<SnapshotState>>,
    pub include_repository: bool,
    pub index_names: bool,
    pub index_details: bool,
    pub human: bool,
}

impl Default for GetSnapshotsRequest {
    fn default() -> Self {
        Self {
            repositories: vec!["_all".into()],
            snapshots: vec!["_all".into()],
            ignore_unavailable: false,
            verbose: true,
            sort: SortBy::StartTime,
            desc: false,
            size: None,
            offset: 0,
            after: None,
            from_sort_value: None,
            slm_policy_filter: vec![],
            states: None,
            include_repository: true,
            index_names: true,
            index_details: false,
            human: false,
        }
    }
}

impl GetSnapshotsRequest {
    /// Build from path parts and query-string params.
    pub fn from_params(repository: &str, snapshot: &str, params: &BTreeMap<String, String>) -> Result<Self> {
        let mut r = Self {
            repositories: repository.split(',').map(|s| s.to_string()).collect(),
            snapshots: snapshot.split(',').map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let b = |k: &str, d: bool| -> Result<bool> {
            match params.get(k).map(|s| s.as_str()) {
                None => Ok(d),
                Some("" | "true") => Ok(true),
                Some("false") => Ok(false),
                Some(o) => Err(SnapshotError::illegal_argument(format!(
                    "Failed to parse value [{o}] as only [true] or [false] are allowed."
                ))),
            }
        };
        r.ignore_unavailable = b("ignore_unavailable", false)?;
        r.verbose = b("verbose", true)?;
        r.include_repository = b("include_repository", true)?;
        r.index_names = b("index_names", true)?;
        r.index_details = b("index_details", false)?;
        r.human = b("human", false)?;
        if let Some(s) = params.get("sort") {
            r.sort = SortBy::parse(s)?;
        }
        if let Some(o) = params.get("order") {
            r.desc = match o.as_str() {
                "asc" => false,
                "desc" => true,
                _ => return Err(SnapshotError::illegal_argument(format!("Unknown SortOrder [{o}]"))),
            };
        }
        if let Some(s) = params.get("size") {
            let n: i64 = s.parse().map_err(|_| SnapshotError::illegal_argument(format!("Failed to parse int parameter [size] with value [{s}]")))?;
            if n == 0 || n < -1 {
                return Err(SnapshotError::validation(&["size must be -1 or greater than 0".into()]));
            }
            r.size = if n < 0 { None } else { Some(n as usize) };
        }
        if let Some(s) = params.get("offset") {
            let n: i64 = s.parse().map_err(|_| SnapshotError::illegal_argument(format!("Failed to parse int parameter [offset] with value [{s}]")))?;
            if n < 0 {
                return Err(SnapshotError::validation(&["offset must be >= 0".into()]));
            }
            r.offset = n as usize;
        }
        r.after = params.get("after").cloned();
        r.from_sort_value = params.get("from_sort_value").cloned();
        if let Some(p) = params.get("slm_policy_filter") {
            r.slm_policy_filter = p.split(',').map(|s| s.to_string()).collect();
        }
        if let Some(states) = params.get("state") {
            if states.trim().is_empty() {
                return Err(SnapshotError::illegal_argument("[state] parameter must not be empty"));
            }
            r.states = Some(states.split(',').map(|s| match s {
                "IN_PROGRESS" => Ok(SnapshotState::InProgress),
                "SUCCESS" => Ok(SnapshotState::Success),
                "FAILED" => Ok(SnapshotState::Failed),
                "PARTIAL" => Ok(SnapshotState::Partial),
                "INCOMPATIBLE" => Ok(SnapshotState::Incompatible),
                other => Err(SnapshotError::illegal_argument(format!("No enum constant org.elasticsearch.snapshots.SnapshotState.{other}"))),
            }).collect::<Result<Vec<_>>>()?);
        }
        r.validate()?;
        Ok(r)
    }

    /// ES `GetSnapshotsRequest.validate`.
    pub fn validate(&self) -> Result<()> {
        let mut errs = Vec::new();
        if !self.verbose {
            if self.sort != SortBy::StartTime {
                errs.push("can't use non-default sort with verbose=false".to_string());
            }
            if self.size.is_some() {
                errs.push("can't use size limit with verbose=false".into());
            }
            if self.offset > 0 {
                errs.push("can't use offset with verbose=false".into());
            }
            if self.after.is_some() {
                errs.push("can't use after with verbose=false".into());
            }
            if self.desc {
                errs.push("can't use non-default sort order with verbose=false".into());
            }
            if !self.slm_policy_filter.is_empty() {
                errs.push("can't use slm policy filter with verbose=false".into());
            }
            if self.from_sort_value.is_some() {
                errs.push("can't use from_sort_value with verbose=false".into());
            }
        } else if self.after.is_some() && self.from_sort_value.is_some() {
            errs.push("can't use after and from_sort_value simultaneously".into());
        }
        if self.after.is_some() && self.offset > 0 {
            errs.push("can't use after and offset simultaneously".into());
        }
        if errs.is_empty() { Ok(()) } else { Err(SnapshotError::validation(&errs)) }
    }
}

fn encode_after(v: &SortValue, repo: &str, name: &str) -> String {
    base64::engine::general_purpose::URL_SAFE.encode(format!("{},{repo},{name}", v.render()))
}

fn decode_after(s: &str) -> Result<(String, String, String)> {
    let err = || SnapshotError::illegal_argument(format!("invalid ?after parameter [{s}]"));
    let raw = base64::engine::general_purpose::URL_SAFE
        .decode(s)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s))
        .map_err(|_| err())?;
    let txt = String::from_utf8(raw).map_err(|_| err())?;
    let parts: Vec<&str> = txt.split(',').collect();
    if parts.len() != 3 {
        return Err(err());
    }
    Ok((parts[0].to_string(), parts[1].to_string(), parts[2].to_string()))
}

/// A snapshot currently being taken by this node.
pub struct RunningSnapshot {
    pub repository: String,
    pub name: String,
    pub uuid: String,
    pub include_global_state: bool,
    pub partial: bool,
    pub metadata: Option<Map<String, Value>>,
    pub feature_states: Vec<FeatureState>,
    pub data_streams: Vec<String>,
    pub indices: Vec<String>,
    pub start_time: i64,
    pub node_id: Option<String>,
    pub shards: BTreeMap<String, BTreeMap<u32, Arc<Mutex<ShardStatus>>>>,
    pub cancel: CancelToken,
    done: tokio::sync::watch::Sender<bool>,
}

impl RunningSnapshot {
    /// `SnapshotInfo` of an in-progress snapshot (ES `SnapshotInfo.inProgress`).
    pub fn info(&self) -> SnapshotInfo {
        SnapshotInfo {
            repository: self.repository.clone(),
            snapshot: self.name.clone(),
            uuid: self.uuid.clone(),
            version_id: Some(ES_VERSION_ID),
            version: Some(ES_VERSION.into()),
            indices: self.indices.clone(),
            data_streams: self.data_streams.clone(),
            feature_states: self.feature_states.clone(),
            include_global_state: Some(self.include_global_state),
            metadata: self.metadata.clone(),
            state: Some(SnapshotState::InProgress),
            reason: None,
            start_time: self.start_time,
            end_time: 0,
            total_shards: self.shards.values().map(|s| s.len() as u32).sum(),
            successful_shards: 0,
            failures: vec![],
            index_details: BTreeMap::new(),
        }
    }

    pub fn status(&self) -> SnapshotStatus {
        let indices = self
            .shards
            .iter()
            .map(|(n, s)| (n.clone(), s.iter().map(|(id, st)| (*id, st.lock().clone())).collect()))
            .collect();
        let now = now_millis();
        SnapshotStatus {
            snapshot: self.name.clone(),
            repository: self.repository.clone(),
            uuid: self.uuid.clone(),
            state: if self.cancel.is_cancelled() { "ABORTED".into() } else { "STARTED".into() },
            include_global_state: self.include_global_state,
            indices,
            start_time: self.start_time,
            time: now - self.start_time,
        }
    }
}

pub struct SnapshotsService {
    ctx: RwLock<RepositoryContext>,
    repos: RwLock<BTreeMap<String, Arc<Repository>>>,
    running: Mutex<Vec<Arc<RunningSnapshot>>>,
    /// Shards snapshotted concurrently per snapshot by this node.
    pub shard_concurrency: usize,
    /// Remote shard snapshots in flight per snapshot (each node bounds its
    /// own uploads).
    pub remote_shard_concurrency: usize,
}

impl SnapshotsService {
    pub fn new(ctx: RepositoryContext) -> Arc<Self> {
        Arc::new(Self {
            ctx: RwLock::new(ctx),
            repos: RwLock::new(BTreeMap::new()),
            running: Mutex::new(Vec::new()),
            shard_concurrency: 4,
            remote_shard_concurrency: 64,
        })
    }

    pub fn set_context(&self, ctx: RepositoryContext) {
        *self.ctx.write() = ctx;
    }

    // ----- repositories ----------------------------------------------------

    /// Register (or replace) a repository. `verify` runs `_verify` with no nodes.
    pub async fn put_repository(&self, meta: RepositoryMetadata, verify: bool) -> Result<()> {
        if self.running.lock().iter().any(|r| r.repository == meta.name) {
            return Err(SnapshotError::illegal_state(format!(
                "trying to modify or unregister repository [{}] that is currently used",
                meta.name
            )));
        }
        let ctx = self.ctx.read().clone();
        let repo = Repository::open(meta.clone(), &ctx).await?;
        if verify {
            repo.verify(&[]).await?;
        }
        self.repos.write().insert(meta.name.clone(), repo);
        Ok(())
    }

    /// Register an already-built repository (custom stores, tests).
    pub fn register_repository(&self, repo: Arc<Repository>) {
        self.repos.write().insert(repo.name().to_string(), repo);
    }

    /// `DELETE _snapshot/{names}` (wildcards allowed).
    pub fn delete_repositories(&self, names: &[String]) -> Result<Vec<String>> {
        let available: Vec<String> = self.repos.read().keys().cloned().collect();
        let (selected, missing) = resolve_names(names, &available);
        if let Some(m) = missing.first() {
            return Err(SnapshotError::repository_missing(m));
        }
        if selected.is_empty() {
            if let Some(n) = names.first() {
                return Err(SnapshotError::repository_missing(n));
            }
        }
        for n in &selected {
            if self.running.lock().iter().any(|r| &r.repository == n) {
                return Err(SnapshotError::illegal_state(format!(
                    "trying to modify or unregister repository [{n}] that is currently used"
                )));
            }
        }
        let mut repos = self.repos.write();
        for n in &selected {
            repos.remove(n);
        }
        Ok(selected)
    }

    pub fn repository(&self, name: &str) -> Result<Arc<Repository>> {
        self.repos.read().get(name).cloned().ok_or_else(|| SnapshotError::repository_missing(name))
    }

    pub fn repository_names(&self) -> Vec<String> {
        self.repos.read().keys().cloned().collect()
    }

    fn resolve_repositories(&self, names: &[String]) -> Result<Vec<Arc<Repository>>> {
        let available = self.repository_names();
        let (selected, missing) = resolve_names(names, &available);
        if let Some(m) = missing.first() {
            return Err(SnapshotError::repository_missing(m));
        }
        selected.iter().map(|n| self.repository(n)).collect()
    }

    /// `GET _snapshot` / `GET _snapshot/{names}` body.
    pub fn get_repositories_json(&self, names: &[String]) -> Result<Value> {
        let names = if names.is_empty() { vec!["_all".to_string()] } else { names.to_vec() };
        let mut m = Map::new();
        for r in self.resolve_repositories(&names)? {
            m.insert(r.name().to_string(), r.metadata().to_json());
        }
        Ok(Value::Object(m))
    }

    pub async fn verify_repository(&self, repo: &str, nodes: &[(String, String)]) -> Result<Value> {
        self.repository(repo)?.verify(nodes).await
    }

    pub async fn cleanup_repository(&self, repo: &str) -> Result<Value> {
        let r = self.repository(repo)?;
        Ok(r.cleanup().await?.to_json())
    }

    // ----- create ----------------------------------------------------------

    pub fn running_snapshots(&self, repo: Option<&str>) -> Vec<Arc<RunningSnapshot>> {
        self.running.lock().iter().filter(|r| repo.is_none_or(|n| r.repository == n)).cloned().collect()
    }

    /// Create a snapshot and wait for it to complete.
    pub async fn create_snapshot(self: &Arc<Self>, req: CreateSnapshotRequest) -> Result<SnapshotInfo> {
        let (_, handle) = self.start_snapshot(req).await?;
        handle.await.map_err(|e| SnapshotError::illegal_state(format!("snapshot task failed: {e}")))?
    }

    /// Validate and start a snapshot in the background
    /// (`wait_for_completion=false` → respond `{"accepted":true}`).
    pub async fn start_snapshot(
        self: &Arc<Self>,
        req: CreateSnapshotRequest,
    ) -> Result<(Arc<RunningSnapshot>, tokio::task::JoinHandle<Result<SnapshotInfo>>)> {
        let repo = self.repository(&req.repository)?;
        crate::repository::settings::validate_snapshot_name(repo.name(), &req.snapshot)?;
        repo.ensure_writable("create snapshot")?;
        let data = repo.repository_data().await?;
        if data.snapshot_by_name(&req.snapshot).is_some() {
            return Err(SnapshotError::invalid_snapshot_name(
                repo.name(),
                &req.snapshot,
                "snapshot with the same name already exists",
            ));
        }
        if !req.partial {
            let mut missing: Vec<String> = req
                .indices
                .iter()
                .filter(|i| i.shards.iter().any(|s| s.unavailable.is_some()))
                .map(|i| i.index_name.clone())
                .collect();
            missing.sort();
            if !missing.is_empty() {
                return Err(SnapshotError::snapshot(
                    repo.name(),
                    &req.snapshot,
                    format!("Indices don't have primary shards [{}]", missing.join(", ")),
                ));
            }
        }
        let mut seen = BTreeSet::new();
        for i in &req.indices {
            if !seen.insert(i.index_name.clone()) {
                return Err(SnapshotError::illegal_argument(format!("duplicate index [{}] in snapshot request", i.index_name)));
            }
        }
        let uuid = req.uuid.clone().unwrap_or_else(random_base64_uuid);
        repo.register_in_progress(&req.snapshot, &uuid)?;

        let mut shards = BTreeMap::new();
        for i in &req.indices {
            let mut m = BTreeMap::new();
            for s in &i.shards {
                let mut st = ShardStatus::new(ShardStage::Init);
                st.node = s.remote.as_ref().map(|r| r.node_id.clone()).or_else(|| req.node_id.clone());
                if let Some(r) = &s.unavailable {
                    st.stage = ShardStage::Failure;
                    st.reason = Some(r.clone());
                }
                m.insert(s.shard, Arc::new(Mutex::new(st)));
            }
            shards.insert(i.index_name.clone(), m);
        }
        let mut indices: Vec<String> = req.indices.iter().map(|i| i.index_name.clone()).collect();
        indices.sort();
        let (done, _) = tokio::sync::watch::channel(false);
        let running = Arc::new(RunningSnapshot {
            repository: repo.name().to_string(),
            name: req.snapshot.clone(),
            uuid,
            include_global_state: req.include_global_state,
            partial: req.partial,
            metadata: req.metadata.clone(),
            feature_states: req.feature_states.clone(),
            data_streams: req.data_streams.clone(),
            indices,
            start_time: now_millis(),
            node_id: req.node_id.clone(),
            shards,
            cancel: CancelToken::new(),
            done,
        });
        self.running.lock().push(running.clone());
        let me = self.clone();
        let r2 = running.clone();
        let handle = tokio::spawn(async move {
            let result = me.run_snapshot(&repo, &r2, data, req).await;
            repo.unregister_in_progress(&r2.name);
            me.running.lock().retain(|x| !Arc::ptr_eq(x, &r2));
            let _ = r2.done.send(true);
            result
        });
        Ok((running, handle))
    }

    async fn run_snapshot(
        &self,
        repo: &Arc<Repository>,
        running: &Arc<RunningSnapshot>,
        data: RepositoryData,
        req: CreateSnapshotRequest,
    ) -> Result<SnapshotInfo> {
        let index_ids: BTreeMap<String, String> =
            req.indices.iter().map(|i| (i.index_name.clone(), repo.resolve_index_id(&data, &i.index_name))).collect();
        let id_list: Vec<String> = index_ids.values().cloned().collect();
        repo.acquire_index_ids(&id_list);
        let r = self.run_snapshot_inner(repo, running, &data, &req, &index_ids).await;
        repo.release_index_ids(&id_list);
        r
    }

    async fn run_snapshot_inner(
        &self,
        repo: &Arc<Repository>,
        running: &Arc<RunningSnapshot>,
        data: &RepositoryData,
        req: &CreateSnapshotRequest,
        index_ids: &BTreeMap<String, String>,
    ) -> Result<SnapshotInfo> {
        let mut tasks = Vec::new();
        let mut remote_tasks: Vec<(RemoteShardRequest, Arc<Mutex<ShardStatus>>)> = Vec::new();
        for i in &req.indices {
            let incomplete_source = repo.settings().source_only && !crate::restore::has_complete_source(&i.index_metadata_json);
            for s in &i.shards {
                if s.unavailable.is_some() {
                    continue;
                }
                if incomplete_source {
                    let mut st = running.shards[&i.index_name][&s.shard].lock();
                    st.stage = ShardStage::Failure;
                    st.reason = Some(SOURCE_ONLY_INCOMPLETE.into());
                    continue;
                }
                if let Some(r) = &s.remote {
                    remote_tasks.push((
                        RemoteShardRequest {
                            snapshot_name: running.name.clone(),
                            snapshot_uuid: running.uuid.clone(),
                            index_name: i.index_name.clone(),
                            index_id: index_ids[&i.index_name].clone(),
                            shard: s.shard,
                            node_id: r.node_id.clone(),
                            base_gen: None,
                        },
                        running.shards[&i.index_name][&s.shard].clone(),
                    ));
                    continue;
                }
                tasks.push(ShardSnapshotTask {
                    snapshot_name: running.name.clone(),
                    snapshot_uuid: running.uuid.clone(),
                    index_id: index_ids[&i.index_name].clone(),
                    shard: s.shard,
                    dir: s.dir.clone(),
                    files: s.files.clone(),
                    status: running.shards[&i.index_name][&s.shard].clone(),
                    cancel: running.cancel.clone(),
                });
            }
        }
        let task_index: BTreeMap<(String, u32), String> = req
            .indices
            .iter()
            .flat_map(|i| i.shards.iter().map(move |s| ((index_ids[&i.index_name].clone(), s.shard), i.index_name.clone())))
            .collect();
        type ShardResults = Vec<(String, u32, Result<crate::repository::ShardSnapshotResult>)>;
        let local = futures::stream::iter(tasks.into_iter().map(|t| {
            let repo = repo.clone();
            let key = (t.index_id.clone(), t.shard);
            async move { (key.0, key.1, repo.snapshot_shard(data, t).await) }
        }))
        .buffer_unordered(self.shard_concurrency.max(1))
        .collect::<ShardResults>();
        let remote = futures::stream::iter(remote_tasks.into_iter().map(|(r, status)| {
            let repo = repo.clone();
            let uploader = req.remote.clone();
            let cancel = running.cancel.clone();
            async move {
                let (index_id, shard) = (r.index_id.clone(), r.shard);
                let res = match uploader {
                    None => Err(SnapshotError::illegal_state(format!("[{}][{shard}] no node can snapshot this shard", r.index_name))),
                    Some(up) => {
                        let target = repo.clone();
                        let snapshot_uuid = r.snapshot_uuid.clone();
                        repo.snapshot_shard_with(data, &index_id, shard, &snapshot_uuid, move |base| {
                            let mut r = r;
                            r.base_gen = base;
                            async move { up.0.snapshot_shard(target, r, status, cancel).await }
                        })
                        .await
                    }
                };
                (index_id, shard, res)
            }
        }))
        .buffer_unordered(self.remote_shard_concurrency.max(1))
        .collect::<ShardResults>();
        let (mut results, remote_results) = futures::future::join(local, remote).await;
        results.extend(remote_results);

        if running.cancel.is_cancelled() {
            return Err(SnapshotError::snapshot(
                repo.name(),
                &running.name,
                running.cancel.reason().unwrap_or_else(|| "Snapshot was aborted by deletion".into()),
            ));
        }

        let mut failures = Vec::new();
        let mut obsolete = Vec::new();
        let mut sizes: BTreeMap<String, (u64, u32)> = BTreeMap::new();
        for (id, shard, r) in results {
            let index = task_index[&(id.clone(), shard)].clone();
            match r {
                Ok(res) => {
                    {
                        // A remote shard's status follows the node's reports;
                        // its final numbers are the result's.
                        let mut st = running.shards[&index][&shard].lock();
                        if st.stage != ShardStage::Done {
                            st.stage = ShardStage::Done;
                            st.stats = res.stats;
                        }
                    }
                    if let Some(g) = res.old_gen {
                        obsolete.push((id, shard, g));
                    }
                    let e = sizes.entry(index).or_default();
                    e.0 += res.total_size;
                    e.1 = e.1.max(res.segment_count);
                }
                Err(e) => {
                    let st = &running.shards[&index][&shard];
                    let mut st = st.lock();
                    st.stage = ShardStage::Failure;
                    st.reason = Some(e.to_string());
                }
            }
        }
        let index_uuid = |i: &IndexSnapshotInput| -> String {
            crate::restore::flat_settings(&i.index_metadata_json)
                .get("index.uuid")
                .and_then(|v| v.as_str())
                .unwrap_or("_na_")
                .to_string()
        };
        let mut total = 0u32;
        for i in &req.indices {
            for s in &i.shards {
                total += 1;
                let st = running.shards[&i.index_name][&s.shard].lock();
                if st.stage == ShardStage::Failure {
                    failures.push(ShardFailure {
                        index: i.index_name.clone(),
                        index_uuid: index_uuid(i),
                        shard_id: s.shard,
                        reason: st.reason.clone().unwrap_or_default(),
                        node_id: st.node.clone().or_else(|| req.node_id.clone()),
                        status: "INTERNAL_SERVER_ERROR".into(),
                    });
                }
            }
        }
        failures.sort_by(|a, b| (&a.index, a.shard_id).cmp(&(&b.index, b.shard_id)));
        let state = if failures.is_empty() { SnapshotState::Success } else { SnapshotState::Partial };
        let index_details = req
            .indices
            .iter()
            .map(|i| {
                let (size, segs) = sizes.get(&i.index_name).copied().unwrap_or_default();
                (
                    i.index_name.clone(),
                    IndexDetails { shard_count: shard_count_of(i), size_in_bytes: size, max_segments_per_shard: segs },
                )
            })
            .collect();
        let info = SnapshotInfo {
            repository: repo.name().to_string(),
            snapshot: running.name.clone(),
            uuid: running.uuid.clone(),
            version_id: Some(ES_VERSION_ID),
            version: Some(ES_VERSION.into()),
            indices: running.indices.clone(),
            data_streams: req.data_streams.clone(),
            feature_states: req.feature_states.clone(),
            include_global_state: Some(req.include_global_state),
            metadata: req.metadata.clone(),
            state: Some(state),
            reason: None,
            start_time: running.start_time,
            end_time: now_millis(),
            total_shards: total,
            successful_shards: total - failures.len() as u32,
            failures,
            index_details,
        };
        let finalize = FinalizeRequest {
            info: info.clone(),
            indices: req
                .indices
                .iter()
                .map(|i| FinalizeIndex {
                    name: i.index_name.clone(),
                    index_id: index_ids[&i.index_name].clone(),
                    metadata: i.index_metadata_json.clone(),
                    shard_count: shard_count_of(i),
                })
                .collect(),
            global_metadata: if req.include_global_state { req.global_metadata_json.clone() } else { None },
            obsolete_shard_gens: obsolete,
        };
        repo.finalize_snapshot(finalize).await?;
        Ok(info)
    }

    // ----- get -------------------------------------------------------------

    /// `GET _snapshot/{repo}/{snapshot}` response body.
    pub async fn get_snapshots(&self, req: &GetSnapshotsRequest) -> Result<Value> {
        req.validate()?;
        let repos = self.resolve_repositories(&req.repositories)?;
        let mut all: Vec<SnapshotInfo> = Vec::new();
        let mut failures = Map::new();
        for repo in &repos {
            match self.snapshots_in_repo(repo, req).await {
                Ok(mut v) => all.append(&mut v),
                Err(e) if e.kind == "snapshot_missing_exception" => return Err(e),
                Err(e) if repos.len() == 1 => return Err(e),
                Err(e) => {
                    failures.insert(repo.name().to_string(), e.to_json()["error"].clone());
                }
            }
        }

        if let Some(states) = &req.states {
            all.retain(|s| s.state.is_some_and(|state| states.contains(&state)));
        }
        // slm policy filter
        if !req.slm_policy_filter.is_empty() {
            all.retain(|s| policy_matches(&req.slm_policy_filter, s.slm_policy()));
        }
        // from_sort_value
        if let Some(fsv) = &req.from_sort_value {
            let v = req.sort.parse_value(fsv)?;
            all.retain(|s| if req.desc { req.sort.value(s) <= v } else { req.sort.value(s) >= v });
        }
        let sort = req.sort;
        all.sort_by(|a, b| {
            let o = sort
                .value(a)
                .cmp(&sort.value(b))
                .then_with(|| a.snapshot.cmp(&b.snapshot))
                .then_with(|| a.repository.cmp(&b.repository));
            if req.desc { o.reverse() } else { o }
        });
        let total = all.len();
        let mut rest: Vec<SnapshotInfo> = if let Some(after) = &req.after {
            let (val, repo, name) = decode_after(after)?;
            let v = sort.parse_value(&val)?;
            all.into_iter()
                .filter(|s| {
                    let key = (sort.value(s), s.snapshot.clone(), s.repository.clone());
                    let after_key = (v.clone(), name.clone(), repo.clone());
                    if req.desc { key < after_key } else { key > after_key }
                })
                .collect()
        } else {
            all
        };
        let rest: Vec<SnapshotInfo> = rest.drain(..).skip(req.offset).collect();
        let take = req.size.unwrap_or(usize::MAX).min(rest.len());
        let remaining = rest.len() - take;
        let page = &rest[..take];
        let opts = RenderOptions {
            verbose: req.verbose,
            include_repository: req.include_repository,
            index_names: req.index_names,
            index_details: req.index_details,
            human: req.human,
        };
        let mut out = Map::new();
        out.insert("snapshots".into(), Value::Array(page.iter().map(|s| s.render(opts)).collect()));
        if !failures.is_empty() {
            out.insert("failures".into(), Value::Object(failures));
        }
        if remaining > 0 {
            if let Some(last) = page.last() {
                out.insert("next".into(), json!(encode_after(&sort.value(last), &last.repository, &last.snapshot)));
            }
        }
        out.insert("total".into(), json!(total));
        out.insert("remaining".into(), json!(remaining));
        Ok(Value::Object(out))
    }

    async fn snapshots_in_repo(&self, repo: &Arc<Repository>, req: &GetSnapshotsRequest) -> Result<Vec<SnapshotInfo>> {
        let data = repo.repository_data().await?;
        let running = self.running_snapshots(Some(repo.name()));
        let mut names: Vec<String> = data.snapshot_names();
        let running_names: Vec<String> = running.iter().map(|r| r.name.clone()).collect();
        for n in &running_names {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
        let mut exprs = Vec::new();
        let mut want_current = false;
        for s in &req.snapshots {
            if s == "_current" {
                want_current = true;
            } else {
                exprs.push(s.clone());
            }
        }
        let mut selected: BTreeSet<String> = BTreeSet::new();
        if !exprs.is_empty() || !want_current {
            let (sel, missing) = resolve_names(&exprs, &names);
            if !missing.is_empty() && !req.ignore_unavailable {
                return Err(SnapshotError::snapshot_missing(repo.name(), &missing[0]));
            }
            selected.extend(sel);
        }
        if want_current {
            selected.extend(running_names.iter().cloned());
        }
        let mut out = Vec::new();
        for name in selected {
            if let Some(r) = running.iter().find(|r| r.name == name) {
                if data.snapshot_by_name(&name).is_none() {
                    out.push(r.info());
                    continue;
                }
            }
            let Some(entry) = data.snapshot_by_name(&name) else { continue };
            if req.verbose {
                match repo.read_snapshot_info(&entry.uuid).await {
                    Ok(i) => out.push(i),
                    Err(e) if e.is_not_found() && req.ignore_unavailable => {}
                    Err(e) => return Err(e),
                }
            } else {
                out.push(SnapshotInfo {
                    repository: repo.name().to_string(),
                    snapshot: entry.name.clone(),
                    uuid: entry.uuid.clone(),
                    version_id: None,
                    version: None,
                    indices: data.indices_of(&entry.uuid),
                    data_streams: vec![],
                    feature_states: vec![],
                    include_global_state: None,
                    metadata: None,
                    state: Some(entry.snapshot_state()),
                    reason: None,
                    start_time: 0,
                    end_time: 0,
                    total_shards: 0,
                    successful_shards: 0,
                    failures: vec![],
                    index_details: BTreeMap::new(),
                });
            }
        }
        if !req.verbose {
            // Sorting by start time without loading SnapshotInfo: use index-N times.
            out.sort_by_key(|s| data.snapshot_by_name(&s.snapshot).map(|e| e.start_time_millis).unwrap_or(i64::MAX));
        }
        Ok(out)
    }

    /// Completed snapshot's `SnapshotInfo` by name.
    pub async fn snapshot_info(&self, repo: &str, name: &str) -> Result<SnapshotInfo> {
        let r = self.repository(repo)?;
        let data = r.repository_data().await?;
        let e = data.snapshot_by_name(name).ok_or_else(|| SnapshotError::snapshot_missing(repo, name))?;
        r.read_snapshot_info(&e.uuid).await
    }

    // ----- status ----------------------------------------------------------

    /// `GET _snapshot/_status`, `GET _snapshot/{repo}/_status`,
    /// `GET _snapshot/{repo}/{names}/_status`.
    pub async fn snapshot_status(&self, repo: Option<&str>, names: &[String], ignore_unavailable: bool, human: bool) -> Result<Value> {
        let mut out = Vec::new();
        let names: Vec<String> = names.iter().filter(|n| !n.is_empty()).cloned().collect();
        if names.is_empty() || names.iter().all(|n| n == "_current") {
            if let Some(r) = repo {
                self.repository(r)?;
            }
            for r in self.running_snapshots(repo) {
                out.push(r.status().to_json(human));
            }
            return Ok(json!({ "snapshots": out }));
        }
        let repo_name = repo.ok_or_else(|| SnapshotError::illegal_argument("repository is required when snapshots are specified"))?;
        let r = self.repository(repo_name)?;
        let data = r.repository_data().await?;
        let running = self.running_snapshots(Some(repo_name));
        let mut seen = BTreeSet::new();
        for name in &names {
            if name == "_current" {
                for rs in &running {
                    if seen.insert(rs.name.clone()) {
                        out.push(rs.status().to_json(human));
                    }
                }
                continue;
            }
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(rs) = running.iter().find(|x| &x.name == name) {
                out.push(rs.status().to_json(human));
                continue;
            }
            let Some(entry) = data.snapshot_by_name(name) else {
                if ignore_unavailable {
                    continue;
                }
                return Err(SnapshotError::snapshot_missing(repo_name, name));
            };
            out.push(self.completed_status(&r, &data, &entry.uuid).await?.to_json(human));
        }
        Ok(json!({ "snapshots": out }))
    }

    async fn completed_status(&self, repo: &Arc<Repository>, data: &RepositoryData, uuid: &str) -> Result<SnapshotStatus> {
        let info = repo.read_snapshot_info(uuid).await?;
        let mut indices = BTreeMap::new();
        for index in &info.indices {
            let Some(ie) = data.indices.get(index) else { continue };
            let shard_count =
                info.index_details.get(index).map(|d| d.shard_count).unwrap_or(ie.shard_generations.len() as u32);
            let mut shards = BTreeMap::new();
            for shard in 0..shard_count {
                if let Some(f) = info.failures.iter().find(|f| &f.index == index && f.shard_id == shard) {
                    let mut st = ShardStatus::new(ShardStage::Failure);
                    st.reason = Some(f.reason.clone());
                    st.node = f.node_id.clone();
                    shards.insert(shard, st);
                    continue;
                }
                let meta = repo.read_shard_snapshot(&ie.id, shard, uuid).await?;
                let mut st = ShardStatus::new(ShardStage::Done);
                st.stats = meta.stats();
                shards.insert(shard, st);
            }
            indices.insert(index.clone(), shards);
        }
        Ok(SnapshotStatus {
            snapshot: info.snapshot.clone(),
            repository: repo.name().to_string(),
            uuid: info.uuid.clone(),
            state: match info.state {
                Some(SnapshotState::Failed) => "FAILED".into(),
                _ => "SUCCESS".into(),
            },
            include_global_state: info.include_global_state.unwrap_or(false),
            indices,
            start_time: info.start_time,
            time: info.duration_millis(),
        })
    }

    // ----- delete / clone --------------------------------------------------

    /// `DELETE _snapshot/{repo}/{names}`: aborts matching running snapshots and
    /// deletes matching completed ones. Returns `{"acknowledged":true}`.
    pub async fn delete_snapshots(&self, repo: &str, names: &[String]) -> Result<Value> {
        let r = self.repository(repo)?;
        r.ensure_writable("delete snapshot")?;
        let data = r.repository_data().await?;
        let running = self.running_snapshots(Some(repo));
        let mut all_names = data.snapshot_names();
        all_names.extend(running.iter().map(|x| x.name.clone()));
        let (selected, missing) = resolve_names(names, &all_names);
        if let Some(m) = missing.first() {
            return Err(SnapshotError::snapshot_missing(repo, m));
        }
        // Abort running ones and wait for them to stop.
        for rs in running.iter().filter(|x| selected.contains(&x.name)) {
            rs.cancel.cancel("Snapshot was aborted by deletion");
            let mut rx = rs.done.subscribe();
            while !*rx.borrow() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        }
        let data = r.repository_data().await?;
        let uuids: BTreeSet<String> = selected
            .iter()
            .filter_map(|n| data.snapshot_by_name(n).map(|e| e.uuid.clone()))
            .collect();
        if !uuids.is_empty() {
            r.delete_snapshots(&uuids).await?;
        }
        Ok(json!({"acknowledged": true}))
    }

    /// `PUT _snapshot/{repo}/{source}/_clone/{target}` with `{"indices": ...}`.
    pub async fn clone_snapshot(&self, repo: &str, source: &str, target: &str, body: &Value) -> Result<Value> {
        let indices = body.get("indices").map(str_list).unwrap_or_default();
        if indices.is_empty() {
            return Err(SnapshotError::validation(&["indices patterns are empty".into()]));
        }
        let r = self.repository(repo)?;
        if self.running_snapshots(Some(repo)).iter().any(|x| x.name == target) {
            return Err(SnapshotError::invalid_snapshot_name(repo, target, "snapshot with the same name is already in-progress"));
        }
        r.clone_snapshot(source, target, &indices).await?;
        Ok(json!({"acknowledged": true}))
    }

    // ----- restore helpers -------------------------------------------------

    /// Restore one shard of `index_name` from `snapshot` into `target_dir`.
    pub async fn restore_shard(
        &self,
        repo: &str,
        snapshot: &str,
        index_name: &str,
        shard: u32,
        target_dir: &Path,
        cancel: &CancelToken,
    ) -> Result<crate::repository::shard::RestoreShardResult> {
        let r = self.repository(repo)?;
        let data = r.repository_data().await?;
        let e = data.snapshot_by_name(snapshot).ok_or_else(|| SnapshotError::snapshot_missing(repo, snapshot))?;
        let ie = data.indices.get(index_name).ok_or_else(|| SnapshotError::index_not_found(index_name))?;
        if !ie.snapshots.contains(&e.uuid) {
            return Err(SnapshotError::index_not_found(index_name));
        }
        r.restore_shard(&e.uuid, &ie.id, shard, target_dir, cancel).await
    }

    pub async fn read_index_metadata(&self, repo: &str, snapshot: &str, index_name: &str) -> Result<Value> {
        let r = self.repository(repo)?;
        let data = r.repository_data().await?;
        let e = data.snapshot_by_name(snapshot).ok_or_else(|| SnapshotError::snapshot_missing(repo, snapshot))?;
        r.read_index_metadata(&data, &e.uuid, index_name).await
    }

    pub async fn read_global_metadata(&self, repo: &str, snapshot: &str) -> Result<Value> {
        let r = self.repository(repo)?;
        let data = r.repository_data().await?;
        let e = data.snapshot_by_name(snapshot).ok_or_else(|| SnapshotError::snapshot_missing(repo, snapshot))?;
        r.read_global_metadata(&e.uuid).await
    }
}

fn shard_count_of(i: &IndexSnapshotInput) -> u32 {
    let from_settings = crate::restore::flat_settings(&i.index_metadata_json)
        .get("index.number_of_shards")
        .and_then(|v| v.as_str().and_then(|s| s.parse::<u32>().ok()).or_else(|| v.as_u64().map(|n| n as u32)));
    let from_shards = i.shards.iter().map(|s| s.shard + 1).max().unwrap_or(0);
    from_settings.unwrap_or(0).max(from_shards)
}

fn policy_matches(filters: &[String], policy: Option<&str>) -> bool {
    let mut included = false;
    let mut matched_any_inclusion = false;
    for f in filters {
        if let Some(neg) = f.strip_prefix('-') {
            if let Some(p) = policy {
                if simple_match(neg, p) {
                    return false;
                }
            }
            continue;
        }
        matched_any_inclusion = true;
        if f == "_none" {
            if policy.is_none() {
                included = true;
            }
        } else if let Some(p) = policy {
            if simple_match(f, p) {
                included = true;
            }
        }
    }
    included || (!matched_any_inclusion && policy.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_validation() {
        let mut p = BTreeMap::new();
        p.insert("verbose".to_string(), "false".to_string());
        p.insert("sort".to_string(), "name".to_string());
        let e = GetSnapshotsRequest::from_params("r", "_all", &p).unwrap_err();
        assert_eq!(e.reason, "Validation Failed: 1: can't use non-default sort with verbose=false;");
        let mut p = BTreeMap::new();
        p.insert("sort".to_string(), "bogus".to_string());
        assert_eq!(GetSnapshotsRequest::from_params("r", "_all", &p).unwrap_err().reason, "unknown sort order [bogus]");
        let mut p = BTreeMap::new();
        p.insert("size".to_string(), "0".to_string());
        assert!(GetSnapshotsRequest::from_params("r", "_all", &p).is_err());
    }

    #[test]
    fn create_ignores_unrecognized_body_entries() {
        let body = json!({"indices": "a", "index_settings": {"index.routing.rebalance.enable": "none"}, "unknown": "value"});
        let req = CreateSnapshotParams::from_json(Some(&body)).unwrap();
        assert_eq!(req.indices, vec!["a"]);
    }

    #[test]
    fn snapshot_state_params() {
        let params = BTreeMap::from([("state".into(), "SUCCESS,PARTIAL".into())]);
        let req = GetSnapshotsRequest::from_params("r", "_all", &params).unwrap();
        assert_eq!(req.states, Some(vec![SnapshotState::Success, SnapshotState::Partial]));
        for invalid in ["", " ", "success", "SUCCESS,BOGUS"] {
            let params = BTreeMap::from([("state".into(), invalid.into())]);
            assert!(GetSnapshotsRequest::from_params("r", "_all", &params).is_err());
        }
    }

    #[test]
    fn after_cursor_roundtrip() {
        let c = encode_after(&SortValue::Num(42), "repo", "snap");
        assert_eq!(decode_after(&c).unwrap(), ("42".into(), "repo".into(), "snap".into()));
        assert!(decode_after("!!").is_err());
    }

    #[test]
    fn policy_filter() {
        assert!(policy_matches(&["_none".into()], None));
        assert!(!policy_matches(&["_none".into()], Some("p")));
        assert!(policy_matches(&["da*".into()], Some("daily")));
        assert!(!policy_matches(&["*".into(), "-daily".into()], Some("daily")));
    }

    #[test]
    fn create_params() {
        let p = CreateSnapshotParams::from_json(Some(&json!({"indices":"a,b*","include_global_state":false,"metadata":{"taken_by":"me"}}))).unwrap();
        assert_eq!(p.indices, vec!["a", "b*"]);
        assert!(!p.include_global_state);
        let big = "x".repeat(2000);
        let e = CreateSnapshotParams::from_json(Some(&json!({"metadata":{"k": big}}))).unwrap_err();
        assert!(e.reason.contains("metadata must be smaller than 1024 bytes"));
    }
}
