//! Blob-store repository (ES `BlobStoreRepository`).
//!
//! Layout (relative to the repository root / `base_path`):
//!
//! ```text
//! index-N                         RepositoryData JSON (generation N)
//! index.latest                    8-byte big-endian N
//! snap-{uuid}.dat                 SnapshotInfo
//! meta-{uuid}.dat                 global metadata JSON
//! indices/{index_id}/meta-{blob}.dat          index metadata JSON
//! indices/{index_id}/{shard}/index-{gen}      shard snapshot list (BlobStoreIndexShardSnapshots)
//! indices/{index_id}/{shard}/snap-{uuid}.dat  shard snapshot (BlobStoreIndexShardSnapshot)
//! indices/{index_id}/{shard}/__{uuid}[.partN] data blobs
//! tests-{seed}/                   verification blobs
//! ```
//! Metadata blobs are JSON, LZ4-compressed (`LZ4\0` prefix) when `compress`
//! is set; data blobs are raw file bytes split into `chunk_size` parts.

pub mod data;
mod ops;
pub mod settings;
pub mod shard;

pub use data::{EMPTY_REPO_GEN, IndexEntry, RepositoryData, SnapshotEntry};
pub use ops::{CleanupResult, FinalizeIndex, FinalizeRequest};
pub use settings::{RepositoryContext, RepositoryMetadata, RepositorySettings, S3ClientSettings};
pub use shard::{FileInfo, ShardSnapshotFiles, ShardSnapshotMeta, ShardSnapshotResult, ShardSnapshots};

use crate::blobstore::{BlobContainer, BlobStore, decode_metadata_blob, encode_metadata_blob};
use crate::error::{Result, SnapshotError};
use crate::util::RateLimiter;
use bytes::Bytes;
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::sync::Arc;

pub const INDEX_FILE_PREFIX: &str = "index-";
pub const INDEX_LATEST_BLOB: &str = "index.latest";
pub const SNAPSHOT_PREFIX: &str = "snap-";
pub const METADATA_PREFIX: &str = "meta-";
pub const DATA_BLOB_PREFIX: &str = "__";
pub const TESTS_PREFIX: &str = "tests-";

pub fn snapshot_blob_name(uuid: &str) -> String {
    format!("{SNAPSHOT_PREFIX}{uuid}.dat")
}

pub fn metadata_blob_name(uuid: &str) -> String {
    format!("{METADATA_PREFIX}{uuid}.dat")
}

type ShardKey = (String, u32);

pub struct Repository {
    pub(crate) meta: RepositoryMetadata,
    pub(crate) settings: RepositorySettings,
    pub(crate) store: Arc<dyn BlobStore>,
    pub(crate) root: BlobContainer,
    pub(crate) snapshot_rate: Arc<RateLimiter>,
    pub(crate) restore_rate: Arc<RateLimiter>,
    /// Serialises root `index-N` updates from this process.
    pub(crate) finalize_lock: tokio::sync::Mutex<()>,
    shard_locks: Mutex<HashMap<ShardKey, Arc<tokio::sync::Mutex<()>>>>,
    /// Shard generations written by this process but not yet (or never)
    /// published in `index-N`. Shard index blobs are cumulative, so the
    /// newest one is always the right base for the next shard operation.
    pub(crate) pending_shard_gens: Mutex<HashMap<ShardKey, String>>,
    /// Snapshots running in this repository (name → uuid); their shard-level
    /// entries must survive concurrent deletes/cleanup.
    pub(crate) in_progress: Mutex<HashMap<String, String>>,
    /// Index ids handed out to running snapshots for indices new to the repo.
    pub(crate) reserved_index_ids: Mutex<HashMap<String, String>>,
    /// Index ids that running snapshots are writing into (refcounted); their
    /// directories must not be removed by deletes.
    pub(crate) active_index_ids: Mutex<HashMap<String, usize>>,
    _input_owner: Option<Box<dyn Send + Sync>>,
}

impl std::fmt::Debug for Repository {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Repository").field("name", &self.meta.name).field("type", &self.meta.repo_type).finish()
    }
}

impl Repository {
    pub async fn open(meta: RepositoryMetadata, ctx: &RepositoryContext) -> Result<Arc<Self>> {
        let (store, settings) = settings::build_store(&meta, ctx).await?;
        Ok(Self::with_store(meta, store, settings))
    }

    /// Retain the caller's admitted input storage with every repository clone.
    /// Storage implementations stay independent of any particular breaker.
    pub async fn open_owned(meta:RepositoryMetadata,ctx:&RepositoryContext,owner:Box<dyn Send+Sync>)->Result<Arc<Self>> {
        let (store,settings)=settings::build_store(&meta,ctx).await?;
        let mut repo=Self::with_store(meta,store,settings);
        Arc::get_mut(&mut repo).expect("new repository has one owner")._input_owner=Some(owner);
        Ok(repo)
    }

    /// Build a repository over an existing store (tests, custom stores).
    pub fn with_store(meta: RepositoryMetadata, store: Arc<dyn BlobStore>, mut settings: RepositorySettings) -> Arc<Self> {
        if store.is_read_only() {
            settings.readonly = true;
        }
        Arc::new(Self {
            snapshot_rate: Arc::new(RateLimiter::new(settings.max_snapshot_bytes_per_sec)),
            restore_rate: Arc::new(RateLimiter::new(settings.max_restore_bytes_per_sec)),
            root: BlobContainer::new(store.clone(), ""),
            store,
            meta,
            settings,
            finalize_lock: tokio::sync::Mutex::new(()),
            shard_locks: Mutex::new(HashMap::new()),
            pending_shard_gens: Mutex::new(HashMap::new()),
            in_progress: Mutex::new(HashMap::new()),
            reserved_index_ids: Mutex::new(HashMap::new()),
            active_index_ids: Mutex::new(HashMap::new()),
            _input_owner: None,
        })
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }

    pub fn metadata(&self) -> &RepositoryMetadata {
        &self.meta
    }

    pub fn settings(&self) -> &RepositorySettings {
        &self.settings
    }

    pub fn is_readonly(&self) -> bool {
        self.settings.readonly
    }

    pub fn store(&self) -> &Arc<dyn BlobStore> {
        &self.store
    }

    pub fn root(&self) -> &BlobContainer {
        &self.root
    }

    pub fn snapshot_rate_limiter(&self) -> &RateLimiter {
        &self.snapshot_rate
    }

    pub fn restore_rate_limiter(&self) -> &RateLimiter {
        &self.restore_rate
    }

    pub(crate) fn ensure_writable(&self, what: &str) -> Result<()> {
        if self.settings.readonly {
            Err(SnapshotError::repository(self.name(), format!("cannot {what} in a readonly repository")))
        } else {
            Ok(())
        }
    }

    pub fn indices_container(&self) -> BlobContainer {
        self.root.child("indices")
    }

    pub fn index_container(&self, index_id: &str) -> BlobContainer {
        self.indices_container().child(index_id)
    }

    pub fn shard_container(&self, index_id: &str, shard: u32) -> BlobContainer {
        self.index_container(index_id).child(&shard.to_string())
    }

    pub(crate) fn acquire_index_ids(&self, ids: &[String]) {
        let mut a = self.active_index_ids.lock();
        for id in ids {
            *a.entry(id.clone()).or_default() += 1;
        }
    }

    pub(crate) fn release_index_ids(&self, ids: &[String]) {
        let mut a = self.active_index_ids.lock();
        for id in ids {
            if let Some(c) = a.get_mut(id) {
                *c -= 1;
                if *c == 0 {
                    a.remove(id);
                }
            }
        }
    }

    pub(crate) fn shard_lock(&self, index_id: &str, shard: u32) -> Arc<tokio::sync::Mutex<()>> {
        self.shard_locks.lock().entry((index_id.to_string(), shard)).or_default().clone()
    }

    // ----- metadata blobs -------------------------------------------------

    pub(crate) async fn read_meta<T: DeserializeOwned>(&self, c: &BlobContainer, name: &str) -> Result<T> {
        let raw = c.read_blob(name).await?;
        let json = decode_metadata_blob(name, &raw)?;
        serde_json::from_slice(&json).map_err(|e| {
            SnapshotError::new("corrupt_state_exception", 500, format!("failed to parse blob [{}{name}]: {e}", c.path()))
        })
    }

    pub(crate) async fn write_meta<T: Serialize>(&self, c: &BlobContainer, name: &str, v: &T, fail_if_exists: bool) -> Result<()> {
        let json = serde_json::to_vec(v)?;
        c.write_blob_atomic(name, encode_metadata_blob(&json, self.settings.compress), fail_if_exists).await
    }

    pub(crate) async fn read_meta_value(&self, c: &BlobContainer, name: &str) -> Result<serde_json::Value> {
        self.read_meta(c, name).await
    }

    // ----- root generations -----------------------------------------------

    /// Latest root generation: max of the `index-N` listing and `index.latest`.
    pub async fn latest_generation(&self) -> Result<i64> {
        let mut best = EMPTY_REPO_GEN;
        let mut listed = false;
        match self.root.list_blobs_by_prefix(INDEX_FILE_PREFIX).await {
            Ok(blobs) => {
                listed = true;
                for name in blobs.keys() {
                    if let Some(g) = name.strip_prefix(INDEX_FILE_PREFIX).and_then(|n| n.parse::<i64>().ok()) {
                        best = best.max(g);
                    }
                }
            }
            Err(e) if e.kind == "unsupported_operation_exception" => {}
            Err(e) => return Err(e),
        }
        match self.root.read_blob(INDEX_LATEST_BLOB).await {
            Ok(b) if b.len() == 8 => {
                let g = i64::from_be_bytes(b[..8].try_into().unwrap());
                // index.latest may point at a generation that was deleted
                // concurrently; only trust it if listing was impossible or
                // it agrees with the listing.
                if !listed || g > best {
                    if !listed || self.root.blob_exists(&format!("{INDEX_FILE_PREFIX}{g}")).await.unwrap_or(false) {
                        best = best.max(g);
                    }
                }
            }
            Ok(_) => {}
            Err(e) if e.is_not_found() => {}
            Err(e) => return Err(e),
        }
        Ok(best)
    }

    /// Read the current `RepositoryData`.
    pub async fn repository_data(&self) -> Result<RepositoryData> {
        for _ in 0..5 {
            let gen_ = self.latest_generation().await?;
            if gen_ == EMPTY_REPO_GEN {
                return Ok(RepositoryData::empty());
            }
            match self.read_meta::<RepositoryData>(&self.root, &format!("{INDEX_FILE_PREFIX}{gen_}")).await {
                Ok(mut d) => {
                    d.generation = gen_;
                    return Ok(d);
                }
                // Deleted by a concurrent writer between list and read: retry.
                Err(e) if e.is_not_found() => continue,
                Err(e) => return Err(e),
            }
        }
        Err(SnapshotError::repository(self.name(), "could not read repository data: index-N blobs keep changing"))
    }

    /// Write `data` as generation `expected_gen + 1`, failing with ES's
    /// concurrent-modification error if the repository moved on.
    pub async fn write_repository_data(&self, data: &RepositoryData, expected_gen: i64) -> Result<i64> {
        self.ensure_writable("modify repository data")?;
        let current = self.latest_generation().await?;
        if current != expected_gen {
            return Err(self.concurrent_modification(expected_gen, current));
        }
        let new_gen = expected_gen + 1;
        let json = data.to_json_bytes();
        match self
            .root
            .write_blob_atomic(
                &format!("{INDEX_FILE_PREFIX}{new_gen}"),
                encode_metadata_blob(&json, self.settings.compress),
                true,
            )
            .await
        {
            Ok(()) => {}
            Err(e) if e.is_already_exists() => {
                let actual = self.latest_generation().await.unwrap_or(new_gen);
                return Err(self.concurrent_modification(expected_gen, actual));
            }
            Err(e) => return Err(e),
        }
        self.root
            .write_blob_atomic(INDEX_LATEST_BLOB, Bytes::copy_from_slice(&new_gen.to_be_bytes()), false)
            .await?;
        if expected_gen >= 0 {
            let old: Vec<String> = ((expected_gen - 1000).max(0)..new_gen).map(|g| format!("{INDEX_FILE_PREFIX}{g}")).collect();
            if let Err(e) = self.root.delete_blobs_ignoring_if_not_exists(&old).await {
                tracing::warn!("[{}] failed to delete old index-N blobs: {e}", self.name());
            }
        }
        Ok(new_gen)
    }

    pub(crate) fn concurrent_modification(&self, expected: i64, actual: i64) -> SnapshotError {
        SnapshotError::repository(
            self.name(),
            format!(
                "concurrent modification of the index-N file, expected current generation [{expected}], actual current generation [{actual}]"
            ),
        )
    }

    /// Current shard generation for (index_id, shard): newest locally written,
    /// else the one recorded in `data`.
    pub(crate) fn current_shard_gen(&self, data: &RepositoryData, index_id: &str, shard: u32) -> Option<String> {
        if let Some(g) = self.pending_shard_gens.lock().get(&(index_id.to_string(), shard)) {
            return Some(g.clone());
        }
        data.index_by_id(index_id)
            .and_then(|(_, e)| e.shard_generations.get(shard as usize).cloned().flatten())
    }

    pub(crate) async fn read_shard_snapshots(&self, index_id: &str, shard: u32, gen_: Option<&str>) -> Result<ShardSnapshots> {
        match gen_ {
            None => Ok(ShardSnapshots::default()),
            Some(g) => {
                let c = self.shard_container(index_id, shard);
                match self.read_meta(&c, &format!("{INDEX_FILE_PREFIX}{g}")).await {
                    Ok(s) => Ok(s),
                    Err(e) if e.is_not_found() => Err(SnapshotError::repository(
                        self.name(),
                        format!("shard generation [{g}] for [{index_id}][{shard}] is missing"),
                    )),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Read a snapshot's `SnapshotInfo` (`snap-{uuid}.dat`).
    pub async fn read_snapshot_info(&self, uuid: &str) -> Result<crate::snapshot::SnapshotInfo> {
        let mut info: crate::snapshot::SnapshotInfo = self.read_meta(&self.root, &snapshot_blob_name(uuid)).await?;
        info.repository = self.name().to_string();
        Ok(info)
    }

    /// Global metadata stored with a snapshot.
    pub async fn read_global_metadata(&self, uuid: &str) -> Result<serde_json::Value> {
        self.read_meta_value(&self.root, &metadata_blob_name(uuid)).await
    }

    /// Index metadata JSON for `index_name` as stored in snapshot `uuid`.
    pub async fn read_index_metadata(&self, data: &RepositoryData, uuid: &str, index_name: &str) -> Result<serde_json::Value> {
        let entry = data
            .snapshot_by_uuid(uuid)
            .ok_or_else(|| SnapshotError::snapshot_missing(self.name(), uuid))?;
        let ie = data.indices.get(index_name).ok_or_else(|| SnapshotError::index_not_found(index_name))?;
        let ident = entry
            .index_metadata_lookup
            .get(&ie.id)
            .ok_or_else(|| SnapshotError::index_not_found(index_name))?;
        let blob = data
            .index_metadata_identifiers
            .get(ident)
            .ok_or_else(|| SnapshotError::repository(self.name(), format!("missing index metadata identifier [{ident}]")))?;
        self.read_meta_value(&self.index_container(&ie.id), &metadata_blob_name(blob)).await
    }
}
