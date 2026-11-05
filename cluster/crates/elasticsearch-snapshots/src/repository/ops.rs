//! Repository-level operations: snapshot finalization, deletion, clone,
//! cleanup and verification.

use super::{
    DATA_BLOB_PREFIX, INDEX_FILE_PREFIX, METADATA_PREFIX, Repository, RepositoryData, SNAPSHOT_PREFIX, ShardSnapshots,
    SnapshotEntry, TESTS_PREFIX, metadata_blob_name, snapshot_blob_name,
};
use crate::blobstore::DeleteResult;
use crate::error::{Result, SnapshotError};
use crate::snapshot::{IndexDetails, SnapshotInfo, SnapshotState};
use crate::util::{random_base64_uuid, sha256_hex};
use bytes::Bytes;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// One index of a snapshot being finalized.
#[derive(Debug, Clone)]
pub struct FinalizeIndex {
    pub name: String,
    pub index_id: String,
    pub metadata: Value,
    pub shard_count: u32,
}

#[derive(Debug, Clone)]
pub struct FinalizeRequest {
    pub info: SnapshotInfo,
    pub indices: Vec<FinalizeIndex>,
    pub global_metadata: Option<Value>,
    /// Shard generations superseded by this snapshot: (index_id, shard, gen).
    pub obsolete_shard_gens: Vec<(String, u32, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CleanupResult {
    pub deleted_bytes: u64,
    pub deleted_blobs: u64,
}

impl CleanupResult {
    /// `POST _snapshot/{repo}/_cleanup` response.
    pub fn to_json(&self) -> Value {
        json!({"results": {"deleted_bytes": self.deleted_bytes, "deleted_blobs": self.deleted_blobs}})
    }
}

/// Metadata identifier: content hash, so identical metadata is stored once.
fn metadata_identifier(index_id: &str, metadata: &Value) -> String {
    let bytes = serde_json::to_vec(metadata).unwrap_or_default();
    format!("{index_id}-{}", &sha256_hex(&bytes)[..32])
}

impl Repository {
    /// Index id for `index_name`: existing in the repository, reserved by a
    /// running snapshot, or freshly generated.
    pub fn resolve_index_id(&self, data: &RepositoryData, index_name: &str) -> String {
        if let Some(e) = data.indices.get(index_name) {
            return e.id.clone();
        }
        self.reserved_index_ids.lock().entry(index_name.to_string()).or_insert_with(random_base64_uuid).clone()
    }

    pub(crate) fn register_in_progress(&self, name: &str, uuid: &str) -> Result<()> {
        let mut ip = self.in_progress.lock();
        if ip.contains_key(name) {
            return Err(SnapshotError::invalid_snapshot_name(
                self.name(),
                name,
                "snapshot with the same name is already in-progress",
            ));
        }
        ip.insert(name.to_string(), uuid.to_string());
        Ok(())
    }

    pub(crate) fn unregister_in_progress(&self, name: &str) {
        self.in_progress.lock().remove(name);
    }

    pub fn running_snapshot_names(&self) -> Vec<String> {
        self.in_progress.lock().keys().cloned().collect()
    }

    /// Publish a finished snapshot: index + global metadata, `snap-{uuid}.dat`
    /// and a new root `index-N`.
    pub async fn finalize_snapshot(&self, req: FinalizeRequest) -> Result<RepositoryData> {
        self.ensure_writable("create snapshot")?;
        let _g = self.finalize_lock.lock().await;
        let data = self.repository_data().await?;
        let info = &req.info;
        if data.snapshot_by_name(&info.snapshot).is_some() {
            return Err(SnapshotError::invalid_snapshot_name(
                self.name(),
                &info.snapshot,
                "snapshot with the same name already exists",
            ));
        }

        let mut lookup = BTreeMap::new();
        let mut new_identifiers = BTreeMap::new();
        let mut index_rows = Vec::new();
        for idx in &req.indices {
            let ident = metadata_identifier(&idx.index_id, &idx.metadata);
            if !data.index_metadata_identifiers.contains_key(&ident) && !new_identifiers.contains_key(&ident) {
                let blob = random_base64_uuid();
                self.write_meta(&self.index_container(&idx.index_id), &metadata_blob_name(&blob), &idx.metadata, false)
                    .await?;
                new_identifiers.insert(ident.clone(), blob);
            }
            lookup.insert(idx.index_id.clone(), ident);
            let gens: Vec<Option<String>> =
                (0..idx.shard_count).map(|s| self.current_shard_gen(&data, &idx.index_id, s)).collect();
            index_rows.push((idx.name.clone(), idx.index_id.clone(), gens));
        }

        let global = req.global_metadata.clone().unwrap_or_else(|| json!({}));
        self.write_meta(&self.root, &metadata_blob_name(&info.uuid), &global, false).await?;
        let mut stored = info.clone();
        stored.repository = self.name().to_string();
        self.write_meta(&self.root, &snapshot_blob_name(&info.uuid), &stored, false).await?;

        let entry = SnapshotEntry {
            name: info.snapshot.clone(),
            uuid: info.uuid.clone(),
            state: info.state.unwrap_or(SnapshotState::Success).value(),
            index_metadata_lookup: lookup,
            version: info.version.clone().unwrap_or_default(),
            start_time_millis: info.start_time,
            end_time_millis: info.end_time,
            slm_policy: info.slm_policy().unwrap_or_default().to_string(),
        };
        let mut new_data = data.with_snapshot(entry, &index_rows, &new_identifiers);
        new_data.generation = self.write_repository_data(&new_data, data.generation).await?;

        {
            let mut reserved = self.reserved_index_ids.lock();
            for idx in &req.indices {
                reserved.remove(&idx.name);
            }
        }
        let published: BTreeSet<(String, u32, String)> = index_rows
            .iter()
            .flat_map(|(_, id, gens)| {
                gens.iter().enumerate().filter_map(|(s, g)| g.clone().map(|g| (id.clone(), s as u32, g)))
            })
            .collect();
        for (id, shard, gen_) in &req.obsolete_shard_gens {
            if published.contains(&(id.clone(), *shard, gen_.clone())) {
                continue;
            }
            let _ = self
                .shard_container(id, *shard)
                .delete_blobs_ignoring_if_not_exists(&[format!("{INDEX_FILE_PREFIX}{gen_}")])
                .await;
        }
        Ok(new_data)
    }

    /// Delete snapshots by uuid, removing every blob no longer referenced.
    pub async fn delete_snapshots(&self, uuids: &BTreeSet<String>) -> Result<(RepositoryData, DeleteResult)> {
        self.ensure_writable("delete snapshot")?;
        let _g = self.finalize_lock.lock().await;
        let data = self.repository_data().await?;
        for u in uuids {
            if data.snapshot_by_uuid(u).is_none() {
                return Err(SnapshotError::snapshot_missing(self.name(), u));
            }
        }
        let deleted_names: BTreeSet<String> =
            data.snapshots.iter().filter(|s| uuids.contains(&s.uuid)).map(|s| s.name.clone()).collect();
        let (mut new_data, removed_indices) = data.without_snapshots(uuids);
        let keep_names: BTreeSet<String> = new_data
            .snapshots
            .iter()
            .map(|s| s.name.clone())
            .chain(self.in_progress.lock().keys().cloned())
            .collect();

        struct ShardCleanup {
            index_id: String,
            shard: u32,
            old_gen: Option<String>,
            blobs: Vec<String>,
        }
        let mut shard_cleanups = Vec::new();
        let mut guards = Vec::new();
        for (name, entry) in new_data.indices.iter_mut() {
            let old = &data.indices[name];
            if !old.snapshots.iter().any(|u| uuids.contains(u)) {
                continue;
            }
            for shard in 0..entry.shard_generations.len() as u32 {
                let lock = self.shard_lock(&entry.id, shard);
                let guard = lock.clone().lock_owned().await;
                let old_gen = self.current_shard_gen(&data, &entry.id, shard);
                if old_gen.is_none() {
                    continue;
                }
                let mut idx: ShardSnapshots = self.read_shard_snapshots(&entry.id, shard, old_gen.as_deref()).await?;
                let before = idx.referenced_blobs();
                let removed_entries: Vec<(String, String)> = idx
                    .snapshots
                    .iter()
                    .filter(|(n, _)| deleted_names.contains(*n) || !keep_names.contains(*n))
                    .map(|(n, e)| (n.clone(), e.uuid.clone()))
                    .collect();
                for (n, _) in &removed_entries {
                    idx.snapshots.remove(n);
                }
                idx.compact();
                let after = idx.referenced_blobs();
                let new_gen = random_base64_uuid();
                self.write_meta(&self.shard_container(&entry.id, shard), &format!("{INDEX_FILE_PREFIX}{new_gen}"), &idx, true)
                    .await?;
                self.pending_shard_gens.lock().insert((entry.id.clone(), shard), new_gen.clone());
                entry.shard_generations[shard as usize] = Some(new_gen);
                let mut blobs: Vec<String> = before.difference(&after).cloned().collect();
                for (_, u) in &removed_entries {
                    if !u.is_empty() {
                        blobs.push(snapshot_blob_name(u));
                    }
                }
                for u in uuids {
                    blobs.push(snapshot_blob_name(u));
                }
                shard_cleanups.push(ShardCleanup { index_id: entry.id.clone(), shard, old_gen, blobs });
                guards.push(guard);
            }
        }
        new_data.generation = self.write_repository_data(&new_data, data.generation).await?;
        drop(guards);

        // Unreferenced blobs are only removed after the new index-N is durable.
        let mut res = DeleteResult::default();
        let root_blobs: Vec<String> =
            uuids.iter().flat_map(|u| [snapshot_blob_name(u), metadata_blob_name(u)]).collect();
        res.add(self.delete_counted(&self.root, &root_blobs).await);
        for sc in shard_cleanups {
            let mut blobs = sc.blobs;
            if let Some(g) = sc.old_gen {
                blobs.push(format!("{INDEX_FILE_PREFIX}{g}"));
            }
            res.add(self.delete_counted(&self.shard_container(&sc.index_id, sc.shard), &blobs).await);
        }
        let active: BTreeSet<String> = self.active_index_ids.lock().keys().cloned().collect();
        for (_, e) in &removed_indices {
            if active.contains(&e.id) {
                continue;
            }
            match self.index_container(&e.id).delete().await {
                Ok(r) => res.add(r),
                Err(e2) => tracing::warn!("[{}] failed to delete index dir [{}]: {e2}", self.name(), e.id),
            }
            let mut p = self.pending_shard_gens.lock();
            p.retain(|(id, _), _| id != &e.id);
        }
        // Index metadata blobs whose identifier is gone.
        let mut ident_to_index: BTreeMap<String, String> = BTreeMap::new();
        for s in &data.snapshots {
            for (id, ident) in &s.index_metadata_lookup {
                ident_to_index.insert(ident.clone(), id.clone());
            }
        }
        let removed_ids: BTreeSet<&String> = removed_indices.iter().map(|(_, e)| &e.id).collect();
        for (ident, blob) in &data.index_metadata_identifiers {
            if new_data.index_metadata_identifiers.contains_key(ident) {
                continue;
            }
            if let Some(id) = ident_to_index.get(ident) {
                if removed_ids.contains(id) {
                    continue;
                }
                res.add(self.delete_counted(&self.index_container(id), &[metadata_blob_name(blob)]).await);
            }
        }
        Ok((new_data, res))
    }

    /// Delete blobs, counting those that existed (sizes from a listing).
    async fn delete_counted(&self, c: &crate::blobstore::BlobContainer, names: &[String]) -> DeleteResult {
        let mut res = DeleteResult::default();
        if names.is_empty() {
            return res;
        }
        let listing = c.list_blobs().await.unwrap_or_default();
        let existing: Vec<String> = names.iter().filter(|n| listing.contains_key(*n)).cloned().collect();
        for n in &existing {
            res.blobs_deleted += 1;
            res.bytes_deleted += listing[n];
        }
        if let Err(e) = c.delete_blobs_ignoring_if_not_exists(&existing).await {
            tracing::warn!("[{}] failed to delete blobs in [{}]: {e}", self.name(), c.path());
            return DeleteResult::default();
        }
        res
    }

    /// Clone `source` into a new snapshot `target` containing `indices`.
    pub async fn clone_snapshot(&self, source_name: &str, target_name: &str, indices: &[String]) -> Result<SnapshotInfo> {
        self.ensure_writable("clone snapshot")?;
        crate::repository::settings::validate_snapshot_name(self.name(), target_name)?;
        let data = self.repository_data().await?;
        let src = data
            .snapshot_by_name(source_name)
            .ok_or_else(|| SnapshotError::snapshot_missing(self.name(), source_name))?
            .clone();
        if data.snapshot_by_name(target_name).is_some() {
            return Err(SnapshotError::invalid_snapshot_name(
                self.name(),
                target_name,
                "snapshot with the same name already exists",
            ));
        }
        let src_info = self.read_snapshot_info(&src.uuid).await?;
        let (selected, _missing) = crate::util::resolve_names(indices, &src_info.indices);
        if selected.is_empty() {
            return Err(SnapshotError::illegal_argument(format!(
                "No indices in the source snapshot [{}] matched requested pattern [{}]",
format!("{}/{}", source_name, src.uuid),
                indices.join(", ")
            )));
        }
        let target_uuid = random_base64_uuid();
        self.register_in_progress(target_name, &target_uuid)?;
        let result = self.clone_inner(&data, &src, &src_info, target_name, &target_uuid, &selected).await;
        self.unregister_in_progress(target_name);
        result
    }

    async fn clone_inner(
        &self,
        data: &RepositoryData,
        src: &SnapshotEntry,
        src_info: &SnapshotInfo,
        target_name: &str,
        target_uuid: &str,
        selected: &[String],
    ) -> Result<SnapshotInfo> {
        let start = crate::util::now_millis();
        let mut finalize_indices = Vec::new();
        let mut obsolete = Vec::new();
        let mut total_shards = 0;
        let mut details = BTreeMap::new();
        for name in selected {
            if src_info.failures.iter().any(|f| &f.index == name) {
                return Err(SnapshotError::snapshot(
                    self.name(),
                    target_name,
                    format!("Can't clone index [{name}] because its snapshot was not successful."),
                ));
            }
            let ie = &data.indices[name];
            let shard_count = ie.shard_generations.len() as u32;
            let mut size = 0;
            for shard in 0..shard_count {
                let lock = self.shard_lock(&ie.id, shard);
                let _guard = lock.lock().await;
                let gen_ = self.current_shard_gen(data, &ie.id, shard);
                let mut idx = self.read_shard_snapshots(&ie.id, shard, gen_.as_deref()).await?;
                let files = idx.snapshots.get(&src.name).cloned().ok_or_else(|| {
                    SnapshotError::snapshot(
                        self.name(),
                        target_name,
                        format!("Can't clone index [{name}] because its snapshot was not successful."),
                    )
                })?;
                idx.snapshots.insert(
                    target_name.to_string(),
                    super::ShardSnapshotFiles { uuid: target_uuid.to_string(), files: files.files.clone() },
                );
                let c = self.shard_container(&ie.id, shard);
                let mut meta = self.read_shard_snapshot(&ie.id, shard, &src.uuid).await?;
                meta.name = target_name.to_string();
                meta.number_of_files = 0;
                meta.total_size = 0;
                meta.start_time = start;
                meta.time = 0;
                size += meta.files.iter().map(|f| f.length).sum::<u64>();
                self.write_meta(&c, &snapshot_blob_name(target_uuid), &meta, false).await?;
                let new_gen = random_base64_uuid();
                self.write_meta(&c, &format!("{INDEX_FILE_PREFIX}{new_gen}"), &idx, true).await?;
                self.pending_shard_gens.lock().insert((ie.id.clone(), shard), new_gen);
                if let Some(g) = gen_ {
                    obsolete.push((ie.id.clone(), shard, g));
                }
            }
            total_shards += shard_count;
            let metadata = self.read_index_metadata(data, &src.uuid, name).await?;
            details.insert(
                name.clone(),
                IndexDetails {
                    shard_count,
                    size_in_bytes: size,
                    max_segments_per_shard: src_info.index_details.get(name).map(|d| d.max_segments_per_shard).unwrap_or(0),
                },
            );
            finalize_indices.push(FinalizeIndex { name: name.clone(), index_id: ie.id.clone(), metadata, shard_count });
        }
        let global = self.read_global_metadata(&src.uuid).await.ok();
        let end = crate::util::now_millis();
        let info = SnapshotInfo {
            repository: self.name().to_string(),
            snapshot: target_name.to_string(),
            uuid: target_uuid.to_string(),
            version_id: Some(crate::snapshot::ES_VERSION_ID),
            version: Some(crate::snapshot::ES_VERSION.to_string()),
            indices: selected.to_vec(),
            data_streams: src_info.data_streams.clone(),
            feature_states: vec![],
            include_global_state: Some(false),
            metadata: src_info.metadata.clone(),
            state: Some(SnapshotState::Success),
            reason: None,
            start_time: start,
            end_time: end,
            total_shards,
            successful_shards: total_shards,
            failures: vec![],
            index_details: details,
        };
        self.finalize_snapshot(FinalizeRequest {
            info: info.clone(),
            indices: finalize_indices,
            global_metadata: global,
            obsolete_shard_gens: obsolete,
        })
        .await?;
        Ok(info)
    }

    /// `POST _snapshot/{repo}/_cleanup`: remove every blob not reachable from
    /// the current `index-N`.
    pub async fn cleanup(&self) -> Result<CleanupResult> {
        self.ensure_writable("clean up")?;
        let _g = self.finalize_lock.lock().await;
        if !self.in_progress.lock().is_empty() {
            return Err(SnapshotError::illegal_state(format!(
                "Cannot cleanup [{}] - a snapshot is currently running",
                self.name()
            )));
        }
        let data = self.repository_data().await?;
        let mut res = DeleteResult::default();
        let uuids: BTreeSet<&str> = data.snapshots.iter().map(|s| s.uuid.as_str()).collect();
        let names: BTreeSet<&str> = data.snapshots.iter().map(|s| s.name.as_str()).collect();

        // Root blobs.
        let mut stale = Vec::new();
        for name in self.root.list_blobs().await?.keys() {
            let keep = if name == super::INDEX_LATEST_BLOB {
                true
            } else if let Some(g) = name.strip_prefix(INDEX_FILE_PREFIX) {
                g.parse::<i64>().map(|g| g >= data.generation).unwrap_or(false)
            } else if let Some(u) = name.strip_prefix(SNAPSHOT_PREFIX).and_then(|n| n.strip_suffix(".dat")) {
                uuids.contains(u)
            } else if let Some(u) = name.strip_prefix(METADATA_PREFIX).and_then(|n| n.strip_suffix(".dat")) {
                uuids.contains(u)
            } else {
                false
            };
            if !keep {
                stale.push(name.clone());
            }
        }
        res.add(self.delete_counted(&self.root, &stale).await);
        for child in self.root.children().await? {
            if child.starts_with(TESTS_PREFIX) {
                res.add(self.root.child(&child).delete().await?);
            }
        }

        // Index directories.
        let indices_c = self.indices_container();
        let live_ids: BTreeMap<&str, (&String, &super::IndexEntry)> =
            data.indices.iter().map(|(n, e)| (e.id.as_str(), (n, e))).collect();
        let live_blobs: BTreeSet<&String> = data.index_metadata_identifiers.values().collect();
        for id in indices_c.children().await? {
            let Some((_, entry)) = live_ids.get(id.as_str()) else {
                res.add(indices_c.child(&id).delete().await?);
                continue;
            };
            let ic = indices_c.child(&id);
            let stale_meta: Vec<String> = ic
                .list_blobs_by_prefix(METADATA_PREFIX)
                .await?
                .into_keys()
                .filter(|n| {
                    n.strip_prefix(METADATA_PREFIX)
                        .and_then(|n| n.strip_suffix(".dat"))
                        .map(|b| !live_blobs.contains(&b.to_string()))
                        .unwrap_or(true)
                })
                .collect();
            res.add(self.delete_counted(&ic, &stale_meta).await);
            for shard_dir in ic.children().await? {
                let sc = ic.child(&shard_dir);
                let shard: Option<u32> = shard_dir.parse().ok().filter(|s| (*s as usize) < entry.shard_generations.len());
                let Some(shard) = shard else {
                    res.add(sc.delete().await?);
                    continue;
                };
                let gen_ = entry.shard_generations[shard as usize].clone();
                self.pending_shard_gens.lock().remove(&(id.clone(), shard));
                let idx = self.read_shard_snapshots(&id, shard, gen_.as_deref()).await?;
                let referenced = idx.referenced_blobs();
                let snap_uuids: BTreeSet<String> = idx
                    .snapshots
                    .iter()
                    .filter(|(n, _)| names.contains(n.as_str()))
                    .map(|(_, e)| e.uuid.clone())
                    .collect();
                let mut stale = Vec::new();
                for blob in sc.list_blobs().await?.into_keys() {
                    let keep = if blob.starts_with(DATA_BLOB_PREFIX) {
                        referenced.contains(&blob)
                    } else if let Some(g) = blob.strip_prefix(INDEX_FILE_PREFIX) {
                        Some(g) == gen_.as_deref()
                    } else if let Some(u) = blob.strip_prefix(SNAPSHOT_PREFIX).and_then(|n| n.strip_suffix(".dat")) {
                        snap_uuids.contains(u)
                    } else {
                        false
                    };
                    if !keep {
                        stale.push(blob);
                    }
                }
                res.add(self.delete_counted(&sc, &stale).await);
            }
        }
        Ok(CleanupResult { deleted_bytes: res.bytes_deleted, deleted_blobs: res.blobs_deleted })
    }

    /// `POST _snapshot/{repo}/_verify`: write, read back and delete a test
    /// blob per node. `nodes` = (node id, node name).
    pub async fn verify(&self, nodes: &[(String, String)]) -> Result<Value> {
        let location = self.store.describe();
        if self.settings.readonly {
            self.latest_generation().await.map_err(|e| {
                SnapshotError::repository_verification(
                    self.name(),
                    format!("path [{location}] is not accessible on master node: {}", e.reason),
                )
            })?;
        } else {
            let seed = random_base64_uuid();
            let c = self.root.child(&format!("{TESTS_PREFIX}{seed}"));
            let r: Result<()> = async {
                c.write_blob_atomic("master.dat", Bytes::from(seed.clone().into_bytes()), true).await?;
                let back = c.read_blob("master.dat").await?;
                if back != seed.as_bytes() {
                    return Err(SnapshotError::io("verify", "read back different content"));
                }
                for (id, _) in nodes {
                    let name = format!("data-{id}.dat");
                    c.write_blob(&name, Bytes::from(seed.clone().into_bytes()), true).await?;
                    if c.read_blob(&name).await? != seed.as_bytes() {
                        return Err(SnapshotError::io("verify", "read back different content"));
                    }
                }
                Ok(())
            }
            .await;
            let cleanup = c.delete().await;
            if let Err(e) = r {
                return Err(SnapshotError::repository_verification(
                    self.name(),
                    format!("path [{location}] is not accessible on master node: {}", e.reason),
                ));
            }
            cleanup?;
        }
        let mut m = serde_json::Map::new();
        for (id, name) in nodes {
            m.insert(id.clone(), json!({"name": name}));
        }
        Ok(json!({"nodes": Value::Object(m)}))
    }
}
