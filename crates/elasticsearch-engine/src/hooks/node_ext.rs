//! Hooks 1 and 2 on [`Node`]: create an index with a given uuid, adopt an
//! index directory placed on disk at runtime (file-based peer recovery,
//! snapshot restore), reopen an index from disk. Compiled as a private
//! child module of `crate::node`.

use super::*;
use crate::hooks::files::{read_manifest, verify_index_dir};

/// Options of [`Node::adopt_index_with`].
#[derive(Clone, Debug)]
pub struct AdoptOptions {
    /// Register under this name instead of the one in `index_meta.json`
    /// (snapshot restore with `rename_pattern`).
    pub rename: Option<String>,
    /// Register under this uuid instead of the source's (restore as a new index).
    pub uuid: Option<String>,
    /// Verify every file against `transfer_manifest.json` when present.
    pub verify: bool,
    /// Fail when there is no `transfer_manifest.json` (i.e. the directory
    /// did not come from a completed transfer).
    pub require_manifest: bool,
    /// Copy the directory instead of moving it (the source stays intact).
    pub copy: bool,
    /// Make the adopted copy operate at this primary term, even if lower than
    /// the one it was captured at (snapshot restore resets it): the term
    /// history becomes `[(0, term)]`. Documents keep their stored
    /// `_primary_term`. `None` keeps the source's history.
    pub term: Option<u64>,
}

impl Default for AdoptOptions {
    fn default() -> Self {
        AdoptOptions { rename: None, uuid: None, verify: true, require_manifest: false, copy: false, term: None }
    }
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), &to)?;
            rustic_core::platform::sync_file(&to)?;
        }
    }
    crate::hooks::sync_dir(Some(dst));
    Ok(())
}

fn validate_uuid(uuid: &str) -> Result<()> {
    if uuid.is_empty() || uuid.len() > 64 || !uuid.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(EsError::illegal_argument(format!("invalid index uuid [{uuid}]")));
    }
    Ok(())
}

impl Node {
    /// Hook 1: create an index whose `index.uuid` is `uuid` (the cluster-wide
    /// uuid chosen by the master) instead of a random one. Same body and
    /// semantics as [`Node::create_index`]; a uuid already used by another
    /// index on this node is rejected.
    pub fn create_index_with_uuid(&self, name: &str, uuid: &str, body: &Value) -> Result<Arc<IndexService>> {
        validate_uuid(uuid)?;
        if let Some(other) = self.all_indices().into_iter().find(|s| s.uuid == uuid) {
            return Err(EsError::illegal_argument(format!("index uuid [{uuid}] is already used by index [{}]", other.name)));
        }
        self.create_index_impl(name, body, Some(uuid), true)
    }

    /// Register an index whose metadata the cluster already resolved,
    /// with **no shard open** (the cluster opens the copies allocated here
    /// with [`IndexService::open_shard`]). `metadata` is taken verbatim —
    /// no templates, no defaults: `{"settings": {..}, "mappings": {..},
    /// "aliases": {..}}` (settings flat or nested; `index.uuid` is set to
    /// `uuid`). The service is *sparse*: whole-index conveniences (write /
    /// get / update / search over "the index") act on the copies open here;
    /// a document op routed to a shard with no local copy fails with
    /// `shard_not_found_exception`, a search covers only the open copies.
    pub fn create_index_sparse(&self, name: &str, uuid: &str, metadata: &Value) -> Result<Arc<IndexService>> {
        validate_uuid(uuid)?;
        if let Err(why) = is_valid_name(name) {
            return Err(EsError::invalid_index_name(name, &why));
        }
        let _g = self.create_lock.lock();
        if let Some(existing) = self.get_index(name) {
            return Err(EsError::index_already_exists(name, &existing.uuid));
        }
        if let Some(other) = self.all_indices().into_iter().find(|s| s.uuid == uuid) {
            return Err(EsError::illegal_argument(format!("index uuid [{uuid}] is already used by index [{}]", other.name)));
        }
        let mut flat = settings::normalize_index_settings(metadata.get("settings").unwrap_or(&Value::Null));
        flat.insert("index.uuid".into(), json!(uuid));
        // The cluster recovers the copies itself; load the synonyms sets when
        // they build, validate without them otherwise (ES `CREATE_INDEX`).
        let registry = crate::index::registry_for_in(&flat, Some(&self.data), true).or_else(|_| crate::index::registry_for(&flat))?;
        let mut mappings = crate::mapping::Mappings::parse(metadata.get("mappings").filter(|m| !m.is_null()).unwrap_or(&json!({})), &registry)?;
        // Published REST mappings omit internal TSID fields; reconstruct them
        // before opening the physical shard copies, just as normal creation does.
        crate::tsdb::prepare_mappings(&flat, &mut mappings, &registry)?;
        mappings.assign_core_ids();
        let lookup = crate::mapping::FieldLookup::build(&mappings, &registry);
        let aliases: std::collections::BTreeMap<String, Value> = metadata.get("aliases").and_then(Value::as_object).map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
        let dir = self.data.join("indices").join(new_uuid());
        std::fs::create_dir_all(&dir)?;
        let created = flat.get("index.creation_date").and_then(|v| v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_i64())).unwrap_or_else(crate::date::now_millis);
        let core = crate::index::IndexCore::new_sparse(name, uuid, created, &dir, flat, mappings, lookup, registry, aliases);
        let svc = Arc::new(IndexService::sparse_from(core));
        if let Err(e) = svc.persist() {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
        svc.set_mapping_update_hook(self.mapping_hook.read().clone());
        self.indices.write().insert(name.to_string(), svc.clone());
        Ok(svc)
    }

    /// Install the node-level dynamic mapping hook (A6) on every index,
    /// present and future. `None` restores local application.
    pub fn set_mapping_update_hook(&self, hook: Option<Arc<dyn crate::index::MappingUpdateHook>>) {
        *self.mapping_hook.write() = hook.clone();
        for svc in self.all_indices() {
            svc.set_mapping_update_hook(hook.clone());
        }
    }

    /// A fresh directory (outside `indices/`, so a restart never opens a
    /// half-received copy) to receive an index's files into before
    /// [`Node::adopt_index`].
    pub fn new_staging_dir(&self) -> Result<PathBuf> {
        let d = self.data.join("staging").join(new_uuid());
        std::fs::create_dir_all(&d)?;
        Ok(d)
    }

    /// Hook 2: register the index whose files are in `src` (copied in from
    /// a peer or a snapshot, e.g. by
    /// [`IndexFileSet::write_to_dir`](crate::hooks::IndexFileSet::write_to_dir))
    /// without a restart. See [`Node::adopt_index_with`].
    pub fn adopt_index(&self, src: &Path) -> Result<Arc<IndexService>> {
        self.adopt_index_with(src, &AdoptOptions::default())
    }

    /// Adopt an index directory: verify it (against its transfer manifest),
    /// check the name/uuid are free, move (or copy) it under `indices/`,
    /// drop per-copy state (retention leases), open it — metadata, mappings,
    /// settings, aliases, seq_no / version / primary-term state, translog
    /// replay — and register it. On failure the node is unchanged and a
    /// moved directory is moved back.
    pub fn adopt_index_with(&self, src: &Path, opts: &AdoptOptions) -> Result<Arc<IndexService>> {
        let meta_path = src.join("index_meta.json");
        let raw: Value = match std::fs::read(&meta_path) {
            Ok(b) => serde_json::from_slice(&b)?,
            Err(e) => return Err(EsError::illegal_argument(format!("no index to adopt at [{}]: {e}", src.display()))),
        };
        match read_manifest(src)? {
            Some(m) if opts.verify => verify_index_dir(src, &m)?,
            Some(_) => {}
            None if opts.require_manifest => return Err(EsError::illegal_argument(format!("[{}] has no transfer manifest", src.display()))),
            None => {}
        }
        let name = match &opts.rename {
            Some(n) => resolve_date_math(n)?,
            None => raw["name"].as_str().unwrap_or_default().to_string(),
        };
        if let Err(why) = is_valid_name(&name) {
            return Err(EsError::invalid_index_name(&name, &why));
        }
        let uuid = opts.uuid.clone().unwrap_or_else(|| raw["uuid"].as_str().unwrap_or_default().to_string());
        validate_uuid(&uuid)?;
        let _g = self.create_lock.lock();
        if let Some(existing) = self.get_index(&name) {
            return Err(EsError::index_already_exists(&name, &existing.uuid));
        }
        for svc in self.all_indices() {
            if svc.aliases().contains_key(&name) {
                return Err(EsError::new(400, "invalid_index_name_exception", format!("Invalid index name [{name}], already exists as alias")).with("index_uuid", "_na_").with("index", name.clone()));
            }
            if svc.uuid == uuid {
                return Err(EsError::illegal_argument(format!("index uuid [{uuid}] is already used by index [{}]", svc.name)));
            }
        }
        let root = self.data.join("indices");
        let in_place = src.canonicalize().ok().and_then(|p| p.parent().map(Path::to_path_buf)) == root.canonicalize().ok();
        let dest = if in_place { src.to_path_buf() } else { root.join(new_uuid()) };
        let mut moved = false;
        if !in_place {
            if opts.copy {
                copy_dir(src, &dest)?;
            } else if std::fs::rename(src, &dest).is_ok() {
                moved = true;
            } else {
                copy_dir(src, &dest)?;
                let _ = std::fs::remove_dir_all(src);
            }
            crate::hooks::sync_dir(Some(&root));
        }
        let undo = |dest: &Path| {
            if moved {
                let _ = std::fs::rename(dest, src);
            } else if !in_place {
                let _ = std::fs::remove_dir_all(dest);
            }
        };
        let result = (|| -> Result<IndexService> {
            // Per-copy state never transfers.
            let _ = std::fs::remove_file(dest.join("shard-0").join(crate::hooks::retention::LEASES_FILE));
            if opts.rename.is_some() || opts.uuid.is_some() {
                let mut m = raw.clone();
                m["name"] = json!(name);
                m["uuid"] = json!(uuid);
                if let Some(s) = m["settings"].as_object_mut() {
                    s.insert("index.provided_name".into(), json!(name));
                    s.insert("index.uuid".into(), json!(uuid));
                }
                crate::hooks::write_durable(&dest.join("index_meta.json"), &serde_json::to_vec_pretty(&m)?)?;
            }
            if let Some(t) = opts.term {
                if t == 0 {
                    return Err(EsError::illegal_argument("primary term must be >= 1"));
                }
                let h = crate::hooks::TermHistory { current: t, ranges: vec![(0, t)] };
                h.persist(&dest.join("shard-0"))?;
            }
            IndexService::open(&dest)
        })();
        match result {
            Ok(svc) => {
                let svc = Arc::new(svc);
                svc.set_mapping_update_hook(self.mapping_hook.read().clone());
                self.indices.write().insert(name, svc.clone());
                Ok(svc)
            }
            Err(e) => {
                undo(&dest);
                Err(e)
            }
        }
    }

    /// Re-open `name` from its directory (e.g. after its files were
    /// replaced in place by a segment transfer). The new instance is opened
    /// (translog replayed) before it replaces the old one, so on error the
    /// old instance stays registered. The caller must stop writes to the
    /// index for the duration (the cluster holds its write gate): requests
    /// still holding the old `Arc` would write into a detached copy.
    pub fn reopen_index(&self, name: &str) -> Result<Arc<IndexService>> {
        let _g = self.create_lock.lock();
        let old = self.get_index(name).ok_or_else(|| EsError::index_not_found(name))?;
        // The new instance needs the shard locks: close the old copies
        // (durably) first; if the reopen fails, reopen the old state.
        let open_before: Vec<(u32, String, bool, u64)> = old.local_shards().iter().map(|s| (s.id, s.allocation_id.clone(), s.is_primary(), s.shard.primary_term())).collect();
        for (n, ..) in &open_before {
            old.close_shard(*n)?;
        }
        // A sparse index reopens with no copy: reopen the ones that were open.
        let reopen_copies = |svc: &IndexService| -> Result<()> {
            if svc.is_sparse() {
                for (n, alloc, primary, term) in &open_before {
                    svc.open_shard(*n, crate::index::ShardOpenSpec { allocation_id: alloc.clone(), primary: *primary, primary_term: *term, recovery: crate::index::RecoverySource::ExistingStore })?;
                }
            }
            Ok(())
        };
        let svc = match IndexService::open(old.path()).and_then(|s| reopen_copies(&s).map(|_| s)) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                if let Ok(again) = IndexService::open(old.path()).and_then(|s| reopen_copies(&s).map(|_| s)) {
                    let again = Arc::new(again);
                    again.set_mapping_update_hook(self.mapping_hook.read().clone());
                    self.indices.write().insert(name.to_string(), again);
                }
                return Err(e);
            }
        };
        svc.set_mapping_update_hook(self.mapping_hook.read().clone());
        self.indices.write().insert(name.to_string(), svc.clone());
        self.scrolls.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.pits.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.fold_contexts.lock().retain(|_, s| !s.targets.iter().any(|t| t.svc.name == name));
        self.drop_fetch_contexts(name);
        Ok(svc)
    }
}

#[cfg(test)]
mod tests {
    // The whole-index hook shims stay covered here.
    #![allow(deprecated)]
    use super::*;
    use crate::hooks::changes::{ApplyOutcome, ChangeOp, Durability};
    use crate::index::WriteReq;
    use crate::shard::WriteCond;

    fn put(svc: &IndexService, id: &str, body: &str) {
        let r = svc.write(vec![WriteReq::Index { id: Some(id.into()), source: body.into(), routing: Some(format!("r-{id}")), cond: WriteCond::default() }], true);
        r.into_iter().next().unwrap().unwrap();
    }

    fn count(svc: &IndexService) -> u64 {
        svc.doc_count_realtime().0
    }

    #[test]
    fn create_with_uuid() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = node.create_index_with_uuid("a", "clusterUuid123", &json!({"settings": {"number_of_replicas": 0}})).unwrap();
        assert_eq!(svc.uuid, "clusterUuid123");
        assert_eq!(svc.setting("index.uuid").as_deref(), Some("clusterUuid123"));
        assert!(node.create_index_with_uuid("b", "clusterUuid123", &json!({})).is_err());
        assert!(node.create_index_with_uuid("c", "bad/uuid", &json!({})).is_err());
        drop(svc);
        drop(node);
        let node = Node::open(tmp.path()).unwrap();
        assert_eq!(node.get_index("a").unwrap().uuid, "clusterUuid123");
    }

    #[test]
    fn adopt_from_files_round_trip() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let src = Node::open(a.path()).unwrap();
        let svc = src.create_index_with_uuid("logs", "U1", &json!({"mappings": {"properties": {"msg": {"type": "text"}, "n": {"type": "long"}}}, "aliases": {"l": {}}})).unwrap();
        for i in 0..30 {
            put(&svc, &format!("d{i}"), &format!(r#"{{"msg": "hello {i}", "n": {i}, "dyn": "x{i}"}}"#));
        }
        svc.shard(0).unwrap().shard.flush().unwrap();
        svc.shard(0).unwrap().shard.set_primary_term(4).unwrap();
        for i in 30..40 {
            put(&svc, &format!("d{i}"), &format!(r#"{{"msg": "late {i}", "n": {i}}}"#));
        }
        svc.write(vec![WriteReq::Delete { id: "d3".into(), routing: None, cond: WriteCond::default() }], true);
        let max = svc.shard(0).unwrap().shard.max_seq_no();

        // 1. Full copy with translog: everything, including ops above the commit.
        let set = svc.snapshot_files(true).unwrap();
        assert!(set.manifest.includes_translog && set.manifest.max_seq_no == max);
        // Writes after the snapshot are not in it.
        put(&svc, "after", r#"{"msg": "after"}"#);
        let dst = Node::open(b.path()).unwrap();
        let staging = dst.new_staging_dir().unwrap();
        set.write_to_dir(&staging, false).unwrap();
        let copy = dst.adopt_index(&staging).unwrap();
        assert!(!staging.exists(), "moved into indices/");
        assert_eq!((copy.name.as_str(), copy.uuid.as_str()), ("logs", "U1"));
        assert_eq!(count(&copy), 39);
        assert!(copy.get("after").unwrap().is_none());
        let g = copy.get("d35").unwrap().unwrap();
        assert_eq!((g.seq_no, g.version, g.routing.as_deref()), (35, 1, Some("r-d35")));
        assert_eq!(copy.shard(0).unwrap().shard.doc_primary_term("d35"), Some(4));
        assert_eq!(copy.shard(0).unwrap().shard.doc_primary_term("d1"), Some(1));
        assert_eq!(copy.shard(0).unwrap().shard.primary_term(), 4);
        assert_eq!(copy.shard(0).unwrap().shard.max_seq_no(), max);
        assert_eq!(copy.shard(0).unwrap().shard.local_checkpoint(), max);
        assert!(copy.lookup().get("dyn").is_some(), "dynamic mappings travel");
        assert!(copy.aliases().contains_key("l"));
        assert!(dst.resolve("l", ResolveOpts::default()).is_ok());
        // 2. Catch up with the primary's history, then the copy continues it.
        let ops: Vec<ChangeOp> = svc.shard(0).unwrap().shard.changes_since(copy.shard(0).unwrap().shard.local_checkpoint() + 1).unwrap().map(|op| (*op).clone()).collect();
        assert_eq!(ops.len(), 1);
        let r = copy.apply_replica_ops(svc.shard(0).unwrap().shard.primary_term(), ops, Durability::Request).unwrap();
        assert_eq!(r.outcomes, vec![ApplyOutcome::Applied]);
        assert!(copy.get("after").unwrap().is_some());
        // 3. Restart of the receiving node: the adopted index comes back.
        drop(copy);
        drop(dst);
        let dst = Node::open(b.path()).unwrap();
        let copy = dst.get_index("logs").unwrap();
        assert_eq!(count(&copy), 40);
        assert_eq!(copy.shard(0).unwrap().shard.local_checkpoint(), max + 1);
        // 3b. A hard-linked copy is identical (and shares inodes for commit files).
        svc.shard(0).unwrap().shard.flush().unwrap();
        let set = svc.snapshot_files(false).unwrap();
        let linked = dst.new_staging_dir().unwrap();
        set.write_to_dir(&linked, true).unwrap();
        verify_index_dir(&linked, &set.manifest).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let seg = set.manifest.files.iter().find(|f| f.path.ends_with(".rstc")).unwrap();
            assert!(std::fs::metadata(linked.join(&seg.path)).unwrap().nlink() >= 2, "segment file is hard-linked");
        }
        std::fs::remove_dir_all(&linked).unwrap();
        // 4. Snapshot-style copy (commit only, no translog), restored under
        //    another name and uuid; name / uuid clashes are rejected.
        svc.shard(0).unwrap().shard.flush().unwrap();
        let set = svc.snapshot_files(false).unwrap();
        assert!(!set.manifest.includes_translog);
        let staging = dst.new_staging_dir().unwrap();
        set.write_to_dir(&staging, false).unwrap();
        assert_eq!(dst.adopt_index(&staging).unwrap_err().kind, "resource_already_exists_exception");
        assert!(staging.exists(), "rejected adopt leaves the source alone");
        let opts = AdoptOptions { rename: Some("restored".into()), ..Default::default() };
        assert!(dst.adopt_index_with(&staging, &opts).is_err(), "uuid U1 is taken");
        let opts = AdoptOptions { rename: Some("restored".into()), uuid: Some("U2".into()), term: Some(1), ..Default::default() };
        let r = dst.adopt_index_with(&staging, &opts).unwrap();
        assert_eq!(r.shard(0).unwrap().shard.primary_term(), 1, "reset below the source's term 4");
        assert_eq!(r.shard(0).unwrap().shard.term_history().ranges, vec![(0, 1)]);
        assert_eq!((r.name.as_str(), r.uuid.as_str()), ("restored", "U2"));
        assert_eq!(r.setting("index.provided_name").as_deref(), Some("restored"));
        assert_eq!(count(&r), 40);
        // 5. Corruption is detected before anything is registered.
        let set = svc.snapshot_files(false).unwrap();
        let staging = dst.new_staging_dir().unwrap();
        set.write_to_dir(&staging, false).unwrap();
        let seg = set.manifest.files.iter().find(|f| f.path.ends_with(".rstc")).unwrap();
        let p = staging.join(&seg.path);
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[40] ^= 0xff;
        std::fs::write(&p, bytes).unwrap();
        let opts = AdoptOptions { rename: Some("corrupt".into()), uuid: Some("U3".into()), ..Default::default() };
        assert_eq!(dst.adopt_index_with(&staging, &opts).unwrap_err().kind, "corrupt_index_exception");
        assert!(dst.get_index("corrupt").is_none());
    }

    #[test]
    fn reopen_index_picks_up_replaced_files() {
        let a = tempfile::tempdir().unwrap();
        let node = Node::open(a.path()).unwrap();
        let svc = node.create_index("x", &json!({})).unwrap();
        put(&svc, "1", r#"{"a": 1}"#);
        let reopened = node.reopen_index("x").unwrap();
        assert_eq!(count(&reopened), 1, "translog replayed");
        assert!(node.reopen_index("missing").is_err());
    }
}
